# Design: Calibration error modelling

**Status:** Research / exploratory
**Complexity:** Medium to high depending on approach

---

## Problem

Camera calibration errors produce systematic, spatially-correlated residuals that the
current Gaussian noise model misrepresents. Specifically:

- The reprojection error of an extrinsic/intrinsic calibration is a **smooth function of
  3D position** — a marker observed in one part of the image may be consistently biased
  towards one direction, while the same marker in a different part of the image has a
  different bias.
- These errors are **not random per frame** — a poorly calibrated camera will have
  approximately the same bias for a given marker position across many frames.
- The magnitude can be **comparable to, or larger than, pose estimation error** (5–30 px
  for a typical Pose2Sim calibration).

The existing velocity mode (`velocity_measurement_noise_std`) addresses a different problem:
a *single* camera with severely bad calibration, handled by excluding it from the normal
update and using its frame-to-frame pixel differences instead, which requires enough
well-calibrated cameras to anchor the pose.

The goal here is distinct: improving tracking quality when **all cameras have roughly
comparable, moderate calibration error** — good enough to use normally, but with residuals
that limit accuracy for fine motion. The primary use cases are:

- Small gestures (finger movement, subtle wrist rotation)
- Hand–object interaction (person touching or manipulating an object)
- Two-person interaction (hands in proximity, contact between subjects)

In all these cases even a few pixels of systematic calibration error can prevent the filter
from resolving the motion correctly, or can cause model penetration when two body parts
(e.g. both hands) should be close together.

---

## Approach 1 — Per-camera bias states in the UKF

**Idea:** Augment the UKF state vector with a 2D pixel-offset bias `b_k = (bx_k, by_k)`
per camera. The bias is modelled as a slow random walk (small process noise), so the filter
estimates it online and updates it as the person moves through the scene.

The measurement model becomes:
```
z_ik = project(marker_i, cam_k, x) + b_k + noise(sigma)
```

The bias is subtracted from each observation before the normal position update:
effectively the filter learns and removes the systematic offset.

**State augmentation:**

If there are `K` cameras, add `2K` DOFs to the state vector. Process noise for bias states
is much smaller than for pose DOFs — e.g. `sigma_bias ≈ 0.1–1.0 px/sqrt(s)` (a few pixels
of drift per second at most). The bias initial covariance should be large (e.g. `10–30 px`)
to allow the filter to converge quickly.

**Implementation sketch (C++):**

1. Extend `State` to carry a `bias` vector of size `2K` (one 2D entry per camera in the
   active camera list). Alternatively keep it as a separate field in `Tracker` and augment
   the UKF externally.
2. In `UKF::update()`, for each sigma point, add the corresponding `b_k` to the predicted
   measurement before computing the innovation.
3. In `UKF::predict()`, propagate bias states as identity (random walk — no dynamics).
4. The Kalman gain will then naturally partition updates between pose DOFs and bias DOFs.

**Pros:** Principled; handles slowly-varying biases; online; no offline pass needed.

**Cons:** State size grows linearly with number of cameras (8 cameras → 16 extra DOFs on
top of ~218 pose DOFs — manageable). If the true bias varies rapidly with 3D position, the
random-walk model will lag. Does not model the spatial correlation directly.

**Fit to the stated goal:** The random-walk process model is well matched to *temporal*
drift (e.g. a camera nudged mid-session) but less well matched to *spatial* calibration
error, where the bias is a fixed function of 3D position that repeats every time the
person returns to the same region of the scene. The filter may track the spatial variation
as apparent temporal drift, but it will converge slowly and may confuse pose with bias when
the subject spends little time in any given region. For the fine-motion / interaction use
cases above, Approaches 3 and 4 are better targeted.

**Relationship to velocity mode:** Velocity mode is the limiting case where `sigma_bias → 0`
and the bias is essentially constant — the difference operation eliminates it exactly.
Per-camera bias states are the online, adaptive generalisation of this.

**Suggested experiment:** Start with a single "bad" camera that is known to have poor
calibration. Add its 2D bias as state, track NIS before and after, and inspect whether the
estimated bias matches the visual reprojection residuals.

---

## Approach 2 — Robust measurement likelihood

**Idea:** Replace the Gaussian measurement likelihood with a heavier-tailed distribution
that is less sensitive to occasional large calibration errors. No state changes needed.

### Student-t likelihood

Replace `noise ~ N(0, sigma²)` with `noise ~ t_ν(0, sigma²)`. For ν = 3–5 degrees of
freedom the tails are significantly heavier than Gaussian, suppressing the influence of
outlier observations without discarding them entirely.

In a UKF context the standard approach is to scale the measurement covariance adaptively
per update step. At each step, after computing the innovation `v_i` for observation `i`,
scale the R matrix entry:

```
R_ii ← R_ii * (ν + 1) / (ν + v_i² / R_ii)
```

This is the M-step of the variational Bayes EM update for a Student-t observation model
(Agamennoni et al. 2011, "Robust Inference for State Estimation"). Iterate the update
2–3 times per timestep for convergence. The result is that observations with large
innovations contribute less than they would under the Gaussian model.

### Huber loss (simpler)

Use a Huber-style covariance inflation: observations with Mahalanobis distance `d_i > k`
(a tunable threshold, e.g. `k = 1.5`) have their individual R entry scaled by `d_i / k`.
This is easier to implement than the Student-t EM loop and produces similar behaviour.

```cpp
for each observation i:
    d_i = |innovation_i| / noise_std_i
    if d_i > huber_k:
        noise_std_i *= d_i / huber_k    // inflate R for this observation
```

This is distinct from the existing Mahalanobis gate (which discards outliers entirely at
a hard threshold). Huber keeps the observation but down-weights it — useful when the
outlier contains real information about slow drift.

**Pros:** Requires no state augmentation; works within the existing UKF update loop;
easy to add as an optional config flag.

**Cons:** Does not *model* the calibration error — just reduces its influence. Does not
improve estimates in the long run; cannot tell a calibration error from a genuine outlier.

**Suggested experiment:** Set `outlier_threshold` to a very high value (effectively
disable hard gating), enable Huber weighting, and compare NIS vs. the current approach
on a session with known calibration problems.

---

## Approach 3 — Gaussian Process offline calibration correction

**Idea:** After a full tracking pass, compute per-camera signed residuals as a function of
the marker's 3D world position. Fit a Gaussian Process (GP) to these residuals and use the
posterior mean as a per-camera calibration correction map, then re-run the tracker.

### Offline pipeline

```
1. Run tracker (existing UKF, no change).
2. From tracking_obs_results, extract per-step (camera_id, marker_id, actual_x, actual_y,
   pred_x, pred_y) for inliers only.
3. For each camera k, build dataset {(3D_pos_i, residual_i)} where
       3D_pos_i = triangulated position of marker at that step (from smoothed state),
       residual_i = (actual_x - pred_x, actual_y - pred_y).
4. Fit two independent GPs (one for x, one for y residual) using a Matérn-3/2 kernel
   over 3D world coordinates.  Length-scale ≈ 0.3–1.0 m (tune by cross-validation).
5. On re-run: query GP(3D_pos_i) to predict expected residual for each observation, and
   subtract it from the observed pixel before handing to the UKF. Also use GP posterior
   variance to inflate calib_noise_std locally.
```

### Practical notes

- **Input for GP:** Use smoothed 3D positions from the RTS pass (already implemented) to
  avoid contamination of the calibration map by pre-convergence tracking errors.
- **GP library:** `GPyTorch` (PyTorch backend) or `scikit-learn` `GaussianProcessRegressor`
  for a simpler implementation. The input is 3D so a sparse GP (e.g. inducing points) may
  be needed if the session is long.
- **Iteration:** One correction pass is usually sufficient; two passes can refine if the
  first correction shifts the 3D trajectory enough to change the residuals.
- **Scope:** Correction is per camera, per run — not a permanent calibration fix. To
  improve the calibration itself, feed the GP mean back into the extrinsic refinement step.

**Pros:** Captures the full spatial structure of the calibration error; interpretable
(the GP mean map is a calibration error heatmap); can be visualised in the MCP diagnostic
server.

**Cons:** Offline — requires a first tracking pass; adds a GP fitting dependency; requires
enough observations spread across the scene to fit the GP (may be sparse for short sessions
or limited range of motion).

---

## Approach 4 — Relative keypoint measurements

**Idea:** Instead of giving the filter the absolute pixel coordinate of keypoint `c`, give
it the pixel coordinate of `c` **relative to its anatomical parent keypoint** `p` in the
same camera frame. Because calibration errors are approximately the same for nearby image
points, the bias nearly cancels in the difference.

This is the spatial analog of velocity mode: velocity mode differences successive frames
(cancelling temporal bias); relative mode differences spatially nearby keypoints in the
same frame (cancelling spatial bias).

### Measurement model

For a pair (parent marker `p`, child marker `c`) both visible in camera `k`:

```
z = pixel(c, cam_k) - pixel(p, cam_k)        [2D observed difference]

h(x) = project(c, cam_k, x) - project(p, cam_k, x)   [predicted difference]
```

If the camera bias is approximately constant across the image region containing both
keypoints, it cancels:
```
z ≈ (project_true(c) + b_k) - (project_true(p) + b_k)
  = project_true(c) - project_true(p)
```

The noise of the difference is `sqrt(2) * sigma_kp` assuming independence — a penalty
of ×1.4 in noise, but immunity to the calibration bias.

The parent/child pairing should follow the skeleton's kinematic hierarchy (e.g. wrist
relative to elbow, elbow relative to shoulder) so that the two markers are physically
close in 3D and therefore close in image space, maximising bias cancellation.

### Implementation sketch (C++)

1. Add `MeasurementMode::RELATIVE` to the enum in `observation.hpp`.
2. Add `int ref_marker_id = -1` to `Observation` (the parent marker, analogous to
   `prev_position` for VELOCITY mode).
3. In `Tracker::_prepare_observations()`: for each (camera, child marker) where both
   the child and its skeleton parent are observed with sufficient confidence, construct a
   RELATIVE observation. Also keep the absolute POSITION observation — they provide
   complementary information.
4. In `UKF::update()`, add a `ref_projections` map (like the existing `prev_projections`)
   computed from the *current* sigma-point state (not the previous frame):
   ```cpp
   // Inside the sigma-point loop:
   auto ref_proj = project(sigma_points[i], obs.ref_marker_id, camera);
   predicted[i] = project(sigma_points[i], obs.marker_id, camera) - ref_proj;
   ```
5. The reference projection must use the same sigma point as the child — this ensures the
   innovation covariance captures their correlation correctly.

### Skeleton parent map

The skeleton YAML already encodes the kinematic parent of each joint. Extend
`SkeletonLayout` or the observation builder to expose a `parent_marker_id(marker_id) →
int` lookup for use in step 3.

### Variant A — Parent-child pairs (kinematic neighbours)

The basic case described above: pair each keypoint with its direct parent in the skeleton
hierarchy. Both keypoints are guaranteed to be physically close in 3D.

Relative mode should be selective:
- Only when both parent and child confidence are above a threshold (e.g. 0.5).
- Only for kinematic neighbours (one joint apart in the skeleton hierarchy).
- Not for the root marker (no parent).
- A simple pixel-distance guard: only use RELATIVE if the expected image-distance between
  parent and child (from the FK prior) is less than ~100 px, ensuring the calibration bias
  at the two points is similar.

### Variant B — Spatially-close, hierarchically-distant pairs

A more powerful extension: find pairs of keypoints that are **close in image space** but
**far apart in the skeleton hierarchy**. When such a pair appears in the same camera frame,
their pixel difference is largely free of calibration bias (both are in the same image
region) and carries a strong constraint on the relative pose of two body parts.

**Why this matters for the stated use cases:** When both hands are close together
(handling an object, clapping, two-person contact), the wrists of the two kinematic chains
are spatially near in the image but separated by the full length of both arms in the
skeleton. Adding `pixel(right_wrist) - pixel(left_wrist)` as a measurement tells the
filter: "these two points are close together in this camera." Under the Gaussian
calibration noise model the filter sees them as independently noisy, so they may drift
apart or cause model penetration. The relative measurement has noise `pose_noise * sqrt(2)`
(calibration bias cancels) and no calibration-error floor — exactly the regime that matters
for fine hand motion.

**Measurement model:** Identical to the parent-child case:

```
z     = pixel(a, cam_k) - pixel(b, cam_k)
h(x)  = project(a, cam_k, x) - project(b, cam_k, x)
sigma = pose_noise_std * sqrt(2)
```

**Pair selection:** At each frame, for each camera:
1. Compute projected positions of all visible keypoints from the FK prior.
2. Find candidate pairs with image distance < threshold (e.g. 80 px) and skeleton
   hierarchy distance > 2 (not parent/child/grandchild — those are already handled by
   Variant A).
3. Rank candidates by a score: `image_closeness / skeleton_distance` — favouring pairs
   that are very close in the image but very far in the skeleton.
4. Take the top N pairs (e.g. N = 10) to bound the number of extra measurements per frame.

**Implementation note:** Pair selection runs on the FK prior, not the observations, so it
is computed once per frame before the UKF update. The same `MeasurementMode::RELATIVE`
infrastructure used for Variant A applies; the only difference is that `ref_marker_id`
points to an arbitrary marker rather than the skeleton parent.

### Pros and cons

**Pros:**
- Cancels per-camera calibration biases for any spatially-close pair, not just limb segments
- Directly constrains relative pose of distant body parts when they happen to be close —
  the primary mechanism for preventing model penetration and resolving hand interactions
- Works online, no offline pass needed; drops gracefully when either keypoint is occluded
- Noise floor is `pose_noise * sqrt(2)` with no calibration-error contribution

**Cons:**
- Only cancels *approximately constant* calibration bias within the inter-keypoint image
  region; cancellation degrades for pairs far apart in the image
- The number of candidate pairs can be large; a selection strategy is needed to keep the
  measurement vector manageable
- Noise increases by `sqrt(2)` vs. absolute observations — tune `pose_noise_std` and
  `calib_noise_std` together
- The filter must account for correlation between child/parent projections through the
  sigma-point cross-covariance (handled automatically if reference uses the same sigma point)

---

## Recommended exploration sequence

1. **Diagnose first.** After any tracking run, plot per-camera signed residuals
   (`actual - predicted` pixel positions) from `tracking_obs_results.obs_blob` as a
   function of 3D marker position (from the smoothed state). If residuals show a clear
   spatial pattern, calibration error dominates. If they look like white noise, the
   current model is already adequate.

2. **Try Approach 4 Variant A (parent-child relative)** for hand/wrist/finger markers —
   these are most affected by calibration error relative to limb length. Small
   implementation on top of existing VELOCITY mode infrastructure. Compare NIS and visual
   finger tracking quality before and after.

3. **Try Approach 4 Variant B (spatially-close pairs)** once Variant A is in place. The
   same `MeasurementMode::RELATIVE` machinery is reused; only the pair-selection logic
   is new. Start with two-hand interaction sequences where penetration or separation is
   visible, and check whether the constraint prevents it.

4. **Try Approach 2 (robust likelihood)** — one afternoon of work, no architecture
   changes. Useful as a complement to Approach 4 for handling occasional large outliers
   that survive the Mahalanobis gate.

5. **Try Approach 1 (bias states)** only if diagnostics show clear *temporal* drift in
   residuals (e.g. a camera bumped mid-session). For the primary use case of spatially-
   fixed calibration error it is less well suited than Approaches 3 and 4.

6. **Use Approach 3 (GP correction)** as an offline refinement for long sessions where
   the subject covers a wide range of the scene and the calibration error has clear spatial
   structure. Use the GP posterior as input to an extrinsic refinement step if possible.
