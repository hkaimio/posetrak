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

The existing `velocity_measurement_noise_std` / velocity mode already addresses the
degenerate case of a *constant* per-camera bias: differencing successive frames cancels the
bias exactly. The goal here is to handle the more general, spatially-varying case.

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

### When to use relative observations

Relative mode should be selective:
- Only when both parent and child confidence are above a threshold (e.g. 0.5).
- Only for kinematic neighbours (one joint apart in the skeleton hierarchy).
- Not for the root marker (no parent) and not across body segments where the markers are
  far apart in the image (bias cancellation degrades with image distance).

A simple heuristic: use RELATIVE if the expected image-distance between parent and child
(from the FK prior) is less than some fraction of the image width, e.g. < 100 px.

### Pros and cons

**Pros:**
- Cancels per-camera constant biases *without* augmenting the state vector
- Works online, no offline pass needed
- Conceptually simple — the existing VELOCITY mode infrastructure is a template
- Naturally handles hands and fine limb segments (where calibration error is large
  relative to limb length)

**Cons:**
- Only cancels *constant* (not spatially-varying) calibration biases within the
  inter-keypoint image region; long limbs see less cancellation
- Requires both parent and child to be visible; drops gracefully when parent is occluded
  (fall back to absolute POSITION observation)
- Noise increases by `sqrt(2)` — may require lowering `calib_noise_std` to compensate
- Adds coupling between adjacent marker updates; the filter must correctly account for
  the correlation between child and parent projection through the cross-covariance

---

## Recommended exploration sequence

1. **Diagnose first.** After any tracking run, plot per-camera signed residuals
   (`actual - predicted` pixel positions) from `tracking_obs_results.obs_blob` as a
   function of 3D marker position (from the smoothed state). If residuals show a clear
   spatial pattern, calibration error dominates. If they look like white noise, the
   current model is already adequate.

2. **Try Approach 2 (robust likelihood)** — one afternoon of work, no architecture
   changes. Does it reduce the NIS variance on sessions with known calibration problems?

3. **Try Approach 4 (relative keypoints)** for hand/wrist markers first — these are
   most affected by calibration error relative to limb length. Compare NIS for
   wrist/finger markers before and after.

4. **Try Approach 1 (bias states)** if residual diagnostics show slow temporal drift
   in the residuals, suggesting a poorly constrained camera.

5. **Use Approach 3 (GP correction)** as an offline refinement step for sessions where
   the calibration is known to be poor and re-calibration is not an option.
