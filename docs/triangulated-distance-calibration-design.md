# Skeleton Scaling via Triangulated Marker Distances — Design

**Status:** Proposed replacement for Option A (prismatic state DOFs)
**Motivation:** Option A (prismatic joints in UKF state) is confirmed unstable in practice.
This document specifies an alternative calibration algorithm that avoids modifying the
filter state entirely.

---

## 1. Why Option A Failed

The prismatic-joint approach augments the UKF state vector with one extra position DOF
and one velocity DOF per calibrated bone. In practice this causes two compounding
problems:

**Ill-conditioned covariance.** Prismatic DOFs have near-zero process noise (σ ≈
0.1 mm/√s). After accumulating frames the posterior covariance has min eigenvalue ~1×10⁻⁸
(velocity-limit-clamped DOFs) and max eigenvalue ~360 (unconstrained velocity DOFs),
giving condition number ~3.6×10¹⁰. The RTS smoother gain G = D · P_prior⁻¹ has
spectral norm 2–20, producing exponential backward divergence. See
`docs/rts-smoother-calibration-failure.md` for full diagnostics.

**Scale–pose degeneracy in 2D.** The UKF measurement model reprojects 3D marker
predictions onto 2D image coordinates. A physically impossible bone length (e.g., a
forearm that is 30 % too long but angled away from cameras) can fit 2D observations
equally well or better than the correct length. Short bones with sparse marker coverage
(clavicles) diverge most severely because the 2D reprojection residual provides almost
no leverage on the true 3D length. Camera extrinsics systematic errors amplify this
further.

**Root cause in one sentence.** Bone length is a 3D concept; the UKF only ever sees 2D
projections, so it cannot reliably infer lengths from reprojection residuals alone.

---

## 2. Proposed Algorithm: Post-Hoc Triangulated Distance Calibration

### 2.1 Overview

1. Run UKF tracking on the calibration sequence with the **default skeleton** (no
   prismatic DOFs, no extra state). Track all frames normally.
2. After each frame's UKF update, **triangulate** each marker that has ≥ N inlier
   camera observations (N is configurable; see §2.4).
3. For each calibration scale group, compute a **per-frame scale estimate** from the
   ratio of the triangulated 3D distance to the model-predicted 3D distance for the
   marker pair spanning that group (see §2.2–2.3).
4. After all frames, aggregate per-frame estimates across all joints in the group: take
   the **median** over all valid (frame, joint) samples, discarding samples that fail
   quality criteria (§2.4). All joints in a group receive the same final scale factor.
5. Write the calibrated YAML: multiply each calibrated joint's `offset` vector by the
   group's scale factor.
6. Optionally iterate (§2.5): re-run tracking with the updated skeleton and repeat
   steps 2–5 until scale changes are below tolerance.

No changes to the UKF, process model, sigma-point generator, RTS smoother, or
SkeletonLayout are required.

### 2.2 Per-Frame Scale Estimator

For each (joint, frame) pair, define a marker pair `(A, B)` such that `A` is attached
near the **proximal end** of the bone and `B` is attached near the **distal end**. Then
for frame `t`:

```
ŝ(t) = |p_A_tri(t) − p_B_tri(t)| / |p_A_model(t) − p_B_model(t)|
```

where:
- `p_A_tri(t)`, `p_B_tri(t)` — 3D positions from triangulation of inlier observations
- `p_A_model(t)`, `p_B_model(t)` — 3D marker positions from FK evaluated at the UKF
  posterior state `x̂_{t|t}`

The ratio is the multiplicative factor by which the current model bone length must
change to match the observed 3D geometry. It is independent of joint angles: the
division cancels the pose-dependent rotation of the bone.

When one endpoint is a **model joint position** rather than a triangulated marker (see
§2.3), `p_B_tri(t) ≡ p_B_model(t)`, so the formula reduces to:

```
ŝ(t) = |p_A_tri(t) − p_parent_model(t)| / |p_A_model(t) − p_parent_model(t)|
```

This is valid because the parent joint position is determined by bones *other than* the
one being estimated, so there is no circular dependency.

### 2.3 Scale Groups and Shared Scale Factors

All joints listed within a single `scale_group` entry share one scale factor, estimated
by pooling per-frame samples from every joint in the group:

```
ŝ_group = median { ŝ_j(t) : all joints j in group, all valid frames t for joint j }
```

**Grouping choice determines what is shared.** Putting `shin.L` and `shin.R` in the
same group imposes a symmetric femur length estimate — sensible as the default
assumption. Putting them in separate groups gives independent left/right estimates —
appropriate when significant body asymmetry is known or suspected. The skeleton YAML
controls this; no code changes are needed to switch modes.

For well-observed symmetric pairs (e.g., upper arm), pooling L and R samples roughly
doubles the number of valid observations and reduces the median's variance.

### 2.4 Bone-to-Marker-Pair Mapping

The table below specifies the marker pair for each calibration bone. "Tri" means the
point must be triangulated from inlier observations. "Model joint" means the point is
taken from FK at the UKF posterior state (no triangulation required for that endpoint).
A frame is **skipped** if any required triangulated endpoint is unavailable.

| Scale group | Joint(s) | Proximal ref | Distal ref |
|---|---|---|---|
| `hip_socket` | `thigh.L`, `thigh.R` | `MRK-hip.L` / `MRK-hip.R` (tri) | Midpoint of both hip markers (tri)¹ |
| `femur` | `shin.L`, `shin.R` | `MRK-hip.L` / `MRK-hip.R` (tri) | `MRK-knee.L` / `MRK-knee.R` (tri) |
| `tibia` | `foot.L`, `foot.R` | `MRK-knee.L` / `MRK-knee.R` (tri) | `MRK-Ankle.L` / `MRK-Ankle.R` (tri) |
| `spine` | `spine1`, `spine2` (chain²) | Midpoint `MRK-hip.L`, `MRK-hip.R` (tri) | Midpoint `MRK-shoulder.L`, `MRK-shoulder.R` (tri) |
| `shoulder_reach` | `shoulder.L`, `shoulder.R` | `MRK-shoulder.L` / `MRK-shoulder.R` (tri) | Model `spine2` joint pos |
| `upper_arm` | `upper_arm.L`, `upper_arm.R` | `MRK-shoulder.L` / `MRK-shoulder.R` (tri) | `MRK-elbow.L` / `MRK-elbow.R` (tri) |
| `forearm` | `forearm.L`, `forearm.R` | `MRK-elbow.L` / `MRK-elbow.R` (tri) | `MRK-wrist.L` / `MRK-wrist.R` (tri) |

**Notes:**

¹ **Hip socket.** The `thigh.L/R` offsets encode a 3D vector from the body centre
(`hips` root) to each hip socket. The body centre is not directly marked, so it is
estimated as `(tri(MRK-hip.L) + tri(MRK-hip.R)) / 2`. This requires **both** hip
markers to be triangulated in the same frame. Frames where only one hip marker is
available are discarded for this group entirely — experience shows such frames tend to
be problematic in other ways (partial occlusion, detection noise). No fallback to model
joint position is used.

² **Spine chain.** Spine1 and spine2 form a two-joint chain with no mid-spine marker.
A single scale factor is estimated from the hip-midpoint to shoulder-midpoint distance
and applied uniformly to both joints. Independent scaling of the two segments is not
observable without a thoracic or sternum marker.

### 2.5 Frame Selection and Quality Filtering

A (frame, joint) sample contributes to the group's estimate only if all of the
following pass:

1. **Inlier count.** Every triangulated endpoint must have ≥ `min_inlier_cameras`
   inlier camera observations. The UKF Mahalanobis gating step (already applied during
   tracking) is the primary outlier rejection mechanism: only cameras that pass the
   Mahalanobis chi-squared test are counted as inliers. `min_inlier_cameras` is a
   configurable parameter (suggested default: 2).

2. **Triangulation condition number.** The DLT normal-matrix condition number for each
   triangulated endpoint must be below a threshold (suggested: `cond < 200`). This
   catches degenerate viewing geometries (near-collinear camera rays) that survive the
   inlier count check but produce poor depth estimates.

3. **Sanity clamp.** The raw scale estimate `ŝ(t) ∈ [0.5, 2.0]`. Values outside this
   range indicate a bad triangulation or severe tracking failure and are discarded.

### 2.6 Aggregation

```
ŝ_group = median { ŝ_j(t) : joint j in group, frame t passes §2.5 for joint j }
```

The median is robust to outlier frames from occlusion recovery, marker swaps, or poor
triangulation geometry. For well-observed groups the median and precision-weighted mean
converge to the same value.

A trimmed mean (discard bottom and top 10 % of samples, average the rest) is an
acceptable alternative for groups with few valid samples where the median is unstable.

### 2.7 Convergence Reporting

Report per group:

| Status | Criterion |
|---|---|
| CONVERGED | ≥ 240 valid samples AND IQR < 0.02 |
| UNCERTAIN | ≥ 240 valid samples AND 0.02 ≤ IQR < 0.10 |
| NOT_OBSERVABLE | < 240 valid samples OR IQR ≥ 0.10 |

IQR is the interquartile range of the per-sample `ŝ(t)` values before aggregation. A
small IQR reflects consistent triangulation across many frames; a large IQR indicates
either poor observability or conflicting estimates between left and right sides.

NOT_OBSERVABLE groups retain their default offsets unchanged.

If a group contains bilateral pairs (`.L` and `.R` joints) that have been pooled, also
compute separate medians per side and emit a warning if they diverge by more than 5 %:
`|ŝ.L − ŝ.R| / ((ŝ.L + ŝ.R) / 2) > 0.05`. The pooled median is still written to the
YAML; the warning informs the user to consider splitting the group into independent
left/right groups.

### 2.8 Iterative Refinement

A single pass captures the dominant systematic bone-length error. Iteration is optional
but improves accuracy when initial bone lengths are far from truth (> 20 % error),
because first-pass pose estimates are biased and the triangulated distances are slightly
inconsistent with the model.

```
k = 0; skeleton = default
loop:
    run UKF on calibration sequence with skeleton_k
    compute ŝ_group for all groups
    apply scale: offset_j ← ŝ_group * offset_j  for each joint j in group
    skeleton_{k+1} = updated skeleton
    if max_group |ŝ_group − 1| < 0.005: break   # < 5 mm on a 1 m bone
    if k >= 5: break
    k += 1
output: skeleton_{k+1}
```

In practice 1–2 iterations are expected to suffice.

---

## 3. Comparison to Previous Approaches

| Property | Option A (prismatic in state) | Option B (alternating LS) | This proposal |
|---|---|---|---|
| Filter changes required | Major (new joint type, state augmentation) | None | None |
| Smoother stability | Fails (documented) | N/A | N/A |
| Scale constraint space | 2D reprojection | 3D (via FK + q_t) | 3D (via triangulation) |
| Model dependence of scale estimate | High (UKF must resolve scale+pose jointly) | Medium (uses smoothed q_t) | Low (triangulation is model-independent) |
| Camera systematic error sensitivity | High (extrinsics error → 2D bias → scale bias) | Medium | Medium (same triangulation used in tracker) |
| Robustness to short sequences | Poor (convergence requires many frames) | Medium | Good (median of independent frames) |
| Robustness to degenerate poses | Poor (2D degeneracy amplified) | Medium | Good (frame-level filtering) |
| Implementation complexity | High | Medium | Low (post-processing, ~200 lines Python) |
| Iterations needed | 1 (online) | 3–10 | 1–2 |

This proposal is most similar to Option B but replaces the linear least-squares inner
loop with a direct distance ratio estimator. The key advantage over Option B is that the
reference distances come from triangulation (model-independent) rather than from the
model's own FK (model-dependent). This breaks the scale–pose circular dependency that
makes Option B slow to converge when initial bone lengths are wrong.

---

## 4. Known Limitations

### 4.1 Camera extrinsics systematic errors

Triangulation accuracy depends on extrinsics quality. Systematic extrinsics errors
(translation bias, small rotation errors) produce biased triangulated positions. The
effect on the **ratio** `|p_A_tri − p_B_tri| / |p_A_model − p_B_model|` is partially
attenuated because both endpoints are triangulated using the same cameras, so a uniform
spatial bias partially cancels. Non-uniform bias (e.g., different cameras observe
proximal vs. distal markers for the same bone) does not cancel and remains a systematic
error source.

**Mitigation:** Prefer frames where both markers of a pair are seen by the same set of
cameras (check inlier camera overlap before accepting a frame).

### 4.2 Marker attachment offset absorption

The scale estimator measures the distance between marker positions, not between joint
centres. Marker attachment offsets contribute to the measured distance. If attachment
offsets are wrong, the estimated bone length absorbs part of that error.

For anatomical-landmark markers (hip, knee, ankle, shoulder, elbow, wrist), attachment
offsets are typically 1–3 cm against bone lengths of 30–50 cm, giving a relative bias
< 10 %. This is acceptable for initial calibration. A future refinement would calibrate
marker attachment offsets separately (static T-pose) before running bone-length
calibration.

### 4.3 Spine chain — single shared scale factor

Spine1 and spine2 are estimated jointly. A person with an unusually long lumbar and
short thoracic segment will have both scaled by the same average factor. This is a
practical limitation of not having a mid-torso marker; it could be resolved by adding a
sternum or xiphoid process marker.

### 4.4 Hip socket — body-centre estimated from markers

Using the midpoint of the two hip markers as the body centre introduces an error when
the pose is asymmetric (asymmetric poses cause the marker midpoint to shift away from
the true pelvis origin). Since frames where either marker is missing are already
discarded (§2.4), this reduces to a second-order pose-dependent bias that averages out
over a diverse motion sequence.

---

## 5. Implementation Plan

### 5.1 Phased approach: Python first, C++ for production

**Phase 1 — Python post-processor** (`scripts/calibrate_scale.py`).

The algorithm operates entirely on data already available from a completed tracking run:
the per-frame state vectors (`state_vectors.csv`), the raw observation JSON files, and
the camera calibration TOML. FK can be re-evaluated in Python (the existing notebook
infrastructure already does this). Triangulation requires ~50 lines of numpy DLT. This
phase requires **zero C++ changes** and can be developed and validated against real data
immediately.

**Phase 2 — C++ `posetrak scale` subcommand.**

For production use and iterative refinement, integrating the algorithm into the C++
binary is better: the C++ tracker is already fast and the TOML config loading,
triangulation, and FK are already implemented and tested there. The iterative loop
(re-run tracker → recompute scale → update skeleton → repeat) runs with one command
invocation and no file round-tripping. The Python prototype informs the C++
implementation directly; no algorithmic redesign is expected between phases.

### 5.2 Data sources from a tracking run

All required data is present in the standard tracker output:

**`tracking_results.csv`** — model marker 3D positions from FK evaluated at the UKF
posterior state. Columns: `frame`, `timestamp`, `marker_id`, `marker_name`, `x_3d`,
`y_3d`, `z_3d`, `is_visible`. These provide `p_A_model` and `p_B_model` in the scale
estimator formula. One row per marker per frame.

**`marker_projections.csv`** — per-camera 2D observations with Mahalanobis outlier
flags. Columns: `frame`, `timestamp`, `marker_id`, `marker_name`, `camera_id`,
`proj_x`, `proj_y`, `obs_x`, `obs_y`, `error_x`, `error_y`, `is_outlier`. Rows with
`is_outlier == false` are the inlier set used for triangulation. The count of inlier
rows per (frame, marker) pair is the inlier camera count checked against
`min_inlier_cameras`.

**`state_vectors.csv`** — full UKF posterior state (root pose + all joint angles) per
frame. Needed only for scale groups that reference a `_model_joint()` endpoint (e.g.,
the `spine2` joint position for the `shoulder_reach` group). For those cases, FK must
be re-evaluated in Python from the state vector to obtain the joint frame position.
Groups that reference only named markers do not require this file.

**Triangulation.** The calibration script triangulates each marker's 3D position from
its inlier `(obs_x, obs_y)` observations using the camera projection matrices from the
Pose2Sim TOML. This is a standard DLT solve (~50 lines of numpy); the condition number
of the normal matrix serves as the quality gate (`max_tri_cond`). No access to the raw
OpenPose JSON files is required — `marker_projections.csv` already contains the
filtered and synchronised 2D observations.

### 5.3 Python implementation outline

```
Input:
  --tracking-dir        directory containing tracking_results.csv,
                        marker_projections.csv, state_vectors.csv
  --cameras             Pose2Sim camera TOML
  --skeleton            reference skeleton YAML (with scale_groups)
  --output              output calibrated YAML path
  --min-inlier-cameras  minimum Mahalanobis inlier count for triangulation (default: 2)
  --max-tri-cond        triangulation condition number threshold (default: 200)
  --iterations          number of refinement iterations (default: 1)

Steps:
  1. Load skeleton (scale_groups), camera calibration
  2. Load tracking_results.csv  →  dict[frame][marker_name] = (x, y, z)
  3. Load marker_projections.csv  →  dict[frame][marker_name][camera_id] = (obs_x, obs_y)
     filtered to is_outlier == false
  4. (If any group uses _model_joint) load state_vectors.csv; build FK evaluator
  5. For each frame t:
     a. For each marker with ≥ min_inlier_cameras inlier cameras:
        triangulate 3D position; accept only if tri_cond < max_tri_cond
     b. (If needed) evaluate FK at state_t → joint frame positions
     c. For each scale group: for each joint in group, resolve endpoint positions
        (triangulated marker, or midpoint of two triangulated markers, or model joint);
        if both endpoints available, compute
        ŝ_j(t) = |p_A_tri − p_B_tri| / |p_A_model − p_B_model|
        and append to group sample list
  6. For each group: ŝ_group = median(samples); compute IQR and sample count
  7. Print convergence table (CONVERGED / UNCERTAIN / NOT_OBSERVABLE per group)
  8. Emit bilateral divergence warnings where applicable
  9. Write calibrated YAML: update offset_j ← ŝ_group * offset_j for each joint
```

### 5.4 Calibration bone specification in YAML

Reuse the `scale_groups` key from the original design, extended with `marker_pair`
fields. No `prismatic` joint type is needed. The `_midpoint(...)` and
`_model_joint(...)` pseudo-references are resolved at runtime by the calibration
script.

```yaml
scale_groups:
  - name: hip_socket
    description: "position of hip joints relative to pelvis origin"
    joints:
      - name: thigh.L
        marker_pair: [MRK-hip.L, _midpoint(MRK-hip.L, MRK-hip.R)]
      - name: thigh.R
        marker_pair: [MRK-hip.R, _midpoint(MRK-hip.L, MRK-hip.R)]

  - name: femur
    joints:
      - name: shin.L
        marker_pair: [MRK-hip.L, MRK-knee.L]
      - name: shin.R
        marker_pair: [MRK-hip.R, MRK-knee.R]

  - name: tibia
    joints:
      - name: foot.L
        marker_pair: [MRK-knee.L, MRK-Ankle.L]
      - name: foot.R
        marker_pair: [MRK-knee.R, MRK-Ankle.R]

  - name: spine
    # Single scale factor applied to both joints in this chain
    chain: [spine1, spine2]
    marker_pair:
      proximal: _midpoint(MRK-hip.L, MRK-hip.R)
      distal:   _midpoint(MRK-shoulder.L, MRK-shoulder.R)

  - name: shoulder_reach
    joints:
      - name: shoulder.L
        marker_pair: [MRK-shoulder.L, _model_joint(spine2)]
      - name: shoulder.R
        marker_pair: [MRK-shoulder.R, _model_joint(spine2)]

  - name: upper_arm
    joints:
      - name: upper_arm.L
        marker_pair: [MRK-shoulder.L, MRK-elbow.L]
      - name: upper_arm.R
        marker_pair: [MRK-shoulder.R, MRK-elbow.R]

  - name: forearm
    joints:
      - name: forearm.L
        marker_pair: [MRK-elbow.L, MRK-wrist.L]
      - name: forearm.R
        marker_pair: [MRK-elbow.R, MRK-wrist.R]
```

### 5.5 Definition of done

- `scripts/calibrate_scale.py` produces a calibrated YAML from an existing tracking run.
- The calibrated YAML is a drop-in replacement: same format, no `scale_groups` key,
  updated `offset` values only.
- Mean marker RMSE with the calibrated skeleton is lower than with the default skeleton
  on the calibration sequence.
- Per-group convergence table is printed with CONVERGED / UNCERTAIN / NOT_OBSERVABLE
  status, sample count, and IQR.

---

## 6. Open Questions

- **Minimum cameras for triangulation.** The default `min_inlier_cameras = 2` is the
  geometric minimum. With only 2 cameras and a short baseline, depth estimation is
  noisy. For spine and hip groups (longer chains, more sensitive to depth error),
  consider requiring `min_inlier_cameras = 3` as a group-level override in the YAML.

- **Relation to marker attachment offset calibration.** If both bone lengths and marker
  offsets are wrong, this algorithm partially absorbs marker offset errors into bone
  length estimates. Running a static-pose marker offset calibration first (T-pose +
  linear solve) would improve accuracy. The two calibrations could eventually be
  combined in a single script.
