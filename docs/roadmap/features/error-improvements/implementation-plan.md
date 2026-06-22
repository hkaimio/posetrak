# Implementation plan: measurement error model improvements

**Feature branch:** `feature/measurement-error-model`
**Base branch:** `main`

Implements the two design briefs in this folder. Phase 1 is a prerequisite for all later
phases. Phases 2–4 can be developed independently once Phase 1 is merged, but should be
PRed in order so each builds on a clean base.

---

## Phase 1 — Split measurement noise into `ep` and `ec`

*Design brief: `design-crop-scale-noise.md`*
*Estimated effort: 1–2 days*

This is the prerequisite for everything else. It makes `pose_noise_std` (`ep`) a
first-class parameter, which Phase 3 and 4 rely on for correct noise modelling of relative
measurements.

### Deliverables

**C++ — `include/posetrak/core/observation.hpp`**
- Add `double crop_scale = 1.0` field to `Observation`.
- Add two-argument `measurement_noise_std(double ep, double ec)` overload.
- Keep existing single-argument overload mapping to `measurement_noise_std(0.0, base)`.

**C++ — `include/posetrak/core/config.hpp` + `src/core/config.cpp`**
- Replace `measurement_noise_std` with `pose_noise_std` (default 0.0) and
  `calib_noise_std` (default 5.0) in `TrackerConfig`.
- TOML parser: read new keys; fall back to `measurement_noise_std` for old configs.
- Propagate through `HierarchicalConfig` child configs the same way.

**C++ — `src/db/session_reader.cpp`**
- Extend `load_observations()` SQL to also select
  `COALESCE(noise_scale, 1.0) AS crop_scale` from `pose_observations`.
- Set `obs.crop_scale` for every keypoint in that frame row.

**C++ — `include/posetrak/filters/ukf.hpp` + `src/filters/ukf.cpp`**
- Change `UKF::update()` signature: replace single `measurement_noise_std` parameter with
  `double pose_noise_std, double calib_noise_std`.
- Update R-assembly block to call the two-argument `measurement_noise_std()`.

**C++ — `src/filters/subset_ukf.cpp`**
- Apply same signature change to `SubsetUKF::update()` wrapper.

**C++ — `src/tracking/tracker.cpp`**
- Forward `config_.pose_noise_std` and `config_.calib_noise_std` to all `ukf_->update()`
  calls.

**Python UI**
- `app/pose/run_tracker.py`: add `_pose_noise` spin box (default 0.0); write
  `pose_noise_std` to config on run.
- `app/ui/content_panels.py`: update config summary to show both terms.
- `app/mcp/tools/runs.py`: update `describe_config` output.

### Tests

- Unit test: `Observation::measurement_noise_std(ep, ec)` with known `crop_scale` values.
- Regression: existing tracker test configs (which use `measurement_noise_std`) continue
  to produce identical output (backward-compat path maps to `ec` only).

### Acceptance criteria

- All existing C++ unit tests pass unchanged.
- Tracker run with `pose_noise_std=0, calib_noise_std=X` produces the same result as the
  same run with the old `measurement_noise_std=X`.
- Tracker run with `pose_noise_std=4, calib_noise_std=3` produces lower NIS variance than
  either parameter alone on a session where crop scale varies significantly across cameras.

---

## Phase 2 — Robust measurement likelihood (Huber)

*Design brief: `design-calibration-error.md` — Approach 2*
*Estimated effort: 0.5–1 day*
*Depends on: Phase 1 (uses `calib_noise_std` as the base for Huber threshold scaling)*

Quickest architectural change; validates whether calibration-like outliers are limiting
tracking before investing in more complex approaches.

### Deliverables

**C++ — `include/posetrak/core/config.hpp` + `src/core/config.cpp`**
- Add `double huber_k = 0.0` to `TrackerConfig` (0 = disabled, i.e. current behaviour).
- Parse from TOML `[tracking] huber_k = 1.5`.

**C++ — `src/filters/ukf.cpp`**
- In `UKF::update()`, after computing the initial innovation for each observation, apply
  Huber inflation if `huber_k > 0`:
  ```
  d_i = |innovation_i| / noise_std_i
  if d_i > huber_k: noise_std_i *= d_i / huber_k
  ```
  Re-assemble R with inflated values. This is a single pass after the initial sigma-point
  spread; no iteration needed for the Huber variant.

**Python UI**
- `run_tracker.py`: add `_huber_k` spin box (default 0.0, labelled "Huber k (0=off)").
- `content_panels.py`: show `huber_k` in config summary if non-zero.

### Tests

- Unit test: with a single observation whose innovation is exactly `huber_k * noise_std`,
  verify R entry is unchanged; at `2 * huber_k * noise_std`, verify it doubles.
- Integration: run on a session with known large residuals; compare NIS distribution with
  and without Huber.

### Acceptance criteria

- `huber_k = 0` produces bitwise-identical output to Phase 1 baseline.
- On a session with calibration outliers, NIS median is lower and NIS tail (> 95th
  percentile) is reduced with `huber_k = 1.5`.

---

## Phase 3 — Relative keypoint measurements: parent-child pairs (Variant A)

*Design brief: `design-calibration-error.md` — Approach 4, Variant A*
*Estimated effort: 2–3 days*
*Depends on: Phase 1 (`pose_noise_std` must be a standalone parameter)*

Introduces `MeasurementMode::RELATIVE` following the same pattern as the existing
`MeasurementMode::VELOCITY`. Targets hand/wrist/finger tracking quality.

### Deliverables

**C++ — `include/posetrak/core/observation.hpp`**
- Add `RELATIVE` to `MeasurementMode` enum.
- Add `int ref_marker_id = -1` field to `Observation` (parent marker for RELATIVE mode).

**C++ — `include/posetrak/db/session_reader.hpp` / `src/db/session_reader.cpp`**
*(or `src/tracking/tracker.cpp` if observation building happens there)*
- In `load_observations()`, after building the standard `POSITION` observations, add a
  second pass: for each `(camera, child marker)` where both the child and its skeleton
  parent are present in the same frame with confidence ≥ threshold (e.g. 0.5), emit a
  `RELATIVE` observation with `ref_marker_id` set to the parent marker index.
- Keep the absolute `POSITION` observation alongside — they provide complementary info.
- Noise for RELATIVE observations: `pose_noise_std * sqrt(2)` stored as
  `noise_std_override` (so it ignores `calib_noise_std` which has cancelled).

**C++ — `include/posetrak/core/skeleton_layout.hpp`**
- Expose `parent_marker_id(int marker_id) → int` lookup (returns -1 for root markers).
  Build from the skeleton joint parent chain during `SkeletonLayout` construction.

**C++ — `src/filters/ukf.cpp`**
- In `UKF::update()`, before the sigma-point loop, build a
  `ref_projections[sigma_idx][marker_id]` map for RELATIVE observations, computed from
  the *current* sigma point (not the previous frame — this is the key difference from
  VELOCITY mode).
- In `predict_measurements()`, for `RELATIVE` mode:
  ```cpp
  auto ref = project(sigma_pt, obs.ref_marker_id, camera);
  auto child = project(sigma_pt, obs.marker_id, camera);
  predicted = child - ref;   // NaN if either is behind camera
  ```
- Observed innovation for RELATIVE: `z = pixel(child) - pixel(parent)` — computed in
  `tracker.cpp` when building the observation, stored as `obs.position`.

**C++ — `include/posetrak/core/config.hpp` + `src/core/config.cpp`**
- Add `double relative_min_confidence = 0.5` config key.
- Add `bool use_relative_observations = false` flag (off by default for safety).

### Tests

- Unit test: with two markers projected at known pixel positions, verify that
  `predict_measurements()` returns the correct difference for `RELATIVE` mode.
- Unit test: `parent_marker_id()` returns correct values for skeleton fixtures.
- Integration: run with `use_relative_observations = true` on a wrist-tracking sequence;
  compare NIS for wrist/hand markers before and after.

### Acceptance criteria

- `use_relative_observations = false` is bitwise-identical to Phase 2 baseline.
- With the flag enabled, NIS for finger/wrist markers is lower than without.
- No regression in NIS for shoulder/hip markers (large joints should be unaffected).

---

## Phase 4 — Relative keypoint measurements: spatially-close pairs (Variant B)

*Design brief: `design-calibration-error.md` — Approach 4, Variant B*
*Estimated effort: 2–3 days*
*Depends on: Phase 3 (reuses `MeasurementMode::RELATIVE` infrastructure)*

Extends Phase 3 to pairs that are close in image space but distant in the skeleton
hierarchy. Primary target: two-hand interactions, hand–object contact, model penetration.

### Deliverables

**C++ — `src/tracking/tracker.cpp`** (or observation builder)
- Per frame, per camera: compute projected positions of all visible markers from FK prior.
- Find candidate pairs: image distance < `cross_pair_max_px`, skeleton distance > 2 hops.
- Score: `image_closeness / skeleton_distance`; take top `cross_pair_max_n` pairs.
- Emit `RELATIVE` observations for selected pairs (uses same machinery as Phase 3).
- Noise: `pose_noise_std * sqrt(2)` via `noise_std_override`.

**C++ — `include/posetrak/core/config.hpp` + `src/core/config.cpp`**
- Add `double cross_pair_max_px = 0.0` (0 = disabled).
- Add `int cross_pair_max_n = 10`.

**C++ — skeleton distance helper**
- `SkeletonLayout::hierarchy_distance(int marker_a, int marker_b) → int` — number of
  edges between two markers in the kinematic tree. Cache on first call or precompute at
  layout construction time.

### Tests

- Unit test: `hierarchy_distance()` returns correct values for known skeleton topology.
- Unit test: pair selection correctly excludes parent-child pairs (distance ≤ 2) and
  pairs beyond the pixel threshold.
- Integration: run on a two-hand interaction sequence; check whether the constraint
  reduces inter-wrist distance error and eliminates model penetration frames.

### Acceptance criteria

- `cross_pair_max_px = 0` is bitwise-identical to Phase 3 baseline.
- On a sequence with hands in contact, the number of frames with model penetration is
  reduced.
- No regression in wrist/shoulder NIS on sequences where hands are far apart (no cross
  pairs should be selected, so behaviour is identical to Phase 3).

---

## Deferred — Approaches 1 and 3

**Approach 1 (per-camera bias states):** Revisit if post-Phase-4 residual analysis shows
clear temporal drift in per-camera residuals (suggesting a bumped or thermally drifting
camera). Not well matched to spatially-fixed calibration error.

**Approach 3 (GP offline correction):** Revisit for long sessions with wide subject motion
range once Phase 4 is validated. Requires `GPyTorch` or `scikit-learn` dependency and a
Marimo analysis notebook for the offline correction workflow. The GP posterior can also
serve as input to an extrinsic refinement step.

---

## Branch and PR strategy

```
main
 └── feature/measurement-error-model
       ├── phase-1/split-noise-model          → PR 1: prerequisite, standalone
       ├── phase-2/huber-robust-likelihood    → PR 2: quick win, minimal risk
       ├── phase-3/relative-parent-child      → PR 3: core relative mode
       └── phase-4/relative-cross-pairs       → PR 4: interaction constraint
```

Each phase is a self-contained PR against `feature/measurement-error-model`. The feature
branch is merged to `main` after Phase 4 (or after Phase 3 if Phase 4 is deferred).

Phases 2, 3, and 4 each have a config flag that defaults to the Phase 1 baseline behaviour,
so the feature branch can be used for normal tracking throughout development without
enabling experimental features.
