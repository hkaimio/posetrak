# Open Issues

## Failing Tests — to fix after scaling feature (2026-03-01)

16 test assertions fail across 8 distinct root causes.  None were introduced by CP1;
all pre-date the prismatic joint work (confirmed by stashing CP1 changes and re-running).

---

### 1 — ~~`yaml-cpp` bad conversion when loading skeleton YAML with `scale_groups` / `depends_on`~~ **FIXED**

**Affected tests** (3 assertions):
- `test_skeleton_loader.cpp:22` — `load_skeleton_from_yaml("tests/data/simple_humanoid.yaml")`
- `test_tracker_integration.cpp:222`
- `test_triangulation.cpp:423`

**Root cause**: `src/io/skeleton_loader.cpp:73` did
`group_node["depends_on"].as<std::string>()`, but `simple_humanoid.yaml:174` stores it as a
YAML sequence (`depends_on: ["core"]`), not a scalar.  `yaml-cpp` throws `bad conversion`.

**Fix applied**: Parse `depends_on` as either a scalar or a sequence.  Changed
`group_dependencies` to `unordered_map<string, vector<string>>` and the parser now handles
both forms (scalar string and YAML sequence).

---

### 2 — `Camera::project()` returns `nullopt` for points that project outside image bounds

**Affected tests** (3 assertions): `test_camera.cpp:92, 138, 229`

**Root cause**: The camera has image size 640×480.  The test point `(1, 0.5, 2)` projects to
pixel `(720, 440)` — outside the image — so `project()` returns `nullopt`.  The tests were
written assuming `project()` does *not* clip to image bounds (test comment says "x_pixel = 720").

**Fix**: Remove the image-bounds check from `project()` (or add a separate
`project_unclamped()` overload); bounds checking is the caller's responsibility.

---

### 3 — Camera loader loads extrinsics with wrong position values

**Affected tests** (1 assertion): `test_camera_loader.cpp:60`

**Root cause**: Loaded `extrinsics.position[0] = +9.08` instead of expected `-4.37`.
Likely a frame-convention mismatch: the loader stores the raw TOML translation vector
instead of the camera position in world frame (`position = -R^T * t`).

**Fix**: Audit `src/io/camera_loader.cpp`; ensure the stored position is the camera
origin in world coordinates, consistent with what `Intrinsics`/`Extrinsics` documents.

---

### 4 — `Observation::measurement_noise_std()` ignores confidence

**Affected tests** (3 assertions): `test_observation.cpp:32, 38, 44`

**Root cause**: `measurement_noise_std(base_std)` returns `base_std` unchanged instead of
`base_std / confidence`.  The implementation is a stub that forgets to divide.

**Fix**: In `src/core/observation.cpp` (or wherever `measurement_noise_std` is defined):
```cpp
double Observation::measurement_noise_std(double base_std) const {
    double conf = std::max(confidence, 0.1);   // clamp to avoid div-by-zero
    return base_std / conf;
}
```

---

### 5 — `ConstantVelocityModel::propagate()` does not enforce joint limits or locked DOFs

**Affected tests** (2 assertions): `test_process_model.cpp:162, 296`

- `:162` — revolute joint with limits `[-1, 1]`; angle `0.95 + 0.1*0.1 = 1.05` should be
  clamped to `1.0` but the model returns `1.05`.
- `:296` — spherical joint with X/Y locked (`min == max == 0`); after propagation the
  locked DOFs should stay at `0` but one drifts to `0.2`.

**Root cause**: `propagate()` integrates position and velocity but does not call
`enforce_joint_limits()` (or equivalent) on the resulting state.

**Fix**: Call the limit-enforcement step at the end of `propagate()`, mirroring what the
UKF does after its prediction step.

---

### 6 — UKF `error_dim()` wrong for spherical joint with locked DOFs

**Affected tests** (1 assertion): `test_ukf.cpp:317`

**Scenario**: shoulder is `SPHERICAL` with X/Y locked (only Z active; `active_dof = 1`).
Test expects `error_dim() = 2*(6+3) = 18`, but UKF returns `2*(6+1) = 14`.

**Root cause**: UKF error space uses *active* DOFs only.  A state vector of size 3 is
allocated for the spherical joint (always), but only 1 active DOF contributes to error
space — so `error_dim = 14`.  The test was written under a different assumption (error
space = storage space).

**Fix options**:
- A. Update the test to reflect the correct active-DOF policy (`REQUIRE(ukf.error_dim() == 14)`).
- B. Change UKF error space to equal storage space (simpler but wastes sigma points on
  locked DOFs).

Option A is preferred (correct policy, save computation).

---

### 7 — UKF update fails with "Failed to compute eigenvalues for covariance conditioning"

**Affected tests** (1 assertion): `test_ukf_update.cpp:429`

**Root cause**: After a purely translational predict step with a very stiff covariance, the
innovation covariance `S = Pyy + R` becomes numerically ill-conditioned (near-singular
because marker reprojection residuals have ~zero spread across sigma points).
The covariance conditioning (likely a Cholesky or eigen-decomposition) fails.

**Fix**: Add a floor to the diagonal of `S` before decomposition, or detect and skip
the update when `S` is singular (return current state unchanged with a warning).

---

### 8 — `test_ukf_frame0_comparison` needs pre-generated Python debug fixture

**Affected tests** (4 assertions): `test_ukf_frame0_comparison.cpp:39, 94, 98, 128`

**Root cause**: Tests compare C++ UKF frame-0 output against reference data at
`tracking_tests/cpp-python-comparison/python_results/debug/frame_0000/all_observations.csv`
which does not exist in the repo.  These are golden-file comparison tests that require the
Python tracker to be run first to generate the fixture.

**Fix**: Either commit the fixture (if small and stable), or add a `[!shouldfail]` / skip
tag until the fixture generation script is documented and run in CI.

---

### 9 — `State::apply_error_update` quaternion multiplication order mismatch

**Affected tests** (1 assertion): `test_state.cpp:221`

**Root cause**: Test computes `expected_quat = delta_q * quat` (left-compose, global frame
update), but the implementation uses `quat * delta_q` (right-compose, body-frame update),
or vice-versa.  One of the two is wrong for the established convention.

**Fix**: Audit `State::apply_error_update` in `src/core/state.cpp` and decide the correct
convention (global vs body frame perturbation); update the test or implementation to match.

---

## Active

- **`Skeleton::active_dof()` does not count root DOFs** (2026-01-26)
  - The method still exists on `Skeleton` but hot-path code (UKF, sigma points, process model)
    now uses `SkeletonLayout::joint_active_dof_count()` and `error_state_dim()` exclusively.
    `Skeleton::active_dof()` is only called internally inside `skeleton_layout.cpp` to build
    the layout. The dangerous ad-hoc usage in filter code is gone, but the method with its
    misleading semantics (root not counted) is still there. Consider removing or renaming it.

- **Hardcoded marker names in `Tracker::initialize`** (2026-02-28)
  - `tracker.cpp` references `MRK-hip.L`, `MRK-hip.R`, `MRK-shoulder.L`, `MRK-shoulder.R`
    by name to estimate the analytic root position and orientation during initialization.
  - Fix: replace with skeleton-agnostic full-cloud Procrustes. Collect all markers that have
    both a triangulated observation and a rest-pose FK position. Solve the weighted point-cloud
    alignment (SVD of the cross-covariance matrix, possibly RANSAC-based to handle bad
    triangulations). No skeleton YAML changes needed.

## Fixed

- **FIXED (2026-02-28)**: \`root_velocity_\` was \`VectorXd\` instead of \`Vector3d\`
  - Now \`Eigen::Vector3d\` in \`State\` with a matching typed accessor.

- **FIXED (2026-02-28)**: \`ObservationsSet::get_all_at_time()\` caused time-point issues
  - Method removed; all callers now use \`get_all_in_range()\`.

- **FIXED (2026-02-28)**: State vector contained inactive (out-of-group) joints
  - \`SkeletonLayout::from_groups()\` only includes selected-group joints in
    \`total_storage_dof_count_\`; UKF state is always layout-scoped, not full-skeleton-sized.

- **FIXED (2026-02-28)**: Joint/state/error-state index mapping scattered across codebase
  - \`SkeletonLayout\` is now the single source of truth for all DOF index arithmetic,
    replacing ad-hoc loops that previously appeared independently in \`UnscentedKalmanFilter\`,
    \`ConstantVelocityModel\`, \`SigmaPointGenerator\`, etc. All indices are precomputed at
    construction; O(1) hot-path access via \`JointDesc::state_index\` / \`error_index\`.

- **FIXED (2026-02-12)**: Reprojection errors in \`tracking_stats.csv\` were zero for most frames
  - **Root cause**: After outlier rejection, UKF recomputed \`measurement_mean\` using only
    inliers, but \`observation_results\` still held innovation values from the OLD
    \`measurement_mean\` (including outliers). \`StatisticsTracker\` read these stale values.
  - **Fix**: Added recomputation of innovation values in \`observation_results\` after
    \`measurement_mean\` is updated for inliers (\`src/filters/ukf.cpp\`, lines ~962-1010).

```

---

## Architectural debt: raw `skeleton.joints()` iteration mixed with state-vector indexing

**Status**: Partially mitigated; root cause not yet resolved.

**Problem**:
`SkeletonLayout` exists precisely to decouple the raw skeleton joint list from the state-vector
layout.  It handles follower collapsing, group filtering, locked DOFs, and all DOF index
arithmetic.  Yet numerous sites iterate `skeleton.joints()` directly while simultaneously
advancing a `joint_angle_idx` / `angle_idx` counter into the state vector.  This is a layering
violation: the raw joint list includes scale-group *follower* prismatic joints that share a state
slot with their leader, so the index counter goes out of bounds whenever followers exist.

**Known affected sites** (each caused an `Eigen` out-of-bounds abort, found 2026-03-01):
- `Skeleton::total_dof_count()` — counted followers as independent slots
- `InverseKinematics::config_to_state()` — DOF count loop + extraction loop
- `Tracker::estimate_analytic_state()` — init loop
- `UnscentedKalmanFilter::enforce_joint_limits()` — two passes (angle clamping + velocity zeroing)
- `UnscentedKalmanFilter::write_sigma_points_csv()` — four loops (header + data, angle + vel)
- `TrackingExporter::write_frame()` — joint angle export loop

**Mitigations applied** (2026-03-01):
Added `Joint::is_scale_follower` flag so loops that must iterate raw joints can skip followers
explicitly.  Removed the `!include_all` guard in `SkeletonLayout::build()` so follower collapsing
applies to all layouts including the full-skeleton one used by IK.

**Correct long-term fix**:
All code that iterates joints in state-vector order should use `SkeletonLayout::joints()`, which
already presents only leaders in the correct order with precomputed `state_index` values.
Raw `skeleton.joints()` iteration is legitimate only for:
- Building the Pinocchio model (`pinocchio_model_builder.cpp`) — operates in `q`-space, not state-space
- Structural queries (root finding, group membership) that do not touch state indices

Any new code that combines `skeleton.joints()` with a manual `state_index` counter should be
treated as a bug by default and refactored to use `SkeletonLayout::joints()` instead.
