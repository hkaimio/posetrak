# Open Issues

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
