## State

- why root_velocity_ is VectorXd, not VEctor3D (2026-01-26)

- Skeleton::active_dof() does not count root position & orientation. Should it? Or is it only part of the tracker state?

- ObservationsSet still has the method get_all_at_time() which seems to cause issues. We really should use get_all_in_range always.

- State vector has now also inactve joitns (those not in any active group) I think this is design error as those ar enot used for anything

- Mapping between joints, state and error state vector indices is not doe in many places and there ahve been many errors. We should have a single component that doers thjis,.

## Fixed Issues (2026-02-12)

- **FIXED**: Reprojection errors in tracking_stats.csv were zero for most frames (2026-02-12)
  - **Root Cause**: After outlier rejection, UKF recomputes measurement_mean using only inliers, but observation_results still contained innovation values computed with the OLD measurement_mean (including outliers). StatisticsTracker then computed reprojection errors from these stale innovations.
  - **Fix**: Added code in ukf.cpp to recompute innovation values in observation_results after measurement_mean is updated for inliers. This ensures reported reprojection errors match actual filter behavior.
  - **File**: src/filters/ukf.cpp, lines ~962-1010
  - **Impact**: Reprojection error statistics now correctly reflect tracking quality
