# Frame 0 Debug Status - 2026-02-02

## Problem Summary
C++ tracker diverges from Python tracker immediately at frame 0, with 8m position difference and only 51 inliers vs Python's 115 inliers.

## Completed Work

### 1. std::optional Refactoring (✅ Complete)
- Changed `Camera::project()` and `Camera::project_undistorted()` to return `std::optional<Eigen::Vector2d>`
- Updated all call sites across tests, CLI, tracking code, and triangulation
- Eliminates NaN propagation, makes projection failures type-safe

### 2. NaN-Safe Covariance Computation (✅ Complete)
- Fixed innovation covariance computation for inliers (lines 531-572 in ukf.cpp)
- Applied same NaN-safe handling as initial computation (nanmean for sigma points)
- Tracker now completes all 130 frames without crashing

### 3. Test Infrastructure Implementation (⏳ In Progress)

#### Created Files:
- `tests/test_helpers/python_data_loader.{hpp,cpp}` - Load Python debug data from JSON/CSV
- `tests/test_helpers/matrix_comparison.{hpp,cpp}` - Numerical comparison utilities with Catch2 matchers
- `tests/test_ukf_frame0_comparison.cpp` - Integration test comparing C++ and Python UKF update
- `include/posetrak/filters/ukf.hpp` - Added testing accessor `generate_sigma_points_for_testing()`
- `src/filters/ukf.cpp` - Implemented testing accessor

#### Test Structure:
Sections 1-3 implemented:
1. **Load Python debug data** - Verify files exist, data loads correctly
2. **Initialize C++ UKF** - Set prior state/covariance, verify match
3. **Compare sigma points** - Generate 145 sigma points, compare with Python

## Current Status

### Compilation: ✅ Success
All code compiles without errors after resolving:
- State default constructor issue (→ std::optional<State>)
- State to_vector() missing (→ manual conversion using segments)
- Loader API mismatch (→ free functions instead of class methods)
- Camera map type mismatch (→ string keys → int keys conversion)
- Camera default constructor (→ use .insert() instead of operator[])
- JSON structure (joint_angles/velocities are dicts with arrays, not flat arrays)

### Test Execution: ❌ Partially Failing
**Current Errors:**
1. **Sigma points shape mismatch**: CSV loaded as 421×244, expected 145×error_dim
   - File has 422 lines (1 header + 421 data), 252 columns
   - Need to investigate why 421 rows instead of 145
   - May be loading wrong file or file format issue

2. **Covariance dimension mismatch**: "Covariance size must match error dimension"
   - UKF constructor validation failing
   - Likely related to state dimension calculation

### Next Steps

1. **Debug sigma points loading**:
   - Check if CSV has duplicate rows or unexpected format
   - Verify error_dim calculation matches between Python and C++
   - May need to filter or properly parse the CSV

2. **Fix covariance dimension**:
   - Verify prior_covariance.csv dimensions
   - Check skeleton.total_dof_count() matches expected error_dim
   - State error dimension should be: 3 (root_pos) + 3 (root_quat_error) + n_dof + 3 (root_vel) + 3 (root_ang_vel) + n_dof

3. **Complete sections 4-9** (once sections 1-3 pass):
   - Section 4: Compare measurement prediction
   - Section 5: Compare outlier rejection
   - Section 6: Compare innovation covariance
   - Section 7: Compare Kalman gain
   - Section 8: Compare posterior state
   - Section 9: Compare posterior covariance

## Python Debug Data Structure
Located in: `tracking_tests/cpp-python-comparison/python_results/debug/frame_0000/`

Files:
- `prior_state.json` - Root pose, joint angles/velocities (JSON with named joints)
- `prior_covariance.csv` - 72×72 covariance matrix
- `all_observations.csv` - 341 observations
- `sigma_points.csv` - 145 sigma points × state dimensions (422 lines × 252 cols - investigate)
- `predicted_observations.csv` - Predicted measurements for each sigma point
- `innovation_covariance.csv` - Innovation covariance after outlier rejection
- `kalman_gain.csv` - Kalman gain matrix
- `posterior_state.json` - Updated state after UKF
- `posterior_covariance.csv` - Updated covariance after UKF
- `outlier_flags.csv` - Boolean flags for each observation

## Git Branch
Working in: `hkaimio/first-frame-debug`
