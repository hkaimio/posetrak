# Error-State Dimension Refactoring Plan

## Problem Statement

The C++ UKF implementation uses **all DOFs** (including locked ones) in the error-state space, while the Python implementation correctly uses only **active DOFs** (excluding locked ball joint axes). This causes:

- Different sigma point counts: C++ 501 vs Python 421
- Different covariance matrix sizes: 250×250 vs 210×210  
- Different sigma point spreads and covariance scaling
- Incompatible results between implementations

**Root Cause**: 
- **C++**: `error_dim = 2 * (6 + skeleton.total_dof_count())` ← includes locked DOFs
- **Python**: `error_dim = 2 * skeleton.get_total_active_dof()` ← only active DOFs

## Architecture Comparison

### Python Design ✓
- `JointSpaceState` = simple data container (no skeleton knowledge)
- `JointSpaceFilter._compute_state_error()` = has skeleton, handles locked DOFs
- `JointSpaceFilter._apply_error_to_state()` = has skeleton, handles locked DOFs
- Error space = tangent space with only active DOFs

### C++ Current Design ✗
- `State` = data + error logic (`apply_error_update()` method)
- `State::apply_error_update()` = **no skeleton access**, can't handle locked DOFs
- `UnscentedKalmanFilter::compute_state_error()` = has skeleton, **doesn't** handle locked DOFs
- Error space = incorrectly includes all DOFs

### C++ Target Design ✓
- `State` = data container only (like Python)
- `UnscentedKalmanFilter::compute_state_error()` = has skeleton, handles locked DOFs
- `UnscentedKalmanFilter::apply_error_to_state()` = **new method**, has skeleton, handles locked DOFs
- Error space = tangent space with only active DOFs

## Required Changes

### Priority 1: Core Dimension Fix

#### 1.1 Fix SigmaPointGenerator Error Dimension
**File**: `src/filters/sigma_points.cpp:19`
```cpp
// BEFORE:
error_dim_(2 * (6 + skeleton.total_dof_count()))

// AFTER:
error_dim_(2 * (6 + skeleton.active_dof()))
```
**Impact**: Changes n_sigma from 501 to 421, matches Python

#### 1.2 Fix Tracker Initialization Covariance Sizing
**File**: `src/tracking/tracker.cpp:173-183`
```cpp
// BEFORE:
int const error_dim = initial_state.error_state_dim();  // Uses wrong size
int const pos_dim = error_dim / 2;
int joint_dof = pos_dim - 6;

// AFTER:
int const active_dof_count = skeleton_.active_dof();
int const error_dim = 2 * (6 + active_dof_count);
int const pos_dim = error_dim / 2;
int const joint_active_dof = active_dof_count - 6;
```

### Priority 2: Error Space Operations

#### 2.1 Fix compute_state_error() - Handle Locked DOFs
**File**: `src/filters/ukf.cpp:336-395`

**Current Logic**: Always includes all 3 DOFs for SPHERICAL joints using full SO(3) error

**New Logic**: 
- Check `joint.active_dof()` and `joint.get_active_dof_mask()`
- If `num_active == 3`: Use full SO(3) error (current code)
- If `num_active < 3`: Use simple angle differences for active DOFs only
- Advance `error_pos_idx` by `num_active` (not always 3)
- Advance `joint_angles_idx` by 3 (storage size)

**Code Pattern**:
```cpp
else if (joint.type == JointType::SPHERICAL) {
    auto active_mask = joint.get_active_dof_mask();
    int num_active = joint.active_dof();
    
    if (num_active == 3) {
        // All DOFs active - use full SO(3) error
        // ... existing rotation matrix logic ...
        error.segment<3>(error_pos_idx) = axis_angle_error;
        error_pos_idx += 3;
    } else {
        // Some DOFs locked - simple angle differences for active DOFs
        Eigen::Vector3d const aa_ref = reference.joint_angles().segment<3>(joint_angles_idx);
        Eigen::Vector3d const aa_state = state.joint_angles().segment<3>(joint_angles_idx);
        Eigen::Vector3d angle_error = aa_state - aa_ref;
        
        int active_idx = 0;
        for (int i = 0; i < 3; ++i) {
            if (active_mask[i]) {
                error(error_pos_idx + active_idx) = angle_error(i);
                active_idx++;
            }
        }
        error_pos_idx += num_active;
    }
    joint_angles_idx += 3;  // State storage always 3
}
```

**Also fix velocity errors** using same logic at line ~360

#### 2.2 Add apply_error_to_state() Method
**File**: `include/posetrak/filters/ukf.hpp` (add to private section)
```cpp
/**
 * @brief Apply error-state update with manifold operations
 * @param nominal_state Nominal state
 * @param error_vec Error vector in tangent space (active DOFs only)
 * @return Updated state
 */
State apply_error_to_state(State const& nominal_state, Eigen::VectorXd const& error_vec) const;
```

**File**: `src/filters/ukf.cpp` (new implementation)

Mirror Python's `_apply_error_to_state()` logic:
- Position: additive `new_pos = nominal_pos + error[0:3]`
- **Root quaternion**: `new_quat = nominal_quat * error_quat` (RIGHT multiplication, body frame)
- REVOLUTE joints: additive `new_angle = nominal_angle + error[idx]`
- SPHERICAL joints:
  - If `num_active == 3`: Use SO(3) composition via rotation matrices
  - If `num_active < 3`: Additive for active DOFs only, skip locked DOFs
- Velocities: Same pattern as positions
- Track two indices:
  - `error_pos_idx`: position in error vector (only active DOFs)
  - `state_pos_idx`: position in full state (all DOFs)

#### 2.3 Update Call Site
**File**: `src/filters/ukf.cpp:671`
```cpp
// BEFORE:
state_.apply_error_update(state_correction);

// AFTER:
state_ = apply_error_to_state(state_, state_correction);
```

### Priority 3: Bug Fixes

#### 3.1 Fix Quaternion Multiplication Order Bug
**File**: `src/core/state.cpp:75`
```cpp
// BEFORE (WRONG - left multiplication, world frame):
root_orientation_ = (delta_q * root_orientation_).normalized();

// AFTER (CORRECT - right multiplication, body frame):
root_orientation_ = (root_orientation_ * delta_q).normalized();
```

**Rationale**: Error-state formulation uses body-frame errors, requiring right multiplication.

### Priority 4: Cleanup

#### 4.1 Deprecate State::apply_error_update()
**File**: `include/posetrak/core/state.hpp:118-120`

Add deprecation comment:
```cpp
/// @deprecated This method doesn't handle locked DOFs correctly.
///             Use UnscentedKalmanFilter::apply_error_to_state() instead.
void apply_error_update(Eigen::VectorXd const& error_delta);
```

Consider removing entirely if not used outside tests.

#### 4.2 Deprecate State::error_state_dim()
**File**: `include/posetrak/core/state.hpp:95`

Add deprecation comment:
```cpp
/// @deprecated This returns storage dimension, not error-state dimension.
///             Use filter's error_dim() method instead.
int error_state_dim() const { return 2 * (3 + 3 + joint_angles_.size()); }
```

#### 4.3 Update Tests
**Files**: `tests/test_state.cpp`, `tests/test_ukf.cpp`

Update any tests using `State::apply_error_update()` to use filter methods instead.

## Implementation Order

1. **Commit existing changes** (debug features, config updates)
2. **Priority 1**: Fix dimensions (sigma points, tracker init)
3. **Priority 2**: Implement error-space operations (compute_error, apply_error)
4. **Priority 3**: Fix quaternion bug
5. **Priority 4**: Add deprecation warnings, update tests
6. **Verify**: Run comparison tests, check sigma point counts match

## Expected Results

After refactoring:
- ✓ Error dimension: 210 (matches Python)
- ✓ Sigma points: 421 (matches Python)
- ✓ Covariance: 210×210 (matches Python)
- ✓ Initial covariance: diagonal values ~1.0 (matches Python)
- ✓ Sigma point spread: comparable to Python
- ✓ Locked DOFs: properly excluded from error space
- ✓ Quaternion updates: proper body-frame composition

## Testing Strategy

1. **Unit Tests**: Verify locked DOF handling
2. **Dimension Tests**: Check error_dim == 210
3. **Sigma Point Tests**: Verify 421 sigma points
4. **Comparison Tests**: Run cpp-python-comparison
5. **Integration Tests**: Full tracking pipeline

## Risks and Considerations

- **Breaking Change**: Changes error-state dimension, invalidates saved states
- **Test Updates**: May need to update expected values in tests
- **Performance**: Slightly faster (smaller matrices)
- **Correctness**: Critical fix - current code is fundamentally wrong

## References

- Python implementation: `kalman_tracker/joint_space/filter_base.py`
- Error-state formulation: Quaternion kinematics for error-state Kalman filter (Sola, 2017)
- Ball joint DOF locking: Implemented via joint limits in skeleton YAML
