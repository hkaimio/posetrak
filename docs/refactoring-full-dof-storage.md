# Refactoring Plan: Full 3-DOF Storage for Spherical Joints

## Goal
Adopt Python prototype's approach of always storing 3 DOFs for SPHERICAL joints, regardless of locked DOFs, to fix index mapping bugs and simplify the codebase.

## Problem Summary
**Current State**:
- State stores only active DOFs (e.g., 1 element for spherical joint with 2 locked axes)
- Process model and UKF expect full 3-DOF layout for spherical joints
- Manual index tracking is error-prone and causes out-of-bounds access

**Root Cause**: Inconsistent representation between State vector (compact) and iteration logic (expects full layout)

## Python's Approach (Validated)
- Always allocates 3 elements for BALL joints in `position_state` vector
- No special index mapping or compact storage
- Locked DOFs enforced through constraint/limit system, not storage
- Simple iteration: `dof_idx += 3` always works for BALL joints

**Advantages**:
1. Simple indexing - no mapping complexity
2. Uniform representation - all SPHERICAL joints treated same
3. Small memory overhead - only 8-16 bytes per joint for 1-2 locked DOFs
4. No index tracking bugs - consistent layout everywhere

## Design Decision
**Option 3 (Full 3-DOF Storage)** + **Builder Pattern**

### Core Changes:
1. **State representation**: Always allocate 3 elements for SPHERICAL joints
2. **Skeleton**: Add precomputed index mappings (optional builder pattern for future)
3. **No IndexMapper class needed**: Simple, consistent iteration

## Implementation Plan

### Phase 1: Update State Representation

#### Files to Modify:
- `include/posetrak/core/state.hpp`
- `src/core/state.cpp`
- `tests/test_state.cpp`

#### Changes:
1. **Update State constructors**:
   - Constructor now takes **total DOF count** (not active DOF count)
   - For SPHERICAL joints: always allocate 3 elements
   - Document new invariant

2. **Update State::num_dof() semantics**:
   - Returns **total** DOF count (including locked DOFs for SPHERICAL)
   - Add comment clarifying this is storage size, not active DOF count

3. **Keep State interface unchanged**:
   - `joint_angles()` returns VectorXd with full storage
   - Accessors remain the same
   - Locked DOFs stored as values (typically 0.0)

#### Example Changes:
```cpp
// Before: State state(skeleton.active_dof_count());
// After:  State state(skeleton.total_dof_count());

// Spherical joint with 1 locked DOF:
// Before: joint_angles has 2 elements
// After:  joint_angles has 3 elements (locked one stored as 0.0)
```

### Phase 2: Update Skeleton to Provide Total DOF Count

#### Files to Modify:
- `include/posetrak/core/skeleton.hpp`
- `src/core/skeleton.cpp`
- `tests/test_skeleton.cpp`

#### Changes:
1. **Add `total_dof_count()` method**:
   ```cpp
   /// @brief Get total DOF count (always 3 for SPHERICAL, regardless of locked DOFs)
   /// @return Total storage DOFs needed for state vector
   int total_dof_count() const;
   ```

2. **Keep `active_dof_count()` for information**:
   - Used for constraint enforcement, limit checking
   - Not used for State allocation

3. **Add joint index mapping helpers** (optional, for later):
   ```cpp
   /// @brief Get starting index in state vector for a joint
   /// @param joint_name Name of joint
   /// @return Index into State::joint_angles() vector
   int get_joint_state_index(std::string const& joint_name) const;
   ```

4. **Implementation**:
   ```cpp
   int Skeleton::total_dof_count() const {
       int total = 0;
       for (auto const& joint : joints_) {
           if (joint.type == JointType::REVOLUTE) {
               total += 1;
           } else if (joint.type == JointType::SPHERICAL) {
               total += 3;  // Always 3, regardless of locked DOFs
           }
           // FIXED has 0 DOF
       }
       return total;
   }
   ```

### Phase 3: Update Process Model

#### Files to Modify:
- `src/filters/process_model.cpp`
- `tests/test_process_model.cpp`

#### Changes:
1. **Simplify index tracking**:
   ```cpp
   // Before: Complex active_mask iteration
   if (num_active == 3) {
       // Full 3-DOF
   } else {
       // Locked DOF case - manual index tracking
   }

   // After: Always 3 DOFs
   Eigen::Vector3d current_axis_angle = state.joint_angles().segment<3>(angle_idx);
   // ... process model logic ...
   new_joint_angles.segment<3>(angle_idx) = new_axis_angle;
   angle_idx += 3;  // Always advance by 3
   ```

2. **Remove locked DOF special case** (lines 84-98 in process_model.cpp)

3. **Locked DOF enforcement**:
   - Apply after prediction via limit enforcement
   - Or zero out velocity for locked DOFs in process model

### Phase 4: Update UKF Mean/Covariance Computation

#### Files to Modify:
- `src/filters/ukf.cpp`
- `include/posetrak/filters/ukf.hpp`

#### Changes:
1. **Simplify `compute_state_mean()`** (lines 142-195):
   ```cpp
   // Before: Manual dof_idx tracking with active_mask checks
   else if (joint.type == JointType::SPHERICAL) {
       int num_active = joint.active_dof();
       if (num_active == 3) {
           // Full DOF
       } else {
           // Complex active_mask iteration
       }
   }

   // After: Always 3 DOFs
   else if (joint.type == JointType::SPHERICAL) {
       // Iterative mean on SO(3) manifold (Karcher mean)
       Eigen::Matrix3d R_mean = axis_angle_to_rotation(
           sigma_points[0].joint_angles().segment<3>(dof_idx)
       );

       // ... iterative refinement ...

       angles_mean.segment<3>(dof_idx) = rotation_to_axis_angle(R_mean);
       dof_idx += 3;  // Always advance by 3
   }
   ```

2. **Simplify `compute_state_error()`** (lines 260-315):
   - Remove active_mask iteration
   - Always use `segment<3>()` for SPHERICAL joints

3. **Simplify `compute_state_covariance()`**:
   - Index tracking becomes trivial
   - No special cases needed

### Phase 5: Update Tests

#### Files to Modify:
- `tests/test_ukf.cpp`
- `tests/test_state.cpp`
- `tests/test_process_model.cpp`
- `tests/test_skeleton.cpp`

#### Changes:
1. **Update State construction in tests**:
   ```cpp
   // Before: State state(2);  // 2 active DOFs
   // After:  State state(3);  // 3 total DOFs (one locked)
   ```

2. **Re-enable locked DOF test** (test_ukf.cpp:278-315):
   - Remove `[!mayfail]` tag
   - Update expectations for 3-element storage

3. **Add tests for locked DOF enforcement**:
   - Verify locked DOFs remain at 0.0 after prediction
   - Test limit enforcement clamps locked DOFs

4. **Update skeleton construction in tests**:
   - Use `skeleton.total_dof_count()` not `active_dof_count()`

### Phase 6: Update Forward Kinematics

#### Files to Modify:
- `src/kinematics/forward_kinematics.cpp`
- `tests/test_forward_kinematics.cpp`

#### Changes:
1. **Update FK to expect full 3-DOF storage**:
   - Change `segment<2>()` to `segment<3>()` where needed
   - Simplify locked DOF handling

2. **Locked DOF handling in FK**:
   - Read all 3 values, locked ones should be 0.0
   - No special logic needed

### Phase 7: Documentation and Validation

#### Tasks:
1. **Update documentation**:
   - Add comments to State class explaining storage invariant
   - Update CONTRIBUTING.md with new design
   - Document locked DOF enforcement strategy

2. **Run full test suite**:
   ```bash
   cd builddir
   meson test
   ```

3. **Verify locked DOF behavior**:
   - Check test_locked_dofs passes
   - Verify UKF prediction preserves locked DOFs at 0.0

4. **Check code coverage**:
   ```bash
   ./run_tests.sh
   ```

## Migration Strategy

### Backward Compatibility:
- **Breaking change**: State constructor signature changes
- **Impact**: Minimal - State is internal, not part of public API yet
- **Migration**: Update all `State(skeleton.active_dof_count())` → `State(skeleton.total_dof_count())`

### Testing Strategy:
1. **Phase-by-phase validation**:
   - After each phase, run affected tests
   - Commit working changes incrementally

2. **Before/After validation**:
   - Save test results before refactoring
   - Compare after refactoring (should be same or better)

3. **Locked DOF specific tests**:
   - Test with 0, 1, 2, 3 locked DOFs
   - Verify values remain at 0.0 throughout pipeline

## Expected Outcomes

### Code Quality:
- ✅ Remove complex index tracking logic
- ✅ Eliminate manual active_mask iteration
- ✅ Consistent representation throughout codebase
- ✅ Match proven Python implementation

### Bug Fixes:
- ✅ Fix out-of-bounds access in process model
- ✅ Fix index errors in UKF mean computation
- ✅ Enable locked DOF test (currently [!mayfail])

### Performance:
- ⚠️ Slight memory increase: ~8-16 bytes per locked DOF
- ✅ Simpler code may improve CPU performance (less branching)
- ≈ Negligible impact overall

### Maintainability:
- ✅ Easier to understand and modify
- ✅ Matches working Python prototype
- ✅ Reduces cognitive load for future developers

## Future Enhancements (Optional)

### Builder Pattern (Phase 8+):
If needed for multi-person tracking or complex scenarios:

1. **Add SkeletonBuilder**:
   - Mutable builder with validation
   - Computes index mappings on `build()`
   - Returns immutable Skeleton

2. **Add ModelBuilder** (for multi-person):
   - Combines multiple Skeletons
   - Manages global state indexing
   - Returns immutable Model

**Note**: Not needed for current implementation. Consider only if tracking multiple people or complex filtering scenarios.

## Risk Assessment

### Low Risk:
- ✅ Small memory overhead
- ✅ Proven approach (Python prototype)
- ✅ Incremental changes possible

### Medium Risk:
- ⚠️ Many files touched (coordination needed)
- ⚠️ Test expectations need updates

### Mitigation:
- Implement in phases with validation
- Commit after each working phase
- Keep old tests as reference during migration

## Estimated Effort

- **Phase 1-2**: 30-45 minutes (State + Skeleton changes)
- **Phase 3-4**: 45-60 minutes (Process model + UKF)
- **Phase 5**: 30 minutes (Test updates)
- **Phase 6**: 15-20 minutes (FK updates)
- **Phase 7**: 15 minutes (Documentation + validation)

**Total**: ~2.5-3 hours for complete refactoring

## Approval Checklist

Before starting implementation:
- [ ] Design approved by user
- [ ] Understand all phases
- [ ] Have backup/commit before starting
- [ ] Tests currently passing (baseline)

## Success Criteria

- [ ] All existing tests pass (87 tests)
- [ ] Locked DOF test re-enabled and passing
- [ ] No increase in test failures
- [ ] Code coverage maintained or improved
- [ ] Documentation updated
- [ ] Git commit with clear description
