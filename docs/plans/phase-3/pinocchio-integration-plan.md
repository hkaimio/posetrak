# Phase 3.1: Pinocchio Integration for Forward Kinematics

## Status: Planning

## Objective

Port the working Pinocchio-based forward kinematics from `/home/harri/projects/cpp-tracker-test` to posetrak, adapting it to work with posetrak's Skeleton data structures while preserving the proven, zero-error implementation.

## Reference Implementation

The cpp-tracker-test implementation has:
- ✅ Zero difference in marker positions vs Python
- ✅ Working meson build configuration for Pinocchio
- ✅ Proven handling of all joint types (ROOT/FREE_FLYER, BALL/SPHERICAL, REVOLUTE)
- ✅ Correct quaternion conversions and Euler angle handling

## Architecture Comparison

### Skeleton Structures

**cpp-tracker-test:**
```cpp
enum class JointType { ROOT, BALL, REVOLUTE };
struct Joint {
    std::string name;
    std::optional<std::string> parent;  // nullopt for root
    JointType type;
    int dof;
    Eigen::Vector3d offset_from_parent;
    Eigen::Vector3d rest_orientation;  // ZYX Euler
    bool has_rest_orientation;
};
```

**posetrak:**
```cpp
enum class JointType { REVOLUTE, SPHERICAL, FIXED };
struct Joint {
    std::string name;
    std::string parent;  // empty for root
    JointType type;
    int dof;
    Eigen::Vector3d offset;
};
```

### Key Differences

1. **Root representation**: cpp-tracker-test has explicit ROOT joint type, posetrak uses empty parent string
2. **Ball joints**: cpp-tracker-test calls them BALL, posetrak calls them SPHERICAL
3. **Rest orientation**: cpp-tracker-test supports rest_orientation (ZYX Euler), posetrak doesn't (yet)
4. **Fixed joints**: posetrak has FIXED type, cpp-tracker-test doesn't

## Implementation Plan

### 3.1.1: Meson Build Integration ✅ (Easy - copy proven config)

**From cpp-tracker-test meson.build:**
```meson
# Pinocchio - manually construct dependency to avoid pkg-config issues
pinocchio_inc = include_directories('/opt/openrobots/include', is_system: true)
pinocchio_dep = declare_dependency(
  include_directories : pinocchio_inc,
  dependencies : [
    cpp_compiler.find_library('pinocchio_default', dirs : ['/opt/openrobots/lib'], required : true),
    cpp_compiler.find_library('pinocchio_parsers', dirs : ['/opt/openrobots/lib'], required : false),
  ],
  link_args : ['-L/opt/openrobots/lib', '-Wl,-rpath,/opt/openrobots/lib'],
  compile_args : ['-DPINOCCHIO_WITH_URDFDOM', '-DPINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION']
)
```

**Action**: Add to posetrak's meson.build

### 3.1.2: PinocchioModelBuilder Class (Port with adaptations)

**File**: `include/posetrak/kinematics/pinocchio_model_builder.hpp`

**Key adaptations needed:**
1. Map posetrak::JointType to Pinocchio joint models:
   - `REVOLUTE` → `JointModelRX/RY/RZ` (axis from rotation limits)
   - `SPHERICAL` → `JointModelSpherical` (like cpp-tracker-test's BALL)
   - `FIXED` → No Pinocchio joint (or Identity joint)
   - Root (empty parent) → `JointModelFreeFlyer`

2. Handle posetrak's Skeleton accessor methods:
   - `skeleton.get_joints()` returns `std::vector<Joint> const&`
   - `skeleton.get_markers()` returns `std::vector<Marker> const&`
   - `skeleton.find_joint(name)` returns `Joint const*`

3. Extract rotation axis for REVOLUTE joints from limits:
   - If only limits[0] is active → X axis
   - If only limits[1] is active → Y axis
   - If only limits[2] is active → Z axis

**Port verbatim** (proven correct):
- `add_joint_recursive()` - Core recursive build logic
- `add_marker_frames()` - Marker frame attachment
- Rest orientation handling (ZYX Euler → rotation matrix)
- Frame ID lookup and mapping

### 3.1.3: ForwardKinematics Class (Port with State adaptations)

**File**: `include/posetrak/kinematics/forward_kinematics.hpp`

**Key adaptations:**
1. Replace `StateFrame` (cpp-tracker-test) with posetrak's `State`:
   ```cpp
   // cpp-tracker-test StateFrame:
   struct StateFrame {
       int frame_idx;
       double timestamp;
       std::map<std::string, double> joint_angles;
   };

   // posetrak State:
   class State {
       Eigen::VectorXd q;          // Configuration vector
       Eigen::VectorXd v;          // Velocity vector
       Eigen::VectorXd a;          // Acceleration vector
       double timestamp;
       std::optional<int> frame_idx;
       // ... accessors ...
   };
   ```

2. State conversion:
   - cpp-tracker-test: `state_to_config()` parses map of joint_angles
   - posetrak: State already has `q` vector, but we need to ensure it matches Pinocchio's expectations

3. Return type:
   - cpp-tracker-test: `MarkerPositions` struct with map
   - posetrak: Could return `std::unordered_map<std::string, Eigen::Vector3d>` directly

**Port verbatim** (proven correct):
- Quaternion conversion for spherical joints (angle-axis → quaternion)
- `pinocchio::forwardKinematics()` call
- `pinocchio::updateFramePlacements()` call
- Frame transform extraction: `data_.oMf[frame_id].translation()`

### 3.1.4: Tests (Adapt from cpp-tracker-test)

**From cpp-tracker-test:**
- `test_model_builder.cpp` - Tests model building from skeleton
- `test_yaml_skeleton.cpp` - Tests skeleton loading → model → FK

**For posetrak:**
- Test model building from simple skeleton (1 root + 1 revolute + 1 spherical)
- Test marker frame creation
- Test FK computation with known configuration
- Test that nq/nv match expected DOF counts

## Critical Details to Preserve (Hard-won knowledge)

### 1. Quaternion Order
Pinocchio uses **[x, y, z, w]** order for quaternions (not [w, x, y, z])

### 2. Root Joint Handling
- Root joint offset should be **ignored** (root is at origin)
- Only non-root joints use `offset_from_parent` in SE3 placement

### 3. Euler Angle Convention
- cpp-tracker-test uses **ZYX intrinsic** Euler angles
- Conversion: `R = Rx(x) * Ry(y) * Rz(z)` in extrinsic (fixed-frame) order
- This matches the Python implementation

### 4. Spherical Joint Representation
- State stores 3 rotation angles
- Convert to quaternion via angle-axis:
  ```cpp
  Eigen::Vector3d v(rx, ry, rz);
  double angle = v.norm();
  if (angle == 0.0) {
      quat = Eigen::Quaterniond::Identity();
  } else {
      Eigen::Vector3d axis = v / angle;
      quat = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
  }
  ```

### 5. Frame Updates
Must call **both**:
1. `pinocchio::forwardKinematics(model, data, q)` - Update joint positions
2. `pinocchio::updateFramePlacements(model, data)` - Update operational frames

Forgetting step 2 results in incorrect marker positions.

## Testing Strategy

1. **Unit test**: Build model from simple skeleton, verify nq/nv
2. **Integration test**: Load YAML skeleton from Phase 2, build model, compute FK
3. **Comparison test**: Compare with Python FK results (should be identical)
4. **Performance test**: Benchmark FK computation speed

## Files to Create

### Headers
- `include/posetrak/kinematics/pinocchio_model_builder.hpp`
- `include/posetrak/kinematics/forward_kinematics.hpp`

### Implementation
- `src/kinematics/pinocchio_model_builder.cpp`
- `src/kinematics/forward_kinematics.cpp`

### Tests
- `tests/test_pinocchio_model_builder.cpp`
- `tests/test_forward_kinematics.cpp`

## Dependencies

- ✅ Pinocchio 2.x (installed at /opt/openrobots)
- ✅ Eigen 3.x (already in posetrak via meson wrap)
- ✅ Skeleton loader (Phase 2.1)
- ⏳ State representation (exists but may need FK-specific methods)

## Success Criteria

- [ ] Meson build successfully links Pinocchio
- [ ] PinocchioModelBuilder builds model from posetrak Skeleton
- [ ] ForwardKinematics computes marker positions
- [ ] All tests pass
- [ ] FK results match Python implementation (zero difference)
- [ ] No memory leaks (valgrind clean)

## Risk Mitigation

**Risk**: Introducing bugs during adaptation
**Mitigation**: Port cpp-tracker-test code as literally as possible, only changing:
- Namespace (fk_proto → posetrak)
- Type names (StateFrame → State, BALL → SPHERICAL)
- Accessor methods (direct member access → getter calls)

**Risk**: Euler angle conversion errors
**Mitigation**: Use exact same conversion code from cpp-tracker-test

**Risk**: Quaternion ordering mistakes
**Mitigation**: Keep [x,y,z,w] order explicit in comments everywhere

## Next Steps After 3.1

Once FK is working:
- **3.3**: Triangulation (multi-view 2D → 3D reconstruction)
- **3.4**: Inverse kinematics (3D markers → joint angles)
- **Skip 3.2**: URDF export (postponed, not critical for tracking)
