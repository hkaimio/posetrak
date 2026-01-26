# FK Validation Test Debug Notes

**Date**: January 26, 2026
**Status**: Test failing with ~1.745m marker position errors

## Goal

Validate C++ forward kinematics implementation against Python tracker ground truth data using the full ~120 DOF Harri skeleton.

## Test Setup

### Data Sources
- **Skeleton**: `tests/data/Harri_skeleton-shouldery-rot.yaml` (60 joints, 61 markers)
  - Loaded via `load_skeleton_from_yaml()`
  - Results in: 60 joints, 119 DOF total (6 root + 113 non-root)
  - Joint types: 1 root, 27 ball/spherical, 32 revolute

- **Ground Truth States**: `tests/data/states_0.json` (10 frames)
  - Joint angles stored as `joint_<name>_angle_<idx>` keys
  - 240 total keys per frame (includes velocities, metadata)
  - Angles are in **axis-angle representation** (radians)

- **Ground Truth Markers**: `tests/data/marker_positions_0.json` (10 frames)
  - 61 markers per frame with 3D positions

### Pinocchio Model
- Built successfully: `nq=147`, `nv=119`
- Configuration vector (q):
  - Root: 7 DOF (3 pos + 4 quat [x,y,z,w])
  - Spherical joints: 4 DOF each (quaternion)
  - Revolute joints: 1 DOF each (angle)

### Test Implementation
Location: `tests/test_pinocchio_integration.cpp` - "ForwardKinematics validates against Python ground truth"

Currently builds configuration vector q by:
1. Extracting root position and quaternion from JSON
2. Iterating through skeleton joints (via Pinocchio model.names iteration)
3. Looking up joint angles from JSON by joint name
4. For spherical joints: converting 3-element axis-angle to quaternion
5. For revolute joints: copying single angle value

## Observed Error

### Symptoms
- First marker tested: `MRK-Ankle.L`
- **Computed position**: `[9.16769, 2.63875, 1.97921]`
- **Ground truth position**: `[9.15835, 0.899874, 2.12893]`
- **Error magnitude**: 1.74534 meters (~1.745 radians ≈ 100 degrees)

### Key Observations
1. X and Z coordinates are close (errors: 0.009m and 0.15m)
2. Y coordinate has massive error: 2.639 vs 0.900 (diff: 1.739m)
3. Error magnitude (1.745m) suspiciously equals ~100 degrees in radians (1.745 rad = 100.02°)
4. Root position from JSON: `[9.08, 1.78, 1.98]`
5. Frame 0 joint angles are all very small (~1e-4 radians), so this isn't accumulated error

### Configuration Vector
- Successfully builds 147-element q vector
- Successfully looks up joint angles by name from JSON
- Successfully converts axis-angle to quaternions for spherical joints

## Theories on Root Cause

### Theory 1: Joint Ordering Mismatch ❌ (Eliminated)
**Hypothesis**: `get_joints_ordered()` returns depth-first traversal (hips → thigh.L → shin.L → ...) but JSON has joints in YAML file order (hips → spine1 → spine2 → ...)

**Evidence Against**:
- Test was rewritten to look up joint angles by NAME from JSON
- Uses `model.names[i]` to iterate through Pinocchio joints
- Each joint's angles are fetched via `"joint_" + joint_name + "_angle_" + idx`
- Order shouldn't matter with name-based lookup

### Theory 2: Axis-Angle Conversion Issue
**Hypothesis**: The 3-element vectors in JSON represent axis-angle but conversion is incorrect

**Status**: Need to verify
- cpp-tracker-test converts axis-angle → quaternion exactly like we do
- Our code in `forward_kinematics.cpp`:
  ```cpp
  double angle = angles.norm();
  Eigen::Vector3d axis = angles.normalized();
  Eigen::Quaterniond quat(Eigen::AngleAxisd(angle, axis));
  q[idx++] = quat.x(); q[idx++] = quat.y();
  q[idx++] = quat.z(); q[idx++] = quat.w();
  ```
- This matches cpp-tracker-test implementation

**Question**: Are we applying axis-angle conversion when we shouldn't? Or vice versa?

### Theory 3: Pinocchio Model Building Mismatch
**Hypothesis**: Model is built in wrong order or with wrong joint mappings

**Status**: Need to investigate
- `PinocchioModelBuilder` iterates through `skeleton.joints()` (unordered_map)
- Adds joints recursively starting from root
- Question: Does iteration order of unordered_map matter for Pinocchio?
- cpp-tracker-test uses `skeleton.joints` which is a **vector** in file order

**Key Difference**:
- cpp-tracker-test skeleton: `std::vector<Joint>` (ordered)
- posetrak skeleton: `std::unordered_map<string, Joint>` (unordered)

### Theory 4: Skeleton/Marker Definition Mismatch
**Hypothesis**: YAML skeleton definition differs from JSON expectations

**Status**: Unlikely but possible
- Both use same source file: `Harri_skeleton-shouldery-rot.yaml`
- JSON was exported from Python tracker using this same skeleton
- But: Did JSON export use different marker offsets or joint structure?

### Theory 5: Coordinate System / Frame Mismatch
**Hypothesis**: World frame, joint frames, or marker frames differ between implementations

**Status**: Need to check
- Root position looks reasonable (within 0.1m of marker position)
- But marker offset from root seems wrong (0.86m vs -0.88m in Y)
- This 1.74m difference suggests a 180° flip or similar transformation issue

## Debug Data

### Frame 0 State
```
Root position: [9.080116, 1.780279, 1.982511]
Root quaternion: [0.999999, -8.76e-06, 1.62e-10, -6.40e-09] (nearly identity)
First few joint angles (axis-angle):
  - spine1: [-0.000413, 1.97e-11, -3.98e-09]  (tiny rotation)
  - spine2: [-0.000368, -8.95e-11, -3.34e-09]
  - etc. (all very small, ~1e-4 magnitude)
```

### MRK-Ankle.L Error Breakdown
```
Root: [9.08, 1.78, 1.98]
Marker computed: [9.17, 2.64, 1.98]  →  offset from root: [0.09, 0.86, 0.00]
Marker GT:       [9.16, 0.90, 2.13]  →  offset from root: [0.08, -0.88, 0.15]

Y-axis error: 0.86 - (-0.88) = 1.74m ← This is the problem!
```

The Y offset has the WRONG SIGN and wrong magnitude. This suggests:
- Joint is rotating in opposite direction, OR
- Parent joint chain is wrong, OR
- Coordinate frame is flipped

## Next Steps

### Immediate Actions
1. **Compare Pinocchio model structure**:
   - Print joint hierarchy from both cpp-tracker-test and posetrak
   - Check if joint parent relationships match
   - Verify frame IDs and names match

2. **Test with simple skeleton**:
   - Create minimal test: root → 1 joint → 1 marker
   - Verify FK with known configuration
   - Compare against cpp-tracker-test with same skeleton

3. **Debug ankle chain specifically**:
   - Trace joint chain from root to ankle marker
   - Print each joint's transformation
   - Compare intermediate transforms with Python

4. **Check PinocchioModelBuilder ordering**:
   - Add debug output showing order joints are added to model
   - Compare with cpp-tracker-test's joint order
   - Verify `model.names` order matches JSON expectations

### Questions to Answer
1. Does `PinocchioModelBuilder` need to add joints in a specific order?
2. Is there a coordinate frame convention difference (Y-up vs Z-up)?
3. Are we correctly handling the root joint (type=FIXED but should be FreeFlyer in Pinocchio)?
4. Do we need to maintain joint insertion order (switch from unordered_map to ordered structure)?

## References

### Relevant Code Locations
- Test: `tests/test_pinocchio_integration.cpp:192`
- FK implementation: `src/kinematics/forward_kinematics.cpp`
- State to config: `forward_kinematics.cpp:57-145`
- Model builder: `src/kinematics/pinocchio_model_builder.cpp`
- cpp-tracker-test reference: `/home/harri/projects/cpp-tracker-test/src/benchmark_fk.cpp`

### Known Good Implementation
cpp-tracker-test achieves <1e-10 error with same data format and skeleton structure. The implementation should be nearly identical.
