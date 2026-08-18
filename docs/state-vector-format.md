# State Vector Format

## Overview

The `state_vectors.csv` file contains the tracker's estimated pose state at each frame. It represents the full configuration of the skeleton including root position/orientation and all joint angles, plus their time derivatives (velocities).

## File Format

### CSV Structure

- **Header row**: Column names defining each state component
- **Data rows**: One row per frame, ordered by `tracker_frame_idx`
- **Separator**: Comma (`,`)
- **Numeric format**: Floating point, typically scientific notation for small values

### Frame Identification

| Column | Type | Description |
|--------|------|-------------|
| `tracker_frame_idx` | integer | Frame index (0-based) in tracking sequence |
| `timestamp` | float | Time in seconds (can be negative for pre-trigger data) |

## State Components

### Root Transform (6 DOF)

The root joint represents the base of the skeleton with full 6-degree-of-freedom positioning.

#### Position (3 DOF)

| Column | Units | Description |
|--------|-------|-------------|
| `root_position_x` | meters | X-coordinate in world frame |
| `root_position_y` | meters | Y-coordinate in world frame |
| `root_position_z` | meters | Z-coordinate in world frame |

**Coordinate system**: Right-handed, typically Z-up or Y-up depending on skeleton convention.

#### Orientation (3 DOF, 4 parameters)

| Column | Range | Description |
|--------|-------|-------------|
| `root_quaternion_w` | [-1, 1] | Quaternion scalar (real) part |
| `root_quaternion_x` | [-1, 1] | Quaternion X component (i) |
| `root_quaternion_y` | [-1, 1] | Quaternion Y component (j) |
| `root_quaternion_z` | [-1, 1] | Quaternion Z component (k) |

**Quaternion convention**:
- Format: `q = w + xi + yj + zk`
- Normalized: `w² + x² + y² + z² = 1`
- Represents rotation from world frame to root's local frame
- Storage order: `[w, x, y, z]` (scalar-first)

**Rotation application**:
```
R_world_root = quaternion_to_matrix([w, x, y, z])
```

#### Velocities (6 DOF)

| Column | Units | Description |
|--------|-------|-------------|
| `root_velocity_x` | m/s | Linear velocity in X |
| `root_velocity_y` | m/s | Linear velocity in Y |
| `root_velocity_z` | m/s | Linear velocity in Z |
| `root_angular_velocity_x` | rad/s | Angular velocity around X |
| `root_angular_velocity_y` | rad/s | Angular velocity around Y |
| `root_angular_velocity_z` | rad/s | Angular velocity around Z |

**Note**: Velocities are in the **world frame**, not body frame.

### Joint Angles

Joint angles represent **deviations from the rest pose** defined in the skeleton YAML. The final joint orientation is:

```
R_final = R_rest @ R_animation
```

Where:
- `R_rest` = Rest orientation from skeleton YAML (`orientation` field, Euler angles)
- `R_animation` = Rotation from state vector angles (see below)

#### Revolute Joints (1 DOF)

Single-axis rotation around the joint's defined axis.

**Columns**: `joint_{name}_angle_0`, `joint_{name}_velocity_0`

| Column Pattern | Units | Description |
|----------------|-------|-------------|
| `joint_{name}_angle_0` | radians | Rotation angle around joint axis |
| `joint_{name}_velocity_0` | rad/s | Angular velocity |

**Example**:
```csv
joint_elbow.L_angle_0, joint_elbow.L_velocity_0
1.5708,                 0.0
```
→ Left elbow bent 90° (π/2 radians), not moving

**Rotation computation**:
```python
axis = joint.axis  # From skeleton YAML, e.g., [0, 1, 0]
angle = state_vector['joint_{name}_angle_0']
R_animation = axis_angle_to_matrix(axis * angle)
```

#### Spherical Joints (3 DOF)

Full 3-axis rotation represented as **axis-angle** (also called rotation vector).

**Columns**: `joint_{name}_angle_{0,1,2}`, `joint_{name}_velocity_{0,1,2}`

| Column Pattern | Units | Description |
|----------------|-------|-------------|
| `joint_{name}_angle_0` | radians | X-component of axis-angle |
| `joint_{name}_angle_1` | radians | Y-component of axis-angle |
| `joint_{name}_angle_2` | radians | Z-component of axis-angle |
| `joint_{name}_velocity_0` | rad/s | Angular velocity X-component |
| `joint_{name}_velocity_1` | rad/s | Angular velocity Y-component |
| `joint_{name}_velocity_2` | rad/s | Angular velocity Z-component |

**Example**:
```csv
joint_shoulder.L_angle_0, joint_shoulder.L_angle_1, joint_shoulder.L_angle_2
0.5,                      0.3,                      -0.2
```
→ Shoulder rotated by `√(0.5² + 0.3² + 0.2²) = 0.618` radians around axis `[0.809, 0.485, -0.324]`

**Axis-angle representation**:
- Vector `v = [angle_0, angle_1, angle_2]`
- Rotation magnitude: `θ = ||v|| = √(angle_0² + angle_1² + angle_2²)`
- Rotation axis: `n = v / θ` (unit vector)
- Rotation: `θ` radians around axis `n`

**Special case (zero rotation)**:
- If `θ = 0` (all components zero), no rotation is applied
- Rest pose orientation only

**Rotation computation (Rodrigues' formula)**:
```python
axis_angle = np.array([angle_0, angle_1, angle_2])
theta = np.linalg.norm(axis_angle)

if theta < 1e-10:
    R_animation = np.eye(3)  # No rotation
else:
    axis = axis_angle / theta  # Normalize
    K = skew_symmetric(axis)   # Cross-product matrix
    R_animation = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
```

Where skew-symmetric matrix of `[a, b, c]`:
```
K = [ 0  -c   b]
    [ c   0  -a]
    [-b   a   0]
```

### Fixed Joints

Fixed joints have **no state columns** - they maintain constant transforms defined in the skeleton YAML.

## Column Naming Convention

### Pattern

```
{category}_{joint_name}_{component}_{index}
```

Components:
- **Position/Orientation**: `position_{x,y,z}`, `quaternion_{w,x,y,z}`
- **Linear velocity**: `velocity_{x,y,z}`
- **Angular velocity**: `angular_velocity_{x,y,z}`
- **Joint angles**: `angle_{0,1,2}` (0 for revolute, 0-2 for spherical)
- **Joint velocities**: `velocity_{0,1,2}`

### Special Characters in Joint Names

Joint names may contain dots (`.`) for left/right suffixes:
- Example: `shoulder.L`, `f_index.01.R`
- CSV column: `joint_shoulder.L_angle_0`

The dot is **not** a separator, it's part of the joint name.

## Complete Example

For a skeleton with:
- Root (free-flyer)
- `spine1` (spherical)
- `elbow.L` (revolute, axis=[0,0,1])

CSV header:
```csv
tracker_frame_idx,timestamp,root_position_x,root_position_y,root_position_z,root_quaternion_w,root_quaternion_x,root_quaternion_y,root_quaternion_z,root_velocity_x,root_velocity_y,root_velocity_z,root_angular_velocity_x,root_angular_velocity_y,root_angular_velocity_z,joint_spine1_angle_0,joint_spine1_angle_1,joint_spine1_angle_2,joint_spine1_velocity_0,joint_spine1_velocity_1,joint_spine1_velocity_2,joint_elbow.L_angle_0,joint_elbow.L_velocity_0
```

Sample data row:
```csv
100,1.667,5.21,4.02,0.82,0.43,0.90,0.02,0.06,0.25,0.009,0.12,-0.002,-0.040,0.027,0.449,0.262,-0.262,-0.007,0,0,1.571,0.002
```

Interpretation:
- Frame 100 at 1.667 seconds
- Root at (5.21, 4.02, 0.82) meters
- Root quaternion [0.43, 0.90, 0.02, 0.06] (normalized)
- Root moving at 0.25 m/s in X, 0.009 m/s in Y, 0.12 m/s in Z
- Root rotating slowly (small angular velocities)
- Spine1 axis-angle: [0.449, 0.262, -0.262] rad → 29° around [0.766, 0.447, -0.447]
- Left elbow at 1.571 rad (90°), opening at 0.002 rad/s

## Forward Kinematics Algorithm

To reconstruct 3D marker positions from state vectors:

```python
def compute_forward_kinematics(skeleton, state_vector):
    """Compute world positions of all joints and markers."""
    transforms = {}  # joint_name -> 4x4 transform matrix

    for joint in skeleton.joints_in_hierarchy_order():
        # Get parent transform (identity for root)
        if joint.parent is None:
            T_parent = np.eye(4)
        else:
            T_parent = transforms[joint.parent]

        # Build local transform
        T_local = np.eye(4)

        if joint.is_root:
            # Root: absolute position + quaternion
            T_local[:3, 3] = [
                state_vector['root_position_x'],
                state_vector['root_position_y'],
                state_vector['root_position_z']
            ]
            quat = [
                state_vector['root_quaternion_w'],
                state_vector['root_quaternion_x'],
                state_vector['root_quaternion_y'],
                state_vector['root_quaternion_z']
            ]
            R_quat = quaternion_to_matrix(quat)
            R_rest = euler_to_matrix(joint.rest_orientation)
            T_local[:3, :3] = R_quat @ R_rest

        else:
            # Non-root: offset + rest + animation
            T_local[:3, 3] = joint.offset
            R_rest = euler_to_matrix(joint.rest_orientation)
            R_anim = get_animation_rotation(joint, state_vector)
            T_local[:3, :3] = R_rest @ R_anim

        # Compose with parent
        transforms[joint.name] = T_parent @ T_local

    # Marker positions
    marker_positions = {}
    for marker in skeleton.markers:
        T_joint = transforms[marker.parent_joint]
        pos_local = np.array([*marker.offset, 1.0])
        pos_world = T_joint @ pos_local
        marker_positions[marker.name] = pos_world[:3]

    return marker_positions

def get_animation_rotation(joint, state_vector):
    """Extract animation rotation from state vector."""
    if joint.type == 'revolute':
        angle = state_vector[f'joint_{joint.name}_angle_0']
        return axis_angle_to_matrix(joint.axis * angle)

    elif joint.type in ['ball', 'spherical']:
        axis_angle = np.array([
            state_vector[f'joint_{joint.name}_angle_0'],
            state_vector[f'joint_{joint.name}_angle_1'],
            state_vector[f'joint_{joint.name}_angle_2']
        ])
        return axis_angle_to_matrix(axis_angle)

    else:  # Fixed
        return np.eye(3)
```

## Data Properties

### Frame Coverage

- **Continuous frames**: `tracker_frame_idx` typically increases by 1 each row
- **Gaps**: Missing frames indicate tracking failure or dropped frames
- **Interpolation**: Consumers may need to interpolate missing frames

### Numeric Precision

- Angles: Typically 6-8 decimal places (sub-millimeter precision in 3D)
- Positions: Meter-level with millimeter precision
- Very small values: Scientific notation (e.g., `1.234e-09`)

### Value Ranges

- **Quaternions**: Normalized, but may accumulate tiny errors (re-normalize if needed)
- **Joint angles**: Within limits specified in skeleton YAML (may violate during tracking instability)
- **Velocities**: Unbounded, but physically realistic values for human motion:
  - Linear: < 10 m/s
  - Angular: < 50 rad/s

### Missing Data

Some implementations use:
- `NaN` for uninitialized/unavailable values
- `0.0` for default/rest values

Check specific tracker implementation for conventions.

## Relationship to Pinocchio Configuration

The tracker uses Pinocchio for kinematics. The state vector maps to Pinocchio's configuration vector `q`:

### Configuration Vector (`q`)

| Joint Type | Pinocchio Representation | Config Size | State Vector Mapping |
|------------|-------------------------|-------------|---------------------|
| Root (free-flyer) | Position (3) + Quaternion [x,y,z,w] | 7 | `root_position_{x,y,z}`, `root_quaternion_{w,x,y,z}` (reordered) |
| Revolute | Single angle | 1 | `joint_{name}_angle_0` |
| Spherical | Quaternion [x,y,z,w] | 4 | Convert axis-angle to quaternion |
| Fixed | — | 0 | — |

**Critical notes**:
- Pinocchio quaternion order: `[x, y, z, w]` (scalar-last)
- State vector quaternion order: `[w, x, y, z]` (scalar-first)
- Must reorder when passing to Pinocchio!
- Spherical joints: State uses axis-angle, Pinocchio uses quaternion - conversion required

### Velocity Vector (`v`)

Similar structure but all components are velocities (6 for root, 3 for spherical, 1 for revolute).

## Implementation References

### C++ (Tracker)

- **State storage**: `cpp/src/core/state.cpp` and `cpp/include/posetrak/core/state.hpp`
- **Pinocchio conversion**: `cpp/src/kinematics/forward_kinematics.cpp` (`state_to_config()` function)
- **CSV export**: Tracker outputs state vectors directly

### Python (Visualization)

- **Loading**: `pandas.read_csv()` with automatic column detection
- **FK computation**: `scripts/visualize_tracking_results.py` (`compute_joint_transforms()`)

## Debugging Tips

### Visualizing State

1. **Check quaternion normalization**: `w² + x² + y² + z² ≈ 1`
2. **Plot joint angles over time**: Should be smooth curves
3. **Check velocity consistency**: Numerical derivative of positions should match velocities
4. **Validate against limits**: Angles should stay within skeleton YAML limits

### Common Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Skeleton in wrong position | Incorrect root position/orientation | Check coordinate system, quaternion order |
| Limbs pointing wrong way | Mismatched Euler angle convention | Verify `orientation` computation (Rx*Ry*Rz) |
| Discontinuous motion | Dropped frames, tracking failure | Interpolate missing frames |
| Impossibly large angles | Units mismatch (degrees vs radians) | State vectors always use radians |
| Jittery motion | Noise in state estimates | Apply smoothing filter |

## See Also

- [Skeleton YAML Format](skeleton-format.md) - Defines joint hierarchy and rest poses
- `cpp/src/kinematics/forward_kinematics.cpp` - Reference implementation
- `scripts/visualize_tracking_results.py` - Python visualization example
