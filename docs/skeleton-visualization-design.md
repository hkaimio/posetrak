# Skeleton Visualization Design

## Overview

Add animated skeleton visualization to the Rerun viewer, showing the articulated bone structure in 3D with proper joint transformations. This will overlay on top of the existing marker point cloud visualization.

## Data Sources

### Skeleton Structure (`skeleton.yaml`)
- **Joint hierarchy**: parent-child relationships defining the kinematic tree
- **Joint types**:
  - `root`: Fixed position with quaternion orientation
  - `revolute`: Single-axis rotation (1 DOF)
  - `ball`/`spherical`: 3-axis rotation (3 DOF)
  - `fixed`: No DOF (virtual joint for marker attachment)
- **Rest pose**: `orientation` field contains ZYX Euler angles (radians) for joint's default orientation
- **Bone geometry**:
  - `offset`: Translation from parent joint in parent's frame
  - `bone_tip_offset`: Direction and length of bone visual (from joint origin to bone tip)

### Animation Data (`state_vectors.csv`)
- **Frame sync**: `tracker_frame_idx`, `timestamp` columns
- **Root transform**:
  - Position: `root_position_x/y/z`
  - Orientation: `root_quaternion_w/x/y/z`
- **Joint angles**:
  - Revolute: `joint_<name>_angle_0` (single scalar in radians)
  - Ball/Spherical: `joint_<name>_angle_0/1/2` (3 scalars in axis-angle representation)

## Visualization Architecture

### Hierarchy in Rerun

```
points/person_{id}/
  ├─ markers/posterior           # Existing: 3D marker points
  └─ skeleton/                   # NEW: Animated skeleton
      ├─ root                     # Root joint transform
      │   ├─ bone                 # Bone visual (cylinder/rod)
      │   └─ spine1               # Child joint
      │       ├─ bone
      │       └─ spine2           # ...continues recursively
      └─ [annotation_context]     # Joint labels/styling (static)
```

### Coordinate Frame Conventions

1. **World frame**: Right-handed, Z-up (matching existing setup)
2. **Joint local frames**:
   - Defined by rest pose orientation (ZYX Euler)
   - Animated by applying joint angle rotations
3. **Bone direction**: From joint origin to `bone_tip_offset` in joint's local frame

### Transform Chain

For each joint, compute world transform by walking up hierarchy:

```
T_world_joint = T_world_parent × T_parent_joint
```

Where `T_parent_joint` is composed of:
1. **Translation**: joint `offset` (in parent's frame)
2. **Rest orientation**: `orientation` (ZYX Euler → rotation matrix)
3. **Animation rotation**: joint angles from `state_vectors.csv`
   - Root: quaternion directly
   - Revolute: rotation around joint axis by angle
   - Ball: axis-angle representation → rotation matrix

## Implementation Plan

### Phase 1: Core Infrastructure

**Add skeleton loading to visualizer:**
- Parse skeleton YAML (reuse existing Python YAML parser or write minimal parser)
- Build joint hierarchy data structures
- Map joint names to state vector columns

**Add state vector loading:**
- Parse `state_vectors.csv` header to find column indices for each joint
- Load frame-by-frame state data keyed by `tracker_frame_idx`

### Phase 2: Forward Kinematics

**Transform computation:**
```python
def compute_joint_transforms(skeleton, state_vector):
    """Compute world transforms for all joints in a frame."""
    transforms = {}  # joint_name -> 4x4 matrix

    for joint in skeleton.joints_in_hierarchy_order():
        # Get parent transform (identity for root)
        T_parent = transforms.get(joint.parent_name, np.eye(4))

        # Compose local transform
        T_local = compose_transform(
            translation=joint.offset,
            rest_rotation=euler_zyx_to_matrix(joint.rest_orientation),
            animation_rotation=get_joint_rotation(joint, state_vector)
        )

        transforms[joint.name] = T_parent @ T_local

    return transforms
```

**Rotation extraction from state vector:**
- Root: quaternion → rotation matrix
- Revolute: `angle * axis` → rotation matrix (axis defined in skeleton)
- Ball/Spherical: axis-angle `[angle_0, angle_1, angle_2]` → rotation matrix

### Phase 3: Bone Visualization

**Geometry generation:**
For each joint, create a cylindrical rod from origin to `bone_tip_offset`:

```python
def log_skeleton_bones(skeleton, transforms, frame_num, timestamp):
    """Log animated skeleton bones for one frame."""
    rr.set_time("frame", sequence=frame_num)
    rr.set_time("timestamp", timestamp=timestamp)

    for joint in skeleton.joints:
        if joint.parent_index is None:
            continue  # Skip root (no parent to draw bone to)

        # Get joint world position
        T_world = transforms[joint.name]
        joint_pos = T_world[:3, 3]

        # Compute bone tip in world frame
        bone_tip_local = joint.bone_tip_offset
        bone_tip_world = T_world @ [*bone_tip_local, 1]
        bone_tip_pos = bone_tip_world[:3]

        # Compute bone properties
        bone_vector = bone_tip_pos - joint_pos
        bone_length = np.linalg.norm(bone_vector)
        bone_radius = compute_bone_radius(bone_length)

        # Log as LineStrips3D or custom arrow/cylinder
        entity_path = f"points/person_0/skeleton/{joint.name}/bone"
        rr.log(
            entity_path,
            rr.LineStrips3D([joint_pos, bone_tip_pos],
                           radii=bone_radius,
                           colors=[180, 180, 255])  # Light blue
        )
```

**Bone radius scaling:**
```python
def compute_bone_radius(bone_length):
    """Scale bone visual thickness based on bone length."""
    # Fingers (2-5cm) -> thin (2-3mm radius)
    # Arms/legs (20-50cm) -> thick (10-15mm radius)
    return np.clip(bone_length * 0.05, 0.002, 0.015)
```

### Phase 4: Joint Markers/Coordinate Frames (Optional Enhancement)

**Visualize joint positions:**
- Small spheres at each joint origin
- Color-coded by joint type (revolute vs ball)
- Optional: show local coordinate axes for debugging

**Example:**
```python
# Show joint positions as small spheres
joint_positions = [T[:3, 3] for T in transforms.values()]
joint_colors = [joint_type_color(j.type) for j in skeleton.joints]

rr.log(
    "points/person_0/skeleton/joints",
    rr.Points3D(joint_positions,
               radii=0.015,
               colors=joint_colors)
)
```

## Integration with Existing Visualization

### Selective Export Compatibility

The skeleton visualization should respect the `--only-tracking` and `--only-cameras` flags:

- `--only-cameras`: Skip skeleton (not camera-related)
- `--only-tracking`: Include skeleton with marker points
- Default (no flags): Include everything

### Performance Considerations

**Optimization strategies:**
1. **Skip finger joints** when `--no-fingers` flag present (or based on groups)
2. **LOD (Level of Detail)**: Optionally simplify skeleton for distant views
3. **Frame decimation**: For large datasets, optionally visualize every Nth frame

### New Command-Line Options

```bash
--skeleton-bones          # Show animated skeleton bones (default: off)
--skeleton-joints         # Show joint positions as spheres (default: off)
--skeleton-axes           # Show local coordinate axes at joints (default: off)
--skeleton-groups GROUPS  # Only show skeleton for specific groups (e.g., "main,HandL")
```

## Visual Design

### Color Scheme
- **Bones**: Light blue `[180, 180, 255]` - distinct from red markers
- **Joints**:
  - Revolute: Cyan `[0, 255, 255]`
  - Ball/Spherical: Yellow `[255, 255, 0]`
  - Root: Green `[0, 255, 0]`

### Rendering Order
1. Skeleton bones (background layer, semi-transparent?)
2. Joint spheres (mid-layer)
3. Marker points (foreground, existing visualization)

This layering ensures markers are always visible on top of skeleton structure.

## Testing Strategy

1. **Static pose validation**: Load frame 0, verify joint positions match expected FK
2. **Animation continuity**: Verify smooth bone movement across frames
3. **Hierarchy correctness**: Check that child joints move with parents
4. **Edge cases**:
   - Missing state vector data for some joints
   - Joints with locked DOFs (limits min == max)
   - Very short bones (fingers) render correctly

## Future Enhancements

1. **Interactive joint selection**: Click joint to see DOF values in UI
2. **Velocity visualization**: Arrow indicators showing joint angular velocities
3. **Constraint violation indicators**: Highlight joints exceeding limits in red
4. **Multi-person support**: Handle multiple skeletons if tracking multiple people
5. **Mesh skinning**: Replace simple bones with skinned character mesh

## References

- Existing marker visualization: `visualize_tracking_results()` in `scripts/visualize_tracking_results.py`
- Skeleton loader: `src/io/skeleton_loader.cpp`
- State vector format: `tracking_tests/*/state_vectors.csv`
- Rerun entity hierarchy pattern: `docs/rerun-visualization-design.md`
