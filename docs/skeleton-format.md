# Skeleton YAML Format

## Overview

The skeleton YAML file defines the hierarchical structure of the articulated skeleton used for motion tracking. It specifies joints (nodes in the kinematic tree), markers (observation points), and optional grouping for modular tracking.

## File Structure

```yaml
name: "skeleton_name"
units: "meters"  # Optional, documentation only

joints:
  - name: "joint_name"
    type: "joint_type"
    parent: "parent_joint_name"  # null for root joint
    offset: [x, y, z]
    orientation: [z, y, x]  # ZYX Euler angles in radians
    bone_tip_offset: [x, y, z]
    limits: ...  # Joint-type specific
    axis: [x, y, z]  # For revolute joints only
    group: "group_name"  # Optional

markers:
  - name: "marker_name"
    parent: "joint_name"
    offset: [x, y, z]
    openpose_keypoint: 5  # Optional COCO keypoint ID

groups:
  - name: "group_name"
    optional: true
    depends_on: "other_group"
    joints: ["joint1", "joint2", ...]
    markers: ["marker1", "marker2", ...]
    freeflyer_joint: "joint_name"  # Optional, hierarchical solver only
    ref_marker: "marker_name"      # Optional, hierarchical solver only
```

## Joint Definitions

### Joint Types

| Type | Description | Config DOF | Velocity DOF | State Representation |
|------|-------------|------------|--------------|---------------------|
| `root` or `fixed` | Fixed/root joint | 0 (or 7 for root) | 0 (or 6 for root) | Root: position (3) + quaternion (4) |
| `revolute` | Single-axis rotation | 1 | 1 | Single angle (radians) |
| `ball` or `spherical` | 3-axis rotation | 4 (quaternion) | 3 | Axis-angle (3 components) |

**Note**: The root joint is special - it represents the base of the skeleton with 6 DOF (3 translational + 3 rotational). In Pinocchio, this is implemented as a free-flyer joint (7 config DOF due to quaternion, 6 velocity DOF).

### Required Fields

- **`name`** (string): Unique identifier for the joint
- **`type`** (string): One of: `root`, `fixed`, `revolute`, `ball`, `spherical`
- **`parent`** (string or null): Name of parent joint; `null` for root joint
- **`offset`** ([x, y, z]): Translation from parent joint to this joint's origin, expressed in parent's local frame (meters)

### Optional Fields

#### `orientation` - Rest Pose Orientation

**Format**: `[z, y, x]` - Three Euler angles in radians

**Critical Implementation Details**:
- Array is indexed as `[0]=z, [1]=y, [2]=x`
- Rotation matrix is computed as **`R = Rx(x) * Ry(y) * Rz(z)`** (extrinsic XYZ order)
- This is equivalent to applying rotations in order: X-axis, then Y-axis, then Z-axis, all in the parent frame
- Defines the joint's local coordinate frame at rest (zero animation angles)
- Animation rotations from `state_vectors.csv` are applied **relative to this rest orientation**

**Example**:
```yaml
orientation: [-1.2681, 0.1232, -1.5346]
# z = -1.2681 rad, y = 0.1232 rad, x = -1.5346 rad
# Rotation: Rx(-1.5346) * Ry(0.1232) * Rz(-1.2681)
```

**Mathematical Expansion**:
```
R = Rx(x) * Ry(y) * Rz(z)
  = [1   0    0  ]   [cy  0  sy]   [cz -sz  0]
    [0  cx  -sx]   [0   1   0]   [sz  cz  0]
    [0  sx   cx]   [-sy 0  cy]   [0   0   1]

  = [      cy*cz,       -cy*sz,        sy]
    [sx*sy*cz+cx*sz, -sx*sy*sz+cx*cz, -sx*cy]
    [-cx*sy*cz+sx*sz, cx*sy*sz+sx*cz,  cx*cy]
```

#### `bone_tip_offset` - Visualization Geometry

**Format**: `[x, y, z]` - Vector from joint origin to bone tip, in joint's local frame

**Purpose**: Defines the visual representation of the bone segment
- Starting point: Joint origin
- Ending point: Joint origin + `bone_tip_offset` (transformed to world frame)
- Used by visualization tools to draw skeleton bones
- Does **not** affect kinematics or marker positions
- Typically points toward the child joint but can be artistic

**Example**:
```yaml
bone_tip_offset: [0.0, 0.323, 0.0]  # Bone extends 32.3cm along Y-axis
```

#### `limits` - Joint Range of Motion

Specifies allowed rotation ranges for active joints.

**For revolute joints** (single axis):
```yaml
limits: [min_angle, max_angle]  # Radians
# Example:
limits: [-1.57, 1.57]  # ±90 degrees
```

**For spherical joints** (3 axes):
```yaml
limits:
  x: [min_x, max_x]
  y: [min_y, max_y]
  z: [min_z, max_z]
# Example:
limits:
  x: [-0.523, 0.523]  # ±30 degrees
  y: [-0.785, 0.785]  # ±45 degrees
  z: [-1.57, 1.57]    # ±90 degrees
```

#### `axis` - Revolute Joint Rotation Axis

**Format**: `[x, y, z]` - Unit vector defining rotation axis in joint's local frame

**Required for**: `revolute` joints
**Example**:
```yaml
type: revolute
axis: [0, 1, 0]  # Rotates around local Y-axis
limits: [-1.57, 1.57]
```

**Common axes**:
- `[1, 0, 0]` - X-axis (sagittal plane rotation)
- `[0, 1, 0]` - Y-axis (frontal plane rotation)
- `[0, 0, 1]` - Z-axis (transverse plane rotation)

#### `group` - Modular Tracking Groups

**Format**: String identifier

**Purpose**: Enables hierarchical tracking where groups can be toggled independently
- Used by tracker to activate/deactivate subsets of skeleton
- Enables modular estimation (e.g., body without fingers)
- Useful for debugging and performance optimization

## Marker Definitions

Markers define observation points on the skeleton that correspond to physical tracking markers or keypoints.

### Required Fields

- **`name`** (string): Unique marker identifier
- **`parent`** (string): Joint name to which marker is attached
- **`offset`** ([x, y, z]): Position relative to parent joint's origin, in joint's local frame

### Optional Fields

- **`openpose_keypoint`** (integer): COCO keypoint ID for 2D pose detection integration
- **`group`** (string): Assigned via groups section (see below)

### Example

```yaml
markers:
  - name: "left_shoulder_marker"
    parent: "shoulder.L"
    offset: [0.02, 0.05, 0.0]
    openpose_keypoint: 5
```

## Groups Section

Optional section for organizing joints and markers into logical groups.

### Fields

- **`name`** (string): Group identifier
- **`optional`** (boolean): If true, group can be disabled during tracking (default: true)
- **`depends_on`** (string): Name of another group this group depends on
- **`joints`** (list): Joint names in this group
- **`markers`** (list): Marker names in this group
- **`freeflyer_joint`** (string, optional): See "Hierarchical solver fields" below
- **`ref_marker`** (string, optional): See "Hierarchical solver fields" below

A joint or marker name listed here that doesn't actually exist in the
skeleton's `joints:`/`markers:` sections produces a loader warning (printed
to stderr) rather than a load failure — this is a stale-reference bug in the
file, not something the loader silently tolerates.

### Example

```yaml
groups:
  - name: "body"
    optional: false
    joints: ["hips", "spine1", "spine2", "neck1", "head"]
    markers: ["head_top", "neck_front", "chest_center"]

  - name: "left_hand_fingers"
    optional: true
    depends_on: "body"
    joints: ["f_index.01.L", "f_index.02.L", "thumb.01.L"]
    markers: ["fingertip_index.L", "fingertip_thumb.L"]
```

### Hierarchical solver fields

`freeflyer_joint` and `ref_marker` are only meaningful for a group that runs
as its own **hierarchical-solver child stage** (see
`docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md`) —
a filter that tracks a sub-region of the skeleton (e.g. a hand) with its root
held fixed at another group's already-solved joint, instead of estimating a
free-floating root of its own. Both are empty/absent for an ordinary group
like `main` that isn't wired up as a child stage.

- **`freeflyer_joint`** (string): The joint this group's child filter treats
  as its externally-supplied, fixed root — `PinocchioModelBuilder::
  build_subtree_model()`'s `freeflyer_joint_name` parameter. Must be a joint
  **outside** the group (typically the group's parent group, one level up
  the kinematic tree from the group's own topmost joint) — it is the anchor
  the child filter is fixed to, not a joint the child estimates itself.
- **`ref_marker`** (string): The marker every other marker in this group is
  measured relative to, via a `PAIR_DIFF` (pixel-difference) observation
  instead of an absolute-pixel one — `build_ref_marker_pair_observations()`'s
  reference marker. Conventionally a marker attached at (or very near) the
  freeflyer joint, and conventionally also a member of the *other* group
  that owns `freeflyer_joint` (shared markers across a parent/child group
  pair are expected and fine — see the design doc's "wrist ownership"
  section).

Example — a hand group anchored at the forearm, with the wrist marker as
its `PAIR_DIFF` reference (matches this repo's production skeletons):

```yaml
groups:
  - name: "main"
    joints: ["hips", "...", "forearm.L", "hand.L", "..."]
    markers: ["MRK-wrist.L", "..."]
    optional: false
    # No freeflyer_joint/ref_marker: "main" is not a child stage.

  - name: "HandL"
    depends_on: "main"
    joints: ["hand.L", "thumb.01.L", "f_index.01.L", "..."]
    markers: ["MRK-wrist.L", "MRK-index.L", "..."]
    freeflyer_joint: "forearm.L"  # in "main", not "HandL"
    ref_marker: "MRK-wrist.L"     # shared with "main"
```

## Coordinate System Conventions

- **Right-handed coordinates**: Standard in robotics/graphics
- **Default up-axis**: Y-up (can vary by skeleton)
- **Units**: Meters (specified in file header)
- **Frame hierarchy**: Each joint's local frame is defined relative to its parent

### Transform Chain

For any joint `J` with parent `P`:

```
T_world_J = T_world_P * Translate(offset) * Rotate(orientation) * Rotate(animation)
```

Where:
1. `T_world_P` - Parent's world transform (from root)
2. `Translate(offset)` - Joint offset in parent's frame
3. `Rotate(orientation)` - Rest orientation (Rx*Ry*Rz)
4. `Rotate(animation)` - Animation angles from state vector

For markers attached to joint `J`:
```
P_world_marker = T_world_J * [marker.offset; 1]
```

## Complete Example

```yaml
name: "SimpleHumanoid"
units: "meters"

joints:
  # Root joint (pelvis)
  - name: "hips"
    type: "root"
    parent: null
    offset: [0.0, 0.0, 0.95]  # Initial height
    bone_tip_offset: [0.0, 0.15, 0.0]

  # Spine chain
  - name: "spine1"
    type: "ball"
    parent: "hips"
    offset: [0.0, 0.15, 0.0]
    orientation: [0.0, 0.0, -0.087]  # -5° forward tilt
    bone_tip_offset: [0.0, 0.20, 0.0]
    limits:
      x: [-0.35, 0.52]
      y: [-0.26, 0.26]
      z: [-0.26, 0.26]

  # Left arm
  - name: "shoulder.L"
    type: "ball"
    parent: "spine2"
    offset: [0.18, 0.42, 0.0]
    orientation: [-1.268, 0.123, -1.535]  # Arms point sideways in rest pose
    bone_tip_offset: [0.0, 0.12, 0.0]
    limits:
      x: [-0.35, 2.6]
      y: [-1.2, 1.2]
      z: [-1.57, 0.78]

  - name: "elbow.L"
    type: "revolute"
    parent: "shoulder.L"
    offset: [0.0, 0.30, 0.0]
    axis: [0, 0, 1]  # Bend in Z-axis
    bone_tip_offset: [0.0, 0.28, 0.0]
    limits: [0.0, 2.79]  # 0-160°

markers:
  - name: "head_top"
    parent: "head"
    offset: [0.0, 0.12, 0.0]

  - name: "left_wrist"
    parent: "elbow.L"
    offset: [0.0, 0.28, 0.0]
    openpose_keypoint: 7

groups:
  - name: "upper_body"
    optional: false
    joints: ["hips", "spine1", "spine2", "neck", "head"]

  - name: "left_arm"
    optional: false
    joints: ["shoulder.L", "elbow.L"]
    markers: ["left_wrist"]
```

## Implementation Notes

### Loading in C++ (Pinocchio)

The `skeleton_loader.cpp` reads the YAML and builds a Pinocchio model:
- Root joint becomes `JointModelFreeFlyer` (6 DOF)
- Revolute joints become `JointModelRX/RY/RZ` based on axis
- Spherical joints become `JointModelSpherical` (quaternion representation)
- Rest orientation is baked into the SE3 placement of each joint frame

See `src/io/skeleton_loader.cpp` and `src/kinematics/pinocchio_model_builder.cpp` for details.

### Loading in Python (Visualization)

The visualizer script loads skeleton structure and computes forward kinematics manually:
- Parses YAML to build joint hierarchy
- Computes world transforms by walking tree from root
- Applies rest orientation first, then animation rotation
- Uses transforms to position bones and markers in 3D space

See `scripts/visualize_tracking_results.py` for implementation.

## References

- Pinocchio documentation: https://github.com/stack-of-tasks/pinocchio
- URDF format (similar concepts): http://wiki.ros.org/urdf/XML/joint
- Euler angles: https://en.wikipedia.org/wiki/Euler_angles
