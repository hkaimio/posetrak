"""
Export posetrak tracking results to BVH format.

Usage:
    uv run scripts/export_bvh.py <tracking_dir> --skeleton <skel.yaml> [options]

The HIERARCHY uses joint offsets from the rest pose (local translations only —
BVH does not store rotations in the hierarchy).  Frame 0 is always the skeleton
rest pose; subsequent frames carry the tracking result.

Rotation convention
-------------------
All BVH channels use ZXY intrinsic Euler angles (Zrotation Xrotation Yrotation),
which is the most broadly compatible order across Blender, MotionBuilder and Unity.

For each non-root joint the BVH local rotation at frame t is:

    R_bvh = R_rest  ·  R_tracking(t)

where
  R_rest       = rest-pose rotation from the skeleton YAML `orientation` field
                 (ZYX intrinsic Euler [z, y, x] → R = Rx(x)·Ry(y)·Rz(z))
  R_tracking   = axis-angle rotation from joint_angles.csv (3-vector, magnitude
                 = angle in radians, direction = axis; 0-vector → identity)

The root joint encodes the full state (position + quaternion) from root_pose.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Maths helpers
# ---------------------------------------------------------------------------

def axis_angle_to_matrix(vec: np.ndarray) -> np.ndarray:
    """Convert a 3-vector axis-angle to a 3×3 rotation matrix."""
    angle = float(np.linalg.norm(vec))
    if angle < 1e-10:
        return np.eye(3)
    axis = vec / angle
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


def quat_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Convert unit quaternion (w, x, y, z) to 3×3 rotation matrix."""
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)    ],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)    ],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def zyx_euler_to_matrix(z: float, y: float, x: float) -> np.ndarray:
    """ZYX intrinsic Euler angles (radians) → rotation matrix R = Rx·Ry·Rz."""
    cz, sz = math.cos(z), math.sin(z)
    cy, sy = math.cos(y), math.sin(y)
    cx, sx = math.cos(x), math.sin(x)
    Rz = np.array([[cz, -sz, 0], [sz,  cz, 0], [0, 0, 1]])
    Ry = np.array([[cy,  0, sy], [0,    1,  0], [-sy, 0, cy]])
    Rx = np.array([[1,   0,  0], [0,   cx, -sx], [0, sx,  cx]])
    return Rx @ Ry @ Rz


def matrix_to_zxy_euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    """
    Decompose 3×3 rotation matrix into ZXY intrinsic Euler angles in degrees.

    Convention: R = Rz(α) · Rx(β) · Ry(γ)

    Returns (alpha_deg, beta_deg, gamma_deg) = (Z, X, Y) angles in degrees.
    Handles gimbal lock (β ≈ ±90°) by setting γ = 0.
    """
    # R[2,1] = sin(β)
    sin_beta = float(np.clip(R[2, 1], -1.0, 1.0))
    beta = math.asin(sin_beta)
    cos_beta = math.cos(beta)

    if abs(cos_beta) > 1e-6:
        alpha = math.atan2(-R[0, 1], R[1, 1])   # Zrotation
        gamma = math.atan2(-R[2, 0], R[2, 2])   # Yrotation
    else:
        # Gimbal lock: fix γ = 0, absorb into α
        gamma = 0.0
        alpha = math.atan2(R[1, 0], R[0, 0])

    return math.degrees(alpha), math.degrees(beta), math.degrees(gamma)


# ---------------------------------------------------------------------------
# Skeleton loading
# ---------------------------------------------------------------------------

class Joint:
    __slots__ = ("name", "parent", "joint_type", "offset", "rest_rot",
                 "bone_tip_offset", "children")

    def __init__(self, name: str, parent: str | None, joint_type: str,
                 offset: np.ndarray, rest_rot: np.ndarray,
                 bone_tip_offset: np.ndarray | None):
        self.name = name
        self.parent = parent
        self.joint_type = joint_type          # 'root', 'ball', 'revolute'
        self.offset = offset                  # local translation from parent (m)
        self.rest_rot = rest_rot              # 3×3 rest rotation matrix
        self.bone_tip_offset = bone_tip_offset  # end-site offset (m) or None
        self.children: list[str] = []


def load_skeleton(yaml_path: Path) -> tuple[dict[str, Joint], str]:
    """Return (joints_by_name, root_name)."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    joints: dict[str, Joint] = {}
    root_name = None

    for jd in data["joints"]:
        name = jd["name"]
        parent = jd.get("parent") or None
        jtype = jd.get("type", "ball")

        offset_raw = jd.get("offset") or [0.0, 0.0, 0.0]
        offset = np.array(offset_raw, dtype=float)

        # rest orientation: ZYX intrinsic Euler [z, y, x] in radians
        ori_raw = jd.get("orientation") or [0.0, 0.0, 0.0]
        oz, oy, ox = float(ori_raw[0]), float(ori_raw[1]), float(ori_raw[2])
        rest_rot = zyx_euler_to_matrix(oz, oy, ox)

        tip_raw = jd.get("bone_tip_offset")
        bone_tip = np.array(tip_raw, dtype=float) if tip_raw else None

        joints[name] = Joint(name, parent, jtype, offset, rest_rot, bone_tip)
        if parent is None:
            root_name = name

    # Wire up children
    for j in joints.values():
        if j.parent and j.parent in joints:
            joints[j.parent].children.append(j.name)

    if root_name is None:
        raise ValueError("No root joint found in skeleton YAML")
    return joints, root_name


# ---------------------------------------------------------------------------
# Tracking data loading
# ---------------------------------------------------------------------------

def load_root_pose(csv_path: Path) -> dict[int, np.ndarray]:
    """Return {frame: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]}."""
    result: dict[int, np.ndarray] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            result[frame] = np.array([
                float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"]),
                float(row["quat_w"]), float(row["quat_x"]),
                float(row["quat_y"]), float(row["quat_z"]),
            ])
    return result


def load_joint_angles(csv_path: Path) -> dict[int, dict[str, np.ndarray]]:
    """Return {frame: {joint_name: np.array([ax, ay, az])}}."""
    result: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            name = row["joint_name"]
            result[frame][name] = np.array([
                float(row["angle_x"]),
                float(row["angle_y"]),
                float(row["angle_z"]),
            ])
    return dict(result)


# ---------------------------------------------------------------------------
# BVH writing
# ---------------------------------------------------------------------------

def _scale(v: np.ndarray, unit: str) -> np.ndarray:
    return v * 100.0 if unit == "cm" else v


def _write_hierarchy(
    f,
    joints: dict[str, Joint],
    root_name: str,
    unit: str,
) -> None:
    f.write("HIERARCHY\n")
    _write_joint_recursive(f, joints, root_name, unit, depth=0, is_root=True)
    f.write("\n")


def _write_joint_recursive(
    f,
    joints: dict[str, Joint],
    name: str,
    unit: str,
    depth: int,
    is_root: bool,
) -> None:
    indent = "\t" * depth
    j = joints[name]
    offset = _scale(j.offset, unit)

    if is_root:
        f.write(f"{indent}ROOT {name}\n")
    else:
        f.write(f"{indent}JOINT {name}\n")

    f.write(f"{indent}{{\n")
    f.write(f"{indent}\tOFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")

    if is_root:
        f.write(f"{indent}\tCHANNELS 6 Xposition Yposition Zposition"
                f" Zrotation Xrotation Yrotation\n")
    else:
        f.write(f"{indent}\tCHANNELS 3 Zrotation Xrotation Yrotation\n")

    if not j.children:
        # End site
        if j.bone_tip_offset is not None:
            tip = _scale(j.bone_tip_offset, unit)
        else:
            # Fallback: 10 cm / 0.1 m along local Y
            tip = _scale(np.array([0.0, 0.1, 0.0]), unit)
        f.write(f"{indent}\tEnd Site\n")
        f.write(f"{indent}\t{{\n")
        f.write(f"{indent}\t\tOFFSET {tip[0]:.6f} {tip[1]:.6f} {tip[2]:.6f}\n")
        f.write(f"{indent}\t}}\n")
    else:
        for child_name in sorted(j.children):
            _write_joint_recursive(f, joints, child_name, unit, depth + 1,
                                   is_root=False)

    f.write(f"{indent}}}\n")


def _rotation_for_frame(
    joint: Joint,
    angles: np.ndarray | None,
) -> tuple[float, float, float]:
    """
    Compute ZXY Euler angles (degrees) for a single joint at one frame.

    R_bvh = R_rest · R_tracking
    """
    R_track = axis_angle_to_matrix(angles) if angles is not None else np.eye(3)
    R_bvh = joint.rest_rot @ R_track
    return matrix_to_zxy_euler_deg(R_bvh)


def _write_motion(
    f,
    joints: dict[str, Joint],
    root_name: str,
    root_poses: dict[int, np.ndarray],
    joint_angles_by_frame: dict[int, dict[str, np.ndarray]],
    unit: str,
    fps: float,
    include_rest_frame: bool,
    start_frame: int | None,
    end_frame: int | None,
) -> None:
    all_frames = sorted(root_poses.keys())
    if start_frame is not None:
        all_frames = [f for f in all_frames if f >= start_frame]
    if end_frame is not None:
        all_frames = [f for f in all_frames if f <= end_frame]

    # DFS order — must match the HIERARCHY section traversal exactly
    dfs_order = _dfs_order(joints, root_name)

    frame_time = 1.0 / fps
    n_frames = len(all_frames) + (1 if include_rest_frame else 0)

    f.write("MOTION\n")
    f.write(f"Frames: {n_frames}\n")
    f.write(f"Frame Time: {frame_time:.8f}\n")

    def write_frame(root_pose_vec: np.ndarray | None,
                    angles_dict: dict[str, np.ndarray]) -> None:
        values: list[float] = []

        for name in dfs_order:
            jnt = joints[name]
            if name == root_name:
                if root_pose_vec is not None:
                    pos = _scale(root_pose_vec[:3], unit)
                    qw, qx, qy, qz = root_pose_vec[3:]
                    R_root = quat_to_matrix(qw, qx, qy, qz)
                    rz, rx, ry = matrix_to_zxy_euler_deg(R_root)
                else:
                    # Rest frame: use skeleton offset as position, identity rotation
                    pos = _scale(jnt.offset, unit)
                    rz, rx, ry = matrix_to_zxy_euler_deg(jnt.rest_rot)
                values.extend([pos[0], pos[1], pos[2], rz, rx, ry])
            else:
                angles = angles_dict.get(name)
                rz, rx, ry = _rotation_for_frame(jnt, angles)
                values.extend([rz, rx, ry])

        f.write(" ".join(f"{v:.6f}" for v in values) + "\n")

    # Frame 0: rest pose
    if include_rest_frame:
        # All joint angles = 0 → R_bvh = R_rest for every joint
        write_frame(None, {})

    # Tracking frames
    for frame_idx in all_frames:
        rp = root_poses[frame_idx]
        angles = joint_angles_by_frame.get(frame_idx, {})
        write_frame(rp, angles)


def _dfs_order(joints: dict[str, Joint], root_name: str) -> list[str]:
    """Return joint names in DFS pre-order with sorted children.

    This MUST match the order joints appear in the HIERARCHY section
    (which is also written by DFS), because BVH motion channels are
    laid out in hierarchy-appearance order.
    """
    order: list[str] = []

    def visit(name: str) -> None:
        order.append(name)
        for child in sorted(joints[name].children):
            visit(child)

    visit(root_name)
    return order


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export posetrak tracking results to BVH.")
    parser.add_argument("tracking_dir", type=Path,
                        help="Directory containing root_pose.csv and joint_angles.csv")
    parser.add_argument("--skeleton", "-s", type=Path, required=True,
                        help="Skeleton YAML file")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output .bvh file (default: <tracking_dir>/tracking.bvh)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Frame rate (default: auto-detect from timestamps)")
    parser.add_argument("--units", choices=["m", "cm"], default="m",
                        help="Position units in output BVH (m for Blender, cm for "
                             "most other tools; default: m)")
    parser.add_argument("--no-rest-frame", action="store_true",
                        help="Omit frame 0 rest pose (not recommended)")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="First tracking frame to export (1-based)")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="Last tracking frame to export (1-based, inclusive)")
    args = parser.parse_args()

    tracking_dir: Path = args.tracking_dir
    root_pose_csv = tracking_dir / "root_pose.csv"
    joint_angles_csv = tracking_dir / "joint_angles.csv"

    for p in (root_pose_csv, joint_angles_csv, args.skeleton):
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    output: Path = args.output or (tracking_dir / "tracking.bvh")

    print(f"Loading skeleton: {args.skeleton}")
    joints, root_name = load_skeleton(args.skeleton)
    print(f"  {len(joints)} joints, root: {root_name}")

    print(f"Loading root pose: {root_pose_csv}")
    root_poses = load_root_pose(root_pose_csv)
    print(f"  {len(root_poses)} frames")

    print(f"Loading joint angles: {joint_angles_csv}")
    joint_angles = load_joint_angles(joint_angles_csv)

    # Auto-detect FPS from timestamp difference
    fps = args.fps
    if fps is None:
        sorted_frames = sorted(root_poses.keys())
        if len(sorted_frames) >= 2:
            # Read timestamps from CSV for two consecutive frames
            with open(root_pose_csv) as f:
                reader = csv.DictReader(f)
                ts = {int(r["frame"]): float(r["timestamp"]) for r in reader}
            f1, f2 = sorted_frames[0], sorted_frames[1]
            dt = ts[f2] - ts[f1]
            fps = round(1.0 / dt) if dt > 0 else 120.0
        else:
            fps = 120.0
    print(f"  FPS: {fps}, units: {args.units}")

    print(f"Writing: {output}")
    with open(output, "w") as f:
        _write_hierarchy(f, joints, root_name, args.units)
        _write_motion(
            f, joints, root_name,
            root_poses, joint_angles,
            unit=args.units,
            fps=fps,
            include_rest_frame=not args.no_rest_frame,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )

    print("Done.")


if __name__ == "__main__":
    main()
