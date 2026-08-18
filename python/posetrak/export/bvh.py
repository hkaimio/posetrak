# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""
Export posetrak tracking results to BVH format.

Callable API
------------
    from posetrak.export.bvh import export_bvh

    export_bvh(
        "take1.bvh",
        session_db="session.db",
        run_id="<uuid>",
        smoothed=True,
    )

CLI wrapper lives at python/tools/export_bvh.py.

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

import csv
import math
import tempfile
from pathlib import Path

import numpy as np

from posetrak.export.common import (
    Joint,
    _coord_matrices,
    _detect_fps_from_df,
    _df_to_joint_angles,
    _df_to_root_poses,
    _dfs_order,
    _load_fps_from_db,
    _scale,
    axis_angle_to_matrix,
    load_joint_angles,
    load_root_pose,
    load_skeleton,
    quat_to_matrix,
)


# ---------------------------------------------------------------------------
# BVH-specific maths
# ---------------------------------------------------------------------------

def matrix_to_zxy_euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    """
    Decompose 3×3 rotation matrix into ZXY intrinsic Euler angles in degrees.

    Convention: R = Rz(α) · Rx(β) · Ry(γ)

    Returns (alpha_deg, beta_deg, gamma_deg) = (Z, X, Y) angles in degrees.
    Handles gimbal lock (β ≈ ±90°) by setting γ = 0.
    """
    sin_beta = float(np.clip(R[2, 1], -1.0, 1.0))
    beta = math.asin(sin_beta)
    cos_beta = math.cos(beta)

    if abs(cos_beta) > 1e-6:
        alpha = math.atan2(-R[0, 1], R[1, 1])   # Zrotation
        gamma = math.atan2(-R[2, 0], R[2, 2])   # Yrotation
    else:
        gamma = 0.0
        alpha = math.atan2(R[1, 0], R[0, 0])

    return math.degrees(alpha), math.degrees(beta), math.degrees(gamma)


# ---------------------------------------------------------------------------
# BVH hierarchy writing
# ---------------------------------------------------------------------------

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
        if j.bone_tip_offset is not None:
            tip = _scale(j.bone_tip_offset, unit)
        else:
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


# ---------------------------------------------------------------------------
# BVH motion writing
# ---------------------------------------------------------------------------

def _rotation_for_frame(
    joint: Joint,
    angles: np.ndarray | None,
) -> tuple[float, float, float]:
    """Compute ZXY Euler angles (degrees) for one joint at one frame.

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
    coord: str = "yup",
) -> None:
    all_frames = sorted(root_poses.keys())
    if start_frame is not None:
        all_frames = [f for f in all_frames if f >= start_frame]
    if end_frame is not None:
        all_frames = [f for f in all_frames if f <= end_frame]

    dfs_order = _dfs_order(joints, root_name)
    M_track, M_rest_rot = _coord_matrices(coord)

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
                    pos = _scale(M_track @ root_pose_vec[:3], unit)
                    qw, qx, qy, qz = root_pose_vec[3:]
                    R_root = quat_to_matrix(qw, qx, qy, qz)
                    rz, rx, ry = matrix_to_zxy_euler_deg(M_track @ R_root)
                else:
                    pos = _scale(M_track @ jnt.offset, unit)
                    rz, rx, ry = matrix_to_zxy_euler_deg(M_rest_rot @ jnt.rest_rot)
                values.extend([pos[0], pos[1], pos[2], rz, rx, ry])
            else:
                angles = angles_dict.get(name)
                rz, rx, ry = _rotation_for_frame(jnt, angles)
                values.extend([rz, rx, ry])

        f.write(" ".join(f"{v:.6f}" for v in values) + "\n")

    if include_rest_frame:
        write_frame(None, {})

    for frame_idx in all_frames:
        rp = root_poses[frame_idx]
        angles = joint_angles_by_frame.get(frame_idx, {})
        write_frame(rp, angles)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_bvh(
    output: Path | str,
    *,
    # DB mode
    session_db: str | None = None,
    run_id: str | None = None,
    person_id: int = 0,
    # CSV mode
    tracking_dir: Path | None = None,
    skeleton_path: Path | None = None,
    # Options
    fps: float | None = None,
    units: str = "m",
    coord: str = "yup",
    smoothed: bool = False,
    include_rest_frame: bool = True,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> None:
    """Export posetrak tracking results to a BVH file.

    Parameters
    ----------
    output:
        Destination path for the .bvh file.
    session_db:
        Path to a session SQLite database.  Requires *run_id*.
    run_id:
        Tracking run UUID (required with *session_db*).
    person_id:
        Person ID to export (default 0).
    tracking_dir:
        Directory with root_pose.csv and joint_angles.csv (CSV mode).
        Requires *skeleton_path*.
    skeleton_path:
        Skeleton YAML file (CSV mode).
    fps:
        Output frame rate.  Auto-detected from timestamps when omitted.
    units:
        ``"m"`` (metres, Blender default) or ``"cm"`` (most other tools).
    coord:
        ``"yup"`` (Y-up, Z-forward — Blender/Unity/Maya) or
        ``"zup"`` (unchanged tracker frame — 3ds Max etc.).
    smoothed:
        Export RTS-smoothed results (DB mode) or smoothed_*.csv (CSV mode).
    include_rest_frame:
        Prepend a frame-0 rest pose (recommended; default True).
    start_frame:
        First tracking frame to include (1-based).
    end_frame:
        Last tracking frame to include (1-based, inclusive).
    """
    output = Path(output)

    if session_db is not None:
        # ---- DB mode ----
        if run_id is None:
            raise ValueError("run_id is required when using session_db")

        from posetrak.db.load_session import load_tracking_run_data

        data = load_tracking_run_data(
            session_db, run_id,
            person_id=person_id,
            smoothed=smoothed,
        )

        if data["root_pose_df"].empty:
            raise RuntimeError("No tracking results found for the specified run")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp:
            tmp.write(data["skeleton_yaml"])
            skel_tmp_path = Path(tmp.name)

        try:
            joints, root_name = load_skeleton(skel_tmp_path)
        finally:
            skel_tmp_path.unlink(missing_ok=True)

        root_poses = _df_to_root_poses(data["root_pose_df"])
        joint_angles = _df_to_joint_angles(data["joint_angles_df"])

        if fps is None:
            fps = _load_fps_from_db(session_db, run_id)
            if fps == 120.0:
                fps = _detect_fps_from_df(data["root_pose_df"])

    else:
        # ---- CSV mode ----
        if tracking_dir is None:
            raise ValueError("tracking_dir is required when not using session_db")
        if skeleton_path is None:
            raise ValueError("skeleton_path is required when not using session_db")

        tracking_dir = Path(tracking_dir)
        skeleton_path = Path(skeleton_path)
        root_pose_csv = tracking_dir / (
            "smoothed_root_pose.csv" if smoothed else "root_pose.csv"
        )
        joint_angles_csv = tracking_dir / (
            "smoothed_joint_angles.csv" if smoothed else "joint_angles.csv"
        )

        for p in (root_pose_csv, joint_angles_csv, skeleton_path):
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")

        joints, root_name = load_skeleton(skeleton_path)
        root_poses = load_root_pose(root_pose_csv)
        joint_angles = load_joint_angles(joint_angles_csv)

        if fps is None:
            sorted_frames = sorted(root_poses.keys())
            if len(sorted_frames) >= 2:
                with open(root_pose_csv) as f:
                    reader = csv.DictReader(f)
                    ts = {int(r["frame"]): float(r["timestamp"]) for r in reader}
                f1, f2 = sorted_frames[0], sorted_frames[1]
                dt = ts[f2] - ts[f1]
                fps = round(1.0 / dt) if dt > 0 else 120.0
            else:
                fps = 120.0

    with open(output, "w") as f:
        _write_hierarchy(f, joints, root_name, units)
        _write_motion(
            f, joints, root_name,
            root_poses, joint_angles,
            unit=units,
            fps=fps,
            include_rest_frame=include_rest_frame,
            start_frame=start_frame,
            end_frame=end_frame,
            coord=coord,
        )
