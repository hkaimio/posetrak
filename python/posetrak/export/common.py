"""
Shared maths, skeleton loading, and data-loading helpers for BVH / USD export.

All symbols here are format-agnostic.  Format-specific logic (Euler
decomposition for BVH, quaternion serialisation for USD, etc.) lives in
the individual exporter modules.
"""

from __future__ import annotations

import csv
import math
import sqlite3
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


# ---------------------------------------------------------------------------
# Skeleton
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
        self.bone_tip_offset = bone_tip_offset
        self.children: list[str] = []


def load_skeleton(yaml_path: Path) -> tuple[dict[str, Joint], str]:
    """Return (joints_by_name, root_name) parsed from a skeleton YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    joints: dict[str, Joint] = {}
    root_name: str | None = None

    for jd in data["joints"]:
        name = jd["name"]
        parent = jd.get("parent") or None
        jtype = jd.get("type", "ball")

        offset_raw = jd.get("offset") or [0.0, 0.0, 0.0]
        offset = np.array(offset_raw, dtype=float)

        ori_raw = jd.get("orientation") or [0.0, 0.0, 0.0]
        oz, oy, ox = float(ori_raw[0]), float(ori_raw[1]), float(ori_raw[2])
        rest_rot = zyx_euler_to_matrix(oz, oy, ox)

        tip_raw = jd.get("bone_tip_offset")
        bone_tip = np.array(tip_raw, dtype=float) if tip_raw else None

        joints[name] = Joint(name, parent, jtype, offset, rest_rot, bone_tip)
        if parent is None:
            root_name = name

    for j in joints.values():
        if j.parent and j.parent in joints:
            joints[j.parent].children.append(j.name)

    if root_name is None:
        raise ValueError("No root joint found in skeleton YAML")
    return joints, root_name


# ---------------------------------------------------------------------------
# Tracking data — CSV
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
# Tracking data — DB (DataFrame → dict converters)
# ---------------------------------------------------------------------------

def _df_to_root_poses(df) -> dict[int, np.ndarray]:
    """Convert a root_pose DataFrame to {frame: array}."""
    result: dict[int, np.ndarray] = {}
    for _, row in df.iterrows():
        frame = int(row["frame"])
        result[frame] = np.array([
            float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"]),
            float(row["quat_w"]), float(row["quat_x"]),
            float(row["quat_y"]), float(row["quat_z"]),
        ])
    return result


def _df_to_joint_angles(df) -> dict[int, dict[str, np.ndarray]]:
    """Convert a joint_angles DataFrame to {frame: {name: array}}."""
    result: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    for _, row in df.iterrows():
        frame = int(row["frame"])
        name = row["joint_name"]
        result[frame][name] = np.array([
            float(row["angle_x"]),
            float(row["angle_y"]),
            float(row["angle_z"]),
        ])
    return dict(result)


def _detect_fps_from_df(df) -> float:
    """Auto-detect FPS from the first two rows of a root_pose DataFrame."""
    if df is None or len(df) < 2:
        return 120.0
    sorted_df = df.sort_values("frame")
    ts = sorted_df["timestamp"].values
    dt = ts[1] - ts[0]
    return round(1.0 / dt) if dt > 0 else 120.0


def _load_fps_from_db(session_db: str, run_id: str) -> float:
    """Query actual_fps from capture_videos via tracking_run → observation_sequence → capture."""
    try:
        conn = sqlite3.connect(session_db, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT sv.actual_fps
            FROM tracking_runs tr
            JOIN pose_observation_sequences pos ON pos.id = tr.observation_sequence_id
            JOIN captures s ON s.id = pos.shot_id
            JOIN capture_videos sv ON sv.shot_id = s.id
            WHERE tr.id = ?
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        conn.close()
        if row and row["actual_fps"]:
            return float(row["actual_fps"])
    except Exception:
        pass
    return 120.0


# ---------------------------------------------------------------------------
# Quaternion extraction (format-agnostic, returns raw floats)
# ---------------------------------------------------------------------------

def matrix_to_quat_components(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3×3 rotation matrix to quaternion (w, x, y, z) via Shepperd's method."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(w), float(x), float(y), float(z)


# ---------------------------------------------------------------------------
# Coordinate-system helpers
# ---------------------------------------------------------------------------

def _coord_matrices(coord: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (M_track, M_rest_rot) for the chosen coordinate system.

    M_track    — applied to root positions and rotations from the tracker's
                 Z-up world frame.
    M_rest_rot — applied to the rest-frame root rotation only.

    yup: Y-up, Z-forward (Blender / Unity / Maya default).
    zup: Z-up, Y-backward (unchanged tracker frame, for 3ds Max etc.).
    """
    if coord == "yup":
        M = np.array([[1, 0,  0],
                      [0, 0,  1],
                      [0, -1, 0]], dtype=float)
        return M, np.eye(3)
    else:  # zup
        Rx90 = np.array([[1, 0,  0],
                         [0, 0, -1],
                         [0, 1,  0]], dtype=float)
        return np.eye(3), Rx90


# ---------------------------------------------------------------------------
# Tree traversal
# ---------------------------------------------------------------------------

def _dfs_order(joints: dict[str, Joint], root_name: str) -> list[str]:
    """Return joint names in DFS pre-order with children sorted alphabetically.

    The order must match however the HIERARCHY / Skeleton prim lists joints,
    so that per-frame channel arrays are consistent.
    """
    order: list[str] = []

    def visit(name: str) -> None:
        order.append(name)
        for child in sorted(joints[name].children):
            visit(child)

    visit(root_name)
    return order


# ---------------------------------------------------------------------------
# Unit scaling
# ---------------------------------------------------------------------------

def _scale(v: np.ndarray, unit: str) -> np.ndarray:
    return v * 100.0 if unit == "cm" else v
