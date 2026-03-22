"""skeleton_layout.py — Parse skeleton YAML and provide state-vector indexing + FK.

This module mirrors the C++ SkeletonLayout class for use in Python analysis scripts.
It parses the skeleton YAML, assigns DOF indices to joints (including prismatic
scale-group followers), and provides:

  - decode_state_blob()     : bytes → dict with pos, quat, joint_angles, velocities
  - compute_marker_positions(): decoded state → {marker_name: world_pos_3d}
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _euler_zyx_to_rot(angles: np.ndarray) -> np.ndarray:
    """[z, y, x] radians → R = Rx(x) @ Ry(y) @ Rz(z)."""
    z, y, x = float(angles[0]), float(angles[1]), float(angles[2])
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    return np.array([
        [cy * cz,             -cy * sz,              sy   ],
        [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
        [-cx * sy * cz + sx * sz,  cx * sy * sz + sx * cz,  cx * cy],
    ])


def _axis_angle_to_rot(vec: np.ndarray) -> np.ndarray:
    """Rodrigues: vec = axis * angle → 3×3 rotation matrix."""
    angle = float(np.linalg.norm(vec))
    if angle < 1e-10:
        return np.eye(3)
    ax = vec / angle
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    x, y, z = ax
    return np.array([
        [t * x * x + c,   t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c,   t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c  ],
    ])


def _quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Unit quaternion [w, x, y, z] → 3×3 rotation matrix."""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-10:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z),  2 * (x * y - w * z),   2 * (x * z + w * y)],
        [2 * (x * y + w * z),    1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),    2 * (y * z + w * x),   1 - 2 * (x * x + y * y)],
    ])


def _axis_angle_to_quat(vec: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-angle vector → (w, x, y, z) quaternion."""
    angle = float(np.linalg.norm(vec))
    if angle < 1e-10:
        return 1.0, 0.0, 0.0, 0.0
    ax = vec / angle
    half = angle / 2.0
    s = math.sin(half)
    return math.cos(half), s * ax[0], s * ax[1], s * ax[2]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class JointInfo:
    name: str
    joint_type: str          # 'revolute', 'spherical', 'fixed', 'prismatic', 'root'
    state_index: int         # index into joint_angles portion of state vector (-1 if no slot)
    storage_dof: int         # storage slots in state vector (0 for followers / fixed)
    is_scale_follower: bool  # True if this prismatic shares a leader's state slot
    active_mask: list        # which of [x,y,z] are free DOFs (for SPHERICAL)
    axis: np.ndarray         # for revolute: rotation axis
    offset: np.ndarray       # translation from parent in parent frame
    rest_orientation: np.ndarray  # ZYX euler [z,y,x] radians
    bone_tip_offset: np.ndarray   # for visualization
    parent: Optional[str]


@dataclass
class MarkerInfo:
    name: str
    parent: str
    local_pos: np.ndarray    # 3-vector in parent joint frame
    openpose_keypoint: Optional[int]  # COCO keypoint ID


# ---------------------------------------------------------------------------
# SkeletonLayout
# ---------------------------------------------------------------------------

class SkeletonLayout:
    """Parses skeleton YAML and provides DOF indexing and FK computation."""

    def __init__(self, yaml_content: str):
        data = yaml.safe_load(yaml_content)
        self._joints: list[JointInfo] = []
        self._joint_by_name: dict[str, JointInfo] = {}
        self._markers: list[MarkerInfo] = []
        self._n_dof: int = 0
        self._root_name: str = ""

        self._parse(data)

    def _parse(self, data: dict) -> None:
        joints_data = data.get("joints", [])

        # ---- Build scale_group maps ----
        scale_group_joints: dict[str, str] = {}   # joint_name → group_name
        scale_group_limits: dict[str, list] = {}  # group_name → [min, max]
        for sg in data.get("scale_groups", []):
            gname = sg["name"]
            limits = sg.get("limits", [0.0, 1.0])
            scale_group_limits[gname] = limits
            for jn in sg.get("joints", []):
                scale_group_joints[jn] = gname

        # ---- Track scale group leaders ----
        # group_name → state_index of the leader prismatic
        scale_group_leader_idx: dict[str, int] = {}

        n_dof = 0

        for jd in joints_data:
            name = jd["name"]
            parent = jd.get("parent") or None
            jtype = jd.get("type", "ball").lower()

            offset_raw = jd.get("offset") or [0.0, 0.0, 0.0]
            offset = np.array(offset_raw, dtype=float)

            ori_raw = jd.get("orientation") or [0.0, 0.0, 0.0]
            rest_orientation = np.array(ori_raw, dtype=float)

            tip_raw = jd.get("bone_tip_offset")
            bone_tip = np.array(tip_raw, dtype=float) if tip_raw else np.zeros(3)

            axis_raw = jd.get("axis")
            axis = np.array(axis_raw, dtype=float) if axis_raw else np.array([1.0, 0.0, 0.0])

            # ---- Insert prismatic for scale group if applicable ----
            if name in scale_group_joints and parent is not None:
                gname = scale_group_joints[name]
                pris_name = f"prismatic_{name}"
                if gname not in scale_group_leader_idx:
                    # First time: this is the leader
                    leader_idx = n_dof
                    scale_group_leader_idx[gname] = leader_idx
                    pris_info = JointInfo(
                        name=pris_name,
                        joint_type="prismatic",
                        state_index=leader_idx,
                        storage_dof=1,
                        is_scale_follower=False,
                        active_mask=[True],
                        axis=np.array([1.0, 0.0, 0.0]),
                        offset=np.zeros(3),
                        rest_orientation=np.zeros(3),
                        bone_tip_offset=np.zeros(3),
                        parent=parent,
                    )
                    n_dof += 1
                else:
                    # Follower: shares the leader's state index
                    leader_idx = scale_group_leader_idx[gname]
                    pris_info = JointInfo(
                        name=pris_name,
                        joint_type="prismatic",
                        state_index=leader_idx,
                        storage_dof=0,
                        is_scale_follower=True,
                        active_mask=[True],
                        axis=np.array([1.0, 0.0, 0.0]),
                        offset=np.zeros(3),
                        rest_orientation=np.zeros(3),
                        bone_tip_offset=np.zeros(3),
                        parent=parent,
                    )
                self._joints.append(pris_info)
                self._joint_by_name[pris_name] = pris_info

            # ---- Process the joint itself ----
            if parent is None:
                # Root joint: no state slot
                jinfo = JointInfo(
                    name=name,
                    joint_type="root",
                    state_index=-1,
                    storage_dof=0,
                    is_scale_follower=False,
                    active_mask=[],
                    axis=axis,
                    offset=offset,
                    rest_orientation=rest_orientation,
                    bone_tip_offset=bone_tip,
                    parent=None,
                )
                self._root_name = name
            elif jtype in ("fixed",):
                jinfo = JointInfo(
                    name=name,
                    joint_type="fixed",
                    state_index=-1,
                    storage_dof=0,
                    is_scale_follower=False,
                    active_mask=[],
                    axis=axis,
                    offset=offset,
                    rest_orientation=rest_orientation,
                    bone_tip_offset=bone_tip,
                    parent=parent,
                )
            elif jtype == "revolute":
                jinfo = JointInfo(
                    name=name,
                    joint_type="revolute",
                    state_index=n_dof,
                    storage_dof=1,
                    is_scale_follower=False,
                    active_mask=[True],
                    axis=axis,
                    offset=offset,
                    rest_orientation=rest_orientation,
                    bone_tip_offset=bone_tip,
                    parent=parent,
                )
                n_dof += 1
            elif jtype in ("ball", "spherical"):
                # Determine active DOFs from limits
                limits = jd.get("limits", {})
                active_mask = []
                for axis_key in ("x", "y", "z"):
                    if axis_key in limits:
                        lmin, lmax = limits[axis_key]
                        locked = abs(lmax - lmin) <= 1e-4
                        active_mask.append(not locked)
                    else:
                        active_mask.append(True)
                jinfo = JointInfo(
                    name=name,
                    joint_type="spherical",
                    state_index=n_dof,
                    storage_dof=3,  # ALWAYS 3, even if some DOFs locked
                    is_scale_follower=False,
                    active_mask=active_mask,
                    axis=axis,
                    offset=offset,
                    rest_orientation=rest_orientation,
                    bone_tip_offset=bone_tip,
                    parent=parent,
                )
                n_dof += 3
            elif jtype == "prismatic":
                # Standalone prismatic (not scale-group) — treat as fixed
                jinfo = JointInfo(
                    name=name,
                    joint_type="prismatic",
                    state_index=-1,
                    storage_dof=0,
                    is_scale_follower=False,
                    active_mask=[],
                    axis=axis,
                    offset=offset,
                    rest_orientation=rest_orientation,
                    bone_tip_offset=bone_tip,
                    parent=parent,
                )
            else:
                # Unknown type — treat as fixed
                jinfo = JointInfo(
                    name=name,
                    joint_type="fixed",
                    state_index=-1,
                    storage_dof=0,
                    is_scale_follower=False,
                    active_mask=[],
                    axis=axis,
                    offset=offset,
                    rest_orientation=rest_orientation,
                    bone_tip_offset=bone_tip,
                    parent=parent,
                )

            self._joints.append(jinfo)
            self._joint_by_name[name] = jinfo

        self._n_dof = n_dof

        # ---- Parse markers ----
        for md in data.get("markers", []):
            mname = md["name"]
            mparent = md.get("parent", "")
            offset_raw = md.get("offset") or [0.0, 0.0, 0.0]
            local_pos = np.array(offset_raw, dtype=float)
            coco_id = md.get("openpose_keypoint")
            self._markers.append(MarkerInfo(
                name=mname,
                parent=mparent,
                local_pos=local_pos,
                openpose_keypoint=coco_id,
            ))

    # -----------------------------------------------------------------------
    # Public properties
    # -----------------------------------------------------------------------

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def joints(self) -> list:
        return [j for j in self._joints if j.joint_type not in ("root", "fixed")]

    @property
    def markers(self) -> list:
        return [
            {"name": m.name, "parent": m.parent, "local_pos": m.local_pos,
             "openpose_keypoint": m.openpose_keypoint}
            for m in self._markers
        ]

    def root_joint_name(self) -> str:
        return self._root_name

    # -----------------------------------------------------------------------
    # State blob decoding
    # -----------------------------------------------------------------------

    def decode_state_blob(self, blob: bytes) -> dict:
        """Decode a little-endian float64 state blob.

        Returns dict with keys:
          pos         : np.ndarray shape (3,)   root position
          axis_angle  : np.ndarray shape (3,)   root orientation as axis-angle
          quat        : np.ndarray shape (4,)   [w, x, y, z]
          joint_angles: dict {joint_name: np.ndarray shape (3,)} — angle_x/y/z
          root_vel    : np.ndarray shape (3,)
          root_angvel : np.ndarray shape (3,)
          joint_vels  : dict {joint_name: np.ndarray shape (3,)}
        """
        n = self._n_dof
        expected = (12 + 2 * n) * 8
        if len(blob) != expected:
            raise ValueError(
                f"State blob size mismatch: got {len(blob)} bytes, "
                f"expected {expected} (n_dof={n})"
            )

        values = np.frombuffer(blob, dtype="<f8")

        pos = values[0:3].copy()
        axis_angle_raw = values[3:6].copy()

        # Convert axis-angle → quaternion
        w, x, y, z = _axis_angle_to_quat(axis_angle_raw)
        quat = np.array([w, x, y, z])

        joint_angles_flat = values[6 : 6 + n].copy()
        root_vel = values[6 + n : 9 + n].copy()
        root_angvel = values[9 + n : 12 + n].copy()
        joint_vels_flat = values[12 + n : 12 + 2 * n].copy()

        # Map flat arrays to per-joint dicts
        joint_angles: dict[str, np.ndarray] = {}
        joint_vels: dict[str, np.ndarray] = {}

        for ji in self._joints:
            if ji.state_index < 0 or ji.is_scale_follower:
                continue
            if ji.joint_type in ("spherical",) or (ji.joint_type == "ball"):
                ang = joint_angles_flat[ji.state_index : ji.state_index + 3]
                vel = joint_vels_flat[ji.state_index : ji.state_index + 3]
                joint_angles[ji.name] = ang.copy()
                joint_vels[ji.name] = vel.copy()
            elif ji.joint_type in ("revolute", "prismatic"):
                a = joint_angles_flat[ji.state_index]
                v = joint_vels_flat[ji.state_index]
                joint_angles[ji.name] = np.array([a, 0.0, 0.0])
                joint_vels[ji.name] = np.array([v, 0.0, 0.0])

        return {
            "pos": pos,
            "axis_angle": axis_angle_raw,
            "quat": quat,
            "joint_angles": joint_angles,
            "root_vel": root_vel,
            "root_angvel": root_angvel,
            "joint_vels": joint_vels,
        }

    # -----------------------------------------------------------------------
    # Forward kinematics
    # -----------------------------------------------------------------------

    def compute_joint_transforms(self, decoded_state: dict) -> dict[str, np.ndarray]:
        """Compute 4×4 world transforms for all joints via FK.

        Parameters
        ----------
        decoded_state:
            Output of decode_state_blob().

        Returns
        -------
        dict mapping joint_name → 4×4 homogeneous transform (world frame).
        """
        pos = decoded_state["pos"]
        quat = decoded_state["quat"]
        joint_angles = decoded_state["joint_angles"]

        transforms: dict[str, np.ndarray] = {}

        def process(ji: JointInfo) -> np.ndarray:
            if ji.name in transforms:
                return transforms[ji.name]

            T = np.eye(4)

            if ji.joint_type == "root":
                w, x, y, z = quat
                R_quat = _quat_to_rot(w, x, y, z)
                R_rest = _euler_zyx_to_rot(ji.rest_orientation)
                T[:3, :3] = R_quat @ R_rest
                T[:3, 3] = pos
            else:
                parent_ji = self._joint_by_name.get(ji.parent)
                if parent_ji is None:
                    T_parent = np.eye(4)
                else:
                    T_parent = process(parent_ji)

                R_rest = _euler_zyx_to_rot(ji.rest_orientation)
                jtype = ji.joint_type

                if jtype == "spherical":
                    angles = joint_angles.get(ji.name, np.zeros(3))
                    R_anim = _axis_angle_to_rot(angles)
                elif jtype == "revolute":
                    angles = joint_angles.get(ji.name, np.zeros(3))
                    R_anim = _axis_angle_to_rot(ji.axis * angles[0])
                else:
                    # fixed, prismatic — identity animation
                    R_anim = np.eye(3)

                T[:3, :3] = R_rest @ R_anim
                T[:3, 3] = ji.offset
                T = T_parent @ T

            transforms[ji.name] = T
            return T

        for ji in self._joints:
            process(ji)

        return transforms

    def compute_marker_positions(self, decoded_state: dict) -> dict[str, np.ndarray]:
        """Compute world-space marker positions via FK.

        Parameters
        ----------
        decoded_state:
            Output of decode_state_blob().

        Returns
        -------
        dict mapping marker_name → world position np.ndarray shape (3,).
        """
        transforms = self.compute_joint_transforms(decoded_state)
        result: dict[str, np.ndarray] = {}

        for m in self._markers:
            T = transforms.get(m.parent)
            if T is None:
                continue
            world_pos = T[:3, :3] @ m.local_pos + T[:3, 3]
            result[m.name] = world_pos

        return result

    # -----------------------------------------------------------------------
    # Helpers for CSV-compatible output
    # -----------------------------------------------------------------------

    def decoded_to_root_pose_row(self, step: int, timestamp: float, decoded: dict) -> dict:
        """Convert decoded state to a root_pose.csv-compatible dict."""
        pos = decoded["pos"]
        quat = decoded["quat"]
        vel = decoded["root_vel"]
        angvel = decoded["root_angvel"]
        return {
            "frame": step,
            "timestamp": timestamp,
            "pos_x": pos[0], "pos_y": pos[1], "pos_z": pos[2],
            "quat_w": quat[0], "quat_x": quat[1], "quat_y": quat[2], "quat_z": quat[3],
            "vel_x": vel[0], "vel_y": vel[1], "vel_z": vel[2],
            "omega_x": angvel[0], "omega_y": angvel[1], "omega_z": angvel[2],
        }

    def decoded_to_joint_angle_rows(self, step: int, timestamp: float, decoded: dict) -> list:
        """Convert decoded state to joint_angles.csv-compatible list of dicts.

        Scale followers are skipped. Each joint produces one row per DOF group.
        """
        rows = []
        joint_angles = decoded["joint_angles"]
        joint_vels = decoded["joint_vels"]

        for ji in self._joints:
            if ji.is_scale_follower:
                continue
            if ji.state_index < 0:
                continue
            if ji.joint_type not in ("spherical", "revolute", "prismatic"):
                continue

            angles = joint_angles.get(ji.name, np.zeros(3))
            vels = joint_vels.get(ji.name, np.zeros(3))

            rows.append({
                "frame": step,
                "timestamp": timestamp,
                "joint_name": ji.name,
                "angle_x": float(angles[0]),
                "angle_y": float(angles[1]),
                "angle_z": float(angles[2]),
                "velocity_x": float(vels[0]),
                "velocity_y": float(vels[1]),
                "velocity_z": float(vels[2]),
            })

        return rows
