# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""
Export posetrak tracking results to glTF 2.0 skeletal animation format.

Callable API
------------
    from posetrak.export.gltf import export_gltf

    export_gltf(
        "take1.glb",            # or .gltf for JSON + embedded base64
        session_db="session.db",
        run_id="<uuid>",
        smoothed=True,
    )

CLI wrapper lives at python/tools/export_gltf.py.

Output structure
----------------
glTF 2.0 document with:
  nodes        — one node per joint in DFS order; rest-pose TRS
  skins        — "Skeleton" skin with inverseBindMatrices
  animations   — "Take" animation with translation + rotation channels per joint

Rotation encoding
-----------------
glTF quaternion convention is [x, y, z, w]. The local rotation for each
joint at frame t is R_rest · R_tracking(t), identical to BVH and USD.

No extra Python package required — only numpy and stdlib (json, struct, base64).
"""

from __future__ import annotations

import base64
import csv
import json
import struct
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
    matrix_to_quat_components,
    quat_to_matrix,
)

# glTF GL_FLOAT component type constant
_FLOAT = 5126


# ---------------------------------------------------------------------------
# Binary-buffer builder
# ---------------------------------------------------------------------------

class _GltfBuilder:
    """Accumulates glTF buffer, bufferViews, and accessors for a single buffer."""

    def __init__(self) -> None:
        self._buf: bytearray = bytearray()
        self.buffer_views: list[dict] = []
        self.accessors: list[dict] = []

    def _align4(self) -> None:
        pad = (-len(self._buf)) % 4
        self._buf.extend(b"\x00" * pad)

    def add(
        self,
        arr: np.ndarray,
        accessor_type: str,
        *,
        add_min_max: bool = False,
    ) -> int:
        """Append *arr* (cast to float32) to the buffer; return accessor index."""
        arr = np.asarray(arr, dtype=np.float32)
        self._align4()
        byte_offset = len(self._buf)
        raw = arr.tobytes()
        self._buf.extend(raw)

        bv_idx = len(self.buffer_views)
        self.buffer_views.append(
            {"buffer": 0, "byteOffset": byte_offset, "byteLength": len(raw)}
        )

        acc: dict = {
            "bufferView": bv_idx,
            "componentType": _FLOAT,
            "count": arr.shape[0],
            "type": accessor_type,
        }
        if add_min_max:
            flat = arr.ravel().astype(float)
            acc["min"] = [float(flat.min())]
            acc["max"] = [float(flat.max())]

        acc_idx = len(self.accessors)
        self.accessors.append(acc)
        return acc_idx

    def buf_bytes(self) -> bytes:
        return bytes(self._buf)


# ---------------------------------------------------------------------------
# GLB writer
# ---------------------------------------------------------------------------

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN  = 0x004E4942


def _write_glb(output: Path, gltf_dict: dict, bin_data: bytes) -> None:
    json_bytes = json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8")
    # Both chunks must be 4-byte aligned: JSON padded with spaces, BIN with zeros
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    bin_data   += b"\x00" * ((-len(bin_data)) % 4)

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    with open(output, "wb") as f:
        f.write(struct.pack("<III", _GLB_MAGIC, 2, total))
        f.write(struct.pack("<II", len(json_bytes), _CHUNK_JSON))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(bin_data), _CHUNK_BIN))
        f.write(bin_data)


# ---------------------------------------------------------------------------
# Core glTF stage writer
# ---------------------------------------------------------------------------

def _write_gltf(
    output: Path,
    joints: dict[str, Joint],
    root_name: str,
    root_poses: dict[int, np.ndarray],
    joint_angles_by_frame: dict[int, dict[str, np.ndarray]],
    unit: str,
    fps: float,
    include_rest_frame: bool,
    start_frame: int | None,
    end_frame: int | None,
    coord: str,
) -> None:
    all_frames = sorted(root_poses.keys())
    if start_frame is not None:
        all_frames = [f for f in all_frames if f >= start_frame]
    if end_frame is not None:
        all_frames = [f for f in all_frames if f <= end_frame]

    dfs_order = _dfs_order(joints, root_name)
    M_track, M_rest_rot = _coord_matrices(coord)
    n_joints = len(dfs_order)
    joint_idx: dict[str, int] = {name: i for i, name in enumerate(dfs_order)}

    # ------------------------------------------------------------------
    # Per-frame animation arrays
    # glTF quaternion convention: [x, y, z, w]
    # ------------------------------------------------------------------
    n_rest = 1 if include_rest_frame else 0
    n_frames = n_rest + len(all_frames)

    translations = np.zeros((n_frames, n_joints, 3), dtype=np.float32)
    rotations    = np.zeros((n_frames, n_joints, 4), dtype=np.float32)

    def _fill_frame(
        tc: int,
        root_pose_vec: np.ndarray | None,
        angles: dict[str, np.ndarray],
    ) -> None:
        for j_i, name in enumerate(dfs_order):
            j = joints[name]
            if name == root_name:
                if root_pose_vec is not None:
                    t = _scale(M_track @ root_pose_vec[:3], unit)
                    qw, qx, qy, qz = root_pose_vec[3:]
                    R = M_track @ quat_to_matrix(qw, qx, qy, qz)
                else:
                    t = _scale(M_track @ j.offset, unit)
                    R = M_rest_rot @ j.rest_rot
            else:
                t = _scale(j.offset, unit)
                ang = angles.get(name)
                R_track = axis_angle_to_matrix(ang) if ang is not None else np.eye(3)
                R = j.rest_rot @ R_track

            translations[tc, j_i] = t
            w, x, y, z = matrix_to_quat_components(R)
            rotations[tc, j_i] = [x, y, z, w]  # glTF: [x,y,z,w]

    if include_rest_frame:
        _fill_frame(0, None, {})
    for k, frame_idx in enumerate(all_frames):
        _fill_frame(
            n_rest + k,
            root_poses[frame_idx],
            joint_angles_by_frame.get(frame_idx, {}),
        )

    # ------------------------------------------------------------------
    # Inverse bind matrices — world-space 4×4, column-major for glTF MAT4
    # In glTF p' = M * p (column vectors), so M[:3,:3] = R, M[:3,3] = t.
    # Column-major storage = numpy M.T.ravel().
    # ------------------------------------------------------------------
    world: dict[str, np.ndarray] = {}
    for name in dfs_order:
        j = joints[name]
        R = (M_rest_rot @ j.rest_rot) if name == root_name else j.rest_rot
        t = _scale((M_track @ j.offset) if name == root_name else j.offset, unit)
        M_local = np.eye(4, dtype=np.float64)
        M_local[:3, :3] = R
        M_local[:3, 3] = t
        world[name] = M_local if j.parent is None else (world[j.parent] @ M_local)

    ibm = np.array(
        [np.linalg.inv(world[n]).T.ravel() for n in dfs_order],
        dtype=np.float32,
    )  # shape (N, 16) — column-major per matrix

    # ------------------------------------------------------------------
    # Binary buffer — time, inverse-bind matrices, per-joint channels
    # ------------------------------------------------------------------
    gb = _GltfBuilder()
    time_arr = (np.arange(n_frames, dtype=np.float32) / fps)
    time_acc = gb.add(time_arr, "SCALAR", add_min_max=True)
    ibm_acc  = gb.add(ibm, "MAT4")

    trans_accs = [gb.add(translations[:, j_i, :], "VEC3") for j_i in range(n_joints)]
    rot_accs   = [gb.add(rotations[:, j_i, :],    "VEC4") for j_i in range(n_joints)]

    # ------------------------------------------------------------------
    # Nodes (rest-pose TRS, children by DFS index)
    # ------------------------------------------------------------------
    nodes: list[dict] = []
    for name in dfs_order:
        j = joints[name]
        R_rest = (M_rest_rot @ j.rest_rot) if name == root_name else j.rest_rot
        t_rest = _scale((M_track @ j.offset) if name == root_name else j.offset, unit)
        w, x, y, z = matrix_to_quat_components(R_rest)
        node: dict = {
            "name": name,
            "translation": [float(t_rest[0]), float(t_rest[1]), float(t_rest[2])],
            "rotation": [float(x), float(y), float(z), float(w)],
            "scale": [1.0, 1.0, 1.0],
        }
        children = [joint_idx[c] for c in sorted(j.children)]
        if children:
            node["children"] = children
        nodes.append(node)

    # ------------------------------------------------------------------
    # Animation — one translation + one rotation channel per joint
    # ------------------------------------------------------------------
    samplers: list[dict] = []
    channels: list[dict] = []
    for j_i in range(n_joints):
        t_si = len(samplers)
        samplers.append({"input": time_acc, "interpolation": "LINEAR", "output": trans_accs[j_i]})
        channels.append({"sampler": t_si, "target": {"node": j_i, "path": "translation"}})

        r_si = len(samplers)
        samplers.append({"input": time_acc, "interpolation": "LINEAR", "output": rot_accs[j_i]})
        channels.append({"sampler": r_si, "target": {"node": j_i, "path": "rotation"}})

    # ------------------------------------------------------------------
    # glTF JSON document
    # ------------------------------------------------------------------
    buf_bytes = gb.buf_bytes()

    gltf: dict = {
        "asset": {"version": "2.0", "generator": "posetrak"},
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": [0]}],
        "nodes": nodes,
        "skins": [{
            "name": "Skeleton",
            "skeleton": 0,
            "joints": list(range(n_joints)),
            "inverseBindMatrices": ibm_acc,
        }],
        "animations": [{"name": "Take", "channels": channels, "samplers": samplers}],
        "accessors": gb.accessors,
        "bufferViews": gb.buffer_views,
        "buffers": [],
    }

    if output.suffix.lower() == ".glb":
        gltf["buffers"] = [{"byteLength": len(buf_bytes)}]
        _write_glb(output, gltf, buf_bytes)
    else:
        b64 = base64.b64encode(buf_bytes).decode("ascii")
        gltf["buffers"] = [
            {
                "uri": f"data:application/octet-stream;base64,{b64}",
                "byteLength": len(buf_bytes),
            }
        ]
        output.write_text(json.dumps(gltf, indent=2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_gltf(
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
    """Export posetrak tracking results to a glTF 2.0 skeletal animation file.

    Parameters
    ----------
    output:
        Destination path. Extension controls format:
        ``.glb`` = binary (recommended), ``.gltf`` = JSON with embedded base64.
    session_db:
        Path to a session SQLite database. Requires *run_id*.
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
        Output frame rate. Auto-detected from timestamps when omitted.
    units:
        ``"m"`` (metres) or ``"cm"``.
    coord:
        ``"yup"`` (Y-up, Z-forward — Blender/Unity/Maya) or
        ``"zup"`` (unchanged tracker frame).
    smoothed:
        Export RTS-smoothed results (DB mode) or smoothed_*.csv (CSV mode).
    include_rest_frame:
        Prepend a time-code-0 rest pose frame (recommended; default True).
    start_frame:
        First tracking frame to include (1-based).
    end_frame:
        Last tracking frame to include (1-based, inclusive).
    """
    output = Path(output)

    if session_db is not None:
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp.write(data["skeleton_yaml"])
            skel_tmp_path = Path(tmp.name)

        try:
            joints, root_name = load_skeleton(skel_tmp_path)
        finally:
            skel_tmp_path.unlink(missing_ok=True)

        root_poses    = _df_to_root_poses(data["root_pose_df"])
        joint_angles  = _df_to_joint_angles(data["joint_angles_df"])

        if fps is None:
            fps = _load_fps_from_db(session_db, run_id)
            if fps == 120.0:
                fps = _detect_fps_from_df(data["root_pose_df"])

    else:
        if tracking_dir is None:
            raise ValueError("tracking_dir is required when not using session_db")
        if skeleton_path is None:
            raise ValueError("skeleton_path is required when not using session_db")

        tracking_dir  = Path(tracking_dir)
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
        root_poses   = load_root_pose(root_pose_csv)
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

    _write_gltf(
        output, joints, root_name,
        root_poses, joint_angles,
        unit=units,
        fps=fps,
        include_rest_frame=include_rest_frame,
        start_frame=start_frame,
        end_frame=end_frame,
        coord=coord,
    )
