"""
Export posetrak tracking results to USD skeletal animation format.

Callable API
------------
    from posetrak.export.usd import export_usd

    export_usd(
        "take1.usda",           # or .usdc for binary
        session_db="session.db",
        run_id="<uuid>",
        smoothed=True,
    )

CLI wrapper lives at python/tools/export_usd.py.

Output USD structure
--------------------
    /Root               UsdSkelRoot  (stage default prim)
      /Root/Skel        UsdSkelSkeleton
                          joints          — DFS-ordered joint paths
                          restTransforms  — local-space 4×4 at rest pose
                          bindTransforms  — world-space 4×4 at rest pose
      /Root/Anim        UsdSkelAnimation (bound to /Root/Skel)
                          rotations       — quatf[] per frame, all joints
                          translations    — point3f[] per frame, all joints

Rotation encoding
-----------------
Rotations are stored as quaternions — no Euler decomposition, no gimbal lock.
The local rotation for each joint at frame t is:

    R = R_rest · R_tracking(t)

which is identical to the BVH convention, but stored directly as a quaternion.

Requires the 'usd-core' package:  uv pip install usd-core
"""

from __future__ import annotations

import csv
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


# ---------------------------------------------------------------------------
# USD helpers
# ---------------------------------------------------------------------------

def _check_usd() -> None:
    try:
        import pxr  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "USD export requires the 'usd-core' package.\n"
            "Install it with:  uv pip install usd-core"
        ) from exc


def _make_local_xform_np(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """4×4 local transform for USD row-vector convention (p' = p * M).

    M[:3, :3] = R^T  so that p_out[j] = sum_i p_in[i] * M[i, j] = (R*p_in)[j].
    M[3, :3] = t     (translation in the last row).
    """
    M = np.zeros((4, 4))
    M[:3, :3] = R.T
    M[3, :3] = t
    M[3, 3] = 1.0
    return M


def _np_to_gf_matrix4d(M: np.ndarray):
    from pxr import Gf
    return Gf.Matrix4d(tuple(tuple(float(v) for v in row) for row in M))


def _rotation_to_quatf(R: np.ndarray):
    from pxr import Gf
    w, x, y, z = matrix_to_quat_components(R)
    return Gf.Quatf(w, Gf.Vec3f(x, y, z))


def _sanitize_token(name: str) -> str:
    """Map a joint name to a valid USD SdfPath identifier.

    SdfPath components must match [A-Za-z_][A-Za-z0-9_]*.
    Blender-derived skeletons commonly use dots (e.g. 'shoulder.R',
    'f_pinky.01.R') which are illegal in SdfPath and cause parse errors.
    """
    s = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if s and s[0].isdigit():
        s = "_" + s
    return s or "_joint"


def _build_joint_paths(joints: dict[str, Joint], root_name: str) -> dict[str, str]:
    """Return {joint_name: USD skel path}, e.g. 'Hips', 'Hips/Spine', 'Hips/Spine/Chest'."""
    paths: dict[str, str] = {}

    def visit(name: str, parent_path: str) -> None:
        token = _sanitize_token(name)
        path = token if not parent_path else f"{parent_path}/{token}"
        paths[name] = path
        for child in sorted(joints[name].children):
            visit(child, path)

    visit(root_name, "")
    return paths


def _build_bind_transforms(
    joints: dict[str, Joint],
    dfs_order: list[str],
    root_name: str,
    M_track: np.ndarray,
    M_rest_rot: np.ndarray,
    unit: str,
) -> list[np.ndarray]:
    """World-space 4×4 bind transforms (accumulated local transforms from root)."""
    world: dict[str, np.ndarray] = {}
    for name in dfs_order:
        j = joints[name]
        R = (M_rest_rot @ j.rest_rot) if name == root_name else j.rest_rot
        t = _scale((M_track @ j.offset) if name == root_name else j.offset, unit)
        local_M = _make_local_xform_np(R, t)
        world[name] = local_M if j.parent is None else (local_M @ world[j.parent])
    return [world[name] for name in dfs_order]


# ---------------------------------------------------------------------------
# USD stage writing
# ---------------------------------------------------------------------------

def _write_usd(
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
    from pxr import Gf, Usd, UsdGeom, UsdSkel, Vt

    all_frames = sorted(root_poses.keys())
    if start_frame is not None:
        all_frames = [f for f in all_frames if f >= start_frame]
    if end_frame is not None:
        all_frames = [f for f in all_frames if f <= end_frame]

    dfs_order = _dfs_order(joints, root_name)
    M_track, M_rest_rot = _coord_matrices(coord)

    # Joint path tokens for USD ("Hips", "Hips/Spine", …)
    path_map = _build_joint_paths(joints, root_name)
    joint_tokens = Vt.TokenArray([path_map[n] for n in dfs_order])

    # Rest transforms (local space, one per joint)
    rest_xforms = []
    for name in dfs_order:
        j = joints[name]
        R = (M_rest_rot @ j.rest_rot) if name == root_name else j.rest_rot
        t = _scale((M_track @ j.offset) if name == root_name else j.offset, unit)
        rest_xforms.append(_np_to_gf_matrix4d(_make_local_xform_np(R, t)))

    # Bind transforms (world space, for skinning reference)
    bind_xforms_np = _build_bind_transforms(
        joints, dfs_order, root_name, M_track, M_rest_rot, unit
    )
    bind_xforms = [_np_to_gf_matrix4d(M) for M in bind_xforms_np]

    # ---- USD stage ----
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(
        stage, UsdGeom.Tokens.y if coord == "yup" else UsdGeom.Tokens.z
    )
    UsdGeom.SetStageMetersPerUnit(stage, 1.0 if unit == "m" else 0.01)

    n_anim_frames = len(all_frames) + (1 if include_rest_frame else 0)
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(float(n_anim_frames - 1))
    stage.SetTimeCodesPerSecond(fps)

    # ---- Prims ----
    skel_root = UsdSkel.Root.Define(stage, "/Root")
    stage.SetDefaultPrim(skel_root.GetPrim())

    skeleton = UsdSkel.Skeleton.Define(stage, "/Root/Skel")
    skeleton.GetJointsAttr().Set(joint_tokens)
    skeleton.GetRestTransformsAttr().Set(Vt.Matrix4dArray(rest_xforms))
    skeleton.GetBindTransformsAttr().Set(Vt.Matrix4dArray(bind_xforms))

    animation = UsdSkel.Animation.Define(stage, "/Root/Anim")
    animation.GetJointsAttr().Set(joint_tokens)

    # Bind skeleton + animation on the SkelRoot (most DCC tools — Blender, Cascadeur —
    # resolve skel:animationSource from the SkelRoot scope, not from the Skeleton prim).
    binding = UsdSkel.BindingAPI.Apply(skel_root.GetPrim())
    binding.GetSkeletonRel().AddTarget(skeleton.GetPath())
    binding.GetAnimationSourceRel().AddTarget(animation.GetPath())

    # ---- Per-frame data ----
    def _rotations(
        root_pose_vec: np.ndarray | None,
        angles_dict: dict[str, np.ndarray],
    ) -> Vt.QuatfArray:
        quats = []
        for name in dfs_order:
            j = joints[name]
            if name == root_name:
                if root_pose_vec is not None:
                    qw, qx, qy, qz = root_pose_vec[3:]
                    R = M_track @ quat_to_matrix(qw, qx, qy, qz)
                else:
                    R = M_rest_rot @ j.rest_rot
            else:
                angles = angles_dict.get(name)
                R_track = axis_angle_to_matrix(angles) if angles is not None else np.eye(3)
                R = j.rest_rot @ R_track
            quats.append(_rotation_to_quatf(R))
        return Vt.QuatfArray(quats)

    def _translations(root_pose_vec: np.ndarray | None) -> Vt.Vec3fArray:
        vecs = []
        for name in dfs_order:
            j = joints[name]
            if name == root_name:
                raw = root_pose_vec[:3] if root_pose_vec is not None else j.offset
                pos = _scale(M_track @ raw, unit)
            else:
                # USD animation transforms are full local transforms, not incremental
                # on top of restTransforms, so non-root joints must carry their rest
                # offset here (else they'd collapse to the parent's origin).
                pos = _scale(j.offset, unit)
            vecs.append(Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])))
        return Vt.Vec3fArray(vecs)

    n_joints = len(dfs_order)
    unit_scales = Vt.Vec3hArray(
        [Gf.Vec3h(1.0, 1.0, 1.0)] * n_joints
    )

    tc = 0
    if include_rest_frame:
        animation.GetRotationsAttr().Set(_rotations(None, {}), tc)
        animation.GetTranslationsAttr().Set(_translations(None), tc)
        animation.GetScalesAttr().Set(unit_scales, tc)
        tc += 1

    for frame_idx in all_frames:
        rp = root_poses[frame_idx]
        angles = joint_angles_by_frame.get(frame_idx, {})
        animation.GetRotationsAttr().Set(_rotations(rp, angles), tc)
        animation.GetTranslationsAttr().Set(_translations(rp), tc)
        animation.GetScalesAttr().Set(unit_scales, tc)
        tc += 1

    stage.Save()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_usd(
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
    """Export posetrak tracking results to a USD skeletal animation file.

    Parameters
    ----------
    output:
        Destination path.  Extension controls format:
        ``.usda`` = ASCII (human-readable), ``.usdc`` = binary crate.
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
        ``"m"`` (metres) or ``"cm"``.
    coord:
        ``"yup"`` (Y-up, Z-forward — Blender/Unity/Maya) or
        ``"zup"`` (unchanged tracker frame — 3ds Max etc.).
    smoothed:
        Export RTS-smoothed results (DB mode) or smoothed_*.csv (CSV mode).
    include_rest_frame:
        Prepend a time-code-0 rest pose frame (recommended; default True).
    start_frame:
        First tracking frame to include (1-based).
    end_frame:
        Last tracking frame to include (1-based, inclusive).

    Raises
    ------
    ImportError
        If the ``usd-core`` package is not installed.
    """
    _check_usd()
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

    _write_usd(
        output, joints, root_name,
        root_poses, joint_angles,
        unit=units,
        fps=fps,
        include_rest_frame=include_rest_frame,
        start_frame=start_frame,
        end_frame=end_frame,
        coord=coord,
    )
