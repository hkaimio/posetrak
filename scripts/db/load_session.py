"""load_session.py — Load posetrak session data from SQLite session databases.

Provides DataFrame-oriented access to tracking results, cameras, sync points,
and observations stored in per-session SQLite files.
"""

from __future__ import annotations

import json
import sqlite3
import warnings
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .skeleton_layout import SkeletonLayout
except ImportError:
    # Fallback for scripts run directly with project root on sys.path
    from scripts.db.skeleton_layout import SkeletonLayout


# ---------------------------------------------------------------------------
# DB connection helper
# ---------------------------------------------------------------------------

def _open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ---------------------------------------------------------------------------
# Skeleton loading
# ---------------------------------------------------------------------------

def _load_skeleton_yaml_for_run(conn: sqlite3.Connection, run_id: str) -> str:
    """Return skeleton YAML string for the given run.

    Checks tracking_run_persons first (per-person override), then falls back
    to tracking_runs.skeleton_id.
    """
    # Try per-person overrides — for person_id 0
    row = conn.execute(
        "SELECT s.yaml_content FROM tracking_run_persons trp "
        "JOIN skeletons s ON s.id = trp.skeleton_id "
        "WHERE trp.run_id = ? AND trp.person_id = 0",
        (run_id,),
    ).fetchone()
    if row and row["yaml_content"]:
        return row["yaml_content"]

    row = conn.execute(
        "SELECT s.yaml_content FROM tracking_runs tr "
        "JOIN skeletons s ON s.id = tr.skeleton_id "
        "WHERE tr.id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"No tracking run found with id={run_id!r}")
    if not row["yaml_content"]:
        raise ValueError(f"Skeleton has no yaml_content for run={run_id!r}")
    return row["yaml_content"]


# ---------------------------------------------------------------------------
# Main data loader
# ---------------------------------------------------------------------------

def load_tracking_run_data(
    session_db: str,
    run_id: str,
    person_id: int = 0,
    smoothed: bool = False,
) -> dict:
    """Load and decode all tracking data for a run.

    Parameters
    ----------
    session_db : str
        Path to the session SQLite file.
    run_id : str
        UUID of the tracking run.
    person_id : int
        Person ID to load (default 0).
    smoothed : bool
        If True, load smoothed results (is_smoothed=1).

    Returns
    -------
    dict with keys:
      root_pose_df       : DataFrame matching root_pose.csv columns
      joint_angles_df    : DataFrame matching joint_angles.csv columns
      tracking_stats_df  : DataFrame with frame/timestamp/tracking_lost/num_inliers/cov_condition_number
      skeleton_yaml      : str
      marker_names       : list[str]
      n_dof              : int
      run_row            : dict of tracking_runs row data
    """
    conn = _open_db(session_db)
    try:
        run_row = conn.execute(
            "SELECT * FROM tracking_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if run_row is None:
            raise ValueError(f"No tracking run found with id={run_id!r}")

        run_dict = dict(run_row)
        marker_names: list[str] = json.loads(run_dict.get("marker_names") or "[]")

        skeleton_yaml = _load_skeleton_yaml_for_run(conn, run_id)
        layout = SkeletonLayout(skeleton_yaml)

        is_smoothed_val = 1 if smoothed else 0
        rows = conn.execute(
            "SELECT tracker_step, timestamp_s, tracking_lost, "
            "n_inlier_observations, cov_condition_number, state, cov_diag "
            "FROM tracking_results "
            "WHERE run_id = ? AND person_id = ? AND is_smoothed = ? "
            "ORDER BY tracker_step",
            (run_id, person_id, is_smoothed_val),
        ).fetchall()

        if not rows:
            warnings.warn(
                f"No tracking results found for run={run_id!r}, "
                f"person_id={person_id}, smoothed={smoothed}"
            )
            empty_df = pd.DataFrame()
            return {
                "root_pose_df": empty_df,
                "joint_angles_df": empty_df,
                "tracking_stats_df": empty_df,
                "skeleton_yaml": skeleton_yaml,
                "marker_names": marker_names,
                "n_dof": layout.n_dof,
                "run_row": run_dict,
            }

        root_pose_records = []
        joint_angle_records = []
        stats_records = []

        for row in rows:
            step = row["tracker_step"]
            ts = row["timestamp_s"]
            state_blob = row["state"]

            try:
                decoded = layout.decode_state_blob(bytes(state_blob))
            except Exception as exc:
                warnings.warn(f"Could not decode state at step {step}: {exc}")
                continue

            root_pose_records.append(
                layout.decoded_to_root_pose_row(step, ts, decoded)
            )
            joint_angle_records.extend(
                layout.decoded_to_joint_angle_rows(step, ts, decoded)
            )
            stats_records.append({
                "frame": step,
                "timestamp": ts,
                "tracking_lost": bool(row["tracking_lost"]),
                "num_inliers": row["n_inlier_observations"],
                "cov_condition_number": row["cov_condition_number"],
            })

        root_pose_df = pd.DataFrame(root_pose_records)
        joint_angles_df = pd.DataFrame(joint_angle_records)
        tracking_stats_df = pd.DataFrame(stats_records)

        return {
            "root_pose_df": root_pose_df,
            "joint_angles_df": joint_angles_df,
            "tracking_stats_df": tracking_stats_df,
            "skeleton_yaml": skeleton_yaml,
            "marker_names": marker_names,
            "n_dof": layout.n_dof,
            "run_row": run_dict,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Camera loading
# ---------------------------------------------------------------------------

def load_cameras_from_session(
    session_db: str,
    extrinsic_calibration_id: str,
    session_id: str,
) -> list[dict]:
    """Load camera data from a session DB.

    Parameters
    ----------
    session_db : str
        Path to the session SQLite file.
    extrinsic_calibration_id : str
        UUID of the extrinsic calibration to use.
    session_id : str
        UUID of the mocap session (used to look up session_cameras).

    Returns
    -------
    list of dicts sorted by label, each containing:
      label      : str
      camera_id  : int  (0-based index in alphabetical label order)
      K          : np.ndarray (3, 3)
      R          : np.ndarray (3, 3)
      t          : np.ndarray (3,)
      dist       : np.ndarray (4,)
      P          : np.ndarray (3, 4)
    """
    conn = _open_db(session_db)
    try:
        # Get extrinsic entries joined with intrinsics via session_cameras
        rows = conn.execute(
            """
            SELECT
                ee.camera_instance_id,
                ee.R AS R_blob,
                ee.t AS t_blob,
                sc.label,
                ic.fx, ic.fy, ic.cx, ic.cy,
                ic.dist_coeffs AS dist_blob
            FROM extrinsic_entries ee
            JOIN session_cameras sc
                ON sc.camera_instance_id = ee.camera_instance_id
                AND sc.session_id = ?
            JOIN intrinsics_calibrations ic
                ON ic.id = sc.intrinsics_calibration_id
            WHERE ee.extrinsic_calibration_id = ?
            """,
            (session_id, extrinsic_calibration_id),
        ).fetchall()

        if not rows:
            warnings.warn(
                f"No extrinsic entries found for calibration={extrinsic_calibration_id!r}"
            )
            return []

        cams = []
        for row in rows:
            label = row["label"] or row["camera_instance_id"]
            R = np.frombuffer(bytes(row["R_blob"]), "<f8").reshape(3, 3).copy()
            t = np.frombuffer(bytes(row["t_blob"]), "<f8").copy()
            fx, fy = float(row["fx"]), float(row["fy"])
            cx, cy = float(row["cx"]), float(row["cy"])
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
            dist_blob = row["dist_blob"]
            if dist_blob:
                dist = np.frombuffer(bytes(dist_blob), "<f8").copy()
                if len(dist) < 4:
                    dist = np.pad(dist, (0, 4 - len(dist)))
                dist = dist[:4]
            else:
                dist = np.zeros(4)
            P = K @ np.hstack([R, t.reshape(3, 1)])
            cams.append({
                "label": label,
                "K": K,
                "R": R,
                "t": t,
                "dist": dist,
                "P": P,
            })

        # Sort by label alphabetically, then assign integer camera_id
        cams.sort(key=lambda c: c["label"])
        for i, c in enumerate(cams):
            c["camera_id"] = i

        return cams
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Sync loading
# ---------------------------------------------------------------------------

def load_sync_from_session(session_db: str, sync_config_id: str) -> dict:
    """Load sync data compatible with SyncTable in visualize_tracking.py.

    Parameters
    ----------
    session_db : str
        Path to the session SQLite file.
    sync_config_id : str
        UUID of the sync_config.

    Returns
    -------
    dict[cam_label, {'syncpoints': [{'frame': int, 'timestamp': float}, ...]}]
    """
    conn = _open_db(session_db)
    try:
        rows = conn.execute(
            """
            SELECT sp.video_frame, sp.timestamp_s, ci.label AS cam_label
            FROM sync_points sp
            JOIN shot_videos sv ON sv.id = sp.shot_video_id
            JOIN camera_instances ci ON ci.id = sp.camera_instance_id
            WHERE sp.sync_config_id = ?
            ORDER BY ci.label, sp.video_frame
            """,
            (sync_config_id,),
        ).fetchall()

        result: dict[str, dict] = {}
        for row in rows:
            cam_label = row["cam_label"] or row["camera_instance_id"]
            if cam_label not in result:
                result[cam_label] = {"syncpoints": []}
            result[cam_label]["syncpoints"].append({
                "frame": row["video_frame"],
                "timestamp": row["timestamp_s"],
            })

        return result
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Observations loading
# ---------------------------------------------------------------------------

def load_observations_from_session(
    session_db: str,
    sequence_id: str,
    camera_label_map: dict,
) -> pd.DataFrame:
    """Load 2D pose observations from the database.

    Parameters
    ----------
    session_db : str
        Path to the session SQLite file.
    sequence_id : str
        UUID of the pose_observation_sequence.
    camera_label_map : dict
        Maps camera_instance_id (str) → integer camera_id.
        Build from load_cameras_from_session() results.

    Returns
    -------
    DataFrame with columns:
      frame, timestamp, camera_id, pixel_x, pixel_y, confidence
    One row per (camera, frame, keypoint) combination with confidence > 0.
    """
    conn = _open_db(session_db)
    try:
        rows = conn.execute(
            "SELECT camera_instance_id, video_frame, timestamp_s, person_id, kp_blob "
            "FROM pose_observations "
            "WHERE sequence_id = ? "
            "ORDER BY video_frame, camera_instance_id",
            (sequence_id,),
        ).fetchall()

        records = []
        for row in rows:
            cam_id_str = row["camera_instance_id"]
            cam_id = camera_label_map.get(cam_id_str)
            if cam_id is None:
                continue
            kp_data = np.frombuffer(bytes(row["kp_blob"]), dtype="<f4")
            n_kp = len(kp_data) // 3
            kp = kp_data.reshape(n_kp, 3)
            frame = row["video_frame"]
            ts = row["timestamp_s"]
            for kp_idx in range(n_kp):
                x, y, conf = float(kp[kp_idx, 0]), float(kp[kp_idx, 1]), float(kp[kp_idx, 2])
                if conf <= 0.0:
                    continue
                records.append({
                    "frame": frame,
                    "timestamp": ts,
                    "camera_id": cam_id,
                    "keypoint_index": kp_idx,
                    "pixel_x": x,
                    "pixel_y": y,
                    "confidence": conf,
                })

        if not records:
            return pd.DataFrame(columns=[
                "frame", "timestamp", "camera_id", "keypoint_index",
                "pixel_x", "pixel_y", "confidence",
            ])
        return pd.DataFrame(records)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Combined loader with marker positions
# ---------------------------------------------------------------------------

def load_tracking_run_with_markers(
    session_db: str,
    run_id: str,
    person_id: int = 0,
    smoothed: bool = False,
) -> dict:
    """Load tracking run data and compute 3D marker positions via FK.

    Parameters
    ----------
    session_db : str
        Path to the session SQLite file.
    run_id : str
        UUID of the tracking run.
    person_id : int
        Person to load (default 0).
    smoothed : bool
        Load smoothed results if True.

    Returns
    -------
    Same dict as load_tracking_run_data(), plus:
      marker_positions_df : DataFrame with columns:
                            frame, timestamp, marker_name, x_3d, y_3d, z_3d
    """
    result = load_tracking_run_data(session_db, run_id, person_id, smoothed)
    skeleton_yaml = result["skeleton_yaml"]

    if result["root_pose_df"].empty:
        result["marker_positions_df"] = pd.DataFrame(
            columns=["frame", "timestamp", "marker_name", "x_3d", "y_3d", "z_3d"]
        )
        return result

    layout = SkeletonLayout(skeleton_yaml)
    conn = _open_db(session_db)
    try:
        is_smoothed_val = 1 if smoothed else 0
        rows = conn.execute(
            "SELECT tracker_step, timestamp_s, state "
            "FROM tracking_results "
            "WHERE run_id = ? AND person_id = ? AND is_smoothed = ? "
            "ORDER BY tracker_step",
            (run_id, person_id, is_smoothed_val),
        ).fetchall()
    finally:
        conn.close()

    marker_records = []
    for row in rows:
        step = row["tracker_step"]
        ts = row["timestamp_s"]
        try:
            decoded = layout.decode_state_blob(bytes(row["state"]))
        except Exception:
            continue
        marker_positions = layout.compute_marker_positions(decoded)
        for mname, pos in marker_positions.items():
            marker_records.append({
                "frame": step,
                "timestamp": ts,
                "marker_name": mname,
                "x_3d": float(pos[0]),
                "y_3d": float(pos[1]),
                "z_3d": float(pos[2]),
            })

    result["marker_positions_df"] = pd.DataFrame(marker_records) if marker_records else pd.DataFrame(
        columns=["frame", "timestamp", "marker_name", "x_3d", "y_3d", "z_3d"]
    )
    return result
