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
    from posetrak.db.skeleton_layout import SkeletonLayout


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
        # NIS is only computed during the forward pass (is_smoothed=0); join to
        # forward rows so NIS is always available even when loading smoothed state.
        rows = conn.execute(
            "SELECT tr.tracker_step, tr.timestamp_s, tr.tracking_lost, "
            "tr.n_inlier_observations, tr.cov_condition_number, "
            "fwd.nis_value, fwd.nis_dof, tr.state, tr.cov_diag "
            "FROM tracking_results tr "
            "LEFT JOIN tracking_results fwd "
            "  ON fwd.run_id = tr.run_id AND fwd.person_id = tr.person_id "
            "  AND fwd.tracker_step = tr.tracker_step AND fwd.is_smoothed = 0 "
            "WHERE tr.run_id = ? AND tr.person_id = ? AND tr.is_smoothed = ? "
            "ORDER BY tr.tracker_step",
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
        cov_diag_records = []
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
            cov_blob = row["cov_diag"]
            if cov_blob is not None and len(cov_blob) > 0:
                try:
                    decoded_cov = layout.decode_cov_diag(bytes(cov_blob))
                    cov_diag_records.extend(
                        layout.decoded_cov_to_joint_std_rows(step, ts, decoded_cov)
                    )
                except Exception as exc:
                    warnings.warn(f"Could not decode cov_diag at step {step}: {exc}")
            stats_records.append({
                "frame": step,
                "timestamp": ts,
                "tracking_lost": bool(row["tracking_lost"]),
                "num_inliers": row["n_inlier_observations"],
                "cov_condition_number": row["cov_condition_number"],
                "nis_value": row["nis_value"],
                "nis_dof": row["nis_dof"],
            })

        root_pose_df = pd.DataFrame(root_pose_records)
        joint_angles_df = pd.DataFrame(joint_angle_records)
        cov_diag_df = pd.DataFrame(cov_diag_records) if cov_diag_records else pd.DataFrame(
            columns=["frame", "timestamp", "joint_name",
                     "std_x", "std_y", "std_z",
                     "vel_std_x", "vel_std_y", "vel_std_z"]
        )
        tracking_stats_df = pd.DataFrame(stats_records)

        return {
            "root_pose_df": root_pose_df,
            "joint_angles_df": joint_angles_df,
            "cov_diag_df": cov_diag_df,
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
        # Get extrinsic entries; intrinsics are stored on capture_videos (v11+).
        # For each camera instance in the session, pick the intrinsics from the
        # first capture_video that has a non-NULL intrinsics_calibration_id.
        rows = conn.execute(
            """
            SELECT
                ee.camera_instance_id,
                ee.R AS R_blob,
                ee.t AS t_blob,
                sc.label,
                ci.label AS instance_label,
                ic.fx, ic.fy, ic.cx, ic.cy,
                ic.dist_coeffs AS dist_blob
            FROM extrinsic_entries ee
            LEFT JOIN session_cameras sc
                ON sc.camera_instance_id = ee.camera_instance_id
                AND sc.session_id = :session_id
            LEFT JOIN camera_instances ci
                ON ci.id = ee.camera_instance_id
            LEFT JOIN (
                SELECT sv.camera_instance_id, sv.intrinsics_calibration_id
                FROM capture_videos sv
                JOIN captures sh ON sh.id = sv.shot_id
                WHERE sh.session_id = :session_id
                  AND sv.intrinsics_calibration_id IS NOT NULL
                GROUP BY sv.camera_instance_id
            ) sv_intr ON sv_intr.camera_instance_id = ee.camera_instance_id
            LEFT JOIN intrinsics_calibrations ic
                ON ic.id = sv_intr.intrinsics_calibration_id
            WHERE ee.extrinsic_calibration_id = :ext_cal_id
            """,
            {"session_id": session_id, "ext_cal_id": extrinsic_calibration_id},
        ).fetchall()

        if not rows:
            warnings.warn(
                f"No extrinsic entries found for calibration={extrinsic_calibration_id!r}"
            )
            return []

        cams = []
        for row in rows:
            if row["fx"] is None:
                warnings.warn(
                    f"No intrinsics found for camera_instance_id={row['camera_instance_id']!r}; "
                    "camera skipped. Set intrinsics_calibration_id on shot_videos to fix this."
                )
                continue
            label = row["label"] or row["instance_label"] or row["camera_instance_id"]
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
            instance_label = row["instance_label"] or label
            P = K @ np.hstack([R, t.reshape(3, 1)])
            cams.append({
                "label": label,
                "instance_label": instance_label,
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
    dict[cam_label, {'fps': float, 'syncpoints': [{'frame': int, 'timestamp': float}, ...]}]
    """
    conn = _open_db(session_db)
    try:
        rows = conn.execute(
            """
            SELECT sp.video_frame, sp.timestamp_s, sv.actual_fps,
                   sp.camera_instance_id,
                   COALESCE(sc.label, ci.label) AS cam_label
            FROM sync_points sp
            JOIN capture_videos sv ON sv.id = sp.shot_video_id
            JOIN camera_instances ci ON ci.id = sp.camera_instance_id
            JOIN sync_configs scfg ON scfg.id = sp.sync_config_id
            JOIN captures sh ON sh.id = scfg.shot_id
            LEFT JOIN session_cameras sc
                ON sc.camera_instance_id = sp.camera_instance_id
               AND sc.session_id = sh.session_id
            WHERE sp.sync_config_id = ?
            ORDER BY cam_label, sp.video_frame
            """,
            (sync_config_id,),
        ).fetchall()

        result: dict[str, dict] = {}
        for row in rows:
            cam_label = row["cam_label"] or row["camera_instance_id"]
            if cam_label not in result:
                result[cam_label] = {"fps": row["actual_fps"] or 0.0, "syncpoints": []}
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
# Inlier observations from tracking run
# ---------------------------------------------------------------------------

def load_inlier_obs_from_tracking_run(
    session_db: str,
    run_id: str,
    person_id: int = 0,
    inliers_only: bool = True,
) -> pd.DataFrame:
    """Load inlier 2D observations from tracking_obs_results for a run.

    Parses the obs_blob (float32[n_cams, n_markers, 8]) and returns only
    observations where slot[6] (is_outlier) == 0.  Absent slots (NaN) and
    outlier slots are excluded.  Pixels are in undistorted pixel space (K_new).

    Parameters
    ----------
    session_db : str
        Path to the session SQLite file.
    run_id : str
        UUID of the tracking run.
    person_id : int
        Person ID (default 0).

    Returns
    -------
    DataFrame with columns:
      tracker_step, timestamp_s, camera_label, marker_name, pixel_x, pixel_y
    """
    conn = _open_db(session_db)
    try:
        run_row = conn.execute(
            "SELECT active_camera_ids, marker_names FROM tracking_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run_row is None:
            raise ValueError(f"No tracking run found with id={run_id!r}")

        active_camera_labels: list[str] = json.loads(run_row["active_camera_ids"] or "[]")
        marker_names: list[str] = json.loads(run_row["marker_names"] or "[]")
        n_cams = len(active_camera_labels)
        n_markers = len(marker_names)

        if n_cams == 0 or n_markers == 0:
            return pd.DataFrame(columns=[
                "tracker_step", "timestamp_s", "camera_label",
                "marker_name", "pixel_x", "pixel_y",
            ])

        rows = conn.execute(
            """
            SELECT tor.tracker_step, tor.obs_blob, tr.timestamp_s
            FROM tracking_obs_results tor
            JOIN tracking_results tr
                ON tr.run_id = tor.run_id
               AND tr.person_id = tor.person_id
               AND tr.tracker_step = tor.tracker_step
               AND tr.is_smoothed = 0
            WHERE tor.run_id = ? AND tor.person_id = ?
            ORDER BY tor.tracker_step
            """,
            (run_id, person_id),
        ).fetchall()

        records = []
        expected_floats = n_cams * n_markers * 8
        for row in rows:
            step = row["tracker_step"]
            ts = row["timestamp_s"]
            obs_data = np.frombuffer(bytes(row["obs_blob"]), dtype="<f4")
            if len(obs_data) != expected_floats:
                warnings.warn(
                    f"Unexpected obs_blob size at step {step}: "
                    f"got {len(obs_data)}, expected {expected_floats}"
                )
                continue
            obs = obs_data.reshape(n_cams, n_markers, 8)
            for ci, cam_label in enumerate(active_camera_labels):
                for mi, marker_name in enumerate(marker_names):
                    # slot[6] = is_outlier: 0.0 = inlier, 1.0 = outlier, NaN = absent
                    is_outlier = obs[ci, mi, 6]
                    if not np.isfinite(is_outlier):
                        continue  # absent slot — no observation
                    if inliers_only and is_outlier != 0.0:
                        continue
                    px, py = float(obs[ci, mi, 0]), float(obs[ci, mi, 1])
                    if not (np.isfinite(px) and np.isfinite(py)):
                        continue
                    records.append({
                        "tracker_step": step,
                        "timestamp_s": ts,
                        "camera_label": cam_label,
                        "marker_name": marker_name,
                        "pixel_x": px,
                        "pixel_y": py,
                    })

        if not records:
            return pd.DataFrame(columns=[
                "tracker_step", "timestamp_s", "camera_label",
                "marker_name", "pixel_x", "pixel_y",
            ])
        return pd.DataFrame(records)
    finally:
        conn.close()


def load_obs_results(
    session_db: str,
    run_id: str,
    person_id: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load observations and projected markers from tracking_obs_results.

    Decodes the obs_blob (float32[n_cams, n_markers, 8]) where slots are:
    ``[obs_u, obs_v, pred_u, pred_v, mahal_dist, used_in_update, is_outlier, pad]``.
    NaN values indicate an absent slot.

    Returns
    -------
    (observations_df, projected_markers_df)

    observations_df columns:
        frame, camera_id, marker_name, pixel_x, pixel_y, is_outlier

    projected_markers_df columns:
        frame, camera_id, marker_name, proj_x, proj_y,
        error_x, error_y, error_dist, is_outlier
    """
    _OBS_COLS = ["frame", "camera_id", "marker_name", "pixel_x", "pixel_y", "is_outlier"]
    _PROJ_COLS = ["frame", "camera_id", "marker_name", "proj_x", "proj_y",
                  "error_x", "error_y", "error_dist", "is_outlier"]

    conn = _open_db(session_db)
    try:
        run_row = conn.execute(
            "SELECT active_camera_ids, marker_names FROM tracking_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run_row is None:
            return pd.DataFrame(columns=_OBS_COLS), pd.DataFrame(columns=_PROJ_COLS)

        cam_labels: list[str] = json.loads(run_row["active_camera_ids"] or "[]")
        marker_names: list[str] = json.loads(run_row["marker_names"] or "[]")
        n_cams = len(cam_labels)
        n_markers = len(marker_names)

        if n_cams == 0 or n_markers == 0:
            return pd.DataFrame(columns=_OBS_COLS), pd.DataFrame(columns=_PROJ_COLS)

        blob_rows = conn.execute(
            "SELECT tracker_step, obs_blob FROM tracking_obs_results "
            "WHERE run_id = ? AND person_id = ? ORDER BY tracker_step",
            (run_id, person_id),
        ).fetchall()

        obs_records: list[dict] = []
        proj_records: list[dict] = []
        expected = n_cams * n_markers * 8
        for row in blob_rows:
            frame = row["tracker_step"]
            data = np.frombuffer(bytes(row["obs_blob"]), dtype="<f4")
            if len(data) != expected:
                warnings.warn(
                    f"Unexpected obs_blob size at step {frame}: "
                    f"got {len(data)}, expected {expected}"
                )
                continue
            blob = data.reshape(n_cams, n_markers, 8)
            for ci, cam in enumerate(cam_labels):
                for mi, mname in enumerate(marker_names):
                    slot = blob[ci, mi]
                    is_outlier_raw = slot[6]
                    if not np.isfinite(is_outlier_raw):
                        continue  # absent slot
                    is_out = bool(is_outlier_raw != 0.0)
                    obs_x, obs_y = float(slot[0]), float(slot[1])
                    pred_x, pred_y = float(slot[2]), float(slot[3])
                    if np.isfinite(obs_x) and np.isfinite(obs_y):
                        obs_records.append({
                            "frame": frame, "camera_id": cam,
                            "marker_name": mname,
                            "pixel_x": obs_x, "pixel_y": obs_y,
                            "is_outlier": is_out,
                        })
                    if np.isfinite(pred_x) and np.isfinite(pred_y):
                        err_x = obs_x - pred_x if np.isfinite(obs_x) else float("nan")
                        err_y = obs_y - pred_y if np.isfinite(obs_y) else float("nan")
                        err_d = (float(np.sqrt(err_x**2 + err_y**2))
                                 if np.isfinite(err_x) and np.isfinite(err_y)
                                 else float("nan"))
                        proj_records.append({
                            "frame": frame, "camera_id": cam,
                            "marker_name": mname,
                            "proj_x": pred_x, "proj_y": pred_y,
                            "error_x": err_x, "error_y": err_y,
                            "error_dist": err_d,
                            "is_outlier": is_out,
                        })

        obs_df = (pd.DataFrame(obs_records) if obs_records
                  else pd.DataFrame(columns=_OBS_COLS))
        proj_df = (pd.DataFrame(proj_records) if proj_records
                   else pd.DataFrame(columns=_PROJ_COLS))
        return obs_df, proj_df
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
