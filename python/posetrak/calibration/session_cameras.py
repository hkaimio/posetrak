# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""session_cameras.py — the cameras and sync table of a capture, ready for geometry.

Loads what multi-view geometry needs from a session database: one calibrated
camera state per camera of a capture (intrinsics and solved extrinsics), and the
sync table that maps a video's frames to global time. Headless: nothing here
depends on the GUI.
"""
from __future__ import annotations

import sqlite3
import struct
import sys

import numpy as np

from app.setup.db_context import SyncPoint, SyncTable
from app.setup.extrinsics_solver import CamCalibState


def _resolve_intrinsics(ic: sqlite3.Row) -> dict:
    """Build the K/K_orig/dist/fisheye dict a CamCalibState needs from an
    ``intrinsics_calibrations`` row."""
    fx, fy, cx, cy = ic["fx"], ic["fy"], ic["cx"], ic["cy"]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    K_orig = K.copy()
    if ic["matrix_original"]:
        vals = struct.unpack("<9d", bytes(ic["matrix_original"]))
        K_orig = np.array(vals).reshape(3, 3)
    if ic["dist_coeffs"]:
        n = len(bytes(ic["dist_coeffs"])) // 8
        dist = np.array(struct.unpack(f"<{n}d", bytes(ic["dist_coeffs"]))).reshape(1, -1)
    else:
        dist = np.zeros((1, 4))
    return {
        "K": K, "K_orig": K_orig, "dist": dist,
        "fisheye": ic["distortion_model"] == "fisheye",
    }


def load_camera_states(conn: sqlite3.Connection, shot_id: str) -> dict[str, CamCalibState]:
    """Load one CamCalibState per camera covering *shot_id*, with R/t filled
    from the capture's solved extrinsics.

    Keyed by ``camera_instance_id`` rather than by label, because the tables to
    join against (sync_points, capture_videos, extrinsic_entries) all key on it.
    A camera without solved extrinsics or an intrinsics calibration is skipped,
    with a note on stderr.

    Parameters
    ----------
    conn:
        Connection to a session database, with ``sqlite3.Row`` rows.
    shot_id:
        The capture.

    Raises
    ------
    ValueError
        If the capture has no solved extrinsic calibration.
    """
    row = conn.execute(
        "SELECT extrinsic_calibration_id FROM captures WHERE id = ?", (shot_id,)
    ).fetchone()
    if row is None or row["extrinsic_calibration_id"] is None:
        raise ValueError(f"capture {shot_id!r} has no solved extrinsic_calibration_id")
    calib_id = row["extrinsic_calibration_id"]

    extr_rows = conn.execute(
        "SELECT camera_instance_id, R, t FROM extrinsic_entries WHERE extrinsic_calibration_id = ?",
        (calib_id,),
    ).fetchall()
    extrinsics = {
        r["camera_instance_id"]: (
            np.frombuffer(bytes(r["R"]), dtype=np.float64).reshape(3, 3),
            np.frombuffer(bytes(r["t"]), dtype=np.float64).reshape(3, 1),
        )
        for r in extr_rows
    }

    cam_rows = conn.execute(
        """
        SELECT cv.id AS shot_video_id, cv.camera_instance_id, cv.file_path,
               cv.first_video_frame, cv.last_video_frame, cv.actual_fps,
               cv.intrinsics_calibration_id AS cv_calib_id,
               cm.default_intrinsics_calibration_id AS mode_default_calib_id
        FROM capture_videos cv
        LEFT JOIN camera_modes cm ON cm.id = cv.camera_mode_id
        WHERE cv.shot_id = ?
        """,
        (shot_id,),
    ).fetchall()

    states: dict[str, CamCalibState] = {}
    for r in cam_rows:
        cam_id = r["camera_instance_id"]
        if cam_id not in extrinsics:
            print(f"  SKIP {cam_id[:8]}: no solved extrinsics entry", file=sys.stderr)
            continue
        intr_calib_id = r["cv_calib_id"] or r["mode_default_calib_id"]
        if intr_calib_id is None:
            print(f"  SKIP {cam_id[:8]}: no intrinsics calibration", file=sys.stderr)
            continue
        ic = conn.execute(
            "SELECT * FROM intrinsics_calibrations WHERE id = ?", (intr_calib_id,)
        ).fetchone()
        if ic is None:
            continue
        R, t = extrinsics[cam_id]
        states[cam_id] = CamCalibState(
            video_id=cam_id, label=cam_id, R=R, t=t,
            file_path=r["file_path"], first_frame=r["first_video_frame"],
            last_frame=r["last_video_frame"], **_resolve_intrinsics(ic),
        )
    return states


def load_sync_table(conn: sqlite3.Connection, shot_id: str) -> tuple[SyncTable, dict]:
    """Load the newest sync configuration of a capture.

    Returns
    -------
    tuple[SyncTable, dict]
        The table, and the ``capture_videos`` id of each ``camera_instance_id``.

    Raises
    ------
    ValueError
        If the capture has no sync configuration.
    """
    row = conn.execute(
        "SELECT id FROM sync_configs WHERE shot_id = ? ORDER BY rowid DESC LIMIT 1", (shot_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"capture {shot_id!r} has no sync_configs row")
    sync_id = row["id"]

    sp_rows = conn.execute(
        "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, cv.actual_fps, "
        "       cv.camera_instance_id "
        "FROM sync_points sp JOIN capture_videos cv ON cv.id = sp.shot_video_id "
        "WHERE sp.sync_config_id = ?",
        (sync_id,),
    ).fetchall()
    sync_points = []
    fps_by_video = {}
    svid_by_cam = {}
    for r in sp_rows:
        svid = r["shot_video_id"]
        sync_points.append(SyncPoint(
            camera_instance_id=svid, shot_video_id=svid,
            video_frame=int(r["video_frame"]), timestamp_s=float(r["timestamp_s"]),
        ))
        fps_by_video[svid] = float(r["actual_fps"])
        svid_by_cam[r["camera_instance_id"]] = svid
    return SyncTable(sync_points, fps_by_video), svid_by_cam
