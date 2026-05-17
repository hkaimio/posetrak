"""finalise.py — Convert detection keypoints into pose_observation_sequences."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from posetrak.db.db import generate_id
from app.setup.db_context import SyncPoint, SyncTable


@dataclass
class TrackAssignment:
    shot_video_id: str
    track_id: int
    person_name: str
    first_frame: int   # inclusive
    last_frame: int    # inclusive


def finalise_to_db(
    session: sqlite3.Connection,
    detection_run_id: str,
    shot_id: str,
    sync_config_id: str,
    assignments: list[TrackAssignment],
    pose_model: str,
    notes: str = "",
) -> str:
    """Write pose_observation_sequences + pose_observations from detection_keypoints.

    Returns the new sequence ID.
    """
    # Build SyncTable from DB
    rows = session.execute(
        "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, sv.actual_fps "
        "FROM sync_points sp "
        "JOIN capture_videos sv ON sv.id = sp.shot_video_id "
        "WHERE sp.sync_config_id = ?",
        (sync_config_id,),
    ).fetchall()

    points = [
        SyncPoint(
            camera_instance_id="",  # not needed for frame_to_global_time
            shot_video_id=r["shot_video_id"],
            video_frame=r["video_frame"],
            timestamp_s=r["timestamp_s"],
        )
        for r in rows
    ]
    fps_by_video = {r["shot_video_id"]: float(r["actual_fps"]) for r in rows}
    sync_table = SyncTable(points, fps_by_video)

    # Get camera_instance_id per shot_video_id
    sv_rows = session.execute(
        "SELECT id, camera_instance_id FROM capture_videos WHERE shot_id = ?",
        (shot_id,),
    ).fetchall()
    camera_by_svid = {r["id"]: r["camera_instance_id"] for r in sv_rows}

    # Get time_start/end_s from detection_runs row
    run_row = session.execute(
        "SELECT time_start_s, time_end_s FROM detection_runs WHERE id = ?",
        (detection_run_id,),
    ).fetchone()
    time_start_s = float(run_row["time_start_s"])
    time_end_s = float(run_row["time_end_s"])

    # Assign stable person_id per person_name
    name_to_pid: dict[str, int] = {}
    for asgn in assignments:
        if asgn.person_name not in name_to_pid:
            name_to_pid[asgn.person_name] = len(name_to_pid)

    seq_id = generate_id()
    session.execute(
        "INSERT INTO pose_observation_sequences "
        "(id, shot_id, sync_config_id, time_start_s, time_end_s, pose_model, notes, "
        " pixels_are_undistorted, detection_run_id) "
        "VALUES (?,?,?,?,?,?,?,0,?)",
        (seq_id, shot_id, sync_config_id, time_start_s, time_end_s,
         pose_model, notes, detection_run_id),
    )

    obs_rows = []
    for asgn in assignments:
        camera_instance_id = camera_by_svid.get(asgn.shot_video_id)
        if camera_instance_id is None:
            continue

        person_id = name_to_pid[asgn.person_name]

        kp_rows = session.execute(
            "SELECT video_frame, keypoints, noise_scale "
            "FROM detection_keypoints "
            "WHERE detection_run_id=? AND shot_video_id=? AND track_id=? "
            "AND video_frame BETWEEN ? AND ? "
            "AND region_type='full_body' "
            "ORDER BY video_frame",
            (detection_run_id, asgn.shot_video_id, asgn.track_id,
             asgn.first_frame, asgn.last_frame),
        ).fetchall()

        for kp_row in kp_rows:
            frame_idx = int(kp_row["video_frame"])
            timestamp_s = sync_table.frame_to_global_time(frame_idx, asgn.shot_video_id)
            if timestamp_s is None:
                continue
            kp_bytes = bytes(kp_row["keypoints"])
            noise_scale = kp_row["noise_scale"]
            obs_rows.append((
                seq_id, camera_instance_id, frame_idx,
                timestamp_s, person_id, kp_bytes, noise_scale,
            ))

    if obs_rows:
        session.executemany(
            "INSERT INTO pose_observations "
            "(sequence_id, camera_instance_id, video_frame, timestamp_s, "
            " person_id, kp_blob, noise_scale) "
            "VALUES (?,?,?,?,?,?,?)",
            obs_rows,
        )

    session.executemany(
        "INSERT OR IGNORE INTO sequence_persons (sequence_id, person_id, person_name) "
        "VALUES (?, ?, ?)",
        [(seq_id, pid, name) for name, pid in name_to_pid.items()],
    )

    session.execute(
        "DELETE FROM detection_track_assignments WHERE detection_run_id = ?",
        (detection_run_id,),
    )
    session.executemany(
        "INSERT INTO detection_track_assignments "
        "(detection_run_id, shot_video_id, track_id, person_name, first_frame, last_frame) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (detection_run_id, a.shot_video_id, a.track_id,
             a.person_name, a.first_frame, a.last_frame)
            for a in assignments
        ],
    )

    session.commit()
    return seq_id
