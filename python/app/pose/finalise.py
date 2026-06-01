"""finalise.py — Convert detection keypoints into pose_observation_sequences."""
from __future__ import annotations

import sqlite3
from collections import defaultdict
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


def _scale_kp_confidence(kp_blob: bytes, scale: float) -> bytes:
    """Multiply the confidence channel of a float32[N,3] keypoint blob by *scale*."""
    arr = np.frombuffer(kp_blob, dtype="<f4").copy().reshape(-1, 3)
    arr[:, 2] *= scale
    return arr.tobytes()


def conf_scale_for_model(pose_model: str) -> float:
    """Return the confidence normalisation factor for *pose_model*.

    RTMPose outputs SimCC logit scores (3–8 for well-detected joints).
    VITpose outputs heatmap peak values (0–1).  The C++ UKF uses
    ``measurement_noise_std = base_noise / confidence``, so a scale factor
    is needed to bring ViTPose values into the same effective noise range.
    """
    try:
        from app.pose.backends_rtmpose import _KNOWN_MODELS
        entry = _KNOWN_MODELS.get(pose_model)
        if entry is not None:
            return float(entry[3])
    except Exception:
        pass
    return 1.0


def finalise_to_db(
    session: sqlite3.Connection,
    detection_run_id: str,
    shot_id: str,
    sync_config_id: str,
    assignments: list[TrackAssignment],
    pose_model: str,
    notes: str = "",
    confidence_scale: float = 1.0,
) -> list[str]:
    """Write pose_observation_sequences + pose_observations from detection_keypoints.

    Creates one sequence per person.  Replaces any existing sequences for this
    detection run (including their tracking runs and results).

    Returns the list of new sequence IDs (one per person).
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
            camera_instance_id="",
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

    # Delete existing sequences for this detection run (cascade manually)
    existing_ids = [
        r[0] for r in session.execute(
            "SELECT id FROM pose_observation_sequences WHERE detection_run_id = ?",
            (detection_run_id,),
        )
    ]
    for sid in existing_ids:
        session.execute(
            "DELETE FROM tracking_results WHERE run_id IN "
            "(SELECT id FROM tracking_runs WHERE observation_sequence_id = ?)", (sid,)
        )
        session.execute(
            "DELETE FROM tracking_obs_results WHERE run_id IN "
            "(SELECT id FROM tracking_runs WHERE observation_sequence_id = ?)", (sid,)
        )
        session.execute(
            "DELETE FROM tracking_runs WHERE observation_sequence_id = ?", (sid,)
        )
        session.execute("DELETE FROM sequence_persons WHERE sequence_id = ?", (sid,))
        session.execute("DELETE FROM pose_observations WHERE sequence_id = ?", (sid,))
    session.execute(
        "DELETE FROM pose_observation_sequences WHERE detection_run_id = ?",
        (detection_run_id,),
    )

    # Group assignments by person name
    by_person: dict[str, list[TrackAssignment]] = defaultdict(list)
    for asgn in assignments:
        by_person[asgn.person_name].append(asgn)

    seq_ids: list[str] = []

    for person_name, person_assignments in by_person.items():
        seq_id = generate_id()
        seq_ids.append(seq_id)

        session.execute(
            "INSERT INTO pose_observation_sequences "
            "(id, shot_id, sync_config_id, time_start_s, time_end_s, pose_model, notes, "
            " pixels_are_undistorted, detection_run_id) "
            "VALUES (?,?,?,?,?,?,?,0,?)",
            (seq_id, shot_id, sync_config_id, time_start_s, time_end_s,
             pose_model, notes, detection_run_id),
        )

        # One person per sequence — person_id is always 0 within the sequence
        session.execute(
            "INSERT OR IGNORE INTO sequence_persons (sequence_id, person_id, person_name) "
            "VALUES (?, 0, ?)",
            (seq_id, person_name),
        )

        # Keyed by (camera_instance_id, video_frame) so that adjacent segments
        # sharing a boundary frame don't produce duplicates — later segment wins.
        obs_by_key: dict[tuple, tuple] = {}
        for asgn in person_assignments:
            camera_instance_id = camera_by_svid.get(asgn.shot_video_id)
            if camera_instance_id is None:
                continue

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
                kp_blob = bytes(kp_row["keypoints"])
                if confidence_scale != 1.0:
                    kp_blob = _scale_kp_confidence(kp_blob, confidence_scale)
                obs_by_key[(camera_instance_id, frame_idx)] = (
                    seq_id, camera_instance_id, frame_idx,
                    timestamp_s, 0,  # person_id=0, single person per sequence
                    kp_blob, kp_row["noise_scale"],
                )

        if obs_by_key:
            session.executemany(
                "INSERT INTO pose_observations "
                "(sequence_id, camera_instance_id, video_frame, timestamp_s, "
                " person_id, kp_blob, noise_scale) "
                "VALUES (?,?,?,?,?,?,?)",
                obs_by_key.values(),
            )

    # Persist the track→person mapping for future restore
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
    return seq_ids
