# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""sequence_dots.py — attach a detection run's reflective-dot candidates to a sequence.

``SessionReader::load_unlabeled_candidates()`` reads the anonymous dot
candidates of a tracking subject from its own observation sequence
(``pose_observations`` rows with ``source='dots'``). A sequence made by
``finalise_object_to_db`` gets them from the marker run that produced it. A
person's sequence is made from a pose run and carries none, so the dots worn
on the person, detected by a separate dots run, are added afterwards with
:func:`add_dots_to_sequence`.

The copy is additive: body and hand rows of the sequence are untouched, and
each copied row records the dots run it came from in
``pose_observations.detection_run_id``.
"""
from __future__ import annotations

import sqlite3
from typing import NamedTuple


class AddDotsResult(NamedTuple):
    """Rows written by :func:`add_dots_to_sequence`, and rows it had to leave out."""

    inserted: int
    skipped_no_camera: int
    skipped_no_timestamp: int


def add_dots_to_sequence(
    session: sqlite3.Connection,
    detection_run_id: str,
    sequence_id: str,
    *,
    replace: bool = False,
) -> AddDotsResult:
    """Copy one run's dot candidates into an existing sequence.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    detection_run_id:
        A marker run (``detector_type='aruco'``) that detected dots.
    sequence_id:
        The ``pose_observation_sequences`` row that receives them. Its sync
        configuration must be the run's, or frames would map to the wrong
        timestamps.
    replace:
        Delete the sequence's existing ``dots`` rows first. Refused for a
        sequence that has tracking runs or manual edits, because the dots those
        results were computed from would no longer exist.

    Returns
    -------
    AddDotsResult
        Rows inserted, and rows skipped because their video has no camera or
        their frame has no sync timestamp.

    Raises
    ------
    ValueError
        If the sequence or run does not exist, their sync configurations
        differ, or the run has no dot candidates.
    RuntimeError
        If the sequence already has ``dots`` rows and *replace* is false, or is
        tracked or edited and *replace* is true.
    """
    # Imported here: these live in the GUI application's packages.
    from app.pose.db_cache import DOT_REGION_TYPE, DOT_TRACK_ID
    from app.setup.db_context import SyncPoint, SyncTable

    seq = session.execute(
        "SELECT shot_id, sync_config_id FROM pose_observation_sequences WHERE id = ?",
        (sequence_id,),
    ).fetchone()
    if seq is None:
        raise ValueError(f"no pose_observation_sequences row with id {sequence_id!r}")
    shot_id, sync_config_id = seq[0], seq[1]

    run = session.execute(
        "SELECT sync_config_id FROM detection_runs WHERE id = ?", (detection_run_id,)
    ).fetchone()
    if run is None:
        raise ValueError(f"no detection_runs row with id {detection_run_id!r}")
    if run[0] != sync_config_id:
        raise ValueError(
            f"sync_config_id mismatch: sequence uses {sync_config_id!r}, detection run uses "
            f"{run[0]!r}; the frame-to-timestamp mapping would be wrong"
        )

    kp_rows = session.execute(
        "SELECT shot_video_id, video_frame, keypoints, noise_scale FROM detection_keypoints "
        "WHERE detection_run_id = ? AND track_id = ? AND region_type = ? "
        "ORDER BY shot_video_id, video_frame",
        (detection_run_id, DOT_TRACK_ID, DOT_REGION_TYPE),
    ).fetchall()
    if not kp_rows:
        raise ValueError(
            f"detection run {detection_run_id!r} has no dot candidates; run "
            "`detect run --type dots` (or --dots-camera with --type aruco) first"
        )

    existing = session.execute(
        "SELECT COUNT(*) FROM pose_observations WHERE sequence_id = ? AND source = 'dots'",
        (sequence_id,),
    ).fetchone()[0]
    if existing:
        if not replace:
            raise RuntimeError(
                f"sequence {sequence_id!r} already has {existing} 'dots' rows; "
                "pass replace (--replace on the command line) to delete them and copy again"
            )
        tracked_or_edited = session.execute(
            "SELECT 1 FROM tracking_runs WHERE observation_sequence_id = ? "
            "UNION SELECT 1 FROM pose_observation_edits WHERE sequence_id = ? LIMIT 1",
            (sequence_id, sequence_id),
        ).fetchone()
        if tracked_or_edited:
            raise RuntimeError(
                f"sequence {sequence_id!r} has tracking results or manual edits, so its dots "
                "cannot be replaced; make a new sequence instead"
            )

    sync_rows = session.execute(
        "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, sv.actual_fps "
        "FROM sync_points sp JOIN capture_videos sv ON sv.id = sp.shot_video_id "
        "WHERE sp.sync_config_id = ?",
        (sync_config_id,),
    ).fetchall()
    sync_table = SyncTable(
        [
            SyncPoint(camera_instance_id="", shot_video_id=r[0], video_frame=r[1], timestamp_s=r[2])
            for r in sync_rows
        ],
        {r[0]: float(r[3]) for r in sync_rows},
    )
    camera_by_video = {
        r[0]: r[1]
        for r in session.execute(
            "SELECT id, camera_instance_id FROM capture_videos WHERE shot_id = ?", (shot_id,)
        )
    }

    obs_rows = []
    skipped_no_camera = skipped_no_timestamp = 0
    for shot_video_id, video_frame, keypoints, noise_scale in kp_rows:
        camera_instance_id = camera_by_video.get(shot_video_id)
        if camera_instance_id is None:
            skipped_no_camera += 1
            continue
        timestamp_s = sync_table.frame_to_global_time(int(video_frame), shot_video_id)
        if timestamp_s is None:
            skipped_no_timestamp += 1
            continue
        # person_id is unused for dots: the candidates are scene-wide.
        obs_rows.append((
            sequence_id, camera_instance_id, int(video_frame), timestamp_s,
            0, "dots", detection_run_id, bytes(keypoints), noise_scale,
        ))

    if not obs_rows:
        raise ValueError(
            f"none of the {len(kp_rows)} dot rows of run {detection_run_id!r} could be mapped to a "
            "camera and a sync timestamp of this sequence"
        )
    if existing:
        session.execute(
            "DELETE FROM pose_observations WHERE sequence_id = ? AND source = 'dots'",
            (sequence_id,),
        )
    session.executemany(
        "INSERT INTO pose_observations "
        "(sequence_id, camera_instance_id, video_frame, timestamp_s, "
        " person_id, source, detection_run_id, kp_blob, noise_scale) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        obs_rows,
    )
    session.commit()
    return AddDotsResult(len(obs_rows), skipped_no_camera, skipped_no_timestamp)
