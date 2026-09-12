# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""copy_dot_candidates_to_sequence.py -- copy reflective-dot candidates from
a standalone dot-detection run into an EXISTING person pose_observation_sequence,
so `SessionReader::load_unlabeled_candidates()` (keyed only on `sequence_id` +
`source='dots'`) can see them during tracking.

Why this exists rather than reusing `finalise_object_to_db()`
---------------------------------------------------------------
`app/pose/finalise.py`'s `finalise_object_to_db()` already copies a run's
`detection_keypoints` rows (region_type='dots', track_id=DOT_TRACK_ID) into
`pose_observations` under `source='dots'` -- but it does so by creating a NEW
sequence tied to a `capture_object_id` (the rigid-prop use case,
aruco-prop-tracking-design.md). An articulated-body dot-detection run has no
`capture_object_id` (there is no rigid prop; the dots are worn on a tracked
PERSON), so that function's own guard refuses it outright.

Since `pose_observations.person_id` is already documented as "repurposed as
the object; ignored for dots" (finalise.py's own `_copy_region` comment) and
`load_unlabeled_candidates()` never filters on it, the copy target that
actually matters is `sequence_id` alone -- so this script does the same
copy `_copy_region()` does, just against an *existing* person sequence
instead of minting a new object sequence. This is intentionally narrow (an
additive INSERT into `pose_observations`, no schema change, no change to the
GUI finalisation flow) rather than a generalisation of
`finalise_object_to_db()` to a person/no-object case -- that's real design
work belonging to a future pass, not something to invent solo mid-validation.

Usage:
    python tools/copy_dot_candidates_to_sequence.py \\
        --session D:/mocap/2026-09-06-kare-tests/2026-09-06-kare-tests.db \\
        --detection-run 75cbf678-2066-4a58-ab81-d27ea4c58d02 \\
        --sequence ec1b3e2f-1ef8-4e31-806c-33102a969ecd
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.setup.db_context import SyncPoint, SyncTable  # noqa: E402
from app.pose.db_cache import DOT_REGION_TYPE, DOT_TRACK_ID  # noqa: E402


def copy_dot_candidates(
    session: sqlite3.Connection,
    detection_run_id: str,
    sequence_id: str,
    *,
    replace: bool = False,
) -> int:
    """Copy one run's dot detection_keypoints into an existing sequence's
    pose_observations (source='dots'). Returns the number of rows inserted.

    Refuses (unless replace=True) if the sequence already has 'dots' rows --
    same immutability spirit as finalise.py's guards, scoped to just this
    source rather than the whole sequence (body/hand rows are untouched
    either way).
    """
    seq = session.execute(
        "SELECT shot_id, sync_config_id FROM pose_observation_sequences WHERE id = ?",
        (sequence_id,),
    ).fetchone()
    if seq is None:
        raise ValueError(f"no pose_observation_sequences row with id {sequence_id!r}")
    shot_id, sync_config_id = seq

    run = session.execute(
        "SELECT sync_config_id FROM detection_runs WHERE id = ?", (detection_run_id,)
    ).fetchone()
    if run is None:
        raise ValueError(f"no detection_runs row with id {detection_run_id!r}")
    if run[0] != sync_config_id:
        raise ValueError(
            f"sync_config_id mismatch: sequence uses {sync_config_id!r}, "
            f"detection run uses {run[0]!r} -- frame/timestamp mapping would be wrong"
        )

    existing = session.execute(
        "SELECT COUNT(*) FROM pose_observations WHERE sequence_id = ? AND source = 'dots'",
        (sequence_id,),
    ).fetchone()[0]
    if existing:
        if not replace:
            raise RuntimeError(
                f"sequence {sequence_id!r} already has {existing} 'dots' rows -- "
                "pass --replace to delete and re-copy"
            )
        session.execute(
            "DELETE FROM pose_observations WHERE sequence_id = ? AND source = 'dots'",
            (sequence_id,),
        )

    rows = session.execute(
        "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, sv.actual_fps "
        "FROM sync_points sp "
        "JOIN capture_videos sv ON sv.id = sp.shot_video_id "
        "WHERE sp.sync_config_id = ?",
        (sync_config_id,),
    ).fetchall()
    points = [
        SyncPoint(camera_instance_id="", shot_video_id=r[0], video_frame=r[1], timestamp_s=r[2])
        for r in rows
    ]
    fps_by_video = {r[0]: float(r[3]) for r in rows}
    sync_table = SyncTable(points, fps_by_video)

    sv_rows = session.execute(
        "SELECT id, camera_instance_id FROM capture_videos WHERE shot_id = ?", (shot_id,)
    ).fetchall()
    camera_by_svid = {r[0]: r[1] for r in sv_rows}

    kp_rows = session.execute(
        "SELECT shot_video_id, video_frame, keypoints, noise_scale FROM detection_keypoints "
        "WHERE detection_run_id = ? AND track_id = ? AND region_type = ? "
        "ORDER BY shot_video_id, video_frame",
        (detection_run_id, DOT_TRACK_ID, DOT_REGION_TYPE),
    ).fetchall()

    obs_rows = []
    skipped_no_camera = 0
    skipped_no_timestamp = 0
    for shot_video_id, video_frame, keypoints, noise_scale in kp_rows:
        camera_instance_id = camera_by_svid.get(shot_video_id)
        if camera_instance_id is None:
            skipped_no_camera += 1
            continue
        frame_idx = int(video_frame)
        timestamp_s = sync_table.frame_to_global_time(frame_idx, shot_video_id)
        if timestamp_s is None:
            skipped_no_timestamp += 1
            continue
        obs_rows.append((
            sequence_id, camera_instance_id, frame_idx, timestamp_s,
            0,  # person_id column, repurposed placeholder for dots -- see module docstring
            "dots", detection_run_id, bytes(keypoints), noise_scale,
        ))

    if obs_rows:
        session.executemany(
            "INSERT INTO pose_observations "
            "(sequence_id, camera_instance_id, video_frame, timestamp_s, "
            " person_id, source, detection_run_id, kp_blob, noise_scale) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            obs_rows,
        )
    session.commit()

    if skipped_no_camera or skipped_no_timestamp:
        print(f"  (skipped {skipped_no_camera} rows with unmapped camera, "
              f"{skipped_no_timestamp} rows with no sync timestamp)")
    return len(obs_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--detection-run", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--replace", action="store_true",
                     help="Delete any existing 'dots' rows for this sequence first.")
    args = ap.parse_args()

    conn = sqlite3.connect(args.session)
    n = copy_dot_candidates(conn, args.detection_run, args.sequence, replace=args.replace)
    print(f"inserted {n} pose_observations rows (source='dots') into sequence {args.sequence}")


if __name__ == "__main__":
    main()
