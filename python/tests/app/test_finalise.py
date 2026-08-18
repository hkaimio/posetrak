# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for app.pose.finalise.finalise_to_db — multi-source write path.

Phase 2 of hand-detection refinement lets detection_keypoints hold
'hand_l'/'hand_r' rows alongside the whole-body 'full_body' row for the same
(run, video, frame, track). finalise_to_db must carry all three into
pose_observations as separate source-tagged rows rather than only the
whole-body pass.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from posetrak.db.db import create_session, generate_id

from app.pose.finalise import TrackAssignment, auto_assign_and_finalise, finalise_to_db

_SHOT_ID = "test-shot-id"
_SYNC_ID = "test-sync-id"
_SVID = "test-sv-id"
_CAM_ID = "test-cam-id"


@pytest.fixture
def session(tmp_path):
    db_path = tmp_path / "test.db"
    conn = create_session(db_path)
    conn.row_factory = sqlite3.Row
    session_id = generate_id()
    conn.executescript(f"""
        INSERT INTO mocap_sessions (id, recorded_at) VALUES ('{session_id}', '2026-01-01');
        INSERT INTO captures (id, session_id, capture_number, label)
            VALUES ('{_SHOT_ID}', '{session_id}', 1, 'test');
        INSERT INTO sync_configs (id, shot_id, created_by)
            VALUES ('{_SYNC_ID}', '{_SHOT_ID}', 'test');
        INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,
                                 first_video_frame, last_video_frame, actual_fps)
            VALUES ('{_SVID}', '{_SHOT_ID}', '{_CAM_ID}', '/fake/video.mp4', 0, 1000, 120.0);
        INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id,
                                 video_frame, timestamp_s)
            VALUES ('{_SYNC_ID}', '{_CAM_ID}', '{_SVID}', 0, 0.0);
        INSERT INTO detection_runs (id, shot_id, sync_config_id, time_start_s, time_end_s,
                                 detector_model, pose_model, status, created_at)
            VALUES ('run1', '{_SHOT_ID}', '{_SYNC_ID}', 0.0, 1.0, 'yolo', 'rtmpose-l-133kp',
                    'complete', '2026-01-01');
    """)
    conn.commit()
    return conn


def _kp(fill: float, n: int) -> bytes:
    arr = np.zeros((n, 3), dtype=np.float32)
    arr[:] = [fill, fill, fill]
    return arr.tobytes()


def test_finalise_writes_one_row_per_source(session):
    session.execute(
        "INSERT INTO detection_keypoints"
        " (detection_run_id, shot_video_id, video_frame, track_id, region_type, keypoints, noise_scale)"
        " VALUES ('run1', ?, 0, 1, 'full_body', ?, 0.5)",
        (_SVID, _kp(1.0, 133)),
    )
    session.execute(
        "INSERT INTO detection_keypoints"
        " (detection_run_id, shot_video_id, video_frame, track_id, region_type, keypoints, noise_scale)"
        " VALUES ('run1', ?, 0, 1, 'hand_l', ?, 0.1)",
        (_SVID, _kp(2.0, 21)),
    )
    session.execute(
        "INSERT INTO detection_keypoints"
        " (detection_run_id, shot_video_id, video_frame, track_id, region_type, keypoints, noise_scale)"
        " VALUES ('run1', ?, 0, 1, 'hand_r', ?, 0.2)",
        (_SVID, _kp(3.0, 21)),
    )
    session.commit()

    assignment = TrackAssignment(
        shot_video_id=_SVID, track_id=1, person_name="alice",
        first_frame=0, last_frame=0,
    )
    seq_ids = finalise_to_db(
        session, detection_run_id="run1", shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
        assignments=[assignment], pose_model="rtmpose-l-133kp",
    )
    assert len(seq_ids) == 1
    seq_id = seq_ids[0]

    rows = session.execute(
        "SELECT source, detection_run_id, kp_blob, noise_scale FROM pose_observations"
        " WHERE sequence_id = ? ORDER BY source",
        (seq_id,),
    ).fetchall()
    by_source = {r["source"]: r for r in rows}
    assert set(by_source) == {"body", "hand_l", "hand_r"}

    for source in by_source:
        assert by_source[source]["detection_run_id"] == "run1"

    body_kp = np.frombuffer(bytes(by_source["body"]["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    assert body_kp.shape == (133, 3)
    np.testing.assert_allclose(body_kp[0], [1.0, 1.0, 1.0])
    assert by_source["body"]["noise_scale"] == pytest.approx(0.5)

    hand_l_kp = np.frombuffer(bytes(by_source["hand_l"]["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    assert hand_l_kp.shape == (21, 3)
    np.testing.assert_allclose(hand_l_kp[0], [2.0, 2.0, 2.0])
    assert by_source["hand_l"]["noise_scale"] == pytest.approx(0.1)

    hand_r_kp = np.frombuffer(bytes(by_source["hand_r"]["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    np.testing.assert_allclose(hand_r_kp[0], [3.0, 3.0, 3.0])
    assert by_source["hand_r"]["noise_scale"] == pytest.approx(0.2)


def test_finalise_confidence_scale_applies_only_to_body(session):
    session.execute(
        "INSERT INTO detection_keypoints"
        " (detection_run_id, shot_video_id, video_frame, track_id, region_type, keypoints, noise_scale)"
        " VALUES ('run1', ?, 0, 1, 'full_body', ?, 0.5)",
        (_SVID, _kp(1.0, 133)),
    )
    session.execute(
        "INSERT INTO detection_keypoints"
        " (detection_run_id, shot_video_id, video_frame, track_id, region_type, keypoints, noise_scale)"
        " VALUES ('run1', ?, 0, 1, 'hand_l', ?, 0.1)",
        (_SVID, _kp(2.0, 21)),
    )
    session.commit()

    assignment = TrackAssignment(
        shot_video_id=_SVID, track_id=1, person_name="alice",
        first_frame=0, last_frame=0,
    )
    seq_ids = finalise_to_db(
        session, detection_run_id="run1", shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
        assignments=[assignment], pose_model="rtmpose-l-133kp", confidence_scale=2.0,
    )
    seq_id = seq_ids[0]
    rows = {
        r["source"]: r for r in session.execute(
            "SELECT source, kp_blob FROM pose_observations WHERE sequence_id=?", (seq_id,)
        ).fetchall()
    }
    body_kp = np.frombuffer(bytes(rows["body"]["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    assert body_kp[0, 2] == pytest.approx(2.0)  # 1.0 * confidence_scale

    hand_kp = np.frombuffer(bytes(rows["hand_l"]["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    assert hand_kp[0, 2] == pytest.approx(2.0)  # unscaled — hand rows keep their own conf


def test_refinalise_refuses_once_sequence_has_edits(session):
    """Re-finalising a detection run must refuse, not crash, once one of its
    existing sequences has manual keypoint edits -- see
    docs/roadmap/features/hand-detection-refinement (finalise_to_db edit-loss
    bug): the delete cascade never removed pose_observation_edits, so a plain
    re-finalise used to hit an unhandled FOREIGN KEY constraint failed instead
    of a clear, intentional refusal.
    """
    session.execute(
        "INSERT INTO detection_keypoints"
        " (detection_run_id, shot_video_id, video_frame, track_id, region_type, keypoints, noise_scale)"
        " VALUES ('run1', ?, 0, 1, 'full_body', ?, 0.5)",
        (_SVID, _kp(1.0, 133)),
    )
    session.commit()

    assignment = TrackAssignment(
        shot_video_id=_SVID, track_id=1, person_name="alice",
        first_frame=0, last_frame=0,
    )
    seq_ids = finalise_to_db(
        session, detection_run_id="run1", shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
        assignments=[assignment], pose_model="rtmpose-l-133kp",
    )
    seq_id = seq_ids[0]

    session.execute(
        "INSERT INTO pose_observation_edits"
        " (id, sequence_id, camera_instance_id, video_frame, kp_blob, kp_mask)"
        " VALUES (?, ?, ?, 0, ?, ?)",
        (generate_id(), seq_id, _CAM_ID, _kp(9.0, 133), b"\x01" * 133),
    )
    session.commit()

    with pytest.raises(RuntimeError, match="already has tracking results and/or manual"):
        finalise_to_db(
            session, detection_run_id="run1", shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
            assignments=[assignment], pose_model="rtmpose-l-133kp",
        )

    # Refused before touching anything -- the original sequence and its edit
    # must both still exist untouched.
    assert session.execute(
        "SELECT count(*) FROM pose_observation_sequences WHERE id=?", (seq_id,)
    ).fetchone()[0] == 1
    assert session.execute(
        "SELECT count(*) FROM pose_observation_edits WHERE sequence_id=?", (seq_id,)
    ).fetchone()[0] == 1


# ---------------------------------------------------------------------------
# capture_person_id resolution (config-improvements design doc, "Person
# model: promote identity to capture level")
# ---------------------------------------------------------------------------


def test_finalise_creates_capture_person_and_links_both_tables(session):
    session.execute(
        "INSERT INTO detection_keypoints"
        " (detection_run_id, shot_video_id, video_frame, track_id, region_type, keypoints, noise_scale)"
        " VALUES ('run1', ?, 0, 1, 'full_body', ?, 0.5)",
        (_SVID, _kp(1.0, 133)),
    )
    session.commit()

    assignment = TrackAssignment(
        shot_video_id=_SVID, track_id=1, person_name="alice",
        first_frame=0, last_frame=0,
    )
    seq_ids = finalise_to_db(
        session, detection_run_id="run1", shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
        assignments=[assignment], pose_model="rtmpose-l-133kp",
    )
    seq_id = seq_ids[0]

    from posetrak.db.manage_person import get_person, list_persons

    persons = list_persons(session, _SHOT_ID)
    assert len(persons) == 1
    assert persons[0]["name"] == "alice"
    person_id = persons[0]["id"]

    seq_person = session.execute(
        "SELECT capture_person_id FROM sequence_persons WHERE sequence_id = ?", (seq_id,)
    ).fetchone()
    assert seq_person["capture_person_id"] == person_id

    assignment_row = session.execute(
        "SELECT capture_person_id FROM detection_track_assignments "
        "WHERE detection_run_id = 'run1'"
    ).fetchone()
    assert assignment_row["capture_person_id"] == person_id
    assert get_person(session, person_id) is not None


def test_finalise_reuses_existing_capture_person_across_runs(session):
    """The same name finalised from a second detection run in the same
    capture must resolve to the same capture_persons row, not create a
    second one -- otherwise identity wouldn't actually carry across runs."""
    from posetrak.db.manage_person import create_person, list_persons

    existing_id = create_person(session, _SHOT_ID, "alice")

    session.execute(
        "INSERT INTO detection_keypoints"
        " (detection_run_id, shot_video_id, video_frame, track_id, region_type, keypoints, noise_scale)"
        " VALUES ('run1', ?, 0, 1, 'full_body', ?, 0.5)",
        (_SVID, _kp(1.0, 133)),
    )
    session.commit()

    assignment = TrackAssignment(
        shot_video_id=_SVID, track_id=1, person_name="alice",
        first_frame=0, last_frame=0,
    )
    finalise_to_db(
        session, detection_run_id="run1", shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
        assignments=[assignment], pose_model="rtmpose-l-133kp",
    )

    persons = list_persons(session, _SHOT_ID)
    assert len(persons) == 1
    assert persons[0]["id"] == existing_id


# ---------------------------------------------------------------------------
# auto_assign_and_finalise (segmentation-reuse gap 3)
# ---------------------------------------------------------------------------


def test_auto_assign_maps_track_id_to_persons_ordered(session):
    """track_id N -> persons_ordered[N-1], the same mask-label convention
    app.pose.pose_worker._bboxes_from_mask uses -- no manual stitching."""
    for track_id, fill in [(1, 1.0), (2, 2.0)]:
        session.execute(
            "INSERT INTO detection_keypoints"
            " (detection_run_id, shot_video_id, video_frame, track_id, region_type,"
            "  keypoints, noise_scale)"
            " VALUES ('run1', ?, 0, ?, 'full_body', ?, 0.5)",
            (_SVID, track_id, _kp(fill, 133)),
        )
        session.execute(
            "INSERT INTO person_tracks"
            " (id, detection_run_id, shot_video_id, track_id, first_frame, last_frame)"
            " VALUES (?, 'run1', ?, ?, 0, 0)",
            (f"pt{track_id}", _SVID, track_id),
        )
    session.commit()

    seq_ids = auto_assign_and_finalise(
        session, detection_run_id="run1", shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
        persons_ordered=["alice", "bob"], pose_model="rtmpose-l-133kp",
    )
    assert len(seq_ids) == 2

    rows = session.execute(
        "SELECT person_name, track_id FROM detection_track_assignments "
        "WHERE detection_run_id = 'run1' ORDER BY track_id"
    ).fetchall()
    assert [(r["person_name"], r["track_id"]) for r in rows] == [
        ("alice", 1), ("bob", 2),
    ]

    seq_persons = {
        r["person_name"] for r in session.execute(
            "SELECT person_name FROM sequence_persons WHERE sequence_id IN ({})".format(
                ",".join("?" * len(seq_ids))
            ),
            seq_ids,
        ).fetchall()
    }
    assert seq_persons == {"alice", "bob"}


def test_auto_assign_skips_track_id_outside_persons_ordered(session):
    """A stray track_id with no matching label is skipped, not fatal --
    finalise_to_db still runs for whatever assignments remain valid."""
    session.execute(
        "INSERT INTO detection_keypoints"
        " (detection_run_id, shot_video_id, video_frame, track_id, region_type,"
        "  keypoints, noise_scale)"
        " VALUES ('run1', ?, 0, 1, 'full_body', ?, 0.5)",
        (_SVID, _kp(1.0, 133)),
    )
    session.execute(
        "INSERT INTO person_tracks"
        " (id, detection_run_id, shot_video_id, track_id, first_frame, last_frame)"
        " VALUES ('pt1', 'run1', ?, 1, 0, 0)",
        (_SVID,),
    )
    # track_id 5 has no corresponding entry in persons_ordered (len 1).
    session.execute(
        "INSERT INTO person_tracks"
        " (id, detection_run_id, shot_video_id, track_id, first_frame, last_frame)"
        " VALUES ('pt5', 'run1', ?, 5, 0, 0)",
        (_SVID,),
    )
    session.commit()

    seq_ids = auto_assign_and_finalise(
        session, detection_run_id="run1", shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
        persons_ordered=["alice"], pose_model="rtmpose-l-133kp",
    )
    assert len(seq_ids) == 1
    rows = session.execute(
        "SELECT person_name FROM detection_track_assignments WHERE detection_run_id = 'run1'"
    ).fetchall()
    assert [r["person_name"] for r in rows] == ["alice"]
