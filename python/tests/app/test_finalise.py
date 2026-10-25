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


def test_auto_assign_skips_a_label_no_subject_has(session):
    """Labels are permanent subject labels, so persons_ordered can hold ""
    for an unused label -- a track on such a label is skipped, never turned
    into a blank-named person."""
    for track_id in (1, 2):
        session.execute(
            "INSERT INTO detection_keypoints"
            " (detection_run_id, shot_video_id, video_frame, track_id, region_type,"
            "  keypoints, noise_scale)"
            " VALUES ('run1', ?, 0, ?, 'full_body', ?, 0.5)",
            (_SVID, track_id, _kp(1.0, 133)),
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
        persons_ordered=["", "bob"], pose_model="rtmpose-l-133kp",
    )
    assert len(seq_ids) == 1
    names = [r["person_name"] for r in session.execute(
        "SELECT person_name FROM detection_track_assignments WHERE detection_run_id = 'run1'"
    ).fetchall()]
    assert names == ["bob"]


def test_auto_assign_records_the_persons_subject_id(session):
    """The person's registry (subject) id lands on the sequence and on the
    stitched track assignment, not just the name."""
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
    session.commit()

    seq_ids = auto_assign_and_finalise(
        session, detection_run_id="run1", shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
        persons_ordered=["alice"], pose_model="rtmpose-l-133kp",
    )
    person = session.execute(
        "SELECT id FROM capture_subjects WHERE name = 'alice'"
    ).fetchone()["id"]
    sp = session.execute(
        "SELECT capture_person_id, subject_id FROM sequence_persons WHERE sequence_id = ?",
        (seq_ids[0],),
    ).fetchone()
    assert (sp["capture_person_id"], sp["subject_id"]) == (person, person)
    dta = session.execute(
        "SELECT subject_id FROM detection_track_assignments WHERE detection_run_id = 'run1'"
    ).fetchone()
    assert dta["subject_id"] == person


def _finalise_with_hands(session, pose_model, body_width):
    for region, n in (("full_body", body_width), ("hand_l", 21), ("hand_r", 21)):
        session.execute(
            "INSERT OR REPLACE INTO detection_keypoints"
            " (detection_run_id, shot_video_id, video_frame, track_id, region_type, keypoints, noise_scale)"
            " VALUES ('run1', ?, 0, 1, ?, ?, 0.5)",
            (_SVID, region, _kp(1.0, n)),
        )
    session.commit()
    assignment = TrackAssignment(shot_video_id=_SVID, track_id=1, person_name="alice", first_frame=0, last_frame=0)
    return finalise_to_db(
        session, detection_run_id="run1", shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
        assignments=[assignment], pose_model=pose_model,
    )[0]


def _manifest(session, seq_id):
    rows = session.execute(
        "SELECT obs_source, keypoint_idx, name, source FROM pose_sequence_keypoints WHERE sequence_id = ?"
        " ORDER BY obs_source, keypoint_idx", (seq_id,)).fetchall()
    by_source = {}
    for r in rows:
        by_source.setdefault(r["obs_source"], []).append((r["keypoint_idx"], r["name"], r["source"]))
    return by_source


def test_finalise_writes_a_manifest_per_source_for_a_coco133_sequence(session):
    manifest = _manifest(session, _finalise_with_hands(session, "rtmpose-l-133kp", 133))
    assert set(manifest) == {"body", "hand_l", "hand_r"}
    assert len(manifest["body"]) == 133 and manifest["body"][7] == (7, "coco133:left_elbow", "coco133")
    assert manifest["hand_l"][5] == (5, "hand21:left_index_1", "hand21")
    assert manifest["hand_r"][0] == (0, "hand21:right_hand_root", "hand21")


def test_finalise_writes_the_soma_body_manifest_and_replaces_it_on_refinalising(session):
    seq_id = _finalise_with_hands(session, "soma77-dinov3-vitpose", 77)
    manifest = _manifest(session, seq_id)
    assert len(manifest["body"]) == 77 and manifest["body"][13] == (13, "soma77:LeftForeArm", "soma77")
    assert len(manifest["hand_r"]) == 21
    second = _finalise_with_hands(session, "soma77-dinov3-vitpose", 77)
    assert session.execute("SELECT COUNT(*) FROM pose_sequence_keypoints WHERE sequence_id = ?", (seq_id,)).fetchone()[0] == 0
    assert len(_manifest(session, second)["body"]) == 77


def test_finalise_scales_the_candidate_peaks_like_the_body_confidence(session):
    from posetrak.detection.kp_candidates import decode_kp_candidates_blob, encode_kp_candidates_blob

    blob = encode_kp_candidates_blob([0, 0], [1.0, 2.0], [3.0, 4.0], [0.5, 0.25], [0, 1], [5, 6])
    for region, data in (("full_body", _kp(0.5, 77)), ("kp_candidates", blob)):
        session.execute(
            "INSERT INTO detection_keypoints"
            " (detection_run_id, shot_video_id, video_frame, track_id, region_type, keypoints, noise_scale)"
            " VALUES ('run1', ?, 0, 1, ?, ?, 0.5)", (_SVID, region, data))
    session.commit()
    seq_id = finalise_to_db(
        session, detection_run_id="run1", shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
        assignments=[TrackAssignment(_SVID, 1, "alice", 0, 0)], pose_model="soma77-vitpose-h", confidence_scale=5.0)[0]
    rows = {r["source"]: bytes(r["kp_blob"]) for r in session.execute(
        "SELECT source, kp_blob FROM pose_observations WHERE sequence_id = ?", (seq_id,))}
    table = decode_kp_candidates_blob(rows["kp_candidates"])
    np.testing.assert_allclose(table[:, 3], [2.5, 1.25])
    np.testing.assert_array_equal(table[:, [0, 1, 2, 4, 5]], decode_kp_candidates_blob(blob)[:, [0, 1, 2, 4, 5]])
    np.testing.assert_allclose(np.frombuffer(rows["body"], np.float32).reshape(-1, 3)[0, 2], 2.5)


def test_finalise_copies_candidate_rows_with_the_body_manifest(session):
    from posetrak.detection.kp_candidates import encode_kp_candidates_blob

    blob = encode_kp_candidates_blob([0, 0], [1.0, 2.0], [3.0, 4.0], [0.9, 0.2], [0, 1], [5, 6])
    for region, data in (("full_body", _kp(1.0, 77)), ("kp_candidates", blob)):
        session.execute(
            "INSERT INTO detection_keypoints"
            " (detection_run_id, shot_video_id, video_frame, track_id, region_type, keypoints, noise_scale)"
            " VALUES ('run1', ?, 0, 1, ?, ?, 0.5)", (_SVID, region, data))
    session.commit()
    seq_id = finalise_to_db(
        session, detection_run_id="run1", shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
        assignments=[TrackAssignment(_SVID, 1, "alice", 0, 0)], pose_model="soma77-vitpose-h")[0]

    rows = {r["source"]: r for r in session.execute(
        "SELECT source, kp_blob, noise_scale FROM pose_observations WHERE sequence_id = ?", (seq_id,))}
    assert set(rows) == {"body", "kp_candidates"}
    assert bytes(rows["kp_candidates"]["kp_blob"]) == blob            # copied as it is, not confidence-scaled
    assert rows["kp_candidates"]["noise_scale"] == pytest.approx(0.5)
    names = {src: {r[0]: r[1] for r in session.execute(
        "SELECT keypoint_idx, name FROM pose_sequence_keypoints WHERE sequence_id = ? AND obs_source = ?",
        (seq_id, src))} for src in ("body", "kp_candidates")}
    assert len(names["kp_candidates"]) == 77 and names["kp_candidates"] == names["body"]

    from app.pose.db_cache import read_observations_with_edits

    merged = read_observations_with_edits(session, seq_id, _CAM_ID)    # readers of keypoint blobs skip the candidates
    assert set(merged) == {0}
