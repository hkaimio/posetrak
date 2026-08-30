# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for app.pose.finalise.finalise_object_to_db (marker-based-mocap
design doc §4.3, §7.1 sub-phase 1d).

See status.md's 2026-08-30 entry for why finalisation is sub-phase 1d
(before review, 1e) rather than the original 1d/1e order: an object has
no track-to-person stitching decision to make, so finalisation is
automatic/immediate rather than gated behind a review step.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from posetrak.db.db import create_session, generate_id
from app.pose.db_cache import create_detection_run, create_marker_detection_run, MarkerKeypointWriter
from app.pose.finalise import finalise_object_to_db
from posetrak.db.manage_capture_object import create_capture_object
from posetrak.db.manage_marker_body import import_marker_body_str

_SHOT_ID = "test-shot-id"
_SYNC_ID = "test-sync-id"
_SVID = "test-sv-id"
_CAM_ID = "test-cam-id"

_MARKER_BODY_YAML = """\
name: test-bokken
units: meters
markers:
  - name: hilt
    type: aruco
    dictionary: DICT_4X4_50
    id: "3"
    size: 0.05
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
  - name: tip
    type: aruco
    dictionary: DICT_4X4_50
    id: "7"
    size: 0.03
    center: [0.0, 0.9, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
"""


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
            VALUES ('{_SVID}', '{_SHOT_ID}', '{_CAM_ID}', '/fake/video.mp4', 0, 1000, 30.0);
        INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id,
                                 video_frame, timestamp_s)
            VALUES ('{_SYNC_ID}', '{_CAM_ID}', '{_SVID}', 0, 0.0);
    """)
    conn.commit()
    return conn


def _write_marker_run(session, marker_ids, **kwargs) -> str:
    run_id = create_marker_detection_run(
        session, shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
        time_start_s=0.0, time_end_s=1.0, dictionary="DICT_4X4_50",
        marker_ids=marker_ids, **kwargs,
    )
    writer = MarkerKeypointWriter(session, run_id, _SVID, marker_ids=marker_ids)
    for frame in (0, 1):
        writer.add_frame(frame, [])  # buffered only -- writes NaN placeholders
    writer.finalise()  # flush the buffer to detection_keypoints before overwriting

    # Overwrite with a deterministic, known-good blob for the test to assert on.
    for frame in (0, 1):
        kp = np.zeros((4 * len(marker_ids), 3), dtype=np.float32)
        kp[:, 0] = 100.0 + frame
        kp[:, 1] = 200.0 + frame
        kp[:, 2] = 1.0
        session.execute(
            "UPDATE detection_keypoints SET keypoints=? "
            "WHERE detection_run_id=? AND shot_video_id=? AND video_frame=?",
            (kp.tobytes(), run_id, _SVID, frame),
        )
    session.commit()
    return run_id


def test_finalise_object_creates_sequence_and_manifest_with_marker_body(session):
    body_id = import_marker_body_str(session, _MARKER_BODY_YAML, name="Test Bokken")
    object_id = create_capture_object(session, _SHOT_ID, "bokken-A", body_id)
    run_id = _write_marker_run(
        session, ["3", "7"], marker_body_definition_id=body_id, capture_object_id=object_id,
    )

    seq_id = finalise_object_to_db(session, run_id)

    seq = session.execute(
        "SELECT shot_id, sync_config_id, detection_run_id FROM pose_observation_sequences "
        "WHERE id=?", (seq_id,),
    ).fetchone()
    assert seq["shot_id"] == _SHOT_ID
    assert seq["detection_run_id"] == run_id

    manifest = session.execute(
        "SELECT keypoint_idx, name, source FROM pose_sequence_keypoints "
        "WHERE sequence_id=? ORDER BY keypoint_idx", (seq_id,),
    ).fetchall()
    names = [r["name"] for r in manifest]
    assert names == [
        "hilt:c0", "hilt:c1", "hilt:c2", "hilt:c3",
        "tip:c0", "tip:c1", "tip:c2", "tip:c3",
    ]
    assert all(r["source"] == "aruco" for r in manifest)

    # No sequence_persons / detection_track_assignments rows -- an object
    # is not a person, and finalise_object_to_db never touches those tables.
    assert session.execute(
        "SELECT COUNT(*) FROM sequence_persons WHERE sequence_id=?", (seq_id,)
    ).fetchone()[0] == 0


def test_finalise_object_without_marker_body_uses_bare_marker_ids(session):
    # A real capture_object, but the run wasn't told its marker_body_definition_id
    # (an unusual state -- create_marker_detection_run always passes both
    # together in practice -- but finalise_object_to_db's two lookups are
    # independent, so this exercises the fallback deliberately).
    body_id = import_marker_body_str(session, _MARKER_BODY_YAML, name="Test Bokken")
    object_id = create_capture_object(session, _SHOT_ID, "bokken-A", body_id)
    run_id = _write_marker_run(session, ["3", "7"], capture_object_id=object_id)

    seq_id = finalise_object_to_db(session, run_id)
    names = [
        r["name"] for r in session.execute(
            "SELECT name FROM pose_sequence_keypoints WHERE sequence_id=? ORDER BY keypoint_idx",
            (seq_id,),
        )
    ]
    assert names == ["3:c0", "3:c1", "3:c2", "3:c3", "7:c0", "7:c1", "7:c2", "7:c3"]


def test_finalise_object_writes_observations_per_frame(session):
    body_id = import_marker_body_str(session, _MARKER_BODY_YAML, name="Test Bokken")
    object_id = create_capture_object(session, _SHOT_ID, "bokken-A", body_id)
    run_id = _write_marker_run(
        session, ["3", "7"], marker_body_definition_id=body_id, capture_object_id=object_id,
    )

    seq_id = finalise_object_to_db(session, run_id)

    obs = session.execute(
        "SELECT video_frame, person_id, source, kp_blob FROM pose_observations "
        "WHERE sequence_id=? ORDER BY video_frame", (seq_id,),
    ).fetchall()
    assert [r["video_frame"] for r in obs] == [0, 1]
    assert all(r["person_id"] == 0 for r in obs)
    assert all(r["source"] == "markers" for r in obs)
    kp0 = np.frombuffer(bytes(obs[0]["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    assert kp0.shape == (8, 3)
    assert np.allclose(kp0[:, 0], 100.0)
    assert np.allclose(kp0[:, 2], 1.0)


def test_finalise_object_rejects_non_marker_run(session):
    run_id = create_detection_run(
        session, shot_id=_SHOT_ID, sync_config_id=_SYNC_ID, time_start_s=0.0, time_end_s=1.0,
        detector_model="yolox-x", pose_model="rtmpose-l-133kp",
    )
    with pytest.raises(ValueError, match="not a marker run"):
        finalise_object_to_db(session, run_id)


def test_finalise_object_rejects_run_without_capture_object_id(session):
    run_id = _write_marker_run(session, ["3"])  # no capture_object_id given
    with pytest.raises(ValueError, match="no capture_object_id"):
        finalise_object_to_db(session, run_id)


def test_refinalise_object_overwrites_when_safe(session):
    body_id = import_marker_body_str(session, _MARKER_BODY_YAML, name="Test Bokken")
    object_id = create_capture_object(session, _SHOT_ID, "bokken-A", body_id)
    run_id = _write_marker_run(
        session, ["3", "7"], marker_body_definition_id=body_id, capture_object_id=object_id,
    )

    seq_id_1 = finalise_object_to_db(session, run_id)
    seq_id_2 = finalise_object_to_db(session, run_id)  # re-finalise, no tracking/edits yet

    assert seq_id_1 != seq_id_2
    assert session.execute(
        "SELECT COUNT(*) FROM pose_observation_sequences WHERE detection_run_id=?", (run_id,)
    ).fetchone()[0] == 1
    assert session.execute(
        "SELECT id FROM pose_observation_sequences WHERE id=?", (seq_id_1,)
    ).fetchone() is None


def test_refinalise_object_refuses_once_edits_exist(session):
    body_id = import_marker_body_str(session, _MARKER_BODY_YAML, name="Test Bokken")
    object_id = create_capture_object(session, _SHOT_ID, "bokken-A", body_id)
    run_id = _write_marker_run(
        session, ["3", "7"], marker_body_definition_id=body_id, capture_object_id=object_id,
    )
    seq_id = finalise_object_to_db(session, run_id)

    session.execute(
        "INSERT INTO pose_observation_edits"
        " (id, sequence_id, camera_instance_id, video_frame, kp_blob, kp_mask)"
        " VALUES (?, ?, ?, 0, ?, ?)",
        (generate_id(), seq_id, _CAM_ID, np.zeros((8, 3), dtype=np.float32).tobytes(), b"\x01"),
    )
    session.commit()

    with pytest.raises(RuntimeError, match="already has tracking results and/or manual"):
        finalise_object_to_db(session, run_id)

    # Refused before touching anything.
    assert session.execute(
        "SELECT count(*) FROM pose_observation_sequences WHERE id=?", (seq_id,)
    ).fetchone()[0] == 1
