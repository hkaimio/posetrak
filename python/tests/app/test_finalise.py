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

from app.pose.finalise import TrackAssignment, finalise_to_db

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
