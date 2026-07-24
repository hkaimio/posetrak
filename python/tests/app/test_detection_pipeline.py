"""Tests for the detection pipeline DB layer (no GPU required)."""
import sqlite3

import numpy as np
import pytest

from posetrak.db.db import create_session, SESSION_SCHEMA_VERSION, get_schema_version
from app.pose.db_cache import (
    create_detection_run,
    mark_run_complete,
    list_detection_runs,
    DetectionBatchWriter,
    read_detections_for_run,
    read_track_spans,
)
from app.pose.backends import PersonDetection, PoseResult


_SHOT_ID = "test-shot-id"
_SYNC_ID = "test-sync-id"
_SVID = "test-sv-id"
_CAM_ID = "test-cam-id"

_TEST_IDS = dict(shot_id=_SHOT_ID, sync_id=_SYNC_ID, svid=_SVID, cam_id=_CAM_ID)


@pytest.fixture
def session(tmp_path):
    db_path = tmp_path / "test.db"
    conn = create_session(db_path)
    conn.row_factory = sqlite3.Row
    _seed_session(conn)
    return conn


def _seed_session(conn):
    from posetrak.db.db import generate_id
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
    """)
    conn.commit()


def test_schema_version(session):
    assert get_schema_version(session) == SESSION_SCHEMA_VERSION


def test_create_and_list_detection_run(session):
    ids = _TEST_IDS
    run_id = create_detection_run(
        session,
        shot_id=ids["shot_id"],
        sync_config_id=ids["sync_id"],
        time_start_s=10.0,
        time_end_s=90.0,
        detector_model="yolo11x",
        pose_model="rtmpose-l-133kp",
        pose_input_width=384,
        pose_input_height=288,
    )
    assert run_id

    runs = list_detection_runs(session, ids["shot_id"])
    assert len(runs) == 1
    assert runs[0]["id"] == run_id
    assert runs[0]["status"] == "running"

    mark_run_complete(session, run_id)
    runs = list_detection_runs(session, ids["shot_id"])
    assert runs[0]["status"] == "complete"


def test_batch_writer_roundtrip(session):
    ids = _TEST_IDS
    run_id = create_detection_run(
        session,
        shot_id=ids["shot_id"],
        sync_config_id=ids["sync_id"],
        time_start_s=0.0,
        time_end_s=10.0,
        detector_model="yolo11x",
        pose_model="rtmpose-l-133kp",
        pose_input_width=384,
        pose_input_height=288,
    )

    writer = DetectionBatchWriter(
        session, run_id, ids["svid"], pose_input_width=384
    )

    # Frame 0: two tracks
    det0 = PersonDetection(
        track_id=1,
        bbox=np.array([100, 200, 80, 160], dtype=np.float32),
        confidence=0.9,
    )
    det1 = PersonDetection(
        track_id=2,
        bbox=np.array([300, 200, 70, 150], dtype=np.float32),
        confidence=0.85,
    )
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 0.9  # confidence
    pose0 = PoseResult(track_id=1, keypoints=kp)
    pose1 = PoseResult(track_id=2, keypoints=kp)

    writer.add_frame(0, [det0, det1], [pose0, pose1], "yolo11x")

    # Frame 1: track 1 only
    det0b = PersonDetection(
        track_id=1,
        bbox=np.array([102, 202, 80, 160], dtype=np.float32),
        confidence=0.88,
    )
    writer.add_frame(1, [det0b], [PoseResult(track_id=1, keypoints=kp)], "yolo11x")

    writer.finalise()

    dets = read_detections_for_run(session, run_id, ids["svid"])
    assert len(dets) == 3  # 2 in frame 0 + 1 in frame 1

    spans = read_track_spans(session, run_id, ids["svid"])
    span_by_id = {s["track_id"]: s for s in spans}
    assert span_by_id[1]["first_frame"] == 0
    assert span_by_id[1]["last_frame"] == 1
    assert span_by_id[2]["first_frame"] == 0
    assert span_by_id[2]["last_frame"] == 0


def test_noise_scale_stored(session):
    """noise_scale = bbox_w / pose_input_width stored in detection_keypoints."""
    ids = _TEST_IDS
    run_id = create_detection_run(
        session,
        shot_id=ids["shot_id"],
        sync_config_id=ids["sync_id"],
        time_start_s=0.0,
        time_end_s=10.0,
        detector_model="yolo11x",
        pose_model="rtmpose-l-133kp",
        pose_input_width=384,
        pose_input_height=288,
    )
    writer = DetectionBatchWriter(session, run_id, ids["svid"], pose_input_width=384)

    bbox_w = 192.0  # half of model input width → noise_scale = 0.5
    det = PersonDetection(
        track_id=1,
        bbox=np.array([100, 200, bbox_w, 160], dtype=np.float32),
        confidence=0.9,
    )
    kp = np.zeros((17, 3), dtype=np.float32)
    writer.add_frame(0, [det], [PoseResult(track_id=1, keypoints=kp)], "yolo11x")
    writer.finalise()

    row = session.execute(
        "SELECT keypoints, noise_scale FROM detection_keypoints WHERE detection_run_id=?",
        (run_id,),
    ).fetchone()
    assert row is not None
    kp_back = np.frombuffer(bytes(row["keypoints"]), dtype=np.float32).reshape(-1, 3)
    assert kp_back.shape == (17, 3)
    assert abs(row["noise_scale"] - 0.5) < 1e-6


def test_detection_run_failed_status(session):
    """mark_run_complete with status='failed' stores correctly."""
    ids = _TEST_IDS
    run_id = create_detection_run(
        session,
        shot_id=ids["shot_id"],
        sync_config_id=ids["sync_id"],
        time_start_s=0.0,
        time_end_s=5.0,
        detector_model="yolo11x",
        pose_model="rtmpose-l-133kp",
    )
    mark_run_complete(session, run_id, status="failed")
    runs = list_detection_runs(session, ids["shot_id"])
    assert runs[0]["status"] == "failed"
    assert runs[0]["completed_at"] is not None


def test_empty_frame_no_crash(session):
    """A frame with no detections does not write any rows."""
    ids = _TEST_IDS
    run_id = create_detection_run(
        session,
        shot_id=ids["shot_id"],
        sync_config_id=ids["sync_id"],
        time_start_s=0.0,
        time_end_s=1.0,
        detector_model="yolo11x",
        pose_model="rtmpose-l-133kp",
        pose_input_width=384,
        pose_input_height=288,
    )
    writer = DetectionBatchWriter(session, run_id, ids["svid"], pose_input_width=384)
    writer.add_frame(0, [], [], "yolo11x")
    writer.finalise()

    dets = read_detections_for_run(session, run_id, ids["svid"])
    assert len(dets) == 0
    spans = read_track_spans(session, run_id, ids["svid"])
    assert len(spans) == 0


# ---------------------------------------------------------------------------
# CameraInfo.label (progress/log messages should show a camera's name, not
# just its UUID -- see hand-detection-refinement live-review feedback)
# ---------------------------------------------------------------------------


def test_load_cameras_populates_label_from_camera_instances(session):
    from posetrak.detection.pipeline import DetectionPipeline

    session.execute(
        "INSERT INTO camera_models (id, manufacturer, model_name) VALUES ('cm1', 'GoPro', 'Hero11 Mini')"
    )
    session.execute(
        "INSERT INTO camera_instances (id, camera_model_id, label) VALUES (?, 'cm1', ?)",
        (_CAM_ID, "gopro-11_mini_01"),
    )
    session.commit()

    pipeline = DetectionPipeline(
        session=session, shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
        time_start_s=0.0, time_end_s=1.0, detector=None, estimator=None,
    )
    assert len(pipeline.cameras) == 1
    assert pipeline.cameras[0].label == "gopro-11_mini_01"


def test_load_cameras_falls_back_to_uuid_when_camera_instance_missing(session):
    """No camera_instances row for this camera_instance_id (e.g. a stale
    reference) -- label should fall back to the UUID, not be blank."""
    from posetrak.detection.pipeline import DetectionPipeline

    pipeline = DetectionPipeline(
        session=session, shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
        time_start_s=0.0, time_end_s=1.0, detector=None, estimator=None,
    )
    assert len(pipeline.cameras) == 1
    assert pipeline.cameras[0].label == _CAM_ID
