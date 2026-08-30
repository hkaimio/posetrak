# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ArUco marker detection (design phase 1a).

See docs/roadmap/features/marker-based-mocap/marker-mocap-design.md §7.1.
Two layers, matching test_detection_pipeline.py's split between DB-layer
tests and (here) a frame-processing test that patches only the frame
source, not the whole pipeline -- no real video file involved, but real
``cv2.aruco`` detection runs against rendered marker images.
"""
from __future__ import annotations

import json
import sqlite3
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from app.pose.db_cache import (
    MarkerKeypointWriter,
    create_marker_detection_run,
    read_marker_keypoints_for_run,
)
from app.setup.fiducial_markers import ARUCO_DICTIONARIES, ArucoDetector, MarkerCornerObs, FiducialDetection
from posetrak.db.db import create_session
from posetrak.detection.marker_pipeline import MarkerDetectionPipeline

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
            VALUES ('{_SVID}', '{_SHOT_ID}', '{_CAM_ID}', '/fake/video.mp4', 0, 1000, 30.0);
        INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id,
                                 video_frame, timestamp_s)
            VALUES ('{_SYNC_ID}', '{_CAM_ID}', '{_SVID}', 0, 0.0);
    """)
    conn.commit()


def _render_marker_image(marker_id: int, dictionary: str = "DICT_4X4_50") -> np.ndarray:
    """A real ArUco marker rendered to a BGR image (see test_fiducial_markers.py)."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary])
    gray = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200)
    padded = cv2.copyMakeBorder(gray, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# DB layer: run creation + keypoint writer round trip
# ---------------------------------------------------------------------------


def test_create_marker_detection_run_stores_config(session):
    ids = _TEST_IDS
    run_id = create_marker_detection_run(
        session,
        shot_id=ids["shot_id"],
        sync_config_id=ids["sync_id"],
        time_start_s=0.0,
        time_end_s=10.0,
        dictionary="DICT_4X4_50",
        marker_ids=["3", "7"],
        min_marker_perimeter_rate=0.01,
        frame_step=2,
    )
    row = session.execute(
        "SELECT detector_type, detector_model, pose_model, config_json "
        "FROM detection_runs WHERE id=?",
        (run_id,),
    ).fetchone()
    assert row["detector_type"] == "aruco"
    assert row["detector_model"] == "aruco:DICT_4X4_50"
    assert row["pose_model"] == ""
    config = json.loads(row["config_json"])
    assert config["marker_ids"] == ["3", "7"]
    assert config["dictionary"] == "DICT_4X4_50"
    assert config["min_marker_perimeter_rate"] == 0.01
    assert config["frame_step"] == 2


def test_existing_pose_runs_default_to_pose_detector_type(session):
    """A plain (non-marker) run created via create_detection_run keeps
    working unchanged -- detector_type defaults to 'pose', config_json to
    NULL, exactly the v46->v47 migration's backward-compat guarantee."""
    from app.pose.db_cache import create_detection_run
    run_id = create_detection_run(
        session,
        shot_id=_TEST_IDS["shot_id"],
        sync_config_id=_TEST_IDS["sync_id"],
        time_start_s=0.0,
        time_end_s=10.0,
        detector_model="yolo11x",
        pose_model="rtmpose-l-133kp",
    )
    row = session.execute(
        "SELECT detector_type, config_json FROM detection_runs WHERE id=?", (run_id,)
    ).fetchone()
    assert row["detector_type"] == "pose"
    assert row["config_json"] is None


def _corner_obs(marker_id: str, corner_index: int, x: float, y: float) -> MarkerCornerObs:
    return MarkerCornerObs(
        marker_type="aruco", marker_id=marker_id, corner_index=corner_index,
        video_id="cam", frame_idx=0, px=x, py=y,
    )


def _fake_detection(marker_id: str, base_xy: tuple[float, float]) -> FiducialDetection:
    bx, by = base_xy
    corners = [_corner_obs(marker_id, i, bx + i, by + i) for i in range(4)]
    return FiducialDetection(marker_type="aruco", marker_id=marker_id, corners=corners)


def test_marker_keypoint_writer_layout_and_missing_marker(session):
    """Blob is 4*n_markers rows, list-position-major by marker_ids; a
    marker absent from the frame keeps NaN x/y and confidence 0 at its
    slot (design §4.1)."""
    ids = _TEST_IDS
    run_id = create_marker_detection_run(
        session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
        time_start_s=0.0, time_end_s=10.0, dictionary="DICT_4X4_50",
        marker_ids=["3", "7"],
    )
    writer = MarkerKeypointWriter(session, run_id, ids["svid"], marker_ids=["3", "7"])

    # Frame 0: only marker "7" seen (slot 1); marker "3" (slot 0) absent.
    writer.add_frame(0, [_fake_detection("7", (10.0, 20.0))])
    writer.finalise()

    kp_by_frame = read_marker_keypoints_for_run(session, run_id, ids["svid"])
    kp = kp_by_frame[0]
    assert kp.shape == (8, 3)  # 4 corners * 2 markers

    # Slot 0 (marker "3"): untouched -> NaN x/y, confidence 0.
    assert np.all(np.isnan(kp[0:4, 0]))
    assert np.all(np.isnan(kp[0:4, 1]))
    assert np.all(kp[0:4, 2] == 0.0)

    # Slot 1 (marker "7"): populated, confidence 1.
    assert np.allclose(kp[4:8, 0], [10.0, 11.0, 12.0, 13.0])
    assert np.allclose(kp[4:8, 1], [20.0, 21.0, 22.0, 23.0])
    assert np.all(kp[4:8, 2] == 1.0)


def test_marker_keypoint_writer_ignores_unconfigured_marker_id(session):
    """A detected marker id outside the prop's configured list is dropped,
    not appended -- the blob width is fixed by marker_ids at run creation."""
    ids = _TEST_IDS
    run_id = create_marker_detection_run(
        session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
        time_start_s=0.0, time_end_s=10.0, dictionary="DICT_4X4_50",
        marker_ids=["3"],
    )
    writer = MarkerKeypointWriter(session, run_id, ids["svid"], marker_ids=["3"])
    writer.add_frame(0, [_fake_detection("99", (1.0, 2.0))])  # not in marker_ids
    writer.finalise()

    kp = read_marker_keypoints_for_run(session, run_id, ids["svid"])[0]
    assert kp.shape == (4, 3)
    assert np.all(np.isnan(kp[:, 0]))
    assert np.all(kp[:, 2] == 0.0)


# ---------------------------------------------------------------------------
# Pipeline: real ArucoDetector against rendered marker images, frame_step,
# and run-status bookkeeping. iter_frames is patched to synthesize frames
# without a real video file, matching how CLI tests mock the pipeline
# boundary rather than decode a real video (test_detect.py).
# ---------------------------------------------------------------------------


def _synthetic_frames(path, first_frame, last_frame):
    """Yields marker '3' present on even frames, absent on odd frames."""
    for i in range(first_frame, last_frame):
        if i % 2 == 0:
            yield i, _render_marker_image(3)
        else:
            yield i, np.full((300, 300, 3), 255, dtype=np.uint8)


def test_pipeline_end_to_end_with_real_detector(session):
    ids = _TEST_IDS
    with patch("posetrak.detection.marker_pipeline.iter_frames", _synthetic_frames):
        pipeline = MarkerDetectionPipeline(
            session,
            shot_id=ids["shot_id"],
            sync_config_id=ids["sync_id"],
            time_start_s=0.0,
            time_end_s=10.0,  # 10s * 30fps sync anchor -> frames [0, 300)
            marker_ids=["3"],
        )
        result = pipeline.run()

    assert result.status == "complete"
    assert result.cameras_processed == [ids["cam_id"]]

    row = session.execute(
        "SELECT status FROM detection_runs WHERE id=?", (result.detection_run_id,)
    ).fetchone()
    assert row["status"] == "complete"

    kp_by_frame = read_marker_keypoints_for_run(session, result.detection_run_id, ids["svid"])
    # Even frames: marker "3" detected -> confidence 1 at all 4 corners.
    assert np.all(kp_by_frame[0][:, 2] == 1.0)
    assert np.all(kp_by_frame[2][:, 2] == 1.0)
    # Odd frames: blank image -> no detection -> NaN/confidence 0.
    assert np.all(kp_by_frame[1][:, 2] == 0.0)
    assert np.all(np.isnan(kp_by_frame[1][:, 0]))


def test_pipeline_frame_step_skips_frames(session):
    ids = _TEST_IDS
    with patch("posetrak.detection.marker_pipeline.iter_frames", _synthetic_frames):
        pipeline = MarkerDetectionPipeline(
            session,
            shot_id=ids["shot_id"],
            sync_config_id=ids["sync_id"],
            time_start_s=0.0,
            time_end_s=10.0,
            marker_ids=["3"],
            frame_step=3,
        )
        result = pipeline.run()

    kp_by_frame = read_marker_keypoints_for_run(session, result.detection_run_id, ids["svid"])
    assert set(kp_by_frame.keys()) == set(range(0, 300, 3))


def test_pipeline_rejects_empty_marker_ids(session):
    ids = _TEST_IDS
    with pytest.raises(ValueError):
        MarkerDetectionPipeline(
            session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
            time_start_s=0.0, time_end_s=10.0, marker_ids=[],
        )


def test_pipeline_rejects_invalid_frame_step(session):
    ids = _TEST_IDS
    with pytest.raises(ValueError):
        MarkerDetectionPipeline(
            session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
            time_start_s=0.0, time_end_s=10.0, marker_ids=["3"], frame_step=0,
        )


def test_real_detector_output_writes_through_correctly(session):
    """ArucoDetector's real output (not the _fake_detection helper above)
    round-trips through MarkerKeypointWriter with the right corner order."""
    ids = _TEST_IDS
    run_id = create_marker_detection_run(
        session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
        time_start_s=0.0, time_end_s=10.0, dictionary="DICT_4X4_50",
        marker_ids=["3"],
    )
    detections = ArucoDetector().detect(_render_marker_image(3), video_id="cam", frame_idx=0)
    assert len(detections) == 1  # sanity: the image really has one marker

    writer = MarkerKeypointWriter(session, run_id, ids["svid"], marker_ids=["3"])
    writer.add_frame(0, detections)
    writer.finalise()

    kp = read_marker_keypoints_for_run(session, run_id, ids["svid"])[0]
    assert kp.shape == (4, 3)
    assert np.all(kp[:, 2] == 1.0)  # all 4 corners detected
    assert np.all(np.isfinite(kp[:, :2]))
