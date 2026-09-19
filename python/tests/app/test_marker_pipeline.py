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
    DotCandidateWriter,
    MarkerKeypointWriter,
    create_marker_detection_run,
    read_dot_candidates_for_run,
    read_marker_keypoints_for_run,
)
from app.setup.fiducial_markers import (
    ARUCO_DICTIONARIES,
    ArucoDetector,
    MarkerCornerObs,
    FiducialDetection,
    load_marker_body_yaml,
)
from posetrak.db.db import create_session
from posetrak.db.manage_capture_object import create_capture_object
from posetrak.db.manage_marker_body import import_marker_body_str
from posetrak.detection.dot_blob_detector import BlobCandidate
from posetrak.detection.marker_pipeline import MarkerDetectionPipeline, load_pipeline_for_capture_object

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


def _fake_blob(cx: float, cy: float, *, area: float, compactness: float) -> BlobCandidate:
    diameter = 2.0 * (area / np.pi) ** 0.5
    return BlobCandidate(cx=cx, cy=cy, area=area, compactness=compactness, bbox=(0, 0, 1, 1),
                          major_axis_px=diameter, minor_axis_px=diameter)


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


def test_marker_keypoint_writer_uses_near_zero_noise_scale(session):
    """noise_scale (-> Observation::crop_scale) must be ~0 for markers, not
    the person pipeline's 1.0 default (code review finding #4, status.md
    2026-08-31 entry): a coded ArUco corner is found by direct sub-pixel
    corner refinement on the full-resolution frame, with no fixed-input-
    resolution network stage for crop_scale to describe -- letting it stay
    1.0 gave marker observations the same detection-algorithm-error
    contribution as an interpolated pose keypoint, silently under-trusting
    the corners' real precision relative to camera/calibration error."""
    ids = _TEST_IDS
    run_id = create_marker_detection_run(
        session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
        time_start_s=0.0, time_end_s=10.0, dictionary="DICT_4X4_50",
        marker_ids=["3"],
    )
    writer = MarkerKeypointWriter(session, run_id, ids["svid"], marker_ids=["3"])
    writer.add_frame(0, [_fake_detection("3", (10.0, 20.0))])
    writer.finalise()

    noise_scale = session.execute(
        "SELECT noise_scale FROM detection_keypoints "
        "WHERE detection_run_id=? AND shot_video_id=? AND video_frame=0",
        (run_id, ids["svid"]),
    ).fetchone()[0]
    assert noise_scale == 0.0


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
# DotCandidateWriter: anonymous reflective-dot candidates, variable-N per
# frame (see dot-assignment-architecture-design.md).
# ---------------------------------------------------------------------------


def test_dot_candidate_writer_round_trips_variable_candidate_counts(session):
    ids = _TEST_IDS
    run_id = create_marker_detection_run(
        session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
        time_start_s=0.0, time_end_s=10.0, dictionary="DICT_4X4_50",
        marker_ids=["3"],
    )
    writer = DotCandidateWriter(session, run_id, ids["svid"])

    writer.add_frame(0, [
        _fake_blob(10.0, 20.0, area=12.5, compactness=0.9),
        _fake_blob(30.0, 40.0, area=8.0, compactness=0.85),
    ])
    writer.add_frame(1, [])  # processed, nothing seen -- still writes a row
    writer.finalise()

    candidates = read_dot_candidates_for_run(session, run_id, ids["svid"])
    assert set(candidates.keys()) == {0, 1}

    d0 = 2.0 * (12.5 / np.pi) ** 0.5
    d1 = 2.0 * (8.0 / np.pi) ** 0.5
    assert candidates[0].shape == (2, 9)
    assert np.allclose(candidates[0][0], [10.0, 20.0, 12.5, 0.9, d0, d0, 0.0, 0.0, -1.0])
    assert np.allclose(candidates[0][1], [30.0, 40.0, 8.0, 0.85, d1, d1, 0.0, 0.0, -1.0])

    assert candidates[1].shape == (0, 9)


def test_dot_candidate_writer_uses_near_zero_noise_scale(session):
    """Same reasoning as the ArUco corner writer's own test above: a dot
    centroid comes from thresholding the full-resolution frame directly, not
    a fixed-input-resolution network, so noise_scale (-> crop_scale) must be
    ~0, not the person pipeline's 1.0 default."""
    ids = _TEST_IDS
    run_id = create_marker_detection_run(
        session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
        time_start_s=0.0, time_end_s=10.0, dictionary="DICT_4X4_50",
        marker_ids=["3"],
    )
    writer = DotCandidateWriter(session, run_id, ids["svid"])
    writer.add_frame(0, [_fake_blob(10.0, 20.0, area=12.5, compactness=0.9)])
    writer.finalise()

    noise_scale = session.execute(
        "SELECT noise_scale FROM detection_keypoints "
        "WHERE detection_run_id=? AND shot_video_id=? AND video_frame=0 AND region_type='dots'",
        (run_id, ids["svid"]),
    ).fetchone()[0]
    assert noise_scale == 0.0


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


def _synthetic_dot_frames(path, first_frame, last_frame):
    """Yields a dark frame with one bright dot at (50, 60) on even frames,
    a plain dark frame on odd frames -- unlike _synthetic_frames' all-white
    background, dark enough that only the drawn dot exceeds the detector's
    default threshold."""
    for i in range(first_frame, last_frame):
        frame = np.full((300, 300, 3), 20, dtype=np.uint8)
        if i % 2 == 0:
            cv2.circle(frame, (50, 60), 6, (250, 250, 250), thickness=-1)
        yield i, frame


def test_pipeline_writes_dot_candidates_for_an_enabled_camera(session):
    ids = _TEST_IDS
    with patch("posetrak.detection.marker_pipeline.iter_frames", _synthetic_dot_frames):
        pipeline = MarkerDetectionPipeline(
            session,
            shot_id=ids["shot_id"],
            sync_config_id=ids["sync_id"],
            time_start_s=0.0,
            time_end_s=10.0,
            marker_ids=["3"],
            detect_dots_for_cameras={ids["cam_id"]},
        )
        result = pipeline.run()

    assert result.status == "complete"
    candidates = read_dot_candidates_for_run(session, result.detection_run_id, ids["svid"])
    # Even frame: the drawn dot is detected. Odd frame: none seen, but the
    # frame was still processed -- an empty row, not a missing one.
    assert candidates[0].shape == (1, 9)
    assert np.allclose(candidates[0][0, :2], [50.0, 60.0], atol=1.0)
    assert candidates[1].shape == (0, 9)


def test_pipeline_assigns_a_consistent_tracklet_id_to_a_stationary_dot(session):
    """The same real dot detected on frame 0, 2, 4, ... (odd frames miss it
    entirely -- see _synthetic_dot_frames) should keep the same tracklet_id
    throughout -- a one-frame gap is well within DotTrackletLinker's own
    default max_missed, so the track shouldn't age out between hits."""
    ids = _TEST_IDS
    with patch("posetrak.detection.marker_pipeline.iter_frames", _synthetic_dot_frames):
        pipeline = MarkerDetectionPipeline(
            session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
            time_start_s=0.0, time_end_s=2.0, marker_ids=["3"],
            detect_dots_for_cameras={ids["cam_id"]},
        )
        result = pipeline.run()

    candidates = read_dot_candidates_for_run(session, result.detection_run_id, ids["svid"])
    tracklet_ids = [candidates[f][0, 8] for f in range(0, 60, 2)]  # every even frame
    assert len(set(tracklet_ids)) == 1
    assert tracklet_ids[0] != -1


def test_pipeline_writes_no_dot_candidates_when_camera_not_enabled(session):
    """detect_dots_for_cameras defaults to disabled for every camera -- no
    detection_keypoints rows with region_type='dots' at all, not even empty
    ones, when a camera isn't in the set."""
    ids = _TEST_IDS
    with patch("posetrak.detection.marker_pipeline.iter_frames", _synthetic_dot_frames):
        pipeline = MarkerDetectionPipeline(
            session,
            shot_id=ids["shot_id"],
            sync_config_id=ids["sync_id"],
            time_start_s=0.0,
            time_end_s=10.0,
            marker_ids=["3"],
        )
        result = pipeline.run()

    candidates = read_dot_candidates_for_run(session, result.detection_run_id, ids["svid"])
    assert candidates == {}


def _synthetic_dim_dot_frames(path, first_frame, last_frame):
    """Dark background everywhere; a DIM dot (value=90, well below the
    detector's default raw-brightness threshold of 235) drawn only in a
    narrow frame window -- background sampling (spread across the whole
    requested range, median of ~40 samples) mostly misses this window, so
    the background frame stays close to plain background there, letting a
    background-subtraction test tell "found via residual" apart from
    "found via raw brightness" cleanly."""
    for i in range(first_frame, last_frame):
        frame = np.full((300, 300, 3), 20, dtype=np.uint8)
        if 100 <= i < 110:
            cv2.circle(frame, (50, 60), 6, (90, 90, 90), thickness=-1)
        yield i, frame


def test_pipeline_bg_subtract_finds_a_dim_highlight_raw_threshold_misses(session):
    """dot_bg_subtract=True + a low dot_threshold recovers a dim highlight
    that the default raw-brightness path (threshold=235) never sees at all
    -- the real 2026-09-06 finding this pipeline wiring exists for."""
    ids = _TEST_IDS
    with patch("posetrak.detection.marker_pipeline.iter_frames", _synthetic_dim_dot_frames):
        without_bg = MarkerDetectionPipeline(
            session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
            time_start_s=0.0, time_end_s=10.0, marker_ids=["3"],
            detect_dots_for_cameras={ids["cam_id"]},
        )
        result_without = without_bg.run()
    candidates_without = read_dot_candidates_for_run(
        session, result_without.detection_run_id, ids["svid"]
    )
    assert all(c.shape[0] == 0 for c in candidates_without.values())

    with patch("posetrak.detection.marker_pipeline.iter_frames", _synthetic_dim_dot_frames):
        with_bg = MarkerDetectionPipeline(
            session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
            time_start_s=0.0, time_end_s=10.0, marker_ids=["3"],
            detect_dots_for_cameras={ids["cam_id"]},
            dot_bg_subtract=True, dot_threshold=40,
        )
        result_with = with_bg.run()
    candidates_with = read_dot_candidates_for_run(session, result_with.detection_run_id, ids["svid"])
    assert candidates_with[104].shape == (1, 9)
    assert np.allclose(candidates_with[104][0, :2], [50.0, 60.0], atol=1.0)

    run_row = session.execute(
        "SELECT config_json FROM detection_runs WHERE id=?", (result_with.detection_run_id,)
    ).fetchone()
    dot_config = json.loads(run_row["config_json"])["dot_detection"]
    assert dot_config["bg_subtract"] is True
    assert dot_config["threshold"] == 40


def _synthetic_fused_blob_frames(path, first_frame, last_frame):
    """Dark background everywhere; a bright rectangle (simulating a subject
    in a pose the background samples mostly don't cover) plus a real,
    brighter dot on top of it, in a narrow frame window."""
    for i in range(first_frame, last_frame):
        frame = np.full((300, 300, 3), 20, dtype=np.uint8)
        if 100 <= i < 110:
            cv2.rectangle(frame, (20, 20), (140, 140), (90, 90, 90), thickness=-1)
            cv2.circle(frame, (50, 60), 6, (250, 250, 250), thickness=-1)
        yield i, frame


def test_pipeline_background_mode_blacklist_recovers_a_marker_subtract_fuses_away(session):
    """The real 2026-09-08 pipeline-level wiring check: 'subtract' mode
    fuses the marker into the surrounding atypical-pose blob and loses it
    (same failure as dot_blob_detector.py's own unit test, checked here at
    the pipeline level to confirm background_mode is actually threaded
    through); 'blacklist' mode keeps the marker as its own small contour."""
    ids = _TEST_IDS
    with patch("posetrak.detection.marker_pipeline.iter_frames", _synthetic_fused_blob_frames):
        subtract = MarkerDetectionPipeline(
            session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
            time_start_s=0.0, time_end_s=10.0, marker_ids=["3"],
            detect_dots_for_cameras={ids["cam_id"]},
            dot_bg_subtract=True, dot_threshold=40, dot_background_mode="subtract",
        )
        result_subtract = subtract.run()
    candidates_subtract = read_dot_candidates_for_run(session, result_subtract.detection_run_id, ids["svid"])
    assert candidates_subtract[104].shape == (0, 9)

    with patch("posetrak.detection.marker_pipeline.iter_frames", _synthetic_fused_blob_frames):
        blacklist = MarkerDetectionPipeline(
            session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
            time_start_s=0.0, time_end_s=10.0, marker_ids=["3"],
            detect_dots_for_cameras={ids["cam_id"]},
            dot_threshold=200, dot_background_mode="blacklist",
        )
        result_blacklist = blacklist.run()
    candidates_blacklist = read_dot_candidates_for_run(session, result_blacklist.detection_run_id, ids["svid"])
    assert candidates_blacklist[104].shape == (1, 9)
    assert np.allclose(candidates_blacklist[104][0, :2], [50.0, 60.0], atol=1.0)

    run_row = session.execute(
        "SELECT config_json FROM detection_runs WHERE id=?", (result_blacklist.detection_run_id,)
    ).fetchone()
    dot_config = json.loads(run_row["config_json"])["dot_detection"]
    assert dot_config["background_mode"] == "blacklist"


def test_pipeline_dot_threshold_by_camera_overrides_the_global_default(session):
    """A camera-specific threshold in dot_threshold_by_camera should win
    over dot_threshold for that camera -- the real 2026-09-08 finding that
    different cameras' sensors/tone-mapping cap real markers at very
    different absolute brightness levels."""
    ids = _TEST_IDS
    with patch("posetrak.detection.marker_pipeline.iter_frames", _synthetic_dim_dot_frames):
        pipeline = MarkerDetectionPipeline(
            session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
            time_start_s=0.0, time_end_s=10.0, marker_ids=["3"],
            detect_dots_for_cameras={ids["cam_id"]},
            dot_threshold=235,  # would miss the dim (value=90) dot on its own
            dot_threshold_by_camera={ids["cam_id"]: 80},
            dot_background_mode="blacklist",
        )
        result = pipeline.run()
    candidates = read_dot_candidates_for_run(session, result.detection_run_id, ids["svid"])
    assert candidates[104].shape == (1, 9)
    assert np.allclose(candidates[104][0, :2], [50.0, 60.0], atol=1.0)


def test_pipeline_passes_per_camera_max_saturation_and_blacklist_frac(session):
    """dot_max_saturation_by_camera / dot_blacklist_frac_by_camera override
    their scalar defaults for a specific camera, exactly like
    dot_threshold_by_camera above -- the 2026-09-11 finding (person-marker
    redesign phase P-D, status.md) that a real capture mixing camera
    models needs both overridden per camera too (one camera's markers
    render meaningfully colour-tinted; the glare veto tuned on a
    reflective-prop capture proved too tight for person-worn markers).
    Checked at the detect_blobs() call site itself (mocked) since the
    per-camera .get() plumbing is what's under test, not detect_blobs()'s
    own thresholding (covered elsewhere)."""
    ids = _TEST_IDS
    calls = []

    def _fake_detect_blobs(gray, **kwargs):
        calls.append(kwargs)
        return []

    with patch("posetrak.detection.marker_pipeline.iter_frames", _synthetic_dim_dot_frames), \
         patch("posetrak.detection.marker_pipeline.detect_blobs", _fake_detect_blobs):
        pipeline = MarkerDetectionPipeline(
            session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
            time_start_s=0.0, time_end_s=10.0, marker_ids=["3"],
            detect_dots_for_cameras={ids["cam_id"]},
            dot_background_mode="blacklist",
            dot_max_saturation=255.0, dot_max_saturation_by_camera={ids["cam_id"]: 90.0},
            dot_blacklist_frac=0.7, dot_blacklist_frac_by_camera={ids["cam_id"]: 0.95},
        )
        pipeline.run()

    assert calls, "detect_blobs was never called"
    assert all(c["max_saturation"] == 90.0 for c in calls)
    assert all(c["blacklist_frac"] == 0.95 for c in calls)


def test_run_parallel_matches_run_on_a_real_tiny_video_file(session, tmp_path):
    """run_parallel() spawns a real subprocess (ProcessPoolExecutor, needed
    to be picklable across Windows' spawn start method) -- unlike every
    other test in this file, a mock.patch on iter_frames only patches this
    *process's* copy of the module, not a freshly spawned worker's, so this
    one needs a genuine small video file on disk rather than a patched
    frame generator. Draws the same ArUco marker _render_marker_image()
    already builds for the non-dot pipeline tests, confirming run_parallel()
    finds it via a real subprocess exactly like run() does via the same
    process."""
    frame = _render_marker_image(3)
    video_path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                              (frame.shape[1], frame.shape[0]))
    for _ in range(5):
        writer.write(frame)
    writer.release()

    ids = _TEST_IDS
    session.execute("UPDATE capture_videos SET file_path=? WHERE id=?", (str(video_path), ids["svid"]))
    session.commit()

    pipeline = MarkerDetectionPipeline(
        session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
        time_start_s=0.0, time_end_s=10.0, marker_ids=["3"],
    )
    result = pipeline.run_parallel(max_workers=1)

    assert result.status == "complete"
    assert result.cameras_processed == [ids["cam_id"]]
    assert result.frames_processed == 5
    keypoints = read_marker_keypoints_for_run(session, result.detection_run_id, ids["svid"])
    assert len(keypoints) == 5
    assert any((kp[:, 2] > 0).any() for kp in keypoints.values())  # the marker was actually found


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


# ---------------------------------------------------------------------------
# Marker-body-driven mode (design phase 1c) -- rig_config constructor path
# and the load_pipeline_for_capture_object factory.
# ---------------------------------------------------------------------------

_ONE_MARKER_BODY_YAML = """\
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
"""

_DOT_ONLY_BODY_YAML = "name: dot-prop\nunits: meters\nmarkers:\n  - name: a\n    type: reflective_dot\n    center: [0,0,0]\n"


def test_rig_config_constructor_derives_marker_ids_from_config():
    config = load_marker_body_yaml(_ONE_MARKER_BODY_YAML)
    # marker_corners keys are the coded markers' dictionary ids ("3" here);
    # constructing with rig_config= should derive marker_ids from that,
    # not require it separately.
    assert list(config.marker_corners.keys()) == ["3"]


def test_rig_config_pipeline_rejects_dot_only_body(session):
    ids = _TEST_IDS
    config = load_marker_body_yaml(_DOT_ONLY_BODY_YAML)
    with pytest.raises(ValueError, match="no coded markers"):
        MarkerDetectionPipeline(
            session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
            time_start_s=0.0, time_end_s=10.0, rig_config=config,
        )


def test_rig_config_pipeline_end_to_end(session):
    ids = _TEST_IDS
    body_id = import_marker_body_str(session, _ONE_MARKER_BODY_YAML, name="Test Bokken")
    object_id = create_capture_object(session, ids["shot_id"], "bokken-A", body_id)
    config = load_marker_body_yaml(_ONE_MARKER_BODY_YAML)
    with patch("posetrak.detection.marker_pipeline.iter_frames", _synthetic_frames):
        pipeline = MarkerDetectionPipeline(
            session, shot_id=ids["shot_id"], sync_config_id=ids["sync_id"],
            time_start_s=0.0, time_end_s=1.0, rig_config=config,
            capture_object_id=object_id, marker_body_definition_id=body_id,
        )
        result = pipeline.run()

    row = session.execute(
        "SELECT capture_object_id, config_json FROM detection_runs WHERE id=?",
        (result.detection_run_id,),
    ).fetchone()
    assert row["capture_object_id"] == object_id
    config_json = json.loads(row["config_json"])
    assert config_json["marker_body_definition_id"] == body_id
    assert config_json["capture_object_id"] == object_id
    assert config_json["marker_ids"] == ["3"]

    kp_by_frame = read_marker_keypoints_for_run(session, result.detection_run_id, ids["svid"])
    assert np.all(kp_by_frame[0][:, 2] == 1.0)  # marker "3" seen on even frames


def test_load_pipeline_for_capture_object(session):
    ids = _TEST_IDS
    body_id = import_marker_body_str(session, _ONE_MARKER_BODY_YAML, name="Test Bokken")
    object_id = create_capture_object(session, ids["shot_id"], "bokken-A", body_id)

    with patch("posetrak.detection.marker_pipeline.iter_frames", _synthetic_frames):
        pipeline = load_pipeline_for_capture_object(
            session, capture_object_id=object_id, sync_config_id=ids["sync_id"],
            time_start_s=0.0, time_end_s=1.0,
        )
        result = pipeline.run()

    row = session.execute(
        "SELECT capture_object_id FROM detection_runs WHERE id=?", (result.detection_run_id,)
    ).fetchone()
    assert row["capture_object_id"] == object_id

    kp_by_frame = read_marker_keypoints_for_run(session, result.detection_run_id, ids["svid"])
    assert np.all(kp_by_frame[0][:, 2] == 1.0)


def test_load_pipeline_for_capture_object_missing_object(session):
    with pytest.raises(ValueError, match="capture_objects"):
        load_pipeline_for_capture_object(
            session, capture_object_id="does-not-exist", sync_config_id=_TEST_IDS["sync_id"],
            time_start_s=0.0, time_end_s=1.0,
        )


def test_load_pipeline_for_capture_object_dot_only_body_raises(session):
    body_id = import_marker_body_str(session, _DOT_ONLY_BODY_YAML, name="Dot Prop")
    object_id = create_capture_object(session, _TEST_IDS["shot_id"], "dot-prop-A", body_id)
    with pytest.raises(ValueError, match="no coded markers"):
        load_pipeline_for_capture_object(
            session, capture_object_id=object_id, sync_config_id=_TEST_IDS["sync_id"],
            time_start_s=0.0, time_end_s=1.0,
        )
