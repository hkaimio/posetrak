"""Tests for page_extrinsics._load_states_from_capture.

Covers Phase 1 of docs/roadmap/features/extrinsics-improvements/
extrinsics-improvements-design.md ("Video frame source"): loading
CamCalibState directly from a capture's registered video files instead of
matching filenames in an exported-PNG directory.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from app.setup.page_extrinsics import _load_states_from_capture


def _pack_matrix(K: np.ndarray) -> bytes:
    return struct.pack("<9d", *K.flatten())


def _pack_dist(coeffs: list[float]) -> bytes:
    return struct.pack(f"<{len(coeffs)}d", *coeffs)


@pytest.fixture()
def capture_db(tmp_path):
    """Session DB with one capture, two cameras with video files + intrinsics."""
    from posetrak.db.db import create_session

    conn = create_session(tmp_path / "extrinsics_video_test.db")
    conn.execute("PRAGMA foreign_keys = OFF")

    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01T00:00:00Z')"
    )
    conn.execute("INSERT INTO captures (id, session_id, capture_number) VALUES ('shot1', 'sess1', 1)")

    conn.execute("INSERT INTO camera_models (id, manufacturer, model_name) VALUES ('cm1', 'Test', 'CamModel')")
    conn.execute(
        "INSERT INTO camera_modes (id, camera_model_id, width_px, height_px, nominal_fps)"
        " VALUES ('mode1', 'cm1', 1920, 1080, 30.0)"
    )
    conn.execute("INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('ci1', 'cm1', 'cam_A')")
    conn.execute("INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('ci2', 'cm1', 'cam_B')")

    K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    conn.execute(
        "INSERT INTO intrinsics_calibrations"
        " (id, camera_mode_id, calibrated_at, distortion_model, fx, fy, cx, cy,"
        "  dist_coeffs, matrix_original)"
        " VALUES ('calib_old', 'mode1', '2025-01-01', 'radtan', 1000.0, 1000.0, 960.0, 540.0, ?, ?)",
        (_pack_dist([-0.1, 0.05, 0.001, -0.001]), _pack_matrix(K)),
    )
    conn.execute(
        "INSERT INTO intrinsics_calibrations"
        " (id, camera_mode_id, calibrated_at, distortion_model, fx, fy, cx, cy,"
        "  dist_coeffs, matrix_original)"
        " VALUES ('calib_new', 'mode1', '2026-01-01', 'fisheye', 1010.0, 1010.0, 962.0, 542.0, ?, ?)",
        (_pack_dist([0.01, -0.02, 0.0, 0.0]), _pack_matrix(K)),
    )

    conn.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,"
        " first_video_frame, last_video_frame, actual_fps, camera_mode_id)"
        " VALUES ('sv1', 'shot1', 'ci1', '/videos/cam_A.mp4', 0, 299, 30.0, 'mode1')"
    )
    conn.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,"
        " first_video_frame, last_video_frame, actual_fps, camera_mode_id)"
        " VALUES ('sv2', 'shot1', 'ci2', '/videos/cam_B.mp4', 0, 199, 30.0, 'mode1')"
    )

    conn.commit()
    yield conn
    conn.close()


def test_loads_one_state_per_camera(capture_db) -> None:
    states = _load_states_from_capture(capture_db, "shot1")
    assert {s.label for s in states} == {"cam_A", "cam_B"}


def test_video_id_matches_label(capture_db) -> None:
    states = _load_states_from_capture(capture_db, "shot1")
    for s in states:
        assert s.video_id == s.label


def test_file_path_and_frame_range_set(capture_db) -> None:
    states = {s.label: s for s in _load_states_from_capture(capture_db, "shot1")}
    assert states["cam_A"].file_path == "/videos/cam_A.mp4"
    assert states["cam_A"].first_frame == 0
    assert states["cam_A"].last_frame == 299
    assert states["cam_B"].last_frame == 199


def test_image_left_unset(capture_db) -> None:
    """CamCalibState.image is left None -- scrubbing populates it later."""
    states = _load_states_from_capture(capture_db, "shot1")
    assert all(s.image is None for s in states)


def test_falls_back_to_latest_calibration_for_mode(capture_db) -> None:
    """Neither a per-video override nor a mode default is set: use the newest."""
    states = {s.label: s for s in _load_states_from_capture(capture_db, "shot1")}
    # calib_new (2026-01-01) is newer than calib_old (2025-01-01) and is fisheye.
    assert states["cam_A"].fisheye is True
    assert states["cam_A"].calib_id == "calib_new"


def test_per_video_override_wins(capture_db) -> None:
    capture_db.execute(
        "UPDATE capture_videos SET intrinsics_calibration_id = 'calib_old' WHERE id = 'sv1'"
    )
    capture_db.commit()
    states = {s.label: s for s in _load_states_from_capture(capture_db, "shot1")}
    assert states["cam_A"].calib_id == "calib_old"
    assert states["cam_A"].fisheye is False
    # cam_B is untouched and still falls back to the latest.
    assert states["cam_B"].calib_id == "calib_new"


def test_mode_default_used_when_no_override(capture_db) -> None:
    capture_db.execute(
        "UPDATE camera_modes SET default_intrinsics_calibration_id = 'calib_old' WHERE id = 'mode1'"
    )
    capture_db.commit()
    states = {s.label: s for s in _load_states_from_capture(capture_db, "shot1")}
    assert states["cam_A"].calib_id == "calib_old"
    assert states["cam_B"].calib_id == "calib_old"


def test_per_video_override_beats_mode_default(capture_db) -> None:
    capture_db.execute(
        "UPDATE camera_modes SET default_intrinsics_calibration_id = 'calib_old' WHERE id = 'mode1'"
    )
    capture_db.execute(
        "UPDATE capture_videos SET intrinsics_calibration_id = 'calib_new' WHERE id = 'sv1'"
    )
    capture_db.commit()
    states = {s.label: s for s in _load_states_from_capture(capture_db, "shot1")}
    assert states["cam_A"].calib_id == "calib_new"
    assert states["cam_B"].calib_id == "calib_old"


def test_camera_without_any_calibration_is_skipped(capture_db) -> None:
    capture_db.execute("INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('ci3', 'cm1', 'cam_C')")
    capture_db.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,"
        " first_video_frame, last_video_frame, actual_fps)"
        " VALUES ('sv3', 'shot1', 'ci3', '/videos/cam_C.mp4', 0, 99, 30.0)"
    )
    capture_db.commit()
    states = {s.label: s for s in _load_states_from_capture(capture_db, "shot1")}
    assert "cam_C" not in states
    assert {"cam_A", "cam_B"} <= states.keys()


def test_intrinsics_matrix_decoded_correctly(capture_db) -> None:
    states = {s.label: s for s in _load_states_from_capture(capture_db, "shot1")}
    s = states["cam_A"]
    assert s.K[0, 0] == pytest.approx(1010.0)
    assert s.K[1, 1] == pytest.approx(1010.0)
    assert s.K[0, 2] == pytest.approx(962.0)
    assert s.K[1, 2] == pytest.approx(542.0)
    assert s.dist.shape == (1, 4)


def test_no_cameras_for_unknown_shot_returns_empty(capture_db) -> None:
    assert _load_states_from_capture(capture_db, "no-such-shot") == []
