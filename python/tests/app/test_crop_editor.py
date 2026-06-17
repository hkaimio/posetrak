"""Tests for PersonCropGridWidget data loading logic."""
from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from posetrak.db.db import create_session
from app.pose.db_cache import write_observation_edit


N_KP = 3  # small keypoint count for tests


def _encode_kp(x: float = 100.0, y: float = 200.0, conf: float = 0.9) -> bytes:
    kp = np.full((N_KP, 3), [x, y, conf], dtype=np.float32)
    return kp.tobytes()


def _make_mask(*indices: int) -> bytes:
    n_bytes = math.ceil(N_KP / 8)
    mask = bytearray(n_bytes)
    for i in indices:
        mask[i // 8] |= 1 << (i % 8)
    return bytes(mask)


def _make_jpeg_stub() -> bytes:
    """Tiny valid JPEG stub (1×1 white pixel)."""
    # Minimal valid JPEG bytes for testing (loadFromData will succeed)
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
        b"\xc3\xb1a\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf5\xff\xd9"
    )


@pytest.fixture()
def seq_db(tmp_path):
    """Session DB with a minimal but fully-wired sequence."""
    conn = create_session(tmp_path / "crop_test.db")
    conn.execute("PRAGMA foreign_keys = OFF")

    # Parent rows
    conn.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01T00:00:00Z')")
    conn.execute("INSERT INTO captures (id, session_id, capture_number) VALUES ('shot1', 'sess1', 1)")

    # Camera instance (required for capture_videos join)
    conn.execute("INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('ci1', 'cm1', 'cam_A')")
    conn.execute("INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('ci2', 'cm1', 'cam_B')")

    # capture_videos
    conn.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path, first_video_frame, last_video_frame, actual_fps)"
        " VALUES ('sv1', 'shot1', 'ci1', '/dev/null', 0, 100, 30.0)"
    )
    conn.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path, first_video_frame, last_video_frame, actual_fps)"
        " VALUES ('sv2', 'shot1', 'ci2', '/dev/null', 0, 100, 30.0)"
    )

    # Detection run + assignments
    conn.execute("INSERT INTO detection_runs (id, shot_id, sync_config_id, time_start_s, time_end_s,"
                 " detector_model, pose_model, status, created_at)"
                 " VALUES ('run1', 'shot1', 'sync1', 0.0, 2.0, 'yolo', 'rtmpose', 'complete', '2026-01-01')")
    conn.execute(
        "INSERT INTO detection_track_assignments"
        " (detection_run_id, shot_video_id, track_id, person_name, first_frame, last_frame)"
        " VALUES ('run1', 'sv1', 42, 'Alice', 0, 100)"
    )
    conn.execute(
        "INSERT INTO detection_track_assignments"
        " (detection_run_id, shot_video_id, track_id, person_name, first_frame, last_frame)"
        " VALUES ('run1', 'sv2', 7, 'Alice', 0, 100)"
    )

    # Sequence
    conn.execute(
        "INSERT INTO pose_observation_sequences"
        " (id, shot_id, sync_config_id, time_start_s, time_end_s, detection_run_id, pixels_are_undistorted)"
        " VALUES ('seq1', 'shot1', 'sync1', 0.0, 2.0, 'run1', 0)"
    )
    conn.execute("INSERT INTO sequence_persons (sequence_id, person_id, person_name)"
                 " VALUES ('seq1', 0, 'Alice')")

    # pose_observations — two frames, two cameras each.
    # Both cameras share the same synchronized timestamp_s at each frame.
    for frame, ts in [(10, 0.083), (11, 0.167)]:
        conn.execute(
            "INSERT INTO pose_observations"
            " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, kp_blob)"
            " VALUES ('seq1', 'ci1', ?, ?, 0, ?)",
            (frame, ts, _encode_kp(x=float(frame * 10))),
        )
        conn.execute(
            "INSERT INTO pose_observations"
            " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, kp_blob)"
            " VALUES ('seq1', 'ci2', ?, ?, 0, ?)",
            (frame, ts, _encode_kp(x=float(frame * 10 + 5))),
        )

    # frame_cache_entries for sv1 at frame 10
    jpeg = _make_jpeg_stub()
    conn.execute(
        "INSERT INTO frame_cache_entries"
        " (shot_video_id, frame_idx, cache_type, track_id, region_type,"
        "  width_px, height_px, image_data, detection_run_id,"
        "  src_x, src_y, src_w, src_h)"
        " VALUES ('sv1', 10, 'person_crop', 42, 'full_body',"
        "         320, 240, ?, 'run1', 50, 30, 200, 150)",
        (jpeg,),
    )

    conn.commit()
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Tests (data-layer only, no Qt rendering)
# ---------------------------------------------------------------------------

def test_load_sequence_builds_frame_list(qapp, seq_db):
    from app.pose.crop_editor import PersonCropGridWidget
    w = PersonCropGridWidget()
    w.load_sequence(seq_db, "seq1")

    assert len(w._frames) == 2
    assert len(w._cameras) == 2
    assert w._cameras[0].label in ("cam_A", "cam_B")


def test_frame_slots_have_per_camera_video_frames(qapp, seq_db):
    from app.pose.crop_editor import PersonCropGridWidget
    w = PersonCropGridWidget()
    w.load_sequence(seq_db, "seq1")

    # Both frames should have entries for both cameras
    for fs in w._frames:
        assert "ci1" in fs.per_cam
        assert "ci2" in fs.per_cam


def test_track_id_lookup(qapp, seq_db):
    from app.pose.crop_editor import PersonCropGridWidget, _CameraSlot
    w = PersonCropGridWidget()
    w.load_sequence(seq_db, "seq1")

    cam_sv1 = next(c for c in w._cameras if c.shot_video_id == "sv1")
    assert w._track_id_for_frame(cam_sv1, 10) == 42
    assert w._track_id_for_frame(cam_sv1, 100) == 42
    assert w._track_id_for_frame(cam_sv1, 200) is None  # out of range


def test_load_crop_returns_jpeg(qapp, seq_db):
    from app.pose.crop_editor import PersonCropGridWidget
    w = PersonCropGridWidget()
    w.load_sequence(seq_db, "seq1")

    jpeg, src_x, src_y, src_w, src_h = w._load_crop("sv1", 10, 42)
    assert jpeg is not None and len(jpeg) > 0
    assert src_x == 50
    assert src_y == 30
    assert src_w == 200
    assert src_h == 150


def test_load_crop_returns_none_for_missing(qapp, seq_db):
    from app.pose.crop_editor import PersonCropGridWidget
    w = PersonCropGridWidget()
    w.load_sequence(seq_db, "seq1")

    jpeg, *_ = w._load_crop("sv1", 99, 42)  # frame 99 has no crop
    assert jpeg is None


def test_kp_by_frame_populated(qapp, seq_db):
    from app.pose.crop_editor import PersonCropGridWidget
    w = PersonCropGridWidget()
    w.load_sequence(seq_db, "seq1")

    cam_ci1 = next(c for c in w._cameras if c.camera_instance_id == "ci1")
    assert 10 in cam_ci1.kp_by_frame
    assert cam_ci1.kp_by_frame[10].shape == (N_KP, 3)


def test_edited_mask_detected(qapp, seq_db):
    from app.pose.crop_editor import PersonCropGridWidget

    # Write an edit for ci1 frame 10 before loading
    edit_kp = np.full((N_KP, 3), [55.0, 65.0, 0.0], dtype=np.float32)
    write_observation_edit(seq_db, "seq1", "ci1", 10, edit_kp, _make_mask(1))

    w = PersonCropGridWidget()
    w.load_sequence(seq_db, "seq1")

    mask = w._edited_mask("ci1", 10)
    assert mask is not None
    assert bool(mask[1]) is True
    assert bool(mask[0]) is False
