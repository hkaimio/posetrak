# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the app test suite."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def qt_offscreen():
    """Force Qt to use the offscreen platform so tests run without a display."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp(qt_offscreen):
    """Session-scoped QApplication; re-uses an existing instance if present."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _make_jpeg_stub() -> bytes:
    """Tiny valid JPEG stub (1×1 white pixel)."""
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


_SEQ_DB_N_KP = 3  # keypoint count used by seq_db


@pytest.fixture()
def seq_db(tmp_path):
    """Session DB with a minimal but fully-wired sequence (2 frames, 2 cameras)."""
    from posetrak.db.db import create_session

    conn = create_session(tmp_path / "crop_test.db")
    conn.execute("PRAGMA foreign_keys = OFF")

    conn.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01T00:00:00Z')")
    conn.execute("INSERT INTO captures (id, session_id, capture_number) VALUES ('shot1', 'sess1', 1)")

    conn.execute("INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('ci1', 'cm1', 'cam_A')")
    conn.execute("INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('ci2', 'cm1', 'cam_B')")

    conn.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,"
        " first_video_frame, last_video_frame, actual_fps)"
        " VALUES ('sv1', 'shot1', 'ci1', '/dev/null', 0, 100, 30.0)"
    )
    conn.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,"
        " first_video_frame, last_video_frame, actual_fps)"
        " VALUES ('sv2', 'shot1', 'ci2', '/dev/null', 0, 100, 30.0)"
    )

    conn.execute(
        "INSERT INTO detection_runs (id, shot_id, sync_config_id, time_start_s, time_end_s,"
        " detector_model, pose_model, status, created_at)"
        " VALUES ('run1', 'shot1', 'sync1', 0.0, 2.0, 'yolo', 'rtmpose', 'complete', '2026-01-01')"
    )
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

    conn.execute(
        "INSERT INTO pose_observation_sequences"
        " (id, shot_id, sync_config_id, time_start_s, time_end_s, detection_run_id, pixels_are_undistorted)"
        " VALUES ('seq1', 'shot1', 'sync1', 0.0, 2.0, 'run1', 0)"
    )
    conn.execute("INSERT INTO sequence_persons (sequence_id, person_id, person_name)"
                 " VALUES ('seq1', 0, 'Alice')")

    def _enc(x: float) -> bytes:
        kp = np.full((_SEQ_DB_N_KP, 3), [x, x * 2, 0.9], dtype=np.float32)
        return kp.tobytes()

    for frame, ts in [(10, 0.083), (11, 0.167)]:
        conn.execute(
            "INSERT INTO pose_observations"
            " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, kp_blob)"
            " VALUES ('seq1', 'ci1', ?, ?, 0, ?)",
            (frame, ts, _enc(float(frame * 10))),
        )
        conn.execute(
            "INSERT INTO pose_observations"
            " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, kp_blob)"
            " VALUES ('seq1', 'ci2', ?, ?, 0, ?)",
            (frame, ts, _enc(float(frame * 10 + 5))),
        )

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
