# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ObjectPanel / ObjectCropGridWidget (marker-based-mocap design
doc §7.1 sub-phase 1e).

Builds a real finalised object sequence via the actual pipeline helpers
(create_marker_detection_run -> MarkerKeypointWriter -> finalise_object_to_db)
rather than a hand-rolled fixture, so these tests exercise the same data
shape the real GUI flow produces. Frame decode itself (FrameReader's
background QThread) isn't driven with a real video file -- _on_frame_ready
is called directly with a synthetic frame instead, mirroring how
test_run_detection_dialog*.py stops short of real video decode.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from posetrak.db.db import create_session, generate_id
from app.pose.db_cache import create_marker_detection_run, MarkerKeypointWriter
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
"""


@pytest.fixture
def session_with_object_sequence(tmp_path):
    db_path = tmp_path / "test.db"
    conn = create_session(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF")
    session_id = generate_id()
    conn.executescript(f"""
        INSERT INTO mocap_sessions (id, recorded_at) VALUES ('{session_id}', '2026-01-01');
        INSERT INTO captures (id, session_id, capture_number, label)
            VALUES ('{_SHOT_ID}', '{session_id}', 1, 'test');
        INSERT INTO camera_instances (id, camera_model_id, label)
            VALUES ('{_CAM_ID}', 'cm1', 'cam_A');
        INSERT INTO sync_configs (id, shot_id, created_by)
            VALUES ('{_SYNC_ID}', '{_SHOT_ID}', 'test');
        INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,
                                 first_video_frame, last_video_frame, actual_fps)
            VALUES ('{_SVID}', '{_SHOT_ID}', '{_CAM_ID}', '/dev/null', 0, 1000, 30.0);
        INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id,
                                 video_frame, timestamp_s)
            VALUES ('{_SYNC_ID}', '{_CAM_ID}', '{_SVID}', 0, 0.0);
    """)
    conn.commit()

    body_id = import_marker_body_str(conn, _MARKER_BODY_YAML, name="Test Bokken")
    object_id = create_capture_object(conn, _SHOT_ID, "bokken-A", body_id)

    run_id = create_marker_detection_run(
        conn, shot_id=_SHOT_ID, sync_config_id=_SYNC_ID, time_start_s=0.0, time_end_s=1.0,
        dictionary="DICT_4X4_50", marker_ids=["3"],
        marker_body_definition_id=body_id, capture_object_id=object_id,
    )
    writer = MarkerKeypointWriter(conn, run_id, _SVID, marker_ids=["3"])
    for frame in (0, 15):
        writer.add_frame(frame, [])
    writer.finalise()
    conn.commit()
    # Overwrite the NaN placeholders with a deterministic known-good blob.
    for frame, x in ((0, 100.0), (15, 150.0)):
        kp = np.zeros((4, 3), dtype=np.float32)
        kp[:, 0] = x
        kp[:, 1] = 200.0
        kp[:, 2] = 1.0
        conn.execute(
            "UPDATE detection_keypoints SET keypoints=? "
            "WHERE detection_run_id=? AND shot_video_id=? AND video_frame=?",
            (kp.tobytes(), run_id, _SVID, frame),
        )
    conn.commit()

    seq_id = finalise_object_to_db(conn, run_id)
    yield conn, seq_id, object_id
    conn.close()


def test_object_panel_shows_object_and_body_name(qapp, session_with_object_sequence, tmp_path):
    from app.ui.content_panels import ObjectPanel

    conn, seq_id, _object_id = session_with_object_sequence
    panel = ObjectPanel(conn, seq_id, tmp_path / "test.db")
    assert panel._crop_grid is not None
    panel.shutdown()


def test_object_panel_shows_no_tracking_runs_initially(qapp, session_with_object_sequence, tmp_path):
    from app.ui.content_panels import ObjectPanel

    conn, seq_id, _object_id = session_with_object_sequence
    panel = ObjectPanel(conn, seq_id, tmp_path / "test.db")
    assert panel._run_box._btn.text() == "Tracking runs (0)"
    assert panel._run_list.count() == 1
    assert panel._run_list.item(0).text() == "No tracking runs yet."
    panel.shutdown()


def test_object_panel_lists_existing_tracking_run(qapp, session_with_object_sequence, tmp_path):
    from app.ui.content_panels import ObjectPanel
    from posetrak.db.manage_skeleton import import_skeleton_str

    conn, seq_id, _object_id = session_with_object_sequence
    skel_id = import_skeleton_str(
        conn,
        "name: test-prop\nunits: meters\njoints:\n"
        "  - name: prop_root\n    type: root\n    parent: null\n"
        "    offset: [0.0, 0.0, 0.0]\n",
        name="test-prop",
    )
    conn.execute(
        "INSERT INTO tracking_runs "
        "(id, observation_sequence_id, tracker_config_id, skeleton_id, "
        " extrinsic_calibration_id, sync_config_id, ran_at, posetrak_version, "
        " active_camera_ids, marker_names) "
        "VALUES ('run1', ?, 'cfg1', ?, 'calib1', 'sync1', '2026-01-01T00:00:00Z', "
        "        'test', '[]', '[]')",
        (seq_id, skel_id),
    )
    conn.commit()

    panel = ObjectPanel(conn, seq_id, tmp_path / "test.db")
    assert panel._run_box._btn.text() == "Tracking runs (1)"
    assert "test-prop" in panel._run_list.item(0).text()
    panel.shutdown()


def test_object_crop_grid_creates_one_cell_per_camera(qapp, session_with_object_sequence, tmp_path):
    from app.ui.content_panels import ObjectCropGridWidget

    conn, seq_id, _object_id = session_with_object_sequence
    grid = ObjectCropGridWidget(conn, seq_id)
    assert len(grid._cells) == 1
    assert len(grid._cameras) == 1
    assert grid._cameras[0]["camera_instance_id"] == _CAM_ID
    grid.shutdown()


def test_object_crop_grid_no_observations_shows_message(qapp, tmp_path):
    from posetrak.db.db import create_session
    from app.ui.content_panels import ObjectCropGridWidget

    conn = create_session(tmp_path / "empty.db")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')")
    conn.execute("INSERT INTO captures (id, session_id, capture_number) VALUES ('cap1', 'sess1', 1)")
    conn.execute("INSERT INTO sync_configs (id, shot_id, created_by) VALUES ('sync1', 'cap1', 'x')")
    conn.execute(
        "INSERT INTO pose_observation_sequences (id, shot_id, sync_config_id, time_start_s, time_end_s) "
        "VALUES ('seq1', 'cap1', 'sync1', 0.0, 1.0)"
    )
    conn.commit()

    grid = ObjectCropGridWidget(conn, "seq1")
    assert grid._cells == []
    grid.shutdown()


def test_object_crop_grid_loads_overlay_for_current_frame(qapp, session_with_object_sequence, tmp_path):
    from app.ui.content_panels import ObjectCropGridWidget

    conn, seq_id, _object_id = session_with_object_sequence
    grid = ObjectCropGridWidget(conn, seq_id)
    # Slider starts at 0 -> t = time_start_s = 0.0 -> frame 0.
    assert grid._current_frame_by_cam[_CAM_ID] == 0
    kp = grid._obs_kp[_CAM_ID][0]
    assert kp.shape == (4, 3)
    assert np.allclose(kp[:, 0], 100.0)
    grid.shutdown()


def test_object_crop_grid_edit_toggle_propagates_to_cells(qapp, session_with_object_sequence, tmp_path):
    from app.ui.content_panels import ObjectCropGridWidget

    conn, seq_id, _object_id = session_with_object_sequence
    grid = ObjectCropGridWidget(conn, seq_id)
    assert grid._cells[0]._canvas._edit_mode is False
    grid._edit_check.setChecked(True)
    assert grid._cells[0]._canvas._edit_mode is True
    grid._edit_check.setChecked(False)
    assert grid._cells[0]._canvas._edit_mode is False
    grid.shutdown()


def test_object_crop_grid_kp_moved_writes_edit(qapp, session_with_object_sequence, tmp_path):
    from app.ui.content_panels import ObjectCropGridWidget
    from app.pose.db_cache import read_observations_with_edits

    conn, seq_id, _object_id = session_with_object_sequence
    grid = ObjectCropGridWidget(conn, seq_id)

    grid._on_kp_moved(_CAM_ID, 0, 999.0, 888.0)

    # The edit is persisted (survives a fresh read, not just the in-memory cache).
    fresh = read_observations_with_edits(conn, seq_id, _CAM_ID, primary_source="markers")
    assert fresh[0][0, 0] == pytest.approx(999.0)
    assert fresh[0][0, 1] == pytest.approx(888.0)
    # Other corners on the same frame are untouched.
    assert fresh[0][1, 0] == pytest.approx(100.0)
    # In-memory cache was refreshed too.
    assert grid._obs_kp[_CAM_ID][0][0, 0] == pytest.approx(999.0)
    grid.shutdown()


def test_object_crop_grid_frame_ready_sets_pixmap(qapp, session_with_object_sequence, tmp_path):
    from app.ui.content_panels import ObjectCropGridWidget

    conn, seq_id, _object_id = session_with_object_sequence
    grid = ObjectCropGridWidget(conn, seq_id)
    fake_frame = np.zeros((100, 200, 3), dtype=np.uint8)

    grid._on_frame_ready(_CAM_ID, 0, fake_frame)

    assert grid._cells[0]._canvas._pixmap is not None
    grid.shutdown()


def test_object_crop_grid_frame_ready_ignores_stale_frame(qapp, session_with_object_sequence, tmp_path):
    """A slow background decode for a frame the slider has since moved away
    from must not clobber the current display."""
    from app.ui.content_panels import ObjectCropGridWidget

    conn, seq_id, _object_id = session_with_object_sequence
    grid = ObjectCropGridWidget(conn, seq_id)
    assert grid._current_frame_by_cam[_CAM_ID] == 0

    grid._on_frame_ready(_CAM_ID, 999, np.zeros((10, 10, 3), dtype=np.uint8))
    assert grid._cells[0]._canvas._pixmap is None  # stale result discarded
    grid.shutdown()
