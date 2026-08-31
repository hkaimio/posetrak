# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ObjectRunTrackerDialog (marker-based-mocap design doc §7.1
sub-phase 1f) -- the GUI entry point for running the tracker against a
finalised marker-mocap object sequence, previously reachable only via the
CLI directly.

Pure widget-construction and path-building assertions -- actually launching
the tracker subprocess follows this file's existing convention (see
test_run_tracker.py's own docstring) of manual validation rather than a
unit test.
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
from posetrak.db.manage_skeleton import import_skeleton_str

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

_PROP_SKELETON_YAML = """\
name: test-prop-skeleton
units: meters
joints:
  - name: prop_root
    type: root
    parent: null
    offset: [0.0, 0.0, 0.0]
input_tracks:
  - id: hilt
    type: labeled_points
markers:
  - name: hilt:c0
    parent: prop_root
    offset: [0.0, 0.0, 0.0]
    track: hilt
    landmark: "hilt:c0"
"""


@pytest.fixture
def object_sequence(tmp_path):
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
        conn, shot_id=_SHOT_ID, sync_config_id=_SYNC_ID, time_start_s=5.0, time_end_s=9.0,
        dictionary="DICT_4X4_50", marker_ids=["3"],
        marker_body_definition_id=body_id, capture_object_id=object_id,
    )
    writer = MarkerKeypointWriter(conn, run_id, _SVID, marker_ids=["3"])
    writer.add_frame(0, [])
    writer.finalise()
    kp = np.zeros((4, 3), dtype=np.float32)
    kp[:, 2] = 1.0
    conn.execute(
        "UPDATE detection_keypoints SET keypoints=? "
        "WHERE detection_run_id=? AND shot_video_id=? AND video_frame=0",
        (kp.tobytes(), run_id, _SVID),
    )
    conn.commit()

    seq_id = finalise_object_to_db(conn, run_id)
    yield conn, seq_id
    conn.close()


def test_no_skeletons_disables_run_button(qapp, object_sequence, tmp_path):
    from app.pose.run_tracker import ObjectRunTrackerDialog

    conn, seq_id = object_sequence
    dlg = ObjectRunTrackerDialog(conn, str(tmp_path / "test.db"), seq_id)

    assert dlg._skeleton_combo.count() == 0
    assert not dlg._run_btn.isEnabled()


def test_skeleton_combo_lists_imported_skeleton_and_enables_run(qapp, object_sequence, tmp_path):
    from app.pose.run_tracker import ObjectRunTrackerDialog

    conn, seq_id = object_sequence
    skel_id = import_skeleton_str(conn, _PROP_SKELETON_YAML, name="test-prop")
    conn.commit()

    dlg = ObjectRunTrackerDialog(conn, str(tmp_path / "test.db"), seq_id)

    assert dlg._skeleton_combo.count() == 1
    assert dlg._skeleton_combo.itemData(0) == skel_id
    assert dlg._run_btn.isEnabled()


def test_time_range_defaults_to_sequence_range(qapp, object_sequence, tmp_path):
    from app.pose.run_tracker import ObjectRunTrackerDialog

    conn, seq_id = object_sequence
    dlg = ObjectRunTrackerDialog(conn, str(tmp_path / "test.db"), seq_id)

    assert dlg._start_spin.value() == pytest.approx(5.0)
    assert dlg._end_spin.value() == pytest.approx(9.0)


def test_skeleton_selection_pushes_skeleton_id_to_config_widget(qapp, object_sequence, tmp_path):
    from app.pose.run_tracker import ObjectRunTrackerDialog

    conn, seq_id = object_sequence
    skel_id = import_skeleton_str(conn, _PROP_SKELETON_YAML, name="test-prop")
    conn.commit()

    dlg = ObjectRunTrackerDialog(conn, str(tmp_path / "test.db"), seq_id)
    assert dlg._config_widget._skeleton_ids == [skel_id]


def test_resolve_out_dir_uses_shot_label_and_skeleton_name(qapp, object_sequence, tmp_path):
    from app.pose.run_tracker import ObjectRunTrackerDialog

    conn, seq_id = object_sequence
    skel_id = import_skeleton_str(conn, _PROP_SKELETON_YAML, name="test-prop")
    conn.commit()

    db_path = tmp_path / "test.db"
    dlg = ObjectRunTrackerDialog(conn, str(db_path), seq_id)

    out_dir = dlg._resolve_out_dir(skel_id)
    assert out_dir == db_path.parent / "posetrak_results" / "test" / "test-prop" / "tracking"


def test_resolve_out_dir_honours_explicit_override(qapp, object_sequence, tmp_path):
    from app.pose.run_tracker import ObjectRunTrackerDialog

    conn, seq_id = object_sequence
    skel_id = import_skeleton_str(conn, _PROP_SKELETON_YAML, name="test-prop")
    conn.commit()

    dlg = ObjectRunTrackerDialog(conn, str(tmp_path / "test.db"), seq_id)
    dlg._out_dir_edit.setText(str(tmp_path / "custom_out"))
    assert dlg._resolve_out_dir(skel_id) == tmp_path / "custom_out"
