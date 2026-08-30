# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for RunDetectionDialog's object/marker-detection mode
(marker-based-mocap design doc §7.1 sub-phase 1c).

Mirrors test_run_detection_dialog.py's approach for the segmentation bbox
source: the actual MarkerDetectionPipeline execution needs a real video
file, so these tests stop short of driving the background thread --
_run_marker_detection's job construction is exercised with
MarkerDetectionJob.start() monkeypatched to a no-op.
"""
from __future__ import annotations

import pytest


@pytest.fixture()
def capture_db_with_object(tmp_path):
    """Session DB with a capture, one synced camera, and one tracked object."""
    from posetrak.db.db import create_session
    from posetrak.db.manage_capture_object import create_capture_object
    from posetrak.db.manage_marker_body import import_marker_body_str

    conn = create_session(tmp_path / "run_detection_obj_test.db")
    conn.execute("PRAGMA foreign_keys = OFF")

    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES ('cap1', 'sess1', 1)"
    )
    conn.execute(
        "INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('ci1', 'cm1', 'cam_A')"
    )
    conn.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,"
        " first_video_frame, last_video_frame, actual_fps)"
        " VALUES ('sv1', 'cap1', 'ci1', '/dev/null', 0, 1000, 30.0)"
    )
    conn.execute(
        "INSERT INTO sync_configs (id, shot_id, created_by) VALUES ('sync1', 'cap1', 'x')"
    )
    conn.execute(
        "INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id,"
        " video_frame, timestamp_s) VALUES ('sync1', 'ci1', 'sv1', 0, 0.0)"
    )
    conn.commit()

    body_id = import_marker_body_str(
        conn,
        "name: test-bokken\nunits: meters\nmarkers:\n"
        "  - name: hilt\n    type: aruco\n    dictionary: DICT_4X4_50\n"
        "    id: \"3\"\n    size: 0.05\n    center: [0.0, 0.0, 0.0]\n"
        "    normal: [0.0, 0.0, 1.0]\n    up: [0.0, 1.0, 0.0]\n",
        name="Test Bokken",
    )
    object_id = create_capture_object(conn, "cap1", "bokken-A", body_id)
    conn.commit()

    yield conn, object_id
    conn.close()


def _make_dialog(capture_db, tmp_path):
    from app.pose.run_detection_dialog import RunDetectionDialog

    return RunDetectionDialog(
        conn=capture_db,
        session_path=tmp_path / "run_detection_obj_test.db",
        capture_id="cap1",
        time_start_s=0.0,
        time_end_s=10.0,
    )


def test_no_object_combo_when_no_objects_exist(qapp, tmp_path):
    from posetrak.db.db import create_session
    from app.pose.run_detection_dialog import RunDetectionDialog

    conn = create_session(tmp_path / "empty.db")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')")
    conn.execute("INSERT INTO captures (id, session_id, capture_number) VALUES ('cap1', 'sess1', 1)")
    conn.commit()

    dlg = RunDetectionDialog(conn=conn, session_path=tmp_path / "empty.db", capture_id="cap1")
    assert dlg._object_combo is None
    assert dlg._marker_perimeter_spin is None
    assert dlg._marker_frame_step_spin is None


def test_object_combo_lists_person_and_objects(qapp, capture_db_with_object, tmp_path):
    conn, object_id = capture_db_with_object
    dlg = _make_dialog(conn, tmp_path)

    assert dlg._object_combo is not None
    labels = [dlg._object_combo.itemText(i) for i in range(dlg._object_combo.count())]
    assert labels == ["Person (pose)", "bokken-A"]
    assert dlg._object_combo.itemData(0) is None
    assert dlg._object_combo.itemData(1) == object_id


def test_marker_fields_start_disabled(qapp, capture_db_with_object, tmp_path):
    conn, _object_id = capture_db_with_object
    dlg = _make_dialog(conn, tmp_path)
    assert not dlg._marker_perimeter_spin.isEnabled()
    assert not dlg._marker_frame_step_spin.isEnabled()


def test_choosing_object_toggles_field_enablement(qapp, capture_db_with_object, tmp_path):
    conn, object_id = capture_db_with_object
    dlg = _make_dialog(conn, tmp_path)

    idx = dlg._object_combo.findData(object_id)
    dlg._object_combo.setCurrentIndex(idx)

    assert dlg._marker_perimeter_spin.isEnabled()
    assert dlg._marker_frame_step_spin.isEnabled()
    assert not dlg._detector_combo.isEnabled()
    assert not dlg._pose_combo.isEnabled()
    assert not dlg._conf_spin.isEnabled()
    assert not dlg._refine_hands_check.isEnabled()

    dlg._object_combo.setCurrentIndex(0)  # back to "Person (pose)"
    assert not dlg._marker_perimeter_spin.isEnabled()
    assert not dlg._marker_frame_step_spin.isEnabled()
    assert dlg._detector_combo.isEnabled()
    assert dlg._pose_combo.isEnabled()
    assert dlg._conf_spin.isEnabled()
    assert dlg._refine_hands_check.isEnabled()


def test_run_marker_detection_builds_job_with_chosen_settings(
    qapp, capture_db_with_object, tmp_path, monkeypatch
) -> None:
    from app.pose.main import MarkerDetectionJob

    monkeypatch.setattr(MarkerDetectionJob, "start", lambda self: None)

    conn, object_id = capture_db_with_object
    dlg = _make_dialog(conn, tmp_path)
    idx = dlg._object_combo.findData(object_id)
    dlg._object_combo.setCurrentIndex(idx)
    dlg._marker_perimeter_spin.setValue(0.02)
    dlg._marker_frame_step_spin.setValue(3)

    dlg._run_marker_detection(object_id, "sync1", 0.0, 10.0)

    assert isinstance(dlg._job, MarkerDetectionJob)
    assert dlg._job._capture_object_id == object_id
    assert dlg._job._sync_config_id == "sync1"
    assert dlg._job._time_start_s == 0.0
    assert dlg._job._time_end_s == 10.0
    assert dlg._job._min_marker_perimeter_rate == pytest.approx(0.02)
    assert dlg._job._frame_step == 3


def test_on_run_dispatches_to_marker_detection_when_object_selected(
    qapp, capture_db_with_object, tmp_path, monkeypatch
) -> None:
    from app.pose.main import MarkerDetectionJob, DetectionJob

    monkeypatch.setattr(MarkerDetectionJob, "start", lambda self: None)
    monkeypatch.setattr(DetectionJob, "start", lambda self: pytest.fail("should not run DetectionJob"))

    conn, object_id = capture_db_with_object
    dlg = _make_dialog(conn, tmp_path)
    idx = dlg._object_combo.findData(object_id)
    dlg._object_combo.setCurrentIndex(idx)

    dlg._on_run()

    assert isinstance(dlg._job, MarkerDetectionJob)


def test_on_run_with_person_selected_uses_detection_job(
    qapp, capture_db_with_object, tmp_path, monkeypatch
) -> None:
    from app.pose.main import DetectionJob

    monkeypatch.setattr(DetectionJob, "start", lambda self: None)

    conn, _object_id = capture_db_with_object
    dlg = _make_dialog(conn, tmp_path)
    # _object_combo defaults to index 0, "Person (pose)" -- no selection change needed.

    dlg._on_run()

    assert isinstance(dlg._job, DetectionJob)
