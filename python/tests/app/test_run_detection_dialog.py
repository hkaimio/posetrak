# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for RunDetectionDialog's segmentation bbox source (segmentation-
reuse gap 2: docs/roadmap/features/segmentation-reuse/
segmentation-reuse-design.md). This file had zero prior test coverage.

The detection job needs a real video file and models, so these tests stop short of driving a real
background thread: the dialog's job construction is exercised with DetectionJob.start() monkeypatched
to a no-op, and the job's run() against a fake pipeline.
"""
from __future__ import annotations

import pytest


@pytest.fixture()
def capture_db(tmp_path):
    """Session DB with a capture, two synced cameras, a capture-level
    person, and one existing segmentation covering the capture."""
    from posetrak.db.db import create_session

    conn = create_session(tmp_path / "run_detection_test.db")
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
        "INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('ci2', 'cm1', 'cam_B')"
    )
    conn.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,"
        " first_video_frame, last_video_frame, actual_fps)"
        " VALUES ('sv1', 'cap1', 'ci1', '/dev/null', 0, 1000, 30.0)"
    )
    conn.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,"
        " first_video_frame, last_video_frame, actual_fps)"
        " VALUES ('sv2', 'cap1', 'ci2', '/dev/null', 0, 1000, 30.0)"
    )
    conn.execute(
        "INSERT INTO sync_configs (id, shot_id, created_by) VALUES ('sync1', 'cap1', 'x')"
    )
    # Two sync points per camera (0s->frame 0, 10s->frame 300 @30fps) so
    # SyncTable.lookup has an interval to interpolate within.
    for svid, ci in (("sv1", "ci1"), ("sv2", "ci2")):
        conn.execute(
            "INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id,"
            " video_frame, timestamp_s) VALUES ('sync1', ?, ?, 0, 0.0)",
            (ci, svid),
        )
        conn.execute(
            "INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id,"
            " video_frame, timestamp_s) VALUES ('sync1', ?, ?, 300, 10.0)",
            (ci, svid),
        )
    conn.execute(
        "INSERT INTO capture_persons (id, capture_id, name, created_at) "
        "VALUES ('p1', 'cap1', 'Alice', '2026-01-01T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO seg_quality_runs "
        "(id, shot_id, time_start_s, time_end_s, created_at, persons_json) "
        "VALUES ('seg1', 'cap1', 0.0, 1e9, '2026-01-01T00:00:00Z', '[\"Alice\"]')"
    )
    conn.commit()
    yield conn
    conn.close()


def _make_dialog(qapp, capture_db, tmp_path, trial_id=None):
    from app.pose.run_detection_dialog import RunDetectionDialog

    return RunDetectionDialog(
        conn=capture_db,
        session_path=tmp_path / "run_detection_test.db",
        capture_id="cap1",
        time_start_s=0.0,
        time_end_s=10.0,
        trial_id=trial_id,
    )


def test_no_bbox_source_combo_when_no_segmentation_exists(qapp, tmp_path):
    """The common case (no segmentation ever created) looks exactly as
    before -- no new UI element at all."""
    from posetrak.db.db import create_session
    from app.pose.run_detection_dialog import RunDetectionDialog

    conn = create_session(tmp_path / "empty.db")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')")
    conn.execute("INSERT INTO captures (id, session_id, capture_number) VALUES ('cap1', 'sess1', 1)")
    conn.commit()

    dlg = RunDetectionDialog(conn=conn, session_path=tmp_path / "empty.db", capture_id="cap1")
    assert dlg._bbox_source_combo is None


def test_device_warning_hidden_initially(qapp, tmp_path):
    from posetrak.db.db import create_session
    from app.pose.run_detection_dialog import RunDetectionDialog

    conn = create_session(tmp_path / "empty.db")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')")
    conn.execute("INSERT INTO captures (id, session_id, capture_number) VALUES ('cap1', 'sess1', 1)")
    conn.commit()

    dlg = RunDetectionDialog(conn=conn, session_path=tmp_path / "empty.db", capture_id="cap1")
    assert dlg._device_warning_label.isHidden()
    assert dlg._device_warning_label.text() == ""


def test_device_notice_shows_warning_label(qapp, tmp_path):
    from posetrak.db.db import create_session
    from app.pose.run_detection_dialog import RunDetectionDialog

    conn = create_session(tmp_path / "empty.db")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')")
    conn.execute("INSERT INTO captures (id, session_id, capture_number) VALUES ('cap1', 'sess1', 1)")
    conn.commit()

    dlg = RunDetectionDialog(conn=conn, session_path=tmp_path / "empty.db", capture_id="cap1")
    dlg._on_device_notice("No GPU detected -- detection will run on CPU and will be slow.")
    assert not dlg._device_warning_label.isHidden()
    assert "No GPU detected" in dlg._device_warning_label.text()


def test_bbox_source_combo_lists_yolo_and_segmentation(qapp, capture_db, tmp_path):
    dlg = _make_dialog(qapp, capture_db, tmp_path)
    assert dlg._bbox_source_combo is not None
    labels = [dlg._bbox_source_combo.itemText(i) for i in range(dlg._bbox_source_combo.count())]
    assert labels[0] == "YOLO detection"
    assert any("Alice" in label for label in labels[1:])
    assert dlg._bbox_source_combo.itemData(1) == "seg1"


def test_choosing_segmentation_disables_yolo_only_fields(qapp, capture_db, tmp_path):
    # capture_db's segmentation covers the whole capture (time_end_s=1e9),
    # so it's the default bbox source here -- the YOLO-only fields start
    # disabled, not enabled.
    dlg = _make_dialog(qapp, capture_db, tmp_path)
    assert dlg._bbox_source_combo.currentIndex() == 1
    assert not dlg._detector_combo.isEnabled()
    assert not dlg._conf_spin.isEnabled()

    dlg._bbox_source_combo.setCurrentIndex(0)  # switch to YOLO
    assert dlg._detector_combo.isEnabled()
    assert dlg._conf_spin.isEnabled()

    dlg._bbox_source_combo.setCurrentIndex(1)  # back to the segmentation
    assert not dlg._detector_combo.isEnabled()
    assert not dlg._conf_spin.isEnabled()


def _make_dialog_with_seg_range(
    qapp, tmp_path, seg_start_s, seg_end_s, time_start_s=2.0, time_end_s=8.0
):
    """Minimal session with one segmentation run covering [seg_start_s,
    seg_end_s), then a RunDetectionDialog asking to detect [time_start_s,
    time_end_s)."""
    from posetrak.db.db import create_session, generate_id
    from app.pose.run_detection_dialog import RunDetectionDialog

    conn = create_session(tmp_path / "seg_range_test.db")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')")
    conn.execute("INSERT INTO captures (id, session_id, capture_number) VALUES ('cap1', 'sess1', 1)")
    conn.execute(
        "INSERT INTO seg_quality_runs "
        "(id, shot_id, time_start_s, time_end_s, created_at, persons_json) "
        "VALUES (?, 'cap1', ?, ?, '2026-01-01T00:00:00Z', '[\"Alice\"]')",
        (generate_id(), seg_start_s, seg_end_s),
    )
    conn.commit()

    return RunDetectionDialog(
        conn=conn,
        session_path=tmp_path / "seg_range_test.db",
        capture_id="cap1",
        time_start_s=time_start_s,
        time_end_s=time_end_s,
    )


def test_defaults_to_segmentation_covering_the_full_requested_range(qapp, tmp_path):
    dlg = _make_dialog_with_seg_range(qapp, tmp_path, seg_start_s=0.0, seg_end_s=1e9)
    assert dlg._bbox_source_combo.currentIndex() == 1
    assert dlg._bbox_source_combo.currentData() is not None


def test_stays_on_yolo_when_segmentation_only_covers_part_of_the_range(qapp, tmp_path):
    # Segmentation covers [0, 5) but detection is being run over [2, 8) --
    # frames 5-8 would have no bboxes at all if this were picked silently.
    dlg = _make_dialog_with_seg_range(qapp, tmp_path, seg_start_s=0.0, seg_end_s=5.0)
    assert dlg._bbox_source_combo.currentIndex() == 0
    assert dlg._bbox_source_combo.currentData() is None


def test_stays_on_yolo_when_requested_time_range_is_unknown(qapp, tmp_path):
    # No time_start_s/time_end_s given at all (e.g. a brand new trial
    # before Mark Start/End) -- nothing to check containment against.
    dlg = _make_dialog_with_seg_range(
        qapp, tmp_path, seg_start_s=0.0, seg_end_s=1e9, time_start_s=None, time_end_s=None
    )
    assert dlg._bbox_source_combo.currentIndex() == 0
    assert dlg._bbox_source_combo.currentData() is None


def test_segmentation_source_starts_the_detection_job_with_that_segmentation(qapp, capture_db, tmp_path, monkeypatch):
    """Choosing an existing segmentation as the bbox source runs the same detection job as YOLO boxes, with the
    segmentation as its box source and the pose model the user has the weights for."""
    from app.pose.main import DetectionJob

    started = []
    monkeypatch.setattr(DetectionJob, "start", lambda self: started.append(self))
    monkeypatch.setattr("app.pose.pose_model_gui.prepare_pose_model", lambda parent, requested: (requested, None))

    dlg = _make_dialog(qapp, capture_db, tmp_path)
    assert dlg._bbox_source_combo.itemData(1) == "seg1"
    dlg._bbox_source_combo.setCurrentIndex(1)
    assert dlg._pose_combo.currentText() == "soma77-vitpose-h"          # the recommended model is preselected
    dlg._on_run()

    (job,) = started
    assert job._seg_run_id == "seg1" and job._pose_model_name == "soma77-vitpose-h"
    assert (job._shot_id, job._sync_config_id, job._time_start_s, job._time_end_s) == ("cap1", "sync1", 0.0, 10.0)


def test_declined_licences_run_the_job_with_rtmpose(qapp, capture_db, tmp_path, monkeypatch):
    from app.pose.main import DetectionJob

    started = []
    monkeypatch.setattr(DetectionJob, "start", lambda self: started.append(self))
    monkeypatch.setattr("app.pose.pose_model_gui.prepare_pose_model",
                        lambda parent, requested: ("rtmpose-l-133kp", "licences not accepted"))
    dlg = _make_dialog(qapp, capture_db, tmp_path)
    dlg._bbox_source_combo.setCurrentIndex(0)                  # YOLO boxes
    dlg._on_run()
    assert started[0]._pose_model_name == "rtmpose-l-133kp" and started[0]._seg_run_id is None


def test_detection_job_from_segmentation_finalises_without_manual_stitching(qapp, capture_db, tmp_path, monkeypatch):
    """A segmentation-sourced run is finalised by the job itself: a mask label is already a person."""
    from app.pose.main import DetectionJob
    from posetrak.detection.pipeline import PipelineResult

    pipelines, finalised = [], []

    class FakePipeline:
        cameras = []

        def __init__(self, **kwargs):
            pipelines.append(kwargs)

        def run(self, on_progress=None, on_camera_done=None):
            return PipelineResult(detection_run_id="run-1")

    class FakeEstimator:
        device = "cuda"

    monkeypatch.setattr("posetrak.detection.pipeline.DetectionPipeline", FakePipeline)
    monkeypatch.setattr("posetrak.detection.pose_model_choice.create_estimator", lambda model: FakeEstimator())
    monkeypatch.setattr("app.pose.finalise.auto_assign_and_finalise", lambda **kw: finalised.append(kw))

    job = DetectionJob(
        session_path=str(tmp_path / "run_detection_test.db"), shot_id="cap1", sync_config_id="sync1",
        time_start_s=0.0, time_end_s=10.0, detector_name="yolox-x", pose_model_name="soma77-vitpose-h",
        detector_conf=0.3, refine_hands=True, seg_run_id="seg1",
    )
    done = []
    job.finished.connect(done.append)
    job.run()

    from posetrak.detection.box_sources import MaskBoxSource

    assert isinstance(pipelines[0]["box_source"], MaskBoxSource) and pipelines[0]["hand_pass"] is None   # SOMA: no 133 hands
    assert done == ["run-1"]
    assert finalised[0]["persons_ordered"] == ["Alice"] and finalised[0]["confidence_scale"] == 5.0


# ---------------------------------------------------------------------------
# Marker (object) detection run auto-finalisation (marker-based-mocap design
# doc §7.1). Regression coverage for the bug where a
# completed marker run left the object stuck: no pose_observation_sequence
# was ever created, and the only UI reachable from the run was the person
# stitching tab, which has nothing to show for an object and no way to
# proceed. finalise_object_to_db existed but nothing ever called it.
# ---------------------------------------------------------------------------


def _write_marker_run(capture_db, object_id, marker_body_id, marker_ids) -> str:
    import numpy as np

    from app.pose.db_cache import MarkerKeypointWriter, create_marker_detection_run

    run_id = create_marker_detection_run(
        capture_db, shot_id="cap1", sync_config_id="sync1",
        time_start_s=0.0, time_end_s=1.0, dictionary="DICT_4X4_50",
        marker_ids=marker_ids, capture_object_id=object_id,
        marker_body_definition_id=marker_body_id,
    )
    writer = MarkerKeypointWriter(capture_db, run_id, "sv1", marker_ids=marker_ids)
    writer.add_frame(0, [])  # buffered only -- writes NaN placeholders
    writer.finalise()  # flush the buffer to detection_keypoints before overwriting

    kp = np.zeros((4 * len(marker_ids), 3), dtype=np.float32)
    kp[:, 2] = 1.0
    capture_db.execute(
        "UPDATE detection_keypoints SET keypoints=? "
        "WHERE detection_run_id=? AND shot_video_id=? AND video_frame=0",
        (kp.tobytes(), run_id, "sv1"),
    )
    capture_db.commit()
    return run_id


def test_marker_run_auto_finalises_on_finish(qapp, capture_db, tmp_path):
    """Once a marker detection job completes, _on_finished must finalise it
    into an object pose_observation_sequence right away -- an object has no
    track-to-person stitching decision to make first, so there is nothing
    else to wait on."""
    from posetrak.db.manage_capture_object import create_capture_object
    from posetrak.db.manage_marker_body import import_marker_body_str

    body_id = import_marker_body_str(
        capture_db,
        "name: test-prop\nunits: meters\nmarkers:\n"
        "  - name: hilt\n    type: aruco\n    dictionary: DICT_4X4_50\n"
        "    id: \"3\"\n    size: 0.05\n    center: [0.0, 0.0, 0.0]\n"
        "    normal: [0.0, 0.0, 1.0]\n    up: [0.0, 1.0, 0.0]\n",
        name="Test Prop",
    )
    object_id = create_capture_object(capture_db, "cap1", "prop-A", body_id)
    run_id = _write_marker_run(capture_db, object_id, body_id, ["3"])

    dlg = _make_dialog(qapp, capture_db, tmp_path)
    dlg._is_marker_run = True

    finished = []
    dlg.detection_finished.connect(lambda trial_id, rid: finished.append((trial_id, rid)))
    dlg._on_finished(run_id)

    seqs = capture_db.execute(
        "SELECT id FROM pose_observation_sequences WHERE detection_run_id = ?", (run_id,)
    ).fetchall()
    assert len(seqs) == 1
    assert len(finished) == 1
    assert finished[0][1] == run_id


def test_marker_run_finalise_failure_shows_error_but_still_reports_finished(
    qapp, capture_db, tmp_path, monkeypatch,
):
    """A finalise failure (e.g. a real ValueError/RuntimeError from
    finalise_object_to_db) must not crash the dialog -- it should surface
    the error and still let the caller know the raw run exists, so the user
    can retry from the run's own detail panel instead of losing track of it."""
    from app.pose import finalise as finalise_mod
    from PySide6.QtWidgets import QMessageBox

    def _boom(session, run_id):
        raise RuntimeError("boom")

    monkeypatch.setattr(finalise_mod, "finalise_object_to_db", _boom)
    critical_calls = []
    monkeypatch.setattr(
        QMessageBox, "critical",
        lambda *a, **k: critical_calls.append(a) or QMessageBox.StandardButton.Ok,
    )

    dlg = _make_dialog(qapp, capture_db, tmp_path)
    dlg._is_marker_run = True

    finished = []
    dlg.detection_finished.connect(lambda trial_id, rid: finished.append((trial_id, rid)))
    dlg._on_finished("nonexistent-run-id")

    assert len(critical_calls) == 1
    assert len(finished) == 1
