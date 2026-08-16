"""Tests for RunDetectionDialog's segmentation bbox source (segmentation-
reuse gap 2: docs/roadmap/features/segmentation-reuse/
segmentation-reuse-design.md). This file had zero prior test coverage.

The actual PoseWorker/JobQueueRunner execution needs a real video file and
model, so these tests stop short of driving a real background thread:
_run_from_segmentation's job-construction is exercised with
JobQueueRunner.start() monkeypatched to a no-op, and the finalise wiring
(_on_seg_queue_done) is exercised directly against DB state seeded the way
a completed PoseWorker job would have left it -- the same pattern
test_cutie_init_panel.py uses for the same reason.
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


def test_bbox_source_combo_lists_yolo_and_segmentation(qapp, capture_db, tmp_path):
    dlg = _make_dialog(qapp, capture_db, tmp_path)
    assert dlg._bbox_source_combo is not None
    labels = [dlg._bbox_source_combo.itemText(i) for i in range(dlg._bbox_source_combo.count())]
    assert labels[0] == "YOLO detection"
    assert any("Alice" in label for label in labels[1:])
    assert dlg._bbox_source_combo.itemData(1) == "seg1"


def test_choosing_segmentation_disables_yolo_only_fields(qapp, capture_db, tmp_path):
    dlg = _make_dialog(qapp, capture_db, tmp_path)
    assert dlg._detector_combo.isEnabled()
    assert dlg._conf_spin.isEnabled()

    dlg._bbox_source_combo.setCurrentIndex(1)  # the segmentation entry
    assert not dlg._detector_combo.isEnabled()
    assert not dlg._conf_spin.isEnabled()

    dlg._bbox_source_combo.setCurrentIndex(0)  # back to YOLO
    assert dlg._detector_combo.isEnabled()
    assert dlg._conf_spin.isEnabled()


def test_run_from_segmentation_builds_one_job_per_camera(qapp, capture_db, tmp_path, monkeypatch):
    """_run_from_segmentation creates a detection run and one
    PoseExtractionJob per camera, with frame ranges resolved via the sync
    table (mirroring DetectionPipeline._frame_range) and persons_ordered
    read from the segmentation's own persisted snapshot."""
    from app.pose.job_queue_runner import JobQueueRunner

    monkeypatch.setattr(JobQueueRunner, "start", lambda self: None)

    dlg = _make_dialog(qapp, capture_db, tmp_path)
    dlg._run_from_segmentation("seg1", "sync1", 0.0, 10.0)

    assert dlg._seg_detection_run_id is not None
    run_row = capture_db.execute(
        "SELECT shot_id, sync_config_id, detector_model FROM detection_runs WHERE id=?",
        (dlg._seg_detection_run_id,),
    ).fetchone()
    assert run_row["shot_id"] == "cap1"
    assert run_row["sync_config_id"] == "sync1"
    assert run_row["detector_model"] == "segmentation"

    assert dlg._seg_jobs_total == 2
    jobs = dlg._seg_runner.jobs
    assert {j.shot_video_id for j in jobs} == {"sv1", "sv2"}
    for j in jobs:
        assert j.seg_quality_run_id == "seg1"
        assert j.persons_ordered == ["Alice"]
        assert j.first_frame == 0
        assert j.last_frame == 300  # 10s @ 30fps per the sync points


def test_on_seg_queue_done_finalises_without_manual_stitching(qapp, capture_db, tmp_path, monkeypatch):
    """Once the queue completes, results are finalised automatically --
    no manual track-to-person assignment step (gap 3, reused here)."""
    import numpy as np
    from app.pose.job_queue_runner import JobQueueRunner

    monkeypatch.setattr(JobQueueRunner, "start", lambda self: None)

    dlg = _make_dialog(qapp, capture_db, tmp_path)
    dlg._run_from_segmentation("seg1", "sync1", 0.0, 10.0)

    # Simulate what a completed PoseWorker job would have written.
    run_id = dlg._seg_detection_run_id
    kp = np.zeros((133, 3), dtype=np.float32).tobytes()
    capture_db.execute(
        "INSERT INTO detection_keypoints"
        " (detection_run_id, shot_video_id, video_frame, track_id, region_type,"
        "  keypoints, noise_scale)"
        " VALUES (?, 'sv1', 0, 1, 'full_body', ?, 0.5)",
        (run_id, kp),
    )
    capture_db.execute(
        "INSERT INTO person_tracks"
        " (id, detection_run_id, shot_video_id, track_id, first_frame, last_frame)"
        " VALUES ('pt1', ?, 'sv1', 1, 0, 0)",
        (run_id,),
    )
    capture_db.commit()

    finished = []
    dlg.detection_finished.connect(lambda trial_id, rid: finished.append((trial_id, rid)))
    dlg._on_seg_queue_done()

    seqs = capture_db.execute(
        "SELECT id FROM pose_observation_sequences WHERE detection_run_id = ?", (run_id,)
    ).fetchall()
    assert len(seqs) == 1
    assert len(finished) == 1
    assert finished[0][1] == run_id
    # A trial was created and linked (trial_id was None at dialog construction).
    trial_row = capture_db.execute(
        "SELECT trial_id FROM detection_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert trial_row["trial_id"] == finished[0][0]
