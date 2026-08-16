"""Tests for CutieInitPanel's capture-scoped construction (segmentation
before any detection run exists) -- see docs/roadmap/features/
segmentation-reuse/segmentation-reuse-design.md. This file had zero prior
test coverage; these tests exercise the piece that actually changed:
resolving cameras/persons/an existing segmentation from a capture id
directly, with no detection_runs row required to exist first.
"""
from __future__ import annotations

import pytest


@pytest.fixture()
def capture_db(tmp_path):
    """Session DB with a capture, two cameras, and one capture-level
    person -- deliberately no detection_runs row, the scenario this
    feature is meant to unblock."""
    from posetrak.db.db import create_session

    conn = create_session(tmp_path / "seg_test.db")
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
        " VALUES ('sv1', 'cap1', 'ci1', '/dev/null', 0, 100, 30.0)"
    )
    conn.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,"
        " first_video_frame, last_video_frame, actual_fps)"
        " VALUES ('sv2', 'cap1', 'ci2', '/dev/null', 0, 100, 30.0)"
    )
    conn.execute(
        "INSERT INTO capture_persons (id, capture_id, name, created_at) "
        "VALUES ('p1', 'cap1', 'Alice', '2026-01-01T00:00:00Z')"
    )
    conn.commit()
    yield conn
    conn.close()


def test_loads_capture_person_with_no_detection_run(qapp, capture_db):
    """The whole point of this feature: a person defined at the capture
    level (not via detection_track_assignments) is enough to populate the
    person selector, with no detection_runs row anywhere in the DB."""
    from app.pose.cutie_init_panel import CutieInitPanel

    panel = CutieInitPanel(capture_db, "cap1")
    try:
        assert panel._persons == ["Alice"]
        assert {c["label"] for c in panel._cameras} == {"cam_A", "cam_B"}
        assert panel._seg_run_id is None  # no segmentation created yet
    finally:
        panel.shutdown()


def test_persons_union_detection_track_assignments(qapp, capture_db):
    """A capture that already went through the old detect-first flow
    (person names only in detection_track_assignments, no capture_persons
    row for them) still surfaces those names -- backward compatible."""
    from app.pose.cutie_init_panel import CutieInitPanel

    capture_db.execute(
        "INSERT INTO sync_configs (id, shot_id, created_by) VALUES ('sync1', 'cap1', 'x')"
    )
    capture_db.execute(
        "INSERT INTO detection_runs (id, shot_id, sync_config_id, time_start_s, time_end_s,"
        " detector_model, pose_model, status, created_at)"
        " VALUES ('run1', 'cap1', 'sync1', 0.0, 2.0, 'yolo', 'rtmpose', 'complete', '2026-01-01')"
    )
    capture_db.execute(
        "INSERT INTO detection_track_assignments"
        " (detection_run_id, shot_video_id, track_id, person_name, first_frame, last_frame)"
        " VALUES ('run1', 'sv1', 3, 'Bob', 0, 100)"
    )
    capture_db.commit()

    panel = CutieInitPanel(capture_db, "cap1")
    try:
        assert panel._persons == ["Alice", "Bob"]
    finally:
        panel.shutdown()


def test_ensure_seg_run_creates_capture_scoped_row(qapp, capture_db):
    """_ensure_seg_run() writes shot_id (not detection_run_id) -- the
    schema-migration half of this feature."""
    from app.pose.cutie_init_panel import CutieInitPanel

    panel = CutieInitPanel(capture_db, "cap1")
    try:
        panel._ensure_seg_run()
        row = capture_db.execute(
            "SELECT shot_id, time_start_s, time_end_s FROM seg_quality_runs WHERE id=?",
            (panel._seg_init_run_id,),
        ).fetchone()
        assert row is not None
        assert row["shot_id"] == "cap1"
        assert row["time_start_s"] == 0.0
    finally:
        panel.shutdown()


def test_loads_existing_seg_run_by_capture(qapp, capture_db):
    """A segmentation already exists for this capture (created before any
    detection run) -- resolved by shot_id, no detection_run_id column at
    all in the query."""
    from app.pose.cutie_init_panel import CutieInitPanel
    from posetrak.db.db import generate_id

    seg_id = generate_id()
    capture_db.execute(
        "INSERT INTO seg_quality_runs (id, shot_id, time_start_s, time_end_s, created_at) "
        "VALUES (?, 'cap1', 0.0, 1e9, '2026-01-01T00:00:00Z')",
        (seg_id,),
    )
    capture_db.commit()

    panel = CutieInitPanel(capture_db, "cap1")
    try:
        assert panel._seg_run_id == seg_id
    finally:
        panel.shutdown()


def test_resolve_or_create_detection_run_needs_sync_config(qapp, capture_db):
    """No sync config for the capture yet -- can't create a detection run
    to write pose results into, so this returns None with a status message
    rather than crashing."""
    from app.pose.cutie_init_panel import CutieInitPanel

    panel = CutieInitPanel(capture_db, "cap1")
    try:
        assert panel._resolve_or_create_detection_run("rtmpose-l-133kp") is None
    finally:
        panel.shutdown()


def test_resolve_or_create_detection_run_creates_fresh_run(qapp, capture_db):
    """With a sync config present, a detection run is created lazily and
    on demand -- no pre-existing "parent" detection run needed."""
    from app.pose.cutie_init_panel import CutieInitPanel

    capture_db.execute(
        "INSERT INTO sync_configs (id, shot_id, created_by) VALUES ('sync1', 'cap1', 'x')"
    )
    capture_db.commit()

    panel = CutieInitPanel(capture_db, "cap1")
    try:
        run_id = panel._resolve_or_create_detection_run("rtmpose-l-133kp")
        assert run_id is not None
        row = capture_db.execute(
            "SELECT shot_id, sync_config_id FROM detection_runs WHERE id=?", (run_id,)
        ).fetchone()
        assert row["shot_id"] == "cap1"
        assert row["sync_config_id"] == "sync1"
    finally:
        panel.shutdown()


def test_on_finalise_auto_assigns_from_persons_ordered(qapp, capture_db):
    """_on_finalise() (segmentation-reuse gap 3): once a pose job has
    completed, Finalise builds sequences straight from person_tracks +
    self._persons, no manual stitcher pass. Simulates job completion by
    writing the DB rows PoseWorker would have written and pointing
    _pose_detection_run_id at the run, rather than driving a real
    PoseWorker/JobQueueRunner thread (needs a real video + model)."""
    import numpy as np

    from app.pose.cutie_init_panel import CutieInitPanel

    capture_db.execute(
        "INSERT INTO sync_configs (id, shot_id, created_by) VALUES ('sync1', 'cap1', 'x')"
    )
    capture_db.execute(
        "INSERT INTO detection_runs (id, shot_id, sync_config_id, time_start_s, time_end_s,"
        " detector_model, pose_model, status, created_at)"
        " VALUES ('run1', 'cap1', 'sync1', 0.0, 1.0, 'cutie-interactive', 'rtmpose-l-133kp',"
        " 'complete', '2026-01-01')"
    )
    kp = np.zeros((133, 3), dtype=np.float32).tobytes()
    capture_db.execute(
        "INSERT INTO detection_keypoints"
        " (detection_run_id, shot_video_id, video_frame, track_id, region_type,"
        "  keypoints, noise_scale)"
        " VALUES ('run1', 'sv1', 0, 1, 'full_body', ?, 0.5)",
        (kp,),
    )
    capture_db.execute(
        "INSERT INTO person_tracks"
        " (id, detection_run_id, shot_video_id, track_id, first_frame, last_frame)"
        " VALUES ('pt1', 'run1', 'sv1', 1, 0, 0)"
    )
    capture_db.commit()

    panel = CutieInitPanel(capture_db, "cap1")
    try:
        assert panel._persons == ["Alice"]
        panel._pose_detection_run_id = "run1"
        panel._on_finalise()

        seqs = capture_db.execute(
            "SELECT id FROM pose_observation_sequences WHERE detection_run_id = 'run1'"
        ).fetchall()
        assert len(seqs) == 1
        assignments = capture_db.execute(
            "SELECT person_name, track_id FROM detection_track_assignments "
            "WHERE detection_run_id = 'run1'"
        ).fetchall()
        assert [(r["person_name"], r["track_id"]) for r in assignments] == [("Alice", 1)]
    finally:
        panel.shutdown()
