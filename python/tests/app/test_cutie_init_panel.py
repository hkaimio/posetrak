# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

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


# ---------------------------------------------------------------------------
# Global-time scrubbing (trial start/end visible + adjustable, camera
# switches don't reset marks) -- see docs/roadmap/features/
# segmentation-reuse/status.md's 2026-08-16 note.
# ---------------------------------------------------------------------------


@pytest.fixture()
def synced_capture_db(capture_db):
    """Extends capture_db with a solved sync (sv2 starts 1s later than
    sv1, both 30fps) and a trial spanning global time [2.0, 8.0)s."""
    # capture_db's cameras are only 100 frames long -- widen to cover the
    # sync points below (up to frame 300, 10s @ 30fps), otherwise the
    # scrubber's computed global range gets clamped to a too-short window.
    capture_db.execute("UPDATE capture_videos SET last_video_frame = 300")
    capture_db.execute(
        "INSERT INTO sync_configs (id, shot_id, created_by) VALUES ('sync1', 'cap1', 'x')"
    )
    for svid, ci, offset in (("sv1", "ci1", 0.0), ("sv2", "ci2", 1.0)):
        capture_db.execute(
            "INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id,"
            " video_frame, timestamp_s) VALUES ('sync1', ?, ?, 0, ?)",
            (ci, svid, offset),
        )
        capture_db.execute(
            "INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id,"
            " video_frame, timestamp_s) VALUES ('sync1', ?, ?, 300, ?)",
            (ci, svid, offset + 10.0),
        )
    capture_db.execute(
        "INSERT INTO trials (id, capture_id, name, time_start_s, time_end_s) "
        "VALUES ('trial1', 'cap1', 'T', 2.0, 8.0)"
    )
    capture_db.commit()
    return capture_db


def test_marks_default_to_trial_bounds(qapp, synced_capture_db):
    from app.pose.cutie_init_panel import CutieInitPanel

    panel = CutieInitPanel(synced_capture_db, "cap1", trial_id="trial1")
    try:
        assert panel._sync_table is not None
        assert panel._to_seconds(panel._mark_start) == pytest.approx(2.0)
        assert panel._to_seconds(panel._mark_end) == pytest.approx(8.0)
        assert "00:02.00" in panel._trial_range_label.text()
    finally:
        panel.shutdown()


def test_marks_default_to_full_range_without_trial_id(qapp, synced_capture_db):
    """No trial_id (opened from the Capture page, not a trial) -- marks
    default to the full available range, not zero-width."""
    from app.pose.cutie_init_panel import CutieInitPanel

    panel = CutieInitPanel(synced_capture_db, "cap1")
    try:
        assert panel._sync_table is not None
        assert panel._trial_range_label.text() == ""
        assert panel._mark_start == panel._scrubber.minimum()
        assert panel._mark_end == panel._scrubber.maximum()
    finally:
        panel.shutdown()


def test_mark_start_end_freely_adjustable_away_from_trial_bounds(qapp, synced_capture_db):
    """Explicit requirement: marks default to the trial's bounds but stay
    freely adjustable -- rerun only part of a trial, or cover a wider
    range than one trial on purpose."""
    from app.pose.cutie_init_panel import CutieInitPanel

    panel = CutieInitPanel(synced_capture_db, "cap1", trial_id="trial1")
    try:
        # Narrow to just part of the trial.
        panel._scrubber.setValue(panel._to_units(3.0))
        panel._on_mark_start()
        panel._scrubber.setValue(panel._to_units(5.0))
        panel._on_mark_end()
        assert panel._to_seconds(panel._mark_start) == pytest.approx(3.0)
        assert panel._to_seconds(panel._mark_end) == pytest.approx(5.0)

        # Widen past the trial's own end -- also allowed.
        panel._scrubber.setValue(panel._to_units(9.5))
        panel._on_mark_end()
        assert panel._to_seconds(panel._mark_end) == pytest.approx(9.5)
    finally:
        panel.shutdown()


def test_camera_switch_preserves_marks_and_maps_to_correct_local_frame(qapp, synced_capture_db):
    """The bug this whole feature fixes: switching cameras used to reset
    Mark Start/End to the new camera's full range. With a sync table,
    marks are global and switching only changes which local frame is
    displayed for the same global instant."""
    from app.pose.cutie_init_panel import CutieInitPanel

    panel = CutieInitPanel(synced_capture_db, "cap1", trial_id="trial1")
    try:
        mark_start, mark_end = panel._mark_start, panel._mark_end
        panel._scrubber.setValue(panel._to_units(5.0))

        # sv1 is selected by default (alphabetically first); switch to sv2.
        assert panel._cam_combo.currentData()["id"] == "sv1"
        idx_sv2 = next(
            i for i in range(panel._cam_combo.count())
            if panel._cam_combo.itemData(i)["id"] == "sv2"
        )
        panel._cam_combo.setCurrentIndex(idx_sv2)

        # Marks and scrubber position are untouched by the switch.
        assert panel._mark_start == mark_start
        assert panel._mark_end == mark_end
        assert panel._scrubber.value() == panel._to_units(5.0)

        # sv2 started 1s later than sv1 -- at global t=5.0, sv1's local
        # frame is 150 (5.0*30) and sv2's is 120 ((5.0-1.0)*30).
        cam_sv1 = next(c for c in panel._cameras if c["id"] == "sv1")
        cam_sv2 = next(c for c in panel._cameras if c["id"] == "sv2")
        assert panel._local_frame_for(cam_sv1, panel._to_units(5.0)) == 150
        assert panel._local_frame_for(cam_sv2, panel._to_units(5.0)) == 120
    finally:
        panel.shutdown()


def test_global_units_for_local_round_trips(qapp, synced_capture_db):
    from app.pose.cutie_init_panel import CutieInitPanel

    panel = CutieInitPanel(synced_capture_db, "cap1")
    try:
        cam_sv2 = next(c for c in panel._cameras if c["id"] == "sv2")
        g = panel._to_units(6.0)
        local = panel._local_frame_for(cam_sv2, g)
        assert local == 150  # (6.0 - 1.0) * 30
        assert panel._global_units_for_local(cam_sv2, local) == g
    finally:
        panel.shutdown()


def test_ensure_seg_run_persists_real_marked_range(qapp, synced_capture_db):
    """With a sync table, seg_quality_runs.time_start_s/time_end_s record
    the actual selected range instead of the 0.0/1e9 "covers everything"
    sentinel -- closes the gap flagged in status.md's 2026-08-16 note."""
    from app.pose.cutie_init_panel import CutieInitPanel

    panel = CutieInitPanel(synced_capture_db, "cap1", trial_id="trial1")
    try:
        panel._ensure_seg_run()
        row = synced_capture_db.execute(
            "SELECT time_start_s, time_end_s FROM seg_quality_runs WHERE id=?",
            (panel._seg_init_run_id,),
        ).fetchone()
        assert row["time_start_s"] == pytest.approx(2.0)
        assert row["time_end_s"] == pytest.approx(8.0)
    finally:
        panel.shutdown()


def test_no_sync_points_falls_back_to_legacy_per_camera_domain(qapp, capture_db):
    """capture_db has no sync_points at all -- the whole global-time
    feature must stay inert, matching pre-refactor behaviour exactly."""
    from app.pose.cutie_init_panel import CutieInitPanel

    panel = CutieInitPanel(capture_db, "cap1")
    try:
        assert panel._sync_table is None
        cam = panel._cam_combo.currentData()
        assert panel._scrubber.minimum() == cam["track_first"]
        assert panel._scrubber.maximum() == cam["track_last"]
        # Legacy fallback: scrubber value IS the local frame directly.
        assert panel._local_frame_for(cam, 42) == 42
    finally:
        panel.shutdown()


def test_queue_tracking_range_reflects_direction(qapp, synced_capture_db):
    """A Cutie tracking pass only ever propagates in one direction from
    the current frame -- CutieWorker._run_forward reads only last_frame
    (init_frame -> last_frame) and _run_backward reads only first_frame
    (first_frame -> init_frame). Forward should show current frame -> mark
    end; backward, mark start -> current frame -- not the full mark_start-
    mark_end range for both, which used to make the Job Queue list show
    the identical range regardless of direction."""
    import cv2
    import numpy as np
    from app.pose.cutie_init_panel import CutieInitPanel

    synced_capture_db.execute(
        "INSERT INTO seg_quality_runs (id, shot_id, time_start_s, time_end_s, created_at) "
        "VALUES ('seg1', 'cap1', 0.0, 1e9, '2026-01-01T00:00:00Z')"
    )
    ok, buf = cv2.imencode(".png", np.ones((4, 4), dtype=np.uint8))
    assert ok
    synced_capture_db.execute(
        "INSERT INTO seg_masks (seg_quality_run_id, shot_video_id, frame_idx, mask_blob) "
        "VALUES ('seg1', 'sv1', 150, ?)",
        (buf.tobytes(),),
    )
    synced_capture_db.commit()

    panel = CutieInitPanel(synced_capture_db, "cap1", trial_id="trial1")
    try:
        assert panel._cam_combo.currentData()["id"] == "sv1"
        panel._scrubber.setValue(panel._to_units(5.0))  # sv1 local frame 150

        # Trial range [2.0, 8.0)s @ 30fps -> sv1 local frames [60, 240].
        panel._queue_tracking("forward")
        fwd_job = panel._runner.jobs[-1]
        assert fwd_job.direction == "forward"
        assert fwd_job.init_frame == 150
        assert fwd_job.first_frame == 150
        assert fwd_job.last_frame == 240

        panel._queue_tracking("backward")
        bwd_job = panel._runner.jobs[-1]
        assert bwd_job.direction == "backward"
        assert bwd_job.init_frame == 150
        assert bwd_job.first_frame == 60
        assert bwd_job.last_frame == 150
    finally:
        panel.shutdown()
