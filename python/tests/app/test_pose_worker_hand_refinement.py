"""Tests for PoseWorker's hand-refinement wiring (pose_worker.py).

The segmentation-driven pose extraction path (PoseWorker, queued from
CutieInitPanel's "Queue Pose" buttons) is a separate code path from the
YOLO-based DetectionPipeline/DetectionJob — it does not automatically get
HandRefinementPipeline just because DetectionJob does. These tests pin
down that _run_hand_refinement calls HandRefinementPipeline with the right
run/camera, using a fake pipeline class so no rtmlib checkpoint load is
required.
"""
from __future__ import annotations

import sqlite3

import pytest

from app.pose.pose_worker import PoseExtractionJob, PoseWorker


@pytest.fixture
def session(tmp_path):
    from posetrak.db.db import create_session, generate_id

    db_path = tmp_path / "test.db"
    conn = create_session(db_path)
    conn.row_factory = sqlite3.Row
    session_id = generate_id()
    conn.executescript(f"""
        INSERT INTO mocap_sessions (id, recorded_at) VALUES ('{session_id}', '2026-01-01');
        INSERT INTO captures (id, session_id, capture_number, label)
            VALUES ('shot-1', '{session_id}', 1, 'test');
        INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,
                                 first_video_frame, last_video_frame, actual_fps)
            VALUES ('svid-1', 'shot-1', 'cam-1', '/fake/video.mp4', 0, 1000, 120.0);
    """)
    conn.commit()
    return conn


def _make_job(**overrides) -> PoseExtractionJob:
    defaults = dict(
        job_id="job1",
        camera_label="cam-1",
        shot_video_id="svid-1",
        video_path="/fake/video.mp4",
        detection_run_id="run-1",
        seg_quality_run_id="seg-1",
        persons_ordered=["alice"],
        first_frame=0,
        last_frame=10,
    )
    defaults.update(overrides)
    return PoseExtractionJob(**defaults)


class TestRunHandRefinement:
    def test_calls_hand_refinement_pipeline_for_this_camera(self, qapp, session, monkeypatch):
        calls = []

        class _FakePipeline:
            def __init__(self, conn):
                calls.append(("init", conn))

            def run(self, run_id, cameras, on_progress=None):
                calls.append(("run", run_id, [c.shot_video_id for c in cameras]))
                return 3

        monkeypatch.setattr(
            "posetrak.detection.hand_refinement.HandRefinementPipeline", _FakePipeline
        )

        job = _make_job()
        worker = PoseWorker(job, db_path=":memory:")
        worker._run_hand_refinement(session, job)

        assert calls[0][0] == "init"
        assert calls[1] == ("run", "run-1", ["svid-1"])

    def test_camera_instance_id_resolved_from_capture_videos(self, qapp, session, monkeypatch):
        captured_cameras = []

        class _FakePipeline:
            def __init__(self, conn):
                pass

            def run(self, run_id, cameras, on_progress=None):
                captured_cameras.extend(cameras)
                return 0

        monkeypatch.setattr(
            "posetrak.detection.hand_refinement.HandRefinementPipeline", _FakePipeline
        )

        job = _make_job()
        worker = PoseWorker(job, db_path=":memory:")
        worker._run_hand_refinement(session, job)

        assert captured_cameras[0].camera_instance_id == "cam-1"
        assert captured_cameras[0].file_path == "/fake/video.mp4"

    def test_missing_rtmlib_is_non_fatal(self, qapp, session, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "posetrak.detection.hand_refinement":
                raise ImportError("no rtmlib")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        job = _make_job()
        worker = PoseWorker(job, db_path=":memory:")
        worker._run_hand_refinement(session, job)  # must not raise
