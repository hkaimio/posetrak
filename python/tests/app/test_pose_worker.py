# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""PoseWorker runs the detection frame loop for one camera of a segmentation run (no GPU, no video)."""
from __future__ import annotations

import json
import sqlite3

import cv2
import numpy as np
import pytest

from app.pose.pose_worker import PoseExtractionJob, PoseWorker
from posetrak.db.db import create_session
from posetrak.detection.backends import PoseResult

H, W = 60, 80


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "s.db"
    conn = create_session(path)
    conn.executescript("""
        INSERT INTO mocap_sessions (id, recorded_at) VALUES ('ms', '2026-01-01');
        INSERT INTO captures (id, session_id, capture_number, label) VALUES ('shot', 'ms', 1, 't');
        INSERT INTO sync_configs (id, shot_id, created_by) VALUES ('sync', 'shot', 't');
        INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path, first_video_frame, last_video_frame, actual_fps)
            VALUES ('svid', 'shot', 'cam-1', '/fake/video.mp4', 0, 100, 10.0);
        INSERT INTO seg_quality_runs (id, shot_id, time_start_s, time_end_s, created_at) VALUES ('seg', 'shot', 0, 1, '2026');
        INSERT INTO detection_runs (id, shot_id, sync_config_id, time_start_s, time_end_s, detector_model, pose_model,
            status, created_at) VALUES ('run', 'shot', 'sync', 0, 1, 'segmentation', 'rtmpose-l-133kp', 'running', '2026');
    """)
    mask = np.zeros((H, W), np.uint8)
    mask[10:50, 20:40] = 1
    blob = cv2.imencode(".png", mask)[1].tobytes()
    conn.executemany("INSERT INTO seg_masks VALUES ('seg', 'svid', ?, ?)", [(f, blob) for f in range(5)])
    conn.commit()
    conn.close()
    return path


class FakeEstimator:
    name, version, input_size = "rtmpose-l-133kp", "1", (288, 384)

    def estimate_batch(self, items):
        return [[PoseResult(d.track_id, np.ones((133, 3), np.float32)) for d in dets] for _, dets in items]

    def options(self):
        return {"model": self.name}


def run_worker(db_path, monkeypatch, qapp, **job_overrides):
    monkeypatch.setattr("posetrak.detection.pose_model_choice.create_estimator", lambda model, device=None: FakeEstimator())
    monkeypatch.setattr("posetrak.detection.frame_readers.get_reader", lambda mode: (
        lambda path, first, last: ((f, np.zeros((H, W, 3), np.uint8)) for f in range(first, last))))
    job = PoseExtractionJob(
        job_id="j", camera_label="cam", shot_video_id="svid", video_path="/fake/video.mp4", detection_run_id="run",
        seg_quality_run_id="seg", persons_ordered=["alice"], first_frame=1, last_frame=3, **job_overrides)
    worker = PoseWorker(job, db_path=str(db_path))
    worker.run()
    return worker


def test_worker_writes_the_mask_boxes_poses_and_run_options(db_path, monkeypatch, qapp):
    worker = run_worker(db_path, monkeypatch, qapp, refine_hands=False)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    frames = [r[0] for r in conn.execute("SELECT video_frame FROM detection_keypoints ORDER BY 1")]
    assert frames == [1, 2, 3] and worker.get_keypoints_written() == 3
    run = conn.execute("SELECT * FROM detection_runs WHERE id='run'").fetchone()
    assert run["status"] == "complete" and run["seg_quality_run_id"] == "seg" and run["seg_revision"] is None
    config = json.loads(run["config_json"])
    assert config["box_source"]["name"] == "segmentation" and config["hand_pass"] is False
    box = conn.execute("SELECT bbox_x, bbox_y, bbox_w, bbox_h, track_id FROM person_detections LIMIT 1").fetchone()
    assert tuple(box) == (pytest.approx(29.5), pytest.approx(29.5), 19.0, 39.0, 1)
    assert [tuple(r) for r in conn.execute("SELECT first_frame, last_frame FROM person_tracks")] == [(1, 3)]


def test_worker_replaces_the_rows_of_its_range_only(db_path, monkeypatch, qapp):
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO person_detections (detection_run_id, shot_video_id, video_frame, track_id, region_type, model_name,"
        " bbox_x, bbox_y, bbox_w, bbox_h, confidence) VALUES ('run', 'svid', ?, 9, 'full_body', 'old', 1, 1, 1, 1, 1)",
        [(0,), (2,), (4,)])
    conn.commit()
    conn.close()
    run_worker(db_path, monkeypatch, qapp, refine_hands=False)
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT video_frame, track_id FROM person_detections ORDER BY 1, 2").fetchall()
    assert rows == [(0, 9), (1, 1), (2, 1), (3, 1), (4, 9)]


def test_worker_runs_the_hand_pass_in_the_loop_for_133_point_models(db_path, monkeypatch, qapp):
    seen = []

    class FakeHandPass:
        supports = staticmethod(lambda pose_model: True)

        def __init__(self, pose_model=None):
            pass

        def refine(self, items):
            seen.append(len(items))
            return [[("hand_l", np.zeros((21, 3), np.float32), 1.0)] for _ in items]

    monkeypatch.setattr("posetrak.detection.hand_refinement.HandPass", FakeHandPass)
    run_worker(db_path, monkeypatch, qapp, refine_hands=True)
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM detection_keypoints WHERE region_type='hand_l'").fetchone()[0] == 3
    assert json.loads(conn.execute("SELECT config_json FROM detection_runs").fetchone()[0])["hand_pass"] is True
    assert sum(seen) == 3


def test_a_failing_model_marks_the_run_failed(db_path, monkeypatch, qapp):
    def broken(model, device=None):
        raise RuntimeError("no weights")

    monkeypatch.setattr("posetrak.detection.pose_model_choice.create_estimator", broken)
    job = PoseExtractionJob("j", "cam", "svid", "/fake/video.mp4", "run", "seg", ["alice"], 1, 3)
    errors = []
    worker = PoseWorker(job, db_path=str(db_path))
    worker.error.connect(errors.append)
    worker.run()
    assert errors and sqlite3.connect(db_path).execute("SELECT status FROM detection_runs").fetchone()[0] == "failed"
