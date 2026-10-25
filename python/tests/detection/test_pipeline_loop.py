# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""The detection frame loop: box sources, batching across cameras, run inputs, errors, stopping."""
from __future__ import annotations

import json
import sqlite3
import threading

import cv2
import numpy as np
import pytest

from app.pose.db_cache import read_detections_for_run, read_track_spans
from posetrak.db.db import create_session
from posetrak.detection.backends import PersonDetection, PoseResult
from posetrak.detection.box_sources import MaskBoxSource, RunBoxSource, boxes_from_labels
from posetrak.detection.pipeline import CameraInfo, CameraJob, DetectionLoop, DetectionPipeline

SHOT, SYNC = "shot", "sync"
CAMERAS = ("cam-a", "cam-b")
H, W = 90, 160


@pytest.fixture
def session(tmp_path) -> sqlite3.Connection:
    conn = create_session(tmp_path / "s.db")
    conn.executescript(f"""
        INSERT INTO mocap_sessions (id, recorded_at) VALUES ('ms', '2026-01-01');
        INSERT INTO captures (id, session_id, capture_number, label) VALUES ('{SHOT}', 'ms', 1, 't');
        INSERT INTO sync_configs (id, shot_id, created_by) VALUES ('{SYNC}', '{SHOT}', 't');
    """)
    for cam in CAMERAS:
        conn.execute(
            "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path, first_video_frame,"
            " last_video_frame, actual_fps) VALUES (?,?,?,?,0,100,10.0)", (f"v-{cam}", SHOT, cam, f"/fake/{cam}.mp4"))
        conn.execute(
            "INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id, video_frame, timestamp_s)"
            " VALUES (?,?,?,0,0.0)", (SYNC, cam, f"v-{cam}"))
    conn.commit()
    return conn


def fake_reader(path, first, last):
    for frame in range(first, last):
        yield frame, np.full((H, W, 3), frame % 255, np.uint8)


class Boxes:
    """A box source with one person (track 1) on every frame but those in *skip*."""

    name, version, conf = "fake-boxes", "", 0.5

    def __init__(self, skip=()):
        self.skip, self.started = set(skip), []

    def start_camera(self, cam, first, last):
        self.started.append((cam.shot_video_id, first, last))

    def wants(self, cam, frame):
        return frame not in self.skip

    def prepare(self, cam, frame, image):
        return [PersonDetection(1, np.array([80, 45, 40, 80], np.float32), 0.9)]

    def detect(self, works):
        return [w.prepared for w in works]

    def run_inputs(self):
        return {}

    def options(self):
        return {"fake": True}


class Estimator:
    name, version, input_size = "fake-pose", "1", (32, 24)

    def __init__(self, fail_after=None):
        self.batches, self.fail_after = [], fail_after

    def estimate_batch(self, items):
        self.batches.append(len(items))
        if self.fail_after is not None and len(self.batches) > self.fail_after:
            raise RuntimeError("model exploded")
        return [[PoseResult(d.track_id, np.full((3, 3), frame_value(img), np.float32)) for d in dets]
                for img, dets in items]

    def estimate(self, frame, detections):
        return self.estimate_batch([(frame, detections)])[0]

    def options(self):
        return {"model": self.name}


def frame_value(img) -> float:
    return float(img[0, 0, 0])


def make_loop(session, run_id, boxes, estimator, **kw):
    return DetectionLoop(session, run_id, boxes, estimator, reader=fake_reader, **kw)


def new_run(session):
    from app.pose.db_cache import create_detection_run

    return create_detection_run(session, SHOT, SYNC, 0.0, 1.0, detector_model="fake-boxes", pose_model="fake-pose")


def cam_jobs(first=0, last=12):
    return [CameraJob(CameraInfo(f"v-{c}", c, f"/fake/{c}.mp4", 10.0, 0, 0.0, label=c), first, last) for c in CAMERAS]


def test_loop_writes_every_camera_and_batches_across_cameras(session):
    run_id, estimator = new_run(session), Estimator()
    progress = []
    frames = make_loop(session, run_id, Boxes(), estimator, batch_frames=4).run(
        cam_jobs(), on_progress=lambda d, t, label: progress.append((label, d, t)))
    assert frames == 24
    for cam in CAMERAS:
        rows = read_detections_for_run(session, run_id, f"v-{cam}")
        assert sorted(r["video_frame"] for r in rows) == list(range(12))
        spans = read_track_spans(session, run_id, f"v-{cam}")
        assert [(s["first_frame"], s["last_frame"]) for s in spans] == [(0, 11)]
    kp = session.execute("SELECT keypoints, noise_scale FROM detection_keypoints WHERE video_frame=5 LIMIT 1").fetchone()
    assert np.frombuffer(kp["keypoints"], np.float32)[0] == 5.0
    assert kp["noise_scale"] == pytest.approx(40 / 24)           # box width over model input width
    assert sum(estimator.batches) == 24 and max(estimator.batches) > 1     # frames of one call were batched
    assert {label for label, _, _ in progress} == set(CAMERAS) and max(d for _, d, _ in progress) == 12


def test_frames_the_source_does_not_want_are_counted_but_not_written(session):
    run_id = new_run(session)
    frames = make_loop(session, run_id, Boxes(skip={3, 4}), Estimator()).run(cam_jobs(0, 6)[:1])
    assert frames == 6
    rows = read_detections_for_run(session, run_id, "v-cam-a")
    assert sorted(r["video_frame"] for r in rows) == [0, 1, 2, 5]


def test_pose_noise_scale_overrides_the_box_width(session):
    class Scaled(Estimator):
        def estimate_batch(self, items):
            return [[PoseResult(d.track_id, np.zeros((3, 3), np.float32), noise_scale=2.5) for d in dets]
                    for _, dets in items]

    run_id = new_run(session)
    make_loop(session, run_id, Boxes(), Scaled()).run(cam_jobs(0, 2)[:1])
    assert {r[0] for r in session.execute("SELECT noise_scale FROM detection_keypoints")} == {2.5}


def test_an_estimator_error_surfaces_and_does_not_hang(session):
    run_id = new_run(session)
    loop = make_loop(session, run_id, Boxes(), Estimator(fail_after=1), batch_frames=2)
    with pytest.raises(RuntimeError, match="exploded"):
        loop.run(cam_jobs(0, 40))


def test_a_decode_error_surfaces_and_does_not_hang(session):
    def broken(path, first, last):
        yield first, np.zeros((H, W, 3), np.uint8)
        raise OSError("corrupt video")

    run_id = new_run(session)
    loop = DetectionLoop(session, run_id, Boxes(), Estimator(), reader=broken)
    with pytest.raises(OSError, match="corrupt"):
        loop.run(cam_jobs(0, 40))


def test_stop_event_ends_the_loop_early_without_error(session):
    stop = threading.Event()
    run_id = new_run(session)
    seen = []

    def on_progress(done, total, label):
        seen.append(done)
        if sum(1 for _ in seen) == 5:
            stop.set()

    frames = make_loop(session, run_id, Boxes(), Estimator(), stop_event=stop).run(cam_jobs(0, 400), on_progress)
    assert 5 <= frames < 800


def test_run_box_source_reuses_boxes_and_track_ids_of_a_run(session):
    from app.pose.db_cache import create_detection_run

    source = create_detection_run(session, SHOT, SYNC, 0.0, 1.0, detector_model="yolox-x", pose_model="rtmpose-l-133kp")
    session.executemany(
        "INSERT INTO person_detections (detection_run_id, shot_video_id, video_frame, track_id, region_type,"
        " model_name, bbox_x, bbox_y, bbox_w, bbox_h, confidence) VALUES (?, 'v-cam-a', ?, ?, 'full_body', 'm', 80, 45, 40, 80, 0.8)",
        [(source, 2, 7), (source, 2, 9), (source, 4, 7)])
    session.commit()
    run_id = new_run(session)
    boxes = RunBoxSource(session, source)
    make_loop(session, run_id, boxes, Estimator()).run(cam_jobs(0, 6)[:1])
    rows = session.execute(
        "SELECT video_frame, track_id FROM person_detections WHERE detection_run_id=? ORDER BY 1, 2", (run_id,)).fetchall()
    assert [tuple(r) for r in rows] == [(2, 7), (2, 9), (4, 7)]
    assert boxes.run_inputs() == {"source_detection_run_id": source}


def test_boxes_from_labels_are_inclusive_pixel_boxes():
    mask = np.zeros((10, 20), np.uint8)
    mask[2:5, 3:8] = 1
    mask[7, 15] = 2
    assert boxes_from_labels(mask, [1, 2, 3]) == {1: (3, 2, 7, 4), 2: (15, 7, 15, 7)}


def add_seg_run(session, mask: np.ndarray, frame: int = 3) -> None:
    session.execute(
        "INSERT INTO seg_quality_runs (id, shot_id, time_start_s, time_end_s, created_at)"
        " VALUES ('seg', ?, 0, 1, '2026-01-01')", (SHOT,))
    session.execute(
        "INSERT INTO seg_masks (seg_quality_run_id, shot_video_id, frame_idx, mask_blob) VALUES ('seg', 'v-cam-a', ?, ?)",
        (frame, cv2.imencode(".png", mask)[1].tobytes()))
    session.commit()


def test_mask_box_source_scales_boxes_to_the_frame_and_attaches_the_grown_mask(session):
    mask = np.zeros((45, 80), np.uint8)             # half the frame's size
    mask[10:30, 20:40] = 1                          # label 1: x 20..39, y 10..29
    mask[1, 70] = 2                                 # label 2: one mask pixel, 2x2 in the frame: too small
    add_seg_run(session, mask)
    source = MaskBoxSource(session, "seg", ["ann", "bob"], gate_margin_px=8.0, work_width=80)
    cam = CameraInfo("v-cam-a", "cam-a", "x", 10.0, 0, 0.0)
    assert source.prepare(cam, 99, np.zeros((H, W, 3), np.uint8)) == []        # no mask stored for the frame
    (ann,) = source.prepare(cam, 3, np.zeros((H, W, 3), np.uint8))             # frame twice the mask's size
    assert ann.track_id == 1
    # x 20..39 -> pixels 40..79, y 10..29 -> 20..59 in the frame; centre-format box
    assert ann.bbox.tolist() == [pytest.approx((40 + 79) / 2), pytest.approx((20 + 59) / 2), 39.0, 39.0]
    assert ann.mask.shape == (45, 80)                                          # work width 80 of a 160-wide frame
    assert ann.mask[10:30, 20:40].all() and ann.mask.sum() > 20 * 20           # grown by the margin, never shrunk
    assert source.run_inputs() == {"seg_quality_run_id": "seg", "seg_revision": 0}


def test_pipeline_records_the_run_inputs_and_options(session, monkeypatch):
    monkeypatch.setattr("posetrak.detection.pipeline.get_reader", lambda mode: fake_reader)

    class FromRun(Boxes):
        name = "detection-run"

        def run_inputs(self):
            return {"source_detection_run_id": None, "seg_revision": 3}

    pipeline = DetectionPipeline(session, SHOT, SYNC, 0.0, 0.5, box_source=FromRun(), estimator=Estimator(),
                                 decode_mode="thread")
    result = pipeline.run()
    assert result.status == "complete" and result.frames_processed == 10
    row = session.execute("SELECT * FROM detection_runs WHERE id=?", (result.detection_run_id,)).fetchone()
    assert (row["detector_model"], row["pose_model"], row["status"], row["seg_revision"]) == (
        "detection-run", "fake-pose", "complete", 3)
    config = json.loads(row["config_json"])
    assert config["estimator"] == {"model": "fake-pose"} and config["box_source"]["fake"] is True
    assert config["decode_mode"] == "thread" and config["hand_pass"] is False
