# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""pose_worker.py — Segmentation-driven pose extraction as a queued job.

A thin QThread around the detection frame loop (``posetrak.detection.pipeline.DetectionLoop``): the boxes come
from the labels of a segmentation run (``MaskBoxSource``), the loop runs the job's pose model on one camera's
frame range, and the rows go directly to the DB via DetectionBatchWriter into the detection run the caller
created.

The worker opens its own SQLite connection so DB writes happen in the worker thread, avoiding the main-thread
signal queue problems entirely. Only progress and tracking_done signals cross the thread boundary.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from posetrak.db.db import _connect
from posetrak.detection.box_sources import load_mask  # noqa: F401  (re-exported for app.pose.finalise)

log = logging.getLogger(__name__)

_PROGRESS_EVERY = 50


@dataclass
class PoseExtractionJob:
    job_id: str
    camera_label: str
    shot_video_id: str
    video_path: str
    detection_run_id: str       # write into this run (created by caller before enqueue)
    seg_quality_run_id: str     # read masks from this run
    persons_ordered: list[str]  # index i → label i+1 = track_id
    first_frame: int
    last_frame: int
    pose_model: str = "rtmpose-l-133kp"
    overwrite_range: bool = True    # delete existing keypoints in range first
    refine_hands: bool = True       # run HandRefinementPipeline after estimation
    status: str = "pending"
    keypoints_written: int = 0
    error: str = ""

    @property
    def summary(self) -> str:
        model_short = self.pose_model.split("-")[0].upper()
        return f"🎯 {self.camera_label}  {self.first_frame}–{self.last_frame}  [{model_short}]"


class PoseWorker(QThread):
    """Runs pose estimation for one camera over a mask-covered frame range.

    Signals
    -------
    progress(done, total):
        Emitted every 50 frames.
    tracking_done():
        Emitted when the pass completes or is stopped.
    error(message):
        Emitted on unrecoverable failure.
    """

    progress      = Signal(int, int)
    tracking_done = Signal()
    error         = Signal(str)

    def __init__(
        self,
        job: PoseExtractionJob,
        db_path: str,
        device: str = "cuda",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._job = job
        self._db_path = db_path
        self._device = device
        self._stop_event = threading.Event()
        self._keypoints_written = 0

    def get_keypoints_written(self) -> int:
        return self._keypoints_written

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            self._run_pose()
        except Exception:
            log.exception("PoseWorker error")
            self.error.emit("Pose extraction failed — see console for details.")
        finally:
            log.info("PoseWorker: emitting tracking_done  t=%.3f", time.monotonic())
            self.tracking_done.emit()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_pose(self) -> None:
        from app.pose.db_cache import mark_run_complete
        from posetrak.detection.box_sources import MaskBoxSource
        from posetrak.detection.frame_readers import get_reader
        from posetrak.detection.pipeline import (
            DEFAULT_DECODE_MODE, CameraInfo, CameraJob, DetectionLoop, record_run_options, run_config,
        )
        from posetrak.detection.pose_model_choice import create_estimator

        job = self._job
        t0 = time.monotonic()
        log.info("PoseWorker: start  %s  frames %d-%d  model=%s  t=%.3f",
                 job.camera_label, job.first_frame, job.last_frame, job.pose_model, t0)

        conn = _connect(Path(self._db_path))

        try:
            if job.overwrite_range:
                _delete_range(conn, job)

            estimator = create_estimator(job.pose_model, device=self._device)
            box_source = MaskBoxSource(conn, job.seg_quality_run_id, job.persons_ordered)
            hand_pass = None
            if job.refine_hands and "133" in job.pose_model:
                try:
                    from posetrak.detection.hand_refinement import HandPass
                    hand_pass = HandPass()
                except ImportError:
                    log.warning("PoseWorker: rtmlib unavailable, skipping hand refinement")

            row = conn.execute("SELECT camera_instance_id FROM capture_videos WHERE id=?",
                               (job.shot_video_id,)).fetchone()
            cam = CameraInfo(
                shot_video_id=job.shot_video_id,
                camera_instance_id=row["camera_instance_id"] if row else job.shot_video_id,
                file_path=job.video_path, actual_fps=0.0, ref_frame=0, ref_timestamp_s=0.0,
                label=job.camera_label,
            )
            record_run_options(conn, job.detection_run_id, box_source,
                               run_config(box_source, estimator, hand_pass, DEFAULT_DECODE_MODE, 6))

            def on_progress(done: int, total: int, label: str) -> None:
                if done % _PROGRESS_EVERY == 0:
                    self.progress.emit(done, total)

            loop = DetectionLoop(conn, job.detection_run_id, box_source, estimator, hand_pass=hand_pass,
                                 reader=get_reader(DEFAULT_DECODE_MODE), stop_event=self._stop_event)
            done = loop.run([CameraJob(cam, job.first_frame, job.last_frame + 1)], on_progress)
            self._keypoints_written = conn.execute(
                "SELECT COUNT(*) FROM detection_keypoints WHERE detection_run_id=? AND shot_video_id=?"
                " AND region_type='full_body' AND video_frame BETWEEN ? AND ?",
                (job.detection_run_id, job.shot_video_id, job.first_frame, job.last_frame)).fetchone()[0]
            mark_run_complete(conn, job.detection_run_id)
            log.info("PoseWorker: done  %d frames  %d keypoint rows  %.2fs",
                     done, self._keypoints_written, time.monotonic() - t0)

        except Exception:
            mark_run_complete(conn, job.detection_run_id, "failed")
            raise
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _delete_range(conn: sqlite3.Connection, job: PoseExtractionJob) -> None:
    """Delete existing detection data for this camera + frame range."""
    params = (job.detection_run_id, job.shot_video_id, job.first_frame, job.last_frame)
    conn.execute(
        "DELETE FROM person_detections "
        "WHERE detection_run_id=? AND shot_video_id=? AND video_frame BETWEEN ? AND ?",
        params,
    )
    conn.execute(
        "DELETE FROM detection_keypoints "
        "WHERE detection_run_id=? AND shot_video_id=? AND video_frame BETWEEN ? AND ?",
        params,
    )
    conn.execute(
        "DELETE FROM frame_cache_entries "
        "WHERE detection_run_id=? AND shot_video_id=? AND frame_idx BETWEEN ? AND ?",
        params,
    )
    conn.commit()
    log.info("PoseWorker: deleted existing data  svid=%s  frames %d-%d",
             job.shot_video_id, job.first_frame, job.last_frame)
