"""job_queue_runner.py — FIFO queue of Cutie tracking jobs.

TrackingJob holds all parameters for one CutieWorker pass (including the
init mask as a PNG blob so the job is fully self-contained).  JobQueueRunner
owns a single CutieWorker and starts the next pending job automatically
when the current one finishes.

The panel creates jobs via enqueue(); signals forward mask_ready per frame
and lifecycle events (job_started/finished/failed, queue_done) so the panel
can handle DB writes and UI updates without knowing which job is running.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

log = logging.getLogger(__name__)


@dataclass
class TrackingJob:
    job_id: str
    camera_label: str
    shot_video_id: str
    video_path: str
    init_frame: int
    init_mask_png: bytes        # PNG-encoded uint8 labeled mask
    persons_ordered: list[str]
    first_frame: int
    last_frame: int
    direction: str              # "forward" | "backward"
    max_dim: int = 1920
    status: str = "pending"     # pending | running | done | failed | cancelled
    masks_written: int = 0
    error: str = ""

    @property
    def summary(self) -> str:
        arrow = "▶" if self.direction == "forward" else "◀"
        return f"{arrow} {self.camera_label}  {self.first_frame}–{self.last_frame}"


class JobQueueRunner(QObject):
    """Executes TrackingJob items one at a time using CutieWorker.

    Jobs are run in FIFO order.  When a job finishes (or fails), the next
    pending job starts automatically.  Calling stop_current() stops the
    running worker after the current frame; the queue then continues with
    the next job.  cancel_all() stops the current job and marks all pending
    jobs as cancelled.
    """

    # Forwarded from CutieWorker — batch of (frame_idx, png_bytes) tuples
    mask_ready = Signal(str, object)        # svid, list[tuple[int, bytes]]
    progress   = Signal(int, int)           # done, total

    # Job lifecycle
    job_started  = Signal(str)              # job_id
    job_finished = Signal(str, int)         # job_id, masks_written
    job_failed   = Signal(str, str)         # job_id, error_message
    queue_done   = Signal()                 # all jobs complete/cancelled

    def __init__(self, db_path: str = "", parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._db_path = db_path
        self._jobs: list = []           # TrackingJob | PoseExtractionJob
        self._worker = None
        self._current = None
        self._masks_this_job: int = 0
        self._batch_count: int = 0
        self._t_job_start: float = 0.0
        self._cutie_model = None        # cached after first Cutie job; avoids Hydra re-init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def jobs(self) -> list[TrackingJob]:
        return self._jobs

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def enqueue(self, job: TrackingJob) -> None:
        """Add *job* to the queue.  Does NOT auto-start — call start() to run."""
        self._jobs.append(job)

    def start(self) -> None:
        """Begin executing pending jobs (no-op if already running)."""
        if not self.is_running:
            self._start_next()

    def stop_current(self) -> None:
        """Request the running worker to stop after the current frame."""
        if self._worker is not None:
            self._worker.stop()

    def cancel_all(self) -> None:
        """Stop current job and mark all pending jobs as cancelled."""
        self.stop_current()
        for job in self._jobs:
            if job.status == "pending":
                job.status = "cancelled"

    def remove_pending(self, job_id: str) -> bool:
        """Cancel a specific pending job. Returns True if found."""
        for job in self._jobs:
            if job.job_id == job_id and job.status == "pending":
                job.status = "cancelled"
                return True
        return False

    def shutdown(self) -> None:
        """Stop the running worker and wait for it (called on panel close)."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_next(self) -> None:
        pending = [j for j in self._jobs if j.status == "pending"]
        if not pending:
            self.queue_done.emit()
            return
        self._run_job(pending[0])

    def _run_job(self, job) -> None:
        from app.pose.pose_worker import PoseExtractionJob
        if isinstance(job, PoseExtractionJob):
            self._run_pose_job(job)
        else:
            self._run_tracking_job(job)

    def _run_tracking_job(self, job: TrackingJob) -> None:
        from app.pose.cutie_worker import CutieWorker

        buf = np.frombuffer(job.init_mask_png, dtype=np.uint8)
        init_mask = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if init_mask is None:
            job.status = "failed"
            job.error  = "Could not decode init mask PNG"
            self.job_failed.emit(job.job_id, job.error)
            self._start_next()
            return

        job.status = "running"
        self._current = job
        self._masks_this_job = 0
        self._batch_count = 0
        self._t_job_start = time.monotonic()
        log.info("JobQueueRunner: starting job %s  %s %s  frames %d-%d  t=%.3f",
                 job.job_id, job.direction, job.camera_label,
                 job.first_frame, job.last_frame, self._t_job_start)
        self.job_started.emit(job.job_id)

        self._worker = CutieWorker(
            video_path=job.video_path,
            init_frame=job.init_frame,
            init_mask=init_mask,
            persons_ordered=job.persons_ordered,
            first_frame=job.first_frame,
            last_frame=job.last_frame,
            direction=job.direction,
            max_dim=job.max_dim,
            model=self._cutie_model,    # None on first job; cached thereafter
        )
        svid = job.shot_video_id
        self._worker.mask_ready.connect(
            lambda batch, s=svid: self._on_batch_ready(s, batch)
        )
        self._worker.progress.connect(self.progress)
        self._worker.tracking_done.connect(self._on_worker_finished)
        self._worker.error.connect(
            lambda msg, jid=job.job_id: self._on_worker_error(jid, msg)
        )
        self._worker.start()

    def _run_pose_job(self, job) -> None:
        from app.pose.pose_worker import PoseWorker

        job.status = "running"
        self._current = job
        self._masks_this_job = 0
        self._batch_count = 0
        self._t_job_start = time.monotonic()
        log.info("JobQueueRunner: starting pose job %s  %s  frames %d-%d  model=%s  t=%.3f",
                 job.job_id, job.camera_label, job.first_frame, job.last_frame,
                 job.pose_model, self._t_job_start)
        self.job_started.emit(job.job_id)

        self._worker = PoseWorker(job, self._db_path)
        self._worker.progress.connect(self.progress)
        self._worker.tracking_done.connect(self._on_worker_finished)
        self._worker.error.connect(
            lambda msg, jid=job.job_id: self._on_worker_error(jid, msg)
        )
        self._worker.start()

    def _on_batch_ready(self, svid: str, batch: list) -> None:
        self._masks_this_job += len(batch)
        self._batch_count += 1
        t = time.monotonic() - self._t_job_start
        if self._batch_count == 1:
            log.info("JobQueueRunner: first batch received  masks=%d  t=%.3f", len(batch), t)
        elif self._batch_count % 10 == 0:
            log.debug("JobQueueRunner: batch %d received  total_masks=%d  t=%.3f",
                      self._batch_count, self._masks_this_job, t)
        self.mask_ready.emit(svid, batch)

    def _on_worker_finished(self) -> None:
        t = time.monotonic() - self._t_job_start
        log.info("JobQueueRunner: _on_worker_finished  batches=%d  masks=%d  t=%.3f",
                 self._batch_count, self._masks_this_job, t)
        job = self._current
        if job is not None and job.status == "running":
            job.status = "done"
            # For PoseWorker, get the count from the worker itself (it writes to DB
            # directly and doesn't use batch signals).  For CutieWorker, _masks_this_job
            # is populated by _on_batch_ready.
            if hasattr(self._worker, "get_keypoints_written"):
                count = self._worker.get_keypoints_written() if self._worker else 0
                job.keypoints_written = count
            else:
                count = self._masks_this_job
                job.masks_written = count
            self.job_finished.emit(job.job_id, count)

        # Cache the loaded Cutie model so subsequent Cutie workers skip Hydra re-init.
        # PoseWorker has no get_loaded_model(); only CutieWorker does.
        if self._worker is not None and hasattr(self._worker, "get_loaded_model"):
            loaded = self._worker.get_loaded_model()
            if loaded is not None:
                self._cutie_model = loaded
            # Wait for the OS thread to fully exit before releasing the worker.
            # Dropping the Python reference before the thread exits can cause
            # a SIGABRT in native (CUDA/Hydra) cleanup code.
            self._worker.wait()

        self._worker  = None
        self._current = None
        self._start_next()

    def _on_worker_error(self, job_id: str, error: str) -> None:
        for job in self._jobs:
            if job.job_id == job_id:
                job.status = "failed"
                job.error  = error
                self.job_failed.emit(job_id, error)
                break
        # CutieWorker.run() emits error then finished, so _on_worker_finished
        # will fire next and call _start_next().
