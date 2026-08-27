# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""pose_worker.py — Segmentation-driven pose extraction as a queued job.

Reads Cutie seg masks from the DB per frame, derives per-person tight
bounding boxes, runs RTMPose or VITpose via rtmlib, and writes results
directly to the DB via DetectionBatchWriter — the same schema used by
the YOLO-based pipeline.

The worker opens its own SQLite connection so DB writes happen in the
worker thread, avoiding the main-thread signal queue problems entirely.
Only progress and tracking_done signals cross the thread boundary.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from posetrak.detection.mask_treatment import TEMPORAL_WINDOW_RADIUS

log = logging.getLogger(__name__)


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
    apply_mask_treatment: bool = True  # suppress other people in each person's own
                                        # crop before estimation -- see mask_treatment.py
                                        # and docs/roadmap/features/segmentation-pose-treatment/.
                                        # Defaults on: validated by the study that motivated
                                        # it, and this path only ever runs against a
                                        # human-curated mask in the first place. Left as a
                                        # real field (not hardcoded) since whether to expose
                                        # a UI toggle is still an open question in that doc.
    temporal_mask_smoothing: bool = False  # smooth the treatment boundary over a small
                                        # frame window instead of deriving it from a single
                                        # frame's mask -- mask_treatment.suppress_others_temporal().
                                        # 2026-08-27: a real tracking run showed
                                        # apply_mask_treatment measurably increases jerkiness
                                        # in the target's own hand-joint angles during grabs
                                        # (not other arm joints -- isolated by re-tracking the
                                        # same pose data under the baseline's own tracker
                                        # config, ruling out a config confound). Working
                                        # hypothesis: single-frame mask-boundary jitter feeds
                                        # a different "context edit" to the pose model every
                                        # frame; this is the mitigation being validated.
                                        # Defaults off pending that validation -- see the
                                        # design doc's open question 0.
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
        self._stop_requested = False
        self._keypoints_written = 0

    def get_keypoints_written(self) -> int:
        return self._keypoints_written

    def stop(self) -> None:
        self._stop_requested = True

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

    def _run_hand_refinement(self, conn: sqlite3.Connection, job: "PoseExtractionJob") -> None:
        """Patch refined hand keypoints into the just-written detection_keypoints.

        Same mechanism the YOLO-based pipeline uses (posetrak.detection.
        hand_refinement) — mask-derived bboxes give better crops for the
        body pass, but the hand-specific crop/gate is unaffected by bbox
        source, so it's worth applying here too. See
        docs/roadmap/features/hand-detection-refinement/hand-detection-refinement-design.md.
        """
        try:
            from posetrak.detection.hand_refinement import HandRefinementPipeline
        except ImportError:
            log.warning("PoseWorker: rtmlib unavailable, skipping hand refinement")
            return
        from posetrak.detection.pipeline import CameraInfo

        cam_row = conn.execute(
            "SELECT camera_instance_id FROM capture_videos WHERE id=?",
            (job.shot_video_id,),
        ).fetchone()
        cam = CameraInfo(
            shot_video_id=job.shot_video_id,
            camera_instance_id=cam_row["camera_instance_id"] if cam_row else job.shot_video_id,
            file_path=job.video_path,
            actual_fps=0.0,
            ref_frame=0,
            ref_timestamp_s=0.0,
        )

        def on_progress(done: int, total: int, cam_id: str) -> None:
            self.progress.emit(done, total)

        t0 = time.monotonic()
        n_refined = HandRefinementPipeline(conn).run(
            job.detection_run_id, cameras=[cam], on_progress=on_progress
        )
        log.info("PoseWorker: hand refinement done  %d refined  %.2fs",
                 n_refined, time.monotonic() - t0)

    def _run_pose(self) -> None:
        from app.pose.backends_rtmpose import RTMPoseEstimator
        from app.pose.db_cache import DetectionBatchWriter, mark_run_complete

        job = self._job
        t0 = time.monotonic()
        log.info("PoseWorker: start  %s  frames %d-%d  model=%s  t=%.3f",
                 job.camera_label, job.first_frame, job.last_frame, job.pose_model, t0)

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row

        try:
            if job.overwrite_range:
                _delete_range(conn, job)

            estimator = RTMPoseEstimator(job.pose_model, device=self._device)
            pose_input_w = estimator.input_size[1]   # input_size is (height, width)

            writer = DetectionBatchWriter(
                conn, job.detection_run_id, job.shot_video_id, pose_input_w,
            )

            total = job.last_frame - job.first_frame + 1
            done = 0
            frames_with_kp = 0

            cap = cv2.VideoCapture(job.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, job.first_frame)

            def _load_and_scale_mask(frame_idx: int, fh: int, fw: int) -> np.ndarray | None:
                # Stored at max_dim=1920p by CutieWorker; scale UP to native
                # video resolution so bboxes/keypoints come out at native
                # resolution, matching the camera calibration matrices the
                # C++ tracker uses (same as the YOLO pipeline).
                mask = _load_mask(conn, job.seg_quality_run_id, job.shot_video_id, frame_idx)
                if mask is not None and mask.shape != (fh, fw):
                    mask = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
                return mask

            def _handle_frame(
                frame_idx: int, frame_bgr: np.ndarray, treatment_mask: np.ndarray,
            ) -> None:
                nonlocal frames_with_kp
                detections = _bboxes_from_mask(treatment_mask, job.persons_ordered)
                if detections:
                    results = _estimate_per_person(
                        estimator, frame_bgr, treatment_mask, detections, job.apply_mask_treatment,
                    )
                    writer.add_frame(frame_idx, detections, results,
                                    job.pose_model, img=frame_bgr)
                    self._keypoints_written += len(results)
                    frames_with_kp += 1

            try:
                if not job.temporal_mask_smoothing:
                    for frame_idx in range(job.first_frame, job.last_frame + 1):
                        if self._stop_requested:
                            break
                        ret, frame_bgr = cap.read()
                        if not ret:
                            log.warning("PoseWorker: video read failed at frame %d", frame_idx)
                            break

                        fh, fw = frame_bgr.shape[:2]
                        mask = _load_and_scale_mask(frame_idx, fh, fw)
                        if mask is None:
                            done += 1
                            continue

                        _handle_frame(frame_idx, frame_bgr, mask)

                        done += 1
                        if done % 50 == 0:
                            self.progress.emit(done, total)
                else:
                    # Temporal smoothing needs a small lookahead (offline
                    # batch job, no live/causal constraint) -- buffer
                    # TEMPORAL_WINDOW_RADIUS frames of context on each side
                    # before processing the center frame. A deque holding
                    # exactly window_size (frame_idx, frame, mask) entries:
                    # each time it's freshly full, index RADIUS is exactly
                    # the frame with a complete window on both sides, so
                    # sliding the window one frame at a time visits every
                    # frame's "complete-context" moment exactly once. The
                    # first/last RADIUS frames of the whole range never get
                    # a turn at that position before the loop ends, so the
                    # tail flush below processes them explicitly with
                    # whatever (necessarily smaller) window is available --
                    # suppress_others_temporal() already shrinks gracefully
                    # when handed a short window.
                    from posetrak.detection.mask_treatment import suppress_others_temporal

                    window_size = 2 * TEMPORAL_WINDOW_RADIUS + 1
                    pending: deque[tuple[int, np.ndarray, np.ndarray | None]] = deque()

                    def _process_pending_index(i: int) -> None:
                        nonlocal done, frames_with_kp
                        frame_idx, frame_bgr, mask = pending[i]
                        if mask is not None:
                            lo = max(0, i - TEMPORAL_WINDOW_RADIUS)
                            hi = min(len(pending), i + TEMPORAL_WINDOW_RADIUS + 1)
                            masks_window = []
                            center_in_window = None
                            for j in range(lo, hi):
                                m = pending[j][2]
                                if m is None:
                                    continue
                                if j == i:
                                    center_in_window = len(masks_window)
                                masks_window.append(m)

                            detections = _bboxes_from_mask(mask, job.persons_ordered)
                            if detections:
                                results = []
                                for det in detections:
                                    treated = suppress_others_temporal(
                                        frame_bgr, masks_window, center_in_window, det.track_id,
                                    )
                                    results.extend(estimator.estimate(treated, [det]))
                                writer.add_frame(frame_idx, detections, results,
                                                job.pose_model, img=frame_bgr)
                                self._keypoints_written += len(results)
                                frames_with_kp += 1
                        done += 1
                        if done % 50 == 0:
                            self.progress.emit(done, total)

                    for frame_idx in range(job.first_frame, job.last_frame + 1):
                        if self._stop_requested:
                            break
                        ret, frame_bgr = cap.read()
                        if not ret:
                            log.warning("PoseWorker: video read failed at frame %d", frame_idx)
                            break
                        fh, fw = frame_bgr.shape[:2]
                        mask = _load_and_scale_mask(frame_idx, fh, fw)
                        pending.append((frame_idx, frame_bgr, mask))
                        if len(pending) > window_size:
                            pending.popleft()
                        if len(pending) == window_size:
                            _process_pending_index(TEMPORAL_WINDOW_RADIUS)

                    # Tail flush: if the whole range never filled the
                    # window (a very short clip), every frame is still
                    # unprocessed -- otherwise only the last RADIUS frames
                    # (indices RADIUS+1..end of the final buffer state) are.
                    tail_start = 0 if len(pending) < window_size else TEMPORAL_WINDOW_RADIUS + 1
                    for i in range(tail_start, len(pending)):
                        _process_pending_index(i)
            finally:
                cap.release()

            writer.finalise()

            if job.refine_hands and not self._stop_requested:
                self._run_hand_refinement(conn, job)

            # Recompute person_track spans from full person_detections — handles
            # partial re-runs where overwrite_range covered only part of the video.
            _update_track_spans(conn, job.detection_run_id, job.shot_video_id)
            mark_run_complete(conn, job.detection_run_id)

            log.info("PoseWorker: done  %d frames  %d kp_frames  %.2fs",
                     done, frames_with_kp, time.monotonic() - t0)

        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_mask(
    conn: sqlite3.Connection,
    seg_quality_run_id: str,
    shot_video_id: str,
    frame_idx: int,
) -> np.ndarray | None:
    """Return (H, W) uint8 labeled mask, or None if not stored."""
    row = conn.execute(
        "SELECT mask_blob FROM seg_masks "
        "WHERE seg_quality_run_id=? AND shot_video_id=? AND frame_idx=?",
        (seg_quality_run_id, shot_video_id, frame_idx),
    ).fetchone()
    if row is None:
        return None
    buf = np.frombuffer(bytes(row["mask_blob"]), dtype=np.uint8)
    mask = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def _bboxes_from_mask(
    mask: np.ndarray,
    persons_ordered: list[str],
) -> list:
    """Return PersonDetection list from a labeled mask.

    Uses tight pixel bboxes in centre-format (cx, cy, w, h).
    DetectionBatchWriter._encode_crop adds the 20% margin when storing
    person crop images, matching the YOLO-based pipeline exactly.
    """
    from app.pose.backends import PersonDetection

    detections = []
    for i, _name in enumerate(persons_ordered):
        label = i + 1
        ys, xs = np.where(mask == label)
        if len(xs) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4:
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        detections.append(PersonDetection(
            track_id=label,
            bbox=np.array([cx, cy, float(w), float(h)], dtype=np.float32),
            confidence=1.0,
        ))
    return detections


def _estimate_per_person(
    estimator,
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    detections: list,
    apply_mask_treatment: bool,
) -> list:
    """Run pose estimation once per detection instead of once per frame.

    Segmentation-driven estimation historically batched every person in a
    frame through one estimator.estimate(frame, detections) call. Applying
    a per-person mask treatment (suppress everyone but the target person,
    mask_treatment.py) means each person needs their own treated frame, so
    this can no longer be a single call across all detections -- checked
    against rtmlib's own source (RTMPose/ViTPose.__call__) before making
    this change: it already loops one ONNX inference per bbox internally
    regardless of how many bboxes arrive in one call, so this is not a
    throughput regression, just a restructuring of the same work.
    """
    if not apply_mask_treatment:
        return estimator.estimate(frame_bgr, detections)

    from posetrak.detection.mask_treatment import suppress_others

    results = []
    for det in detections:
        treated = suppress_others(frame_bgr, mask, det.track_id)
        results.extend(estimator.estimate(treated, [det]))
    return results


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


def _update_track_spans(
    conn: sqlite3.Connection,
    detection_run_id: str,
    shot_video_id: str,
) -> None:
    """Recompute person_track first/last spans from person_detections.

    DetectionBatchWriter.finalise() writes spans only for frames seen in
    the current job.  For partial re-runs this overwrites the full span
    with only the partial range.  This function corrects by querying all
    detections for the run+camera.
    """
    spans = conn.execute(
        "SELECT track_id, MIN(video_frame) AS first_f, MAX(video_frame) AS last_f "
        "FROM person_detections "
        "WHERE detection_run_id=? AND shot_video_id=? "
        "GROUP BY track_id",
        (detection_run_id, shot_video_id),
    ).fetchall()
    for row in spans:
        conn.execute(
            "UPDATE person_tracks "
            "SET first_frame=?, last_frame=? "
            "WHERE detection_run_id=? AND shot_video_id=? AND track_id=?",
            (row["first_f"], row["last_f"], detection_run_id, shot_video_id, row["track_id"]),
        )
    conn.commit()
