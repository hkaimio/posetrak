"""hand_redetect_worker.py — Idea 3: automated hand redetection after a
manual edit, in the interactive keypoint editor.

Two request shapes, matching two access patterns already used elsewhere in
this codebase for the same underlying reason (seeking per-frame in
compressed video is expensive):

- `request_frame()`: a single (camera, frame) -- e.g. after a debounced
  post-edit trigger. Decoded with one direct random seek, the same shape as
  `CropBackfillWorker`'s single-frame priority path (`content_panels.py`).
- `request_range()`: many frames across one contiguous span -- e.g. after an
  interpolation fill. Decoded with one sequential walk over the span, the
  same shape as `WideCropExtractWorker`'s epoch walk (`wide_crop_cache.py`).

Both call the same crop/candidate-selection/gate function the batch pipeline
already uses (`posetrak.detection.hand_refinement.detect_hand_in_crop`), and
on a pass, write the result via `app.pose.db_cache.write_hand_refinement`
(`source='hand_l.refined'`/`'hand_r.refined'`).

Per this codebase's established testing convention (see
`test_wide_crop_cache.py`'s own docstring), the QThread's decode/queue
mechanics are validated manually, not unit-tested; `redetect_hand` below is
pulled out as a small, pure function specifically so it *can* be unit-tested
directly, the same way `HandRefinementPipeline._refine_one` is tested in
`test_hand_refinement.py`.

See docs/roadmap/features/hand-detection-refinement/hand-detection-refinement-design.md,
"Idea 3" section.
"""
from __future__ import annotations

import collections
import logging
import sqlite3
import threading

import numpy as np
from PySide6.QtCore import QThread, Signal

_log = logging.getLogger(__name__)

try:
    # Importing backends_rtmpose first runs its Windows onnxruntime-gpu
    # DLL-path setup, which rtmlib.Hand also depends on internally -- same
    # reason posetrak.detection.hand_refinement does this.
    import posetrak.detection.backends_rtmpose as _rtmpose_backend  # noqa: F401
    from rtmlib import Hand as _Hand
    _RTMLIB_AVAILABLE = True
except ImportError:
    _RTMLIB_AVAILABLE = False


def redetect_hand(
    hand_model,
    image: np.ndarray,
    wrist: tuple[float, float],
    elbow: tuple[float, float] | None,
) -> tuple[np.ndarray, float] | None:
    """Detect+gate one hand in *image*, returning (hand_kp[21,3], noise_scale)
    ready for `write_hand_refinement`, or None if no candidate passed the gate.

    Thin wrapper around `detect_hand_in_crop` -- kept separate from any
    video/DB/Qt machinery so it's directly unit-testable with a fake
    *hand_model* and a synthetic *image*, mirroring
    `HandRefinementPipeline._refine_one`'s same split in
    `posetrak.detection.hand_refinement`.
    """
    from posetrak.detection.hand_refinement import (
        _HAND_CONF_SCALE,
        _HAND_N_KP,
        _HAND_POSE_INPUT_WIDTH,
        detect_hand_in_crop,
    )

    result = detect_hand_in_crop(hand_model, image, wrist, elbow)
    if result is None:
        return None
    hand_kp = np.empty((_HAND_N_KP, 3), dtype=np.float32)
    hand_kp[:, 0] = result.keypoints[:, 0]
    hand_kp[:, 1] = result.keypoints[:, 1]
    hand_kp[:, 2] = result.scores * _HAND_CONF_SCALE
    noise_scale = result.crop_w_px / _HAND_POSE_INPUT_WIDTH
    return hand_kp, noise_scale


class HandRedetectWorker(QThread):
    """Background worker: redetect a hand on request, one frame or a range.

    Uses its own SQLite connection, like every other background worker in
    this codebase, so it never contends with the main thread's reads.
    Scoped to one (sequence, person) -- created/torn down alongside edit
    mode, the same lifecycle as `CropBackfillWorker`/`FrameCropCacheManager`
    (see `content_panels.py`'s `_set_edit_mode`).
    """

    result_ready = Signal(str, int)  # (camera_instance_id, video_frame)

    def __init__(
        self,
        db_path: str,
        sequence_id: str,
        person_id: int,
        cameras: list[dict],  # [{shot_video_id, camera_instance_id, file_path}, ...]
        parent=None,
    ) -> None:
        super().__init__(parent)
        if not _RTMLIB_AVAILABLE:
            raise ImportError(
                "rtmlib is required for HandRedetectWorker. "
                "Install from the rtmlib repository."
            )
        self._db_path = db_path
        self._sequence_id = sequence_id
        self._person_id = person_id
        self._file_path_by_svid = {c["shot_video_id"]: c["file_path"] for c in cameras}
        self._cam_id_by_svid = {c["shot_video_id"]: c["camera_instance_id"] for c in cameras}
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._queue: collections.deque = collections.deque()
        self._hand_model = None

    def stop(self) -> None:
        """Signal the worker to stop and block until it actually has.

        Never return early -- see `CropBackfillWorker.stop()`/
        `WideCropExtractWorker.stop()` for why dropping the last reference to
        a QThread while it's still running is undefined behaviour in Qt.
        """
        with self._cv:
            self._stop_event.set()
            self._cv.notify_all()
        if not self.wait(3000):
            _log.warning("hand-redetect worker: still running 3s after stop() -- waiting")
            self.wait()

    def request_frame(
        self,
        svid: str,
        frame_idx: int,
        timestamp_s: float,
        side: str,
        wrist: tuple[float, float],
        elbow: tuple[float, float] | None,
    ) -> None:
        """Queue a single-frame redetect request (the debounced post-edit case)."""
        _log.info(
            "hand-redetect: queued single-frame request  svid=%s frame=%d side=%s",
            svid, frame_idx, side,
        )
        with self._cv:
            self._queue.append(("frame", svid, frame_idx, timestamp_s, side, wrist, elbow))
            self._cv.notify_all()

    def request_range(
        self,
        svid: str,
        side: str,
        anchor_by_frame: dict[int, tuple[float, tuple[float, float], tuple[float, float] | None]],
    ) -> None:
        """Queue a range redetect request (the interpolation-fill case).

        *anchor_by_frame* maps each video_frame in the range to
        (timestamp_s, wrist, elbow). Decoded as one sequential walk over
        `[min(frames), max(frames)]`, not a seek per frame.
        """
        if not anchor_by_frame:
            _log.info(
                "hand-redetect: range request for svid=%s side=%s has no usable anchors, skipping",
                svid, side,
            )
            return
        _log.info(
            "hand-redetect: queued range request  svid=%s side=%s frames=[%d..%d] (%d frames)",
            svid, side, min(anchor_by_frame), max(anchor_by_frame), len(anchor_by_frame),
        )
        with self._cv:
            self._queue.append(("range", svid, side, dict(anchor_by_frame)))
            self._cv.notify_all()

    def _get_hand_model(self):
        if self._hand_model is None:
            device = _rtmpose_backend._auto_device()
            self._hand_model = _Hand(to_openpose=False, backend="onnxruntime", device=device)
        return self._hand_model

    def run(self) -> None:  # noqa: C901
        import cv2

        from app.pose.db_cache import clear_disabled_hand_edits, write_hand_refinement

        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row  # clear_disabled_hand_edits needs dict-style access
        except Exception:
            _log.exception("hand-redetect worker: failed to open DB")
            return

        caps: dict[str, object] = {}  # None sentinel = failed to open, don't retry

        def get_cap(svid: str):
            if svid not in caps:
                cap = cv2.VideoCapture(self._file_path_by_svid.get(svid, ""))
                caps[svid] = cap if cap.isOpened() else None
                if caps[svid] is None:
                    _log.warning("hand-redetect worker: cannot open video svid=%s", svid)
            return caps[svid]

        def process_and_write(
            svid: str, frame_idx: int, timestamp_s: float, side: str, bgr,
            wrist: tuple[float, float], elbow: tuple[float, float] | None,
        ) -> None:
            # A bug in here must not kill the whole worker thread -- it did
            # exactly that during real-session testing (an uncaught
            # exception propagated out of the while-loop body, silently
            # ending run() after the very first request; every request
            # after that was queued but never processed again, with no
            # crash and no further log output). One bad request should log
            # and move on, not take the rest of the session down with it.
            try:
                _process_and_write_unsafe(
                    svid, frame_idx, timestamp_s, side, bgr, wrist, elbow,
                )
            except Exception:
                _log.exception(
                    "hand-redetect: unhandled error processing request  cam=%s frame=%d side=%s",
                    self._cam_id_by_svid.get(svid), frame_idx, side,
                )

        def _process_and_write_unsafe(
            svid: str, frame_idx: int, timestamp_s: float, side: str, bgr,
            wrist: tuple[float, float], elbow: tuple[float, float] | None,
        ) -> None:
            cam_id = self._cam_id_by_svid.get(svid)
            if cam_id is None:
                _log.warning(
                    "hand-redetect: no camera_instance_id for svid=%s -- check cameras list", svid,
                )
                return
            _log.info(
                "hand-redetect: processing  cam=%s frame=%d side=%s wrist=(%.1f,%.1f) elbow=%s",
                cam_id, frame_idx, side, wrist[0], wrist[1], elbow,
            )
            result = redetect_hand(self._get_hand_model(), bgr, wrist, elbow)
            if result is None:
                _log.info(
                    "hand-redetect: REJECTED (no candidate passed the gate)  cam=%s frame=%d side=%s",
                    cam_id, frame_idx, side,
                )
                return
            hand_kp, noise_scale = result
            write_hand_refinement(
                conn, self._sequence_id, cam_id, frame_idx, self._person_id,
                timestamp_s=timestamp_s, side=side, kp=hand_kp, noise_scale=noise_scale,
            )
            # "Auto-detect" mode's whole premise: a fresh redetection
            # supersedes stale disable-edits for this hand (never a
            # deliberate repositioning edit) -- see
            # clear_disabled_hand_edits' docstring. Requests only ever
            # reach this worker when the editor's "auto-detect" toggle is
            # on (see content_panels.py's _maybe_queue_hand_redetect /
            # _queue_hand_redetect_range), so this runs unconditionally
            # here -- the toggle is what gates it, not this call site.
            clear_disabled_hand_edits(conn, self._sequence_id, cam_id, frame_idx, side)
            _log.info(
                "hand-redetect: WROTE hand_%s.refined  cam=%s frame=%d  mean_conf=%.2f  noise_scale=%.3f",
                "l" if side == "left" else "r", cam_id, frame_idx,
                float(hand_kp[:, 2].mean()), noise_scale,
            )
            self.result_ready.emit(cam_id, frame_idx)

        def process_task(task) -> None:
            if task[0] == "frame":
                _, svid, frame_idx, timestamp_s, side, wrist, elbow = task
                cap = get_cap(svid)
                if cap is None:
                    return
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, bgr = cap.read()
                if not ok:
                    _log.warning(
                        "hand-redetect: frame read failed  svid=%s frame=%d", svid, frame_idx,
                    )
                    return
                actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if actual != frame_idx:
                    # cv2's frame-index seeking is known to drift on
                    # inter-frame-coded video -- surface it rather than
                    # silently cropping around the wrong image.
                    _log.warning(
                        "hand-redetect: seek drift  requested frame=%d  actually decoded=%d"
                        "  svid=%s -- gate may reject due to wrong crop",
                        frame_idx, actual, svid,
                    )
                process_and_write(svid, frame_idx, timestamp_s, side, bgr, wrist, elbow)

            else:  # "range"
                _, svid, side, anchor_by_frame = task
                cap = get_cap(svid)
                if cap is None:
                    return
                frames = sorted(anchor_by_frame)
                frame_set = set(frames)
                last_frame = frames[-1]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
                while True:
                    if self._stop_event.is_set():
                        break
                    ok, bgr = cap.read()
                    if not ok:
                        _log.warning(
                            "hand-redetect: range read failed, stopping early  svid=%s"
                            " last_requested=%d", svid, last_frame,
                        )
                        break
                    # POS_FRAMES after read() reports the index of the
                    # *next* frame to decode -- one past what we just got.
                    # Trust this over a manually incremented counter: cv2's
                    # seek is known to drift on inter-frame-coded video,
                    # and a naive counter would silently process an anchor
                    # against the wrong image once that happens (this used
                    # to be exactly such a counter -- fixed after
                    # real-session testing showed the range path failing
                    # more than the single-frame path).
                    cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    if cur in frame_set:
                        timestamp_s, wrist, elbow = anchor_by_frame[cur]
                        process_and_write(svid, cur, timestamp_s, side, bgr, wrist, elbow)
                    if cur >= last_frame:
                        break

        try:
            while True:
                with self._cv:
                    while not self._queue and not self._stop_event.is_set():
                        self._cv.wait(0.5)
                    if not self._queue and self._stop_event.is_set():
                        return
                    task = self._queue.popleft()

                try:
                    process_task(task)
                except Exception:
                    # Belt-and-braces on top of process_and_write's own
                    # try/except: anything unexpected here (video I/O,
                    # dequeue bookkeeping) must not silently end run() --
                    # see process_and_write's comment for why that's a real
                    # failure mode this worker already hit once.
                    _log.exception("hand-redetect: unhandled error processing task %r", task[0])
        finally:
            for cap in caps.values():
                if cap is not None:
                    cap.release()
            conn.close()
