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
            return
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

        from app.pose.db_cache import write_hand_refinement

        try:
            conn = sqlite3.connect(self._db_path)
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
            cam_id = self._cam_id_by_svid.get(svid)
            if cam_id is None:
                return
            result = redetect_hand(self._get_hand_model(), bgr, wrist, elbow)
            if result is None:
                return
            hand_kp, noise_scale = result
            write_hand_refinement(
                conn, self._sequence_id, cam_id, frame_idx, self._person_id,
                timestamp_s=timestamp_s, side=side, kp=hand_kp, noise_scale=noise_scale,
            )
            self.result_ready.emit(cam_id, frame_idx)

        try:
            while True:
                with self._cv:
                    while not self._queue and not self._stop_event.is_set():
                        self._cv.wait(0.5)
                    if not self._queue and self._stop_event.is_set():
                        return
                    task = self._queue.popleft()

                if task[0] == "frame":
                    _, svid, frame_idx, timestamp_s, side, wrist, elbow = task
                    cap = get_cap(svid)
                    if cap is None:
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, bgr = cap.read()
                    if not ok:
                        continue
                    process_and_write(svid, frame_idx, timestamp_s, side, bgr, wrist, elbow)

                else:  # "range"
                    _, svid, side, anchor_by_frame = task
                    cap = get_cap(svid)
                    if cap is None:
                        continue
                    frames = sorted(anchor_by_frame)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
                    cur = frames[0]
                    remaining = collections.deque(frames)
                    while remaining:
                        if self._stop_event.is_set():
                            break
                        ok, bgr = cap.read()
                        if not ok:
                            break
                        if cur == remaining[0]:
                            remaining.popleft()
                            timestamp_s, wrist, elbow = anchor_by_frame[cur]
                            process_and_write(svid, cur, timestamp_s, side, bgr, wrist, elbow)
                        cur += 1
        finally:
            for cap in caps.values():
                if cap is not None:
                    cap.release()
            conn.close()
