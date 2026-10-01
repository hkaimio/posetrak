# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Persistent per-camera video decoders with read-ahead, for previewing frames.

See docs/roadmap/features/preview-caching/preview-caching-design.md.

One :class:`FrameDecoder` per video file runs on its own thread, keeps the
container open, and answers "give me frame N" with the latest request winning.
It decodes with PyAV (the decoder detection uses, so a frame index means the
same frame here as in a detection or tracking result) and keeps display-size
JPEGs of recent frames in a byte-budgeted cache:

- Forward stepping and playback reuse the open decoder, so each new frame is
  one sequential decode instead of a seek.
- A random seek must decode from the previous keyframe, so on a jump the frames
  between that keyframe and the target are decoded without colour conversion
  and only the ones we keep are converted and encoded.
- Backward stepping decodes a whole keyframe interval at once and keeps it, so
  stepping back through it costs nothing after the first step.
- Prefetch starts only after the requests have settled, and is abandoned the
  moment a new request arrives.

A whole-frame result is limited to DISPLAY_LONG_EDGE, which is not enough when
the user zooms in. :meth:`FrameDecoder.request_region` therefore also produces,
for the frame on screen, a crop of a requested rectangle at up to the video's
native resolution (the full frame is kept from the exact decode, or decoded
again). It is an extra result for that one frame; the whole-frame entry stays
so the old image can be shown until the crop arrives.

Results use the same tuple as the person-crop layers, ``(jpeg, width_px,
height_px, src_x, src_y, src_w, src_h)``, with the source rectangle covering
the whole decoded frame, so the existing display and overlay code can show
them unchanged.

A decoder never blocks its owner on shutdown: the decoder thread closes its own
container, and the owner waits only briefly and then abandons a thread that
does not finish (threaded decoding is documented to hang on close in
``posetrak/detection/frame_source.py``).
"""

from __future__ import annotations

import collections
import logging
import threading
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal

_log = logging.getLogger(__name__)

Result = tuple  # (jpeg, width_px, height_px, src_x, src_y, src_w, src_h)

DISPLAY_LONG_EDGE = 1920
JPEG_QUALITY = 85
DEFAULT_AHEAD = 60          # frames decoded ahead of the current one
STEP_AHEAD = 8              # frames decoded ahead at once while stepping forward, without settling
BACK_WANTED = 60            # frames kept behind the current one when stepping backwards
BACK_KEEP = 15              # frames kept behind the current one otherwise
MAX_SKIP_FORWARD = 90       # continue decoding rather than seek for a target this far ahead
SETTLE_SECONDS = 0.15
DEFAULT_BUDGET_BYTES = 256 * 1024 * 1024
_STEP_THRESHOLD = 3         # a request this close to the previous one counts as stepping


class FrameDecoder:
    """Decodes one video file on a background thread. See the module docstring."""

    def __init__(
        self,
        file_path: str,
        on_ready: Callable[[int], None],
        *,
        budget_bytes: int = DEFAULT_BUDGET_BYTES,
        ahead: int = DEFAULT_AHEAD,
        settle_s: float = SETTLE_SECONDS,
        thread_type: str = "AUTO",
        on_failed: Callable[[], None] | None = None,
        on_missing: Callable[[int], None] | None = None,
    ) -> None:
        self._file_path = str(file_path)
        self._on_ready = on_ready
        self._on_failed = on_failed
        self._on_missing = on_missing
        self._budget = budget_bytes
        self._ahead = ahead
        self._settle = settle_s
        self._thread_type = thread_type

        self._cond = threading.Condition()
        self._target: int | None = None
        self._serial = 0           # bumped by every request(); the worker sleeps until it changes
        self._owes_ready = False   # the current target was not cached when it was requested
        self._last_target: int | None = None
        self._direction = 0
        self._stop = False
        self._cache: collections.OrderedDict[int, Result] = collections.OrderedDict()
        self._cache_bytes = 0
        self._missing: set[int] = set()   # requested frames the video does not have
        self._region_req: tuple | None = None   # (frame_idx, rect, scale) not yet produced
        self._region: tuple[int, Result] | None = None   # (frame_idx, crop) last produced
        self.failed = False

        # Decoder state, touched only by the worker thread.
        self._container = None
        self._stream = None
        self._gen = None
        self._pos = -1
        self._fps = 0.0
        self._tb = 0.0
        self._rotation = 0
        self._held: tuple[int, object] | None = None   # (frame_idx, full-resolution av frame)

        self._thread = threading.Thread(
            target=self._run, name=f"frame-decoder-{self._file_path[-24:]}", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API (any thread)
    # ------------------------------------------------------------------

    def request(self, frame_idx: int) -> bool:
        """Make *frame_idx* the frame to show; supersedes any earlier request.
        Returns True if it is already cached (the caller can display it now; no
        ready notification will follow for a frame that was cached when asked)."""
        with self._cond:
            if frame_idx == self._target:
                # Already the frame being shown: nothing to do. Re-requesting
                # it must not wake the worker, or every redraw of the same
                # frame would ask for it again and again.
                return frame_idx in self._cache
            if self._last_target is not None:
                diff = frame_idx - self._last_target
                if diff != 0:
                    self._direction = (1 if diff > 0 else -1) if abs(diff) <= _STEP_THRESHOLD else 0
            self._last_target = frame_idx
            self._target = frame_idx
            self._region_req = None
            self._owes_ready = frame_idx not in self._cache
            self._serial += 1
            self._cond.notify_all()
            return not self._owes_ready

    def get(self, frame_idx: int) -> Result | None:
        """The cached result for *frame_idx*, or None if it is not decoded yet."""
        with self._cond:
            return self._cache.get(frame_idx)

    def request_region(self, frame_idx: int, rect: tuple, scale: float) -> None:
        """Ask for a crop of *rect* (x0, y0, x1, y1 in frame pixels) of the
        current target frame, at *scale* output pixels per frame pixel (at most
        1). Ignored if *frame_idx* is no longer the target. The result is
        available from :meth:`get_region` and announced through on_ready."""
        with self._cond:
            if frame_idx != self._target:
                return
            self._region_req = (frame_idx, tuple(rect), min(1.0, float(scale)))
            self._serial += 1
            self._cond.notify_all()

    def get_region(self, frame_idx: int) -> Result | None:
        """The last produced crop if it is for *frame_idx*. Its source rectangle
        (the last four tuple items) says which part of the frame it covers."""
        with self._cond:
            if self._region is not None and self._region[0] == frame_idx:
                return self._region[1]
            return None

    def is_missing(self, frame_idx: int) -> bool:
        """True if *frame_idx* was requested and the video has no such frame
        (for example a sync mapping that runs past the end of a shorter clip)."""
        with self._cond:
            return frame_idx in self._missing

    def cached_frames(self) -> list[int]:
        """Sorted indices of the frames currently cached (for diagnostics/tests)."""
        with self._cond:
            return sorted(self._cache)

    def close(self, timeout: float = 2.0) -> bool:
        """Stop the thread. Returns False if it did not finish within *timeout*
        (it is then abandoned; it is a daemon thread and holds no lock)."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _superseded(self, target: int) -> bool:
        return self._stop or self._target != target

    def _run(self) -> None:
        try:
            if not self._open():
                self.failed = True
                if self._on_failed is not None:
                    self._on_failed()
                return
            handled = -1
            while True:
                with self._cond:
                    self._cond.wait_for(lambda: self._stop or self._serial != handled)
                    if self._stop:
                        return
                    target, direction, handled = self._target, self._direction, self._serial
                    owes_ready = self._owes_ready
                    self._owes_ready = False
                assert target is not None
                if self.get(target) is None:
                    owes_ready = True
                    keep_from = 0 if direction < 0 else target - BACK_KEEP
                    if not self._decode_to(target, keep_from):
                        if not self._superseded(target):
                            # Not superseded, so the video ended before it.
                            with self._cond:
                                self._missing.add(target)
                            if self._on_missing is not None:
                                self._on_missing(target)
                        continue  # sleep until the next request
                if owes_ready:
                    self._on_ready(target)
                if self._make_region(target):
                    self._on_ready(target)
                if direction > 0:
                    # Stepping forward: a few frames ahead cost almost nothing
                    # (the decoder is already there), so do them at once and the
                    # next steps are cache hits even if the user never pauses.
                    self._prefetch_forward(target, STEP_AHEAD)
                # The full read-ahead waits until requests have settled; a new
                # request cancels it.
                with self._cond:
                    self._cond.wait_for(
                        lambda: self._stop or self._target != target, timeout=self._settle
                    )
                    if self._stop:
                        return
                    if self._target != target:
                        continue
                self._prefetch(target, direction)
        except Exception:  # noqa: BLE001
            _log.exception("frame decoder failed: %s", self._file_path)
            self.failed = True
        finally:
            self._close_container()

    def _open(self) -> bool:
        try:
            from posetrak.detection.proxy_cache import open_video

            self._container, self._stream, self._rotation = open_video(
                self._file_path, self._thread_type
            )
            self._tb = float(self._stream.time_base)
            self._fps = float(self._stream.average_rate)
            return True
        except Exception:  # noqa: BLE001
            _log.warning("frame decoder: cannot open %s", self._file_path, exc_info=True)
            return False

    def _close_container(self) -> None:
        try:
            self._gen = None
            if self._container is not None:
                self._container.close()
        except Exception:  # noqa: BLE001
            _log.debug("frame decoder: close failed", exc_info=True)

    def _seek(self, target: int) -> None:
        # Half a frame early so rounding cannot land after the target.
        seek_s = max(0.0, (target - 0.5) / self._fps)
        self._container.seek(
            int(seek_s / self._tb), stream=self._stream, backward=True, any_frame=False
        )
        self._gen = self._container.decode(self._stream)
        self._pos = -1

    def _next_frame(self):
        """(frame_idx, av_frame) of the next decoded frame, or None at the end."""
        while True:
            try:
                frame = next(self._gen)
            except StopIteration:
                self._gen = None
                return None
            if frame.pts is None:
                continue
            idx = round(float(frame.pts) * self._tb * self._fps)
            self._pos = idx
            return idx, frame

    def _decode_to(self, target: int, keep_from: int, owner: int | None = None) -> bool:
        """Decode until *target* is cached, caching frames with index >=
        *keep_from* on the way. *owner* is the frame the user asked for (the
        target itself unless this is prefetch); the decode is abandoned when a
        newer request replaces it. Returns False if abandoned or unreadable."""
        owner = target if owner is None else owner
        if self._gen is None or target <= self._pos or target - self._pos > MAX_SKIP_FORWARD:
            self._seek(target)
        count = 0
        while True:
            item = self._next_frame()
            if item is None:
                return False
            idx, frame = item
            if idx >= target and owner == target and self._held_idx() != target:
                self._held = (target, frame)   # for request_region
            if idx >= keep_from or idx >= target:
                self._store(idx, frame)
                if idx > target and self.get(target) is None:
                    # A missing frame (variable frame rate): show the next one.
                    with self._cond:
                        self._cache[target] = self._cache[idx]
            if idx >= target:
                return True
            count += 1
            if count % 16 == 0 and self._superseded(owner):
                return False

    def _held_idx(self) -> int | None:
        return self._held[0] if self._held is not None else None

    def _make_region(self, target: int) -> bool:
        """Produce the pending region crop for *target*; True if one was made."""
        import cv2
        import numpy as np

        from posetrak.detection.frame_source import _apply_rotation

        with self._cond:
            req, self._region_req = self._region_req, None
        if req is None or req[0] != target:
            return False
        if self._held_idx() != target and not self._decode_to(target, keep_from=target):
            return False
        if self._held_idx() != target:
            return False
        rect, scale = req[1], req[2]
        img = _apply_rotation(self._held[1].to_ndarray(format="bgr24"), self._rotation)
        h, w = img.shape[:2]
        x0 = int(np.clip(np.floor(rect[0]), 0, w - 1))
        y0 = int(np.clip(np.floor(rect[1]), 0, h - 1))
        x1 = int(np.clip(np.ceil(rect[2]), x0 + 1, w))
        y1 = int(np.clip(np.ceil(rect[3]), y0 + 1, h))
        crop = img[y0:y1, x0:x1]
        if scale < 1.0:
            crop = cv2.resize(
                crop,
                (max(1, int((x1 - x0) * scale)), max(1, int((y1 - y0) * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return False
        with self._cond:
            self._region = (target, (buf.tobytes(), crop.shape[1], crop.shape[0], x0, y0, x1 - x0, y1 - y0))
        return True

    def _store(self, idx: int, frame) -> None:
        if self.get(idx) is not None:
            return
        result = self._encode(frame)
        with self._cond:
            self._cache[idx] = result
            self._cache_bytes += len(result[0])
            self._evict()

    def _evict(self) -> None:
        """Drop the frames farthest from the current target until under budget.
        Called with the lock held."""
        centre = self._target if self._target is not None else 0
        while self._cache_bytes > self._budget and len(self._cache) > 1:
            far = max(self._cache, key=lambda k: abs(k - centre))
            self._cache_bytes -= len(self._cache.pop(far)[0])

    def _encode(self, frame) -> Result:
        import cv2

        from posetrak.detection.frame_source import _apply_rotation

        img = _apply_rotation(frame.to_ndarray(format="bgr24"), self._rotation)
        h, w = img.shape[:2]
        long_edge = max(h, w)
        if long_edge > DISPLAY_LONG_EDGE:
            scale = DISPLAY_LONG_EDGE / long_edge
            img = cv2.resize(
                img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
            )
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes(), img.shape[1], img.shape[0], 0, 0, w, h

    def _prefetch(self, target: int, direction: int) -> None:
        if direction < 0:
            self._prefetch_backward(target)
        else:
            self._prefetch_forward(target)

    def _prefetch_forward(self, target: int, count: int | None = None) -> None:
        limit = target + (self._ahead if count is None else count)
        if self._gen is None or self._pos < target or self._pos >= limit:
            return
        while self._pos < limit and not self._superseded(target):
            item = self._next_frame()
            if item is None:
                return
            self._store(*item)

    def _prefetch_backward(self, target: int) -> None:
        while not self._superseded(target):
            first = target
            with self._cond:
                while first - 1 in self._cache:
                    first -= 1
            if target - first >= BACK_WANTED or first <= 0:
                return
            if not self._decode_to(first - 1, keep_from=0, owner=target):
                return


class FrameService(QObject):
    """One :class:`FrameDecoder` per camera video, sharing a memory budget.

    ``frame_ready(shot_video_id, frame_idx)`` is emitted (from a decoder
    thread; Qt queues it to the receiver's thread) when a requested frame has
    been decoded.
    """

    frame_ready = Signal(str, int)
    unavailable = Signal(str)   # shot_video_id whose video could not be opened
    frame_missing = Signal(str, int)   # the video has no such frame (past its end)

    def __init__(
        self,
        cameras: list[dict],
        *,
        budget_bytes: int = DEFAULT_BUDGET_BYTES,
        parent: QObject | None = None,
        **decoder_kwargs,
    ) -> None:
        super().__init__(parent)
        per_decoder = max(1, budget_bytes // max(1, len(cameras)))
        self._decoders: dict[str, FrameDecoder] = {}
        for cam in cameras:
            svid = cam["shot_video_id"]
            self._decoders[svid] = FrameDecoder(
                cam["file_path"],
                on_ready=lambda idx, s=svid: self._emit(self.frame_ready, s, idx),
                on_failed=lambda s=svid: self._emit(self.unavailable, s),
                on_missing=lambda idx, s=svid: self._emit(self.frame_missing, s, idx),
                budget_bytes=per_decoder,
                **decoder_kwargs,
            )

    @staticmethod
    def _emit(signal, *args) -> None:
        """Emit from a decoder thread; the service may already be deleted while a
        decode that was in flight finishes."""
        try:
            signal.emit(*args)
        except RuntimeError:
            pass

    def request(self, shot_video_id: str, frame_idx: int) -> bool:
        """Ask for a frame; True if it is already cached."""
        decoder = self._decoders.get(shot_video_id)
        return decoder.request(frame_idx) if decoder is not None else False

    def get(self, shot_video_id: str, frame_idx: int) -> Result | None:
        decoder = self._decoders.get(shot_video_id)
        return decoder.get(frame_idx) if decoder is not None else None

    def request_region(self, shot_video_id: str, frame_idx: int, rect: tuple, scale: float) -> None:
        decoder = self._decoders.get(shot_video_id)
        if decoder is not None:
            decoder.request_region(frame_idx, rect, scale)

    def get_region(self, shot_video_id: str, frame_idx: int) -> Result | None:
        decoder = self._decoders.get(shot_video_id)
        return decoder.get_region(frame_idx) if decoder is not None else None

    def is_missing(self, shot_video_id: str, frame_idx: int) -> bool:
        decoder = self._decoders.get(shot_video_id)
        return decoder is not None and decoder.is_missing(frame_idx)

    def is_unavailable(self, shot_video_id: str) -> bool:
        decoder = self._decoders.get(shot_video_id)
        return decoder is None or decoder.failed

    def shutdown(self) -> None:
        for decoder in self._decoders.values():
            decoder.close(timeout=0.5)
        self._decoders.clear()
