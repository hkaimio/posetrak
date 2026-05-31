"""frame_cache.py — LRU video frame cache backed by temp JPEG files.

Decodes video frames on demand via OpenCV, caches them as JPEG files in a
temp directory, and evicts the least-recently-used entries when the cache
exceeds its size limit.  Thread-safe; intended for use from the UI thread
with an optional background pre-fetch thread (Phase 3).
"""
from __future__ import annotations

import os
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


_DEFAULT_MAX_FRAMES = 300
_JPEG_QUALITY = 85


class FrameCache:
    """Disk-backed LRU cache of decoded video frames.

    Parameters
    ----------
    max_frames:
        Maximum number of frames to keep on disk before evicting LRU entries.
    max_dim:
        If the video is larger than this in either dimension, frames are
        scaled down to fit.  0 means no scaling.
    """

    def __init__(self, max_frames: int = _DEFAULT_MAX_FRAMES, max_dim: int = 1920) -> None:
        self._max_frames = max_frames
        self._max_dim = max_dim
        self._lock = threading.Lock()
        self._tmp_dir = tempfile.mkdtemp(prefix="posetrak_frames_")
        # OrderedDict: insertion order == LRU order; popitem(last=False) removes oldest.
        self._lru: OrderedDict[tuple[str, int], Path] = OrderedDict()
        # Cache of open VideoCapture objects keyed by video path.
        self._caps: dict[str, cv2.VideoCapture] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self, video_path: str, frame_idx: int) -> np.ndarray | None:
        """Return BGR frame at *frame_idx*, or None if the video can't be read."""
        key = (video_path, frame_idx)
        with self._lock:
            cached_path = self._lru.get(key)
            if cached_path is not None:
                # Move to end (most recently used).
                self._lru.pop(key)
                self._lru[key] = cached_path
                return cv2.imdecode(
                    np.frombuffer(cached_path.read_bytes(), dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
        # Decode outside the lock to avoid blocking UI during seek.
        frame = self._decode(video_path, frame_idx)
        if frame is None:
            return None
        with self._lock:
            self._store(key, frame)
        return frame

    def invalidate_video(self, video_path: str) -> None:
        """Remove all cached frames for *video_path* (e.g. after file change)."""
        with self._lock:
            to_remove = [k for k in self._lru if k[0] == video_path]
            for k in to_remove:
                path = self._lru.pop(k)
                path.unlink(missing_ok=True)
            cap = self._caps.pop(video_path, None)
            if cap is not None:
                cap.release()

    def close(self) -> None:
        """Release all VideoCapture handles and delete the temp directory."""
        with self._lock:
            for cap in self._caps.values():
                cap.release()
            self._caps.clear()
            self._lru.clear()
        import shutil
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _decode(self, video_path: str, frame_idx: int) -> np.ndarray | None:
        with self._lock:
            cap = self._caps.get(video_path)
            if cap is None:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return None
                self._caps[video_path] = cap
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        if self._max_dim > 0:
            h, w = frame.shape[:2]
            if max(h, w) > self._max_dim:
                scale = self._max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        return frame

    def _store(self, key: tuple[str, int], frame: np.ndarray) -> None:
        """Write *frame* to a temp JPEG file and register in LRU (lock held)."""
        # Evict LRU if at capacity.
        while len(self._lru) >= self._max_frames:
            _, oldest_path = self._lru.popitem(last=False)
            oldest_path.unlink(missing_ok=True)

        tmp_path = Path(self._tmp_dir) / f"{abs(hash(key))}.jpg"
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        if ok:
            tmp_path.write_bytes(buf.tobytes())
        self._lru[key] = tmp_path

    def get_frame_count(self, video_path: str) -> int:
        """Return the total frame count for *video_path*, or 0 on error."""
        with self._lock:
            cap = self._caps.get(video_path)
            if cap is None:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return 0
                self._caps[video_path] = cap
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_fps(self, video_path: str) -> float:
        """Return the FPS of *video_path*, or 30.0 on error."""
        with self._lock:
            cap = self._caps.get(video_path)
            if cap is None:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return 30.0
                self._caps[video_path] = cap
            fps = cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else 30.0
