"""FrameCache — central decoded-frame provider for the setup wizard UI.

All widgets that display video pixels go through this class.  Frames are
served from a three-level stack:

1. In-memory LRU (``_lru``, ~200 entries).
2. ``frame_cache_entries`` table in the session DB (compressed JPEG).
3. ``cv2.VideoCapture`` pool, with sequential-read optimisation.

Reads are synchronous (called from the Qt main thread).
DB writes are asynchronous: a daemon thread drains a ``queue.Queue`` so
the UI is never blocked by disk I/O.
"""

from __future__ import annotations

import enum
import queue
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Enums and key type
# ---------------------------------------------------------------------------


class CacheType(enum.Enum):
    FULL_FRAME  = "full_frame"   # full-resolution decoded frame
    THUMB       = "thumb"        # small thumbnail (e.g. 320 × 180)
    PERSON_CROP = "person_crop"  # tight crop for one detector track + region


@dataclass(frozen=True)
class CacheKey:
    """Identity for every cache entry.

    ``track_id`` and ``region_type`` are required for ``PERSON_CROP``.
    ``width_px`` and ``height_px`` are required for ``THUMB``.
    """
    shot_video_id: str
    frame_idx:     int
    cache_type:    CacheType
    track_id:      int | None = None
    region_type:   str | None = None
    width_px:      int | None = None
    height_px:     int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LRU_MAX = 200
_JPEG_QUALITY = 85


def _encode_jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def _decode_jpeg(data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("JPEG decode failed")
    return img


def _db_key(key: CacheKey) -> tuple:
    """Flatten a CacheKey to a tuple suitable for DB comparisons."""
    return (
        key.shot_video_id,
        key.frame_idx,
        key.cache_type.value,
        key.track_id if key.track_id is not None else -1,
        key.region_type if key.region_type is not None else "",
        key.width_px if key.width_px is not None else 0,
    )


# ---------------------------------------------------------------------------
# FrameCache
# ---------------------------------------------------------------------------


class FrameCache:
    """Three-level frame cache: LRU → DB → VideoCapture.

    Parameters
    ----------
    conn:
        Open connection to the session database (for ``frame_cache_entries``).
        Pass ``None`` to disable DB persistence (useful in tests).
    lru_max:
        Maximum number of frames to keep in the in-memory LRU cache.
    """

    def __init__(
        self,
        conn: sqlite3.Connection | None = None,
        lru_max: int = _LRU_MAX,
    ) -> None:
        self._conn = conn
        self._lru: OrderedDict[CacheKey, np.ndarray] = OrderedDict()
        self._lru_max = lru_max
        # VideoCapture pool
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._last_frame: dict[str, int] = {}
        # Async DB write queue
        self._write_q: queue.Queue[tuple | None] = queue.Queue()
        if conn is not None:
            self._writer = threading.Thread(
                target=self._write_worker, daemon=True, name="frame-cache-writer"
            )
            self._writer.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        key: CacheKey,
        *,
        file_path: str | None = None,
        undistort: bool = False,
    ) -> np.ndarray:
        """Return the image for *key*.

        Parameters
        ----------
        file_path:
            Path to the video file.  Required on a cache miss for
            ``FULL_FRAME``, ``THUMB``, and ``PERSON_CROP`` types.
        undistort:
            If ``True`` apply ``cv2.undistort`` before returning.
            Not yet implemented; flag is accepted for API compatibility.

        Raises
        ------
        ValueError
            If ``file_path`` is ``None`` on a cache miss.
        """
        # 1. In-memory LRU
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]

        # 2. DB persistence layer
        img = self._read_from_db(key)
        if img is not None:
            self._lru_put(key, img)
            return img

        # 3. Decode from video
        if file_path is None:
            raise ValueError(
                f"Cache miss for {key!r} and no file_path provided"
            )
        img = self._decode(key, file_path)

        self._lru_put(key, img)
        self._write_q.put((key, img))
        return img

    def open_video(self, shot_video_id: str, file_path: str) -> None:
        """Pre-open a VideoCapture so subsequent gets don't pay open latency."""
        if shot_video_id not in self._caps:
            cap = cv2.VideoCapture(file_path)
            self._caps[shot_video_id] = cap
            self._last_frame[shot_video_id] = -1

    def close_video(self, shot_video_id: str) -> None:
        """Release the VideoCapture for *shot_video_id*."""
        cap = self._caps.pop(shot_video_id, None)
        if cap is not None:
            cap.release()
        self._last_frame.pop(shot_video_id, None)

    def close_all(self) -> None:
        """Release all open VideoCaptures and stop the write worker."""
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        self._last_frame.clear()
        if self._conn is not None:
            self._write_q.put(None)  # sentinel — stop worker

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lru_put(self, key: CacheKey, img: np.ndarray) -> None:
        self._lru[key] = img
        self._lru.move_to_end(key)
        while len(self._lru) > self._lru_max:
            self._lru.popitem(last=False)

    def _read_from_db(self, key: CacheKey) -> np.ndarray | None:
        if self._conn is None:
            return None
        svid, fidx, ctype, tid, rtype, wpx = _db_key(key)
        row = self._conn.execute(
            "SELECT image_data FROM frame_cache_entries "
            "WHERE shot_video_id=? AND frame_idx=? AND cache_type=? "
            "AND track_id=? AND region_type=? AND width_px=?",
            (svid, fidx, ctype, tid, rtype, wpx),
        ).fetchone()
        if row is None:
            return None
        return _decode_jpeg(row[0])

    def _get_cap(self, shot_video_id: str, file_path: str) -> cv2.VideoCapture:
        if shot_video_id not in self._caps:
            cap = cv2.VideoCapture(file_path)
            self._caps[shot_video_id] = cap
            self._last_frame[shot_video_id] = -1
        return self._caps[shot_video_id]

    def _decode(self, key: CacheKey, file_path: str) -> np.ndarray:
        cap = self._get_cap(key.shot_video_id, file_path)
        last = self._last_frame.get(key.shot_video_id, -1)

        if key.frame_idx != last + 1:
            # Random seek required
            cap.set(cv2.CAP_PROP_POS_FRAMES, key.frame_idx)

        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(
                f"Failed to read frame {key.frame_idx} from {file_path!r}"
            )
        self._last_frame[key.shot_video_id] = key.frame_idx

        return self._post_process(frame, key)

    def _post_process(self, frame: np.ndarray, key: CacheKey) -> np.ndarray:
        if key.cache_type == CacheType.THUMB:
            w = key.width_px or 320
            h = key.height_px or 180
            return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        if key.cache_type == CacheType.PERSON_CROP:
            # Crop bounding box from person_detections — caller is responsible
            # for providing a key whose bbox is already known; for now return
            # the full frame as a placeholder (real impl queries the DB).
            return frame
        return frame  # FULL_FRAME — return as-is

    def _write_worker(self) -> None:
        """Daemon thread: drain the write queue and insert JPEG blobs into DB."""
        while True:
            item = self._write_q.get()
            if item is None:
                break
            key, img = item
            if self._conn is None:
                continue
            svid, fidx, ctype, tid, rtype, wpx = _db_key(key)
            hpx = key.height_px or 0
            try:
                jpeg = _encode_jpeg(img)
                self._conn.execute(
                    "INSERT OR REPLACE INTO frame_cache_entries "
                    "(shot_video_id, frame_idx, cache_type, track_id, "
                    "region_type, width_px, height_px, image_data) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (svid, fidx, ctype, tid, rtype, wpx, hpx, jpeg),
                )
                self._conn.commit()
            except Exception:  # noqa: BLE001
                pass  # best-effort persistence — never crash the UI thread
