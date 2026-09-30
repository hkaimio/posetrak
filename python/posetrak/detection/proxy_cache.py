# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Scrub proxy cache: small whole-frame JPEGs of every Nth video frame.

See docs/roadmap/features/preview-caching/preview-caching-design.md.

Seeking in a 4K video costs a keyframe decode, so dragging a time slider over
it stalls. The proxy holds a reduced image of every ``step``-th frame, made once
by one sequential pass over the video, and a lookup by frame index returns the
nearest one at once. It lives in a sidecar database beside the session
database (``<session>.cache.db``): it is disposable, safe to delete at any time,
and never part of the session data.

A frame index means the same frame here as in the frame service and in
detection: ``round(pts * time_base * fps)`` of the PyAV frame.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path

_log = logging.getLogger(__name__)

PROXY_WIDTH = 640
PROXY_STEP = 8
PROXY_JPEG_QUALITY = 70
_COMMIT_EVERY = 50

_SCHEMA = """
-- One reduced whole-frame image per stored frame of a capture video.
CREATE TABLE IF NOT EXISTS proxy_frames (
    -- capture_videos.id of the session database the frame belongs to.
    shot_video_id TEXT    NOT NULL,
    -- Frame index in the video (PyAV pts-based, same as detection results).
    frame_idx     INTEGER NOT NULL,
    -- Size of the stored JPEG in pixels.
    width_px      INTEGER NOT NULL,
    height_px     INTEGER NOT NULL,
    -- Size of the full frame the JPEG was reduced from, after display rotation;
    -- the coordinate system overlays are drawn in.
    src_w         INTEGER NOT NULL,
    src_h         INTEGER NOT NULL,
    -- JPEG bytes.
    jpeg          BLOB    NOT NULL,
    PRIMARY KEY (shot_video_id, frame_idx)
) WITHOUT ROWID;

-- A pass over part of a video that ran to completion, so the same range is not
-- decoded again. Frames outside completed ranges may still exist (interrupted
-- passes keep what they wrote).
CREATE TABLE IF NOT EXISTS proxy_ranges (
    shot_video_id TEXT    NOT NULL,
    -- Every step-th frame index (frame_idx % step_frames == 0) was stored.
    step_frames   INTEGER NOT NULL,
    -- Width in pixels of the stored images.
    width_px      INTEGER NOT NULL,
    -- Inclusive frame range that was covered.
    start_frame   INTEGER NOT NULL,
    end_frame     INTEGER NOT NULL
);
"""


def sidecar_path(session_db: str | Path) -> Path:
    """``session.db`` -> ``session.cache.db`` in the same folder."""
    p = Path(session_db)
    return p.with_name(p.stem + ".cache.db")


def session_db_path(conn: sqlite3.Connection) -> Path | None:
    """The file behind an open session connection, or None (in-memory)."""
    for row in conn.execute("PRAGMA database_list"):
        if row[1] == "main" and row[2]:
            return Path(row[2])
    return None


def open_video(path: str, thread_type: str = "AUTO"):
    """Open *path* with PyAV: ``(container, stream, rotation_degrees)``."""
    import av

    from posetrak.detection.frame_source import _parse_displaymatrix, _stream_rotation

    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = thread_type
    rotation = _stream_rotation(stream)
    if rotation == 0:
        for probe in container.decode(stream):
            for sd in probe.side_data or []:
                if "DISPLAYMATRIX" in str(sd.type).upper():
                    rotation = _parse_displaymatrix(bytes(sd))
            break
        container.seek(0, stream=stream, backward=True, any_frame=False)
    return container, stream, rotation


class ProxyCache:
    """The sidecar database. Each instance has its own connection, so create one
    per thread."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn = sqlite3.connect(str(self.path), timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def nearest(self, shot_video_id: str, frame_idx: int, max_distance: int) -> tuple | None:
        """The stored frame closest to *frame_idx* within *max_distance*, as the
        result tuple ``(jpeg, width_px, height_px, 0, 0, src_w, src_h)`` the
        display code uses, or None."""
        row = self._conn.execute(
            "SELECT jpeg, width_px, height_px, src_w, src_h FROM proxy_frames "
            "WHERE shot_video_id = ? AND frame_idx BETWEEN ? AND ? "
            "ORDER BY ABS(frame_idx - ?) LIMIT 1",
            (shot_video_id, frame_idx - max_distance, frame_idx + max_distance, frame_idx),
        ).fetchone()
        if row is None:
            return None
        return bytes(row[0]), row[1], row[2], 0, 0, row[3], row[4]

    def is_covered(self, shot_video_id: str, start: int, end: int, step: int, width: int) -> bool:
        """True if a finished pass at *step* and *width* covers [start, end]."""
        row = self._conn.execute(
            "SELECT 1 FROM proxy_ranges WHERE shot_video_id = ? AND step_frames = ? "
            "AND width_px = ? AND start_frame <= ? AND end_frame >= ? LIMIT 1",
            (shot_video_id, step, width, start, end),
        ).fetchone()
        return row is not None

    def stats(self) -> list[tuple]:
        """``(shot_video_id, frames, bytes)`` per video."""
        return self._conn.execute(
            "SELECT shot_video_id, COUNT(*), SUM(LENGTH(jpeg)) FROM proxy_frames "
            "GROUP BY shot_video_id"
        ).fetchall()

    def clear(self, shot_video_id: str | None = None) -> None:
        """Delete the proxy of one video, or of everything."""
        where, args = ("", ()) if shot_video_id is None else (" WHERE shot_video_id = ?", (shot_video_id,))
        with self._conn:
            self._conn.execute("DELETE FROM proxy_frames" + where, args)
            self._conn.execute("DELETE FROM proxy_ranges" + where, args)

    def build(
        self,
        shot_video_id: str,
        video_path: str,
        *,
        start: int = 0,
        end: int | None = None,
        step: int = PROXY_STEP,
        width: int = PROXY_WIDTH,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> int:
        """Store every *step*-th frame in [start, end] (end None = to the end of
        the video) by one sequential decode. Returns the number of frames
        written; a cancelled pass keeps what it wrote and records no range."""
        import cv2

        from posetrak.detection.frame_source import _apply_rotation

        if end is not None and self.is_covered(shot_video_id, start, end, step, width):
            return 0
        container, stream, rotation = open_video(video_path)
        written = 0
        try:
            tb, fps = float(stream.time_base), float(stream.average_rate)
            container.seek(
                int(max(0.0, (start - 0.5) / fps) / tb), stream=stream, backward=True,
                any_frame=False,
            )
            have = {
                r[0]
                for r in self._conn.execute(
                    "SELECT frame_idx FROM proxy_frames WHERE shot_video_id = ? "
                    "AND frame_idx >= ? AND frame_idx <= ?",
                    (shot_video_id, start, end if end is not None else 1 << 40),
                )
            }
            last = start
            pending: list[tuple] = []

            def flush() -> None:
                # Rows are held in memory and written in one short transaction:
                # several videos are built at once, and a write transaction that
                # stays open across many decoded frames would block the others.
                if pending:
                    with self._conn:
                        self._conn.executemany(
                            "INSERT OR REPLACE INTO proxy_frames VALUES (?,?,?,?,?,?,?)", pending
                        )
                    pending.clear()

            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                idx = round(float(frame.pts) * tb * fps)
                if end is not None and idx > end:
                    break
                last = idx
                if idx < start or idx % step or idx in have:
                    continue
                if written % 8 == 0 and cancelled():
                    flush()
                    return written
                img = _apply_rotation(frame.to_ndarray(format="bgr24"), rotation)
                h, w = img.shape[:2]
                if w > width:
                    img = cv2.resize(
                        img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA
                    )
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, PROXY_JPEG_QUALITY])
                if not ok:
                    continue
                pending.append(
                    (shot_video_id, idx, img.shape[1], img.shape[0], w, h, buf.tobytes())
                )
                written += 1
                if written % _COMMIT_EVERY == 0:
                    flush()
            flush()
            with self._conn:
                self._conn.execute(
                    "INSERT INTO proxy_ranges VALUES (?,?,?,?,?)",
                    (shot_video_id, step, width, start, end if end is not None else last),
                )
        finally:
            try:
                container.close()
            except Exception:  # noqa: BLE001
                _log.debug("proxy build: close failed", exc_info=True)
        return written


CHUNK_FRAMES = 1200          # granularity of the build order (10 s at 120 fps)
_FOCUS_SETTLE_S = 0.5        # the focus must stay put this long to interrupt a chunk


class ProxyBuilder:
    """Builds the proxy of one video over [first, last] on a background thread.

    The range is cut into chunks and the chunk nearest the *focus* frame (the
    frame being looked at) is built next, so the part in use is ready first and
    a jump elsewhere is followed. A chunk in progress is interrupted when the
    focus has settled in another unfinished chunk; the frames it already wrote
    are kept and not decoded again when it is resumed.
    """

    def __init__(
        self,
        cache_path: str | Path,
        shot_video_id: str,
        video_path: str,
        first: int,
        last: int,
        focus: int,
        *,
        step: int = PROXY_STEP,
        width: int = PROXY_WIDTH,
        chunk: int = CHUNK_FRAMES,
    ) -> None:
        import threading

        self._cache_path = Path(cache_path)
        self._svid, self._video_path = shot_video_id, video_path
        self._step, self._width = step, width
        self._pending = [(a, min(a + chunk - 1, last)) for a in range(first, last + 1, chunk)]
        self.total = len(self._pending)
        self._lock = threading.Lock()
        self._focus = focus
        self._focus_since = 0.0
        self._current: tuple[int, int] | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="proxy-build")

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self._stop = True

    @property
    def done(self) -> int:
        with self._lock:
            return self.total - len(self._pending)

    def set_focus(self, frame: int) -> None:
        """Tell the builder which frame is being looked at (cheap; any thread)."""
        import time

        with self._lock:
            if frame != self._focus:
                self._focus = frame
                self._focus_since = time.monotonic()

    def _nearest_pending(self) -> tuple[int, int] | None:
        """Called with the lock held."""
        if not self._pending:
            return None

        def distance(c: tuple[int, int]) -> int:
            if c[0] <= self._focus <= c[1]:
                return 0
            return min(abs(self._focus - c[0]), abs(self._focus - c[1]))

        return min(self._pending, key=distance)

    def _interrupted(self) -> bool:
        import time

        if self._stop:
            return True
        with self._lock:
            if time.monotonic() - self._focus_since < _FOCUS_SETTLE_S:
                return False
            best = self._nearest_pending()
            return best is not None and best != self._current

    def _run(self) -> None:
        try:
            cache = ProxyCache(self._cache_path)
        except Exception:  # noqa: BLE001
            _log.warning("proxy builder: cannot open %s", self._cache_path, exc_info=True)
            return
        try:
            while not self._stop:
                with self._lock:
                    chunk = self._current = self._nearest_pending()
                if chunk is None:
                    return
                try:
                    cache.build(
                        self._svid, self._video_path, start=chunk[0], end=chunk[1],
                        step=self._step, width=self._width, cancelled=self._interrupted,
                    )
                except Exception:  # noqa: BLE001
                    _log.warning("proxy build failed: %s", self._video_path, exc_info=True)
                    return
                if cache.is_covered(self._svid, chunk[0], chunk[1], self._step, self._width):
                    with self._lock:
                        self._pending.remove(chunk)
        finally:
            cache.close()
