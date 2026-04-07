"""DBContext — session-DB access layer for the setup wizard.

All wizard pages read and write through this object; they do not open their
own connections.  A single transaction is held open within each page so that
"Back" can roll it back via ``begin_page()`` / ``rollback_page()``.
"""

from __future__ import annotations

import bisect
import sqlite3
import struct
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from posetrak.db.db import generate_id


# ---------------------------------------------------------------------------
# Typed return types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncPoint:
    """One sync anchor: a (frame, timestamp) pair for a single camera video."""
    camera_instance_id: str
    shot_video_id: str
    video_frame: int
    timestamp_s: float


@dataclass(frozen=True)
class ExtrinsicEntry:
    """Rotation + translation for one camera in a calibration set."""
    camera_instance_id: str
    R: np.ndarray  # shape (3, 3), float64, row-major
    t: np.ndarray  # shape (3,), float64

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtrinsicEntry):
            return NotImplemented
        return (
            self.camera_instance_id == other.camera_instance_id
            and np.array_equal(self.R, other.R)
            and np.array_equal(self.t, other.t)
        )

    def __hash__(self) -> int:
        return hash((self.camera_instance_id, self.R.tobytes(), self.t.tobytes()))


class ShotVideoInfo(NamedTuple):
    """Row from shot_videos, enriched with camera label."""
    id: str
    shot_id: str
    camera_instance_id: str
    file_path: str
    actual_fps: float
    first_video_frame: int
    last_video_frame: int


# ---------------------------------------------------------------------------
# SyncTable
# ---------------------------------------------------------------------------


class SyncTable:
    """Piecewise-linear timestamp → frame index mapping for each shot video.

    Mirrors the lookup algorithm used by visualize_tracking.py but indexed
    by ``shot_video_id`` rather than camera label.
    """

    def __init__(
        self,
        points: list[SyncPoint],
        fps_by_video: dict[str, float],
    ) -> None:
        # Build per-video sorted anchor lists
        self._tables: dict[str, tuple[list[float], list[int], float]] = {}
        grouped: dict[str, list[SyncPoint]] = {}
        for pt in points:
            grouped.setdefault(pt.shot_video_id, []).append(pt)

        for vid_id, pts in grouped.items():
            pts_sorted = sorted(pts, key=lambda p: p.timestamp_s)
            timestamps = [p.timestamp_s for p in pts_sorted]
            frames = [p.video_frame for p in pts_sorted]
            fps = fps_by_video.get(vid_id, 0.0)
            self._tables[vid_id] = (timestamps, frames, fps)

    def lookup(self, timestamp_s: float, shot_video_id: str) -> int | None:
        """Return the video frame index for *shot_video_id* at *timestamp_s*.

        Returns ``None`` if no sync data is available for this video.
        """
        if shot_video_id not in self._tables:
            return None
        timestamps, frames, fps = self._tables[shot_video_id]
        if not timestamps:
            return None

        idx = bisect.bisect_right(timestamps, timestamp_s)
        if idx == 0:
            anchor_ts, anchor_frame = timestamps[0], frames[0]
        elif idx >= len(timestamps):
            anchor_ts, anchor_frame = timestamps[-1], frames[-1]
        else:
            anchor_ts, anchor_frame = timestamps[idx - 1], frames[idx - 1]

        if fps > 0:
            return anchor_frame + round((timestamp_s - anchor_ts) * fps)
        # No fps: snap to nearest anchor
        if idx == 0:
            return frames[0]
        if idx >= len(timestamps):
            return frames[-1]
        if abs(timestamps[idx] - timestamp_s) < abs(timestamps[idx - 1] - timestamp_s):
            return frames[idx]
        return frames[idx - 1]

    def frame_to_global_time(self, frame_idx: int, shot_video_id: str) -> float | None:
        """Return the approximate global timestamp for a local frame index.

        Inverts the ``lookup`` relationship: given a frame in *shot_video_id*,
        finds the anchor closest to that frame and extrapolates using the stored
        fps.  Returns ``None`` if no sync data is available for the video.
        """
        if shot_video_id not in self._tables:
            return None
        timestamps, frames, fps = self._tables[shot_video_id]
        if not timestamps:
            return None
        if fps <= 0:
            return float(timestamps[0])
        best_i = min(range(len(frames)), key=lambda i: abs(frames[i] - frame_idx))
        return float(timestamps[best_i]) + (frame_idx - frames[best_i]) / fps

    def video_ids(self) -> list[str]:
        return list(self._tables.keys())

    def time_range(self) -> tuple[float, float] | None:
        """Return (min_timestamp, max_timestamp) across all videos, or None if empty."""
        all_ts: list[float] = []
        for timestamps, _frames, _fps in self._tables.values():
            all_ts.extend(timestamps)
        if not all_ts:
            return None
        return (min(all_ts), max(all_ts))


# ---------------------------------------------------------------------------
# DBContext
# ---------------------------------------------------------------------------

# Preference order for sync method selection in get_active_sync().
_SYNC_METHOD_RANK: dict[str, int] = {"led-auto": 0, "manual-rough": 1}


class DBContext:
    """Session-DB write gateway for the setup wizard.

    Parameters
    ----------
    conn:
        Open connection to a session database (v8+).
    session_id:
        UUID of the ``mocap_sessions`` row this wizard is operating on.
    """

    def __init__(self, conn: sqlite3.Connection, session_id: str) -> None:
        self._conn = conn
        self._session_id = session_id
        self._savepoint_active = False

    # ------------------------------------------------------------------
    # Page transaction helpers
    # ------------------------------------------------------------------

    def begin_page(self) -> None:
        """Open a savepoint so the current page's writes can be rolled back."""
        self._conn.execute("SAVEPOINT wizard_page")
        self._savepoint_active = True

    def commit_page(self) -> None:
        """Release the current page savepoint (makes writes durable)."""
        self._conn.execute("RELEASE SAVEPOINT wizard_page")
        self._savepoint_active = False

    def rollback_page(self) -> None:
        """Roll back all writes since the last ``begin_page()``.

        No-op if no savepoint is currently active (e.g. cleanupPage called
        after the page was already committed and the user later closes the wizard).
        """
        if not self._savepoint_active:
            return
        self._conn.execute("ROLLBACK TO SAVEPOINT wizard_page")
        self._conn.execute("RELEASE SAVEPOINT wizard_page")
        self._savepoint_active = False

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def create_shot(self, label: str, shot_number: int) -> str:
        """Insert a ``shots`` row and return its ID."""
        shot_id = generate_id()
        self._conn.execute(
            "INSERT INTO shots (id, session_id, shot_number, label) VALUES (?, ?, ?, ?)",
            (shot_id, self._session_id, shot_number, label),
        )
        return shot_id

    def create_shot_video(
        self,
        shot_id: str,
        cam_instance_id: str,
        path: str,
        fps: float,
        frame_count: int,
        width: int,   # noqa: ARG002 — stored for future schema extension
        height: int,  # noqa: ARG002 — stored for future schema extension
    ) -> str:
        """Insert a ``shot_videos`` row and return its ID.

        ``width`` and ``height`` are accepted to match the interface spec but
        are not persisted (not in the current ``shot_videos`` schema).
        """
        video_id = generate_id()
        self._conn.execute(
            "INSERT INTO shot_videos "
            "(id, shot_id, camera_instance_id, file_path, "
            "first_video_frame, last_video_frame, actual_fps) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (video_id, shot_id, cam_instance_id, path, 0, frame_count - 1, fps),
        )
        return video_id

    def write_sync_config(
        self,
        shot_id: str,
        method: str,
        points: dict[str, list[SyncPoint]],
    ) -> str:
        """Write a sync_configs row + sync_points and return the config ID.

        Parameters
        ----------
        shot_id:
            Parent shot UUID.
        method:
            Sync method string, e.g. ``"manual-rough"`` or ``"led-auto"``.
            Stored in ``sync_configs.created_by`` for retrieval by
            ``get_active_sync()``.
        points:
            Mapping from camera_instance_id → list of SyncPoint.
        """
        config_id = generate_id()
        self._conn.execute(
            "INSERT INTO sync_configs (id, shot_id, created_by) VALUES (?, ?, ?)",
            (config_id, shot_id, method),
        )
        for pts in points.values():
            for pt in pts:
                self._conn.execute(
                    "INSERT INTO sync_points "
                    "(sync_config_id, camera_instance_id, shot_video_id, "
                    "video_frame, timestamp_s) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        config_id,
                        pt.camera_instance_id,
                        pt.shot_video_id,
                        pt.video_frame,
                        pt.timestamp_s,
                    ),
                )
        return config_id

    def write_extrinsics(
        self,
        shot_id: str,
        entries: list[ExtrinsicEntry],
        *,
        rms_error: float | None = None,
    ) -> str:
        """Write an extrinsic calibration set and link it to the shot.

        Returns the ``extrinsic_calibrations.id``.
        """
        import datetime

        calib_id = generate_id()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO extrinsic_calibrations "
            "(id, session_id, calibrated_at, rms_error) VALUES (?, ?, ?, ?)",
            (calib_id, self._session_id, now, rms_error),
        )
        for entry in entries:
            r_blob = entry.R.astype(np.float64).flatten().tobytes()
            t_blob = entry.t.astype(np.float64).flatten().tobytes()
            self._conn.execute(
                "INSERT INTO extrinsic_entries "
                "(extrinsic_calibration_id, camera_instance_id, R, t) "
                "VALUES (?, ?, ?, ?)",
                (calib_id, entry.camera_instance_id, r_blob, t_blob),
            )
        # Link calibration to the shot
        self._conn.execute(
            "UPDATE shots SET extrinsic_calibration_id = ? WHERE id = ?",
            (calib_id, shot_id),
        )
        return calib_id

    def update_shot_video_fps(self, shot_video_id: str, fps: float) -> None:
        """Persist a corrected fps value to shot_videos.actual_fps."""
        self._conn.execute(
            "UPDATE shot_videos SET actual_fps = ? WHERE id = ?",
            (fps, shot_video_id),
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_shot_videos(self, shot_id: str) -> list[ShotVideoInfo]:
        """Return all shot_videos rows for *shot_id*."""
        rows = self._conn.execute(
            "SELECT id, shot_id, camera_instance_id, file_path, "
            "actual_fps, first_video_frame, last_video_frame "
            "FROM shot_videos WHERE shot_id = ? ORDER BY rowid",
            (shot_id,),
        ).fetchall()
        return [
            ShotVideoInfo(
                id=r["id"],
                shot_id=r["shot_id"],
                camera_instance_id=r["camera_instance_id"],
                file_path=r["file_path"],
                actual_fps=r["actual_fps"],
                first_video_frame=r["first_video_frame"],
                last_video_frame=r["last_video_frame"],
            )
            for r in rows
        ]

    def get_sync_configs(self, shot_id: str) -> list[tuple[str, str]]:
        """Return ``[(config_id, created_by), …]`` for a shot, newest first."""
        rows = self._conn.execute(
            "SELECT id, created_by FROM sync_configs WHERE shot_id = ? ORDER BY rowid DESC",
            (shot_id,),
        ).fetchall()
        return [(r["id"], r["created_by"] or "") for r in rows]

    def load_sync_config(self, config_id: str) -> SyncTable | None:
        """Load a specific sync config into a SyncTable."""
        rows = self._conn.execute(
            "SELECT sp.camera_instance_id, sp.shot_video_id, "
            "sp.video_frame, sp.timestamp_s, sv.actual_fps "
            "FROM sync_points sp "
            "JOIN shot_videos sv ON sv.id = sp.shot_video_id "
            "WHERE sp.sync_config_id = ?",
            (config_id,),
        ).fetchall()
        if not rows:
            return None
        points = [
            SyncPoint(
                camera_instance_id=r["camera_instance_id"],
                shot_video_id=r["shot_video_id"],
                video_frame=r["video_frame"],
                timestamp_s=r["timestamp_s"],
            )
            for r in rows
        ]
        fps_by_video = {r["shot_video_id"]: r["actual_fps"] for r in rows}
        return SyncTable(points, fps_by_video)

    def get_active_sync(self, shot_id: str) -> SyncTable | None:
        """Return a ``SyncTable`` for the best available sync config, or ``None``.

        Prefers ``led-auto`` over ``manual-rough``; falls back to the most
        recently inserted config if neither method is present.
        """
        configs = self._conn.execute(
            "SELECT id, created_by FROM sync_configs WHERE shot_id = ? ORDER BY rowid",
            (shot_id,),
        ).fetchall()
        if not configs:
            return None

        # Pick best config
        best = min(
            configs,
            key=lambda r: (_SYNC_METHOD_RANK.get(r["created_by"] or "", 99), r.keys()),
        )
        config_id = best["id"]

        # Load sync points
        rows = self._conn.execute(
            "SELECT sp.camera_instance_id, sp.shot_video_id, "
            "sp.video_frame, sp.timestamp_s, sv.actual_fps "
            "FROM sync_points sp "
            "JOIN shot_videos sv ON sv.id = sp.shot_video_id "
            "WHERE sp.sync_config_id = ?",
            (config_id,),
        ).fetchall()

        if not rows:
            return None

        points = [
            SyncPoint(
                camera_instance_id=r["camera_instance_id"],
                shot_video_id=r["shot_video_id"],
                video_frame=r["video_frame"],
                timestamp_s=r["timestamp_s"],
            )
            for r in rows
        ]
        fps_by_video = {r["shot_video_id"]: r["actual_fps"] for r in rows}
        return SyncTable(points, fps_by_video)
