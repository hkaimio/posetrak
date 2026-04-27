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


class CaptureVideoInfo(NamedTuple):
    """Row from capture_videos, enriched with camera label."""
    id: str
    shot_id: str  # historical column name; references captures(id)
    camera_instance_id: str
    file_path: str
    actual_fps: float
    first_video_frame: int
    last_video_frame: int
    camera_label: str = ""


ShotVideoInfo = CaptureVideoInfo  # backwards-compat alias


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
        Open connection to a session database (v11+).
    session_id:
        UUID of the ``mocap_sessions`` row this wizard is operating on.
    registry_conn:
        Optional open connection to a registry database.  When present,
        ``upsert_camera_records()`` copies camera rows from the registry into
        the session-local tables so the session DB stays self-contained.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        registry_conn: sqlite3.Connection | None = None,
    ) -> None:
        self._conn = conn
        self._session_id = session_id
        self._registry_conn = registry_conn
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
        """Insert a ``captures`` row and return its ID."""
        shot_id = generate_id()
        self._conn.execute(
            "INSERT INTO captures (id, session_id, capture_number, label) VALUES (?, ?, ?, ?)",
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
        width: int,   # noqa: ARG002 — not persisted in current schema
        height: int,  # noqa: ARG002 — not persisted in current schema
        camera_mode_id: str | None = None,
        intrinsics_calibration_id: str | None = None,
    ) -> str:
        """Insert a ``capture_videos`` row and return its ID."""
        video_id = generate_id()
        self._conn.execute(
            "INSERT INTO capture_videos "
            "(id, shot_id, camera_instance_id, file_path, "
            "first_video_frame, last_video_frame, actual_fps, "
            "camera_mode_id, intrinsics_calibration_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (video_id, shot_id, cam_instance_id, path, 0, frame_count - 1, fps,
             camera_mode_id, intrinsics_calibration_id),
        )
        return video_id

    def upsert_session_camera(self, camera_instance_id: str) -> None:
        """Record that *camera_instance_id* participated in this session.

        Uses INSERT OR IGNORE so calling it multiple times for the same camera
        is safe.
        """
        self._conn.execute(
            "INSERT OR IGNORE INTO session_cameras (session_id, camera_instance_id) "
            "VALUES (?, ?)",
            (self._session_id, camera_instance_id),
        )

    def upsert_camera_records(
        self,
        camera_instance_id: str,
        camera_mode_id: str | None = None,
        intrinsics_calibration_id: str | None = None,
    ) -> None:
        """Copy camera records from the registry into the session-local tables.

        Copies the camera_instances row, its camera_models row, and (if
        *camera_mode_id* is given) the camera_modes row plus the selected
        intrinsics_calibrations row (including undistort maps).  Uses
        INSERT OR IGNORE so re-calling with the same IDs is harmless.

        No-op if no registry connection was provided at construction time.
        """
        reg = self._registry_conn
        if reg is None:
            return

        # camera_instances → camera_model_id
        inst_row = reg.execute(
            "SELECT id, camera_model_id, serial_number, label FROM camera_instances WHERE id = ?",
            (camera_instance_id,),
        ).fetchone()
        if inst_row is None:
            return

        model_id = inst_row["camera_model_id"]
        model_row = reg.execute(
            "SELECT id, manufacturer, model_name, sensor_size FROM camera_models WHERE id = ?",
            (model_id,),
        ).fetchone()
        if model_row:
            self._conn.execute(
                "INSERT OR IGNORE INTO camera_models (id, manufacturer, model_name, sensor_size) "
                "VALUES (?, ?, ?, ?)",
                (model_row["id"], model_row["manufacturer"],
                 model_row["model_name"], model_row["sensor_size"]),
            )

        self._conn.execute(
            "INSERT OR IGNORE INTO camera_instances "
            "(id, camera_model_id, serial_number, label) VALUES (?, ?, ?, ?)",
            (inst_row["id"], inst_row["camera_model_id"],
             inst_row["serial_number"], inst_row["label"]),
        )

        if camera_mode_id:
            mode_row = reg.execute(
                "SELECT id, camera_model_id, width_px, height_px, nominal_fps, codec, notes,"
                "       default_intrinsics_calibration_id "
                "FROM camera_modes WHERE id = ?",
                (camera_mode_id,),
            ).fetchone()
            if mode_row:
                self._conn.execute(
                    "INSERT OR IGNORE INTO camera_modes "
                    "(id, camera_model_id, width_px, height_px, nominal_fps, codec, notes,"
                    " default_intrinsics_calibration_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (mode_row["id"], mode_row["camera_model_id"],
                     mode_row["width_px"], mode_row["height_px"],
                     mode_row["nominal_fps"], mode_row["codec"],
                     mode_row["notes"], mode_row["default_intrinsics_calibration_id"]),
                )

        if intrinsics_calibration_id:
            ic_row = reg.execute(
                "SELECT id, camera_mode_id, calibrated_at, calibration_tool, distortion_model,"
                "       fx, fy, cx, cy, dist_coeffs, rms_error, notes,"
                "       image_width, image_height, matrix_original, undistort_mapx, undistort_mapy "
                "FROM intrinsics_calibrations WHERE id = ?",
                (intrinsics_calibration_id,),
            ).fetchone()
            if ic_row:
                self._conn.execute(
                    "INSERT OR IGNORE INTO intrinsics_calibrations "
                    "(id, camera_mode_id, calibrated_at, calibration_tool, distortion_model,"
                    " fx, fy, cx, cy, dist_coeffs, rms_error, notes,"
                    " image_width, image_height, matrix_original, undistort_mapx, undistort_mapy) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ic_row["id"], ic_row["camera_mode_id"], ic_row["calibrated_at"],
                     ic_row["calibration_tool"], ic_row["distortion_model"],
                     ic_row["fx"], ic_row["fy"], ic_row["cx"], ic_row["cy"],
                     ic_row["dist_coeffs"], ic_row["rms_error"], ic_row["notes"],
                     ic_row["image_width"], ic_row["image_height"],
                     ic_row["matrix_original"], ic_row["undistort_mapx"], ic_row["undistort_mapy"]),
                )

    def list_camera_instances(self) -> list[sqlite3.Row]:
        """Return all camera_instances from the session DB (and registry if available).

        Registry instances not yet in the session DB are included with a
        ``from_registry`` marker.  Rows are sqlite3.Row objects with columns:
        id, label, model_name, from_registry (0 or 1).
        """
        session_rows = self._conn.execute(
            "SELECT ci.id, ci.label, COALESCE(cm.model_name, '') AS model_name, 0 AS from_registry"
            " FROM camera_instances ci"
            " LEFT JOIN camera_models cm ON cm.id = ci.camera_model_id"
            " ORDER BY ci.label"
        ).fetchall()

        if self._registry_conn is None:
            return session_rows

        session_ids = {r["id"] for r in session_rows}
        reg_rows = self._registry_conn.execute(
            "SELECT ci.id, ci.label, COALESCE(cm.model_name, '') AS model_name, 1 AS from_registry"
            " FROM camera_instances ci"
            " LEFT JOIN camera_models cm ON cm.id = ci.camera_model_id"
            " ORDER BY ci.label"
        ).fetchall()

        extra = [r for r in reg_rows if r["id"] not in session_ids]
        return session_rows + extra

    def list_camera_modes(self, model_id: str) -> list[sqlite3.Row]:
        """Return camera_modes for *model_id*, querying session DB first, then registry.

        Rows have columns: id, width_px, height_px, nominal_fps, notes,
        default_intrinsics_calibration_id.
        """
        rows = self._conn.execute(
            "SELECT id, width_px, height_px, nominal_fps, notes, default_intrinsics_calibration_id"
            " FROM camera_modes WHERE camera_model_id = ? ORDER BY width_px DESC, nominal_fps DESC",
            (model_id,),
        ).fetchall()
        if rows:
            return rows
        if self._registry_conn is None:
            return []
        return self._registry_conn.execute(
            "SELECT id, width_px, height_px, nominal_fps, notes, default_intrinsics_calibration_id"
            " FROM camera_modes WHERE camera_model_id = ? ORDER BY width_px DESC, nominal_fps DESC",
            (model_id,),
        ).fetchall()

    def get_camera_model_id(self, camera_instance_id: str) -> str | None:
        """Return the camera_model_id for *camera_instance_id*, or None if not found."""
        row = self._conn.execute(
            "SELECT camera_model_id FROM camera_instances WHERE id = ?",
            (camera_instance_id,),
        ).fetchone()
        if row:
            return row["camera_model_id"]
        if self._registry_conn:
            row = self._registry_conn.execute(
                "SELECT camera_model_id FROM camera_instances WHERE id = ?",
                (camera_instance_id,),
            ).fetchone()
            return row["camera_model_id"] if row else None
        return None

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
        # Link calibration to the capture
        self._conn.execute(
            "UPDATE captures SET extrinsic_calibration_id = ? WHERE id = ?",
            (calib_id, shot_id),
        )
        return calib_id

    def update_shot_video_fps(self, shot_video_id: str, fps: float) -> None:
        """Persist a corrected fps value to capture_videos.actual_fps."""
        self._conn.execute(
            "UPDATE capture_videos SET actual_fps = ? WHERE id = ?",
            (fps, shot_video_id),
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_shot_videos(self, shot_id: str) -> list[CaptureVideoInfo]:
        """Return all capture_videos rows for *shot_id* (capture id)."""
        rows = self._conn.execute(
            "SELECT sv.id, sv.shot_id, sv.camera_instance_id, sv.file_path, "
            "sv.actual_fps, sv.first_video_frame, sv.last_video_frame, "
            "COALESCE(ci.label, sv.camera_instance_id) AS camera_label "
            "FROM capture_videos sv "
            "LEFT JOIN camera_instances ci ON ci.id = sv.camera_instance_id "
            "WHERE sv.shot_id = ? ORDER BY sv.rowid",
            (shot_id,),
        ).fetchall()
        return [
            CaptureVideoInfo(
                id=r["id"],
                shot_id=r["shot_id"],
                camera_instance_id=r["camera_instance_id"],
                file_path=r["file_path"],
                actual_fps=r["actual_fps"],
                first_video_frame=r["first_video_frame"],
                last_video_frame=r["last_video_frame"],
                camera_label=r["camera_label"],
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
            "JOIN capture_videos sv ON sv.id = sp.shot_video_id "
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
            "JOIN capture_videos sv ON sv.id = sp.shot_video_id "
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
