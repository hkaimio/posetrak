# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""video_export.py — Plan and extract per-camera video clips for a time
range on a capture's synced master timeline.

A capture's ``sync_points`` map each camera's own raw video frame numbers to
a shared master timeline (see ``docs/architecture/data-model.md`` and
``app/setup/page_sync.py``, which produces them). Given a [start, end] range
on that master timeline, :func:`plan_clip` interpolates the corresponding
frame range for one camera and converts it into a seek time on that
camera's *own* video file, so :func:`run_ffmpeg_extract` can cut a clip with
ffmpeg.

The frame→seek-time conversion needs the file's own container frame rate,
not ``capture_videos.actual_fps`` — the two can diverge sharply for some
phones' slow-motion recordings, which tag the container with a much lower
nominal playback rate than the true capture rate (e.g. Android's
``com.android.capture.fps`` metadata) so that normal playback shows slow
motion. Trusting ``actual_fps`` there seeks to the wrong point in the file.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable


class VideoExportError(RuntimeError):
    """Raised for clip-planning or extraction failures."""


@dataclass
class ClipPlan:
    camera_label: str
    camera_instance_id: str
    source_path: str
    container_start_s: float
    container_duration_s: float
    frame_start: float
    frame_end: float


def resolve_sync_config(
    conn: sqlite3.Connection, capture_id: str, sync_config_id: str | None = None
) -> str:
    """Return the sync_config_id to use for *capture_id*.

    Auto-resolves when the capture has exactly one; otherwise requires
    *sync_config_id* to already pick one (raises listing the options).
    """
    if sync_config_id is not None:
        return sync_config_id
    rows = conn.execute(
        "SELECT id FROM sync_configs WHERE shot_id = ?", (capture_id,)
    ).fetchall()
    if len(rows) == 1:
        return rows[0][0]
    if len(rows) == 0:
        raise VideoExportError(f"No sync config found for capture {capture_id!r}.")
    ids = ", ".join(r[0] for r in rows)
    raise VideoExportError(
        f"Capture {capture_id!r} has {len(rows)} sync configs ({ids}) — pass "
        "--sync-config to pick one."
    )


def _frame_at_time(points: list[tuple[int, float]], t: float) -> float:
    """Linear fit of video_frame vs. timestamp_s from *points*, evaluated at *t*.

    Two points give an exact line (the common case: two flash/clap anchors).
    More than two are least-squares fit via numpy.polyfit.
    """
    if len(points) < 2:
        raise VideoExportError(
            "Need at least 2 sync points to interpolate a frame number; got "
            f"{len(points)}."
        )
    if len(points) == 2:
        (f0, t0), (f1, t1) = points
        if t1 == t0:
            raise VideoExportError("Sync points have identical timestamps.")
        slope = (f1 - f0) / (t1 - t0)
        return f0 + slope * (t - t0)

    import numpy as np
    frames = np.array([p[0] for p in points], dtype=float)
    times = np.array([p[1] for p in points], dtype=float)
    slope, intercept = np.polyfit(times, frames, 1)
    return float(slope * t + intercept)


def probe_container_fps(file_path: Path) -> float:
    """Return the video stream's own declared frame rate (ffprobe's
    ``r_frame_rate``) as a float — see module docstring for why this, and
    not the DB's ``actual_fps``, is the right clock for seeking."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise VideoExportError("ffprobe not found on PATH — required to plan a clip export.")
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(file_path)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise VideoExportError(f"ffprobe failed on {file_path}: {exc.stderr}") from exc
    text = result.stdout.strip()
    if not text:
        raise VideoExportError(f"ffprobe returned no frame rate for {file_path}")
    return float(Fraction(text))


def plan_clip(
    conn: sqlite3.Connection,
    *,
    capture_id: str,
    sync_config_id: str,
    camera_instance_id: str,
    camera_label: str,
    master_start_s: float,
    master_end_s: float,
    probe_fps: Callable[[Path], float] | None = None,
) -> ClipPlan:
    """Compute the container-time [start, duration] to extract for one
    camera, from a [master_start_s, master_end_s] range on the capture's
    synced master timeline.

    *probe_fps* is injectable so planning can be unit-tested without
    invoking real ffprobe. Looked up as a module attribute at call time
    (rather than bound as the parameter's default) so tests can
    monkeypatch ``video_export.probe_container_fps`` even for callers —
    e.g. the CLI — that don't pass *probe_fps* explicitly.
    """
    if probe_fps is None:
        probe_fps = probe_container_fps
    if master_end_s <= master_start_s:
        raise VideoExportError(
            f"End ({master_end_s}) must be after start ({master_start_s})."
        )

    video_row = conn.execute(
        "SELECT id, file_path FROM capture_videos "
        "WHERE shot_id = ? AND camera_instance_id = ?",
        (capture_id, camera_instance_id),
    ).fetchone()
    if video_row is None:
        raise VideoExportError(
            f"Camera {camera_label!r} has no video registered for capture {capture_id!r}."
        )

    rows = conn.execute(
        "SELECT video_frame, timestamp_s FROM sync_points "
        "WHERE sync_config_id = ? AND shot_video_id = ?",
        (sync_config_id, video_row["id"]),
    ).fetchall()
    if len(rows) < 2:
        raise VideoExportError(
            f"Camera {camera_label!r} has fewer than 2 sync points for sync "
            f"config {sync_config_id!r} — cannot map the time range onto its footage."
        )
    points = sorted((r["video_frame"], r["timestamp_s"]) for r in rows)

    frame_start = _frame_at_time(points, master_start_s)
    frame_end = _frame_at_time(points, master_end_s)
    if frame_end <= frame_start:
        raise VideoExportError(
            f"Camera {camera_label!r}: computed end frame ({frame_end:.1f}) is "
            f"not after start frame ({frame_start:.1f})."
        )
    if frame_start < 0:
        raise VideoExportError(
            f"Camera {camera_label!r}: computed start frame ({frame_start:.1f}) "
            "is before the start of the recording — reduce --before."
        )

    file_path = Path(video_row["file_path"])
    container_fps = probe_fps(file_path)
    if container_fps <= 0:
        raise VideoExportError(f"Camera {camera_label!r}: invalid container fps {container_fps}")

    return ClipPlan(
        camera_label=camera_label,
        camera_instance_id=camera_instance_id,
        source_path=str(file_path),
        container_start_s=frame_start / container_fps,
        container_duration_s=(frame_end - frame_start) / container_fps,
        frame_start=frame_start,
        frame_end=frame_end,
    )


def run_ffmpeg_extract(plan: ClipPlan, output_path: Path, *, overwrite: bool = False) -> None:
    """Stream-copy the planned clip to *output_path* via ffmpeg.

    Uses ``-c copy`` (no re-encode): fast and lossless, at the cost of
    snapping the actual start to the nearest preceding keyframe — the clip
    can start slightly earlier than planned, never later, so it never cuts
    into the requested range.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise VideoExportError("ffmpeg not found on PATH — required to extract a clip.")
    cmd = [
        ffmpeg, "-y" if overwrite else "-n",
        "-ss", f"{plan.container_start_s:.6f}",
        "-i", plan.source_path,
        "-t", f"{plan.container_duration_s:.6f}",
        "-c", "copy",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise VideoExportError(
            f"ffmpeg failed extracting {plan.camera_label!r}: {exc.stderr}"
        ) from exc
