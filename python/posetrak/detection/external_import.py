# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""external_import.py — import 2D point tracks made outside PoseTrak as a detection run.

A track made in another tool (a hand-tracked point in Blender's Movie Clip
Editor, another tracker's output) is a sequence of pixel positions of one
marker in one camera's video. It becomes a ``detection_runs`` row with
``detector_type='external_2d'`` whose rows are in the ``dots`` layout that the
reflective-dot detector writes: per camera and frame, a candidate count and one
candidate per track that has a point on that frame. Nothing downstream tells it
from automatic detection: ``sequence finalise-object`` and ``sequence add-dots``
copy its rows into a sequence like any dots run's.

File format
-----------
One CSV file per track, with a header row and the columns ``video_frame``,
``pixel_x`` and ``pixel_y``: the frame number in the camera's video and the
position in the raw (distorted) image in pixels, origin top left, y down. Other
columns are ignored, so ``python/tools/blender/blender_export_2d_tracks.py``'s
output can be imported as it is. A track may have gaps.

Layouts
-------
``config_json["layout"]`` names how tracks are read into observations. Only
``"anonymous"`` exists: each point is an unlabeled dot candidate, and a track's
points share a tracklet id (per camera), so the assignment can follow the track
from frame to frame. ``"labeled"`` is reserved and not implemented: it would write the
fixed-slot corner layout plus a ``pose_sequence_keypoints`` manifest, and read a
``"label_map"`` in ``config_json`` from external track name to PoseTrak landmark
name (``{"Track.003": "hilt:c2"}``), with a track missing from the map imported
as anonymous rather than dropped.
"""
from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from posetrak.db.sync_lookup import load_sync_table

# A hand-placed point has no measured size. The size only feeds the streak
# noise inflation of the dot assignment, which a round candidate does not trigger.
_NOMINAL_AREA_PX = 50.0
_REQUIRED_COLUMNS = ("video_frame", "pixel_x", "pixel_y")


@dataclass
class External2DResult:
    """What :func:`import_external_2d` wrote."""

    detection_run_id: str
    rows: int                        # (camera, frame) rows written
    points: int                      # points in those rows
    skipped_no_timestamp: int        # points on frames the sync configuration does not cover
    rows_by_camera: dict[str, int]


def import_external_2d(
    session: sqlite3.Connection,
    capture_id: str,
    sync_config_id: str,
    tracks: Sequence[tuple[str, Path]],
    *,
    trial_id: str | None = None,
    capture_object_id: str | None = None,
    source: str = "external",
) -> External2DResult:
    """Import track files as one ``external_2d`` detection run.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    capture_id:
        The capture the tracks were made for.
    sync_config_id:
        Sync configuration that maps the tracks' video frames to global time.
    tracks:
        ``(camera label, CSV path)`` per track. A camera may have several tracks
        (a track restarted after an occlusion, or two markers); they are kept
        apart by tracklet id.
    trial_id:
        Trial to record on the run.
    capture_object_id:
        Object to bind the run to, so that ``sequence finalise-object`` can make
        its sequence.
    source:
        Free text naming the tool or method the tracks come from, recorded in
        ``config_json``.

    Returns
    -------
    External2DResult
        The new run's ID and what was written.

    Raises
    ------
    ValueError
        If there are no tracks, a camera label has no video in the capture, a file
        lacks a required column, has a value that is not a number or a frame outside
        its video, or no point maps to a sync timestamp. Nothing is written in these
        cases.
    """
    # Imported here: it lives in the GUI application's package.
    from app.pose.db_cache import DotCandidateWriter, create_detection_run, mark_run_complete

    if not tracks:
        raise ValueError("no tracks to import")
    video_rows = session.execute(
        "SELECT ci.label, cv.id, cv.first_video_frame, cv.last_video_frame FROM capture_videos cv "
        "JOIN camera_instances ci ON ci.id = cv.camera_instance_id WHERE cv.shot_id = ?",
        (capture_id,),
    ).fetchall()
    video_by_label = {r[0]: r[1] for r in video_rows}
    frame_range = {r[1]: (r[2], r[3]) for r in video_rows}
    points_by_video: dict[str, dict[int, list[tuple[float, float, int]]]] = {}
    next_tracklet: dict[str, int] = {}
    for label, path in tracks:
        video_id = video_by_label.get(label)
        if video_id is None:
            raise ValueError(
                f"no video of camera {label!r} in this capture (cameras: {', '.join(sorted(video_by_label))})"
            )
        tracklet = next_tracklet.get(label, 0)
        next_tracklet[label] = tracklet + 1
        frames = points_by_video.setdefault(video_id, {})
        first, last = frame_range[video_id]
        for frame, x, y in _read_track(Path(path)):
            if not first <= frame <= last:
                raise ValueError(
                    f"{Path(path).name}: frame {frame} is outside the frames of camera {label!r} "
                    f"({first}-{last}); check the export's frame numbering"
                )
            frames.setdefault(frame, []).append((x, y, tracklet))

    sync_table = load_sync_table(session, sync_config_id)
    times: list[float] = []
    skipped = 0
    for video_id, frames in points_by_video.items():
        for frame in list(frames):
            t = sync_table.frame_to_global_time(frame, video_id)
            if t is None:
                skipped += len(frames.pop(frame))
            else:
                times.append(t)
    if not times:
        raise ValueError("none of the points falls on a frame the sync configuration covers")

    config = {
        "layout": "anonymous",
        "source": source,
        "files": [{"camera": label, "file": Path(path).name} for label, path in tracks],
    }
    run_id = create_detection_run(
        session,
        shot_id=capture_id,
        sync_config_id=sync_config_id,
        time_start_s=min(times),
        time_end_s=max(times),
        detector_model=f"external_2d:{source}",
        pose_model="",
        trial_id=trial_id,
        detector_type="external_2d",
        config_json=json.dumps(config),
    )
    if capture_object_id is not None:
        session.execute(
            "UPDATE detection_runs SET capture_object_id = ? WHERE id = ?", (capture_object_id, run_id)
        )
        session.commit()

    label_by_video = {video_id: label for label, video_id in video_by_label.items()}
    rows_by_camera: dict[str, int] = {}
    n_points = 0
    for video_id, frames in points_by_video.items():
        writer = DotCandidateWriter(session, run_id, video_id)
        for frame in sorted(frames):
            writer.add_frame(frame, [_candidate(x, y, tracklet) for x, y, tracklet in frames[frame]])
            n_points += len(frames[frame])
        writer.finalise()
        rows_by_camera[label_by_video[video_id]] = len(frames)
    mark_run_complete(session, run_id)
    return External2DResult(run_id, sum(rows_by_camera.values()), n_points, skipped, rows_by_camera)


def _read_track(path: Path) -> list[tuple[int, float, float]]:
    """Read ``(frame, x, y)`` of every point of a track file."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in _REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name}: missing column(s) {', '.join(missing)} (has {reader.fieldnames})")
        points = []
        for line, row in enumerate(reader, start=2):
            try:
                frame, x, y = int(float(row["video_frame"])), float(row["pixel_x"]), float(row["pixel_y"])
            except (TypeError, ValueError):
                raise ValueError(f"{path.name} line {line}: video_frame, pixel_x and pixel_y must be numbers") from None
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"{path.name} line {line}: pixel position is not finite")
            points.append((frame, x, y))
    return points


def _candidate(x: float, y: float, tracklet_id: int):
    from posetrak.detection.dot_blob_detector import BlobCandidate

    diameter = 2.0 * math.sqrt(_NOMINAL_AREA_PX / math.pi)
    r = diameter / 2.0
    return BlobCandidate(
        cx=x, cy=y, area=_NOMINAL_AREA_PX, compactness=1.0,
        bbox=(int(x - r), int(y - r), int(diameter), int(diameter)),
        major_axis_px=diameter, minor_axis_px=diameter,
        dir_x=0.0, dir_y=0.0, tracklet_id=float(tracklet_id),
    )
