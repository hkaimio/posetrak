"""import_sync_json.py — Import camera synchronisation data from a sync JSON file.

The sync JSON file produced by the posetrak pipeline has the following
structure::

    {
        "cam1": {
            "fps": 119.88,
            "syncpoints": [
                {"frame": 0, "timestamp": 0.0},
                {"frame": 1, "timestamp": 0.00834},
                ...
            ]
        },
        "cam2": {...},
        ...
    }

All syncpoints for each camera are stored. The tracker uses them for
piecewise-linear timestamp interpolation between anchor frames.

One ``sync_configs`` row is created per call, and one ``sync_points`` row is
created per (camera, video_frame) pair.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from scripts.db.posetrak_db import generate_id


@dataclass
class SyncImportResult:
    """Result of a sync JSON import operation.

    Attributes
    ----------
    sync_config_id:
        ID of the newly created ``sync_configs`` row.
    camera_instance_ids:
        Mapping from camera key in the JSON (e.g. ``"cam1"``) to the
        ``camera_instances.id`` used for each sync_points row.
    skipped:
        Set of JSON camera keys that were skipped because they were not
        listed in the *camera_instances* mapping.
    """

    sync_config_id: str = ""
    camera_instance_ids: dict[str, str] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)


def import_sync_json(
    session: sqlite3.Connection,
    shot_id: str,
    sync_json_path: Path,
    camera_instances: str | dict[str, str],
    *,
    created_by: str = "posetrak-db",
    notes: str = "",
) -> SyncImportResult:
    """Import camera synchronisation anchors from a sync JSON file.

    Creates one ``sync_configs`` row and one ``sync_points`` row per (camera,
    video_frame) pair. All inserts are executed in a single transaction.

    The ``shot_video_id`` for each camera is looked up from ``shot_videos``
    using ``shot_id`` and ``camera_instance_id``. A ``ValueError`` is raised
    if no matching ``shot_videos`` row is found.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    shot_id:
        ID of the ``shots`` row this sync configuration belongs to.
    sync_json_path:
        Path to the sync JSON file.
    camera_instances:
        Camera instance assignment. Two forms:

        - **Homogeneous** (``str``): a single ``camera_instances.id`` UUID
          applied to all cameras found in the JSON.
        - **Per-camera** (``dict[str, str]``): mapping from JSON camera key
          (e.g. ``"cam1"``) to ``camera_instances.id``. Keys not listed are
          silently skipped.
    created_by:
        Identifier of the tool or user that created this sync config
        (default ``"posetrak-db"``).
    notes:
        Optional free-text notes stored with the ``sync_configs`` row.

    Returns
    -------
    SyncImportResult
        IDs of all rows created, plus the set of skipped camera keys.

    Raises
    ------
    ValueError
        If a listed camera has no matching ``shot_videos`` row for the given
        ``shot_id``.
    """
    with sync_json_path.open(encoding="utf-8") as fh:
        raw: dict[str, object] = json.load(fh)

    cam_keys = sorted(raw.keys())

    result = SyncImportResult()
    sync_config_id = generate_id()
    result.sync_config_id = sync_config_id

    rows_points: list[tuple[str, str, str, int, float]] = []

    for cam_key in cam_keys:
        if isinstance(camera_instances, str):
            instance_id = camera_instances
        else:
            if cam_key not in camera_instances:
                result.skipped.add(cam_key)
                continue
            instance_id = camera_instances[cam_key]

        cam_data: dict[str, object] = raw[cam_key]  # type: ignore[assignment]
        syncpoints: list[dict[str, object]] = cam_data["syncpoints"]  # type: ignore[assignment]

        # Look up the shot_video row for this camera/shot combination.
        sv_row = session.execute(
            "SELECT id FROM shot_videos "
            "WHERE shot_id = ? AND camera_instance_id = ?",
            (shot_id, instance_id),
        ).fetchone()
        if sv_row is None:
            raise ValueError(
                f"No shot_videos row found for shot_id={shot_id!r} "
                f"and camera_instance_id={instance_id!r} (cam_key={cam_key!r}). "
                "Add shot video rows with add_shot_video() before importing sync data."
            )

        shot_video_id = sv_row["id"]
        for sp in syncpoints:
            frame = int(sp["frame"])
            ts = float(sp["timestamp"])
            rows_points.append((sync_config_id, instance_id, shot_video_id, frame, ts))
        result.camera_instance_ids[cam_key] = instance_id

    with session:
        session.execute(
            "INSERT INTO sync_configs (id, shot_id, created_by, notes) "
            "VALUES (?, ?, ?, ?)",
            (sync_config_id, shot_id, created_by, notes),
        )
        session.executemany(
            "INSERT INTO sync_points "
            "(sync_config_id, camera_instance_id, shot_video_id, video_frame, timestamp_s) "
            "VALUES (?, ?, ?, ?, ?)",
            rows_points,
        )

    return result
