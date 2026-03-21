"""import_pose_json.py — Import 2-D pose observations from OpenPose/Halpe JSON files.

Pose JSON files follow the structure produced by the posetrak pipeline::

    {
        "version": 1.3,
        "people": [
            {
                "person_id": [0],
                "pose_keypoints_2d": [x, y, conf, x, y, conf, ...]
            }
        ]
    }

Files are located at ``{pose_dir}/{cam_key}/{cam_key}_{frame:06d}.json``.

Timestamps are computed from the sync anchor stored in ``sync_points``:

    timestamp = ref_timestamp + (frame - ref_frame) / fps

where ``ref_frame``/``ref_timestamp`` come from ``sync_points`` and ``fps``
comes from ``shot_videos``.

Observations are inserted in batches of 500 rows per transaction to avoid
overly large single transactions.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.db.posetrak_db import generate_id

_BATCH_SIZE = 500


@dataclass
class PoseImportResult:
    """Result of a pose JSON import operation.

    Attributes
    ----------
    sequence_id:
        ID of the newly created ``pose_observation_sequences`` row.
    n_observations:
        Total number of ``pose_observations`` rows inserted.
    skipped_cameras:
        Set of camera keys that were skipped (not in *camera_instances* mapping).
    """

    sequence_id: str = ""
    n_observations: int = 0
    skipped_cameras: set[str] = field(default_factory=set)


def import_pose_json(
    session: sqlite3.Connection,
    shot_id: str,
    sync_config_id: str,
    pose_dir: Path,
    camera_instances: str | dict[str, str],
    *,
    person_id: int = 0,
    time_start: float | None = None,
    time_end: float | None = None,
    pose_model: str = "",
    notes: str = "",
) -> PoseImportResult:
    """Import 2-D pose keypoint observations from a directory of JSON files.

    For each listed camera, this function discovers all frame JSON files in
    ``{pose_dir}/{cam_key}/``, computes a timestamp for each frame using the
    sync anchor, applies the optional time window filter, and inserts
    ``pose_observations`` rows in batches of 500.

    A single ``pose_observation_sequences`` row is created covering the full
    imported time range (``time_start_s`` = minimum timestamp,
    ``time_end_s`` = maximum timestamp of all inserted rows).

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    shot_id:
        ID of the ``shots`` row these observations belong to.
    sync_config_id:
        ID of the ``sync_configs`` row that provides the time anchor.
    pose_dir:
        Root directory containing per-camera subdirectories of pose JSON files.
    camera_instances:
        Camera instance assignment. Two forms:

        - **Homogeneous** (``str``): a single ``camera_instances.id`` UUID
          applied to every camera directory found under *pose_dir*.
        - **Per-camera** (``dict[str, str]``): mapping from camera key
          (e.g. ``"cam1"``) to ``camera_instances.id``. Keys not in the
          mapping are skipped.
    person_id:
        Index into the ``people`` list to select; rows where the person is
        absent are skipped (default ``0``).
    time_start:
        If provided, only observations with ``timestamp >= time_start`` are
        imported.
    time_end:
        If provided, only observations with ``timestamp <= time_end`` are
        imported.
    pose_model:
        Optional identifier of the pose estimation model (e.g. ``"halpe133"``).
    notes:
        Optional free-text notes stored with the sequence row.

    Returns
    -------
    PoseImportResult
        Sequence ID, total observation count, and skipped camera keys.
    """
    result = PoseImportResult()

    # Discover all camera dirs present in pose_dir.
    available_cam_dirs: set[str] = set()
    if pose_dir.exists():
        available_cam_dirs = {d.name for d in pose_dir.iterdir() if d.is_dir()}

    # Determine which camera keys to process and which to skip.
    if isinstance(camera_instances, dict):
        cam_keys = sorted(camera_instances.keys())
        # Mark dirs that exist in pose_dir but are not in the mapping as skipped.
        for dk in available_cam_dirs:
            if dk not in camera_instances:
                result.skipped_cameras.add(dk)
    else:
        # Homogeneous: process all discovered camera dirs.
        cam_keys = sorted(available_cam_dirs)

    # Build per-camera sync info: {cam_key: (ref_frame, ref_timestamp, fps)}
    sync_info: dict[str, tuple[int, float, float]] = {}

    for cam_key in cam_keys:
        if isinstance(camera_instances, dict):
            if cam_key not in camera_instances:
                result.skipped_cameras.add(cam_key)
                continue
            instance_id = camera_instances[cam_key]
        else:
            instance_id = camera_instances

        sp_row = session.execute(
            "SELECT video_frame, timestamp_s FROM sync_points "
            "WHERE sync_config_id = ? AND camera_instance_id = ?",
            (sync_config_id, instance_id),
        ).fetchone()

        sv_row = session.execute(
            "SELECT actual_fps FROM shot_videos "
            "WHERE shot_id = ? AND camera_instance_id = ?",
            (shot_id, instance_id),
        ).fetchone()

        if sp_row is None or sv_row is None:
            # Cannot compute timestamps without sync info; skip this camera.
            result.skipped_cameras.add(cam_key)
            continue

        sync_info[cam_key] = (
            int(sp_row["video_frame"]),
            float(sp_row["timestamp_s"]),
            float(sv_row["actual_fps"]),
        )

    # Collect all observation rows before creating the sequence row.
    # (We need to know time_start_s / time_end_s for the sequence.)
    all_rows: list[tuple[str, str, int, float, int, bytes]] = []
    sequence_id = generate_id()

    for cam_key in sorted(sync_info.keys()):
        if isinstance(camera_instances, dict):
            instance_id = camera_instances[cam_key]
        else:
            instance_id = camera_instances

        ref_frame, ref_ts, fps = sync_info[cam_key]
        cam_dir = pose_dir / cam_key

        if not cam_dir.exists():
            continue

        json_files = sorted(cam_dir.glob(f"{cam_key}_*.json"))

        for json_path in json_files:
            stem = json_path.stem  # e.g. "cam1_000042"
            frame_str = stem[len(cam_key) + 1:]
            try:
                frame_num = int(frame_str)
            except ValueError:
                continue

            timestamp = ref_ts + (frame_num - ref_frame) / fps

            # Apply time window filter.
            if time_start is not None and timestamp < time_start:
                continue
            if time_end is not None and timestamp > time_end:
                continue

            with json_path.open(encoding="utf-8") as fh:
                data = json.load(fh)

            people = data.get("people", [])
            # Find the entry matching person_id.
            kp_flat: list[float] | None = None
            for person_entry in people:
                pid_val = person_entry.get("person_id", [])
                pid = pid_val[0] if isinstance(pid_val, list) and pid_val else pid_val
                if int(pid) == person_id:
                    kp_flat = person_entry["pose_keypoints_2d"]
                    break

            if kp_flat is None:
                continue

            kp_blob = np.array(kp_flat, dtype=np.float32).reshape(-1, 3).tobytes()

            all_rows.append(
                (sequence_id, instance_id, frame_num, timestamp, person_id, kp_blob)
            )

    # Determine actual time range from imported observations.
    if all_rows:
        timestamps = [r[3] for r in all_rows]
        seq_time_start = min(timestamps)
        seq_time_end = max(timestamps)
    else:
        seq_time_start = time_start if time_start is not None else 0.0
        seq_time_end = time_end if time_end is not None else 0.0

    # Insert sequence row.
    with session:
        session.execute(
            "INSERT INTO pose_observation_sequences "
            "(id, shot_id, sync_config_id, time_start_s, time_end_s, pose_model, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sequence_id, shot_id, sync_config_id, seq_time_start, seq_time_end,
             pose_model, notes),
        )

    # Insert observations in batches of _BATCH_SIZE.
    for batch_start in range(0, len(all_rows), _BATCH_SIZE):
        batch = all_rows[batch_start: batch_start + _BATCH_SIZE]
        with session:
            session.executemany(
                "INSERT INTO pose_observations "
                "(sequence_id, camera_instance_id, video_frame, timestamp_s, "
                "person_id, kp_blob) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                batch,
            )

    result.sequence_id = sequence_id
    result.n_observations = len(all_rows)
    return result
