#!/usr/bin/env python3
"""apply_seg_weighting.py — Create quality-weighted pose_observation_sequences.

For each pose_observation_sequences row linked to a detection_run, creates a
cloned sequence where each keypoint confidence is multiplied by its segmentation
quality score:

  new_conf = original_conf * quality_score

Quality scores come from ``keypoint_obs_quality`` (written by add_seg_quality.py):
  1.0  — keypoint clearly inside person mask → no change
  0.5  — boundary zone → confidence halved
  0.0  — outside mask → keypoint suppressed (confidence → 0)
 -1.0  — unavailable (no mask data) → treated as 1.0 (no change)

The resulting sequences can be used as inputs to the C++ tracker to evaluate
the effect of segmentation-based quality weighting without any C++ changes.

Usage
-----
::

    python apply_seg_weighting.py \\
        --db ~/projects/mocap_videos/ukemi-tommi-20260509.db \\
        --detection-run-id 8bfded7f-8f42-46a6-9ae8-c51a4f0dbd2d \\
        [--sequence-id SEQ_ID]   # if omitted, processes all sequences for the run
        [--dry-run]              # print summary without writing to DB
        [--name-suffix " [seg-weighted]"]
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sqlite3
import uuid
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

N_KEYPOINTS = 133  # RTMPose-133


# ---------------------------------------------------------------------------
# Blob codec (matches segmentation.py / add_seg_quality.py conventions)
# ---------------------------------------------------------------------------

def decode_kp_blob(blob: bytes) -> np.ndarray:
    """Decode pose_observations.kp_blob → (N_KP, 3) float32 [x, y, conf]."""
    arr = np.frombuffer(blob, dtype="<f4").copy()
    return arr.reshape(-1, 3)


def encode_kp_blob(arr: np.ndarray) -> bytes:
    """Encode (N_KP, 3) float32 → bytes."""
    return arr.astype("<f4").tobytes()


def decode_quality_blob(blob: bytes) -> np.ndarray:
    """Decode keypoint_obs_quality.quality_blob → (N_KP,) float32."""
    return np.frombuffer(blob, dtype="<f4").copy()


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def _build_camera_map(conn: sqlite3.Connection, shot_id: str) -> dict[str, str]:
    """Return {camera_instance_id → shot_video_id (capture_videos.id)} for a shot."""
    rows = conn.execute(
        "SELECT id, camera_instance_id FROM capture_videos WHERE shot_id = ?",
        (shot_id,),
    ).fetchall()
    return {row["camera_instance_id"]: row["id"] for row in rows}


def _build_person_map(conn: sqlite3.Connection, sequence_id: str) -> dict[int, str]:
    """Return {person_id → person_name} for a sequence."""
    rows = conn.execute(
        "SELECT person_id, person_name FROM sequence_persons WHERE sequence_id = ?",
        (sequence_id,),
    ).fetchall()
    return {row["person_id"]: row["person_name"] for row in rows}


def _build_track_lookup(
    conn: sqlite3.Connection, detection_run_id: str, shot_video_id: str
) -> list[tuple[int, int, int, str]]:
    """Return list of (first_frame, last_frame, track_id, person_name) for a video."""
    rows = conn.execute(
        """SELECT first_frame, last_frame, track_id, person_name
           FROM detection_track_assignments
           WHERE detection_run_id = ? AND shot_video_id = ?
           ORDER BY first_frame""",
        (detection_run_id, shot_video_id),
    ).fetchall()
    return [(r["first_frame"], r["last_frame"], r["track_id"], r["person_name"]) for r in rows]


def _find_track_id(
    track_lookup: list[tuple[int, int, int, str]], person_name: str, frame: int
) -> int | None:
    """Find track_id for person_name at the given frame."""
    for first, last, tid, name in track_lookup:
        if name == person_name and first <= frame <= last:
            return tid
    return None


def _build_quality_cache(
    conn: sqlite3.Connection, detection_run_id: str, shot_video_id: str
) -> dict[tuple[int, int], np.ndarray]:
    """Load all quality blobs for (shot_video_id) across all seg_quality_runs.

    Returns {(video_frame, track_id) → quality_arr (N_KP,)}.
    Uses the most recently created seg_run entry when multiple exist for the
    same (video_frame, track_id) (rare due to re-run fragmentation).
    """
    # Get seg_quality_runs for this detection_run, ordered so latest is last
    seg_runs = conn.execute(
        """SELECT id FROM seg_quality_runs
           WHERE detection_run_id = ?
           ORDER BY created_at""",
        (detection_run_id,),
    ).fetchall()
    seg_run_ids = [r["id"] for r in seg_runs]
    if not seg_run_ids:
        return {}

    placeholders = ",".join("?" * len(seg_run_ids))
    rows = conn.execute(
        f"""SELECT video_frame, track_id, quality_blob
            FROM keypoint_obs_quality
            WHERE seg_run_id IN ({placeholders})
              AND shot_video_id = ?
            ORDER BY seg_run_id""",
        (*seg_run_ids, shot_video_id),
    ).fetchall()

    cache: dict[tuple[int, int], np.ndarray] = {}
    for row in rows:
        key = (row["video_frame"], row["track_id"])
        cache[key] = decode_quality_blob(row["quality_blob"])
    return cache


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def apply_quality_weighting(
    conn: sqlite3.Connection,
    sequence_id: str,
    detection_run_id: str,
    name_suffix: str,
    dry_run: bool,
) -> str | None:
    """Clone a pose_observation_sequence with quality-weighted confidences.

    Returns the new sequence_id (or None on dry-run / failure).
    """
    seq = conn.execute(
        "SELECT * FROM pose_observation_sequences WHERE id = ?", (sequence_id,)
    ).fetchone()
    if seq is None:
        log.error("Sequence %s not found", sequence_id)
        return None

    shot_id = seq["shot_id"]
    camera_map = _build_camera_map(conn, shot_id)
    person_map = _build_person_map(conn, sequence_id)

    old_name = seq["name"] or ""
    new_name = old_name + name_suffix

    # Pre-load quality data for all cameras we'll encounter
    # camera_instance_id → shot_video_id
    quality_caches: dict[str, dict] = {}  # shot_video_id → cache
    track_lookups: dict[str, list] = {}   # shot_video_id → list

    for cam_inst_id, svid in camera_map.items():
        quality_caches[svid] = _build_quality_cache(conn, detection_run_id, svid)
        track_lookups[svid] = _build_track_lookup(conn, detection_run_id, svid)

    # Count stats
    total = 0
    weighted = 0
    outside = 0
    missing_quality = 0

    # Collect new rows
    new_obs_rows = []

    obs_rows = conn.execute(
        """SELECT camera_instance_id, video_frame, timestamp_s, person_id, kp_blob, noise_scale
           FROM pose_observations WHERE sequence_id = ?
           ORDER BY camera_instance_id, video_frame, person_id""",
        (sequence_id,),
    ).fetchall()

    for row in obs_rows:
        cam_inst_id = row["camera_instance_id"]
        video_frame = row["video_frame"]
        person_id = row["person_id"]
        person_name = person_map.get(person_id)

        shot_video_id = camera_map.get(cam_inst_id)
        kp = decode_kp_blob(row["kp_blob"])  # (N_KP, 3)

        quality_arr = None
        if shot_video_id and person_name:
            track_lookup = track_lookups.get(shot_video_id, [])
            tid = _find_track_id(track_lookup, person_name, video_frame)
            if tid is not None:
                cache = quality_caches.get(shot_video_id, {})
                quality_arr = cache.get((video_frame, tid))

        if quality_arr is None:
            missing_quality += 1
            # Keep original kp unchanged
            new_kp = kp
        else:
            # Clamp -1.0 (UNAVAILABLE) → 1.0 (no change)
            q = np.where(quality_arr < 0.0, 1.0, quality_arr).astype(np.float32)
            new_kp = kp.copy()
            new_kp[:, 2] *= q  # multiply confidence column by quality
            n_outside = int(np.sum(q == 0.0))
            outside += n_outside
            weighted += 1

        total += 1
        new_obs_rows.append((
            row["camera_instance_id"],
            row["video_frame"],
            row["timestamp_s"],
            row["person_id"],
            encode_kp_blob(new_kp),
            row["noise_scale"],
        ))

    log.info("Sequence %s: %d obs rows, %d quality-weighted, %d outside-kps, %d missing-quality",
             sequence_id, total, weighted, outside, missing_quality)

    if dry_run:
        log.info("  DRY RUN — not writing to DB")
        return None

    new_seq_id = str(uuid.uuid4())
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    conn.execute(
        """INSERT INTO pose_observation_sequences
           (id, shot_id, sync_config_id, time_start_s, time_end_s,
            name, pose_model, notes, pixels_are_undistorted, detection_run_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            new_seq_id,
            seq["shot_id"],
            seq["sync_config_id"],
            seq["time_start_s"],
            seq["time_end_s"],
            new_name,
            seq["pose_model"],
            f"Quality-weighted clone of {sequence_id} (created {now})",
            seq["pixels_are_undistorted"],
            seq["detection_run_id"],
        ),
    )

    # Copy sequence_persons
    for pid, pname in person_map.items():
        conn.execute(
            "INSERT INTO sequence_persons (sequence_id, person_id, person_name) VALUES (?,?,?)",
            (new_seq_id, pid, pname),
        )

    # Insert new obs rows
    conn.executemany(
        """INSERT INTO pose_observations
           (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id,
            kp_blob, noise_scale)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [(new_seq_id, *r) for r in new_obs_rows],
    )

    conn.commit()
    log.info("  Created new sequence: %s  name=%r", new_seq_id, new_name)
    return new_seq_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Path to SQLite DB")
    parser.add_argument("--detection-run-id", required=True,
                        help="Detection run UUID to process")
    parser.add_argument("--sequence-id", default=None,
                        help="Process only this pose_observation_sequences ID")
    parser.add_argument("--name-suffix", default=" [seg-weighted]",
                        help="Suffix appended to new sequence name (default: ' [seg-weighted]')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print statistics without writing to DB")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        log.error("DB not found: %s", db_path)
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Find sequences to process
    if args.sequence_id:
        seq_ids = [args.sequence_id]
    else:
        rows = conn.execute(
            "SELECT id FROM pose_observation_sequences WHERE detection_run_id = ?",
            (args.detection_run_id,),
        ).fetchall()
        seq_ids = [r["id"] for r in rows]

    if not seq_ids:
        log.warning("No pose_observation_sequences found for detection_run_id=%s",
                    args.detection_run_id)
        return

    log.info("Processing %d sequence(s) for detection_run %s",
             len(seq_ids), args.detection_run_id)

    created = []
    for sid in seq_ids:
        new_id = apply_quality_weighting(
            conn, sid, args.detection_run_id, args.name_suffix, args.dry_run
        )
        if new_id:
            created.append(new_id)

    if created:
        log.info("Done. Created %d new sequence(s):", len(created))
        for nid in created:
            log.info("  %s", nid)
    elif args.dry_run:
        log.info("Done (dry-run, nothing written).")


if __name__ == "__main__":
    main()
