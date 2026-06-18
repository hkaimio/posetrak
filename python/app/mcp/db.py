"""Shared database helpers for the MCP server.

Opens session DBs read-only and provides blob decoders so tools never
have to touch raw bytes.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# HALPE-133 keypoint index → name (body landmarks relevant for diagnostics)
# ---------------------------------------------------------------------------

HALPE_NAMES: dict[int, str] = {
    0: "nose", 1: "L_Eye", 2: "R_Eye", 3: "L_Ear", 4: "R_Ear",
    5: "L_Shoulder", 6: "R_Shoulder", 7: "L_Elbow", 8: "R_Elbow",
    9: "L_Wrist", 10: "R_Wrist",
    11: "L_Hip", 12: "R_Hip", 13: "L_Knee", 14: "R_Knee",
    15: "L_Ankle", 16: "R_Ankle",
    17: "head", 18: "neck",
    19: "L_BigToe", 20: "R_BigToe", 21: "L_SmallToe", 22: "R_SmallToe",
    23: "L_Heel", 24: "R_Heel",
}

# obs_blob field layout: float32[n_cam, n_mrk, 8]
OBS_ACTUAL_X = 0
OBS_ACTUAL_Y = 1
OBS_PRED_X = 2
OBS_PRED_Y = 3
OBS_MAHAL = 4
OBS_USED = 5    # 1.0 = inlier, 0.0 = outlier/absent
OBS_OUTLIER = 6
OBS_PAD = 7


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open a session DB in read-only mode (no migrations, no writes)."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Blob decoders
# ---------------------------------------------------------------------------

def decode_obs_blob(blob: bytes, n_cam: int, n_mrk: int) -> np.ndarray:
    """Decode tracking_obs_results.obs_blob → float32[n_cam, n_mrk, 8]."""
    return np.frombuffer(bytes(blob), dtype=np.float32).reshape(n_cam, n_mrk, 8)


def decode_kp_mask(mask_bytes: bytes, n_kps: int = 133) -> list[int]:
    """Decode pose_observation_edits.kp_mask bitmask → list of edited KP indices."""
    mask = np.frombuffer(bytes(mask_bytes), dtype=np.uint8)
    return [i for i in range(n_kps) if (mask[i // 8] >> (i % 8)) & 1]


def decode_extrinsics(R_bytes: bytes, t_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode extrinsic_entries blobs → (world_position, view_direction).

    world_position = -R^T t  (camera origin in world space)
    view_direction = R[2, :] (camera +Z axis in world space)
    """
    R = np.frombuffer(bytes(R_bytes), dtype=np.float64).reshape(3, 3)
    t = np.frombuffer(bytes(t_bytes), dtype=np.float64).reshape(3, 1)
    world_pos = (-R.T @ t).flatten()
    view_dir = R[2, :]
    return world_pos, view_dir


# ---------------------------------------------------------------------------
# Name resolution helpers
# ---------------------------------------------------------------------------

def resolve_camera_names(conn: sqlite3.Connection, camera_ids: list[str]) -> dict[str, str]:
    """Map camera_instance UUIDs → human-readable labels."""
    names: dict[str, str] = {}
    for cid in camera_ids:
        row = conn.execute(
            "SELECT label FROM camera_instances WHERE id = ?", (cid,)
        ).fetchone()
        names[cid] = row["label"] if row and row["label"] else cid[:8]
    return names


def short_label(name: str, max_len: int = 8) -> str:
    """Abbreviate a camera label for compact tables.

    For labels with a numeric suffix (e.g. gopro-11_mini_01) combines the
    penultimate segment with the stripped number (→ mini_01).
    Otherwise returns the first underscore-delimited segment, truncated.
    """
    parts = name.split("_")
    last = parts[-1]
    if last.isdigit() and len(parts) >= 2:
        candidate = f"{parts[-2]}_{last}"
        if len(candidate) <= max_len:
            return candidate
        # Truncate penultimate segment to fit
        num_chars = len(last) + 1  # _N
        return parts[-2][: max(1, max_len - num_chars)] + "_" + last
    return parts[0][:max_len]


def get_run_cameras(conn: sqlite3.Connection, run_id: str) -> tuple[list[str], dict[str, str]]:
    """Return (ordered camera UUIDs, uuid→label mapping) for a run.

    tracking_runs.active_camera_ids stores camera labels (not UUIDs).
    We resolve each label to the matching camera_instances.id UUID so that
    callers can join against extrinsic_entries, pose_observation_edits, etc.
    """
    row = conn.execute(
        "SELECT active_camera_ids FROM tracking_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Run not found: {run_id}")

    labels: list[str] = json.loads(row["active_camera_ids"])

    camera_ids: list[str] = []
    names: dict[str, str] = {}
    for label in labels:
        ci = conn.execute(
            "SELECT id FROM camera_instances WHERE label = ?", (label,)
        ).fetchone()
        uuid = ci["id"] if ci else label
        camera_ids.append(uuid)
        names[uuid] = label

    return camera_ids, names


def get_run_markers(conn: sqlite3.Connection, run_id: str) -> list[str]:
    """Return ordered marker name list for a run."""
    row = conn.execute(
        "SELECT marker_names FROM tracking_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Run not found: {run_id}")
    return json.loads(row["marker_names"])


def marker_indices(marker_names: list[str], targets: list[str]) -> dict[str, int]:
    """Map requested marker names → obs_blob column indices.

    Raises ValueError listing any names not found.
    """
    idx_map = {name: i for i, name in enumerate(marker_names)}
    missing = [t for t in targets if t not in idx_map]
    if missing:
        raise ValueError(
            f"Marker(s) not found in this run: {missing}. "
            f"Available: {marker_names}"
        )
    return {t: idx_map[t] for t in targets}


def get_steps_in_range(
    conn: sqlite3.Connection, run_id: str, start_s: float, end_s: float
) -> list[tuple[int, float]]:
    """Return [(tracker_step, timestamp_s)] between start_s and end_s (unsmoothed)."""
    rows = conn.execute(
        """SELECT tracker_step, timestamp_s FROM tracking_results
           WHERE run_id = ? AND person_id = 0 AND is_smoothed = 0
             AND timestamp_s BETWEEN ? AND ?
           ORDER BY tracker_step""",
        (run_id, start_s, end_s),
    ).fetchall()
    return [(r["tracker_step"], r["timestamp_s"]) for r in rows]
