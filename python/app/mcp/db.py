# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Shared database helpers for the MCP server.

Opens session DBs read-only and provides blob decoders so tools never
have to touch raw bytes.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import yaml

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
OBS_PAD = 7     # mode flag as of the hierarchical-solver feature -- see below

# OBS_PAD values, written by ResultWriter::write_obs_results() (always 0 --
# the "normal", single-stage tracking path) and
# ResultWriter::patch_obs_results() (a hierarchical-solver child stage's
# read-modify-write patch into a parent-owned run -- see
# docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md).
# A shared marker (e.g. a wrist both a body and a hand group solve) always
# keeps the PARENT's entry -- patch_obs_results() never overwrites it -- so
# OBS_MODE_PAIR_DIFF_RECONSTRUCTED only ever appears on a child stage's own
# markers, never on a marker also owned by the parent.
OBS_MODE_ABSOLUTE = 0.0
OBS_MODE_PAIR_DIFF_RECONSTRUCTED = 1.0


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


def get_run_stages(
    conn: sqlite3.Connection, run_id: str, person_id: int = 0
) -> list[sqlite3.Row]:
    """Return tracking_run_stages rows for a run/person, ordered by group_name.

    Empty list means this run is monolithic -- the existence-based
    hierarchical-mode toggle used throughout this feature (see
    hierarchical_solver.hpp: "a tracker_config_id with any tracker_config_stages
    rows runs hierarchically; one without runs monolithic"). Non-empty means
    the run has one or more child stages (e.g. HandL/HandR) merged into the
    same tracking_results/tracking_obs_results rows the parent wrote. Callers
    surfacing per-DOF or per-marker confidence on such a run must label
    parent-only scalars accordingly and consult COV_DIAG_HIERARCHICAL_CAVEAT.
    """
    return conn.execute(
        """SELECT group_name, status, started_at, completed_at
           FROM tracking_run_stages WHERE run_id = ? AND person_id = ?
           ORDER BY group_name""",
        (run_id, person_id),
    ).fetchall()


def get_marker_groups(conn: sqlite3.Connection, skeleton_id: str) -> dict[str, list[str]]:
    """Return marker name -> list of skeleton group names it belongs to.

    Parses the skeleton's own `groups:` YAML section (the same source
    python/tools/upgrade_skeleton_hand_groups.py and the C++ SkeletonGroup
    machinery read) -- a marker can belong to more than one group, e.g. a
    wrist marker shared between "main" and "HandL"/"HandR" (parent-wins per
    ResultWriter::patch_obs_results()). Returns {} if the skeleton has no
    groups: section at all (pre-hierarchical-solver skeletons).
    """
    row = conn.execute(
        "SELECT yaml_content FROM skeletons WHERE id = ?", (skeleton_id,)
    ).fetchone()
    if row is None:
        return {}
    skel = yaml.safe_load(row["yaml_content"])
    out: dict[str, list[str]] = {}
    for group in skel.get("groups") or []:
        for marker in group.get("markers") or []:
            out.setdefault(marker, []).append(group["name"])
    return out


# cov_diag on a hierarchical run (fixed 2026-07-23; see
# docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md):
# the parent stage expands cov_diag to full-skeleton error-state width at
# write time (SkeletonLayout::error_state_dim(), see db/session_schema.sql's
# tracking_results comment), filling every DOF it doesn't own with a
# placeholder variance derived from the run's tracker_configs.init_joint_std/
# init_velocity_std -- NOT a real per-DOF uncertainty. Each child stage then
# patches its own owned DOFs with its real solved covariance
# (hierarchical_solver.cpp's run_one_stage(), via
# SkeletonLayout::build_error_index_map_from()). So a DOF's cov_diag entry is
# only a real confidence value once that DOF's owning group's stage has
# completed -- check get_run_stages()' status for that group, don't infer it
# from the blob itself (a placeholder and a genuinely tiny real variance are
# not distinguishable by value alone). Any tool surfacing cov_diag-derived
# confidence should carry this caveat whenever get_run_stages() is non-empty
# for the run being inspected.
COV_DIAG_HIERARCHICAL_CAVEAT = (
    "cov_diag for a DOF on a hierarchical run is only a real confidence "
    "value once that DOF's owning group's stage status is 'complete' (see "
    "get_run_stages()) -- until then it still holds a placeholder variance "
    "from the run's own init_joint_std/init_velocity_std, not a real "
    "per-DOF uncertainty."
)


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
