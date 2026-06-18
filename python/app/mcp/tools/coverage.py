"""Tools: get_camera_coverage, get_edit_coverage."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict

import numpy as np

from app.mcp.db import (
    HALPE_NAMES,
    OBS_OUTLIER,
    OBS_USED,
    decode_kp_mask,
    decode_obs_blob,
    get_run_cameras,
    get_run_markers,
    get_steps_in_range,
    marker_indices,
    short_label,
)


def get_camera_coverage(
    conn: sqlite3.Connection,
    run_id: str,
    start_s: float,
    end_s: float,
    markers: list[str],
    stride: int = 4,
) -> str:
    """Per-step inlier/outlier/absent grid.

    I = inlier, x = outlier, · = no observation.
    Every `stride`-th step is shown; summary percentages shown per camera.
    """
    camera_ids, cam_names = get_run_cameras(conn, run_id)
    marker_names = get_run_markers(conn, run_id)
    n_cam = len(camera_ids)
    n_mrk = len(marker_names)

    try:
        midx_map = marker_indices(marker_names, markers)
    except ValueError as e:
        return str(e)

    steps = get_steps_in_range(conn, run_id, start_s, end_s)
    if not steps:
        return f"No steps found between {start_s}s and {end_s}s."

    _W = 8  # column width for camera labels and cell symbols
    cam_labels = [short_label(cam_names[cid]) for cid in camera_ids]

    # Accumulate: counts[marker][cam_idx] = {'I': n, 'x': n, '.': n}
    counts: dict[str, list[dict[str, int]]] = {
        m: [{"I": 0, "x": 0, ".": 0} for _ in camera_ids] for m in markers
    }
    shown_rows: list[str] = []

    # Header
    shown_rows.append(
        f"{'step':>5} {'ts':>7} | "
        + "  ".join(f"{'--- ' + m + ' ---':^{_W * n_cam - 1}}" for m in markers)
    )
    shown_rows.append(
        f"{'':>5} {'':>7} | "
        + "  ".join(" ".join(f"{lbl:>{_W}}" for lbl in cam_labels) for _ in markers)
    )
    shown_rows.append("-" * (16 + (_W * n_cam + 2) * len(markers)))

    for i, (step, ts) in enumerate(steps):
        obs_row = conn.execute(
            "SELECT obs_blob FROM tracking_obs_results "
            "WHERE run_id = ? AND person_id = 0 AND tracker_step = ?",
            (run_id, step),
        ).fetchone()
        if obs_row is None:
            continue

        blob = decode_obs_blob(obs_row["obs_blob"], n_cam, n_mrk)

        # Accumulate counts always
        for mname in markers:
            midx = midx_map[mname]
            for ci in range(n_cam):
                ax = blob[ci, midx, 0]
                if np.isnan(ax):
                    counts[mname][ci]["."] += 1
                elif blob[ci, midx, OBS_OUTLIER] > 0.5:
                    counts[mname][ci]["x"] += 1
                else:
                    counts[mname][ci]["I"] += 1

        if i % stride != 0:
            continue

        # Build row
        blocks = []
        for mname in markers:
            midx = midx_map[mname]
            cells = []
            for ci in range(n_cam):
                ax = blob[ci, midx, 0]
                if np.isnan(ax):
                    cells.append(".")
                elif blob[ci, midx, OBS_OUTLIER] > 0.5:
                    cells.append("x")
                else:
                    cells.append("I")
            blocks.append(" ".join(f"{c:>{_W}}" for c in cells))
        shown_rows.append(f"{step:>5} {ts:>7.3f} | " + "  ".join(blocks))

    # Summary percentages
    total_steps = len(steps)
    shown_rows.append("")
    shown_rows.append("Inlier % per camera (over shown window):")
    for mname in markers:
        shown_rows.append(f"  {mname}:")
        for ci, cid in enumerate(camera_ids):
            c = counts[mname][ci]
            n_obs = c["I"] + c["x"]
            pct_inlier = 100 * c["I"] / total_steps if total_steps else 0
            pct_present = 100 * n_obs / total_steps if total_steps else 0
            shown_rows.append(
                f"    {cam_names[cid]:<30} {pct_inlier:>5.1f}% inlier  "
                f"({pct_present:.1f}% present, {c['x']} outliers)"
            )

    return (
        f"Camera coverage: {', '.join(markers)} | {start_s}s – {end_s}s | "
        f"every {stride} steps\nI=inlier  x=outlier  .=absent\n\n"
        + "\n".join(shown_rows)
    )


def get_edit_coverage(conn: sqlite3.Connection, run_id: str) -> str:
    """Which keypoints are edited per camera, and — critically — which are NOT."""
    run = conn.execute(
        "SELECT observation_sequence_id, active_camera_ids FROM tracking_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    if run is None:
        return f"Run not found: {run_id}"

    seq_id = run["observation_sequence_id"]
    camera_ids, cam_names = get_run_cameras(conn, run_id)

    edits = conn.execute(
        "SELECT camera_instance_id, video_frame, kp_mask "
        "FROM pose_observation_edits WHERE sequence_id = ? "
        "ORDER BY camera_instance_id, video_frame",
        (seq_id,),
    ).fetchall()

    if not edits:
        return "No keypoint edits found for this run's observation sequence."

    # Group by camera
    by_cam: dict[str, list[tuple[int, bytes]]] = defaultdict(list)
    for e in edits:
        by_cam[e["camera_instance_id"]].append((e["video_frame"], e["kp_mask"]))

    # Key HALPE body-landmark indices worth tracking
    KEY_KPS = {
        "L_Hip": 11, "R_Hip": 12, "L_Knee": 13, "R_Knee": 14,
        "L_Ankle": 15, "R_Ankle": 16,
        "L_BigToe": 19, "R_BigToe": 20,
        "L_SmallToe": 21, "R_SmallToe": 22,
        "L_Heel": 23, "R_Heel": 24,
        "L_Shoulder": 5, "R_Shoulder": 6,
        "L_Elbow": 7, "R_Elbow": 8,
        "L_Wrist": 9, "R_Wrist": 10,
    }

    lines: list[str] = []
    lines.append(f"Keypoint edit coverage for run {run_id[:8]}…")
    lines.append(f"Observation sequence: {seq_id[:8]}…")
    lines.append("")

    # Determine which key KPs each camera has ANY edit for
    cam_edited_kps: dict[str, set[int]] = {}
    for cid in camera_ids:
        cam_label = cam_names[cid]
        frames_masks = by_cam.get(cid, [])
        if not frames_masks:
            cam_edited_kps[cid] = set()
            lines.append(f"  {cam_label:<30} — no edits")
            continue

        frames = [f for f, _ in frames_masks]
        all_edited: set[int] = set()
        for _, mask in frames_masks:
            all_edited.update(decode_kp_mask(mask))
        cam_edited_kps[cid] = all_edited

        frame_range = f"{min(frames)}–{max(frames)} ({len(frames)} frames)"

        if len(all_edited) == 133:
            edited_summary = "ALL keypoints (full-frame correction)"
        else:
            kp_names = [HALPE_NAMES.get(i, f"kp{i}") for i in sorted(all_edited) if i <= 24]
            extra = len([i for i in all_edited if i > 24])
            edited_summary = ", ".join(kp_names)
            if extra:
                edited_summary += f" + {extra} face/hand kps"

        lines.append(f"  {cam_label:<30}  frames {frame_range}")
        lines.append(f"    edited: {edited_summary}")

    # Flag key landmarks NOT edited in any camera that has observations for this run
    lines.append("")
    lines.append("Key body landmarks — edit status in active cameras:")
    lines.append(f"  {'KP':<15} | " + " | ".join(f"{short_label(cam_names[cid], 8):>8}" for cid in camera_ids))
    lines.append("  " + "-" * (17 + 13 * len(camera_ids)))

    for kp_name, kp_idx in KEY_KPS.items():
        row_parts = []
        for cid in camera_ids:
            if cid not in by_cam:
                row_parts.append("  —   ")
            elif kp_idx in cam_edited_kps.get(cid, set()) or 133 in {133} and len(cam_edited_kps.get(cid, set())) == 133:
                # Check if all 133 edited (full replacement) or specifically this KP
                if len(cam_edited_kps.get(cid, set())) == 133:
                    row_parts.append(" ALL  ")
                else:
                    row_parts.append(" yes  ")
            else:
                if cid in by_cam:
                    row_parts.append(" NO   ")  # has edits but not this KP
                else:
                    row_parts.append("  —   ")
        flag = ""
        unedited_active = [
            cam_names[cid] for cid in camera_ids
            if cid in by_cam and kp_idx not in cam_edited_kps.get(cid, set())
            and len(cam_edited_kps.get(cid, set())) < 133
        ]
        if unedited_active:
            flag = f"  ← not edited in: {', '.join(unedited_active)}"
        lines.append(f"  {kp_name:<15} | " + " | ".join(f"{p:>8}" for p in row_parts) + flag)

    return "\n".join(lines)
