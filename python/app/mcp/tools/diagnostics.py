"""Tools: get_filter_stats, get_observation_gaps."""

from __future__ import annotations

import sqlite3

import numpy as np

from app.mcp.db import (
    OBS_ACTUAL_X, OBS_ACTUAL_Y, OBS_OUTLIER, OBS_PRED_X, OBS_PRED_Y, OBS_USED,
    decode_obs_blob,
    get_run_cameras,
    get_run_markers,
    get_steps_in_range,
    marker_indices,
    short_label,
)

_NIS_HIGH = 1.5   # filter is overconfident (surprised by observations)
_NIS_LOW  = 0.3   # filter is very underconfident (accepting everything)
_COV_COND_WARN = 1_000_000
_GAP_HIGHLIGHT = 30  # pixels


def get_filter_stats(
    conn: sqlite3.Connection,
    run_id: str,
    start_s: float,
    end_s: float,
) -> str:
    rows = conn.execute(
        """SELECT tracker_step, timestamp_s, n_inlier_observations,
                  cov_condition_number, nis_value, nis_dof, tracking_lost
           FROM tracking_results
           WHERE run_id = ? AND person_id = 0 AND is_smoothed = 0
             AND timestamp_s BETWEEN ? AND ?
           ORDER BY tracker_step""",
        (run_id, start_s, end_s),
    ).fetchall()

    if not rows:
        return f"No tracking results found between {start_s}s and {end_s}s."

    # Anomaly detection
    high_nis: list[tuple[float, float]] = []  # (ts, nis_dof)
    low_nis: list[tuple[float, float]] = []
    ill_cond: list[tuple[float, float]] = []

    lines: list[str] = []
    lines.append(
        f"{'step':>5} {'ts':>7} | {'NIS/DOF':>8} {'n_inlier':>9} {'cov_cond':>12} | flags"
    )
    lines.append("-" * 58)

    for r in rows:
        nis_dof = r["nis_value"] / r["nis_dof"] if r["nis_dof"] else 0.0
        flags = []
        if r["tracking_lost"]:
            flags.append("LOST")
        if nis_dof > _NIS_HIGH:
            flags.append(f"NIS↑{nis_dof:.2f}")
            high_nis.append((r["timestamp_s"], nis_dof))
        elif nis_dof < _NIS_LOW and nis_dof > 0:
            flags.append(f"NIS↓{nis_dof:.2f}")
            low_nis.append((r["timestamp_s"], nis_dof))
        cond = r["cov_condition_number"] or 0
        if cond > _COV_COND_WARN:
            flags.append(f"cond={cond:.1e}")
            ill_cond.append((r["timestamp_s"], cond))

        lines.append(
            f"{r['tracker_step']:>5} {r['timestamp_s']:>7.3f} | "
            f"{nis_dof:>8.3f} {r['n_inlier_observations']:>9} {cond:>12.0f} | "
            f"{'  '.join(flags)}"
        )

    # Summary
    summary: list[str] = []
    if high_nis:
        ts_range = f"{high_nis[0][0]:.2f}s–{high_nis[-1][0]:.2f}s"
        summary.append(
            f"NIS/DOF > {_NIS_HIGH} at {len(high_nis)} steps ({ts_range}): "
            f"filter is OVERCONFIDENT — observations are surprising it. "
            f"The state has likely drifted from reality."
        )
    if low_nis:
        ts_range = f"{low_nis[0][0]:.2f}s–{low_nis[-1][0]:.2f}s"
        summary.append(
            f"NIS/DOF < {_NIS_LOW} at {len(low_nis)} steps ({ts_range}): "
            f"filter is very UNDERCONFIDENT — measurement_noise_std may be too large."
        )
    if ill_cond:
        ts_range = f"{ill_cond[0][0]:.2f}s–{ill_cond[-1][0]:.2f}s"
        summary.append(
            f"Covariance condition number > 1e6 at {len(ill_cond)} steps ({ts_range}): "
            f"ill-conditioned — depth direction may be underdetermined."
        )

    header = [
        f"Filter statistics: {start_s}s – {end_s}s  ({len(rows)} steps)",
        "",
    ]
    if summary:
        header.append("Anomaly summary:")
        for s in summary:
            header.append(f"  • {s}")
        header.append("")

    return "\n".join(header + lines)


def get_observation_gaps(
    conn: sqlite3.Connection,
    run_id: str,
    start_s: float,
    end_s: float,
    markers: list[str],
    stride: int = 3,
) -> str:
    """Show actual vs predicted pixel positions and the gap for specified markers.

    Gaps ≥ 30px are flagged with *. Only steps where at least one camera has
    an observation for the marker are shown.
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

    cam_labels = [short_label(cam_names[cid]) for cid in camera_ids]

    sections: list[str] = []
    sections.append(
        f"Observation gaps: {start_s}s – {end_s}s  (every {stride} steps, gaps ≥{_GAP_HIGHLIGHT}px flagged *)"
    )

    for mname in markers:
        midx = midx_map[mname]
        sections.append(f"\n=== {mname} ===")
        sections.append(
            f"{'step':>5} {'ts':>7} | {'camera':<10} {'act_x':>6} {'act_y':>6} "
            f"{'pred_x':>7} {'pred_y':>7} {'gap':>6} {'status'}"
        )
        sections.append("-" * 65)

        for i, (step, ts) in enumerate(steps):
            if i % stride != 0:
                continue
            obs_row = conn.execute(
                "SELECT obs_blob FROM tracking_obs_results "
                "WHERE run_id = ? AND person_id = 0 AND tracker_step = ?",
                (run_id, step),
            ).fetchone()
            if obs_row is None:
                continue

            blob = decode_obs_blob(obs_row["obs_blob"], n_cam, n_mrk)
            found_any = False
            for ci, cid in enumerate(camera_ids):
                ax = blob[ci, midx, OBS_ACTUAL_X]
                if np.isnan(ax):
                    continue
                found_any = True
                ay = blob[ci, midx, OBS_ACTUAL_Y]
                px = blob[ci, midx, OBS_PRED_X]
                py = blob[ci, midx, OBS_PRED_Y]
                used = blob[ci, midx, OBS_USED]
                outlier = blob[ci, midx, OBS_OUTLIER]
                gap = np.sqrt((ax - px) ** 2 + (ay - py) ** 2)
                flag = "*" if gap >= _GAP_HIGHLIGHT else " "
                status = "I" if used > 0.5 else ("x" if outlier > 0.5 else "?")

                sections.append(
                    f"{step:>5} {ts:>7.3f} | {cam_labels[ci]:<10} "
                    f"{ax:>6.0f} {ay:>6.0f} {px:>7.0f} {py:>7.0f} "
                    f"{gap:>5.0f}{flag} {status}"
                )

            if not found_any:
                sections.append(f"{step:>5} {ts:>7.3f} | (no observations)")

    return "\n".join(sections)
