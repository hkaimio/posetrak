"""Tools: list_tracking_runs, get_run_info."""

from __future__ import annotations

import json
import sqlite3

from app.mcp.db import get_run_cameras, get_run_markers


# Noise values considered suspiciously large (worth flagging in get_run_info)
_NOISE_WARN_THRESHOLD = 25.0


def list_tracking_runs(conn: sqlite3.Connection) -> str:
    runs = conn.execute(
        """SELECT tr.id, tr.ran_at, tr.notes,
                  sk.name AS skeleton_name,
                  MIN(tres.timestamp_s) AS t_start,
                  MAX(tres.timestamp_s) AS t_end
           FROM tracking_runs tr
           LEFT JOIN skeletons sk ON tr.skeleton_id = sk.id
           LEFT JOIN tracking_results tres
                  ON tres.run_id = tr.id AND tres.is_smoothed = 0
           GROUP BY tr.id
           ORDER BY tr.ran_at DESC"""
    ).fetchall()

    if not runs:
        return "No tracking runs found in this database."

    lines = ["Tracking runs (newest first):\n"]
    for r in runs:
        t_start = f"{r['t_start']:.1f}s" if r["t_start"] is not None else "?"
        t_end = f"{r['t_end']:.1f}s" if r["t_end"] is not None else "?"
        notes = f"  notes: {r['notes']}" if r["notes"] else ""
        lines.append(
            f"  {r['id']}\n"
            f"    ran_at:   {r['ran_at']}\n"
            f"    skeleton: {r['skeleton_name']}\n"
            f"    time:     {t_start} – {t_end}{notes}\n"
        )

    # Persons per run
    for r in runs:
        persons = conn.execute(
            """SELECT sp.person_name FROM tracking_run_persons trp
               JOIN sequence_persons sp
                 ON sp.sequence_id = (
                     SELECT observation_sequence_id FROM tracking_runs WHERE id = trp.run_id
                 )
                 AND sp.person_id = trp.person_id
               WHERE trp.run_id = ?""",
            (r["id"],),
        ).fetchall()
        if persons:
            names = ", ".join(p["person_name"] for p in persons if p["person_name"])
            # Append person name to the corresponding block
            lines = [ln.replace(r["id"] + "\n", r["id"] + f"\n    persons:  {names}\n", 1) for ln in lines]

    return "".join(lines)


def get_run_info(conn: sqlite3.Connection, run_id: str) -> str:
    run = conn.execute(
        "SELECT * FROM tracking_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if run is None:
        return f"Run not found: {run_id}"

    # Time range
    trange = conn.execute(
        """SELECT MIN(timestamp_s) AS t0, MAX(timestamp_s) AS t1,
                  COUNT(*) AS n_steps
           FROM tracking_results
           WHERE run_id = ? AND is_smoothed = 0""",
        (run_id,),
    ).fetchone()

    # Smoothing
    n_smoothed = conn.execute(
        "SELECT COUNT(*) FROM tracking_results WHERE run_id = ? AND is_smoothed = 1",
        (run_id,),
    ).fetchone()[0]

    # Tracker config
    cfg = conn.execute(
        "SELECT * FROM tracker_configs WHERE id = ?", (run["tracker_config_id"],)
    ).fetchone()

    # Skeleton
    sk = conn.execute(
        "SELECT name FROM skeletons WHERE id = ?", (run["skeleton_id"],)
    ).fetchone()

    # Cameras
    camera_ids, cam_names = get_run_cameras(conn, run_id)

    # Markers
    marker_names = get_run_markers(conn, run_id)

    # Persons
    persons = conn.execute(
        """SELECT sp.person_id, sp.person_name
           FROM tracking_run_persons trp
           JOIN sequence_persons sp
             ON sp.sequence_id = (
                 SELECT observation_sequence_id FROM tracking_runs WHERE id = trp.run_id
             )
             AND sp.person_id = trp.person_id
           WHERE trp.run_id = ?""",
        (run_id,),
    ).fetchall()

    lines: list[str] = []
    lines.append(f"Run: {run_id}")
    lines.append(f"Ran at:   {run['ran_at']}")
    lines.append(f"Skeleton: {sk['name'] if sk else run['skeleton_id']}")
    if trange and trange["t0"] is not None:
        lines.append(
            f"Time:     {trange['t0']:.3f}s – {trange['t1']:.3f}s  "
            f"({trange['n_steps']} steps)"
        )
    lines.append(f"Smoothed: {'yes' if n_smoothed > 0 else 'no'}")
    if persons:
        for p in persons:
            lines.append(f"Person:   {p['person_name']} (id={p['person_id']})")

    lines.append("")
    lines.append("Tracker config:")
    if cfg:
        noise = cfg["measurement_noise_std"]
        noise_warn = "  ← large! drift may go undetected" if noise and noise > _NOISE_WARN_THRESHOLD else ""
        threshold = cfg["outlier_threshold"]
        if noise and threshold:
            max_accepted = noise * threshold
            lines.append(f"  measurement_noise_std:  {noise} px{noise_warn}")
            lines.append(f"  outlier_threshold:      {threshold} σ  (accepts up to {max_accepted:.0f} px)")
        else:
            lines.append(f"  measurement_noise_std:  {noise}")
            lines.append(f"  outlier_threshold:      {threshold}")
        lines.append(f"  process_noise_std:      {cfg['process_noise_std']} rad/s²")
        lines.append(f"  process_noise_vel_std:  {cfg['process_noise_vel_std']}")
        lines.append(f"  velocity_half_life_s:   {cfg['velocity_half_life_s']} s")
        lines.append(f"  tracker_fps:            {cfg['tracker_fps']} Hz")
    else:
        lines.append("  (config not found)")

    lines.append("")
    lines.append(f"Active cameras ({len(camera_ids)}):")
    for cid in camera_ids:
        lines.append(f"  {cam_names[cid]:<30} {cid[:8]}")

    lines.append("")
    lines.append(f"Skeleton markers ({len(marker_names)} total):")
    # Show all, grouped: highlight legs
    leg_kws = {"hip", "knee", "ankle", "toe", "heel"}
    leg_markers = [(i, n) for i, n in enumerate(marker_names)
                   if any(kw in n.lower() for kw in leg_kws)]
    other_markers = [(i, n) for i, n in enumerate(marker_names)
                     if not any(kw in n.lower() for kw in leg_kws)]

    if leg_markers:
        lines.append("  Leg/hip markers (for diagnostics):")
        for idx, name in leg_markers:
            lines.append(f"    [{idx:>3}] {name}")
    lines.append("  Other markers:")
    for idx, name in other_markers[:20]:
        lines.append(f"    [{idx:>3}] {name}")
    if len(other_markers) > 20:
        lines.append(f"    ... ({len(other_markers) - 20} more)")

    return "\n".join(lines)
