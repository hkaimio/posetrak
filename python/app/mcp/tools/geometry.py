# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tool: get_camera_geometry — 3D positions, viewing directions, parallax table."""

from __future__ import annotations

import sqlite3

import numpy as np

from app.mcp.db import decode_extrinsics, get_run_cameras

_GOOD_ANGLE = 90.0    # degrees — cameras face mostly opposite directions
_POOR_ANGLE = 45.0    # degrees — cameras face similar directions


def get_camera_geometry(conn: sqlite3.Connection, run_id: str) -> str:
    camera_ids, cam_names = get_run_cameras(conn, run_id)

    # Get latest extrinsic calibration
    calib = conn.execute(
        "SELECT id FROM extrinsic_calibrations ORDER BY calibrated_at DESC LIMIT 1"
    ).fetchone()
    if calib is None:
        return "No extrinsic calibration found in this database."

    entries = conn.execute(
        "SELECT camera_instance_id, R, t FROM extrinsic_entries "
        "WHERE extrinsic_calibration_id = ?",
        (calib["id"],),
    ).fetchall()

    # Build geometry per camera
    cam_geo: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for e in entries:
        cid = e["camera_instance_id"]
        if cid not in set(camera_ids):
            continue
        pos, view = decode_extrinsics(e["R"], e["t"])
        cam_geo[cid] = (pos, view)

    if not cam_geo:
        return "No extrinsic data found for the active cameras in this run."

    lines: list[str] = []
    lines.append("Camera positions and viewing directions (world space, metres):")
    lines.append(f"  {'camera':<30} {'x':>8} {'y':>8} {'z':>8}   view_dir (x, y, z)")
    lines.append("  " + "-" * 72)
    for cid in camera_ids:
        if cid not in cam_geo:
            lines.append(f"  {cam_names[cid]:<30}  (no extrinsic data)")
            continue
        pos, view = cam_geo[cid]
        lines.append(
            f"  {cam_names[cid]:<30} {pos[0]:>8.3f} {pos[1]:>8.3f} {pos[2]:>8.3f}"
            f"   ({view[0]:>6.3f}, {view[1]:>6.3f}, {view[2]:>6.3f})"
        )

    lines.append("")
    lines.append("Pairwise parallax (baseline & angle between viewing directions):")
    lines.append(
        f"  {'pair':<45} {'baseline':>10} {'angle':>8}   depth_quality"
    )
    lines.append("  " + "-" * 80)

    cam_list = [cid for cid in camera_ids if cid in cam_geo]
    for i in range(len(cam_list)):
        for j in range(i + 1, len(cam_list)):
            cid_a, cid_b = cam_list[i], cam_list[j]
            pos_a, view_a = cam_geo[cid_a]
            pos_b, view_b = cam_geo[cid_b]

            baseline = float(np.linalg.norm(pos_a - pos_b))
            cos_angle = float(np.clip(np.dot(view_a, view_b), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos_angle)))

            if angle_deg >= _GOOD_ANGLE:
                quality = "GOOD  — near-opposite views, strong depth constraint"
            elif angle_deg >= _POOR_ANGLE:
                quality = "MODERATE"
            else:
                quality = "POOR  — similar viewing direction, weak depth constraint"

            pair_label = f"{cam_names[cid_a]} ↔ {cam_names[cid_b]}"
            lines.append(
                f"  {pair_label:<45} {baseline:>9.2f}m {angle_deg:>7.1f}°   {quality}"
            )

    lines.append("")
    lines.append(
        "Note: 'depth constraint' describes how well this camera pair resolves "
        "3-D position along the viewing axis. When the ONLY cameras with "
        "observations for a marker all have POOR pairwise angles, 3-D depth "
        "is underdetermined and the filter may drift."
    )

    return "\n".join(lines)
