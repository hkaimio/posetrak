# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prototype_track_pen_trajectory.py -- per-instant 6-DOF pen trajectory
across the real writing-scene footage, using the already-validated
calibration from this capture (see status.md, 2026-09-13):
  - marker 2 (reference) and marker 3's corners in marker 2's own local
    frame, from calibrate_rigid_marker_body.py's real-capture run.
  - the pen-tip reflective band's offset in marker 2's local frame, from
    prototype_pen_band_tracking.py's cross-time/cross-camera fit.

This is a standalone trajectory export, NOT the production object-tracking
path (ObjectPanel/ObjectRunTrackerDialog + the C++ UKF tracker) -- that
path exists and already works for a single object, but needs the pen's
ArUco/dot observations finalized into a pose_observation_sequences row and
a capture_objects entry first, neither of which exist yet for this capture
(a real integration task, not yet exercised for this specific object --
see marker-mocap-productization-plan.md §3.4/§4). This script instead
directly reuses the same per-bucket multi-camera solve_marker_pose()
mechanism already validated for the calibration itself, to get a real
trajectory now and validate the idea before that integration work.

For each time bucket in the requested window: solve marker 2's world pose
if it's visible (>=2 cameras); if only marker 3 is visible, recover
marker 2's pose via the already-known marker3-to-marker2 transform (from
the same calibration) instead of dropping the bucket -- this roughly
doubles trajectory coverage versus marker-2-only, since the two tags face
opposite directions and are rarely both hidden at once. Exports one CSV
row per solved bucket: marker-2 world position + quaternion, and the
tip band's derived world position.

Usage:
    python tools/prototype_track_pen_trajectory.py \\
        --session D:/mocap/2026-09-06-kare-tests/2026-09-06-kare-tests.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --time-start 98.0 --time-end 113.0 \\
        --marker-size 0.095 \\
        --output pen_trajectory.csv
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.setup.extrinsics_solver import _undistort_pts, solve_marker_pose  # noqa: E402
from app.setup.fiducial_markers import ArucoDetector  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states, load_sync_table  # noqa: E402

MARKER_IDS = {"2", "3", "1"}
REFERENCE_MARKER_ID = "2"

# Pad's own reference marker. NOT "0": the extrinsics calibration box carries
# a same-ID (DICT_4X4_50, id "0") ArUco tag on one face by mistake (Harri,
# 2026-09-13), and that box's tag dominates "0" sightings in every camera
# checked (near-frozen pixel position across the whole writing window) --
# the real pad's own tag-0 is only rarely glimpsed underneath that
# contamination. Marker "1" has no known ID conflict and shows clear,
# consistent real motion in every camera, so it -- not "0" -- is this
# script's pad anchor; "0" is never solved from the real capture at all.
PAD_REFERENCE_MARKER_ID = "1"

# From calibrate_rigid_marker_body.py's real-capture run (pen_body_with_bands2.yaml /
# the earlier pen_body_from_capture.yaml -- corners unchanged between the two, only
# the dot-detection side differed): marker 3's own corners in marker 2's local frame.
MARKER3_LOCAL_CORNERS = np.array([
    [0.033873, 0.046793, -0.021456],
    [-0.060842, 0.043939, -0.021080],
    [-0.057792, -0.050779, -0.019745],
    [0.036934, -0.047898, -0.019123],
])

# From prototype_pen_band_tracking.py: tip-band offset in marker 2's local frame.
TIP_LOCAL_OFFSET = np.array([0.30634122, 0.03274555, -0.00472446])


def _rotation_to_quat(m: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> quaternion (x,y,z,w), standard branch-on-trace formula."""
    tr = np.trace(m)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / S
        qx = 0.25 * S
        qy = (m[0, 1] + m[1, 0]) / S
        qz = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / S
        qx = (m[0, 1] + m[1, 0]) / S
        qy = 0.25 * S
        qz = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / S
        qx = (m[0, 2] + m[2, 0]) / S
        qy = (m[1, 2] + m[2, 1]) / S
        qz = 0.25 * S
    return qx, qy, qz, qw


def _kabsch(local_pts: np.ndarray, world_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid registration (rotation only) -- same closed-form SVD solve
    used throughout this project's other calibration/refit tools."""
    lc = local_pts.mean(axis=0)
    wc = world_pts.mean(axis=0)
    H = (local_pts - lc).T @ (world_pts - wc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = wc - R @ lc
    return R, t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--time-start", type=float, required=True)
    ap.add_argument("--time-end", type=float, required=True)
    ap.add_argument("--marker-size", type=float, required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--min-cameras", type=int, default=2)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    states = load_camera_states(conn, args.shot_id)
    sync_table, svid_by_cam = load_sync_table(conn, args.shot_id)
    detector = ArucoDetector(dictionary="DICT_4X4_50")

    frame_obs: dict[float, dict[str, dict[str, np.ndarray]]] = {}
    for cam_id, state in states.items():
        svid = svid_by_cam.get(cam_id)
        if svid is None:
            continue
        first = sync_table.lookup(args.time_start, svid)
        last = sync_table.lookup(args.time_end, svid)
        if first is None or last is None:
            print(f"  SKIP {cam_id[:8]}: no sync coverage")
            continue
        print(f"camera {cam_id[:8]}: frames {first}-{last} ({state.file_path})")
        n_decoded = 0
        for video_frame, img in iter_frames(state.file_path, first, last):
            n_decoded += 1
            if (video_frame - first) % args.stride != 0:
                continue
            dets = detector.detect(img, video_id=cam_id, frame_idx=video_frame)
            gt = sync_table.frame_to_global_time(video_frame, svid)
            if gt is None:
                continue
            bucket = round(gt / 0.05) * 0.05
            for d in dets:
                if d.marker_id not in MARKER_IDS:
                    continue
                pts = np.array([(c.px, c.py) for c in d.corners], dtype=np.float64)
                pts_undist = _undistort_pts(pts, state)
                frame_obs.setdefault(bucket, {}).setdefault(d.marker_id, {})[cam_id] = pts_undist
        print(f"  {n_decoded} frames decoded")

    rows_out = []
    n_via_marker2, n_via_marker3 = 0, 0
    n_pad_solved = [0]
    for bucket, by_marker in sorted(frame_obs.items()):
        R2, t2 = None, None
        obs2 = by_marker.get(REFERENCE_MARKER_ID)
        if obs2 is not None and len(obs2) >= args.min_cameras:
            try:
                rvec, tvec, _ = solve_marker_pose(obs2, states, args.marker_size)
                R2, _ = cv2.Rodrigues(rvec)
                t2 = tvec.flatten()
                n_via_marker2 += 1
            except (ValueError, RuntimeError):
                pass
        if R2 is None:
            obs3 = by_marker.get("3")
            if obs3 is not None and len(obs3) >= args.min_cameras:
                try:
                    rvec, tvec, _ = solve_marker_pose(obs3, states, args.marker_size)
                    R3, _ = cv2.Rodrigues(rvec)
                    t3 = tvec.flatten()
                    world_corners3 = (R3 @ MARKER3_LOCAL_CORNERS.T).T + t3
                    # marker-2-local corners are just the flat aruco template
                    # (its own local frame, size args.marker_size) -- reuse
                    # the same helper the calibration script itself used.
                    from app.setup.extrinsics_solver import marker_local_corners
                    local_corners2 = marker_local_corners(args.marker_size)
                    # We know marker-3's corners in marker-2 frame already;
                    # find (R2,t2) mapping marker-2-local -> world such that
                    # applying it to MARKER3_LOCAL_CORNERS reproduces
                    # world_corners3 -- i.e. rigid-register the two.
                    R2, t2 = _kabsch(MARKER3_LOCAL_CORNERS, world_corners3)
                    n_via_marker3 += 1
                except (ValueError, RuntimeError):
                    pass
        if R2 is None:
            continue
        tip_world = R2 @ TIP_LOCAL_OFFSET + t2
        qx, qy, qz, qw = _rotation_to_quat(R2)

        row = {
            "time_s": bucket,
            "marker2_x": t2[0], "marker2_y": t2[1], "marker2_z": t2[2],
            "marker2_qx": qx, "marker2_qy": qy, "marker2_qz": qz, "marker2_qw": qw,
            "tip_x": tip_world[0], "tip_y": tip_world[1], "tip_z": tip_world[2],
        }

        obs_pad = by_marker.get(PAD_REFERENCE_MARKER_ID)
        if obs_pad is not None and len(obs_pad) >= args.min_cameras:
            try:
                rvec_p, tvec_p, _ = solve_marker_pose(obs_pad, states, args.marker_size)
                Rp, _ = cv2.Rodrigues(rvec_p)
                tp = tvec_p.flatten()
                pqx, pqy, pqz, pqw = _rotation_to_quat(Rp)
                row.update({
                    "pad_x": tp[0], "pad_y": tp[1], "pad_z": tp[2],
                    "pad_qx": pqx, "pad_qy": pqy, "pad_qz": pqz, "pad_qw": pqw,
                })
                n_pad_solved[0] += 1
            except (ValueError, RuntimeError):
                pass

        rows_out.append(row)

    print(f"\n{len(rows_out)} buckets solved ({n_via_marker2} via marker 2 directly, "
          f"{n_via_marker3} via marker 3 + known transform)")
    print(f"pad (marker {PAD_REFERENCE_MARKER_ID}) solved in {n_pad_solved[0]} of those buckets")
    if rows_out:
        out_path = Path(args.output)
        fieldnames = [
            "time_s",
            "marker2_x", "marker2_y", "marker2_z",
            "marker2_qx", "marker2_qy", "marker2_qz", "marker2_qw",
            "tip_x", "tip_y", "tip_z",
            "pad_x", "pad_y", "pad_z",
            "pad_qx", "pad_qy", "pad_qz", "pad_qw",
        ]
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"Wrote {out_path}")
        times = [r["time_s"] for r in rows_out]
        print(f"time range covered: {min(times):.2f}s - {max(times):.2f}s")
        gaps = [b - a for a, b in zip(times, times[1:]) if b - a > 0.15]
        print(f"gaps > 0.15s in coverage: {len(gaps)} (largest: {max(gaps):.2f}s)" if gaps else "no gaps > 0.15s")


if __name__ == "__main__":
    main()
