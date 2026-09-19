# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prototype_fused_person_marker_assignment.py — PROTOTYPE: combines P7's
multi-camera triangulation fusion with P1's catalog-keypoint gating, doing
the one-to-one assignment in 3D instead of independently per 2D camera
view.

Scope for this iteration (deliberately partial -- see person-marker-
assignment-design.md's phased plan): only frames where a catalog marker's
anchor keypoint is triangulable (seen by >=2 cameras) get a 3D anchor to
gate against, and only fused (>=2-view) candidate points are assigned
against them. Genuinely single-view-only keypoints/candidates are not
handled by a 2D fallback pass yet -- this iteration is checking whether
the 3D-primary path itself recovers more multi-marker-joint coverage than
P1's per-camera-2D approach did, not yet trying to recover every last
single-view observation too.

Usage:
    python tools/prototype_fused_person_marker_assignment.py \\
        --session /path/to/session.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --marker-detection-run 01c2e3c1-204b-49f8-9b14-6fd611b765a4 \\
        --pose-sequence ec1b3e2f-1ef8-4e31-806c-33102a969ecd \\
        --start-time 40.0 --end-time 60.0
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.pose.db_cache import decode_dot_candidates  # noqa: E402
from app.setup.extrinsics_solver import _undistort_pts  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states, load_sync_table  # noqa: E402
from tools.prototype_multi_camera_fusion import fuse_frame, triangulate_multiview  # noqa: E402
from tools.prototype_person_marker_assignment import _CATALOG, _GROUP_SIZE, _UNAMBIGUOUS_GROUPS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--marker-detection-run", required=True)
    ap.add_argument("--pose-sequence", required=True)
    ap.add_argument("--start-time", type=float, required=True)
    ap.add_argument("--end-time", type=float, required=True)
    ap.add_argument("--fusion-max-reproj-px", type=float, default=3.0)
    ap.add_argument("--fusion-merge-radius-m", type=float, default=0.03)
    ap.add_argument("--match-radius-m", type=float, default=0.15)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    states = load_camera_states(conn, args.shot_id)
    sync_table, svid_by_cam = load_sync_table(conn, args.shot_id)
    cam_id_by_svid = {svid: cam for cam, svid in svid_by_cam.items()}

    identity_dists: dict[str, list[float]] = {}
    group_coverage_totals: dict[str, list[tuple[int, int]]] = {}
    n_frames = 0
    n_frames_with_3d = 0

    # Drive the loop off one reference camera's own frame indices, resolving
    # every other camera's frame at the same global time -- same pattern
    # render_tracking_debug_frames.py's render_grid_video() already uses.
    ref_cam = next(iter(states))
    ref_svid = svid_by_cam[ref_cam]
    frame_lo = sync_table.lookup(args.start_time, ref_svid)
    frame_hi = sync_table.lookup(args.end_time, ref_svid)

    for ref_frame in range(frame_lo, frame_hi):
        t = sync_table.frame_to_global_time(ref_frame, ref_svid)
        if t is None:
            continue

        dots_by_cam: dict[str, np.ndarray] = {}
        kp_by_cam: dict[str, np.ndarray] = {}
        for cam_id, svid in svid_by_cam.items():
            if cam_id not in states:
                continue
            frame_idx = sync_table.lookup(t, svid)
            if frame_idx is None:
                continue
            dot_row = conn.execute(
                "SELECT keypoints FROM detection_keypoints WHERE detection_run_id = ? AND shot_video_id = ? "
                "AND video_frame = ? AND region_type = 'dots'",
                (args.marker_detection_run, svid, frame_idx),
            ).fetchone()
            if dot_row is not None:
                dots = decode_dot_candidates(bytes(dot_row["keypoints"]))
                dots_by_cam[cam_id] = _undistort_pts(dots[:, :2], states[cam_id])
            pose_row = conn.execute(
                "SELECT kp_blob FROM pose_observations WHERE sequence_id = ? AND camera_instance_id = ? "
                "AND video_frame = ? AND source = 'body'",
                (args.pose_sequence, cam_id, frame_idx),
            ).fetchone()
            if pose_row is not None:
                kp = np.frombuffer(bytes(pose_row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
                kp_by_cam[cam_id] = kp

        if not dots_by_cam or not kp_by_cam:
            continue

        fused = fuse_frame(dots_by_cam, states, args.fusion_max_reproj_px, args.fusion_merge_radius_m)

        # Triangulate each catalog marker's anchor keypoint from every camera
        # that sees it (kp[:,2] > 0), when 2+ cameras agree on visibility --
        # single-camera-only anchors are out of scope this iteration (see
        # module docstring).
        marker_anchors: dict[str, np.ndarray] = {}
        for name, (idx, _group) in _CATALOG.items():
            views_P, views_pt = [], []
            for cam_id, kp in kp_by_cam.items():
                if kp[idx, 2] > 0:
                    pt = _undistort_pts(kp[idx:idx + 1, :2], states[cam_id])[0]
                    from app.setup.extrinsics_solver import _proj_matrix
                    views_P.append(_proj_matrix(states[cam_id]))
                    views_pt.append((pt[0], pt[1]))
            if len(views_P) >= 2:
                xyz = triangulate_multiview(views_P, views_pt)
                if xyz is not None:
                    marker_anchors[name] = xyz

        if not marker_anchors or not fused:
            n_frames += 1
            continue
        n_frames += 1
        n_frames_with_3d += 1

        names = list(marker_anchors)
        cost = np.full((len(names), len(fused)), 1e6)
        for ni, name in enumerate(names):
            for fi, fp in enumerate(fused):
                d = np.linalg.norm(marker_anchors[name] - fp.xyz)
                if d <= args.match_radius_m:
                    cost[ni, fi] = d
        row_ind, col_ind = linear_sum_assignment(cost)
        assigned = {names[ri]: (ci, cost[ri, ci]) for ri, ci in zip(row_ind, col_ind) if cost[ri, ci] < 1e6}

        for name, (_ci, dist) in assigned.items():
            if _CATALOG[name][1] in _UNAMBIGUOUS_GROUPS:
                identity_dists.setdefault(name, []).append(dist * 100)  # cm, more readable than meters
        by_group: dict[str, set[int]] = {}
        for name, (ci, _dist) in assigned.items():
            group = _CATALOG[name][1]
            if group not in _UNAMBIGUOUS_GROUPS:
                by_group.setdefault(group, set()).add(ci)
        for group, cand_set in by_group.items():
            group_coverage_totals.setdefault(group, []).append((len(cand_set), _GROUP_SIZE[group]))

    print(f"{n_frames} frames considered ({n_frames_with_3d} had both a 3D anchor and a fused candidate)\n")
    print("--- identity-accurate slots (unambiguous, single marker per joint) ---")
    for name in sorted(identity_dists):
        dists = identity_dists[name]
        print(f"  {name:12s} matched {len(dists)}/{n_frames} frames  "
              f"mean_dist={np.mean(dists):.1f}cm  median={np.median(dists):.1f}cm")

    print()
    print("--- group coverage (ambiguous, 2+ markers share one joint) ---")
    for group in sorted(group_coverage_totals):
        totals = group_coverage_totals[group]
        avg_covered = np.mean([n for n, _size in totals])
        size = totals[0][1]
        full_coverage_frac = np.mean([n == size for n, _size in totals])
        print(f"  {group:10s} avg {avg_covered:.2f}/{size} distinct candidates assigned per frame  "
              f"(all {size} found: {100*full_coverage_frac:.0f}% of frames)  n_frames={len(totals)}")


if __name__ == "__main__":
    main()
