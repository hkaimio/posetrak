# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prototype_person_marker_assignment.py — PROTOTYPE (no production code
touched): phase P1 of person-marker-assignment-design.md's phased plan.

Extends prototype_marker_person_filter.py's validated one-to-one
(Hungarian) match from the 10 pose-keypoint-anchored slots it used (one
marker assumed per keypoint) to the *real* marker catalog for this rig --
16 leg markers, several sharing one joint (3 at a knee: medial/lateral/
front; 2 at an ankle: medial/lateral).

Honest scope limit (see person-marker-assignment-design.md §3, phase P1's
own row): 2D proximity to a shared anchor keypoint cannot yet tell WHICH
same-joint marker is which -- that needs P2's marker-normal-based
backface culling, or real 3D. So this script reports two different kinds
of number and does not conflate them:

- **identity accuracy** for the 6 *unambiguous* single-marker slots (both
  hips, both heels, both toes) -- a real, trustworthy per-marker match.
- **group coverage** for the 4 *ambiguous* multi-marker joints (both
  knees, both ankles) -- how many distinct real candidates the group as a
  whole picked up, out of how many named markers share that joint, with
  no claim about which specific candidate is which specific marker.

Usage:
    python tools/prototype_person_marker_assignment.py \\
        --session D:/mocap/2026-09-06-kare-tests/2026-09-06-kare-tests.db \\
        --marker-detection-run 01c2e3c1-204b-49f8-9b14-6fd611b765a4 \\
        --pose-sequence ec1b3e2f-1ef8-4e31-806c-33102a969ecd \\
        --camera-label gopro-11_mini_01 \\
        --start-time 40.0 --end-time 45.0
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pose.db_cache import decode_dot_candidates  # noqa: E402

# name -> (vitpose-l-133kp anchor index, group). See module docstring for
# which groups are ambiguous (2+ names sharing an anchor) vs. not.
_CATALOG: dict[str, tuple[int, str]] = {
    "hip_L": (11, "hip_L"), "hip_R": (12, "hip_R"),
    "knee_L_medial": (13, "knee_L"), "knee_L_lateral": (13, "knee_L"), "knee_L_front": (13, "knee_L"),
    "knee_R_medial": (14, "knee_R"), "knee_R_lateral": (14, "knee_R"), "knee_R_front": (14, "knee_R"),
    "ankle_L_medial": (15, "ankle_L"), "ankle_L_lateral": (15, "ankle_L"),
    "ankle_R_medial": (16, "ankle_R"), "ankle_R_lateral": (16, "ankle_R"),
    "heel_L": (19, "heel_L"), "heel_R": (22, "heel_R"),
    "toe_L": (17, "toe_L"), "toe_R": (20, "toe_R"),
}
_GROUP_SIZE = Counter(group for _idx, group in _CATALOG.values())
_UNAMBIGUOUS_GROUPS = {g for g, n in _GROUP_SIZE.items() if n == 1}


@dataclass
class FrameResult:
    n_gt_kp: int
    identity_matches: dict[str, float]       # marker name -> match distance, unambiguous groups only
    group_coverage: dict[str, tuple[int, int]]  # group -> (n_distinct_candidates_assigned, group_size)


def assign_frame(dots: np.ndarray, kp: np.ndarray, max_dist: float) -> FrameResult:
    names = [n for n, (idx, _g) in _CATALOG.items() if kp[idx, 2] > 0]
    if not names or dots.shape[0] == 0:
        return FrameResult(len(names), {}, {})

    cost = np.full((len(names), dots.shape[0]), 1e6)
    for ni, name in enumerate(names):
        idx, _group = _CATALOG[name]
        kx, ky = kp[idx, 0], kp[idx, 1]
        d = np.hypot(dots[:, 0] - kx, dots[:, 1] - ky)
        cost[ni, :] = np.where(d <= max_dist, d, 1e6)

    row_ind, col_ind = linear_sum_assignment(cost)
    assigned = {names[ri]: (ci, cost[ri, ci]) for ri, ci in zip(row_ind, col_ind) if cost[ri, ci] < 1e6}

    identity_matches = {
        name: dist for name, (_ci, dist) in assigned.items() if _CATALOG[name][1] in _UNAMBIGUOUS_GROUPS
    }
    group_coverage: dict[str, tuple[int, int]] = {}
    by_group: dict[str, set[int]] = {}
    for name, (ci, _dist) in assigned.items():
        group = _CATALOG[name][1]
        if group in _UNAMBIGUOUS_GROUPS:
            continue
        by_group.setdefault(group, set()).add(ci)
    for group, cand_set in by_group.items():
        group_coverage[group] = (len(cand_set), _GROUP_SIZE[group])

    return FrameResult(len(names), identity_matches, group_coverage)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--marker-detection-run", required=True)
    ap.add_argument("--pose-sequence", required=True)
    ap.add_argument("--camera-label", required=True)
    ap.add_argument("--start-time", type=float, required=True)
    ap.add_argument("--end-time", type=float, required=True)
    ap.add_argument("--match-radius-px", type=float, default=80.0)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    cam_id = conn.execute("SELECT id FROM camera_instances WHERE label = ?", (args.camera_label,)).fetchone()["id"]
    shot_id = conn.execute(
        "SELECT shot_id FROM pose_observation_sequences WHERE id = ?", (args.pose_sequence,)
    ).fetchone()["shot_id"]
    svid = conn.execute(
        "SELECT id FROM capture_videos WHERE camera_instance_id = ? AND shot_id = ?", (cam_id, shot_id)
    ).fetchone()["id"]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from calibrate_rigid_marker_body import load_sync_table  # noqa: E402
    sync_table, _ = load_sync_table(conn, shot_id)
    frame_lo = sync_table.lookup(args.start_time, svid)
    frame_hi = sync_table.lookup(args.end_time, svid)

    identity_dists: dict[str, list[float]] = {}
    group_coverage_totals: dict[str, list[tuple[int, int]]] = {}
    n_frames = 0

    for video_frame in range(frame_lo, frame_hi):
        dot_row = conn.execute(
            "SELECT keypoints FROM detection_keypoints WHERE detection_run_id = ? AND shot_video_id = ? "
            "AND video_frame = ? AND region_type = 'dots'",
            (args.marker_detection_run, svid, video_frame),
        ).fetchone()
        if dot_row is None:
            continue
        pose_row = conn.execute(
            "SELECT kp_blob FROM pose_observations WHERE sequence_id = ? AND camera_instance_id = ? "
            "AND video_frame = ? AND source = 'body'",
            (args.pose_sequence, cam_id, video_frame),
        ).fetchone()
        if pose_row is None:
            continue

        dots = decode_dot_candidates(bytes(dot_row["keypoints"]))
        kp = np.frombuffer(bytes(pose_row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
        result = assign_frame(dots, kp, args.match_radius_px)
        if result.n_gt_kp == 0:
            continue
        n_frames += 1
        for name, dist in result.identity_matches.items():
            identity_dists.setdefault(name, []).append(dist)
        for group, (n_assigned, group_size) in result.group_coverage.items():
            group_coverage_totals.setdefault(group, []).append((n_assigned, group_size))

    print(f"{args.camera_label}: {n_frames} frames with any leg keypoint\n")
    print("--- identity-accurate slots (unambiguous, single marker per joint) ---")
    for name in sorted(identity_dists):
        dists = identity_dists[name]
        print(f"  {name:12s} matched {len(dists)}/{n_frames} frames  "
              f"mean_dist={np.mean(dists):.1f}px  median={np.median(dists):.1f}px")

    print()
    print("--- group coverage (ambiguous, 2+ markers share one joint -- identity NOT resolved) ---")
    for group in sorted(group_coverage_totals):
        totals = group_coverage_totals[group]
        avg_covered = np.mean([n for n, _size in totals])
        size = totals[0][1]
        full_coverage_frac = np.mean([n == size for n, _size in totals])
        print(f"  {group:10s} avg {avg_covered:.2f}/{size} distinct candidates assigned per frame  "
              f"(all {size} found: {100*full_coverage_frac:.0f}% of frames)  n_frames={len(totals)}")


if __name__ == "__main__":
    main()
