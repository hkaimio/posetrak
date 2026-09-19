# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""sweep_linker_and_b2.py -- sweeps `MotionGatedLinker` parameters (e.g.
max_missed) *and* B2's own grouping thresholds together, evaluated
against real GT, without needing a new detection run per combination.

Motivation (status.md, 2026-09-11): re-running full-capture detection
with `max_missed=3` (chosen from a per-camera regression/purity sweep)
fixed the per-tracklet identity-switch problem it was tuned for, but
made B2's own cross-camera grouping measurably *worse* against real GT
(precision 0.61 -> 0.39 on the 42-48s window) -- shorter, more numerous
tracklet fragments give B2's pairwise reprojection test thinner, weaker
overlap evidence per pair, and re-tuning B2's own thresholds alone
couldn't recover the old precision. The open question this answers:
would a *less* aggressive max_missed (still fixing the coasting bug)
give a better combined per-tracklet + per-group outcome than either
extreme tested so far (6 with the bug, 3 without it)?

Re-links each camera's raw candidates directly from the DB (via
`prototype_motion_gated_linker.relink_camera_to_tracklets`) for a given
`max_missed`, then runs `build_tracklet_groups`'s own prefilter/
build_groups/validate directly against that in-memory tracklet
population -- bypassing collect_tracklets() and its dependency on
whatever's actually stored in a `detection_runs` row. This makes an
N x M sweep (linker params x B2 params) cost N re-link passes, not N x M
full detection runs.

Usage:
    python tools/sweep_linker_and_b2.py \\
        --session /path/to/session.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --tracking-run 990fb01a-6c72-47bc-bdb7-4d31147d2ef7 \\
        --detection-run 75cbf678-2066-4a58-ab81-d27ea4c58d02 \\
        --start-time 42.0 --end-time 48.0 \\
        --camera-label gopro-11_mini_01 gopro13_02 pixel9 oneplus9pro-01 \\
        --validate-against-gt scratch/dot_ground_truth/pc_labels_refined.json \\
        --max-missed 2 3 4 6 8
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.build_tracklet_groups import (  # noqa: E402
    FKPredictor, build_groups, prefilter, validate,
)
from tools.calibrate_rigid_marker_body import load_camera_states, load_sync_table  # noqa: E402
from tools.prototype_fk_marker_prediction import _DEFAULT_TRIAL  # noqa: E402
from tools.prototypes.prototype_motion_gated_linker import relink_camera_to_tracklets  # noqa: E402

_UNLABELED = "unlabeled"


def derive_gt_slots_from_tracklets(
    gt_frames: list, tracklets_by_cam: dict, label_to_cam: dict, match_radius_px: float,
) -> dict[tuple, str]:
    """Like build_tracklet_groups.derive_gt_tracklet_slots, but matches
    each labelled point against a given in-memory `tracklets_by_cam`
    (an arbitrary re-link) instead of the DB's stored tracklet_id --
    needed to validate a linker parameterization that was never actually
    written to a detection run."""
    frame_index: dict[str, dict[int, list[tuple[int, float, float]]]] = {}
    for cam_id, tracklets in tracklets_by_cam.items():
        idx: dict[int, list] = defaultdict(list)
        for tid, frames in tracklets.items():
            for vf, (px, py) in frames.items():
                idx[vf].append((tid, px, py))
        frame_index[cam_id] = idx

    votes: dict[tuple, Counter] = defaultdict(Counter)
    for e in gt_frames:
        pts = e.get("points", [])
        pm = e.get("point_meta", [])
        if not pts:
            continue
        cam_id = label_to_cam.get(e["camera_label"])
        if cam_id is None or cam_id not in frame_index:
            continue
        candidates = frame_index[cam_id].get(e["video_frame"], [])
        if not candidates:
            continue
        for k, (x, y) in enumerate(pts):
            slot = pm[k].get("slot") if k < len(pm) else None
            if not slot or slot == _UNLABELED:
                continue
            best_tid, best_d = None, None
            for tid, px, py in candidates:
                d = math.hypot(px - x, py - y)
                if best_d is None or d < best_d:
                    best_tid, best_d = tid, d
            if best_d is not None and best_d <= match_radius_px:
                votes[(cam_id, best_tid)][slot] += 1
    return {k: c.most_common(1)[0][0] for k, c in votes.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--tracking-run", required=True)
    ap.add_argument("--detection-run", required=True)
    ap.add_argument("--start-time", type=float, required=True)
    ap.add_argument("--end-time", type=float, required=True)
    ap.add_argument("--camera-label", nargs="+", required=True)
    ap.add_argument("--validate-against-gt", required=True)
    ap.add_argument("--gt-match-radius-px", type=float, default=5.0)
    def _float_or_none(s: str) -> float | None:
        return None if s.lower() == "none" else float(s)

    ap.add_argument("--max-missed", type=int, nargs="+", default=[3])
    ap.add_argument("--max-noise-dt", type=_float_or_none, nargs="+", default=[None],
                     help="Cap on the Kalman process-noise dt (see dot_tracklet.py's own docstring, "
                          "2026-09-12) -- pass 'none' for today's unbounded default.")
    ap.add_argument("--accel-std-px", type=float, nargs="+", default=[2.0])
    ap.add_argument("--disambiguation-margin-linker", type=float, default=2.0)
    ap.add_argument("--min-lifetime", type=int, default=15)
    ap.add_argument("--proximity-px", type=float, default=80.0)
    ap.add_argument("--min-shared-frames", type=int, default=5)
    ap.add_argument("--median-thresh-px", type=float, default=3.0)
    ap.add_argument("--p90-thresh-px", type=float, default=6.0)
    ap.add_argument("--disambiguation-margin-b2", type=float, default=1.5)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    states = load_camera_states(conn, args.shot_id)
    sync_table, svid_by_cam = load_sync_table(conn, args.shot_id)
    label_to_cam = {r["label"]: r["id"] for r in conn.execute("SELECT id, label FROM camera_instances")}
    cam_to_label = {v: k for k, v in label_to_cam.items()}
    fkp = FKPredictor(conn, args.tracking_run, states, _DEFAULT_TRIAL)
    gt_frames = json.loads(Path(args.validate_against_gt).read_text())

    wanted_cams = {label_to_cam[label] for label in args.camera_label}

    for mm in args.max_missed:
        for mnd in args.max_noise_dt:
            for asp in args.accel_std_px:
                tracklets_by_cam = {}
                for cam_id in wanted_cams:
                    svid = svid_by_cam[cam_id]
                    lo = sync_table.lookup(args.start_time, svid)
                    hi = sync_table.lookup(args.end_time, svid)
                    tracklets_by_cam[cam_id] = relink_camera_to_tracklets(
                        conn, args.detection_run, svid, lo, hi,
                        max_missed=mm, accel_std_px=asp, max_noise_dt=mnd,
                        disambiguation_margin=args.disambiguation_margin_linker,
                    )
                n_raw = {cam_to_label[c]: len(t) for c, t in tracklets_by_cam.items()}

                filtered = prefilter(tracklets_by_cam, sync_table, svid_by_cam, fkp, args.min_lifetime, args.proximity_px)
                groups, _ = build_groups(
                    filtered, states, sync_table, svid_by_cam,
                    args.min_shared_frames, args.median_thresh_px, args.p90_thresh_px, args.disambiguation_margin_b2,
                )
                gt_slots = derive_gt_slots_from_tracklets(gt_frames, tracklets_by_cam, label_to_cam, args.gt_match_radius_px)

                print(f"=== max_missed={mm} max_noise_dt={mnd} accel_std_px={asp} === raw tracklets: {n_raw}  groups: {len(groups)}")
                validate(groups, gt_slots, tracklets_by_cam, sync_table, svid_by_cam, args.min_shared_frames)
            print()


if __name__ == "__main__":
    main()
