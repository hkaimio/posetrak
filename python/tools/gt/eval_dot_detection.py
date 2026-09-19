# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""eval_dot_detection.py — Stage A metric (redesign doc §2.2): dot
detection recall / precision **per camera**, against a slot-labelled
ground-truth set (label_marker_slots_gui.py output, optionally refined).

For each labelled frame, a detection is matched to a GT dot if within
--match-radius-px, one-to-one (Hungarian on distance, gated by the
radius). Then, per camera:

  recall       = GT slot-dots matched by a detection / all GT slot-dots
  precision    = detections matched to a GT slot-dot        / all detections
  precision*   = detections matched to ANY GT dot (slot or  / all detections
                 `unlabeled`)
                 -- separates "false positive" from "found a real
                 reflective thing that just isn't a body marker"

The aggregate across cameras is meaningless when per-camera detection
quality differs (the 2026-09-10 finding: gopro13_01 / oneplus / insta
under-detect); the per-camera rows are the point. Run it against
different `--detection-run` ids to compare threshold / max_saturation
settings (P-D).

Usage:
    python tools/eval_dot_detection.py \\
        --labels scratch/dot_ground_truth/pc_labels_refined.json \\
        --session nelli=/path/to/session.db \\
        --detection-run 01c2e3c1-204b-49f8-9b14-6fd611b765a4
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.pose.db_cache import decode_dot_candidates  # noqa: E402

_UNLABELED = "unlabeled"


def _match(gt: np.ndarray, det: np.ndarray, radius: float) -> list[tuple[int, int, float]]:
    """One-to-one nearest matching gated at *radius*. Returns (gi, di, dist)."""
    if len(gt) == 0 or len(det) == 0:
        return []
    d = np.linalg.norm(gt[:, None, :] - det[None, :, :], axis=2)
    cost = np.where(d <= radius, d, 1e6)
    gi, di = linear_sum_assignment(cost)
    return [(int(a), int(b), float(d[a, b])) for a, b in zip(gi, di) if cost[a, b] < 1e6]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--session", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--detection-run", required=True)
    ap.add_argument("--match-radius-px", type=float, default=6.0)
    ap.add_argument("--list-misses", action="store_true", help="Print each unmatched GT slot-dot.")
    args = ap.parse_args()

    session_paths = {}
    for entry in args.session:
        name, _, path = entry.partition("=")
        session_paths[name] = path
    conns = {n: sqlite3.connect(f"file:{p}?mode=ro", uri=True) for n, p in session_paths.items()}

    frames = json.loads(Path(args.labels).read_text())

    # per camera: [n_gt_slot, n_gt_slot_matched, n_gt_any, n_det, n_det_hit_slot, n_det_hit_any]
    agg: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    dists: dict[str, list[float]] = defaultdict(list)

    for e in frames:
        pts = e.get("points", [])
        if not pts:
            continue
        pm = e.get("point_meta", [])
        slots = [(pm[k].get("slot") if k < len(pm) else _UNLABELED) or _UNLABELED for k in range(len(pts))]
        gt_all = np.array(pts, dtype=float)
        is_slot = np.array([s != _UNLABELED for s in slots])
        gt_slot = gt_all[is_slot]

        conn = conns[e["dataset"]]
        row = conn.execute(
            "SELECT keypoints FROM detection_keypoints WHERE detection_run_id = ? "
            "AND shot_video_id = ? AND video_frame = ? AND region_type = 'dots'",
            (args.detection_run, e["shot_video_id"], e["video_frame"]),
        ).fetchone()
        det = decode_dot_candidates(bytes(row[0]))[:, :2].astype(float) if row is not None else np.zeros((0, 2))

        cam = e["camera_label"]
        a = agg[cam]
        a[0] += len(gt_slot)
        a[2] += len(gt_all)
        a[3] += len(det)

        m_slot = _match(gt_slot, det, args.match_radius_px)
        a[1] += len(m_slot)
        matched_det_slot = {di for _gi, di, _d in m_slot}
        a[4] += len(matched_det_slot)
        dists[cam] += [dd for _g, _d, dd in m_slot]

        m_any = _match(gt_all, det, args.match_radius_px)
        a[5] += len({di for _gi, di, _d in m_any})

        if args.list_misses:
            hit_gi = {gi for gi, _di, _d in m_slot}
            slot_idx = np.where(is_slot)[0]
            for local_gi in range(len(gt_slot)):
                if local_gi not in hit_gi:
                    print(f"  MISS  {cam} f={e['video_frame']}  {slots[slot_idx[local_gi]]}  "
                          f"@({gt_slot[local_gi][0]:.0f},{gt_slot[local_gi][1]:.0f})")

    print(f"\nmatch radius = {args.match_radius_px:.1f}px   detection_run = {args.detection_run}\n")
    hdr = f"{'camera':20s} {'GTslot':>7s} {'recall':>8s} {'prec':>8s} {'prec*':>8s} {'#det':>6s} {'medpx':>7s}"
    print(hdr)
    print("-" * len(hdr))
    tot = [0, 0, 0, 0, 0, 0]
    all_d: list[float] = []
    for cam in sorted(agg):
        n_gt_slot, n_gt_slot_m, _n_gt_any, n_det, n_det_slot, n_det_any = agg[cam]
        rec = n_gt_slot_m / n_gt_slot if n_gt_slot else float("nan")
        prec = n_det_slot / n_det if n_det else float("nan")
        preca = n_det_any / n_det if n_det else float("nan")
        med = np.median(dists[cam]) if dists[cam] else float("nan")
        print(f"{cam:20s} {n_gt_slot:7d} {rec:8.2f} {prec:8.2f} {preca:8.2f} {n_det:6d} {med:7.1f}")
        for i in range(6):
            tot[i] += agg[cam][i]
        all_d += dists[cam]
    rec = tot[1] / tot[0] if tot[0] else float("nan")
    prec = tot[4] / tot[3] if tot[3] else float("nan")
    preca = tot[5] / tot[3] if tot[3] else float("nan")
    med = np.median(all_d) if all_d else float("nan")
    print("-" * len(hdr))
    print(f"{'TOTAL (not meaningful)':20s} {tot[0]:7d} {rec:8.2f} {prec:8.2f} {preca:8.2f} {tot[3]:6d} {med:7.1f}")


if __name__ == "__main__":
    main()
