# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""sweep_dot_detection.py — P-D: sweep per-camera dot-detection
`threshold` x `max_saturation` on the slot-labelled ground-truth frames
and report recall / precision per combination, per camera, so
`dot_threshold_by_camera` (and a new per-camera `max_saturation`) can be
picked from data rather than guessed.

Runs `dot_blob_detector.detect_blobs()` directly on each labelled frame
(no pipeline, no DB write) in `background_mode='blacklist'` with a
per-camera median background (computed once, cached) so the blacklist
glare veto matches production. For each (threshold, max_saturation) cell:

    recall    = GT slot-dots matched by a detection / all GT slot-dots
    precision = detections matched to a GT slot-dot   / all detections

Match is one-to-one nearest within --match-radius-px.

Usage:
    python tools/sweep_dot_detection.py \\
        --labels scratch/dot_ground_truth/pc_labels_refined.json \\
        --session nelli=/path/to/session.db \\
        --background-cache-dir scratch/dot_ground_truth/bg_cache \\
        --camera gopro13_01 --camera oneplus9pro-01 --camera pixel9 \\
        --thresholds 130 150 170 190 210 \\
        --max-saturations 45 80 120 255
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from posetrak.detection.dot_blob_detector import detect_blobs  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402
from tools.gt.refine_dot_ground_truth import _get_background, _resolve_file_path  # noqa: E402

_UNLABELED = "unlabeled"


def _match_count(gt: np.ndarray, det: np.ndarray, radius: float) -> int:
    if len(gt) == 0 or len(det) == 0:
        return 0
    d = np.linalg.norm(gt[:, None, :] - det[None, :, :], axis=2)
    cost = np.where(d <= radius, d, 1e6)
    gi, di = linear_sum_assignment(cost)
    return int(sum(1 for a, b in zip(gi, di) if cost[a, b] < 1e6))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--session", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--camera", action="append", default=None,
                     help="Restrict to these camera labels (repeatable). Default: all in the labels.")
    ap.add_argument("--thresholds", type=int, nargs="+", default=[130, 150, 170, 190, 210])
    ap.add_argument("--max-saturations", type=float, nargs="+", default=[45, 80, 120, 255])
    ap.add_argument("--match-radius-px", type=float, default=4.0)
    ap.add_argument("--blacklist-frac", type=float, nargs="+", default=[0.7],
                     help="Sweep the glare-veto fraction. Higher = looser veto (fewer real markers "
                          "vetoed, more glare kept). Each value gets its own table.")
    ap.add_argument("--blacklist-radius-px", type=int, default=6)
    ap.add_argument("--background-cache-dir", default=None)
    ap.add_argument("--bg-sample-count", type=int, default=40)
    ap.add_argument("--bg-frame-span", type=int, default=6000)
    ap.add_argument("--no-blacklist", action="store_true",
                     help="Skip the blacklist glare veto (background=None) -- isolates the pure "
                          "threshold/max_saturation/shape effect, no per-camera background decode.")
    args = ap.parse_args()

    session_paths = {}
    for entry in args.session:
        name, _, path = entry.partition("=")
        session_paths[name] = path
    conns = {n: sqlite3.connect(f"file:{p}?mode=ro", uri=True) for n, p in session_paths.items()}
    cache_dir = Path(args.background_cache_dir) if args.background_cache_dir else None

    frames = json.loads(Path(args.labels).read_text())
    want_cams = set(args.camera) if args.camera else None

    # group labelled frames by (dataset, camera_label)
    by_cam: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in frames:
        if not e.get("points"):
            continue
        if want_cams and e["camera_label"] not in want_cams:
            continue
        by_cam[(e["dataset"], e["camera_label"])].append(e)

    for (dataset, cam), entries in sorted(by_cam.items()):
        conn = conns[dataset]
        svid = entries[0]["shot_video_id"]
        file_path = _resolve_file_path(conn, svid)
        if args.no_blacklist:
            background = None
        else:
            vfs = [e["video_frame"] for e in entries]
            lo, hi = min(vfs) - args.bg_frame_span, max(vfs) + args.bg_frame_span
            background = _get_background(cache_dir, dataset, svid, file_path,
                                        args.bg_sample_count, max(0, lo), hi)

        # decode each labelled frame once; keep gray + bgr + GT slot points
        per_frame: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for e in entries:
            img = None
            for _, dec in iter_frames(file_path, e["video_frame"], e["video_frame"] + 1):
                img = dec
                break
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pm = e.get("point_meta", [])
            gt = np.array([xy for k, xy in enumerate(e["points"])
                           if k < len(pm) and pm[k].get("slot") not in (None, _UNLABELED)], dtype=float)
            per_frame.append((gray, img, gt.reshape(-1, 2)))

        n_gt = sum(len(gt) for _g, _b, gt in per_frame)
        for frac in args.blacklist_frac:
            bg_arg = None if (background is None or frac >= 1.0) else background
            tag = "NO VETO" if bg_arg is None else f"veto frac={frac}"
            print(f"\n=== {cam}  ({len(per_frame)} frames, {n_gt} GT markers)  "
                  f"match r={args.match_radius_px:.0f}px  {tag} ===")
            head = "thr \\ maxsat |" + "".join(f"{s:>13.0f}" for s in args.max_saturations)
            print(head)
            print("-" * len(head))
            for thr in args.thresholds:
                cells = []
                for sat in args.max_saturations:
                    matched = n_det = 0
                    for gray, bgr, gt in per_frame:
                        blobs = detect_blobs(
                            gray, threshold=thr, background=bg_arg, background_mode="blacklist",
                            blacklist_frac=frac, blacklist_radius_px=args.blacklist_radius_px,
                            bgr=bgr, max_saturation=sat,
                        )
                        det = np.array([[b.cx, b.cy] for b in blobs], dtype=float).reshape(-1, 2)
                        n_det += len(det)
                        matched += _match_count(gt, det, args.match_radius_px)
                    rec = matched / n_gt if n_gt else float("nan")
                    prec = matched / n_det if n_det else float("nan")
                    cells.append(f"{rec:4.2f}/{prec:4.2f}")
                print(f"{thr:11d} |" + "".join(f"{c:>13s}" for c in cells))


if __name__ == "__main__":
    main()
