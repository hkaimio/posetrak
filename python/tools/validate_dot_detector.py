# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""validate_dot_detector.py — check `dot_blob_detector.detect_blobs()`
against a hand-labeled ground-truth set (see label_dot_ground_truth.py) in
seconds, instead of running the full multi-hour detection pipeline on real
data to find out whether a parameter/logic change helped or hurt.

For each labeled frame: decodes just that one frame (plus, if
--bg-subtract, a per-(dataset, camera) background sampled once and cached
to --background-cache-dir so re-running against the same footage doesn't
re-pay the sampling cost), runs detect_blobs() with the given parameters,
and reports:

- recall: fraction of hand-labeled real markers with a detected candidate
  within --match-radius-px.
- precision (local): fraction of detected candidates *within a padded
  bounding box of the labeled markers themselves* that are near a real
  one -- deliberately not "fraction of every candidate in the whole 4K
  frame", since a person/prop is a small part of the image and whatever's
  happening in the rest of the room is a separate (real, but different)
  concern from whether detection is confusing candidates *near the actual
  subject*. --precision-pad-px controls how generous that box is.

Usage:
    python tools/validate_dot_detector.py \\
        --labels scratch/dot_ground_truth/labels.json \\
        --session nelli=D:/mocap/2026-09-06-kare-tests/2026-09-06-kare-tests.db \\
        --session sword=E:/mocap/vanhaa/ukemi-tommi-20260509.db \\
        --background-cache-dir scratch/dot_ground_truth/bg_cache \\
        --bg-subtract --threshold 60 --max-saturation 45 --bg-sample-count 40
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posetrak.detection.dot_blob_detector import compute_background, detect_blobs  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402


def _resolve_file_path(conn: sqlite3.Connection, shot_video_id: str) -> str:
    row = conn.execute("SELECT file_path FROM capture_videos WHERE id = ?", (shot_video_id,)).fetchone()
    if row is None:
        raise SystemExit(f"no capture_videos row for shot_video_id={shot_video_id!r}")
    return row[0]


def _background_cache_path(
    cache_dir: Path, dataset: str, shot_video_id: str, sample_count: int, first_frame: int, last_frame: int,
) -> Path:
    # Keyed on the sampled frame range too -- see refine_dot_ground_truth.py's identical helper
    # for why (two labeled frames far apart in time must not silently share one cached background).
    h = hashlib.sha1(f"{dataset}:{shot_video_id}:{sample_count}:{first_frame}:{last_frame}".encode()).hexdigest()[:16]
    return cache_dir / f"{dataset}_{shot_video_id[:8]}_{h}.npy"


def _get_background(
    cache_dir: Path | None, dataset: str, shot_video_id: str, file_path: str,
    sample_count: int, first_frame: int, last_frame: int,
) -> np.ndarray:
    if cache_dir is not None:
        cache_path = _background_cache_path(cache_dir, dataset, shot_video_id, sample_count, first_frame, last_frame)
        if cache_path.exists():
            return np.load(cache_path)
    span = max(1, last_frame - first_frame)
    stride = max(1, span // sample_count)
    frames = []
    for video_frame, img in iter_frames(file_path, first_frame, last_frame):
        if (video_frame - first_frame) % stride == 0:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if not frames:
        raise SystemExit(f"no frames sampled for background ({dataset}/{shot_video_id})")
    bg = compute_background(frames)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(_background_cache_path(cache_dir, dataset, shot_video_id, sample_count, first_frame, last_frame), bg)
    return bg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--session", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--match-radius-px", type=float, default=15.0)
    ap.add_argument("--precision-pad-px", type=float, default=150.0)
    ap.add_argument("--threshold", type=int, default=235)
    ap.add_argument("--max-saturation", type=float, default=255.0)
    ap.add_argument("--bg-subtract", action="store_true")
    ap.add_argument("--bg-sample-count", type=int, default=40)
    ap.add_argument("--bg-frame-span", type=int, default=6000,
                     help="Frame range (video_frame - this, video_frame + this) to sample the "
                          "background from around each labeled frame, when --bg-subtract is set "
                          "and no wider run range is known. Default 6000 (~50s at 120fps).")
    ap.add_argument("--background-cache-dir", default=None)
    ap.add_argument("--verbose", action="store_true", help="Print per-frame unmatched/false-positive detail.")
    args = ap.parse_args()

    session_paths = {}
    for entry in args.session:
        name, _, path = entry.partition("=")
        if not path:
            raise SystemExit(f"--session must be NAME=PATH, got {entry!r}")
        session_paths[name] = path
    session_conns = {name: sqlite3.connect(f"file:{p}?mode=ro", uri=True) for name, p in session_paths.items()}

    with open(args.labels, encoding="utf-8") as f:
        frames = json.load(f)
    frames = [fr for fr in frames if fr.get("points")]
    if not frames:
        raise SystemExit("no labeled frames with points found in --labels")

    cache_dir = Path(args.background_cache_dir) if args.background_cache_dir else None
    bg_cache: dict[tuple[str, str], np.ndarray] = {}

    total_gt = total_matched = total_cands_in_roi = total_cands_matched = 0
    per_frame_rows = []

    for entry in frames:
        dataset, cam_label, video_frame = entry["dataset"], entry["camera_label"], entry["video_frame"]
        svid = entry["shot_video_id"]
        conn = session_conns.get(dataset)
        if conn is None:
            raise SystemExit(f"no --session given for dataset {dataset!r}")
        file_path = _resolve_file_path(conn, svid)

        background = None
        if args.bg_subtract:
            cache_key = (dataset, svid)
            if cache_key not in bg_cache:
                bg_cache[cache_key] = _get_background(
                    cache_dir, dataset, svid, file_path, args.bg_sample_count,
                    max(0, video_frame - args.bg_frame_span), video_frame + args.bg_frame_span,
                )
            background = bg_cache[cache_key]

        img = None
        for _, decoded in iter_frames(file_path, video_frame, video_frame + 1):
            img = decoded
            break
        if img is None:
            raise SystemExit(f"could not decode frame {video_frame} from {file_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blobs = detect_blobs(
            gray, threshold=args.threshold, background=background, bgr=img,
            max_saturation=args.max_saturation,
        )
        cand_pts = np.array([[b.cx, b.cy] for b in blobs]) if blobs else np.zeros((0, 2))
        gt_pts = np.array(entry["points"], dtype=np.float64)

        # Recall: each GT point vs. nearest candidate.
        n_matched = 0
        unmatched = []
        for gx, gy in gt_pts:
            if cand_pts.shape[0] == 0:
                unmatched.append((gx, gy))
                continue
            d = np.hypot(cand_pts[:, 0] - gx, cand_pts[:, 1] - gy)
            if d.min() <= args.match_radius_px:
                n_matched += 1
            else:
                unmatched.append((gx, gy))

        # Precision within a padded box around the GT points.
        pad = args.precision_pad_px
        x0, y0 = gt_pts[:, 0].min() - pad, gt_pts[:, 1].min() - pad
        x1, y1 = gt_pts[:, 0].max() + pad, gt_pts[:, 1].max() + pad
        roi_cands = [
            (cx, cy) for cx, cy in cand_pts
            if x0 <= cx <= x1 and y0 <= cy <= y1
        ]
        n_cand_matched = 0
        false_positives = []
        for cx, cy in roi_cands:
            d = np.hypot(gt_pts[:, 0] - cx, gt_pts[:, 1] - cy)
            if d.min() <= args.match_radius_px:
                n_cand_matched += 1
            else:
                false_positives.append((cx, cy))

        total_gt += len(gt_pts)
        total_matched += n_matched
        total_cands_in_roi += len(roi_cands)
        total_cands_matched += n_cand_matched

        per_frame_rows.append(
            f"{dataset}/{cam_label} frame={video_frame} tag={entry.get('tag', ''):14s} "
            f"recall={n_matched}/{len(gt_pts)}  precision={n_cand_matched}/{len(roi_cands)}  "
            f"total_cands_frame={len(blobs)}"
        )
        if args.verbose:
            if unmatched:
                per_frame_rows.append(f"    unmatched GT points: {[(round(x), round(y)) for x, y in unmatched]}")
            if false_positives:
                per_frame_rows.append(f"    false positives (in ROI): {[(round(x), round(y)) for x, y in false_positives]}")

    print("\n".join(per_frame_rows))
    print()
    print(f"AGGREGATE: recall={total_matched}/{total_gt} ({100 * total_matched / max(total_gt, 1):.1f}%)  "
          f"precision={total_cands_matched}/{total_cands_in_roi} "
          f"({100 * total_cands_matched / max(total_cands_in_roi, 1):.1f}%)  "
          f"over {len(frames)} frames")


if __name__ == "__main__":
    main()
