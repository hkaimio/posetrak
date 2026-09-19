# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""refine_dot_ground_truth.py — snap each hand-clicked ground-truth point
(from label_dot_ground_truth.py) to the real local intensity centroid, in a
small window around the click, using the same background-subtracted
connected-components + moment-centroid logic `dot_blob_detector.detect_blobs()`
itself uses -- so a "correct" detection is judged against the same
definition of "centroid" the detector produces, not against wherever a
mouse click happened to land.

A hand click is good enough to say "there's a real marker near here" but
not to say precisely where its centroid is -- a few pixels off is normal,
and for a motion-blur streak (labeled at its visual midpoint per Harri's
own labeling convention) a few pixels off *along the streak* is expected
too. This tries a ladder of residual thresholds from strict to loose (a
tighter threshold gives a smaller, more centroid-accurate blob when it
finds one at all; a streak's dimmer peak may need a looser one to register
as a connected blob at all) and takes the first one that yields a
connected component within --snap-radius-px of the original click.

Writes a *new* labels file by default (never silently overwrites your
hand-labeled one) with a report of how far each point moved, so you can
sanity-check before adopting it -- e.g. by re-opening the refined file in
label_dot_ground_truth.py itself (it resumes/loads any existing --output
file) and spot-checking the ones that moved the most.

Usage:
    python tools/refine_dot_ground_truth.py \\
        --labels scratch/dot_ground_truth/labels.json \\
        --session nelli=D:/mocap/2026-09-06-kare-tests/2026-09-06-kare-tests.db \\
        --session sword=E:/mocap/vanhaa/ukemi-tommi-20260509.db \\
        --background-cache-dir scratch/dot_ground_truth/bg_cache \\
        --output scratch/dot_ground_truth/labels_refined.json
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

from posetrak.detection.dot_blob_detector import compute_background  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402

# Residual thresholds tried strict -> loose. Strict-first: a tighter
# threshold on a real marker's own bright core gives a smaller, more
# accurate centroid; only fall back to a looser one (needed for a dim
# motion-blur streak) if nothing connected turns up yet.
_THRESHOLD_LADDER = [120, 90, 60, 40, 25, 15]


def _resolve_file_path(conn: sqlite3.Connection, shot_video_id: str) -> str:
    row = conn.execute("SELECT file_path FROM capture_videos WHERE id = ?", (shot_video_id,)).fetchone()
    if row is None:
        raise SystemExit(f"no capture_videos row for shot_video_id={shot_video_id!r}")
    return row[0]


def _background_cache_path(
    cache_dir: Path, dataset: str, shot_video_id: str, sample_count: int, first_frame: int, last_frame: int,
) -> Path:
    # Keyed on the sampled frame range too, not just (dataset, video, sample_count) -- two labeled
    # frames far apart in time get their own local background window (see --bg-frame-span), and
    # without the range in the key the second one would silently reuse the first's cached (and
    # for a genuinely different scene moment, wrong) background instead of computing its own.
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


def _refine_point(
    gray: np.ndarray, background: np.ndarray, gx: float, gy: float,
    window_radius: int, snap_radius_px: float,
    centroid_mode: str, blacklist_threshold: int,
) -> tuple[float, float, bool, int | None]:
    """Returns (refined_x, refined_y, was_refined, threshold_used).

    centroid_mode:
      'blacklist' (default) -- threshold *raw* brightness at
        `blacklist_threshold`, take the contour-moments centroid of the
        connected component nearest the click. This is exactly what
        `dot_blob_detector.detect_blobs(background_mode='blacklist')` --
        the production detector -- does, so a marker the detector found
        lands the GT point on the detector's own centroid (metric epsilon
        can then be tiny), and a marker it missed still gets a real
        centroid.
      'subtract-ladder' -- the original: threshold the residual
        (`gray - background`) down a strict->loose ladder, first component
        within snap radius wins. Use for captures detected in
        `background_mode='subtract'`.
    """
    h, w = gray.shape
    x0, y0 = max(0, int(gx - window_radius)), max(0, int(gy - window_radius))
    x1, y1 = min(w, int(gx + window_radius)), min(h, int(gy + window_radius))
    if x1 <= x0 or y1 <= y0:
        return gx, gy, False, None
    local_gx, local_gy = gx - x0, gy - y0

    if centroid_mode == "blacklist":
        levels = [blacklist_threshold]
        source = gray[y0:y1, x0:x1]
    else:
        levels = _THRESHOLD_LADDER
        source = cv2.subtract(gray[y0:y1, x0:x1], background[y0:y1, x0:x1])

    for thresh in levels:
        _, mask = cv2.threshold(source, thresh, 255, cv2.THRESH_BINARY)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        best_i, best_d = None, None
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 2:
                continue
            cx, cy = centroids[i]
            d = float(np.hypot(cx - local_gx, cy - local_gy))
            if best_d is None or d < best_d:
                best_i, best_d = i, d
        if best_i is not None and best_d <= snap_radius_px:
            comp_mask = (labels == best_i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                m = cv2.moments(max(contours, key=cv2.contourArea))
                if m["m00"] != 0:
                    rx, ry = m["m10"] / m["m00"], m["m01"] / m["m00"]
                    return x0 + rx, y0 + ry, True, thresh
    return gx, gy, False, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--session", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--output", required=True)
    ap.add_argument("--background-cache-dir", default=None)
    ap.add_argument("--bg-sample-count", type=int, default=40)
    ap.add_argument("--bg-frame-span", type=int, default=6000)
    ap.add_argument("--window-radius-px", type=int, default=40,
                     help="How far around each click to search (default 40px -- generous enough "
                          "for a motion-blur streak's own length, not just a round dot).")
    ap.add_argument("--snap-radius-px", type=float, default=30.0,
                     help="Max distance a found component's centroid may be from the original "
                          "click to be accepted as a refinement of it.")
    ap.add_argument("--centroid-mode", choices=["blacklist", "subtract-ladder"], default="blacklist",
                     help="'blacklist' (default) matches the production detector's raw-brightness "
                          "centroid; 'subtract-ladder' is the original residual method.")
    ap.add_argument("--blacklist-threshold", type=int, default=200,
                     help="Raw-brightness threshold for --centroid-mode blacklist (match the "
                          "detection run's own dot_threshold).")
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

    cache_dir = Path(args.background_cache_dir) if args.background_cache_dir else None

    # One background per (dataset, shot_video_id), spanning the union of that camera's own
    # labeled frames (padded) -- not one per individual labeled frame. The room/scene doesn't
    # change across a capture; a single well-sampled background is valid for any frame in range,
    # and computing (then caching) it once per camera is what makes this fast, matching the
    # production pipeline's own one-background-per-camera-per-run design.
    labeled_frames_by_cam: dict[tuple[str, str], list[int]] = {}
    for entry in frames:
        if entry.get("points"):
            labeled_frames_by_cam.setdefault((entry["dataset"], entry["shot_video_id"]), []).append(entry["video_frame"])

    # 'blacklist' centroid mode thresholds raw brightness only -- no background needed,
    # so skip the (slow) per-camera background decode entirely.
    bg_cache: dict[tuple[str, str], np.ndarray] = {}
    for (dataset, svid), video_frames in ([] if args.centroid_mode == "blacklist" else labeled_frames_by_cam.items()):
        conn = session_conns.get(dataset)
        if conn is None:
            raise SystemExit(f"no --session given for dataset {dataset!r}")
        file_path = _resolve_file_path(conn, svid)
        lo, hi = min(video_frames) - args.bg_frame_span, max(video_frames) + args.bg_frame_span
        bg_cache[(dataset, svid)] = _get_background(
            cache_dir, dataset, svid, file_path, args.bg_sample_count, max(0, lo), hi,
        )

    n_points = n_refined = n_unrefined = 0
    moved_report = []

    for entry in frames:
        if not entry.get("points"):
            continue
        dataset, svid, video_frame = entry["dataset"], entry["shot_video_id"], entry["video_frame"]
        conn = session_conns.get(dataset)
        file_path = _resolve_file_path(conn, svid)
        background = bg_cache.get((dataset, svid))  # None in --centroid-mode blacklist

        img = None
        for _, decoded in iter_frames(file_path, video_frame, video_frame + 1):
            img = decoded
            break
        if img is None:
            raise SystemExit(f"could not decode frame {video_frame} from {file_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        new_points = []
        for gx, gy in entry["points"]:
            n_points += 1
            rx, ry, refined, thresh = _refine_point(
                gray, background, gx, gy, args.window_radius_px, args.snap_radius_px,
                args.centroid_mode, args.blacklist_threshold,
            )
            new_points.append([rx, ry])
            moved = float(np.hypot(rx - gx, ry - gy))
            if refined:
                n_refined += 1
                if moved > 1.0:
                    moved_report.append(
                        f"{dataset}/{entry['camera_label']} frame={video_frame}: "
                        f"({gx:.1f},{gy:.1f}) -> ({rx:.1f},{ry:.1f})  moved={moved:.1f}px  thresh={thresh}"
                    )
            else:
                n_unrefined += 1
                moved_report.append(
                    f"{dataset}/{entry['camera_label']} frame={video_frame}: "
                    f"({gx:.1f},{gy:.1f}) -- NOT REFINED (no connected component found within "
                    f"{args.snap_radius_px:.0f}px at any threshold down to {_THRESHOLD_LADDER[-1]}) -- kept as-is"
                )
        entry["points"] = new_points

    print("\n".join(moved_report))
    print()
    print(f"Refined {n_refined}/{n_points} points ({n_unrefined} left unchanged -- no nearby signal found).")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(frames, f, indent=2)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
