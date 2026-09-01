# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prototype_dot_blob_detector.py — spike for reflective-dot detection
(marker-mocap design doc's Phase C, see
docs/roadmap/features/marker-based-mocap/reflective-dot-detection-design.md).

Not integrated with anything -- a throwaway script to characterize real
detector behavior (candidate counts, false-positive sources, achievable
centroid precision) on real GoPro footage before committing to the
Hungarian/Mahalanobis assignment architecture. Read-only against the
session DB; writes only annotated PNGs + a CSV to the scratchpad/output dir
given on the command line.

Detection method matches marker-detection-analysis.md's Question A
(threshold + connected components + centroid) -- confirmed empirically
2026-09-01 against real capture frames: the sword's dots are retroreflective
(blown-out white, high contrast against the dark wood/case), but other
bright/reflective scene elements (a shiny case edge threw a comparable
glare in one frame) can trigger the same threshold, so a shape filter
(compactness/aspect ratio, not just brightness+area) is included from the
start rather than added after the fact.

Usage:
    python tools/prototype_dot_blob_detector.py \\
        --session /path/to/session.db \\
        --camera-instance-id <camera_instance_id> \\
        --time-start 34.4 --time-end 100.6 --stride 6 \\
        --out-dir /path/to/output
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.setup.db_context import SyncPoint, SyncTable  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402


@dataclass
class BlobCandidate:
    cx: float
    cy: float
    area: float
    compactness: float  # 4*pi*area / perimeter^2 -- 1.0 for a perfect circle
    bbox: tuple[int, int, int, int]


def detect_blobs(
    gray: np.ndarray,
    *,
    threshold: int = 235,
    min_area: float = 4.0,
    max_area: float = 400.0,
    min_compactness: float = 0.5,
) -> list[BlobCandidate]:
    """Threshold + connected components + centroid, with a compactness
    filter to reject elongated glare streaks (light fixtures, shiny edges)
    that pass a brightness+area filter alone but aren't a round dot."""
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        compactness = 4 * np.pi * area / (perimeter * perimeter)
        if compactness < min_compactness:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        x, y, w, h = cv2.boundingRect(c)
        out.append(BlobCandidate(cx, cy, area, compactness, (x, y, w, h)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True, help="captures.id -- disambiguates cameras reused across captures")
    ap.add_argument("--camera-instance-id", required=True)
    ap.add_argument("--time-start", type=float, required=True)
    ap.add_argument("--time-end", type=float, required=True)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--threshold", type=int, default=235)
    ap.add_argument("--min-area", type=float, default=4.0)
    ap.add_argument("--max-area", type=float, default=400.0)
    ap.add_argument("--min-compactness", type=float, default=0.5)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--save-every-annotated", type=int, default=30,
                    help="Save an annotated frame every Nth processed frame (0 = never)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    sync_row = conn.execute(
        "SELECT id FROM sync_configs WHERE shot_id = ? ORDER BY rowid DESC LIMIT 1",
        (args.shot_id,),
    ).fetchone()
    if sync_row is None:
        raise ValueError("no sync_configs row found for this capture")
    sync_id = sync_row["id"]

    sp_rows = conn.execute(
        "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, cv.actual_fps, cv.file_path "
        "FROM sync_points sp JOIN capture_videos cv ON cv.id = sp.shot_video_id "
        "WHERE sp.sync_config_id = ? AND cv.camera_instance_id = ? AND cv.shot_id = ?",
        (sync_id, args.camera_instance_id, args.shot_id),
    ).fetchall()
    if not sp_rows:
        raise ValueError("no sync_points for this camera")
    svid = sp_rows[0]["shot_video_id"]
    file_path = sp_rows[0]["file_path"]
    sync_points = [
        SyncPoint(camera_instance_id=svid, shot_video_id=svid,
                 video_frame=int(r["video_frame"]), timestamp_s=float(r["timestamp_s"]))
        for r in sp_rows
    ]
    sync_table = SyncTable(sync_points, {svid: float(sp_rows[0]["actual_fps"])})

    first = sync_table.lookup(args.time_start, svid)
    last = sync_table.lookup(args.time_end, svid)
    if first is None or last is None:
        raise ValueError("no sync coverage for the requested time range")

    print(f"Scanning {file_path} frames {first}-{last}")

    csv_path = out_dir / "blob_candidates.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["video_frame", "timestamp_s", "n_candidates", "cx", "cy", "area", "compactness"])

    n_decoded = 0
    n_processed = 0
    candidate_counts = []
    for video_frame, img in iter_frames(file_path, first, last):
        n_decoded += 1
        if (video_frame - first) % args.stride != 0:
            continue
        n_processed += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blobs = detect_blobs(
            gray, threshold=args.threshold, min_area=args.min_area,
            max_area=args.max_area, min_compactness=args.min_compactness,
        )
        candidate_counts.append(len(blobs))
        t = sync_table.frame_to_global_time(video_frame, svid) or 0.0
        if not blobs:
            writer.writerow([video_frame, f"{t:.3f}", 0, "", "", "", ""])
        for b in blobs:
            writer.writerow([video_frame, f"{t:.3f}", len(blobs),
                             f"{b.cx:.2f}", f"{b.cy:.2f}", f"{b.area:.1f}", f"{b.compactness:.3f}"])

        if args.save_every_annotated and n_processed % args.save_every_annotated == 0:
            annotated = img.copy()
            for b in blobs:
                cv2.circle(annotated, (int(b.cx), int(b.cy)), 12, (0, 0, 255), 2)
                cv2.putText(annotated, f"{b.area:.0f}", (int(b.cx) + 14, int(b.cy)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            out_path = out_dir / f"annotated_f{video_frame}_t{t:.2f}_n{len(blobs)}.png"
            cv2.imwrite(str(out_path), annotated)

    csv_file.close()
    print(f"{n_decoded} frames decoded, {n_processed} processed")
    if candidate_counts:
        arr = np.array(candidate_counts)
        print(f"candidates per frame: mean={arr.mean():.2f} median={np.median(arr):.0f} "
              f"min={arr.min()} max={arr.max()}")
        print(f"frames with 0 candidates: {(arr == 0).sum()}/{len(arr)}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
