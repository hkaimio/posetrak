# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""build_gt_frame_manifest.py — generate a frame manifest for
label_dot_ground_truth.py: for each requested global time and each
camera, resolve that camera's own video_frame via the shot's sync table
(different cameras don't share frame numbering) and emit one manifest
entry per (camera, time).

Usage:
    python tools/build_gt_frame_manifest.py \\
        --session /path/to/session.db \\
        --dataset nelli \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --camera-label gopro-11_mini_01 gopro13_01 gopro13_02 insta_ace2_pro oneplus9pro-01 pixel9 \\
        --start-time 40.0 --end-time 60.0 --count 25 \\
        --tag person-markers \\
        --output scratch/dot_ground_truth/frame_manifest.json

    # or pass explicit times instead of --start/--end/--count:
    #   --times 40.0 42.5 45.1 48.7 ...

Append to an existing manifest with --append (dedupes on
dataset/camera_label/video_frame).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.calibrate_rigid_marker_body import load_sync_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--dataset", required=True, help="Short name matching label_dot_ground_truth's --session NAME=PATH.")
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--camera-label", required=True, nargs="+")
    ap.add_argument("--start-time", type=float)
    ap.add_argument("--end-time", type=float)
    ap.add_argument("--count", type=int, help="Number of evenly-spaced times between start and end.")
    ap.add_argument("--times", type=float, nargs="+", help="Explicit global times (overrides start/end/count).")
    ap.add_argument("--tag", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    if args.times:
        times = list(args.times)
    elif args.start_time is not None and args.end_time is not None and args.count:
        times = list(np.linspace(args.start_time, args.end_time, args.count))
    else:
        raise SystemExit("give either --times, or all of --start-time --end-time --count")

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    sync_table, svid_by_cam = load_sync_table(conn, args.shot_id)
    label_to_id = {r["label"]: r["id"] for r in conn.execute("SELECT id, label FROM camera_instances")}

    out_path = Path(args.output)
    existing: list[dict] = []
    if args.append and out_path.exists():
        existing = json.loads(out_path.read_text())
    seen = {(e["dataset"], e["camera_label"], e["video_frame"]) for e in existing}

    entries = list(existing)
    n_new = 0
    for label in args.camera_label:
        cam_id = label_to_id.get(label)
        if cam_id is None or cam_id not in svid_by_cam:
            print(f"  skip {label}: no camera / no sync entry", file=sys.stderr)
            continue
        svid = svid_by_cam[cam_id]
        for t in times:
            fidx = sync_table.lookup(float(t), svid)
            if fidx is None:
                print(f"  skip {label} @ {t:.3f}s: no sync frame", file=sys.stderr)
                continue
            key = (args.dataset, label, int(fidx))
            if key in seen:
                continue
            seen.add(key)
            entries.append({
                "dataset": args.dataset,
                "camera_label": label,
                "shot_video_id": svid,
                "video_frame": int(fidx),
                "time_s": round(float(t), 3),
                "tag": args.tag,
            })
            n_new += 1

    entries.sort(key=lambda e: (e["dataset"], e["camera_label"], e["video_frame"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2))
    print(f"wrote {len(entries)} entries ({n_new} new) -> {out_path}")


if __name__ == "__main__":
    main()
