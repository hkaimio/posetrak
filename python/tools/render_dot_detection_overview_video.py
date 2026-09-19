# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""render_dot_detection_overview_video.py — PROTOTYPE: raw diagnostic
visualization of dot detection, upstream of any person-marker assignment.

Two modes, both a multi-camera grid over a time window, cropped to each
camera's own leg-keypoint region by default (--full-frame to disable):

    --mode dots       every raw dot candidate this frame, one uniform
                      colour, drawn as a circle sized to the candidate's
                      own minor axis (so a fast-swing motion-blur streak
                      reads as a bigger blob). Purely "what did the
                      detector find", no filtering, no assignment.

    --mode tracklets  same candidates, coloured by MotionGatedLinker's
                      per-camera tracklet_id (deterministic hash -> BGR),
                      with the id number drawn next to each -- so a
                      tracklet that persists across frames keeps its
                      colour and number, and a tracklet that fragments or
                      swaps identity is visible as a colour/number change
                      on what looks like one continuous dot.

Neither mode runs fusion or assignment -- this is for answering "is the
detector even seeing the dots" and "how good is the per-camera temporal
linking" before trusting anything built on top.

Usage:
    python tools/render_dot_detection_overview_video.py \\
        --session /path/to/session.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --marker-detection-run 01c2e3c1-204b-49f8-9b14-6fd611b765a4 \\
        --pose-sequence ec1b3e2f-1ef8-4e31-806c-33102a969ecd \\
        --camera-label gopro-11_mini_01 gopro13_01 gopro13_02 insta_ace2_pro oneplus9pro-01 pixel9 \\
        --start-time 40.0 --end-time 50.0 --slow-factor 4 \\
        --mode tracklets \\
        --video-out scratch/dot_ground_truth/tracklets_overview.mp4
"""
from __future__ import annotations

import argparse
import colorsys
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pose.db_cache import decode_dot_candidates  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states, load_sync_table  # noqa: E402
from tools.prototype_person_marker_assignment import _CATALOG  # noqa: E402

_PAD_PX = 300
_STAMP_FONT_SCALE = 0.55
_STAMP_THICKNESS = 1


def _tracklet_color(tid: int) -> tuple[int, int, int]:
    if tid < 0:
        return (128, 128, 128)
    h = (tid * 0.61803398875) % 1.0  # golden-ratio hop -> well-spread hues
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def _prepare_cell(img, window, target_w, target_h):
    x0, y0, x1, y1 = window
    crop = img[y0:min(y1, img.shape[0]), x0:min(x1, img.shape[1])]
    crop_h, crop_w = crop.shape[:2]
    scale = min(target_w / crop_w, target_h / crop_h)
    new_w, new_h = max(1, round(crop_w * scale)), max(1, round(crop_h * scale))
    resized = cv2.resize(crop, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    off_x, off_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized

    def transform(x, y):
        return (x - x0) * scale + off_x, (y - y0) * scale + off_y

    return canvas, transform, scale


def _crop_window(conn, pose_sequence, cam_id, frame_lo, frame_hi):
    all_x, all_y = [], []
    for vf in range(frame_lo, frame_hi):
        row = conn.execute(
            "SELECT kp_blob FROM pose_observations WHERE sequence_id = ? AND camera_instance_id = ? "
            "AND video_frame = ? AND source = 'body'",
            (pose_sequence, cam_id, vf),
        ).fetchone()
        if row is None:
            continue
        kp = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
        for idx, _g in _CATALOG.values():
            if kp[idx, 2] > 0:
                all_x.append(kp[idx, 0])
                all_y.append(kp[idx, 1])
    if not all_x:
        return None
    return (max(0, int(min(all_x) - _PAD_PX)), max(0, int(min(all_y) - _PAD_PX)),
            int(max(all_x) + _PAD_PX), int(max(all_y) + _PAD_PX))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--marker-detection-run", required=True)
    ap.add_argument("--pose-sequence", required=True)
    ap.add_argument("--camera-label", required=True, nargs="+")
    ap.add_argument("--start-time", type=float, required=True)
    ap.add_argument("--end-time", type=float, required=True)
    ap.add_argument("--slow-factor", type=float, default=4.0)
    ap.add_argument("--video-out", required=True)
    ap.add_argument("--mode", choices=["dots", "tracklets"], default="dots")
    ap.add_argument("--full-frame", action="store_true", help="Show the whole frame, not a leg-region crop.")
    ap.add_argument("--cell-width", type=int, default=700)
    ap.add_argument("--cell-height", type=int, default=520)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    states = load_camera_states(conn, args.shot_id)
    sync_table, svid_by_cam = load_sync_table(conn, args.shot_id)

    cam_ids, file_paths, native_fps, cam_range = [], {}, {}, {}
    for label in args.camera_label:
        cam_id = conn.execute("SELECT id FROM camera_instances WHERE label = ?", (label,)).fetchone()["id"]
        cam_ids.append(cam_id)
        svid = svid_by_cam[cam_id]
        row = conn.execute("SELECT file_path, actual_fps FROM capture_videos WHERE id = ?", (svid,)).fetchone()
        file_paths[cam_id] = row["file_path"]
        native_fps[cam_id] = float(row["actual_fps"])

    ref_cam = cam_ids[0]
    ref_svid = svid_by_cam[ref_cam]
    frame_lo = sync_table.lookup(args.start_time, ref_svid)
    frame_hi = sync_table.lookup(args.end_time, ref_svid)
    if frame_lo is None or frame_hi is None:
        raise SystemExit("no sync data for the reference camera in this range")

    windows = {}
    for cam_id in cam_ids:
        svid = svid_by_cam[cam_id]
        lo = sync_table.lookup(args.start_time, svid) or frame_lo
        hi = sync_table.lookup(args.end_time, svid) or frame_hi
        cam_range[cam_id] = (lo, hi)
        if args.full_frame:
            windows[cam_id] = None
        else:
            w = _crop_window(conn, args.pose_sequence, cam_id, lo, hi)
            windows[cam_id] = w  # may be None -> falls back to full frame below

    cols = int(np.ceil(np.sqrt(len(cam_ids))))
    rows = int(np.ceil(len(cam_ids) / cols))
    cell_w, cell_h = args.cell_width, args.cell_height

    decoders = {c: iter_frames(file_paths[c], cam_range[c][0], cam_range[c][1] + 1) for c in cam_ids}
    next_frame = {c: next(decoders[c]) for c in cam_ids}

    def _advance_to(cam_id, target):
        fidx, img = next_frame[cam_id]
        try:
            while fidx < target:
                fidx, img = next(decoders[cam_id])
        except StopIteration:
            pass
        next_frame[cam_id] = (fidx, img)
        return img

    out_path = Path(args.video_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    out_fps = native_fps[ref_cam] / args.slow_factor
    n_written = 0

    for ref_frame in range(frame_lo, frame_hi):
        t = sync_table.frame_to_global_time(ref_frame, ref_svid)
        if t is None:
            continue
        canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        for i, cam_id in enumerate(cam_ids):
            svid = svid_by_cam[cam_id]
            fidx = sync_table.lookup(t, svid)
            if fidx is None:
                continue
            img = _advance_to(cam_id, fidx)
            dot_row = conn.execute(
                "SELECT keypoints FROM detection_keypoints WHERE detection_run_id = ? AND shot_video_id = ? "
                "AND video_frame = ? AND region_type = 'dots'",
                (args.marker_detection_run, svid, fidx),
            ).fetchone()
            dots = decode_dot_candidates(bytes(dot_row["keypoints"])) if dot_row else np.zeros((0, 9))

            win = windows[cam_id]
            if win is None:
                win = (0, 0, img.shape[1], img.shape[0])
            cell, transform, scale = _prepare_cell(img, win, cell_w, cell_h)

            for cand in dots:
                px, py, _area, _comp, major, minor, _dx, _dy, tid = cand
                tx, ty = transform(px, py)
                p = (int(round(tx)), int(round(ty)))
                if args.mode == "tracklets":
                    color = _tracklet_color(int(tid))
                    cv2.circle(cell, p, 6, color, -1)
                    cv2.putText(cell, str(int(tid)), (p[0] + 7, p[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                else:
                    r = max(4, int(round(minor * scale * 0.5)))
                    cv2.circle(cell, p, r, (0, 255, 255), 2)

            cv2.putText(cell, f"{args.camera_label[i]}  f={fidx}  {len(dots)} dots",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, _STAMP_FONT_SCALE, (0, 0, 0),
                        _STAMP_THICKNESS + 2, cv2.LINE_AA)
            cv2.putText(cell, f"{args.camera_label[i]}  f={fidx}  {len(dots)} dots",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, _STAMP_FONT_SCALE, (255, 255, 255),
                        _STAMP_THICKNESS, cv2.LINE_AA)
            r, c = divmod(i, cols)
            canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cell

        if writer is None:
            h, w = canvas.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
        writer.write(canvas)
        n_written += 1

    if writer is not None:
        writer.release()
    print(f"wrote {n_written} frames at {out_fps:.1f}fps ({args.slow_factor}x) [{args.mode}] -> {out_path}")


if __name__ == "__main__":
    main()
