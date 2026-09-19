# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""render_person_marker_assignment_video.py — PROTOTYPE: visual validation
video for prototype_person_marker_assignment.py's per-frame one-to-one
(Hungarian) assignment (person-marker-assignment-design.md phase P1).

Built specifically to check something the P1 script's own aggregate
recall/coverage numbers cannot show: *temporal stability*. P1 assigns
every frame completely independently, with no tracklet continuity at all
-- a real risk is that even a genuinely well-matched marker flickers
between two nearby candidates from one frame to the next, which would be
invisible in a per-frame hit-rate statistic but immediately obvious
watching a clip. This is the validation step the design doc's phased plan
calls for before building multi-camera fusion (P7) on top of this
foundation.

Each output frame shows, cropped to the region around the person's leg
keypoints:
    dim cyan ring       every raw dot candidate this frame (available,
                         whether assigned or not)
    faint colored cross  each catalog marker's pose-keypoint anchor
                         position (where its search was centered)
    bright colored dot
      + name label       a catalog marker actually assigned this frame,
                         at its assigned candidate's position -- color is
                         fixed per marker NAME (16 distinct hues), so
                         watching the same color jump between two
                         different physical dots frame-to-frame is the
                         signature of exactly the instability this video
                         exists to catch.

Usage:
    python tools/render_person_marker_assignment_video.py \\
        --session /path/to/session.db \\
        --marker-detection-run 01c2e3c1-204b-49f8-9b14-6fd611b765a4 \\
        --pose-sequence ec1b3e2f-1ef8-4e31-806c-33102a969ecd \\
        --camera-label gopro-11_mini_01 \\
        --start-time 40.0 --end-time 50.0 \\
        --slow-factor 4 \\
        --video-out scratch/dot_ground_truth/p1_assignment_check.mp4
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.pose.db_cache import decode_dot_candidates  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402
from tools.calibrate_rigid_marker_body import load_sync_table  # noqa: E402
from tools.prototype_person_marker_assignment import _CATALOG  # noqa: E402

_PAD_PX = 250


def _marker_colors() -> dict[str, tuple[int, int, int]]:
    """16 perceptually-spread BGR colors, one per catalog marker name, in a
    fixed (sorted) order so the same name always gets the same color across
    runs -- matters for judging temporal stability by eye across separate
    invocations too, not just within one video."""
    import colorsys
    names = sorted(_CATALOG)
    colors = {}
    for i, name in enumerate(names):
        h = i / len(names)
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
        colors[name] = (int(b * 255), int(g * 255), int(r * 255))
    return colors


_COLORS = _marker_colors()


def _draw_frame(img: np.ndarray, dots: np.ndarray, kp: np.ndarray, match_radius_px: float) -> np.ndarray:
    out = img.copy()

    for cx, cy, area, *_rest in dots:
        r = max(3, int(np.sqrt(max(area, 1.0) / np.pi)))
        cv2.circle(out, (int(round(cx)), int(round(cy))), r, (255, 255, 0), 1)

    # assign_frame() (prototype_person_marker_assignment.py) only returns
    # distances/coverage counts, not which candidate index each name got --
    # cheap to recompute the actual per-name assignment directly here for
    # drawing purposes, same cost matrix construction either way.
    names = [n for n, (idx, _g) in _CATALOG.items() if kp[idx, 2] > 0]
    from scipy.optimize import linear_sum_assignment
    name_to_dot: dict[str, int] = {}
    if names and dots.shape[0] > 0:
        cost = np.full((len(names), dots.shape[0]), 1e6)
        for ni, name in enumerate(names):
            idx, _g = _CATALOG[name]
            kx, ky = kp[idx, 0], kp[idx, 1]
            d = np.hypot(dots[:, 0] - kx, dots[:, 1] - ky)
            cost[ni, :] = np.where(d <= match_radius_px, d, 1e6)
        row_ind, col_ind = linear_sum_assignment(cost)
        name_to_dot = {names[ri]: ci for ri, ci in zip(row_ind, col_ind) if cost[ri, ci] < 1e6}

    for name, (idx, _group) in _CATALOG.items():
        if kp[idx, 2] <= 0:
            continue
        color = _COLORS[name]
        kx, ky = kp[idx, 0], kp[idx, 1]
        cv2.drawMarker(out, (int(round(kx)), int(round(ky))), color,
                        markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
        if name in name_to_dot:
            dx, dy = dots[name_to_dot[name], 0], dots[name_to_dot[name], 1]
            cv2.circle(out, (int(round(dx)), int(round(dy))), 7, color, 2)
            cv2.putText(out, name, (int(dx) + 8, int(dy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--marker-detection-run", required=True)
    ap.add_argument("--pose-sequence", required=True)
    ap.add_argument("--camera-label", required=True)
    ap.add_argument("--start-time", type=float, required=True)
    ap.add_argument("--end-time", type=float, required=True)
    ap.add_argument("--match-radius-px", type=float, default=80.0)
    ap.add_argument("--slow-factor", type=float, default=4.0)
    ap.add_argument("--video-out", required=True)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    cam_id = conn.execute("SELECT id FROM camera_instances WHERE label = ?", (args.camera_label,)).fetchone()["id"]
    shot_id = conn.execute(
        "SELECT shot_id FROM pose_observation_sequences WHERE id = ?", (args.pose_sequence,)
    ).fetchone()["shot_id"]
    svid_row = conn.execute(
        "SELECT id, file_path, actual_fps FROM capture_videos WHERE camera_instance_id = ? AND shot_id = ?",
        (cam_id, shot_id),
    ).fetchone()
    svid, file_path, native_fps = svid_row["id"], svid_row["file_path"], float(svid_row["actual_fps"])

    sync_table, _ = load_sync_table(conn, shot_id)
    frame_lo = sync_table.lookup(args.start_time, svid)
    frame_hi = sync_table.lookup(args.end_time, svid)
    if frame_lo is None or frame_hi is None:
        raise SystemExit("no sync data for this camera in the requested range")

    # Fixed crop window over the whole clip (union of leg-keypoint positions,
    # padded) -- same reasoning as render_tracking_debug_frames.py's own
    # _compute_crop_window: a per-frame crop would jump around and make
    # instability impossible to judge by eye, which is the whole point here.
    all_x, all_y = [], []
    for video_frame in range(frame_lo, frame_hi):
        pose_row = conn.execute(
            "SELECT kp_blob FROM pose_observations WHERE sequence_id = ? AND camera_instance_id = ? "
            "AND video_frame = ? AND source = 'body'",
            (args.pose_sequence, cam_id, video_frame),
        ).fetchone()
        if pose_row is None:
            continue
        kp = np.frombuffer(bytes(pose_row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
        for idx, _group in _CATALOG.values():
            if kp[idx, 2] > 0:
                all_x.append(kp[idx, 0])
                all_y.append(kp[idx, 1])
    if not all_x:
        raise SystemExit("no leg keypoints found in this time range -- nothing to crop to")
    x0, x1 = max(0, int(min(all_x) - _PAD_PX)), int(max(all_x) + _PAD_PX)
    y0, y1 = max(0, int(min(all_y) - _PAD_PX)), int(max(all_y) + _PAD_PX)

    out_path = Path(args.video_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    out_fps = native_fps / args.slow_factor
    n_written = 0

    for video_frame, img in iter_frames(file_path, frame_lo, frame_hi):
        dot_row = conn.execute(
            "SELECT keypoints FROM detection_keypoints WHERE detection_run_id = ? AND shot_video_id = ? "
            "AND video_frame = ? AND region_type = 'dots'",
            (args.marker_detection_run, svid, video_frame),
        ).fetchone()
        dots = decode_dot_candidates(bytes(dot_row["keypoints"])) if dot_row is not None else np.zeros((0, 9))
        pose_row = conn.execute(
            "SELECT kp_blob FROM pose_observations WHERE sequence_id = ? AND camera_instance_id = ? "
            "AND video_frame = ? AND source = 'body'",
            (args.pose_sequence, cam_id, video_frame),
        ).fetchone()
        kp = (np.frombuffer(bytes(pose_row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
              if pose_row is not None else np.zeros((133, 3), dtype=np.float32))

        rendered = _draw_frame(img, dots, kp, args.match_radius_px)
        rendered = rendered[y0:min(y1, rendered.shape[0]), x0:min(x1, rendered.shape[1])]
        t = sync_table.frame_to_global_time(video_frame, svid) or args.start_time
        cv2.putText(rendered, f"t={t:.3f}s frame={video_frame}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if writer is None:
            h, w = rendered.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
        writer.write(rendered)
        n_written += 1

    if writer is not None:
        writer.release()
    print(f"wrote {n_written} frames ({n_written / native_fps:.2f}s of real time) at "
          f"{out_fps:.1f}fps ({args.slow_factor}x slow-motion) -> {out_path}")


if __name__ == "__main__":
    main()
