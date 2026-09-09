# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""render_hybrid_assignment_video.py — PROTOTYPE: visual validation video
for prototype_hybrid_person_marker_assignment.py (P7's 3D-fused pass +
P1's per-camera 2D fallback pass, combined).

Same purpose as render_person_marker_assignment_video.py (checking
temporal stability by eye, catching what per-frame aggregate stats can't
show) but for the hybrid assignment, and additionally distinguishing the
two passes visually so it's obvious at a glance how much of a clip's
assignment is cross-camera-confirmed vs. single-view-only:

    filled dot + solid ring   3d-pass match (cross-camera triangulation
                               confirmed this candidate) -- higher
                               confidence
    open ring only            2d-pass match (this camera's own fallback,
                               no cross-camera confirmation this frame)
    dim cyan ring             every other raw candidate, unassigned

Renders ONE camera's own view per invocation (matching every other
per-camera debug video this project already has) -- the hybrid assignment
itself runs across all cameras every frame (that's the whole point of the
3D pass), this just draws the chosen camera's own resulting labels.

Usage:
    python tools/render_hybrid_assignment_video.py \\
        --session D:/mocap/2026-09-06-kare-tests/2026-09-06-kare-tests.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --marker-detection-run 01c2e3c1-204b-49f8-9b14-6fd611b765a4 \\
        --pose-sequence ec1b3e2f-1ef8-4e31-806c-33102a969ecd \\
        --camera-label gopro-11_mini_01 \\
        --start-time 40.0 --end-time 50.0 \\
        --slow-factor 4 \\
        --video-out scratch/dot_ground_truth/hybrid_assignment_check.mp4
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pose.db_cache import decode_dot_candidates  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states, load_sync_table  # noqa: E402
from tools.prototype_hybrid_person_marker_assignment import hybrid_assign_frame  # noqa: E402
from tools.prototype_person_marker_assignment import _CATALOG  # noqa: E402
from tools.prototype_tracklet_smoothed_assignment import run_tracklet_smoothed  # noqa: E402
from tools.render_person_marker_assignment_video import _marker_colors  # noqa: E402

_PAD_PX = 250
_COLORS = _marker_colors()


def _draw_frame(
    img: np.ndarray, dots_raw: np.ndarray, assigns: dict[str, tuple[float, float, str]],
) -> np.ndarray:
    out = img.copy()
    assigned_px = {(round(px), round(py)) for px, py, _conf in assigns.values()}
    for cx, cy in dots_raw:
        if (round(cx), round(cy)) in assigned_px:
            continue
        cv2.circle(out, (int(round(cx)), int(round(cy))), 5, (255, 255, 0), 1)

    for name, (px, py, conf) in assigns.items():
        color = _COLORS[name]
        p = (int(round(px)), int(round(py)))
        if conf == "3d":
            cv2.circle(out, p, 3, color, -1)
            cv2.circle(out, p, 9, color, 2)
            label = f"{name} (3D)"
        else:
            cv2.circle(out, p, 7, color, 1)
            label = name
        cv2.putText(out, label, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--marker-detection-run", required=True)
    ap.add_argument("--pose-sequence", required=True)
    ap.add_argument("--camera-label", required=True)
    ap.add_argument("--start-time", type=float, required=True)
    ap.add_argument("--end-time", type=float, required=True)
    ap.add_argument("--slow-factor", type=float, default=4.0)
    ap.add_argument("--video-out", required=True)
    ap.add_argument("--tracklet-smoothing", action="store_true",
                     help="Apply prototype_tracklet_smoothed_assignment.py's majority-vote pass "
                          "on top of the per-frame hybrid assignment before drawing.")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    states = load_camera_states(conn, args.shot_id)
    sync_table, svid_by_cam = load_sync_table(conn, args.shot_id)
    target_cam_id = conn.execute(
        "SELECT id FROM camera_instances WHERE label = ?", (args.camera_label,)
    ).fetchone()["id"]
    target_svid = svid_by_cam[target_cam_id]
    file_path = conn.execute("SELECT file_path, actual_fps FROM capture_videos WHERE id = ?", (target_svid,)).fetchone()
    native_fps = float(file_path["actual_fps"])
    file_path = file_path["file_path"]

    frame_lo = sync_table.lookup(args.start_time, target_svid)
    frame_hi = sync_table.lookup(args.end_time, target_svid)
    if frame_lo is None or frame_hi is None:
        raise SystemExit("no sync data for this camera in the requested range")

    # Fixed crop window from the target camera's own leg-keypoint extent over the clip.
    all_x, all_y = [], []
    for video_frame in range(frame_lo, frame_hi):
        row = conn.execute(
            "SELECT kp_blob FROM pose_observations WHERE sequence_id = ? AND camera_instance_id = ? "
            "AND video_frame = ? AND source = 'body'",
            (args.pose_sequence, target_cam_id, video_frame),
        ).fetchone()
        if row is None:
            continue
        kp = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
        for idx, _group in _CATALOG.values():
            if kp[idx, 2] > 0:
                all_x.append(kp[idx, 0])
                all_y.append(kp[idx, 1])
    if not all_x:
        raise SystemExit("no leg keypoints found for the target camera in this range")
    x0, x1 = max(0, int(min(all_x) - _PAD_PX)), int(max(all_x) + _PAD_PX)
    y0, y1 = max(0, int(min(all_y) - _PAD_PX)), int(max(all_y) + _PAD_PX)

    smoothed_results = None
    if args.tracklet_smoothing:
        smoothed_results = run_tracklet_smoothed(
            conn, args.shot_id, args.marker_detection_run, args.pose_sequence,
            args.start_time, args.end_time, ref_camera_id=target_cam_id,
        )

    out_path = Path(args.video_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    out_fps = native_fps / args.slow_factor
    n_written = 0

    for video_frame, img in iter_frames(file_path, frame_lo, frame_hi):
        t = sync_table.frame_to_global_time(video_frame, target_svid) or args.start_time

        dots_by_cam, kp_by_cam = {}, {}
        for cam_id, svid in svid_by_cam.items():
            if cam_id not in states:
                continue
            fidx = sync_table.lookup(t, svid)
            if fidx is None:
                continue
            dot_row = conn.execute(
                "SELECT keypoints FROM detection_keypoints WHERE detection_run_id = ? AND shot_video_id = ? "
                "AND video_frame = ? AND region_type = 'dots'",
                (args.marker_detection_run, svid, fidx),
            ).fetchone()
            dots_by_cam[cam_id] = decode_dot_candidates(bytes(dot_row["keypoints"]))[:, :2] if dot_row else np.zeros((0, 2))
            pose_row = conn.execute(
                "SELECT kp_blob FROM pose_observations WHERE sequence_id = ? AND camera_instance_id = ? "
                "AND video_frame = ? AND source = 'body'",
                (args.pose_sequence, cam_id, fidx),
            ).fetchone()
            if pose_row is not None:
                kp_by_cam[cam_id] = np.frombuffer(bytes(pose_row["kp_blob"]), dtype=np.float32).reshape(-1, 3)

        if smoothed_results is not None:
            result = smoothed_results.get(video_frame, {})
        else:
            result = hybrid_assign_frame(dots_by_cam, kp_by_cam, states) if kp_by_cam else {}
        this_cam_assigns = result.get(target_cam_id, {})
        this_cam_dots = dots_by_cam.get(target_cam_id, np.zeros((0, 2)))

        rendered = _draw_frame(img, this_cam_dots, this_cam_assigns)
        rendered = rendered[y0:min(y1, rendered.shape[0]), x0:min(x1, rendered.shape[1])]
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
