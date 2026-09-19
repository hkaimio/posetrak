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
    dim cyan ring              every other raw candidate, unassigned

Accepts one or more --camera-label values. A single camera renders exactly
as before (full-resolution crop); two or more are tiled into a grid, one
cell per camera, so a same-frame issue can be checked across every
camera's own view at once instead of guessing whether it's camera-specific
(2026-09-09: the frame-7920 heel/ankle swap review asked for exactly this
-- see status.md).

Usage:
    python tools/render_hybrid_assignment_video.py \\
        --session D:/mocap/2026-09-06-kare-tests/2026-09-06-kare-tests.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --marker-detection-run 01c2e3c1-204b-49f8-9b14-6fd611b765a4 \\
        --pose-sequence ec1b3e2f-1ef8-4e31-806c-33102a969ecd \\
        --camera-label gopro-11_mini_01 gopro-12_mini_02 \\
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
from tools.prototype_marker_normal_assignment import normal_aware_hybrid_assign_frame  # noqa: E402
from tools.prototype_person_marker_assignment import _CATALOG  # noqa: E402
from tools.prototype_tracklet_smoothed_assignment import run_tracklet_smoothed  # noqa: E402
from tools.render_person_marker_assignment_video import _marker_colors  # noqa: E402

_PAD_PX = 250
_COLORS = _marker_colors()
_LABEL_FONT_SCALE = 0.6
_LABEL_THICKNESS = 2
_STAMP_FONT_SCALE = 0.9
_STAMP_THICKNESS = 2


def _prepare_cell(img: np.ndarray, window: tuple[int, int, int, int], target_w: int, target_h: int):
    """Crop to *window*, then resize to fit *target_w* x *target_h* -- one
    uniform scale factor for both axes (so the crop's aspect ratio is
    preserved, never stretched), letterboxed (centered, black bars) into
    the exact target size. Returns (canvas, transform), transform mapping
    an (x, y) point in img's own raw pixel space into canvas space.

    Callers must draw annotations onto the returned canvas (via the
    returned transform), not onto the pre-resize crop -- circle/line/font
    sizes are specified in canvas pixels, so drawing before this resize and
    scaling the whole image down afterwards blurs them away and defeats
    the point of choosing a font size at all (2026-09-09, see status.md).
    """
    x0, y0, x1, y1 = window
    crop = img[y0:min(y1, img.shape[0]), x0:min(x1, img.shape[1])]
    crop_h, crop_w = crop.shape[:2]
    scale = min(target_w / crop_w, target_h / crop_h)
    new_w, new_h = max(1, round(crop_w * scale)), max(1, round(crop_h * scale))
    resized = cv2.resize(crop, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    off_x, off_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized

    def transform(x: float, y: float) -> tuple[float, float]:
        return (x - x0) * scale + off_x, (y - y0) * scale + off_y

    return canvas, transform


def _draw_frame(
    canvas: np.ndarray, dots_raw: np.ndarray, assigns: dict[str, tuple[float, float, str]], transform,
) -> None:
    """Draws in place onto *canvas* (already cropped+resized to its final
    output size -- see _prepare_cell), mapping every point through
    *transform* first."""
    assigned_orig = {(round(px), round(py)) for px, py, _conf in assigns.values()}
    for cx, cy in dots_raw:
        if (round(cx), round(cy)) in assigned_orig:
            continue
        tx, ty = transform(cx, cy)
        cv2.circle(canvas, (int(round(tx)), int(round(ty))), 5, (255, 255, 0), 1)

    for name, (px, py, conf) in assigns.items():
        color = _COLORS[name]
        tx, ty = transform(px, py)
        p = (int(round(tx)), int(round(ty)))
        if conf == "3d":
            cv2.circle(canvas, p, 3, color, -1)
            cv2.circle(canvas, p, 9, color, 2)
            label = f"{name} (3D)"
        else:
            cv2.circle(canvas, p, 7, color, 1)
            label = name
        cv2.putText(canvas, label, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    _LABEL_FONT_SCALE, color, _LABEL_THICKNESS, cv2.LINE_AA)


def _crop_window(conn: sqlite3.Connection, pose_sequence: str, cam_id: str, frame_lo: int, frame_hi: int):
    """Fixed crop window from this camera's own leg-keypoint extent over the clip."""
    all_x, all_y = [], []
    for video_frame in range(frame_lo, frame_hi):
        row = conn.execute(
            "SELECT kp_blob FROM pose_observations WHERE sequence_id = ? AND camera_instance_id = ? "
            "AND video_frame = ? AND source = 'body'",
            (pose_sequence, cam_id, video_frame),
        ).fetchone()
        if row is None:
            continue
        kp = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
        for idx, _group in _CATALOG.values():
            if kp[idx, 2] > 0:
                all_x.append(kp[idx, 0])
                all_y.append(kp[idx, 1])
    if not all_x:
        return None
    x0, x1 = max(0, int(min(all_x) - _PAD_PX)), int(max(all_x) + _PAD_PX)
    y0, y1 = max(0, int(min(all_y) - _PAD_PX)), int(max(all_y) + _PAD_PX)
    return x0, y0, x1, y1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--marker-detection-run", required=True)
    ap.add_argument("--pose-sequence", required=True)
    ap.add_argument("--camera-label", required=True, nargs="+",
                     help="One camera renders full-resolution; two or more are tiled into a grid.")
    ap.add_argument("--start-time", type=float, required=True)
    ap.add_argument("--end-time", type=float, required=True)
    ap.add_argument("--slow-factor", type=float, default=4.0)
    ap.add_argument("--video-out", required=True)
    ap.add_argument("--cell-width", type=int, default=640,
                     help="Grid cell size when rendering 2+ cameras. Ignored for a single camera, "
                          "which renders at its own native crop resolution unless this is given.")
    ap.add_argument("--cell-height", type=int, default=480)
    ap.add_argument("--tracklet-smoothing", action="store_true",
                     help="Apply prototype_tracklet_smoothed_assignment.py's majority-vote pass "
                          "on top of the per-frame hybrid assignment before drawing.")
    ap.add_argument("--use-normals", action="store_true",
                     help="Use prototype_marker_normal_assignment.py's directionally-aware "
                          "assignment for ambiguous same-joint groups (knee/ankle) instead of "
                          "hybrid_assign_frame's undifferentiated shared-anchor matching.")
    ap.add_argument("--min-direction-score", type=float, default=0.0,
                     help="Only with --use-normals: reject a candidate/name pairing below this "
                          "cosine similarity instead of merely preferring better-aligned ones.")
    ap.add_argument("--disambiguate-2d-fallback", action="store_true",
                     help="Only with --use-normals: also attempt directional disambiguation in "
                          "the single-camera 2D fallback pass. Off by default -- found to "
                          "confidently mislabel a real case (see prototype_marker_normal_"
                          "assignment.py's docstring), since a single camera's projected-offset "
                          "geometry is much less reliable than the 3D pass.")
    args = ap.parse_args()
    assign_fn = hybrid_assign_frame
    if args.use_normals:
        from functools import partial
        assign_fn = partial(normal_aware_hybrid_assign_frame, min_direction_score=args.min_direction_score,
                             disambiguate_2d_fallback=args.disambiguate_2d_fallback)

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    states = load_camera_states(conn, args.shot_id)
    sync_table, svid_by_cam = load_sync_table(conn, args.shot_id)

    cam_ids, file_paths, native_fps_by_cam = [], {}, {}
    for label in args.camera_label:
        cam_id = conn.execute("SELECT id FROM camera_instances WHERE label = ?", (label,)).fetchone()["id"]
        cam_ids.append(cam_id)
        svid = svid_by_cam[cam_id]
        row = conn.execute("SELECT file_path, actual_fps FROM capture_videos WHERE id = ?", (svid,)).fetchone()
        file_paths[cam_id] = row["file_path"]
        native_fps_by_cam[cam_id] = float(row["actual_fps"])

    ref_cam = cam_ids[0]
    ref_svid = svid_by_cam[ref_cam]
    frame_lo = sync_table.lookup(args.start_time, ref_svid)
    frame_hi = sync_table.lookup(args.end_time, ref_svid)
    if frame_lo is None or frame_hi is None:
        raise SystemExit("no sync data for the reference camera in the requested range")

    windows, cam_frame_range = {}, {}
    for cam_id in cam_ids:
        svid = svid_by_cam[cam_id]
        cam_frame_lo = sync_table.lookup(args.start_time, svid) or frame_lo
        cam_frame_hi = sync_table.lookup(args.end_time, svid) or frame_hi
        cam_frame_range[cam_id] = (cam_frame_lo, cam_frame_hi)
        w = _crop_window(conn, args.pose_sequence, cam_id, cam_frame_lo, cam_frame_hi)
        if w is None:
            raise SystemExit(f"no leg keypoints found for camera {cam_id} in this range")
        windows[cam_id] = w

    smoothed_results = None
    if args.tracklet_smoothing:
        smoothed_results = run_tracklet_smoothed(
            conn, args.shot_id, args.marker_detection_run, args.pose_sequence,
            args.start_time, args.end_time, ref_camera_id=ref_cam, assign_fn=assign_fn,
        )

    grid = len(cam_ids) > 1
    cols = int(np.ceil(np.sqrt(len(cam_ids)))) if grid else 1
    rows = int(np.ceil(len(cam_ids) / cols)) if grid else 1
    cell_w, cell_h = args.cell_width, args.cell_height

    # Sequential per-camera decoders, all driven off the reference camera's own frame
    # timeline -- each camera's frame index at every reference timestamp is looked up
    # independently via the shared sync table, same pattern as render_grid_video().
    decoders = {
        cam_id: iter_frames(file_paths[cam_id], cam_frame_range[cam_id][0], cam_frame_range[cam_id][1] + 1)
        for cam_id in cam_ids
    }
    next_frame = {cam_id: next(decoders[cam_id]) for cam_id in cam_ids}

    def _advance_to(cam_id: str, target_fidx: int):
        fidx, img = next_frame[cam_id]
        try:
            while fidx < target_fidx:
                fidx, img = next(decoders[cam_id])
        except StopIteration:
            pass  # ran off the end of this camera's own decode range -- reuse its last frame
        next_frame[cam_id] = (fidx, img)
        return img

    out_path = Path(args.video_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    out_fps = native_fps_by_cam[ref_cam] / args.slow_factor
    n_written = 0

    for ref_frame in range(frame_lo, frame_hi):
        t = sync_table.frame_to_global_time(ref_frame, ref_svid)
        if t is None:
            continue

        dots_by_cam, kp_by_cam, fidx_by_cam = {}, {}, {}
        for cam_id in cam_ids:
            svid = svid_by_cam[cam_id]
            fidx = sync_table.lookup(t, svid)
            if fidx is None:
                continue
            fidx_by_cam[cam_id] = fidx
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
            result = smoothed_results.get(ref_frame, {})
        else:
            result = assign_fn(dots_by_cam, kp_by_cam, states) if kp_by_cam else {}

        canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) if grid else None

        for i, cam_id in enumerate(cam_ids):
            fidx = fidx_by_cam.get(cam_id)
            if fidx is None:
                continue
            img = _advance_to(cam_id, fidx)
            this_dots = dots_by_cam.get(cam_id, np.zeros((0, 2)))
            this_assigns = result.get(cam_id, {})
            x0, y0, x1, y1 = windows[cam_id]
            if grid:
                target_w, target_h = cell_w, cell_h
            else:
                # No forced downscale for a single camera -- render at its own native crop size.
                target_w = min(x1, img.shape[1]) - x0
                target_h = min(y1, img.shape[0]) - y0
            cell, transform = _prepare_cell(img, (x0, y0, x1, y1), target_w, target_h)
            _draw_frame(cell, this_dots, this_assigns, transform)
            cv2.putText(cell, f"{args.camera_label[i]} t={t:.3f}s frame={fidx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, _STAMP_FONT_SCALE, (255, 255, 255), _STAMP_THICKNESS, cv2.LINE_AA)

            if grid:
                r, c = divmod(i, cols)
                canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cell
            else:
                canvas = cell

        if canvas is None:
            continue
        if writer is None:
            h, w = canvas.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
        writer.write(canvas)
        n_written += 1

    if writer is not None:
        writer.release()
    print(f"wrote {n_written} frames ({n_written / native_fps_by_cam[ref_cam]:.2f}s of real time) at "
          f"{out_fps:.1f}fps ({args.slow_factor}x slow-motion) -> {out_path}")


if __name__ == "__main__":
    main()
