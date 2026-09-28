# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""render_calibration_overlay.py — Render a `marker-body calibrate-video` run
back onto its source footage, with detected and reprojected corners/dots
annotated, for visually checking a solved marker body against the real prop.

Runs the exact same solve as `posetrak marker-body calibrate-video` (same
options, same private helpers from `posetrak.calibration.video_marker_body`)
to get the per-frame camera poses and solved marker/dot geometry, then
re-decodes the video and draws, per fitted frame:

  - yellow rings: the raw detected marker corners for that frame
  - magenta rings: raw detected dot candidates for that frame
  - red quad + label: each marker's corners, reprojected from its solved pose
  - green circle + label: each solved dot, reprojected from its 3D position

A dot whose reprojected (green) position never sits anywhere near the ball's
other dots across the whole video is a real, physical clue that it is not on
the prop at all (a mount/harness, or background clutter the fit mistakenly
kept) -- exactly what plain 3D-offset numbers can't show as directly as a
picture can.

Both detected and reprojected points are drawn in *undistorted* pixel space
(the same space `collect_video_observations` stores its own detections in),
so the background frame is undistorted to match before drawing anything on
it -- overlaying either set directly on the raw, distorted frame would be
off, growing with distance from the image centre.

Example::

    uv run python/tools/render_calibration_overlay.py \\
        --session session.db --video orbit.mp4 --intrinsics <id> \\
        --body-markers 29:0.0457 --reference-id 29 \\
        --anchor-markers 0:0.095,1:0.095,14:0.095,15:0.095 \\
        --dictionary DICT_4X4_1000 --detect-dots --dot-max-area 400 \\
        --output overlay.mp4
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.setup.extrinsics_solver import marker_local_corners  # noqa: E402
from posetrak.calibration.session_cameras import load_intrinsics  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402
from posetrak.calibration.video_marker_body import (  # noqa: E402
    VideoCalibrationOptions,
    _bundle_adjust,
    _chain_from_reference,
    _drop_thin_frames,
    _initial_camera_poses,
    _measure,
    _solve_dots,
    collect_video_observations,
    parse_marker_sizes,
)

_DETECTED_MARKER_COLOR = (0, 255, 255)   # yellow
_DETECTED_DOT_COLOR = (255, 0, 255)      # magenta
_REPROJECTED_MARKER_COLOR = (0, 0, 255)  # red
_REPROJECTED_DOT_COLOR = (0, 255, 0)     # green


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", required=True, help="Session DB (read-only; for the intrinsics lookup).")
    p.add_argument("--video", required=True)
    p.add_argument("--intrinsics", required=True, help="Intrinsics calibration id (prefix ok).")
    p.add_argument("--body-markers", required=True, help="ID[:SIZE],... -- same syntax as calibrate-video.")
    p.add_argument("--body-marker-size", type=float, default=None)
    p.add_argument("--reference-id", required=True)
    p.add_argument("--anchor-markers", default="")
    p.add_argument("--anchor-marker-size", type=float, default=None)
    p.add_argument("--dictionary", default="DICT_4X4_50")
    p.add_argument("--stride", type=int, default=6, help="Same meaning as calibrate-video's own --stride.")
    p.add_argument("--first-frame", type=int, default=None)
    p.add_argument("--last-frame", type=int, default=None)
    p.add_argument("--min-frame-markers", type=int, default=2)
    p.add_argument("--detect-dots", action="store_true")
    p.add_argument("--dot-threshold", type=int, default=235)
    p.add_argument("--dot-min-area", type=float, default=4.0)
    p.add_argument("--dot-max-area", type=float, default=400.0)
    p.add_argument("--dot-min-compactness", type=float, default=0.5)
    p.add_argument("--dot-max-distance-m", type=float, default=1.0)
    p.add_argument("--dot-min-views", type=int, default=12)
    p.add_argument("--dot-min-parallax-deg", type=float, default=2.0)
    p.add_argument("--dot-max-reprojection-px", type=float, default=3.0)
    p.add_argument("--dot-cluster-tolerance-m", type=float, default=0.02)
    p.add_argument("--dot-match-gate-px", type=float, default=8.0)
    p.add_argument("--dot-marker-exclude-scale", type=float, default=1.15)
    p.add_argument("--dot-max-rms-px", type=float, default=None)
    p.add_argument("--output", required=True, help="Annotated video to write (e.g. overlay.mp4).")
    p.add_argument("--render-every", type=int, default=1, help="Render every Nth *fitted* frame (thinning the output further).")
    p.add_argument("--scale", type=float, default=0.5, help="Downscale factor for the output video.")
    p.add_argument("--output-fps", type=float, default=8.0,
                   help="Fixed playback rate for the output video. Not derived from the source "
                        "footage's own fps: consecutive fitted frames are not evenly spaced in "
                        "original-video time (dropped frames), so a derived rate would desync "
                        "perceived motion from the on-screen frame index.")
    return p.parse_args()


def _solve(video: str, state, options: VideoCalibrationOptions, log):
    """Same sequence as `video_marker_body.solve_video_body`, but keeping the
    internal `_Fit` (per-frame camera poses) and per-frame observations this
    tool needs, instead of only the final YAML the public function returns."""
    frames = collect_video_observations(video, state, options, log=log)
    sizes = options.marker_sizes()
    K = state.K
    ref_id = options.reference_id

    measurements = _measure(frames, sizes, K, options.max_pnp_rms_px)
    placed = _chain_from_reference(measurements.pairs, ref_id)
    if len(placed) < 2:
        raise ValueError(f"reference marker {ref_id!r} never seen with another known marker")
    log(f"{len(placed)} markers linked to the reference: {', '.join(sorted(placed))}")

    cameras = _initial_camera_poses(measurements, placed)
    detections = [(f, m) for f, poses in measurements.in_camera.items() for m in poses if m in placed and f in cameras]
    detections = _drop_thin_frames(detections, options.min_frame_markers)
    if not detections:
        raise ValueError("no frame shows enough linked markers for a fit")

    fit = _bundle_adjust(detections, frames, sizes, K, placed, cameras, ref_id)
    rms = np.sqrt(np.mean(np.sum(fit.residuals_px ** 2, axis=2), axis=1))
    keep = rms <= max(3.0, 3.0 * float(np.median(rms)))
    if not keep.all():
        log(f"  dropping {int((~keep).sum())} of {len(keep)} marker sightings that fit badly, fitting again")
        detections = _drop_thin_frames([d for d, k in zip(fit.detections, keep) if k], options.min_frame_markers)
        fit = _bundle_adjust(detections, frames, sizes, K, fit.marker_poses, fit.camera_poses, ref_id)
        rms = np.sqrt(np.mean(np.sum(fit.residuals_px ** 2, axis=2), axis=1))
    total_rms = float(np.sqrt(np.mean(rms ** 2)))
    log(f"Fitted {len(fit.camera_poses)} frames and {len(fit.marker_poses)} markers, reprojection rms {total_rms:.2f} px")

    dots = _solve_dots(frames, fit, K, options, total_rms, log) if options.detect_dots else []
    for i, d in enumerate(dots):
        log(f"  dot{i}: center={d.center} views={d.views} rms_px={d.rms_px:.2f}")

    return frames, fit, dots, sizes


def main() -> None:
    args = _parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        intr = load_intrinsics(conn, args.intrinsics)
    finally:
        conn.close()
    state = intr.state

    options = VideoCalibrationOptions(
        body_markers=parse_marker_sizes(args.body_markers, args.body_marker_size, "--body-markers"),
        reference_id=args.reference_id.strip(),
        anchor_markers=(
            parse_marker_sizes(args.anchor_markers, args.anchor_marker_size, "--anchor-markers")
            if args.anchor_markers else {}
        ),
        dictionary=args.dictionary, stride=args.stride, first_frame=args.first_frame, last_frame=args.last_frame,
        min_frame_markers=args.min_frame_markers, detect_dots=args.detect_dots,
        dot_threshold=args.dot_threshold, dot_min_area=args.dot_min_area, dot_max_area=args.dot_max_area,
        dot_min_compactness=args.dot_min_compactness, dot_max_distance_m=args.dot_max_distance_m,
        dot_min_views=args.dot_min_views, dot_min_parallax_deg=args.dot_min_parallax_deg,
        dot_max_reprojection_px=args.dot_max_reprojection_px, dot_cluster_tolerance_m=args.dot_cluster_tolerance_m,
        dot_match_gate_px=args.dot_match_gate_px, dot_marker_exclude_scale=args.dot_marker_exclude_scale,
        dot_max_rms_px=args.dot_max_rms_px,
    )
    options.validate()

    frames, fit, dots, sizes = _solve(args.video, state, options, log=lambda m: print(m, flush=True))
    # fit.camera_poses is keyed by *position in `frames`* (video_marker_body._measure:
    # `for index, frame in enumerate(frames): ... in_camera[index] = ...`), NOT by the
    # real video frame number (FrameObservation.frame) -- easy to conflate since both
    # are small ints starting near 0, but they are different sequences (`frames` only
    # holds the sparse subset of sampled frames with enough known markers, further
    # thinned by outlier-dropping in the fit itself). Using a fit position as if it
    # were a real frame number pairs each solved pose with a real frame from a nearby
    # but wrong point in the video -- confirmed directly: frame_by_idx.get(0) was None
    # even though 0 is fit.camera_poses' own first key. Map position -> real frame
    # number via `frames[position].frame` throughout, never the position itself.
    K = state.K

    def project(pt_ref: np.ndarray, R_cam: np.ndarray, t_cam: np.ndarray) -> tuple[float, float] | None:
        p_cam = R_cam @ pt_ref + t_cam.flatten()
        if p_cam[2] <= 1e-6:
            return None
        return (
            K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2],
            K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2],
        )

    fitted_positions = sorted(fit.camera_poses)[:: max(1, args.render_every)]
    # position -> real video frame number, and back (positions are unique per frame
    # so this inverse is safe)
    pos_to_real = {pos: frames[pos].frame for pos in fitted_positions}
    real_to_pos = {real: pos for pos, real in pos_to_real.items()}
    real_frames_needed = set(pos_to_real.values())
    writer = None
    n_written = 0
    try:
        # cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, idx) is not frame-accurate on
        # long-GOP footage (GoPro H.264/HEVC): it lands on the nearest keyframe,
        # not the requested frame -- exactly the footage this tool targets. Use
        # the same frame-accurate iter_frames() collect_video_observations
        # itself reads through, so what's drawn always matches what was
        # actually detected/fitted at that frame.
        for real_idx, img in iter_frames(args.video, 0, max(real_frames_needed) + 1):
            if real_idx not in real_frames_needed:
                continue
            pos = real_to_pos[real_idx]
            fobs = frames[pos]
            und = cv2.undistort(img, state.K_orig, state.dist, None, K)
            R_cam, t_cam = fit.camera_poses[pos]
            cv2.putText(und, f"video frame {real_idx}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            # Drawn at full resolution, before the final downscale below -- scale
            # font/circle/line sizes by 1/args.scale so labels stay equally
            # legible on screen regardless of --scale (a small --scale would
            # otherwise shrink text along with the frame).
            inv = 1.0 / max(args.scale, 1e-6)
            for mid, corners in fobs.markers.items():
                for c in corners:
                    cv2.circle(und, (int(c[0]), int(c[1])), round(5 * inv), _DETECTED_MARKER_COLOR, max(1, round(inv)))
            for u, v, _tid, _scale in fobs.dots:
                cv2.circle(und, (int(u), int(v)), round(4 * inv), _DETECTED_DOT_COLOR, max(1, round(inv)))

            for mid, (R_m, t_m) in fit.marker_poses.items():
                local = marker_local_corners(sizes[mid])
                world = (R_m @ local.T).T + t_m.flatten()
                pix = [project(p, R_cam, t_cam) for p in world]
                if all(p is not None for p in pix):
                    pts = np.array(pix, dtype=np.int32)
                    cv2.polylines(und, [pts], True, _REPROJECTED_MARKER_COLOR, max(1, round(inv)))
                    cx, cy = pts.mean(axis=0).astype(int)
                    cv2.putText(und, mid, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * inv,
                                _REPROJECTED_MARKER_COLOR, max(1, round(2 * inv)))

            for i, d in enumerate(dots):
                p = project(d.center, R_cam, t_cam)
                if p is not None:
                    cv2.circle(und, (int(p[0]), int(p[1])), round(7 * inv), _REPROJECTED_DOT_COLOR, max(1, round(2 * inv)))
                    cv2.putText(und, f"dot{i}", (int(p[0]) + round(9 * inv), int(p[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * inv, _REPROJECTED_DOT_COLOR, max(1, round(2 * inv)))

            small = cv2.resize(und, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            if writer is None:
                h, w = small.shape[:2]
                # NOT fps/stride: fitted_frames has gaps wherever a sampled frame was
                # dropped (too few markers, or fit badly and excluded) -- real elapsed
                # time between consecutive rendered frames is therefore irregular, so a
                # rate derived from it desyncs perceived motion even though each frame's
                # own overlay is correct. This is a review tool, not a real-time
                # reconstruction: play back at a fixed, deliberately slow rate instead,
                # and rely on the on-screen frame index (above) for correspondence.
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.output_fps, (w, h))
            writer.write(small)
            n_written += 1
    finally:
        if writer is not None:
            writer.release()

    print(f"Wrote {n_written} frames to {args.output}", flush=True)


if __name__ == "__main__":
    main()
