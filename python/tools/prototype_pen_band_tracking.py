# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prototype_pen_band_tracking.py -- tracks the pen's 2 reflective bands
(identified 2026-09-13: one near the ArUco tags, one at the pen's writing
tip -- see status.md) through the real multi-camera writing-scene footage,
and computes each band's fixed 3D offset in marker-2's own local frame.

Why this exists, rather than reusing calibrate_rigid_marker_body.py's
--detect-dots path directly: that path's dot correspondence needs >=2
cameras to each see *exactly one* candidate the same instant, which a
2026-09-13 diagnostic showed never happens in this footage -- even in the
calmest, one-person part of the writing scene, background-subtraction
residue from the subject's own moving clothing/hair yields 14-30
candidates per camera per frame (down from ~150 during the pen handoff,
but nowhere near the 1 that filter needs). A single-frame ambiguity filter
can't resolve that; what's needed is temporal continuity.

Mechanism:
  1. Per camera, bootstrap each band's pixel track from ONE manually-
     confirmed (frame, approximate pixel) seed -- automatic bootstrap
     heuristics (nearest-to-tag, compactness) were tried and shown
     (status.md, 2026-09-13) to be unreliable in this cluttered scene; a
     human-confirmed seed removes the hardest part (initial acquisition),
     leaving only "maintain track across frames", which nearest-candidate-
     within-gate_px already does well (validated for the pen calibration
     *video*'s tag-2 dot in prototype_calibrate_pen_and_pad.py).
  2. Track forward-only from the seed frame to the end of the calm window
     (never reseeds from a skipped frame -- same policy as that earlier
     tracker).
  3. Separately, collect per-bucket marker-2 world poses the same way
     calibrate_rigid_marker_body.py does (multi-camera ArUco corner
     co-occurrence).
  4. Combine: each (camera, frame) band sighting becomes one row in a
     single multi-view DLT triangulation for the band's position in
     marker 2's own local frame, via a "virtual" projection matrix
     P_virtual = P_camera @ [R2(instant) t2(instant); 0 0 0 1] that
     composes the camera's own fixed projection with that instant's
     already-solved marker-2 world pose. No two sightings need be
     simultaneous -- this is the same cross-time/cross-camera
     co-occurrence bridging calibrate_rigid_marker_body.py's own ArUco
     corner registration already relies on, just applied to an unknown
     point instead of a known corner offset.

Usage:
    python tools/prototype_pen_band_tracking.py \\
        --session D:/mocap/2026-09-06-kare-tests/2026-09-06-kare-tests.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --time-start 101.0 --time-end 111.5 \\
        --marker-size 0.095
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.setup.extrinsics_solver import (  # noqa: E402
    _proj_matrix, _undistort_pts, marker_local_corners, solve_marker_pose,
)
from app.setup.fiducial_markers import ArucoDetector  # noqa: E402
from posetrak.detection.dot_blob_detector import compute_background, detect_blobs  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states, load_sync_table  # noqa: E402

# Per-camera bootstrap seeds, one per band, found by direct visual
# inspection of the real capture footage (see status.md, 2026-09-13) --
# not meant to be rediscovered automatically; see module docstring for why.
# (camera_instance_id, video_frame, (px, py) raw/distorted pixel coords)
SEEDS: dict[str, dict[str, tuple[int, tuple[float, float]]]] = {
    "tip": {
        # Found by direct visual inspection of the calm (one-person) part of
        # the writing scene, then confirmed as the nearest real detect_blobs()
        # candidate to the eyeballed pixel location (status.md, 2026-09-13).
        "24c26464-38f5-478d-a98a-d75a7e4eeddd": (14357, (1629.2, 1144.6)),  # gopro13_02
        "50819e8c-6ba4-44b4-aeee-23e1ddce2061": (22033, (1765.6, 955.1)),  # pixel9
    },
    "top": {},  # not seeded this pass -- not cleanly separable from marker-
                # edge residue in the one frame checked; tip band is both
                # the easier target and the one accuracy-critical for the
                # pen-tip-to-pad use case (see status.md).
}

REFERENCE_MARKER_ID = "2"
MARKER_IDS = {"2", "3"}


def _virtual_projection(P_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """P_cam (3x4, world->pixel) composed with a marker's own world pose
    (R, t mapping marker-local -> world) into one 3x4 matrix that maps a
    marker-local-frame point directly to pixel coordinates."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return P_cam @ T


def triangulate_from_virtual_views(
    rows: list[tuple[np.ndarray, tuple[float, float]]],
    max_reproj_px: float = 15.0,
) -> tuple[np.ndarray | None, list[float]]:
    """DLT triangulation of one local-frame point from an arbitrary list of
    (P_virtual, pixel) observations -- unlike calibrate_rigid_marker_body.
    triangulate_point_multi_view(), rows need not come from distinct
    cameras or the same instant (that's the whole point here: pooling
    every tracked sighting across both cameras' entire tracked range).
    One round of reprojection-error outlier trimming, mirroring
    prototype_calibrate_pen_and_pad.py's _robust_triangulate()."""
    def _solve(rows_in):
        A = []
        for P, (u, v) in rows_in:
            A.append(u * P[2] - P[0])
            A.append(v * P[2] - P[1])
        A = np.stack(A)
        _, _, vt = np.linalg.svd(A)
        x = vt[-1]
        if abs(x[3]) < 1e-9:
            return None
        return x[:3] / x[3]

    if len(rows) < 2:
        return None, []
    point = _solve(rows)
    if point is None:
        return None, []
    point_h = np.append(point, 1.0)
    errors = []
    for P, (u, v) in rows:
        proj = P @ point_h
        if abs(proj[2]) < 1e-9:
            errors.append(float("inf"))
            continue
        errors.append(float(np.linalg.norm(proj[:2] / proj[2] - np.array([u, v]))))

    inliers = [r for r, e in zip(rows, errors) if e <= max_reproj_px]
    if len(inliers) < 2 or len(inliers) == len(rows):
        return point, errors
    point2 = _solve(inliers)
    if point2 is None:
        return point, errors
    point2_h = np.append(point2, 1.0)
    errors2 = []
    for P, (u, v) in rows:
        proj = P @ point2_h
        errors2.append(float("inf") if abs(proj[2]) < 1e-9
                        else float(np.linalg.norm(proj[:2] / proj[2] - np.array([u, v]))))
    return point2, errors2


def track_band_forward(
    file_path: str, seed_frame: int, last_frame: int, background: np.ndarray,
    seed_xy: tuple[float, float], threshold: float, min_area: float,
    max_area: float, min_compactness: float, gate_px: float,
) -> dict[int, tuple[float, float]]:
    """Bootstrap at (seed_frame, seed_xy), then track forward to
    last_frame via nearest-candidate-within-gate_px; stops (silently, for
    the rest of the range) the first frame no candidate qualifies --
    never reseeds from a skipped frame."""
    tracks: dict[int, tuple[float, float]] = {seed_frame: seed_xy}
    pos = seed_xy
    lost = False
    for vf, img in iter_frames(file_path, seed_frame, last_frame):
        if vf == seed_frame:
            continue
        if lost:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blobs = detect_blobs(gray, threshold=threshold, min_area=min_area,
                              max_area=max_area, min_compactness=min_compactness,
                              background=background)
        best, best_d = None, gate_px
        for b in blobs:
            d = float(np.hypot(b.cx - pos[0], b.cy - pos[1]))
            if d <= best_d:
                best, best_d = b, d
        if best is None:
            lost = True
            continue
        pos = (best.cx, best.cy)
        tracks[vf] = pos
    return tracks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--time-start", type=float, required=True)
    ap.add_argument("--time-end", type=float, required=True)
    ap.add_argument("--marker-size", type=float, required=True)
    ap.add_argument("--stride", type=int, default=2, help="Corner-detection stride (marker pose only)")
    ap.add_argument("--min-cameras", type=int, default=2)
    ap.add_argument("--dot-bg-samples", type=int, default=40)
    ap.add_argument("--dot-threshold", type=float, default=45.0)
    ap.add_argument("--dot-min-area", type=float, default=3.0)
    ap.add_argument("--dot-max-area", type=float, default=300.0)
    ap.add_argument("--dot-min-compactness", type=float, default=0.3)
    ap.add_argument("--gate-px", type=float, default=35.0)
    ap.add_argument("--max-reproj-px", type=float, default=15.0)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    states = load_camera_states(conn, args.shot_id)
    sync_table, svid_by_cam = load_sync_table(conn, args.shot_id)
    detector = ArucoDetector(dictionary="DICT_4X4_50")

    # ---- Pass 1: per-bucket marker corner collection across ALL cameras
    # (same mechanism as calibrate_rigid_marker_body.py) -- gives per-
    # instant marker-2 world poses, the anchor every band sighting (from
    # either camera, at whatever instant) gets triangulated against.
    frame_obs: dict[float, dict[str, dict[str, np.ndarray]]] = {}
    band_raw_tracks: dict[str, dict[str, dict[int, tuple[float, float]]]] = {
        "tip": {}, "top": {},
    }

    for cam_id, state in states.items():
        svid = svid_by_cam.get(cam_id)
        if svid is None:
            continue
        first = sync_table.lookup(args.time_start, svid)
        last = sync_table.lookup(args.time_end, svid)
        if first is None or last is None:
            print(f"  SKIP {cam_id[:8]}: no sync coverage")
            continue

        needs_bands = cam_id in SEEDS["tip"] or cam_id in SEEDS["top"]
        background = None
        if needs_bands:
            step = max(1, (last - first) // args.dot_bg_samples)
            bg_frames = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                         for vf, img in iter_frames(state.file_path, first, last)
                         if (vf - first) % step == 0]
            background = compute_background(bg_frames)
            print(f"camera {cam_id[:8]}: background from {len(bg_frames)} frames")

        print(f"camera {cam_id[:8]}: frames {first}-{last} ({state.file_path})")
        n_decoded = 0
        for video_frame, img in iter_frames(state.file_path, first, last):
            n_decoded += 1
            if (video_frame - first) % args.stride != 0:
                continue
            dets = detector.detect(img, video_id=cam_id, frame_idx=video_frame)
            gt = sync_table.frame_to_global_time(video_frame, svid)
            if gt is None:
                continue
            bucket = round(gt / 0.05) * 0.05
            for d in dets:
                if d.marker_id not in MARKER_IDS:
                    continue
                pts = np.array([(c.px, c.py) for c in d.corners], dtype=np.float64)
                pts_undist = _undistort_pts(pts, state)
                frame_obs.setdefault(bucket, {}).setdefault(d.marker_id, {})[cam_id] = pts_undist
        print(f"  {n_decoded} frames decoded (corners)")

        if needs_bands:
            for band, seeds in SEEDS.items():
                if cam_id not in seeds:
                    continue
                seed_frame, seed_xy = seeds[cam_id]
                tr = track_band_forward(
                    state.file_path, seed_frame, last, background, seed_xy,
                    args.dot_threshold, args.dot_min_area, args.dot_max_area,
                    args.dot_min_compactness, args.gate_px,
                )
                band_raw_tracks[band][cam_id] = tr
                print(f"  band '{band}' on {cam_id[:8]}: tracked {len(tr)} frames "
                      f"from seed {seed_frame} to {last}")

    # ---- Pass 2: solve marker-2 world pose per bucket.
    ref_pose_by_bucket: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for bucket, by_marker in sorted(frame_obs.items()):
        ref_obs = by_marker.get(REFERENCE_MARKER_ID)
        if ref_obs is None or len(ref_obs) < args.min_cameras:
            continue
        try:
            rvec, tvec, _ = solve_marker_pose(ref_obs, states, args.marker_size)
        except (ValueError, RuntimeError):
            continue
        R, _ = cv2.Rodrigues(rvec)
        ref_pose_by_bucket[bucket] = (R, tvec.flatten())
    print(f"\nMarker '{REFERENCE_MARKER_ID}' world pose solved in {len(ref_pose_by_bucket)} buckets")

    # ---- Pass 3: build virtual-view rows per band, pooling every tracked
    # (camera, frame) sighting whose bucket has a solved marker-2 pose.
    for band, per_cam_tracks in band_raw_tracks.items():
        rows: list[tuple[np.ndarray, tuple[float, float]]] = []
        n_total_tracked = 0
        for cam_id, tr in per_cam_tracks.items():
            state = states[cam_id]
            svid = svid_by_cam.get(cam_id)
            P_cam = _proj_matrix(state)
            n_total_tracked += len(tr)
            for vf, (px, py) in tr.items():
                gt = sync_table.frame_to_global_time(vf, svid)
                if gt is None:
                    continue
                bucket = round(gt / 0.05) * 0.05
                pose = ref_pose_by_bucket.get(bucket)
                if pose is None:
                    continue
                R, t = pose
                pt_undist = _undistort_pts(np.array([[px, py]]), state)[0]
                P_virtual = _virtual_projection(P_cam, R, t)
                rows.append((P_virtual, (float(pt_undist[0]), float(pt_undist[1]))))
        print(f"\nband '{band}': {n_total_tracked} raw tracked sightings, "
              f"{len(rows)} usable (bucket had a solved marker-2 pose)")
        if len(rows) < 2:
            print("  not enough usable rows -- skipping")
            continue
        point, errors = triangulate_from_virtual_views(rows, args.max_reproj_px)
        if point is None:
            print("  triangulation failed")
            continue
        finite = [e for e in errors if np.isfinite(e)]
        inliers = [e for e in finite if e <= args.max_reproj_px]
        print(f"  local-frame offset (marker '{REFERENCE_MARKER_ID}'): {point}")
        print(f"  reprojection error: median={np.median(finite):.2f}px "
              f"p90={np.percentile(finite, 90):.2f}px max={np.max(finite):.2f}px "
              f"({len(inliers)}/{len(finite)} rows <= {args.max_reproj_px}px)")


if __name__ == "__main__":
    main()
