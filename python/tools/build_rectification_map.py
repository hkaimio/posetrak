#!/usr/bin/env python3
"""
build_rectification_map.py — Non-parametric rectification map from a ChArUco video.

Works for cameras with non-standard projection models (e.g. Insta360 Mega mode)
that do not fit OpenCV's standard or fisheye distortion models.

Algorithm
---------
Each detected board position gives observed (distorted) pixel coordinates for
corners whose 3D positions on the flat board are known.  Using an approximate
camera matrix K (and zero distortion assumed) we solve PnP to estimate the board
pose, then project all corners back through K alone.  The result is the "ideal"
pixel coordinate — where that corner would appear under a perfect pinhole camera.

The gap  observed_distorted → ideal_pinhole  is the local distortion correction.
Accumulating these pairs across all frames and fitting a thin-plate spline (TPS)
over them gives a dense warp map.  A second pass re-fits K on undistorted images
and iterates a few times to reduce the dependency on the initial K approximation.

Output
------
A .npz file containing:
  mapx, mapy   float32 (H, W) — source pixel for each output pixel (cv2.remap)
  K            float64 (3, 3) — final camera matrix (for undistorted images)
  K_init       float64 (3, 3) — bootstrap camera matrix
  image_size   [W, H]

Usage
-----
    uv run python python/tools/build_rectification_map.py \\
        --video   /path/to/calib.mp4 \\
        --rows    17 --cols 24 \\
        --dict    DICT_4X4_100 \\
        --square  0.025 \\
        --out     insta_mega_rectmap.npz

    # Then remap any video from the same camera mode:
    uv run python python/tools/remap_video.py \\
        --map    insta_mega_rectmap.npz \\
        --input  mocap.mp4 \\
        --output mocap_rect.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# ChArUco helpers
# ---------------------------------------------------------------------------

_ARUCO_DICT_MAP = {
    "DICT_4X4_50":  cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50":  cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50":  cv2.aruco.DICT_6X6_50,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
}


def make_board(rows: int, cols: int, square: float,
               marker_ratio: float, dict_name: str):
    aruco_dict = cv2.aruco.getPredefinedDictionary(_ARUCO_DICT_MAP[dict_name])
    marker_len = square * marker_ratio
    try:
        return cv2.aruco.CharucoBoard((cols, rows), square, marker_len, aruco_dict)
    except TypeError:
        return cv2.aruco.CharucoBoard_create(cols, rows, square, marker_len, aruco_dict)


def detect_corners(frame: np.ndarray, board, detector,
                   min_corners: int = 8) -> tuple[np.ndarray | None, np.ndarray | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _, _ = detector.detectBoard(gray)
    if corners is None or ids is None or len(ids) < min_corners:
        return None, None
    return corners, ids


def match_points(corners, ids, board):
    """Return (obj_pts N×3, img_pts N×2) or (None, None)."""
    try:
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
    except AttributeError:
        flat = ids.flatten()
        obj_pts = board.getChessboardCorners()[flat]
        img_pts = corners.reshape(-1, 2)
    if obj_pts is None or len(obj_pts) < 6:
        return None, None
    return obj_pts.reshape(-1, 3).astype(np.float64), \
           img_pts.reshape(-1, 2).astype(np.float64)


# ---------------------------------------------------------------------------
# Video scanning
# ---------------------------------------------------------------------------

def scan_video(video_path: Path, board, detector,
               skip: int = 4, sharpness_thresh: float = 50.0,
               min_corners: int = 8, log=print):
    """
    Scan video, returning a list of (frame_idx, corners, ids) for every frame
    where corners are detected.  Uses a local-maxima sharpness filter to avoid
    redundant near-identical frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    n_total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log(f"Video: {W}×{H} @ {fps:.1f} fps  ({n_total} frames, ~{n_total/fps:.0f} s)")

    # Pass 1: compute sharpness for every skip-th frame
    lap_vals = []
    lap_idxs = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_vals.append(cv2.Laplacian(gray, cv2.CV_64F).var())
            lap_idxs.append(fi)
            if len(lap_idxs) % 500 == 0:
                log(f"  sharpness scan: {fi}/{n_total} frames")
        fi += 1
    cap.release()

    lap_arr  = np.array(lap_vals)
    log(f"Sharpness: mean={lap_arr.mean():.1f}  max={lap_arr.max():.1f}")

    # Local maxima within window=8 analysed frames, above threshold.
    # If sharpness_thresh <= 0: accept all analysed frames (no local-maxima filter).
    window = 8
    if sharpness_thresh <= 0:
        sharp_idxs = lap_idxs[:]
        log(f"Sharp frames selected: {len(sharp_idxs)} (all analysed, no sharpness filter)")
    else:
        sharp_idxs = []
        for i in range(window, len(lap_arr) - window):
            sl = lap_arr[i - window: i + window + 1]
            if lap_arr[i] == sl.max() and np.sum(sl == lap_arr[i]) == 1 \
                    and lap_arr[i] > sharpness_thresh:
                sharp_idxs.append(lap_idxs[i])
        log(f"Sharp frames selected: {len(sharp_idxs)}")

    # Pass 2: detect ChArUco corners — scan sequentially (no random seeking).
    # Random seeking into large compressed 4K files is very slow and forces
    # decoder resets; a full sequential pass is consistently faster.
    sharp_set = set(sharp_idxs)
    cap = cv2.VideoCapture(str(video_path))
    detections: list[tuple[int, np.ndarray, np.ndarray]] = []
    fi = 0
    n_checked = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in sharp_set:
            corners, ids = detect_corners(frame, board, detector, min_corners)
            if corners is not None:
                detections.append((fi, corners, ids))
            n_checked += 1
            if n_checked % 20 == 0:
                log(f"  detection: checked {n_checked}/{len(sharp_idxs)} candidates, "
                    f"{len(detections)} found so far")
        fi += 1
    cap.release()

    log(f"Frames with ≥{min_corners} ChArUco corners: {len(detections)}")
    return detections, (W, H)


# ---------------------------------------------------------------------------
# Bootstrap K from near-central board positions
# ---------------------------------------------------------------------------

def bootstrap_K(detections, board, image_size, log=print,
                centre_fraction: float = 0.35) -> np.ndarray:
    """
    Estimate K from frames where the board centre is close to the image centre.
    Near-centre, any wide-angle lens is approximately pinhole so
    calibrateCamera (no distortion) gives a usable focal length.
    """
    W, H = image_size
    cx0, cy0 = W / 2.0, H / 2.0
    max_dx, max_dy = W * centre_fraction, H * centre_fraction

    obj_pts_list, img_pts_list = [], []
    for item in detections:
        corners, ids = item[1], item[2]
        board_centre = corners.reshape(-1, 2).mean(axis=0)
        if abs(board_centre[0] - cx0) > max_dx or abs(board_centre[1] - cy0) > max_dy:
            continue
        obj_pts, img_pts = match_points(corners, ids, board)
        if obj_pts is None:
            continue
        obj_pts_list.append(obj_pts)
        img_pts_list.append(img_pts)

    if len(obj_pts_list) < 4:
        log("Warning: few central-board frames for bootstrap K; using all frames.")
        for item in detections:
            corners, ids = item[1], item[2]
            obj_pts, img_pts = match_points(corners, ids, board)
            if obj_pts is not None:
                obj_pts_list.append(obj_pts)
                img_pts_list.append(img_pts)

    if not obj_pts_list:
        raise RuntimeError("No frames suitable for K bootstrap.")

    ret, K, _dist, _rv, _tv = cv2.calibrateCamera(
        obj_pts_list, img_pts_list, image_size, None, None,
        flags=cv2.CALIB_FIX_ASPECT_RATIO,
    )
    log(f"Bootstrap K from {len(obj_pts_list)} central frames  (reproj error {ret:.2f} px)")
    log(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    return K


# ---------------------------------------------------------------------------
# Build (ideal → distorted) correspondences
# ---------------------------------------------------------------------------

def build_correspondences(
    detections, board, K, image_size, log=print
) -> tuple[np.ndarray, np.ndarray]:
    """
    For every detected frame: solve PnP (zero distortion) → project corners
    through K alone → "ideal" pixel.  Return:
      ideal      (N, 2)  — where K+pose says corner should be (ideal pinhole)
      distorted  (N, 2)  — where corner was observed (includes lens warp)
    """
    W, H = image_size
    ideal_list, dist_list = [], []

    n_skipped = 0
    for item in detections:
        corners, ids = item[1], item[2]
        obj_pts, img_pts = match_points(corners, ids, board)
        if obj_pts is None:
            continue

        ret, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ret:
            n_skipped += 1
            continue

        # Project with ideal K, no distortion
        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
        ideal = projected.reshape(-1, 2)

        # Filter: ideal and observed must both be inside the image
        in_bounds = (
            (ideal[:, 0] >= 0) & (ideal[:, 0] < W) &
            (ideal[:, 1] >= 0) & (ideal[:, 1] < H) &
            (img_pts[:, 0] >= 0) & (img_pts[:, 0] < W) &
            (img_pts[:, 1] >= 0) & (img_pts[:, 1] < H)
        )
        if in_bounds.sum() < 4:
            n_skipped += 1
            continue

        ideal_list.append(ideal[in_bounds])
        dist_list.append(img_pts[in_bounds])

    if not ideal_list:
        raise RuntimeError("No valid frames for correspondence building.")

    ideal = np.vstack(ideal_list)
    distorted = np.vstack(dist_list)
    if n_skipped:
        log(f"  PnP: {n_skipped} frames skipped (solver failed or out-of-bounds)")
    log(f"  Total corner pairs: {len(ideal)}")
    return ideal, distorted


# ---------------------------------------------------------------------------
# Spatial binning (averages in ideal space for RBF stability)
# ---------------------------------------------------------------------------

def spatially_bin(
    ideal: np.ndarray, distorted: np.ndarray,
    image_size: tuple[int, int], n_bins: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin (ideal, distorted) pairs into a grid of n_bins × n_bins cells in ideal
    space.  Return (binned_ideal, binned_distorted, counts) — one median point
    per occupied cell.  Counts lets callers flag cells with little data.
    """
    W, H = image_size
    bw, bh = W / n_bins, H / n_bins
    ix = np.clip((ideal[:, 0] / bw).astype(int), 0, n_bins - 1)
    iy = np.clip((ideal[:, 1] / bh).astype(int), 0, n_bins - 1)
    cell = iy * n_bins + ix

    bins_i: dict[int, list] = {}
    bins_d: dict[int, list] = {}
    for k in range(len(ideal)):
        c = int(cell[k])
        bins_i.setdefault(c, []).append(ideal[k])
        bins_d.setdefault(c, []).append(distorted[k])

    rep_i, rep_d, counts = [], [], []
    for c in sorted(bins_i):
        rep_i.append(np.median(bins_i[c], axis=0))
        rep_d.append(np.median(bins_d[c], axis=0))
        counts.append(len(bins_i[c]))

    return np.array(rep_i), np.array(rep_d), np.array(counts)


# ---------------------------------------------------------------------------
# RBF warp map fitting
# ---------------------------------------------------------------------------

def fit_warp_map(
    ideal_pts: np.ndarray,
    distorted_pts: np.ndarray,
    image_size: tuple[int, int],
    smoothing: float = 0.5,
    eval_scale: int = 8,
    log=print,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit TPS RBF:  ideal_pixel → distorted_pixel  (= inverse warp for cv2.remap).
    Evaluates on a coarse grid (1/eval_scale of full resolution) then upsamples.

    Returns (mapx, mapy) as float32 arrays of shape (H, W).
    """
    from scipy.interpolate import RBFInterpolator

    W, H = image_size
    N = len(ideal_pts)
    log(f"  Fitting TPS RBF on {N} binned points (smoothing={smoothing})...")
    t0 = time.perf_counter()

    rbf_x = RBFInterpolator(ideal_pts, distorted_pts[:, 0],
                             kernel="thin_plate_spline", smoothing=smoothing)
    rbf_y = RBFInterpolator(ideal_pts, distorted_pts[:, 1],
                             kernel="thin_plate_spline", smoothing=smoothing)
    log(f"  RBF fit: {time.perf_counter()-t0:.1f}s")

    # Evaluate on coarse grid
    gW, gH = W // eval_scale, H // eval_scale
    gx = np.linspace(0, W - 1, gW, dtype=np.float32)
    gy = np.linspace(0, H - 1, gH, dtype=np.float32)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=1, dtype=np.float64)

    t0 = time.perf_counter()
    mapx_c = rbf_x(grid_pts).reshape(gH, gW).astype(np.float32)
    mapy_c = rbf_y(grid_pts).reshape(gH, gW).astype(np.float32)
    log(f"  RBF eval on {gW}×{gH} grid: {time.perf_counter()-t0:.1f}s")

    # Upsample to full resolution (distortion is smooth, so INTER_CUBIC is fine)
    mapx = cv2.resize(mapx_c, (W, H), interpolation=cv2.INTER_CUBIC)
    mapy = cv2.resize(mapy_c, (W, H), interpolation=cv2.INTER_CUBIC)
    return mapx, mapy


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------

def coverage_report(
    ideal_pts: np.ndarray, image_size: tuple[int, int],
    n_bins: int = 24, radius_px: int = 200, log=print,
) -> float:
    """Report image regions with no correspondences within radius_px pixels."""
    W, H = image_size
    bw, bh = W / n_bins, H / n_bins
    covered = 0
    empty_cells = []
    for gy in range(n_bins):
        for gx in range(n_bins):
            cx = (gx + 0.5) * bw
            cy = (gy + 0.5) * bh
            dists = np.hypot(ideal_pts[:, 0] - cx, ideal_pts[:, 1] - cy)
            if dists.min() <= radius_px:
                covered += 1
            else:
                empty_cells.append((gx, gy))

    total = n_bins * n_bins
    pct = 100.0 * covered / total
    log(f"Coverage: {covered}/{total} cells have data within {radius_px}px  ({pct:.1f}%)")
    if empty_cells:
        # Print a simple ASCII coverage map
        grid = [["·"] * n_bins for _ in range(n_bins)]
        for gx, gy in empty_cells:
            grid[gy][gx] = "░"
        for bx, by in [
            (int(p[0] * n_bins / W), int(p[1] * n_bins / H)) for p in ideal_pts
        ]:
            if 0 <= by < n_bins and 0 <= bx < n_bins:
                grid[by][bx] = "█"
        log("Coverage map (█=data  ·=covered by interp  ░=extrapolation needed):")
        for row in grid:
            log("  " + "".join(row))
    return pct


# ---------------------------------------------------------------------------
# K refinement pass: apply current map, re-detect, re-calibrate K
# ---------------------------------------------------------------------------

def refit_K(
    video_path: Path, detections, board, detector, mapx, mapy, image_size,
    min_corners: int = 8, log=print,
) -> np.ndarray:
    """
    Apply the current warp map to each detected frame, re-detect ChArUco corners,
    calibrate K with a standard (no-distortion) model on the undistorted images.
    Scans the video sequentially to avoid storing frames in RAM.
    """
    det_set = {item[0] for item in detections}
    obj_pts_list, img_pts_list = [], []
    n_ok = 0

    cap = cv2.VideoCapture(str(video_path))
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in det_set:
            undist = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            corners, ids = detect_corners(undist, board, detector, min_corners)
            if corners is not None:
                obj_pts, img_pts = match_points(corners, ids, board)
                if obj_pts is not None:
                    obj_pts_list.append(obj_pts)
                    img_pts_list.append(img_pts)
                    n_ok += 1
        fi += 1
    cap.release()

    if not obj_pts_list:
        raise RuntimeError("No detections in undistorted frames for K refinement.")

    ret, K, _dist, _rv, _tv = cv2.calibrateCamera(
        obj_pts_list, img_pts_list, image_size, None, None,
        flags=cv2.CALIB_FIX_ASPECT_RATIO,
    )
    log(f"  Re-fit K from {n_ok} undistorted frames  (reproj error {ret:.2f} px)")
    log(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    return K


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_map(args) -> None:
    def log(*a, **kw):
        print(*a, **kw, flush=True)
    video  = Path(args.video)
    board  = make_board(args.rows, args.cols, args.square, args.marker_ratio, args.dict)
    detector = cv2.aruco.CharucoDetector(board)

    # ── 1. Scan video ──────────────────────────────────────────────────────
    log("\n=== Step 1: Scanning video for ChArUco detections ===")
    raw_detections, image_size = scan_video(
        video, board, detector,
        skip=args.skip,
        sharpness_thresh=args.sharpness_threshold,
        min_corners=args.min_corners,
        log=log,
    )
    if len(raw_detections) < 6:
        raise RuntimeError(
            f"Only {len(raw_detections)} frames detected — need ≥6. "
            "Check --dict matches the printed board, or lower --min-corners."
        )
    W, H = image_size
    detections = raw_detections  # list of (fi, corners, ids) — no frames stored

    # ── 2. Bootstrap K ─────────────────────────────────────────────────────
    log("\n=== Step 2: Bootstrap K from central-board frames ===")
    K = bootstrap_K(detections, board, image_size, log=log)
    K_init = K.copy()

    # ── 3. Iterate: correspondences → warp → re-fit K ──────────────────────
    mapx = mapy = None
    for it in range(1, args.iterations + 1):
        log(f"\n=== Iteration {it}/{args.iterations} ===")

        log("Building (ideal → distorted) correspondence pairs...")
        ideal, distorted = build_correspondences(detections, board, K, image_size, log=log)

        log(f"Spatial binning into {args.grid_bins}×{args.grid_bins} cells...")
        bin_ideal, bin_dist, counts = spatially_bin(
            ideal, distorted, image_size, n_bins=args.grid_bins
        )
        log(f"  {len(bin_ideal)} occupied bins  "
            f"(median {int(np.median(counts))} pairs/bin, max {counts.max()})")

        log("Fitting warp map...")
        mapx, mapy = fit_warp_map(
            bin_ideal, bin_dist, image_size,
            smoothing=args.smoothing,
            eval_scale=args.eval_scale,
            log=log,
        )

        if it < args.iterations:
            log("Re-fitting K on undistorted frames (sequential video scan)...")
            K_new = refit_K(
                video, detections, board, detector, mapx, mapy, image_size,
                min_corners=args.min_corners, log=log,
            )
            delta = np.abs(K_new - K) / (np.abs(K) + 1e-6)
            log(f"  K change: max {delta.max()*100:.2f}%")
            K = K_new
            if delta.max() < 0.005:
                log("  K converged (< 0.5%), stopping early.")
                break

    # ── 4. Coverage report ─────────────────────────────────────────────────
    log("\n=== Coverage analysis ===")
    coverage_pct = coverage_report(bin_ideal, image_size, n_bins=24,
                                   radius_px=args.coverage_radius, log=log)

    # ── 5. Save ────────────────────────────────────────────────────────────
    out = Path(args.out)
    np.savez_compressed(
        out,
        mapx=mapx,
        mapy=mapy,
        K=K,
        K_init=K_init,
        image_size=np.array([W, H]),
        n_correspondences=len(ideal),
        coverage_pct=float(coverage_pct),
    )
    log(f"\nSaved rectification map → {out}")
    log(f"  mapx/mapy: {W}×{H}  K: fx={K[0,0]:.1f} fy={K[1,1]:.1f}")
    log(f"  Coverage: {coverage_pct:.1f}% of image area has nearby data")
    if coverage_pct < 80:
        log("  ⚠  Low coverage — peripheral regions will be extrapolated. "
            "Move the board to the image corners during calibration.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build a non-parametric rectification map from a ChArUco calibration video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video",   required=True, help="Calibration video path")
    p.add_argument("--rows",    type=int,   default=17,           help="Board rows (squares)")
    p.add_argument("--cols",    type=int,   default=24,           help="Board columns (squares)")
    p.add_argument("--dict",    default="DICT_4X4_100",
                   choices=list(_ARUCO_DICT_MAP), help="ArUco dictionary")
    p.add_argument("--square",  type=float, default=0.025,        help="Square size in metres")
    p.add_argument("--marker-ratio", type=float, default=0.75,
                   help="ArUco marker side as fraction of square size")
    p.add_argument("--out",     required=True, help="Output .npz path")
    p.add_argument("--skip",    type=int,   default=4,
                   help="Analyse every Nth frame for sharpness (default 4)")
    p.add_argument("--sharpness-threshold", type=float, default=50.0,
                   help="Laplacian variance floor for sharp-frame selection")
    p.add_argument("--min-corners", type=int, default=8,
                   help="Minimum detected corners to accept a frame")
    p.add_argument("--iterations", type=int, default=2,
                   help="Number of K-refinement iterations (default 2)")
    p.add_argument("--grid-bins",  type=int, default=48,
                   help="Spatial bins per axis for averaging correspondences (default 48)")
    p.add_argument("--smoothing",  type=float, default=0.5,
                   help="TPS smoothing regularisation (smaller = tighter fit, default 0.5)")
    p.add_argument("--eval-scale", type=int, default=8,
                   help="Evaluate RBF at 1/N resolution then upsample (default 8)")
    p.add_argument("--coverage-radius", type=int, default=200,
                   help="Pixel radius for coverage check (default 200)")
    return p.parse_args()


if __name__ == "__main__":
    build_map(parse_args())
