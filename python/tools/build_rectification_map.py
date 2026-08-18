#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

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


def make_detector(board, min_marker_perim_rate: float = 0.01):
    """
    Create a CharucoDetector with relaxed parameters for detecting small or
    distant markers in wide-angle footage.

    Default OpenCV minMarkerPerimeterRate=0.03 means a marker must be ≥115 px
    perimeter in a 3840 px frame — too large when the board is far from the
    camera.  0.01 allows markers down to ~38 px perimeter.
    """
    det_params = cv2.aruco.DetectorParameters()
    det_params.minMarkerPerimeterRate  = min_marker_perim_rate
    det_params.maxMarkerPerimeterRate  = 4.0
    det_params.adaptiveThreshWinSizeMin  = 3
    det_params.adaptiveThreshWinSizeMax  = 53   # larger range catches blurry markers
    det_params.adaptiveThreshWinSizeStep = 10
    det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    charuco_params = cv2.aruco.CharucoParameters()
    return cv2.aruco.CharucoDetector(board, charuco_params, det_params)


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
    return obj_pts.reshape(-1, 3).astype(np.float32), \
           img_pts.reshape(-1, 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Video scanning
# ---------------------------------------------------------------------------

def scan_video(video_path: Path, board, detector,
               skip: int = 4, sharpness_thresh: float = 50.0,
               min_corners: int = 8, dump_dir: Path | None = None,
               dump_scale: float = 0.5, log=print):
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
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        log(f"Debug frames will be saved to: {dump_dir}  (scale={dump_scale})")

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

            if dump_dir is not None:
                _dump_frame(frame, fi, corners, ids, board, dump_dir, dump_scale, min_corners)
        fi += 1
    cap.release()

    log(f"Frames with ≥{min_corners} ChArUco corners: {len(detections)}")
    return detections, (W, H)


def _dump_frame(frame, fi, corners, ids, board, dump_dir, scale, min_corners):
    """Save a debug image: sharp-frame candidate with detection result overlaid."""
    vis = frame.copy()

    if corners is not None and ids is not None:
        # Draw detected ChArUco corners
        cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids, (0, 255, 0))
        n = len(ids)
        label     = f"f{fi}  OK  {n} corners"
        bg_colour = (0, 128, 0)
    else:
        # Show why it failed: try to detect ArUco markers even without enough corners
        aruco_dict = board.getDictionary() if hasattr(board, "getDictionary") else None
        if aruco_dict is not None:
            gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
            det  = cv2.aruco.ArucoDetector(aruco_dict)
            m_corners, m_ids, _ = det.detectMarkers(gray)
            if m_ids is not None and len(m_ids):
                cv2.aruco.drawDetectedMarkers(vis, m_corners, m_ids, (0, 165, 255))
                label = f"f{fi}  ArUco: {len(m_ids)} markers, but <{min_corners} ChArUco corners"
            else:
                label = f"f{fi}  NO DETECTION (no ArUco markers found)"
        else:
            label = f"f{fi}  NO DETECTION"
        bg_colour = (0, 0, 180)

    # Resize before text so the text renders at a legible size
    if scale != 1.0:
        h, w = vis.shape[:2]
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Label banner at top
    font, fscale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2
    (tw, th), baseline = cv2.getTextSize(label, font, fscale, thick)
    banner_h = th + baseline + 12
    banner = np.full((banner_h, vis.shape[1], 3), bg_colour, dtype=np.uint8)
    cv2.putText(banner, label, (8, th + 6), font, fscale, (255, 255, 255), thick, cv2.LINE_AA)
    vis = np.vstack([banner, vis])

    status = "ok" if corners is not None else "fail"
    path   = dump_dir / f"frame_{fi:05d}_{status}.jpg"
    cv2.imwrite(str(path), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])


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
) -> tuple[np.ndarray, np.ndarray, list, list]:
    """
    For every detected frame: solve PnP (zero distortion) → project corners
    through K alone → "ideal" pixel.  Return:
      ideal            (N, 2)  — where K+pose says corner should be (ideal pinhole)
      distorted        (N, 2)  — where corner was observed (includes lens warp)
      obj_pts_per_frame        — list of (n, 3) arrays, one per frame (for K refit)
      ideal_per_frame          — list of (n, 2) arrays, one per frame (for K refit)

    The ideal positions are already the corners as they appear in an undistorted
    image, so K can be re-estimated from (obj_pts, ideal_pts) directly without
    ever re-reading video frames.
    """
    W, H = image_size
    ideal_list, dist_list = [], []
    obj_pts_per_frame: list[np.ndarray] = []
    ideal_per_frame:   list[np.ndarray] = []

    n_skipped = 0
    for item in detections:
        corners, ids = item[1], item[2]
        obj_pts, img_pts = match_points(corners, ids, board)
        if obj_pts is None:
            continue

        obj64 = obj_pts.astype(np.float64)
        img64 = img_pts.astype(np.float64)
        ret, rvec, tvec = cv2.solvePnP(
            obj64, img64, K, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ret:
            n_skipped += 1
            continue

        # Project with ideal K, no distortion
        projected, _ = cv2.projectPoints(obj64, rvec, tvec, K, None)
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
        obj_pts_per_frame.append(obj_pts[in_bounds])
        ideal_per_frame.append(ideal[in_bounds])

    if not ideal_list:
        raise RuntimeError("No valid frames for correspondence building.")

    ideal = np.vstack(ideal_list)
    distorted = np.vstack(dist_list)
    if n_skipped:
        log(f"  PnP: {n_skipped} frames skipped (solver failed or out-of-bounds)")
    log(f"  Total corner pairs: {len(ideal)}  ({len(obj_pts_per_frame)} frames)")
    return ideal, distorted, obj_pts_per_frame, ideal_per_frame


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
# Warp map fitting
# ---------------------------------------------------------------------------

def _poly_features(pts: np.ndarray, degree: int, W: float, H: float) -> np.ndarray:
    """
    Build polynomial feature matrix for 2D points, normalised to [-1, 1].
    Returns (N, n_features) array including the bias column.
    """
    x = pts[:, 0] / W * 2 - 1   # normalise to [-1, 1]
    y = pts[:, 1] / H * 2 - 1
    cols = [np.ones(len(pts))]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            cols.append((x ** (d - i)) * (y ** i))
    return np.column_stack(cols)


def _n_poly_features(degree: int) -> int:
    return (degree + 1) * (degree + 2) // 2


# ---------------------------------------------------------------------------
# Collinearity-based warp fitting
# ---------------------------------------------------------------------------

def extract_collinearity_groups(
    detections, board_cols: int, board_rows: int, min_pts: int = 4,
) -> list[np.ndarray]:
    """
    For each detected frame, group corners by board row and column index.

    ChArUco corner at board grid position (r, c) has ID = r*(board_cols-1) + c.
    Points in the same board row (or column) must be collinear in the undistorted
    image — they lie on a straight 3D line projected through a pinhole.

    Returns list of (N≥min_pts, 2) float64 arrays of distorted pixel positions.
    """
    n_cc = board_cols - 1  # chess-corner columns per row

    groups: list[np.ndarray] = []
    for _fi, corners, ids in detections:
        pts     = corners.reshape(-1, 2).astype(np.float64)
        ids_flat = ids.flatten()

        row_pts: dict[int, list] = {}
        col_pts: dict[int, list] = {}
        for pt, cid in zip(pts, ids_flat):
            r = int(cid) // n_cc
            c = int(cid) % n_cc
            row_pts.setdefault(r, []).append(pt)
            col_pts.setdefault(c, []).append(pt)

        for bucket in (row_pts, col_pts):
            for pts_list in bucket.values():
                if len(pts_list) >= min_pts:
                    groups.append(np.array(pts_list, dtype=np.float64))
    return groups


def fit_collinearity_poly(
    groups: list[np.ndarray],
    image_size: tuple[int, int],
    degree: int = 5,
    n_iters: int = 20,
    lambda_reg: float = 10.0,
    log=print,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iterative linearised least-squares: find polynomial forward warp coefficients
    (coef_x, coef_y) such that after applying the correction
        corrected = observed + [poly_x(observed), poly_y(observed)]
    every group of points is collinear.

    No PnP, no K, no distortion model assumed.  The only constraint is that
    board rows/columns are straight lines — a consequence of the flat board and
    the pinhole projection model.

    Regularisation (Tikhonov) + fixing the constant term = 0 prevents global
    translation drift (the null space of the collinearity constraint).
    """
    W, H = image_size
    n_coef = _n_poly_features(degree)
    coef_x = np.zeros(n_coef)
    coef_y = np.zeros(n_coef)

    for iteration in range(n_iters):
        A_parts, b_parts = [], []

        for group in groups:
            phi     = _poly_features(group, degree, W, H)    # (N, n_coef)
            corrected = group + np.column_stack([phi @ coef_x, phi @ coef_y])

            # Best-fit line direction via SVD
            centroid  = corrected.mean(axis=0)
            centered  = corrected - centroid
            try:
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            normal = np.array([-Vt[0, 1], Vt[0, 0]])  # perpendicular to line direction

            # Current collinearity residuals (signed distance from the line)
            resid = centered @ normal   # (N,)

            # Centred features: subtracting the group mean makes the centroid
            # translation drop out of the gradient, so we don't fight the centroid
            # drifting while fitting the shape.
            phi_c    = phi - phi.mean(axis=0)              # (N, n_coef)
            nx, ny   = normal
            A_g      = np.hstack([nx * phi_c, ny * phi_c]) # (N, 2*n_coef)
            b_g      = -resid                               # want residuals → 0

            A_parts.append(A_g)
            b_parts.append(b_g)

        if not A_parts:
            break

        A   = np.vstack(A_parts)
        b   = np.concatenate(b_parts)
        ATA = A.T @ A + lambda_reg * np.eye(2 * n_coef)
        delta = np.linalg.solve(ATA, A.T @ b)

        coef_x += delta[:n_coef]
        coef_y += delta[n_coef:]
        # Zero constant term: no global translation (anchors image centre)
        coef_x[0] = 0.0
        coef_y[0] = 0.0

        rms = np.sqrt(np.mean(b ** 2))
        log(f"  Iter {iteration+1:2d}: RMS collinearity residual = {rms:.3f} px  "
            f"|Δ| = {np.linalg.norm(delta):.5f}")
        if np.linalg.norm(delta) < 1e-4:
            log("  Converged.")
            break

    return coef_x, coef_y


def build_inverse_map_from_forward_poly(
    coef_x: np.ndarray, coef_y: np.ndarray,
    image_size: tuple[int, int],
    degree: int,
    subsample: int = 4,
    log=print,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a forward polynomial warp (distorted→ideal) to the inverse map
    (ideal→distorted) required by cv2.remap.

    Strategy: evaluate the forward polynomial on a subsampled grid of distorted
    coordinates to get scattered (ideal, distorted) pairs, then fit a second
    polynomial inverse from ideal→distorted and evaluate on the full grid.
    """
    W, H = image_size

    # --- Forward warp on subsampled distorted grid ---
    gx = np.arange(0, W, subsample, dtype=np.float64)
    gy = np.arange(0, H, subsample, dtype=np.float64)
    GX, GY = np.meshgrid(gx, gy)
    dist_pts  = np.column_stack([GX.ravel(), GY.ravel()])
    phi       = _poly_features(dist_pts, degree, W, H)
    ideal_pts = dist_pts + np.column_stack([phi @ coef_x, phi @ coef_y])

    # Filter: keep points whose ideal position is close to the valid image area
    margin = max(W, H) * 0.15
    ok = (
        (ideal_pts[:, 0] >= -margin) & (ideal_pts[:, 0] < W + margin) &
        (ideal_pts[:, 1] >= -margin) & (ideal_pts[:, 1] < H + margin)
    )
    ideal_pts = ideal_pts[ok]
    dist_pts  = dist_pts[ok]
    log(f"  Forward warp: {ok.sum()}/{len(ok)} grid points within bounds")

    # --- Fit inverse polynomial: ideal → distorted ---
    log(f"  Fitting inverse degree-{degree} polynomial on {len(ideal_pts)} points...")
    A     = _poly_features(ideal_pts, degree, W, H)
    reg   = 1e-3 * np.eye(A.shape[1])
    ATA   = A.T @ A + reg
    ci_x  = np.linalg.solve(ATA, A.T @ dist_pts[:, 0])
    ci_y  = np.linalg.solve(ATA, A.T @ dist_pts[:, 1])
    rx    = np.std(A @ ci_x - dist_pts[:, 0])
    ry    = np.std(A @ ci_y - dist_pts[:, 1])
    log(f"  Inverse poly residual std: x={rx:.3f} px  y={ry:.3f} px")

    # --- Evaluate inverse polynomial on every pixel row ---
    log("  Building full-resolution map...")
    t0   = time.perf_counter()
    gx_r = np.arange(W, dtype=np.float64)
    mapx = np.empty((H, W), dtype=np.float32)
    mapy = np.empty((H, W), dtype=np.float32)
    for row in range(H):
        row_pts      = np.column_stack([gx_r, np.full(W, row, np.float64)])
        A_r          = _poly_features(row_pts, degree, W, H)
        mapx[row]    = (A_r @ ci_x).astype(np.float32)
        mapy[row]    = (A_r @ ci_y).astype(np.float32)
    log(f"  Map eval: {time.perf_counter() - t0:.1f}s")
    return mapx, mapy


def fit_warp_map(
    ideal_pts: np.ndarray,
    distorted_pts: np.ndarray,
    image_size: tuple[int, int],
    method: str = "poly",
    poly_degree: int = 5,
    smoothing: float = 0.5,
    eval_scale: int = 8,
    log=print,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit inverse warp:  ideal_pixel → distorted_pixel  (for cv2.remap).

    method="poly"  — polynomial least-squares (default, recommended).
        Degree-5 polynomial has 21 parameters per axis — globally smooth,
        immune to local oscillations, fits well when all 7000+ raw pairs
        are used directly (no binning needed).  Ridge regularisation prevents
        extrapolation blow-up.

    method="tps"   — thin-plate spline RBF on spatially-binned points.
        Flexible but can overfit to noisy/sparse data, producing local
        oscillations. Use only if the polynomial residual map shows systematic
        local structure that the polynomial can't capture.

    Returns (mapx, mapy) float32 (H, W).
    """
    W, H = image_size

    if method == "poly":
        return _fit_poly(ideal_pts, distorted_pts, image_size, poly_degree, log)
    else:
        return _fit_tps(ideal_pts, distorted_pts, image_size, smoothing, eval_scale, log)


def _fit_poly(
    ideal_pts: np.ndarray,
    distorted_pts: np.ndarray,
    image_size: tuple[int, int],
    degree: int,
    log=print,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Polynomial warp: use ALL raw correspondence pairs (no binning) with Ridge
    regression.  The global polynomial is inherently smooth and can't produce
    local oscillations, so noisy individual pairs average out correctly.
    """
    W, H = image_size
    N = len(ideal_pts)
    log(f"  Fitting degree-{degree} polynomial on {N} pairs...")
    t0 = time.perf_counter()

    A = _poly_features(ideal_pts, degree, W, H)
    # Ridge regression (alpha=1e-3 in normalised space) for numerical stability
    ATA = A.T @ A
    n_feat = A.shape[1]
    ATA[np.arange(n_feat), np.arange(n_feat)] += 1e-3
    coef_x = np.linalg.solve(ATA, A.T @ distorted_pts[:, 0])
    coef_y = np.linalg.solve(ATA, A.T @ distorted_pts[:, 1])

    resid_x = np.std(A @ coef_x - distorted_pts[:, 0])
    resid_y = np.std(A @ coef_y - distorted_pts[:, 1])
    log(f"  Poly fit: {time.perf_counter()-t0:.1f}s  "
        f"residual std: x={resid_x:.2f}px  y={resid_y:.2f}px")

    # Evaluate on full grid
    log("  Evaluating polynomial on full grid...")
    t0 = time.perf_counter()
    gy_all = np.arange(H, dtype=np.float64)
    mapx = np.empty((H, W), dtype=np.float32)
    mapy = np.empty((H, W), dtype=np.float32)
    gx_row = np.arange(W, dtype=np.float64)
    for row in range(H):
        pts_row = np.column_stack([gx_row, np.full(W, row, dtype=np.float64)])
        A_row = _poly_features(pts_row, degree, W, H)
        mapx[row] = (A_row @ coef_x).astype(np.float32)
        mapy[row] = (A_row @ coef_y).astype(np.float32)
    log(f"  Grid eval: {time.perf_counter()-t0:.1f}s")
    return mapx, mapy


def _fit_tps(
    ideal_pts: np.ndarray,
    distorted_pts: np.ndarray,
    image_size: tuple[int, int],
    smoothing: float,
    eval_scale: int,
    log=print,
) -> tuple[np.ndarray, np.ndarray]:
    """TPS RBF fallback — expects spatially-binned (not raw) points."""
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
    gW, gH = W // eval_scale, H // eval_scale
    gx = np.linspace(0, W - 1, gW, dtype=np.float32)
    gy = np.linspace(0, H - 1, gH, dtype=np.float32)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=1).astype(np.float64)
    t0 = time.perf_counter()
    mapx_c = rbf_x(grid_pts).reshape(gH, gW).astype(np.float32)
    mapy_c = rbf_y(grid_pts).reshape(gH, gW).astype(np.float32)
    log(f"  RBF eval on {gW}×{gH} grid: {time.perf_counter()-t0:.1f}s")
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
    obj_pts_per_frame: list, ideal_per_frame: list,
    image_size: tuple[int, int], log=print,
) -> np.ndarray:
    """
    Re-calibrate K from per-frame (obj_pts, ideal_pts) produced by
    build_correspondences.  No video or frames needed: the ideal_pts are
    already the corner positions in the undistorted image (they are the
    PnP-projected positions assuming zero distortion), so calibrateCamera
    on them directly gives an improved K.
    """
    # calibrateCamera requires float32 Point2f/Point3f
    obj32  = [p.astype(np.float32) for p in obj_pts_per_frame]
    img32  = [p.astype(np.float32) for p in ideal_per_frame]
    ret, K, _dist, _rv, _tv = cv2.calibrateCamera(
        obj32, img32, image_size, None, None,
        flags=cv2.CALIB_FIX_ASPECT_RATIO,
    )
    log(f"  Re-fit K from {len(obj_pts_per_frame)} frames  (reproj error {ret:.2f} px)")
    log(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    return K


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_map(args) -> None:
    def log(*a, **kw):
        print(*a, **kw, flush=True)
    video  = Path(args.video)
    board    = make_board(args.rows, args.cols, args.square, args.marker_ratio, args.dict)
    detector = make_detector(board, min_marker_perim_rate=args.min_marker_perim_rate)

    # ── 1. Scan video ──────────────────────────────────────────────────────
    log("\n=== Step 1: Scanning video for ChArUco detections ===")
    dump_dir = Path(args.dump_frames) if args.dump_frames else None
    raw_detections, image_size = scan_video(
        video, board, detector,
        skip=args.skip,
        sharpness_thresh=args.sharpness_threshold,
        min_corners=args.min_corners,
        dump_dir=dump_dir,
        dump_scale=args.dump_scale,
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

    # ── 3. Fit warp ────────────────────────────────────────────────────────
    if args.fit_method == "collinear":
        # ── Collinearity path: no PnP, no K required ──────────────────────
        log("\n=== Collinearity fitting (no PnP or K required) ===")
        groups = extract_collinearity_groups(
            detections, args.cols, args.rows, min_pts=4
        )
        log(f"  {len(groups)} row/column groups from {len(detections)} frames")
        if len(groups) < 10:
            raise RuntimeError(
                f"Only {len(groups)} collinearity groups — need ≥10. "
                "Check --dict and --rows/--cols match the printed board."
            )

        log("Fitting forward polynomial via collinearity constraint...")
        coef_x, coef_y = fit_collinearity_poly(
            groups, image_size,
            degree=args.poly_degree,
            n_iters=args.collinear_iters,
            lambda_reg=args.collinear_reg,
            log=log,
        )

        log("Building inverse remap map from forward polynomial...")
        mapx, mapy = build_inverse_map_from_forward_poly(
            coef_x, coef_y, image_size, args.poly_degree, subsample=4, log=log,
        )

        # Coverage: use all detected corner positions (distorted space)
        ideal = np.vstack([g for g in groups])

    else:
        # ── PnP correspondence path (poly or tps) ─────────────────────────
        mapx = mapy = None
        for it in range(1, args.iterations + 1):
            log(f"\n=== Iteration {it}/{args.iterations} ===")

            log("Building (ideal → distorted) correspondence pairs...")
            ideal, distorted, obj_per_frame, ideal_per_frame = build_correspondences(
                detections, board, K, image_size, log=log
            )

            log("Fitting warp map...")
            if args.fit_method == "poly":
                fit_ideal, fit_dist = ideal, distorted
            else:
                log(f"Spatial binning into {args.grid_bins}×{args.grid_bins} cells...")
                fit_ideal, fit_dist, counts = spatially_bin(
                    ideal, distorted, image_size, n_bins=args.grid_bins
                )
                log(f"  {len(fit_ideal)} occupied bins  "
                    f"(median {int(np.median(counts))} pairs/bin, max {counts.max()})")
            mapx, mapy = fit_warp_map(
                fit_ideal, fit_dist, image_size,
                method=args.fit_method,
                poly_degree=args.poly_degree,
                smoothing=args.smoothing,
                eval_scale=args.eval_scale,
                log=log,
            )

            if it < args.iterations:
                log("Re-fitting K from ideal corner positions...")
                K_new = refit_K(obj_per_frame, ideal_per_frame, image_size, log=log)
                delta = np.abs(K_new - K) / (np.abs(K) + 1e-6)
                log(f"  K change: max {delta.max()*100:.2f}%")
                K = K_new
                if delta.max() < 0.005:
                    log("  K converged (< 0.5%), stopping early.")
                    break

    # ── 4. Coverage report ─────────────────────────────────────────────────
    log("\n=== Coverage analysis ===")
    coverage_pct = coverage_report(ideal, image_size, n_bins=24,
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
    p.add_argument("--fit-method", default="poly", choices=["poly", "tps", "collinear"],
                   help="Warp fitting method. "
                        "poly (default): PnP-based correspondences + polynomial least-squares. "
                        "collinear: no PnP — fits the warp so that board rows/columns are "
                        "straight lines in the undistorted image. Works for large distortions "
                        "where PnP-with-zero-distortion fails. "
                        "tps: thin-plate spline on spatially-binned PnP correspondences.")
    p.add_argument("--collinear-iters", type=int, default=20,
                   help="Max iterations for collinearity optimizer (default 20)")
    p.add_argument("--collinear-reg", type=float, default=10.0,
                   help="Tikhonov regularisation for collinearity fit (default 10.0). "
                        "Increase if the map drifts wildly at the edges.")
    p.add_argument("--poly-degree", type=int, default=5,
                   help="Polynomial degree (default 5 = 21 parameters per axis). "
                        "Increase to 6-7 if residuals show systematic structure.")
    p.add_argument("--smoothing",  type=float, default=0.5,
                   help="TPS smoothing regularisation — only used with --fit-method tps")
    p.add_argument("--eval-scale", type=int, default=8,
                   help="TPS coarse-grid scale factor — only used with --fit-method tps")
    p.add_argument("--coverage-radius", type=int, default=200,
                   help="Pixel radius for coverage check (default 200)")
    p.add_argument("--min-marker-perim-rate", type=float, default=0.01,
                   help="Minimum ArUco marker perimeter as fraction of image width. "
                        "Default 0.01 (~38 px at 4K) to detect distant markers. "
                        "OpenCV default is 0.03 (~115 px) which misses small markers.")
    p.add_argument("--dump-frames", default=None, metavar="DIR",
                   help="Save all sharp-frame candidates as JPEG debug images in DIR. "
                        "Detected boards are drawn in green; failed frames show any "
                        "ArUco markers found in orange so you can diagnose dict mismatches.")
    p.add_argument("--dump-scale", type=float, default=0.5,
                   help="Scale factor for saved debug images (default 0.5 = half-res)")
    return p.parse_args()


if __name__ == "__main__":
    build_map(parse_args())
