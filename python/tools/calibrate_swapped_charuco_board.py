#!/usr/bin/env python3
"""calibrate_swapped_charuco_board.py — Intrinsics calibration for a specific
physical ChArUco board that has two of its four printed A4 quadrant pages
glued in the wrong place.

Background
----------
An 8x12-square, DICT_4X4 ChArUco board was assembled from 4 printed A4 pages
glued onto a backing. Debugging a "no ChArUco boards detected" failure
(2026-07-25) found that the top-right and bottom-left quadrant pages were
glued into each other's position, each rotated 180 degrees; top-left and
bottom-right are correctly placed. This was confirmed independently three
ways: a pure-geometry nearest-neighbour lattice reconstruction from marker
pixel positions (48/48 markers matched with zero error), a marker-orientation
check (swapped-quadrant markers measured ~178-180 degrees rotated relative to
correctly-placed ones), and a with/without-correction reprojection
comparison.

cv2.aruco.CharucoBoard's Python API accepts a custom per-position marker-id
array, but not custom per-marker corner ORDER -- so it cannot represent a
locally-rotated page and interpolateCornersCharuco()/CharucoDetector never
finds valid chessboard corners on this board no matter what rows/cols/
dictionary/legacy-pattern combination is tried (all exhaustively tested).
This script sidesteps that entirely: it detects raw ArUco markers (which
decode perfectly regardless of physical rotation) and calibrates directly
from each marker's own 4 corners, using object points corrected for the
swapped+rotated quadrants.

Known limitation (as of the 2026-07-25 session): even with this correction,
calibration on the original handheld capture converged to ~50px RMS
regardless of frame count, distortion model (standard/rational/fisheye), or
marker subset -- evidence pointed to the physical board not being flat
(uniform residual across every frame, unaffected by row/column rescaling).
Re-shoot with a flattened/reinforced board before trusting this script's
output; a good calibration should converge to well under 1px RMS.

Usage
-----
    python calibrate_swapped_charuco_board.py VIDEO --output-file calib.h5 \\
        --camera-name "OnePlus 10" --camera-mode "4K 120fps"

    # Faster iteration: sample every Nth frame directly instead of scanning
    # the whole video for local-maximum sharpness.
    python calibrate_swapped_charuco_board.py VIDEO --output-file calib.h5 \\
        --camera-name "OnePlus 10" --camera-mode "4K 120fps" --frame-stride 200
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.calibration.calibrate_intrinsics import (  # noqa: E402
    CalibrationResult,
    UndistortionMaps,
    find_sharp_frames,
    get_video_properties,
    save_calibration_h5,
)

# Board layout -- update these if a different physical board is used.
ROWS, COLS = 8, 12          # squares
MARKER_COLS = COLS // 2     # 6 marker-columns per row (markers on alternating squares)
DICT_NAME = "DICT_4X4_50"
DEFAULT_SQUARE_SIZE = 0.025  # metres -- measure the actual printed square and override if different
DEFAULT_MARKER_RATIO = 0.75

MIN_MARKERS_PER_FRAME = 10  # skip frames with too little board coverage


def _true_rc(r: int, c: int) -> tuple[int, int, bool]:
    """Map a marker's nominal (row, col) -- per a standard, un-scrambled
    CharucoBoard's row-major numbering -- to its TRUE physical (row, col) on
    this specific mis-assembled board, and whether it sits on a
    180-degree-rotated page.

    Top-left (rows 0-3, cols 0-2) and bottom-right (rows 4-7, cols 3-5) are
    correctly placed. Top-right and bottom-left were glued in each other's
    spot, each rotated 180 degrees -- which for a rectangular sub-grid is
    equivalent to (row, col) -> (ROWS-1-row, MARKER_COLS-1-col).
    """
    if (0 <= r <= 3 and 0 <= c <= 2) or (4 <= r <= 7 and 3 <= c <= 5):
        return r, c, False
    return ROWS - 1 - r, MARKER_COLS - 1 - c, True


def build_corrected_object_points(
    dict_name: str = DICT_NAME,
    square_size: float = DEFAULT_SQUARE_SIZE,
    marker_ratio: float = DEFAULT_MARKER_RATIO,
) -> tuple[dict[int, np.ndarray], "cv2.aruco.Dictionary"]:
    """Return ({marker_id: 4x3 float64 object points}, aruco_dictionary) for
    this board's TRUE physical layout, corner order corrected for the two
    rotated quadrants.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    std_board = cv2.aruco.CharucoBoard(
        (COLS, ROWS), square_size, square_size * marker_ratio, aruco_dict
    )
    std_ids = std_board.getIds().flatten()
    std_obj = std_board.getObjPoints()
    id_to_stdidx = {int(v): i for i, v in enumerate(std_ids)}

    corrected: dict[int, np.ndarray] = {}
    for marker_id in range(ROWS * MARKER_COLS):
        r, c = divmod(marker_id, MARKER_COLS)
        tr, tc, rotated = _true_rc(r, c)
        true_pos_id = tr * MARKER_COLS + tc
        obj = std_obj[id_to_stdidx[true_pos_id]].copy()
        if rotated:
            # Corner order returned by the ArUco decoder for a marker rotated
            # 180 degrees in the world is the original order rolled by 2
            # (opposite corner comes first): [TL,TR,BR,BL] -> [BR,BL,TL,TR].
            obj = obj[[2, 3, 0, 1], :]
        corrected[marker_id] = obj
    return corrected, aruco_dict


def collect_marker_correspondences(
    frames: list[np.ndarray],
    aruco_dict,
    corrected_obj: dict[int, np.ndarray],
    min_markers: int = MIN_MARKERS_PER_FRAME,
    log_fn=print,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Detect raw ArUco markers in each frame and build (object_pts, image_pts)
    arrays per frame using the corrected object points. Suitable directly for
    cv2.calibrateCamera (float32 in, one array per frame)."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    all_obj, all_img = [], []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        if marker_ids is None or len(marker_ids) < min_markers:
            continue
        obj_pts, img_pts = [], []
        for mc, mid in zip(marker_corners, marker_ids.flatten()):
            mid = int(mid)
            if mid not in corrected_obj:
                continue
            obj_pts.append(corrected_obj[mid])
            img_pts.append(mc.reshape(-1, 2))
        if len(obj_pts) < min_markers:
            continue
        all_obj.append(np.concatenate(obj_pts, axis=0).astype(np.float32))
        all_img.append(np.concatenate(img_pts, axis=0).astype(np.float32))

    log_fn(
        f"Collected marker-corner correspondences from {len(all_obj)} frames "
        f"(of {len(frames)} candidate frames)."
    )
    return all_obj, all_img


def _sample_frames_by_stride(video_path: Path, stride: int, log_fn=print) -> list[np.ndarray]:
    """Grab every `stride`-th frame directly, skipping the (slow) full-video
    Laplacian sharpness scan. Faster for quick iteration; use
    find_sharp_frames (the default) for a real calibration run."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in range(0, total, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    log_fn(f"Sampled {len(frames)} frames at stride {stride} (of {total} total).")
    return frames


def calibrate(
    video_path: Path,
    square_size: float = DEFAULT_SQUARE_SIZE,
    marker_ratio: float = DEFAULT_MARKER_RATIO,
    dict_name: str = DICT_NAME,
    frame_stride: Optional[int] = None,
    sharpness_window: int = 10,
    sharpness_threshold: float = 200.0,
    log_fn=print,
) -> tuple[CalibrationResult, UndistortionMaps, dict]:
    """Run the full pipeline: sample frames, detect markers, calibrate.

    Returns (calibration_result, undistort_maps, video_properties).
    """
    corrected_obj, aruco_dict = build_corrected_object_points(
        dict_name, square_size, marker_ratio
    )

    video_properties = get_video_properties(video_path)
    if frame_stride is not None:
        frames = _sample_frames_by_stride(video_path, frame_stride, log_fn=log_fn)
    else:
        sharp_indices = find_sharp_frames(
            video_path, window=sharpness_window, threshold=sharpness_threshold, log_fn=log_fn
        )
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        for idx in sharp_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        log_fn(f"Loaded {len(frames)} sharp frames.")

    image_size = (frames[0].shape[1], frames[0].shape[0])
    all_obj, all_img = collect_marker_correspondences(
        frames, aruco_dict, corrected_obj, log_fn=log_fn
    )
    if len(all_obj) < 5:
        raise ValueError(
            f"Only {len(all_obj)} usable frames -- need more board coverage "
            f"before calibrating."
        )

    log_fn("Running camera calibration...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj, all_img, image_size, None, None, flags=cv2.CALIB_RATIONAL_MODEL
    )
    log_fn(f"Calibration done: RMS error = {ret:.3f} px")
    if ret > 2.0:
        log_fn(
            f"WARNING: RMS error {ret:.1f}px is far above what a good "
            f"calibration should give (well under 1px). Check board flatness "
            f"and capture sharpness before trusting this result."
        )

    newcameramat, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, image_size, 0, image_size)
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramat, image_size, cv2.CV_32FC1)

    calib_result = CalibrationResult(
        error=float(ret),
        matrix=mtx.copy(),
        matrix_undistorted=newcameramat,
        distortion=dist,
        size=image_size,
        model_type="standard",
    )
    return calib_result, UndistortionMaps(mapx=mapx, mapy=mapy), video_properties


@click.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--camera-name", required=True, help="Camera name/model")
@click.option("--camera-mode", required=True, help="Camera mode/settings")
@click.option("--output-file", type=click.Path(), default="calibration.h5",
              help="Output calibration file path (.h5)")
@click.option("--square-size", default=DEFAULT_SQUARE_SIZE, help="Printed square side length, metres")
@click.option("--marker-ratio", default=DEFAULT_MARKER_RATIO, help="Marker side as a fraction of square size")
@click.option("--frame-stride", default=None, type=int,
              help="Sample every Nth frame directly instead of scanning the whole "
                   "video for locally-sharp frames (much faster, lower quality)")
@click.option("--sharpness-threshold", default=200.0,
              help="Raw Laplacian-variance threshold for the sharp-frame scan (ignored with --frame-stride)")
def main(video_path, camera_name, camera_mode, output_file, square_size, marker_ratio,
         frame_stride, sharpness_threshold):
    """Calibrate intrinsics from VIDEO_PATH using the corrected board layout
    for the swapped-and-rotated-quadrant ChArUco board (see module docstring)."""
    video_path_obj = Path(video_path)
    calib_result, undistort_maps, video_properties = calibrate(
        video_path_obj,
        square_size=square_size,
        marker_ratio=marker_ratio,
        frame_stride=frame_stride,
        sharpness_threshold=sharpness_threshold,
    )
    save_calibration_h5(
        Path(output_file), camera_name, camera_mode,
        calib_result, calib_result, undistort_maps, video_properties,
    )


if __name__ == "__main__":
    main()
