"""
Camera intrinsics calibration tool.

This script performs intrinsics calibration for cameras using checkerboard patterns.
It can process either video files or directories of images, and optionally saves
intermediate results including detected checkerboards and undistorted images.
"""

import cv2
import h5py
import click
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List


@dataclass
class CalibrationResult:
    """Results from camera calibration process.

    Attributes:
        error: RMS reprojection error in pixels
        matrix: 3x3 camera matrix (ORIGINAL from calibration)
        matrix_undistorted: 3x3 new camera matrix for undistorted images
        distortion: Distortion coefficients [k1, k2, p1, p2, k3, ...]
        size: Image size as (width, height)
        model_type: Calibration model type ('standard' or 'fisheye')
    """
    error: float
    matrix: np.ndarray
    matrix_undistorted: np.ndarray
    distortion: np.ndarray
    size: Tuple[int, int]
    model_type: str = 'standard'

    def to_dict(self) -> dict:
        """Convert to dictionary with lists instead of numpy arrays."""
        return {
            'error': float(self.error),
            'matrix': self.matrix.tolist(),
            'matrix_undistorted': self.matrix_undistorted.tolist(),
            'distortion': self.distortion.flatten().tolist(),
            'size': list(self.size),
            'model_type': self.model_type
        }


@dataclass
class UndistortionMaps:
    """Undistortion mapping arrays from OpenCV.

    Attributes:
        mapx: X-coordinate remapping array
        mapy: Y-coordinate remapping array
    """
    mapx: np.ndarray
    mapy: np.ndarray


def find_sharp_frames(
    video_path: Path,
    window: int = 10,
    threshold: float = 0.8,
    skip: int = 1,
    use_global_metric: bool = False,
    log_fn=None,
    save_debug: bool = False,
) -> List[int]:
    """Find frames in a video that are sharp enough for calibration.

    Uses Laplacian variance as a sharpness metric and finds local maxima.

    Args:
        video_path: Path to video file
        window: Window size for finding local maxima
        threshold: Minimum sharpness threshold (normalized if use_global_metric=True,
                   raw Laplacian variance otherwise)
        skip: Process every nth frame (1 = process all frames)
        use_global_metric: If True, normalize sharpness across entire video;
                          if False, use raw Laplacian values for local comparison

    Returns:
        List of frame indices that are sharp local maxima

    Raises:
        IOError: If video file cannot be opened
    """
    log = log_fn or print
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    laplacians = []
    frame_indices = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
            laplacians.append(laplacian)
            frame_indices.append(frame_count)

            if len(frame_indices) % 100 == 0:
                log(f"Processed {len(frame_indices)} frames (total: {frame_count})")

        frame_count += 1

    cap.release()

    laplacians_array = np.array(laplacians)

    if use_global_metric:
        metric = (laplacians_array - np.mean(laplacians_array)) / np.std(laplacians_array)
        log(f"Global sharpness metric: mean={np.mean(laplacians_array):.2f}, std={np.std(laplacians_array):.2f}")
    else:
        metric = laplacians_array
        log(f"Raw Laplacian metric: mean={np.mean(laplacians_array):.2f}, max={np.max(laplacians_array):.2f}")

    if save_debug:
        np.savez('laplacians.npz', laplacians_array=laplacians_array, metric=metric)

    maxima = []
    for i in range(window, len(metric) - window):
        window_slice = metric[i - window:i + window + 1]
        if metric[i] == np.max(window_slice) and metric[i] > threshold:
            if np.sum(window_slice == metric[i]) == 1:
                maxima.append(frame_indices[i])

    log(f"Found {len(maxima)} sharp frames out of {len(frame_indices)} analyzed ({frame_count} total)")
    return maxima


def extract_checkerboard_corners(
    image: np.ndarray,
    rows: int,
    cols: int,
    refine: bool = True
) -> Optional[np.ndarray]:
    """Extract checkerboard corners from an image.

    Args:
        image: Input image (color or grayscale)
        rows: Number of internal corners in rows
        cols: Number of internal corners in columns
        refine: Whether to refine corner positions with sub-pixel accuracy

    Returns:
        Array of corner positions if found, None otherwise
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

    if found and refine:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    return corners if found else None


def process_video_for_checkerboards(
    video_path: Path,
    rows: int,
    cols: int,
    window: int = 10,
    threshold: float = 0.8,
    skip: int = 1,
    use_global_metric: bool = False,
    log_fn=None,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Extract checkerboard frames and corners from a video.

    Args:
        video_path: Path to video file
        rows: Number of internal corners in rows
        cols: Number of internal corners in columns
        window: Window size for sharpness detection
        threshold: Sharpness threshold
        skip: Process every nth frame
        use_global_metric: Use global normalized metric instead of local comparison

    Returns:
        List of tuples (frame_idx, frame, corners) for each detected checkerboard

    Raises:
        IOError: If video file cannot be opened
    """
    log = log_fn or print
    sharp_frames = find_sharp_frames(video_path, window, threshold, skip, use_global_metric, log_fn=log_fn)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    checkerboards = []

    for frame_idx in sharp_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            log(f"Warning: Failed to read frame {frame_idx}")
            continue

        corners = extract_checkerboard_corners(frame, rows, cols)
        if corners is not None:
            log(f"Checkerboard detected in frame {frame_idx}")
            checkerboards.append((frame_idx, frame, corners))

    cap.release()
    return checkerboards


def process_images_for_checkerboards(
    image_dir: Path,
    rows: int,
    cols: int,
    log_fn=None,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Extract checkerboards from images in a directory.

    Args:
        image_dir: Path to directory containing images
        rows: Number of internal corners in rows
        cols: Number of internal corners in columns

    Returns:
        List of tuples (filename, image, corners) for each detected checkerboard
    """
    log = log_fn or print
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in image_dir.iterdir()
                   if f.suffix.lower() in image_extensions]

    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")

    checkerboards = []

    for img_path in sorted(image_files):
        image = cv2.imread(str(img_path))
        if image is None:
            log(f"Warning: Failed to read {img_path}")
            continue

        corners = extract_checkerboard_corners(image, rows, cols)
        if corners is not None:
            log(f"Checkerboard detected in {img_path.name}")
            checkerboards.append((img_path.name, image, corners))

    return checkerboards


def calibrate_camera(
    image_points: List[np.ndarray],
    object_points: List[np.ndarray],
    image_size: Tuple[int, int],
    use_fisheye: bool = False,
    log_fn=None,
) -> Tuple[CalibrationResult, UndistortionMaps]:
    """Perform camera calibration using detected checkerboard corners.

    Args:
        image_points: List of detected corner arrays (one per image)
        object_points: List of corresponding 3D object point arrays
        image_size: Size of images as (width, height)
        use_fisheye: Use fisheye calibration model instead of standard pinhole model

    Returns:
        Tuple of (calibration_result, undistortion_maps)
    """
    if use_fisheye:
        # Fisheye calibration
        # Use minimal flags for better compatibility with diverse checkerboard configurations
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

        # Fisheye expects shape (N, 1, 3) for object points and (N, 1, 2) for image points.
        # Sort each view so the corner farthest from the image centre comes first —
        # this prevents the InitExtrinsics "fabs(norm_u1) > 0" assertion from firing
        # when the first point normalises to (0,0) near the principal point.
        f_init = max(image_size)
        cx, cy = image_size[0] / 2.0, image_size[1] / 2.0
        obj_sorted, img_sorted = [], []
        for obj_p, img_p in zip(object_points, image_points):
            img_2d = img_p.reshape(-1, 2)
            order = np.argsort(-np.hypot(img_2d[:, 0] - cx, img_2d[:, 1] - cy))
            obj_sorted.append(obj_p.reshape(-1, 3)[order])
            img_sorted.append(img_2d[order])
        object_points_fisheye = [np.ascontiguousarray(p.reshape(-1, 1, 3), dtype=np.float64) for p in obj_sorted]
        image_points_fisheye  = [np.ascontiguousarray(p.reshape(-1, 1, 2), dtype=np.float64) for p in img_sorted]
        K = np.array([[f_init, 0, cx],
                      [0, f_init, cy],
                      [0, 0, 1]], dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(object_points))]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(object_points))]

        ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            object_points_fisheye,
            image_points_fisheye,
            image_size,
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        )

        # For fisheye, get the optimal camera matrix
        newcameramat = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            mtx, dist, image_size, np.eye(3), balance=0.0
        )

        mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
            mtx, dist, np.eye(3), newcameramat, image_size, cv2.CV_32FC1
        )

        # Store original mtx (not newcameramat!)
        original_mtx = mtx.copy()
        model_type = 'fisheye'
    else:
        # Standard pinhole calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
            flags=cv2.CALIB_RATIONAL_MODEL
        )

        newcameramat, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, image_size, 0, image_size
        )

        mapx, mapy = cv2.initUndistortRectifyMap(
            mtx, dist, None, newcameramat, image_size, cv2.CV_32FC1
        )

        # Store original mtx (not newcameramat!)
        original_mtx = mtx.copy()
        model_type = 'standard'

    calib_result = CalibrationResult(
        error=float(ret),
        matrix=original_mtx,
        matrix_undistorted=newcameramat,
        distortion=dist,
        size=image_size,
        model_type=model_type
    )

    undistort_maps = UndistortionMaps(mapx=mapx, mapy=mapy)

    log = log_fn or print
    log(f"Calibration done ({model_type}): RMS error = {ret:.3f} px")

    return calib_result, undistort_maps


def save_checkerboard_images(
    checkerboards: List[Tuple],
    output_dir: Path,
    subdir_name: str
) -> None:
    """Save checkerboard images to a directory.

    Args:
        checkerboards: List of (identifier, image, corners) tuples
        output_dir: Base output directory
        subdir_name: Name of subdirectory to create
    """
    save_dir = output_dir / subdir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for identifier, image, _ in checkerboards:
        if isinstance(identifier, int):
            filename = f"frame_{identifier:04d}.png"
        else:
            filename = identifier

        output_path = save_dir / filename
        cv2.imwrite(str(output_path), image)

    print(f"Saved {len(checkerboards)} images to {save_dir}")


def undistort_checkerboards(
    checkerboards: List[Tuple],
    undistort_maps: UndistortionMaps
) -> List[Tuple]:
    """Apply undistortion to checkerboard images.

    Args:
        checkerboards: List of (identifier, image, corners) tuples
        undistort_maps: Undistortion mapping arrays

    Returns:
        List of (identifier, undistorted_image, new_corners) tuples
    """
    undistorted = []

    for identifier, image, _ in checkerboards:
        undist_image = cv2.remap(
            image, undistort_maps.mapx, undistort_maps.mapy, cv2.INTER_LINEAR
        )
        undistorted.append((identifier, undist_image, None))

    return undistorted


def get_video_properties(video_path: Path) -> dict:
    """Extract video properties using OpenCV.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary containing video properties
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    props = {
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': float(cap.get(cv2.CAP_PROP_FPS)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
        'format': chr(int(cap.get(cv2.CAP_PROP_FOURCC)) & 0xff) +
                  chr((int(cap.get(cv2.CAP_PROP_FOURCC)) >> 8) & 0xff) +
                  chr((int(cap.get(cv2.CAP_PROP_FOURCC)) >> 16) & 0xff) +
                  chr((int(cap.get(cv2.CAP_PROP_FOURCC)) >> 24) & 0xff)
    }

    cap.release()
    return props


def save_calibration_h5(
    output_path: Path,
    camera_name: str,
    camera_mode: str,
    original_calib: CalibrationResult,
    undistorted_calib: CalibrationResult,
    undistort_maps: UndistortionMaps,
    video_properties: Optional[dict] = None
) -> None:
    """Save calibration results to HDF5 file.

    Args:
        output_path: Path for output HDF5 file
        camera_name: Name/model of the camera
        camera_mode: Camera mode/settings description
        original_calib: Calibration from original images
        undistorted_calib: Calibration from undistorted images
        undistort_maps: Undistortion mapping arrays
        video_properties: Optional video properties from OpenCV
    """
    with h5py.File(output_path, 'w') as f:
        # Camera metadata
        f.attrs['camera_name'] = camera_name
        f.attrs['camera_mode'] = camera_mode
        f.attrs['calibration_model'] = original_calib.model_type

        # Intrinsics group - stores ORIGINAL calibration parameters
        intrinsics = f.create_group('intrinsics')
        intrinsics.create_dataset('matrix', data=original_calib.matrix)  # Original K
        intrinsics.create_dataset('matrix_undistorted', data=original_calib.matrix_undistorted)  # New K for undistorted images
        intrinsics.create_dataset('distortions', data=original_calib.distortion.flatten())
        intrinsics.attrs['size'] = original_calib.size
        intrinsics.attrs['error'] = original_calib.error
        intrinsics.attrs['model_type'] = original_calib.model_type

        # Undistortion maps
        maps = f.create_group('undistortion_maps')
        maps.create_dataset('mapx', data=undistort_maps.mapx, compression='gzip', compression_opts=9)
        maps.create_dataset('mapy', data=undistort_maps.mapy, compression='gzip', compression_opts=9)

        # Calibration after undistortion
        calib_undist = f.create_group('calibration_undistorted')
        calib_undist.create_dataset('matrix', data=undistorted_calib.matrix)
        calib_undist.create_dataset('distortions', data=undistorted_calib.distortion.flatten())
        calib_undist.attrs['size'] = undistorted_calib.size
        calib_undist.attrs['error'] = undistorted_calib.error
        calib_undist.attrs['model_type'] = undistorted_calib.model_type

        # Video properties if available
        if video_properties:
            video = f.create_group('video_properties')
            for key, value in video_properties.items():
                video.attrs[key] = value

    print(f"Calibration results saved to {output_path}")
    print(f"Use h5py or HDFView to inspect the file contents")


# ---------------------------------------------------------------------------
# ChArUco board detection
# ---------------------------------------------------------------------------

#: Mapping from human-readable name to cv2.aruco dictionary constant.
#: Built lazily so importing this module does not fail if cv2.aruco is absent.
def _aruco_dicts() -> dict[str, int]:
    try:
        d = cv2.aruco
        return {
            "DICT_4X4_50":  d.DICT_4X4_50,
            "DICT_4X4_100": d.DICT_4X4_100,
            "DICT_5X5_50":  d.DICT_5X5_50,
            "DICT_6X6_50":  d.DICT_6X6_50,
            "DICT_6X6_250": d.DICT_6X6_250,
            "DICT_7X7_250": d.DICT_7X7_250,
        }
    except AttributeError:
        return {}


def charuco_available() -> bool:
    """Return True if cv2.aruco is present."""
    return hasattr(cv2, "aruco") and hasattr(cv2.aruco, "CharucoBoard")


def create_charuco_board(
    rows: int,
    cols: int,
    square_length: float,
    marker_length: float,
    dict_name: str = "DICT_6X6_250",
):
    """Create a cv2.aruco.CharucoBoard.

    Args:
        rows: Number of squares in the vertical direction.
        cols: Number of squares in the horizontal direction.
        square_length: Physical side length of a chessboard square.
        marker_length: Physical side length of an ArUco marker (must be < square_length).
        dict_name: Name key from _aruco_dicts() (default DICT_6X6_250).

    Returns:
        cv2.aruco.CharucoBoard instance.
    """
    dicts = _aruco_dicts()
    if dict_name not in dicts:
        raise ValueError(f"Unknown ArUco dictionary '{dict_name}'. "
                         f"Choose from: {list(dicts)}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(dicts[dict_name])
    # OpenCV 4.7+ uses tuple constructor; older builds use CharucoBoard_create
    try:
        board = cv2.aruco.CharucoBoard((cols, rows), square_length, marker_length, aruco_dict)
    except TypeError:
        board = cv2.aruco.CharucoBoard_create(cols, rows, square_length, marker_length, aruco_dict)
    return board


def extract_charuco_corners(
    image: np.ndarray,
    board,
    min_corners: int = 4,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Detect ChArUco corners in *image*.

    Returns:
        (charuco_corners, charuco_ids) or (None, None) if detection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    try:
        # OpenCV 4.7+ new API
        detector = cv2.aruco.CharucoDetector(board)
        corners, ids, _, _ = detector.detectBoard(gray)
    except AttributeError:
        # Older API
        aruco_dict = board.getDictionary() if hasattr(board, "getDictionary") else board.dictionary
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        if marker_ids is None or len(marker_ids) == 0:
            return None, None
        _, corners, ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )

    if corners is None or ids is None or len(ids) < min_corners:
        return None, None
    return corners, ids


def process_video_for_charuco(
    video_path: Path,
    board,
    window: int = 10,
    threshold: float = 0.8,
    skip: int = 1,
    use_global_metric: bool = False,
    min_corners: int = 6,
    log_fn=None,
) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Extract ChArUco frames from *video_path*.

    Returns:
        List of (frame_idx, frame, charuco_corners, charuco_ids).
    """
    log = log_fn or print
    sharp_frames = find_sharp_frames(
        video_path, window, threshold, skip, use_global_metric, log_fn=log_fn
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    results = []
    for frame_idx in sharp_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            log(f"Warning: Failed to read frame {frame_idx}")
            continue
        corners, ids = extract_charuco_corners(frame, board, min_corners)
        if corners is not None:
            log(f"ChArUco detected in frame {frame_idx} ({len(ids)} corners)")
            results.append((frame_idx, frame, corners, ids))

    cap.release()
    return results


def process_images_for_charuco(
    image_dir: Path,
    board,
    min_corners: int = 6,
    log_fn=None,
) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Detect ChArUco corners in images from a directory.

    Returns:
        List of (filename, image, charuco_corners, charuco_ids).
    """
    log = log_fn or print
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    files = [f for f in image_dir.iterdir() if f.suffix.lower() in extensions]
    if not files:
        raise ValueError(f"No image files found in {image_dir}")

    results = []
    for img_path in sorted(files):
        img = cv2.imread(str(img_path))
        if img is None:
            log(f"Warning: Failed to read {img_path}")
            continue
        corners, ids = extract_charuco_corners(img, board, min_corners)
        if corners is not None:
            log(f"ChArUco detected in {img_path.name} ({len(ids)} corners)")
            results.append((img_path.name, img, corners, ids))

    return results


def calibrate_camera_charuco(
    all_corners: List[np.ndarray],
    all_ids: List[np.ndarray],
    board,
    image_size: Tuple[int, int],
    use_fisheye: bool = False,
    log_fn=None,
) -> Tuple[CalibrationResult, UndistortionMaps]:
    """Run camera calibration from ChArUco corner detections.

    Args:
        all_corners: List of charuco_corners arrays (one per image).
        all_ids:     List of charuco_ids arrays (one per image).
        board:       The cv2.aruco.CharucoBoard used for detection.
        image_size:  (width, height) of the images.
        use_fisheye: Use fisheye calibration model.

    Returns:
        (CalibrationResult, UndistortionMaps)
    """
    log = log_fn or print

    if use_fisheye:
        # cv2.aruco.calibrateCameraCharuco doesn't support fisheye, so we extract
        # the 3D-2D correspondences manually and call cv2.fisheye.calibrate directly.
        object_points: List[np.ndarray] = []
        image_points: List[np.ndarray] = []
        for corners, ids in zip(all_corners, all_ids):
            try:
                # OpenCV 4.7+ Board.matchImagePoints
                obj_pts, img_pts = board.matchImagePoints(corners, ids)
            except AttributeError:
                # Older API: index chessboardCorners by detected id
                flat_ids = ids.flatten()
                obj_pts = board.chessboardCorners[flat_ids]          # (N, 3)
                img_pts = corners.reshape(-1, 2)                      # (N, 2)
            if obj_pts is None or len(obj_pts) < 4:
                continue
            object_points.append(obj_pts.reshape(-1, 3).astype(np.float64))
            image_points.append(img_pts.reshape(-1, 2).astype(np.float64))

        if not object_points:
            raise RuntimeError("No valid ChArUco frames for fisheye calibration.")

        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

        # Sort points in each view so the corner farthest from the image centre comes
        # first.  cv2.fisheye.calibrate calls InitExtrinsics which normalises the
        # FIRST image point with the initial K; if that point is at or near the
        # principal point the normalised vector is (0,0) and an assertion fires:
        #   "fabs(norm_u1) > 0"  (fisheye.cpp)
        # Placing the outermost corner first guarantees a non-zero norm regardless of
        # the initial K estimate.
        cx_est, cy_est = image_size[0] / 2.0, image_size[1] / 2.0
        sorted_object_points = []
        sorted_image_points = []
        for obj_p, img_p in zip(object_points, image_points):
            img_2d = img_p.reshape(-1, 2)
            dists = np.hypot(img_2d[:, 0] - cx_est, img_2d[:, 1] - cy_est)
            order = np.argsort(-dists)  # farthest first
            sorted_object_points.append(obj_p[order])
            sorted_image_points.append(img_2d[order])

        obj_fe = [np.ascontiguousarray(p.reshape(-1, 1, 3), dtype=np.float64) for p in sorted_object_points]
        img_fe = [np.ascontiguousarray(p.reshape(-1, 1, 2), dtype=np.float64) for p in sorted_image_points]

        f_init = max(image_size)
        cx, cy = cx_est, cy_est
        K = np.array([[f_init, 0, cx], [0, f_init, cy], [0, 0, 1]], dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in obj_fe]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in obj_fe]

        try:
            ret, K, dist, rvecs, tvecs = cv2.fisheye.calibrate(
                obj_fe, img_fe, image_size, K, D, rvecs, tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            )
        except cv2.error as exc:
            # If the assertion still fires for a specific view, remove it and retry.
            # This handles frames where all detected corners happen to cluster near
            # the principal point (degenerate view coverage for fisheye).
            if "norm_u1" not in str(exc) or len(obj_fe) <= 6:
                raise
            log(f"Warning: fisheye InitExtrinsics failed — scanning for degenerate views…")
            good_obj, good_img = [], []
            for i, (o, im, rv, tv) in enumerate(zip(obj_fe, img_fe, rvecs, tvecs)):
                probe_obj = good_obj + [o]
                probe_img = good_img + [im]
                probe_rv  = [np.zeros((1,1,3), dtype=np.float64)] * len(probe_obj)
                probe_tv  = [np.zeros((1,1,3), dtype=np.float64)] * len(probe_obj)
                try:
                    cv2.fisheye.calibrate(
                        probe_obj, probe_img, image_size,
                        K.copy(), D.copy(), probe_rv, probe_tv,
                        calibration_flags,
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1e-3),
                    )
                    good_obj.append(o)
                    good_img.append(im)
                except cv2.error:
                    log(f"  Removed degenerate view {i} (kept {len(good_obj)} so far)")
            if not good_obj:
                raise RuntimeError("All views are degenerate for fisheye calibration — "
                                   "check board parameters and coverage.") from exc
            K = np.array([[f_init, 0, cx], [0, f_init, cy], [0, 0, 1]], dtype=np.float64)
            D = np.zeros((4, 1), dtype=np.float64)
            rv2 = [np.zeros((1,1,3), dtype=np.float64)] * len(good_obj)
            tv2 = [np.zeros((1,1,3), dtype=np.float64)] * len(good_obj)
            log(f"Retrying fisheye calibration with {len(good_obj)} non-degenerate views…")
            ret, K, dist, rvecs, tvecs = cv2.fisheye.calibrate(
                good_obj, good_img, image_size, K, D, rv2, tv2,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            )

        newcameramat = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, dist, image_size, np.eye(3), balance=0.0
        )
        mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
            K, dist, np.eye(3), newcameramat, image_size, cv2.CV_32FC1
        )
        result = CalibrationResult(
            error=float(ret),
            matrix=K.copy(),
            matrix_undistorted=newcameramat,
            distortion=dist,
            size=image_size,
            model_type="fisheye",
        )
        log(f"ChArUco fisheye calibration done: RMS error = {ret:.3f} px")
        return result, UndistortionMaps(mapx=mapx, mapy=mapy)

    ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, image_size, None, None
    )

    K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, image_size, 0, image_size)
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K_new, image_size, cv2.CV_32FC1)

    result = CalibrationResult(
        error=float(ret),
        matrix=K.copy(),
        matrix_undistorted=K_new,
        distortion=dist,
        size=image_size,
        model_type="standard",
    )
    log(f"ChArUco calibration done: RMS error = {ret:.3f} px")
    return result, UndistortionMaps(mapx=mapx, mapy=mapy)


# ---------------------------------------------------------------------------
# Unified pipeline (callable from CLI and from UI thread)
# ---------------------------------------------------------------------------

def collect_sharp_frames(
    video_path: Path,
    window: int = 10,
    threshold: float = 0.8,
    skip: int = 1,
    use_global_metric: bool = False,
    log_fn=None,
) -> List[np.ndarray]:
    """Scan *video_path* for sharp frames and return them as BGR arrays.

    This is the slow step (reads every frame to compute sharpness).
    Cache the result and pass as *preloaded_frames* to
    :func:`run_intrinsics_pipeline` to skip video scanning on re-runs
    with changed board or model parameters.
    """
    log = log_fn or print
    sharp_indices = find_sharp_frames(
        video_path, window, threshold, skip, use_global_metric, log_fn=log_fn
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    frames: List[np.ndarray] = []
    for frame_idx in sharp_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            log(f"Warning: Failed to read frame {frame_idx}")
    cap.release()
    log(f"Loaded {len(frames)} sharp frames.")
    return frames


def _detect_checkerboard_in_frames(
    frames: List[np.ndarray],
    rows: int,
    cols: int,
    log_fn=None,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    log = log_fn or print
    results = []
    for i, frame in enumerate(frames):
        corners = extract_checkerboard_corners(frame, rows, cols)
        if corners is not None:
            log(f"Checkerboard detected in frame {i}")
            results.append((i, frame, corners))
    return results


def _detect_charuco_in_frames(
    frames: List[np.ndarray],
    board,
    min_corners: int = 6,
    log_fn=None,
) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    log = log_fn or print
    results = []
    for i, frame in enumerate(frames):
        corners, ids = extract_charuco_corners(frame, board, min_corners)
        if corners is not None:
            log(f"ChArUco detected in frame {i} ({len(ids)} corners)")
            results.append((i, frame, corners, ids))
    return results


def run_intrinsics_pipeline(
    input_path: Path,
    rows: int,
    cols: int,
    pattern: str = "checkerboard",
    square_size: float = 1.0,
    marker_size_ratio: float = 0.75,
    aruco_dict_name: str = "DICT_6X6_250",
    use_fisheye: bool = False,
    window: int = 10,
    threshold: float = 0.8,
    skip: int = 1,
    use_global_metric: bool = False,
    preloaded_frames: Optional[List[np.ndarray]] = None,
    log_fn=None,
) -> Tuple[CalibrationResult, UndistortionMaps]:
    """Run the full intrinsics calibration pipeline.

    Detects calibration pattern corners from *input_path* (video file or image
    directory), calibrates the camera, and returns the result.

    Args:
        input_path:        Video file or directory of images.
        rows:              Number of internal corners (rows) for checkerboard,
                           or number of squares (rows) for ChArUco.
        cols:              Columns counterpart.
        pattern:           "checkerboard" or "charuco".
        square_size:       Physical square size (arbitrary units for checkerboard;
                           metres recommended for ChArUco so scale is meaningful).
        marker_size_ratio: ChArUco only — marker side as a fraction of square_size.
        aruco_dict_name:   ChArUco only — dictionary name from _aruco_dicts().
        use_fisheye:       Use OpenCV fisheye model (checkerboard and ChArUco).
        window, threshold, skip, use_global_metric:
                           Passed to find_sharp_frames (video mode only).
        log_fn:            Optional callable for progress messages.  Defaults to print.

    Returns:
        (CalibrationResult, UndistortionMaps)
    """
    log = log_fn or print
    input_path = Path(input_path)

    if pattern == "charuco":
        if not charuco_available():
            raise RuntimeError("cv2.aruco is not available in this OpenCV build.")
        board = create_charuco_board(
            rows, cols,
            square_length=square_size,
            marker_length=square_size * marker_size_ratio,
            dict_name=aruco_dict_name,
        )
        if preloaded_frames is not None:
            log(f"Detecting ChArUco patterns in {len(preloaded_frames)} cached frames…")
            detections = _detect_charuco_in_frames(preloaded_frames, board, log_fn=log_fn)
        elif input_path.is_file():
            log("Scanning video for ChArUco boards…")
            detections = process_video_for_charuco(
                input_path, board, window, threshold, skip, use_global_metric, log_fn=log_fn
            )
        else:
            log("Scanning image directory for ChArUco boards…")
            detections = process_images_for_charuco(input_path, board, log_fn=log_fn)

        if not detections:
            raise ValueError("No ChArUco boards detected. "
                             "Check pattern settings and sharpness threshold.")
        log(f"Found {len(detections)} usable frames.")

        all_corners = [d[2] for d in detections]
        all_ids = [d[3] for d in detections]
        first_img = detections[0][1]
        image_size = (first_img.shape[1], first_img.shape[0])

        return calibrate_camera_charuco(
            all_corners, all_ids, board, image_size, use_fisheye=use_fisheye, log_fn=log_fn
        )

    else:  # checkerboard
        if preloaded_frames is not None:
            log(f"Detecting checkerboard patterns in {len(preloaded_frames)} cached frames…")
            detections = _detect_checkerboard_in_frames(preloaded_frames, rows, cols, log_fn=log_fn)
        elif input_path.is_file():
            log("Scanning video for checkerboard corners…")
            detections = process_video_for_checkerboards(
                input_path, rows, cols, window, threshold, skip, use_global_metric, log_fn=log_fn
            )
        else:
            log("Scanning image directory for checkerboard corners…")
            detections = process_images_for_checkerboards(
                input_path, rows, cols, log_fn=log_fn
            )

        if not detections:
            raise ValueError("No checkerboards detected. "
                             "Check rows/cols settings and sharpness threshold.")
        log(f"Found {len(detections)} usable frames.")

        if len(detections) < 10:
            log(f"WARNING: Only {len(detections)} frames — at least 10 recommended.")

        first_img = detections[0][1]
        image_size = (first_img.shape[1], first_img.shape[0])

        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
        object_points = [objp for _ in detections]
        image_points = [d[2] for d in detections]

        log("Running camera calibration…")
        return calibrate_camera(
            image_points, object_points, image_size,
            use_fisheye=use_fisheye, log_fn=log_fn,
        )


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--camera-name', required=True, help='Camera name/model')
@click.option('--camera-mode', required=True, help='Camera mode/settings')
@click.option('--rows', default=7, help='Number of internal corners in rows')
@click.option('--cols', default=10, help='Number of internal corners in columns')
@click.option('--output-dir', type=click.Path(), help='Directory to save intermediate images')
@click.option('--output-file', type=click.Path(), default='calibration.h5',
              help='Output calibration file path (.h5 or .hdf5)')
@click.option('--window', default=10, help='Window size for sharpness detection (video only)')
@click.option('--threshold', default=0.8, help='Sharpness threshold (video only). Use ~10 for local (default), ~0.8 for global metric')
@click.option('--skip', default=1, help='Process every nth frame (video only, 1 = all frames)')
@click.option('--global-sharpness-metric', is_flag=True,
              help='Use global normalized sharpness metric instead of local Laplacian comparison')
@click.option('--fisheye', is_flag=True,
              help='Use fisheye camera calibration model instead of standard pinhole + distortion model')
def main(
    input_path: str,
    camera_name: str,
    camera_mode: str,
    rows: int,
    cols: int,
    output_dir: Optional[str],
    output_file: str,
    window: int,
    threshold: float,
    skip: int,
    global_sharpness_metric: bool,
    fisheye: bool
) -> None:
    """Calibrate camera intrinsics from video or image directory.

    INPUT_PATH can be either a video file or a directory containing images.

    Examples:
        calibrate_intrinsics video.mp4 --camera-name "GoPro HERO11" --camera-mode "4K"

        calibrate_intrinsics images/ --camera-name "Canon R5" --camera-mode "8K" --output-dir results/

        calibrate_intrinsics video.mp4 --camera-name "Insta360" --camera-mode "5.7K" --skip 5 --output-file calib.h5

        calibrate_intrinsics video.mp4 --camera-name "GoPro" --camera-mode "2.7K" --global-sharpness-metric --threshold 0.8

        calibrate_intrinsics video.mp4 --camera-name "Insta360 X3" --camera-mode "5.7K" --fisheye
    """
    input_path_obj = Path(input_path)
    output_dir_obj = Path(output_dir) if output_dir else None
    output_file_obj = Path(output_file)

    # Get video properties if input is a video file
    video_properties = None
    if input_path_obj.is_file():
        print(f"Processing video: {input_path_obj}")
        video_properties = get_video_properties(input_path_obj)
        if video_properties:
            print(f"Video properties: {video_properties['width']}x{video_properties['height']} @ {video_properties['fps']:.2f} fps, "
                  f"{video_properties['frame_count']} frames, format: {video_properties['format']}")

        checkerboards = process_video_for_checkerboards(
            input_path_obj, rows, cols, window, threshold, skip, global_sharpness_metric
        )
    elif input_path_obj.is_dir():
        print(f"Processing images from: {input_path_obj}")
        checkerboards = process_images_for_checkerboards(
            input_path_obj, rows, cols
        )
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    if not checkerboards:
        raise ValueError("No checkerboards detected in input")

    print(f"Found {len(checkerboards)} checkerboards")

    if len(checkerboards) < 10:
        print(f"WARNING: Only {len(checkerboards)} checkerboards found. "
              f"At least 10 are recommended for good calibration.")

    # Save original checkerboard images if output directory specified
    if output_dir_obj:
        save_checkerboard_images(checkerboards, output_dir_obj, 'checkerboard_orig')

    # Prepare calibration data
    _, first_image, _ = checkerboards[0]
    image_size = (first_image.shape[1], first_image.shape[0])

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    object_points = [objp for _ in checkerboards]
    image_points = [corners for _, _, corners in checkerboards]

    # Calibrate with original images
    print(f"\nCalibrating with original images using {'fisheye' if fisheye else 'standard'} model...")
    original_calib, undistort_maps = calibrate_camera(
        image_points, object_points, image_size, use_fisheye=fisheye
    )

    # Undistort checkerboard images
    print("\nUndistorting checkerboard images...")
    undistorted_boards = undistort_checkerboards(checkerboards, undistort_maps)

    # Re-extract corners from undistorted images
    print("\nRe-extracting corners from undistorted images...")
    undist_with_corners = []
    for identifier, undist_image, _ in undistorted_boards:
        corners = extract_checkerboard_corners(undist_image, rows, cols)
        if corners is not None:
            undist_with_corners.append((identifier, undist_image, corners))
        else:
            print(f"Warning: Checkerboard not found in undistorted image {identifier}")

    if output_dir_obj:
        save_checkerboard_images(
            undist_with_corners, output_dir_obj, 'checkerboard_undistort'
        )

    # Calibrate with undistorted images
    print(f"\nCalibrating with undistorted images using {'fisheye' if fisheye else 'standard'} model...")
    undist_image_points = [corners for _, _, corners in undist_with_corners]
    undist_object_points = [objp for _ in undist_with_corners]

    undistorted_calib, _ = calibrate_camera(
        undist_image_points, undist_object_points, image_size, use_fisheye=fisheye
    )

    # Save results
    save_calibration_h5(
        output_file_obj,
        camera_name,
        camera_mode,
        original_calib,
        undistorted_calib,
        undistort_maps,
        video_properties
    )

    print("\n" + "="*60)
    print("Calibration Complete!")
    print("="*60)
    print(f"Original calibration error: {original_calib.error:.3f} pixels")
    print(f"Undistorted calibration error: {undistorted_calib.error:.3f} pixels")
    print(f"Results saved to: {output_file_obj}")


if __name__ == '__main__':
    main()
