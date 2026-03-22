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
    use_global_metric: bool = False
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
                print(f"Processed {len(frame_indices)} frames (total: {frame_count})")

        frame_count += 1

    cap.release()

    laplacians_array = np.array(laplacians)

    if use_global_metric:
        # Normalize sharpness metric across entire video
        metric = (laplacians_array - np.mean(laplacians_array)) / np.std(laplacians_array)
        print(f"Using global sharpness metric based on image Laplacians: (mean={np.mean(laplacians_array):.2f}, std={np.std(laplacians_array):.2f})")
    else:
        # Use raw Laplacian values
        metric = laplacians_array
        print(f"Using raw Laplacian values as sharpness metric: (mean={np.mean(laplacians_array):.2f}, max={np.max(laplacians_array):.2f})")

    np.savez('laplacians.npz', laplacians_array=laplacians_array, metric=metric)

    # Find local maxima above threshold
    maxima = []
    for i in range(window, len(metric) - window):
        window_slice = metric[i - window:i + window + 1]
        if metric[i] == np.max(window_slice) and metric[i] > threshold:
            if np.sum(window_slice == metric[i]) == 1:
                maxima.append(frame_indices[i])

    print(f"Found {len(maxima)} sharp frames out of {len(frame_indices)} analyzed frames ({frame_count} total)")
    print(f"Threshold used: {threshold}, window size: {window}, skip: {skip}, global metric: {use_global_metric}")
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
    use_global_metric: bool = False
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
    sharp_frames = find_sharp_frames(video_path, window, threshold, skip, use_global_metric)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    checkerboards = []

    for frame_idx in sharp_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {frame_idx}")
            continue

        corners = extract_checkerboard_corners(frame, rows, cols)
        if corners is not None:
            print(f"Found checkerboard in frame {frame_idx}")
            checkerboards.append((frame_idx, frame, corners))

    cap.release()
    return checkerboards


def process_images_for_checkerboards(
    image_dir: Path,
    rows: int,
    cols: int
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Extract checkerboards from images in a directory.

    Args:
        image_dir: Path to directory containing images
        rows: Number of internal corners in rows
        cols: Number of internal corners in columns

    Returns:
        List of tuples (filename, image, corners) for each detected checkerboard
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in image_dir.iterdir()
                   if f.suffix.lower() in image_extensions]

    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")

    checkerboards = []

    for img_path in sorted(image_files):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Failed to read {img_path}")
            continue

        corners = extract_checkerboard_corners(image, rows, cols)
        if corners is not None:
            print(f"Found checkerboard in {img_path.name}")
            checkerboards.append((img_path.name, image, corners))

    return checkerboards


def calibrate_camera(
    image_points: List[np.ndarray],
    object_points: List[np.ndarray],
    image_size: Tuple[int, int],
    use_fisheye: bool = False
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

        # Reshape object_points and image_points for fisheye calibration
        # Fisheye expects shape (N, 1, 3) for object points and (N, 1, 2) for image points
        # Ensure all arrays are contiguous and float64
        object_points_fisheye = [np.ascontiguousarray(pts.reshape(-1, 1, 3), dtype=np.float64) for pts in object_points]
        image_points_fisheye = [np.ascontiguousarray(pts.reshape(-1, 1, 2), dtype=np.float64) for pts in image_points]

        # Initialize camera matrix with reasonable estimate
        # Focal length approximation: ~image_width for fisheye lenses
        f_init = max(image_size)
        cx, cy = image_size[0] / 2.0, image_size[1] / 2.0
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
        matrix=original_mtx,  # Original camera matrix from calibration
        matrix_undistorted=newcameramat,  # New camera matrix for undistorted images
        distortion=dist,
        size=image_size,
        model_type=model_type
    )

    undistort_maps = UndistortionMaps(mapx=mapx, mapy=mapy)

    print(f"Calibration successful ({model_type} model): error = {ret:.3f} pixels")

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
