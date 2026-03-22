import json
import re
import yaml
import h5py
import click
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.optimize import least_squares
from collections import defaultdict


def load_project_config(yaml_path: str) -> dict:
    """Load project configuration from YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_intrinsics_from_yaml(yaml_path: str) -> dict:
    """Load camera intrinsics calibration from a YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    intrinsics = data.get("intrinsics", {})
    size = tuple(int(x) for x in intrinsics["size"])
    matrix = np.array(intrinsics["matrix"], dtype=np.float64)
    distortions = np.array([intrinsics["distortions"]], dtype=np.float64)
    fisheye = data.get("fisheye", False)  # Default to standard model

    return {
        "size": size,
        "matrix": matrix,
        "distortion": distortions,
        "fisheye": fisheye,
    }


def load_intrinsics_from_h5(h5_path: str) -> dict:
    """Load camera intrinsics calibration from an HDF5 file.

    Handles both old and new file formats:
    - New format: Has 'matrix' (original K) and 'matrix_undistorted' (new K)
    - Old format: Only has 'matrix' (which is actually the new K - confusing!)
    """
    with h5py.File(h5_path, 'r') as f:
        if 'intrinsics' not in f:
            raise ValueError(f"HDF5 file {h5_path} does not contain 'intrinsics' group")

        intrinsics = f['intrinsics']
        size = tuple(intrinsics.attrs['size'])
        distortions = intrinsics['distortions'][:].reshape(1, -1)

        # Check if this is new format (has matrix_undistorted)
        if 'matrix_undistorted' in intrinsics:
            # New format: has both original and undistorted matrices
            matrix_original = intrinsics['matrix'][:]
            matrix_undistorted = intrinsics['matrix_undistorted'][:]
            print(f"  Loaded NEW format calibration from {h5_path}")
        else:
            # Old format: 'matrix' is actually the new camera matrix
            # We'll use it as both - this is not ideal but works for existing files
            matrix_original = intrinsics['matrix'][:]
            matrix_undistorted = intrinsics['matrix'][:]
            print(f"  Loaded OLD format calibration from {h5_path} (matrix used as both original and undistorted)")

        # Load undistortion maps if available (for coordinate conversion)
        undistort_mapx = None
        undistort_mapy = None
        if 'undistortion_maps' in f:
            undistort_mapx = f['undistortion_maps/mapx'][:]
            undistort_mapy = f['undistortion_maps/mapy'][:]

        # Read calibration model (default to 'standard' for older files)
        calibration_model = f.attrs.get('calibration_model', 'standard')
        if isinstance(calibration_model, bytes):
            calibration_model = calibration_model.decode('utf-8')
        fisheye = (calibration_model == 'fisheye')

        return {
            "size": size,
            "matrix": matrix_undistorted,  # The new camera matrix for undistorted images
            "matrix_original": matrix_original,  # Original K from calibration
            "distortion": distortions,  # Original distortion coefficients
            "fisheye": fisheye,
            "h5_path": h5_path,  # Store path to reload maps when needed
            "undistort_mapx": undistort_mapx,
            "undistort_mapy": undistort_mapy,
        }


_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)


def load_intrinsics_from_db(registry_path: str, intrinsics_id: str) -> dict:
    """Load camera intrinsics from the posetrak registry database.

    The DB stores K_new (undistorted optimal matrix) in fx/fy/cx/cy and the
    original K in matrix_original.  This matches what load_intrinsics_from_h5
    returns — callers use ``matrix`` (= K_new) with zero distortion for
    undistorted images.
    """
    import sqlite3
    import struct
    import zlib

    conn = sqlite3.connect(registry_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM intrinsics_calibrations WHERE id = ?", (intrinsics_id,)
    ).fetchone()
    conn.close()

    if row is None:
        raise ValueError(f"intrinsics_calibration {intrinsics_id!r} not found in {registry_path}")

    # K_new (undistorted optimal matrix) — stored as fx/fy/cx/cy
    fx, fy, cx_val, cy = row["fx"], row["fy"], row["cx"], row["cy"]
    matrix_undistorted = np.array([[fx, 0.0, cx_val], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    # Original K — stored as 9 float64s little-endian
    if row["matrix_original"]:
        vals = struct.unpack("<9d", bytes(row["matrix_original"]))
        matrix_original = np.array(vals).reshape(3, 3)
    else:
        matrix_original = matrix_undistorted.copy()

    # Distortion coefficients
    if row["dist_coeffs"]:
        n = len(bytes(row["dist_coeffs"])) // 8
        dist = np.array(struct.unpack(f"<{n}d", bytes(row["dist_coeffs"]))).reshape(1, -1)
    else:
        dist = np.zeros((1, 4))

    image_width = row["image_width"]
    image_height = row["image_height"]
    size = (image_width, image_height) if image_width and image_height else (0, 0)

    # Undistortion maps (optional)
    undistort_mapx = None
    undistort_mapy = None
    if row["undistort_mapx"] and image_width and image_height:
        mapx_bytes = zlib.decompress(bytes(row["undistort_mapx"]))
        mapy_bytes = zlib.decompress(bytes(row["undistort_mapy"]))
        undistort_mapx = np.frombuffer(mapx_bytes, dtype=np.float32).reshape(image_height, image_width)
        undistort_mapy = np.frombuffer(mapy_bytes, dtype=np.float32).reshape(image_height, image_width)

    fisheye = (row["distortion_model"] == "fisheye")

    print(f"  Loaded intrinsics from DB: {intrinsics_id}")
    print(f"  Size: {size}  fx={fx:.2f} fy={fy:.2f}  rms={row['rms_error']}")

    return {
        "size": size,
        "matrix": matrix_undistorted,      # K_new — for undistorted images
        "matrix_original": matrix_original, # original K
        "distortion": dist,
        "fisheye": fisheye,
        "undistort_mapx": undistort_mapx,
        "undistort_mapy": undistort_mapy,
    }


def load_intrinsics(calib_path: str, registry_path: str | None = None) -> dict:
    """Load intrinsics from a UUID (DB), HDF5 file, or YAML file."""
    if _UUID_RE.match(calib_path):
        if not registry_path:
            raise ValueError(
                f"calib.intrinsics is a UUID ({calib_path!r}) but --registry was not provided."
            )
        return load_intrinsics_from_db(registry_path, calib_path)

    p = Path(calib_path)
    if p.suffix.lower() in ['.h5', '.hdf5']:
        return load_intrinsics_from_h5(str(p))
    elif p.suffix.lower() in ['.yaml', '.yml']:
        return load_intrinsics_from_yaml(str(p))
    else:
        raise ValueError(f"Unsupported calibration file format: {p.suffix}")


def convert_undistorted_to_distorted_coords(undist_x: float, undist_y: float,
                                             mapx: np.ndarray, mapy: np.ndarray) -> Tuple[float, float]:
    """Convert undistorted pixel coordinates to distorted using undistortion maps.

    The undistortion maps contain the distorted coordinates for each undistorted pixel.
    We need to find which undistorted pixel maps to our desired distorted location.
    This is done by finding the closest match in the maps (inverse lookup).

    Args:
        undist_x, undist_y: Undistorted pixel coordinates
        mapx, mapy: Undistortion maps (from calibration)

    Returns:
        Distorted pixel coordinates (x, y)
    """
    # Round to nearest pixel in undistorted space
    ux = int(round(undist_x))
    uy = int(round(undist_y))

    # Check bounds
    h, w = mapx.shape
    if 0 <= ux < w and 0 <= uy < h:
        # The map values at undistorted coordinates give us the distorted coordinates
        # mapx[uy, ux] and mapy[uy, ux] tell us where this undistorted pixel came from
        dist_x = mapx[uy, ux]
        dist_y = mapy[uy, ux]
        return float(dist_x), float(dist_y)
    else:
        # Out of bounds, return original coordinates
        return undist_x, undist_y


def convert_distorted_to_undistorted_coords(dist_x: float, dist_y: float,
                                              mapx: np.ndarray, mapy: np.ndarray) -> Tuple[float, float]:
    """Convert distorted pixel coordinates to undistorted using undistortion maps.

    Args:
        dist_x, dist_y: Distorted pixel coordinates
        mapx, mapy: Undistortion maps (from calibration)

    Returns:
        Undistorted pixel coordinates (x, y)
    """
    # The maps tell us: for each undistorted pixel (uy, ux), sample from (mapx[uy,ux], mapy[uy,ux]) in distorted
    # To go distorted→undistorted, we need to find (uy, ux) such that (mapx[uy,ux], mapy[uy,ux]) ≈ (dist_x, dist_y)

    # Find the closest undistorted pixel(s) that map to our distorted coordinate
    distances = (mapx - dist_x)**2 + (mapy - dist_y)**2
    min_idx = np.argmin(distances)
    uy, ux = np.unravel_index(min_idx, distances.shape)

    return float(ux), float(uy)


def undistort_points(cameras: Dict[str, dict],
                     annotations: Dict[str, dict],
                     annotations_distorted: bool) -> Dict[str, dict]:
    """
    Undistort annotation points if they are in distorted coordinates.

    Args:
        cameras: Dictionary of camera data with intrinsics
        annotations: Dictionary of annotations (will be modified in-place)
        annotations_distorted: Whether annotations are in distorted coordinates

    Returns:
        Updated annotations dictionary with undistorted coordinates
    """
    if not annotations_distorted:
        print("  Annotations already in undistorted coordinates, no conversion needed")
        return annotations

    print("  Converting annotation points from distorted to undistorted coordinates...")

    for cam_name, ann_data in annotations.items():
        if cam_name not in cameras:
            continue

        intrinsics = cameras[cam_name]["intrinsics"]
        # Check if we have undistortion maps for accurate conversion
        mapx = intrinsics.get("undistort_mapx")
        mapy = intrinsics.get("undistort_mapy")
        use_maps = (mapx is not None and mapy is not None)

        # Get image dimensions for bounds checking
        img_width, img_height = intrinsics["size"]

        # Fallback parameters for when maps are not available
        K_original = intrinsics.get("matrix_original", intrinsics["matrix"])
        K_new = intrinsics["matrix"]
        dist = intrinsics["distortion"]
        is_fisheye = intrinsics["fisheye"]

        # Undistort known points
        if ann_data.get("known_points"):
            known_points_distorted = np.array([[cx, cy] for _, _, _, cx, cy in ann_data["known_points"]], dtype=np.float32)

            if use_maps:
                # Use undistortion maps for exact conversion (matches how images are undistorted)
                known_points_undistorted = np.array([
                    convert_distorted_to_undistorted_coords(cx, cy, mapx, mapy)
                    for cx, cy in known_points_distorted
                ])
            elif is_fisheye:
                known_points_undistorted = cv2.fisheye.undistortPoints(
                    known_points_distorted.reshape(-1, 1, 2), K_original, dist, np.eye(3), K_new
                ).reshape(-1, 2)
            else:
                known_points_undistorted = cv2.undistortPoints(
                    known_points_distorted.reshape(-1, 1, 2), K_original, dist, np.eye(3), K_new
                ).reshape(-1, 2)
            # Update with undistorted coordinates, filtering out points outside image bounds
            new_known_points = []
            n_filtered = 0
            for i, (X, Y, Z, _, _) in enumerate(ann_data["known_points"]):
                cx_undist, cy_undist = known_points_undistorted[i]
                # Check if point is within undistorted image bounds
                if 0 < cx_undist < img_width-1 and 0 < cy_undist < img_height-1:
                    new_known_points.append((X, Y, Z, cx_undist, cy_undist))
                else:
                    n_filtered += 1
                    if n_filtered <= 3:  # Only print first few warnings
                        print(f"      Warning: Filtered out point at distorted ({ann_data['known_points'][i][3]:.1f}, {ann_data['known_points'][i][4]:.1f}) "
                              f"-> undistorted ({cx_undist:.1f}, {cy_undist:.1f}) [outside image bounds]")

            ann_data["known_points"] = new_known_points
            conversion_method = "maps" if use_maps else ("fisheye" if is_fisheye else "standard")
            filter_msg = f", filtered {n_filtered}" if n_filtered > 0 else ""
            print(f"    {cam_name} ({conversion_method}): Undistorted {len(new_known_points)} known points{filter_msg}")

        # Process feature point observations (if present)
        new_feature_points = {}
        n_filtered = 0
        for feat_name, observations in ann_data.get("feature_points", {}).items():
            if not observations:
                continue

            obs_distorted = np.array(observations, dtype=np.float32)

            if use_maps:
                # Use undistortion maps for exact conversion
                obs_undistorted = np.array([
                    convert_distorted_to_undistorted_coords(cx, cy, mapx, mapy)
                    for cx, cy in obs_distorted
                ])
            elif is_fisheye:
                obs_undistorted = cv2.fisheye.undistortPoints(
                    obs_distorted.reshape(-1, 1, 2), K_original, dist, np.eye(3), K_new
                ).reshape(-1, 2)
            else:
                obs_undistorted = cv2.undistortPoints(
                    obs_distorted.reshape(-1, 1, 2), K_original, dist, np.eye(3), K_new
                ).reshape(-1, 2)

            # Filter out points outside image bounds. Undistortion projects these to edge of the frame
            # so we need to ensure there is 1 pixel margin
            valid_observations = []
            for i, (cx, cy) in enumerate(obs_undistorted):
                if 0 < cx < img_width-1 and 0 < cy < img_height-1:
                    valid_observations.append((cx, cy))
                else:
                    n_filtered += 1

            if valid_observations:
                new_feature_points[feat_name] = valid_observations

        ann_data["feature_points"] = new_feature_points
        total_feat = sum(len(obs) for obs in new_feature_points.values())
        conversion_method = "maps" if use_maps else ("fisheye" if is_fisheye else "standard")
        filter_msg = f", filtered {n_filtered}" if n_filtered > 0 else ""
        print(f"    {cam_name} ({conversion_method}): Undistorted {total_feat} feature point observations{filter_msg}")

    return annotations


def parse_via_json(json_path: str) -> Dict[str, dict]:
    """
    Parse VIA JSON annotation file and organize by camera.

    Returns:
        Dictionary mapping camera names to their annotations:
        {
            "cam1": {
                "filename": "cam1_frame_1000.png",
                "known_points": [(X, Y, Z, cx, cy), ...],  # Points with row/col
                "feature_points": {
                    "point_name": [(cx, cy), ...],
                    ...
                }
            },
            ...
        }
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        via_data = json.load(f)

    cameras_data = defaultdict(lambda: {
        "filename": None,
        "known_points": [],
        "feature_points": defaultdict(list)
    })

    img_metadata = via_data.get('_via_img_metadata', {})

    for img_key, img_data in img_metadata.items():
        filename = img_data['filename']

        # Extract camera name from filename (e.g., "cam1_frame_1000.png" -> "cam1")
        # If no underscore, use the whole filename without extension
        if '_' in filename:
            cam_name = filename.split('_')[0]
        else:
            cam_name = Path(filename).stem
        cameras_data[cam_name]["filename"] = filename

        # Parse regions (point annotations)
        for region in img_data.get('regions', []):
            shape_attrs = region.get('shape_attributes', {})
            region_attrs = region.get('region_attributes', {})

            # Skip empty regions
            if not shape_attrs or shape_attrs.get('name') != 'point':
                continue

            cx = shape_attrs.get('cx')
            cy = shape_attrs.get('cy')

            if cx is None or cy is None:
                continue

            # Check if this is a known 3D point (has row/col)
            if 'row' in region_attrs and 'col' in region_attrs and region_attrs['row'] and region_attrs['col']:
                row = region_attrs['row']
                col = region_attrs['col']

                try:
                    # Convert to float (row=Y, col=X, Z=0)
                    X = float(col)
                    Y = float(row)
                    Z = 0.0
                    cameras_data[cam_name]["known_points"].append((X, Y, Z, cx, cy))
                except (ValueError, TypeError):
                    print(f"Warning: Invalid row/col values in {filename}: row={row}, col={col}")
                    continue

            # Check if this is a feature point (has name)
            elif 'name' in region_attrs and region_attrs['name']:
                point_name = region_attrs['name']
                cameras_data[cam_name]["feature_points"][point_name].append((cx, cy))

    # Convert defaultdicts to regular dicts
    result = {}
    for cam_name, data in cameras_data.items():
        result[cam_name] = {
            "filename": data["filename"],
            "known_points": data["known_points"],
            "feature_points": dict(data["feature_points"])
        }

    print(f"Parsed annotations for {len(result)} cameras from {json_path}")

    return result


def estimate_camera_pose_pnp(object_points: np.ndarray,
                               image_points: np.ndarray,
                               camera_matrix: np.ndarray,
                               dist_coeffs: np.ndarray,
                               use_ransac: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate camera pose using PnP.

    Returns:
        rvec: Rotation vector (3,)
        tvec: Translation vector (3,)
        reprojection_error: Mean reprojection error in pixels
    """
    if len(object_points) < 6:
        raise ValueError(f"Need at least 6 points for PnP, got {len(object_points)}")

    # Check if all points are coplanar (same Z). When coplanar, PnP has two
    # mirror solutions (camera above or below the plane). Use IPPE to get both
    # and pick the one where the camera has positive Z in world coordinates.
    z_vals = object_points[:, 2]
    is_coplanar = np.all(np.abs(z_vals - z_vals[0]) < 1e-6)

    if is_coplanar:
        retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )
        if retval == 0:
            raise RuntimeError("PnP IPPE failed to find a solution")

        # Pick solution with camera center at positive Z in world space
        rvec, tvec = rvecs[0], tvecs[0]  # default to first
        for rv, tv in zip(rvecs, tvecs):
            R, _ = cv2.Rodrigues(rv)
            cam_center_z = (-R.T @ tv)[2, 0]
            if cam_center_z > 0:
                rvec, tvec = rv, tv
                break
        print(f"  PnP IPPE (coplanar): selected solution with camera Z = {(-cv2.Rodrigues(rvec)[0].T @ tvec)[2, 0]:.3f}")
    elif use_ransac:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, camera_matrix, dist_coeffs,
            iterationsCount=1000, reprojectionError=8.0, confidence=0.99
        )
        if not success:
            raise RuntimeError("PnP RANSAC failed to find a solution")

        print(f"  PnP RANSAC: {len(inliers)}/{len(object_points)} inliers")
    else:
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs
        )
        if not success:
            raise RuntimeError("PnP failed to find a solution")

    # Calculate reprojection error
    projected_points, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected_points = projected_points.reshape(-1, 2)
    errors = np.linalg.norm(projected_points - image_points, axis=1)
    mean_error = np.mean(errors)

    return rvec, tvec, mean_error


def triangulate_point_multiview(camera_matrices: List[np.ndarray],
                                  image_points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Triangulate a 3D point from multiple views using DLT.

    Args:
        camera_matrices: List of 3x4 projection matrices [K[R|t]]
        image_points: List of (x, y) pixel coordinates

    Returns:
        3D point in world coordinates (3,)
    """
    if len(camera_matrices) != len(image_points):
        raise ValueError("Number of cameras must match number of image points")

    if len(camera_matrices) < 2:
        raise ValueError("Need at least 2 views for triangulation")

    # Build the linear system A * X = 0
    A = []
    for P, (x, y) in zip(camera_matrices, image_points):
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])

    A = np.array(A)

    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    X_homogeneous = Vt[-1]

    # Convert from homogeneous to 3D coordinates
    X = X_homogeneous[:3] / X_homogeneous[3]

    return X


def triangulate_feature_points(cameras: Dict[str, dict],
                                 annotations: Dict[str, dict]) -> Dict[str, np.ndarray]:
    """
    Triangulate all feature points visible in multiple views.

    Returns:
        Dictionary mapping feature point names to 3D coordinates
    """
    # Collect all feature point names and their observations
    feature_observations = defaultdict(list)

    for cam_name, ann_data in annotations.items():
        if cam_name not in cameras or cameras[cam_name].get("rvec") is None:
            continue

        for point_name, observations in ann_data["feature_points"].items():
            for cx, cy in observations:
                feature_observations[point_name].append({
                    "camera": cam_name,
                    "pixel": (cx, cy)
                })

    # Triangulate each feature point
    triangulated_points = {}

    for point_name, observations in feature_observations.items():
        if len(observations) < 2:
            print(f"Warning: Feature point '{point_name}' visible in only {len(observations)} view(s), skipping")
            continue

        # Prepare camera matrices and image points
        camera_matrices = []
        image_points = []

        for obs in observations:
            cam_name = obs["camera"]
            cam = cameras[cam_name]

            # Build projection matrix P = K[R|t]
            K = cam["intrinsics"]["matrix"]
            rvec = cam["rvec"]
            tvec = cam["tvec"]
            R, _ = cv2.Rodrigues(rvec)
            Rt = np.hstack([R, tvec.reshape(3, 1)])
            P = K @ Rt

            camera_matrices.append(P)
            image_points.append(obs["pixel"])

        # Triangulate
        try:
            point_3d = triangulate_point_multiview(camera_matrices, image_points)

            # Validate by computing reprojection error
            errors = []
            for P, img_pt in zip(camera_matrices, image_points):
                pt_homo = np.append(point_3d, 1.0)
                projected_homo = P @ pt_homo
                projected = projected_homo[:2] / projected_homo[2]
                error = np.linalg.norm(projected - np.array(img_pt))
                errors.append(error)

            mean_error = np.mean(errors)
            max_error = np.max(errors)

            if max_error > 20.0:  # Threshold for outlier rejection
                print(f"Warning: Feature point '{point_name}' has high reprojection error ({max_error:.1f} px), skipping")
                continue

            triangulated_points[point_name] = point_3d
            print(f"  Triangulated '{point_name}': {point_3d}, reprojection error: {mean_error:.2f} ± {np.std(errors):.2f} px")

        except Exception as e:
            print(f"Warning: Failed to triangulate '{point_name}': {e}")
            continue

    return triangulated_points


def bundle_adjustment(cameras: Dict[str, dict],
                       annotations: Dict[str, dict],
                       feature_points_3d: Dict[str, np.ndarray],
                       fix_reference: bool = False,
                       reference_camera: Optional[str] = None) -> Tuple[Dict[str, dict], Dict[str, np.ndarray]]:
    """
    Perform bundle adjustment to jointly optimize camera poses and 3D feature points.

    Returns:
        Updated cameras dict and feature_points_3d dict
    """
    print("\nPerforming bundle adjustment...")

    # Prepare data structures
    camera_names = [name for name in cameras.keys() if cameras[name].get("rvec") is not None]
    feature_names = list(feature_points_3d.keys())

    # Create parameter vector
    # Format: [cam1_rvec(3), cam1_tvec(3), cam2_rvec(3), cam2_tvec(3), ..., point1(3), point2(3), ...]
    params = []

    # Add camera parameters
    cam_param_start = {}
    for i, cam_name in enumerate(camera_names):
        cam_param_start[cam_name] = len(params)
        params.extend(cameras[cam_name]["rvec"].flatten())
        params.extend(cameras[cam_name]["tvec"].flatten())

    # Add feature point parameters
    feature_param_start = {}
    for i, feat_name in enumerate(feature_names):
        feature_param_start[feat_name] = len(params)
        params.extend(feature_points_3d[feat_name])

    params = np.array(params)
    n_cameras = len(camera_names)
    n_points = len(feature_names)

    print(f"  Optimizing {n_cameras} cameras and {n_points} feature points")
    print(f"  Total parameters: {len(params)}")

    # Prepare observations
    observations = []

    # Known 3D points observations
    for cam_name in camera_names:
        ann_data = annotations.get(cam_name, {})
        known_points = ann_data.get("known_points", [])

        for X, Y, Z, cx, cy in known_points:
            observations.append({
                "type": "known",
                "camera": cam_name,
                "point_3d": np.array([X, Y, Z]),
                "pixel": np.array([cx, cy])
            })

    # Feature points observations
    for feat_name in feature_names:
        for cam_name in camera_names:
            ann_data = annotations.get(cam_name, {})
            feat_observations = ann_data.get("feature_points", {}).get(feat_name, [])

            for cx, cy in feat_observations:
                observations.append({
                    "type": "feature",
                    "camera": cam_name,
                    "feature_name": feat_name,
                    "pixel": np.array([cx, cy])
                })

    print(f"  Total observations: {len(observations)}")

    # Residual function
    def residual_fn(params):
        residuals = []

        for obs in observations:
            cam_name = obs["camera"]
            cam_idx = cam_param_start[cam_name]

            # Extract camera parameters
            rvec = params[cam_idx:cam_idx+3]
            tvec = params[cam_idx+3:cam_idx+6]

            # Get 3D point
            if obs["type"] == "known":
                point_3d = obs["point_3d"]
            else:  # feature
                feat_idx = feature_param_start[obs["feature_name"]]
                point_3d = params[feat_idx:feat_idx+3]

            # Project point
            K = cameras[cam_name]["intrinsics"]["matrix"]
            dist = np.zeros(5)  # No distortion for undistorted images

            projected, _ = cv2.projectPoints(
                point_3d.reshape(1, 3), rvec, tvec, K, dist
            )
            projected = projected.reshape(2)

            # Compute residual
            observed = obs["pixel"]
            residual = projected - observed
            residuals.extend(residual)

        return np.array(residuals)

    # Set up parameter bounds and fixed parameters
    if fix_reference and reference_camera and reference_camera in cam_param_start:
        # Fix reference camera parameters
        ref_idx = cam_param_start[reference_camera]
        lower_bounds = np.full_like(params, -np.inf)
        upper_bounds = np.full_like(params, np.inf)
        lower_bounds[ref_idx:ref_idx+6] = params[ref_idx:ref_idx+6]
        upper_bounds[ref_idx:ref_idx+6] = params[ref_idx:ref_idx+6]
        bounds = (lower_bounds, upper_bounds)
    else:
        bounds = (-np.inf, np.inf)

    # Optimize
    result = least_squares(
        residual_fn,
        params,
        bounds=bounds,
        method='trf',
        ftol=1e-6,
        xtol=1e-6,
        max_nfev=1000,
        verbose=2
    )

    print(f"\nBundle adjustment completed:")
    print(f"  Success: {result.success}")
    print(f"  Iterations: {result.nfev}")
    print(f"  Final cost: {result.cost:.6f}")
    print(f"  Initial cost: {np.sum(residual_fn(params)**2) / 2:.6f}")

    # Extract optimized parameters
    optimized_params = result.x

    # Update cameras
    for cam_name in camera_names:
        cam_idx = cam_param_start[cam_name]
        cameras[cam_name]["rvec"] = optimized_params[cam_idx:cam_idx+3].reshape(3, 1)
        cameras[cam_name]["tvec"] = optimized_params[cam_idx+3:cam_idx+6].reshape(3, 1)

    # Update feature points
    for feat_name in feature_names:
        feat_idx = feature_param_start[feat_name]
        feature_points_3d[feat_name] = optimized_params[feat_idx:feat_idx+3]

    return cameras, feature_points_3d


def compute_reprojection_errors(cameras: Dict[str, dict],
                                  annotations: Dict[str, dict],
                                  feature_points_3d: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """
    Compute reprojection errors for all cameras and points.

    Returns:
        Dictionary with per-camera statistics
    """
    camera_stats = {}

    for cam_name, cam in cameras.items():
        if cam.get("rvec") is None:
            continue

        ann_data = annotations.get(cam_name, {})
        K = cam["intrinsics"]["matrix"]
        rvec = cam["rvec"]
        tvec = cam["tvec"]
        dist = np.zeros(5)  # No distortion for undistorted images

        errors = []

        # Known points
        for X, Y, Z, cx, cy in ann_data.get("known_points", []):
            point_3d = np.array([[X, Y, Z]])
            projected, _ = cv2.projectPoints(point_3d, rvec, tvec, K, dist)
            projected = projected.reshape(2)
            error = np.linalg.norm(projected - np.array([cx, cy]))
            errors.append(("known", error))

        # Feature points
        for feat_name, observations in ann_data.get("feature_points", {}).items():
            if feat_name not in feature_points_3d:
                continue

            point_3d = feature_points_3d[feat_name].reshape(1, 3)

            for cx, cy in observations:
                projected, _ = cv2.projectPoints(point_3d, rvec, tvec, K, dist)
                projected = projected.reshape(2)
                error = np.linalg.norm(projected - np.array([cx, cy]))
                errors.append(("feature", error))

        if errors:
            error_values = [e[1] for e in errors]
            camera_stats[cam_name] = {
                "mean": np.mean(error_values),
                "std": np.std(error_values),
                "max": np.max(error_values),
                "min": np.min(error_values),
                "n_known": sum(1 for e in errors if e[0] == "known"),
                "n_feature": sum(1 for e in errors if e[0] == "feature")
            }

    return camera_stats


def save_results_toml(output_path: str, cameras: Dict[str, dict]):
    """Save calibration results in TOML format compatible with Pose2Sim."""
    with open(output_path, 'w') as f:
        for cam_name, cam in cameras.items():
            if cam.get("rvec") is None:
                print(f"Warning: Camera '{cam_name}' has no extrinsics, skipping")
                continue

            intrinsics = cam["intrinsics"]
            K = intrinsics["matrix"]
            dist = np.zeros(4)  # Extrinsics calibrated on undistorted images; distortion is zero
            rvec = cam["rvec"].flatten()
            tvec = cam["tvec"].flatten()
            size = intrinsics["size"]

            f.write(f'[{cam_name}]\n')
            f.write(f'name = "{cam_name}"\n')
            f.write(f'size = [ {size[0]}, {size[1]}]\n')
            f.write(f'matrix = [ [ {K[0,0]}, 0.0, {K[0,2]}], [ 0.0, {K[1,1]}, {K[1,2]}], [ 0.0, 0.0, 1.0]]\n')
            f.write(f'distortions = [ {dist[0]}, {dist[1]}, {dist[2]}, {dist[3]}]\n')
            f.write(f'rotation = [ {rvec[0]}, {rvec[1]}, {rvec[2]}]\n')
            f.write(f'translation = [ {tvec[0]}, {tvec[1]}, {tvec[2]}]\n')
            f.write(f'fisheye = false\n\n')

        # Calculate overall error
        overall_error = 0.0
        f.write('[metadata]\n')
        f.write('adjusted = true\n')
        f.write(f'error = {overall_error}\n')

def visualize_reprojections(project_path: Path,
                             cameras: Dict[str, dict],
                             annotations: Dict[str, dict],
                             feature_points_3d: Dict[str, np.ndarray],
                             images_distorted: bool = False,
                             undistort_visualization: bool = False):
    """
    Create visualization of reprojection errors on calibration images.

    Loads images from <project_path>/calibration/extrinsics/ext_<cam_name>_ext/
    and saves annotated versions to <project_path>/calibration/output/<cam_name>.png

    - Blue circles: Annotated points (with labels)
    - Green X: Projected points
    - Red lines: Error vectors between annotated and projected

    Args:
        project_path: Path to project directory
        cameras: Camera data with poses and intrinsics
        annotations: Point annotations (in undistorted coordinates)
        feature_points_3d: Triangulated 3D feature points
        images_distorted: Whether input images are distorted
        undistort_visualization: Whether to undistort the output visualization image
    """
    extr_dir = project_path / "calibration" / "extrinsics"
    output_dir = project_path / "calibration" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nCreating visualization images...")

    for cam_name, cam in cameras.items():
        if cam.get("rvec") is None:
            continue

        # Find the calibration image
        cam_extr_dir = extr_dir / f"ext_{cam_name}_ext"
        if not cam_extr_dir.exists():
            print(f"  Warning: No extrinsics directory found for '{cam_name}' at {cam_extr_dir}")
            continue

        # Look for PNG files in the directory
        # If images_distorted is True, look in no_undistort subfolder first
        if images_distorted:
            no_undistort_dir = cam_extr_dir / "no_undistort"
            if no_undistort_dir.exists():
                png_files = list(no_undistort_dir.glob("*.png"))
                if png_files:
                    print(f"  Using distorted images from {no_undistort_dir}")

        # If not found or images_distorted is False, use regular directory
        if not images_distorted or not png_files:
            png_files = list(cam_extr_dir.glob("*.png"))

        if not png_files:
            print(f"  Warning: No PNG files found for '{cam_name}' in {cam_extr_dir}")
            continue

        # Use the first PNG file found
        img_path = png_files[0]
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"  Warning: Failed to load image for '{cam_name}' from {img_path}")
            continue

        print(f"  Visualizing '{cam_name}' from {img_path.absolute()}")

        # Undistort the visualization image if requested
        if images_distorted and undistort_visualization:
            # Use the original camera matrix and the new camera matrix from calibration
            K_original = cam["intrinsics"].get("matrix_original", cam["intrinsics"]["matrix"])
            K_new = cam["intrinsics"]["matrix"]
            dist = cam["intrinsics"]["distortion"]
            is_fisheye = cam["intrinsics"]["fisheye"]

            if is_fisheye:
                # Use fisheye undistortion
                h, w = img.shape[:2]
                mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
                    K_original, dist, np.eye(3), K_new, (w, h), cv2.CV_32FC1
                )
                img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            else:
                # Use standard undistortion
                img = cv2.undistort(img, K_original, dist, None, K_new)
            print(f"    Undistorted visualization image")

        # Get camera parameters
        K_new = cam["intrinsics"]["matrix"]  # New camera matrix for undistorted space
        K_original = cam["intrinsics"].get("matrix_original", K_new)  # Original camera matrix
        dist = cam["intrinsics"]["distortion"]
        rvec = cam["rvec"]
        tvec = cam["tvec"]
        is_fisheye = cam["intrinsics"]["fisheye"]
        mapx = cam["intrinsics"].get("undistort_mapx")
        mapy = cam["intrinsics"].get("undistort_mapy")
        ann_data = annotations.get(cam_name, {})

        # Check if we have undistortion maps for coordinate conversion
        if images_distorted and not undistort_visualization and (mapx is None or mapy is None):
            print(f"    Warning: No undistortion maps available for '{cam_name}', coordinate conversion may be inaccurate")
            # Fall back to manual conversion (less accurate)
            use_maps = False
        else:
            use_maps = (mapx is not None and mapy is not None)

        # Draw known points
        for X, Y, Z, cx, cy in ann_data.get("known_points", []):
            # Annotations are in undistorted K_new space
            # Convert to distorted K_original space if needed for drawing
            if images_distorted and not undistort_visualization:
                if use_maps:
                    # Use undistortion maps for accurate conversion
                    cx_draw, cy_draw = convert_undistorted_to_distorted_coords(cx, cy, mapx, mapy)
                else:
                    # Fall back to manual conversion (less accurate)
                    # Normalize with K_new to get undistorted normalized coordinates
                    x_n_undist = (cx - K_new[0, 2]) / K_new[0, 0]
                    y_n_undist = (cy - K_new[1, 2]) / K_new[1, 1]

                    if is_fisheye:
                        # For fisheye: use distortPoints which expects normalized undistorted coords
                        pts_norm_undist = np.array([[[x_n_undist, y_n_undist]]], dtype=np.float32)
                        pts_norm_dist = cv2.fisheye.distortPoints(pts_norm_undist, K_original, dist)
                        x_n_dist, y_n_dist = pts_norm_dist[0, 0]
                    else:
                        # For standard: manually apply distortion formula
                        r2 = x_n_undist*x_n_undist + y_n_undist*y_n_undist
                        k1, k2, p1, p2 = dist.flatten()[:4]
                        k3 = dist.flatten()[4] if len(dist.flatten()) > 4 else 0
                        radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
                        x_n_dist = x_n_undist*radial + 2*p1*x_n_undist*y_n_undist + p2*(r2 + 2*x_n_undist*x_n_undist)
                        y_n_dist = y_n_undist*radial + p1*(r2 + 2*y_n_undist*y_n_undist) + 2*p2*x_n_undist*y_n_undist

                    # Denormalize with K_original to get distorted pixel coordinates
                    cx_draw = x_n_dist * K_original[0, 0] + K_original[0, 2]
                    cy_draw = y_n_dist * K_original[1, 1] + K_original[1, 2]
            else:
                cx_draw, cy_draw = cx, cy

            # Draw annotated point (blue circle)
            cv2.circle(img, (int(cx_draw), int(cy_draw)), 5, (255, 0, 0), 2)

            # Project the 3D point in undistorted K_new space
            point_3d = np.array([[X, Y, Z]])
            projected, _ = cv2.projectPoints(point_3d, rvec, tvec, K_new, np.zeros(5))
            proj_x_undist, proj_y_undist = projected.reshape(2)

            # Convert projected point from undistorted to distorted space if needed
            if images_distorted and not undistort_visualization:
                if use_maps:
                    # Use undistortion maps for accurate conversion
                    proj_x, proj_y = convert_undistorted_to_distorted_coords(proj_x_undist, proj_y_undist, mapx, mapy)
                else:
                    # Fall back to manual conversion (less accurate)
                    # Normalize with K_new
                    x_n_undist = (proj_x_undist - K_new[0, 2]) / K_new[0, 0]
                    y_n_undist = (proj_y_undist - K_new[1, 2]) / K_new[1, 1]

                    if is_fisheye:
                        pts_norm_undist = np.array([[[x_n_undist, y_n_undist]]], dtype=np.float32)
                        pts_norm_dist = cv2.fisheye.distortPoints(pts_norm_undist, K_original, dist)
                        x_n_dist, y_n_dist = pts_norm_dist[0, 0]
                    else:
                        r2 = x_n_undist*x_n_undist + y_n_undist*y_n_undist
                        k1, k2, p1, p2 = dist.flatten()[:4]
                        k3 = dist.flatten()[4] if len(dist.flatten()) > 4 else 0
                        radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
                        x_n_dist = x_n_undist*radial + 2*p1*x_n_undist*y_n_undist + p2*(r2 + 2*x_n_undist*x_n_undist)
                        y_n_dist = y_n_undist*radial + p1*(r2 + 2*y_n_undist*y_n_undist) + 2*p2*x_n_undist*y_n_undist

                    # Denormalize with K_original
                    proj_x = x_n_dist * K_original[0, 0] + K_original[0, 2]
                    proj_y = y_n_dist * K_original[1, 1] + K_original[1, 2]
            else:
                proj_x, proj_y = proj_x_undist, proj_y_undist

            # Draw projected point (green X)
            cv2.drawMarker(img, (int(proj_x), int(proj_y)), (0, 255, 0),
                          cv2.MARKER_TILTED_CROSS, 10, 2)  # Green X

            # Draw error line (red)
            cv2.line(img, (int(cx_draw), int(cy_draw)), (int(proj_x), int(proj_y)),
                    (0, 0, 255), 1)  # Red line

            # Add label
            label = f"({int(Y)},{int(X)})"
            cv2.putText(img, label, (int(cx_draw) + 8, int(cy_draw) - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Draw feature points
        for feat_name, observations in ann_data.get("feature_points", {}).items():
            if feat_name not in feature_points_3d:
                continue

            point_3d = feature_points_3d[feat_name].reshape(1, 3)

            for cx, cy in observations:
                # Annotations are in undistorted K_new space
                # Convert to distorted K_original space if needed for drawing
                if images_distorted and not undistort_visualization:
                    if use_maps:
                        # Use undistortion maps for accurate conversion
                        cx_draw, cy_draw = convert_undistorted_to_distorted_coords(cx, cy, mapx, mapy)
                    else:
                        # Fall back to manual conversion (less accurate)
                        # Normalize with K_new
                        x_n_undist = (cx - K_new[0, 2]) / K_new[0, 0]
                        y_n_undist = (cy - K_new[1, 2]) / K_new[1, 1]

                        if is_fisheye:
                            pts_norm_undist = np.array([[[x_n_undist, y_n_undist]]], dtype=np.float32)
                            pts_norm_dist = cv2.fisheye.distortPoints(pts_norm_undist, K_original, dist)
                            x_n_dist, y_n_dist = pts_norm_dist[0, 0]
                        else:
                            r2 = x_n_undist*x_n_undist + y_n_undist*y_n_undist
                            k1, k2, p1, p2 = dist.flatten()[:4]
                            k3 = dist.flatten()[4] if len(dist.flatten()) > 4 else 0
                            radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
                            x_n_dist = x_n_undist*radial + 2*p1*x_n_undist*y_n_undist + p2*(r2 + 2*x_n_undist*x_n_undist)
                            y_n_dist = y_n_undist*radial + p1*(r2 + 2*y_n_undist*y_n_undist) + 2*p2*x_n_undist*y_n_undist

                        # Denormalize with K_original
                        cx_draw = x_n_dist * K_original[0, 0] + K_original[0, 2]
                        cy_draw = y_n_dist * K_original[1, 1] + K_original[1, 2]
                else:
                    cx_draw, cy_draw = cx, cy

                # Draw annotated point (blue circle)
                cv2.circle(img, (int(cx_draw), int(cy_draw)), 5, (255, 0, 0), 2)

                # Project the 3D point in undistorted K_new space
                projected, _ = cv2.projectPoints(point_3d, rvec, tvec, K_new, np.zeros(5))
                proj_x_undist, proj_y_undist = projected.reshape(2)

                # Convert projected point from undistorted to distorted space if needed
                if images_distorted and not undistort_visualization:
                    if use_maps:
                        # Use undistortion maps for accurate conversion
                        proj_x, proj_y = convert_undistorted_to_distorted_coords(proj_x_undist, proj_y_undist, mapx, mapy)
                    else:
                        # Fall back to manual conversion (less accurate)
                        # Normalize with K_new
                        x_n_undist = (proj_x_undist - K_new[0, 2]) / K_new[0, 0]
                        y_n_undist = (proj_y_undist - K_new[1, 2]) / K_new[1, 1]

                        if is_fisheye:
                            pts_norm_undist = np.array([[[x_n_undist, y_n_undist]]], dtype=np.float32)
                            pts_norm_dist = cv2.fisheye.distortPoints(pts_norm_undist, K_original, dist)
                            x_n_dist, y_n_dist = pts_norm_dist[0, 0]
                        else:
                            r2 = x_n_undist*x_n_undist + y_n_undist*y_n_undist
                            k1, k2, p1, p2 = dist.flatten()[:4]
                            k3 = dist.flatten()[4] if len(dist.flatten()) > 4 else 0
                            radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
                            x_n_dist = x_n_undist*radial + 2*p1*x_n_undist*y_n_undist + p2*(r2 + 2*x_n_undist*x_n_undist)
                            y_n_dist = y_n_undist*radial + p1*(r2 + 2*y_n_undist*y_n_undist) + 2*p2*x_n_undist*y_n_undist

                        # Denormalize with K_original
                        proj_x = x_n_dist * K_original[0, 0] + K_original[0, 2]
                        proj_y = y_n_dist * K_original[1, 1] + K_original[1, 2]
                else:
                    proj_x, proj_y = proj_x_undist, proj_y_undist

                # Draw projected point (green X)
                cv2.drawMarker(img, (int(proj_x), int(proj_y)), (0, 255, 0),
                              cv2.MARKER_TILTED_CROSS, 10, 2)  # Green X

                # Draw error line (red)
                cv2.line(img, (int(cx_draw), int(cy_draw)), (int(proj_x), int(proj_y)),
                        (0, 0, 255), 1)  # Red line

                # Add label
                cv2.putText(img, feat_name, (int(cx_draw) + 8, int(cy_draw) - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Save the annotated image
        output_path = output_dir / f"{cam_name}.png"
        cv2.imwrite(str(output_path), img)
        print(f"    Saved visualization to {output_path}")

def generate_report(output_path: str,
                     cameras: Dict[str, dict],
                     camera_stats: Dict[str, dict],
                     feature_points_3d: Dict[str, np.ndarray],
                     annotations: Dict[str, dict]):
    """Generate a detailed calibration report."""
    with open(output_path, 'w') as f:
        f.write("Camera Extrinsics Calibration Report\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Cameras: {len(cameras)}\n")
        f.write(f"Feature points: {len(feature_points_3d)}\n")

        total_known = sum(len(ann.get("known_points", [])) for ann in annotations.values())
        total_features = sum(len(ann.get("feature_points", {})) for ann in annotations.values())
        f.write(f"Known 3D points: {total_known}\n")
        f.write(f"Feature point observations: {total_features}\n\n")

        f.write("Per-Camera Results:\n")
        f.write("-" * 50 + "\n")

        all_errors = []
        for cam_name, cam in cameras.items():
            if cam.get("rvec") is None:
                continue

            rvec = cam["rvec"].flatten()
            tvec = cam["tvec"].flatten()

            # Convert rotation vector to degrees
            angle = np.linalg.norm(rvec)
            axis = rvec / angle if angle > 0 else np.array([0, 0, 1])
            angle_deg = np.degrees(angle)

            f.write(f"\n{cam_name}:\n")
            f.write(f"  Rotation (axis-angle): [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}] × {angle_deg:.1f}°\n")
            f.write(f"  Rotation (vector): [{rvec[0]:.4f}, {rvec[1]:.4f}, {rvec[2]:.4f}]\n")
            f.write(f"  Translation: [{tvec[0]:.4f}, {tvec[1]:.4f}, {tvec[2]:.4f}]\n")

            if cam_name in camera_stats:
                stats = camera_stats[cam_name]
                f.write(f"  Reprojection error: {stats['mean']:.2f} ± {stats['std']:.2f} px ")
                f.write(f"(min: {stats['min']:.2f}, max: {stats['max']:.2f})\n")
                f.write(f"  Points used: {stats['n_known']} known, {stats['n_feature']} features\n")
                all_errors.append(stats['mean'])

        f.write("\n\nFeature Point 3D Coordinates:\n")
        f.write("-" * 50 + "\n")
        for feat_name, point_3d in sorted(feature_points_3d.items()):
            f.write(f"{feat_name}: [{point_3d[0]:.4f}, {point_3d[1]:.4f}, {point_3d[2]:.4f}]\n")

        if all_errors:
            f.write("\n\nOverall Statistics:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Mean reprojection error: {np.mean(all_errors):.2f} px\n")
            f.write(f"Std reprojection error: {np.std(all_errors):.2f} px\n")
            f.write(f"Max reprojection error: {np.max(all_errors):.2f} px\n")


@click.command()
@click.option("--config", required=True, help="Path to project YAML configuration file")
@click.option("--annotations", required=True, help="Path to VIA JSON annotation file")
@click.option("--registry", default=None, help="Path to posetrak registry DB (required when calib.intrinsics is a UUID)")
@click.option("--output-toml", help="Path to output TOML file (default: <project_path>/calibration/intrinsics.toml)")
@click.option("--output-report", help="Path to validation report (default: <project_path>/calibration/extrinsics_report.txt)")
@click.option("--bundle-adjust", is_flag=True, help="Enable bundle adjustment")
@click.option("--reprojection-threshold", default=5.0, help="Maximum reprojection error threshold in pixels")
@click.option("--fix-reference", is_flag=True, help="Keep reference camera at origin")
@click.option("--visualize", is_flag=True, help="Generate visualization images with reprojection errors")
@click.option("--annotations-distorted", is_flag=True, help="Annotation points are in distorted (original) image coordinates")
@click.option("--images-distorted", is_flag=True, help="Calibration images are distorted (original, not undistorted)")
@click.option("--undistort-visualization", is_flag=True, help="Undistort the visualization output images (only used with --images-distorted)")
def main(config, annotations, registry, output_toml, output_report, bundle_adjust, reprojection_threshold, fix_reference, visualize, annotations_distorted, images_distorted, undistort_visualization):
    """
    Estimate camera extrinsics from annotated point correspondences.
    """
    print("=" * 60)
    print("Camera Extrinsics Calibration")
    print("=" * 60)

    # Load project configuration
    print("\n[1/6] Loading project configuration...")
    project_config = load_project_config(config)
    project_path = Path(project_config["path"])

    # Set default output paths if not specified
    if output_toml is None:
        output_toml = project_path / "calibration" / "intrinsics.toml"
    if output_report is None:
        output_report = project_path / "calibration" / "extrinsics_report.txt"

    output_toml = Path(output_toml)
    output_report = Path(output_report)

    # Load intrinsics for each camera
    print("\n[2/6] Loading camera intrinsics...")
    cameras = {}
    for cam_config in project_config["cameras"]:
        cam_name = cam_config["name"]

        if "calib" not in cam_config or "intrinsics" not in cam_config["calib"]:
            print(f"  Warning: Camera '{cam_name}' has no intrinsics calibration, skipping")
            continue

        intrinsics_path = cam_config["calib"]["intrinsics"]

        if isinstance(intrinsics_path, str):
            intrinsics = load_intrinsics(intrinsics_path, registry_path=registry)
            cameras[cam_name] = {
                "intrinsics": intrinsics,
                "config": cam_config
            }
            print(f"  Loaded intrinsics for '{cam_name}' from {intrinsics_path}")
        else:
            print(f"  Warning: Camera '{cam_name}' intrinsics is not a string, skipping")

    # Parse VIA JSON annotations
    print("\n[3/6] Parsing annotation file...")
    annotations = parse_via_json(annotations)

    for cam_name, ann_data in annotations.items():
        n_known = len(ann_data["known_points"])
        n_features = len(ann_data["feature_points"])
        print(f"  {cam_name}: {n_known} known points, {n_features} feature points")

    # Undistort annotation points if they are in distorted coordinates
    if annotations_distorted:
        print("\n[3b/6] Converting annotations from distorted to undistorted coordinates...")
        annotations = undistort_points(cameras, annotations, annotations_distorted)

    # Estimate initial camera poses using PnP
    print("\n[4/6] Estimating initial camera poses via PnP...")
    reference_camera = project_config.get("ref_camera")

    for cam_name in cameras.keys():
        if cam_name not in annotations:
            print(f"  Warning: No annotations for camera '{cam_name}', skipping")
            continue

        ann_data = annotations[cam_name]
        known_points = ann_data["known_points"]

        if len(known_points) < 6:
            print(f"  Warning: Camera '{cam_name}' has only {len(known_points)} known points (need ≥6), skipping")
            continue

        # Prepare data for PnP
        object_points = np.array([[X, Y, Z] for X, Y, Z, _, _ in known_points], dtype=np.float64)
        image_points = np.array([[cx, cy] for _, _, _, cx, cy in known_points], dtype=np.float64)

        K = cameras[cam_name]["intrinsics"]["matrix"]
        dist = np.zeros(5)  # Undistorted images, no distortion

        print(f"  Estimating pose for '{cam_name}' with {len(known_points)} points...")

        try:
            rvec, tvec, error = estimate_camera_pose_pnp(
                object_points, image_points, K, dist, use_ransac=False
            )

            cameras[cam_name]["rvec"] = rvec
            cameras[cam_name]["tvec"] = tvec

            print(f"    Rotation: {rvec.flatten()}")
            print(f"    Translation: {tvec.flatten()}")
            print(f"    Reprojection error: {error:.2f} px")

            if error > reprojection_threshold:
                print(f"    WARNING: High reprojection error!")

        except Exception as e:
            print(f"    ERROR: {e}")

    # Triangulate feature points
    print("\n[5/6] Triangulating feature points...")
    feature_points_3d = triangulate_feature_points(cameras, annotations)
    print(f"  Successfully triangulated {len(feature_points_3d)} feature points")

    # Bundle adjustment (optional)
    if bundle_adjust:
        print("\n[6/6] Bundle adjustment...")
        cameras, feature_points_3d = bundle_adjustment(
            cameras, annotations, feature_points_3d,
            fix_reference=fix_reference,
            reference_camera=reference_camera
        )
    else:
        print("\n[6/6] Skipping bundle adjustment")

    # Compute final reprojection errors
    print("\nComputing final reprojection errors...")
    camera_stats = compute_reprojection_errors(cameras, annotations, feature_points_3d)

    # Generate visualization (optional)
    if visualize:
        visualize_reprojections(project_path, cameras, annotations, feature_points_3d,
                                images_distorted=images_distorted,
                                undistort_visualization=undistort_visualization)

    # Save results
    print(f"\nSaving results to {output_toml}...")
    output_toml.parent.mkdir(parents=True, exist_ok=True)
    save_results_toml(str(output_toml), cameras)

    print(f"Generating report to {output_report}...")
    output_report.parent.mkdir(parents=True, exist_ok=True)
    generate_report(str(output_report), cameras, camera_stats, feature_points_3d, annotations)

    print("\n" + "=" * 60)
    print("Calibration complete!")
    print("=" * 60)

    # Print summary
    print("\nSummary:")
    for cam_name, stats in camera_stats.items():
        print(f"  {cam_name}: {stats['mean']:.2f} ± {stats['std']:.2f} px "
              f"({stats['n_known']} known, {stats['n_feature']} features)")

    if camera_stats:
        all_means = [s['mean'] for s in camera_stats.values()]
        print(f"\nOverall mean reprojection error: {np.mean(all_means):.2f} px")


if __name__ == "__main__":
    main()
