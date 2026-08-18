# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import zlib
from pathlib import Path

import click
import cv2
import h5py
import numpy as np
import yaml

def get_fps(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def parse_timecode(tc, fps):
    """
    Parse timecode string (hh:mm:ss.ff or mm:ss.ff or ss.ff) to frame number.
    """
    if isinstance(tc, int):
        return tc
    if isinstance(tc, float):
        return int(round(tc))
    if isinstance(tc, str):
        # Remove whitespace
        tc = tc.strip()
        # Match hh:mm:ss.ff or mm:ss.ff or ss.ff
        m = re.match(r'^(?:(\d+):)?(?:(\d+):)?(\d+)(?:[.,](\d+))?$', tc)
        if not m:
            raise ValueError(f"Invalid timecode format: {tc}")
        h = int(m.group(1)) if m.group(2) else 0
        m_ = int(m.group(2)) if m.group(2) else (int(m.group(1)) if m.group(1) and not m.group(2) else 0)
        s = int(m.group(3))
        f = int(m.group(4)) if m.group(4) else 0
        print(f"Parsed timecode {tc} to h:{h}, m:{m_}, s:{s}, f:{f}")
        total_seconds = h * 3600 + m_ * 60 + s + f / (10 ** len(m.group(4))) if m.group(4) else 0
        if m.group(4):
            total_seconds = h * 3600 + m_ * 60 + s + float("0." + m.group(4))
        else:
            total_seconds = h * 3600 + m_ * 60 + s
        return int(round(total_seconds * fps))
    raise ValueError(f"Invalid frame/timecode value: {tc}")

def time_diff(frame1:int, frame2:int, fps:float) -> float:
    """ Return time difference in seconds between two frame indices.

    Args:
        frame1: First frame index.
        frame2: Second frame index.
        fps: Frames per second.
    """
    return (frame2 - frame1) / fps

def extract_video_clip(src_path, dst_path, start_frame, end_frame, mapx=None, mapy=None):
    """ Extract a video clip from src_path from start_frame to end_frame.
        Optionally apply undistortion using mapx and mapy.  """
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {src_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Extracting video from {start_frame} to {end_frame} at {fps} FPS")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(dst_path), fourcc, fps, (int(width), int(height)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if mapx is not None and mapy is not None:
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        out.write(frame)
    out.release()
    cap.release()

def save_frame_as_png(video_path, frame_idx, out_path, mapx=None, mapy=None):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if mapx is not None and mapy is not None:
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    if ret:
        cv2.imwrite(str(out_path), frame)
    cap.release()

def find_sharp_frames(video_path, window=10, threshold=0.8):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    laplacians = []
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        laplacians.append(laplacian)
        n+= 1
        if n % 100 == 0:
            print(f"Processed {n} frames from {video_path}")

    cap.release()

    metric = (np.array(laplacians) - np.mean(laplacians)) / np.std(laplacians)

    maxima = []
    for i in range(window, metric.shape[0] - window):
        window_slice = metric[i-window:i+window+1]
        if metric[i] == np.max(window_slice) and metric[i] > threshold:
            # Ensure it's strictly greater than all neighbors
            if np.sum(window_slice == metric[i]) == 1:
                maxima.append(i)
    return maxima

def toml_write(calib_path: Path, cams):
    with open(os.path.join(calib_path), 'w+') as cal_f:
        for name, cam in cams.items():
            # Skip cameras without calibration data
            if "calib" not in cam or "calibration" not in cam["calib"]:
                continue

            calib = cam["calib"]["calibration"]
            cam_str = f'[{name}]\n'
            name_str = f'name = "{name}"\n'
            size = f'size = [ {calib["size"][0]}, {calib["size"][1]}]\n'
            mat = f'matrix = [ [ {calib["matrix"][0,0]}, 0.0, {calib["matrix"][0,2]}], [ 0.0, {calib["matrix"][1,1]}, {calib["matrix"][1,2]}], [ 0.0, 0.0, 1.0]]\n'
            dist = f'distortions = [ {calib["distortion"][0][0]}, {calib["distortion"][0][1]}, {calib["distortion"][0][2]}, {calib["distortion"][0][3]}]\n'

            # Check if we have valid extrinsics (rotation and translation)
            has_extrinsics = False
            if calib["rotation"] and len(calib["rotation"]) > 0:
                rot_vals = calib["rotation"][0] if isinstance(calib["rotation"][0], np.ndarray) else calib["rotation"]
                if isinstance(rot_vals, np.ndarray) and len(rot_vals) >= 3:
                    # Check if it's not just zeros
                    if not np.allclose(rot_vals[:3], [0.0, 0.0, 0.0]):
                        has_extrinsics = True
                        rot = f'rotation = [ {rot_vals[0]}, {rot_vals[1]}, {rot_vals[2]}]\n'

            if has_extrinsics and calib["translation"] and len(calib["translation"]) > 0:
                trans_vals = calib["translation"][0] if isinstance(calib["translation"][0], np.ndarray) else calib["translation"]
                if isinstance(trans_vals, np.ndarray) and len(trans_vals) >= 3:
                    tran = f'translation = [ {trans_vals[0]}, {trans_vals[1]}, {trans_vals[2]}]\n'

            fish = f'fisheye = false\n\n'

            # Write to file - only include rotation/translation if we have valid extrinsics
            if has_extrinsics:
                cal_f.write(cam_str + name_str + size + mat + dist + rot + tran + fish)
            else:
                cal_f.write(cam_str + name_str + size + mat + dist + fish)

        meta = '[metadata]\nadjusted = false\nerror = 0.0\n'
        cal_f.write(meta)

def extract_checkerboard_frames(video_path, img_path, rows, cols, window=10, threshold=0.8, calibrate=False):
    """
    Find frames in a video that are sharp enough to be used for calibration.
    Uses a Laplacian variance metric to determine sharpness.
    """
    sharp_frames = find_sharp_frames(video_path, window, threshold)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)

    mapx, mapy = None, None

    imgpoints = []
    objpoints = []
    checkerboard_frames = []
    for frame_idx in sharp_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_idx} from {video_path}")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if found:
            print(f"Found checkerboard in frame {frame_idx} of {video_path}")
            if calibrate:
                # Refine corners for better accuracy
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners)
                objpoints.append(objp)
                checkerboard_frames.append((frame_idx, frame))
            else:
                cv2.imwrite(str(img_path / f"frame_{frame_idx:04d}.png"), frame)


    if calibrate:
        print(f"Calibrating camera")
        print("Image points:")
        np.savez(img_path / "imgpoints.npz", *imgpoints)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=(cv2.CALIB_RATIONAL_MODEL))
        newcameramat, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, gray.shape[::-1], 0, gray.shape[::-1])
        print(f"Calibration successful with {len(checkerboard_frames)} frames, error {np.around(ret, decimals=3)} px.")
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramat, gray.shape[::-1], cv2.CV_32FC1)
        imgpoints = []
        objpoints = []
        print(f"Undistorting and saving checkerboard frames to {img_path}")
        for idx, frame in checkerboard_frames:
            undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners)
                objpoints.append(objp)
                cv2.imwrite(str(img_path / f"calib_frame_{idx:04d}.png"), undistorted_frame)
            else:
                print(f"Checkerboard not found after undistortion in frame {idx} of {video_path}")
        # Recalibrate with undistorted points to get rvecs and tvecs
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=(cv2.CALIB_RATIONAL_MODEL))


        calib = {
            "size": undistorted_frame.shape[:2],
            "matrix": newcameramat,
            "distortion": dist,
            "rotation": rvecs,
            "translation": tvecs,
            "error": ret
        }
        if len(checkerboard_frames) < 10:
            print(f"WARNING!!! Not enough checkerboard frames found for calibration in {video_path}. Found {len(checkerboard_frames)} frames.")

    else:
        calib = None

    cap.release()

    return mapx, mapy, calib

def load_intrinsics_from_yaml(yaml_path):
    """
    Load camera intrinsics calibration from a YAML file.

    Args:
        yaml_path: Path to YAML file containing calibration data

    Returns:
        tuple: (mapx, mapy, calib) where calib contains size, matrix, distortion
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    intrinsics = data.get("intrinsics", {})
    size = tuple(int(x) for x in intrinsics["size"])  # Convert to integers
    matrix = np.array(intrinsics["matrix"], dtype=np.float64)
    distortions = np.array([intrinsics["distortions"]], dtype=np.float64)

    # Create undistortion maps
    newcameramat, roi = cv2.getOptimalNewCameraMatrix(matrix, distortions, size, 0, size)
    mapx, mapy = cv2.initUndistortRectifyMap(matrix, distortions, None, newcameramat, size, cv2.CV_32FC1)

    calib = {
        "size": size,
        "matrix": newcameramat,
        "distortion": distortions,
        "rotation": [np.array([0.0, 0.0, 0.0])],  # Initialize as list with single array
        "translation": [np.array([0.0, 0.0, 0.0])],  # Initialize as list with single array
        "error": 0.0
    }

    print(f"Loaded intrinsics calibration from {yaml_path}")
    print(f"  Size: {size}")
    print(f"  Matrix diagonal: [{matrix[0,0]:.2f}, {matrix[1,1]:.2f}]")
    print(f"  Distortions: {distortions.flatten()}")

    return mapx, mapy, calib


def load_intrinsics_from_h5(h5_path):
    """
    Load camera intrinsics calibration from an HDF5 file.

    Args:
        h5_path: Path to HDF5 file containing calibration data

    Returns:
        tuple: (mapx, mapy, calib) where calib contains size, matrix, distortion
    """
    with h5py.File(h5_path, 'r') as f:
        # Check if file has the expected structure
        if 'intrinsics' not in f:
            raise ValueError(f"HDF5 file {h5_path} does not contain 'intrinsics' group")


        # Read intrinsics data

        # Load undistortion maps if available
        if 'undistortion_maps' in f and "calibration_undistorted" in f:
            mapx = f['undistortion_maps/mapx'][:]
            mapy = f['undistortion_maps/mapy'][:]
            intrinsics = f['calibration_undistorted']
            size = tuple(intrinsics.attrs['size'])
            matrix = intrinsics['matrix'][:]
            distortions = intrinsics['distortions'][:]
            error = intrinsics.attrs.get('error', 0.0)
            print(f"Loaded pre-computed undistortion maps from {h5_path}")
        else:
            # Generate undistortion maps if not stored
            print(f"Generating undistortion maps from intrinsics in {h5_path}")
            intrinsics = f['intrinsics']
            size = tuple(intrinsics.attrs['size'])
            matrix = intrinsics['matrix'][:]
            distortions = intrinsics['distortions'][:]
            error = intrinsics.attrs.get('error', 0.0)
            newcameramat, roi = cv2.getOptimalNewCameraMatrix(
                matrix, distortions.reshape(1, -1), size, 0, size
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                matrix, distortions.reshape(1, -1), None, newcameramat, size, cv2.CV_32FC1
            )

        calib = {
            "size": size,
            "matrix": matrix,
            "distortion": distortions.reshape(1, -1),
            "rotation": [np.array([0.0, 0.0, 0.0])],
            "translation": [np.array([0.0, 0.0, 0.0])],
            "error": error
        }

        print(f"Loaded intrinsics calibration from {h5_path}")
        print(f"  Size: {size}")
        print(f"  Matrix diagonal: [{matrix[0,0]:.2f}, {matrix[1,1]:.2f}]")
        print(f"  Distortions: {distortions.flatten()}")
        print(f"  Error: {error:.4f} pixels")

        return mapx, mapy, calib


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


def _looks_like_uuid(s: str) -> bool:
    return bool(_UUID_RE.match(s))


def load_intrinsics_from_db(registry_path: str | Path, intrinsics_id: str):
    """Load undistortion maps and camera matrix from the posetrak registry DB.

    Args:
        registry_path: Path to the registry SQLite database.
        intrinsics_id: ``intrinsics_calibrations.id`` (full UUID).

    Returns:
        tuple: (mapx, mapy, calib) matching the format of the other loaders,
               or (None, None, calib) if maps are not stored.
    """
    import sqlite3

    conn = sqlite3.connect(str(registry_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM intrinsics_calibrations WHERE id = ?", (intrinsics_id,)
    ).fetchone()
    conn.close()

    if row is None:
        raise ValueError(f"intrinsics_calibration {intrinsics_id!r} not found in {registry_path}")

    image_width = row["image_width"]
    image_height = row["image_height"]
    fx, fy, cx, cy = row["fx"], row["fy"], row["cx"], row["cy"]
    matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    dist_blob = bytes(row["dist_coeffs"]) if row["dist_coeffs"] else None
    if dist_blob:
        n = len(dist_blob) // 8
        dist = np.frombuffer(dist_blob, dtype=np.float64).reshape(1, n)
    else:
        dist = np.zeros((1, 4), dtype=np.float64)

    mapx, mapy = None, None
    if row["undistort_mapx"] and image_width and image_height:
        mapx_bytes = zlib.decompress(bytes(row["undistort_mapx"]))
        mapy_bytes = zlib.decompress(bytes(row["undistort_mapy"]))
        mapx = np.frombuffer(mapx_bytes, dtype=np.float32).reshape(image_height, image_width)
        mapy = np.frombuffer(mapy_bytes, dtype=np.float32).reshape(image_height, image_width)
        size = (image_width, image_height)
    elif image_width and image_height:
        # Maps not stored — generate them on the fly
        size = (image_width, image_height)
        newcameramat, _ = cv2.getOptimalNewCameraMatrix(matrix, dist, size, 0, size)
        mapx, mapy = cv2.initUndistortRectifyMap(matrix, dist, None, newcameramat, size, cv2.CV_32FC1)
        matrix = newcameramat
    else:
        size = (0, 0)

    calib = {
        "size": size,
        "matrix": matrix,
        "distortion": dist,
        "rotation": [np.array([0.0, 0.0, 0.0])],
        "translation": [np.array([0.0, 0.0, 0.0])],
        "error": row["rms_error"] or 0.0,
    }

    print(f"Loaded intrinsics from DB: {intrinsics_id}")
    print(f"  Size: {size}  fx={fx:.2f} fy={fy:.2f}  rms={row['rms_error']}")

    return mapx, mapy, calib


@click.command()
@click.argument("yaml_path")
@click.option("--scene", multiple=True, help="Process only specified scene(s). Can be used multiple times. If not specified, all scenes are processed.")
@click.option("--registry", default=None, metavar="DB_PATH",
              help="Path to the posetrak registry DB. Required when a camera's "
                   "calib.intrinsics is an intrinsics_calibration_id UUID.")
def main(yaml_path, scene, registry):
    # Load YAML
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    project_path = Path(config["path"])
    project_path.mkdir(parents=True, exist_ok=True)

    # Prepare calibration dirs
    calib_dir = project_path / "calibration"
    extr_dir = calib_dir / "extrinsics"
    extr_dir.mkdir(parents=True, exist_ok=True)
    intr_dir = calib_dir / "intrinsics"

    # Prepare cameras dict with FPS and parsed sync_frame
    cameras = {}
    for cam in config["cameras"]:
        if "fps" in cam and cam["fps"] is not None:
            fps = float(cam["fps"])
        else:
            fps = get_fps(cam["path"])
        print(f"Camera {cam['name']} FPS: {fps}")
        sync_frame = parse_timecode(cam["sync_frame"], fps)
        cam_copy = cam.copy()
        cam_copy["fps"] = fps
        cam_copy["sync_frame"] = sync_frame
        cameras[cam["name"]] = cam_copy

    ref_camera = cameras[config["ref_camera"]]
    sync_ref = ref_camera["sync_frame"]
    ref_camera_fps = ref_camera["fps"]



    # Extract intrinsics checkerboard frames if needed
    for cam in cameras.values():
        if "calib" in cam and "intrinsics" in cam["calib"]:
            intrinsics_spec = cam["calib"]["intrinsics"]

            # Check if intrinsics is a string (UUID, file path)
            if isinstance(intrinsics_spec, str):
                if _looks_like_uuid(intrinsics_spec):
                    if not registry:
                        raise ValueError(
                            f"Camera {cam['name']}: intrinsics is a UUID but --registry was not provided."
                        )
                    print(f"Loading intrinsics for camera {cam['name']} from registry DB: {intrinsics_spec}")
                    mapx, mapy, calib = load_intrinsics_from_db(registry, intrinsics_spec)
                else:
                    calib_path = Path(intrinsics_spec)
                    if calib_path.suffix.lower() in ['.h5', '.hdf5']:
                        print(f"Loading existing intrinsics for camera {cam['name']} from HDF5 file: {intrinsics_spec}")
                        mapx, mapy, calib = load_intrinsics_from_h5(intrinsics_spec)
                    elif calib_path.suffix.lower() in ['.yaml', '.yml']:
                        print(f"Loading existing intrinsics for camera {cam['name']} from YAML file: {intrinsics_spec}")
                        mapx, mapy, calib = load_intrinsics_from_yaml(intrinsics_spec)
                    else:
                        raise ValueError(
                            f"Unsupported calibration file format: {calib_path.suffix}. "
                            f"Supported formats: UUID, .h5, .hdf5, .yaml, .yml"
                        )

                cam["mapx"] = mapx
                cam["mapy"] = mapy
                cam["calib"]["calibration"] = calib

            # Otherwise, it's a dict with video path for calibration
            elif isinstance(intrinsics_spec, dict) and "video" in intrinsics_spec:
                print(f"Processing intrinsics for camera {cam['name']}")
                cam_intr_dir = intr_dir / f"int_{cam['name']}_img"
                cam_intr_dir.mkdir(parents=True, exist_ok=True)
                png_files = list(cam_intr_dir.glob("*.png"))
                if not png_files:
                    video_path = intrinsics_spec["video"]
                    rows = intrinsics_spec.get("rows", 7)
                    cols = intrinsics_spec.get("cols", 10)
                    print(
                        f"Extracting checkerboard frames for {cam['name']} from {video_path} to {cam_intr_dir} (rows={rows}, cols={cols})"
                    )
                    mapx, mapy, calib = extract_checkerboard_frames(
                        video_path, cam_intr_dir, rows, cols, window=10, threshold=0.2, calibrate=True
                    )
                    cam["mapx"] = mapx
                    cam["mapy"] = mapy
                    cam["calib"]["calibration"] = calib
                    print(f"Calibration data for {cam['name']}: {calib}")

    toml_write(calib_dir / "intrinsics.toml", cameras)

    # Filter scenes if --scene option(s) provided
    scenes_to_process = config["scenes"]
    if scene:
        scene_names = set(scene)
        scenes_to_process = [s for s in config["scenes"] if s["name"] in scene_names]
        if not scenes_to_process:
            print(f"Warning: No scenes matched the specified names: {scene_names}")
            return
        print(f"Processing {len(scenes_to_process)} scene(s): {[s['name'] for s in scenes_to_process]}")
    else:
        print(f"Processing all {len(scenes_to_process)} scenes")

    # Process scenes
    for scene_config in scenes_to_process:
        scene_dir = project_path / scene_config["name"]
        videos_dir = scene_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        # Parse start/end frames as timecodes if needed
        start_frame_raw = scene_config["start_frame"]
        end_frame_raw = scene_config["end_frame"]
        print(f"Processing scene {scene_config['name']} from {start_frame_raw} to {end_frame_raw}")
        start_time_diff = time_diff(sync_ref, parse_timecode(start_frame_raw, ref_camera_fps), ref_camera_fps)
        end_time_diff = time_diff(sync_ref, parse_timecode(end_frame_raw, ref_camera_fps), ref_camera_fps)
        print(f"Scene {scene_config['name']} start time diff: {start_time_diff:.3f} s, end time diff: {end_time_diff:.3f} s")
        for cam in cameras.values():
            # Use the scene timecodes and camera FPS for conversion
            cam_start = cam["sync_frame"] + int(round(start_time_diff * cam["fps"]))
            cam_end = cam["sync_frame"] + int(round(end_time_diff * cam["fps"]))
            src_path = cam["path"]
            dst_path = videos_dir / f"{cam['name']}.mp4"
            print(f"Extracting {src_path} frames {cam_start}:{cam_end} to {dst_path}")
            extract_video_clip(src_path, dst_path, cam_start, cam_end, cam.get("mapx", None), cam.get("mapy", None))

    # Extract extrinsics frames if specified
    for cam in cameras.values():
        if "calib" in cam and "extrinsics" in cam["calib"] and "frame" in cam["calib"]["extrinsics"]:
            extr_frame_raw = cam["calib"]["extrinsics"]["frame"]
            frame_idx = parse_timecode(extr_frame_raw, cam["fps"])
            video_path = cam["path"]

            # Save undistorted version
            out_path = extr_dir / f"ext_{cam['name']}_ext" / f"frame_{frame_idx:04d}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving extrinsics frame {frame_idx} from {video_path} to {out_path}")
            save_frame_as_png(video_path, frame_idx, out_path, cam.get("mapx", None), cam.get("mapy", None))

            # Save non-undistorted version
            no_undistort_path = extr_dir / "no_undistort" / f"{cam['name']}_original.png"
            no_undistort_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving non-undistorted extrinsics frame {frame_idx} from {video_path} to {no_undistort_path}")
            save_frame_as_png(video_path, frame_idx, no_undistort_path, None, None)

if __name__ == "__main__":
    main()
