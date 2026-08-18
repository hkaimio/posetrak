#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""
Visualize tracking results in Rerun 3D viewer.

Reads marker 3D positions from tracking_results.csv and displays them
in Rerun using the entity hierarchy defined in rerun-visualization-design.md.

Usage:
    python scripts/visualize_tracking_results.py tracking_tests/full-alpha-0_1/tracking_results.csv
    python scripts/visualize_tracking_results.py tracking_tests/full-alpha-0_1/tracking_results.csv --live
"""

import argparse
import pandas as pd
import rerun as rr
import numpy as np
import yaml
import cv2
import toml
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import uuid

# 1. Generate or define a consistent ID for this specific video dataset
# You can use a static string or a UUID based on the video filename
consistent_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, "posetrak_visualization"))

def setup_rerun(recording_path: Path | None = None, live: bool = False, app_id: str = "posetrak"):
    """Initialize Rerun recording."""
    rr.init(app_id, recording_id=consistent_uuid)

    if live:
        # Stream to viewer
        rr.connect()
        print("📡 Streaming to Rerun viewer at localhost:9876")
    elif recording_path:
        # Save to file for later viewing
        rr.save(str(recording_path))
        print(f"💾 Saving recording to {recording_path}")
    else:
        # Memory only (for quick testing) - save to temp file
        import tempfile
        temp_path = Path(tempfile.mktemp(suffix=".rrd"))
        rr.save(str(temp_path))
        print(f"🔍 Saving to {temp_path}")
        print("   Use 'rerun <path>' to view later")


def load_skeleton_hierarchy(skeleton_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load skeleton hierarchy from YAML file.

    Returns:
        Tuple of (joint_parent_map, marker_parent_map)
        - joint_parent_map: {joint_name: parent_joint_name}
        - marker_parent_map: {marker_name: parent_joint_name}
    """
    with open(skeleton_path, 'r') as f:
        skeleton = yaml.safe_load(f)

    # Build joint hierarchy
    joint_parent_map = {}
    for joint in skeleton.get('joints', []):
        joint_name = joint['name']
        parent_name = joint.get('parent')
        if parent_name:
            joint_parent_map[joint_name] = parent_name

    # Build marker -> joint mapping
    marker_parent_map = {}
    for marker in skeleton.get('markers', []):
        marker_name = marker['name']
        parent_joint = marker.get('parent')
        if parent_joint:
            marker_parent_map[marker_name] = parent_joint

    return joint_parent_map, marker_parent_map


def compute_marker_connections(marker_parent_map: Dict[str, str],
                              joint_parent_map: Dict[str, str],
                              marker_ids: Dict[str, int]) -> List[Tuple[int, int]]:
    """
    Compute marker connections based on joint hierarchy.

    Two markers are connected if their parent joints have a direct
    parent-child relationship in the skeleton hierarchy.

    Args:
        marker_parent_map: {marker_name: parent_joint_name}
        joint_parent_map: {joint_name: parent_joint_name}
        marker_ids: {marker_name: marker_id}

    Returns:
        List of (marker_id_1, marker_id_2) tuples representing connections
    """
    connections = set()

    # For each marker, find markers on the parent joint
    for marker_name, parent_joint in marker_parent_map.items():
        if marker_name not in marker_ids:
            continue

        marker_id = marker_ids[marker_name]

        # Find the parent joint's parent
        if parent_joint in joint_parent_map:
            grandparent_joint = joint_parent_map[parent_joint]

            # Find all markers attached to the grandparent joint
            for other_marker, other_parent in marker_parent_map.items():
                if other_marker in marker_ids and other_parent == grandparent_joint:
                    other_id = marker_ids[other_marker]
                    # Add connection (ensure ordering for deduplication)
                    conn = tuple(sorted([marker_id, other_id]))
                    connections.add(conn)

    return list(connections)


class SkeletonJoint:
    """Represents a joint in the skeleton hierarchy."""
    def __init__(self, name: str, parent: Optional[str], joint_type: str,
                 offset: np.ndarray, rest_orientation: np.ndarray,
                 bone_tip_offset: np.ndarray, axis: Optional[np.ndarray] = None):
        self.name = name
        self.parent = parent
        self.joint_type = joint_type
        self.offset = offset  # Translation from parent
        self.rest_orientation = rest_orientation  # ZYX Euler angles
        self.bone_tip_offset = bone_tip_offset  # Bone visual end point
        self.axis = axis if axis is not None else np.array([1.0, 0.0, 0.0])  # For revolute joints


def load_skeleton_structure(skeleton_path: Path) -> Dict[str, SkeletonJoint]:
    """
    Load full skeleton structure from YAML file.

    Returns:
        Dictionary mapping joint names to SkeletonJoint objects
    """
    with open(skeleton_path, 'r') as f:
        skeleton = yaml.safe_load(f)

    joints = {}
    for joint_data in skeleton.get('joints', []):
        name = joint_data['name']
        parent = joint_data.get('parent')
        joint_type = joint_data.get('type', 'fixed')

        # Parse vectors with defaults
        offset = np.array(joint_data.get('offset', [0.0, 0.0, 0.0]))
        rest_orientation = np.array(joint_data.get('orientation', [0.0, 0.0, 0.0]))
        bone_tip_offset = np.array(joint_data.get('bone_tip_offset', [0.0, 0.0, 0.0]))
        axis = np.array(joint_data.get('axis', [1.0, 0.0, 0.0]))

        joints[name] = SkeletonJoint(
            name=name,
            parent=parent,
            joint_type=joint_type,
            offset=offset,
            rest_orientation=rest_orientation,
            bone_tip_offset=bone_tip_offset,
            axis=axis
        )

    return joints


def load_state_vectors(state_csv_path: Path) -> pd.DataFrame:
    """
    Load state vectors CSV file with joint angles.

    Returns:
        DataFrame indexed by tracker_frame_idx with columns for each joint's angles
    """
    if not state_csv_path.exists():
        print(f"⚠️  State vectors file not found: {state_csv_path}")
        return None

    df = pd.read_csv(state_csv_path)
    return df


def euler_zyx_to_matrix(euler_zyx: np.ndarray) -> np.ndarray:
    """
    Convert ZYX Euler angles to rotation matrix.

    CRITICAL: Matches pinocchio_model_builder.cpp exactly:
    - Array stores [z, y, x] in that order
    - Computes R = Rx(x) * Ry(y) * Rz(z) (extrinsic XYZ order)

    Args:
        euler_zyx: [z, y, x] Euler angles in radians

    Returns:
        3x3 rotation matrix
    """
    z, y, x = euler_zyx

    # Individual rotation matrices
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    # R = Rx(x) * Ry(y) * Rz(z) as per C++ implementation
    return np.array([
        [cy*cz, -cy*sz, sy],
        [sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy],
        [-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy]
    ])


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle representation to rotation matrix using Rodrigues' formula.

    Matches the C++ implementation in forward_kinematics.cpp.

    Args:
        axis_angle: 3D vector where direction is axis and magnitude is angle in radians

    Returns:
        3x3 rotation matrix
    """
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-10:
        return np.eye(3)

    # Normalize to get unit axis
    axis = axis_angle / angle

    # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    # where K is the skew-symmetric matrix of the axis
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.

    Args:
        quat: [w, x, y, z] quaternion

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat

    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-10:
        return np.eye(3)

    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


def compute_joint_transforms(skeleton: Dict[str, SkeletonJoint],
                            state_vector: pd.Series,
                            rest_pose_only: bool = False) -> Dict[str, np.ndarray]:
    """
    Compute world transforms for all joints using forward kinematics.

    Args:
        skeleton: Dictionary of SkeletonJoint objects
        state_vector: Single row from state_vectors.csv with joint angles
        rest_pose_only: If True, ignore animation angles and show only rest pose

    Returns:
        Dictionary mapping joint names to 4x4 transformation matrices
    """
    transforms = {}

    # Process joints in hierarchy order (parents before children)
    def process_joint(joint_name: str):
        if joint_name in transforms:
            return transforms[joint_name]

        joint = skeleton[joint_name]

        # Build local transform
        T_local = np.eye(4)

        if joint.joint_type == 'root':
            # Root joint: use absolute position and orientation from state vector
            # The state position IS the world position, don't add offset
            root_pos = np.array([
                state_vector.get('root_position_x', 0.0),
                state_vector.get('root_position_y', 0.0),
                state_vector.get('root_position_z', 0.0)
            ])
            T_local[:3, 3] = root_pos

            # Root orientation from quaternion
            qw = state_vector.get('root_quaternion_w', 1.0)
            qx = state_vector.get('root_quaternion_x', 0.0)
            qy = state_vector.get('root_quaternion_y', 0.0)
            qz = state_vector.get('root_quaternion_z', 0.0)
            R_quat = quaternion_to_matrix(np.array([qw, qx, qy, qz]))

            # Apply rest orientation in the root's frame
            R_rest = euler_zyx_to_matrix(joint.rest_orientation)
            T_local[:3, :3] = R_quat @ R_rest

            # Root has no parent, so T_world = T_local
            transforms[joint_name] = T_local
            return T_local

        # Get parent transform for non-root joints
        if joint.parent is None:
            raise ValueError(f"Non-root joint {joint_name} has no parent")

        T_parent = process_joint(joint.parent)

        # Translation from parent
        T_local[:3, 3] = joint.offset

        # Rest orientation (defines joint's local frame in parent's coordinate system)
        R_rest = euler_zyx_to_matrix(joint.rest_orientation)

        # Animation rotation from state vector (deviation from rest pose)
        R_anim = np.eye(3)

        # Skip animation if showing rest pose only
        if not rest_pose_only:
            if joint.joint_type == 'revolute':
                # Single angle around axis
                angle_key = f'joint_{joint.name}_angle_0'
                if angle_key in state_vector and not pd.isna(state_vector[angle_key]):
                    angle = state_vector[angle_key]
                    axis_angle = joint.axis * angle
                    R_anim = axis_angle_to_matrix(axis_angle)

            elif joint.joint_type in ['ball', 'spherical']:
                # Axis-angle representation (3 components)
                angle_keys = [f'joint_{joint.name}_angle_{i}' for i in range(3)]
                if all(k in state_vector for k in angle_keys):
                    axis_angle = np.array([
                        state_vector.get(angle_keys[0], 0.0),
                        state_vector.get(angle_keys[1], 0.0),
                        state_vector.get(angle_keys[2], 0.0)
                    ])
                    if not all(pd.isna(state_vector[k]) for k in angle_keys):
                        R_anim = axis_angle_to_matrix(axis_angle)

        # Combine: first establish rest frame, then apply animation in that frame
        # This matches Pinocchio's behavior
        T_local[:3, :3] = R_rest @ R_anim

        # Compute world transform
        transforms[joint_name] = T_parent @ T_local
        return transforms[joint_name]

    # Process all joints
    for joint_name in skeleton.keys():
        process_joint(joint_name)

    return transforms


def compute_bone_radius(bone_length: float) -> float:
    """
    Scale bone visual thickness based on bone length.

    Args:
        bone_length: Length of bone in meters

    Returns:
        Radius for bone visualization in meters
    """
    # Fingers (2-5cm) -> thin (2-3mm radius)
    # Arms/legs (20-50cm) -> thick (10-15mm radius)
    if bone_length > 0.05:  # Longer than 5cm
        return 0.015

    return 0.005


def log_skeleton_bones(skeleton: Dict[str, SkeletonJoint],
                       transforms: Dict[str, np.ndarray],
                       person_id: int = 0,
                       rest_pose: bool = False):
    """
    Log animated skeleton bones for one frame.

    Draws bones from joint origin to bone_tip_offset (defining bone geometry).

    Args:
        skeleton: Dictionary of SkeletonJoint objects
        transforms: Joint world transforms from forward kinematics
        person_id: Person identifier for entity path
        rest_pose: If True, logs to separate entity path for rest pose visualization
    """
    bone_starts = []
    bone_ends = []
    bone_radii = []

    for joint_name, joint in skeleton.items():
        # Get joint world transform
        T_world = transforms[joint_name]
        joint_pos = T_world[:3, 3]

        # Compute bone tip in world frame (bone_tip_offset is in joint's local frame)
        bone_tip_local = np.array([*joint.bone_tip_offset, 1.0])
        bone_tip_world = T_world @ bone_tip_local
        bone_tip_pos = bone_tip_world[:3]

        # Compute bone length and radius
        bone_vector = bone_tip_pos - joint_pos
        bone_length = np.linalg.norm(bone_vector)

        if bone_length < 0.001:  # Skip very short bones (< 1mm)
            continue

        bone_starts.append(joint_pos)
        bone_ends.append(bone_tip_pos)
        bone_radii.append(compute_bone_radius(bone_length))

    if len(bone_starts) == 0:
        return

    # Create line segments for all bones
    segments = [[start, end] for start, end in zip(bone_starts, bone_ends)]

    # Choose entity path and color based on whether this is rest pose
    if rest_pose:
        entity_path = f"points/person_{person_id}/skeleton/rest_pose"
        color = [255, 128, 0]  # Orange for rest pose
    else:
        entity_path = f"points/person_{person_id}/skeleton/bones"
        color = [0, 255, 0]

    # Log as LineStrips3D with varying radii
    rr.log(
        entity_path,
        rr.LineStrips3D(
            segments,
            radii=bone_radii,
            colors=color
        )
    )


def log_world_setup():
    """Log static world setup: coordinate axes."""
    # Root coordinate system (structure_from_motion pattern)
    rr.log(
        "/",
        rr.ViewCoordinates.RIGHT_HAND_Z_UP,
        static=True,
    )

    # Coordinate axes for reference
    rr.log(
        "points/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["X", "Y", "Z"],
        ),
        static=True,
    )


def open_video_capture(video_path: Path) -> cv2.VideoCapture:
    """Open video file for reading frames on-demand."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return cap


def read_video_frame(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    """
    Read a specific frame from video.

    Args:
        cap: OpenCV VideoCapture object
        frame_idx: Frame index to read (0-based)

    Returns:
        Frame as RGB numpy array, or None if frame not available
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    # Convert BGR to RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_camera_config(cameras_toml: Path) -> Dict:
    """Load camera calibration data from TOML file."""
    with open(cameras_toml, 'r') as f:
        return toml.load(f)


def load_sync_metadata(sync_json: Path) -> Dict:
    """Load camera synchronization metadata from JSON file."""
    with open(sync_json, 'r') as f:
        return json.load(f)


def get_video_frame_for_tracking_frame(tracking_frame: int, timestamp: float,
                                       sync_points: List[Dict], fps: float = 120.0) -> int:
    """
    Get video frame index for a given tracking frame.

    Args:
        tracking_frame: Tracking frame number (1-based)
        timestamp: Tracking timestamp
        sync_points: List of {"frame": int, "timestamp": float} sync points
        fps: Default FPS if no sync points

    Returns:
        Video frame index (0-based)
    """
    if not sync_points:
        # No sync data, use simple FPS calculation
        return int(timestamp * fps)

    # Find video frame by interpolating/extrapolating from sync points
    # Sync points map video frame -> timestamp
    for i in range(len(sync_points) - 1):
        t1 = sync_points[i]['timestamp']
        t2 = sync_points[i+1]['timestamp']
        if t1 <= timestamp <= t2:
            # Interpolate
            f1 = sync_points[i]['frame']
            f2 = sync_points[i+1]['frame']
            ratio = (timestamp - t1) / (t2 - t1) if t2 > t1 else 0
            return int(f1 + ratio * (f2 - f1))

    # Before first or after last sync point - extrapolate
    if timestamp < sync_points[0]['timestamp']:
        # Before first point
        if len(sync_points) >= 2:
            rate = (sync_points[1]['frame'] - sync_points[0]['frame']) / \
                   (sync_points[1]['timestamp'] - sync_points[0]['timestamp'])
        else:
            rate = fps
        return int(sync_points[0]['frame'] + (timestamp - sync_points[0]['timestamp']) * rate)
    else:
        # After last point
        if len(sync_points) >= 2:
            rate = (sync_points[-1]['frame'] - sync_points[-2]['frame']) / \
                   (sync_points[-1]['timestamp'] - sync_points[-2]['timestamp'])
        else:
            rate = fps
        return int(sync_points[-1]['frame'] + (timestamp - sync_points[-1]['timestamp']) * rate)


def log_camera_3d(camera_name: str, camera_data: Dict, entity_path_prefix: str = "camera"):
    """
    Log camera in 3D space with proper intrinsics and extrinsics.
    Uses structure_from_motion hierarchy pattern.

    Args:
        camera_name: Camera identifier (e.g., 'cam1')
        camera_data: Camera calibration data with matrix, rotation, translation
        entity_path_prefix: Root entity path (default: "camera")
    """
    # Extract intrinsics
    intrinsics = np.array(camera_data['matrix'])
    width, height = camera_data['size']

    # Extract extrinsics
    # Rotation is Rodrigues vector (world-to-camera), convert to rotation matrix
    rvec = np.array(camera_data['rotation'])
    R_world_to_cam, _ = cv2.Rodrigues(rvec)

    # Translation vector (world-to-camera)
    t_world_to_cam = np.array(camera_data['translation'])

    # Camera extrinsics give world-to-camera transform: P_cam = R * P_world + t
    # For Rerun, we need camera-to-world transform to position the camera
    # Inverse: R_cam_to_world = R^T, t_cam_to_world = -R^T * t
    R_cam_to_world = R_world_to_cam.T
    t_cam_to_world = -R_cam_to_world @ t_world_to_cam

    # Structure_from_motion pattern:
    # camera/{name} - Transform3D only (extrinsics/position in 3D space)
    # camera/{name}/image - Pinhole only (intrinsics/image plane)

    camera_entity = f"{entity_path_prefix}/{camera_name}"
    image_entity = f"{camera_entity}/image"

    # Log camera position and orientation (extrinsics)
    rr.log(
        camera_entity,
        rr.Transform3D(
            translation=t_cam_to_world,
            mat3x3=R_cam_to_world,
        ),
        static=True,
    )

    # Log pinhole camera intrinsics at image child entity
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    rr.log(
        image_entity,
        rr.Pinhole(
            resolution=[width, height],
            focal_length=[fx, fy],
            principal_point=[cx, cy],
        ),
        static=True,
    )


def log_camera_images_only(video_cap: cv2.VideoCapture, camera_name: str,
                          tracking_csv: Path, sync_points: Dict):
    """
    Log ONLY camera images without any marker observations (for base layer).

    Args:
        video_cap: OpenCV VideoCapture object for reading frames
        camera_name: Camera name (e.g., 'cam1')
        tracking_csv: Path to tracking_results.csv (for frame/timestamp info)
        sync_points: Sync metadata dict with sync_points list and fps
    """
    print(f"📹 Loading camera frames from video...")

    # Load tracking results to get frame/timestamp info
    tracking_df = pd.read_csv(tracking_csv)
    unique_frames = tracking_df[['frame', 'timestamp']].drop_duplicates().sort_values('frame')

    entity_base = f"camera/{camera_name}"

    print(f"🎬 Logging {len(unique_frames)} camera frames (images only, no markers)...")

    prev_frame_idx = None
    for _, row in unique_frames.iterrows():
        frame_num = row['frame']
        timestamp = row['timestamp']

        # Map tracking frame to video frame using sync metadata
        video_frame_idx = get_video_frame_for_tracking_frame(
            frame_num, timestamp,
            sync_points.get("sync_points", []),
            sync_points.get("fps", 120.0)
        )

        if video_frame_idx != prev_frame_idx:
            # Read frame directly from video
            frame_img_rgb = read_video_frame(video_cap, video_frame_idx)
            if frame_img_rgb is None:
                continue

            prev_frame_idx = video_frame_idx

            # Set timelines
            rr.set_time("frame", sequence=int(frame_num))
            rr.set_time("timestamp", timestamp=float(timestamp))

            # Log the video frame as background image
            rr.log(
                f"{entity_base}/image",
                rr.Image(frame_img_rgb).compress(jpeg_quality=75),
            )

            # Progress indicator
            if frame_num % 30 == 0:
                print(f"  Frame {frame_num}")

    print(f"✅ Logged camera images for {camera_name}")


def log_camera_observations(observations_csv: Path, camera_name: str,
                           camera_id: int, marker_ids_map: Dict[str, int],
                           debug_dir: Optional[Path] = None):
    """
    Log ONLY 2D marker observations without camera images (for tracking overlay).

    Args:
        observations_csv: Path to observations.csv with 2D marker positions
        camera_name: Camera name (e.g., 'cam1')
        camera_id: Camera ID in observations CSV (e.g., 0 for cam1)
        marker_ids_map: Map from marker name to marker ID
        debug_dir: Optional path to debug directory with all_observations.csv files
    """
    print(f"📹 Loading camera observations from {observations_csv}...")
    obs_df = pd.read_csv(observations_csv)

    # Filter for this camera
    cam_obs = obs_df[obs_df['camera_id'] == camera_id].copy()
    print(f"📊 Loaded {len(cam_obs)} observations for camera {camera_id}")

    # Group by frame
    frames = cam_obs.groupby('frame')

    # Use structure_from_motion hierarchy: camera/{name}/image for 2D view
    entity_base = f"camera/{camera_name}"

    print(f"🎬 Logging 2D marker observations...")

    processed_frames = 0
    for frame_num, frame_data in frames:
        # Get timestamp for this tracking frame
        timestamp = frame_data['timestamp'].iloc[0]

        # Set timelines
        rr.set_time("frame", sequence=int(frame_num))
        rr.set_time("timestamp", timestamp=float(timestamp))

        # Extract 2D marker positions
        positions_2d = frame_data[['pixel_x', 'pixel_y']].values
        marker_ids = frame_data['marker_id'].values
        confidences = frame_data['confidence'].values

        # Load debug data to get outlier status and metrics
        is_outlier = np.zeros(len(frame_data), dtype=bool)
        mahalanobis_dist = np.zeros(len(frame_data))
        residual_norms = np.zeros(len(frame_data))
        has_predicted = np.zeros(len(frame_data), dtype=bool)
        predicted_positions = np.zeros_like(positions_2d)

        if debug_dir:
            debug_frame_dir = debug_dir / f"frame_{frame_num:04d}"
            debug_csv = debug_frame_dir / "all_observations.csv"

            if debug_csv.exists():
                debug_df = pd.read_csv(debug_csv)
                cam_debug = debug_df[debug_df['camera_id'] == camera_id].copy()

                # Match debug data to observations by marker_name
                for idx, row in frame_data.iterrows():
                    marker_name = row['marker_name']
                    debug_row = cam_debug[cam_debug['marker_name'] == marker_name]
                    if len(debug_row) > 0:
                        debug_row = debug_row.iloc[0]
                        local_idx = list(frame_data.index).index(idx)
                        is_outlier[local_idx] = debug_row['is_outlier']
                        mahalanobis_dist[local_idx] = debug_row['mahalanobis_distance']
                        residual_norms[local_idx] = debug_row['residual_norm']
                        if not pd.isna(debug_row['predicted_u']):
                            has_predicted[local_idx] = True
                            predicted_positions[local_idx] = [debug_row['predicted_u'], debug_row['predicted_v']]

        # Color markers by confidence and outlier status
        colors = []
        for i, conf in enumerate(confidences):
            if conf > 5.0:
                marker_color = [0, 255, 0]  # Green for high confidence
            elif conf > 3.0:
                marker_color = [255, 255, 0]  # Yellow for medium
            else:
                marker_color = [255, 100, 0]  # Orange for low confidence
            if is_outlier[i]:
                for j in range(3):
                    marker_color[j] = int(marker_color[j] * 0.5 + 128 * 0.5)
            colors.append(marker_color)

        # Log all 2D observed points with scalar attributes
        # Use class_ids for annotation context labels
        rr.log(
            f"{entity_base}/image/keypoints",
            rr.Points2D(
                positions=positions_2d,
                colors=colors,
                radii=np.where(is_outlier, 3.0, 6.0),  # Larger radius for outliers
                class_ids=marker_ids.astype(np.uint16),
            ),
            rr.AnyValues(
                mahalanobis_distance=mahalanobis_dist,
                residual_norm=residual_norms,
                confidence=confidences,
            )
        )

        # Log scalar timeseries for metrics (visible in plot view)
        for i, (marker_id, marker_name) in enumerate(zip(marker_ids, frame_data['marker_name'])):
            if mahalanobis_dist[i] > 0:  # Only log if we have debug data
                rr.log(
                    f"{entity_base}//metrics/{marker_name}/mahalanobis",
                    rr.Scalars(float(mahalanobis_dist[i])),
                )
                rr.log(
                    f"{entity_base}/metrics/{marker_name}/residual",
                    rr.Scalars(float(residual_norms[i])),
                )
                rr.log(
                    f"{entity_base}/metrics/{marker_name}/confidence",
                    rr.Scalars(float(confidences[i])),
                )

        # Log predicted points where available (separate series)
        if has_predicted.any():
            pred_pos = predicted_positions[has_predicted]
            pred_ids = marker_ids[has_predicted]
            rr.log(
                f"{entity_base}/image/predictions",
                rr.Points2D(
                    positions=pred_pos,
                    colors=[100, 150, 255],  # Light blue
                    radii=4.0,
                    class_ids=pred_ids.astype(np.uint16),
                ),
            )

        processed_frames += 1
        # Progress indicator
        if frame_num % 30 == 0:
            print(f"  Frame {frame_num}: {len(frame_data)} markers")

    print(f"✅ Processed {processed_frames} frames for {camera_name}")


def visualize_camera_view(observations_csv: Path, video_cap: cv2.VideoCapture,
                         camera_name: str, camera_id: int, sync_points: List[Dict],
                         marker_ids_map: Dict[str, int],
                         debug_dir: Optional[Path] = None):
    """
    Visualize camera view with 2D marker observations overlaid.
    This is kept for backwards compatibility - it combines both image and observation logging.

    Args:
        observations_csv: Path to observations.csv with 2D marker positions
        video_cap: OpenCV VideoCapture object for reading frames
        camera_name: Camera name (e.g., 'cam1')
        camera_id: Camera ID in observations CSV (e.g., 0 for cam1)
        sync_points: List of sync points for frame/timestamp mapping
        marker_ids_map: Map from marker name to marker ID
        debug_dir: Optional path to debug directory with all_observations.csv files
    """
    print(f"📹 Loading camera observations from {observations_csv}...")
    obs_df = pd.read_csv(observations_csv)

    # Filter for this camera
    cam_obs = obs_df[obs_df['camera_id'] == camera_id].copy()
    print(f"📊 Loaded {len(cam_obs)} observations for camera {camera_id}")

    # Group by frame
    frames = cam_obs.groupby('frame')

    # Use structure_from_motion hierarchy: camera/{name}/image for 2D view
    entity_base = f"camera/{camera_name}"

    print(f"🎬 Logging camera frames with 2D observations...")

    processed_frames = 0
    prev_frame_idx = None
    for frame_num, frame_data in frames:
        # Get timestamp for this tracking frame
        timestamp = frame_data['timestamp'].iloc[0]

        # Map tracking frame to video frame using sync metadata
        video_frame_idx = get_video_frame_for_tracking_frame(frame_num, timestamp, sync_points["sync_points"], sync_points.get("fps", 120.0))
        if video_frame_idx != prev_frame_idx:

            # Read frame directly from video
            frame_img_rgb = read_video_frame(video_cap, video_frame_idx)
            if frame_img_rgb is None:
                continue

            prev_frame_idx = video_frame_idx

            # Set timelines
            rr.set_time("frame", sequence=int(frame_num))
            rr.set_time("timestamp", timestamp=float(timestamp))

            # Log the video frame as background image
            rr.log(
                f"{entity_base}/image",
                rr.Image(frame_img_rgb).compress(jpeg_quality=75),
            )

        # Extract 2D marker positions
        positions_2d = frame_data[['pixel_x', 'pixel_y']].values
        marker_ids = frame_data['marker_id'].values
        confidences = frame_data['confidence'].values

        # Load debug data to get outlier status and metrics
        is_outlier = np.zeros(len(frame_data), dtype=bool)
        mahalanobis_dist = np.zeros(len(frame_data))
        residual_norms = np.zeros(len(frame_data))
        has_predicted = np.zeros(len(frame_data), dtype=bool)
        predicted_positions = np.zeros_like(positions_2d)

        if debug_dir:
            debug_frame_dir = debug_dir / f"frame_{frame_num:04d}"
            debug_csv = debug_frame_dir / "all_observations.csv"

            if debug_csv.exists():
                debug_df = pd.read_csv(debug_csv)
                cam_debug = debug_df[debug_df['camera_id'] == camera_id].copy()

                # Match debug data to observations by marker_name
                for idx, row in frame_data.iterrows():
                    marker_name = row['marker_name']
                    debug_row = cam_debug[cam_debug['marker_name'] == marker_name]
                    if len(debug_row) > 0:
                        debug_row = debug_row.iloc[0]
                        local_idx = list(frame_data.index).index(idx)
                        is_outlier[local_idx] = debug_row['is_outlier']
                        mahalanobis_dist[local_idx] = debug_row['mahalanobis_distance']
                        residual_norms[local_idx] = debug_row['residual_norm']
                        if not pd.isna(debug_row['predicted_u']):
                            has_predicted[local_idx] = True
                            predicted_positions[local_idx] = [debug_row['predicted_u'], debug_row['predicted_v']]

        # Color markers by confidence and outlier status
        colors = []
        for i, conf in enumerate(confidences):
            if conf > 5.0:
                marker_color = [0, 255, 0]  # Green for high confidence
            elif conf > 3.0:
                marker_color = [255, 255, 0]  # Yellow for medium
            else:
                marker_color = [255, 100, 0]  # Orange for low confidence
            if is_outlier[i]:
                for j in range(3):
                    marker_color[j] = int(marker_color[j] * 0.5 + 128 * 0.5)
            colors.append(marker_color)

        # Log all 2D observed points with scalar attributes
        # Use class_ids for annotation context labels
        rr.log(
            f"{entity_base}/image/keypoints",
            rr.Points2D(
                positions=positions_2d,
                colors=colors,
                radii=np.where(is_outlier, 3.0, 6.0),  # Larger radius for outliers
                class_ids=marker_ids.astype(np.uint16),
            ),
            rr.AnyValues(
                mahalanobis_distance=mahalanobis_dist,
                residual_norm=residual_norms,
                confidence=confidences,
            )
        )

        # Log scalar timeseries for metrics (visible in plot view)
        for i, (marker_id, marker_name) in enumerate(zip(marker_ids, frame_data['marker_name'])):
            if mahalanobis_dist[i] > 0:  # Only log if we have debug data
                rr.log(
                    f"{entity_base}//metrics/{marker_name}/mahalanobis",
                    rr.Scalars(float(mahalanobis_dist[i])),
                )
                rr.log(
                    f"{entity_base}/metrics/{marker_name}/residual",
                    rr.Scalars(float(residual_norms[i])),
                )
                rr.log(
                    f"{entity_base}/metrics/{marker_name}/confidence",
                    rr.Scalars(float(confidences[i])),
                )

        # Log predicted points where available (separate series)
        if has_predicted.any():
            pred_pos = predicted_positions[has_predicted]
            pred_ids = marker_ids[has_predicted]
            rr.log(
                f"{entity_base}/image/predictions",
                rr.Points2D(
                    positions=pred_pos,
                    colors=[100, 150, 255],  # Light blue
                    radii=4.0,
                    class_ids=pred_ids.astype(np.uint16),
                ),
            )

        processed_frames += 1
        # Progress indicator
        if frame_num % 30 == 0:
            print(f"  Frame {frame_num}: {len(frame_data)} markers")

    print(f"✅ Processed {processed_frames} frames for {camera_name}")


def visualize_tracking_results(csv_path: Path, skeleton_path: Path | None = None,
                              show_skeleton_bones: bool = False) -> Dict[str, int]:
    """
    Load and visualize tracking results.

    Args:
        csv_path: Path to tracking_results.csv
        skeleton_path: Optional path to skeleton YAML for marker connections
        show_skeleton_bones: If True, render animated skeleton bones

    Returns:
        Dictionary mapping marker names to marker IDs
    """
    print(f"📂 Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"📊 Loaded {len(df)} marker observations across {df['frame'].nunique()} frames")
    print(f"🎯 Tracking {df['marker_id'].nunique()} markers")

    # Set up world coordinate system
    #log_world_setup()

    # Create class descriptions for all markers (logged once, statically)
    person_id = 0
    # Use 'points' hierarchy to separate from camera hierarchy
    entity_base = f"points/person_{person_id}/markers"

    # Get unique markers
    unique_markers = df[['marker_id', 'marker_name']].drop_duplicates().sort_values('marker_id')

    # Build marker name -> ID mapping
    marker_ids_map = {row['marker_name']: int(row['marker_id'])
                      for _, row in unique_markers.iterrows()}

    # Load skeleton hierarchy if provided
    keypoint_connections = None
    skeleton_structure = None
    state_vectors_df = None

    if skeleton_path:
        print(f"📐 Loading skeleton hierarchy from {skeleton_path}...")
        joint_parent_map, marker_parent_map = load_skeleton_hierarchy(skeleton_path)
        keypoint_connections = compute_marker_connections(
            marker_parent_map, joint_parent_map, marker_ids_map
        )
        print(f"🔗 Computed {len(keypoint_connections)} marker connections")

        # Load full skeleton structure for bone visualization
        if show_skeleton_bones:
            skeleton_structure = load_skeleton_structure(skeleton_path)
            print(f"🦴 Loaded skeleton with {len(skeleton_structure)} joints")

            # Load state vectors for animation
            state_csv_path = csv_path.parent / "state_vectors.csv"
            state_vectors_df = load_state_vectors(state_csv_path)
            if state_vectors_df is not None:
                print(f"📊 Loaded {len(state_vectors_df)} state vector frames")

    # Create annotation context with single class for skeleton
    # All markers are keypoints within this class (Rerun face_tracking pattern)
    keypoint_annotations = [
        rr.AnnotationInfo(id=int(row['marker_id']), label=str(row['marker_name']))
        for _, row in unique_markers.iterrows()
    ]

    class_descriptions = [
        rr.ClassDescription(
            info=rr.AnnotationInfo(
                id=person_id,
                label=f"person_{person_id}",
            ),
            keypoint_annotations=keypoint_annotations,
            # keypoint_connections=keypoint_connections,
        )
    ]

    # Log annotation context once (static)
    rr.log(
        entity_base,
        rr.AnnotationContext(class_descriptions),
        static=True,
    )
    print(f"📝 Registered {len(class_descriptions)} marker class descriptions")

    # Group by frame for temporal logging
    frames = df.groupby('frame')

    print(f"🎬 Logging {len(frames)} frames...")

    for frame_num, frame_data in frames:
        # Get timestamp (use first row's timestamp for this frame)
        timestamp = frame_data['timestamp'].iloc[0]

        # Set dual timelines as per design doc (using keyword arguments)
        rr.set_time("frame", sequence=int(frame_num))
        rr.set_time("timestamp", timestamp=float(timestamp))

        # Visualize skeleton bones if enabled
        if show_skeleton_bones and skeleton_structure is not None and state_vectors_df is not None:
            # Find matching state vector for this frame
            state_row = state_vectors_df[state_vectors_df['tracker_frame_idx'] == frame_num]
            if len(state_row) > 0:
                state_vector = state_row.iloc[0]

                # For frame 0, also log rest pose for validation
                if frame_num == 1:
                    rest_transforms = compute_joint_transforms(skeleton_structure, state_vector, rest_pose_only=True)
                    log_skeleton_bones(skeleton_structure, rest_transforms, person_id, rest_pose=True)

                # Compute forward kinematics with animation
                transforms = compute_joint_transforms(skeleton_structure, state_vector, rest_pose_only=False)

                # Log animated skeleton bones
                log_skeleton_bones(skeleton_structure, transforms, person_id, rest_pose=False)

        # Filter visible markers
        visible_markers = frame_data[frame_data['is_visible'] == True].copy()

        if len(visible_markers) == 0:
            continue

        # Extract 3D positions
        positions = visible_markers[['x_3d', 'y_3d', 'z_3d']].values

        # Extract marker IDs for keypoint annotation
        marker_ids = visible_markers['marker_id'].values

        # Log posterior markers (final tracking results)
        # Single class with multiple keypoints enables skeleton connections
        rr.log(
            f"{entity_base}/posterior",
            rr.Points3D(
                positions=positions,
                colors=[255, 0, 0],  # Red
                radii=0.030,
                class_ids=[person_id] * len(positions),  # Single class for whole skeleton
                keypoint_ids=marker_ids.astype(np.uint16),  # Which keypoint each point is
            ),
        )

        # Progress indicator (every 100 frames)
        if frame_num % 100 == 0:
            print(f"  Frame {frame_num}: {len(visible_markers)} markers")

    print("✅ Visualization complete!")
    print("\n📖 To view the recording:")
    print("  rerun tracking_tests/full-alpha-0_1/tracking_viz.rrd")
    print("\n🎮 Rerun viewer controls:")
    print("  - Use timeline slider to scrub through frames")
    print("  - Click 'frame' or 'timestamp' to switch timeline")
    print("  - Select entities in left panel to toggle visibility")
    print("  - Hover over markers to see their labels from AnnotationContext")
    print("  - Right-click + drag to rotate 3D view")

    return marker_ids_map


def main():
    parser = argparse.ArgumentParser(
        description="Visualize tracking results in Rerun",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to tracking_results.csv",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Save recording to .rrd file (optional)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Stream to live Rerun viewer instead of spawning new window",
    )
    parser.add_argument(
        "--skeleton",
        "-s",
        type=Path,
        help="Path to skeleton YAML file (adds connection lines between markers)",
    )
    parser.add_argument(
        "--cameras",
        "-c",
        type=Path,
        help="Path to camera calibration TOML file (adds camera views)",
    )
    parser.add_argument(
        "--video-dir",
        "-v",
        type=Path,
        help="Directory containing camera video files (cam1.mp4, cam2.mp4, etc.)",
    )
    parser.add_argument(
        "--sync",
        type=Path,
        help="Path to sync metadata JSON file (for camera frame timing)",
    )
    parser.add_argument(
        "--app-id",
        type=str,
        default="posetrak",
        help="Rerun application ID (use same ID to overlay multiple recordings)",
    )
    parser.add_argument(
        "--only-cameras",
        action="store_true",
        help="Only export camera images and 2D observations (for base layer)",
    )
    parser.add_argument(
        "--only-tracking",
        action="store_true",
        help="Only export 3D tracking data (for overlay on camera base layer)",
    )
    parser.add_argument(
        "--skeleton-bones",
        action="store_true",
        help="Show animated skeleton bones (requires --skeleton and state_vectors.csv)",
    )

    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"❌ Error: {args.csv_path} not found")
        return 1

    if args.skeleton and not args.skeleton.exists():
        print(f"❌ Error: {args.skeleton} not found")
        return 1

    # Validate flags
    if args.only_cameras and args.only_tracking:
        print("❌ Error: Cannot use both --only-cameras and --only-tracking")
        return 1

    # Setup Rerun with custom app ID
    setup_rerun(recording_path=args.output, live=args.live, app_id=args.app_id)

    # Determine what to log
    log_tracking = not args.only_cameras
    log_cameras = not args.only_tracking

    # Visualize 3D tracking results and get marker IDs map
    marker_ids_map = None
    if log_tracking:
        marker_ids_map = visualize_tracking_results(
            args.csv_path,
            skeleton_path=args.skeleton,
            show_skeleton_bones=args.skeleton_bones
        )
    else:
        print("ℹ️  Skipping 3D tracking data (--only-cameras mode)")
        # Still need to load marker IDs for camera view annotations
        tracking_df = pd.read_csv(args.csv_path)
        marker_ids_map = {row['marker_name']: int(row['marker_id'])
                         for _, row in tracking_df[['marker_id', 'marker_name']].drop_duplicates().iterrows()}

    # Camera visualization
    if args.cameras and args.video_dir:
        print(f"\n{'='*60}")
        if log_cameras and not log_tracking:
            print(f"📹 Adding camera base layer (images only, no markers)")
        elif log_tracking and not log_cameras:
            print(f"📊 Adding tracking observations to camera views")
        else:
            print(f"📹 Adding full camera visualizations")
        print(f"{'='*60}\n")

        # Load camera config
        camera_config = load_camera_config(args.cameras)
        observations_csv = args.csv_path.parent / "observations.csv"

        # Load sync metadata if provided
        sync_data = {}
        if args.sync and args.sync.exists():
            print(f"📅 Loading sync metadata from {args.sync}")
            sync_data = load_sync_metadata(args.sync)
        else:
            print("⚠️  No sync metadata provided, using default timing")

        # Check for debug directory
        debug_dir = args.csv_path.parent / "debug"
        if debug_dir.exists():
            print(f"🐛 Debug data found at {debug_dir}")
        else:
            debug_dir = None
            print("ℹ️  No debug data available")



        # Handle camera images (base layer)
        if log_cameras:
            # Log 3D camera positions and frustums (for both modes)
            for camera_name, camera_data in camera_config.items():
                if camera_name == 'metadata':
                    continue
                print(f"📷 Setting up {camera_name} in 3D space...")
                log_camera_3d(camera_name, camera_data)

            print(f"\n🎥 Processing camera videos (base layer - images only)...")

            for camera_name, camera_data in camera_config.items():
                if camera_name == 'metadata':
                    continue

                video_path = args.video_dir / f"{camera_name}.mp4"
                sync_key = camera_name
                sync_points = sync_data.get(sync_key, {})

                if video_path.exists():
                    print(f"\n📹 Processing {camera_name} video...")
                    video_cap = open_video_capture(video_path)

                    try:
                        log_camera_images_only(video_cap, camera_name,
                                             args.csv_path, sync_points)
                    finally:
                        video_cap.release()
                else:
                    print(f"⚠️  Video not found for {camera_name}: {video_path}")

        # Handle tracking observations (overlay layer)
        if log_tracking:
            if not observations_csv.exists():
                print(f"⚠️  Observations not found: {observations_csv}")
            else:
                print(f"\n📊 Processing camera observations (tracking overlay)...")

                # Create annotation context for camera views (for marker labels)
                tracking_df = pd.read_csv(args.csv_path)
                unique_markers = tracking_df[['marker_id', 'marker_name']].drop_duplicates().sort_values('marker_id')

                # Create class descriptions for each marker (for camera 2D views)
                marker_class_descriptions = [
                    rr.ClassDescription(
                        info=rr.AnnotationInfo(
                            id=int(row['marker_id']),
                            label=str(row['marker_name']),
                        ),
                    )
                    for _, row in unique_markers.iterrows()
                ]

                # Log annotation context for each camera's 2D view
                for camera_name, camera_data in camera_config.items():
                    if camera_name == 'metadata':
                        continue
                    # rr.log(
                    #     f"camera/{camera_name}/image",
                    #     rr.AnnotationContext(marker_class_descriptions),
                    #     static=True,
                    # )

                # Process each camera's observations
                for camera_name, camera_data in camera_config.items():
                    if camera_name == 'metadata':
                        continue

                    # Extract camera ID from name (cam1->0, cam2->1, etc.)
                    camera_id = int(camera_name.replace('cam', '')) - 1

                    print(f"\n📹 Processing {camera_name} observations (camera_id={camera_id})...")
                    log_camera_observations(observations_csv, camera_name, camera_id,
                                          marker_ids_map, debug_dir=debug_dir)
    elif args.cameras or args.video_dir:
        print(f"\n⚠️  Skipping camera visualization: need both --cameras and --video-dir")

    return 0


if __name__ == "__main__":
    exit(main())
