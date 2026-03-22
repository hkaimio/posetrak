#!/usr/bin/env python3
"""
visualize_tracking.py — Overlay UKF tracking results on source camera videos.

Produces either:
  - A single-camera video with tracking overlays, or
  - A mosaic (grid) video with all cameras arranged in a grid.

Observations (2D detections) are drawn as dots: inlier observations in the
person colour, outlier observations in grey.  The skeleton is drawn as
wireframe bones from each joint's origin to its bone_tip_offset endpoint
(computed via forward kinematics from state_vectors.csv).

Usage:
    python3 scripts/visualize_tracking.py \\
        --tracking-dir tracking_tests/run1 [--tracking-dir tracking_tests/run2] \\
        --cameras /path/to/Calib_scene.toml \\
        --sync /path/to/sync_data.json \\
        --video-dir /path/to/videos \\
        [--skeleton my_skeleton.yaml]   # auto-detected from TOML if omitted \\
        [--camera cam3]                 # single-camera mode; omit for mosaic \\
        [--resolution 1920x1080]        \\
        [--fps 60]                      \\
        --output result.mp4
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
import tomllib
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Colours — BGR order for OpenCV
# ---------------------------------------------------------------------------

PERSON_BONE_COLORS_BGR: list[tuple[int, int, int]] = [
    (  0, 220,   0),   # green
    (139, 170, 255),   # orange
    (200,  40,  40),   # blue
    (  0, 200, 220),   # yellow
    (210,   0, 210),   # magenta
    (255, 180,   0),   # cyan
]
PERSON_MARKER_COLORS_BGR: list[tuple[int, int, int]] = [
    (  33, 255,   0),   # green
    (  0, 120, 255),   # orange
    (200,  40,  40),   # blue
    (  0, 200, 220),   # yellow
    (210,   0, 210),   # magenta
    (255, 180,   0),   # cyan
]
OUTLIER_COLOR_BGR: tuple[int, int, int] = (130, 130, 130)
OVERLAY_TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Camera(NamedTuple):
    """Calibrated camera with precomputed projection matrix."""
    csv_id: int         # 0-based camera_id used in marker_projections.csv
    toml_name: str      # section key in the TOML (cam1, cam2, …)
    display_name: str   # human-readable name from TOML 'name' field
    K: np.ndarray       # 3×3 intrinsic matrix
    R: np.ndarray       # 3×3 rotation (world → camera)
    t: np.ndarray       # 3-vector translation (world → camera)
    P: np.ndarray       # 3×4 projection matrix K @ [R | t]
    dist: np.ndarray = None  # 4-vector distortion coeffs [k1,k2,p1,p2] (optional)


class SyncTable:
    """Per-camera lookup: tracker timestamp → nearest video frame index."""

    def __init__(self, sync_data: dict):
        self._tables: dict[str, tuple[list[float], list[int]]] = {}
        for cam_name, info in sync_data.items():
            pts = info.get("syncpoints", [])
            timestamps = [sp["timestamp"] for sp in pts]
            frames     = [sp["frame"]     for sp in pts]
            self._tables[cam_name] = (timestamps, frames)

    def lookup(self, cam_toml_name: str, tracker_timestamp: float) -> int | None:
        """Return the video frame index whose timestamp is nearest to tracker_timestamp."""
        if cam_toml_name not in self._tables:
            return None
        timestamps, frames = self._tables[cam_toml_name]
        if not timestamps:
            return None
        idx = bisect.bisect_left(timestamps, tracker_timestamp)
        if idx == 0:
            return frames[0]
        if idx >= len(timestamps):
            return frames[-1]
        if abs(timestamps[idx] - tracker_timestamp) < abs(timestamps[idx - 1] - tracker_timestamp):
            return frames[idx]
        return frames[idx - 1]

    def camera_names(self) -> list[str]:
        return list(self._tables.keys())


# Per-frame, per-camera marker data loaded from marker_projections.csv
MarkerFrameData = dict[int, dict[int, dict[str, tuple[float, float, float, float, bool]]]]

# Tracker frame timestamps
FrameTimestamps = dict[int, float]

# FK bone positions: frame_id → joint_name → (head_world [3], tail_world [3])
BoneWorldData = dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]


# ---------------------------------------------------------------------------
# Camera loading
# ---------------------------------------------------------------------------

def _rodrigues(rvec: list[float]) -> np.ndarray:
    v = np.array(rvec, dtype=float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-10:
        return np.eye(3)
    ax = v / angle
    c, s = math.cos(angle), math.sin(angle)
    tt = 1.0 - c
    x, y, z = ax
    return np.array([
        [tt*x*x+c,   tt*x*y-s*z, tt*x*z+s*y],
        [tt*x*y+s*z, tt*y*y+c,   tt*y*z-s*x],
        [tt*x*z-s*y, tt*y*z+s*x, tt*z*z+c  ],
    ])


def load_cameras(toml_path: Path) -> list[Camera]:
    """Load cameras from a Pose2Sim TOML file.

    cam1 → csv_id=0, matching the 0-based camera_id in marker_projections.csv.
    """
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    cams: list[Camera] = []
    for key, vals in data.items():
        if not key.startswith("cam") or key == "metadata":
            continue
        try:
            one_based = int(key[3:])
        except ValueError:
            continue
        csv_id = one_based - 1
        K = np.array(vals["matrix"], dtype=float)
        R = _rodrigues(vals["rotation"])
        t = np.array(vals["translation"], dtype=float)
        P = K @ np.hstack([R, t.reshape(3, 1)])
        cams.append(Camera(
            csv_id=csv_id,
            toml_name=key,
            display_name=vals.get("name", key),
            K=K, R=R, t=t, P=P,
        ))

    cams.sort(key=lambda c: c.csv_id)
    return cams


# ---------------------------------------------------------------------------
# Skeleton loading and forward kinematics
# ---------------------------------------------------------------------------
#
# FK implementation adapted from scripts/visualize_tracking_results.py.
# Transform chain (non-root joints):
#   T_world[j] = T_world[parent] @ Translation(offset) @ R_rest @ R_anim
# where:
#   R_rest = Rx(x) @ Ry(y) @ Rz(z)   (ZYX Euler, stored as [z, y, x])
#   R_anim = Rodrigues(axis_angle)     for ball/spherical joints
#          = Rodrigues(axis * angle)   for revolute joints
#          = I                         for fixed joints
# Root joint: T_world[root] = R_quat(state) @ R_rest, t = pos(state)
# ---------------------------------------------------------------------------

class SkeletonJoint:
    """Represents a joint in the skeleton hierarchy."""
    def __init__(self, name: str, parent: str | None, joint_type: str,
                 offset: np.ndarray, rest_orientation: np.ndarray,
                 bone_tip_offset: np.ndarray, axis: np.ndarray | None = None):
        self.name = name
        self.parent = parent
        self.joint_type = joint_type
        self.offset = offset
        self.rest_orientation = rest_orientation
        self.bone_tip_offset = bone_tip_offset
        self.axis = axis if axis is not None else np.array([1.0, 0.0, 0.0])


def load_skeleton_structure(yaml_path: Path) -> dict[str, SkeletonJoint]:
    """Load skeleton YAML → dict[joint_name → SkeletonJoint]."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    joints: dict[str, SkeletonJoint] = {}
    for jd in data.get("joints", []):
        name = jd["name"]
        joints[name] = SkeletonJoint(
            name=name,
            parent=jd.get("parent"),
            joint_type=jd.get("type", "fixed"),
            offset=np.array(jd.get("offset", [0.0, 0.0, 0.0])),
            rest_orientation=np.array(jd.get("orientation", [0.0, 0.0, 0.0])),
            bone_tip_offset=np.array(jd.get("bone_tip_offset", [0.0, 0.0, 0.0])),
            axis=np.array(jd["axis"]) if "axis" in jd else None,
        )
    return joints


def _euler_zyx_to_rot(angles: np.ndarray) -> np.ndarray:
    """[z, y, x] → R = Rx(x) @ Ry(y) @ Rz(z)  (matches pinocchio_model_builder.cpp)."""
    z, y, x = float(angles[0]), float(angles[1]), float(angles[2])
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    return np.array([
        [cy*cz,             -cy*sz,              sy   ],
        [sx*sy*cz + cx*sz,  -sx*sy*sz + cx*cz,  -sx*cy],
        [-cx*sy*cz + sx*sz,  cx*sy*sz + sx*cz,   cx*cy],
    ])


def _axis_angle_to_rot(vec: np.ndarray) -> np.ndarray:
    """Rodrigues: vec = axis * angle."""
    angle = float(np.linalg.norm(vec))
    if angle < 1e-10:
        return np.eye(3)
    K = np.array([
        [0,      -vec[2], vec[1]],
        [vec[2],  0,     -vec[0]],
        [-vec[1], vec[0], 0     ],
    ]) / angle
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def _quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Quaternion [w, x, y, z] → 3×3 rotation matrix (normalises first)."""
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-10:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),    1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),    2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


def _compute_joint_transforms(
    skeleton: dict[str, SkeletonJoint],
    state_row: pd.Series,
) -> dict[str, np.ndarray]:
    """Compute 4×4 world transforms for all joints via FK."""
    transforms: dict[str, np.ndarray] = {}

    def process(name: str) -> np.ndarray:
        if name in transforms:
            return transforms[name]
        joint = skeleton[name]
        T = np.eye(4)
        if joint.joint_type == "root":
            px = float(state_row.get("root_position_x", 0.0))
            py = float(state_row.get("root_position_y", 0.0))
            pz = float(state_row.get("root_position_z", 0.0))
            qw = float(state_row.get("root_quaternion_w", 1.0))
            qx = float(state_row.get("root_quaternion_x", 0.0))
            qy = float(state_row.get("root_quaternion_y", 0.0))
            qz = float(state_row.get("root_quaternion_z", 0.0))
            R_rest = _euler_zyx_to_rot(joint.rest_orientation)
            T[:3, :3] = _quat_to_rot(qw, qx, qy, qz) @ R_rest
            T[:3, 3] = [px, py, pz]
        else:
            T_parent = process(joint.parent)
            T[:3, 3] = joint.offset
            R_rest = _euler_zyx_to_rot(joint.rest_orientation)
            jtype = joint.joint_type.lower()
            if jtype in ("ball", "spherical"):
                keys = [f"joint_{name}_angle_{i}" for i in range(3)]
                if all(k in state_row.index for k in keys):
                    aa = np.array([float(state_row[k]) for k in keys])
                    R_anim = _axis_angle_to_rot(aa)
                else:
                    R_anim = np.eye(3)
            elif jtype == "revolute":
                key = f"joint_{name}_angle_0"
                if key in state_row.index:
                    aa = joint.axis * float(state_row[key])
                    R_anim = _axis_angle_to_rot(aa)
                else:
                    R_anim = np.eye(3)
            else:
                R_anim = np.eye(3)
            T[:3, :3] = R_rest @ R_anim
            T = T_parent @ T
        transforms[name] = T
        return T

    for name in skeleton:
        process(name)
    return transforms


def find_skeleton_for_tracking_dir(tdir: Path, fallback: Path | None) -> Path | None:
    """Find skeleton YAML for a tracking dir by scanning TOML files in parent dir.

    Looks for a TOML file in tdir.parent whose [output].directory equals tdir.
    Falls back to `fallback` if not found.
    """
    parent = tdir.parent
    for toml_path in parent.glob("*.toml"):
        try:
            with open(toml_path, "rb") as f:
                cfg = tomllib.load(f)
            out_dir_str = cfg.get("output", {}).get("directory", "")
            if not out_dir_str:
                continue
            out_dir = Path(out_dir_str)
            if out_dir == tdir or out_dir.resolve() == tdir.resolve():
                skel_rel = cfg.get("data", {}).get("skeleton")
                if skel_rel:
                    # Resolve relative to CWD (same as when the tracker ran)
                    skel_path = Path(skel_rel)
                    if not skel_path.is_absolute():
                        skel_path = Path.cwd() / skel_path
                    if skel_path.exists():
                        print(f"  Auto-detected skeleton for {tdir.name}: {skel_path}")
                        return skel_path
        except Exception:
            continue
    if fallback and fallback.exists():
        return fallback
    return None


def load_bone_world_data(
    skeleton: dict[str, SkeletonJoint],
    state_csv: Path,
) -> BoneWorldData:
    """Compute 3D bone world positions per frame using forward kinematics.

    Uses smoothed_state_vectors.csv if present alongside state_csv.
    Returns: dict[frame_id → dict[joint_name → (head_3d, tail_3d)]]
    """
    smoothed = state_csv.parent / "smoothed_state_vectors.csv"
    csv_to_use = smoothed if smoothed.exists() else state_csv
    if csv_to_use != state_csv:
        print(f"    Using smoothed state vectors: {csv_to_use.name}")

    df = pd.read_csv(csv_to_use)

    result: BoneWorldData = {}
    for _, row in df.iterrows():
        frame_id = int(row["tracker_frame_idx"])
        transforms = _compute_joint_transforms(skeleton, row)
        bones: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for jname, T in transforms.items():
            bto = skeleton[jname].bone_tip_offset
            head = T[:3, 3].copy()
            tail = head + T[:3, :3] @ bto
            bone_len = float(np.linalg.norm(tail - head))
            if bone_len > 0.001:  # skip < 1 mm
                bones[jname] = (head, tail)
        result[frame_id] = bones

    return result


# ---------------------------------------------------------------------------
# Tracking data loading
# ---------------------------------------------------------------------------

def load_marker_projections(csv_path: Path) -> tuple[MarkerFrameData, FrameTimestamps]:
    """Load marker_projections.csv."""
    from collections import defaultdict

    marker_data: MarkerFrameData = defaultdict(lambda: defaultdict(dict))
    timestamps: FrameTimestamps = {}

    with open(csv_path) as f:
        header = f.readline().rstrip().split(",")
        col = {name: i for i, name in enumerate(header)}
        for line in f:
            parts = line.rstrip().split(",")
            frame  = int(parts[col["frame"]])
            ts     = float(parts[col["timestamp"]])
            cam_id = int(parts[col["camera_id"]])
            mname  = parts[col["marker_name"]]
            px     = float(parts[col["proj_x"]])
            py     = float(parts[col["proj_y"]])
            ox     = float(parts[col["obs_x"]])
            oy     = float(parts[col["obs_y"]])
            outlier = parts[col["is_outlier"]].strip().lower() == "true"
            marker_data[frame][cam_id][mname] = (px, py, ox, oy, outlier)
            timestamps[frame] = ts

    return (
        {f: dict(cams) for f, cams in marker_data.items()},
        timestamps,
    )


def fill_forward_marker_observations(marker_data: MarkerFrameData) -> MarkerFrameData:
    """For each camera, carry the last known observations forward to frames that have none.

    Cameras that capture at a lower rate than the tracker (e.g. 60 fps cameras in a
    120 fps tracker) only have observations at every 2nd (or 4th, etc.) tracker frame.
    Without fill-forward the visualizer shows empty dots on the in-between frames even
    though the displayed video frame is identical.  Fill-forward makes markers visible
    continuously, matching what is actually visible in the source image.
    """
    all_frames = sorted(marker_data.keys())
    result: MarkerFrameData = {}
    last_per_cam: dict[int, dict[str, tuple]] = {}

    for frame in all_frames:
        # Start from carried-forward data, then overlay this frame's actual observations
        merged: dict[int, dict] = {cam: dict(obs) for cam, obs in last_per_cam.items()}
        for cam, obs in marker_data[frame].items():
            merged[cam] = dict(obs)
            last_per_cam[cam] = obs
        result[frame] = merged

    return result


# ---------------------------------------------------------------------------
# Video file discovery
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV", ".mkv", ".MKV"}


def find_video_files(video_dir: Path, cameras: list[Camera]) -> dict[int, Path]:
    """Match camera csv_ids to video files in video_dir."""
    all_videos = [p for p in video_dir.iterdir() if p.suffix in VIDEO_EXTENSIONS]
    result: dict[int, Path] = {}

    for cam in cameras:
        candidates = [cam.display_name.lower(), cam.toml_name.lower()]
        matched = None
        for vid in all_videos:
            stem = vid.stem.lower()
            if stem in candidates:
                matched = vid
                break
        if matched is None:
            for vid in all_videos:
                stem = vid.stem.lower()
                if any(c in stem for c in candidates):
                    matched = vid
                    break
        if matched is not None:
            result[cam.csv_id] = matched
        else:
            print(f"  [warn] No video file found for camera {cam.display_name} in {video_dir}",
                  file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Bounding box computation
# ---------------------------------------------------------------------------

def compute_sequence_bboxes(
    all_marker_data: list[MarkerFrameData],
    camera_csv_ids: list[int],
    margin: float = 0.10,
) -> dict[int, tuple[int, int, int, int]]:
    """Compute per-camera crop boxes covering all persons' inlier observations."""
    points: dict[int, list[tuple[float, float]]] = {cid: [] for cid in camera_csv_ids}

    for marker_data in all_marker_data:
        for frame_cams in marker_data.values():
            for cam_id, markers in frame_cams.items():
                if cam_id not in points:
                    continue
                for _mname, (px, py, ox, oy, outlier) in markers.items():
                    if not outlier:
                        points[cam_id].append((ox, oy))

    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for cam_id, pts in points.items():
        if not pts:
            continue
        arr = np.array(pts)
        min_x, min_y = arr.min(axis=0)
        max_x, max_y = arr.max(axis=0)
        w = max_x - min_x
        h = max_y - min_y
        bboxes[cam_id] = (
            min_x - margin * w,
            min_y - margin * h,
            max_x + margin * w,
            max_y + margin * h,
        )

    return bboxes


def adjust_bbox_to_aspect(
    bbox: tuple[float, float, float, float],
    target_aspect: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return bbox
    current_aspect = w / h
    if current_aspect > target_aspect:
        new_h = w / target_aspect
        dy = (new_h - h) / 2
        return x1, y1 - dy, x2, y2 + dy
    else:
        new_w = h * target_aspect
        dx = (new_w - w) / 2
        return x1 - dx, y1, x2 + dx, y2


def clamp_bbox(
    bbox: tuple[float, float, float, float],
    vid_w: int,
    vid_h: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        int(max(0, min(x1, vid_w))),
        int(max(0, min(y1, vid_h))),
        int(max(0, min(x2, vid_w))),
        int(max(0, min(y2, vid_h))),
    )


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def draw_bone(
    img: np.ndarray,
    head: tuple[int, int],
    tail: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw a wireframe bone: diamond outline narrow at head, widest at 30%, tapering to tail."""
    hx, hy = head
    tx, ty = tail
    dx, dy = tx - hx, ty - hy
    length = math.hypot(dx, dy)
    if length < 2:
        cv2.circle(img, (int(hx), int(hy)), 2, color, 1)
        return

    perp_x = -dy / length
    perp_y =  dx / length

    half_w = length * 0.12
    wide_x = hx + 0.30 * dx
    wide_y = hy + 0.30 * dy

    left_w  = (int(wide_x + half_w * perp_x), int(wide_y + half_w * perp_y))
    right_w = (int(wide_x - half_w * perp_x), int(wide_y - half_w * perp_y))

    outline = np.array([head, left_w, (int(tx), int(ty)), right_w], dtype=np.int32)
    cv2.polylines(img, [outline], isClosed=True, color=color, thickness=2)


def draw_marker_dot(
    img: np.ndarray,
    x: int,
    y: int,
    color: tuple[int, int, int],
    radius: int = 3,
) -> None:
    cv2.circle(img, (x, y), radius, color, -1)
    # cv2.circle(img, (x, y), radius, (0, 0, 0), 1)


# ---------------------------------------------------------------------------
# Per-frame cell rendering
# ---------------------------------------------------------------------------

def _project_to_cell(
    p_world: np.ndarray,
    cam_P: np.ndarray,
    x1c: float,
    y1c: float,
    sx: float,
    sy: float,
) -> tuple[int, int] | None:
    """Project a 3D world point through camera P matrix, then map to cell coords."""
    q = cam_P @ np.array([p_world[0], p_world[1], p_world[2], 1.0])
    if q[2] < 0.01:
        return None
    px, py = q[0] / q[2], q[1] / q[2]
    return int((px - x1c) * sx), int((py - y1c) * sy)


def render_cell(
    video_frame: np.ndarray,
    cam_id: int,
    frame_id: int,
    video_frame_idx: int,
    cell_w: int,
    cell_h: int,
    crop: tuple[int, int, int, int],
    persons: list[dict],
    cam: Camera,
) -> np.ndarray:
    """Render one camera cell.

    persons: list of {
        mdata:    dict[marker_name → (proj_x, proj_y, obs_x, obs_y, is_outlier)],
        bones_3d: dict[joint_name → (head_world [3], tail_world [3])],
        color:    BGR tuple,
    }
    """
    x1, y1, x2, y2 = crop
    vid_h, vid_w = video_frame.shape[:2]
    x1c = max(0, min(x1, vid_w))
    x2c = max(0, min(x2, vid_w))
    y1c = max(0, min(y1, vid_h))
    y2c = max(0, min(y2, vid_h))

    cropped = video_frame[y1c:y2c, x1c:x2c]
    if cropped.size == 0:
        cropped = video_frame
        x1c, y1c = 0, 0

    cell = cv2.resize(cropped, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)

    crop_w = x2c - x1c
    crop_h = y2c - y1c
    if crop_w <= 0 or crop_h <= 0:
        return cell

    sx = cell_w / crop_w
    sy = cell_h / crop_h

    def to_cell(px: float, py: float) -> tuple[int, int]:
        return int((px - x1c) * sx), int((py - y1c) * sy)

    # Draw skeleton bones for each person (from FK 3D positions)
    for person in persons:
        bones_3d = person.get("bones_3d", {})
        color = person["bone_color"]
        for _jname, (head_3d, tail_3d) in bones_3d.items():
            h2d = _project_to_cell(head_3d, cam.P, x1c, y1c, sx, sy)
            t2d = _project_to_cell(tail_3d, cam.P, x1c, y1c, sx, sy)
            if h2d is None or t2d is None:
                continue
            # Skip bones where both endpoints are far outside the cell
            margin_px = max(cell_w, cell_h) * 0.5
            if (max(h2d[0], t2d[0]) < -margin_px or
                    min(h2d[0], t2d[0]) > cell_w + margin_px or
                    max(h2d[1], t2d[1]) < -margin_px or
                    min(h2d[1], t2d[1]) > cell_h + margin_px):
                continue
            draw_bone(cell, h2d, t2d, color)

    # Draw observation dots (outliers first, inliers on top)
    for person in persons:
        mdata = person["mdata"]
        color = person["marker_color"]
        for _mname, (px, py, ox, oy, outlier) in mdata.items():
            cx, cy = to_cell(ox, oy)
            if 0 <= cx < cell_w and 0 <= cy < cell_h:
                dot_color = OUTLIER_COLOR_BGR if outlier else color
                draw_marker_dot(cell, cx, cy, dot_color)

    # Camera name and frame numbers
    text = f"{cam.display_name}  vid:{video_frame_idx}  trk:{frame_id}"
    cv2.putText(cell, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(cell, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, OVERLAY_TEXT_COLOR, 1, cv2.LINE_AA)

    return cell


# ---------------------------------------------------------------------------
# Grid layout helper
# ---------------------------------------------------------------------------

def grid_dims(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize Posetrak tracking results on camera videos.")
    p.add_argument("--tracking-dir", dest="tracking_dirs", action="append", default=None,
                   metavar="DIR",
                   help="Tracking output directory.  Repeat for multiple persons.")
    p.add_argument("--cameras", default=None, type=Path,
                   help="Pose2Sim camera calibration TOML (not required with --session-db).")
    p.add_argument("--sync", default=None, type=Path,
                   help="Sync JSON file (not required with --session-db).")
    p.add_argument("--skeleton", default=None, type=Path,
                   help="Skeleton YAML (fallback if auto-detection from TOML fails; "
                        "not required with --session-db).")
    p.add_argument("--session-db", default=None, type=str,
                   help="Path to session SQLite database (alternative to --cameras/--sync).")
    p.add_argument("--run-id", default=None, type=str,
                   help="Tracking run UUID (required with --session-db).")
    p.add_argument("--person-id", default=0, type=int,
                   help="Person ID for DB mode (default: 0).")
    p.add_argument("--video-dir", required=True, type=Path,
                   help="Directory containing one video file per camera.")
    p.add_argument("--camera", default=None,
                   help="Render only this camera.  Omit for full mosaic.")
    p.add_argument("--resolution", default="1920x1080",
                   help="Output resolution WxH (default: 1920x1080).")
    p.add_argument("--fps", type=float, default=None,
                   help="Output frame rate.  Defaults to tracker rate.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output MP4 file path.")
    return p.parse_args()


def infer_fps(timestamps: FrameTimestamps) -> float:
    sorted_ts = sorted(timestamps.values())
    if len(sorted_ts) < 2:
        return 60.0
    diffs = [sorted_ts[i+1] - sorted_ts[i] for i in range(min(20, len(sorted_ts)-1))
             if sorted_ts[i+1] > sorted_ts[i]]
    if not diffs:
        return 60.0
    return round(1.0 / (sum(diffs) / len(diffs)))


def _load_cameras_from_db(session_db: str, run_id: str) -> list[Camera]:
    """Load cameras from DB for a given tracking run.

    Returns list of Camera namedtuples sorted by label.
    """
    import sqlite3 as _sqlite3
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.db.load_session import load_cameras_from_session

    conn = _sqlite3.connect(session_db, check_same_thread=False)
    conn.row_factory = _sqlite3.Row
    run_row = conn.execute(
        "SELECT extrinsic_calibration_id FROM tracking_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if run_row is None:
        conn.close()
        raise ValueError(f"No tracking run found: {run_id!r}")
    extrinsic_id = run_row["extrinsic_calibration_id"]
    ext_row = conn.execute(
        "SELECT session_id FROM extrinsic_calibrations WHERE id = ?", (extrinsic_id,)
    ).fetchone()
    session_id = ext_row["session_id"] if ext_row else None
    conn.close()

    if session_id is None:
        raise ValueError("Cannot determine session_id from extrinsic_calibration_id")

    cam_dicts = load_cameras_from_session(session_db, extrinsic_id, session_id)
    cameras = []
    for c in cam_dicts:
        label = c["label"]
        cam_id = c["camera_id"]
        cameras.append(Camera(
            csv_id=cam_id,
            toml_name=label,
            display_name=label,
            K=c["K"],
            R=c["R"],
            t=c["t"],
            P=c["P"],
            dist=c["dist"],
        ))
    return cameras


def _load_sync_from_db(session_db: str, run_id: str) -> SyncTable:
    """Load sync table from DB for a given tracking run."""
    import sqlite3 as _sqlite3
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.db.load_session import load_sync_from_session

    conn = _sqlite3.connect(session_db, check_same_thread=False)
    conn.row_factory = _sqlite3.Row
    run_row = conn.execute(
        "SELECT sync_config_id FROM tracking_runs WHERE id = ?", (run_id,)
    ).fetchone()
    conn.close()
    if run_row is None:
        raise ValueError(f"No tracking run found: {run_id!r}")
    sync_raw = load_sync_from_session(session_db, run_row["sync_config_id"])
    return SyncTable(sync_raw)


def _load_obs_mdata_from_db(
    session_db: str,
    run_id: str,
    person_id: int,
    cameras: list,
    skeleton_yaml: str,
    tracker_timestamps: dict,
    min_confidence: float = 0.1,
) -> dict:
    """Load 2D observations from DB, keyed by tracker step.

    Returns dict[step → dict[csv_id → dict[mname → (obs_x, obs_y, pred_x, pred_y, is_outlier)]]].

    Tries tracking_obs_results first (accurate pred + is_outlier).
    Falls back to pose_observations matched by nearest timestamp (pred=obs, is_outlier=False).
    """
    import bisect
    import math
    import sqlite3 as _sqlite3
    from collections import defaultdict

    conn = _sqlite3.connect(session_db, check_same_thread=False)
    conn.row_factory = _sqlite3.Row

    run_row = conn.execute(
        "SELECT observation_sequence_id, extrinsic_calibration_id, "
        "       active_camera_ids, marker_names "
        "FROM tracking_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    if run_row is None:
        conn.close()
        return {}

    # --- Try tracking_obs_results first ---
    obs_result_row = conn.execute(
        "SELECT obs_blob FROM tracking_obs_results "
        "WHERE run_id = ? AND person_id = ? LIMIT 1",
        (run_id, person_id),
    ).fetchone()

    if obs_result_row is not None:
        # Has tracking_obs_results: load all steps from it
        import json as _json
        cam_labels = _json.loads(run_row["active_camera_ids"])  # sorted labels
        marker_names = _json.loads(run_row["marker_names"])
        n_cams = len(cam_labels)
        n_markers = len(marker_names)

        # Build label → csv_id (same sort order as cameras list)
        label_to_csv_id = {c.display_name: c.csv_id for c in cameras}

        all_rows = conn.execute(
            "SELECT tracker_step, obs_blob FROM tracking_obs_results "
            "WHERE run_id = ? AND person_id = ? ORDER BY tracker_step",
            (run_id, person_id),
        ).fetchall()
        conn.close()

        result: dict[int, dict] = {}
        for row in all_rows:
            step = row["tracker_step"]
            blob = np.frombuffer(bytes(row["obs_blob"]), dtype="<f4")
            if len(blob) != n_cams * n_markers * 8:
                result[step] = {}
                continue
            obs = blob.reshape(n_cams, n_markers, 8)
            frame_cams: dict[int, dict] = {}
            for ci, label in enumerate(cam_labels):
                csv_id = label_to_csv_id.get(label)
                if csv_id is None:
                    continue
                cam_entry: dict[str, tuple] = {}
                for mi, mname in enumerate(marker_names):
                    slot = obs[ci, mi]
                    ox, oy = float(slot[0]), float(slot[1])
                    px, py = float(slot[2]), float(slot[3])
                    is_outlier = bool(slot[6] > 0.5)
                    if math.isnan(ox):
                        continue  # no observation for this slot
                    cam_entry[mname] = (ox, oy, px, py, is_outlier)
                if cam_entry:
                    frame_cams[csv_id] = cam_entry
            result[step] = frame_cams
        return result

    # --- Fallback: pose_observations matched by nearest timestamp ---
    try:
        import yaml as _yaml
        _skel_data = _yaml.safe_load(skeleton_yaml)
    except Exception:
        conn.close()
        return {}

    coco_to_marker: dict[int, str] = {}
    for m in _skel_data.get("markers", []):
        kid = m.get("openpose_keypoint")
        if kid is not None:
            coco_to_marker[int(kid)] = m["name"]
    if not coco_to_marker:
        conn.close()
        return {}

    seq_id = run_row["observation_sequence_id"]
    ext_cal_id = run_row["extrinsic_calibration_id"]

    ext_rows = conn.execute(
        "SELECT ee.camera_instance_id, sc.label "
        "FROM extrinsic_entries ee "
        "JOIN extrinsic_calibrations exc ON exc.id = ee.extrinsic_calibration_id "
        "JOIN session_cameras sc "
        "    ON sc.camera_instance_id = ee.camera_instance_id "
        "    AND sc.session_id = exc.session_id "
        "WHERE ee.extrinsic_calibration_id = ? "
        "ORDER BY sc.label",
        (ext_cal_id,),
    ).fetchall()
    inst_to_csv_id: dict[str, int] = {}
    for i, r in enumerate(ext_rows):
        inst_to_csv_id[r["camera_instance_id"]] = i

    obs_rows = conn.execute(
        "SELECT camera_instance_id, timestamp_s, kp_blob "
        "FROM pose_observations "
        "WHERE sequence_id = ? AND person_id = ? "
        "ORDER BY timestamp_s",
        (seq_id, person_id),
    ).fetchall()
    conn.close()

    cam_ts_data: dict[str, list] = defaultdict(list)
    for row in obs_rows:
        inst_id = row["camera_instance_id"]
        if inst_id not in inst_to_csv_id:
            continue
        ts = float(row["timestamp_s"])
        kp_arr = np.frombuffer(bytes(row["kp_blob"]), dtype="<f4").reshape(-1, 3)
        markers: dict[str, tuple] = {}
        for coco_id, mname in coco_to_marker.items():
            if coco_id < len(kp_arr):
                x, y, conf = float(kp_arr[coco_id, 0]), float(kp_arr[coco_id, 1]), float(kp_arr[coco_id, 2])
                if conf >= min_confidence:
                    markers[mname] = (x, y)
        cam_ts_data[inst_id].append((ts, markers))

    cam_ts_arrays = {iid: [x[0] for x in data] for iid, data in cam_ts_data.items()}

    result2: dict[int, dict] = {}
    for step, step_ts in tracker_timestamps.items():
        frame_cams2: dict[int, dict] = {}
        for inst_id, csv_id in inst_to_csv_id.items():
            data_list = cam_ts_data.get(inst_id)
            if not data_list:
                continue
            ts_list = cam_ts_arrays[inst_id]
            idx = bisect.bisect_left(ts_list, step_ts)
            if idx == 0:
                nearest = 0
            elif idx >= len(ts_list):
                nearest = len(ts_list) - 1
            else:
                nearest = idx if abs(ts_list[idx] - step_ts) < abs(ts_list[idx - 1] - step_ts) else idx - 1
            _, markers = data_list[nearest]
            if markers:
                # pred=None signals caller to use FK projection; is_outlier unknown → False
                frame_cams2[csv_id] = {mname: (ox, oy, None, None, False)
                                       for mname, (ox, oy) in markers.items()}
        result2[step] = frame_cams2
    return result2


def _load_skeleton_from_db(session_db: str, run_id: str) -> tuple[dict, Path]:
    """Load skeleton from DB, write to temp file.

    Returns (skeleton_dict, temp_yaml_path).
    The caller is responsible for cleaning up temp_yaml_path.
    """
    import sqlite3 as _sqlite3
    import tempfile

    conn = _sqlite3.connect(session_db, check_same_thread=False)
    conn.row_factory = _sqlite3.Row
    row = conn.execute(
        "SELECT s.yaml_content FROM tracking_runs tr "
        "JOIN skeletons s ON s.id = tr.skeleton_id "
        "WHERE tr.id = ?",
        (run_id,),
    ).fetchone()
    conn.close()

    if row is None or not row["yaml_content"]:
        raise ValueError(f"No skeleton YAML for run {run_id!r}")

    yaml_content = row["yaml_content"]
    skeleton = load_skeleton_structure.__wrapped__ if hasattr(load_skeleton_structure, "__wrapped__") \
        else load_skeleton_structure

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = Path(f.name)

    skel_dict = load_skeleton_structure(tmp_path)
    return skel_dict, tmp_path


def main() -> None:
    args = parse_args()

    # Validate arguments
    db_mode = args.session_db is not None
    if db_mode:
        if args.run_id is None:
            print("Error: --run-id is required with --session-db", file=sys.stderr)
            sys.exit(1)
        if not (args.tracking_dirs or True):  # tracking_dirs is optional in db mode
            pass
    else:
        if args.tracking_dirs is None:
            print("Error: --tracking-dir is required without --session-db", file=sys.stderr)
            sys.exit(1)
        if args.cameras is None:
            print("Error: --cameras is required without --session-db", file=sys.stderr)
            sys.exit(1)
        if args.sync is None:
            print("Error: --sync is required without --session-db", file=sys.stderr)
            sys.exit(1)

    try:
        out_w, out_h = [int(v) for v in args.resolution.lower().split("x")]
    except Exception:
        print(f"Bad --resolution format: {args.resolution}  (expected WxH)", file=sys.stderr)
        sys.exit(1)

    print("Loading cameras …")
    if db_mode:
        cameras = _load_cameras_from_db(args.session_db, args.run_id)
    else:
        cameras = load_cameras(args.cameras)
    cam_by_name = {c.display_name: c for c in cameras}
    cam_by_name.update({c.toml_name: c for c in cameras})
    print(f"  {len(cameras)} cameras: {[c.display_name for c in cameras]}")

    if args.camera is not None:
        if args.camera not in cam_by_name:
            print(f"Camera '{args.camera}' not found.  Available: {list(cam_by_name)}", file=sys.stderr)
            sys.exit(1)
        active_cameras = [cam_by_name[args.camera]]
        print(f"  Single-camera mode: {active_cameras[0].display_name}")
    else:
        active_cameras = cameras

    print("Loading sync …")
    if db_mode:
        sync = _load_sync_from_db(args.session_db, args.run_id)
    else:
        with open(args.sync) as f:
            sync_raw = json.load(f)
        sync = SyncTable(sync_raw)

    # Load tracking data per person
    print("Loading tracking data …")
    persons_data: list[dict] = []
    all_frames: set[int] = set()
    primary_timestamps: FrameTimestamps | None = None

    # In DB mode, derive timestamps from tracking_results and use skeleton from DB.
    if db_mode:
        import sqlite3 as _sqlite3
        import tempfile as _tempfile
        _conn_db = _sqlite3.connect(args.session_db, check_same_thread=False)
        _conn_db.row_factory = _sqlite3.Row

        bone_color = PERSON_BONE_COLORS_BGR[0]
        marker_color = PERSON_MARKER_COLORS_BGR[0]

        # Load skeleton from DB
        _skel_row = _conn_db.execute(
            "SELECT s.yaml_content FROM tracking_runs tr "
            "JOIN skeletons s ON s.id = tr.skeleton_id "
            "WHERE tr.id = ?",
            (args.run_id,),
        ).fetchone()

        if _skel_row and _skel_row["yaml_content"]:
            with _tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as _tf:
                _tf.write(_skel_row["yaml_content"])
                _skel_tmp = Path(_tf.name)
            skeleton_db = load_skeleton_structure(_skel_tmp)
            _skel_tmp.unlink(missing_ok=True)
            print(f"  Loaded skeleton from DB ({len(skeleton_db)} joints)")
        else:
            skeleton_db = {}
            print("  [warn] No skeleton in DB — bones disabled", file=sys.stderr)

        # Build bone_data from DB state blobs
        _state_rows = _conn_db.execute(
            "SELECT tracker_step, timestamp_s, state FROM tracking_results "
            "WHERE run_id = ? AND person_id = ? AND is_smoothed = 0 "
            "ORDER BY tracker_step",
            (args.run_id, args.person_id),
        ).fetchall()

        # Try smoothed first
        _smoothed_rows = _conn_db.execute(
            "SELECT tracker_step, timestamp_s, state FROM tracking_results "
            "WHERE run_id = ? AND person_id = ? AND is_smoothed = 1 "
            "ORDER BY tracker_step",
            (args.run_id, args.person_id),
        ).fetchall()
        if _smoothed_rows:
            _state_rows = _smoothed_rows
            print("  Using smoothed state from DB")

        _conn_db.close()

        bone_data_db: BoneWorldData = {}
        primary_timestamps_db: FrameTimestamps = {}

        if skeleton_db and _state_rows:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from scripts.db.skeleton_layout import SkeletonLayout as _SkeletonLayout
            # Need skeleton YAML again for SkeletonLayout
            _skel_row2 = None
            _conn_db2 = _sqlite3.connect(args.session_db, check_same_thread=False)
            _conn_db2.row_factory = _sqlite3.Row
            _skel_row2 = _conn_db2.execute(
                "SELECT s.yaml_content FROM tracking_runs tr "
                "JOIN skeletons s ON s.id = tr.skeleton_id WHERE tr.id = ?",
                (args.run_id,),
            ).fetchone()
            _conn_db2.close()

            if _skel_row2:
                _layout = _SkeletonLayout(_skel_row2["yaml_content"])
                print(f"  Computing FK from DB ({len(_state_rows)} frames) …")
                for _srow in _state_rows:
                    _step = _srow["tracker_step"]
                    _ts = _srow["timestamp_s"]
                    primary_timestamps_db[_step] = _ts
                    try:
                        _decoded = _layout.decode_state_blob(bytes(_srow["state"]))
                        _transforms = _layout.compute_joint_transforms(_decoded)
                        _bones: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                        for _jname, _T in _transforms.items():
                            if _jname not in skeleton_db:
                                continue
                            _bto = skeleton_db[_jname].bone_tip_offset
                            _head = _T[:3, 3].copy()
                            _tail = _head + _T[:3, :3] @ _bto
                            if float(np.linalg.norm(_tail - _head)) > 0.001:
                                _bones[_jname] = (_head, _tail)
                        bone_data_db[_step] = _bones
                    except Exception:
                        pass
                print(f"    {len(bone_data_db)} frames")

        # Build marker_data for DB mode: actual observations + FK projections
        _tracking_dirs_db = args.tracking_dirs or []
        if not _tracking_dirs_db:
            mdata_db: MarkerFrameData = {}

            # Load actual 2D observations from DB (keyed by tracker step via timestamp match)
            _skel_yaml_for_obs = _skel_row["yaml_content"] if _skel_row else ""
            _obs_by_step = _load_obs_mdata_from_db(
                args.session_db, args.run_id, args.person_id,
                cameras, _skel_yaml_for_obs, primary_timestamps_db,
            )

            # Check if obs data already includes pred positions (from tracking_obs_results)
            _obs_has_pred = any(
                v[2] is not None
                for step_cams in _obs_by_step.values()
                for cam_markers in step_cams.values()
                for v in cam_markers.values()
            ) if _obs_by_step else False

            if _obs_has_pred:
                # tracking_obs_results available: tuples are (obs_x, obs_y, pred_x, pred_y, is_outlier)
                # Rearrange to MarkerFrameData convention: (proj_x, proj_y, obs_x, obs_y, is_outlier)
                for _step, _obs_cams in _obs_by_step.items():
                    _frame_cams: dict[int, dict] = {}
                    for _csv_id, _obs_markers in _obs_cams.items():
                        _frame_cams[_csv_id] = {
                            _mname: (_px, _py, _ox, _oy, _outlier)
                            for _mname, (_ox, _oy, _px, _py, _outlier) in _obs_markers.items()
                        }
                    mdata_db[_step] = _frame_cams
                for _step in primary_timestamps_db:
                    if _step not in mdata_db:
                        mdata_db[_step] = {}
            elif bone_data_db and cameras:
                # Fallback: use FK for pred positions; obs from pose_observations
                _conn_proj = _sqlite3.connect(args.session_db, check_same_thread=False)
                _conn_proj.row_factory = _sqlite3.Row
                _proj_rows = _conn_proj.execute(
                    "SELECT tracker_step, state FROM tracking_results "
                    "WHERE run_id = ? AND person_id = ? AND is_smoothed = ? "
                    "ORDER BY tracker_step",
                    (args.run_id, args.person_id, 1 if _smoothed_rows else 0),
                ).fetchall()
                _conn_proj.close()

                for _prow in _proj_rows:
                    _step = _prow["tracker_step"]
                    _obs_cams = _obs_by_step.get(_step, {})
                    try:
                        _decoded = _layout.decode_state_blob(bytes(_prow["state"]))
                        _mpos = _layout.compute_marker_positions(_decoded)
                    except Exception:
                        mdata_db[_step] = {}
                        continue
                    _frame_cams = {}
                    for _cam in cameras:
                        _cam_entry: dict[str, tuple] = {}
                        _cam_obs_markers = _obs_cams.get(_cam.csv_id, {})
                        for _mname, _pos3d in _mpos.items():
                            _p_cam = _cam.R @ _pos3d + _cam.t
                            if _p_cam[2] <= 0:
                                continue
                            _pu = _cam.K[0, 0] * _p_cam[0] / _p_cam[2] + _cam.K[0, 2]
                            _pv = _cam.K[1, 1] * _p_cam[1] / _p_cam[2] + _cam.K[1, 2]
                            if _mname in _cam_obs_markers:
                                _ox_raw, _oy_raw = _cam_obs_markers[_mname][:2]
                            else:
                                _ox_raw, _oy_raw = _pu, _pv
                            _cam_entry[_mname] = (_pu, _pv, _ox_raw, _oy_raw, False)
                        if _cam_entry:
                            _frame_cams[_cam.csv_id] = _cam_entry
                    mdata_db[_step] = _frame_cams
            else:
                for _step in primary_timestamps_db:
                    mdata_db[_step] = {}

            persons_data.append({
                "mdata": mdata_db,
                "timestamps": primary_timestamps_db,
                "bone_color": bone_color,
                "marker_color": marker_color,
                "label": f"run:{args.run_id[:8]}",
                "bone_data": bone_data_db,
            })
            all_frames |= set(mdata_db.keys())
            primary_timestamps = primary_timestamps_db
        else:
            # In DB mode with tracking dirs: still collect timestamps for reference
            primary_timestamps = primary_timestamps_db

    tracking_dirs_to_process = [] if db_mode and not args.tracking_dirs else (args.tracking_dirs or [])

    for idx, tdir_str in enumerate(tracking_dirs_to_process):
        tdir = Path(tdir_str)
        bone_color = PERSON_BONE_COLORS_BGR[idx % len(PERSON_BONE_COLORS_BGR)]
        marker_color = PERSON_MARKER_COLORS_BGR[idx % len(PERSON_MARKER_COLORS_BGR)]

        # Load marker projections
        proj_path = tdir / "marker_projections.csv"
        if not proj_path.exists():
            print(f"  [warn] {proj_path} not found — skipping", file=sys.stderr)
            continue
        mdata, timestamps = load_marker_projections(proj_path)
        mdata = fill_forward_marker_observations(mdata)
        if primary_timestamps is None:
            primary_timestamps = timestamps
        all_frames |= set(mdata.keys())

        # Find skeleton for this tracking dir
        # In DB mode, use bone_data already computed from the DB state blobs
        if db_mode and bone_data_db:
            bone_data = bone_data_db
            if idx == 0:
                print(f"  Using FK bone data from DB ({len(bone_data)} frames)")
        else:
            skel_path = find_skeleton_for_tracking_dir(tdir, args.skeleton)
            if skel_path is None:
                print(f"  [warn] Cannot find skeleton for {tdir.name} — bones disabled for this person",
                      file=sys.stderr)
                bone_data = {}
            else:
                skeleton = load_skeleton_structure(skel_path)
                print(f"  Loaded skeleton: {skel_path.name} ({len(skeleton)} joints)")
                state_csv = tdir / "state_vectors.csv"
                if state_csv.exists():
                    print(f"  Computing FK for {tdir.name} …")
                    bone_data = load_bone_world_data(skeleton, state_csv)
                    print(f"    {len(bone_data)} frames, {sum(len(v) for v in bone_data.values())} bone-frames")
                else:
                    print(f"  [warn] {state_csv} not found — bones disabled", file=sys.stderr)
                    bone_data = {}

        persons_data.append({
            "mdata": mdata,
            "timestamps": timestamps,
            "bone_color": bone_color,
            "marker_color": marker_color,
            "label": tdir.name,
            "bone_data": bone_data,
        })
        print(f"  Person {idx}: {tdir.name}  ({len(mdata)} frames, bone color {bone_color}, marker color {marker_color})")

    if not persons_data:
        print("No tracking data loaded.", file=sys.stderr)
        sys.exit(1)

    if primary_timestamps is None:
        print("No timestamps available.", file=sys.stderr)
        sys.exit(1)

    fps = args.fps or infer_fps(primary_timestamps)
    print(f"  Output FPS: {fps}")

    # Compute per-camera bounding boxes
    print("Computing bounding boxes …")
    active_ids = [c.csv_id for c in active_cameras]
    raw_bboxes = compute_sequence_bboxes(
        [p["mdata"] for p in persons_data], active_ids, margin=0.10
    )

    n_cams = len(active_cameras)
    rows, cols = (1, 1) if n_cams == 1 else grid_dims(n_cams)
    cell_w = out_w // cols
    cell_h = out_h // rows
    cell_aspect = cell_w / cell_h

    crops: dict[int, tuple] = {}
    for cam in active_cameras:
        cid = cam.csv_id
        bbox = raw_bboxes.get(cid)
        if bbox is None:
            print(f"  [warn] No observations for camera {cam.display_name} — using full frame")
            crops[cid] = (0, 0, 9999, 9999)
            continue
        adjusted = adjust_bbox_to_aspect(bbox, cell_aspect)
        crops[cid] = adjusted
        w = adjusted[2] - adjusted[0]
        h = adjusted[3] - adjusted[1]
        print(f"  {cam.display_name}: crop ({adjusted[0]:.0f},{adjusted[1]:.0f})"
              f"–({adjusted[2]:.0f},{adjusted[3]:.0f})  {w:.0f}×{h:.0f}px")

    print("Finding video files …")
    video_paths = find_video_files(args.video_dir, active_cameras)
    if not video_paths:
        print("No video files matched — cannot continue.", file=sys.stderr)
        sys.exit(1)

    caps: dict[int, cv2.VideoCapture] = {}
    video_sizes: dict[int, tuple[int, int]] = {}
    for cam in active_cameras:
        cid = cam.csv_id
        if cid not in video_paths:
            continue
        cap = cv2.VideoCapture(str(video_paths[cid]))
        if not cap.isOpened():
            print(f"  [warn] Cannot open {video_paths[cid]}", file=sys.stderr)
            continue
        caps[cid] = cap
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_sizes[cid] = (vid_w, vid_h)
        print(f"  {cam.display_name}: {video_paths[cid].name}  {vid_w}×{vid_h}")

    for cid, (vid_w, vid_h) in video_sizes.items():
        bbox = crops.get(cid, (0.0, 0.0, float(vid_w), float(vid_h)))
        crops[cid] = clamp_bbox(bbox, vid_w, vid_h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"Cannot open output: {args.output}", file=sys.stderr)
        sys.exit(1)

    print(f"\nRendering {len(all_frames)} frames → {args.output}  ({out_w}×{out_h} @ {fps}fps) …")

    blank_cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

    for frame_id in sorted(all_frames):
        timestamp = primary_timestamps.get(frame_id, frame_id / fps)
        grid = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        for cam_idx, cam in enumerate(active_cameras):
            cid = cam.csv_id
            row_g = cam_idx // cols
            col_g = cam_idx % cols
            y0, y1_g = row_g * cell_h, (row_g + 1) * cell_h
            x0, x1_g = col_g * cell_w, (col_g + 1) * cell_w

            if cid not in caps:
                grid[y0:y1_g, x0:x1_g] = blank_cell
                continue

            vid_frame_idx = sync.lookup(cam.toml_name, timestamp)
            if vid_frame_idx is None:
                vid_frame_idx = 0

            cap = caps[cid]
            cap.set(cv2.CAP_PROP_POS_FRAMES, vid_frame_idx)
            ret, vid_frame = cap.read()
            if not ret:
                grid[y0:y1_g, x0:x1_g] = blank_cell
                continue

            persons_for_cell: list[dict] = []
            for person in persons_data:
                mdata_frame = person["mdata"].get(frame_id, {})
                mdata_cam   = mdata_frame.get(cid, {})
                bones_3d    = person["bone_data"].get(frame_id, {})
                persons_for_cell.append({
                    "mdata":    mdata_cam,
                    "bones_3d": bones_3d,
                    "bone_color":    person["bone_color"],
                    "marker_color":  person["marker_color"],
                })

            crop = crops.get(cid, (0, 0,
                                   video_sizes.get(cid, (1, 1))[0],
                                   video_sizes.get(cid, (1, 1))[1]))
            cell = render_cell(
                video_frame=vid_frame,
                cam_id=cid,
                frame_id=frame_id,
                video_frame_idx=vid_frame_idx,
                cell_w=cell_w,
                cell_h=cell_h,
                crop=crop,
                persons=persons_for_cell,
                cam=cam,
            )

            grid[y0:y1_g, x0:x1_g] = cell

        label = f"frame {frame_id}  t={timestamp:.3f}s"
        cv2.putText(grid, label, (10, out_h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(grid, label, (10, out_h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, OVERLAY_TEXT_COLOR, 1, cv2.LINE_AA)

        writer.write(grid)

        if frame_id % 50 == 0:
            print(f"  frame {frame_id}/{max(all_frames)}", end="\r", flush=True)

    print()

    for cap in caps.values():
        cap.release()
    writer.release()
    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()
