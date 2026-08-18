#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""
visualize_tracking.py — Overlay UKF tracking results on source camera videos.

Produces a mosaic (grid) video with all cameras arranged in a grid.
Each camera view is auto-cropped to the area containing active detections.
Person segmentation masks (if available in the DB) are blended on each frame.
Detected keypoints and the tracked skeleton are drawn per person, using the
same DAVIS palette as the posetrak app.

Usage
-----
List available tracking runs in a session database::

    python3 visualize_tracking.py --session-db SESSION.db

Render a tracking run (UUID prefix is accepted)::

    python3 visualize_tracking.py \\
        --session-db SESSION.db \\
        --run-id RUN_UUID_PREFIX \\
        --output result.mp4

Options::

    --run-id      Tracking run UUID or prefix.  Repeat to overlay multiple
                  runs (e.g. two persons tracked separately).
    --camera      Render only this camera label instead of full mosaic.
    --resolution  Output WxH (default 1920x1080).
    --fps         Override output frame rate (default: inferred from tracker).
    --no-masks    Skip segmentation mask overlay even if masks are available.
    --no-bones    Skip 3-D skeleton bone overlay.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# DAVIS palette — RGB and BGR, matching the posetrak app exactly
# ---------------------------------------------------------------------------

_DAVIS_RGB: list[tuple[int, int, int]] = [
    (0,   0,   0  ),  # 0  background (not drawn)
    (240, 80,  80 ),  # 1  red-ish
    (80,  200, 120),  # 2  green-ish
    (80,  120, 240),  # 3  blue-ish
    (240, 200, 60 ),  # 4  yellow
    (180, 80,  240),  # 5  purple
    (60,  220, 220),  # 6  cyan
    (240, 140, 60 ),  # 7  orange
    (160, 160, 160),  # 8  gray
    (120, 240, 80 ),  # 9  lime
    (240, 60,  160),  # 10 pink
    (60,  160, 240),  # 11 sky blue
    (200, 200, 60 ),  # 12 olive
    (240, 100, 120),  # 13 salmon
    (100, 240, 200),  # 14 aqua
    (200, 140, 240),  # 15 lavender
]

_DAVIS_BGR: list[tuple[int, int, int]] = [(b, g, r) for r, g, b in _DAVIS_RGB]

_MASK_ALPHA = 0.45
_OUTLIER_BGR = (110, 110, 110)
_TEXT_BGR    = (255, 255, 255)


def _davis_bgr(label: int) -> tuple[int, int, int]:
    if 1 <= label < len(_DAVIS_BGR):
        return _DAVIS_BGR[label]
    return (160, 160, 160)


def _build_palette_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i, bgr in enumerate(_DAVIS_BGR):
        if i >= 256:
            break
        lut[i] = bgr
    return lut

_PALETTE_LUT = _build_palette_lut()


def blend_mask_davis(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a labeled uint8 mask onto *frame_bgr* using the DAVIS palette."""
    if mask.shape[:2] != frame_bgr.shape[:2]:
        mask = cv2.resize(mask, (frame_bgr.shape[1], frame_bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    colours = _PALETTE_LUT[mask]
    fg = mask > 0
    out = frame_bgr.copy()
    out[fg] = (
        out[fg].astype(np.float32) * (1 - _MASK_ALPHA)
        + colours[fg].astype(np.float32) * _MASK_ALPHA
    ).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Camera(NamedTuple):
    label: str
    csv_id: int        # 0-based index matching active_camera_ids order
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    P: np.ndarray      # K @ [R | t]
    dist: np.ndarray | None = None


class SyncTable:
    """Tracker timestamp → video frame index per camera."""

    def __init__(self, sync_data: dict):
        self._tables: dict[str, tuple[list[float], list[int], float]] = {}
        for cam_label, info in sync_data.items():
            pts = info.get("syncpoints", [])
            self._tables[cam_label] = (
                [sp["timestamp"] for sp in pts],
                [sp["frame"]     for sp in pts],
                float(info.get("fps", 0.0)),
            )

    def lookup(self, cam_label: str, timestamp: float) -> int | None:
        if cam_label not in self._tables:
            return None
        timestamps, frames, fps = self._tables[cam_label]
        if not timestamps:
            return None
        idx = bisect.bisect_right(timestamps, timestamp)
        if idx == 0:
            anchor_ts, anchor_frame = timestamps[0], frames[0]
        elif idx >= len(timestamps):
            anchor_ts, anchor_frame = timestamps[-1], frames[-1]
        else:
            anchor_ts, anchor_frame = timestamps[idx - 1], frames[idx - 1]
        if fps > 0:
            return anchor_frame + round((timestamp - anchor_ts) * fps)
        return frames[min(idx, len(frames) - 1)]

    def camera_labels(self) -> list[str]:
        return list(self._tables.keys())


# ---------------------------------------------------------------------------
# Skeleton loading and forward kinematics
# ---------------------------------------------------------------------------

class SkeletonJoint:
    def __init__(self, name: str, parent: str | None, joint_type: str,
                 offset: np.ndarray, rest_orientation: np.ndarray,
                 bone_tip_offset: np.ndarray, axis: np.ndarray | None = None):
        self.name            = name
        self.parent          = parent
        self.joint_type      = joint_type
        self.offset          = offset
        self.rest_orientation= rest_orientation
        self.bone_tip_offset = bone_tip_offset
        self.axis            = axis if axis is not None else np.array([1.0, 0.0, 0.0])


def _load_skeleton_from_yaml_str(yaml_content: str) -> dict[str, SkeletonJoint]:
    import yaml
    data = yaml.safe_load(yaml_content)
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
    z, y, x = float(angles[0]), float(angles[1]), float(angles[2])
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    return np.array([
        [cy*cz,             -cy*sz,              sy    ],
        [sx*sy*cz + cx*sz,  -sx*sy*sz + cx*cz,  -sx*cy],
        [-cx*sy*cz + sx*sz,  cx*sy*sz + sx*cz,   cx*cy],
    ])


def _axis_angle_to_rot(vec: np.ndarray) -> np.ndarray:
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
    state_blob: bytes,
    layout,
) -> dict[str, np.ndarray]:
    """Decode state blob via SkeletonLayout and run FK."""
    try:
        decoded = layout.decode_state_blob(state_blob)
        return layout.compute_joint_transforms(decoded)
    except Exception:
        return {}


BoneWorldData = dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _list_tracking_runs(session_db: str) -> None:
    conn = _open_db(session_db)
    rows = conn.execute(
        """
        SELECT tr.id, tr.ran_at, sp.person_name,
               dr.detector_model, dr.pose_model,
               pos.time_start_s, pos.time_end_s,
               cap.label AS capture_name,
               t.name AS trial_name
        FROM tracking_runs tr
        JOIN pose_observation_sequences pos ON pos.id = tr.observation_sequence_id
        JOIN sequence_persons sp ON sp.sequence_id = pos.id AND sp.person_id = 0
        LEFT JOIN detection_runs dr ON dr.id = pos.detection_run_id
        LEFT JOIN trials t ON t.id = dr.trial_id
        LEFT JOIN captures cap ON cap.id = dr.shot_id
        ORDER BY tr.ran_at DESC
        LIMIT 40
        """
    ).fetchall()
    conn.close()

    if not rows:
        print("No tracking runs found in this database.")
        return

    print(f"\n{'RUN ID (prefix)':<14}  {'Ran at':<20}  {'Person':<12}  "
          f"{'Detector':<20}  {'Capture / Trial'}")
    print("-" * 90)
    for r in rows:
        cap_trial = f"{r['capture_name'] or '?'} / {r['trial_name'] or '?'}"
        det = f"{r['detector_model'] or ''} + {r['pose_model'] or ''}"
        print(f"{r['id'][:13]:<14}  {str(r['ran_at'])[:19]:<20}  "
              f"{r['person_name'] or '?':<12}  {det:<20}  {cap_trial}")
    print(f"\n  Use --run-id <prefix> to render.  Repeat for multiple persons.")


def _resolve_run_id(conn: sqlite3.Connection, prefix: str) -> str:
    row = conn.execute(
        "SELECT id FROM tracking_runs WHERE id LIKE ? ORDER BY ran_at DESC LIMIT 1",
        (prefix + "%",)
    ).fetchone()
    if row is None:
        raise ValueError(f"No tracking run found matching prefix: {prefix!r}")
    return row["id"]


def _find_latest_runs_for_trial(
    conn: sqlite3.Connection,
    run_id: str,
) -> list[tuple[str, str]]:
    """Return (run_id, person_name) pairs for the latest tracking run per person
    in the same trial as *run_id*.  The result is sorted by person_name and
    always includes *run_id* itself.

    Uses the trial of the detection run that backs the given tracking run's
    observation sequence.  Returns [(run_id, person_name)] if no trial is set.
    """
    trial_row = conn.execute(
        """
        SELECT dr.trial_id
        FROM tracking_runs tr
        JOIN pose_observation_sequences pos ON pos.id = tr.observation_sequence_id
        JOIN detection_runs dr ON dr.id = pos.detection_run_id
        WHERE tr.id = ?
        """,
        (run_id,),
    ).fetchone()

    trial_id = trial_row["trial_id"] if trial_row else None
    if not trial_id:
        # No trial linkage — just return the given run
        sp = conn.execute(
            """
            SELECT sp.person_name FROM tracking_runs tr
            JOIN pose_observation_sequences pos ON pos.id = tr.observation_sequence_id
            JOIN sequence_persons sp ON sp.sequence_id = pos.id AND sp.person_id = 0
            WHERE tr.id = ?
            """,
            (run_id,),
        ).fetchone()
        return [(run_id, sp["person_name"] if sp else "unknown")]

    # For each person in the trial, pick the single latest tracking run overall
    rows = conn.execute(
        """
        SELECT tr.id AS run_id, sp.person_name, tr.ran_at
        FROM tracking_runs tr
        JOIN pose_observation_sequences pos ON pos.id = tr.observation_sequence_id
        JOIN sequence_persons sp ON sp.sequence_id = pos.id AND sp.person_id = 0
        JOIN detection_runs dr ON dr.id = pos.detection_run_id
        WHERE dr.trial_id = ?
          AND tr.ran_at = (
              SELECT MAX(t2.ran_at)
              FROM tracking_runs t2
              JOIN pose_observation_sequences p2 ON p2.id = t2.observation_sequence_id
              JOIN sequence_persons s2 ON s2.sequence_id = p2.id AND s2.person_id = 0
              JOIN detection_runs d2 ON d2.id = p2.detection_run_id
              WHERE d2.trial_id = ? AND s2.person_name = sp.person_name
          )
        ORDER BY sp.person_name
        """,
        (trial_id, trial_id),
    ).fetchall()

    # Deduplicate in case of tied ran_at (keep first per person)
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for r in rows:
        if r["person_name"] not in seen:
            seen.add(r["person_name"])
            result.append((r["run_id"], r["person_name"]))
    return result


def _load_cameras_from_db(
    session_db: str,
    conn: sqlite3.Connection,
    extrinsic_id: str,
) -> list[Camera]:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from posetrak.db.load_session import load_cameras_from_session

    ext_row = conn.execute(
        "SELECT session_id FROM extrinsic_calibrations WHERE id = ?",
        (extrinsic_id,)
    ).fetchone()
    session_id = ext_row["session_id"] if ext_row else None
    if session_id is None:
        raise ValueError("Cannot determine session_id from extrinsic_calibration_id")

    cam_dicts = load_cameras_from_session(session_db, extrinsic_id, session_id)
    cameras = []
    for i, c in enumerate(cam_dicts):
        cameras.append(Camera(
            label=c["label"],
            csv_id=i,
            K=c["K"], R=c["R"], t=c["t"], P=c["P"],
            dist=c.get("dist"),
        ))
    return cameras


def _load_sync_from_db(
    conn: sqlite3.Connection,
    sync_config_id: str,
) -> SyncTable:
    """Load sync table from DB.

    sync_points.shot_video_id is capture_videos.id; join through capture_videos
    to get camera_instances.label and actual_fps.
    """
    from collections import defaultdict
    rows = conn.execute(
        """
        SELECT sp.video_frame, sp.timestamp_s, cv.actual_fps, ci.label
        FROM sync_points sp
        JOIN capture_videos cv ON cv.id = sp.shot_video_id
        JOIN camera_instances ci ON ci.id = cv.camera_instance_id
        WHERE sp.sync_config_id = ?
        ORDER BY ci.label, sp.video_frame
        """,
        (sync_config_id,)
    ).fetchall()
    data: dict[str, dict] = defaultdict(lambda: {"fps": 0.0, "syncpoints": []})
    for row in rows:
        label = row["label"]
        data[label]["fps"] = float(row["actual_fps"] or 0.0)
        data[label]["syncpoints"].append({
            "frame": int(row["video_frame"]),
            "timestamp": float(row["timestamp_s"]),
        })
    return SyncTable(dict(data))


def _load_bone_data(
    conn: sqlite3.Connection,
    run_id: str,
    person_id: int,
    skeleton: dict[str, SkeletonJoint],
    layout,
) -> tuple[BoneWorldData, dict[int, float]]:
    """Compute FK bone positions and timestamps from tracking_results."""
    rows = conn.execute(
        """
        SELECT tracker_step, timestamp_s, state FROM tracking_results
        WHERE run_id = ? AND person_id = ? AND is_smoothed = 1
        ORDER BY tracker_step
        """,
        (run_id, person_id)
    ).fetchall()
    if not rows:
        rows = conn.execute(
            """
            SELECT tracker_step, timestamp_s, state FROM tracking_results
            WHERE run_id = ? AND person_id = ? AND is_smoothed = 0
            ORDER BY tracker_step
            """,
            (run_id, person_id)
        ).fetchall()

    bone_data: BoneWorldData = {}
    timestamps: dict[int, float] = {}

    for row in rows:
        step = row["tracker_step"]
        timestamps[step] = float(row["timestamp_s"])
        state_blob = bytes(row["state"])
        try:
            transforms = _compute_joint_transforms(skeleton, state_blob, layout)
        except Exception:
            bone_data[step] = {}
            continue
        bones: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for jname, T in transforms.items():
            if jname not in skeleton:
                continue
            bto = skeleton[jname].bone_tip_offset
            head = T[:3, 3].copy()
            tail = head + T[:3, :3] @ bto
            if float(np.linalg.norm(tail - head)) > 0.001:
                bones[jname] = (head, tail)
        bone_data[step] = bones

    return bone_data, timestamps


def _load_obs_data(
    conn: sqlite3.Connection,
    run_id: str,
    person_id: int,
    cam_label_to_csv_id: dict[str, int],
    extrinsic_id: str,
) -> dict[int, dict[int, dict[str, tuple]]]:
    """Load obs from tracking_obs_results.

    Returns step → csv_id → marker_name → (obs_x, obs_y, pred_x, pred_y, is_outlier)
    """
    run_row = conn.execute(
        "SELECT active_camera_ids, marker_names FROM tracking_runs WHERE id = ?",
        (run_id,)
    ).fetchone()
    cam_labels: list[str] = json.loads(run_row["active_camera_ids"])
    marker_names: list[str] = json.loads(run_row["marker_names"])
    n_cams, n_markers = len(cam_labels), len(marker_names)

    ci_rows = conn.execute(
        """
        SELECT ci.label AS ci_label, sc.label AS sc_label
        FROM extrinsic_entries ee
        JOIN extrinsic_calibrations exc ON exc.id = ee.extrinsic_calibration_id
        JOIN session_cameras sc
            ON sc.camera_instance_id = ee.camera_instance_id
           AND sc.session_id = exc.session_id
        JOIN camera_instances ci ON ci.id = ee.camera_instance_id
        WHERE ee.extrinsic_calibration_id = ?
        """,
        (extrinsic_id,)
    ).fetchall()
    ci_label_to_sc_label: dict[str, str] = {r["ci_label"]: r["sc_label"] for r in ci_rows}

    all_rows = conn.execute(
        """
        SELECT tracker_step, obs_blob FROM tracking_obs_results
        WHERE run_id = ? AND person_id = ?
        ORDER BY tracker_step
        """,
        (run_id, person_id)
    ).fetchall()

    result: dict[int, dict[int, dict[str, tuple]]] = {}
    expected = n_cams * n_markers * 8
    for row in all_rows:
        step = row["tracker_step"]
        blob = np.frombuffer(bytes(row["obs_blob"]), dtype="<f4")
        if len(blob) != expected:
            continue
        obs = blob.reshape(n_cams, n_markers, 8)
        step_data: dict[int, dict[str, tuple]] = {}
        for ci, cl in enumerate(cam_labels):
            sc_label = ci_label_to_sc_label.get(cl) or cl
            csv_id = cam_label_to_csv_id.get(sc_label)
            if csv_id is None:
                continue
            cam_markers: dict[str, tuple] = {}
            for mi, mname in enumerate(marker_names):
                slot = obs[ci, mi]
                obs_x, obs_y   = float(slot[0]), float(slot[1])
                pred_x, pred_y = float(slot[2]), float(slot[3])
                is_outlier = bool(slot[6] > 0.5)
                if not math.isnan(obs_x):
                    cam_markers[mname] = (obs_x, obs_y, pred_x, pred_y, is_outlier)
            if cam_markers:
                step_data[csv_id] = cam_markers
        result[step] = step_data
    return result


def _get_person_track_id(
    conn: sqlite3.Connection,
    run_id: str,
    person_name: str,
) -> int | None:
    """Return the mask label (track_id) for a person via detection_track_assignments."""
    row = conn.execute(
        """
        SELECT dta.track_id
        FROM detection_track_assignments dta
        JOIN pose_observation_sequences pos
            ON pos.detection_run_id = dta.detection_run_id
        JOIN tracking_runs tr ON tr.observation_sequence_id = pos.id
        WHERE tr.id = ? AND dta.person_name = ?
        LIMIT 1
        """,
        (run_id, person_name)
    ).fetchone()
    return int(row["track_id"]) if row else None


def _find_seg_source(
    conn: sqlite3.Connection,
    shot_video_id: str,
    frame_range: tuple[int, int],
) -> str | None:
    """Return the seg_quality_run_id with the most masks covering frame_range."""
    f0, f1 = frame_range
    row = conn.execute(
        """
        SELECT seg_quality_run_id, COUNT(*) n
        FROM seg_masks
        WHERE shot_video_id = ? AND frame_idx BETWEEN ? AND ?
        GROUP BY seg_quality_run_id
        ORDER BY n DESC LIMIT 1
        """,
        (shot_video_id, f0, f1)
    ).fetchone()
    return row["seg_quality_run_id"] if row else None


def _load_seg_mask(
    conn: sqlite3.Connection,
    seg_quality_run_id: str,
    shot_video_id: str,
    frame_idx: int,
) -> np.ndarray | None:
    row = conn.execute(
        "SELECT mask_blob FROM seg_masks "
        "WHERE seg_quality_run_id = ? AND shot_video_id = ? AND frame_idx = ?",
        (seg_quality_run_id, shot_video_id, frame_idx)
    ).fetchone()
    if row is None:
        return None
    buf = np.frombuffer(bytes(row["mask_blob"]), dtype=np.uint8)
    mask = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def _get_shot_video_ids(
    conn: sqlite3.Connection,
    shot_id: str,
) -> dict[str, str]:
    """Return {camera_label → shot_video_id} for a shot."""
    rows = conn.execute(
        """
        SELECT ci.label, cv.id AS svid
        FROM capture_videos cv
        JOIN camera_instances ci ON ci.id = cv.camera_instance_id
        WHERE cv.shot_id = ?
        """,
        (shot_id,)
    ).fetchall()
    return {r["label"]: r["svid"] for r in rows}


def _get_video_paths(
    conn: sqlite3.Connection,
    shot_id: str,
) -> dict[str, str]:
    """Return {camera_label → file_path} for a shot."""
    rows = conn.execute(
        """
        SELECT ci.label, cv.file_path
        FROM capture_videos cv
        JOIN camera_instances ci ON ci.id = cv.camera_instance_id
        WHERE cv.shot_id = ?
        """,
        (shot_id,)
    ).fetchall()
    return {r["label"]: r["file_path"] for r in rows}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _project_to_cell(
    p_world: np.ndarray,
    cam_P: np.ndarray,
    x1c: float, y1c: float,
    sx: float,  sy: float,
) -> tuple[int, int] | None:
    q = cam_P @ np.array([p_world[0], p_world[1], p_world[2], 1.0])
    if q[2] < 0.01:
        return None
    px, py = q[0] / q[2], q[1] / q[2]
    return int((px - x1c) * sx), int((py - y1c) * sy)


def _draw_bone(
    img: np.ndarray,
    head: tuple[int, int],
    tail: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    hx, hy = head
    tx, ty = tail
    dx, dy = tx - hx, ty - hy
    length = math.hypot(dx, dy)
    if length < 2:
        cv2.circle(img, (int(hx), int(hy)), 2, color, 1)
        return
    perp_x, perp_y = -dy / length, dx / length
    half_w = length * 0.12
    wide_x = hx + 0.30 * dx
    wide_y = hy + 0.30 * dy
    left_w  = (int(wide_x + half_w * perp_x), int(wide_y + half_w * perp_y))
    right_w = (int(wide_x - half_w * perp_x), int(wide_y - half_w * perp_y))
    pts = np.array([head, left_w, (int(tx), int(ty)), right_w], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)


def render_cell(
    video_frame: np.ndarray,
    cam: Camera,
    crop: tuple[int, int, int, int],
    cell_w: int,
    cell_h: int,
    persons: list[dict],
    frame_label: str = "",
) -> np.ndarray:
    """Render one camera cell with mask, bones, and observation dots."""
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
        x2c, y2c = vid_w, vid_h

    cell = cv2.resize(cropped, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)

    crop_w = x2c - x1c
    crop_h = y2c - y1c
    if crop_w <= 0 or crop_h <= 0:
        return cell

    sx = cell_w / crop_w
    sy = cell_h / crop_h

    # Skeleton bones (behind observations)
    for person in persons:
        color = person["color"]
        bones_3d = person.get("bones_3d", {})
        for _jname, (head_3d, tail_3d) in bones_3d.items():
            h2d = _project_to_cell(head_3d, cam.P, x1c, y1c, sx, sy)
            t2d = _project_to_cell(tail_3d, cam.P, x1c, y1c, sx, sy)
            if h2d is None or t2d is None:
                continue
            margin = max(cell_w, cell_h) * 0.5
            if (max(h2d[0], t2d[0]) < -margin or min(h2d[0], t2d[0]) > cell_w + margin or
                    max(h2d[1], t2d[1]) < -margin or min(h2d[1], t2d[1]) > cell_h + margin):
                continue
            _draw_bone(cell, h2d, t2d, color)

    # Observation dots (outliers first so inliers paint on top)
    def _to_cell(px: float, py: float) -> tuple[int, int]:
        return int((px - x1c) * sx), int((py - y1c) * sy)

    for pass_outlier in (True, False):
        for person in persons:
            color = person["color"]
            for _mname, (ox, oy, _px, _py, is_outlier) in person.get("obs", {}).items():
                if is_outlier != pass_outlier:
                    continue
                cx, cy = _to_cell(ox, oy)
                if 0 <= cx < cell_w and 0 <= cy < cell_h:
                    dot_color = _OUTLIER_BGR if is_outlier else color
                    cv2.circle(cell, (cx, cy), 4, dot_color, -1)

    if frame_label:
        cv2.putText(cell, frame_label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(cell, frame_label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, _TEXT_BGR, 1, cv2.LINE_AA)

    return cell


# ---------------------------------------------------------------------------
# Crop / aspect helpers
# ---------------------------------------------------------------------------

def _compute_bboxes(
    obs_by_step_by_cam: dict[int, dict[int, dict]],
    cam_csv_ids: list[int],
    margin: float = 0.12,
) -> dict[int, tuple[float, float, float, float]]:
    points: dict[int, list[tuple[float, float]]] = {c: [] for c in cam_csv_ids}
    for step_cams in obs_by_step_by_cam.values():
        for csv_id, markers in step_cams.items():
            if csv_id not in points:
                continue
            for ox, oy, _px, _py, is_outlier in markers.values():
                if not is_outlier:
                    points[csv_id].append((ox, oy))
    bboxes: dict[int, tuple] = {}
    for cid, pts in points.items():
        if not pts:
            continue
        arr = np.array(pts)
        mn, mx = arr.min(axis=0), arr.max(axis=0)
        w, h = mx[0] - mn[0], mx[1] - mn[1]
        bboxes[cid] = (mn[0] - margin*w, mn[1] - margin*h,
                       mx[0] + margin*w, mx[1] + margin*h)
    return bboxes


def _adjust_aspect(bbox: tuple, target: float) -> tuple:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return bbox
    if w / h > target:
        new_h = w / target
        dy = (new_h - h) / 2
        return x1, y1 - dy, x2, y2 + dy
    new_w = h * target
    dx = (new_w - w) / 2
    return x1 - dx, y1, x2 + dx, y2


def _clamp_bbox(bbox: tuple, vid_w: int, vid_h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (int(max(0, min(x1, vid_w))),
            int(max(0, min(y1, vid_h))),
            int(max(0, min(x2, vid_w))),
            int(max(0, min(y2, vid_h))))


def _infer_fps(timestamps: dict[int, float]) -> float:
    s = sorted(timestamps.values())
    if len(s) < 2:
        return 60.0
    diffs = [s[i+1] - s[i] for i in range(min(20, len(s)-1)) if s[i+1] > s[i]]
    return round(1.0 / (sum(diffs) / len(diffs))) if diffs else 60.0


def _grid_dims(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    return math.ceil(n / cols), cols


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize posetrak tracking results on source camera videos."
    )
    p.add_argument("--session-db", required=True,
                   help="Path to session SQLite database.")
    p.add_argument("--run-id", action="append", default=None, dest="run_ids",
                   metavar="RUN_ID",
                   help="Tracking run UUID or unique prefix. "
                        "Repeat for multiple persons. "
                        "Omit to list available runs.")
    p.add_argument("--output", default=None, type=Path,
                   help="Output MP4 file path (required when --run-id is given).")
    p.add_argument("--camera", default=None,
                   help="Render only this camera label (default: full mosaic).")
    p.add_argument("--resolution", default="1920x1080",
                   help="Output WxH (default: 1920x1080).")
    p.add_argument("--fps", type=float, default=None,
                   help="Override output FPS.")
    p.add_argument("--all-persons", action="store_true",
                   help="Auto-load the latest tracking run for every person in the "
                        "same trial as the first --run-id.  Ignored when multiple "
                        "--run-id values are already given.")
    p.add_argument("--no-masks", action="store_true",
                   help="Skip segmentation mask overlay.")
    p.add_argument("--no-bones", action="store_true",
                   help="Skip 3-D skeleton bone overlay.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.run_ids:
        _list_tracking_runs(args.session_db)
        return

    if args.output is None:
        print("Error: --output is required when --run-id is given.", file=sys.stderr)
        sys.exit(1)

    try:
        out_w, out_h = [int(v) for v in args.resolution.lower().split("x")]
    except Exception:
        print(f"Bad --resolution: {args.resolution}", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from posetrak.db.skeleton_layout import SkeletonLayout

    conn = _open_db(args.session_db)

    # Resolve run IDs, then optionally expand to all persons in the trial
    run_ids = [_resolve_run_id(conn, r) for r in args.run_ids]

    if args.all_persons and len(run_ids) == 1:
        pairs = _find_latest_runs_for_trial(conn, run_ids[0])
        run_ids = [rid for rid, _name in pairs]
        print(f"Auto-expanded to {len(run_ids)} persons: "
              f"{[name for _, name in pairs]}")

    # Load metadata from first run (all runs assumed same shot/calib)
    first_run = conn.execute(
        "SELECT observation_sequence_id, extrinsic_calibration_id, "
        "       sync_config_id, skeleton_id "
        "FROM tracking_runs WHERE id = ?",
        (run_ids[0],)
    ).fetchone()
    if first_run is None:
        print(f"Run not found: {run_ids[0]}", file=sys.stderr)
        sys.exit(1)

    extrinsic_id  = first_run["extrinsic_calibration_id"]
    sync_cfg_id   = first_run["sync_config_id"]

    # Get shot_id from observation sequence
    obs_seq = conn.execute(
        "SELECT shot_id, detection_run_id FROM pose_observation_sequences WHERE id = ?",
        (first_run["observation_sequence_id"],)
    ).fetchone()
    shot_id = obs_seq["shot_id"]

    print("Loading cameras …")
    cameras = _load_cameras_from_db(args.session_db, conn, extrinsic_id)
    cam_by_label   = {c.label: c for c in cameras}
    cam_label_to_csv = {c.label: c.csv_id for c in cameras}
    print(f"  {len(cameras)} cameras: {[c.label for c in cameras]}")

    if args.camera:
        if args.camera not in cam_by_label:
            print(f"Camera '{args.camera}' not found. Available: {list(cam_by_label)}", file=sys.stderr)
            sys.exit(1)
        active_cameras = [cam_by_label[args.camera]]
    else:
        active_cameras = cameras

    print("Loading sync …")
    sync = _load_sync_from_db(conn, sync_cfg_id)

    print("Loading video file paths …")
    video_paths = _get_video_paths(conn, shot_id)
    label_to_svid = _get_shot_video_ids(conn, shot_id)
    for label, path in video_paths.items():
        print(f"  {label}: {path}")

    # Load skeleton (shared across runs if same skeleton_id)
    skeleton_id = first_run["skeleton_id"]
    skel_row = conn.execute(
        "SELECT yaml_content FROM skeletons WHERE id = ?", (skeleton_id,)
    ).fetchone()
    skeleton: dict = {}
    layout = None
    if skel_row and skel_row["yaml_content"]:
        skeleton = _load_skeleton_from_yaml_str(skel_row["yaml_content"])
        layout   = SkeletonLayout(skel_row["yaml_content"])
        print(f"  Skeleton: {len(skeleton)} joints")

    # Load per-run / per-person data
    print("\nLoading tracking data …")
    persons_data: list[dict] = []
    primary_timestamps: dict[int, float] = {}
    all_steps: set[int] = set()

    for run_id in run_ids:
        run_row = conn.execute(
            "SELECT observation_sequence_id, extrinsic_calibration_id "
            "FROM tracking_runs WHERE id = ?",
            (run_id,)
        ).fetchone()
        seq_id = run_row["observation_sequence_id"]

        pid_rows = conn.execute(
            "SELECT DISTINCT person_id FROM tracking_results WHERE run_id = ? ORDER BY person_id",
            (run_id,)
        ).fetchall()
        person_ids = [r["person_id"] for r in pid_rows] or [0]

        for person_id in person_ids:
            sp_row = conn.execute(
                "SELECT person_name FROM sequence_persons WHERE sequence_id = ? AND person_id = ?",
                (seq_id, person_id)
            ).fetchone()
            person_name = sp_row["person_name"] if sp_row else f"person_{person_id}"

            track_id = _get_person_track_id(conn, run_id, person_name)
            color = _davis_bgr(track_id) if track_id is not None else _davis_bgr(person_id + 1)
            print(f"  run={run_id[:8]}  person={person_name}  "
                  f"track_id={track_id}  color={color}")

            if skeleton and layout and not args.no_bones:
                print(f"    Computing FK …")
                bone_data, timestamps = _load_bone_data(conn, run_id, person_id, skeleton, layout)
                print(f"    {len(bone_data)} frames with FK")
            else:
                bone_data, timestamps = {}, {}
                ts_rows = conn.execute(
                    "SELECT tracker_step, timestamp_s FROM tracking_results "
                    "WHERE run_id = ? AND person_id = ? AND is_smoothed = 0",
                    (run_id, person_id)
                ).fetchall()
                for r in ts_rows:
                    timestamps[r["tracker_step"]] = float(r["timestamp_s"])

            obs_data = _load_obs_data(
                conn, run_id, person_id, cam_label_to_csv, extrinsic_id
            )
            print(f"    {len(obs_data)} steps with observations")

            persons_data.append({
                "person_name": person_name,
                "color": color,
                "bone_data": bone_data,
                "obs_data": obs_data,
                "timestamps": timestamps,
            })
            all_steps |= set(obs_data.keys()) | set(timestamps.keys())
            if not primary_timestamps:
                primary_timestamps = timestamps

    if not primary_timestamps:
        primary_timestamps = persons_data[0]["timestamps"] if persons_data else {}

    fps = args.fps or _infer_fps(primary_timestamps)
    print(f"\nOutput FPS: {fps}")

    # Compute crop bboxes
    print("Computing crop boxes …")
    merged_obs: dict[int, dict[int, dict]] = {}
    for p in persons_data:
        for step, cam_data in p["obs_data"].items():
            if step not in merged_obs:
                merged_obs[step] = {}
            for csv_id, markers in cam_data.items():
                if csv_id not in merged_obs[step]:
                    merged_obs[step][csv_id] = {}
                merged_obs[step][csv_id].update(markers)

    active_csv_ids = [c.csv_id for c in active_cameras]
    raw_bboxes = _compute_bboxes(merged_obs, active_csv_ids)

    n_cams = len(active_cameras)
    rows_g, cols_g = (1, 1) if n_cams == 1 else _grid_dims(n_cams)
    cell_w = out_w // cols_g
    cell_h = out_h // rows_g
    cell_aspect = cell_w / cell_h

    crops: dict[int, tuple] = {}
    for cam in active_cameras:
        cid = cam.csv_id
        bbox = raw_bboxes.get(cid)
        if bbox is None:
            crops[cid] = (0, 0, 9999, 9999)
            print(f"  {cam.label}: no observations — using full frame")
        else:
            adjusted = _adjust_aspect(bbox, cell_aspect)
            crops[cid] = adjusted
            print(f"  {cam.label}: crop {adjusted[0]:.0f},{adjusted[1]:.0f}"
                  f"–{adjusted[2]:.0f},{adjusted[3]:.0f}")

    # Open video captures and find seg mask sources
    print("\nOpening videos …")
    caps: dict[str, cv2.VideoCapture] = {}
    vid_sizes: dict[str, tuple[int, int]] = {}
    seg_sources: dict[str, str | None] = {}  # label → seg_quality_run_id

    for cam in active_cameras:
        vpath = video_paths.get(cam.label)
        if not vpath:
            print(f"  [warn] no video path for {cam.label}", file=sys.stderr)
            continue
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  [warn] cannot open {vpath}", file=sys.stderr)
            continue
        caps[cam.label] = cap
        vid_sizes[cam.label] = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        # Clamp crops now that we know video dimensions
        svid = label_to_svid.get(cam.label)
        if cam.csv_id in crops:
            vw, vh = vid_sizes[cam.label]
            crops[cam.csv_id] = _clamp_bbox(crops[cam.csv_id], vw, vh)

        if not args.no_masks and svid:
            all_vid_frames = list(range(
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ))
            f0 = all_vid_frames[0] if all_vid_frames else 0
            f1 = all_vid_frames[-1] if all_vid_frames else 0
            seg_src = _find_seg_source(conn, svid, (f0, f1))
            seg_sources[cam.label] = seg_src
            if seg_src:
                print(f"  {cam.label}: seg masks from {seg_src[:13]}")
            else:
                print(f"  {cam.label}: no seg masks")
        else:
            seg_sources[cam.label] = None

    # Write output
    print(f"\nRendering {len(all_steps)} frames → {args.output}  "
          f"({out_w}×{out_h} @ {fps} fps) …")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"Cannot open output: {args.output}", file=sys.stderr)
        sys.exit(1)

    blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

    for step in sorted(all_steps):
        timestamp = primary_timestamps.get(step, step / fps)
        grid = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        for cam_idx, cam in enumerate(active_cameras):
            row_g  = cam_idx // cols_g
            col_g  = cam_idx % cols_g
            y0, y1_g = row_g * cell_h, (row_g + 1) * cell_h
            x0, x1_g = col_g * cell_w, (col_g + 1) * cell_w

            cap = caps.get(cam.label)
            if cap is None:
                grid[y0:y1_g, x0:x1_g] = blank
                continue

            vid_frame_idx = sync.lookup(cam.label, timestamp)
            if vid_frame_idx is None:
                vid_frame_idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, vid_frame_idx)
            ret, vid_frame = cap.read()
            if not ret:
                grid[y0:y1_g, x0:x1_g] = blank
                continue

            # Seg mask overlay
            seg_src = seg_sources.get(cam.label)
            if seg_src:
                svid = label_to_svid.get(cam.label)
                mask = _load_seg_mask(conn, seg_src, svid, vid_frame_idx)
                if mask is not None:
                    vid_frame = blend_mask_davis(vid_frame, mask)

            # Assemble per-person data for this step × cam
            persons_for_cell = []
            for p in persons_data:
                persons_for_cell.append({
                    "color":    p["color"],
                    "bones_3d": p["bone_data"].get(step, {}),
                    "obs":      p["obs_data"].get(step, {}).get(cam.csv_id, {}),
                })

            crop = crops.get(cam.csv_id, (0, 0, *vid_sizes.get(cam.label, (1920, 1080))))
            label = f"{cam.label}  f{vid_frame_idx}"
            cell = render_cell(vid_frame, cam, crop, cell_w, cell_h,
                               persons_for_cell, label)
            grid[y0:y1_g, x0:x1_g] = cell

        footer = f"step {step}  t={timestamp:.3f}s"
        cv2.putText(grid, footer, (10, out_h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(grid, footer, (10, out_h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, _TEXT_BGR, 1, cv2.LINE_AA)

        writer.write(grid)

        if step % 50 == 0:
            print(f"  step {step}/{max(all_steps)}", end="\r", flush=True)

    print()
    for cap in caps.values():
        cap.release()
    writer.release()
    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()
