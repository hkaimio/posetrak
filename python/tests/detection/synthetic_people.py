# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""synthetic_people.py — a calibrated multi-camera scene with walking people, for the assignment tests.

Persons are COCO-17 skeletons that walk in straight lines; each camera sees every joint (projected, with a little
noise). The detector's track ids can be swapped at a chosen frame, as a detector does where persons overlap.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from posetrak.db.db import create_session, generate_id

K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
FPS = 30.0

# COCO-17 joint offsets from the person's ground position (x forward, y left, z up), metres
_JOINTS = {
    0: (0, 0, 1.65), 1: (0.03, 0.03, 1.69), 2: (0.03, -0.03, 1.69), 3: (0, 0.08, 1.67), 4: (0, -0.08, 1.67),
    5: (0, 0.19, 1.45), 6: (0, -0.19, 1.45), 7: (0.05, 0.22, 1.2), 8: (0.05, -0.22, 1.2),
    9: (0.15, 0.22, 0.98), 10: (0.15, -0.22, 0.98), 11: (0, 0.1, 0.95), 12: (0, -0.1, 0.95),
    13: (0.05, 0.1, 0.5), 14: (0.05, -0.1, 0.5), 15: (0, 0.1, 0.08), 16: (0, -0.1, 0.08),
}


def look_at(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera rotation and translation of a camera at *eye* looking at *target* (z up)."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ eye


def cameras(n: int = 3, radius: float = 6.0) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for k in range(n):
        a = 2 * np.pi * k / n + 0.4
        out.append(look_at(np.array([radius * np.cos(a), radius * np.sin(a), 1.5]), np.array([0.0, 0.0, 1.0])))
    return out


def person_points(ground: np.ndarray) -> np.ndarray:
    """(17, 3) world joints of a person standing at the ground position (x, y)."""
    return np.array([[ground[0] + dx, ground[1] + dy, dz] for dx, dy, dz in (_JOINTS[j] for j in range(17))])


def project(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    c = X @ R.T + t
    uv = c @ K.T
    return uv[:, :2] / uv[:, 2:3]


def blob(xy: np.ndarray, conf: float = 0.9) -> bytes:
    """A COCO-133 keypoint blob with the 17 body keypoints filled."""
    a = np.zeros((133, 3), dtype="<f4")
    a[:17, :2] = xy
    a[:17, 2] = conf
    return a.tobytes()


def soma_points(ground: np.ndarray) -> dict[str, np.ndarray]:
    """World positions of the SOMA-77 body joints of a person standing at the ground position (x, y)."""
    c = person_points(ground)
    hips, neck = (c[11] + c[12]) / 2, (c[5] + c[6]) / 2
    return {
        "Hips": hips, "Spine1": hips + (neck - hips) * 0.3, "Spine2": hips + (neck - hips) * 0.6,
        "Chest": hips + (neck - hips) * 0.8, "Neck1": neck, "Head": c[0],
        "LeftShoulder": neck + (c[5] - neck) * 0.5, "LeftArm": c[5], "LeftForeArm": c[7], "LeftHand": c[9],
        "RightShoulder": neck + (c[6] - neck) * 0.5, "RightArm": c[6], "RightForeArm": c[8], "RightHand": c[10],
        "LeftLeg": c[11], "LeftShin": c[13], "LeftFoot": c[15], "LeftToeBase": c[15] + [0.12, 0, 0],
        "RightLeg": c[12], "RightShin": c[14], "RightFoot": c[16], "RightToeBase": c[16] + [0.12, 0, 0],
    }


def soma_blob(xy_by_name: dict[str, np.ndarray], conf: float = 0.9) -> bytes:
    """A SOMA-77 keypoint blob with the body joints filled."""
    from posetrak.detection.pose_models import SOMA77_NAMES

    a = np.zeros((len(SOMA77_NAMES), 3), dtype="<f4")
    for name, xy in xy_by_name.items():
        a[SOMA77_NAMES.index(name)] = [*xy, conf]
    return a.tobytes()


def build_session(
    path: Path, starts: list[tuple[float, float]], velocities: list[tuple[float, float]], n_frames: int,
    swap_at: int | None = None, n_cams: int = 3, existing_person: str | None = None, rng_seed: int = 0,
    soma: bool = False,
) -> tuple[sqlite3.Connection, str, list[str]]:
    """A session database with a calibrated capture and one detection run of the walking persons.

    Detector track id of person i is i; from frame *swap_at* on, persons 0 and 1 trade ids. Returns the open
    connection, the detection run id and the ids of the capture's videos.
    """
    rng = np.random.default_rng(rng_seed)
    conn = create_session(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF")
    session_id, capture_id, sync_id, calib_id, run_id = (generate_id() for _ in range(5))
    conn.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES (?, '2026-01-01')", (session_id,))
    conn.execute("INSERT INTO extrinsic_calibrations (id, session_id, calibrated_at) VALUES (?, ?, '2026-01-01')",
                 (calib_id, session_id))
    conn.execute("INSERT INTO captures (id, session_id, extrinsic_calibration_id, capture_number, label) "
                 "VALUES (?, ?, ?, 1, 'synthetic')", (capture_id, session_id, calib_id))
    conn.execute("INSERT INTO sync_configs (id, shot_id, created_by) VALUES (?, ?, 'test')", (sync_id, capture_id))
    conn.execute("INSERT INTO intrinsics_calibrations (id, camera_mode_id, calibrated_at, distortion_model, "
                 "fx, fy, cx, cy, image_width, image_height) VALUES ('intr', 'mode', '2026-01-01', 'radtan', "
                 "1000, 1000, 960, 540, 1920, 1080)")
    videos = []
    for k, (R, t) in enumerate(cameras(n_cams)):
        cam_id, video_id = generate_id(), generate_id()
        videos.append(video_id)
        conn.execute("INSERT INTO camera_instances (id, camera_model_id, label) VALUES (?, 'm', ?)",
                     (cam_id, f"cam{k + 1}"))
        conn.execute("INSERT INTO extrinsic_entries (extrinsic_calibration_id, camera_instance_id, R, t) "
                     "VALUES (?, ?, ?, ?)", (calib_id, cam_id, R.astype("<f8").tobytes(), t.astype("<f8").tobytes()))
        conn.execute("INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path, first_video_frame, "
                     "last_video_frame, actual_fps, intrinsics_calibration_id) "
                     "VALUES (?, ?, ?, 'x.mp4', 0, ?, ?, 'intr')", (video_id, capture_id, cam_id, n_frames, FPS))
        conn.execute("INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id, video_frame, "
                     "timestamp_s) VALUES (?, ?, ?, 0, 0.0)", (sync_id, cam_id, video_id))
    conn.execute("INSERT INTO detection_runs (id, shot_id, sync_config_id, time_start_s, time_end_s, detector_model, "
                 "pose_model, status, created_at) VALUES (?, ?, ?, 0.0, ?, 'yolox-x', ?, "
                 "'complete', '2026-01-01')",
                 (run_id, capture_id, sync_id, n_frames / FPS, "soma77-dinov3-vitpose" if soma else "rtmpose-l-133kp"))
    cams = cameras(n_cams)
    for f in range(n_frames):
        for p, (x0, y0) in enumerate(starts):
            ground = np.array([x0 + velocities[p][0] * f / FPS, y0 + velocities[p][1] * f / FPS])
            tid = p
            if swap_at is not None and f >= swap_at and p < 2:
                tid = 1 - p
            for video_id, (R, t) in zip(videos, cams):
                xy = project(R, t, person_points(ground)) + rng.normal(0, 1.0, (17, 2))
                if soma:
                    named = soma_points(ground)
                    uv = project(R, t, np.array(list(named.values()))) + rng.normal(0, 1.0, (len(named), 2))
                    kp = soma_blob(dict(zip(named, uv)))
                else:
                    kp = blob(xy)
                conn.execute("INSERT INTO detection_keypoints (detection_run_id, shot_video_id, video_frame, "
                             "track_id, region_type, keypoints) VALUES (?, ?, ?, ?, 'full_body', ?)",
                             (run_id, video_id, f, tid, kp))
                conn.execute("INSERT INTO person_detections (detection_run_id, shot_video_id, video_frame, track_id, "
                             "region_type, bbox_x, bbox_y, bbox_w, bbox_h, confidence) "
                             "VALUES (?, ?, ?, ?, 'full_body', ?, ?, ?, ?, 0.9)",
                             (run_id, video_id, f, tid, float(xy[:, 0].min()), float(xy[:, 1].min()),
                              float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1]))))
    if existing_person:
        conn.execute("INSERT INTO capture_persons (id, capture_id, name, created_at) VALUES (?, ?, ?, '2026-01-01')",
                     (generate_id(), capture_id, existing_person))
    conn.commit()
    return conn, run_id, videos
