# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""bones.py — which bone a marker rides on, from how it moves.

A marker rigidly attached to a bone stays at a fixed place in that bone's frame
however the bone moves. Given the 3D track of a marker (triangulated from a group
of tracklets, see grouping.py) and the tracked skeleton (a markerless tracking
run), the marker's track is expressed in the frame of each candidate bone; the
bone it rides on is the one where the track has almost no spread. The mean of the
track in that frame is the marker's calibrated offset.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import cv2
import numpy as np

from app.setup.extrinsics_solver import CamCalibState, _proj_matrix, _undistort_pts
from posetrak.db.skeleton_layout import SkeletonLayout

# Observations of one marker: {time bucket: [(camera_id, (x, y) in distorted pixels), ...]}
Observations = dict[int, list[tuple[str, tuple[float, float]]]]
TIME_BUCKETS_PER_SECOND = 120


class BodyPose:
    """Joint frames of a tracked skeleton over time (from a tracking run's smoothed states)."""

    def __init__(self, conn: sqlite3.Connection, tracking_run_id: str, max_gap_s: float = 0.02) -> None:
        run = conn.execute("SELECT skeleton_id FROM tracking_runs WHERE id = ?", (tracking_run_id,)).fetchone()
        yaml_text = conn.execute(
            "SELECT yaml_content FROM skeletons WHERE id = ?", (run["skeleton_id"],)
        ).fetchone()["yaml_content"]
        self.layout = SkeletonLayout(yaml_text)
        rows = conn.execute(
            "SELECT timestamp_s, state FROM tracking_results WHERE run_id = ? AND is_smoothed = 1 "
            "AND tracking_lost = 0 ORDER BY tracker_step", (tracking_run_id,),
        ).fetchall()
        self.times = np.array([r["timestamp_s"] for r in rows])
        self._blobs = [bytes(r["state"]) for r in rows]
        self._max_gap = max_gap_s
        self._cache: dict[int, dict[str, np.ndarray] | None] = {}

    def joint_frames(self, t: float) -> dict[str, np.ndarray] | None:
        """World 4x4 frame of every joint at time *t*, or None if the run has no state near it."""
        i = int(np.argmin(np.abs(self.times - t))) if len(self.times) else -1
        if i < 0 or abs(self.times[i] - t) > self._max_gap:
            return None
        if i not in self._cache:
            self._cache[i] = self.layout.compute_joint_transforms(self.layout.decode_state_blob(self._blobs[i]))
        return self._cache[i]


def marker_observations(
    members: list[tuple[str, int]], tracklets_by_cam: dict, sync_table, svid_by_cam: dict,
) -> Observations:
    """Group the frames of a marker's tracklets by time, across cameras."""
    obs: Observations = {}
    for cam_id, tracklet_id in members:
        svid = svid_by_cam[cam_id]
        for frame, xy in tracklets_by_cam[cam_id][tracklet_id].items():
            t = sync_table.frame_to_global_time(frame, svid)
            if t is not None:
                obs.setdefault(round(t * TIME_BUCKETS_PER_SECOND), []).append((cam_id, xy))
    return obs


def triangulate(observations: list[tuple[str, tuple[float, float]]], states: dict[str, CamCalibState]) -> np.ndarray | None:
    """Linear (DLT) triangulation of one marker seen by two or more cameras; None if a
    single camera, or the point is behind a camera. One observation per camera is used."""
    seen: dict[str, tuple[float, float]] = {}
    for cam_id, xy in observations:
        seen.setdefault(cam_id, xy)
    if len(seen) < 2:
        return None
    rows = []
    for cam_id, xy in seen.items():
        point = _undistort_pts(np.array([xy]), states[cam_id])[0]
        P = _proj_matrix(states[cam_id])
        rows.append(point[0] * P[2] - P[0])
        rows.append(point[1] * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.array(rows))
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def marker_track(observations: Observations, states: dict[str, CamCalibState]) -> tuple[np.ndarray, np.ndarray]:
    """Times (s) and 3D points of a marker, for the moments two cameras saw it."""
    times, points, _ = marker_track_with_cameras(observations, states)
    return times, points


def marker_track_with_cameras(
    observations: Observations, states: dict[str, CamCalibState],
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, ...]]]:
    """Like :func:`marker_track`, and the cameras that saw the marker at each of those moments."""
    times, points, cameras = [], [], []
    for bucket in sorted(observations):
        X = triangulate(observations[bucket], states)
        if X is not None:
            times.append(bucket / TIME_BUCKETS_PER_SECOND)
            points.append(X)
            cameras.append(tuple(sorted({cam for cam, _ in observations[bucket]})))
    return np.array(times), np.array(points).reshape(-1, 3), cameras


@dataclass
class BoneFit:
    joint: str
    spread_m: float       # RMS distance of the track from its mean, in the bone's frame
    offset: np.ndarray    # mean position in the bone's frame (metres)
    n: int                # samples


def fit_bones(times: np.ndarray, points: np.ndarray, pose: BodyPose, joints: list[str] | None = None) -> list[BoneFit]:
    """For every candidate bone, the marker's track in the bone's frame. Best (least spread) first."""
    frames = [(t, X, pose.joint_frames(t)) for t, X in zip(times, points)]
    frames = [(t, X, F) for t, X, F in frames if F is not None]
    if len(frames) < 3:
        return []
    names = joints or sorted(frames[0][2])
    fits = []
    for name in names:
        local = np.array([(np.linalg.inv(F[name]) @ np.append(X, 1.0))[:3] for _, X, F in frames if name in F])
        if len(local) < 3:
            continue
        mean = local.mean(axis=0)
        fits.append(BoneFit(name, float(np.sqrt(np.mean(np.sum((local - mean) ** 2, axis=1)))), mean, len(local)))
    return sorted(fits, key=lambda f: f.spread_m)


@dataclass
class RayFit:
    """Offset of a marker in a bone's frame fitted to one camera's 2D track."""

    joint: str
    offset: np.ndarray
    reproj_rms_px: float
    n: int
    weakest_direction: np.ndarray   # offset direction (bone frame) the track constrains least
    weakest_strength: float         # m^-2 summed over samples; small = depth still undetermined


def fit_offset_from_rays(
    times: np.ndarray, pixels: np.ndarray, state: CamCalibState, pose: BodyPose, joint: str,
    prior: np.ndarray, prior_weight: float = 1.0,
) -> RayFit | None:
    """Fit a marker's offset in *joint*'s frame to its 2D track in a single camera.

    Each frame says the marker lies on a ray through the camera centre; with the bone
    moving, the offset in the bone's frame is the point closest to all those rays. A
    track in one camera leaves the offset undetermined along the ray directions when the
    bone does not turn, so the fit is pulled towards *prior* with a small weight
    (*prior_weight*, in units of the per-sample information) and the weakest direction is
    reported, to tell a fit that is anchored by the data from one held by the prior.
    """
    centre = -state.R.T @ state.t.ravel()
    rays = _unit_rays(pixels, state)
    A = prior_weight * np.eye(3) * 1e-2
    b = prior_weight * 1e-2 * prior.copy()
    A_data = np.zeros((3, 3))
    frames = []
    for t, ray in zip(times, rays):
        F = pose.joint_frames(t)
        if F is None or joint not in F:
            continue
        R, c = F[joint][:3, :3], F[joint][:3, 3]
        M = R.T @ (np.eye(3) - np.outer(ray, ray)) @ R
        A_data += M
        b += R.T @ (np.eye(3) - np.outer(ray, ray)) @ (centre - c)
        frames.append((F[joint], ray))
    if len(frames) < 10:
        return None
    A += A_data
    offset = np.linalg.solve(A, b)
    errors = []
    for F, ray in frames:
        X = (F @ np.append(offset, 1.0))[:3]
        v = X - centre
        errors.append(np.degrees(np.arccos(np.clip(v @ ray / np.linalg.norm(v), -1, 1))))
    # angle to pixels: about focal length * tan(angle)
    f_px = float(state.K_orig[0, 0])
    rms_px = float(np.sqrt(np.mean((f_px * np.tan(np.radians(errors))) ** 2)))
    w, V = np.linalg.eigh(A_data)
    return RayFit(joint, offset, rms_px, len(frames), V[:, 0], float(w[0] / len(frames)))


def _unit_rays(pixels: np.ndarray, state: CamCalibState) -> np.ndarray:
    undistorted = _undistort_pts(np.asarray(pixels, float), state)
    cam = np.column_stack([
        (undistorted[:, 0] - state.K[0, 2]) / state.K[0, 0],
        (undistorted[:, 1] - state.K[1, 2]) / state.K[1, 1],
        np.ones(len(undistorted)),
    ])
    world = (state.R.T @ cam.T).T
    return world / np.linalg.norm(world, axis=1, keepdims=True)
