# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Box-video calibration on synthetic cameras: anchoring, alignment and the failure verdict.

The video and sync parts are replaced by synthetic detections of a box carried through the
scene; the solver, window ranking, alignment and judging are the real code.
"""

from __future__ import annotations

import sqlite3

import cv2
import numpy as np
import pytest

from app.setup import box_video
from app.setup.box_video import BoxVideoParams, calibrate_from_box_video, format_report, rank_key, _rigid_fit
from app.setup.extrinsics_solver import run_calibration
from app.setup.fiducial_markers import (
    FiducialDetection,
    MarkerCornerObs,
    MarkerRigConfig,
    marker_local_corners,
)

_SIZE, _HALF = 0.1, 0.15
_K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])


def _box() -> MarkerRigConfig:
    local = marker_local_corners(_SIZE)
    to_x = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    to_y = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    return MarkerRigConfig(rig_id="box", marker_corners={
        "10": local @ to_x.T + [_HALF, 0, 0],
        "11": local @ to_y.T + [0, _HALF, 0],
        "12": local + [0, 0, _HALF],
    })


def _look_at(center: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World->camera R, t for a camera at *center* looking at *target* (z forward, y down)."""
    z = (target - center) / np.linalg.norm(target - center)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ center


def _rot(axis: str, deg: float) -> np.ndarray:
    return cv2.Rodrigues(np.array({"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis], float) * np.deg2rad(deg))[0]


# Scene frame S: cameras around the origin; the box at window k is X_S = R_k q + t_k.
_CAMERAS = {
    "camA": _look_at(np.array([3.0, 0.0, 1.0]), np.array([0, 0, 0.5])),
    "camB": _look_at(np.array([0.0, 3.0, 1.2]), np.array([0, 0, 0.5])),
    "camC": _look_at(np.array([-3.0, 0.5, 0.8]), np.array([0, 0, 0.5])),
    "camD": _look_at(np.array([0.5, -3.0, 1.5]), np.array([0, 0, 0.5])),
}
_BOX_POSES = [  # (R, t, cameras that see the box at rest)
    (_rot("z", 10), np.array([0.2, 0.1, 0.4]), set(_CAMERAS)),
    (_rot("x", 40) @ _rot("z", 70), np.array([-0.3, 0.2, 0.9]), set(_CAMERAS)),
    (_rot("y", 30) @ _rot("z", 150), np.array([0.1, -0.4, 0.6]), set(_CAMERAS)),
    (_rot("z", 200), np.array([0.5, 0.5, 0.1]), {"camA", "camB"}),  # final rest: two cameras
]


class _FakeSync:
    def lookup(self, ts, video_id):
        return int(ts * 100)

    def frame_to_global_time(self, frame, video_id):
        return frame / 100.0


def _detections(config, R_box, t_box, cams, rng, noise_px, bad=None):
    out = {}
    for name in cams:
        R, t = _CAMERAS[name]
        dets = []
        for marker_id, corners in config.marker_corners.items():
            pts = corners @ R_box.T + t_box
            px, _ = cv2.projectPoints(pts, cv2.Rodrigues(R)[0], t, _K, None)
            px = px.reshape(-1, 2) + rng.normal(0, noise_px, (4, 2))
            if bad == name:
                px = px + rng.normal(0, 30.0, (4, 2))
            dets.append(FiducialDetection("aruco", marker_id, [
                MarkerCornerObs("aruco", marker_id, i, name, 0, float(p[0]), float(p[1]))
                for i, p in enumerate(px)]))
        out[name] = dets
    return out


@pytest.fixture()
def scenario(monkeypatch):
    """Patch video access with synthetic detections; returns a function to run the calibration."""
    config = _box()
    times = [10.0, 11.0, 12.0, 13.0]
    cams = [box_video._Cam(n, n, "x.mp4", 100.0, "i") for n in _CAMERAS]

    def run(*, noise_px=0.3, bad=None, final_cams=None, params=None, poses=None):
        rng = np.random.default_rng(1)
        poses = poses or _BOX_POSES
        det_by_time = {}
        for ts, (R, t, seen) in zip(times, poses):
            if ts == times[-1] and final_cams is not None:
                seen = final_cams
            det_by_time[ts] = _detections(config, R, t, seen, rng, noise_px, bad)
        monkeypatch.setattr(box_video, "load_capture_cameras", lambda conn, cap: (cams, _FakeSync()))
        monkeypatch.setattr(box_video, "find_stationary_windows", lambda *a, **k: [int(t * 100) for t in times])
        monkeypatch.setattr(box_video, "_detect_at", lambda ts, c, s, d, frames=None: (
            frames.update({n: np.zeros((4, 4, 3), np.uint8) for n in _CAMERAS}) if frames is not None else None,
            det_by_time[round(ts, 3)])[1])
        monkeypatch.setattr(box_video, "_load_intrinsics", lambda conn, cid: {
            "K": _K, "K_orig": _K, "dist": np.zeros((1, 4)), "fisheye": False})
        real = run_calibration
        # no images to match features in: rely on the box control points only
        monkeypatch.setattr(box_video, "run_calibration", lambda *a, **k: real(*a, **{**k, "cp_only": True}))
        return calibrate_from_box_video(
            sqlite3.connect(":memory:"), "cap", config,
            params or BoxVideoParams(reference_camera="camA"))
    return run


def _world_camera_centres():
    """Ground truth: camera centres in the box frame of the final rest position."""
    R_f, t_f, _ = _BOX_POSES[-1]
    return {n: R_f.T @ (-R.T @ t - t_f) for n, (R, t) in _CAMERAS.items()}


def test_anchor_is_a_window_every_camera_sees_and_world_is_the_final_rest_position(scenario):
    res = scenario()
    assert res.ok, format_report(res)
    assert res.anchor_ts != res.final_ts  # the final rest position is seen by 2 of 4 cameras only
    assert res.aligned_to_final
    truth = _world_camera_centres()
    for s in res.calibration.cameras.values():
        centre = -s.R.T @ s.t.flatten()
        assert np.linalg.norm(centre - truth[s.label]) < 0.02, s.label
    assert max(r.median_px for r in res.rows) < 1.0


def test_a_camera_with_large_errors_fails_the_calibration(scenario):
    res = scenario(bad="camC")
    assert not res.ok
    # the corrupted observations also pull on the shared solution, so others may fail too
    assert "camC" in [r.label for r in res.rows if not r.ok]
    assert "FAILED" in format_report(res)


def test_a_camera_that_never_sees_the_box_fails(scenario):
    poses = [(R, t, seen - {"camD"}) for R, t, seen in _BOX_POSES]
    res = scenario(poses=poses)
    assert not res.ok
    bad = {r.label: r for r in res.rows if not r.ok}
    assert "camD" in bad and bad["camD"].median_px is None


def test_windows_with_too_few_corners_are_dropped(scenario):
    # the third window is seen by one camera only: it must be dropped, not used
    poses = [_BOX_POSES[0], _BOX_POSES[1], (_BOX_POSES[2][0], _BOX_POSES[2][1], {"camA"}), _BOX_POSES[3]]
    res = scenario(poses=poses)
    assert any("1 camera" in why for _, why in res.dropped)
    assert len(res.windows) == 3


def test_excluded_window_is_dropped(scenario):
    res = scenario(params=BoxVideoParams(reference_camera="camA", exclude_window_ts=[11.2]))
    assert (11.0, "excluded by option") in res.dropped


def test_window_ranking_prefers_coverage_then_final_then_corners():
    w = lambda ts, n_cams, corners: box_video.Window(  # noqa: E731
        ts, {f"c{i}": corners // n_cams for i in range(n_cams)})
    four_early, four_final, two_final = w(1.0, 4, 40), w(3.0, 4, 32), w(3.0, 2, 40)
    assert rank_key(four_final, 3.0) > rank_key(four_early, 3.0)
    assert rank_key(four_early, 3.0) > rank_key(two_final, 3.0)


def test_rigid_fit_recovers_a_transform():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(12, 3))
    R = _rot("z", 33) @ _rot("x", -20)
    t = np.array([1.0, -2.0, 0.5])
    R2, t2, rms = _rigid_fit(q, q @ R.T + t)
    assert np.allclose(R2, R) and np.allclose(t2, t) and rms < 1e-9
