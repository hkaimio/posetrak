# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for calibrate_rigid_marker_body.py's reflective-dot calibration
pieces (Phase C1, see
docs/roadmap/features/marker-based-mocap/reflective-dot-detection-design.md
§3.1): triangulate_point_multi_view() and cluster_dot_samples(). The rest
of the script (DB loading, ArUco corner calibration) is exercised only by
running it against real data (this file adds no fixture for that -- see
the module's own "standalone, not yet validated as a CLI subcommand"
status), so this covers the two genuinely new pure functions in isolation.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "calibrate_rigid_marker_body.py"
)
_spec = importlib.util.spec_from_file_location("calibrate_rigid_marker_body", _MODULE_PATH)
calibrate_rigid_marker_body = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = calibrate_rigid_marker_body
_spec.loader.exec_module(calibrate_rigid_marker_body)

cluster_dot_samples = calibrate_rigid_marker_body.cluster_dot_samples
triangulate_point_multi_view = calibrate_rigid_marker_body.triangulate_point_multi_view

from app.setup.extrinsics_solver import CamCalibState  # noqa: E402


def _make_camera(video_id: str, position: np.ndarray, look_at: np.ndarray) -> CamCalibState:
    """A simple pinhole camera looking at *look_at* from *position*, world-up (0,0,1)."""
    forward = (look_at - position)
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # World->camera rotation: camera's own axes (right, -up, forward) as rows.
    R = np.stack([right, -up, forward])
    t = -R @ position
    K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    return CamCalibState(
        video_id=video_id, label=video_id, K=K, K_orig=K, dist=np.zeros((1, 4)), fisheye=False,
        R=R, t=t,
    )


def _project(state: CamCalibState, world_pt: np.ndarray) -> tuple[float, float]:
    p_cam = state.R @ world_pt + state.t.flatten()
    p_pix = state.K @ p_cam
    return float(p_pix[0] / p_pix[2]), float(p_pix[1] / p_pix[2])


def test_triangulate_point_multi_view_recovers_a_known_point():
    world_pt = np.array([0.05, -0.02, 0.15])
    cam_a = _make_camera("a", np.array([2.0, 0.0, 1.0]), np.zeros(3))
    cam_b = _make_camera("b", np.array([0.0, 2.0, 1.0]), np.zeros(3))
    cam_c = _make_camera("c", np.array([-2.0, 0.5, 1.5]), np.zeros(3))
    states = {"a": cam_a, "b": cam_b, "c": cam_c}

    observations = {cid: _project(s, world_pt) for cid, s in states.items()}
    recovered = triangulate_point_multi_view(observations, states)

    assert recovered is not None
    assert np.allclose(recovered, world_pt, atol=1e-6)


def test_triangulate_point_multi_view_needs_at_least_two_views():
    world_pt = np.array([0.0, 0.0, 0.2])
    cam_a = _make_camera("a", np.array([2.0, 0.0, 1.0]), np.zeros(3))
    observations = {"a": _project(cam_a, world_pt)}

    assert triangulate_point_multi_view(observations, {"a": cam_a}) is None


def test_triangulate_point_multi_view_rejects_a_cross_camera_false_match():
    """Two cameras each seeing exactly one candidate does not mean those
    candidates are the same physical point -- confirmed against real
    footage (see the function's own doc comment). Simulates that: cam_b's
    "observation" is an unrelated pixel, not world_pt's real projection."""
    world_pt = np.array([0.05, -0.02, 0.15])
    cam_a = _make_camera("a", np.array([2.0, 0.0, 1.0]), np.zeros(3))
    cam_b = _make_camera("b", np.array([0.0, 2.0, 1.0]), np.zeros(3))
    states = {"a": cam_a, "b": cam_b}

    observations = {
        "a": _project(cam_a, world_pt),
        "b": (900.0, 50.0),  # unrelated to world_pt -- a different real feature
    }
    assert triangulate_point_multi_view(observations, states) is None


def test_triangulate_point_multi_view_ignores_unknown_camera_ids():
    world_pt = np.array([0.01, 0.03, 0.1])
    cam_a = _make_camera("a", np.array([2.0, 0.0, 1.0]), np.zeros(3))
    cam_b = _make_camera("b", np.array([0.0, 2.0, 1.0]), np.zeros(3))
    states = {"a": cam_a, "b": cam_b}

    observations = {
        "a": _project(cam_a, world_pt),
        "b": _project(cam_b, world_pt),
        "ghost": (123.0, 456.0),  # not in states -- must be skipped, not crash
    }
    recovered = triangulate_point_multi_view(observations, states)
    assert recovered is not None
    assert np.allclose(recovered, world_pt, atol=1e-6)


def test_cluster_dot_samples_separates_two_distinct_dots():
    rng = np.random.default_rng(7)
    dot_a_center = np.array([0.0, 0.0, 0.0])
    dot_b_center = np.array([0.10, 0.0, 0.0])  # 10cm away -- well past tolerance
    samples = [
        dot_a_center + rng.normal(scale=0.001, size=3) for _ in range(8)
    ] + [
        dot_b_center + rng.normal(scale=0.001, size=3) for _ in range(5)
    ]
    rng.shuffle(samples)

    clusters = cluster_dot_samples(samples, tolerance_m=0.02)

    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [5, 8]


def test_cluster_dot_samples_merges_within_tolerance():
    samples = [np.array([0.0, 0.0, 0.0]), np.array([0.005, 0.0, 0.0]), np.array([0.01, 0.0, 0.0])]
    clusters = cluster_dot_samples(samples, tolerance_m=0.02)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_cluster_dot_samples_empty_input():
    assert cluster_dot_samples([], tolerance_m=0.02) == []


def test_cluster_dot_samples_single_sample():
    clusters = cluster_dot_samples([np.array([1.0, 2.0, 3.0])], tolerance_m=0.02)
    assert len(clusters) == 1
    assert np.allclose(clusters[0][0], [1.0, 2.0, 3.0])
