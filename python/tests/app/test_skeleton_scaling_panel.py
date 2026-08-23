# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for skeleton_scaling_panel.py's measurement-triangulation helpers.

_MeasWorker's actual DB-driven flow needs a real tracking run (DLT
triangulation from raw pose_observations, camera calibration, a skeleton
YAML) to exercise meaningfully -- these tests cover the pure building blocks
it's made of instead: robust triangulation, undistortion, and reading marker
keypoint indices / raw detections back out of the DB, matching
docs/roadmap/features/observation-results-semantics.md's fix.
"""
from __future__ import annotations

import sqlite3
import struct

import numpy as np
import pytest

from app.ui.skeleton_scaling_panel import (
    _dlt,
    _load_raw_observations_by_camera,
    _marker_openpose_indices,
    _nearest_raw_keypoint,
    _reprojection_error_px,
    _robust_triangulate,
    _undistort_point,
)


def _camera_P(center: np.ndarray, look_at: np.ndarray, fx: float = 1000.0) -> np.ndarray:
    """A simple camera looking from *center* toward *look_at*, for building
    synthetic multi-view test fixtures without needing real calibration data."""
    forward = look_at - center
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    R = np.array([right, -true_up, forward])
    K = np.array([[fx, 0, 320], [0, fx, 240], [0, 0, 1]])
    t = -R @ center
    return K @ np.hstack([R, t.reshape(3, 1)])


def _project(P: np.ndarray, point: np.ndarray) -> tuple[float, float]:
    u, v, w = P @ np.append(point, 1.0)
    return u / w, v / w


# ---------------------------------------------------------------------------
# _robust_triangulate
# ---------------------------------------------------------------------------


def test_robust_triangulate_agrees_with_dlt_when_all_views_clean():
    point = np.array([0.1, 0.2, 1.5])
    Ps = [
        _camera_P(np.array([2.0, 0.0, 1.0]), point),
        _camera_P(np.array([-2.0, 1.0, 1.2]), point),
        _camera_P(np.array([0.0, -2.0, 0.8]), point),
    ]
    pts = [_project(P, point) for P in Ps]

    dlt_pos, _ = _dlt(pts, Ps)
    result = _robust_triangulate(pts, Ps)

    assert result is not None
    np.testing.assert_allclose(result[0], dlt_pos, atol=1e-9)
    np.testing.assert_allclose(result[0], point, atol=1e-6)


def test_robust_triangulate_drops_single_bad_camera():
    # Cameras all look toward a common scene center rather than straight at
    # `point`, so its clean projection lands at a genuinely different pixel
    # per camera (all-cameras-look-at-the-point-itself would make every
    # clean projection coincide at the principal point, which makes "worst
    # reprojecting camera" ambiguous and isn't how a real rig looks anyway).
    scene_center = np.array([0.0, 0.0, 1.2])
    point = scene_center + np.array([0.1, 0.2, 0.3])
    Ps = [
        _camera_P(np.array([2.0, 0.0, 1.0]), scene_center),
        _camera_P(np.array([-2.0, 1.0, 1.2]), scene_center),
        _camera_P(np.array([0.0, -2.0, 0.8]), scene_center),
        _camera_P(np.array([1.0, 2.0, 1.4]), scene_center),
    ]
    pts = [_project(P, point) for P in Ps]
    # Corrupt one camera's observation the way a low-confidence miss did in
    # the real bug: wildly off, but not so inconsistent that cond > 200.
    pts[1] = (pts[1][0] + 150.0, pts[1][1] - 150.0)

    result = _robust_triangulate(pts, Ps)

    assert result is not None
    np.testing.assert_allclose(result[0], point, atol=1e-3)


def test_robust_triangulate_none_when_too_few_views():
    result = _robust_triangulate([(1.0, 2.0)], [np.eye(3, 4)])
    assert result is None


def test_reprojection_error_zero_for_exact_projection():
    point = np.array([0.0, 0.0, 2.0])
    P = _camera_P(np.array([1.0, 1.0, 1.0]), point)
    pt = _project(P, point)
    assert _reprojection_error_px(point, pt, P) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _undistort_point
# ---------------------------------------------------------------------------


def test_undistort_point_is_noop_for_zero_distortion():
    K = np.array([[1000.0, 0, 640], [0, 1000.0, 480], [0, 0, 1]])
    dist = np.zeros(4)
    assert _undistort_point(700.0, 500.0, K, dist) == (700.0, 500.0)


def test_undistort_point_moves_off_center_pixel_with_real_distortion():
    K = np.array([[1000.0, 0, 640], [0, 1000.0, 480], [0, 0, 1]])
    dist = np.array([0.05, -0.01, 0.0, 0.0])  # mild barrel distortion
    px, py = _undistort_point(900.0, 480.0, K, dist)
    # Off-center along x, on the horizontal centerline -- undistortion should
    # shift x noticeably and leave y untouched by symmetry.
    assert px != pytest.approx(900.0)
    assert py == pytest.approx(480.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _marker_openpose_indices
# ---------------------------------------------------------------------------


def test_marker_openpose_indices_reads_from_skeleton_yaml():
    yaml_content = """
markers:
  - name: MRK-hip.R
    openpose_keypoint: 12
  - name: MRK-knee.R
    openpose_keypoint: 14
  - name: MRK-no-index
"""
    result = _marker_openpose_indices(yaml_content)
    assert result == {"MRK-hip.R": 12, "MRK-knee.R": 14}


# ---------------------------------------------------------------------------
# _load_raw_observations_by_camera / _nearest_raw_keypoint
# ---------------------------------------------------------------------------


def _kp_blob(keypoints: list[tuple[float, float, float]]) -> bytes:
    flat = [v for kp in keypoints for v in kp]
    return struct.pack(f"<{len(flat)}f", *flat)


@pytest.fixture()
def raw_obs_db(tmp_path):
    conn = sqlite3.connect(tmp_path / "raw_obs.db")
    conn.execute(
        "CREATE TABLE pose_observations (sequence_id TEXT, camera_instance_id TEXT, "
        "timestamp_s REAL, source TEXT, kp_blob BLOB)"
    )
    # Two frames for cam1, one for cam2 (a different sequence's row is
    # included too, to confirm sequence_id scoping works).
    conn.execute(
        "INSERT INTO pose_observations VALUES ('seq1','cam1',1.0,'body',?)",
        (_kp_blob([(10.0, 20.0, 0.9), (30.0, 40.0, 0.0)]),),
    )
    conn.execute(
        "INSERT INTO pose_observations VALUES ('seq1','cam1',2.0,'body',?)",
        (_kp_blob([(12.0, 22.0, 0.8), (32.0, 42.0, 0.7)]),),
    )
    conn.execute(
        "INSERT INTO pose_observations VALUES ('seq1','cam2',1.5,'body',?)",
        (_kp_blob([(50.0, 60.0, 0.5), (70.0, 80.0, 0.6)]),),
    )
    conn.execute(
        "INSERT INTO pose_observations VALUES ('other-seq','cam1',1.0,'body',?)",
        (_kp_blob([(999.0, 999.0, 0.9), (999.0, 999.0, 0.9)]),),
    )
    conn.row_factory = sqlite3.Row
    conn.commit()
    return conn


def test_load_raw_observations_scoped_to_sequence(raw_obs_db):
    result = _load_raw_observations_by_camera(raw_obs_db, "seq1")
    assert set(result) == {"cam1", "cam2"}
    ts, kps = result["cam1"]
    np.testing.assert_array_equal(ts, [1.0, 2.0])
    assert kps.shape == (2, 2, 3)


def test_nearest_raw_keypoint_picks_closest_frame(raw_obs_db):
    result = _load_raw_observations_by_camera(raw_obs_db, "seq1")
    raw = result["cam1"]
    # ts=1.3 is closer to frame at ts=1.0 than ts=2.0
    assert _nearest_raw_keypoint(raw, 1.3, 0) == (10.0, 20.0)
    # ts=1.8 is closer to frame at ts=2.0
    assert _nearest_raw_keypoint(raw, 1.8, 0) == (12.0, 22.0)


def test_nearest_raw_keypoint_none_for_zero_confidence(raw_obs_db):
    result = _load_raw_observations_by_camera(raw_obs_db, "seq1")
    raw = result["cam1"]
    # keypoint 1 at ts=1.0 has confidence 0.0 -- not actually detected
    assert _nearest_raw_keypoint(raw, 1.0, 1) is None


def test_nearest_raw_keypoint_none_for_out_of_range_index(raw_obs_db):
    result = _load_raw_observations_by_camera(raw_obs_db, "seq1")
    raw = result["cam1"]
    assert _nearest_raw_keypoint(raw, 1.0, 99) is None
