"""Synthetic-data validation for the Phase 3 rigid marker-pose BA residual.

See docs/roadmap/features/extrinsics-improvements/
extrinsics-improvements-design.md, section 5 ("Rigid marker-group residual")
and the "Open questions" entry flagging this as needing a synthetic-data
prototype before Phase 3's UI/detection work begins. These tests build a
small multi-camera rig with known poses and a known marker pose, generate
exact (and then noisy) corner projections, and check that
`solve_marker_pose` recovers the marker's pose from those projections alone
-- proving the corner-projection residual math is correct in isolation,
before it's wired into the real BA's parameter vector.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.setup.extrinsics_solver import (
    CamCalibState,
    marker_local_corners,
    project_marker_corners,
    solve_marker_pose,
)


def _look_at_pose(cam_pos: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World->camera (R, t) for a camera at *cam_pos* looking at *target*.

    Standard "look-at" construction: camera-space +Z points at the target,
    +X is to the right, +Y is down (OpenCV camera convention).
    """
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=0)  # world -> camera
    t = -R @ cam_pos
    return R, t


def _make_camera(video_id: str, R: np.ndarray, t: np.ndarray) -> CamCalibState:
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    return CamCalibState(
        video_id=video_id, label=video_id, K=K, K_orig=K.copy(),
        dist=np.zeros((1, 4)), fisheye=False, R=R, t=t.reshape(3, 1),
    )


@pytest.fixture()
def rig():
    """Three cameras arranged around the origin, all looking roughly at it."""
    marker_target = np.array([0.0, 0.0, 0.0])
    cam_positions = [
        np.array([0.0, -3.0, 1.5]),
        np.array([2.6, 1.5, 1.8]),
        np.array([-2.6, 1.5, 1.2]),
    ]
    states = {}
    for i, pos in enumerate(cam_positions):
        R, t = _look_at_pose(pos, marker_target)
        states[f"cam_{i}"] = _make_camera(f"cam_{i}", R, t)
    return states


def _project_all(states, rvec_m, tvec_m, local_corners) -> dict[str, np.ndarray]:
    return {
        vid: project_marker_corners(rvec_m, tvec_m, local_corners, state)
        for vid, state in states.items()
    }


# ---------------------------------------------------------------------------
# marker_local_corners
# ---------------------------------------------------------------------------


def test_marker_local_corners_shape_and_centring():
    corners = marker_local_corners(0.2)
    assert corners.shape == (4, 3)
    # Centred at the local origin.
    np.testing.assert_allclose(corners.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-12)
    # All corners lie in the local Z=0 plane.
    np.testing.assert_allclose(corners[:, 2], 0.0)


def test_marker_local_corners_side_length():
    size = 0.15
    corners = marker_local_corners(size)
    # Adjacent corners (0-1, 1-2, 2-3, 3-0) are exactly `size` apart.
    for i in range(4):
        edge = np.linalg.norm(corners[i] - corners[(i + 1) % 4])
        assert edge == pytest.approx(size)


# ---------------------------------------------------------------------------
# project_marker_corners
# ---------------------------------------------------------------------------


def test_project_marker_corners_identity_pose_matches_manual_projection(rig):
    local_corners = marker_local_corners(0.2)
    rvec_m = np.zeros(3)
    tvec_m = np.array([0.0, 0.0, 0.0])
    state = rig["cam_0"]

    proj = project_marker_corners(rvec_m, tvec_m, local_corners, state)
    assert proj.shape == (4, 2)

    # Manually project the same (world == local, since pose is identity)
    # corners with cv2.projectPoints directly, using the camera's own pose.
    rvec_cam, _ = cv2.Rodrigues(state.R)
    expected, _ = cv2.projectPoints(
        local_corners, rvec_cam, state.t.reshape(3, 1), state.K, np.zeros(4)
    )
    np.testing.assert_allclose(proj, expected.reshape(-1, 2), atol=1e-6)


# ---------------------------------------------------------------------------
# solve_marker_pose — exact (noise-free) recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rvec_m_true,tvec_m_true",
    [
        (np.zeros(3), np.array([0.0, 0.0, 0.0])),
        (np.array([0.3, -0.2, 0.1]), np.array([0.1, -0.05, 0.2])),
        (np.array([0.0, 0.0, np.pi / 2]), np.array([0.0, 0.0, 0.0])),  # 90 deg about Z
        (np.array([1.2, 0.4, -0.7]), np.array([-0.3, 0.4, -0.1])),
    ],
)
def test_solve_marker_pose_recovers_exact_pose(rig, rvec_m_true, tvec_m_true):
    size = 0.2
    local_corners = marker_local_corners(size)
    corner_obs = _project_all(rig, rvec_m_true, tvec_m_true, local_corners)

    rvec, tvec, rms = solve_marker_pose(corner_obs, rig, size)

    R_true, _ = cv2.Rodrigues(rvec_m_true)
    R_solved, _ = cv2.Rodrigues(rvec)
    np.testing.assert_allclose(R_solved, R_true, atol=1e-5)
    np.testing.assert_allclose(tvec, tvec_m_true, atol=1e-5)
    assert rms == pytest.approx(0.0, abs=1e-3)


def test_solve_marker_pose_two_cameras_sufficient(rig):
    """Two (non-degenerate) views are enough to resolve a planar target."""
    size = 0.2
    local_corners = marker_local_corners(size)
    rvec_m_true = np.array([0.2, -0.1, 0.05])
    tvec_m_true = np.array([0.05, 0.1, -0.05])
    two_cam_rig = {k: rig[k] for k in ("cam_0", "cam_1")}
    corner_obs = _project_all(two_cam_rig, rvec_m_true, tvec_m_true, local_corners)

    rvec, tvec, rms = solve_marker_pose(corner_obs, two_cam_rig, size)

    R_true, _ = cv2.Rodrigues(rvec_m_true)
    R_solved, _ = cv2.Rodrigues(rvec)
    np.testing.assert_allclose(R_solved, R_true, atol=1e-4)
    np.testing.assert_allclose(tvec, tvec_m_true, atol=1e-4)
    assert rms == pytest.approx(0.0, abs=1e-2)


def test_solve_marker_pose_single_camera_raises(rig):
    size = 0.2
    local_corners = marker_local_corners(size)
    corner_obs = _project_all(
        {"cam_0": rig["cam_0"]}, np.zeros(3), np.zeros(3), local_corners
    )
    with pytest.raises(ValueError):
        solve_marker_pose(corner_obs, rig, size)


def test_solve_marker_pose_ignores_unsolved_cameras(rig):
    """A camera present in corner_obs but with R=None (not yet solved) must
    not be counted toward the >=2-camera requirement, nor crash the solve."""
    size = 0.2
    local_corners = marker_local_corners(size)
    rvec_m_true = np.array([0.1, 0.1, 0.1])
    tvec_m_true = np.array([0.0, 0.0, 0.1])
    corner_obs = _project_all(rig, rvec_m_true, tvec_m_true, local_corners)

    unsolved = _make_camera("cam_unsolved", np.eye(3), np.zeros(3))
    unsolved.R = None
    unsolved.t = None
    states = dict(rig)
    states["cam_unsolved"] = unsolved
    corner_obs["cam_unsolved"] = np.zeros((4, 2))  # garbage; must be ignored

    rvec, tvec, rms = solve_marker_pose(corner_obs, states, size)
    R_true, _ = cv2.Rodrigues(rvec_m_true)
    R_solved, _ = cv2.Rodrigues(rvec)
    np.testing.assert_allclose(R_solved, R_true, atol=1e-5)
    np.testing.assert_allclose(tvec, tvec_m_true, atol=1e-5)


# ---------------------------------------------------------------------------
# solve_marker_pose — robustness to small pixel noise
# ---------------------------------------------------------------------------


def test_solve_marker_pose_converges_under_small_noise(rig):
    """With a few pixels of Gaussian noise on each corner, the solve should
    still land close to the true pose (this is a convergence/robustness
    smoke test, not a precision guarantee)."""
    rng = np.random.default_rng(0)
    size = 0.2
    local_corners = marker_local_corners(size)
    rvec_m_true = np.array([0.15, -0.1, 0.2])
    tvec_m_true = np.array([0.02, -0.03, 0.05])
    corner_obs = _project_all(rig, rvec_m_true, tvec_m_true, local_corners)
    noisy_obs = {
        vid: pts + rng.normal(scale=0.5, size=pts.shape) for vid, pts in corner_obs.items()
    }

    rvec, tvec, rms = solve_marker_pose(noisy_obs, rig, size)

    R_true, _ = cv2.Rodrigues(rvec_m_true)
    R_solved, _ = cv2.Rodrigues(rvec)
    np.testing.assert_allclose(R_solved, R_true, atol=5e-2)
    np.testing.assert_allclose(tvec, tvec_m_true, atol=5e-3)
    assert rms < 2.0  # sub-pixel-ish residual with 0.5px input noise


def test_solve_marker_pose_custom_initial_guess_still_converges(rig):
    """A deliberately poor initial guess should still converge to the true
    pose via the least_squares refinement (sanity check that the seed is
    just a seed, not load-bearing for correctness)."""
    size = 0.2
    local_corners = marker_local_corners(size)
    rvec_m_true = np.array([0.2, 0.0, 0.0])
    tvec_m_true = np.array([0.0, 0.0, 0.0])
    corner_obs = _project_all(rig, rvec_m_true, tvec_m_true, local_corners)

    bad_guess = (np.array([0.0, 1.0, 0.0]), np.array([0.5, 0.5, 0.5]))
    rvec, tvec, rms = solve_marker_pose(corner_obs, rig, size, initial_guess=bad_guess)

    R_true, _ = cv2.Rodrigues(rvec_m_true)
    R_solved, _ = cv2.Rodrigues(rvec)
    np.testing.assert_allclose(R_solved, R_true, atol=1e-4)
    np.testing.assert_allclose(tvec, tvec_m_true, atol=1e-4)
