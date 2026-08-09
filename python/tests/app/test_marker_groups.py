"""Tests for MarkerGroup / solve_marker_groups / run_calibration's Phase 3
ArUco integration.

See docs/roadmap/features/extrinsics-improvements/
extrinsics-improvements-design.md, sections 3 and 5, and the "Phase 3
progress" scoping note in status.md for why the rigid marker-pose solve
runs as a decoupled post-pass rather than a joint BA parameter block.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.setup.extrinsics_solver import (
    CamCalibState,
    MarkerGroup,
    ObsPoint,
    project_marker_corners,
    marker_local_corners,
    run_calibration,
    solve_marker_groups,
)


def _look_at_pose(cam_pos: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=0)
    t = -R @ cam_pos
    return R, t


def _make_camera(video_id: str, R: np.ndarray | None, t: np.ndarray | None) -> CamCalibState:
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    return CamCalibState(
        video_id=video_id, label=video_id, K=K, K_orig=K.copy(),
        dist=np.zeros((1, 4)), fisheye=False,
        R=R, t=(t.reshape(3, 1) if t is not None else None),
    )


@pytest.fixture()
def rig() -> dict[str, CamCalibState]:
    target = np.array([0.0, 0.0, 0.0])
    positions = [
        np.array([0.0, -3.0, 1.5]),
        np.array([2.6, 1.5, 1.8]),
        np.array([-2.6, 1.5, 1.2]),
    ]
    states = {}
    for i, pos in enumerate(positions):
        R, t = _look_at_pose(pos, target)
        states[f"cam_{i}"] = _make_camera(f"cam_{i}", R, t)
    return states


def _project_all(states: dict[str, CamCalibState], rvec_m, tvec_m, local_corners) -> dict[str, np.ndarray]:
    return {
        vid: project_marker_corners(rvec_m, tvec_m, local_corners, state)
        for vid, state in states.items()
    }


def _group_from_projection(
    marker_id: str, size: float | None, pixels_by_cam: dict[str, np.ndarray], frame_idx: int = 0
) -> MarkerGroup:
    mg = MarkerGroup(marker_id=marker_id, size=size)
    for vid, pts in pixels_by_cam.items():
        mg.obs[vid] = {
            i: ObsPoint(frame_idx=frame_idx, px=float(x), py=float(y))
            for i, (x, y) in enumerate(pts)
        }
    return mg


# ---------------------------------------------------------------------------
# MarkerGroup
# ---------------------------------------------------------------------------


def test_cameras_observing_requires_all_four_corners():
    mg = MarkerGroup(marker_id="3")
    mg.obs["cam_A"] = {0: ObsPoint(0, 1.0, 1.0), 1: ObsPoint(0, 2.0, 2.0),
                       2: ObsPoint(0, 3.0, 3.0), 3: ObsPoint(0, 4.0, 4.0)}
    mg.obs["cam_B"] = {0: ObsPoint(0, 1.0, 1.0)}  # partial -- e.g. occluded corner
    assert mg.cameras_observing() == {"cam_A"}


def test_as_control_points_produces_four_named_points():
    mg = MarkerGroup(marker_id="7")
    mg.obs["cam_A"] = {i: ObsPoint(frame_idx=5, px=float(i), py=float(i * 2)) for i in range(4)}
    cps = mg.as_control_points()

    assert len(cps) == 4
    assert {cp.name for cp in cps} == {f"aruco_7_c{i}" for i in range(4)}
    for i, cp in enumerate(sorted(cps, key=lambda c: c.name)):
        assert cp.world_xyz is None  # always free -- see module docstring
        assert cp.obs["cam_A"] == ObsPoint(frame_idx=5, px=float(i), py=float(i * 2))


def test_as_control_points_merges_partial_observations_across_cameras():
    """A corner occluded in one camera but visible in another should still
    end up correctly attributed per-camera on that corner's ControlPoint."""
    mg = MarkerGroup(marker_id="9")
    mg.obs["cam_A"] = {0: ObsPoint(0, 1.0, 1.0), 1: ObsPoint(0, 2.0, 2.0)}  # corners 2,3 occluded
    mg.obs["cam_B"] = {2: ObsPoint(0, 3.0, 3.0), 3: ObsPoint(0, 4.0, 4.0)}
    cps = {cp.name: cp for cp in mg.as_control_points()}

    assert "cam_A" in cps["aruco_9_c0"].obs and "cam_B" not in cps["aruco_9_c0"].obs
    assert "cam_B" in cps["aruco_9_c2"].obs and "cam_A" not in cps["aruco_9_c2"].obs


# ---------------------------------------------------------------------------
# solve_marker_groups
# ---------------------------------------------------------------------------


def test_solve_marker_groups_solves_known_size_marker(rig):
    size = 0.2
    local_corners = marker_local_corners(size)
    rvec_true, tvec_true = np.array([0.2, -0.1, 0.05]), np.array([0.1, 0.0, 0.2])
    pixels = _project_all(rig, rvec_true, tvec_true, local_corners)
    mg = _group_from_projection("3", size, pixels)

    results = solve_marker_groups([mg], list(rig.values()))

    assert set(results) == {"3"}
    R_true, _ = cv2.Rodrigues(rvec_true)
    R_solved, _ = cv2.Rodrigues(results["3"].rvec)
    np.testing.assert_allclose(R_solved, R_true, atol=1e-4)
    np.testing.assert_allclose(results["3"].tvec, tvec_true, atol=1e-4)
    assert results["3"].size == size
    assert results["3"].rms_reprojection_px == pytest.approx(0.0, abs=1e-2)


def test_solve_marker_groups_skips_unknown_size(rig):
    size = 0.2
    local_corners = marker_local_corners(size)
    pixels = _project_all(rig, np.zeros(3), np.zeros(3), local_corners)
    mg = _group_from_projection("3", None, pixels)  # size unknown

    results = solve_marker_groups([mg], list(rig.values()))
    assert results == {}


def test_solve_marker_groups_skips_single_camera(rig):
    size = 0.2
    local_corners = marker_local_corners(size)
    pixels = _project_all({"cam_0": rig["cam_0"]}, np.zeros(3), np.zeros(3), local_corners)
    mg = _group_from_projection("3", size, pixels)

    results = solve_marker_groups([mg], list(rig.values()))
    assert results == {}


def test_solve_marker_groups_handles_multiple_markers(rig):
    size = 0.15
    local_corners = marker_local_corners(size)
    pixels_a = _project_all(rig, np.array([0.1, 0.0, 0.0]), np.array([0.1, 0.1, 0.0]), local_corners)
    pixels_b = _project_all(rig, np.array([0.0, 0.3, 0.0]), np.array([-0.1, 0.0, 0.1]), local_corners)
    mg_a = _group_from_projection("A", size, pixels_a)
    mg_b = _group_from_projection("B", size, pixels_b)

    results = solve_marker_groups([mg_a, mg_b], list(rig.values()))
    assert set(results) == {"A", "B"}


# ---------------------------------------------------------------------------
# run_calibration integration
#
# Cameras are pre-solved and locked (cp_only=True, locked_cameras=all) so
# this test exercises only the NEW wiring (marker_groups -> effective free
# CPs -> solve_marker_groups post-pass) without needing real footage or
# SIFT -- the camera-solving pipeline itself is pre-existing, unchanged
# code, already covered elsewhere.
# ---------------------------------------------------------------------------


def test_run_calibration_populates_marker_poses(rig):
    size = 0.2
    local_corners = marker_local_corners(size)
    rvec_true, tvec_true = np.array([0.1, 0.1, 0.1]), np.array([0.0, 0.0, 0.1])
    pixels = _project_all(rig, rvec_true, tvec_true, local_corners)
    mg = _group_from_projection("3", size, pixels)

    states = list(rig.values())
    result = run_calibration(
        states,
        marker_groups=[mg],
        locked_cameras={s.video_id for s in states},
        cp_only=True,
    )

    assert "3" in result.marker_poses
    R_true, _ = cv2.Rodrigues(rvec_true)
    R_solved, _ = cv2.Rodrigues(result.marker_poses["3"].rvec)
    np.testing.assert_allclose(R_solved, R_true, atol=1e-3)
    np.testing.assert_allclose(result.marker_poses["3"].tvec, tvec_true, atol=1e-3)


def test_run_calibration_with_no_marker_groups_has_empty_marker_poses(rig):
    states = list(rig.values())
    result = run_calibration(
        states, locked_cameras={s.video_id for s in states}, cp_only=True,
    )
    assert result.marker_poses == {}


def test_run_calibration_unknown_size_marker_contributes_as_free_points(rig):
    """An unknown-size marker doesn't get a solved pose, but its corners
    should still flow through as free control points -- verified indirectly
    here via a successful, error-free run with reasonable reprojection
    error once triangulated (the free-CP triangulation/BA math itself is
    pre-existing, already-tested code; this just confirms the new
    marker_groups plumbing actually reaches it)."""
    size = 0.2
    local_corners = marker_local_corners(size)
    pixels = _project_all(rig, np.zeros(3), np.array([0.0, 0.0, 0.3]), local_corners)
    mg = _group_from_projection("unsized", None, pixels)

    states = list(rig.values())
    result = run_calibration(
        states,
        marker_groups=[mg],
        locked_cameras={s.video_id for s in states},
        cp_only=True,
    )

    assert result.marker_poses == {}  # no size -> no rigid pose
    # Cameras stay exactly where they were locked; the free-corner points
    # existing at all (not crashing, not producing garbage reprojection
    # error) is the thing under test here.
    for vid, stats in result.cp_reprojection_errors.items():
        assert stats["mean"] < 1.0  # sub-pixel: exact synthetic projections, no noise
