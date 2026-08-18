# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ObsPoint / ControlPoint and the control-point file format.

Covers Phase 2 of docs/roadmap/features/extrinsics-improvements/
extrinsics-improvements-design.md ("Per-control-point, per-frame
observations"): each camera's observation of a control point now carries its
own frame_idx, independent of every other camera and every other point.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from app.setup.extrinsics_solver import (
    CamCalibState,
    ControlPoint,
    ObsPoint,
    _undistort_control_obs,
    init_poses_pnp,
    load_control_points,
    save_control_points,
)


def _cam_state(video_id: str, label: str) -> CamCalibState:
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    return CamCalibState(
        video_id=video_id,
        label=label,
        K=K,
        K_orig=K.copy(),
        dist=np.zeros((1, 4)),
        fisheye=False,
    )


# ---------------------------------------------------------------------------
# ObsPoint / ControlPoint basics
# ---------------------------------------------------------------------------


def test_obs_point_fields():
    obs = ObsPoint(frame_idx=42, px=1.5, py=2.5)
    assert obs.frame_idx == 42
    assert obs.px == 1.5
    assert obs.py == 2.5


def test_control_point_obs_holds_obs_points():
    cp = ControlPoint(name="CP1")
    cp.obs["cam_A"] = ObsPoint(frame_idx=100, px=10.0, py=20.0)
    cp.obs["cam_B"] = ObsPoint(frame_idx=250, px=30.0, py=40.0)
    assert cp.obs["cam_A"].frame_idx == 100
    assert cp.obs["cam_B"].frame_idx == 250
    # Different cameras keep independent frame indices for the same point.
    assert cp.obs["cam_A"].frame_idx != cp.obs["cam_B"].frame_idx


# ---------------------------------------------------------------------------
# _undistort_control_obs
# ---------------------------------------------------------------------------


def test_undistort_control_obs_passthrough_with_zero_distortion():
    state = _cam_state("cam_A", "Cam A")
    cp = ControlPoint(name="CP1")
    cp.obs["cam_A"] = ObsPoint(frame_idx=7, px=100.0, py=150.0)
    result = _undistort_control_obs(cp, {"cam_A": state})
    assert result["cam_A"][0] == pytest.approx(100.0, abs=1e-3)
    assert result["cam_A"][1] == pytest.approx(150.0, abs=1e-3)


def test_undistort_control_obs_skips_unknown_camera():
    cp = ControlPoint(name="CP1")
    cp.obs["cam_A"] = ObsPoint(frame_idx=0, px=100.0, py=150.0)
    cp.obs["cam_ghost"] = ObsPoint(frame_idx=0, px=1.0, py=1.0)
    result = _undistort_control_obs(cp, {"cam_A": _cam_state("cam_A", "Cam A")})
    assert set(result) == {"cam_A"}


# ---------------------------------------------------------------------------
# File round-trip (version 2) and version-1 backward compatibility
# ---------------------------------------------------------------------------


def test_save_control_points_writes_version_2(tmp_path):
    states = [_cam_state("cam_A", "Cam A")]
    cp = ControlPoint(name="CP1")
    cp.obs["cam_A"] = ObsPoint(frame_idx=42, px=1.0, py=2.0)
    path = tmp_path / "cps.json"
    save_control_points([cp], states, str(path))
    data = json.loads(path.read_text())
    assert data["version"] == 2
    assert data["control_points"][0]["obs"]["Cam A"] == [42, 1.0, 2.0]


def test_round_trip_preserves_per_camera_frame_indices(tmp_path):
    states = [_cam_state("cam_A", "Cam A"), _cam_state("cam_B", "Cam B")]
    cp = ControlPoint(name="CP1", world_xyz=np.array([1.0, 2.0, 3.0]))
    # Same logical point, placed on different frames in each camera.
    cp.obs["cam_A"] = ObsPoint(frame_idx=100, px=10.0, py=20.0)
    cp.obs["cam_B"] = ObsPoint(frame_idx=250, px=30.0, py=40.0)

    path = tmp_path / "cps.json"
    save_control_points([cp], states, str(path))
    loaded = load_control_points(str(path), states)

    assert len(loaded) == 1
    loaded_cp = loaded[0]
    assert loaded_cp.obs["cam_A"] == ObsPoint(frame_idx=100, px=10.0, py=20.0)
    assert loaded_cp.obs["cam_B"] == ObsPoint(frame_idx=250, px=30.0, py=40.0)
    np.testing.assert_allclose(loaded_cp.world_xyz, [1.0, 2.0, 3.0])


def test_loading_version_1_file_uses_default_frame(tmp_path):
    states = [_cam_state("cam_A", "Cam A")]
    v1_data = {
        "version": 1,
        "control_points": [
            {"name": "CP1", "world_xyz": None, "obs": {"Cam A": [10.0, 20.0]}}
        ],
    }
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(v1_data))

    loaded = load_control_points(str(path), states, default_frame_by_id={"cam_A": 77})
    assert loaded[0].obs["cam_A"] == ObsPoint(frame_idx=77, px=10.0, py=20.0)


def test_loading_version_1_file_defaults_to_zero_without_default_frame(tmp_path):
    states = [_cam_state("cam_A", "Cam A")]
    v1_data = {
        "version": 1,
        "control_points": [
            {"name": "CP1", "world_xyz": None, "obs": {"Cam A": [10.0, 20.0]}}
        ],
    }
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(v1_data))

    loaded = load_control_points(str(path), states)
    assert loaded[0].obs["cam_A"].frame_idx == 0


def test_loading_unversioned_file_treated_as_version_1(tmp_path):
    """A file with no "version" key at all predates even version 1."""
    states = [_cam_state("cam_A", "Cam A")]
    data = {"control_points": [{"name": "CP1", "world_xyz": None, "obs": {"Cam A": [5.0, 6.0]}}]}
    path = tmp_path / "unversioned.json"
    path.write_text(json.dumps(data))

    loaded = load_control_points(str(path), states)
    assert loaded[0].obs["cam_A"] == ObsPoint(frame_idx=0, px=5.0, py=6.0)


def test_unsupported_version_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 99, "control_points": []}))
    with pytest.raises(ValueError):
        load_control_points(str(path), [])


def test_unmatched_camera_label_is_skipped(tmp_path):
    """A CP observation for a camera not present in *states* is silently dropped."""
    states = [_cam_state("cam_A", "Cam A")]
    data = {
        "version": 2,
        "control_points": [
            {
                "name": "CP1",
                "world_xyz": None,
                "obs": {"Cam A": [1, 1.0, 2.0], "Unknown Cam": [1, 3.0, 4.0]},
            }
        ],
    }
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(data))

    loaded = load_control_points(str(path), states)
    assert set(loaded[0].obs) == {"cam_A"}


# ---------------------------------------------------------------------------
# frame_idx does not affect the solver — it is provenance only
# ---------------------------------------------------------------------------


def _project_point(K: np.ndarray, R: np.ndarray, t: np.ndarray, xyz: np.ndarray) -> tuple[float, float]:
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(xyz.reshape(1, 3), rvec, t.reshape(3, 1), K, np.zeros(4))
    return float(proj[0, 0, 0]), float(proj[0, 0, 1])


@pytest.fixture()
def synthetic_pnp_setup():
    """A camera with a known pose and four non-coplanar world points."""
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    # A modest rotation + a translation that keeps all points in front of the camera.
    R, _ = cv2.Rodrigues(np.array([0.1, 0.2, 0.05]))
    t = np.array([0.3, -0.1, 5.0])
    world_pts = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    pixels = [_project_point(K, R, t, p) for p in world_pts]
    return K, R, t, world_pts, pixels


def _make_control_points(world_pts, pixels, video_id: str, frame_indices: list[int]) -> list[ControlPoint]:
    cps = []
    for i, (xyz, (px, py)) in enumerate(zip(world_pts, pixels)):
        cp = ControlPoint(name=f"CP{i}", world_xyz=xyz)
        cp.obs[video_id] = ObsPoint(frame_idx=frame_indices[i], px=px, py=py)
        cps.append(cp)
    return cps


def test_init_poses_pnp_recovers_known_pose(synthetic_pnp_setup):
    K, R, t, world_pts, pixels = synthetic_pnp_setup
    state = CamCalibState(
        video_id="cam_A", label="Cam A", K=K, K_orig=K.copy(),
        dist=np.zeros((1, 4)), fisheye=False,
    )
    cps = _make_control_points(world_pts, pixels, "cam_A", frame_indices=[0, 0, 0, 0])

    initialised = init_poses_pnp([state], cps)

    assert initialised == ["cam_A"]
    np.testing.assert_allclose(state.R, R, atol=1e-4)
    np.testing.assert_allclose(state.t.flatten(), t, atol=1e-3)


def test_init_poses_pnp_result_independent_of_frame_idx(synthetic_pnp_setup):
    """The exact same pixel observations, recorded on wildly different frames
    per point, must solve to the same pose -- frame_idx is provenance only,
    never a solver input."""
    K, R, t, world_pts, pixels = synthetic_pnp_setup

    def solve_with_frames(frame_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        state = CamCalibState(
            video_id="cam_A", label="Cam A", K=K, K_orig=K.copy(),
            dist=np.zeros((1, 4)), fisheye=False,
        )
        cps = _make_control_points(world_pts, pixels, "cam_A", frame_indices)
        init_poses_pnp([state], cps)
        return state.R, state.t

    R_a, t_a = solve_with_frames([0, 0, 0, 0])
    R_b, t_b = solve_with_frames([100, 4231, 7, 999999])

    np.testing.assert_allclose(R_a, R_b, atol=1e-9)
    np.testing.assert_allclose(t_a, t_b, atol=1e-9)
