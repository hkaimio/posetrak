# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Marker-to-bone fitting (posetrak.markers.bones) and tracklet filtering on synthetic data."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from app.setup.extrinsics_solver import CamCalibState  # noqa: E402
from posetrak.markers.bones import BodyPose, fit_bones, marker_track, triangulate  # noqa: E402


def _camera(position: np.ndarray, look_at: np.ndarray) -> CamCalibState:
    forward = look_at - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.array([right, down, forward])
    K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]])
    return CamCalibState(
        video_id="c", label="c", R=R, t=(-R @ position).reshape(3, 1), K=K, K_orig=K.copy(),
        dist=np.zeros((1, 4)), fisheye=False, file_path="", first_frame=0, last_frame=0,
    )


def _project(state: CamCalibState, X: np.ndarray) -> tuple[float, float]:
    p = state.K @ (state.R @ X + state.t.ravel())
    return float(p[0] / p[2]), float(p[1] / p[2])


def test_triangulation_recovers_the_point_seen_by_two_cameras():
    states = {
        "a": _camera(np.array([3.0, 0.0, 1.5]), np.array([0.0, 0.0, 1.0])),
        "b": _camera(np.array([0.0, 3.0, 1.5]), np.array([0.0, 0.0, 1.0])),
    }
    X = np.array([0.1, -0.2, 1.1])
    got = triangulate([("a", _project(states["a"], X)), ("b", _project(states["b"], X))], states)
    assert np.allclose(got, X, atol=1e-6)
    assert triangulate([("a", _project(states["a"], X))], states) is None


class _FakePose(BodyPose):
    """Two bones: 'moving' swings about the origin, 'still' stays put."""

    def __init__(self) -> None:  # noqa: D107
        self.times = np.arange(0, 2, 0.05)

    def joint_frames(self, t: float):
        c, s = np.cos(t * 2), np.sin(t * 2)
        moving = np.eye(4)
        moving[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        moving[:3, 3] = [0.0, 0.0, 1.0]
        return {"moving": moving, "still": np.eye(4)}


def test_a_marker_on_a_moving_bone_has_no_spread_in_that_bone_frame():
    pose = _FakePose()
    offset = np.array([0.2, 0.05, 0.1])
    times = pose.times
    points = np.array([(pose.joint_frames(t)["moving"] @ np.append(offset, 1.0))[:3] for t in times])
    fits = fit_bones(times, points, pose)
    assert fits[0].joint == "moving" and fits[0].spread_m < 1e-9
    assert np.allclose(fits[0].offset, offset)
    assert fits[1].joint == "still" and fits[1].spread_m > 0.1


def test_marker_track_needs_two_cameras_at_the_same_time():
    states = {
        "a": _camera(np.array([3.0, 0.0, 1.5]), np.array([0.0, 0.0, 1.0])),
        "b": _camera(np.array([0.0, 3.0, 1.5]), np.array([0.0, 0.0, 1.0])),
    }
    X = np.array([0.0, 0.0, 1.0])
    obs = {
        0: [("a", _project(states["a"], X)), ("b", _project(states["b"], X))],
        1: [("a", _project(states["a"], X))],
    }
    times, points = marker_track(obs, states)
    assert len(times) == 1 and np.allclose(points[0], X, atol=1e-6)


def test_a_fit_is_decisive_only_when_clearly_the_best_bone():
    from posetrak.markers.bones import BoneFit
    from posetrak.markers.calibrate import GroupFit

    def fit(joint, spread):
        return BoneFit(joint, spread, np.zeros(3), 100)

    clear = GroupFit((0, 6), [], 100, [fit("shin.R", 0.005), fit("thigh.R", 0.012)])
    close = GroupFit((0, 6), [], 100, [fit("shin.R", 0.005), fit("foot.R", 0.006)])
    loose = GroupFit((0, 6), [], 100, [fit("shin.R", 0.05), fit("thigh.R", 0.2)])
    assert clear.decisive().joint == "shin.R"
    assert close.decisive() is None      # a neighbouring bone fits almost as well
    assert loose.decisive() is None      # not rigid enough on any bone


def test_votes_of_different_windows_for_the_same_place_are_clustered():
    from posetrak.markers.bones import BoneFit
    from posetrak.markers.calibrate import GroupFit, cluster_votes

    def vote(window, joint, offset):
        return GroupFit(window, [], 100, [BoneFit(joint, 0.004, np.array(offset), 100), BoneFit("hips", 0.05, np.zeros(3), 100)])

    clusters = cluster_votes([
        vote((0, 6), "shin.R", [0.0, 0.05, 0.10]),
        vote((3, 9), "shin.R", [0.005, 0.052, 0.10]),   # same marker, a different window
        vote((3, 9), "shin.R", [0.0, -0.05, 0.10]),     # the medial one, 10 cm away
        vote((6, 12), "thigh.R", [0.0, 0.05, 0.10]),    # same offset, other bone
    ])
    by_joint = sorted((c.joint, len(c.windows)) for c in clusters)
    assert by_joint == [("shin.R", 1), ("shin.R", 2), ("thigh.R", 1)]
