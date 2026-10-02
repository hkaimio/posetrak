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


def test_votes_of_different_windows_for_the_same_place_are_clustered():
    from posetrak.markers.calibrate import GroupTrack, SlotTest, cluster_tests

    def track(window):
        return GroupTrack(window, [], np.zeros(0), np.zeros((0, 3)))

    def test(slot, joint, offset, spread=0.004):
        return SlotTest(slot, joint, spread, 0.01, np.array(offset), 100)

    clusters = cluster_tests([
        (track((0, 6)), [test("knee_lat_R", "shin.R", [0.0, 0.05, 0.10])]),
        (track((3, 9)), [test("knee_lat_R", "shin.R", [0.005, 0.052, 0.10])]),   # same marker, another window
        (track((3, 9)), [test("knee_med_R", "shin.R", [0.0, -0.05, 0.10])]),     # 10 cm away
        (track((6, 12)), [test("hip_R", "thigh.R", [0.0, 0.05, 0.10])]),         # same offset, other bone
        (track((6, 12)), []),                                                    # fits no slot
    ])
    assert sorted((c.joint, len(c.windows)) for c in clusters) == [("shin.R", 1), ("shin.R", 2), ("thigh.R", 1)]


def test_a_track_must_be_rigid_on_the_bone_and_near_where_the_slot_is_expected():
    from types import SimpleNamespace

    from posetrak.markers.calibrate import CalibrationContext, GroupTrack, test_track
    from posetrak.markers.catalog import Slot

    pose = _FakePose()
    pose.layout = SimpleNamespace(joints=[SimpleNamespace(name="moving", parent="", bone_tip_offset=np.array([0, 0.4, 0]),
                                                           offset=np.zeros(3))])
    knee = Slot("knee", "moving", along=0.0)
    ankle = Slot("ankle", "moving", along=1.0)
    ctx = CalibrationContext(None, "", "", {}, None, {}, None, pose, [knee, ankle])
    times = pose.times
    near_knee = np.array([0.0, 0.03, 0.04])
    points = np.array([(pose.joint_frames(t)["moving"] @ np.append(near_knee, 1.0))[:3] for t in times])
    tests = test_track(ctx, GroupTrack((0, 2), [], times, points))
    assert [t.slot for t in tests] == ["knee"]            # the ankle is 0.4 m further along the bone
    assert tests[0].spread_m < 1e-9


def test_single_camera_fit_recovers_the_offset_when_the_bone_turns():
    from posetrak.markers.bones import fit_offset_from_rays

    state = _camera(np.array([3.0, 0.0, 1.5]), np.array([0.0, 0.0, 1.0]))
    pose = _FakePose()
    offset = np.array([0.2, 0.05, 0.1])
    times = pose.times
    pixels = np.array([_project(state, (pose.joint_frames(t)["moving"] @ np.append(offset, 1.0))[:3]) for t in times])
    fit = fit_offset_from_rays(times, pixels, state, pose, "moving", prior=np.array([0.0, 0.0, 0.0]))
    assert np.linalg.norm(fit.offset - offset) < 0.01 and fit.reproj_rms_px < 0.5


def test_one_marker_fills_one_slot_and_two_markers_on_one_bone_take_different_slots():
    from types import SimpleNamespace

    from posetrak.markers.calibrate import CalibrationContext, Chain, ChainFit, SlotTest, assign_slots
    from posetrak.markers.catalog import Slot

    ctx = CalibrationContext(None, "", "", {}, None, {}, None, None,
                             [Slot("knee_lat_R", "shin.R", along=0.0), Slot("knee_med_R", "shin.R", along=0.0),
                              Slot("hip_R", "thigh.R", along=0.0)])

    def fit(window, tests):
        return ChainFit(Chain([window], set(), np.zeros(0), np.zeros((0, 3))), tests)

    def test(slot, joint, offset, cost_dist):
        return SlotTest(slot, joint, 0.005, cost_dist, np.array(offset), 200)

    a = [test("knee_lat_R", "shin.R", [0.05, 0, 0], 0.02), test("knee_med_R", "shin.R", [0.05, 0, 0], 0.03)]
    b = [test("knee_lat_R", "shin.R", [-0.05, 0, 0], 0.02), test("knee_med_R", "shin.R", [-0.05, 0, 0], 0.025)]
    c = [test("hip_R", "thigh.R", [0.0, 0.05, 0.0], 0.02), test("knee_lat_R", "thigh.R", [0.0, 0.05, 0.0], 0.1)]
    placements, left = assign_slots(ctx, [fit((0, 6), a), fit((3, 9), a), fit((0, 6), b), fit((0, 6), c)])
    got = {p.slot: p.offset[0] for p in placements}
    assert sorted(got) == ["hip_R", "knee_lat_R", "knee_med_R"]     # every slot once
    assert got["knee_lat_R"] != got["knee_med_R"] and not left      # the two markers got different slots
