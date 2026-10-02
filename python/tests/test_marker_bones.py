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


def test_names_ask_for_directions_relative_to_the_limbs_own_side():
    from posetrak.markers.catalog import Slot
    from posetrak.markers.naming import role_direction

    assert role_direction(Slot("knee_lat_R", "shin.R")) == (1.0, 0.0)
    assert role_direction(Slot("knee_lat_L", "shin.L")) == (-1.0, 0.0)     # lateral is away from the midline
    assert role_direction(Slot("knee_med_L", "shin.L")) == (1.0, 0.0)
    assert role_direction(Slot("knee_front_R", "shin.R")) == (0.0, 1.0)
    assert role_direction(Slot("toe_pinky_L", "foot.L")) == (-1.0, 0.0)    # the little toe is lateral
    assert role_direction(Slot("toe_L", "foot.L")) == (1.0, 0.0)
    assert role_direction(Slot("temple_L", "head")) == (-1.0, 0.0)
    assert role_direction(Slot("sacrum", "hips")) == (0.0, -1.0)
    assert role_direction(Slot("wrist_rad_R", "forearm.R")) is None        # depends on how the hand is held


def test_a_disagreeing_direction_costs_and_an_agreeing_one_does_not():
    from types import SimpleNamespace

    from posetrak.markers.catalog import Slot
    from posetrak.markers.naming import direction_penalty

    # a pose where the bone stands upright and the body faces +y (right is +x)
    frames = {"shoulder.R": np.eye(4), "shoulder.L": np.eye(4), "bone": np.eye(4)}
    frames["shoulder.R"][:3, 3] = [0.2, 0, 1.4]
    frames["shoulder.L"][:3, 3] = [-0.2, 0, 1.4]
    up_bone = np.eye(4)
    up_bone[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]       # local y -> world z (the bone is upright)
    frames["bone"] = up_bone
    pose = SimpleNamespace(times=np.array([0.0]), joint_frames=lambda t: frames)
    lateral = Slot("knee_lat_R", "bone")
    assert direction_penalty(pose, lateral, "bone", np.array([0.05, 0.0, 0.0])) < 1e-6       # to the right: yes
    assert direction_penalty(pose, lateral, "bone", np.array([-0.05, 0.0, 0.0])) > 5.9       # to the left: no
    assert direction_penalty(pose, lateral, "bone", np.array([0.005, 0.0, 0.0])) == 1.5      # hugs the axis
    assert direction_penalty(pose, Slot("wrist_rad_R", "bone"), "bone", np.array([-0.05, 0, 0])) == 0.0


def test_reference_placement_round_trips_and_mirroring_negates_the_lateral_part():
    from posetrak.markers.reference import body_axes, from_reference, to_reference

    right, front, up = body_axes(np.array([0.2, 0, 1.4]), np.array([-0.2, 0, 1.4]), np.array([0, 0.05, 0.6]))
    assert np.allclose(right, [1, 0, 0]) and np.allclose(front @ up, 0, atol=1e-9)
    head_r, tail_r = np.array([0.1, 0, 0.9]), np.array([0.1, 0.0, 0.45])          # right shin, pointing down
    head_l, tail_l = np.array([-0.1, 0, 0.9]), np.array([-0.1, 0.0, 0.45])        # its mirror image
    position = np.array([0.16, 0.03, 0.8])                                         # outside and in front of it
    normal = np.array([1.0, 0.2, 0.0])
    placement = to_reference(head_r, tail_r, front, up, position, normal)
    back, back_normal = from_reference(head_r, tail_r, front, up, placement)
    assert np.allclose(back, position) and np.allclose(back_normal, normal / np.linalg.norm(normal))
    mirrored = to_reference(head_l, tail_l, front, up, position * [-1, 1, 1], normal * [-1, 1, 1])
    assert np.isclose(mirrored["along"], placement["along"]) and np.isclose(mirrored["anterior"], placement["anterior"])
    assert np.isclose(mirrored["lateral"], -placement["lateral"])
    assert np.isclose(mirrored["normal"][1], -placement["normal"][1])


def test_a_bone_pointing_forward_uses_up_as_its_anterior_direction():
    from posetrak.markers.reference import bone_frame

    front, up = np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    frame = bone_frame(np.zeros(3), np.array([0, 0.15, -0.03]), front, up)          # a foot
    assert frame[:, 2] @ up > 0.9                                                    # anterior is dorsal


def test_placements_are_written_into_a_module_keeping_its_comments_and_other_markers():
    from posetrak.markers.catalog import load_marker_module
    from posetrak.markers.reference import apply_placements

    text = (
        "# a comment\nmodule: m\nmarkers:\n"
        "  - {name: knee_R, parent_joint: shin.R, along: 0.0, mirror: true}   # lateral\n"
        "  - {name: nose, parent_joint: head}\n"
    )
    new, changed = apply_placements(text, {"knee_R": {"along": 0.05, "lateral": 0.06, "anterior": -0.01,
                                                      "normal": [0.0, 1.0, 0.1]}})
    assert changed == ["knee_R"] and "# a comment" in new and "# lateral" in new and "{name: nose, parent_joint: head}" in new
    module = load_marker_module(new)
    knee_r, knee_l = module.slots[0], module.slots[1]
    assert (knee_r.along, knee_r.lateral, knee_r.anterior, knee_r.normal) == (0.05, 0.06, -0.01, (0.0, 1.0, 0.1))
    assert (knee_l.lateral, knee_l.normal) == (-0.06, (0.0, -1.0, 0.1))        # the mirror image


def test_along_is_kept_as_a_distance_when_the_armature_ends_its_bone_elsewhere():
    from posetrak.markers.reference import rescale_along

    placement = {"along": 2.5, "bone_length": 0.024}                 # 6 cm up a pelvis bone the armature made short
    assert abs(rescale_along(placement, 0.12)["along"] - 0.5) < 1e-9  # half of a 12 cm pelvis bone
    assert rescale_along({"along": 0.3}, 0.12)["along"] == 0.3        # nothing to rescale from


def test_a_slot_with_a_full_placement_is_expected_where_the_reference_body_has_it():
    from types import SimpleNamespace

    from posetrak.markers.calibrate import CalibrationContext, nominal_placement
    from posetrak.markers.catalog import Slot

    # a bone from (0,0,1) down to (0,0,0.5); the joint frame is rotated so that its local y runs down
    bones = {"bone": (np.array([0.0, 0, 1.0]), np.array([0.0, 0, 0.5]))}
    frame = np.eye(4)
    frame[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]   # local y -> world -z, local z -> world +y... (front)
    frame[:3, 3] = [0, 0, 1.0]
    ctx = CalibrationContext(None, "", "", {}, None, {}, None, SimpleNamespace(layout=None), [], bones, {"bone": frame})
    slot = Slot("s", "bone", along=0.5, lateral=0.0, anterior=0.06, normal=(0.0, 0.0, 1.0))
    offset, normal = nominal_placement(ctx, slot)
    # half way down the bone, 6 cm in front of it: in the bone's own frame 0.25 m along y
    assert abs(offset[1] - 0.25) < 1e-9 and abs(np.linalg.norm(offset) - np.hypot(0.25, 0.06)) < 1e-9
    assert abs(np.linalg.norm(normal) - 1.0) < 1e-9


def test_the_attachment_set_keeps_markers_the_calibration_did_not_find_at_their_reference_place():
    import yaml
    from types import SimpleNamespace

    from posetrak.markers.attachment import attachment_document
    from posetrak.markers.calibrate import Placement

    layout = SimpleNamespace(joints=[SimpleNamespace(name="shin.R", parent="thigh.R", bone_tip_offset=np.array([0, 0.4, 0]),
                                                      offset=np.zeros(3))])
    pose = SimpleNamespace(layout=layout)
    found = Placement("knee_lat_R", "shin.R", np.array([0.05, 0.02, 0.0]), 0.004, 800, 5)
    nominal = {"knee_lat_R": ("shin.R", np.array([0.04, 0.0, 0.0]), None),
               "knee_front_R": ("shin.R", np.array([0.0, 0.0, 0.06]), np.array([0.0, 0.0, 1.0]))}
    doc = yaml.safe_load(attachment_document([found], pose, module="m", topology=None, topology_hash=None,
                                             calibration={}, nominal=nominal,
                                             normals={"knee_lat_R": np.array([1.0, 0.0, 0.0])}))
    by_name = {m["name"]: m for m in doc["markers"]}
    assert by_name["knee_lat_R"]["offset"] == [0.05, 0.02, 0.0] and by_name["knee_lat_R"]["confidence"] == "high"
    assert by_name["knee_front_R"]["confidence"] == "nominal" and by_name["knee_front_R"]["normal"] == [0.0, 0.0, 1.0]
