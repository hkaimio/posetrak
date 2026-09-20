# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for marker body calibration from one moving camera (posetrak.calibration.video_marker_body).

The solving stage is checked on synthetic scenes: a camera orbits a stationary prop, and the
corners of its markers and of the surrounding anchor markers, and the reflective dots, are
projected with a little pixel noise. The geometry the solver recovers must match the scene.
"""
from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np
import pytest
import yaml

from app.setup.extrinsics_solver import CamCalibState, marker_local_corners
from app.setup.fiducial_markers import FiducialDetection, MarkerCornerObs
from posetrak.calibration.video_marker_body import (
    FrameObservation,
    VideoCalibrationOptions,
    collect_video_observations,
    _marker_scales,
    parse_marker_sizes,
    solve_video_body,
)

_K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
_WIDTH, _HEIGHT = 1920, 1080
_NOISE_PX = 0.3


def _state() -> CamCalibState:
    return CamCalibState(video_id="cam", label="cam", K=_K, K_orig=_K, dist=np.zeros((1, 4)), fisheye=False)


def _look_at(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera pose of a camera at *position* looking at *target*, world up +y."""
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    return R, -R @ position


def _face(position: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pose (marker to world) of a marker at *position* whose normal, its local +z, is *normal*."""
    z = normal / np.linalg.norm(normal)
    up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1), position


def _project(R: np.ndarray, t: np.ndarray, points: np.ndarray) -> np.ndarray:
    cam = points @ R.T + t
    return np.stack([_K[0, 0] * cam[:, 0] / cam[:, 2] + _K[0, 2], _K[1, 1] * cam[:, 1] / cam[:, 2] + _K[1, 2]], axis=1)


def _visible(R_cam, t_cam, pose, corners_world: np.ndarray) -> bool:
    centre_world = corners_world.mean(axis=0)
    to_camera = -R_cam.T @ t_cam - centre_world
    if pose[0][:, 2] @ to_camera < 0.35 * np.linalg.norm(to_camera):          # facing away, or too oblique
        return False
    pixels = _project(R_cam, t_cam, corners_world)
    return bool(((pixels > 5) & (pixels < [_WIDTH - 5, _HEIGHT - 5])).all())


class _Scene:
    """A prop at the origin, markers of known pose and size, dots, and a camera that orbits it."""

    def __init__(self, markers: dict[str, tuple[float, tuple[np.ndarray, np.ndarray]]], dots=(), lamps=(), cells=(), wobbly=()):
        self.markers = markers                       # id -> (size, (R, t) of the marker in the world)
        self.dots = [(np.array(p), np.array(n)) for p, n in dots]     # (position, outward normal)
        self.lamps = [np.array(p) for p in lamps]    # static bright points that are not on the prop
        self.cells = [np.array(p) for p in cells]    # bright cells inside a marker: reported as inside its quad
        self.wobbly = [np.array(p) for p in wobbly]  # near the prop, but their pixels do not agree on one point

    def corners(self, marker_id: str) -> np.ndarray:
        size, (R, t) = self.markers[marker_id]
        return marker_local_corners(size) @ R.T + t

    def frames(self, camera_poses, rng, *, outlier_rate: float = 0.0, tracklet_gap: int = 0) -> list[FrameObservation]:
        frames, tracklet_of, last_seen, next_id = [], {}, {}, 0
        for index, (R, t) in enumerate(camera_poses):
            seen = {}
            for marker_id in self.markers:
                corners = self.corners(marker_id)
                if _visible(R, t, self.markers[marker_id][1], corners):
                    pixels = _project(R, t, corners) + rng.normal(scale=_NOISE_PX, size=(4, 2))
                    if rng.random() < outlier_rate:
                        pixels[rng.integers(4)] += rng.choice([-40.0, 40.0], size=2)
                    seen[marker_id] = pixels
            dots = []
            for d, (position, normal) in enumerate(self.dots):
                to_camera = -R.T @ t - position
                if normal @ to_camera > 0.3 * np.linalg.norm(to_camera):
                    if d not in tracklet_of or (tracklet_gap and index - last_seen[d] > tracklet_gap):
                        tracklet_of[d], next_id = next_id, next_id + 1
                    last_seen[d] = index
                    u, v = _project(R, t, position[None])[0] + rng.normal(scale=_NOISE_PX, size=2)
                    dots.append((float(u), float(v), tracklet_of[d], np.inf))
            for cell in self.cells:
                if _visible(R, t, self.markers["2"][1], self.corners("2")):
                    u, v = _project(R, t, cell[None])[0]
                    dots.append((float(u), float(v), 2000, 0.6))
            for point in self.wobbly:
                u, v = _project(R, t, point[None])[0] + rng.normal(scale=2.5, size=2)
                dots.append((float(u), float(v), 3000, np.inf))
            for lamp in self.lamps:
                u, v = _project(R, t, lamp[None])[0]
                dots.append((float(u), float(v), 1000, np.inf))
            frames.append(FrameObservation(index, seen, dots))
        return frames


def _orbit(n: int, radius: float = 1.5, heights=(0.2, 0.7)) -> list[tuple[np.ndarray, np.ndarray]]:
    poses = []
    for height in heights:
        for k in range(n):
            angle = 2 * np.pi * k / n
            poses.append(_look_at(np.array([radius * np.sin(angle), height, radius * np.cos(angle)]), np.zeros(3)))
    return poses


def _prop_with_anchors() -> _Scene:
    """Marker 2 (the reference) on the front, marker 3 on the back facing away, anchors all around."""
    markers = {
        "2": (0.10, (np.eye(3), np.zeros(3))),
        "3": (0.08, _face(np.array([0.03, 0.02, -0.2]), np.array([0.15, 0.0, -1.0]))),
    }
    for k in range(8):
        angle = 2 * np.pi * k / 8
        radial = np.array([np.sin(angle), 0.0, np.cos(angle)])
        markers[str(10 + k)] = (0.15, _face(0.55 * radial + np.array([0.0, -0.05, 0.0]), radial))
    dots = [([0.05, 0.10, 0.02], [0.0, 0.0, 1.0]), ([-0.07, -0.05, 0.03], [0.0, 0.0, 1.0]),
            ([0.0, 0.02, -0.22], [0.0, 0.0, -1.0])]
    return _Scene(markers, dots, lamps=[[1.4, 0.3, 0.5]], cells=[[0.02, -0.03, 0.0]], wobbly=[[0.25, 0.1, 0.1]])


def _prop_options(**changes) -> VideoCalibrationOptions:
    options = VideoCalibrationOptions(
        body_markers={"2": 0.10, "3": 0.08}, reference_id="2", anchor_markers={str(10 + k): 0.15 for k in range(8)},
        **changes,
    )
    options.validate()
    return options


def _marker_corners(result_yaml: str, marker_id: str) -> np.ndarray:
    return np.array(next(m for m in yaml.safe_load(result_yaml)["markers"] if m.get("id") == marker_id)["corners"])


class TestPropWithAnchors:
    def test_the_prop_markers_are_recovered_in_the_reference_frame_and_anchors_are_left_out(self) -> None:
        scene = _prop_with_anchors()
        frames = scene.frames(_orbit(48), np.random.default_rng(1))

        result = solve_video_body(frames, _state(), _prop_options())

        assert np.allclose(_marker_corners(result.yaml, "2"), marker_local_corners(0.10), atol=1e-9)
        assert np.allclose(_marker_corners(result.yaml, "3"), scene.corners("3"), atol=3e-3)
        assert [m["id"] for m in yaml.safe_load(result.yaml)["markers"]] == ["2", "3"]
        assert result.unconnected == [] and result.reprojection_rms_px < 0.6
        assert result.marker_stats["3"].frames > 5

    def test_sizes_are_kept_per_marker(self) -> None:
        frames = _prop_with_anchors().frames(_orbit(48), np.random.default_rng(1))

        markers = yaml.safe_load(solve_video_body(frames, _state(), _prop_options()).yaml)["markers"]

        assert {m["id"]: m["size"] for m in markers} == {"2": 0.10, "3": 0.08}

    def test_wrong_corners_do_not_spoil_the_result(self) -> None:
        scene = _prop_with_anchors()
        frames = scene.frames(_orbit(48), np.random.default_rng(2), outlier_rate=0.04)

        result = solve_video_body(frames, _state(), _prop_options())

        assert np.allclose(_marker_corners(result.yaml, "3"), scene.corners("3"), atol=4e-3)

    def test_dots_are_triangulated_and_a_distant_bright_point_is_ignored(self) -> None:
        scene = _prop_with_anchors()
        frames = scene.frames(_orbit(60), np.random.default_rng(3))

        result = solve_video_body(frames, _state(), _prop_options(detect_dots=True))

        centers = np.array([d.center for d in result.dots])
        assert len(centers) == 3                                        # the lamp at 1.5 m is beyond the gate
        for position, _ in scene.dots:
            assert np.linalg.norm(centers - position, axis=1).min() < 3e-3
        assert all(d.views >= 12 and d.rms_px < 1.0 for d in result.dots)
        assert [m["type"] for m in yaml.safe_load(result.yaml)["markers"]].count("reflective_dot") == 3

    def test_a_bright_thing_near_the_prop_whose_views_disagree_is_not_a_dot(self) -> None:
        frames = _prop_with_anchors().frames(_orbit(60), np.random.default_rng(3))

        strict = solve_video_body(frames, _state(), _prop_options(detect_dots=True))
        lenient = solve_video_body(frames, _state(), _prop_options(detect_dots=True, dot_max_rms_px=5.0))

        assert (len(strict.dots), len(lenient.dots)) == (3, 4)
        assert max(d.rms_px for d in strict.dots) < 1.0

    def test_a_bright_cell_of_a_marker_is_not_a_dot_unless_the_exclusion_is_turned_off(self) -> None:
        frames = _prop_with_anchors().frames(_orbit(60), np.random.default_rng(3))

        excluded = solve_video_body(frames, _state(), _prop_options(detect_dots=True))
        included = solve_video_body(frames, _state(), _prop_options(detect_dots=True, dot_marker_exclude_scale=0.0))

        assert (len(excluded.dots), len(included.dots)) == (3, 4)

    def test_a_dot_whose_tracklet_breaks_is_still_one_dot(self) -> None:
        scene = _prop_with_anchors()
        frames = scene.frames(_orbit(60), np.random.default_rng(4), tracklet_gap=1)
        for frame in frames:                                            # split every dot's track in two
            frame.dots = [(u, v, t * 2 + (frame.frame % 2) if t < 1000 else t, s) for u, v, t, s in frame.dots]

        result = solve_video_body(frames, _state(), _prop_options(detect_dots=True, dot_min_views=8))

        assert len(result.dots) == 3

    def test_without_detect_dots_no_dot_is_written(self) -> None:
        frames = _prop_with_anchors().frames(_orbit(48), np.random.default_rng(1))
        assert solve_video_body(frames, _state(), _prop_options()).dots == []


class TestPropWithoutAnchors:
    def test_a_box_with_markers_on_every_side_needs_no_anchors(self) -> None:
        edge = 0.15
        faces = {"2": (0, 0, 1), "3": (1, 0, 0), "4": (0, 0, -1), "5": (-1, 0, 0), "6": (0, 1, 0), "7": (0, -1, 0)}
        scene = _Scene({m: (0.10, _face(edge * np.array(n, dtype=float), np.array(n, dtype=float))) for m, n in faces.items()})
        frames = scene.frames(_orbit(40, radius=1.2, heights=(-0.9, -0.3, 0.3, 0.9)), np.random.default_rng(5))
        options = VideoCalibrationOptions(body_markers={m: 0.10 for m in faces}, reference_id="2")

        result = solve_video_body(frames, _state(), options)

        assert result.unconnected == []
        # The reference marker's frame differs from the world frame: compare pairwise distances.
        solved = {m: _marker_corners(result.yaml, m).mean(axis=0) for m in faces}
        for a in faces:
            for b in faces:
                truth = np.linalg.norm(scene.corners(a).mean(axis=0) - scene.corners(b).mean(axis=0))
                assert np.linalg.norm(solved[a] - solved[b]) == pytest.approx(truth, abs=3e-3)


class TestConnectivity:
    def test_a_prop_marker_never_seen_with_a_linked_marker_is_left_out_and_reported(self) -> None:
        scene = _prop_with_anchors()
        frames = scene.frames(_orbit(48), np.random.default_rng(1))
        lonely = frames[0].markers["2"]
        frames.append(FrameObservation(99, {"9": lonely}))                       # marker 9 alone in a frame
        options = _prop_options()
        options.body_markers["9"] = 0.1

        result = solve_video_body(frames, _state(), options)

        assert result.unconnected == ["9"]
        assert '"9"' not in result.yaml

    def test_a_reference_seen_with_nothing_is_an_error(self) -> None:
        frames = [FrameObservation(i, {"2": np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float) + 500}) for i in range(5)]
        options = VideoCalibrationOptions(body_markers={"2": 0.1, "3": 0.1}, reference_id="2")

        with pytest.raises(ValueError, match="never seen in a frame together with another known marker"):
            solve_video_body(frames, _state(), options)


class TestMarkerScales:
    def test_a_point_is_scaled_by_how_far_it_lies_beyond_the_nearest_marker(self) -> None:
        square = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
        far_square = square + 1000.0
        points = np.array([[50.0, 50.0], [100.0, 50.0], [110.0, 50.0], [1050.0, 1050.0]])

        scales = _marker_scales(points, [square, far_square])

        assert scales == pytest.approx([0.0, 1.0, 1.2, 0.0])
        assert np.isinf(_marker_scales(points, [])).all()


class TestOptions:
    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"reference_id": "9"}, "reference id must be one of the body markers"),
            ({"anchor_markers": {"2": 0.2}}, "cannot be both body and anchor markers: 2"),
            ({"body_markers": {"2": 0.0}}, "marker sizes must be positive"),
            ({"stride": 0}, "stride must be at least 1"),
            ({"min_frame_markers": 1}, "at least 2"),
        ],
    )
    def test_unusable_settings_are_rejected(self, changes: dict, message: str) -> None:
        options = VideoCalibrationOptions(body_markers={"2": 0.1, "3": 0.1}, reference_id="2")
        for name, value in changes.items():
            setattr(options, name, value)

        with pytest.raises(ValueError, match=message):
            options.validate()

    def test_marker_sizes_read_ids_with_an_optional_size(self) -> None:
        assert parse_marker_sizes("2, 3:0.06", 0.1, "body markers") == {"2": 0.1, "3": 0.06}
        assert parse_marker_sizes("", 0.1, "anchor markers") == {}

    @pytest.mark.parametrize(
        ("spec", "default", "message"),
        [("2,3", None, "marker '2' has no size"), ("2:big", 0.1, "not a number"), ("2,2", 0.1, "listed twice")],
    )
    def test_marker_size_errors(self, spec: str, default, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_marker_sizes(spec, default, "body markers")


# ---------------------------------------------------------------------- collecting


def _detection(marker_id: str, x0: float) -> FiducialDetection:
    corners = [
        MarkerCornerObs(marker_type="aruco", marker_id=marker_id, corner_index=i, video_id="cam", frame_idx=0,
                        px=x0 + 40.0 * (i in (1, 2)), py=100.0 + 40.0 * (i in (2, 3)))
        for i in range(4)
    ]
    return FiducialDetection(marker_type="aruco", marker_id=marker_id, corners=corners)


class _Detector:
    def __init__(self, per_frame: dict[int, list[str]]):
        self.per_frame = per_frame

    def detect(self, image, video_id: str = "", frame_idx: int = 0):
        return [_detection(m, 100.0 + 200.0 * i) for i, m in enumerate(self.per_frame.get(frame_idx, []))]


def _dot_video(path, first_frame, last_frame):
    for i in range(first_frame, last_frame):
        frame = np.full((300, 300, 3), 20, dtype=np.uint8)
        cv2.circle(frame, (60 + i, 80), 6, (250, 250, 250), thickness=-1)     # a dot drifting 1 px per frame
        yield i, frame


class TestCollect:
    def test_frames_are_sampled_at_the_stride_and_only_frames_with_enough_known_markers_are_kept(self) -> None:
        per_frame = {0: ["2", "10"], 2: ["2", "99"], 4: ["2", "3", "10"], 6: ["3"]}    # 99 is not in the scene
        options = VideoCalibrationOptions(body_markers={"2": 0.1, "3": 0.1}, reference_id="2",
                                          anchor_markers={"10": 0.2}, stride=2, first_frame=0, last_frame=8)

        with patch("posetrak.calibration.video_marker_body.iter_frames", _dot_video):
            frames = collect_video_observations("v.mp4", _state(), options, detector=_Detector(per_frame))

        assert [(f.frame, sorted(f.markers)) for f in frames] == [(0, ["10", "2"]), (4, ["10", "2", "3"])]
        assert frames[0].markers["2"].shape == (4, 2) and frames[0].dots == []

    def test_dot_candidates_are_linked_into_one_tracklet_across_sampled_frames(self) -> None:
        options = VideoCalibrationOptions(
            body_markers={"2": 0.1, "3": 0.1}, reference_id="2", stride=1, first_frame=0, last_frame=6,
            detect_dots=True,
        )
        per_frame = {i: ["2", "3"] for i in range(6)}

        with patch("posetrak.calibration.video_marker_body.iter_frames", _dot_video):
            frames = collect_video_observations("v.mp4", _state(), options, detector=_Detector(per_frame))

        assert len(frames) == 6 and all(len(f.dots) == 1 for f in frames)
        assert len({f.dots[0][2] for f in frames}) == 1 and frames[0].dots[0][2] >= 0
        assert frames[5].dots[0][0] == pytest.approx(65.0, abs=1.0)
