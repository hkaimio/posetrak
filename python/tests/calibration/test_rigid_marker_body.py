# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for rigid marker body calibration (posetrak.calibration.rigid_marker_body).

The solving stage is checked on a synthetic scene: a body of known geometry is moved
through the view of three calibrated cameras, its markers and dots are projected, and
the geometry the solver recovers must match the one the scene was built from.
"""
from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np
import pytest
import yaml

from app.setup.db_context import SyncPoint, SyncTable
from app.setup.extrinsics_solver import CamCalibState, marker_local_corners
from app.setup.fiducial_markers import FiducialDetection, MarkerCornerObs, load_marker_body_yaml
from posetrak.calibration.rigid_marker_body import (
    CalibrationOptions,
    Observations,
    _gate_to_markers,
    cluster_dot_samples,
    collect_observations,
    robust_mean,
    solve_body,
    triangulate_point_multi_view,
)
from posetrak.detection.dot_blob_detector import BlobCandidate

_MARKER_SIZE = 0.1


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


# --------------------------------------------------------------------- helpers


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
    """Two cameras each seeing exactly one candidate does not mean those candidates
    are the same physical point. Simulates that: cam_b's "observation" is an unrelated
    pixel, not world_pt's real projection."""
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
    assert len(clusters) == 1 and len(clusters[0]) == 1


def test_robust_mean_trims_an_outlier_per_axis():
    samples = np.array([[1.0, 0.0, 0.0]] * 9 + [[100.0, 0.0, 0.0]])
    assert robust_mean(samples)[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no samples"):
        robust_mean(np.zeros((0, 3)))


# ------------------------------------------------------------- solving a body

# Marker "3" relative to the reference marker "2": turned 25 degrees about the
# reference's own y axis and 25 cm along x.
_REL_ROTATION = cv2.Rodrigues(np.array([0.0, np.deg2rad(25.0), 0.0]))[0]
_REL_TRANSLATION = np.array([0.25, 0.05, 0.0])
_DOT_A = np.array([0.04, 0.16, 0.02])
_DOT_B = np.array([-0.12, 0.08, 0.0])


def _cameras() -> dict[str, CamCalibState]:
    look_at = np.array([0.0, 0.0, 0.0])
    return {
        "a": _make_camera("a", np.array([1.0, 1.6, 0.4]), look_at),
        "b": _make_camera("b", np.array([-1.0, 1.6, 0.4]), look_at),
        "c": _make_camera("c", np.array([0.0, 1.4, 1.2]), look_at),
    }


def _body_pose(k: int) -> tuple[np.ndarray, np.ndarray]:
    """The body moves and turns a little from one instant to the next; its markers face +y, toward the cameras."""
    face_cameras = cv2.Rodrigues(np.array([-np.pi / 2, 0.0, 0.0]))[0]
    swing = cv2.Rodrigues(np.array([0.0, 0.0, np.deg2rad(8.0 * (k - 4))]))[0]
    return swing @ face_cameras, np.array([0.02 * k - 0.08, 0.0, 0.02 * (k % 3)])


def _scene(states: dict[str, CamCalibState], *, dots: bool) -> Observations:
    template = marker_local_corners(_MARKER_SIZE)
    corners_by_marker = {"2": template, "3": (_REL_ROTATION @ template.T).T + _REL_TRANSLATION}
    observations = Observations(markers={}, dots={})
    for k in range(9):
        R, t = _body_pose(k)
        bucket = 0.5 + 0.05 * k
        observations.markers[bucket] = {
            marker_id: {
                cam_id: np.array([_project(state, R @ c + t) for c in corners])
                for cam_id, state in states.items()
            }
            for marker_id, corners in corners_by_marker.items()
        }
        if dots:
            visible = [_DOT_A] if k % 2 == 0 else [_DOT_B]
            if k == 4:                                  # one camera sees both dots at once
                visible = [_DOT_A, _DOT_B]
            observations.dots[bucket] = {
                cam_id: [np.array(_project(state, R @ d + t)) for d in (visible if cam_id == "a" else visible[:1])]
                for cam_id, state in states.items()
            }
    return observations


def _corners(solved_yaml: str, marker_id: str) -> np.ndarray:
    marker = next(m for m in yaml.safe_load(solved_yaml)["markers"] if m.get("id") == marker_id)
    return np.array(marker["corners"])


class TestSolveBody:
    def test_the_markers_of_a_moving_body_are_recovered_in_the_reference_frame(self) -> None:
        states = _cameras()
        options = CalibrationOptions(marker_ids=["2", "3"], reference_id="2", marker_size=_MARKER_SIZE)

        result = solve_body(_scene(states, dots=False), states, options)

        template = marker_local_corners(_MARKER_SIZE)
        assert np.allclose(_corners(result.yaml, "2"), template, atol=1e-9)
        assert np.allclose(_corners(result.yaml, "3"), (_REL_ROTATION @ template.T).T + _REL_TRANSLATION, atol=2e-3)
        assert (result.reference_solved, result.marker_samples) == (9, {"3": 9})
        assert result.marker_corner_std["3"].max() < 2e-3

    def test_the_yaml_loads_as_a_marker_body(self) -> None:
        states = _cameras()
        options = CalibrationOptions(marker_ids=["2", "3"], reference_id="2", marker_size=_MARKER_SIZE, name="bokken")

        rig = load_marker_body_yaml(solve_body(_scene(states, dots=False), states, options).yaml)

        assert list(rig.marker_corners) == ["2", "3"]

    def test_reflective_dots_are_clustered_and_an_ambiguous_camera_is_ignored(self) -> None:
        states = _cameras()
        options = CalibrationOptions(marker_ids=["2", "3"], reference_id="2", marker_size=_MARKER_SIZE, detect_dots=True)

        result = solve_body(_scene(states, dots=True), states, options)

        centers = [np.array(m["center"]) for m in yaml.safe_load(result.yaml)["markers"] if m["type"] == "reflective_dot"]
        assert len(centers) == 2
        for expected in (_DOT_A, _DOT_B):
            assert min(np.linalg.norm(c - expected) for c in centers) < 3e-3
        # At k == 4 camera a sees both dots: it is ignored for that instant, and the two cameras that
        # each saw one candidate still give a sample of dot A.
        assert result.dot_ambiguous_buckets == 1
        assert (result.dot_samples, sorted(result.dot_cluster_sizes)) == (9, [4, 5])

    def test_without_detect_dots_no_dot_is_written(self) -> None:
        states = _cameras()
        options = CalibrationOptions(marker_ids=["2", "3"], reference_id="2", marker_size=_MARKER_SIZE)

        result = solve_body(_scene(states, dots=True), states, options)

        assert "reflective_dot" not in result.yaml and result.dot_samples == 0

    def test_a_marker_never_seen_with_the_reference_is_left_out_and_logged(self) -> None:
        states = _cameras()
        options = CalibrationOptions(marker_ids=["2", "3", "9"], reference_id="2", marker_size=_MARKER_SIZE)
        log: list[str] = []

        result = solve_body(_scene(states, dots=False), states, options, log=log.append)

        assert '"9"' not in result.yaml
        assert any("marker '9': never seen together with the reference" in line for line in log)

    def test_a_reference_marker_no_camera_pair_saw_is_an_error(self) -> None:
        states = _cameras()
        scene = _scene(states, dots=False)
        for by_marker in scene.markers.values():
            by_marker["2"] = {"a": by_marker["2"]["a"]}          # one camera only
        options = CalibrationOptions(marker_ids=["2", "3"], reference_id="2", marker_size=_MARKER_SIZE)

        with pytest.raises(ValueError, match="never seen by 2 or more cameras"):
            solve_body(scene, states, options)


class TestOptions:
    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"reference_id": "7"}, "reference id must be one of the marker ids"),
            ({"stride": 0}, "stride must be at least 1"),
            ({"min_cameras": 1}, "min_cameras at least 2"),
            ({"marker_size": 0.0}, "marker size must be positive"),
        ],
    )
    def test_unusable_settings_are_rejected(self, changes: dict, message: str) -> None:
        options = CalibrationOptions(marker_ids=["2", "3"], reference_id="2", marker_size=0.05)
        for name, value in changes.items():
            setattr(options, name, value)

        with pytest.raises(ValueError, match=message):
            options.validate()


# ------------------------------------------------------ collecting observations


def _detection(marker_id: str, x0: float) -> FiducialDetection:
    corners = [
        MarkerCornerObs(marker_type="aruco", marker_id=marker_id, corner_index=i, video_id="cam", frame_idx=0,
                        px=x0 + 10.0 * (i in (1, 2)), py=50.0 + 10.0 * (i in (2, 3)))
        for i in range(4)
    ]
    return FiducialDetection(marker_type="aruco", marker_id=marker_id, corners=corners)


class _FakeDetector:
    def detect(self, image, video_id: str = "", frame_idx: int = 0):
        return [_detection("2", 100.0), _detection("77", 300.0)]   # 77 is not part of the body


def _frames(path, first_frame, last_frame):
    for i in range(first_frame, last_frame):
        yield i, np.zeros((120, 200, 3), dtype=np.uint8)


def _sync(*video_ids: str) -> SyncTable:
    return SyncTable(
        [SyncPoint(camera_instance_id=v, shot_video_id=v, video_frame=0, timestamp_s=0.0) for v in video_ids],
        {v: 30.0 for v in video_ids},
    )


class TestCollectObservations:
    def test_body_markers_are_bucketed_by_time_and_camera_at_the_stride(self) -> None:
        states = {"a": _make_camera("a", np.array([1.0, 1.6, 0.4]), np.zeros(3)),
                  "b": _make_camera("b", np.array([-1.0, 1.6, 0.4]), np.zeros(3))}
        for cam_id, state in states.items():
            state.file_path = f"{cam_id}.mp4"
        options = CalibrationOptions(marker_ids=["2", "3"], reference_id="2", marker_size=_MARKER_SIZE, stride=6)

        with patch("posetrak.calibration.rigid_marker_body.iter_frames", _frames):
            observations = collect_observations(
                states, _sync("sv-a", "sv-b"), {"a": "sv-a", "b": "sv-b"}, 0.0, 1.0, options, detector=_FakeDetector(),
            )

        assert sorted(observations.markers) == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])     # frames 0, 6, 12, ...
        for by_marker in observations.markers.values():
            assert list(by_marker) == ["2"]                                                    # 77 ignored
            assert sorted(by_marker["2"]) == ["a", "b"] and by_marker["2"]["a"].shape == (4, 2)
        assert not observations.dots

    def test_a_camera_without_sync_coverage_is_skipped(self) -> None:
        states = {"a": _make_camera("a", np.array([1.0, 1.6, 0.4]), np.zeros(3)),
                  "b": _make_camera("b", np.array([-1.0, 1.6, 0.4]), np.zeros(3))}
        for cam_id, state in states.items():
            state.file_path = f"{cam_id}.mp4"
        options = CalibrationOptions(marker_ids=["2"], reference_id="2", marker_size=_MARKER_SIZE)
        log: list[str] = []

        with patch("posetrak.calibration.rigid_marker_body.iter_frames", _frames):
            observations = collect_observations(
                states, _sync("sv-a"), {"a": "sv-a", "b": "sv-b"}, 0.0, 1.0, options,
                detector=_FakeDetector(), log=log.append,
            )

        assert all(list(by_marker["2"]) == ["a"] for by_marker in observations.markers.values())
        assert any("SKIP b" in line for line in log)


def _blob(cx: float, cy: float) -> BlobCandidate:
    return BlobCandidate(cx=cx, cy=cy, area=20.0, compactness=0.9, bbox=(0, 0, 1, 1),
                         major_axis_px=5.0, minor_axis_px=5.0)


class TestDotGate:
    def _options(self, **changes) -> CalibrationOptions:
        return CalibrationOptions(marker_ids=["2"], reference_id="2", marker_size=_MARKER_SIZE,
                                  dot_gate_radius_mult=2.0, **changes)

    def test_a_dot_near_the_marker_outside_its_footprint_is_kept_and_clutter_is_dropped(self) -> None:
        marker = [_detection("2", 100.0)]         # a 10 px square at (100..110, 50..60), diagonal ~14 px
        near, inside, far = _blob(125.0, 55.0), _blob(105.0, 55.0), _blob(180.0, 55.0)

        kept = _gate_to_markers([near, inside, far], marker, self._options())

        assert kept == [near]

    def test_a_frame_without_a_visible_marker_keeps_no_dot(self) -> None:
        assert _gate_to_markers([_blob(105.0, 55.0)], [], self._options()) == []

    def test_the_footprint_exclusion_can_be_turned_off(self) -> None:
        kept = _gate_to_markers([_blob(105.0, 55.0)], [_detection("2", 100.0)], self._options(dot_quad_exclude_scale=0.0))
        assert len(kept) == 1
