# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for MarkerRigConfig / MarkerRigDetector / anchor_from_marker_rig
(Phase 8, design doc section 9 Tier A -- portable non-planar calibration
rig).

See docs/roadmap/features/extrinsics-improvements/
extrinsics-improvements-design.md, section 9, and status.md's 2026-08-11/12
entries for how the "explicit" rig config this exercises actually gets
built in practice (an orbit-video self-calibration, not a hand-derived
"box" shape -- see load_rig_config's own docstring for why).
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from app.setup.extrinsics_solver import CamCalibState, run_calibration
from app.setup.fiducial_markers import (
    FiducialDetection,
    MarkerCornerObs,
    MarkerRigConfig,
    MarkerRigDetector,
    anchor_from_marker_rig,
    load_rig_config,
    marker_local_corners,
)

# ---------------------------------------------------------------------------
# A small synthetic non-planar rig: 3 markers on 3 mutually orthogonal faces
# of a cube corner (ids "10"/"11"/"12") -- genuinely non-coplanar, unlike a
# ChArUco board, so this is a real test of section 9's actual claim, not
# just a relabelled planar case.
# ---------------------------------------------------------------------------

_SIZE = 0.1
_HALF_CUBE = 0.15

# Cyclic permutation rotations (det=+1) mapping the local +Z-facing marker
# pattern (marker_local_corners' own convention) onto the cube's +X and +Y
# faces; the +Z face needs no rotation.
_R_TO_PLUS_X = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_R_TO_PLUS_Y = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def _cube_rig_config() -> MarkerRigConfig:
    local = marker_local_corners(_SIZE)
    corners_z = local + np.array([0.0, 0.0, _HALF_CUBE])
    corners_x = (local @ _R_TO_PLUS_X.T) + np.array([_HALF_CUBE, 0.0, 0.0])
    corners_y = (local @ _R_TO_PLUS_Y.T) + np.array([0.0, _HALF_CUBE, 0.0])
    return MarkerRigConfig(
        rig_id="test_cube",
        marker_corners={"10": corners_x, "11": corners_y, "12": corners_z},
    )


def _assert_noncoplanar(points: np.ndarray) -> None:
    centred = points - points.mean(axis=0)
    _, sv, _ = np.linalg.svd(centred)
    assert sv[-1] > 1e-6, "test rig is degenerate (all points coplanar or worse)"


def test_cube_rig_fixture_is_genuinely_noncoplanar():
    config = _cube_rig_config()
    all_corners = np.concatenate(list(config.marker_corners.values()), axis=0)
    _assert_noncoplanar(all_corners)


def _det(marker_id: str, config: MarkerRigConfig, video_id: str, px_py) -> FiducialDetection:
    corners = [
        MarkerCornerObs(
            marker_type="aruco", marker_id=marker_id, corner_index=i,
            video_id=video_id, frame_idx=0, px=float(px), py=float(py),
        )
        for i, (px, py) in enumerate(px_py)
    ]
    return FiducialDetection(marker_type="aruco", marker_id=marker_id, corners=corners)


# ---------------------------------------------------------------------------
# load_rig_config
# ---------------------------------------------------------------------------


def test_load_rig_config_explicit_round_trips(tmp_path):
    config = _cube_rig_config()
    path = tmp_path / "rig.json"
    payload = {
        "v": 1, "shape": "explicit", "rig_id": config.rig_id,
        "marker_corners": {k: v.tolist() for k, v in config.marker_corners.items()},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_rig_config(str(path))
    assert loaded.rig_id == config.rig_id
    assert set(loaded.marker_corners) == set(config.marker_corners)
    for marker_id, corners in config.marker_corners.items():
        np.testing.assert_allclose(loaded.marker_corners[marker_id], corners)


def test_load_rig_config_defaults_rig_id_to_filename(tmp_path):
    path = tmp_path / "my_rig.json"
    path.write_text(json.dumps({"v": 1, "shape": "explicit", "marker_corners": {}}), encoding="utf-8")
    loaded = load_rig_config(str(path))
    assert loaded.rig_id == "my_rig"


def test_load_rig_config_box_shape_raises_not_implemented(tmp_path):
    path = tmp_path / "rig.json"
    path.write_text(json.dumps({"v": 1, "shape": "box"}), encoding="utf-8")
    with pytest.raises(NotImplementedError):
        load_rig_config(str(path))


def test_load_rig_config_unknown_shape_raises_valueerror(tmp_path):
    path = tmp_path / "rig.json"
    path.write_text(json.dumps({"v": 1, "shape": "tetrahedron"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_rig_config(str(path))


def test_load_rig_config_unsupported_version_raises_valueerror(tmp_path):
    path = tmp_path / "rig.json"
    path.write_text(json.dumps({"v": 2, "shape": "explicit", "marker_corners": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_rig_config(str(path))


# ---------------------------------------------------------------------------
# MarkerRigDetector.detect -- filters to only this rig's own marker ids
# ---------------------------------------------------------------------------


def test_detect_filters_to_rig_markers_only(monkeypatch):
    config = _cube_rig_config()
    detector = MarkerRigDetector(config)
    fake_dets = [
        _det("10", config, "camA", [(0, 0)] * 4),
        _det("99", config, "camA", [(0, 0)] * 4),  # not part of this rig
    ]
    monkeypatch.setattr(
        detector._aruco_by_dict["DICT_4X4_50"], "detect", lambda *a, **kw: fake_dets
    )
    result = detector.detect(np.zeros((10, 10, 3), np.uint8), video_id="camA")
    assert [d.marker_id for d in result] == ["10"]


def test_detect_empty_when_no_rig_markers_present(monkeypatch):
    config = _cube_rig_config()
    detector = MarkerRigDetector(config)
    monkeypatch.setattr(
        detector._aruco_by_dict["DICT_4X4_50"], "detect",
        lambda *a, **kw: [_det("99", config, "camA", [(0, 0)] * 4)],
    )
    assert detector.detect(np.zeros((10, 10, 3), np.uint8)) == []


# ---------------------------------------------------------------------------
# MarkerRigDetector.__init__ -- doesn't scan a dictionary none of this
# config's own markers use (2026-08-15, Harri's "purple marker mixup"
# report): a fully dictionary-specified config (every marker_corners key
# has its own marker_dictionaries entry -- what a "Load Markers…" config
# always is, see page_extrinsics.py's
# _load_rig_config_from_scene_marker_group) must not get a detector for
# this class' unrelated *dictionary* default/fallback ("DICT_4X4_50"),
# which used to be added unconditionally regardless of whether any marker
# needed it. That spurious scan could pick up a marker from a completely
# different rig/config in that dictionary and, because detect()'s
# membership filter checks only the bare numeric id (not dictionary),
# silently misattribute it as one of this config's own markers if the
# ids happened to coincide.
# ---------------------------------------------------------------------------


def _scene_markers_style_config(dictionary: str = "DICT_5X5_50") -> MarkerRigConfig:
    """A config shaped like _load_rig_config_from_scene_marker_group
    builds -- every marker_corners key has an explicit marker_dictionaries
    entry, unlike _cube_rig_config's (which leaves it empty, the older
    load_rig_config JSON-shape case the fallback exists for)."""
    return MarkerRigConfig(
        rig_id="scene markers (room7)",
        marker_corners={"3": marker_local_corners(_SIZE)},
        marker_dictionaries={"3": dictionary},
    )


def test_fully_specified_config_does_not_scan_the_unused_default_dictionary():
    config = _scene_markers_style_config("DICT_5X5_50")
    detector = MarkerRigDetector(config)  # dictionary=... not overridden, defaults to DICT_4X4_50
    assert set(detector._aruco_by_dict) == {"DICT_5X5_50"}


def test_fully_specified_config_ignores_a_same_id_marker_from_another_dictionary(monkeypatch):
    """The exact reported scenario: a physical rig's own DICT_4X4_50
    marker "3" must not be mistaken for a loaded DICT_5X5_50 scene
    marker "3" just because the numeric ids coincide."""
    config = _scene_markers_style_config("DICT_5X5_50")
    detector = MarkerRigDetector(config)
    assert "DICT_4X4_50" not in detector._aruco_by_dict  # nothing to monkeypatch onto it

    monkeypatch.setattr(
        detector._aruco_by_dict["DICT_5X5_50"], "detect", lambda *a, **kw: [],
    )
    result = detector.detect(np.zeros((10, 10, 3), np.uint8), video_id="camA")
    assert result == []


def test_config_with_some_markers_missing_dictionary_still_gets_fallback():
    """The genuine older-load_rig_config-JSON-shape case the fallback
    exists for: a marker with no per-marker dictionary entry must still
    be detectable via the constructor's *dictionary* default."""
    config = MarkerRigConfig(
        rig_id="mixed", marker_corners={"3": marker_local_corners(_SIZE)},
        marker_dictionaries={},  # "3" has no explicit entry -- needs the fallback
    )
    detector = MarkerRigDetector(config)
    assert "DICT_4X4_50" in detector._aruco_by_dict


# ---------------------------------------------------------------------------
# MarkerRigDetector.estimate_rig_pose (diagnostic-only)
# ---------------------------------------------------------------------------


def test_estimate_rig_pose_recovers_known_pose(monkeypatch):
    config = _cube_rig_config()
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    rvec_true = np.array([0.1, 0.2, -0.05])
    t_true = np.array([0.03, -0.02, 1.5])

    all_local = np.concatenate(list(config.marker_corners.values()), axis=0)
    proj, _ = cv2.projectPoints(all_local, rvec_true, t_true, K, np.zeros(4))
    pixels = proj.reshape(-1, 2)

    detections = []
    i = 0
    for marker_id, corners in config.marker_corners.items():
        detections.append(_det(marker_id, config, "camA", pixels[i:i + 4]))
        i += 4

    detector = MarkerRigDetector(config)
    R, t = detector.estimate_rig_pose(detections, K, np.zeros(4))
    R_true, _ = cv2.Rodrigues(rvec_true)
    np.testing.assert_allclose(R, R_true, atol=1e-4)
    np.testing.assert_allclose(t, t_true, atol=1e-3)


def test_estimate_rig_pose_too_few_points_returns_none():
    config = _cube_rig_config()
    detector = MarkerRigDetector(config)
    partial = _det("10", config, "camA", [(1.0, 1.0)] * 2)
    partial.corners = partial.corners[:1]  # only 1 corner total, need >= 4
    K = np.eye(3)
    assert detector.estimate_rig_pose([partial], K, np.zeros(4)) is None


# ---------------------------------------------------------------------------
# anchor_from_marker_rig
# ---------------------------------------------------------------------------


def test_anchor_produces_fixed_control_points_for_every_rig_corner():
    config = _cube_rig_config()
    dets = [_det(mid, config, "camA", [(i, i) for i in range(4)]) for mid in config.marker_corners]
    cps = anchor_from_marker_rig({"camA": dets}, config)
    assert len(cps) == 12  # 3 markers x 4 corners
    for cp in cps:
        assert cp.world_xyz is not None
        assert "camA" in cp.obs


def test_anchor_excludes_markers_not_in_rig():
    config = _cube_rig_config()
    dets = [_det("99", config, "camA", [(0, 0)] * 4)]
    cps = anchor_from_marker_rig({"camA": dets}, config)
    assert cps == []


def test_anchor_merges_same_corner_across_cameras():
    config = _cube_rig_config()
    dets_a = [_det("10", config, "camA", [(1, 1), (2, 1), (2, 2), (1, 2)])]
    dets_b = [_det("10", config, "camB", [(9, 9), (8, 9), (8, 8), (9, 8)])]
    cps = anchor_from_marker_rig({"camA": dets_a, "camB": dets_b}, config)
    assert len(cps) == 4
    for cp in cps:
        assert set(cp.obs) == {"camA", "camB"}


def test_anchor_with_no_detections_returns_empty():
    assert anchor_from_marker_rig({}, _cube_rig_config()) == []


# ---------------------------------------------------------------------------
# Full run_calibration integration -- the real regression test for section
# 9's stated benefit: a genuinely non-planar anchor should recover a
# single unposed camera's pose directly via PnP, with no IPPE tilt-ambiguity
# branch involved at all (mirrors
# test_charuco_detector.test_anchored_board_corners_solve_unposed_cameras,
# but with a non-coplanar rig instead of a flat board).
# ---------------------------------------------------------------------------


def test_anchored_rig_solves_unposed_camera_without_planar_ambiguity():
    config = _cube_rig_config()
    K = np.array([[900.0, 0.0, 400.0], [0.0, 900.0, 300.0], [0.0, 0.0, 1.0]])
    rvec_true = np.array([0.08, -0.15, 0.03])
    R_true, _ = cv2.Rodrigues(rvec_true)
    # Deliberately place the camera's world CENTER at a NEGATIVE Z --
    # exactly the kind of pose init_poses_pnp's coplanar-ambiguity branch
    # (see extrinsics_solver.py) would otherwise second-guess for a flat
    # board ("camera below the reference plane"). A non-planar rig should
    # recover this directly, with no C_z>0 preference logic involved, since
    # the coplanarity check never fires. (t_true is the world->camera
    # translation, NOT the camera's world position -- C = -R^T @ t.)
    C_true = np.array([0.05, 0.02, -1.2])
    t_true = -R_true @ C_true

    all_local = np.concatenate(list(config.marker_corners.values()), axis=0)
    proj, _ = cv2.projectPoints(all_local, rvec_true, t_true, K, np.zeros(4))
    pixels = proj.reshape(-1, 2)

    detections = []
    i = 0
    for marker_id, corners in config.marker_corners.items():
        detections.append(_det(marker_id, config, "camA", pixels[i:i + 4]))
        i += 4

    cps = anchor_from_marker_rig({"camA": detections}, config)
    assert len(cps) == 12

    # Confirm the mechanism: init_poses_pnp's own coplanarity heuristic
    # (extrinsics_solver.py) would not flag these CPs' world points as
    # coplanar -- this is *why* no ambiguity-resolution branch is needed.
    world_pts = np.array([cp.world_xyz for cp in cps])
    centred = world_pts - world_pts.mean(axis=0)
    _, sv, _ = np.linalg.svd(centred)
    cond = sv[0] / sv[-1]
    assert cond < 1e4, "test rig's anchored CPs are unexpectedly coplanar"

    state = CamCalibState(
        video_id="camA", label="camA", K=K, K_orig=K.copy(),
        dist=np.zeros((1, 4)), fisheye=False,
    )
    result = run_calibration([state], control_points=cps, cp_only=True)

    solved = result.cameras["camA"]
    assert solved.R is not None
    np.testing.assert_allclose(solved.R, R_true, atol=1e-4)
    np.testing.assert_allclose(solved.t.flatten(), t_true, atol=1e-3)
    C_solved = -solved.R.T @ solved.t.flatten()
    assert C_solved[2] < 0, "sanity check: this test's whole point is a negative-C_z camera"
