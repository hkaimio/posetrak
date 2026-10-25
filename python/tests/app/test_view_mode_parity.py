# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for _apply_overlay / _compute_target_rect (content_panels.py) --
the two helpers "View-mode parity" (keypoint-editing-design.md, Phase 32)
factored out of _load_frame so every crop-source layer, in both view and
edit mode, frames the same way and keeps overlays visible even when no
image was found for a frame.

Full _load_frame per-camera orchestration (which layer wins, backfill
prioritisation, DB queries) follows the project's usual manual-validation
convention for this widget (see test_phase5.py, test_keypoint_visibility.py)
and is not covered here -- these tests exercise the two extracted methods
directly via the same __new__ + manual-attribute-injection pattern those
files use, without going through _build()'s DB loading or starting any
real background worker.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_widget(**attrs):
    from app.ui.content_panels import PersonCropGridWidget
    from PySide6.QtWidgets import QWidget

    w = PersonCropGridWidget.__new__(PersonCropGridWidget)
    QWidget.__init__(w)
    w._obs_kp = {}
    w._obs_dots = {}
    w._joint_proj = {}
    w._marker_proj = {}
    w._outlier_masks = {}
    w._resolved_markers = {}
    w._cam_intrinsics = {}
    w._bone_pairs = []
    w._hidden_kp_indices = set()
    w._video_dims = {}
    w._edit_mode = False
    w._primary_kp_idx = None
    w._sel_kp_indices = set()
    w._pose_model = None
    for k, v in attrs.items():
        setattr(w, k, v)
    return w


def _make_cell():
    cell = MagicMock()
    cell._canvas.width.return_value = 200
    cell._canvas.height.return_value = 100
    return cell


# ---------------------------------------------------------------------------
# _apply_overlay
# ---------------------------------------------------------------------------

def test_apply_overlay_sets_cell_overlay_even_without_an_image(qapp):
    kp = np.array([[10.0, 20.0, 0.9]], dtype=np.float32)
    w = _make_widget(_obs_kp={"ci1": {5: kp}})
    cell = _make_cell()

    # No image was ever shown on this cell -- _apply_overlay must still set
    # the overlay so an edited keypoint stays visible over a black
    # background instead of being blanked along with the missing image.
    w._apply_overlay(cell, "ci1", "sv1", 5, None, True, True)

    cell.set_overlay.assert_called_once()
    kwargs = cell.set_overlay.call_args.kwargs
    assert kwargs["obs_kp"] is kp
    assert kwargs["show_detected"] is True
    assert kwargs["show_tracked"] is True
    cell.set_hidden.assert_called_once_with(frozenset())


def test_apply_overlay_sets_tracked_skeleton_from_tracking_step(qapp):
    joint_xy = {"hip": np.array([1.0, 2.0])}
    marker_xy = np.array([[3.0, 4.0]], dtype=np.float32)
    w = _make_widget(
        _joint_proj={"ci1": {7: joint_xy}},
        _marker_proj={"ci1": {7: marker_xy}},
    )
    cell = _make_cell()

    w._apply_overlay(cell, "ci1", "sv1", 100, 7, True, True)

    kwargs = cell.set_overlay.call_args.kwargs
    assert kwargs["joint_xy"] is joint_xy
    assert kwargs["marker_xy"] is marker_xy


def test_apply_overlay_no_tracking_step_means_no_tracked_overlay(qapp):
    w = _make_widget(_joint_proj={"ci1": {7: {"hip": [1.0, 2.0]}}})
    cell = _make_cell()

    w._apply_overlay(cell, "ci1", "sv1", 100, None, True, True)

    kwargs = cell.set_overlay.call_args.kwargs
    assert kwargs["joint_xy"] is None
    assert kwargs["marker_xy"] is None


def test_apply_overlay_classifies_dot_candidates_against_resolved_markers(qapp):
    # Three raw candidates: one lands on an inlier slot, one on an outlier
    # slot, one nowhere near any resolved marker this step.
    dot_xy = np.array([[10.0, 10.0], [50.0, 50.0], [200.0, 200.0]], dtype=np.float32)
    resolved = {
        "dot0": (10.0, 10.0, False),   # inlier
        "dot1": (50.0, 51.0, True),    # outlier, within match tolerance
    }
    w = _make_widget(_obs_dots={"ci1": {100: dot_xy}}, _resolved_markers={"ci1": {5: resolved}})
    cell = _make_cell()

    w._apply_overlay(cell, "ci1", "sv1", 100, 5, True, True)

    status = cell.set_overlay.call_args.kwargs["dot_status"]
    assert status[0] == ("inlier", "dot0")
    assert status[1] == ("outlier", "dot1")
    assert status[2][0] == "unassigned"


def test_apply_overlay_passes_this_cameras_intrinsics_to_the_classifier(qapp):
    # A distorted candidate that only matches its resolved (undistorted)
    # counterpart once the camera's own intrinsics are applied -- confirms
    # _apply_overlay actually looks up and forwards self._cam_intrinsics[cam_id],
    # not just that _classify_dot_candidates accepts the parameter in isolation.
    distorted_candidate = _distort((1400.0, 900.0))
    dot_xy = distorted_candidate.reshape(1, 2).astype(np.float32)
    resolved = {"dotA": (1400.0, 900.0, False)}
    intrinsics = {
        "fx": _TEST_K[0, 0], "fy": _TEST_K[1, 1], "cx": _TEST_K[0, 2], "cy": _TEST_K[1, 2],
        "K_orig": _TEST_K, "dist": _TEST_DIST,
    }
    w = _make_widget(
        _obs_dots={"ci1": {100: dot_xy}},
        _resolved_markers={"ci1": {5: resolved}},
        _cam_intrinsics={"ci1": intrinsics},
    )
    cell = _make_cell()

    w._apply_overlay(cell, "ci1", "sv1", 100, 5, True, True)

    status = cell.set_overlay.call_args.kwargs["dot_status"]
    assert status[0] == ("inlier", "dotA")


def test_apply_overlay_no_resolved_markers_means_no_dot_status(qapp):
    # No tracking-run obs data for this step (e.g. ObjectCropGridWidget's own
    # use, or a step with no tracking_obs_results row) -- dot_status stays
    # None so _ImageCanvas falls back to its old, uniform-colour behaviour.
    dot_xy = np.array([[10.0, 10.0]], dtype=np.float32)
    w = _make_widget(_obs_dots={"ci1": {100: dot_xy}})
    cell = _make_cell()

    w._apply_overlay(cell, "ci1", "sv1", 100, 5, True, True)

    assert cell.set_overlay.call_args.kwargs["dot_status"] is None


def test_apply_overlay_empty_resolved_markers_means_everything_unassigned(qapp):
    # A tracking_obs_results row existed for this step, but every marker was
    # NaN for this camera (assignment failed for every slot) -- a known
    # answer ("nothing was assigned"), not "no data". Must colour every
    # candidate unassigned (blue), not fall back to the no-data magenta
    # default -- see _TrackingRunLoader's own comment on why it records an
    # empty dict here instead of omitting the step entirely.
    dot_xy = np.array([[10.0, 10.0], [50.0, 50.0]], dtype=np.float32)
    w = _make_widget(_obs_dots={"ci1": {100: dot_xy}}, _resolved_markers={"ci1": {5: {}}})
    cell = _make_cell()

    w._apply_overlay(cell, "ci1", "sv1", 100, 5, True, True)

    status = cell.set_overlay.call_args.kwargs["dot_status"]
    assert status is not None
    assert status[0] == ("unassigned", None)
    assert status[1] == ("unassigned", None)


# ---------------------------------------------------------------------------
# _classify_dot_candidates
# ---------------------------------------------------------------------------

def test_classify_dot_candidates_matches_nearest_within_tolerance(qapp):
    from app.ui.content_panels import _classify_dot_candidates

    dot_xy = np.array([[10.0, 10.0], [100.0, 100.0]], dtype=np.float32)
    resolved = {"dotA": (10.5, 10.2, False)}

    result = _classify_dot_candidates(dot_xy, resolved)

    assert result[0] == ("inlier", "dotA")
    assert result[1] == ("unassigned", None)


def test_classify_dot_candidates_beyond_tolerance_is_unassigned(qapp):
    from app.ui.content_panels import _classify_dot_candidates

    dot_xy = np.array([[10.0, 10.0]], dtype=np.float32)
    resolved = {"dotA": (25.0, 25.0, False)}  # well outside the match distance

    result = _classify_dot_candidates(dot_xy, resolved)

    assert result[0] == ("unassigned", None)


def test_classify_dot_candidates_nan_candidate_is_unassigned(qapp):
    from app.ui.content_panels import _classify_dot_candidates

    dot_xy = np.array([[float("nan"), float("nan")]], dtype=np.float32)
    resolved = {"dotA": (10.0, 10.0, False)}

    result = _classify_dot_candidates(dot_xy, resolved)

    assert result[0] == ("unassigned", None)


# ---------------------------------------------------------------------------
# Distortion-aware matching: dot_xy is distorted pixels (pose_observations'
# raw detector output, matching the displayed image), resolved_markers is
# undistorted (obs_blob's own convention) -- confirmed on real data that the
# gap between them is tens of px, not noise. _classify_dot_candidates must
# undistort dot_xy before comparing when intrinsics are available.
# ---------------------------------------------------------------------------

_TEST_K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
_TEST_DIST = np.array([-0.35, 0.15, 0.0, 0.0, 0.0])


def _distort(undistorted_xy):
    """The inverse of what _undistort_for_matching does -- used only to build
    a realistic *distorted* dot_xy fixture from a known undistorted position,
    mirroring how a real detector's raw pixel would relate to obs_blob's."""
    import cv2

    x = (undistorted_xy[0] - _TEST_K[0, 2]) / _TEST_K[0, 0]
    y = (undistorted_xy[1] - _TEST_K[1, 2]) / _TEST_K[1, 1]
    obj = np.array([[[x, y, 1.0]]], dtype=np.float64)
    out, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), _TEST_K, _TEST_DIST)
    return out.reshape(2)


def test_classify_dot_candidates_without_intrinsics_misses_a_real_match(qapp):
    # The bug this guards against: without undistorting first, a real match
    # tens of pixels away (this fixture's own distortion offset, verified >
    # 50px) reads as "unassigned" even though it's the correct candidate.
    from app.ui.content_panels import _classify_dot_candidates

    undistorted_true_pos = (1400.0, 900.0)
    distorted_candidate = _distort(undistorted_true_pos)
    assert np.linalg.norm(distorted_candidate - np.array(undistorted_true_pos)) > 50.0

    dot_xy = distorted_candidate.reshape(1, 2).astype(np.float32)
    resolved = {"dotA": (undistorted_true_pos[0], undistorted_true_pos[1], False)}

    result = _classify_dot_candidates(dot_xy, resolved)  # no intrinsics given

    assert result[0] == ("unassigned", None)


def test_classify_dot_candidates_with_intrinsics_finds_the_match(qapp):
    from app.ui.content_panels import _classify_dot_candidates

    undistorted_true_pos = (1400.0, 900.0)
    distorted_candidate = _distort(undistorted_true_pos)
    dot_xy = distorted_candidate.reshape(1, 2).astype(np.float32)
    resolved = {"dotA": (undistorted_true_pos[0], undistorted_true_pos[1], False)}
    intrinsics = {
        "fx": _TEST_K[0, 0], "fy": _TEST_K[1, 1], "cx": _TEST_K[0, 2], "cy": _TEST_K[1, 2],
        "K_orig": _TEST_K, "dist": _TEST_DIST,
    }

    result = _classify_dot_candidates(dot_xy, resolved, intrinsics)

    assert result[0] == ("inlier", "dotA")


def test_classify_dot_candidates_missing_dist_falls_back_gracefully(qapp):
    # An intrinsics entry present but without a distortion model (K_orig/dist
    # both None, e.g. an uncalibrated camera) must not raise -- just behaves
    # like no intrinsics were given at all.
    from app.ui.content_panels import _classify_dot_candidates

    dot_xy = np.array([[10.0, 10.0]], dtype=np.float32)
    resolved = {"dotA": (10.2, 10.1, False)}
    intrinsics = {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "K_orig": None, "dist": None}

    result = _classify_dot_candidates(dot_xy, resolved, intrinsics)

    assert result[0] == ("inlier", "dotA")


# ---------------------------------------------------------------------------
# _compute_target_rect
# ---------------------------------------------------------------------------

def test_compute_target_rect_matches_own_rect_when_nothing_else_present(qapp):
    w = _make_widget()
    cell = _make_cell()  # 200x100 canvas, aspect 2:1

    # own_rect alone is already 2:1 -- aspect-fit should be a no-op, and with
    # no keypoints/tracked overlay to widen for, only the fixed margin grows it.
    own_rect = (0.0, 0.0, 200.0, 100.0)
    desired = w._compute_target_rect(cell, "ci1", "sv1", 5, None, own_rect)

    from app.ui.content_panels import _DISPLAY_MARGIN_FRAC
    mx, my = 200.0 * _DISPLAY_MARGIN_FRAC, 100.0 * _DISPLAY_MARGIN_FRAC
    assert desired == pytest.approx((-mx, -my, 200.0 + mx, 100.0 + my))


def test_compute_target_rect_widens_to_cover_keypoints_outside_own_rect(qapp):
    # A keypoint sitting well outside own_rect (e.g. an edit placed far from
    # a stale/wrong detection) must widen the result, not be clipped to it.
    kp = np.array([[500.0, 500.0, 0.9]], dtype=np.float32)
    w = _make_widget(_obs_kp={"ci1": {5: kp}})
    cell = _make_cell()

    own_rect = (0.0, 0.0, 50.0, 50.0)
    desired = w._compute_target_rect(cell, "ci1", "sv1", 5, None, own_rect)

    assert desired[2] >= 500.0
    assert desired[3] >= 500.0


def test_compute_target_rect_ignores_implausible_keypoint(qapp):
    # A garbage/diverged coordinate must not blow the rect up -- see the
    # _sane_bbox safety-cap regression test in test_wide_crop_cache.py.
    kp = np.array([[5_000_000.0, 5_000_000.0, 0.9]], dtype=np.float32)
    w = _make_widget(_obs_kp={"ci1": {5: kp}})
    cell = _make_cell()

    own_rect = (0.0, 0.0, 50.0, 50.0)
    desired = w._compute_target_rect(cell, "ci1", "sv1", 5, None, own_rect)

    assert desired[2] < 1000.0
    assert desired[3] < 1000.0


# ---------------------------------------------------------------------------
# _display_crop_result -- debug-label wiring ("Debug overlay", Phase 33)
# ---------------------------------------------------------------------------

def _make_result():
    from tests.app.conftest import _make_jpeg_stub
    # (jpeg, wpx, hpx, src_x, src_y, src_w, src_h) -- 1x1 stub, scale 1:1.
    return (_make_jpeg_stub(), 1, 1, 0.0, 0.0, 1.0, 1.0)


def test_display_crop_result_sets_layer_label_when_debug_enabled(qapp):
    w = _make_widget()
    cell = _make_cell()

    w._display_crop_result(
        cell, "ci1", "sv1", 5, None, True, True, _make_result(),
        layer_label="wide-cache", show_debug=True,
    )

    cell.set_debug_label.assert_called_with("wide-cache")


def test_display_crop_result_appends_black_fill_suffix(qapp):
    w = _make_widget()
    cell = _make_cell()

    # target_rect (100x100) far exceeds the 1x1 decoded stub -- black-fill
    # must engage, and the debug label must say so.
    w._display_crop_result(
        cell, "ci1", "sv1", 5, None, True, True, _make_result(),
        target_rect=(0.0, 0.0, 100.0, 100.0),
        layer_label="wide-cache", show_debug=True,
    )

    cell.set_debug_label.assert_called_with("wide-cache +black-fill")


def test_display_crop_result_no_label_when_debug_disabled(qapp):
    w = _make_widget()
    cell = _make_cell()

    w._display_crop_result(
        cell, "ci1", "sv1", 5, None, True, True, _make_result(),
        layer_label="wide-cache", show_debug=False,
    )

    cell.set_debug_label.assert_called_with(None)
