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
    w._joint_proj = {}
    w._marker_proj = {}
    w._outlier_masks = {}
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
