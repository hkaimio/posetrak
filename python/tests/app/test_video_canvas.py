# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for VideoCanvas's zoom/pan coordinate mapping and tool state
(segmentation-ui-improvements design doc, Issue 5) -- the panel had zero
prior test coverage; these exercise the new zoom math (the part most
likely to have an off-by-one or sign error) alongside the pre-existing
canvas_to_image behavior.

Widget size is always chosen well above VideoCanvas's own
setMinimumSize(320, 240) -- resize() below that gets silently clamped,
which the numbers here would otherwise get wrong.
"""
from __future__ import annotations

import numpy as np

from app.pose.video_canvas import ZOOM_MAX, ZOOM_MIN, VideoCanvas


def _canvas(w=800, h=400, frame_w=400, frame_h=200):
    c = VideoCanvas()
    c.resize(w, h)
    assert (c.width(), c.height()) == (w, h)  # sanity: above the min size clamp
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    c.display(frame)
    return c


def test_canvas_to_image_at_default_zoom_matches_fit_scale(qapp):
    # 400x200 frame into an 800x400 widget -- exact 2:1 fit, no letterbox.
    c = _canvas(w=800, h=400, frame_w=400, frame_h=200)
    assert c.canvas_to_image(0, 0) == (0, 0)
    assert c.canvas_to_image(400, 200) == (200, 100)
    assert c.canvas_to_image(799, 399) == (399, 199)


def test_canvas_to_image_outside_image_area_is_none(qapp):
    c = _canvas(w=800, h=400, frame_w=400, frame_h=200)
    assert c.canvas_to_image(-1, 0) is None
    assert c.canvas_to_image(0, 400) is None


def test_letterboxing_on_mismatched_aspect_ratio(qapp):
    # Square frame in a wide widget -- letterboxed left/right.
    c = _canvas(w=800, h=400, frame_w=400, frame_h=400)
    # fit_scale = min(800/400, 400/400) = 1.0 -> displayed 400x400,
    # centered horizontally: offset_x = 200.
    assert c.canvas_to_image(200, 0) == (0, 0)
    assert c.canvas_to_image(199, 0) is None  # in the left letterbox bar


def test_zoom_in_at_recenters_and_increases_scale(qapp):
    c = _canvas(w=800, h=400, frame_w=400, frame_h=200)
    c.zoom_in_at(200, 100)  # center of the frame
    assert c._zoom > 1.0
    # Zoomed in on the center -- that point should still map back to
    # roughly the canvas center.
    ix, iy = c.canvas_to_image(400, 200)
    assert abs(ix - 200) <= 2
    assert abs(iy - 100) <= 2


def test_zoom_in_then_out_returns_to_original_zoom(qapp):
    c = _canvas(w=800, h=400, frame_w=400, frame_h=200)
    c.zoom_in_at(200, 100)
    c.zoom_out_at(200, 100)
    assert abs(c._zoom - 1.0) < 1e-9


def test_zoom_clamped_to_max(qapp):
    c = _canvas()
    for _ in range(20):
        c.zoom_in_at(200, 100)
    assert c._zoom == ZOOM_MAX


def test_zoom_out_clamped_to_min(qapp):
    c = _canvas()
    c.zoom_out_at(200, 100)  # already at 1.0 -- can't go lower
    assert c._zoom == ZOOM_MIN


def test_reset_zoom_restores_fit_view(qapp):
    c = _canvas(w=800, h=400, frame_w=400, frame_h=200)
    c.zoom_in_at(50, 50)
    c.reset_zoom()
    assert c._zoom == 1.0
    assert c.canvas_to_image(0, 0) == (0, 0)
    assert c.canvas_to_image(799, 399) == (399, 199)


def test_zoomed_crop_stays_within_frame_bounds_near_edge(qapp):
    """Panning centered near the frame edge must clamp the crop, not run
    off the actual image data."""
    c = _canvas(w=800, h=400, frame_w=400, frame_h=200)
    c.zoom_in_at(0, 0)  # top-left corner -- crop would run negative unclamped
    ix, iy = c.canvas_to_image(0, 0)
    assert ix >= 0 and iy >= 0


def test_set_tool_and_brush_radius_do_not_crash_without_a_frame(qapp):
    c = VideoCanvas()
    c.set_tool("paint")
    c.set_brush_radius(15)
    assert c._active_tool == "paint"
    assert c._brush_radius == 15


def test_left_dragged_emitted_only_while_button_held(qapp):
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import Qt as _Qt

    c = _canvas(w=800, h=400, frame_w=400, frame_h=200)
    received = []
    c.left_dragged.connect(lambda x, y: received.append((x, y)))

    # Move with no button held -- no drag signal.
    ev = QMouseEvent(
        QEvent.Type.MouseMove, QPointF(400, 200), QPointF(400, 200),
        _Qt.MouseButton.NoButton, _Qt.MouseButton.NoButton, _Qt.KeyboardModifier.NoModifier,
    )
    c.mouseMoveEvent(ev)
    assert received == []

    # Move with left button held -- drag signal fires with image coords.
    ev2 = QMouseEvent(
        QEvent.Type.MouseMove, QPointF(400, 200), QPointF(400, 200),
        _Qt.MouseButton.LeftButton, _Qt.MouseButton.LeftButton, _Qt.KeyboardModifier.NoModifier,
    )
    c.mouseMoveEvent(ev2)
    assert received == [(200, 100)]
