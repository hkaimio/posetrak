# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for app.setup.camera_cell (CameraCell widget)."""

from __future__ import annotations

import numpy as np
import pytest

from app.setup.overlay import ROIDrawOverlay, SyncAnchorOverlay


@pytest.fixture()
def cell(qapp):
    from app.setup.camera_cell import CameraCell
    w = CameraCell(label="cam1")
    w.resize(320, 180)
    return w


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------


def test_display_to_frame_center(cell) -> None:
    """Centre of the widget should map to centre of the frame."""
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    cell.set_frame(frame)
    cell.resize(320, 180)

    fx, fy = cell.display_to_frame(160, 90)
    assert fx == 80
    assert fy == 45


def test_display_to_frame_top_left(cell) -> None:
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    cell.set_frame(frame)
    cell.resize(320, 180)

    fx, fy = cell.display_to_frame(0, 0)
    assert fx == 0 and fy == 0


def test_display_to_frame_no_frame_returns_input(cell) -> None:
    """With no frame loaded, coordinates are returned unchanged."""
    fx, fy = cell.display_to_frame(50, 75)
    assert fx == 50 and fy == 75


def test_display_to_frame_scales_with_widget_size(cell) -> None:
    """Coordinate mapping must update when the widget is resized."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    cell.set_frame(frame)

    cell.resize(400, 200)
    fx, _ = cell.display_to_frame(200, 0)
    assert fx == 100   # 200/400 * 200 = 100

    cell.resize(200, 100)
    fx, _ = cell.display_to_frame(200, 0)
    assert fx == 200   # 200/200 * 200 = 200


# ---------------------------------------------------------------------------
# Overlay dispatch
# ---------------------------------------------------------------------------


def test_overlays_receive_mouse_press_in_reverse_order(cell) -> None:
    """mouse_press must be dispatched to overlays in reverse order."""
    received: list[int] = []

    class _O:
        def __init__(self, idx):
            self._idx = idx
        def paint(self, *a): pass
        def mouse_press(self, x, y): received.append(self._idx)
        def mouse_move(self, x, y): pass
        def mouse_release(self, x, y): pass

    o0, o1, o2 = _O(0), _O(1), _O(2)
    cell.set_overlays([o0, o1, o2])

    # Simulate via direct call rather than Qt event
    for ov in reversed(cell._overlays):
        ov.mouse_press(0, 0)

    assert received == [2, 1, 0]


def test_set_overlays_replaces_previous(cell) -> None:
    a = SyncAnchorOverlay()
    b = ROIDrawOverlay()
    cell.set_overlays([a])
    assert len(cell._overlays) == 1
    cell.set_overlays([a, b])
    assert len(cell._overlays) == 2
    cell.set_overlays([])
    assert len(cell._overlays) == 0


# ---------------------------------------------------------------------------
# set_frame / clear_frame
# ---------------------------------------------------------------------------


def test_set_frame_stores_array(cell) -> None:
    frame = np.full((90, 160, 3), 128, dtype=np.uint8)
    cell.set_frame(frame)
    assert cell._frame is frame


def test_clear_frame_removes_frame(cell) -> None:
    cell.set_frame(np.zeros((90, 160, 3), dtype=np.uint8))
    cell.clear_frame()
    assert cell._frame is None
