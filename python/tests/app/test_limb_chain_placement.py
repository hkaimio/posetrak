# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the limb chain-placement tool: pick a limb, then click through
its keypoints in order (shoulder->elbow->wrist->..., hip->knee->ankle->...),
one click per keypoint, without re-picking from the toolbar each time.

See the "Set limb…" button next to "Edit keypoints" in content_panels.py.
As with test_keypoint_placement.py, this never drives QMenu.exec() -- the
menu's action lambdas just call _start_chain_placement(limb) directly, so
tests call that instead of exercising the popup.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from PySide6.QtCore import Qt

from app.pose.kp_models import COCO17
from app.ui.content_panels import PersonCropGridWidget


def _make_widget(qapp):
    from PySide6.QtWidgets import QWidget

    w = PersonCropGridWidget.__new__(PersonCropGridWidget)
    QWidget.__init__(w)
    w._pose_model = COCO17
    w._edit_mode = True
    w._sel_kp_indices = set()
    w._primary_kp_idx = None
    w._sel_cam_idx = 0
    w._pending_place_kp_idx = None
    w._chain_limb = None
    w._chain_indices = []
    w._chain_pos = 0
    w._kp_picker = MagicMock()
    w._range_start_v = None
    w._range_end_v = None
    w._slider = None
    w._timeline = None
    w._time_label = None
    w._t_start = 0.0
    w._current_t = 0.0
    w._cells = [MagicMock(), MagicMock()]
    return w


class _FakeKeyEvent:
    def __init__(self, key, modifiers=Qt.KeyboardModifier.NoModifier):
        self._key = key
        self._modifiers = modifiers

    def key(self):
        return self._key

    def modifiers(self):
        return self._modifiers


_LEFT_ARM = COCO17.limb_chain_indices("Left arm")  # shoulder, elbow, wrist


# ---------------------------------------------------------------------------
# _start_chain_placement / _arm_chain_step
# ---------------------------------------------------------------------------

def test_start_chain_placement_arms_first_keypoint(qapp):
    w = _make_widget(qapp)
    w._start_chain_placement("Left arm")

    assert w._chain_limb == "Left arm"
    assert w._chain_indices == _LEFT_ARM
    assert w._chain_pos == 0
    assert w._pending_place_kp_idx == _LEFT_ARM[0]
    for cell in w._cells:
        cell.set_placement_active.assert_called_once_with(True)
        cell.set_placement_label.assert_called_once_with("Left arm: left_shoulder (1/3)")
    w._kp_picker.set_active.assert_called_once_with(_LEFT_ARM[0])


def test_start_chain_placement_moves_keyboard_focus_to_first_cell(qapp):
    # Picking a limb from the "Set limb…" menu leaves keyboard focus on the
    # button unless we move it -- and a focused QPushButton treats Space as
    # "click me" (reopening the menu) rather than "skip this keypoint", since
    # key events only reach _handle_key when a camera canvas holds focus.
    w = _make_widget(qapp)
    w._start_chain_placement("Left arm")
    w._cells[0]._canvas.setFocus.assert_called_once()


def test_start_chain_placement_unknown_limb_is_a_noop(qapp):
    w = _make_widget(qapp)
    w._start_chain_placement("Tail")
    assert w._chain_limb is None
    for cell in w._cells:
        cell.set_placement_active.assert_not_called()


# ---------------------------------------------------------------------------
# _on_placement_clicked during a chain: writes, advances, re-arms
# ---------------------------------------------------------------------------

def test_placement_click_advances_to_next_chain_keypoint(qapp):
    w = _make_widget(qapp)
    w._start_chain_placement("Left arm")
    w._on_kp_moved = MagicMock()
    w._cells[0]._canvas._display_to_full.return_value = (10.0, 20.0)

    w._on_placement_clicked(0, 1.0, 2.0)

    w._on_kp_moved.assert_called_once_with(0, _LEFT_ARM[0], 10.0, 20.0)
    assert w._chain_pos == 1
    assert w._pending_place_kp_idx == _LEFT_ARM[1]
    # Still armed -- chain placement does not disarm mid-chain.
    for cell in w._cells:
        cell.set_placement_label.assert_called_with("Left arm: left_elbow (2/3)")
        cell.set_placement_active.assert_called_with(True)


def test_placement_click_on_last_keypoint_wraps_back_to_first(qapp):
    # The chain stays armed indefinitely (until Esc, a new limb, or a single
    # keypoint pick) so the same limb can be placed again on another frame
    # without re-picking it from the "Set limb…" menu each time.
    w = _make_widget(qapp)
    w._start_chain_placement("Left arm")
    w._on_kp_moved = MagicMock()
    w._cells[0]._canvas._display_to_full.return_value = (0.0, 0.0)

    for _ in range(len(_LEFT_ARM)):
        w._on_placement_clicked(0, 0.0, 0.0)

    assert w._on_kp_moved.call_count == len(_LEFT_ARM)
    assert w._chain_limb == "Left arm"
    assert w._chain_pos == 0
    assert w._pending_place_kp_idx == _LEFT_ARM[0]
    for cell in w._cells:
        cell.set_placement_active.assert_called_with(True)
        cell.set_placement_label.assert_called_with("Left arm: left_shoulder (1/3)")


# ---------------------------------------------------------------------------
# Space skips the current keypoint without writing anything
# ---------------------------------------------------------------------------

def test_space_skips_current_chain_keypoint_without_writing(qapp):
    w = _make_widget(qapp)
    w._start_chain_placement("Left arm")
    w._on_kp_moved = MagicMock()

    handled = w._handle_key(_FakeKeyEvent(Qt.Key.Key_Space))

    assert handled is True
    w._on_kp_moved.assert_not_called()
    assert w._chain_pos == 1
    assert w._pending_place_kp_idx == _LEFT_ARM[1]


def test_space_on_last_chain_keypoint_wraps_back_to_first(qapp):
    w = _make_widget(qapp)
    w._start_chain_placement("Left arm")

    for _ in range(len(_LEFT_ARM)):
        w._handle_key(_FakeKeyEvent(Qt.Key.Key_Space))

    assert w._chain_limb == "Left arm"
    assert w._chain_pos == 0
    assert w._pending_place_kp_idx == _LEFT_ARM[0]


def test_space_without_active_chain_falls_through_to_toggle_outlier(qapp):
    w = _make_widget(qapp)
    w._sel_kp_indices = {1}
    w._toggle_outlier = MagicMock()

    handled = w._handle_key(_FakeKeyEvent(Qt.Key.Key_Space))

    assert handled is True
    w._toggle_outlier.assert_called_once()


# ---------------------------------------------------------------------------
# Esc ends the chain (reuses the existing placement-cancel-first Esc path)
# ---------------------------------------------------------------------------

def test_escape_ends_an_in_progress_chain(qapp):
    w = _make_widget(qapp)
    w._start_chain_placement("Left arm")

    handled = w._handle_key(_FakeKeyEvent(Qt.Key.Key_Escape))

    assert handled is True
    assert w._chain_limb is None
    assert w._pending_place_kp_idx is None


# ---------------------------------------------------------------------------
# Picking a single keypoint from the list ends any in-progress chain
# ---------------------------------------------------------------------------

def test_picking_single_keypoint_ends_in_progress_chain(qapp):
    w = _make_widget(qapp)
    w._start_chain_placement("Left arm")

    w._on_kp_picked(9)

    assert w._chain_limb is None
    assert w._chain_indices == []
    assert w._pending_place_kp_idx == 9


# ---------------------------------------------------------------------------
# Frame change restarts the chain at its first keypoint
# ---------------------------------------------------------------------------

def test_frame_change_restarts_chain_at_first_keypoint(qapp):
    w = _make_widget(qapp)
    w._start_chain_placement("Left arm")
    w._on_kp_moved = MagicMock()
    w._cells[0]._canvas._display_to_full.return_value = (0.0, 0.0)
    w._on_placement_clicked(0, 0.0, 0.0)  # advance to elbow (pos 1)
    assert w._chain_pos == 1
    w.time_changed = MagicMock()
    w._load_frame = MagicMock()

    w._on_slider(0)

    assert w._chain_pos == 0
    assert w._pending_place_kp_idx == _LEFT_ARM[0]
    for cell in w._cells:
        cell.set_placement_label.assert_called_with("Left arm: left_shoulder (1/3)")


def test_frame_change_without_active_chain_does_not_touch_placement(qapp):
    w = _make_widget(qapp)
    w.time_changed = MagicMock()
    w._load_frame = MagicMock()

    w._on_slider(0)

    for cell in w._cells:
        cell.set_placement_label.assert_not_called()
        cell.set_placement_active.assert_not_called()
