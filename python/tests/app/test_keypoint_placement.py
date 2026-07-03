"""Tests for the keypoint-placement toolbar (Phases 28-29): picking a
keypoint from _KeypointPickerPanel, arming placement mode on the camera
canvases, placing on click, and Esc-cancels-placement-first semantics.

See "Keypoint-placement toolbar" in keypoint-editing-design.md.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from PySide6.QtCore import Qt

from app.ui.content_panels import PersonCropGridWidget, _KeypointPickerPanel


def _make_widget(qapp):
    """Minimal PersonCropGridWidget with mock cells for placement-toolbar tests."""
    from PySide6.QtWidgets import QWidget

    w = PersonCropGridWidget.__new__(PersonCropGridWidget)
    QWidget.__init__(w)
    w._edit_mode = True
    w._sel_kp_indices = {1, 2}
    w._primary_kp_idx = 1
    w._sel_cam_idx = 0
    w._pending_place_kp_idx = None
    w._kp_picker = MagicMock()
    w._range_start_v = None
    w._range_end_v = None
    w._slider = None
    w._timeline = None
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


# ---------------------------------------------------------------------------
# _on_kp_picked / _cancel_placement
# ---------------------------------------------------------------------------

def test_on_kp_picked_arms_all_cells_and_highlights_picker(qapp):
    w = _make_widget(qapp)
    w._on_kp_picked(5)
    assert w._pending_place_kp_idx == 5
    for cell in w._cells:
        cell.set_placement_active.assert_called_once_with(True)
    w._kp_picker.set_active.assert_called_once_with(5)


def test_on_kp_picked_retargets_without_needing_cancel_first(qapp):
    w = _make_widget(qapp)
    w._on_kp_picked(3)
    w._on_kp_picked(7)
    assert w._pending_place_kp_idx == 7
    # Both picks armed the cells; no requirement that cancel ran in between.
    assert w._cells[0].set_placement_active.call_count == 2


def test_cancel_placement_disarms_cells_and_clears_picker_highlight(qapp):
    w = _make_widget(qapp)
    w._on_kp_picked(5)
    for cell in w._cells:
        cell.set_placement_active.reset_mock()
    w._kp_picker.reset_mock()

    w._cancel_placement()

    assert w._pending_place_kp_idx is None
    for cell in w._cells:
        cell.set_placement_active.assert_called_once_with(False)
    w._kp_picker.set_active.assert_called_once_with(None)


def test_cancel_placement_is_a_noop_when_nothing_pending(qapp):
    w = _make_widget(qapp)
    w._cancel_placement()
    for cell in w._cells:
        cell.set_placement_active.assert_not_called()
    w._kp_picker.set_active.assert_not_called()


# ---------------------------------------------------------------------------
# _on_placement_clicked
# ---------------------------------------------------------------------------

def test_on_placement_clicked_places_pending_keypoint_at_full_frame_coords(qapp):
    w = _make_widget(qapp)
    w._on_kp_picked(9)
    w._cells[0]._canvas._display_to_full.return_value = (123.0, 456.0)
    w._on_kp_moved = MagicMock()

    w._on_placement_clicked(0, 10.0, 20.0)

    w._cells[0]._canvas._display_to_full.assert_called_once_with(10.0, 20.0)
    w._on_kp_moved.assert_called_once_with(0, 9, 123.0, 456.0)


def test_on_placement_clicked_is_one_shot_and_disarms_afterward(qapp):
    # A single placement should return the editor to normal mode, matching
    # a normal drag-to-move -- not stay armed for repeat placements.
    w = _make_widget(qapp)
    w._on_kp_picked(9)
    w._cells[0]._canvas._display_to_full.return_value = (1.0, 2.0)
    w._on_kp_moved = MagicMock()

    w._on_placement_clicked(0, 10.0, 20.0)

    assert w._pending_place_kp_idx is None
    for cell in w._cells:
        cell.set_placement_active.assert_called_with(False)
    w._kp_picker.set_active.assert_called_with(None)

    # A second click with nothing armed should not place anything again.
    w._on_kp_moved.reset_mock()
    w._on_placement_clicked(0, 30.0, 40.0)
    w._on_kp_moved.assert_not_called()


def test_on_placement_clicked_does_nothing_when_not_armed(qapp):
    w = _make_widget(qapp)
    w._on_kp_moved = MagicMock()
    w._on_placement_clicked(0, 10.0, 20.0)
    w._on_kp_moved.assert_not_called()


# ---------------------------------------------------------------------------
# Esc cancels placement first, existing deselect semantics apply only after
# ---------------------------------------------------------------------------

def test_escape_cancels_placement_before_clearing_selection(qapp):
    w = _make_widget(qapp)
    w._on_kp_picked(4)

    handled = w._handle_key(_FakeKeyEvent(Qt.Key.Key_Escape))

    assert handled is True
    assert w._pending_place_kp_idx is None
    # First Esc only cancels placement -- selection from before is untouched.
    assert w._sel_kp_indices == {1, 2}
    assert w._primary_kp_idx == 1


def test_escape_falls_through_to_deselect_once_nothing_pending(qapp):
    w = _make_widget(qapp)
    assert w._pending_place_kp_idx is None

    handled = w._handle_key(_FakeKeyEvent(Qt.Key.Key_Escape))

    assert handled is True
    assert w._sel_kp_indices == set()
    assert w._primary_kp_idx is None
