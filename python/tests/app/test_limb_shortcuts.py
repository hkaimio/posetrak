"""Tests for the number-key limb shortcuts (see PersonCropGridWidget._LIMB_SHORTCUT_KEYS
and _handle_limb_shortcut):

- plain key: toggle show/hide of the limb's keypoints (reuses
  _on_timeline_visibility_toggled's "all hidden -> show, else hide" rule).
- Shift+key: isolate the limb (hide every other keypoint).
- Ctrl+key: start "Set limb…" chain placement for the limb, if one is
  defined (kp_models.py's limb_chains) -- hands have none yet, so Ctrl+3/
  Ctrl+5 just report that via status_message instead of doing nothing silently.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt

from app.pose.kp_models import COCO17, COCO133
from app.ui.content_panels import PersonCropGridWidget


class _FakeKeyEvent:
    def __init__(self, key, modifiers=Qt.KeyboardModifier.NoModifier):
        self._key = key
        self._modifiers = modifiers

    def key(self):
        return self._key

    def modifiers(self):
        return self._modifiers


def _make_widget(pose_model=COCO133):
    from PySide6.QtWidgets import QWidget

    w = PersonCropGridWidget.__new__(PersonCropGridWidget)
    QWidget.__init__(w)
    w._pose_model = pose_model
    w._edit_mode = True
    w._sel_kp_indices = set()
    w._primary_kp_idx = None
    w._sel_cam_idx = 0
    w._hidden_kp_indices = set()
    w._pending_place_kp_idx = None
    w._chain_limb = None
    w._chain_indices = []
    w._chain_pos = 0
    w._kp_picker = MagicMock()
    w._range_start_v = None
    w._slider = None
    w._timeline = MagicMock()
    w._t_start = 0.0
    w._current_t = 0.0
    w._load_frame = MagicMock()
    w._cells = [MagicMock()]
    return w


def _messages(w) -> list[str]:
    received: list[str] = []
    w.status_message.connect(received.append)
    return received


# ---------------------------------------------------------------------------
# Key -> limb mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "key, limb",
    [
        (Qt.Key.Key_1, "Face"),
        (Qt.Key.Key_2, "Left arm"),
        (Qt.Key.Key_3, "Left hand"),
        (Qt.Key.Key_4, "Right arm"),
        (Qt.Key.Key_5, "Right hand"),
        (Qt.Key.Key_6, "Left leg"),
        (Qt.Key.Key_7, "Right leg"),
    ],
)
def test_plain_key_toggles_limb_visibility(qapp, key, limb):
    w = _make_widget()
    indices = set(COCO133.group_indices(limb))

    handled = w._handle_key(_FakeKeyEvent(key))

    assert handled is True
    assert w._hidden_kp_indices == indices
    w._load_frame.assert_called()


def test_plain_key_pressed_twice_shows_again(qapp):
    w = _make_widget()
    w._handle_key(_FakeKeyEvent(Qt.Key.Key_2))  # hide "Left arm"
    w._handle_key(_FakeKeyEvent(Qt.Key.Key_2))  # press again -> show
    assert w._hidden_kp_indices == set()


# ---------------------------------------------------------------------------
# Shift+key: isolate (hide everything else)
# ---------------------------------------------------------------------------

def test_shift_key_isolates_limb(qapp):
    w = _make_widget()
    left_arm = set(COCO133.group_indices("Left arm"))

    handled = w._handle_key(_FakeKeyEvent(Qt.Key.Key_2, Qt.KeyboardModifier.ShiftModifier))

    assert handled is True
    assert w._hidden_kp_indices == set(COCO133.all_indices) - left_arm
    w._timeline.set_hidden.assert_called_once_with(frozenset(w._hidden_kp_indices))


def test_shift_key_isolate_purges_selection_outside_the_limb(qapp):
    w = _make_widget()
    w._sel_kp_indices = {0, *COCO133.group_indices("Left arm")}
    w._primary_kp_idx = 0  # nose -- outside "Left arm", must be dropped

    w._handle_key(_FakeKeyEvent(Qt.Key.Key_2, Qt.KeyboardModifier.ShiftModifier))

    assert 0 not in w._sel_kp_indices
    assert w._primary_kp_idx != 0


def test_shift_key_emits_isolate_status_message(qapp):
    w = _make_widget()
    received = _messages(w)
    w._handle_key(_FakeKeyEvent(Qt.Key.Key_2, Qt.KeyboardModifier.ShiftModifier))
    assert received == ["Showing only Left arm"]


# ---------------------------------------------------------------------------
# Ctrl+key: start chain placement, or report there isn't one
# ---------------------------------------------------------------------------

def test_ctrl_key_starts_chain_placement_for_arms(qapp):
    w = _make_widget()
    handled = w._handle_key(_FakeKeyEvent(Qt.Key.Key_2, Qt.KeyboardModifier.ControlModifier))
    assert handled is True
    assert w._chain_limb == "Left arm"
    assert w._chain_indices == COCO133.limb_chain_indices("Left arm")


def test_ctrl_key_starts_chain_placement_for_face(qapp):
    w = _make_widget()
    w._handle_key(_FakeKeyEvent(Qt.Key.Key_1, Qt.KeyboardModifier.ControlModifier))
    assert w._chain_limb == "Face"
    assert [COCO133.name_of(i) for i in w._chain_indices] == [
        "nose", "left_ear", "right_ear",
    ]


def test_ctrl_on_hand_with_no_chain_emits_status_and_does_not_start_chain(qapp):
    w = _make_widget()
    received = _messages(w)

    handled = w._handle_key(_FakeKeyEvent(Qt.Key.Key_3, Qt.KeyboardModifier.ControlModifier))

    assert handled is True
    assert w._chain_limb is None
    assert received == ["No limb-placement order defined for Left hand yet"]


# ---------------------------------------------------------------------------
# A pose model without the group (e.g. COCO17 has no hands) reports that too
# ---------------------------------------------------------------------------

def test_limb_not_present_in_pose_model_emits_status(qapp):
    w = _make_widget(pose_model=COCO17)
    received = _messages(w)

    handled = w._handle_key(_FakeKeyEvent(Qt.Key.Key_3))  # "Left hand" -- not in COCO17

    assert handled is True
    assert w._hidden_kp_indices == set()
    assert received == ["No Left hand keypoints in this pose model"]
