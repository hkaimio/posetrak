# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the keypoint-visibility feature (eye icon on each timeline row):

- Clicking a row's eye icon hides/shows that keypoint (leaf) or every
  keypoint in it (group), via `KeypointTimelineWidget.visibility_toggled`.
- Hidden keypoints are excluded from timeline drag-select and Ctrl+click
  keyframe toggling, and from crop-grid drawing/hit-testing/rubber-band/
  group-select/select-all — they can't be selected, moved, or interpolated.
- kp_models.py: COCO133's "Face" group now only covers the keypoints the
  default skeleton actually attaches markers to (nose + ears); the
  remaining 70 face landmarks (eyes + 68 detail points) form a separate
  "Face (detail)" group.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QSlider

from app.pose.kp_models import COCO17, COCO133
from app.ui.keypoint_timeline_widget import (
    LABEL_W,
    ROW_H,
    KeypointTimelineWidget,
    Row,
    _eye_icon_x_range,
    _TimelineCanvas,
)


# ---------------------------------------------------------------------------
# kp_models.py: Face split
# ---------------------------------------------------------------------------

def test_coco133_face_is_only_skeleton_markers():
    assert COCO133.group_indices("Face") == frozenset({0, 3, 4})
    for idx in (0, 3, 4):
        assert COCO133.name_of(idx) in ("nose", "left_ear", "right_ear")


def test_coco133_face_detail_covers_the_rest():
    detail = COCO133.group_indices("Face (detail)")
    assert len(detail) == 70
    assert detail.isdisjoint(COCO133.group_indices("Face"))
    # Eyes moved into the detail group, not left dangling outside both.
    assert {1, 2} <= detail


def test_coco133_tree_groups_include_both_face_groups():
    assert "Face" in COCO133.tree_groups
    assert "Face (detail)" in COCO133.tree_groups


def test_coco17_face_unchanged():
    """COCO17 has no detailed landmarks to split out — leave it alone."""
    assert COCO17.group_indices("Face") == frozenset({0, 1, 2, 3, 4})


# ---------------------------------------------------------------------------
# _TimelineCanvas: eye icon click -> visibility_toggled
# ---------------------------------------------------------------------------

@pytest.fixture()
def canvas(qapp):
    c = _TimelineCanvas(COCO17)
    c.resize(LABEL_W + 200, c.minimumHeight())
    c.set_time_range(0.0, 2.0, "sv1", MagicMock())
    c.set_edit_mode(True)
    return c


def _eye_x() -> int:
    x0, x1 = _eye_icon_x_range()
    return int((x0 + x1) / 2)


def test_eye_icon_click_on_leaf_emits_visibility_toggled(canvas):
    canvas.toggle_group("Face")
    leaf_idx = next(i for i, r in enumerate(canvas._rows) if r.kind == "leaf")
    y = leaf_idx * ROW_H + ROW_H // 2

    received = []
    canvas.visibility_toggled.connect(lambda kp: received.append(frozenset(kp)))

    from PySide6.QtTest import QTest
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(_eye_x(), y))

    assert received == [frozenset(canvas._rows[leaf_idx].kp_indices)]


def test_eye_icon_click_on_group_emits_all_its_indices(canvas):
    assert canvas._rows[0].kind == "group"
    received = []
    canvas.visibility_toggled.connect(lambda kp: received.append(frozenset(kp)))

    from PySide6.QtTest import QTest
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(_eye_x(), ROW_H // 2))

    assert received == [frozenset(canvas._rows[0].kp_indices)]


def test_eye_icon_click_does_not_toggle_group_expand(canvas):
    assert canvas._expanded == set()
    from PySide6.QtTest import QTest
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(_eye_x(), ROW_H // 2))
    assert canvas._expanded == set()


def test_eye_icon_click_does_not_start_a_drag(canvas):
    received = []
    canvas.rubber_band_selected.connect(lambda *a: received.append(a))
    from PySide6.QtTest import QTest
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(_eye_x(), ROW_H // 2))
    QTest.mouseMove(canvas, pos=QPoint(_eye_x() + 40, ROW_H // 2))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(_eye_x() + 40, ROW_H // 2))
    assert received == []


def test_eye_icon_click_outside_edit_mode_still_works(canvas):
    """Visibility is a viewing preference, not an edit action — works regardless of edit mode."""
    canvas.set_edit_mode(False)
    received = []
    canvas.visibility_toggled.connect(lambda kp: received.append(frozenset(kp)))
    from PySide6.QtTest import QTest
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(_eye_x(), ROW_H // 2))
    assert len(received) == 1


# ---------------------------------------------------------------------------
# _TimelineCanvas: hidden-state effects
# ---------------------------------------------------------------------------

def test_is_row_hidden_leaf(canvas):
    row = Row(kind="leaf", label="nose", kp_indices=(0,))
    assert canvas._is_row_hidden(row) is False
    canvas.set_hidden(frozenset({0}))
    assert canvas._is_row_hidden(row) is True


def test_is_row_hidden_group_requires_all_children(canvas):
    row = Row(kind="group", label="Face", kp_indices=(0, 1, 2, 3, 4))
    canvas.set_hidden(frozenset({0, 1}))
    assert canvas._is_row_hidden(row) is False  # only partially hidden
    canvas.set_hidden(frozenset({0, 1, 2, 3, 4}))
    assert canvas._is_row_hidden(row) is True


def test_kp_indices_in_row_range_excludes_hidden(canvas):
    canvas.set_hidden(frozenset(canvas._rows[0].kp_indices))
    result = canvas._kp_indices_in_row_range(0, ROW_H - 1)
    assert result == set()


def test_drag_select_over_partially_hidden_group_only_selects_visible(canvas):
    group = canvas._rows[0]
    one_hidden = next(iter(group.kp_indices))
    canvas.set_hidden(frozenset({one_hidden}))

    received = []
    canvas.rubber_band_selected.connect(lambda *a: received.append(a))
    from PySide6.QtTest import QTest
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W + 10, 2))
    QTest.mouseMove(canvas, pos=QPoint(LABEL_W + 60, 2))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W + 60, 2))

    assert len(received) == 1
    kp_indices, _v0, _v1, _ctrl = received[0]
    assert one_hidden not in kp_indices
    assert kp_indices == set(group.kp_indices) - {one_hidden}


def test_ctrl_click_keyframe_toggle_skips_hidden_leaf(canvas):
    canvas.toggle_group("Face")
    leaf_idx = next(i for i, r in enumerate(canvas._rows) if r.kind == "leaf")
    leaf_row = canvas._rows[leaf_idx]
    canvas.set_hidden(frozenset(leaf_row.kp_indices))
    y = leaf_idx * ROW_H + ROW_H // 2

    received = []
    canvas.keyframe_toggled.connect(lambda *a: received.append(a))
    from PySide6.QtTest import QTest
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier,
                      QPoint(LABEL_W + 10, y))

    assert received == []


# ---------------------------------------------------------------------------
# KeypointTimelineWidget: pass-throughs
# ---------------------------------------------------------------------------

def _make_container(qapp):
    cameras = [{"shot_video_id": "sv1", "camera_instance_id": "ci1", "label": "A"}]
    return KeypointTimelineWidget(COCO17, cameras)


def test_container_forwards_visibility_toggled(qapp):
    w = _make_container(qapp)
    w._canvas.resize(LABEL_W + 200, w._canvas.minimumHeight())
    w._canvas.set_edit_mode(True)
    received = []
    w.visibility_toggled.connect(lambda kp: received.append(frozenset(kp)))

    from PySide6.QtTest import QTest
    QTest.mouseClick(w._canvas, Qt.MouseButton.LeftButton, pos=QPoint(_eye_x(), ROW_H // 2))

    assert len(received) == 1


def test_container_set_hidden_reaches_canvas(qapp):
    w = _make_container(qapp)
    w.set_hidden(frozenset({0, 1}))
    assert w._canvas._hidden_kp == frozenset({0, 1})


# ---------------------------------------------------------------------------
# _ImageCanvas (crop grid): hidden keypoints excluded from hit-testing/drawing
# ---------------------------------------------------------------------------

def _make_image_canvas(qapp, obs_kp: np.ndarray):
    from app.ui.content_panels import _ImageCanvas
    canvas = _ImageCanvas()
    canvas.resize(300, 300)
    canvas.set_overlay(
        obs_kp=obs_kp, joint_xy=None, bone_pairs=[], marker_xy=None,
        show_detected=True, show_tracked=True,
    )
    return canvas


def test_hit_kp_finds_visible_keypoint(qapp):
    obs_kp = np.array([[50.0, 50.0, 0.9], [200.0, 200.0, 0.9]], dtype=np.float32)
    canvas = _make_image_canvas(qapp, obs_kp)
    canvas.set_edit_mode(True)
    assert canvas._hit_kp(50, 50) == 0


def test_hit_kp_skips_hidden_keypoint(qapp):
    obs_kp = np.array([[50.0, 50.0, 0.9], [200.0, 200.0, 0.9]], dtype=np.float32)
    canvas = _make_image_canvas(qapp, obs_kp)
    canvas.set_edit_mode(True)
    canvas.set_hidden(frozenset({0}))
    assert canvas._hit_kp(50, 50) is None


def test_hit_kp_still_finds_other_visible_keypoint_when_one_hidden(qapp):
    obs_kp = np.array([[50.0, 50.0, 0.9], [55.0, 55.0, 0.9]], dtype=np.float32)
    canvas = _make_image_canvas(qapp, obs_kp)
    canvas.set_edit_mode(True)
    canvas.set_hidden(frozenset({0}))
    assert canvas._hit_kp(52, 52) == 1


def test_hit_kp_hidden_ignored_even_with_low_confidence_edit_mode_allowance(qapp):
    """In edit mode, low-confidence (ghost) points are normally still
    hittable — hidden must override that, not just add to it."""
    obs_kp = np.array([[50.0, 50.0, 0.0]], dtype=np.float32)
    canvas = _make_image_canvas(qapp, obs_kp)
    canvas.set_edit_mode(True)
    assert canvas._hit_kp(50, 50) == 0  # ghost point normally hittable in edit mode
    canvas.set_hidden(frozenset({0}))
    assert canvas._hit_kp(50, 50) is None


# ---------------------------------------------------------------------------
# PersonCropGridWidget wiring
# ---------------------------------------------------------------------------

def _make_widget(db):
    from app.ui.content_panels import PersonCropGridWidget
    from PySide6.QtWidgets import QWidget

    w = PersonCropGridWidget.__new__(PersonCropGridWidget)
    QWidget.__init__(w)
    w._conn = db
    w._sequence_id = "seq1"
    w._cells = []
    w._cameras = [{"shot_video_id": "sv1", "camera_instance_id": "ci1", "label": "A"}]
    w._pose_model = COCO17
    w._t_start = 0.0
    w._t_end = 2.0
    w._current_t = 0.0
    w._slider = QSlider(Qt.Orientation.Horizontal)
    w._slider.setMinimum(0)
    w._slider.setMaximum(2000)
    w._slider.setSingleStep(33)
    w._slider.setValue(0)
    w._edit_mode = True
    w._sel_kp_indices = set()
    w._primary_kp_idx = None
    w._sel_cam_idx = 0
    w._obs_kp = {"ci1": {}}
    w._hidden_kp_indices = set()
    w._load_frame = MagicMock()
    w._timeline = MagicMock()
    return w


@pytest.fixture()
def dummy_db(tmp_path):
    from posetrak.db.db import create_session
    conn = create_session(tmp_path / "dummy.db")
    conn.execute("PRAGMA foreign_keys = OFF")
    yield conn
    conn.close()


def test_visibility_toggled_hides_previously_visible(qapp, dummy_db):
    w = _make_widget(dummy_db)
    w._on_timeline_visibility_toggled(frozenset({3, 4}))
    assert w._hidden_kp_indices == {3, 4}
    w._timeline.set_hidden.assert_called_once_with(frozenset({3, 4}))
    w._load_frame.assert_called_once()


def test_visibility_toggled_shows_when_all_already_hidden(qapp, dummy_db):
    w = _make_widget(dummy_db)
    w._hidden_kp_indices = {3, 4, 5}
    w._on_timeline_visibility_toggled(frozenset({3, 4}))
    assert w._hidden_kp_indices == {5}


def test_visibility_toggled_partial_overlap_hides_the_rest(qapp, dummy_db):
    """Matches _is_row_hidden's "all or nothing" rule: a group that's only
    partially hidden reads as visible, so clicking its icon hides the rest
    rather than doing nothing."""
    w = _make_widget(dummy_db)
    w._hidden_kp_indices = {3}
    w._on_timeline_visibility_toggled(frozenset({3, 4}))
    assert w._hidden_kp_indices == {3, 4}


def test_visibility_toggled_purges_selection(qapp, dummy_db):
    w = _make_widget(dummy_db)
    w._sel_kp_indices = {0, 3, 4}
    w._primary_kp_idx = 3
    w._on_timeline_visibility_toggled(frozenset({3, 4}))
    assert w._sel_kp_indices == {0}
    assert w._primary_kp_idx == 0


def test_visibility_toggled_empty_input_is_noop(qapp, dummy_db):
    w = _make_widget(dummy_db)
    w._on_timeline_visibility_toggled(frozenset())
    assert w._hidden_kp_indices == set()
    w._load_frame.assert_not_called()


def test_select_group_excludes_hidden(qapp, dummy_db):
    w = _make_widget(dummy_db)
    w._hidden_kp_indices = {1, 2}  # eyes, part of COCO17's "Face" group {0,1,2,3,4}
    w._select_group("Face")
    assert w._sel_kp_indices == {0, 3, 4}


def test_select_all_excludes_hidden(qapp, dummy_db):
    w = _make_widget(dummy_db)
    w._hidden_kp_indices = {5, 6}
    w._select_all()
    assert 5 not in w._sel_kp_indices
    assert 6 not in w._sel_kp_indices
    assert w._sel_kp_indices == set(COCO17.all_indices) - {5, 6}


def test_rubber_band_selected_excludes_hidden(qapp, dummy_db):
    from app.ui.content_panels import _CropCell

    w = _make_widget(dummy_db)
    cell = _CropCell("cam")
    cell.resize(300, 300)
    w._cells = [cell]
    w._sync_table = MagicMock()
    w._sync_table.lookup = lambda t, svid: 10
    obs_kp = np.array([[10.0, 10.0, 0.9], [20.0, 20.0, 0.9]], dtype=np.float32)
    w._obs_kp = {"ci1": {10: obs_kp}}
    w._hidden_kp_indices = {0}

    w._on_rubber_band_selected(0, 0, 0, 300, 300, ctrl=False)

    assert 0 not in w._sel_kp_indices
    assert 1 in w._sel_kp_indices
