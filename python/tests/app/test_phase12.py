"""Tests for Phase 12: KeypointTimelineWidget skeleton (tree rows, camera tabs,
flat-colored cells, playhead sync).  No selection/rubber-band interaction yet
(that's Phase 13) — this covers the read-only tree/paint-geometry layer.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from app.pose.kp_models import COCO17, PoseModel
from app.pose.timeline_status import STATUS_GREEN, STATUS_GREY
from app.ui.keypoint_timeline_widget import (
    LABEL_W,
    ROW_H,
    KeypointTimelineWidget,
    Row,
    _TimelineCanvas,
    build_rows,
)


# ---------------------------------------------------------------------------
# build_rows
# ---------------------------------------------------------------------------

def test_build_rows_collapsed_shows_only_groups():
    rows = build_rows(COCO17, expanded=set())
    assert [r.label for r in rows] == list(COCO17.tree_groups)
    assert all(r.kind == "group" for r in rows)
    # No leftover for COCO17 (tree_groups exactly partitions all 17 indices).
    assert "Other" not in [r.label for r in rows]


def test_build_rows_expanded_group_shows_leaves_in_order():
    rows = build_rows(COCO17, expanded={"Left arm"})
    labels = [(r.kind, r.label) for r in rows]
    left_arm_pos = labels.index(("group", "Left arm"))
    left_arm_indices = sorted(COCO17.group_indices("Left arm"))
    for offset, kp_idx in enumerate(left_arm_indices, start=1):
        row = rows[left_arm_pos + offset]
        assert row.kind == "leaf"
        assert row.kp_indices == (kp_idx,)
        assert row.label == COCO17.name_of(kp_idx)
        assert row.depth == 1
    # Groups after "Left arm" still appear (nothing else expanded).
    assert ("group", "Right arm") in labels


def test_build_rows_multiple_groups_expanded():
    rows = build_rows(COCO17, expanded={"Face", "Right leg"})
    n_leaves = sum(1 for r in rows if r.kind == "leaf")
    assert n_leaves == len(COCO17.group_indices("Face")) + len(COCO17.group_indices("Right leg"))


def test_build_rows_leftover_indices_become_other_group():
    partial = PoseModel(
        model_id="partial",
        names=("a", "b", "c"),
        groups={"AB": frozenset({0, 1})},
        tree_groups=("AB",),
    )
    rows = build_rows(partial, expanded=set())
    other = [r for r in rows if r.label == "Other"]
    assert len(other) == 1
    assert other[0].kp_indices == (2,)


def test_build_rows_other_group_expands_too():
    partial = PoseModel(
        model_id="partial2",
        names=("a", "b", "c"),
        groups={"AB": frozenset({0, 1})},
        tree_groups=("AB",),
    )
    rows = build_rows(partial, expanded={"Other"})
    leaves = [r for r in rows if r.kind == "leaf" and r.depth == 1]
    assert [r.kp_indices for r in leaves] == [(2,)]


# ---------------------------------------------------------------------------
# _TimelineCanvas geometry helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def canvas(qapp):
    c = _TimelineCanvas(COCO17)
    c.resize(LABEL_W + 200, c.minimumHeight())
    return c


def test_x_at_time_v_endpoints(canvas):
    canvas.set_time_range(0.0, 2.0, "sv1", MagicMock())
    assert abs(canvas._x_at_time_v(0) - LABEL_W) < 1e-6
    assert abs(canvas._x_at_time_v(2000) - canvas.width()) < 1e-6


def test_time_v_at_x_roundtrip(canvas):
    canvas.set_time_range(0.0, 2.0, "sv1", MagicMock())
    for v in (0, 500, 1000, 1999):
        x = canvas._x_at_time_v(v)
        v2 = canvas._time_v_at_x(x)
        assert abs(v2 - v) <= 1  # sub-ms float rounding only


def test_time_v_at_x_clamped_to_range(canvas):
    canvas.set_time_range(0.0, 2.0, "sv1", MagicMock())
    assert canvas._time_v_at_x(0) == 0            # left of label column clamps to 0
    assert canvas._time_v_at_x(10_000) == 2000     # far right clamps to total_ms


def test_row_at_y(canvas):
    rows = canvas._rows
    assert canvas._row_at_y(0) is rows[0]
    assert canvas._row_at_y(ROW_H) is rows[1]
    assert canvas._row_at_y(ROW_H * len(rows) + 100) is None


def test_frame_at_time_v_uses_sync_table(canvas):
    sync = MagicMock()
    sync.lookup = lambda t, svid: round(t * 30)
    canvas.set_time_range(0.0, 2.0, "sv1", sync)
    assert canvas._frame_at_time_v(1000) == 30


def test_frame_at_time_v_none_without_sync_table(canvas):
    assert canvas._frame_at_time_v(0) is None


# ---------------------------------------------------------------------------
# Status / inlier column rendering data
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_canvas(qapp):
    """Narrow canvas: LABEL_W + 20px → 10 buckets of COL_W=2."""
    c = _TimelineCanvas(COCO17)
    c.resize(LABEL_W + 20, c.minimumHeight())
    sync = MagicMock()
    # First half of the width maps to frame 1, second half to frame 2 (30fps, 2s trial).
    sync.lookup = lambda t, svid: 1 if t < 1.0 else 2
    c.set_time_range(0.0, 2.0, "sv1", sync)
    return c


def test_status_columns_two_segments(small_canvas):
    status = {
        1: np.full(len(COCO17.names), STATUS_GREEN, dtype=np.int8),
        2: np.full(len(COCO17.names), STATUS_GREY, dtype=np.int8),
    }
    small_canvas.set_status_data(status, {}, n_cameras=1)
    row = Row(kind="leaf", label="nose", kp_indices=(0,))
    segments = small_canvas._status_columns(row)
    codes = [code for _x, _w, code in segments]
    assert codes == [STATUS_GREEN, STATUS_GREY]
    # Segments should be contiguous and cover the full drawable width.
    total_w = sum(w for _x, w, _c in segments)
    assert total_w == small_canvas.width() - LABEL_W


def test_status_columns_no_data_when_frame_missing(small_canvas):
    row = Row(kind="leaf", label="nose", kp_indices=(0,))
    segments = small_canvas._status_columns(row)
    assert all(code == -1 for _x, _w, code in segments)


def test_status_columns_group_row_aggregates_max_precedence(small_canvas):
    """A group row's code is the worst (max) status among its kp_indices."""
    n = len(COCO17.names)
    frame1 = np.full(n, STATUS_GREEN, dtype=np.int8)
    frame1[1] = STATUS_GREY  # one bad keypoint in the group
    small_canvas.set_status_data({1: frame1, 2: frame1}, {}, n_cameras=1)
    group_row = Row(kind="group", label="Face", kp_indices=tuple(sorted(COCO17.group_indices("Face"))))
    segments = small_canvas._status_columns(group_row)
    assert all(code == STATUS_GREY for _x, _w, code in segments)


def test_inlier_fraction_columns(small_canvas):
    counts = {
        1: np.array([2, 0], dtype=np.int16),
        2: np.array([1, 0], dtype=np.int16),
    }
    small_canvas.set_status_data({}, counts, n_cameras=2)
    segments = small_canvas._inlier_fraction_columns(kp_idx=0)
    fracs = [f for _x, _w, f in segments]
    assert fracs == pytest.approx([1.0, 0.5])


def test_inlier_fraction_columns_no_data_is_zero(small_canvas):
    segments = small_canvas._inlier_fraction_columns(kp_idx=0)
    assert all(f == 0.0 for _x, _w, f in segments)


# ---------------------------------------------------------------------------
# Group expand/collapse via mouse click on the label column
# ---------------------------------------------------------------------------

def test_click_on_group_label_expands(canvas):
    assert canvas._expanded == set()
    first_row = canvas._rows[0]
    assert first_row.kind == "group"
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(10, 2))
    assert first_row.label in canvas._expanded
    assert len(canvas._rows) > 1


def test_click_on_group_label_twice_collapses(canvas):
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(10, 2))
    n_expanded = len(canvas._rows)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(10, 2))
    assert canvas._expanded == set()
    assert len(canvas._rows) < n_expanded


def test_click_in_status_area_does_not_toggle(canvas):
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W + 5, 2))
    assert canvas._expanded == set()


# ---------------------------------------------------------------------------
# KeypointTimelineWidget: camera tabs
# ---------------------------------------------------------------------------

def _make_widget(qapp):
    cameras = [
        {"shot_video_id": "sv1", "camera_instance_id": "ci1", "label": "A"},
        {"shot_video_id": "sv2", "camera_instance_id": "ci2", "label": "B"},
    ]
    return KeypointTimelineWidget(COCO17, cameras)


def test_default_active_camera_is_zero(qapp):
    w = _make_widget(qapp)
    assert w.active_camera_index() == 0
    assert w._cam_buttons[0].isChecked()
    assert not w._cam_buttons[1].isChecked()


def test_clicking_tab_emits_camera_changed_and_updates_state(qapp):
    w = _make_widget(qapp)
    received: list[int] = []
    w.camera_changed.connect(received.append)

    w._cam_buttons[1].click()

    assert received == [1]
    assert w.active_camera_index() == 1
    assert w._cam_buttons[1].isChecked()
    assert not w._cam_buttons[0].isChecked()


def test_set_active_camera_updates_buttons_without_signal(qapp):
    w = _make_widget(qapp)
    received: list[int] = []
    w.camera_changed.connect(received.append)

    w.set_active_camera(1)

    assert received == []  # programmatic sync from the host must not re-emit
    assert w.active_camera_index() == 1
    assert w._cam_buttons[1].isChecked()
