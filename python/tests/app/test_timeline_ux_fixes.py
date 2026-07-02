"""Tests for the timeline UX follow-up fixes requested after Phase 13:

1. The timeline replaces the standalone scrub slider — click/drag on it
   always seeks, in and out of edit mode.
2. The timeline starts collapsed to one line and can be expanded/collapsed.
3. Zoom (Ctrl+wheel, +/-/Fit buttons) and horizontal-scrollbar panning.
4. Selection highlighting covers the whole row, and the active-range
   overlay is confined to selected rows (not every row in the time span).
5. Selecting a keypoint in a crop-grid camera switches the timeline to
   that camera.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QPushButton, QSlider

from app.pose.kp_models import COCO17
from app.ui.keypoint_timeline_widget import (
    LABEL_W,
    MIN_VIEW_SPAN_MS,
    ROW_H,
    KeypointTimelineWidget,
    Row,
    _TimelineCanvas,
)


# ---------------------------------------------------------------------------
# 1. Click/drag-to-seek (_TimelineCanvas.time_scrubbed)
# ---------------------------------------------------------------------------

@pytest.fixture()
def canvas(qapp):
    c = _TimelineCanvas(COCO17)
    c.resize(LABEL_W + 200, c.minimumHeight())
    c.set_time_range(0.0, 2.0, "sv1", MagicMock())
    return c


def test_press_in_track_area_emits_time_scrubbed_outside_edit_mode(canvas):
    """Scrubbing must work even when not in edit mode — it's the only clock now."""
    assert canvas._edit_mode is False
    received = []
    canvas.time_scrubbed.connect(received.append)

    from PySide6.QtTest import QTest
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W + 100, 5))

    assert len(received) == 1
    assert received[0] == pytest.approx(1000, abs=5)  # halfway across a 2000ms window


def test_drag_emits_time_scrubbed_continuously(canvas):
    from PySide6.QtTest import QTest
    received = []
    canvas.time_scrubbed.connect(received.append)

    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W, 5))
    QTest.mouseMove(canvas, pos=QPoint(LABEL_W + 50, 5))
    QTest.mouseMove(canvas, pos=QPoint(LABEL_W + 100, 5))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W + 100, 5))

    assert len(received) >= 3  # press + 2 moves
    assert received[0] < received[-1]


def test_click_on_group_label_does_not_emit_time_scrubbed(canvas):
    """Clicking the label column (x < LABEL_W) is group-toggle territory, not the clock."""
    received = []
    canvas.time_scrubbed.connect(received.append)

    from PySide6.QtTest import QTest
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(10, 5))

    assert received == []


# ---------------------------------------------------------------------------
# 2. Zoom
# ---------------------------------------------------------------------------

def test_zoom_in_narrows_view_span(canvas):
    full_start, full_end = canvas.view_range()
    canvas.zoom(0.5, LABEL_W + 100)
    start, end = canvas.view_range()
    assert (end - start) == pytest.approx((full_end - full_start) * 0.5, abs=1)


def test_zoom_keeps_anchor_time_fixed(canvas):
    anchor_x = LABEL_W + 100
    anchor_v_before = canvas._time_v_at_x(anchor_x)
    canvas.zoom(0.5, anchor_x)
    anchor_v_after = canvas._time_v_at_x(anchor_x)
    assert anchor_v_after == pytest.approx(anchor_v_before, abs=2)


def test_zoom_out_cannot_exceed_total_span(canvas):
    canvas.zoom(10.0, LABEL_W + 100)
    start, end = canvas.view_range()
    assert start == 0
    assert end == canvas.total_ms()


def test_zoom_in_respects_minimum_span(canvas):
    for _ in range(30):
        canvas.zoom(0.5, LABEL_W + 100)
    start, end = canvas.view_range()
    assert (end - start) >= MIN_VIEW_SPAN_MS


def test_zoom_fit_resets_to_full_range(canvas):
    canvas.zoom(0.3, LABEL_W + 50)
    canvas.zoom_fit()
    assert canvas.view_range() == (0, canvas.total_ms())


def test_zoom_emits_view_changed(canvas):
    received = []
    canvas.view_changed.connect(lambda: received.append(True))
    canvas.zoom(0.5, LABEL_W + 100)
    assert received == [True]


def test_set_view_start_pans_without_changing_span(canvas):
    canvas.zoom(0.5, LABEL_W + 100)  # span now ~1000ms
    _start, end0 = canvas.view_range()
    span = end0 - canvas.view_range()[0]

    canvas.set_view_start(500)

    start, end = canvas.view_range()
    assert start == 500
    assert (end - start) == span


def test_set_view_start_clamped_to_valid_range(canvas):
    canvas.zoom(0.5, LABEL_W + 100)
    canvas.set_view_start(-100)
    assert canvas.view_range()[0] == 0

    canvas.set_view_start(100_000)
    start, end = canvas.view_range()
    assert end == canvas.total_ms()


def test_wheel_with_ctrl_zooms(canvas):
    before = canvas.view_range()
    event = QWheelEvent(
        QPointF(LABEL_W + 100, 5), QPointF(LABEL_W + 100, 5),
        QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.ControlModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    canvas.wheelEvent(event)
    after = canvas.view_range()
    assert after != before
    assert (after[1] - after[0]) < (before[1] - before[0])


def test_wheel_without_ctrl_does_not_zoom(canvas):
    before = canvas.view_range()
    event = QWheelEvent(
        QPointF(LABEL_W + 100, 5), QPointF(LABEL_W + 100, 5),
        QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    canvas.wheelEvent(event)
    assert canvas.view_range() == before


# ---------------------------------------------------------------------------
# 4. Selection visualization
# ---------------------------------------------------------------------------

def test_is_row_selected_true_for_intersecting_leaf(canvas):
    row = Row(kind="leaf", label="nose", kp_indices=(0,))
    canvas.set_selection({0, 1}, None, None)
    assert canvas._is_row_selected(row) is True


def test_is_row_selected_false_when_no_overlap(canvas):
    row = Row(kind="leaf", label="nose", kp_indices=(0,))
    canvas.set_selection({5, 6}, None, None)
    assert canvas._is_row_selected(row) is False


def test_is_row_selected_false_when_selection_empty(canvas):
    row = Row(kind="leaf", label="nose", kp_indices=(0,))
    canvas.set_selection(set(), None, None)
    assert canvas._is_row_selected(row) is False


def test_is_row_selected_true_for_group_with_any_selected_child(canvas):
    row = Row(kind="group", label="Face", kp_indices=(0, 1, 2, 3, 4))
    canvas.set_selection({2}, None, None)
    assert canvas._is_row_selected(row) is True


# ---------------------------------------------------------------------------
# KeypointTimelineWidget: collapse/expand
# ---------------------------------------------------------------------------

def _make_container(qapp):
    cameras = [
        {"shot_video_id": "sv1", "camera_instance_id": "ci1", "label": "A"},
        {"shot_video_id": "sv2", "camera_instance_id": "ci2", "label": "B"},
    ]
    return KeypointTimelineWidget(COCO17, cameras)


def test_starts_collapsed(qapp):
    w = _make_container(qapp)
    assert w.is_collapsed() is True
    assert w._canvas_scroll.isHidden() is True
    assert w._hscroll.isHidden() is True


def test_expand_shows_canvas_and_scrollbar(qapp):
    w = _make_container(qapp)
    w.set_collapsed(False)
    assert w.is_collapsed() is False
    assert w._canvas_scroll.isHidden() is False
    assert w._hscroll.isHidden() is False


def test_collapse_button_toggles(qapp):
    w = _make_container(qapp)
    w._collapse_btn.click()
    assert w.is_collapsed() is False
    w._collapse_btn.click()
    assert w.is_collapsed() is True


def test_set_collapsed_emits_signal(qapp):
    w = _make_container(qapp)
    received = []
    w.collapsed_changed.connect(received.append)
    w.set_collapsed(False)
    assert received == [False]


def test_collapsed_height_is_small(qapp):
    w = _make_container(qapp)
    assert w.maximumHeight() < 60  # roughly one row/tab-bar tall


def test_expanded_height_is_unbounded(qapp):
    w = _make_container(qapp)
    w.set_collapsed(False)
    assert w.maximumHeight() > 1000


# ---------------------------------------------------------------------------
# KeypointTimelineWidget: zoom buttons + horizontal scrollbar
# ---------------------------------------------------------------------------

def test_zoom_buttons_call_through_to_canvas(qapp):
    w = _make_container(qapp)
    w.set_time_range(0.0, 2.0, "sv1", MagicMock())
    w._canvas.resize(LABEL_W + 200, w._canvas.minimumHeight())
    before = w._canvas.view_range()

    zoom_in_btn = next(b for b in w.findChildren(QPushButton) if b.text() == "+")
    zoom_in_btn.click()

    after = w._canvas.view_range()
    assert (after[1] - after[0]) < (before[1] - before[0])


def test_fit_button_resets_zoom(qapp):
    w = _make_container(qapp)
    w.set_time_range(0.0, 2.0, "sv1", MagicMock())
    w._canvas.resize(LABEL_W + 200, w._canvas.minimumHeight())
    w._canvas.zoom(0.3, LABEL_W + 50)

    fit_btn = next(b for b in w.findChildren(QPushButton) if b.text() == "Fit")
    fit_btn.click()

    assert w._canvas.view_range() == (0, w._canvas.total_ms())


def test_hscroll_synced_after_set_time_range(qapp):
    w = _make_container(qapp)
    w.set_time_range(0.0, 2.0, "sv1", MagicMock())
    assert w._hscroll.minimum() == 0
    assert w._hscroll.maximum() == 0  # full view visible → nothing to pan
    assert w._hscroll.pageStep() == 2000


def test_hscroll_range_grows_after_zoom(qapp):
    w = _make_container(qapp)
    w.set_time_range(0.0, 2.0, "sv1", MagicMock())
    w._canvas.zoom(0.5, LABEL_W + 100)
    assert w._hscroll.maximum() > 0
    assert w._hscroll.pageStep() == pytest.approx(1000, abs=2)


def test_hscroll_value_pans_canvas(qapp):
    w = _make_container(qapp)
    w.set_time_range(0.0, 2.0, "sv1", MagicMock())
    w._canvas.zoom(0.5, LABEL_W + 100)  # span ~1000ms, hscroll now pannable

    w._hscroll.setValue(w._hscroll.maximum())

    start, _end = w._canvas.view_range()
    assert start == w._hscroll.maximum()


def test_time_scrubbed_forwarded_by_container(qapp):
    w = _make_container(qapp)
    w.set_time_range(0.0, 2.0, "sv1", MagicMock())
    w._canvas.resize(LABEL_W + 200, w._canvas.minimumHeight())
    received = []
    w.time_scrubbed.connect(received.append)

    from PySide6.QtTest import QTest
    QTest.mouseClick(w._canvas, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W + 50, 5))

    assert len(received) == 1


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
    w._cameras = [
        {"shot_video_id": "sv1", "camera_instance_id": "ci1", "label": "A"},
        {"shot_video_id": "sv2", "camera_instance_id": "ci2", "label": "B"},
    ]
    w._t_start = 0.0
    w._t_end = 2.0
    w._current_t = 0.0
    w._slider = QSlider(Qt.Orientation.Horizontal)
    w._slider.setMinimum(0)
    w._slider.setMaximum(2000)
    w._slider.setSingleStep(33)
    w._slider.setValue(0)
    w._time_label = None
    w._edit_mode = False
    w._sel_kp_indices = set()
    w._primary_kp_idx = None
    w._sel_cam_idx = None
    w._obs_kp = {}
    w._range_start_v = None
    w._range_end_v = None
    w._timeline_status_by_cam = {}
    w._timeline_inlier_counts = {}
    w._seg_sources = {}
    w._track_segs = {}
    w._sync_table = MagicMock()
    w._sync_table.lookup = lambda t, svid: round(t * 30)
    w._load_frame = MagicMock()
    w._timeline = MagicMock()
    w._timeline.active_camera_index.return_value = 0
    return w


@pytest.fixture()
def dummy_db(tmp_path):
    from posetrak.db.db import create_session
    conn = create_session(tmp_path / "dummy.db")
    conn.execute("PRAGMA foreign_keys = OFF")
    yield conn
    conn.close()


def test_on_timeline_scrub_updates_slider_value(qapp, dummy_db):
    w = _make_widget(dummy_db)
    w._on_timeline_scrub(750)
    assert w._slider.value() == 750


def test_on_timeline_scrub_drives_on_slider(qapp, dummy_db):
    """Scrubbing routes through the (now headless) slider's valueChanged signal,
    so _on_slider still runs — this keeps the widget's single time-update path."""
    w = _make_widget(dummy_db)
    w._slider.valueChanged.connect(w._on_slider)
    received = []
    w.time_changed.connect(received.append)

    w._on_timeline_scrub(500)

    assert w._current_t == pytest.approx(w._t_start + 0.5)
    assert received == [pytest.approx(0.5)]
    w._load_frame.assert_called_once()


def test_on_timeline_scrub_clamped_to_slider_range(qapp, dummy_db):
    w = _make_widget(dummy_db)
    w._on_timeline_scrub(999_999)
    assert w._slider.value() == w._slider.maximum()


def test_sync_timeline_follows_selected_camera(qapp, dummy_db):
    w = _make_widget(dummy_db)
    w._timeline.active_camera_index.return_value = 0
    w._sel_cam_idx = 1
    w._push_timeline_camera_data = MagicMock()

    w._sync_timeline(0.0)

    w._timeline.set_active_camera.assert_called_once_with(1)
    w._push_timeline_camera_data.assert_called_once_with(1)


def test_sync_timeline_noop_when_camera_already_matches(qapp, dummy_db):
    w = _make_widget(dummy_db)
    w._timeline.active_camera_index.return_value = 1
    w._sel_cam_idx = 1
    w._push_timeline_camera_data = MagicMock()

    w._sync_timeline(0.0)

    w._timeline.set_active_camera.assert_not_called()
    w._push_timeline_camera_data.assert_not_called()


def test_sync_timeline_noop_when_no_camera_selected(qapp, dummy_db):
    w = _make_widget(dummy_db)
    w._sel_cam_idx = None
    w._push_timeline_camera_data = MagicMock()

    w._sync_timeline(0.0)

    w._timeline.set_active_camera.assert_not_called()


def test_on_kp_selected_then_sync_timeline_switches_camera(qapp, dummy_db):
    """End-to-end: selecting a keypoint in camera B's crop cell should make
    the timeline show camera B, without the user touching the timeline tabs."""
    w = _make_widget(dummy_db)
    w._timeline.active_camera_index.return_value = 0
    w._push_timeline_camera_data = MagicMock()
    w._load_frame = lambda t: w._sync_timeline(t)  # mimic real _load_frame's tail call

    w._on_kp_selected(cam_idx=1, kp_idx=3)

    assert w._sel_cam_idx == 1
    w._timeline.set_active_camera.assert_called_once_with(1)
