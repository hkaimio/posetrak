"""Tests for the timeline UX follow-up fixes requested after Phase 13, in
two rounds:

Round 1:
1. The timeline replaces the standalone scrub slider.
2. The timeline starts collapsed to one line and can be expanded/collapsed.
3. Zoom (Ctrl+wheel, +/-/Fit buttons) and horizontal-scrollbar panning.
4. Selection highlighting covers the whole row, and the active-range
   overlay is confined to selected rows (not every row in the time span).
5. Selecting a keypoint in a crop-grid camera switches the timeline to
   that camera.

Round 2 (after using round 1 revealed further issues):
1. The timeline (tab row + ruler) must render from the start, not only
   after the user finds and clicks the collapse arrow.
2. Zoom anchors on the playhead, not the cursor/click position.
3. Seeking and selecting were conflicting when both lived in the row
   tree's click handler — moved seeking into a dedicated, always-visible
   `_RulerWidget`; the row tree is now selection-only (click clears,
   drag selects).
4. Frame cells get a small gap between them once zoomed in past ~6px/frame.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QPushButton, QSlider

from app.pose.kp_models import COCO17
from app.pose.timeline_status import STATUS_GREEN, STATUS_GREY
from app.ui.keypoint_timeline_widget import (
    LABEL_W,
    MIN_VIEW_SPAN_MS,
    ROW_H,
    RULER_H,
    KeypointTimelineWidget,
    Row,
    _fmt_tick,
    _RulerWidget,
    _TimelineCanvas,
)


# ---------------------------------------------------------------------------
# 1. Click/drag-to-seek now lives in _RulerWidget, not the row-tree canvas
# ---------------------------------------------------------------------------

@pytest.fixture()
def canvas(qapp):
    c = _TimelineCanvas(COCO17)
    c.resize(LABEL_W + 200, c.minimumHeight())
    c.set_time_range(0.0, 2.0, "sv1", MagicMock())
    return c


@pytest.fixture()
def ruler(canvas):
    r = _RulerWidget(canvas)
    r.resize(LABEL_W + 200, RULER_H)
    return r


def test_ruler_press_emits_time_scrubbed(ruler):
    received = []
    ruler.time_scrubbed.connect(received.append)

    from PySide6.QtTest import QTest
    QTest.mousePress(ruler, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W + 100, 5))

    assert len(received) == 1
    assert received[0] == pytest.approx(1000, abs=5)  # halfway across a 2000ms window


def test_ruler_drag_emits_time_scrubbed_continuously(ruler):
    from PySide6.QtTest import QTest
    received = []
    ruler.time_scrubbed.connect(received.append)

    QTest.mousePress(ruler, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W, 5))
    QTest.mouseMove(ruler, pos=QPoint(LABEL_W + 50, 5))
    QTest.mouseMove(ruler, pos=QPoint(LABEL_W + 100, 5))
    QTest.mouseRelease(ruler, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W + 100, 5))

    assert len(received) >= 3  # press + 2 moves
    assert received[0] < received[-1]


def test_ruler_ctrl_wheel_zooms_around_playhead(ruler, canvas):
    canvas.set_current_time_v(1000)
    before = canvas.view_range()
    event = QWheelEvent(
        QPointF(LABEL_W + 30, 5), QPointF(LABEL_W + 30, 5),
        QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.ControlModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    ruler.wheelEvent(event)
    after = canvas.view_range()
    assert (after[1] - after[0]) < (before[1] - before[0])
    # Anchored on the playhead (1000), not the wheel event's x position.
    mid = (after[0] + after[1]) / 2
    assert mid == pytest.approx(1000, abs=50)


def test_row_canvas_click_also_emits_time_scrubbed(canvas):
    """Round 3: clicking a keypoint row clears the selection *and* moves the
    playhead — round 2 made row clicks selection-only, but that meant a click
    that unselected everything visibly did nothing else, which read as
    unresponsive. The row tree keeps its own selection-only edit_mode gating;
    scrubbing now works regardless of edit mode, like the ruler."""
    received = []
    canvas.time_scrubbed.connect(received.append)

    from PySide6.QtTest import QTest
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W + 100, 5))

    assert len(received) == 1
    assert received[0] == pytest.approx(1000, abs=5)


def test_row_canvas_click_scrubs_outside_edit_mode_too(canvas):
    assert canvas._edit_mode is False
    received = []
    canvas.time_scrubbed.connect(received.append)

    from PySide6.QtTest import QTest
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W + 50, 5))

    assert len(received) == 1


def test_row_canvas_group_label_click_does_not_scrub(canvas):
    """x < LABEL_W is group-toggle territory, not the clock — same as before."""
    received = []
    canvas.time_scrubbed.connect(received.append)

    from PySide6.QtTest import QTest
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(10, 5))

    assert received == []


# ---------------------------------------------------------------------------
# 2. Zoom — anchored on the playhead (_current_v), not cursor position
# ---------------------------------------------------------------------------

def test_zoom_in_narrows_view_span(canvas):
    full_start, full_end = canvas.view_range()
    canvas.zoom(0.5)
    start, end = canvas.view_range()
    assert (end - start) == pytest.approx((full_end - full_start) * 0.5, abs=1)


def test_zoom_keeps_playhead_time_fixed_on_screen(canvas):
    canvas.set_current_time_v(1234)
    x_before = canvas._x_at_time_v(1234)
    canvas.zoom(0.5)
    x_after = canvas._x_at_time_v(1234)
    assert x_after == pytest.approx(x_before, abs=2)


def test_zoom_anchors_on_playhead_not_cursor(canvas):
    """Regression: zoom used to anchor on the cursor/click x, which felt
    disorienting because it wasn't the thing the user was looking at."""
    canvas.set_current_time_v(1800)  # near the right edge of a 2000ms trial
    canvas.zoom(0.1)  # very tight zoom
    start, end = canvas.view_range()
    assert start <= 1800 <= end


def test_zoom_out_cannot_exceed_total_span(canvas):
    canvas.zoom(10.0)
    start, end = canvas.view_range()
    assert start == 0
    assert end == canvas.total_ms()


def test_zoom_in_respects_minimum_span(canvas):
    for _ in range(30):
        canvas.zoom(0.5)
    start, end = canvas.view_range()
    assert (end - start) >= MIN_VIEW_SPAN_MS


def test_zoom_fit_resets_to_full_range(canvas):
    canvas.zoom(0.3)
    canvas.zoom_fit()
    assert canvas.view_range() == (0, canvas.total_ms())


def test_zoom_emits_view_changed(canvas):
    received = []
    canvas.view_changed.connect(lambda: received.append(True))
    canvas.zoom(0.5)
    assert received == [True]


def test_set_view_start_pans_without_changing_span(canvas):
    canvas.zoom(0.5)  # span now ~1000ms
    _start, end0 = canvas.view_range()
    span = end0 - canvas.view_range()[0]

    canvas.set_view_start(500)

    start, end = canvas.view_range()
    assert start == 500
    assert (end - start) == span


def test_set_view_start_clamped_to_valid_range(canvas):
    canvas.zoom(0.5)
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


def test_collapse_arrow_on_ruler_toggles(qapp):
    """Round 3: the collapse arrow moved from the tab row onto the ruler,
    since the ruler is the row that stays visible while collapsed."""
    w = _make_container(qapp)
    w._ruler.resize(LABEL_W + 200, RULER_H)

    from PySide6.QtTest import QTest
    QTest.mouseClick(w._ruler, Qt.MouseButton.LeftButton, pos=QPoint(8, 5))
    assert w.is_collapsed() is False
    QTest.mouseClick(w._ruler, Qt.MouseButton.LeftButton, pos=QPoint(8, 5))
    assert w.is_collapsed() is True


def test_collapse_arrow_click_does_not_also_scrub(qapp):
    w = _make_container(qapp)
    w._ruler.resize(LABEL_W + 200, RULER_H)
    received = []
    w.time_scrubbed.connect(received.append)

    from PySide6.QtTest import QTest
    QTest.mouseClick(w._ruler, Qt.MouseButton.LeftButton, pos=QPoint(8, 5))

    assert received == []


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
    w._canvas.zoom(0.3)

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
    w._canvas.zoom(0.5)
    assert w._hscroll.maximum() > 0
    assert w._hscroll.pageStep() == pytest.approx(1000, abs=2)


def test_hscroll_value_pans_canvas(qapp):
    w = _make_container(qapp)
    w.set_time_range(0.0, 2.0, "sv1", MagicMock())
    w._canvas.zoom(0.5)  # span ~1000ms, hscroll now pannable

    w._hscroll.setValue(w._hscroll.maximum())

    start, _end = w._canvas.view_range()
    assert start == w._hscroll.maximum()


def test_time_scrubbed_forwarded_by_container(qapp):
    w = _make_container(qapp)
    w.set_time_range(0.0, 2.0, "sv1", MagicMock())
    w._ruler.resize(LABEL_W + 200, RULER_H)
    received = []
    w.time_scrubbed.connect(received.append)

    from PySide6.QtTest import QTest
    QTest.mouseClick(w._ruler, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W + 50, 5))

    assert len(received) == 1


def test_ruler_always_visible_even_when_collapsed(qapp):
    """The ruler (tick marks + playhead) must work as a scrub control without
    expanding the timeline — that's the whole point of pulling it out of the
    collapsible row-tree canvas."""
    w = _make_container(qapp)
    assert w.is_collapsed() is True
    assert w._ruler.isHidden() is False


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


# ---------------------------------------------------------------------------
# 3 (round 2). Plain click on the timeline row tree clears the whole
# selection (keypoints + range), instead of collapsing it to one row.
# ---------------------------------------------------------------------------

def test_timeline_rubber_band_empty_clears_full_selection(qapp, dummy_db):
    w = _make_widget(dummy_db)
    w._sel_kp_indices = {0, 1, 2, 3, 4}
    w._primary_kp_idx = 2
    w._range_start_v = 100
    w._range_end_v = 900

    w._on_timeline_rubber_band(set(), 300, 300, ctrl=False)

    assert w._sel_kp_indices == set()
    assert w._primary_kp_idx is None
    assert w._range_start_v is None
    assert w._range_end_v is None
    w._load_frame.assert_called_once()


def test_timeline_rubber_band_real_selection_still_sets_range(qapp, dummy_db):
    w = _make_widget(dummy_db)
    w._on_timeline_rubber_band({0, 1}, 100, 200, ctrl=False)
    assert w._sel_kp_indices == {0, 1}
    assert w._range_start_v == 100
    assert w._range_end_v == 200


def test_timeline_rubber_band_ctrl_empty_is_not_treated_as_clear(qapp, dummy_db):
    """An empty set with ctrl=True (a Ctrl-drag that happened to touch no
    rows) should be a no-op, not a clear — only a plain, non-Ctrl 'click
    into empty space' clears."""
    w = _make_widget(dummy_db)
    w._sel_kp_indices = {0, 1}
    w._range_start_v = 100
    w._range_end_v = 200

    w._on_timeline_rubber_band(set(), 300, 300, ctrl=True)

    assert w._sel_kp_indices == {0, 1}
    assert w._range_start_v == 100
    assert w._range_end_v == 200


# ---------------------------------------------------------------------------
# 4 (round 2). Frame gaps once zoomed in past ~6px/frame
# ---------------------------------------------------------------------------

@pytest.fixture()
def framegap_canvas(qapp):
    """30fps sync table (~33ms/frame); resized wide so zooming in can exceed
    the 6px/frame gap threshold."""
    c = _TimelineCanvas(COCO17)
    c.resize(LABEL_W + 900, c.minimumHeight())
    sync = MagicMock()
    sync.lookup = lambda t, svid: int(t * 30)
    c.set_time_range(0.0, 2.0, "sv1", sync)
    return c


def test_should_gap_frames_false_when_zoomed_out(framegap_canvas):
    # Full 2s view across 900px ≈ 0.45 px/ms ≈ 15px/frame at 30fps... but the
    # *view* span matters, not raw width; zoom out further to be sure we're
    # under threshold isn't needed since 900px/2000ms/frame(~33ms) ≈ 15px,
    # which is already above 6 — zoom out explicitly to land clearly below it.
    framegap_canvas.zoom_fit()
    framegap_canvas.resize(LABEL_W + 60, framegap_canvas.minimumHeight())
    assert framegap_canvas._should_gap_frames() is False


def test_should_gap_frames_true_when_zoomed_in(framegap_canvas):
    framegap_canvas.set_current_time_v(1000)
    for _ in range(6):
        framegap_canvas.zoom(0.5)
    assert framegap_canvas._should_gap_frames() is True


def test_status_columns_split_by_frame_creates_more_segments(framegap_canvas):
    n = len(COCO17.names)
    status = {f: np.full(n, STATUS_GREEN, dtype=np.int8) for f in range(0, 60)}
    framegap_canvas.set_status_data(status, {}, n_cameras=1)
    framegap_canvas.set_current_time_v(1000)
    for _ in range(6):
        framegap_canvas.zoom(0.5)
    row = Row(kind="leaf", label="nose", kp_indices=(0,))

    unsplit = framegap_canvas._status_columns(row, split_by_frame=False)
    split = framegap_canvas._status_columns(row, split_by_frame=True)

    # Same status throughout → unsplit collapses to one segment; splitting by
    # frame must produce (many) more, one per visible frame.
    assert len(unsplit) == 1
    assert len(split) > 1


def test_estimate_ms_per_frame_matches_sync_table_fps(framegap_canvas):
    ms_per_frame = framegap_canvas._estimate_ms_per_frame()
    assert ms_per_frame == pytest.approx(1000 / 30, rel=0.2)


# ---------------------------------------------------------------------------
# Ruler tick interval selection
# ---------------------------------------------------------------------------

def test_pick_tick_interval_grows_as_span_widens(canvas):
    ruler = _RulerWidget(canvas)
    ruler.resize(LABEL_W + 600, RULER_H)
    narrow = ruler._pick_tick_interval_ms(span_ms=1000, px_width=600)
    wide = ruler._pick_tick_interval_ms(span_ms=600_000, px_width=600)
    assert wide > narrow


# ---------------------------------------------------------------------------
# Round 3, issue 1: ruler tick labels use the capture's global timestamp
# (t_start + v/1000), matching the overlay row's _time_label, not time
# relative to the trial start.
# ---------------------------------------------------------------------------

def test_fmt_tick_matches_time_label_format():
    # Same numeric style as content_panels._fmt_time: f"{s:.3f} s" — same
    # number, so the two rows agree on what "the time" is.
    assert _fmt_tick(12.345) == "12.345"


def test_ruler_ticks_offset_by_t_start(qapp):
    """A trial starting at global t=100s should show ticks near '100.xxx',
    not '0.xxx' — the old (round-2) ruler ignored t_start entirely."""
    c = _TimelineCanvas(COCO17)
    c.resize(LABEL_W + 600, c.minimumHeight())
    c.set_time_range(100.0, 102.0, "sv1", MagicMock())
    ruler = _RulerWidget(c)
    ruler.resize(LABEL_W + 600, RULER_H)

    interval = ruler._pick_tick_interval_ms(2000, 600)
    first_tick_v = (0 // interval) * interval
    label = _fmt_tick(c._t_start + first_tick_v / 1000.0)
    assert label.startswith("100.")


# ---------------------------------------------------------------------------
# Round 3, issue 2: the active-range overlay snaps to whole-frame pixel
# bounds instead of the raw (fractional) millisecond range.
# ---------------------------------------------------------------------------

@pytest.fixture()
def frame_range_canvas(qapp):
    c = _TimelineCanvas(COCO17)
    c.resize(LABEL_W + 900, c.minimumHeight())
    sync = MagicMock()
    sync.lookup = lambda t, svid: int(t * 30)  # 30fps
    c.set_time_range(0.0, 2.0, "sv1", sync)
    return c


def test_range_bounds_snap_to_frame_edges(frame_range_canvas):
    # Frame 15 spans [500, 533.3)ms at 30fps; select a sub-range entirely
    # inside frame 15 (e.g. 510-520ms) — the highlighted pixels should still
    # cover all of frame 15, not just the 10ms sliver that was selected.
    frame_range_canvas.set_selection({0}, 510, 520)
    bounds = frame_range_canvas._range_frame_pixel_bounds()
    assert bounds is not None
    x1, x2 = bounds

    frame15_start_x = frame_range_canvas._x_at_time_v(500)
    frame15_end_x = frame_range_canvas._x_at_time_v(533)
    assert x1 <= frame15_start_x + 2
    assert x2 >= frame15_end_x - 2


def test_range_bounds_none_without_range(frame_range_canvas):
    frame_range_canvas.set_selection({0}, None, None)
    assert frame_range_canvas._range_frame_pixel_bounds() is None


def test_range_bounds_falls_back_without_sync_table(qapp):
    c = _TimelineCanvas(COCO17)
    c.resize(LABEL_W + 200, c.minimumHeight())
    c.set_time_range(0.0, 2.0, None, None)  # no sync table at all
    c.set_selection({0}, 100, 200)
    bounds = c._range_frame_pixel_bounds()
    assert bounds is not None
    x1, x2 = bounds
    assert x1 == pytest.approx(c._x_at_time_v(100), abs=1)
    assert x2 == pytest.approx(c._x_at_time_v(200), abs=1)


# ---------------------------------------------------------------------------
# Round 3, issue "alignment": ruler and canvas map time <-> pixel using
# their own width, and the container keeps a right-margin on the ruler in
# sync with the canvas's vertical scrollbar so ticks line up with rows.
# ---------------------------------------------------------------------------

def test_ruler_mapping_uses_its_own_width_not_canvas(canvas):
    """Regression: the ruler used to delegate to canvas._x_at_time_v, which
    is keyed to canvas.width() — wrong once the two widgets' widths diverge
    (e.g. the canvas's QScrollArea grows a vertical scrollbar)."""
    ruler = _RulerWidget(canvas)
    ruler.resize(LABEL_W + 999, RULER_H)  # deliberately different from canvas's LABEL_W+200
    canvas.set_current_time_v(1000)

    ruler_x = ruler._x_at_time_v(1000)
    canvas_x = canvas._x_at_time_v(1000)
    assert ruler_x != pytest.approx(canvas_x)
    # But it's still correct for the ruler's own width: round-tripping through
    # the ruler's own inverse should land back on 1000.
    assert ruler._time_v_at_x(ruler_x) == pytest.approx(1000, abs=2)


def test_ruler_right_margin_shrinks_drawable_width(canvas):
    ruler = _RulerWidget(canvas)
    ruler.resize(LABEL_W + 200, RULER_H)
    x_before = ruler._x_at_time_v(2000)  # right edge of a 2000ms trial

    ruler.set_right_margin(20)
    x_after = ruler._x_at_time_v(2000)

    assert x_after == pytest.approx(x_before - 20, abs=1)


def test_sync_ruler_margin_zero_when_scrollbar_hidden(qapp):
    w = _make_container(qapp)
    w._sync_ruler_margin()
    assert w._ruler._right_margin == 0


def test_pick_tick_interval_keeps_labels_legibly_spaced(canvas):
    ruler = _RulerWidget(canvas)
    interval = ruler._pick_tick_interval_ms(span_ms=10_000, px_width=600)
    px_per_tick = 600 * interval / 10_000
    assert px_per_tick >= 60
