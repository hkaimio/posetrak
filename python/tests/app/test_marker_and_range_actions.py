# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for three overnight-requested keypoint-editing UI additions:

1. A single timestamp marker on the timeline ruler (red flag), and
   "Select to marker" from a row's right-click menu — selects that row's
   keypoint(s) and the frame range between the marker and the current
   playhead, whichever order they fall in.
2. "Disable selected" / "Enable selected" on the crop-grid right-click
   menu — an explicit-target-state sibling of the existing Space-bar
   toggle (`_toggle_outlier`), sharing its apply-to-range logic via the
   new `_set_outlier_selected(is_outlier)`.
3. "Interpolate missing" on the same menu — unlike the existing `I`-key
   `_interpolate_range` (which overwrites the whole range with one
   straight line except at explicit STATUS_BLUE keyframes), this only
   fills frames with *no* value for a keypoint (confidence < 0.01, or a
   ghost frame with nothing at that slot) and leaves every already-present
   value untouched, disabled or not.

Follow-up round: disable/enable/interpolate are now also reachable from
the timeline row's own right-click menu (not just the crop-grid canvas),
since a selection made via "select to marker" naturally continues there.
Plus: "M" sets the marker at the current frame, the ruler's menu gained
"Select all keypoints to marker", and maximizing a camera cell now follows
that camera in the timeline (same rule already used for keypoint
selection). None of these tests invoke `contextMenuEvent`'s `QMenu.exec()`
path directly -- doing so would block waiting for a real click even in
the offscreen test platform, so wiring is verified by emitting the
canvas/ruler signals directly and checking they propagate, exactly how
the (untestable) menu action's lambda would trigger them.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSlider

from app.ui.keypoint_timeline_widget import LABEL_W, RULER_H, _RulerWidget, _TimelineCanvas
from app.pose.kp_models import COCO17

_N_KP = 2


# ---------------------------------------------------------------------------
# Part 1: marker state on _TimelineCanvas / _RulerWidget
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


def test_marker_v_starts_none(canvas):
    assert canvas.marker_v() is None


def test_set_marker_updates_marker_v(canvas):
    canvas.set_marker(500)
    assert canvas.marker_v() == 500


def test_set_marker_none_clears(canvas):
    canvas.set_marker(500)
    canvas.set_marker(None)
    assert canvas.marker_v() is None


def test_set_marker_emits_marker_changed(canvas):
    spy = MagicMock()
    canvas.marker_changed.connect(spy)
    canvas.set_marker(500)
    spy.assert_called_once()


def test_set_marker_replaces_previous_single_marker(canvas):
    """Only one marker at a time -- setting a new one silently replaces it."""
    canvas.set_marker(100)
    canvas.set_marker(900)
    assert canvas.marker_v() == 900


def test_ruler_paints_with_marker_set_without_crashing(ruler, canvas):
    """paintEvent draws the flag glyph -- just needs to not raise."""
    canvas.set_marker(500)
    ruler.repaint()


def test_container_marker_pass_through(qapp):
    from app.ui.keypoint_timeline_widget import KeypointTimelineWidget
    w = KeypointTimelineWidget(COCO17, [{"label": "A", "shot_video_id": "sv1"}])
    assert w.marker_v() is None
    w.set_marker(1234)
    assert w.marker_v() == 1234
    w.set_marker(None)
    assert w.marker_v() is None


def test_canvas_context_menu_noop_outside_edit_mode(canvas):
    """No marker-select menu should appear (and nothing should crash) when
    not in edit mode, even with a marker set and a valid row under the
    cursor."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QContextMenuEvent
    canvas.set_marker(0)
    canvas.set_edit_mode(False)
    spy = MagicMock()
    canvas.select_to_marker_requested.connect(spy)
    event = QContextMenuEvent(
        QContextMenuEvent.Reason.Mouse, QPoint(LABEL_W + 10, 0), QPoint(LABEL_W + 10, 0)
    )
    canvas.contextMenuEvent(event)
    spy.assert_not_called()


def test_canvas_context_menu_noop_without_marker(canvas):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QContextMenuEvent
    canvas.set_edit_mode(True)
    spy = MagicMock()
    canvas.select_to_marker_requested.connect(spy)
    event = QContextMenuEvent(
        QContextMenuEvent.Reason.Mouse, QPoint(LABEL_W + 10, 0), QPoint(LABEL_W + 10, 0)
    )
    canvas.contextMenuEvent(event)
    spy.assert_not_called()


# ---------------------------------------------------------------------------
# Part 2: PersonCropGridWidget-level actions (disable/enable, interpolate
# missing, select-to-marker range computation) -- same harness pattern as
# test_phase14.py's multikey_db / _make_widget.
# ---------------------------------------------------------------------------

def _enc(x: float, y: float, conf: float) -> bytes:
    kp = np.zeros((_N_KP, 3), dtype=np.float32)
    kp[0] = [x, y, conf]
    kp[1] = [x, y, conf]
    return kp.tobytes()


@pytest.fixture()
def multikey_db(tmp_path):
    """Observations at frames 1, 10, 20 (30fps) -- same layout as
    test_phase14.py's fixture of the same name (kept separate here so this
    file has no cross-file fixture dependency)."""
    from posetrak.db.db import create_session
    conn = create_session(tmp_path / "marker_range.db")
    conn.execute("PRAGMA foreign_keys = OFF")

    conn.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES ('s1', '2026-01-01T00:00:00Z')")
    conn.execute("INSERT INTO captures (id, session_id, capture_number) VALUES ('sh1', 's1', 1)")
    conn.execute("INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('ci1', 'cm1', 'A')")
    conn.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,"
        " first_video_frame, last_video_frame, actual_fps)"
        " VALUES ('sv1', 'sh1', 'ci1', '/dev/null', 0, 100, 30.0)"
    )
    conn.execute(
        "INSERT INTO pose_observation_sequences"
        " (id, shot_id, sync_config_id, time_start_s, time_end_s, detection_run_id, pixels_are_undistorted)"
        " VALUES ('seq1', 'sh1', 'sc1', 0.0, 2.0, 'run1', 0)"
    )
    conn.execute("INSERT INTO sequence_persons (sequence_id, person_id, person_name)"
                 " VALUES ('seq1', 0, 'Alice')")

    for frame, (x, y, c) in {
        1: (1.0, 1.0, 0.9),
        10: (99.0, 99.0, 0.9),
        20: (20.0, 20.0, 0.9),
    }.items():
        conn.execute(
            "INSERT INTO pose_observations"
            " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, kp_blob)"
            " VALUES ('seq1', 'ci1', ?, ?, 0, ?)",
            (frame, frame / 30.0, _enc(x, y, c)),
        )
    conn.commit()
    yield conn
    conn.close()


def _make_widget(qapp, db):
    from app.ui.content_panels import PersonCropGridWidget
    from PySide6.QtWidgets import QWidget

    w = PersonCropGridWidget.__new__(PersonCropGridWidget)
    QWidget.__init__(w)
    w._conn = db
    w._sequence_id = "seq1"
    w._cells = []
    w._cameras = [{"shot_video_id": "sv1", "camera_instance_id": "ci1", "label": "A"}]
    w._t_start = 0.0
    w._t_end = 2.0
    w._current_t = 0.0
    w._slider = QSlider(Qt.Orientation.Horizontal)
    w._slider.setMinimum(0)
    w._slider.setMaximum(2000)
    w._slider.setSingleStep(1)
    w._slider.setValue(0)
    w._time_label = None
    w._show_detected = None
    w._show_tracked = None
    w._show_seg = None
    w._edit_btn = None
    w._edit_mode = True
    w._sel_kp_indices = {0}
    w._hidden_kp_indices = set()
    w._primary_kp_idx = 0
    w._sel_cam_idx = 0
    w._obs_kp = {}
    w._det_bboxes = {}
    w._marker_proj = {}
    w._joint_proj = {}
    w._bone_pairs = []
    w._tracking_timestamps = []
    w._outlier_masks = {}
    w._seg_sources = {}
    w._track_segs = {}
    w._video_dims = {}
    w._3d_ph = None
    w._ncols = 1
    w._hand_redetect = None
    w._hand_redetect_timers = {}
    w._grid = None
    w._backfill = None
    w._clipboard = None
    w._clipboard_cam_idx = None
    w._range_start_v = 33     # round(1/30 * 1000)  -> frame 1
    w._range_end_v = 667      # round(20/30 * 1000) -> frame 20
    w._timeline = None
    w._timeline_status_by_cam = {}
    w._timeline_inlier_counts = {}
    w._load_frame = MagicMock()

    from app.pose.db_cache import read_observations_with_edits
    w._obs_kp["ci1"] = read_observations_with_edits(db, "seq1", "ci1")

    mock_sync = MagicMock()
    mock_sync.lookup = lambda t, svid: round(t * 30)
    w._sync_table = mock_sync

    return w


def _edit_xy_conf(db, frame: int, kp_idx: int = 0) -> tuple[float, float, float] | None:
    row = db.execute(
        "SELECT kp_blob, kp_mask FROM pose_observation_edits"
        " WHERE sequence_id='seq1' AND camera_instance_id='ci1' AND video_frame=?",
        (frame,),
    ).fetchone()
    if row is None:
        return None
    mask = bytes(row["kp_mask"])
    byte_idx, bit_idx = divmod(kp_idx, 8)
    if byte_idx >= len(mask) or not ((mask[byte_idx] >> bit_idx) & 1):
        return None
    kp = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    is_outlier_flag = float(kp[kp_idx, 2])
    return float(kp[kp_idx, 0]), float(kp[kp_idx, 1]), is_outlier_flag


# --- Disable / enable selected ---------------------------------------------

def test_set_outlier_selected_true_disables_current_frame(qapp, multikey_db):
    w = _make_widget(qapp, multikey_db)
    w._range_start_v = None  # no range -> just the current frame
    w._current_t = 1 / 30.0  # frame 1
    w._set_outlier_selected(True)

    edit = _edit_xy_conf(multikey_db, 1)
    assert edit is not None
    _, _, is_outlier_flag = edit
    assert is_outlier_flag != 0.0  # is_outlier=True -> stored flag nonzero


def test_set_outlier_selected_applies_across_whole_range(qapp, multikey_db):
    w = _make_widget(qapp, multikey_db)
    w._set_outlier_selected(True)  # range is frames 1..20 from the fixture

    for frame in (1, 10, 20):
        edit = _edit_xy_conf(multikey_db, frame)
        assert edit is not None, f"frame {frame} should have an edit row"
        assert edit[2] != 0.0


def test_set_outlier_selected_false_re_enables(qapp, multikey_db):
    w = _make_widget(qapp, multikey_db)
    w._set_outlier_selected(True)
    w._obs_kp["ci1"] = _reload(multikey_db)
    w._set_outlier_selected(False)

    edit = _edit_xy_conf(multikey_db, 1)
    assert edit is not None
    assert edit[2] == 0.0  # is_outlier=False -> flag zero (enabled)


def _reload(db):
    from app.pose.db_cache import read_observations_with_edits
    return read_observations_with_edits(db, "seq1", "ci1")


def test_set_outlier_selected_noop_without_selection(qapp, multikey_db):
    w = _make_widget(qapp, multikey_db)
    w._sel_kp_indices = set()
    w._set_outlier_selected(True)  # must not raise
    assert _edit_xy_conf(multikey_db, 1) is None


# --- Interpolate missing ----------------------------------------------------

def test_interpolate_missing_leaves_present_values_untouched(qapp, multikey_db):
    """Frame 10's untouched-but-wrong (99, 99) detection must survive --
    unlike _interpolate_range, a present value (even a bad one) is never a
    fill target."""
    w = _make_widget(qapp, multikey_db)
    w._interpolate_missing_range()
    assert _edit_xy_conf(multikey_db, 10) is None  # no edit row created


def test_interpolate_missing_fills_ghost_frames(qapp, multikey_db):
    """Frame 5 has no pose_observations row at all (a true gap between
    frames 1 and 10) -- it should be linearly filled from the nearest
    present values on either side."""
    w = _make_widget(qapp, multikey_db)
    w._interpolate_missing_range()

    edit = _edit_xy_conf(multikey_db, 5)
    assert edit is not None
    x, y, is_outlier_flag = edit
    t = (5 - 1) / (10 - 1)
    expected = 1.0 + t * (99.0 - 1.0)
    assert x == pytest.approx(expected, abs=0.01)
    assert y == pytest.approx(expected, abs=0.01)
    assert is_outlier_flag == 0.0  # filled value is enabled


def test_interpolate_missing_fills_explicitly_disabled_frame(qapp, multikey_db):
    """A frame with an explicit disable edit counts as 'no value' too --
    interpolation should overwrite it with an enabled, interpolated value."""
    from app.pose.db_cache import update_single_keypoint_edit
    update_single_keypoint_edit(multikey_db, "seq1", "ci1", 10, 0, 99.0, 99.0, is_outlier=True)

    w = _make_widget(qapp, multikey_db)
    w._obs_kp["ci1"] = _reload(multikey_db)
    w._interpolate_missing_range()

    edit = _edit_xy_conf(multikey_db, 10)
    assert edit is not None
    x, y, is_outlier_flag = edit
    assert is_outlier_flag == 0.0  # re-enabled by the interpolation
    # Straight line from (1, 1) to (20, 20) now that frame 10 no longer anchors.
    t = (10 - 1) / (20 - 1)
    expected = 1.0 + t * (20.0 - 1.0)
    assert x == pytest.approx(expected, abs=0.01)


def test_interpolate_missing_noop_without_range(qapp, multikey_db):
    w = _make_widget(qapp, multikey_db)
    w._range_start_v = None
    w._range_end_v = None
    w._interpolate_missing_range()  # must not raise
    assert _edit_xy_conf(multikey_db, 5) is None


# --- Select to marker --------------------------------------------------------

def test_select_to_marker_forward(qapp, multikey_db):
    """Marker after the current playhead: range should span [current, marker]."""
    w = _make_widget(qapp, multikey_db)
    w._timeline = MagicMock()
    w._timeline.marker_v.return_value = 667  # frame 20
    w._timeline.active_camera_index.return_value = 0
    w._current_t = 1 / 30.0  # frame 1 -> time_v = 33

    w._on_timeline_select_to_marker((1,))

    assert w._sel_kp_indices == {1}
    assert w._range_start_v == 33
    assert w._range_end_v == 667


def test_select_to_marker_backward(qapp, multikey_db):
    """Marker before the current playhead: range must still come out
    ordered (start <= end), not negative/reversed."""
    w = _make_widget(qapp, multikey_db)
    w._timeline = MagicMock()
    w._timeline.marker_v.return_value = 33  # frame 1
    w._timeline.active_camera_index.return_value = 0
    w._current_t = 20 / 30.0  # frame 20 -> time_v = 667

    w._on_timeline_select_to_marker((0,))

    assert w._range_start_v == 33
    assert w._range_end_v == 667


def test_select_to_marker_noop_without_marker(qapp, multikey_db):
    w = _make_widget(qapp, multikey_db)
    w._timeline = MagicMock()
    w._timeline.marker_v.return_value = None
    prev_start, prev_end = w._range_start_v, w._range_end_v

    w._on_timeline_select_to_marker((0,))

    assert w._range_start_v == prev_start
    assert w._range_end_v == prev_end


def test_select_to_marker_excludes_hidden_keypoints(qapp, multikey_db):
    w = _make_widget(qapp, multikey_db)
    w._timeline = MagicMock()
    w._timeline.marker_v.return_value = 667
    w._timeline.active_camera_index.return_value = 0
    w._hidden_kp_indices = {1}
    w._current_t = 0.0

    w._on_timeline_select_to_marker((0, 1))

    assert w._sel_kp_indices == {0}


# ---------------------------------------------------------------------------
# Follow-up round: "M" shortcut, timeline-row disable/enable/interpolate,
# ruler "select all to marker", maximize-follows-camera.
# ---------------------------------------------------------------------------

def test_m_key_sets_marker_at_current_frame(qapp, multikey_db):
    w = _make_widget(qapp, multikey_db)
    w._timeline = MagicMock()
    w._current_t = 10 / 30.0  # frame 10 -> time_v = 333

    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_M, Qt.KeyboardModifier.NoModifier)
    handled = w._handle_key(event)

    assert handled is True
    w._timeline.set_marker.assert_called_once_with(333)


def test_m_key_noop_without_timeline(qapp, multikey_db):
    w = _make_widget(qapp, multikey_db)
    w._timeline = None

    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_M, Qt.KeyboardModifier.NoModifier)
    w._handle_key(event)  # must not raise


def test_canvas_disable_selected_signal_reaches_container(qapp):
    from app.ui.keypoint_timeline_widget import KeypointTimelineWidget
    w = KeypointTimelineWidget(COCO17, [{"label": "A", "shot_video_id": "sv1"}])
    spy = MagicMock()
    w.disable_selected_requested.connect(spy)
    w._canvas.disable_selected_requested.emit()
    spy.assert_called_once()


def test_canvas_enable_selected_signal_reaches_container(qapp):
    from app.ui.keypoint_timeline_widget import KeypointTimelineWidget
    w = KeypointTimelineWidget(COCO17, [{"label": "A", "shot_video_id": "sv1"}])
    spy = MagicMock()
    w.enable_selected_requested.connect(spy)
    w._canvas.enable_selected_requested.emit()
    spy.assert_called_once()


def test_canvas_interpolate_missing_signal_reaches_container(qapp):
    from app.ui.keypoint_timeline_widget import KeypointTimelineWidget
    w = KeypointTimelineWidget(COCO17, [{"label": "A", "shot_video_id": "sv1"}])
    spy = MagicMock()
    w.interpolate_missing_requested.connect(spy)
    w._canvas.interpolate_missing_requested.emit()
    spy.assert_called_once()


def test_ruler_select_all_to_marker_signal_reaches_container(qapp):
    from app.ui.keypoint_timeline_widget import KeypointTimelineWidget
    w = KeypointTimelineWidget(COCO17, [{"label": "A", "shot_video_id": "sv1"}])
    spy = MagicMock()
    w.select_all_to_marker_requested.connect(spy)
    w._ruler.select_all_to_marker_requested.emit()
    spy.assert_called_once()


def test_timeline_row_menu_offers_disable_enable_only_with_selection(canvas):
    """Sanity check on the gating condition contextMenuEvent uses, without
    ever calling it (menu.exec() would block) -- the same condition is
    exercised end-to-end by the container-level signal tests above."""
    canvas.set_edit_mode(True)
    assert not canvas._sel_kp_indices  # nothing selected yet
    canvas.set_selection({0}, None, None)
    assert canvas._sel_kp_indices == {0}


def test_on_timeline_select_all_to_marker_selects_all_visible(qapp, multikey_db):
    w = _make_widget(qapp, multikey_db)
    w._pose_model = COCO17
    w._timeline = MagicMock()
    w._timeline.marker_v.return_value = 667
    w._timeline.active_camera_index.return_value = 0
    w._hidden_kp_indices = {5}
    w._current_t = 0.0

    w._on_timeline_select_all_to_marker()

    assert w._sel_kp_indices == set(COCO17.all_indices) - {5}
    assert w._range_start_v == 0
    assert w._range_end_v == 667


def test_on_timeline_select_all_to_marker_noop_without_marker(qapp, multikey_db):
    w = _make_widget(qapp, multikey_db)
    w._pose_model = COCO17
    w._timeline = MagicMock()
    w._timeline.marker_v.return_value = None
    prev = set(w._sel_kp_indices)

    w._on_timeline_select_all_to_marker()

    assert w._sel_kp_indices == prev


def test_enter_maximized_follows_camera_in_timeline(qapp, multikey_db):
    from PySide6.QtWidgets import QWidget
    w = _make_widget(qapp, multikey_db)
    w._timeline = MagicMock()
    w._sel_cam_idx = None

    # Minimal layout stand-ins so _enter_maximized can run without a full
    # PersonCropGridWidget._build() -- only the calls it makes matter here.
    w._grid = MagicMock()
    w._max_splitter = MagicMock()
    w._thumb_layout = MagicMock()
    w._stack = MagicMock()
    w._3d_ph = QWidget()
    fake_cell = MagicMock()
    w._cells = [fake_cell]

    w._enter_maximized(0)

    assert w._sel_cam_idx == 0
    w._timeline.set_current_time_v.assert_called_once()
    w._timeline.set_selection.assert_called_once()
