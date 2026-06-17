"""Tests for Phase 5 keyboard shortcuts in PersonCropGridWidget (content_panels.py)."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QSlider

from tests.app.conftest import _SEQ_DB_N_KP as N_KP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_widget(qapp, seq_db):
    """Create a PersonCropGridWidget with manually-injected state (no _build() DB)."""
    from app.ui.content_panels import PersonCropGridWidget

    w = PersonCropGridWidget.__new__(PersonCropGridWidget)
    # Minimal QWidget init
    from PySide6.QtWidgets import QWidget
    QWidget.__init__(w)
    w._conn = seq_db
    w._sequence_id = "seq1"
    w._cells = []
    w._cameras = [
        {"shot_video_id": "sv1", "camera_instance_id": "ci1", "label": "cam_A"},
        {"shot_video_id": "sv2", "camera_instance_id": "ci2", "label": "cam_B"},
    ]
    w._sync_table = None
    w._det_run_id = None
    w._t_start = 0.0
    w._t_end = 2.0
    w._current_t = 0.083
    w._slider = QSlider(Qt.Orientation.Horizontal)
    w._slider.setMinimum(0)
    w._slider.setMaximum(2000)
    w._slider.setSingleStep(33)
    w._slider.setValue(83)
    w._time_label = None
    w._show_detected = None
    w._show_tracked = None
    w._show_seg = None
    w._edit_btn = None
    w._edit_mode = False
    w._sel_kp_indices = set()
    w._primary_kp_idx = None
    w._sel_cam_idx = None
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
    w._ncols = 2
    w._grid = None
    w._clipboard = None
    w._clipboard_cam_idx = None
    w._range_start_v = None
    w._range_end_v = None

    # Pre-load obs_kp from DB
    from app.pose.db_cache import read_observations_with_edits
    for cam in w._cameras:
        w._obs_kp[cam["camera_instance_id"]] = read_observations_with_edits(
            seq_db, "seq1", cam["camera_instance_id"]
        )

    # Inject a mock sync_table that maps current_t to frame 10 for sv1
    mock_sync = MagicMock()
    mock_sync.lookup = lambda t, svid: 10 if svid == "sv1" else 10
    w._sync_table = mock_sync

    w._edit_mode = True
    w._sel_kp_indices = {0}
    w._primary_kp_idx = 0
    w._sel_cam_idx = 0  # ci1 / sv1

    return w


# ---------------------------------------------------------------------------
# A / D frame navigation
# ---------------------------------------------------------------------------

def test_key_a_steps_slider_back(qapp, seq_db):
    w = _make_widget(qapp, seq_db)
    initial = w._slider.value()
    step = w._slider.singleStep()
    consumed = w._handle_key(_make_key_event(Qt.Key.Key_A))
    assert consumed is True
    assert w._slider.value() == initial - step


def test_key_d_steps_slider_forward(qapp, seq_db):
    w = _make_widget(qapp, seq_db)
    initial = w._slider.value()
    step = w._slider.singleStep()
    consumed = w._handle_key(_make_key_event(Qt.Key.Key_D))
    assert consumed is True
    assert w._slider.value() == initial + step


def test_key_a_works_outside_edit_mode(qapp, seq_db):
    """A/D frame navigation must work regardless of edit mode."""
    w = _make_widget(qapp, seq_db)
    w._edit_mode = False
    initial = w._slider.value()
    consumed = w._handle_key(_make_key_event(Qt.Key.Key_A))
    assert consumed is True
    assert w._slider.value() == initial - w._slider.singleStep()


# ---------------------------------------------------------------------------
# Esc: deselect
# ---------------------------------------------------------------------------

def test_escape_clears_selection(qapp, seq_db):
    w = _make_widget(qapp, seq_db)
    consumed = w._handle_key(_make_key_event(Qt.Key.Key_Escape))
    assert consumed is True
    assert w._sel_kp_indices == set()
    assert w._primary_kp_idx is None
    assert w._sel_cam_idx is None


def test_escape_ignored_outside_edit_mode(qapp, seq_db):
    w = _make_widget(qapp, seq_db)
    w._edit_mode = False
    consumed = w._handle_key(_make_key_event(Qt.Key.Key_Escape))
    assert consumed is False


# ---------------------------------------------------------------------------
# Arrow nudge
# ---------------------------------------------------------------------------

def test_nudge_right_writes_edit(qapp, seq_db):
    w = _make_widget(qapp, seq_db)
    orig_x = float(w._obs_kp["ci1"][10][0, 0])

    consumed = w._handle_key(_make_key_event(Qt.Key.Key_Right))
    assert consumed is True

    row = seq_db.execute(
        "SELECT kp_blob FROM pose_observation_edits"
        " WHERE sequence_id='seq1' AND camera_instance_id='ci1' AND video_frame=10",
    ).fetchone()
    assert row is not None
    edit_kp = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    assert abs(edit_kp[0, 0] - (orig_x + 1.0)) < 0.01


def test_nudge_up_writes_edit(qapp, seq_db):
    w = _make_widget(qapp, seq_db)
    orig_y = float(w._obs_kp["ci1"][10][0, 1])

    w._handle_key(_make_key_event(Qt.Key.Key_Up))

    row = seq_db.execute(
        "SELECT kp_blob FROM pose_observation_edits"
        " WHERE sequence_id='seq1' AND camera_instance_id='ci1' AND video_frame=10",
    ).fetchone()
    edit_kp = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    assert abs(edit_kp[0, 1] - (orig_y - 1.0)) < 0.01


def test_nudge_ignored_without_selection(qapp, seq_db):
    w = _make_widget(qapp, seq_db)
    w._sel_kp_indices = set()
    w._primary_kp_idx = None
    w._sel_cam_idx = None
    consumed = w._handle_key(_make_key_event(Qt.Key.Key_Right))
    assert consumed is False


# ---------------------------------------------------------------------------
# Space: toggle outlier
# ---------------------------------------------------------------------------

def test_space_marks_outlier(qapp, seq_db):
    w = _make_widget(qapp, seq_db)
    # kp[0] has conf=0.9, so not outlier initially
    assert float(w._obs_kp["ci1"][10][0, 2]) > 0.01

    w._handle_key(_make_key_event(Qt.Key.Key_Space))

    # After toggle, should be marked outlier (conf→0)
    merged = w._obs_kp["ci1"][10]
    assert float(merged[0, 2]) < 0.01


def test_space_unmarks_outlier(qapp, seq_db):
    w = _make_widget(qapp, seq_db)
    # First toggle: mark outlier
    w._handle_key(_make_key_event(Qt.Key.Key_Space))
    assert float(w._obs_kp["ci1"][10][0, 2]) < 0.01
    # Second toggle: un-mark outlier
    w._handle_key(_make_key_event(Qt.Key.Key_Space))
    assert float(w._obs_kp["ci1"][10][0, 2]) > 0.01


# ---------------------------------------------------------------------------
# Helper: synthesise a key event
# ---------------------------------------------------------------------------

def _make_key_event(key: Qt.Key):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent
    return QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
