"""Tests for Phase 9: copy/paste keypoints (Ctrl+C / Ctrl+V)."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSlider

from tests.app.conftest import _SEQ_DB_N_KP as N_KP


# ---------------------------------------------------------------------------
# Reuse the _make_widget helper from test_phase5 (duplicated to stay self-contained)
# ---------------------------------------------------------------------------

def _make_widget(qapp, seq_db):
    from app.ui.content_panels import PersonCropGridWidget
    from PySide6.QtWidgets import QWidget

    w = PersonCropGridWidget.__new__(PersonCropGridWidget)
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
    w._backfill = None
    w._clipboard = None
    w._clipboard_cam_idx = None
    w._range_start_v = None
    w._range_end_v = None

    from app.pose.db_cache import read_observations_with_edits
    for cam in w._cameras:
        w._obs_kp[cam["camera_instance_id"]] = read_observations_with_edits(
            seq_db, "seq1", cam["camera_instance_id"]
        )

    mock_sync = MagicMock()
    mock_sync.lookup = lambda t, svid: 10 if svid == "sv1" else 20
    w._sync_table = mock_sync

    w._edit_mode = True
    w._sel_kp_indices = {0}
    w._primary_kp_idx = 0
    w._sel_cam_idx = 0  # cam_A / ci1 / sv1

    return w


def _ctrl_key(key: Qt.Key):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent
    return QKeyEvent(
        QEvent.Type.KeyPress, key, Qt.KeyboardModifier.ControlModifier
    )


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

def test_copy_fills_clipboard(qapp, seq_db):
    w = _make_widget(qapp, seq_db)
    orig_x = float(w._obs_kp["ci1"][10][0, 0])

    consumed = w._handle_key(_ctrl_key(Qt.Key.Key_C))

    assert consumed is True
    assert w._clipboard is not None
    assert 0 in w._clipboard
    assert abs(w._clipboard[0][0] - orig_x) < 0.01
    assert w._clipboard_cam_idx == 0


def test_copy_respects_multi_selection(qapp, seq_db):
    w = _make_widget(qapp, seq_db)
    w._sel_kp_indices = {0, 1}
    w._primary_kp_idx = 0

    w._handle_key(_ctrl_key(Qt.Key.Key_C))

    assert set(w._clipboard.keys()) == {0, 1}


def test_copy_noop_without_selection(qapp, seq_db):
    w = _make_widget(qapp, seq_db)
    w._sel_kp_indices = set()
    w._primary_kp_idx = None

    consumed = w._handle_key(_ctrl_key(Qt.Key.Key_C))

    assert consumed is True  # key is consumed regardless
    assert w._clipboard is None


def test_copy_emits_status_message(qapp, seq_db):
    w = _make_widget(qapp, seq_db)
    messages: list[str] = []
    w.status_message.connect(messages.append)

    w._handle_key(_ctrl_key(Qt.Key.Key_C))

    assert len(messages) == 1
    assert "1 keypoint" in messages[0]


# ---------------------------------------------------------------------------
# Paste
# ---------------------------------------------------------------------------

def test_paste_writes_edit_to_db(qapp, seq_db):
    w = _make_widget(qapp, seq_db)
    # First copy kp 0 from frame 10
    w._handle_key(_ctrl_key(Qt.Key.Key_C))
    copied_x = w._clipboard[0][0]

    # Advance to a different "frame" by changing what the mock sync returns
    w._sync_table.lookup = lambda t, svid: 11 if svid == "sv1" else 20

    # Patch _load_frame to avoid full UI repaint
    w._load_frame = MagicMock()

    w._handle_key(_ctrl_key(Qt.Key.Key_V))

    row = seq_db.execute(
        "SELECT kp_blob FROM pose_observation_edits "
        "WHERE sequence_id='seq1' AND camera_instance_id='ci1' AND video_frame=11",
    ).fetchone()
    assert row is not None
    edit_kp = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    assert abs(edit_kp[0, 0] - copied_x) < 0.01


def test_paste_noop_without_clipboard(qapp, seq_db):
    w = _make_widget(qapp, seq_db)
    w._load_frame = MagicMock()

    consumed = w._handle_key(_ctrl_key(Qt.Key.Key_V))

    assert consumed is True
    w._load_frame.assert_not_called()


def test_paste_emits_status_message(qapp, seq_db):
    w = _make_widget(qapp, seq_db)
    w._handle_key(_ctrl_key(Qt.Key.Key_C))

    messages: list[str] = []
    w.status_message.connect(messages.append)
    w._sync_table.lookup = lambda t, svid: 11 if svid == "sv1" else 20
    w._load_frame = MagicMock()

    w._handle_key(_ctrl_key(Qt.Key.Key_V))

    assert any("Pasted" in m for m in messages)


def test_paste_targets_clipboard_camera(qapp, seq_db):
    """Paste always goes to the camera that was active at copy time."""
    w = _make_widget(qapp, seq_db)
    # Copy from cam_A (idx 0)
    w._handle_key(_ctrl_key(Qt.Key.Key_C))
    assert w._clipboard_cam_idx == 0

    # Switch primary camera to cam_B (idx 1) before pasting
    w._sel_cam_idx = 1
    w._sync_table.lookup = lambda t, svid: 11 if svid == "sv1" else 11
    w._load_frame = MagicMock()

    w._handle_key(_ctrl_key(Qt.Key.Key_V))

    # Paste must still land on ci1 (camera 0), not ci2
    row_ci1 = seq_db.execute(
        "SELECT kp_blob FROM pose_observation_edits "
        "WHERE sequence_id='seq1' AND camera_instance_id='ci1' AND video_frame=11",
    ).fetchone()
    row_ci2 = seq_db.execute(
        "SELECT kp_blob FROM pose_observation_edits "
        "WHERE sequence_id='seq1' AND camera_instance_id='ci2' AND video_frame=11",
    ).fetchone()
    assert row_ci1 is not None
    assert row_ci2 is None
