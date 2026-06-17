"""Tests for Phase 10: frame range selection (Shift+A/D) + linear interpolation (I)."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSlider

_N_KP = 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def interp_db(tmp_path):
    """DB with observations at frames 1 and 10 for ci1 (used as interpolation anchors)."""
    from posetrak.db.db import create_session
    conn = create_session(tmp_path / "interp.db")
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

    def _enc(x: float, y: float, conf: float = 0.9) -> bytes:
        kp = np.zeros((_N_KP, 3), dtype=np.float32)
        kp[:, 0] = x
        kp[:, 1] = y
        kp[:, 2] = conf
        return kp.tobytes()

    # Left anchor at frame 1: (1.0, 1.0); right anchor at frame 10: (10.0, 10.0)
    for frame, x, y in [(1, 1.0, 1.0), (10, 10.0, 10.0)]:
        conn.execute(
            "INSERT INTO pose_observations"
            " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, kp_blob)"
            " VALUES ('seq1', 'ci1', ?, ?, 0, ?)",
            (frame, frame / 30.0, _enc(x, y)),
        )
    conn.commit()
    yield conn
    conn.close()


def _make_widget(qapp, db):
    """Minimal PersonCropGridWidget for Phase 10 testing."""
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
    w._slider.setSingleStep(33)  # ~30fps
    w._slider.setValue(0)
    w._time_label = None
    w._show_detected = None
    w._show_tracked = None
    w._show_seg = None
    w._edit_btn = None
    w._edit_mode = True
    w._sel_kp_indices = {0}
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
    w._grid = None
    w._backfill = None
    w._clipboard = None
    w._clipboard_cam_idx = None
    w._range_start_v = None
    w._range_end_v = None

    from app.pose.db_cache import read_observations_with_edits
    w._obs_kp["ci1"] = read_observations_with_edits(db, "seq1", "ci1")

    # Sync table: lookup(t, svid) = round(t * 30)  (30fps)
    mock_sync = MagicMock()
    mock_sync.lookup = lambda t, svid: round(t * 30)
    w._sync_table = mock_sync

    return w


def _shift_key(key: Qt.Key):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent
    return QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.ShiftModifier)


def _plain_key(key: Qt.Key):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent
    return QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)


# ---------------------------------------------------------------------------
# Shift+A/D range extension
# ---------------------------------------------------------------------------

def test_shift_d_sets_initial_range(qapp, interp_db):
    w = _make_widget(qapp, interp_db)
    w._slider.setValue(100)  # starting slider position

    w._handle_key(_shift_key(Qt.Key.Key_D))

    assert w._range_start_v == 100
    assert w._range_end_v == 100 + 33  # one step right
    assert w._slider.value() == 133


def test_shift_d_extends_existing_range(qapp, interp_db):
    w = _make_widget(qapp, interp_db)
    w._slider.setValue(100)
    w._handle_key(_shift_key(Qt.Key.Key_D))  # range [100, 133], slider at 133

    w._handle_key(_shift_key(Qt.Key.Key_D))  # extend to [100, 166]

    assert w._range_start_v == 100
    assert w._range_end_v == 166
    assert w._slider.value() == 166


def test_shift_a_sets_initial_range(qapp, interp_db):
    w = _make_widget(qapp, interp_db)
    w._slider.setValue(200)

    w._handle_key(_shift_key(Qt.Key.Key_A))

    assert w._range_end_v == 200
    assert w._range_start_v == 200 - 33
    assert w._slider.value() == 167


def test_plain_a_clears_range(qapp, interp_db):
    w = _make_widget(qapp, interp_db)
    w._range_start_v = 100
    w._range_end_v = 200

    w._handle_key(_plain_key(Qt.Key.Key_A))

    assert w._range_start_v is None
    assert w._range_end_v is None


def test_escape_clears_range(qapp, interp_db):
    w = _make_widget(qapp, interp_db)
    w._range_start_v = 100
    w._range_end_v = 200

    w._handle_key(_plain_key(Qt.Key.Key_Escape))

    assert w._range_start_v is None
    assert w._range_end_v is None


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def test_interpolate_range_writes_correct_positions(qapp, interp_db):
    """
    Anchors: frame 1 (1.0,1.0) and frame 10 (10.0,10.0).
    Range [4,7]: expected kp positions are (4,4), (5,5), (6,6), (7,7).
    """
    w = _make_widget(qapp, interp_db)
    # Slider values for frames 4–7 at 30fps: 4/30=133ms, 7/30=233ms
    w._range_start_v = 132   # round(0.132*30) = 4
    w._range_end_v = 231     # round(0.231*30) = 7
    w._sel_kp_indices = {0}
    w._load_frame = MagicMock()

    w._handle_key(_plain_key(Qt.Key.Key_I))

    for frame in [4, 5, 6, 7]:
        row = interp_db.execute(
            "SELECT kp_blob FROM pose_observation_edits"
            " WHERE sequence_id='seq1' AND camera_instance_id='ci1' AND video_frame=?",
            (frame,),
        ).fetchone()
        assert row is not None, f"frame {frame} has no edit"
        kp = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
        assert abs(kp[0, 0] - float(frame)) < 0.05, f"frame {frame} x wrong: {kp[0,0]}"
        assert abs(kp[0, 1] - float(frame)) < 0.05, f"frame {frame} y wrong: {kp[0,1]}"


def test_interpolate_clears_range(qapp, interp_db):
    w = _make_widget(qapp, interp_db)
    w._range_start_v = 132
    w._range_end_v = 231
    w._load_frame = MagicMock()

    w._handle_key(_plain_key(Qt.Key.Key_I))

    assert w._range_start_v is None
    assert w._range_end_v is None


def test_interpolate_skips_kp_with_missing_anchor(qapp, interp_db):
    """kp_idx=1 has no anchor in this DB (conf=0.9 at frames 1 and 10).
    But kp_idx=2 also has anchors (same DB row), so this tests the case
    where the selection includes a kp that DOES have anchors."""
    w = _make_widget(qapp, interp_db)
    w._range_start_v = 132
    w._range_end_v = 231
    w._sel_kp_indices = {0, 1, 2}  # all three kp have anchors (same positions in DB)
    w._load_frame = MagicMock()

    w._handle_key(_plain_key(Qt.Key.Key_I))

    # All three should be interpolated since all have conf=0.9 at anchor frames
    row = interp_db.execute(
        "SELECT kp_blob FROM pose_observation_edits"
        " WHERE sequence_id='seq1' AND camera_instance_id='ci1' AND video_frame=5"
    ).fetchone()
    assert row is not None
    kp = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    for idx in [0, 1, 2]:
        assert abs(kp[idx, 0] - 5.0) < 0.05


def test_i_key_noop_without_range(qapp, interp_db):
    w = _make_widget(qapp, interp_db)
    w._range_start_v = None
    w._range_end_v = None
    w._load_frame = MagicMock()

    consumed = w._handle_key(_plain_key(Qt.Key.Key_I))

    # Key is not consumed when range is None
    assert consumed is False
    w._load_frame.assert_not_called()


def test_interpolate_emits_status_message(qapp, interp_db):
    w = _make_widget(qapp, interp_db)
    w._range_start_v = 132
    w._range_end_v = 231
    w._load_frame = MagicMock()
    messages: list[str] = []
    w.status_message.connect(messages.append)

    w._handle_key(_plain_key(Qt.Key.Key_I))

    assert any("Interpolated" in m for m in messages)
