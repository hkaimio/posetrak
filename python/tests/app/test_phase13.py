"""Tests for Phase 13: timeline selection (rubber-band, plain click) and
Ctrl+click keyframe toggle, wired to PersonCropGridWidget's shared
selection/range state.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from app.pose.kp_models import COCO17
from app.pose.timeline_status import STATUS_BLUE, STATUS_GREEN
from app.ui.keypoint_timeline_widget import LABEL_W, ROW_H, _TimelineCanvas

_N_KP = 3


# ---------------------------------------------------------------------------
# _TimelineCanvas mouse interaction (no host widget involved)
# ---------------------------------------------------------------------------

@pytest.fixture()
def canvas(qapp):
    c = _TimelineCanvas(COCO17)
    c.resize(LABEL_W + 200, c.minimumHeight())
    c.set_time_range(0.0, 2.0, "sv1", MagicMock())
    c.set_edit_mode(True)
    return c


def _row0_y() -> int:
    return ROW_H // 2  # inside the first row's band


def test_drag_in_status_area_emits_rubber_band(canvas):
    received = []
    canvas.rubber_band_selected.connect(lambda *args: received.append(args))

    y = _row0_y()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W + 10, y))
    QTest.mouseMove(canvas, pos=QPoint(LABEL_W + 60, y))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W + 60, y))

    assert len(received) == 1
    kp_indices, v0, v1, ctrl = received[0]
    assert kp_indices == set(canvas._rows[0].kp_indices)
    assert v0 < v1
    assert ctrl is False


def test_drag_disabled_outside_edit_mode(canvas):
    canvas.set_edit_mode(False)
    received = []
    canvas.rubber_band_selected.connect(lambda *args: received.append(args))

    y = _row0_y()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W + 10, y))
    QTest.mouseMove(canvas, pos=QPoint(LABEL_W + 60, y))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W + 60, y))

    assert received == []


def test_plain_click_emits_empty_selection_to_clear(canvas):
    """A click with no drag clears the selection (empty kp_indices), not the clicked row —
    see the follow-up fix: replacing the selection with just the clicked row silently
    discarded the rest of a multi-keypoint selection on a stray click."""
    received = []
    canvas.rubber_band_selected.connect(lambda *args: received.append(args))

    y = _row0_y()
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W + 10, y))

    assert len(received) == 1
    kp_indices, v0, v1, ctrl = received[0]
    assert kp_indices == set()
    assert v0 == v1
    assert ctrl is False


def test_ctrl_drag_emits_rubber_band_with_ctrl_true(canvas):
    received = []
    canvas.rubber_band_selected.connect(lambda *args: received.append(args))

    y = _row0_y()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier,
                      QPoint(LABEL_W + 10, y))
    QTest.mouseMove(canvas, pos=QPoint(LABEL_W + 60, y))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier,
                        QPoint(LABEL_W + 60, y))

    assert len(received) == 1
    _kp_indices, _v0, _v1, ctrl = received[0]
    assert ctrl is True


def test_ctrl_click_on_leaf_row_emits_keyframe_toggled(canvas):
    canvas.toggle_group("Face")  # expand to get a leaf row
    leaf_row_idx = next(i for i, r in enumerate(canvas._rows) if r.kind == "leaf")
    y = leaf_row_idx * ROW_H + ROW_H // 2

    received = []
    canvas.keyframe_toggled.connect(lambda *args: received.append(args))
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier,
                      QPoint(LABEL_W + 10, y))

    assert len(received) == 1
    kp_idx, _v = received[0]
    assert kp_idx == canvas._rows[leaf_row_idx].kp_indices[0]


def test_ctrl_click_on_group_row_does_not_emit_keyframe(canvas):
    """Keyframes are per-keypoint; a group row (multiple kp) can't be a keyframe target."""
    assert canvas._rows[0].kind == "group"
    received = []
    canvas.keyframe_toggled.connect(lambda *args: received.append(args))

    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier,
                      QPoint(LABEL_W + 10, _row0_y()))

    assert received == []


def test_click_on_group_label_still_toggles_in_edit_mode(canvas):
    """Group expand/collapse is navigation, not an edit gesture — always available."""
    assert canvas._expanded == set()
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(10, _row0_y()))
    assert canvas._rows[0].label in canvas._expanded


def test_drag_starting_on_group_label_does_not_emit_rubber_band(canvas):
    received = []
    canvas.rubber_band_selected.connect(lambda *args: received.append(args))

    y = _row0_y()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(10, y))
    QTest.mouseMove(canvas, pos=QPoint(LABEL_W + 60, y))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(LABEL_W + 60, y))

    assert received == []
    # It did toggle the group though (label click handled first).
    assert canvas._rows[0].label in canvas._expanded


# ---------------------------------------------------------------------------
# PersonCropGridWidget wiring: _on_timeline_rubber_band / _on_timeline_keyframe_toggle
# ---------------------------------------------------------------------------

def _enc(rows: list[tuple[float, float, float]]) -> bytes:
    return np.array(rows, dtype=np.float32).tobytes()


@pytest.fixture()
def kf_db(tmp_path):
    from posetrak.db.db import create_session
    conn = create_session(tmp_path / "kf.db")
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
    conn.execute(
        "INSERT INTO pose_observations"
        " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, kp_blob)"
        " VALUES ('seq1', 'ci1', 4, ?, 0, ?)",
        (4 / 30.0, _enc([(4.0, 5.0, 0.9)] * _N_KP)),
    )
    conn.commit()
    yield conn
    conn.close()


def _make_widget(db):
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
    w._current_t = 4 / 30.0
    w._edit_mode = True
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
    w._load_frame = MagicMock()
    w._hand_redetect = None
    w._hand_redetect_timers = {}

    mock_sync = MagicMock()
    mock_sync.lookup = lambda t, svid: round(t * 30)
    w._sync_table = mock_sync

    w._timeline = MagicMock()
    w._timeline.active_camera_index.return_value = 0

    from app.pose.db_cache import read_observations_with_edits
    w._obs_kp["ci1"] = read_observations_with_edits(db, "seq1", "ci1")

    return w


def test_rubber_band_replaces_selection(qapp, kf_db):
    w = _make_widget(kf_db)
    w._sel_kp_indices = {2}

    w._on_timeline_rubber_band({0, 1}, 100, 200, ctrl=False)

    assert w._sel_kp_indices == {0, 1}
    assert w._primary_kp_idx in {0, 1}
    assert w._sel_cam_idx == 0
    assert w._range_start_v == 100
    assert w._range_end_v == 200
    w._load_frame.assert_called_once()


def test_rubber_band_ctrl_adds_to_selection(qapp, kf_db):
    w = _make_widget(kf_db)
    w._sel_kp_indices = {2}

    w._on_timeline_rubber_band({0, 1}, 100, 200, ctrl=True)

    assert w._sel_kp_indices == {0, 1, 2}


def test_degenerate_range_does_not_clear_existing_range(qapp, kf_db):
    """A plain click (v0 == v1) must not wipe out an active drag-selected range."""
    w = _make_widget(kf_db)
    w._range_start_v = 50
    w._range_end_v = 150

    w._on_timeline_rubber_band({0}, 300, 300, ctrl=False)

    assert w._range_start_v == 50
    assert w._range_end_v == 150


def test_rubber_band_empty_selection_clears_primary(qapp, kf_db):
    w = _make_widget(kf_db)
    w._sel_kp_indices = {0}
    w._primary_kp_idx = 0

    w._on_timeline_rubber_band(set(), 100, 100, ctrl=False)

    assert w._sel_kp_indices == set()
    assert w._primary_kp_idx is None


def test_keyframe_toggle_freezes_current_position(qapp, kf_db):
    w = _make_widget(kf_db)
    # Frame 4 is not a keyframe yet (no edit row) → status is None/absent.
    time_v = int(round((4 / 30.0 - w._t_start) * 1000))

    w._on_timeline_keyframe_toggle(kp_idx=1, time_v=time_v)

    row = kf_db.execute(
        "SELECT kp_blob, kp_mask FROM pose_observation_edits"
        " WHERE sequence_id='seq1' AND camera_instance_id='ci1' AND video_frame=4"
    ).fetchone()
    assert row is not None
    kp = np.frombuffer(bytes(row["kp_mask"]), dtype=np.uint8)
    assert (kp[0] >> 1) & 1  # bit for kp_idx=1 set
    edit_kp = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    assert abs(edit_kp[1, 0] - 4.0) < 1e-4
    assert abs(edit_kp[1, 1] - 5.0) < 1e-4
    assert edit_kp[1, 2] == 0.0  # is_outlier = False
    w._load_frame.assert_called_once()


def test_keyframe_toggle_removes_existing_keyframe(qapp, kf_db):
    from app.pose.db_cache import update_single_keypoint_edit
    update_single_keypoint_edit(kf_db, "seq1", "ci1", 4, kp_idx=1, new_x=4.0, new_y=5.0, is_outlier=False)

    w = _make_widget(kf_db)
    status = np.full(_N_KP, STATUS_GREEN, dtype=np.int8)
    status[1] = STATUS_BLUE
    w._timeline_status_by_cam["ci1"] = {4: status}

    time_v = int(round((4 / 30.0 - w._t_start) * 1000))
    w._on_timeline_keyframe_toggle(kp_idx=1, time_v=time_v)

    row = kf_db.execute(
        "SELECT * FROM pose_observation_edits"
        " WHERE sequence_id='seq1' AND camera_instance_id='ci1' AND video_frame=4"
    ).fetchone()
    assert row is None


def test_keyframe_toggle_noop_without_observation(qapp, kf_db):
    w = _make_widget(kf_db)
    # video_frame 99 has no pose_observations row and no edit → nothing to freeze.
    time_v = int(round((99 / 30.0 - w._t_start) * 1000))

    w._on_timeline_keyframe_toggle(kp_idx=0, time_v=time_v)

    row = kf_db.execute("SELECT * FROM pose_observation_edits").fetchone()
    assert row is None
    w._load_frame.assert_not_called()
