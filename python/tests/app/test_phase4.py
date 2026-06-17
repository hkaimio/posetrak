"""Tests for Phase 4 mouse interaction: hit-test, signals, drag-to-move, DB write."""
from __future__ import annotations

import math

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest

from app.pose.db_cache import update_single_keypoint_edit, write_observation_edit
from tests.app.conftest import _SEQ_DB_N_KP as N_KP

# Cell geometry used throughout
_CELL_W = 320
_CELL_H = 240
_SRC_X, _SRC_Y, _SRC_W, _SRC_H = 50, 30, 200, 150


def _make_kp(x: float = 150.0, y: float = 105.0, conf: float = 0.9) -> np.ndarray:
    """Return float32[N_KP, 3] with kp_idx=0 at (x, y, conf), rest far off-screen."""
    kp = np.zeros((N_KP, 3), dtype=np.float32)
    kp[0] = [x, y, conf]
    # Place others outside crop so they don't interfere with hit-tests
    kp[1:] = [-999, -999, 0.9]
    return kp


def _make_cell(qapp, kp: np.ndarray | None = None):
    """Create a _CropCellWidget with known src rect and optional kp."""
    from app.pose.crop_editor import _CropCellWidget
    cell = _CropCellWidget("cam")
    cell.update_frame(None, _SRC_X, _SRC_Y, _SRC_W, _SRC_H, kp)
    return cell


def _kp_display_pos(x_full: float, y_full: float) -> tuple[float, float]:
    """Convert full-frame coords to the display position in the cell."""
    dx = (x_full - _SRC_X) * _CELL_W / _SRC_W
    dy = (y_full - _SRC_Y) * _CELL_H / _SRC_H
    return dx, dy


# ---------------------------------------------------------------------------
# Hit-test
# ---------------------------------------------------------------------------

def test_hit_kp_returns_correct_index(qapp):
    kp = _make_kp(x=150.0, y=105.0)
    cell = _make_cell(qapp, kp)
    dx, dy = _kp_display_pos(150.0, 105.0)
    assert cell._hit_kp(dx, dy) == 0


def test_hit_kp_accepts_nearby_point(qapp):
    """A click 3 px away from the dot centre should still hit."""
    kp = _make_kp(x=150.0, y=105.0)
    cell = _make_cell(qapp, kp)
    dx, dy = _kp_display_pos(150.0, 105.0)
    assert cell._hit_kp(dx + 3, dy) == 0


def test_hit_kp_returns_none_for_miss(qapp):
    kp = _make_kp(x=150.0, y=105.0)
    cell = _make_cell(qapp, kp)
    dx, dy = _kp_display_pos(150.0, 105.0)
    # Move well past the hit radius
    from app.pose.crop_editor import _HIT_RADIUS
    assert cell._hit_kp(dx + _HIT_RADIUS + 5, dy) is None


def test_hit_kp_returns_none_when_no_kp(qapp):
    cell = _make_cell(qapp, kp=None)
    assert cell._hit_kp(100.0, 100.0) is None


def test_display_to_full_roundtrip(qapp):
    cell = _make_cell(qapp)
    x, y = cell._display_to_full(160.0, 120.0)  # centre of display
    # centre of display = src_x + src_w/2, src_y + src_h/2
    assert abs(x - (_SRC_X + _SRC_W / 2)) < 0.5
    assert abs(y - (_SRC_Y + _SRC_H / 2)) < 0.5


# ---------------------------------------------------------------------------
# Signal emission
# ---------------------------------------------------------------------------

def test_keypoint_selected_signal_on_click(qapp):
    kp = _make_kp(x=150.0, y=105.0)
    cell = _make_cell(qapp, kp)
    received: list[int] = []
    cell.keypoint_selected.connect(received.append)
    dx, dy = _kp_display_pos(150.0, 105.0)
    QTest.mouseClick(cell, Qt.MouseButton.LeftButton, pos=QPoint(int(dx), int(dy)))
    assert received == [0]


def test_keypoint_deselected_signal_on_empty_click(qapp):
    kp = _make_kp(x=150.0, y=105.0)
    cell = _make_cell(qapp, kp)
    fired: list[bool] = []
    cell.keypoint_deselected.connect(lambda: fired.append(True))
    # Click far from any dot
    QTest.mouseClick(cell, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    assert fired == [True]


def test_keypoint_moved_signal_on_drag(qapp):
    """Dragging a dot by ≥ threshold emits keypoint_moved with correct new coords."""
    kp = _make_kp(x=150.0, y=105.0)
    cell = _make_cell(qapp, kp)
    moves: list[tuple] = []
    cell.keypoint_moved.connect(lambda idx, x, y: moves.append((idx, x, y)))

    dx, dy = _kp_display_pos(150.0, 105.0)
    # Drag 20 px to the right in display space
    QTest.mousePress(cell, Qt.MouseButton.LeftButton, pos=QPoint(int(dx), int(dy)))
    QTest.mouseMove(cell, pos=QPoint(int(dx) + 20, int(dy)))
    QTest.mouseRelease(cell, Qt.MouseButton.LeftButton, pos=QPoint(int(dx) + 20, int(dy)))

    assert len(moves) == 1
    kp_idx, new_x, new_y = moves[0]
    assert kp_idx == 0
    expected_delta_x = 20 * _SRC_W / _CELL_W
    assert abs(new_x - (150.0 + expected_delta_x)) < 1.0
    assert abs(new_y - 105.0) < 1.0


def test_no_keypoint_moved_signal_on_tiny_drag(qapp):
    """A drag smaller than _DRAG_THRESHOLD should not emit keypoint_moved."""
    kp = _make_kp(x=150.0, y=105.0)
    cell = _make_cell(qapp, kp)
    moves: list = []
    cell.keypoint_moved.connect(lambda idx, x, y: moves.append((idx, x, y)))

    dx, dy = _kp_display_pos(150.0, 105.0)
    QTest.mousePress(cell, Qt.MouseButton.LeftButton, pos=QPoint(int(dx), int(dy)))
    QTest.mouseMove(cell, pos=QPoint(int(dx) + 2, int(dy)))  # 2 px < threshold
    QTest.mouseRelease(cell, Qt.MouseButton.LeftButton, pos=QPoint(int(dx) + 2, int(dy)))

    assert moves == []


# ---------------------------------------------------------------------------
# DB helper: update_single_keypoint_edit
# ---------------------------------------------------------------------------

def _insert_obs(conn, frame: int, kp: np.ndarray) -> None:
    conn.execute(
        "INSERT INTO pose_observations"
        " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, kp_blob)"
        " VALUES ('seq1', 'ci1', ?, ?, 0, ?)",
        (frame, float(frame) * 0.033, kp.tobytes()),
    )
    conn.commit()


def test_update_single_kp_edit_creates_new_row(seq_db):
    """update_single_keypoint_edit creates a new edit row when none exists."""
    video_frame = 10  # inserted by seq_db fixture
    update_single_keypoint_edit(seq_db, "seq1", "ci1", video_frame, 0, 99.0, 77.0)
    row = seq_db.execute(
        "SELECT kp_blob, kp_mask FROM pose_observation_edits"
        " WHERE sequence_id='seq1' AND camera_instance_id='ci1' AND video_frame=?",
        (video_frame,),
    ).fetchone()
    assert row is not None
    edit_kp = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    mask = bytes(row["kp_mask"])
    assert abs(edit_kp[0, 0] - 99.0) < 0.01
    assert abs(edit_kp[0, 1] - 77.0) < 0.01
    assert edit_kp[0, 2] == 0.0  # is_outlier=False
    assert mask[0] & 1  # bit 0 set


def test_outlier_toggle_preserves_moved_position(seq_db):
    """Moving a kp then marking it as outlier must preserve the moved x/y in the merge."""
    from app.pose.db_cache import read_observations_with_edits
    video_frame = 10
    update_single_keypoint_edit(seq_db, "seq1", "ci1", video_frame, 0, 200.0, 150.0, is_outlier=False)
    update_single_keypoint_edit(seq_db, "seq1", "ci1", video_frame, 0, 200.0, 150.0, is_outlier=True)
    merged = read_observations_with_edits(seq_db, "seq1", "ci1")
    kp = merged[video_frame]
    assert abs(kp[0, 0] - 200.0) < 0.01  # moved position preserved
    assert abs(kp[0, 1] - 150.0) < 0.01
    assert kp[0, 2] < 0.01                # outlier flag applied


def test_update_single_kp_edit_merges_with_existing(seq_db):
    """A second call updates only the new slot while keeping the first slot's edit."""
    video_frame = 10
    # Write an edit for slot 0
    update_single_keypoint_edit(seq_db, "seq1", "ci1", video_frame, 0, 10.0, 20.0)
    # Write another edit for slot 1 (without touching slot 0)
    update_single_keypoint_edit(seq_db, "seq1", "ci1", video_frame, 1, 30.0, 40.0)

    row = seq_db.execute(
        "SELECT kp_blob, kp_mask FROM pose_observation_edits"
        " WHERE sequence_id='seq1' AND camera_instance_id='ci1' AND video_frame=?",
        (video_frame,),
    ).fetchone()
    edit_kp = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    mask = bytes(row["kp_mask"])

    assert abs(edit_kp[0, 0] - 10.0) < 0.01  # slot 0 preserved
    assert abs(edit_kp[1, 0] - 30.0) < 0.01  # slot 1 updated
    assert mask[0] & 1          # bit 0 set
    assert mask[0] & (1 << 1)   # bit 1 set


def test_update_single_kp_edit_ghost_frame(seq_db):
    """update_single_keypoint_edit works on a frame with no pose_observations row."""
    ghost_frame = 99  # not in seq_db fixture
    update_single_keypoint_edit(seq_db, "seq1", "ci1", ghost_frame, 0, 123.0, 456.0)

    row = seq_db.execute(
        "SELECT kp_blob, kp_mask FROM pose_observation_edits"
        " WHERE sequence_id='seq1' AND camera_instance_id='ci1' AND video_frame=?",
        (ghost_frame,),
    ).fetchone()
    assert row is not None
    edit_kp = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    assert abs(edit_kp[0, 0] - 123.0) < 0.01
    assert abs(edit_kp[0, 1] - 456.0) < 0.01
    mask = bytes(row["kp_mask"])
    assert mask[0] & 1  # bit 0 set


def test_read_observations_with_edits_includes_ghost_frame(seq_db):
    """Ghost-frame edits (no pose_observations row) appear in the merged result."""
    from app.pose.db_cache import read_observations_with_edits
    ghost_frame = 99
    update_single_keypoint_edit(seq_db, "seq1", "ci1", ghost_frame, 1, 77.0, 88.0)

    merged = read_observations_with_edits(seq_db, "seq1", "ci1")
    assert ghost_frame in merged
    kp = merged[ghost_frame]
    assert abs(kp[1, 0] - 77.0) < 0.01
    assert abs(kp[1, 1] - 88.0) < 0.01
    assert abs(kp[1, 2] - 1.0) < 0.01   # manually placed → full confidence
    # Unedited slots stay at zero
    assert abs(kp[0, 2]) < 0.01


# ---------------------------------------------------------------------------
# Integration: PersonCropGridWidget drag writes to DB
# ---------------------------------------------------------------------------

def test_kp_moved_writes_to_db_and_refreshes(qapp, seq_db):
    """Dragging a keypoint via PersonCropGridWidget writes an edit and refreshes kp_by_frame."""
    from app.pose.crop_editor import PersonCropGridWidget
    w = PersonCropGridWidget()
    w.load_sequence(seq_db, "seq1")

    cam_ci1 = next(c for c in w._cameras if c.camera_instance_id == "ci1")
    # Simulate a move for kp_idx 0 at frame 10
    w._on_cell_kp_moved(cam_ci1, 0, 55.0, 66.0)

    # DB row should exist
    row = seq_db.execute(
        "SELECT kp_blob, kp_mask FROM pose_observation_edits"
        " WHERE sequence_id='seq1' AND camera_instance_id='ci1' AND video_frame=10",
    ).fetchone()
    assert row is not None
    edit_kp = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    assert abs(edit_kp[0, 0] - 55.0) < 0.01

    # In-memory kp cache should reflect the edit (conf → 1.0 for the moved kp)
    merged = cam_ci1.kp_by_frame[10]
    assert abs(merged[0, 0] - 55.0) < 0.01
    assert abs(merged[0, 2] - 1.0) < 0.01
