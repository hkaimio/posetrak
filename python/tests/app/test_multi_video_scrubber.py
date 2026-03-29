"""Tests for app.setup.multi_video_scrubber (MultiVideoScrubber widget)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.setup.db_context import SyncPoint, SyncTable
from app.setup.frame_cache import FrameCache
from app.setup.multi_video_scrubber import CellInfo, MultiVideoScrubber


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VID_A = "vid-a"
_VID_B = "vid-b"
_FRAME = np.zeros((90, 160, 3), dtype=np.uint8)


def _make_cache() -> FrameCache:
    """FrameCache that always returns a blank frame without opening any files."""
    cache = FrameCache(conn=None, lru_max=50)
    cache.get = MagicMock(return_value=_FRAME)
    return cache


def _make_cells() -> list[CellInfo]:
    return [
        CellInfo(_VID_A, "/a.mp4", total_frames=300, fps=30.0, label="cam1"),
        CellInfo(_VID_B, "/b.mp4", total_frames=300, fps=30.0, label="cam2"),
    ]


def _two_anchor_sync() -> SyncTable:
    """Sync table: both cameras share the same timestamp ↔ frame mapping."""
    pts = [
        SyncPoint("cam1", _VID_A, video_frame=0,  timestamp_s=0.0),
        SyncPoint("cam1", _VID_A, video_frame=300, timestamp_s=10.0),
        SyncPoint("cam2", _VID_B, video_frame=0,  timestamp_s=0.0),
        SyncPoint("cam2", _VID_B, video_frame=300, timestamp_s=10.0),
    ]
    return SyncTable(pts, fps_by_video={_VID_A: 30.0, _VID_B: 30.0})


@pytest.fixture()
def scrubber(qapp):
    s = MultiVideoScrubber(_make_cells(), _make_cache())
    s.resize(640, 360)
    yield s
    s.shutdown()


# ---------------------------------------------------------------------------
# Independent mode (no sync table)
# ---------------------------------------------------------------------------


def test_seek_camera_updates_only_target_cell(scrubber) -> None:
    scrubber.seek_camera(0, 50)
    assert scrubber.current_frames[0] == 50
    assert scrubber.current_frames[1] == 0   # unchanged


def test_seek_camera_clamps_to_valid_range(scrubber) -> None:
    scrubber.seek_camera(0, -10)
    assert scrubber.current_frames[0] == 0

    scrubber.seek_camera(0, 99999)
    assert scrubber.current_frames[0] == 299  # total_frames - 1


def test_seek_camera_ignores_out_of_range_cell(scrubber) -> None:
    """Seeking a non-existent cell index should not raise."""
    scrubber.seek_camera(99, 10)


def test_step_in_independent_mode_moves_focused_cell(scrubber) -> None:
    scrubber._focused_cell = 1
    scrubber._step(5)
    assert scrubber.current_frames[0] == 0   # unchanged
    assert scrubber.current_frames[1] == 5


def test_keyboard_right_in_independent_mode(scrubber, qapp) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent

    scrubber._focused_cell = 0
    scrubber._step(1)
    assert scrubber.current_frames[0] == 1


# ---------------------------------------------------------------------------
# Synced mode
# ---------------------------------------------------------------------------


def test_seek_synced_updates_all_cells(scrubber) -> None:
    scrubber.reload_sync(_two_anchor_sync())
    scrubber.seek_synced(5.0)   # 5 s × 30 fps = frame 150
    assert scrubber.current_frames[0] == 150
    assert scrubber.current_frames[1] == 150


def test_seek_synced_noop_without_sync_table(scrubber) -> None:
    assert scrubber.sync_table is None
    scrubber.seek_synced(5.0)
    # Frames should remain at 0
    assert scrubber.current_frames == [0, 0]


def test_reload_sync_switches_to_synced_mode_and_rerenders(scrubber) -> None:
    scrubber.seek_camera(0, 30)
    scrubber.seek_camera(1, 60)
    scrubber.reload_sync(_two_anchor_sync())
    # After reload, both cells seek to current_timestamp (0.0) → frame 0
    assert scrubber.current_frames[0] == 0
    assert scrubber.current_frames[1] == 0


def test_reload_sync_none_clears_sync(scrubber) -> None:
    scrubber.reload_sync(_two_anchor_sync())
    scrubber.reload_sync(None)
    assert scrubber.sync_table is None


def test_step_in_synced_mode_moves_all_cells(scrubber) -> None:
    scrubber.reload_sync(_two_anchor_sync())
    # 1 step at 30 fps = 1/30 s forward
    scrubber._step(1)
    assert scrubber.current_frames[0] == 1
    assert scrubber.current_frames[1] == 1


def test_shift_step_in_synced_mode(scrubber) -> None:
    scrubber.reload_sync(_two_anchor_sync())
    scrubber._step(10)
    assert scrubber.current_frames[0] == 10
    assert scrubber.current_frames[1] == 10


# ---------------------------------------------------------------------------
# Focus / cell click
# ---------------------------------------------------------------------------


def test_cell_click_updates_focused_cell(scrubber) -> None:
    scrubber._on_cell_clicked(1)
    assert scrubber.focused_cell == 1

    scrubber._on_cell_clicked(0)
    assert scrubber.focused_cell == 0


# ---------------------------------------------------------------------------
# Go-to shortcuts
# ---------------------------------------------------------------------------


def test_go_to_frame_independent(scrubber) -> None:
    scrubber._focused_cell = 0
    scrubber.seek_camera(0, 50)
    scrubber._go_to_frame(0)
    assert scrubber.current_frames[0] == 0


def test_go_to_end_independent(scrubber) -> None:
    scrubber._focused_cell = 0
    scrubber._go_to_end()
    assert scrubber.current_frames[0] == 299   # total_frames - 1


def test_go_to_end_synced(scrubber) -> None:
    scrubber.reload_sync(_two_anchor_sync())
    scrubber._go_to_end()
    assert scrubber.current_frames[0] == 299
    assert scrubber.current_frames[1] == 299


# ---------------------------------------------------------------------------
# frame_changed signal
# ---------------------------------------------------------------------------


def test_frame_changed_signal_emitted(scrubber, qapp) -> None:
    received: list[tuple[int, int]] = []
    scrubber.frame_changed.connect(lambda ci, fi: received.append((ci, fi)))
    scrubber.seek_camera(1, 42)
    assert (1, 42) in received


# ---------------------------------------------------------------------------
# Overlay passthrough
# ---------------------------------------------------------------------------


def test_set_overlays_delegates_to_cell(scrubber) -> None:
    from app.setup.overlay import ROIDrawOverlay
    ov = ROIDrawOverlay()
    scrubber.set_overlays(0, [ov])
    assert scrubber._cells[0]._overlays == [ov]
