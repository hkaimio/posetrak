# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Per-frame stepping, playback and prop frame display in PersonCropGridWidget
(docs/roadmap/features/preview-caching/preview-caching-design.md)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSlider, QWidget

from app.ui.content_panels import PersonCropGridWidget

FPS = 119.88
T_START = 300.0


class _Sync:
    """A single-video linear sync table, like SyncTable for one anchor."""

    def lookup(self, timestamp_s: float, shot_video_id: str) -> int:
        return round((timestamp_s - T_START) * FPS)

    def frame_to_global_time(self, frame_idx: int, shot_video_id: str) -> float:
        return T_START + frame_idx / FPS


class _Key:
    def __init__(self, key, shift=False) -> None:
        self._key, self._shift = key, shift

    def key(self):
        return self._key

    def modifiers(self):
        return Qt.KeyboardModifier.ShiftModifier if self._shift else Qt.KeyboardModifier.NoModifier


def _make_widget(frames: int = 600):
    w = PersonCropGridWidget.__new__(PersonCropGridWidget)
    QWidget.__init__(w)
    w._t_start = T_START
    w._current_t = T_START
    w._sync_table = _Sync()
    w._cameras = [{"shot_video_id": "sv1", "camera_instance_id": "cam1"}]
    w._range_start_v = 5
    w._range_end_v = 9
    w._slider = QSlider(Qt.Orientation.Horizontal)
    w._slider.setMaximum(int(frames / FPS * 1000))
    w._slider.setSingleStep(8)   # what the real widget uses: 1000/fps, rounded

    def follow(value: int) -> None:   # what _on_slider does to the time
        w._current_t = w._t_start + value / 1000.0

    w._slider.valueChanged.connect(follow)
    # attributes _handle_key reads
    w._chain_limb = None
    w._sel_kp_indices = set()
    w._sel_cam_idx = None
    w._timeline = None
    w._edit_mode = False
    return w


def _frame(w) -> int:
    return w._sync_table.lookup(w._current_t, "sv1")


def test_stepping_forward_advances_exactly_one_frame_every_time(qapp):
    w = _make_widget()
    w._step_frames(1)   # off frame 0 first, so the loop tests steady state
    seen = [_frame(w)]
    for _ in range(120):
        w._step_frames(1)
        seen.append(_frame(w))
    assert seen == list(range(seen[0], seen[0] + 121))   # no repeat, no skip


def test_the_millisecond_slider_step_would_have_repeated_frames(qapp):
    """Why _step_frames exists: a fixed 8 ms step drifts against 8.34 ms frames."""
    w = _make_widget()
    frames = []
    for _ in range(60):
        w._slider.setValue(w._slider.value() + w._slider.singleStep())
        frames.append(_frame(w))
    assert len(set(frames)) < len(frames)


def test_stepping_back_ten_and_clamping_at_both_ends(qapp):
    w = _make_widget(frames=300)
    w._slider.setValue(w._slider.maximum() // 2)
    start = _frame(w)
    w._step_frames(-10)
    assert _frame(w) == start - 10
    w._step_frames(-100000)
    assert w._slider.value() == 0
    w._step_frames(100000)
    assert w._slider.value() == w._slider.maximum()


def test_stepping_clears_the_selected_range(qapp):
    w = _make_widget()
    w._step_frames(1)
    assert w._range_start_v is None and w._range_end_v is None


def test_arrow_keys_step_when_no_keypoint_is_selected(qapp):
    w = _make_widget()
    w._step_frames(20)
    start = _frame(w)
    assert w._handle_key(_Key(Qt.Key.Key_Right))
    assert _frame(w) == start + 1
    assert w._handle_key(_Key(Qt.Key.Key_Left))
    assert _frame(w) == start
    assert w._handle_key(_Key(Qt.Key.Key_Right, shift=True))
    assert _frame(w) == start + 10


def test_arrow_keys_still_nudge_a_selected_keypoint(qapp):
    w = _make_widget()
    w._edit_mode = True   # a keypoint can only be selected while editing
    w._pending_place_kp_idx = None
    w._range_start_v = None
    w._sel_kp_indices = {3}
    w._sel_cam_idx = 0
    w._nudge_keypoint = MagicMock()
    start = _frame(w)
    assert w._handle_key(_Key(Qt.Key.Key_Right))
    w._nudge_keypoint.assert_called_once_with(1.0, 0.0)
    assert _frame(w) == start


def test_playback_advances_by_wall_clock_and_stops_at_the_end(qapp, monkeypatch):
    import time

    w = _make_widget(frames=120)
    clock = {"t": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
    w._toggle_play()
    assert w._play_timer.isActive()
    clock["t"] += 0.25
    w._play_tick()
    assert w._slider.value() == 250
    clock["t"] += 5.0                # far past the end of a 1 s clip
    w._play_tick()
    assert w._slider.value() == w._slider.maximum()
    assert not w._play_timer.isActive()


def test_space_toggles_playback(qapp):
    w = _make_widget()
    assert w._handle_key(_Key(Qt.Key.Key_Space))
    assert w._play_timer.isActive()
    assert w._handle_key(_Key(Qt.Key.Key_Space))
    assert not w._play_timer.isActive()


# ---------------------------------------------------------------------------
# Prop frames from the frame service
# ---------------------------------------------------------------------------

CAM = {"shot_video_id": "sv1", "camera_instance_id": "cam1"}


def _service_widget(service):
    w = _make_widget()
    w._frame_service = service
    w._apply_overlay = MagicMock()
    w._compute_target_rect = MagicMock(return_value=(0, 0, 10, 10))
    w._display_crop_result = MagicMock()
    return w


def test_a_cached_frame_is_displayed_and_keeps_read_ahead_going(qapp):
    result = (b"jpeg", 960, 540, 0, 0, 3840, 2160)
    service = MagicMock()
    service.get.return_value = result
    w = _service_widget(service)
    cell = MagicMock()

    w._show_service_frame(cell, CAM, 42, 7, True, True, False, False)

    w._display_crop_result.assert_called_once()
    args = w._display_crop_result.call_args
    assert args.args[7] is result                       # the decoded result
    assert args.kwargs["layer_label"] == "frame-service"
    w._compute_target_rect.assert_called_once()
    assert w._compute_target_rect.call_args.args[5] == (0.0, 0.0, 3840.0, 2160.0)
    service.request.assert_called_once_with("sv1", 42)
    cell.show_loading.assert_not_called()


def test_a_missing_frame_shows_loading_and_is_requested(qapp):
    service = MagicMock()
    service.get.return_value = None
    service.request.return_value = False
    service.is_unavailable.return_value = False
    service.is_missing.return_value = False
    w = _service_widget(service)
    cell = MagicMock()

    w._show_service_frame(cell, CAM, 42, None, True, True, False, False)

    cell.show_loading.assert_called_once()
    service.request.assert_called_once_with("sv1", 42)
    w._display_crop_result.assert_not_called()
    w._apply_overlay.assert_called_once()   # overlays stay visible over the placeholder


def test_an_unavailable_video_shows_empty_not_loading_forever(qapp):
    service = MagicMock()
    service.get.return_value = None
    service.is_unavailable.return_value = True
    w = _service_widget(service)
    cell = MagicMock()

    w._show_service_frame(cell, CAM, 42, None, True, True, False, False)

    cell.show_empty.assert_called_once()
    cell.show_loading.assert_not_called()


def test_frame_ready_signals_from_all_cameras_cause_one_redraw(qapp):
    from PySide6.QtCore import QCoreApplication

    w = _make_widget()
    w._load_frame = MagicMock()
    frame = _frame(w)
    for svid in ("sv1", "sv2", "sv3", "sv4", "sv5"):
        w._on_service_frame_ready("sv1", frame)   # same camera lookup: all match the frame
    QCoreApplication.processEvents()
    import time
    time.sleep(0.05)
    QCoreApplication.processEvents()
    assert w._load_frame.call_count == 1


def test_a_frame_that_is_not_on_screen_causes_no_redraw(qapp):
    from PySide6.QtCore import QCoreApplication
    import time

    w = _make_widget()
    w._load_frame = MagicMock()
    w._on_service_frame_ready("sv1", _frame(w) + 40)   # a prefetched frame
    time.sleep(0.05)
    QCoreApplication.processEvents()
    w._load_frame.assert_not_called()


def test_a_frame_past_the_end_of_a_short_video_shows_empty_not_loading(qapp):
    service = MagicMock()
    service.get.return_value = None
    service.is_unavailable.return_value = False
    service.is_missing.return_value = True     # the sync table maps past the clip's end
    w = _service_widget(service)
    cell = MagicMock()

    w._show_service_frame(cell, CAM, 40169, None, True, True, False, False)

    cell.show_empty.assert_called_once()
    cell.show_loading.assert_not_called()
