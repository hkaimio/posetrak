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
    w._object_view = True
    w._apply_overlay = MagicMock()
    w._compute_target_rect = MagicMock(return_value=(0, 0, 10, 10))
    w._display_crop_result = MagicMock()
    return w


def _cell(canvas_width: int = 400, zoom=None):
    cell = MagicMock()
    cell._canvas.width.return_value = canvas_width
    cell._canvas._zoom_rect = zoom
    return cell


def test_a_cached_frame_is_displayed_and_keeps_read_ahead_going(qapp):
    result = (b"jpeg", 960, 540, 0, 0, 3840, 2160)
    service = MagicMock()
    service.get.return_value = result
    service.get_region.return_value = None
    w = _service_widget(service)
    cell = _cell()

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


# ---------------------------------------------------------------------------
# Zoom: sharper image for the view, and no stale crop
# ---------------------------------------------------------------------------

WHOLE = (b"jpeg", 1920, 1080, 0, 0, 3840, 2160)   # a 4K frame held at half size


def _zoom_widget(crop=None):
    service = MagicMock()
    service.get.return_value = WHOLE
    service.get_region.return_value = crop
    w = _service_widget(service)
    w._compute_target_rect.return_value = (0.0, 0.0, 3840.0, 2160.0)
    return w, service


def test_zoomed_view_needing_more_than_the_whole_frame_gives_asks_for_a_crop(qapp):
    w, service = _zoom_widget()
    cell = _cell(canvas_width=800, zoom=(1000.0, 500.0, 1800.0, 950.0))   # 1 px per frame px

    w._show_service_frame(cell, CAM, 42, None, True, True, False, False)

    rect = service.request_region.call_args.args[2]
    assert rect[0] < 1000 and rect[2] > 1800            # the view plus a margin
    assert service.request_region.call_args.args[3] == 1.0
    args = w._display_crop_result.call_args
    assert args.args[7] is WHOLE                       # meanwhile the old image stays
    assert args.kwargs["target_rect"] == (0.0, 0.0, 3840.0, 2160.0)   # whole frame, so zooming out is never clipped


def test_a_crop_covering_the_view_is_shown_instead(qapp):
    crop = (b"crop", 1200, 675, 800, 400, 1200, 675)
    w, service = _zoom_widget(crop)
    cell = _cell(canvas_width=800, zoom=(1000.0, 500.0, 1800.0, 950.0))

    w._show_service_frame(cell, CAM, 42, None, True, True, False, False)

    service.request_region.assert_not_called()
    args = w._display_crop_result.call_args
    assert args.args[7] is crop
    assert args.kwargs["target_rect"] == (800.0, 400.0, 2000.0, 1075.0)


def test_a_crop_that_no_longer_covers_the_view_is_replaced_after_zooming_out(qapp):
    crop = (b"crop", 1200, 675, 800, 400, 1200, 675)
    w, service = _zoom_widget(crop)
    cell = _cell(canvas_width=800, zoom=(200.0, 100.0, 2600.0, 1450.0))   # zoomed out past the crop

    w._show_service_frame(cell, CAM, 42, None, True, True, False, False)

    assert w._display_crop_result.call_args.args[7] is not crop
    # 800 px for 2400 frame px is 0.33 px/px, which the half-size whole frame (0.5) already gives:
    service.request_region.assert_not_called()


def test_an_unzoomed_view_never_asks_for_a_crop(qapp):
    w, service = _zoom_widget()
    w._show_service_frame(_cell(canvas_width=800), CAM, 42, None, True, True, False, False)
    service.request_region.assert_not_called()


def test_no_crop_is_requested_while_the_view_is_still_moving(qapp):
    w, service = _zoom_widget()
    w._on_view_changed()   # starts the settle timer
    cell = _cell(canvas_width=800, zoom=(1000.0, 500.0, 1800.0, 950.0))
    w._show_service_frame(cell, CAM, 42, None, True, True, False, False)
    service.request_region.assert_not_called()
    w._view_timer.stop()


def test_the_segmentation_legend_is_drawn_by_the_canvas_at_the_viewport_corner(qapp):
    from PySide6.QtGui import QColor, QImage, QPixmap

    from app.ui.content_panels import _ImageCanvas

    canvas = _ImageCanvas()
    canvas.resize(300, 200)
    pixmap = QPixmap(1000, 1000)
    pixmap.fill(QColor(0, 0, 0))
    canvas.show_image(pixmap, 0.0, 0.0, 1.0)
    canvas._zoom_rect = (600.0, 600.0, 800.0, 700.0)   # the image's own corner is out of view
    canvas.set_legend([(QColor(255, 0, 0), "Sword")])

    img = canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
    reds = [x for x in range(0, 60) for y in range(0, 40) if QColor(img.pixel(x, y)).red() > 200
            and QColor(img.pixel(x, y)).green() < 50]
    assert reds   # the swatch is inside the top-left corner of the widget

    canvas.set_legend([])
    img = canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
    assert not any(QColor(img.pixel(x, y)).red() > 200 for x in range(0, 60) for y in range(0, 40))


def test_the_nearest_stored_frame_is_shown_while_the_exact_one_is_decoded(qapp):
    proxy = (b"proxy", 640, 360, 0, 0, 3840, 2160)
    service = MagicMock()
    service.get.return_value = None
    service.request.return_value = False
    service.is_unavailable.return_value = False
    service.is_missing.return_value = False
    w = _service_widget(service)
    w._proxy_cache = MagicMock()
    w._proxy_cache.nearest.return_value = proxy
    cell = MagicMock()

    w._show_service_frame(cell, CAM, 42, None, True, True, False, False)

    args = w._display_crop_result.call_args
    assert args.args[7] is proxy and args.kwargs["layer_label"] == "proxy"
    cell.show_loading.assert_not_called()
    service.request.assert_called_once_with("sv1", 42)   # the exact frame is still requested


def test_a_person_view_is_framed_on_the_keypoints_not_the_whole_frame(qapp):
    service = MagicMock()
    service.get.return_value = WHOLE
    service.get_region.return_value = None
    w = _service_widget(service)
    w._object_view = False
    person_rect = (1000.0, 500.0, 1400.0, 1100.0)
    w._compute_target_rect.return_value = person_rect

    w._show_service_frame(_cell(canvas_width=400), CAM, 42, None, True, True, False, False)

    assert w._compute_target_rect.call_args.args[5] is None   # no whole-frame base for a person
    assert w._display_crop_result.call_args.kwargs["target_rect"] == person_rect
    # 400 canvas px for a 400 px wide view is 1 px per frame px; the half-size frame
    # cannot give that, so a native-resolution crop of the person is requested.
    assert service.request_region.called


def test_a_person_without_keypoints_or_skeleton_is_shown_whole(qapp):
    service = MagicMock()
    service.get.return_value = WHOLE
    service.get_region.return_value = None
    w = _service_widget(service)
    w._object_view = False
    w._compute_target_rect.return_value = None

    w._show_service_frame(_cell(canvas_width=400), CAM, 42, None, True, True, False, False)

    assert w._display_crop_result.call_args.kwargs["target_rect"] == (0.0, 0.0, 3840.0, 2160.0)
