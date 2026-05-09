"""Tests for app.setup.pair_scrubber — PairScrubber and _VideoPane."""

from __future__ import annotations

import pytest

from PySide6.QtCore import Qt

from app.setup.pair_scrubber import PairScrubber, _VideoPane


# ---------------------------------------------------------------------------
# _VideoPane — internal pane
# ---------------------------------------------------------------------------


def test_video_pane_constructs(qapp) -> None:
    pane = _VideoPane("Test")
    assert pane._total_frames == 1
    assert pane._current_frame == 0
    assert pane._reader is None


def test_video_pane_step_clamps_at_zero(qapp) -> None:
    pane = _VideoPane("Test")
    pane._total_frames = 100
    pane._current_frame = 0
    pane.step(-10)
    assert pane._current_frame == 0


def test_video_pane_step_clamps_at_max(qapp) -> None:
    pane = _VideoPane("Test")
    pane._total_frames = 100
    pane._current_frame = 99
    pane.step(10)
    assert pane._current_frame == 99


def test_video_pane_step_moves_forward(qapp) -> None:
    pane = _VideoPane("Test")
    pane._total_frames = 100
    pane._current_frame = 50
    pane.step(5)
    assert pane._current_frame == 55


def test_video_pane_step_moves_backward(qapp) -> None:
    pane = _VideoPane("Test")
    pane._total_frames = 100
    pane._current_frame = 50
    pane.step(-15)
    assert pane._current_frame == 35


def test_video_pane_seek_clamps_negative(qapp) -> None:
    pane = _VideoPane("Test")
    pane._total_frames = 100
    pane._current_frame = 50
    pane.seek(-5)
    assert pane._current_frame == 0


def test_video_pane_seek_clamps_beyond_max(qapp) -> None:
    pane = _VideoPane("Test")
    pane._total_frames = 100
    pane._current_frame = 0
    pane.seek(200)
    assert pane._current_frame == 99


def test_video_pane_seek_to_valid_frame(qapp) -> None:
    pane = _VideoPane("Test")
    pane._total_frames = 100
    pane.seek(42)
    assert pane._current_frame == 42


def test_video_pane_frame_label_updates_on_seek(qapp) -> None:
    pane = _VideoPane("Test")
    pane._total_frames = 100
    pane._slider.setMaximum(99)
    pane._seek(77)
    assert "77" in pane._frame_label.text()


def test_video_pane_slider_updates_on_seek(qapp) -> None:
    pane = _VideoPane("Test")
    pane._total_frames = 100
    pane._slider.setMaximum(99)
    pane._seek(33)
    assert pane._slider.value() == 33


def test_video_pane_frame_changed_signal(qapp) -> None:
    pane = _VideoPane("Test")
    pane._total_frames = 100
    pane._slider.setMaximum(99)
    received: list[int] = []
    pane.frame_changed.connect(received.append)
    pane._seek(10)
    assert received == [10]


def test_video_pane_step_no_signal_when_clamped(qapp) -> None:
    pane = _VideoPane("Test")
    pane._total_frames = 100
    pane._current_frame = 0
    received: list[int] = []
    pane.frame_changed.connect(received.append)
    pane.step(-1)  # already at 0, should not emit
    assert received == []


# ---------------------------------------------------------------------------
# PairScrubber — construction
# ---------------------------------------------------------------------------


def test_pair_scrubber_constructs(qapp) -> None:
    ps = PairScrubber()
    assert ps._ref_pane is not None
    assert ps._tgt_pane is not None
    ps.shutdown()


def test_pair_scrubber_mark_btn_disabled_initially(qapp) -> None:
    ps = PairScrubber()
    assert not ps._mark_btn.isEnabled()
    ps.shutdown()


def test_pair_scrubber_ref_frame_is_zero_initially(qapp) -> None:
    ps = PairScrubber()
    assert ps.ref_frame == 0
    ps.shutdown()


def test_pair_scrubber_target_frame_is_zero_initially(qapp) -> None:
    ps = PairScrubber()
    assert ps.target_frame == 0
    ps.shutdown()


# ---------------------------------------------------------------------------
# PairScrubber — set_reference / set_target with fake files
# (FrameReader gracefully fails to open nonexistent path)
# ---------------------------------------------------------------------------


def test_set_reference_does_not_crash_with_missing_file(qapp) -> None:
    ps = PairScrubber()
    ps.set_reference("/nonexistent/video.mp4", 300, "Cam A")
    assert ps._ref_pane._total_frames == 300
    ps.shutdown()


def test_set_target_does_not_crash_with_missing_file(qapp) -> None:
    ps = PairScrubber()
    ps.set_target("/nonexistent/video.mp4", 200, "Cam B")
    assert ps._tgt_pane._total_frames == 200
    ps.shutdown()


def test_mark_btn_enabled_after_both_loaded(qapp) -> None:
    ps = PairScrubber()
    ps.set_reference("/fake/ref.mp4", 300, "Cam A")
    ps.set_target("/fake/tgt.mp4", 200, "Cam B")
    assert ps._mark_btn.isEnabled()
    ps.shutdown()


# ---------------------------------------------------------------------------
# PairScrubber — seek
# ---------------------------------------------------------------------------


def test_seek_reference_updates_frame(qapp) -> None:
    ps = PairScrubber()
    ps.set_reference("/fake/ref.mp4", 300, "Cam A")
    ps.seek_reference(150)
    assert ps.ref_frame == 150
    ps.shutdown()


def test_seek_target_updates_frame(qapp) -> None:
    ps = PairScrubber()
    ps.set_target("/fake/tgt.mp4", 200, "Cam B")
    ps.seek_target(75)
    assert ps.target_frame == 75
    ps.shutdown()


# ---------------------------------------------------------------------------
# PairScrubber — anchor_requested signal
# ---------------------------------------------------------------------------


def test_on_mark_emits_anchor_requested(qapp) -> None:
    ps = PairScrubber()
    ps.set_reference("/fake/ref.mp4", 300, "Cam A")
    ps.set_target("/fake/tgt.mp4", 200, "Cam B")
    ps.seek_reference(100)
    ps.seek_target(50)

    received: list[tuple[int, int]] = []
    ps.anchor_requested.connect(lambda r, t: received.append((r, t)))
    ps._on_mark()

    assert received == [(100, 50)]
    ps.shutdown()


def test_frames_changed_signal_on_seek(qapp) -> None:
    ps = PairScrubber()
    ps.set_reference("/fake/ref.mp4", 300, "Cam A")
    ps.set_target("/fake/tgt.mp4", 200, "Cam B")

    received: list[tuple[int, int]] = []
    ps.frames_changed.connect(lambda r, t: received.append((r, t)))
    ps.seek_reference(10)

    assert len(received) >= 1
    assert received[-1][0] == 10
    ps.shutdown()


# ---------------------------------------------------------------------------
# PairScrubber — keyboard events
# ---------------------------------------------------------------------------


def test_key_a_steps_reference_backward(qapp) -> None:
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent

    ps = PairScrubber()
    ps.set_reference("/fake/ref.mp4", 300, "Cam A")
    ps.seek_reference(50)

    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier)
    ps.keyPressEvent(event)
    assert ps.ref_frame == 49
    ps.shutdown()


def test_key_d_steps_reference_forward(qapp) -> None:
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent

    ps = PairScrubber()
    ps.set_reference("/fake/ref.mp4", 300, "Cam A")
    ps.seek_reference(50)

    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_D, Qt.KeyboardModifier.NoModifier)
    ps.keyPressEvent(event)
    assert ps.ref_frame == 51
    ps.shutdown()


def test_key_left_steps_target_backward(qapp) -> None:
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent

    ps = PairScrubber()
    ps.set_target("/fake/tgt.mp4", 200, "Cam B")
    ps.seek_target(50)

    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Left, Qt.KeyboardModifier.NoModifier)
    ps.keyPressEvent(event)
    assert ps.target_frame == 49
    ps.shutdown()


def test_key_right_steps_target_forward(qapp) -> None:
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent

    ps = PairScrubber()
    ps.set_target("/fake/tgt.mp4", 200, "Cam B")
    ps.seek_target(50)

    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier)
    ps.keyPressEvent(event)
    assert ps.target_frame == 51
    ps.shutdown()


def test_shift_a_steps_reference_by_10(qapp) -> None:
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent

    ps = PairScrubber()
    ps.set_reference("/fake/ref.mp4", 300, "Cam A")
    ps.seek_reference(50)

    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.ShiftModifier)
    ps.keyPressEvent(event)
    assert ps.ref_frame == 40
    ps.shutdown()


def test_shift_right_steps_target_by_10(qapp) -> None:
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent

    ps = PairScrubber()
    ps.set_target("/fake/tgt.mp4", 200, "Cam B")
    ps.seek_target(50)

    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    ps.keyPressEvent(event)
    assert ps.target_frame == 60
    ps.shutdown()


# ---------------------------------------------------------------------------
# PairScrubber — unload
# ---------------------------------------------------------------------------


def test_unload_target_disables_mark_btn(qapp) -> None:
    ps = PairScrubber()
    ps.set_reference("/fake/ref.mp4", 300, "Cam A")
    ps.set_target("/fake/tgt.mp4", 200, "Cam B")
    ps.unload_target()
    assert not ps._mark_btn.isEnabled()
    ps.shutdown()
