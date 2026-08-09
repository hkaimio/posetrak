"""Tests for app.setup.video_scrub_bar.VideoScrubBar.

These cases were originally written against ``pair_scrubber._VideoPane``
before its slider/label/``FrameReader`` logic was extracted into this shared
component (see
``docs/roadmap/features/extrinsics-improvements/extrinsics-improvements-design.md``,
"Frame source & scrubbing"). ``test_pair_scrubber.py`` covers ``_VideoPane``
and ``PairScrubber`` at the public-API level; this file covers the scrub
logic itself.
"""

from __future__ import annotations

from app.setup.video_scrub_bar import VideoScrubBar


def test_constructs(qapp) -> None:
    bar = VideoScrubBar()
    assert bar.total_frames == 1
    assert bar.current_frame == 0
    assert not bar.is_loaded


def test_step_clamps_at_zero(qapp) -> None:
    bar = VideoScrubBar()
    bar._total_frames = 100
    bar._current_frame = 0
    bar.step(-10)
    assert bar.current_frame == 0


def test_step_clamps_at_max(qapp) -> None:
    bar = VideoScrubBar()
    bar._total_frames = 100
    bar._current_frame = 99
    bar.step(10)
    assert bar.current_frame == 99


def test_step_moves_forward(qapp) -> None:
    bar = VideoScrubBar()
    bar._total_frames = 100
    bar._current_frame = 50
    bar.step(5)
    assert bar.current_frame == 55


def test_step_moves_backward(qapp) -> None:
    bar = VideoScrubBar()
    bar._total_frames = 100
    bar._current_frame = 50
    bar.step(-15)
    assert bar.current_frame == 35


def test_seek_clamps_negative(qapp) -> None:
    bar = VideoScrubBar()
    bar._total_frames = 100
    bar._current_frame = 50
    bar.seek(-5)
    assert bar.current_frame == 0


def test_seek_clamps_beyond_max(qapp) -> None:
    bar = VideoScrubBar()
    bar._total_frames = 100
    bar._current_frame = 0
    bar.seek(200)
    assert bar.current_frame == 99


def test_seek_to_valid_frame(qapp) -> None:
    bar = VideoScrubBar()
    bar._total_frames = 100
    bar.seek(42)
    assert bar.current_frame == 42


def test_frame_label_updates_on_seek(qapp) -> None:
    bar = VideoScrubBar()
    bar._total_frames = 100
    bar._slider.setMaximum(99)
    bar._seek(77)
    assert "77" in bar._frame_label.text()


def test_slider_updates_on_seek(qapp) -> None:
    bar = VideoScrubBar()
    bar._total_frames = 100
    bar._slider.setMaximum(99)
    bar._seek(33)
    assert bar._slider.value() == 33


def test_frame_changed_signal(qapp) -> None:
    bar = VideoScrubBar()
    bar._total_frames = 100
    bar._slider.setMaximum(99)
    received: list[int] = []
    bar.frame_changed.connect(received.append)
    bar._seek(10)
    assert received == [10]


def test_step_no_signal_when_clamped(qapp) -> None:
    bar = VideoScrubBar()
    bar._total_frames = 100
    bar._current_frame = 0
    received: list[int] = []
    bar.frame_changed.connect(received.append)
    bar.step(-1)  # already at 0, should not emit
    assert received == []


def test_load_missing_file_does_not_crash(qapp) -> None:
    bar = VideoScrubBar()
    bar.load("/nonexistent/video.mp4", 300, initial_frame=10)
    assert bar.total_frames == 300
    assert bar.current_frame == 10
    assert bar.is_loaded
    bar.unload()
    assert not bar.is_loaded


def test_load_clamps_initial_frame(qapp) -> None:
    bar = VideoScrubBar()
    bar.load("/nonexistent/video.mp4", 50, initial_frame=1000)
    assert bar.current_frame == 49
    bar.unload()


def test_unload_resets_label_and_slider(qapp) -> None:
    bar = VideoScrubBar()
    bar.load("/nonexistent/video.mp4", 50, initial_frame=10)
    bar.unload()
    assert bar._frame_label.text() == "frame —"
    assert bar._slider.maximum() == 0


# ---------------------------------------------------------------------------
# "Go to..." dialog
#
# Regression for a real bug found in UI testing: QInputDialog.getInt() was
# called with min=/max= (PyQt-style keyword names) instead of PySide6's
# minValue=/maxValue=, raising "AttributeError: unsupported keyword 'min'"
# every time the button was clicked. QInputDialog.getInt() itself is not
# exercised here (it blocks on a modal dialog) -- these patch the staticmethod
# to confirm _on_goto() calls it with argument names PySide6 actually accepts,
# and that the returned value is applied on "ok".
# ---------------------------------------------------------------------------


def test_goto_calls_get_int_with_pyside6_keyword_names(qapp, monkeypatch) -> None:
    from app.setup import video_scrub_bar as module

    calls: list[dict] = []

    def fake_get_int(*args, **kwargs):
        calls.append(kwargs)
        return 42, True

    monkeypatch.setattr(module.QInputDialog, "getInt", fake_get_int)

    bar = VideoScrubBar()
    bar._total_frames = 100
    bar._on_goto()

    assert len(calls) == 1
    assert "minValue" in calls[0]
    assert "maxValue" in calls[0]
    assert "min" not in calls[0]
    assert "max" not in calls[0]
    assert bar.current_frame == 42


def test_goto_cancelled_does_not_seek(qapp, monkeypatch) -> None:
    from app.setup import video_scrub_bar as module

    monkeypatch.setattr(
        module.QInputDialog, "getInt", lambda *a, **kw: (999, False)
    )

    bar = VideoScrubBar()
    bar._total_frames = 100
    bar._current_frame = 10
    bar._on_goto()

    assert bar.current_frame == 10  # unchanged -- dialog was cancelled


def test_goto_real_qinputdialog_does_not_raise(qapp) -> None:
    """End-to-end regression: call the *real* QInputDialog.getInt() (not a
    stub) and confirm the min=/max= vs. minValue=/maxValue= mismatch is
    actually gone. Argument-binding errors like the original
    "unsupported keyword 'min'" AttributeError happen before the modal
    dialog is ever shown, so a QTimer.singleShot to close it is only needed
    to let the (otherwise blocking) call return once opened with valid
    arguments.
    """
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QInputDialog

    def close_dialog():
        for w in QApplication.topLevelWidgets():
            if isinstance(w, QInputDialog):
                w.reject()

    bar = VideoScrubBar()
    bar._total_frames = 100
    bar._current_frame = 5

    QTimer.singleShot(0, close_dialog)
    bar._on_goto()  # must not raise AttributeError

    assert bar.current_frame == 5  # rejected -- unchanged
