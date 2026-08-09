"""video_scrub_bar.py — Slider + frame label + "Go to…" scrub control.

Extracted from ``pair_scrubber.py``'s ``_VideoPane`` (see
``docs/roadmap/features/extrinsics-improvements/extrinsics-improvements-design.md``,
"Frame source & scrubbing") so per-camera video scrubbing is a single shared
component instead of being reimplemented per page. ``VideoScrubBar`` owns a
``FrameReader`` and its slider/label/"Go to…" controls but has no opinion on
how a decoded frame is displayed — it reports frames via ``frame_ready`` and
lets the caller wire that to whichever widget shows the image (a
``CameraCell``, a ``_ClickableImageWidget``, ...).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from app.setup.video_reader import FrameReader


class VideoScrubBar(QWidget):
    """Slider + frame-number label + "Go to…" button bound to a FrameReader.

    Signals
    -------
    frame_ready(frame_idx, frame_bgr):
        Emitted when the frame at the *current* scrub position has finished
        decoding (stale, superseded requests are dropped).
    frame_changed(frame_idx):
        Emitted whenever the scrub position itself changes, independent of
        whether the corresponding frame has finished decoding yet.
    """

    frame_ready = Signal(int, object)
    frame_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._total_frames = 1
        self._current_frame = 0
        self._reader: FrameReader | None = None

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider.sliderMoved.connect(self._on_slider_moved)

        self._frame_label = QLabel("frame —")
        self._frame_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        self._frame_label.setFixedWidth(90)

        self._goto_btn = QPushButton("Go to…")
        self._goto_btn.setFixedWidth(54)
        self._goto_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._goto_btn.clicked.connect(self._on_goto)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._slider)
        layout.addWidget(self._frame_label)
        layout.addWidget(self._goto_btn)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def is_loaded(self) -> bool:
        return self._reader is not None

    def load(self, file_path: str, total_frames: int, initial_frame: int = 0) -> None:
        """Load a new video file, replacing any previously loaded one."""
        self._stop_reader()
        self._total_frames = max(total_frames, 1)
        self._current_frame = max(0, min(initial_frame, self._total_frames - 1))

        self._slider.setMaximum(self._total_frames - 1)
        self._slider.setValue(self._current_frame)
        self._frame_label.setText(f"frame {self._current_frame}")

        self._reader = FrameReader(file_path, self)
        self._reader.frame_ready.connect(self._on_frame_ready)
        self._reader.start()
        self._reader.request(self._current_frame)

    def unload(self) -> None:
        """Stop the reader and reset the controls."""
        self._stop_reader()
        self._slider.setMaximum(0)
        self._slider.setValue(0)
        self._frame_label.setText("frame —")

    def step(self, delta: int) -> None:
        """Advance by *delta* frames (negative = backwards)."""
        new_frame = max(0, min(self._total_frames - 1, self._current_frame + delta))
        if new_frame != self._current_frame:
            self._seek(new_frame)

    def seek(self, frame: int) -> None:
        frame = max(0, min(self._total_frames - 1, frame))
        self._seek(frame)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _seek(self, frame: int) -> None:
        self._current_frame = frame
        self._slider.setValue(frame)
        self._frame_label.setText(f"frame {frame}")
        if self._reader is not None:
            self._reader.request(frame)
        self.frame_changed.emit(frame)

    def _on_slider_moved(self, value: int) -> None:
        self._seek(value)

    def _on_goto(self) -> None:
        value, ok = QInputDialog.getInt(
            self, "Go to frame", "Frame number:",
            value=self._current_frame,
            minValue=0, maxValue=max(0, self._total_frames - 1),
        )
        if ok:
            self._seek(value)

    def _on_frame_ready(self, frame_idx: int, frame_data) -> None:
        if frame_idx == self._current_frame:
            self.frame_ready.emit(frame_idx, frame_data)

    def _stop_reader(self) -> None:
        if self._reader is not None:
            self._reader.shutdown()
            self._reader = None
