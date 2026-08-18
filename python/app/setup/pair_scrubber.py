# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""pair_scrubber.py — Two-camera side-by-side video scrubber for sync marking.

Shows a reference video (left, fixed) and a target video (right, swappable).
The user navigates each video independently to find a shared event, then
signals that the current frame pair should be recorded as a sync anchor.

Keyboard shortcuts (when the widget has focus):
  A / D       step reference video ±1 frame
  Shift+A/D   step reference ±10 frames
  ← / →       step target video ±1 frame
  Shift+←/→   step target ±10 frames
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.setup.camera_cell import CameraCell
from app.setup.video_scrub_bar import VideoScrubBar


class _VideoPane(QWidget):
    """One side of the pair scrubber: camera cell + video scrub bar.

    The slider/label/"Go to…"/``FrameReader`` machinery lives in the shared
    ``VideoScrubBar`` (see ``video_scrub_bar.py``); this class only wires a
    decoded frame to its ``CameraCell`` display.
    """

    frame_changed = Signal(int)

    def __init__(self, side_label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cell = CameraCell(label=side_label, parent=self)
        self._cell.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._scrub = VideoScrubBar(self)
        self._scrub.frame_ready.connect(self._on_frame_ready)
        self._scrub.frame_changed.connect(self.frame_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._cell, stretch=1)
        layout.addWidget(self._scrub)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load(self, file_path: str, total_frames: int, initial_frame: int = 0) -> None:
        """Load a new video file, replacing any previously loaded one."""
        self._scrub.load(file_path, total_frames, initial_frame)

    def unload(self) -> None:
        """Stop the reader and clear the display."""
        self._scrub.unload()
        self._cell.set_frame(None)

    @property
    def current_frame(self) -> int:
        return self._scrub.current_frame

    @property
    def total_frames(self) -> int:
        return self._scrub.total_frames

    @property
    def is_loaded(self) -> bool:
        return self._scrub.is_loaded

    def step(self, delta: int) -> None:
        """Advance by *delta* frames (negative = backwards)."""
        self._scrub.step(delta)

    def seek(self, frame: int) -> None:
        self._scrub.seek(frame)

    def set_overlays(self, overlays: list) -> None:
        self._cell.set_overlays(overlays)
        self._cell.update()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _on_frame_ready(self, frame_idx: int, frame_data) -> None:
        self._cell.set_frame(frame_data)


class PairScrubber(QWidget):
    """Two-camera side-by-side scrubber for manual sync anchor marking.

    The left pane is the *reference* video; the right pane is the *target*.
    The target can be swapped without disturbing the reference position.

    Signals
    -------
    anchor_requested(ref_frame, target_frame):
        Emitted when the user clicks "Mark sync pair".
    frames_changed(ref_frame, target_frame):
        Emitted whenever either video advances to a new frame.
    """

    anchor_requested = Signal(int, int)
    frames_changed = Signal(int, int)
    timeline_seek_step = Signal(int)  # ± frame-equivalent steps to move global timeline

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._ref_pane = _VideoPane("Reference", self)
        self._tgt_pane = _VideoPane("Target", self)

        self._ref_pane.frame_changed.connect(self._on_frames_changed)
        self._tgt_pane.frame_changed.connect(self._on_frames_changed)

        panes = QHBoxLayout()
        panes.setSpacing(4)
        panes.addWidget(self._ref_pane)
        panes.addWidget(self._tgt_pane)

        self._mark_btn = QPushButton("Mark sync pair at these frames")
        self._mark_btn.setEnabled(False)
        self._mark_btn.clicked.connect(self._on_mark)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(panes, stretch=1)
        layout.addWidget(self._mark_btn)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_reference(
        self, file_path: str, total_frames: int, label: str = "Reference", initial_frame: int = 0
    ) -> None:
        self._ref_pane._cell._label = label
        self._ref_pane.load(file_path, total_frames, initial_frame)
        self._update_mark_btn()

    def set_target(
        self, file_path: str, total_frames: int, label: str = "Target", initial_frame: int = 0
    ) -> None:
        self._tgt_pane._cell._label = label
        self._tgt_pane.load(file_path, total_frames, initial_frame)
        self._update_mark_btn()

    def unload_target(self) -> None:
        self._tgt_pane.unload()
        self._update_mark_btn()

    @property
    def ref_frame(self) -> int:
        return self._ref_pane.current_frame

    @property
    def target_frame(self) -> int:
        return self._tgt_pane.current_frame

    def seek_reference(self, frame: int) -> None:
        self._ref_pane.seek(frame)

    def seek_target(self, frame: int) -> None:
        self._tgt_pane.seek(frame)

    def set_ref_overlays(self, overlays: list) -> None:
        self._ref_pane.set_overlays(overlays)

    def set_target_overlays(self, overlays: list) -> None:
        self._tgt_pane.set_overlays(overlays)

    def shutdown(self) -> None:
        self._ref_pane.unload()
        self._tgt_pane.unload()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _on_mark(self) -> None:
        self.anchor_requested.emit(self._ref_pane.current_frame, self._tgt_pane.current_frame)

    def _on_frames_changed(self, _frame: int) -> None:
        self.frames_changed.emit(self._ref_pane.current_frame, self._tgt_pane.current_frame)

    def _update_mark_btn(self) -> None:
        self._mark_btn.setEnabled(self._ref_pane.is_loaded and self._tgt_pane.is_loaded)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        mod = event.modifiers()
        step = 10 if (mod & Qt.KeyboardModifier.ShiftModifier) else 1

        if key == Qt.Key.Key_A:
            self._ref_pane.step(-step)
        elif key == Qt.Key.Key_D:
            self._ref_pane.step(step)
        elif key == Qt.Key.Key_J:
            self._tgt_pane.step(-step)
        elif key == Qt.Key.Key_L:
            self._tgt_pane.step(step)
        else:
            super().keyPressEvent(event)
