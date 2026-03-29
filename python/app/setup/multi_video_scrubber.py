"""MultiVideoScrubber — grid of CameraCell widgets for multi-camera scrubbing.

Supports two navigation modes:

- **Synced mode** (a ``SyncTable`` is loaded): ``←``/``→`` advance the
  reference camera's timestamp; all other cameras follow via
  ``SyncTable.lookup()``.
- **Independent mode** (no sync table, or user has focused a specific cell):
  keyboard controls advance only the focused cell.

``Space`` toggles play/pause.  ``Home``/``End`` jump to the first/last frame.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.setup.camera_cell import CameraCell
from app.setup.db_context import SyncTable
from app.setup.frame_cache import CacheKey, CacheType, FrameCache
from app.setup.overlay import Overlay


@dataclass
class CellInfo:
    """Configuration for one camera cell in the scrubber.

    Parameters
    ----------
    shot_video_id:
        Session-DB ``shot_videos.id`` — used as the key into ``FrameCache``
        and ``SyncTable``.
    file_path:
        Path to the video file on disk.
    total_frames:
        Total frame count of the video.
    fps:
        Actual recorded frames per second.
    label:
        Short human-readable camera label shown in the placeholder.
    """
    shot_video_id: str
    file_path: str
    total_frames: int
    fps: float
    label: str = ""


class _FrameDecoder(QThread):
    """Background thread that decodes video frames without blocking the UI.

    Callers post requests via ``request(cell_idx, frame_idx)``.  When many
    requests arrive for the same cell before the previous one completes, only
    the latest is decoded (stale intermediate frames are discarded).
    """

    #: Emitted on the main thread once a frame is ready.
    frame_ready = Signal(int, int, object)  # cell_idx, frame_idx, ndarray

    def __init__(self, cells_info: list, cache, parent=None) -> None:
        super().__init__(parent)
        self._cells_info = cells_info
        self._cache = cache
        self._pending: dict[int, int] = {}
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._stop = False

    def request(self, cell_idx: int, frame_idx: int) -> None:
        """Queue a decode request; supersedes any earlier request for the same cell."""
        with self._lock:
            self._pending[cell_idx] = frame_idx
        self._wakeup.set()

    def shutdown(self) -> None:
        """Stop the thread and wait for it to finish (up to 3 s)."""
        self._stop = True
        self._wakeup.set()
        self.wait(3000)

    def run(self) -> None:
        while not self._stop:
            self._wakeup.wait()
            if self._stop:
                break
            self._wakeup.clear()
            with self._lock:
                batch = dict(self._pending)
                self._pending.clear()
            for cell_idx, frame_idx in batch.items():
                if self._stop:
                    break
                info = self._cells_info[cell_idx]
                frame_idx = max(0, min(frame_idx, info.total_frames - 1))
                try:
                    img = self._cache.get(
                        CacheKey(
                            shot_video_id=info.shot_video_id,
                            frame_idx=frame_idx,
                            cache_type=CacheType.FULL_FRAME,
                        ),
                        file_path=info.file_path,
                    )
                    self.frame_ready.emit(cell_idx, frame_idx, img)
                except Exception:  # noqa: BLE001
                    pass


class MultiVideoScrubber(QWidget):
    """Grid of ``CameraCell`` widgets for multi-camera video scrubbing.

    Parameters
    ----------
    cells_info:
        One ``CellInfo`` per camera; determines grid size and video sources.
    cache:
        ``FrameCache`` instance used to serve decoded frames.
    parent:
        Parent widget.
    """

    #: Emitted whenever any cell's frame changes: (cell_idx, frame_idx).
    frame_changed = Signal(int, int)

    def __init__(
        self,
        cells_info: list[CellInfo],
        cache: FrameCache,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cells_info = cells_info
        self._cache = cache
        self._sync_table: SyncTable | None = None
        self._focused_cell: int = 0
        self._current_frames: list[int] = [0] * len(cells_info)
        self._current_timestamp: float = 0.0

        # Play/pause timer
        self._playing = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)

        # Background frame decoder
        self._decoder = _FrameDecoder(cells_info, cache, self)
        self._decoder.frame_ready.connect(self._on_frame_ready)
        self._decoder.start()

        # Build grid layout — each column is a container with the video cell
        # on top and a (slider + frame label) row below.
        layout = QGridLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)
        self._cells: list[CameraCell] = []
        self._sliders: list[QSlider] = []
        self._frame_labels: list[QLabel] = []
        n = len(cells_info)
        cols = max(1, math.ceil(math.sqrt(n)))
        for i, info in enumerate(cells_info):
            container = QWidget(self)
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(2)

            cell = CameraCell(label=info.label, parent=container)
            cell.clicked.connect(lambda idx=i: self._on_cell_clicked(idx))
            vbox.addWidget(cell, stretch=1)

            # Slider + frame counter row
            slider_row = QHBoxLayout()
            slider_row.setContentsMargins(0, 0, 0, 0)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(max(info.total_frames - 1, 0))
            slider.setValue(0)
            slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            slider.sliderMoved.connect(lambda val, idx=i: self._on_cell_slider_moved(idx, val))
            slider_row.addWidget(slider)

            last = info.total_frames - 1
            frame_label = QLabel(f"0 / {last}")
            frame_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            frame_label.setStyleSheet("font-family: monospace; font-size: 11px;")
            # Size to the widest possible text so it never truncates
            worst = f"{last} / {last}"
            frame_label.setFixedWidth(
                frame_label.fontMetrics().horizontalAdvance(worst) + 8
            )
            slider_row.addWidget(frame_label)

            vbox.addLayout(slider_row)

            layout.addWidget(container, i // cols, i % cols)
            self._cells.append(cell)
            self._sliders.append(slider)
            self._frame_labels.append(frame_label)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Mark cell 0 as selected by default
        if self._cells:
            self._cells[0].set_selected(True)
        # Kick off an initial decode for frame 0 of every cell
        for i in range(len(cells_info)):
            self._decoder.request(i, 0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def focused_cell(self) -> int:
        return self._focused_cell

    @property
    def sync_table(self) -> SyncTable | None:
        return self._sync_table

    @property
    def current_frames(self) -> list[int]:
        """Current frame index for each cell (read-only copy)."""
        return list(self._current_frames)

    @property
    def current_timestamp(self) -> float:
        return self._current_timestamp

    # ------------------------------------------------------------------
    # Navigation API
    # ------------------------------------------------------------------

    def seek_synced(self, timestamp_s: float) -> None:
        """Move all cameras to *timestamp_s* via the sync table.

        No-op if no sync table is loaded.
        """
        if self._sync_table is None:
            return
        self._current_timestamp = timestamp_s
        for i, info in enumerate(self._cells_info):
            frame_idx = self._sync_table.lookup(timestamp_s, info.shot_video_id)
            if frame_idx is not None:
                self._set_cell_frame(i, frame_idx)

    def seek_camera(self, cell_idx: int, frame_idx: int) -> None:
        """Move *cell_idx* to *frame_idx* independently of the sync table."""
        if not (0 <= cell_idx < len(self._cells)):
            return
        self._set_cell_frame(cell_idx, frame_idx)

    @Slot(object)
    def reload_sync(self, sync_table: SyncTable | None) -> None:
        """Update the sync source and immediately re-render all cells."""
        self._sync_table = sync_table
        if sync_table is not None:
            self.seek_synced(self._current_timestamp)

    def set_overlays(self, cell_idx: int, overlays: list[Overlay]) -> None:
        """Replace the overlay list for *cell_idx*."""
        if 0 <= cell_idx < len(self._cells):
            self._cells[cell_idx].set_overlays(overlays)

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        key = event.key()
        mod = event.modifiers()
        step = 10 if (mod & Qt.KeyboardModifier.ShiftModifier) else 1

        if key == Qt.Key.Key_Left:
            self._step(-step)
        elif key == Qt.Key.Key_Right:
            self._step(step)
        elif key == Qt.Key.Key_Space:
            self._toggle_play()
        elif key == Qt.Key.Key_Home:
            self._go_to_frame(0)
        elif key == Qt.Key.Key_End:
            self._go_to_end()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop the background decoder — call before deleting the widget."""
        self._play_timer.stop()
        self._decoder.shutdown()

    def _set_cell_frame(self, cell_idx: int, frame_idx: int) -> None:
        """Clamp frame index, update slider/label immediately, request async decode."""
        info = self._cells_info[cell_idx]
        frame_idx = max(0, min(frame_idx, info.total_frames - 1))
        self._current_frames[cell_idx] = frame_idx
        self._sliders[cell_idx].setValue(frame_idx)
        self._frame_labels[cell_idx].setText(f"{frame_idx} / {info.total_frames - 1}")
        self._decoder.request(cell_idx, frame_idx)
        self.frame_changed.emit(cell_idx, frame_idx)

    def _on_cell_slider_moved(self, cell_idx: int, value: int) -> None:
        """User dragged a per-cell slider — focus that cell and seek it."""
        for i, cell in enumerate(self._cells):
            cell.set_selected(i == cell_idx)
        self._focused_cell = cell_idx
        self._set_cell_frame(cell_idx, value)
        self.setFocus()

    def _on_frame_ready(self, cell_idx: int, frame_idx: int, img) -> None:
        """Display *img* only if *frame_idx* is still the current frame."""
        if self._current_frames[cell_idx] == frame_idx:
            self._cells[cell_idx].set_frame(img)

    def _step(self, delta: int) -> None:
        if self._sync_table is not None:
            ref_fps = self._cells_info[0].fps if self._cells_info else 30.0
            self.seek_synced(self._current_timestamp + delta / max(ref_fps, 1.0))
        else:
            fc = self._focused_cell
            if 0 <= fc < len(self._cells):
                self.seek_camera(fc, self._current_frames[fc] + delta)

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        if self._playing:
            ref_fps = self._cells_info[0].fps if self._cells_info else 30.0
            interval_ms = max(1, int(1000.0 / ref_fps))
            self._play_timer.start(interval_ms)
        else:
            self._play_timer.stop()

    def _go_to_frame(self, frame: int) -> None:
        if self._sync_table is not None:
            ref_fps = self._cells_info[0].fps if self._cells_info else 30.0
            self.seek_synced(frame / max(ref_fps, 1.0))
        else:
            self.seek_camera(self._focused_cell, frame)

    def _go_to_end(self) -> None:
        if self._sync_table is not None and self._cells_info:
            ref = self._cells_info[0]
            self.seek_synced((ref.total_frames - 1) / max(ref.fps, 1.0))
        elif 0 <= self._focused_cell < len(self._cells_info):
            info = self._cells_info[self._focused_cell]
            self.seek_camera(self._focused_cell, info.total_frames - 1)

    def _on_cell_clicked(self, cell_idx: int) -> None:
        for i, cell in enumerate(self._cells):
            cell.set_selected(i == cell_idx)
        self._focused_cell = cell_idx
        self.setFocus()

    def _on_play_tick(self) -> None:
        self._step(1)
        # Auto-stop at end of reference camera
        if self._cells_info:
            ref = self._cells_info[0]
            if self._sync_table is not None:
                if self._current_timestamp >= (ref.total_frames - 1) / max(ref.fps, 1.0):
                    self._toggle_play()
            else:
                fc = self._focused_cell
                if 0 <= fc < len(self._cells_info):
                    if self._current_frames[fc] >= self._cells_info[fc].total_frames - 1:
                        self._toggle_play()
