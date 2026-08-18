# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""page_sync.py — Wizard page for camera synchronisation.

Workflow
--------
1.  The user selects a shot from the combo box.  The MultiVideoScrubber shows
    all camera feeds independently.
2.  **Rough sync** (this page) — the user scrolls each camera to the same
    physical event and presses "Set anchor" for each.  Per-camera fps overrides
    are available here to handle slow-motion clips whose container fps differs
    from the actual capture rate.  "Apply rough sync" writes a
    ``sync_config(method="manual-rough")`` and switches the scrubber into synced
    mode so all cameras track the same global timeline.
3.  **LED sync** (optional, via "LED synchronisation…" button) — opens a dialog
    that automates synchronisation using a blinking LED: the user draws a small
    ROI over the LED in each camera, runs the background job, reviews quality
    metrics and a brightness plot, then accepts to write a
    ``sync_config(method="led-auto")`` that supersedes the rough sync.

Design notes
-----------
- The rough sync panel is always visible; the LED sync dialog is modal and
  opened on demand so the scrubber behind it remains accessible as a reference.
- Per-camera fps overrides are stored on the page and forwarded to the LED sync
  dialog; changing them here before applying rough sync also affects the anchor
  timestamp calculation.
- In synced mode, per-cell sliders convert the dragged frame to a global
  timestamp via ``SyncTable.frame_to_global_time`` so all cameras follow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger(__name__)

from PySide6.QtCore import QEvent, QObject, QThread, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from app.setup.camera_cell import CameraCell
from app.setup.db_context import (
    CaptureVideoInfo,
    DBContext,
    SyncAnchorObservation,
    SyncPoint,
    SyncTable,
)
from posetrak.db.db import generate_id as _generate_id
from app.setup.video_reader import FrameReader
from app.setup.job_runner import BackgroundJob
from app.setup.led_sync import (
    CameraSyncResult,
    LedPairwiseResult,
    LedSyncResult,
    PairMatchResult,
    ROI,
    extract_brightness_changes,
    run_led_sync,
    run_led_sync_pairwise,
    save_brightness_dump,
)
from app.setup.overlay import ROIDrawOverlay
from app.setup.pair_scrubber import PairScrubber
from app.setup.sync_solver import (
    check_connectivity,
    detect_pair_fps_inconsistency,
    solve_sync_graph,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as _FigureCanvas
    from matplotlib.figure import Figure as _Figure
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


@dataclass
class _ShotMeta:
    shot_id: str
    label: str
    videos: list  # list of ShotVideoInfo


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------


class _ClickCaptureOverlay:
    """Fires a callback once on the first mouse press, then acts as a no-op."""

    def __init__(self, on_click) -> None:
        self._on_click = on_click

    def paint(self, painter, fw, fh, cw, ch) -> None:  # noqa: ARG002
        pass

    def mouse_press(self, x: int, y: int) -> None:
        self._on_click(x, y)

    def mouse_move(self, x: int, y: int) -> None:  # noqa: ARG002
        pass

    def mouse_release(self, x: int, y: int) -> None:  # noqa: ARG002
        pass


class _RepaintingROIOverlay:
    """Wraps ``ROIDrawOverlay`` and repaints the cell after each mouse event."""

    def __init__(self, cell: CameraCell) -> None:
        self._inner = ROIDrawOverlay(active=True)
        self._cell = cell

    @property
    def roi(self) -> object:
        return self._inner.roi

    def clear(self) -> None:
        self._inner.clear()

    def paint(self, painter, fw, fh, cw, ch) -> None:
        self._inner.paint(painter, fw, fh, cw, ch)

    def mouse_press(self, x: int, y: int) -> None:
        self._inner.mouse_press(x, y)
        self._cell.update()

    def mouse_move(self, x: int, y: int) -> None:
        self._inner.mouse_move(x, y)
        self._cell.update()

    def mouse_release(self, x: int, y: int) -> None:
        self._inner.mouse_release(x, y)
        self._cell.update()


# ---------------------------------------------------------------------------
# ROI selection dialog
# ---------------------------------------------------------------------------


class _ROISelectDialog(QDialog):
    """Two-phase LED ROI selector.

    Phase 1 — full view
        The full video frame is shown.  Clicking anywhere sets the zoom centre.

    Phase 2 — zoom view
        A 400 × 300 pixel window centred on the click is shown (clamped to the
        frame boundary).  The user scrubs frames and draws a rectangle around
        the LED.  "Back" (or ESC) returns to phase 1.

    Keyboard navigation works in both phases:
    ``←`` / ``→`` step ±1 frame, ``Shift+←`` / ``Shift+→`` step ±10 frames.
    """

    _PHASE_FULL = 1
    _PHASE_ZOOM = 2
    _ZOOM_HW = 200   # half-width of zoom window in video-frame pixels
    _ZOOM_HH = 150   # half-height

    def __init__(
        self,
        file_path: str,
        total_frames: int,
        fps: float,
        initial_frame: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select LED Region of Interest")
        self.setModal(True)
        self.resize(680, 560)

        self._file_path = file_path
        self._total_frames = max(total_frames, 1)
        self._fps = fps
        self._phase = self._PHASE_FULL
        self._current_frame_idx = initial_frame
        self._full_frame = None
        self._zoom_rect: tuple[int, int, int, int] | None = None
        self._roi_overlay: _RepaintingROIOverlay | None = None
        self._confirmed_roi: ROI | None = None
        self._roi_check_timer = None

        # --- widgets ---
        self._instruction = QLabel()
        self._instruction.setWordWrap(True)

        self._cell = CameraCell(label="Loading…", parent=self)
        self._cell.setMinimumSize(640, 360)
        self._cell.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._cell.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(self._total_frames - 1)
        self._slider.setValue(initial_frame)
        self._slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider.sliderMoved.connect(self._on_slider_moved)

        self._frame_label = QLabel(f"Frame {initial_frame}")
        self._frame_label.setStyleSheet("font-size: 11px; font-family: monospace;")

        slider_row = QHBoxLayout()
        slider_row.addWidget(self._slider)
        slider_row.addWidget(self._frame_label)

        self._back_btn = QPushButton("← Back to full view")
        self._back_btn.setVisible(False)
        self._back_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._back_btn.clicked.connect(self._enter_phase_full)

        self._confirm_btn = QPushButton("Confirm ROI")
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._confirm_btn.clicked.connect(self._on_confirm)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._cancel_btn.clicked.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._back_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._confirm_btn)
        btn_row.addWidget(self._cancel_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self._instruction)
        layout.addWidget(self._cell, stretch=1)
        layout.addLayout(slider_row)
        layout.addLayout(btn_row)

        # Dialog keeps keyboard focus so arrow keys always navigate frames
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # --- background frame reader ---
        self._reader = FrameReader(file_path, self)
        self._reader.frame_ready.connect(self._on_frame_ready)
        self._reader.start()

        self._enter_phase_full()
        self._reader.request(initial_frame)

    # ------------------------------------------------------------------
    # Public result
    # ------------------------------------------------------------------

    def selected_roi(self) -> ROI | None:
        return self._confirmed_roi

    # ------------------------------------------------------------------
    # Phase transitions
    # ------------------------------------------------------------------

    def _enter_phase_full(self) -> None:
        self._stop_roi_timer()
        self._phase = self._PHASE_FULL
        self._zoom_rect = None
        self._roi_overlay = None
        self._cell.set_overlays([_ClickCaptureOverlay(self._on_phase1_click)])
        self._instruction.setText(
            "<b>Phase 1 — click on the LED location</b><br>"
            "The view will zoom in so you can draw a precise rectangle.  "
            "Use ←/→ to scrub frames (Shift for ×10)."
        )
        self._back_btn.setVisible(False)
        self._confirm_btn.setEnabled(False)
        if self._full_frame is not None:
            self._cell.set_frame(self._full_frame)
        self.setFocus()

    def _on_phase1_click(self, fx: int, fy: int) -> None:
        if self._full_frame is None:
            return
        fh, fw = self._full_frame.shape[:2]
        hw, hh = self._ZOOM_HW, self._ZOOM_HH
        x1 = max(0, fx - hw)
        x2 = min(fw, fx + hw)
        y1 = max(0, fy - hh)
        y2 = min(fh, fy + hh)
        self._zoom_rect = (x1, y1, x2, y2)
        self._enter_phase_zoom()

    def _enter_phase_zoom(self) -> None:
        self._phase = self._PHASE_ZOOM
        self._roi_overlay = _RepaintingROIOverlay(self._cell)
        self._cell.set_overlays([self._roi_overlay])
        self._instruction.setText(
            "<b>Phase 2 — draw a rectangle around the LED</b><br>"
            "Drag to mark the LED area.  Use ←/→ to scrub (Shift for ×10).  "
            "ESC or ← Back to return to the full view."
        )
        self._back_btn.setVisible(True)
        self._confirm_btn.setEnabled(False)
        if self._full_frame is not None:
            self._show_cropped(self._full_frame)
        self._start_roi_timer()
        self.setFocus()

    def _start_roi_timer(self) -> None:
        from PySide6.QtCore import QTimer
        self._roi_check_timer = QTimer(self)
        self._roi_check_timer.setInterval(100)
        self._roi_check_timer.timeout.connect(self._check_roi_valid)
        self._roi_check_timer.start()

    def _stop_roi_timer(self) -> None:
        if self._roi_check_timer is not None:
            self._roi_check_timer.stop()
            self._roi_check_timer = None

    def _check_roi_valid(self) -> None:
        if self._roi_overlay is not None:
            roi = self._roi_overlay.roi
            self._confirm_btn.setEnabled(roi is not None and roi.is_valid)

    def _show_cropped(self, full_frame) -> None:
        if self._zoom_rect is None:
            return
        x1, y1, x2, y2 = self._zoom_rect
        crop = full_frame[y1:y2, x1:x2]
        if crop.size > 0:
            self._cell.set_frame(crop)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_slider_moved(self, value: int) -> None:
        self._current_frame_idx = value
        self._frame_label.setText(f"Frame {value}")
        self._reader.request(value)

    def _on_frame_ready(self, frame_idx: int, frame) -> None:
        if frame_idx != self._current_frame_idx:
            return
        self._full_frame = frame
        if self._phase == self._PHASE_FULL:
            self._cell.set_frame(frame)
        else:
            self._show_cropped(frame)

    def _on_confirm(self) -> None:
        if self._roi_overlay is None or self._zoom_rect is None:
            return
        inner_roi = self._roi_overlay.roi
        if inner_roi is None or not inner_roi.is_valid:
            return
        x1_crop, y1_crop, _, _ = self._zoom_rect
        n = inner_roi.normalised
        self._confirmed_roi = ROI(
            x1=n.x1 + x1_crop,
            y1=n.y1 + y1_crop,
            x2=n.x2 + x1_crop,
            y2=n.y2 + y1_crop,
        )
        self.accept()

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        key = event.key()
        mod = event.modifiers()
        step = 10 if (mod & Qt.KeyboardModifier.ShiftModifier) else 1

        if key == Qt.Key.Key_Left:
            new_frame = max(0, self._current_frame_idx - step)
            self._slider.setValue(new_frame)
            self._on_slider_moved(new_frame)
        elif key == Qt.Key.Key_Right:
            new_frame = min(self._total_frames - 1, self._current_frame_idx + step)
            self._slider.setValue(new_frame)
            self._on_slider_moved(new_frame)
        elif key == Qt.Key.Key_Escape:
            if self._phase == self._PHASE_ZOOM:
                self._enter_phase_full()
            else:
                self.reject()
        else:
            super().keyPressEvent(event)

    def done(self, result: int) -> None:
        """Ensure the background thread is stopped before the dialog finishes."""
        self._stop_roi_timer()
        if self._reader.isRunning():
            self._reader.shutdown()
        super().done(result)


# ---------------------------------------------------------------------------
# LED sync background job
# ---------------------------------------------------------------------------


class _LedSyncJob(BackgroundJob):
    """Extracts per-camera brightness signals then runs the pairwise LED sync.

    Parameters
    ----------
    cam_data:
        List of ``(file_path, roi, fps_override, cam_id, video_id)`` tuples.
    event_cfg:
        Event-detection parameters forwarded to ``run_led_sync_pairwise``.
    rough_offsets:
        Per-camera rough time offsets (global time at local frame 0).
        Used to bootstrap event matching windows between pairs.
    """

    def __init__(
        self,
        cam_data: list[tuple[str, ROI, float, str, str]],
        event_cfg: dict | None = None,
        rough_offsets: list[float] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._cam_data = cam_data
        self._event_cfg = event_cfg
        self._rough_offsets = rough_offsets

    def run(self) -> None:
        K = len(self._cam_data)
        signals, fps_list, cam_ids, video_ids = [], [], [], []

        for k, (file_path, roi, fps_override, cam_id, video_id) in enumerate(self._cam_data):
            base_pct = k * 80 // K
            end_pct = (k + 1) * 80 // K
            self.progress.emit(base_pct, f"Extracting brightness for {cam_id}…")

            def _prog(fi: int, total: int, _b=base_pct, _e=end_pct) -> None:
                if total > 0:
                    self.progress.emit(_b + (_e - _b) * fi // total, "")

            sig, fps = extract_brightness_changes(
                file_path, roi, fps_override=fps_override, progress_cb=_prog,
            )
            signals.append(sig)
            fps_list.append(fps)
            cam_ids.append(cam_id)
            video_ids.append(video_id)

        self.progress.emit(82, "Running pairwise LED sync…")
        result = run_led_sync_pairwise(
            signals=signals, fps_list=fps_list,
            cam_ids=cam_ids, video_ids=video_ids,
            rough_offsets=self._rough_offsets, event_cfg=self._event_cfg,
        )
        self.progress.emit(100, "Done.")
        self.finished.emit(result)


# ---------------------------------------------------------------------------
# Brightness plot dialog
# ---------------------------------------------------------------------------


class _BrightnessPlotDialog(QDialog):
    """Shows LED brightness for all cameras on an approximate shared timeline.

    The x-axis uses each camera's rough global-time offset so signals are
    approximately aligned even before the graph solver runs.
    """

    def __init__(
        self,
        result: LedPairwiseResult,
        rough_offsets: dict[str, float],
        labels: dict[str, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("LED Brightness — Approximate Global Timeline")
        self.resize(860, 400)

        labels = labels or {}
        import numpy as _np

        layout = QVBoxLayout(self)
        if _HAS_MATPLOTLIB:
            fig = _Figure(figsize=(10, 4), dpi=90, tight_layout=True)
            ax = fig.add_subplot(1, 1, 1)
            colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
            for i, (vid, brightness) in enumerate(result.brightness_by_vid.items()):
                fps = result.fps_by_vid.get(vid, 30.0)
                offset = rough_offsets.get(vid, 0.0)
                t_local = _np.arange(len(brightness)) / fps
                lname = labels.get(vid, vid[:8])
                ax.plot(t_local + offset, brightness, label=lname,
                        color=colors[i % len(colors)], linewidth=0.8, alpha=0.85)
            ax.set_xlabel("Approximate global time (s)")
            ax.set_ylabel("Brightness change")
            ax.set_title("LED brightness — approximate global timeline (rough offsets applied)")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)
            canvas = _FigureCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            layout.addWidget(canvas)
        else:
            layout.addWidget(QLabel(
                "matplotlib is not installed — cannot display brightness plot.\n"
                "Install it with: pip install matplotlib"
            ))

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(close_btn)
        layout.addLayout(row)


# ---------------------------------------------------------------------------
# LED sync dialog
# ---------------------------------------------------------------------------


def _sync_points_from_led_result(
    result: LedSyncResult,
) -> tuple[dict[str, list[SyncPoint]], dict[str, float]]:
    """Convert a ``LedSyncResult`` to sync points and fps map for the DB.

    Every frame is stored as a sync point (matching the marimo-notebook
    behaviour).  The dict is keyed by ``shot_video_id`` — not by
    ``camera_instance_id`` — so that cameras sharing the same
    ``camera_instance_id`` (e.g. ``__unassigned__``) do not overwrite each
    other's entries, and the DB primary-key ``(sync_config_id,
    camera_instance_id, video_frame)`` stays collision-free because we use
    the unique video ID in the ``camera_instance_id`` column.
    """
    points: dict[str, list[SyncPoint]] = {}
    fps_by_video: dict[str, float] = {}

    for cr in result.cameras:
        N = len(cr.frame_times)
        if N == 0:
            continue
        points[cr.shot_video_id] = [
            SyncPoint(
                # Use shot_video_id here so the DB PK never collides when
                # multiple cameras still carry camera_instance_id="__unassigned__".
                camera_instance_id=cr.shot_video_id,
                shot_video_id=cr.shot_video_id,
                video_frame=i,
                timestamp_s=float(cr.frame_times[i]),
            )
            for i in range(N)
        ]
        t_range = abs(float(cr.frame_times[-1]) - float(cr.frame_times[0]))
        eff_fps = (N - 1) / t_range if N > 1 and t_range > 1e-9 else cr.fps_used
        fps_by_video[cr.shot_video_id] = eff_fps
        _log.debug(
            "sync_points [%s / %s]: %d points, "
            "video_frame 0..%d, timestamp %.4f – %.4f s, "
            "effective fps %.3f",
            cr.camera_instance_id, cr.shot_video_id,
            N, N - 1,
            float(cr.frame_times[0]), float(cr.frame_times[-1]),
            eff_fps,
        )

    return points, fps_by_video


class _LedSyncDialog(QDialog):
    """Dialog for automated LED-based synchronisation.

    Opened from the sync page after rough sync has been applied.  Contains
    per-camera ROI selection, fps overrides (pre-filled from rough-sync values),
    a background sync job, quality metrics, a brightness plot, and an accept
    button that writes ``sync_config(method="led-auto")`` to the session DB.

    Parameters
    ----------
    shot:
        Shot metadata including video file paths and camera IDs.
    fps_overrides:
        Per-cell fps override values from the rough sync panel.  The user can
        further adjust them here.
    ctx:
        DB context for writing the accepted sync config.
    current_frames:
        Current frame positions from the scrubber, used as starting frames in
        the ROI dialog.
    on_sync_accepted:
        Callback called with the new ``SyncTable`` when the user accepts.
    rough_offsets:
        Per-camera-index rough sync offset in seconds (global time at local frame 0).
        Derived from the rough sync anchors.  Used to seed DTW when cameras start
        far apart (> ``dtw_band_s``).  Index matches ``shot.videos`` order.
    """

    def __init__(
        self,
        shot: _ShotMeta,
        fps_overrides: dict[int, float],
        ctx: DBContext,
        current_frames: list[int],
        on_sync_accepted,
        anchor_frames: dict[int, int] | None = None,
        sync_table_offsets: dict[int, float] | None = None,
        manual_anchors: list | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"LED Synchronisation — {shot.label}")
        self.resize(640, 480)

        self._shot = shot
        self._ctx = ctx
        self._current_frames = current_frames
        self._on_sync_accepted = on_sync_accepted
        self._anchor_frames = anchor_frames or {}
        # Rough offsets derived from the loaded sync table (more reliable than
        # recomputing from anchor_frames when some cameras were not manually
        # anchored in the current session).
        self._sync_table_offsets = sync_table_offsets or {}
        self._manual_anchors: list = manual_anchors or []

        self._led_rois: dict[int, ROI] = {}
        self._led_result: LedPairwiseResult | None = None
        self._rough_offsets_by_vid: dict[str, float] = {}
        self._led_job: _LedSyncJob | None = None
        self._video_labels: dict[str, str] = {sv.id: sv.camera_label for sv in shot.videos}

        layout = QVBoxLayout(self)

        # --- per-camera rows ---
        cam_group = QGroupBox("Camera setup")
        cam_layout = QVBoxLayout(cam_group)
        cam_layout.setSpacing(3)
        self._roi_labels: list[QLabel] = []
        self._fps_spinboxes: list[QDoubleSpinBox] = []

        for cell_idx, sv in enumerate(shot.videos):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            cam_lbl = QLabel(sv.camera_label)
            cam_lbl.setFixedWidth(70)
            cam_lbl.setStyleSheet("font-weight: bold; font-size: 11px;")

            roi_lbl = QLabel("ROI: not set")
            roi_lbl.setStyleSheet("color: grey; font-size: 11px;")
            roi_lbl.setMinimumWidth(200)

            set_roi_btn = QPushButton("Set ROI…")
            set_roi_btn.setFixedWidth(80)
            set_roi_btn.clicked.connect(
                lambda _checked=False, idx=cell_idx: self._on_set_roi(idx)
            )

            fps_lbl = QLabel("fps:")
            fps_lbl.setStyleSheet("font-size: 11px;")

            fps_spin = QDoubleSpinBox()
            fps_spin.setRange(0.0, 960.0)
            fps_spin.setDecimals(3)
            fps_spin.setValue(fps_overrides.get(cell_idx, sv.actual_fps or 30.0))
            fps_spin.setFixedWidth(75)
            fps_spin.setSpecialValueText("auto")
            fps_spin.setToolTip(
                "Override the fps used for event detection.\n"
                "Set to 0 to use the probed container fps.\n"
                "Use the actual capture rate for slow-motion clips."
            )

            row_layout.addWidget(cam_lbl)
            row_layout.addWidget(roi_lbl, stretch=1)
            row_layout.addWidget(set_roi_btn)
            row_layout.addWidget(fps_lbl)
            row_layout.addWidget(fps_spin)

            cam_layout.addWidget(row)
            self._roi_labels.append(roi_lbl)
            self._fps_spinboxes.append(fps_spin)

        layout.addWidget(cam_group)

        # --- run section ---
        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run LED sync")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        self._progress_label = QLabel()
        self._progress_label.setStyleSheet("font-size: 11px;")
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._progress_bar, stretch=1)
        run_row.addWidget(self._progress_label)
        layout.addLayout(run_row)

        # --- quality section ---
        self._quality_widget = QWidget()
        self._quality_layout = QVBoxLayout(self._quality_widget)
        self._quality_layout.setContentsMargins(0, 4, 0, 0)
        self._quality_layout.setSpacing(2)
        self._quality_widget.setVisible(False)
        layout.addWidget(self._quality_widget)

        # --- accept / close row ---
        accept_row = QHBoxLayout()
        self._accept_label = QLabel()
        self._accept_label.setStyleSheet("font-size: 11px;")
        self._plot_btn = QPushButton("Show brightness plot")
        self._plot_btn.setEnabled(False)
        self._plot_btn.clicked.connect(self._on_show_plot)
        self._dump_btn = QPushButton("Dump brightness data…")
        self._dump_btn.setEnabled(False)
        self._dump_btn.setToolTip(
            "Save per-camera brightness signals to a .npz file.\n"
            "Use load_brightness_dump() in tests or notebooks to reproduce\n"
            "and iterate on the sync algorithm without re-reading videos."
        )
        self._dump_btn.clicked.connect(self._on_dump)
        self._accept_btn = QPushButton("Accept LED sync")
        self._accept_btn.setEnabled(False)
        self._accept_btn.clicked.connect(self._on_accept)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        accept_row.addWidget(self._accept_label, stretch=1)
        accept_row.addWidget(self._dump_btn)
        accept_row.addWidget(self._plot_btn)
        accept_row.addWidget(self._accept_btn)
        accept_row.addWidget(close_btn)
        layout.addLayout(accept_row)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_set_roi(self, cell_idx: int) -> None:
        sv = self._shot.videos[cell_idx]
        total_frames = max(sv.last_video_frame - sv.first_video_frame + 1, 1)
        initial = self._current_frames[cell_idx] if cell_idx < len(self._current_frames) else 0

        dlg = _ROISelectDialog(
            file_path=sv.file_path,
            total_frames=total_frames,
            fps=sv.actual_fps or 30.0,
            initial_frame=initial,
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            roi = dlg.selected_roi()
            if roi is not None and roi.is_valid:
                self._led_rois[cell_idx] = roi
                self._roi_labels[cell_idx].setText(
                    f"ROI: ({roi.x1},{roi.y1})→({roi.x2},{roi.y2})"
                )
                self._roi_labels[cell_idx].setStyleSheet("color: green; font-size: 11px;")
                self._update_run_btn()

    def _compute_rough_offsets(self) -> list[float]:
        """Return per-camera rough time offsets (global time at each camera's frame 0).

        Strategy (in order of preference):
        1. For cameras that were manually anchored in the current session
           (present in ``_anchor_frames``): recompute from the anchor frame and
           the current fps spinbox so that any fps correction made here is
           honoured.
        2. For cameras NOT in ``_anchor_frames``: use the offset derived from
           the loaded sync table (``_sync_table_offsets``).  This covers the
           common case where the user loaded a previous rough-sync config from
           the dropdown without re-anchoring all cameras manually.
        3. Fall back to 0.0 (assume camera is synchronous with reference).
        """
        K = len(self._shot.videos)

        # Build anchor-based offsets for cameras that have a manual anchor.
        anchor_based: dict[int, float] = {}
        if self._anchor_frames:
            ref_cell = min(self._anchor_frames)
            ref_frame = self._anchor_frames[ref_cell]
            ref_fps = self._fps_spinboxes[ref_cell].value() or self._shot.videos[ref_cell].actual_fps or 30.0
            ref_ts = ref_frame / ref_fps
            for cell_idx, sv in enumerate(self._shot.videos):
                if cell_idx in self._anchor_frames:
                    fps = self._fps_spinboxes[cell_idx].value() or sv.actual_fps or 30.0
                    anchor_based[cell_idx] = ref_ts - self._anchor_frames[cell_idx] / fps

        offsets: list[float] = []
        for cell_idx in range(K):
            if cell_idx in anchor_based:
                offsets.append(anchor_based[cell_idx])
            elif cell_idx in self._sync_table_offsets:
                offsets.append(self._sync_table_offsets[cell_idx])
            else:
                offsets.append(0.0)
        return offsets

    def _update_run_btn(self) -> None:
        self._run_btn.setEnabled(len(self._led_rois) >= 2)

    def _on_run(self) -> None:
        cam_data = []
        for cell_idx, sv in enumerate(self._shot.videos):
            roi = self._led_rois.get(cell_idx)
            if roi is None:
                continue
            fps_val = self._fps_spinboxes[cell_idx].value()
            fps_override = fps_val if fps_val > 0 else sv.actual_fps or 30.0
            cam_data.append((sv.file_path, roi, fps_override, sv.camera_instance_id, sv.id))

        if len(cam_data) < 2:
            return

        self._run_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._quality_widget.setVisible(False)
        self._accept_btn.setEnabled(False)
        self._plot_btn.setEnabled(False)
        self._dump_btn.setEnabled(False)
        self._accept_label.setText("")

        # Compute rough offsets using the CURRENT fps spinbox values so that a
        # fps correction made in this dialog is reflected in the time mapping.
        rough_offsets_list = self._compute_rough_offsets()
        _log.debug("LED sync rough_offsets (recomputed at run time): %s", rough_offsets_list)

        # Store per-video rough offsets for the brightness plot.
        self._rough_offsets_by_vid = {
            sv.id: rough_offsets_list[k]
            for k, sv in enumerate(self._shot.videos)
            if k < len(rough_offsets_list)
        }

        self._led_job = _LedSyncJob(
            cam_data, rough_offsets=rough_offsets_list, parent=self,
        )
        self._led_job.progress.connect(self._on_progress)
        self._led_job.finished.connect(self._on_done)
        self._led_job.error.connect(self._on_error)
        self._led_job.start()

    def _on_progress(self, pct: int, msg: str) -> None:
        self._progress_bar.setValue(pct)
        if msg:
            self._progress_label.setText(msg)

    def _on_done(self, result: LedPairwiseResult) -> None:
        self._progress_bar.setVisible(False)
        self._progress_label.setText("")
        self._led_result = result
        self._video_labels = {sv.id: sv.camera_label for sv in self._shot.videos}
        self._run_btn.setEnabled(True)

        while self._quality_layout.count():
            item = self._quality_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Per-camera event count header
        for vid, t_ev in result.events_by_vid.items():
            cam_name = self._video_labels.get(vid, vid[:8])
            lbl = QLabel(f"{cam_name}: {len(t_ev)} events detected")
            lbl.setStyleSheet("color: grey; font-size: 11px;")
            self._quality_layout.addWidget(lbl)

        # Per-pair matching quality
        for (vid_a, vid_b), pair in result.pairs.items():
            name_a = self._video_labels.get(vid_a, vid_a[:8])
            name_b = self._video_labels.get(vid_b, vid_b[:8])
            if not pair.success:
                txt = (
                    f"{name_a} ↔ {name_b}: "
                    f"{pair.n_events_a} / {pair.n_events_b} events → "
                    f"{pair.n_pairs} pairs → {pair.n_inliers} inliers  — failed"
                )
                style = "color: red; font-size: 11px;"
                tooltip = (
                    "RANSAC could not find ≥ 2 consistent inliers for this pair.\n"
                    "Check that the LED ROI is correctly placed on the LED for both cameras\n"
                    "and that there is a sufficient overlap period with LED visible in both."
                )
            else:
                q = "good" if pair.resid_std_s < 0.005 else "poor"
                txt = (
                    f"{name_a} ↔ {name_b}: "
                    f"{pair.n_events_a} / {pair.n_events_b} events → "
                    f"{pair.n_pairs} pairs → {pair.n_inliers} inliers, "
                    f"σ={pair.resid_std_s * 1000:.1f} ms  — {q}"
                )
                style = (
                    "color: green; font-size: 11px;"
                    if q == "good" else "color: orange; font-size: 11px;"
                )
                tooltip = (
                    "Events: LED blink peaks detected in each camera's brightness signal.\n"
                    "Pairs: events matched between the two cameras.\n"
                    "Inliers: matched pairs consistent with the affine timing model\n"
                    "  (within 10 ms tolerance).\n"
                    f"Residual σ: std of inlier timing errors — < 5 ms is good."
                )
            lbl = QLabel(txt)
            lbl.setStyleSheet(style)
            lbl.setToolTip(tooltip)
            self._quality_layout.addWidget(lbl)

        any_success = any(p.success for p in result.pairs.values())
        self._quality_widget.setVisible(True)
        self._accept_btn.setEnabled(True)
        self._plot_btn.setEnabled(True)
        self._dump_btn.setEnabled(any_success)
        self._accept_label.setText(
            "Review quality above, then click Accept."
            if any_success else
            "No successful pairs — check ROI placement and re-run."
        )

    def _on_error(self, msg: str) -> None:
        self._progress_bar.setVisible(False)
        self._run_btn.setEnabled(True)
        self._accept_label.setText(f"Error: {msg}")
        self._accept_label.setStyleSheet("color: red; font-size: 11px;")

    def _on_accept(self) -> None:
        if self._led_result is None:
            return

        # Build combined observation list: LED pairs replace manual anchors for
        # covered camera pairs; manual anchors fill in pairs without LED data.
        combined = _build_combined_observations(self._led_result, self._manual_anchors)

        # Persist corrected fps values from the LED dialog spinboxes.
        for cell_idx, sv in enumerate(self._shot.videos):
            fps = self._fps_spinboxes[cell_idx].value() or sv.actual_fps or 30.0
            if abs(fps - (sv.actual_fps or 0.0)) > 0.01:
                self._ctx.update_shot_video_fps(sv.id, fps)

        # Reload videos so solve_sync_graph sees any updated fps values.
        videos = self._ctx.get_shot_videos(self._shot.shot_id)
        result = solve_sync_graph(combined, videos)

        if not result.sync_points:
            self._accept_label.setText(
                "Solver could not connect any cameras — "
                "check LED ROIs and manual anchors."
            )
            self._accept_label.setStyleSheet("color: red; font-size: 11px;")
            return

        points: dict[str, list[SyncPoint]] = {}
        for sp in result.sync_points:
            points.setdefault(sp.shot_video_id, []).append(sp)

        conn = self._ctx._conn
        conn.execute(
            "DELETE FROM sync_points WHERE sync_config_id IN "
            "(SELECT id FROM sync_configs WHERE shot_id = ? AND created_by = 'led-graph')",
            (self._shot.shot_id,),
        )
        conn.execute(
            "DELETE FROM sync_configs WHERE shot_id = ? AND created_by = 'led-graph'",
            (self._shot.shot_id,),
        )
        self._ctx.write_sync_config(self._shot.shot_id, "led-graph", points)
        conn.commit()

        fps_by_video = {
            v.id: result.effective_fps.get(v.id, v.actual_fps or 30.0)
            for v in videos
        }
        sync_table = SyncTable(result.sync_points, fps_by_video)
        self._on_sync_accepted(sync_table)

        n_conn = len(result.connected_video_ids)
        n_iso = len(result.isolated_video_ids)
        msg = f"LED sync accepted — {n_conn} camera(s) connected."
        if n_iso:
            isolated_names = [
                next((sv.camera_label for sv in self._shot.videos if sv.id == v), v[:8])
                for v in result.isolated_video_ids
            ]
            msg += f"  Isolated: {', '.join(isolated_names)}."
        self._accept_label.setText(msg)
        self._accept_label.setStyleSheet("color: green; font-size: 11px;")
        self._accept_btn.setEnabled(False)

    def _on_show_plot(self) -> None:
        if self._led_result is None:
            return
        if not _HAS_MATPLOTLIB:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "matplotlib not available",
                "Install matplotlib to enable the brightness plot:\n\n"
                "  uv sync --group setup-app",
            )
            return
        _BrightnessPlotDialog(
            self._led_result,
            rough_offsets=self._rough_offsets_by_vid,
            labels=self._video_labels,
            parent=self,
        ).exec()

    def _on_dump(self) -> None:
        if self._led_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save brightness dump",
            f"led_brightness_{self._shot.label}.npz",
            "NumPy archive (*.npz);;All files (*)",
        )
        if not path:
            return
        if not path.endswith(".npz"):
            path += ".npz"
        rough_offsets_list = self._compute_rough_offsets()
        try:
            # Build a minimal LedSyncResult-compatible dump from pairwise data.
            # Uses the first video as the nominal reference for the dump format.
            import numpy as _np
            vids = list(self._led_result.brightness_by_vid.keys())
            arrays: dict = {}
            for i, vid in enumerate(vids):
                arrays[f"signal_{i}"] = self._led_result.brightness_by_vid[vid]
            arrays["fps"] = _np.array(
                [self._led_result.fps_by_vid[v] for v in vids], dtype=float
            )
            arrays["cam_ids"] = _np.array(vids, dtype=object)
            arrays["video_ids"] = _np.array(vids, dtype=object)
            arrays["rough_offsets"] = _np.array(
                rough_offsets_list[:len(vids)] if rough_offsets_list else [0.0] * len(vids),
                dtype=float,
            )
            arrays["ref_camera_idx"] = _np.array(0)
            _np.savez(path, **arrays)
            self._accept_label.setText(f"Brightness data saved to {path}")
            self._accept_label.setStyleSheet("color: green; font-size: 11px;")
        except Exception as exc:  # noqa: BLE001
            self._accept_label.setText(f"Dump failed: {exc}")
            self._accept_label.setStyleSheet("color: red; font-size: 11px;")

    def done(self, result: int) -> None:
        if self._led_job is not None and self._led_job.isRunning():
            self._led_job.requestInterruption()
            self._led_job.wait(3000)
        super().done(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_combined_observations(
    led_result: LedPairwiseResult,
    manual_anchors: list[tuple[str, list]],
) -> list[tuple[str, list[SyncAnchorObservation]]]:
    """Combine LED inlier pairs with manual anchor observations.

    For every camera pair where LED RANSAC succeeded, the LED inlier event
    pairs replace any manual anchor observations that link the same two
    cameras — LED sync is more accurate (many sub-frame measurements vs. one
    manually chosen frame).  Manual anchors are retained for camera pairs
    where LED failed or was not attempted.

    The returned list is in the format expected by
    :func:`~app.setup.sync_solver.solve_sync_graph`.
    """
    # Canonical pair key: (vid_a, vid_b) in the order stored in led_result.pairs.
    # Build a set of unordered pairs covered by LED for fast lookup.
    led_covered: set[frozenset[str]] = {
        frozenset((vid_a, vid_b))
        for (vid_a, vid_b), pair in led_result.pairs.items()
        if pair.success
    }

    combined: list[tuple[str, list[SyncAnchorObservation]]] = []

    # 1. Add synthetic anchor observations from LED inlier pairs.
    for (vid_a, vid_b), pair in led_result.pairs.items():
        if not pair.success:
            continue
        fps_a, fps_b = pair.fps_a, pair.fps_b
        for i, (ta, tb) in enumerate(zip(pair.inlier_t_a, pair.inlier_t_b)):
            anchor_id = f"led_{vid_a[:6]}_{vid_b[:6]}_{i:04d}"
            frame_a_f = ta * fps_a
            frame_a = int(frame_a_f)
            subframe_a = float(frame_a_f - frame_a)
            frame_b_f = tb * fps_b
            frame_b = int(frame_b_f)
            subframe_b = float(frame_b_f - frame_b)
            combined.append((anchor_id, [
                SyncAnchorObservation(
                    id=anchor_id + "_a", sync_anchor_id=anchor_id,
                    shot_video_id=vid_a, video_frame=frame_a, subframe=subframe_a,
                ),
                SyncAnchorObservation(
                    id=anchor_id + "_b", sync_anchor_id=anchor_id,
                    shot_video_id=vid_b, video_frame=frame_b, subframe=subframe_b,
                ),
            ]))

    # 2. Add manual anchors for camera pairs not fully covered by LED.
    for anchor_id, obs_list in manual_anchors:
        # Retain observations whose connection to another camera in the same
        # anchor is NOT covered by a successful LED pair.
        filtered: list[SyncAnchorObservation] = []
        for obs in obs_list:
            other_vids = {o.shot_video_id for o in obs_list if o.shot_video_id != obs.shot_video_id}
            needs_manual = any(
                frozenset((obs.shot_video_id, other)) not in led_covered
                for other in other_vids
            )
            if needs_manual:
                filtered.append(obs)
        if len(filtered) >= 2:
            combined.append((anchor_id, filtered))

    return combined


def _probe_video(path: str) -> tuple[float, int]:
    """Return (fps, frame_count) from a video file using cv2, or (0, 0) on error."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            return fps, frames
        finally:
            cap.release()
    except Exception:
        return 0.0, 0


def _merge_led_and_graph(
    led_table: SyncTable,
    graph_result,
    videos: list[CaptureVideoInfo],
    anchors: list,
) -> SyncTable:
    """Return a SyncTable that keeps LED sync points and adds non-LED cameras
    from the graph result, aligned to the LED global time via a shared anchor."""
    led_video_ids = set(led_table.video_ids())
    fps_by_video = {v.id: (v.actual_fps or 30.0) for v in videos}
    led_points: list[SyncPoint] = []
    for vid_id, (timestamps, frames, _fps) in led_table._tables.items():
        for t, f in zip(timestamps, frames):
            led_points.append(SyncPoint(
                camera_instance_id=vid_id,
                shot_video_id=vid_id,
                video_frame=f,
                timestamp_s=t,
            ))

    if not graph_result.sync_points:
        return SyncTable(led_points, fps_by_video)

    graph_table = SyncTable(graph_result.sync_points, fps_by_video)

    # Find a shared camera and anchor frame to align global time scales
    shared_vid: str | None = None
    shared_frame: int | None = None
    for _anchor_id, obs_list in anchors:
        for obs in obs_list:
            if obs.shot_video_id in led_video_ids and obs.shot_video_id in graph_table.video_ids():
                shared_vid = obs.shot_video_id
                shared_frame = obs.video_frame
                break
        if shared_vid is not None:
            break

    offset = 0.0
    if shared_vid is not None and shared_frame is not None:
        t_led = led_table.frame_to_global_time(shared_frame, shared_vid)
        t_graph = graph_table.frame_to_global_time(shared_frame, shared_vid)
        if t_led is not None and t_graph is not None:
            offset = t_led - t_graph

    extra_points = [
        SyncPoint(
            camera_instance_id=sp.shot_video_id,
            shot_video_id=sp.shot_video_id,
            video_frame=sp.video_frame,
            timestamp_s=sp.timestamp_s + offset,
        )
        for sp in graph_result.sync_points
        if sp.shot_video_id not in led_video_ids
    ]

    return SyncTable(led_points + extra_points, fps_by_video)


# ---------------------------------------------------------------------------
# Camera timeline bar
# ---------------------------------------------------------------------------


class _CameraTimelineBar(QWidget):
    """Compact horizontal timeline showing each camera's active global-time range.

    One row per camera: [label |========bar========|]
    A vertical playhead tracks the current scrubber position.
    Clicking/dragging seeks the scrubber via ``seek_requested(global_s)``.

    Only meaningful after sync is solved — call ``update_cameras`` once a
    SyncTable is available.
    """

    seek_requested = Signal(float)  # global_s

    _BAR_H = 12
    _ROW_H = 16
    _LABEL_W = 100
    _AXIS_H = 18

    # Bar colours
    _COL_SYNCED_GRAPH = "#2196F3"
    _COL_SYNCED_LED   = "#4CAF50"
    _COL_ISOLATED     = "#bbbbbb"
    _COL_ANCHOR_MARK  = "#FF6B00"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cameras: list[tuple[str, float, float, str]] = []  # (label, t0, t1, color_hex)
        self._t_min = 0.0
        self._t_max = 1.0
        self._playhead_s: float | None = None
        # Each anchor mark: (global_time, frozenset of video_ids in that pair)
        self._anchor_marks: list[tuple[float, frozenset[str]]] = []
        self._camera_video_ids: list[str] = []  # parallel to _cameras
        self._led_video_ids: set[str] = set()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(self._AXIS_H)  # collapsed until cameras are loaded
        self.setMouseTracking(True)

    def update_cameras(
        self,
        videos: list[CaptureVideoInfo],
        sync_table: SyncTable,
        led_video_ids: set[str] | None = None,
    ) -> None:
        self._led_video_ids = led_video_ids or set()
        fps_map = {v.id: (v.actual_fps or 30.0) for v in videos}
        cameras: list[tuple[str, float, float, str]] = []
        t_min, t_max = float("inf"), float("-inf")
        for v in videos:
            fps = fps_map[v.id]
            t0 = sync_table.frame_to_global_time(v.first_video_frame, v.id)
            t1 = sync_table.frame_to_global_time(v.last_video_frame, v.id)
            synced = t0 is not None and t1 is not None
            if not synced:
                t0 = 0.0
                t1 = (v.last_video_frame - v.first_video_frame) / fps
                color = self._COL_ISOLATED
            elif v.id in self._led_video_ids:
                color = self._COL_SYNCED_LED
            else:
                color = self._COL_SYNCED_GRAPH
            cameras.append((v.camera_label, t0, t1, color))
            t_min = min(t_min, t0)
            t_max = max(t_max, t1)
        self._cameras = cameras
        self._camera_video_ids = [v.id for v in videos]
        self._t_min = t_min if t_min != float("inf") else 0.0
        self._t_max = t_max if t_max != float("-inf") else 1.0
        h = len(cameras) * self._ROW_H + self._AXIS_H
        self.setFixedHeight(h)
        self.update()

    def update_anchor_marks(self, marks: list[tuple[float, frozenset[str]]]) -> None:
        """Update anchor pair markers.  Each entry is (global_time, video_ids)."""
        self._anchor_marks = list(marks)
        self.update()

    def set_playhead(self, global_s: float | None) -> None:
        self._playhead_s = global_s
        self.update()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _t_to_x(self, t: float) -> int:
        span = self._t_max - self._t_min
        if span <= 0:
            return self._LABEL_W
        frac = (t - self._t_min) / span
        return self._LABEL_W + int(frac * max(self.width() - self._LABEL_W, 1))

    def _x_to_t(self, x: int) -> float:
        w = max(self.width() - self._LABEL_W, 1)
        frac = (x - self._LABEL_W) / w
        return self._t_min + max(0.0, min(1.0, frac)) * (self._t_max - self._t_min)

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:  # noqa: N802
        from PySide6.QtGui import QColor, QPainter, QPen

        if not self._cameras:
            return

        p = QPainter(self)

        text_col = QColor("#333333")
        axis_col = QColor("#cccccc")
        playhead_col = QColor("#E91E63")
        anchor_col = QColor(self._COL_ANCHOR_MARK)

        bars_h = len(self._cameras) * self._ROW_H

        for i, (label, t0, t1, color) in enumerate(self._cameras):
            y = i * self._ROW_H
            # Label
            p.setPen(QPen(text_col))
            p.setFont(self.font())
            p.drawText(
                0, y, self._LABEL_W - 4, self._ROW_H,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            # Bar
            x0 = self._t_to_x(t0)
            x1 = self._t_to_x(t1)
            bar_y = y + (self._ROW_H - self._BAR_H) // 2
            p.fillRect(x0, bar_y, max(x1 - x0, 2), self._BAR_H, QColor(color))

        # Anchor marks — short orange lines on the bar rows of the paired cameras
        if self._anchor_marks:
            p.setPen(QPen(anchor_col, 2))
            for t, video_ids in self._anchor_marks:
                ax = self._t_to_x(t)
                if ax < self._LABEL_W or ax >= self.width():
                    continue
                for row, (label, t0, t1, color) in enumerate(self._cameras):
                    if row >= len(self._camera_video_ids):
                        continue
                    if self._camera_video_ids[row] not in video_ids:
                        continue
                    bar_y = row * self._ROW_H + (self._ROW_H - self._BAR_H) // 2
                    p.drawLine(ax, bar_y, ax, bar_y + self._BAR_H)

        # Time axis
        ay = bars_h
        p.setPen(QPen(axis_col))
        p.drawLine(self._LABEL_W, ay, self.width(), ay)
        step = self._nice_step(self._t_max - self._t_min)
        import math
        t = math.ceil(self._t_min / step) * step
        while t <= self._t_max:
            x = self._t_to_x(t)
            p.setPen(QPen(axis_col))
            p.drawLine(x, ay, x, ay + 3)
            mm, ss = int(t // 60), int(t % 60)
            p.setPen(QPen(text_col))
            p.drawText(x - 20, ay + 3, 40, self._AXIS_H - 3,
                       Qt.AlignmentFlag.AlignHCenter, f"{mm}:{ss:02d}")
            t += step

        # Playhead — clamp to bar area so it's visible even at edges
        if self._playhead_s is not None:
            px = max(self._LABEL_W, min(self.width() - 1, self._t_to_x(self._playhead_s)))
            p.setPen(QPen(playhead_col, 2))
            p.drawLine(px, 0, px, self.height())

        p.end()

    @staticmethod
    def _nice_step(duration: float) -> float:
        for step in (1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600):
            if duration / step <= 12:
                return float(step)
        return 3600.0

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def _emit_seek(self, x: int) -> None:
        if not self._cameras:
            return
        t = self._x_to_t(x)
        self.seek_requested.emit(t)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._emit_seek(event.pos().x())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._emit_seek(event.pos().x())


# ---------------------------------------------------------------------------
# SyncWidget — self-contained graph-based sync editor
# ---------------------------------------------------------------------------


class SyncWidget(QWidget):
    """Multi-camera sync editor using the graph-based anchor model.

    Left panel: QTreeWidget with cameras as top-level nodes; anchor observations
    as children.  Connected cameras are shown in green, isolated cameras in red.

    Centre: PairScrubber — reference (left) and target (right) side by side.
    Select reference and target from the combo boxes above.  Press "Mark sync pair
    at these frames" when both panes show a shared physical event.

    Bottom controls: "Solve & apply" computes timestamps from all recorded anchor
    pairs and writes a sync_config to the session DB.  "LED sync…" opens the LED
    dialog to refine the sync using a blinking LED.
    """

    def __init__(
        self, ctx: DBContext, shot_id: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._ctx = ctx
        self._shot_id = shot_id
        self._videos: list[CaptureVideoInfo] = []
        self._video_id_to_idx: dict[str, int] = {}
        self._sync_table: SyncTable | None = None
        self._current_anchor_id: str | None = None

        self._build_ui()
        self._load_shot()

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # --- Left: anchor tree + delete button ---
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Sync anchors"])
        self._tree.setAlternatingRowColors(True)
        self._tree.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._tree.itemClicked.connect(self._on_tree_item_clicked)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_tree_context_menu)

        self._delete_btn = QPushButton("Delete selected anchor")
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._on_delete_anchor)

        self._add_video_btn = QPushButton("Add video…")
        self._add_video_btn.setToolTip("Add a new video file to this capture.")
        self._add_video_btn.clicked.connect(self._on_add_video)

        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(4)
        left_l.addWidget(self._tree, stretch=1)
        left_l.addWidget(self._delete_btn)
        left_l.addWidget(self._add_video_btn)
        left_w.setMinimumWidth(180)
        left_w.setMaximumWidth(260)

        # --- Right: combos + scrubber + controls ---
        self._ref_combo = QComboBox()
        self._tgt_combo = QComboBox()
        self._ref_combo.currentIndexChanged.connect(self._on_ref_combo_changed)
        self._tgt_combo.currentIndexChanged.connect(self._on_tgt_combo_changed)

        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("Reference:"))
        combo_row.addWidget(self._ref_combo, stretch=1)
        combo_row.addSpacing(8)
        combo_row.addWidget(QLabel("Target:"))
        combo_row.addWidget(self._tgt_combo, stretch=1)

        self._pair = PairScrubber(self)
        self._pair.anchor_requested.connect(self._on_anchor_requested)

        self._connectivity_label = QLabel("No cameras loaded.")
        self._connectivity_label.setStyleSheet("font-size: 11px; color: grey;")

        self._status_label = QLabel()
        self._status_label.setStyleSheet("font-size: 11px; color: grey;")

        self._solve_btn = QPushButton("Solve && apply sync")
        self._solve_btn.setEnabled(False)
        self._solve_btn.setToolTip(
            "Compute consistent timestamps from all recorded anchor pairs\n"
            "and write a sync_config to the session database."
        )
        self._solve_btn.clicked.connect(self._on_solve)

        self._led_btn = QPushButton("LED sync…")
        self._led_btn.setEnabled(False)
        self._led_btn.setToolTip(
            "Open the LED sync dialog to refine the rough graph sync\n"
            "using a blinking LED.  Requires at least one solved sync config."
        )
        self._led_btn.clicked.connect(self._on_led_sync)

        self._time_display = QLabel("t: —")
        self._time_display.setStyleSheet("font-family: monospace; font-size: 11px;")
        self._time_display.setFixedWidth(90)

        self._export_btn = QPushButton("Export frames…")
        self._export_btn.setToolTip(
            "Save the current frame of every camera as a full-resolution PNG.\n"
            "Useful for extrinsics calibration — capture all cameras at the same\n"
            "shared event, then export and feed into Pose2Sim or similar."
        )
        self._export_btn.clicked.connect(self._on_export_frames)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self._connectivity_label, stretch=1)
        bottom_row.addWidget(self._time_display)
        bottom_row.addWidget(self._export_btn)
        bottom_row.addWidget(self._solve_btn)
        bottom_row.addWidget(self._led_btn)

        self._timeline = _CameraTimelineBar(self)
        self._pair.frames_changed.connect(self._on_pair_frames_changed)
        self._timeline.seek_requested.connect(self._on_timeline_seek)

        # Arrow-key global-timeline navigation — use QShortcut so focus
        # doesn't matter (works even when tree widget or other child has focus).
        for key, step in [
            (Qt.Key.Key_Left, -1), (Qt.Key.Key_Right, 1),
        ]:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda s=step: self._on_timeline_step(s))
        for key, step in [
            ("Shift+Left", -10), ("Shift+Right", 10),
        ]:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda s=step: self._on_timeline_step(s))

        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(4)
        right_l.addLayout(combo_row)
        right_l.addWidget(self._pair, stretch=1)
        right_l.addWidget(self._timeline)
        right_l.addWidget(self._status_label)
        right_l.addLayout(bottom_row)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        outer.addWidget(left_w)
        outer.addWidget(right_w, stretch=1)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load_shot(self) -> None:
        self._videos = self._ctx.get_shot_videos(self._shot_id)
        self._video_id_to_idx = {v.id: i for i, v in enumerate(self._videos)}

        self._ref_combo.blockSignals(True)
        self._tgt_combo.blockSignals(True)
        self._ref_combo.clear()
        self._tgt_combo.clear()
        for v in self._videos:
            self._ref_combo.addItem(v.camera_label, v.id)
            self._tgt_combo.addItem(v.camera_label, v.id)
        if len(self._videos) >= 2:
            self._tgt_combo.setCurrentIndex(1)
        self._ref_combo.blockSignals(False)
        self._tgt_combo.blockSignals(False)

        # Always re-solve from anchor observations.  The anchor observations are
        # the authoritative input for the setup wizard; DB sync configs (LED,
        # manual-graph) are written outputs read by the detection dialog.
        # Loading a DB config here would replay stale data and miss anchors
        # added since the last "Solve && apply sync".
        self._solve_and_refresh()

        self._reload_scrubber_ref()
        self._reload_scrubber_tgt()
        self._reload_tree()

    # ------------------------------------------------------------------
    # Scrubber helpers
    # ------------------------------------------------------------------

    def _reload_scrubber_ref(self) -> None:
        idx = self._ref_combo.currentIndex()
        if 0 <= idx < len(self._videos):
            sv = self._videos[idx]
            total = max(sv.last_video_frame - sv.first_video_frame + 1, 1)
            self._pair.set_reference(
                sv.file_path, total, sv.camera_label,
                self._frame_at_playhead(sv.id, total),
            )

    def _reload_scrubber_tgt(self) -> None:
        idx = self._tgt_combo.currentIndex()
        if 0 <= idx < len(self._videos):
            sv = self._videos[idx]
            total = max(sv.last_video_frame - sv.first_video_frame + 1, 1)
            self._pair.set_target(
                sv.file_path, total, sv.camera_label,
                self._frame_at_playhead(sv.id, total),
            )

    def _frame_at_playhead(self, video_id: str, total_frames: int) -> int:
        """Return the frame for *video_id* at the current timeline playhead, or 0."""
        t = self._timeline._playhead_s
        if t is None or self._sync_table is None:
            return 0
        f = self._sync_table.lookup(t, video_id)
        if f is None:
            return 0
        return max(0, min(total_frames - 1, f))

    # ------------------------------------------------------------------
    # Tree
    # ------------------------------------------------------------------

    def _reload_tree(self) -> None:
        self._tree.clear()
        self._current_anchor_id = None
        self._delete_btn.setEnabled(False)

        anchors = self._ctx.get_anchor_observations(self._shot_id)
        vid_map: dict[str, CaptureVideoInfo] = {v.id: v for v in self._videos}
        video_ids = [v.id for v in self._videos]

        ok, isolated = check_connectivity(anchors, video_ids)
        isolated_set = set(isolated)

        for sv in self._videos:
            top = QTreeWidgetItem(self._tree, [sv.camera_label])
            if sv.id in isolated_set:
                top.setForeground(0, QColor("#cc3300"))
                top.setToolTip(0, "Not yet connected to other cameras via any anchor.")
            elif video_ids:
                top.setForeground(0, QColor("#007700"))

            for anchor_id, obs_list in anchors:
                my_obs = next((o for o in obs_list if o.shot_video_id == sv.id), None)
                if my_obs is None:
                    continue
                for partner in obs_list:
                    if partner.shot_video_id == sv.id:
                        continue
                    pcam = vid_map.get(partner.shot_video_id)
                    plabel = pcam.camera_label if pcam else partner.shot_video_id
                    child = QTreeWidgetItem(
                        top,
                        [f"f{my_obs.video_frame} ↔ {plabel}: f{partner.video_frame}"],
                    )
                    child.setData(0, Qt.ItemDataRole.UserRole, {
                        "anchor_id": anchor_id,
                        "ref_video_id": sv.id,
                        "tgt_video_id": partner.shot_video_id,
                        "ref_frame": my_obs.video_frame,
                        "tgt_frame": partner.video_frame,
                    })

        self._tree.expandAll()

        n = len(self._videos)
        if n == 0:
            self._connectivity_label.setText("No cameras.")
            self._connectivity_label.setStyleSheet("font-size: 11px; color: grey;")
        elif ok:
            self._connectivity_label.setText(f"All {n} cameras connected.")
            self._connectivity_label.setStyleSheet("font-size: 11px; color: green;")
        else:
            missing = [
                vid_map[vid].camera_label if vid in vid_map else vid
                for vid in isolated
            ]
            self._connectivity_label.setText(f"Not connected: {', '.join(missing)}")
            self._connectivity_label.setStyleSheet("font-size: 11px; color: orange;")

        has_anchors = any(len(obs_list) >= 2 for _, obs_list in anchors)
        self._solve_btn.setEnabled(has_anchors)
        self._led_btn.setEnabled(self._sync_table is not None)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_ref_combo_changed(self, _idx: int) -> None:
        self._reload_scrubber_ref()

    def _on_tgt_combo_changed(self, _idx: int) -> None:
        self._reload_scrubber_tgt()

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            self._current_anchor_id = None
            self._delete_btn.setEnabled(False)
            return

        self._current_anchor_id = data["anchor_id"]
        self._delete_btn.setEnabled(True)

        ref_idx = self._video_id_to_idx.get(data["ref_video_id"], 0)
        tgt_idx = self._video_id_to_idx.get(data["tgt_video_id"], 0)

        self._ref_combo.blockSignals(True)
        self._tgt_combo.blockSignals(True)
        self._ref_combo.setCurrentIndex(ref_idx)
        self._tgt_combo.setCurrentIndex(tgt_idx)
        self._ref_combo.blockSignals(False)
        self._tgt_combo.blockSignals(False)

        self._reload_scrubber_ref()
        self._reload_scrubber_tgt()
        self._pair.seek_reference(data["ref_frame"])
        self._pair.seek_target(data["tgt_frame"])

    def _on_anchor_requested(self, ref_frame: int, tgt_frame: int) -> None:
        ref_idx = self._ref_combo.currentIndex()
        tgt_idx = self._tgt_combo.currentIndex()
        if ref_idx < 0 or tgt_idx < 0 or ref_idx == tgt_idx:
            self._status_label.setText("Select different cameras for reference and target.")
            self._status_label.setStyleSheet("font-size: 11px; color: orange;")
            return

        ref_sv = self._videos[ref_idx]
        tgt_sv = self._videos[tgt_idx]

        # Count shared anchors between this pair before adding the new one.
        # We only ask about fps when the pair crosses the 2-anchor threshold.
        old_anchors = self._ctx.get_anchor_observations(self._shot_id)
        old_pair_count = sum(
            1 for _aid, obs_list in old_anchors
            if any(o.shot_video_id == ref_sv.id for o in obs_list)
            and any(o.shot_video_id == tgt_sv.id for o in obs_list)
        )

        anchor_id = self._ctx.create_sync_anchor(self._shot_id)
        self._ctx.add_anchor_observation(anchor_id, ref_sv.id, ref_frame)
        self._ctx.add_anchor_observation(anchor_id, tgt_sv.id, tgt_frame)
        self._ctx._conn.commit()

        # Immediately check fps consistency when this pair reaches 2 anchors.
        if old_pair_count == 1:
            new_anchors = self._ctx.get_anchor_observations(self._shot_id)
            self._maybe_prompt_fps_correction(new_anchors, ref_sv, tgt_sv)

        self._status_label.setText(
            f"Anchor: {ref_sv.camera_label} f{ref_frame} ↔ "
            f"{tgt_sv.camera_label} f{tgt_frame}"
        )
        self._status_label.setStyleSheet("font-size: 11px; color: green;")
        self._reload_tree()
        self._solve_and_refresh()

    def _on_delete_anchor(self) -> None:
        if self._current_anchor_id is None:
            return
        self._ctx.delete_sync_anchor(self._current_anchor_id)
        self._ctx._conn.commit()
        self._current_anchor_id = None
        self._delete_btn.setEnabled(False)
        self._status_label.setText("Anchor deleted.")
        self._status_label.setStyleSheet("font-size: 11px; color: grey;")
        self._reload_tree()
        self._solve_and_refresh()

    def _on_tree_context_menu(self, pos) -> None:
        """Show a context menu for anchor-pair tree items.

        Right-clicking a child node (sync pair observation) offers "Delete sync
        pair".  Right-clicking a camera top-level node produces no menu.
        """
        from PySide6.QtWidgets import QMenu

        item = self._tree.itemAt(pos)
        if item is None:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return  # top-level camera node — no actions

        menu = QMenu(self)
        delete_action = menu.addAction("Delete sync pair")
        action = menu.exec(self._tree.viewport().mapToGlobal(pos))

        if action is delete_action:
            self._current_anchor_id = data["anchor_id"]
            self._on_delete_anchor()

    def _maybe_prompt_fps_correction(
        self,
        anchors: list,
        vid_a: "CaptureVideoInfo",
        vid_b: "CaptureVideoInfo",
    ) -> None:
        """Show a two-option dialog if anchor pairs imply an fps inconsistency.

        Presents both cameras as candidates so the user can pick which one has
        the wrong fps.  Updates actual_fps in the DB immediately on confirmation.
        """
        fps_a = vid_a.actual_fps or 30.0
        fps_b = vid_b.actual_fps or 30.0
        result = detect_pair_fps_inconsistency(
            anchors, vid_a.id, vid_b.id, fps_a, fps_b
        )
        if result is None:
            return

        fps_b_implied, fps_a_implied = result

        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("Frame rate mismatch detected")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(
            f"Sync pairs between <b>{vid_a.camera_label}</b> "
            f"({fps_a:.6g} fps) and <b>{vid_b.camera_label}</b> "
            f"({fps_b:.6g} fps) imply an inconsistent frame rate ratio.<br><br>"
            "Which camera has the wrong fps?"
        )
        btn_a = msg.addButton(
            f"{vid_a.camera_label}  →  {fps_a_implied:.6g} fps",
            QMessageBox.ButtonRole.ActionRole,
        )
        btn_b = msg.addButton(
            f"{vid_b.camera_label}  →  {fps_b_implied:.6g} fps",
            QMessageBox.ButtonRole.ActionRole,
        )
        msg.addButton("Skip", QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked is btn_a:
            self._ctx.update_shot_video_fps(vid_a.id, fps_a_implied)
            self._ctx._conn.commit()
            self._videos = self._ctx.get_shot_videos(self._shot_id)
        elif clicked is btn_b:
            self._ctx.update_shot_video_fps(vid_b.id, fps_b_implied)
            self._ctx._conn.commit()
            self._videos = self._ctx.get_shot_videos(self._shot_id)

    def _on_solve(self) -> None:
        anchors = self._ctx.get_anchor_observations(self._shot_id)
        result = solve_sync_graph(anchors, self._videos)

        if not result.sync_points:
            self._status_label.setText("No connected cameras — add anchor pairs first.")
            self._status_label.setStyleSheet("font-size: 11px; color: orange;")
            return

        points: dict[str, list[SyncPoint]] = {}
        for sp in result.sync_points:
            points.setdefault(sp.shot_video_id, []).append(sp)

        # Check every connected camera pair for fps inconsistency.  This catches
        # cases that weren't caught at anchor-marking time (e.g. loaded sessions),
        # and correctly identifies which camera needs correction regardless of
        # which video the BFS solver chose as its reference.
        vid_by_id = {v.id: v for v in self._videos}
        seen_pairs: set[frozenset[str]] = set()
        fps_was_updated = False
        for _anchor_id, obs_list in anchors:
            vid_ids = [o.shot_video_id for o in obs_list]
            for i, a_id in enumerate(vid_ids):
                for b_id in vid_ids[i + 1:]:
                    pair_key = frozenset((a_id, b_id))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    if a_id not in vid_by_id or b_id not in vid_by_id:
                        continue
                    va, vb = vid_by_id[a_id], vid_by_id[b_id]
                    fps_a_before = va.actual_fps or 30.0
                    fps_b_before = vb.actual_fps or 30.0
                    self._maybe_prompt_fps_correction(anchors, va, vb)
                    # Reload to pick up any change made in the dialog.
                    vid_by_id = {v.id: v for v in self._videos}
                    if (
                        vid_by_id[a_id].actual_fps != fps_a_before
                        or vid_by_id[b_id].actual_fps != fps_b_before
                    ):
                        fps_was_updated = True

        if fps_was_updated:
            # Re-solve with updated fps so the written sync_config is correct.
            result = solve_sync_graph(anchors, self._videos)
            points = {}
            for sp in result.sync_points:
                points.setdefault(sp.shot_video_id, []).append(sp)

        fps_by_video = {
            v.id: result.effective_fps.get(v.id, v.actual_fps or 30.0)
            for v in self._videos
        }

        conn = self._ctx._conn
        # Only delete manual-graph configs — LED and other configs may be
        # referenced by detection_runs / tracking_runs (NOT NULL FK) and
        # cannot be deleted without cascading changes to those records.
        conn.execute(
            "DELETE FROM sync_points WHERE sync_config_id IN "
            "(SELECT id FROM sync_configs WHERE shot_id = ? AND created_by = 'manual-graph')",
            (self._shot_id,),
        )
        conn.execute(
            "DELETE FROM sync_configs WHERE shot_id = ? AND created_by = 'manual-graph'",
            (self._shot_id,),
        )
        self._ctx.write_sync_config(self._shot_id, "manual-graph", points)
        conn.commit()

        self._sync_table = SyncTable(result.sync_points, fps_by_video)
        self._timeline.update_cameras(self._videos, self._sync_table)
        anchors = self._ctx.get_anchor_observations(self._shot_id)
        self._timeline.update_anchor_marks(self._compute_anchor_mark_times(anchors))
        self._ensure_combos_show_synced_cameras(result.connected_video_ids)

        n_conn = len(result.connected_video_ids)
        n_iso = len(result.isolated_video_ids)
        msg = f"Sync applied: {n_conn} camera(s) connected."
        if n_iso:
            isolated_labels = [
                next((v.camera_label for v in self._videos if v.id == vid), vid)
                for vid in result.isolated_video_ids
            ]
            msg += f"  Isolated: {', '.join(isolated_labels)}."
        self._status_label.setText(msg)
        self._status_label.setStyleSheet("font-size: 11px; color: green;")
        self._led_btn.setEnabled(True)

    def _on_export_frames(self) -> None:
        import os
        import re

        import cv2
        from PySide6.QtWidgets import QInputDialog, QMessageBox

        prefix, ok = QInputDialog.getText(
            self, "Export frames", "File prefix:", text="frame"
        )
        if not ok or not prefix:
            return

        out_dir = QFileDialog.getExistingDirectory(
            self, "Export frames — select output directory"
        )
        if not out_dir:
            return

        playhead_s = self._timeline._playhead_s

        def _ts(t: float) -> str:
            mm = int(t // 60)
            ss_f = t % 60
            ss = int(ss_f)
            ppp = int(round((ss_f - ss) * 1000))
            return f"{mm:02d}_{ss:02d}_{ppp:03d}"

        def _cam(label: str) -> str:
            return re.sub(r"[^\w]", "_", label).strip("_")

        exported: list[str] = []
        failed: list[str] = []

        for sv in self._videos:
            if playhead_s is not None and self._sync_table is not None:
                frame = self._sync_table.lookup(playhead_s, sv.id)
                if frame is None or frame < sv.first_video_frame:
                    continue  # camera not yet active at this timestamp
            else:
                ref_vid = self._ref_combo.currentData()
                tgt_vid = self._tgt_combo.currentData()
                if sv.id == ref_vid:
                    frame = self._pair.ref_frame
                elif sv.id == tgt_vid:
                    frame = self._pair.target_frame
                else:
                    frame = 0

            t = playhead_s if playhead_s is not None else 0.0
            fname = f"{prefix}_{_ts(t)}_{_cam(sv.camera_label)}_{frame:06d}.png"
            out_path = os.path.join(out_dir, fname)

            cap = cv2.VideoCapture(sv.file_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()
            cap.release()
            if ret:
                cv2.imwrite(out_path, img)
                exported.append(fname)
            else:
                failed.append(sv.camera_label)

        if failed:
            QMessageBox.warning(
                self,
                "Export frames",
                f"Exported {len(exported)} frames.\nFailed to read: {', '.join(failed)}",
            )
        else:
            QMessageBox.information(
                self,
                "Export frames",
                f"Exported {len(exported)} frame(s) to:\n{out_dir}",
            )

    def _on_led_sync(self) -> None:
        row = self._ctx._conn.execute(
            "SELECT COALESCE(label, 'Capture ' || capture_number) FROM captures WHERE id = ?",
            (self._shot_id,),
        ).fetchone()
        label = row[0] if row else "Capture"
        shot = _ShotMeta(shot_id=self._shot_id, label=label, videos=self._videos)
        current_frames = [0] * len(self._videos)
        manual_anchors = self._ctx.get_anchor_observations(self._shot_id)

        sync_table_offsets: dict[int, float] = {}
        if self._sync_table is not None:
            for i, sv in enumerate(self._videos):
                t0 = self._sync_table.frame_to_global_time(0, sv.id)
                if t0 is not None:
                    sync_table_offsets[i] = t0

        def _on_accepted(sync_table: SyncTable) -> None:
            # The LED dialog already ran solve_sync_graph over LED + manual
            # observations, so the resulting sync_table covers all reachable
            # cameras.  No further merging is needed.
            self._sync_table = sync_table
            synced_ids = set(sync_table.video_ids())
            self._timeline.update_cameras(
                self._videos, self._sync_table, led_video_ids=synced_ids,
            )
            anchors = self._ctx.get_anchor_observations(self._shot_id)
            self._timeline.update_anchor_marks(
                self._compute_anchor_mark_times(anchors)
            )
            self._ensure_combos_show_synced_cameras(synced_ids)
            self._status_label.setText("LED sync accepted and applied.")
            self._status_label.setStyleSheet("font-size: 11px; color: green;")

        dlg = _LedSyncDialog(
            shot=shot,
            fps_overrides={},
            ctx=self._ctx,
            current_frames=current_frames,
            on_sync_accepted=_on_accepted,
            sync_table_offsets=sync_table_offsets,
            manual_anchors=manual_anchors,
            parent=self,
        )
        dlg.exec()

    # ------------------------------------------------------------------
    # Timeline bar callbacks
    # ------------------------------------------------------------------

    def _solve_and_refresh(self) -> None:
        """Run solver in-memory and update the timeline without writing to DB."""
        anchors = self._ctx.get_anchor_observations(self._shot_id)
        result = solve_sync_graph(anchors, self._videos)

        vid_label = {v.id: v.camera_label for v in self._videos}
        connected_labels = [vid_label.get(v, v[:8]) for v in sorted(result.connected_video_ids)]
        isolated_labels  = [vid_label.get(v, v[:8]) for v in sorted(result.isolated_video_ids)]
        _log.info(
            "solve_and_refresh: shot=%s anchors=%d sync_points=%d "
            "connected=%s isolated=%s",
            self._shot_id[:8], len(anchors), len(result.sync_points),
            connected_labels, isolated_labels,
        )

        # Detect anchor observations that reference video IDs not in self._videos.
        known_ids = {v.id for v in self._videos}
        orphan_ids: set[str] = set()
        for _aid, obs_list in anchors:
            for obs in obs_list:
                if obs.shot_video_id not in known_ids:
                    orphan_ids.add(obs.shot_video_id)
        if orphan_ids:
            _log.warning(
                "solve_and_refresh: %d anchor observation(s) reference video IDs "
                "not in capture_videos for this shot: %s  "
                "(video may have been deleted and re-added — re-mark those anchors)",
                len(orphan_ids), list(orphan_ids),
            )

        if not result.sync_points:
            self._timeline.update_cameras(self._videos, SyncTable([], {}))
            self._timeline.update_anchor_marks([])
            return

        fps_by_video = {
            v.id: result.effective_fps.get(v.id, v.actual_fps or 30.0)
            for v in self._videos
        }
        self._sync_table = SyncTable(result.sync_points, fps_by_video)
        self._timeline.update_cameras(self._videos, self._sync_table)
        self._timeline.update_anchor_marks(self._compute_anchor_mark_times(anchors))
        self._ensure_combos_show_synced_cameras(result.connected_video_ids)

    def _compute_anchor_mark_times(
        self,
        anchors: list,
    ) -> list[tuple[float, frozenset[str]]]:
        """Return (global_time, video_ids) per anchor for timeline tick marks."""
        if self._sync_table is None:
            return []
        marks: list[tuple[float, frozenset[str]]] = []
        for _anchor_id, obs_list in anchors:
            video_ids = frozenset(obs.shot_video_id for obs in obs_list)
            for obs in obs_list:
                t = self._sync_table.frame_to_global_time(
                    obs.video_frame, obs.shot_video_id
                )
                if t is not None:
                    marks.append((t, video_ids))
                    break  # one representative time per anchor is enough
        return marks

    def _on_timeline_step(self, step: int) -> None:
        """Advance the ref camera by *step* frames and seek all cameras to match.

        Global time is derived from the new ref-camera frame position via the
        sync table, so the step size in seconds automatically matches the ref
        camera's local fps at that point in the video.
        """
        if self._sync_table is None:
            return
        ref_vid = self._ref_combo.currentData()
        if not ref_vid:
            return
        cur_frame = self._pair.ref_frame
        new_frame = max(0, cur_frame + step)
        t = self._sync_table.frame_to_global_time(new_frame, ref_vid)
        if t is None:
            return
        _log.debug(
            "timeline_step: step=%d ref_vid=%.8s frame %d->%d t=%.4f",
            step, ref_vid, cur_frame, new_frame, t,
        )
        self._on_timeline_seek(t)

    def _on_add_video(self) -> None:
        """Open a file picker and add the selected video to this capture."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Add video to capture",
            "",
            "Video files (*.mp4 *.mov *.avi *.mkv *.mts *.m2ts *.MP4 *.MOV);;All files (*)",
        )
        if not path:
            return

        fps, frame_count = _probe_video(path)
        if frame_count <= 0:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Add video", f"Could not read frame count from:\n{path}")
            return

        import os
        cam_id = _generate_id()
        label = os.path.splitext(os.path.basename(path))[0]
        conn = self._ctx._conn
        conn.execute(
            "INSERT OR IGNORE INTO camera_instances (id, label) VALUES (?, ?)",
            (cam_id, label),
        )
        self._ctx.create_shot_video(
            self._shot_id, cam_id, path, fps, frame_count, 0, 0
        )
        conn.commit()

        # Reload everything
        self._videos = self._ctx.get_shot_videos(self._shot_id)
        self._video_id_to_idx = {v.id: i for i, v in enumerate(self._videos)}
        self._ref_combo.blockSignals(True)
        self._tgt_combo.blockSignals(True)
        prev_ref = self._ref_combo.currentData()
        prev_tgt = self._tgt_combo.currentData()
        self._ref_combo.clear()
        self._tgt_combo.clear()
        for v in self._videos:
            self._ref_combo.addItem(v.camera_label, v.id)
            self._tgt_combo.addItem(v.camera_label, v.id)
        # Restore previous selections where possible
        for combo, prev in ((self._ref_combo, prev_ref), (self._tgt_combo, prev_tgt)):
            idx = combo.findData(prev)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        self._ref_combo.blockSignals(False)
        self._tgt_combo.blockSignals(False)
        self._reload_tree()
        if self._sync_table is not None:
            self._timeline.update_cameras(self._videos, self._sync_table)
        self._status_label.setText(f"Added: {label}  ({frame_count} frames @ {fps:.3f} fps)")
        self._status_label.setStyleSheet("font-size: 11px; color: green;")

    def _ensure_combos_show_synced_cameras(self, connected: set[str]) -> None:
        """Switch ref/tgt combos to synced cameras if they currently show isolated ones."""
        ref_vid = self._ref_combo.currentData()
        tgt_vid = self._tgt_combo.currentData()
        if ref_vid in connected and tgt_vid in connected:
            return
        synced_indices = [
            i for i in range(len(self._videos))
            if self._videos[i].id in connected
        ]
        if len(synced_indices) < 1:
            return
        if ref_vid not in connected:
            self._ref_combo.setCurrentIndex(synced_indices[0])
        if tgt_vid not in connected:
            alt = synced_indices[1] if len(synced_indices) > 1 else synced_indices[0]
            self._tgt_combo.setCurrentIndex(alt)

    def _on_pair_frames_changed(self, ref_frame: int, tgt_frame: int) -> None:
        if self._sync_table is None:
            return
        ref_vid = self._ref_combo.currentData()
        tgt_vid = self._tgt_combo.currentData()
        t = None
        if ref_vid:
            t = self._sync_table.frame_to_global_time(ref_frame, ref_vid)
        if t is None and tgt_vid:
            t = self._sync_table.frame_to_global_time(tgt_frame, tgt_vid)
        self._timeline.set_playhead(t)
        if t is not None:
            mm, ss = int(t // 60), t % 60
            self._time_display.setText(f"t: {mm:02d}:{ss:05.2f}")
        else:
            self._time_display.setText("t: —")

    def _on_timeline_seek(self, global_s: float) -> None:
        if self._sync_table is None:
            return
        ref_vid = self._ref_combo.currentData()
        tgt_vid = self._tgt_combo.currentData()
        _log.debug("timeline_seek: global_s=%.4f ref_vid=%.8s tgt_vid=%.8s",
                   global_s, ref_vid or "", tgt_vid or "")
        if ref_vid:
            f = self._sync_table.lookup(global_s, ref_vid)
            if f is not None:
                self._pair.seek_reference(f)
        if tgt_vid:
            f = self._sync_table.lookup(global_s, tgt_vid)
            if f is not None:
                self._pair.seek_target(f)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self._pair.shutdown()


# ---------------------------------------------------------------------------
# SyncPage — wizard page wrapper
# ---------------------------------------------------------------------------


class SyncPage(QWizardPage):
    """Wizard page — camera synchronisation."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTitle("Camera Synchronisation")
        self.setSubTitle(
            "Record sync anchor pairs between cameras using the pair scrubber, "
            "then click 'Solve & apply sync' to build a shared timeline.  "
            "Use 'LED sync…' for more accurate automated alignment."
        )
        self._shots: list[_ShotMeta] = []
        self._widget: Optional[SyncWidget] = None

        self._shot_combo = QComboBox()
        self._shot_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._shot_combo.currentIndexChanged.connect(self._on_shot_changed)

        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)

        shot_row = QHBoxLayout()
        shot_row.addWidget(QLabel("Capture:"))
        shot_row.addWidget(self._shot_combo)
        shot_row.addStretch()
        self._shot_row_w = QWidget()
        self._shot_row_w.setLayout(shot_row)

        layout = QVBoxLayout(self)
        layout.addWidget(self._shot_row_w)
        layout.addWidget(self._container, stretch=1)

    def initializePage(self) -> None:  # noqa: N802
        ctx: DBContext = self.wizard().db_context
        raw = getattr(self.wizard(), "new_shot_ids", None)
        new_ids: list[str] = raw if isinstance(raw, list) and raw else []

        try:
            if new_ids:
                placeholders = ",".join("?" * len(new_ids))
                rows = ctx._conn.execute(
                    f"SELECT id, capture_number, label FROM captures "
                    f"WHERE id IN ({placeholders})",
                    new_ids,
                ).fetchall()
            else:
                rows = ctx._conn.execute(
                    "SELECT id, capture_number, label FROM captures "
                    "WHERE session_id = ? ORDER BY capture_number",
                    (ctx._session_id,),
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            _log.error("Could not read captures: %s", exc)
            return

        self._shots = [
            _ShotMeta(
                shot_id=r["id"],
                label=r["label"] or f"Capture {r['capture_number']}",
                videos=[],
            )
            for r in rows
        ]

        self._shot_combo.blockSignals(True)
        self._shot_combo.clear()
        for s in self._shots:
            self._shot_combo.addItem(s.label, s.shot_id)
        self._shot_row_w.setVisible(len(self._shots) > 1)
        self._shot_combo.blockSignals(False)

        if self._shots:
            self._shot_combo.setCurrentIndex(0)
            self._on_shot_changed(0)

    def cleanupPage(self) -> None:  # noqa: N802
        self._teardown_widget()

    def isComplete(self) -> bool:  # noqa: N802
        return True

    def _on_shot_changed(self, index: int) -> None:
        self._teardown_widget()
        if index < 0 or index >= len(self._shots):
            return
        ctx: DBContext = self.wizard().db_context
        shot_id = self._shots[index].shot_id
        self._widget = SyncWidget(ctx, shot_id, self._container)
        self._container_layout.addWidget(self._widget)

    def _teardown_widget(self) -> None:
        if self._widget is not None:
            self._widget.shutdown()
            self._container_layout.removeWidget(self._widget)
            self._widget.deleteLater()
            self._widget = None


# ---------------------------------------------------------------------------
# SyncDialog — standalone dialog for the session tree
# ---------------------------------------------------------------------------


class SyncDialog(QDialog):
    """Standalone sync dialog opened from the session tree (CapturePanel)."""

    def __init__(
        self,
        ctx: DBContext,
        shot_id: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        row = ctx._conn.execute(
            "SELECT COALESCE(label, 'Capture ' || capture_number) FROM captures WHERE id = ?",
            (shot_id,),
        ).fetchone()
        self.setWindowTitle(f"Sync — {row[0] if row else 'Capture'}")
        self.resize(1100, 700)

        self._widget = SyncWidget(ctx, shot_id, self)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._widget, stretch=1)
        layout.addLayout(btn_row)

    def done(self, result: int) -> None:
        self._widget.shutdown()
        super().done(result)
