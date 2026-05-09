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
from PySide6.QtGui import QColor
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
from app.setup.db_context import CaptureVideoInfo, DBContext, SyncPoint, SyncTable
from app.setup.video_reader import FrameReader
from app.setup.job_runner import BackgroundJob
from app.setup.led_sync import (
    CameraSyncResult,
    LedSyncResult,
    ROI,
    extract_brightness_changes,
    run_led_sync,
    save_brightness_dump,
)
from app.setup.overlay import ROIDrawOverlay
from app.setup.pair_scrubber import PairScrubber
from app.setup.sync_solver import check_connectivity, solve_sync_graph

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
    """Extracts per-camera brightness signals then runs the LED sync algorithm.

    Parameters
    ----------
    cam_data:
        List of ``(file_path, roi, fps_override, cam_id, video_id)`` tuples.
    ref_cam:
        Index of the reference camera (identity map).
    event_cfg:
        Event-detection parameters forwarded to ``run_led_sync``.
    """

    def __init__(
        self,
        cam_data: list[tuple[str, ROI, float, str, str]],
        ref_cam: int = 0,
        event_cfg: dict | None = None,
        rough_offsets: list[float] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._cam_data = cam_data
        self._ref_cam = ref_cam
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

        self.progress.emit(82, "Running LED sync algorithm…")
        result = run_led_sync(
            signals=signals, fps_list=fps_list,
            cam_ids=cam_ids, video_ids=video_ids,
            ref_cam=self._ref_cam, event_cfg=self._event_cfg,
            rough_offsets=self._rough_offsets,
        )
        self.progress.emit(100, "Done.")
        self.finished.emit(result)


# ---------------------------------------------------------------------------
# Brightness plot dialog
# ---------------------------------------------------------------------------


class _BrightnessPlotDialog(QDialog):
    """Shows LED brightness for all cameras on a shared global timeline."""

    def __init__(self, result: LedSyncResult, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("LED Brightness — Synchronized Timeline")
        self.resize(860, 400)

        layout = QVBoxLayout(self)
        if _HAS_MATPLOTLIB:
            fig = _Figure(figsize=(10, 4), dpi=90, tight_layout=True)
            ax = fig.add_subplot(1, 1, 1)
            colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
            for i, cr in enumerate(result.cameras):
                ax.plot(cr.frame_times, cr.brightness, label=cr.camera_instance_id,
                        color=colors[i % len(colors)], linewidth=0.8, alpha=0.85)
            ax.set_xlabel("Global time (s)")
            ax.set_ylabel("Brightness change")
            ax.set_title("LED brightness — synchronized global timeline")
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

        self._led_rois: dict[int, ROI] = {}
        self._led_result: LedSyncResult | None = None
        self._led_job: _LedSyncJob | None = None

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
        self._led_job = _LedSyncJob(
            cam_data, ref_cam=0, rough_offsets=rough_offsets_list, parent=self,
        )
        self._led_job.progress.connect(self._on_progress)
        self._led_job.finished.connect(self._on_done)
        self._led_job.error.connect(self._on_error)
        self._led_job.start()

    def _on_progress(self, pct: int, msg: str) -> None:
        self._progress_bar.setValue(pct)
        if msg:
            self._progress_label.setText(msg)

    def _on_done(self, result: LedSyncResult) -> None:
        self._progress_bar.setVisible(False)
        self._progress_label.setText("")
        self._led_result = result
        self._run_btn.setEnabled(True)

        while self._quality_layout.count():
            item = self._quality_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for cr in result.cameras:
            if cr.map_type == "reference":
                txt = f"{cr.camera_instance_id}: reference camera ({cr.n_events} events)"
                style = "color: grey; font-size: 11px;"
                tooltip = "This camera defines the global time axis (identity map)."
            else:
                offset0 = float(cr.frame_times[0]) if len(cr.frame_times) else 0.0
                is_shift_only = cr.map_type == "shift_only"
                q = "good" if (cr.resid_std_s < 0.005 and not is_shift_only) else "poor"
                txt = (
                    f"{cr.camera_instance_id}: {cr.n_events} events, "
                    f"{cr.n_pairs} DTW pairs, "
                    f"{cr.n_inliers} inliers, "
                    f"σ={cr.resid_std_s * 1000:.1f}ms, "
                    f"offset={offset0:+.3f}s"
                    + (f"  ⚠ {cr.map_type}" if is_shift_only else f"  [{cr.map_type}]")
                    + f" — {q}"
                )
                style = (
                    "color: green; font-size: 11px;"
                    if q == "good" else "color: orange; font-size: 11px;"
                )
                tooltip = (
                    "Events: LED blink peaks detected in the brightness signal.\n"
                    "DTW pairs: events matched between this camera and the reference\n"
                    "  using Dynamic Time Warping.\n"
                    "Inliers: DTW pairs consistent with the fitted affine timing model\n"
                    "  (within 10 ms tolerance). Non-inliers are DTW mismatches or\n"
                    "  noise peaks — they are excluded from the final fit.\n"
                    f"Residual σ: std of inlier timing errors after fitting — < 5 ms is good.\n"
                    f"Offset: global time at camera frame 0 — should match cameras that\n"
                    f"  started recording at the same time.\n"
                    f"Map type: {cr.map_type}\n"
                    f"  affine/pchip = DTW succeeded\n"
                    f"  shift_only   = DTW failed, rough offset or cross-correlation used"
                )
            lbl = QLabel(txt)
            lbl.setStyleSheet(style)
            lbl.setToolTip(tooltip)
            self._quality_layout.addWidget(lbl)

        self._quality_widget.setVisible(True)
        self._accept_btn.setEnabled(True)
        self._plot_btn.setEnabled(True)
        self._dump_btn.setEnabled(True)
        self._accept_label.setText("Review quality above, then click Accept.")

    def _on_error(self, msg: str) -> None:
        self._progress_bar.setVisible(False)
        self._run_btn.setEnabled(True)
        self._accept_label.setText(f"Error: {msg}")
        self._accept_label.setStyleSheet("color: red; font-size: 11px;")

    def _on_accept(self) -> None:
        if self._led_result is None:
            return
        points, fps_by_video = _sync_points_from_led_result(self._led_result)
        # Persist corrected fps values from the LED dialog spinboxes so that
        # loading this sync config later uses the correct fps for interpolation.
        for cell_idx, sv in enumerate(self._shot.videos):
            fps = self._fps_spinboxes[cell_idx].value() or sv.actual_fps or 30.0
            if abs(fps - (sv.actual_fps or 0.0)) > 0.01:
                self._ctx.update_shot_video_fps(sv.id, fps)
        self._ctx.write_sync_config(self._shot.shot_id, "led-auto", points)
        self._ctx._conn.commit()

        all_points = [sp for pts in points.values() for sp in pts]
        sync_table = SyncTable(all_points, fps_by_video)
        self._on_sync_accepted(sync_table)

        self._accept_label.setText("LED sync accepted.")
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
        _BrightnessPlotDialog(self._led_result, parent=self).exec()

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
        rough_offsets = self._compute_rough_offsets()
        try:
            save_brightness_dump(path, self._led_result, rough_offsets)
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

        self._delete_btn = QPushButton("Delete selected anchor")
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._on_delete_anchor)

        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(4)
        left_l.addWidget(self._tree, stretch=1)
        left_l.addWidget(self._delete_btn)
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

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self._connectivity_label, stretch=1)
        bottom_row.addWidget(self._solve_btn)
        bottom_row.addWidget(self._led_btn)

        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(4)
        right_l.addLayout(combo_row)
        right_l.addWidget(self._pair, stretch=1)
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

        # Auto-load the best existing sync config so "LED sync…" is available.
        configs = self._ctx.get_sync_configs(self._shot_id)
        if configs:
            self._sync_table = self._ctx.load_sync_config(configs[0][0])

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
            self._pair.set_reference(sv.file_path, total, sv.camera_label)

    def _reload_scrubber_tgt(self) -> None:
        idx = self._tgt_combo.currentIndex()
        if 0 <= idx < len(self._videos):
            sv = self._videos[idx]
            total = max(sv.last_video_frame - sv.first_video_frame + 1, 1)
            self._pair.set_target(sv.file_path, total, sv.camera_label)

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

        anchor_id = self._ctx.create_sync_anchor(self._shot_id)
        self._ctx.add_anchor_observation(anchor_id, ref_sv.id, ref_frame)
        self._ctx.add_anchor_observation(anchor_id, tgt_sv.id, tgt_frame)
        self._ctx._conn.commit()

        self._status_label.setText(
            f"Anchor: {ref_sv.camera_label} f{ref_frame} ↔ "
            f"{tgt_sv.camera_label} f{tgt_frame}"
        )
        self._status_label.setStyleSheet("font-size: 11px; color: green;")
        self._reload_tree()

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

        fps_by_video = {v.id: (v.actual_fps or 30.0) for v in self._videos}

        conn = self._ctx._conn
        conn.execute(
            "DELETE FROM sync_points WHERE sync_config_id IN "
            "(SELECT id FROM sync_configs WHERE shot_id = ?)",
            (self._shot_id,),
        )
        conn.execute("DELETE FROM sync_configs WHERE shot_id = ?", (self._shot_id,))
        self._ctx.write_sync_config(self._shot_id, "manual-graph", points)
        conn.commit()

        self._sync_table = SyncTable(result.sync_points, fps_by_video)

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

    def _on_led_sync(self) -> None:
        row = self._ctx._conn.execute(
            "SELECT COALESCE(label, 'Capture ' || capture_number) FROM captures WHERE id = ?",
            (self._shot_id,),
        ).fetchone()
        label = row[0] if row else "Capture"
        shot = _ShotMeta(shot_id=self._shot_id, label=label, videos=self._videos)
        current_frames = [0] * len(self._videos)

        sync_table_offsets: dict[int, float] = {}
        if self._sync_table is not None:
            for i, sv in enumerate(self._videos):
                t0 = self._sync_table.frame_to_global_time(0, sv.id)
                if t0 is not None:
                    sync_table_offsets[i] = t0

        def _on_accepted(sync_table: SyncTable) -> None:
            self._sync_table = sync_table
            self._status_label.setText("LED sync accepted and applied.")
            self._status_label.setStyleSheet("font-size: 11px; color: green;")

        dlg = _LedSyncDialog(
            shot=shot,
            fps_overrides={},
            ctx=self._ctx,
            current_frames=current_frames,
            on_sync_accepted=_on_accepted,
            sync_table_offsets=sync_table_offsets,
            parent=self,
        )
        dlg.exec()

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
