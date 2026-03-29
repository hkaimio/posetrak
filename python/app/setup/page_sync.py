"""page_sync.py — Wizard page for camera synchronisation.

The page lets the user inspect all camera feeds together and establish a
common time reference so that downstream tracking and visualisation can
treat frames from different cameras as simultaneous.

Two sync methods are available (in increasing accuracy):

Rough sync (lower panel)
    The user scrolls each camera to the same physical event — a clap, a
    flash, an LED blink — and presses "Set anchor" for each camera.  Once
    every camera has an anchor, "Apply rough sync" computes per-camera
    frame offsets and switches the scrubber into synced mode.  The result
    is written to the session as a sync config with method "manual-rough".

LED sync (upper panel inside LED group)
    Automated brightness-peak detection and cross-correlation against a
    reference camera.  The user draws a small ROI over a blinking LED in
    each camera feed, optionally overrides the fps (useful for Android
    slow-motion clips), then runs the sync job in the background.  On
    completion the per-camera quality metrics and a brightness-vs-global-time
    plot are shown; accepting writes a sync config with method "led-auto".

Layout
------
- Shot selector — one entry per shot in the session.
- MultiVideoScrubber — grid of camera cells, each with its own slider and
  frame counter.  Cells scroll independently until a sync config is applied.
- Rough sync panel — set/clear per-camera anchors, apply the sync.
- LED sync panel — per-camera ROI + fps setup, run button, quality metrics,
  brightness plot, accept button.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from app.setup.camera_cell import CameraCell
from app.setup.db_context import DBContext, SyncPoint, SyncTable
from app.setup.frame_cache import FrameCache
from app.setup.job_runner import BackgroundJob
from app.setup.led_sync import (
    CameraSyncResult,
    LedSyncResult,
    ROI,
    extract_brightness_changes,
    run_led_sync,
)
from app.setup.multi_video_scrubber import CellInfo, MultiVideoScrubber
from app.setup.overlay import ROIDrawOverlay, SyncAnchorOverlay

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
# Overlay helpers used by the ROI selection dialog
# ---------------------------------------------------------------------------


class _ClickCaptureOverlay:
    """Invisible overlay that fires a callback once on the first mouse press.

    Used in the full-view phase of the ROI dialog to capture the zoom-center
    click without permanently modifying the cell's overlay list.
    """

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
    """Wraps ``ROIDrawOverlay`` and triggers a cell repaint after each mouse event.

    Without explicit repaints, the rubber-band rectangle would not update
    during drag.
    """

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
# Background frame reader for the ROI selection dialog
# ---------------------------------------------------------------------------


class _DialogFrameReader(QThread):
    """Decodes individual video frames in a background thread.

    Designed for the (modal) ROI selection dialog where responsiveness matters
    but only one frame at a time is needed.  Rapid slider moves coalesce: only
    the most recently requested frame is decoded.
    """

    frame_ready = Signal(int, object)  # frame_idx, numpy_array

    def __init__(self, file_path: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._file_path = file_path
        self._pending: int | None = None
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop = False

    def request(self, frame_idx: int) -> None:
        with self._lock:
            self._pending = frame_idx
        self._event.set()

    def shutdown(self) -> None:
        self._stop = True
        self._event.set()
        self.wait(2000)

    def run(self) -> None:
        import cv2
        cap = cv2.VideoCapture(str(self._file_path))
        while not self._stop:
            self._event.wait()
            self._event.clear()
            if self._stop:
                break
            with self._lock:
                idx = self._pending
            if idx is None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(idx, frame)
        cap.release()


# ---------------------------------------------------------------------------
# ROI selection dialog
# ---------------------------------------------------------------------------


class _ROISelectDialog(QDialog):
    """Two-phase dialog for selecting a LED ROI in a video.

    Phase 1 — full view
        The full video frame is displayed.  Clicking anywhere sets the zoom
        centre and transitions to phase 2.

    Phase 2 — zoom view
        A 400 × 300 pixel window centred on the click point is displayed
        (clamped to the frame boundary).  The user scrubs frames and draws a
        rectangle around the LED with the mouse.  "Back" returns to phase 1;
        "Confirm ROI" closes the dialog with the selected ROI.  ESC returns to
        phase 1 from phase 2 and cancels from phase 1.

    Parameters
    ----------
    file_path:
        Video file to display.
    total_frames, fps:
        Used to set slider range and label.
    initial_frame:
        Frame shown when the dialog opens.
    """

    _PHASE_FULL = 1
    _PHASE_ZOOM = 2

    # Zoom window size in video-frame pixels (half-widths)
    _ZOOM_HW = 200   # half-width  (→ 400 px total)
    _ZOOM_HH = 150   # half-height (→ 300 px total)

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
        self._full_frame = None     # latest full-frame array
        self._zoom_rect: tuple[int, int, int, int] | None = None  # (x1,y1,x2,y2) full-frame
        self._roi_overlay: _RepaintingROIOverlay | None = None
        self._confirmed_roi: ROI | None = None

        # --- widgets ---
        self._instruction = QLabel()
        self._instruction.setWordWrap(True)

        self._cell = CameraCell(label="Loading…", parent=self)
        self._cell.setMinimumSize(640, 360)
        self._cell.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

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
        self._back_btn.clicked.connect(self._enter_phase_full)

        self._confirm_btn = QPushButton("Confirm ROI")
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.clicked.connect(self._on_confirm)

        self._cancel_btn = QPushButton("Cancel")
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

        # --- background frame reader ---
        self._reader = _DialogFrameReader(file_path, self)
        self._reader.frame_ready.connect(self._on_frame_ready)
        self._reader.start()

        self._enter_phase_full()
        self._reader.request(initial_frame)

    # ------------------------------------------------------------------
    # Public result
    # ------------------------------------------------------------------

    def selected_roi(self) -> ROI | None:
        """Return the confirmed ROI in full-frame coordinates, or ``None``."""
        return self._confirmed_roi

    # ------------------------------------------------------------------
    # Phase transitions
    # ------------------------------------------------------------------

    def _enter_phase_full(self) -> None:
        self._phase = self._PHASE_FULL
        self._zoom_rect = None
        self._roi_overlay = None
        click_capture = _ClickCaptureOverlay(self._on_phase1_click)
        self._cell.set_overlays([click_capture])
        self._instruction.setText(
            "<b>Phase 1 — click on the LED location in the frame</b><br>"
            "The view will zoom into that area so you can draw a precise rectangle."
        )
        self._back_btn.setVisible(False)
        self._confirm_btn.setEnabled(False)
        # Re-show current frame without crop
        if self._full_frame is not None:
            self._cell.set_frame(self._full_frame)

    def _on_phase1_click(self, fx: int, fy: int) -> None:
        """Transition to zoom view centred on the clicked frame coordinate."""
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
            "Drag the mouse to mark the LED area.  Scrub frames to find a blink.  "
            "Press ← Back or ESC to return to the full view."
        )
        self._back_btn.setVisible(True)
        self._confirm_btn.setEnabled(False)
        # Show cropped frame immediately
        if self._full_frame is not None:
            self._show_cropped(self._full_frame)
        # Watch for ROI confirmation via a timer
        from PySide6.QtCore import QTimer
        self._roi_check_timer = QTimer(self)
        self._roi_check_timer.setInterval(100)
        self._roi_check_timer.timeout.connect(self._check_roi_valid)
        self._roi_check_timer.start()

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
        # Translate from cropped-frame coords back to full-frame coords
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
        if event.key() == Qt.Key.Key_Escape:
            if self._phase == self._PHASE_ZOOM:
                self._enter_phase_full()
            else:
                self.reject()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._reader.shutdown()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# LED sync background job
# ---------------------------------------------------------------------------


class _LedSyncJob(BackgroundJob):
    """Extracts per-camera brightness signals and runs the LED sync algorithm.

    Parameters
    ----------
    cam_data:
        List of ``(file_path, roi, fps_override, cam_id, video_id)`` tuples,
        one per camera, in scrubber cell order.
    ref_cam:
        Index of the reference camera.
    event_cfg:
        Event-detection parameters forwarded to ``run_led_sync``.
    """

    def __init__(
        self,
        cam_data: list[tuple[str, ROI, float, str, str]],
        ref_cam: int = 0,
        event_cfg: dict | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._cam_data = cam_data
        self._ref_cam = ref_cam
        self._event_cfg = event_cfg

    def run(self) -> None:
        K = len(self._cam_data)
        signals = []
        fps_list = []
        cam_ids = []
        video_ids = []

        for k, (file_path, roi, fps_override, cam_id, video_id) in enumerate(self._cam_data):
            base_pct = k * 80 // K
            end_pct = (k + 1) * 80 // K
            self.progress.emit(base_pct, f"Extracting brightness for {cam_id}…")

            def _prog(frame_idx: int, total: int, _b=base_pct, _e=end_pct) -> None:
                if total > 0:
                    pct = _b + (_e - _b) * frame_idx // total
                    self.progress.emit(pct, f"Reading {cam_id} frame {frame_idx}/{total}")

            sig, fps = extract_brightness_changes(
                file_path, roi, fps_override=fps_override, progress_cb=_prog,
            )
            signals.append(sig)
            fps_list.append(fps)
            cam_ids.append(cam_id)
            video_ids.append(video_id)

        self.progress.emit(82, "Running LED sync algorithm…")
        result = run_led_sync(
            signals=signals,
            fps_list=fps_list,
            cam_ids=cam_ids,
            video_ids=video_ids,
            ref_cam=self._ref_cam,
            event_cfg=self._event_cfg,
        )
        self.progress.emit(100, "Done.")
        self.finished.emit(result)


# ---------------------------------------------------------------------------
# Brightness plot dialog
# ---------------------------------------------------------------------------


class _BrightnessPlotDialog(QDialog):
    """Shows synchronized LED brightness for all cameras on a shared timeline."""

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
                color = colors[i % len(colors)]
                ax.plot(
                    cr.frame_times,
                    cr.brightness,
                    label=cr.camera_instance_id,
                    color=color,
                    linewidth=0.8,
                    alpha=0.85,
                )
            ax.set_xlabel("Global time (s)")
            ax.set_ylabel("Brightness change (per-frame delta)")
            ax.set_title("LED brightness signals on synchronized global timeline")
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
# SyncPage helpers
# ---------------------------------------------------------------------------


def _sync_points_from_led_result(
    result: LedSyncResult,
) -> tuple[dict[str, list[SyncPoint]], dict[str, float]]:
    """Convert an LedSyncResult to the structures needed by write_sync_config.

    Samples the per-frame global-time array at ~30 uniformly spaced indices to
    give the SyncTable enough anchor points for accurate piecewise interpolation.
    """
    points: dict[str, list[SyncPoint]] = {}
    fps_by_video: dict[str, float] = {}
    n_target = 30

    for cr in result.cameras:
        N = len(cr.frame_times)
        if N == 0:
            continue

        if N <= n_target + 1:
            indices = list(range(N))
        else:
            step = N // n_target
            indices = list(range(0, N, step))
            if indices[-1] != N - 1:
                indices.append(N - 1)

        pts = [
            SyncPoint(
                camera_instance_id=cr.camera_instance_id,
                shot_video_id=cr.shot_video_id,
                video_frame=i,
                timestamp_s=float(cr.frame_times[i]),
            )
            for i in indices
        ]
        points[cr.camera_instance_id] = pts

        # Effective fps for SyncTable extrapolation beyond the sample range
        t_range = abs(float(cr.frame_times[-1]) - float(cr.frame_times[0]))
        if N > 1 and t_range > 1e-9:
            fps_by_video[cr.shot_video_id] = (N - 1) / t_range
        else:
            fps_by_video[cr.shot_video_id] = cr.fps_used

    return points, fps_by_video


# ---------------------------------------------------------------------------
# SyncPage
# ---------------------------------------------------------------------------


class SyncPage(QWizardPage):
    """Wizard page — camera synchronisation."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTitle("Camera Synchronisation")
        self.setSubTitle(
            "Inspect camera feeds together and establish a common time reference. "
            "Use LED sync for automatic alignment, or rough sync for a quick manual offset."
        )

        # ---- shared state ----
        self._shots: list[_ShotMeta] = []
        self._cache: FrameCache | None = None
        self._scrubber: MultiVideoScrubber | None = None

        # Rough-sync state (reset on shot change)
        self._anchors: dict[int, int] = {}
        self._anchor_overlays: list[SyncAnchorOverlay] = []
        self._anchor_labels: list[QLabel] = []

        # LED sync state (reset on shot change)
        self._led_rois: dict[int, ROI] = {}
        self._led_fps_spinboxes: list[QDoubleSpinBox] = []
        self._led_roi_labels: list[QLabel] = []
        self._led_quality_labels: list[QLabel] = []
        self._led_result: LedSyncResult | None = None
        self._led_job: _LedSyncJob | None = None

        # ---- shot selector ----
        self._shot_combo = QComboBox()
        self._shot_combo.currentIndexChanged.connect(self._on_shot_selected)

        shot_bar = QHBoxLayout()
        shot_bar.addWidget(QLabel("Shot:"))
        shot_bar.addWidget(self._shot_combo)
        shot_bar.addStretch()

        # ---- scrubber area ----
        self._scrubber_container = QWidget()
        self._scrubber_layout = QVBoxLayout(self._scrubber_container)
        self._scrubber_layout.setContentsMargins(0, 0, 0, 0)

        # ---- LED sync panel ----
        self._led_panel = QGroupBox("LED synchronisation")
        led_layout = QVBoxLayout(self._led_panel)
        led_layout.setSpacing(4)

        # Per-camera rows (rebuilt dynamically)
        self._led_cam_rows_widget = QWidget()
        self._led_cam_rows_layout = QVBoxLayout(self._led_cam_rows_widget)
        self._led_cam_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._led_cam_rows_layout.setSpacing(2)
        led_layout.addWidget(self._led_cam_rows_widget)

        # Run section
        led_run_row = QHBoxLayout()
        self._led_run_btn = QPushButton("Run LED sync")
        self._led_run_btn.setEnabled(False)
        self._led_run_btn.clicked.connect(self._on_run_led_sync)
        self._led_progress_bar = QProgressBar()
        self._led_progress_bar.setRange(0, 100)
        self._led_progress_bar.setVisible(False)
        self._led_progress_label = QLabel()
        self._led_progress_label.setStyleSheet("font-size: 11px;")
        led_run_row.addWidget(self._led_run_btn)
        led_run_row.addWidget(self._led_progress_bar, stretch=1)
        led_run_row.addWidget(self._led_progress_label)
        led_layout.addLayout(led_run_row)

        # Quality metrics (hidden until results arrive)
        self._led_quality_widget = QWidget()
        self._led_quality_layout = QHBoxLayout(self._led_quality_widget)
        self._led_quality_layout.setContentsMargins(0, 0, 0, 0)
        self._led_quality_widget.setVisible(False)
        led_layout.addWidget(self._led_quality_widget)

        # Accept row
        led_accept_row = QHBoxLayout()
        self._led_plot_btn = QPushButton("Show brightness plot")
        self._led_plot_btn.setEnabled(False)
        self._led_plot_btn.clicked.connect(self._on_show_brightness_plot)
        self._led_accept_btn = QPushButton("Accept LED sync")
        self._led_accept_btn.setEnabled(False)
        self._led_accept_btn.clicked.connect(self._on_accept_led_sync)
        self._led_accept_label = QLabel()
        self._led_accept_label.setStyleSheet("font-size: 11px;")
        led_accept_row.addWidget(self._led_accept_label, stretch=1)
        led_accept_row.addWidget(self._led_plot_btn)
        led_accept_row.addWidget(self._led_accept_btn)
        led_layout.addLayout(led_accept_row)

        # ---- rough sync panel ----
        self._rough_panel = QGroupBox("Rough synchronisation")
        rough_layout = QVBoxLayout(self._rough_panel)
        rough_layout.setSpacing(4)

        btn_row = QHBoxLayout()
        self._set_anchor_btn = QPushButton("Set anchor for focused camera")
        self._set_anchor_btn.setEnabled(False)
        self._set_anchor_btn.clicked.connect(self._on_set_anchor)
        self._clear_anchors_btn = QPushButton("Clear all anchors")
        self._clear_anchors_btn.setEnabled(False)
        self._clear_anchors_btn.clicked.connect(self._on_clear_anchors)
        btn_row.addWidget(self._set_anchor_btn)
        btn_row.addWidget(self._clear_anchors_btn)
        btn_row.addStretch()
        rough_layout.addLayout(btn_row)

        self._anchor_status_widget = QWidget()
        self._anchor_status_layout = QHBoxLayout(self._anchor_status_widget)
        self._anchor_status_layout.setContentsMargins(0, 0, 0, 0)
        rough_layout.addWidget(self._anchor_status_widget)

        apply_row = QHBoxLayout()
        self._rough_status_label = QLabel()
        self._rough_status_label.setStyleSheet("color: grey; font-size: 11px;")
        self._apply_rough_btn = QPushButton("Apply rough sync")
        self._apply_rough_btn.setEnabled(False)
        self._apply_rough_btn.clicked.connect(self._on_apply_rough_sync)
        apply_row.addWidget(self._rough_status_label, stretch=1)
        apply_row.addWidget(self._apply_rough_btn)
        rough_layout.addLayout(apply_row)

        # ---- error label ----
        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: red;")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)

        # ---- main layout (sync panels in a scroll area to handle small windows) ----
        panels_widget = QWidget()
        panels_layout = QVBoxLayout(panels_widget)
        panels_layout.setContentsMargins(0, 0, 0, 0)
        panels_layout.addWidget(self._led_panel)
        panels_layout.addWidget(self._rough_panel)
        panels_layout.addWidget(self._error_label)

        scroll = QScrollArea()
        scroll.setWidget(panels_widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setMaximumHeight(340)

        layout = QVBoxLayout(self)
        layout.addLayout(shot_bar)
        layout.addWidget(self._scrubber_container, stretch=1)
        layout.addWidget(scroll)

    # ------------------------------------------------------------------
    # Qt wizard overrides
    # ------------------------------------------------------------------

    def initializePage(self) -> None:  # noqa: N802
        self._error_label.setVisible(False)
        self._shots.clear()
        self._shot_combo.blockSignals(True)
        self._shot_combo.clear()

        ctx: DBContext = self.wizard().db_context
        try:
            rows = ctx._conn.execute(
                "SELECT id, shot_number, label FROM shots "
                "WHERE session_id = ? ORDER BY shot_number",
                (ctx._session_id,),
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            self._shot_combo.blockSignals(False)
            self._show_error(f"Could not read shots: {exc}")
            return

        for row in rows:
            label = row["label"] or f"Shot {row['shot_number']}"
            videos = ctx.get_shot_videos(row["id"])
            meta = _ShotMeta(shot_id=row["id"], label=label, videos=videos)
            self._shots.append(meta)
            self._shot_combo.addItem(label)

        self._shot_combo.blockSignals(False)

        if self._shots:
            self._on_shot_selected(0)
        else:
            self._show_error(
                "No shots found. Go back to the Shots & Videos page and add at least one."
            )

    def cleanupPage(self) -> None:  # noqa: N802
        self._teardown_scrubber()

    def isComplete(self) -> bool:  # noqa: N802
        return True

    # ------------------------------------------------------------------
    # Slots — shot selection
    # ------------------------------------------------------------------

    def _on_shot_selected(self, index: int) -> None:
        self._teardown_scrubber()
        if index < 0 or index >= len(self._shots):
            return

        shot = self._shots[index]
        if not shot.videos:
            self._show_error(
                f"Shot '{shot.label}' has no videos. "
                "Go back and add video files for this shot."
            )
            return

        cells_info = [
            CellInfo(
                shot_video_id=sv.id,
                file_path=sv.file_path,
                total_frames=max(sv.last_video_frame - sv.first_video_frame + 1, 1),
                fps=sv.actual_fps or 30.0,
                label=sv.camera_instance_id,
            )
            for sv in shot.videos
        ]

        self._cache = FrameCache()
        scrubber = MultiVideoScrubber(cells_info, self._cache, self._scrubber_container)
        self._scrubber_layout.addWidget(scrubber)
        self._scrubber = scrubber

        # Rough-sync overlays
        self._anchor_overlays = [
            SyncAnchorOverlay(total_frames=info.total_frames)
            for info in cells_info
        ]
        for i, ov in enumerate(self._anchor_overlays):
            scrubber.set_overlays(i, [ov])

        self._rebuild_anchor_labels(shot)
        self._rebuild_led_panel(shot)

        self._set_anchor_btn.setEnabled(True)
        self._clear_anchors_btn.setEnabled(True)
        self._update_rough_panel_state()
        scrubber.setFocus()

    # ------------------------------------------------------------------
    # Slots — rough sync
    # ------------------------------------------------------------------

    def _on_set_anchor(self) -> None:
        if self._scrubber is None:
            return
        fc = self._scrubber.focused_cell
        frame = self._scrubber.current_frames[fc]
        self._anchors[fc] = frame
        self._anchor_overlays[fc].set_anchor(frame)
        self._scrubber._cells[fc].update()

        shot = self._shots[self._shot_combo.currentIndex()]
        cam = shot.videos[fc].camera_instance_id
        self._anchor_labels[fc].setText(f"{cam}: {frame}")
        self._update_rough_panel_state()

    def _on_clear_anchors(self) -> None:
        self._anchors.clear()
        for ov in self._anchor_overlays:
            ov.anchor_frame = None
        if self._scrubber:
            for cell in self._scrubber._cells:
                cell.update()
        shot_idx = self._shot_combo.currentIndex()
        if 0 <= shot_idx < len(self._shots):
            shot = self._shots[shot_idx]
            for i, lbl in enumerate(self._anchor_labels):
                cam = shot.videos[i].camera_instance_id
                lbl.setText(f"{cam}: —")
        if self._scrubber:
            self._scrubber.reload_sync(None)
        self._update_rough_panel_state()

    def _on_apply_rough_sync(self) -> None:
        if not self._anchors or self._scrubber is None:
            return

        shot_idx = self._shot_combo.currentIndex()
        shot = self._shots[shot_idx]

        ref_cell = min(self._anchors)
        ref_frame = self._anchors[ref_cell]
        ref_sv = shot.videos[ref_cell]
        ref_fps = ref_sv.actual_fps or 30.0
        ref_ts = ref_frame / ref_fps

        points: dict[str, list[SyncPoint]] = {}
        fps_by_video: dict[str, float] = {}
        for cell_idx, anchor_frame in self._anchors.items():
            sv = shot.videos[cell_idx]
            cam_id = sv.camera_instance_id
            fps = sv.actual_fps or 30.0
            points[cam_id] = [
                SyncPoint(
                    camera_instance_id=cam_id,
                    shot_video_id=sv.id,
                    video_frame=anchor_frame,
                    timestamp_s=ref_ts,
                )
            ]
            fps_by_video[sv.id] = fps

        ctx: DBContext = self.wizard().db_context
        ctx.write_sync_config(shot.shot_id, "manual-rough", points)
        ctx._conn.commit()

        all_points = [sp for pts in points.values() for sp in pts]
        sync_table = SyncTable(all_points, fps_by_video)
        self._scrubber.reload_sync(sync_table)

        n = len(self._anchors)
        self._rough_status_label.setText(
            f"Rough sync applied ({n} camera{'s' if n != 1 else ''})."
        )
        self._rough_status_label.setStyleSheet("color: green; font-size: 11px;")

    # ------------------------------------------------------------------
    # Slots — LED sync setup
    # ------------------------------------------------------------------

    def _on_set_led_roi(self, cell_idx: int) -> None:
        if self._scrubber is None:
            return
        shot_idx = self._shot_combo.currentIndex()
        if shot_idx < 0 or shot_idx >= len(self._shots):
            return
        sv = self._shots[shot_idx].videos[cell_idx]
        initial_frame = self._scrubber.current_frames[cell_idx]
        total_frames = max(sv.last_video_frame - sv.first_video_frame + 1, 1)

        dlg = _ROISelectDialog(
            file_path=sv.file_path,
            total_frames=total_frames,
            fps=sv.actual_fps or 30.0,
            initial_frame=initial_frame,
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            roi = dlg.selected_roi()
            if roi is not None and roi.is_valid:
                self._led_rois[cell_idx] = roi
                self._led_roi_labels[cell_idx].setText(
                    f"ROI: ({roi.x1},{roi.y1})→({roi.x2},{roi.y2})"
                )
                self._led_roi_labels[cell_idx].setStyleSheet("color: green; font-size: 11px;")
                self._update_led_run_btn()

    # ------------------------------------------------------------------
    # Slots — LED sync job
    # ------------------------------------------------------------------

    def _on_run_led_sync(self) -> None:
        shot_idx = self._shot_combo.currentIndex()
        if shot_idx < 0 or shot_idx >= len(self._shots):
            return
        shot = self._shots[shot_idx]

        cam_data = []
        for cell_idx, sv in enumerate(shot.videos):
            roi = self._led_rois.get(cell_idx)
            if roi is None:
                continue
            fps_override = self._led_fps_spinboxes[cell_idx].value()
            if fps_override <= 0.0:
                fps_override = sv.actual_fps or 30.0
            cam_data.append((sv.file_path, roi, fps_override, sv.camera_instance_id, sv.id))

        if len(cam_data) < 2:
            return

        self._led_run_btn.setEnabled(False)
        self._led_progress_bar.setValue(0)
        self._led_progress_bar.setVisible(True)
        self._led_quality_widget.setVisible(False)
        self._led_accept_btn.setEnabled(False)
        self._led_plot_btn.setEnabled(False)
        self._led_accept_label.setText("")

        self._led_job = _LedSyncJob(cam_data, ref_cam=0, parent=self)
        self._led_job.progress.connect(self._on_led_progress)
        self._led_job.finished.connect(self._on_led_sync_done)
        self._led_job.error.connect(self._on_led_sync_error)
        self._led_job.start()

    def _on_led_progress(self, pct: int, msg: str) -> None:
        self._led_progress_bar.setValue(pct)
        self._led_progress_label.setText(msg)

    def _on_led_sync_done(self, result: LedSyncResult) -> None:
        self._led_progress_bar.setVisible(False)
        self._led_progress_label.setText("")
        self._led_result = result
        self._led_run_btn.setEnabled(True)

        # Update quality labels
        while self._led_quality_layout.count():
            item = self._led_quality_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._led_quality_labels = []
        for cr in result.cameras:
            if cr.map_type == "reference":
                txt = f"{cr.camera_instance_id}: reference"
                style = "color: grey; font-size: 11px;"
            else:
                quality = "good" if cr.resid_std_s < 0.005 else "poor"
                txt = (
                    f"{cr.camera_instance_id}: {cr.n_events} events, "
                    f"{cr.n_inliers} inliers, "
                    f"σ={cr.resid_std_s * 1000:.1f} ms — {quality}"
                )
                style = (
                    "color: green; font-size: 11px;"
                    if quality == "good"
                    else "color: orange; font-size: 11px;"
                )
            lbl = QLabel(txt)
            lbl.setStyleSheet(style)
            self._led_quality_layout.addWidget(lbl)
            self._led_quality_labels.append(lbl)
        self._led_quality_layout.addStretch()
        self._led_quality_widget.setVisible(True)

        self._led_accept_btn.setEnabled(True)
        self._led_plot_btn.setEnabled(_HAS_MATPLOTLIB)
        self._led_accept_label.setText("Review quality above, then click Accept.")

    def _on_led_sync_error(self, msg: str) -> None:
        self._led_progress_bar.setVisible(False)
        self._led_run_btn.setEnabled(True)
        self._led_accept_label.setText(f"Error: {msg}")
        self._led_accept_label.setStyleSheet("color: red; font-size: 11px;")

    def _on_accept_led_sync(self) -> None:
        if self._led_result is None or self._scrubber is None:
            return

        shot_idx = self._shot_combo.currentIndex()
        shot = self._shots[shot_idx]

        points, fps_by_video = _sync_points_from_led_result(self._led_result)

        ctx: DBContext = self.wizard().db_context
        ctx.write_sync_config(shot.shot_id, "led-auto", points)
        ctx._conn.commit()

        all_points = [sp for pts in points.values() for sp in pts]
        sync_table = SyncTable(all_points, fps_by_video)
        self._scrubber.reload_sync(sync_table)

        self._led_accept_label.setText("LED sync accepted and applied.")
        self._led_accept_label.setStyleSheet("color: green; font-size: 11px;")
        self._led_accept_btn.setEnabled(False)

    def _on_show_brightness_plot(self) -> None:
        if self._led_result is None:
            return
        dlg = _BrightnessPlotDialog(self._led_result, parent=self)
        dlg.exec()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rebuild_anchor_labels(self, shot: _ShotMeta) -> None:
        while self._anchor_status_layout.count():
            item = self._anchor_status_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._anchor_labels = []
        for sv in shot.videos:
            lbl = QLabel(f"{sv.camera_instance_id}: —")
            lbl.setStyleSheet("font-size: 11px;")
            self._anchor_status_layout.addWidget(lbl)
            self._anchor_labels.append(lbl)
        self._anchor_status_layout.addStretch()

    def _rebuild_led_panel(self, shot: _ShotMeta) -> None:
        """Rebuild per-camera rows in the LED panel for the given shot."""
        # Clear previous rows
        while self._led_cam_rows_layout.count():
            item = self._led_cam_rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._led_rois.clear()
        self._led_fps_spinboxes = []
        self._led_roi_labels = []

        for cell_idx, sv in enumerate(shot.videos):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            cam_lbl = QLabel(sv.camera_instance_id)
            cam_lbl.setFixedWidth(70)
            cam_lbl.setStyleSheet("font-size: 11px; font-weight: bold;")

            roi_lbl = QLabel("ROI: not set")
            roi_lbl.setStyleSheet("color: grey; font-size: 11px;")
            roi_lbl.setMinimumWidth(180)

            set_roi_btn = QPushButton("Set ROI…")
            set_roi_btn.setFixedWidth(80)
            set_roi_btn.clicked.connect(
                lambda checked=False, idx=cell_idx: self._on_set_led_roi(idx)
            )

            fps_lbl = QLabel("fps:")
            fps_lbl.setStyleSheet("font-size: 11px;")

            fps_spin = QDoubleSpinBox()
            fps_spin.setRange(0.0, 960.0)
            fps_spin.setDecimals(3)
            fps_spin.setValue(sv.actual_fps or 30.0)
            fps_spin.setFixedWidth(75)
            fps_spin.setSpecialValueText("auto")
            fps_spin.setToolTip(
                "Override the container fps.\n"
                "Set to 0 to use the value probed from the file.\n"
                "Useful for Android slow-motion clips that record at e.g. "
                "120 fps inside a 30 fps container."
            )

            row_layout.addWidget(cam_lbl)
            row_layout.addWidget(roi_lbl, stretch=1)
            row_layout.addWidget(set_roi_btn)
            row_layout.addWidget(fps_lbl)
            row_layout.addWidget(fps_spin)

            self._led_cam_rows_layout.addWidget(row)
            self._led_roi_labels.append(roi_lbl)
            self._led_fps_spinboxes.append(fps_spin)

        # Reset quality and accept widgets
        while self._led_quality_layout.count():
            item = self._led_quality_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._led_quality_labels = []
        self._led_quality_widget.setVisible(False)
        self._led_result = None
        self._led_accept_btn.setEnabled(False)
        self._led_plot_btn.setEnabled(False)
        self._led_accept_label.setText("")
        self._update_led_run_btn()

    def _update_led_run_btn(self) -> None:
        shot_idx = self._shot_combo.currentIndex()
        if shot_idx < 0 or shot_idx >= len(self._shots):
            self._led_run_btn.setEnabled(False)
            return
        n_videos = len(self._shots[shot_idx].videos)
        n_set = len(self._led_rois)
        self._led_run_btn.setEnabled(n_set >= 2 and n_set == n_videos)

    def _update_rough_panel_state(self) -> None:
        n_anchored = len(self._anchors)
        shot_idx = self._shot_combo.currentIndex()
        n_total = len(self._shots[shot_idx].videos) if 0 <= shot_idx < len(self._shots) else 0
        can_apply = n_anchored >= 2
        self._apply_rough_btn.setEnabled(can_apply)
        if n_anchored == 0:
            msg = "Set an anchor on at least two cameras."
        elif n_anchored < n_total:
            msg = (
                f"{n_anchored} / {n_total} cameras anchored — "
                "can apply (unanchored cameras will not be synced)."
            )
        else:
            msg = f"All {n_total} cameras anchored."
        self._rough_status_label.setText(msg)
        self._rough_status_label.setStyleSheet("color: grey; font-size: 11px;")

    def _teardown_scrubber(self) -> None:
        # Stop any running LED sync job
        if self._led_job is not None and self._led_job.isRunning():
            self._led_job.requestInterruption()
            self._led_job.wait(3000)
            self._led_job = None

        if self._scrubber is not None:
            self._scrubber.shutdown()
            self._scrubber_layout.removeWidget(self._scrubber)
            self._scrubber.deleteLater()
            self._scrubber = None
        if self._cache is not None:
            self._cache.close_all()
            self._cache = None

        self._anchors.clear()
        self._anchor_overlays.clear()
        self._led_rois.clear()
        self._led_result = None

        self._set_anchor_btn.setEnabled(False)
        self._clear_anchors_btn.setEnabled(False)
        self._apply_rough_btn.setEnabled(False)
        self._rough_status_label.setText("")
        self._rough_status_label.setStyleSheet("color: grey; font-size: 11px;")
        self._led_run_btn.setEnabled(False)
        self._led_progress_bar.setVisible(False)
        self._led_progress_label.setText("")
        self._led_accept_btn.setEnabled(False)
        self._led_plot_btn.setEnabled(False)
        self._led_accept_label.setText("")
        self._led_quality_widget.setVisible(False)

    def _show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._error_label.setVisible(True)
        self._scrubber = None  # make tests see None
