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
import threading
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger(__name__)

from PySide6.QtCore import QEvent, QObject, QThread, Qt, Signal
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
    save_brightness_dump,
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
# Background frame reader (used inside the ROI dialog)
# ---------------------------------------------------------------------------


class _DialogFrameReader(QThread):
    """Decodes individual video frames in the background for the ROI dialog.

    Rapid slider moves coalesce: only the most recently requested frame is
    decoded.
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
        self._reader = _DialogFrameReader(file_path, self)
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
        n_videos = len(self._shot.videos)
        self._run_btn.setEnabled(len(self._led_rois) >= 2 and len(self._led_rois) == n_videos)

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
# SyncPage
# ---------------------------------------------------------------------------


class SyncPage(QWizardPage):
    """Wizard page — camera synchronisation."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTitle("Camera Synchronisation")
        self.setSubTitle(
            "Scroll each camera to a common reference event, set an anchor per "
            "camera, then apply rough sync.  Use the LED sync dialog for more "
            "accurate automated alignment.  Click a cell to focus it; use "
            "←/→ (±1 frame) or Shift+←/→ (±10 frames) to navigate."
        )

        self._shots: list[_ShotMeta] = []
        self._cache: FrameCache | None = None
        self._scrubber: MultiVideoScrubber | None = None

        # Rough-sync state (reset on shot change)
        self._anchors: dict[int, int] = {}
        self._anchor_overlays: list[SyncAnchorOverlay] = []
        self._anchor_labels: list[QLabel] = []

        # Per-camera fps override spinboxes (in rough sync panel, rebuilt per shot)
        self._fps_spinboxes: list[QDoubleSpinBox] = []

        # ---- shot selector ----
        self._shot_combo = QComboBox()
        self._shot_combo.currentIndexChanged.connect(self._on_shot_selected)

        # ---- sync config selector ----
        self._sync_config_combo = QComboBox()
        self._sync_config_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._sync_config_combo.setToolTip(
            "Select the sync configuration to apply to this shot.\n"
            "LED-auto configs are preferred; the most recent is pre-selected."
        )
        self._sync_config_combo.currentIndexChanged.connect(self._on_sync_config_selected)

        shot_bar = QHBoxLayout()
        shot_bar.addWidget(QLabel("Shot:"))
        shot_bar.addWidget(self._shot_combo)
        shot_bar.addWidget(QLabel("Sync config:"))
        shot_bar.addWidget(self._sync_config_combo, stretch=1)
        shot_bar.addStretch()

        # ---- scrubber area ----
        self._scrubber_container = QWidget()
        self._scrubber_layout = QVBoxLayout(self._scrubber_container)
        self._scrubber_layout.setContentsMargins(0, 0, 0, 0)

        # ---- rough sync panel ----
        self._rough_panel = QGroupBox("Rough synchronisation")
        rough_layout = QVBoxLayout(self._rough_panel)
        rough_layout.setSpacing(4)

        # Per-camera fps rows (dynamic, rebuilt per shot)
        self._fps_rows_widget = QWidget()
        self._fps_rows_layout = QVBoxLayout(self._fps_rows_widget)
        self._fps_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._fps_rows_layout.setSpacing(2)
        rough_layout.addWidget(self._fps_rows_widget)

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

        # Per-camera anchor status labels (dynamic)
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

        # LED sync entry point
        led_row = QHBoxLayout()
        self._led_sync_btn = QPushButton("LED synchronisation…")
        self._led_sync_btn.setEnabled(False)
        self._led_sync_btn.setToolTip(
            "Open the LED sync dialog to automate synchronisation using a blinking LED.\n"
            "Apply rough sync first to establish a starting alignment."
        )
        self._led_sync_btn.clicked.connect(self._on_open_led_sync)
        led_row.addStretch()
        led_row.addWidget(self._led_sync_btn)
        rough_layout.addLayout(led_row)

        # ---- error label ----
        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: red;")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)

        # ---- main layout ----
        layout = QVBoxLayout(self)
        layout.addLayout(shot_bar)
        layout.addWidget(self._scrubber_container, stretch=1)
        layout.addWidget(self._rough_panel)
        layout.addWidget(self._error_label)

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
                label=sv.camera_label,
            )
            for sv in shot.videos
        ]

        self._cache = FrameCache()
        scrubber = MultiVideoScrubber(cells_info, self._cache, self._scrubber_container)
        self._scrubber_layout.addWidget(scrubber)
        self._scrubber = scrubber

        self._anchor_overlays = [
            SyncAnchorOverlay(total_frames=info.total_frames)
            for info in cells_info
        ]
        for i, ov in enumerate(self._anchor_overlays):
            scrubber.set_overlays(i, [ov])

        self._rebuild_per_camera_widgets(shot)

        self._set_anchor_btn.setEnabled(True)
        self._clear_anchors_btn.setEnabled(True)
        self._update_rough_panel_state()

        # Populate sync config combo and auto-load the best available config
        self._populate_sync_config_combo(shot)

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
        cam = shot.videos[fc].camera_label
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
                cam = shot.videos[i].camera_label
                lbl.setText(f"{cam}: —")
        if self._scrubber:
            self._scrubber.reload_sync(None)
        self._led_sync_btn.setEnabled(False)
        self._update_rough_panel_state()

    def _on_apply_rough_sync(self) -> None:
        if not self._anchors or self._scrubber is None:
            return

        shot_idx = self._shot_combo.currentIndex()
        shot = self._shots[shot_idx]

        ref_cell = min(self._anchors)
        ref_frame = self._anchors[ref_cell]
        ref_sv = shot.videos[ref_cell]
        # Use fps override if the user has set one
        ref_fps = self._fps_overrides().get(ref_cell, ref_sv.actual_fps or 30.0) or 30.0
        ref_ts = ref_frame / ref_fps

        points: dict[str, list[SyncPoint]] = {}
        fps_by_video: dict[str, float] = {}
        for cell_idx, anchor_frame in self._anchors.items():
            sv = shot.videos[cell_idx]
            cam_id = sv.camera_instance_id
            fps = self._fps_overrides().get(cell_idx, sv.actual_fps or 30.0) or 30.0
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
        # Persist corrected fps values so that reloading this sync config later
        # uses the same fps for interpolation (not the container fps).
        for cell_idx, sv in enumerate(shot.videos):
            fps = self._fps_overrides().get(cell_idx, sv.actual_fps or 30.0) or 30.0
            if abs(fps - (sv.actual_fps or 0.0)) > 0.01:
                ctx.update_shot_video_fps(sv.id, fps)
        ctx.write_sync_config(shot.shot_id, "manual-rough", points)
        ctx._conn.commit()

        all_points = [sp for pts in points.values() for sp in pts]
        sync_table = SyncTable(all_points, fps_by_video)
        self._scrubber.reload_sync(sync_table)
        # Seek to the anchor timestamp so the user sees their anchor frames
        self._scrubber.seek_synced(ref_ts)

        n = len(self._anchors)
        self._rough_status_label.setText(
            f"Rough sync applied ({n} camera{'s' if n != 1 else ''})."
        )
        self._rough_status_label.setStyleSheet("color: green; font-size: 11px;")
        self._led_sync_btn.setEnabled(True)
        # Refresh sync config dropdown to show the new entry.
        self._populate_sync_config_combo(shot)

    # ------------------------------------------------------------------
    # Slots — LED sync dialog
    # ------------------------------------------------------------------

    def _on_open_led_sync(self) -> None:
        if self._scrubber is None:
            return
        shot_idx = self._shot_combo.currentIndex()
        if shot_idx < 0 or shot_idx >= len(self._shots):
            return
        shot = self._shots[shot_idx]
        ctx: DBContext = self.wizard().db_context
        current_frames = self._scrubber.current_frames

        def _on_accepted(sync_table: SyncTable) -> None:
            self._scrubber.reload_sync(sync_table)
            self._rough_status_label.setText("LED sync accepted and applied.")
            self._rough_status_label.setStyleSheet("color: green; font-size: 11px;")
            # Clear anchor overlays — LED sync supersedes rough anchors
            for i, ov in enumerate(self._anchor_overlays):
                ov.anchor_frame = None
                self._scrubber.set_overlays(i, [ov])
            # Refresh sync config dropdown to show the new entry.
            self._populate_sync_config_combo(shot)

        # Derive per-camera rough offsets (global time at each camera's frame 0)
        # from the currently loaded sync table.  This is more reliable than
        # recomputing from self._anchors because the sync table was built with
        # the correct fps values and covers all cameras, including those that
        # were not manually anchored in the current UI session.
        sync_table_offsets: dict[int, float] = {}
        if self._scrubber and self._scrubber.sync_table is not None:
            st = self._scrubber.sync_table
            for cell_idx, sv in enumerate(shot.videos):
                t0 = st.frame_to_global_time(0, sv.id)
                if t0 is not None:
                    sync_table_offsets[cell_idx] = t0

        dlg = _LedSyncDialog(
            shot=shot,
            fps_overrides=self._fps_overrides(),
            ctx=ctx,
            current_frames=current_frames,
            on_sync_accepted=_on_accepted,
            anchor_frames=dict(self._anchors),
            sync_table_offsets=sync_table_offsets,
            parent=self,
        )
        dlg.exec()

    # ------------------------------------------------------------------
    # Slots — sync config selection
    # ------------------------------------------------------------------

    def _on_sync_config_selected(self, index: int) -> None:
        config_id = self._sync_config_combo.itemData(index)
        if config_id is None:
            # "No sync config" option
            if self._scrubber is not None:
                self._scrubber.reload_sync(None)
            self._led_sync_btn.setEnabled(False)
            return
        ctx: DBContext = self.wizard().db_context
        sync_table = ctx.load_sync_config(config_id)
        if sync_table is not None and self._scrubber is not None:
            self._scrubber.reload_sync(sync_table)
            self._led_sync_btn.setEnabled(True)
            self._rough_status_label.setText(
                f"Sync config loaded: {self._sync_config_combo.currentText()}"
            )
            self._rough_status_label.setStyleSheet("color: green; font-size: 11px;")
            # Clear anchor overlays — loaded config supersedes rough anchors
            for i, ov in enumerate(self._anchor_overlays):
                ov.anchor_frame = None
                if self._scrubber is not None:
                    self._scrubber.set_overlays(i, [ov])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fps_overrides(self) -> dict[int, float]:
        """Return the current per-camera fps override values."""
        shot_idx = self._shot_combo.currentIndex()
        if shot_idx < 0 or shot_idx >= len(self._shots):
            return {}
        return {
            i: spin.value()
            for i, spin in enumerate(self._fps_spinboxes)
        }

    def _populate_sync_config_combo(self, shot: _ShotMeta) -> None:
        """Fill the sync config combo for *shot* and load the default config."""
        ctx: DBContext = self.wizard().db_context
        configs = ctx.get_sync_configs(shot.shot_id)

        self._sync_config_combo.blockSignals(True)
        self._sync_config_combo.clear()

        if not configs:
            self._sync_config_combo.addItem("— no sync config —", None)
            self._sync_config_combo.blockSignals(False)
            self._on_sync_config_selected(0)
            return

        # Build labels; configs are already newest-first from the DB query.
        # Prefer led-auto → manual-rough for the default selection.
        method_rank = {"led-auto": 0, "manual-rough": 1}
        best_idx = 0
        best_rank = 99
        for i, (cfg_id, method) in enumerate(configs):
            n = len(configs) - i  # descending count label
            label = f"{method} #{n}"
            self._sync_config_combo.addItem(label, cfg_id)
            rank = method_rank.get(method, 99)
            if rank < best_rank:
                best_rank = rank
                best_idx = i

        self._sync_config_combo.blockSignals(False)
        self._sync_config_combo.setCurrentIndex(best_idx)
        # Always call explicitly — setCurrentIndex is a no-op when already at best_idx.
        self._on_sync_config_selected(best_idx)

    def _rebuild_per_camera_widgets(self, shot: _ShotMeta) -> None:
        """Rebuild per-camera fps spinboxes and anchor status labels."""
        # Clear fps rows
        while self._fps_rows_layout.count():
            item = self._fps_rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._fps_spinboxes = []

        for sv in shot.videos:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            cam_lbl = QLabel(sv.camera_label)
            cam_lbl.setFixedWidth(70)
            cam_lbl.setStyleSheet("font-size: 11px; font-weight: bold;")

            fps_lbl = QLabel("fps override:")
            fps_lbl.setStyleSheet("font-size: 11px;")

            fps_spin = QDoubleSpinBox()
            fps_spin.setRange(0.0, 960.0)
            fps_spin.setDecimals(3)
            fps_spin.setValue(sv.actual_fps or 30.0)
            fps_spin.setFixedWidth(75)
            fps_spin.setSpecialValueText("auto")
            fps_spin.setToolTip(
                "Override the fps for anchor-timestamp calculation.\n"
                "Set to 0 to use the probed container fps.\n"
                "Use the actual capture rate for slow-motion clips."
            )

            row_layout.addWidget(cam_lbl)
            row_layout.addWidget(fps_lbl)
            row_layout.addWidget(fps_spin)
            row_layout.addStretch()

            self._fps_rows_layout.addWidget(row)
            self._fps_spinboxes.append(fps_spin)

        # Clear and rebuild anchor status labels
        while self._anchor_status_layout.count():
            item = self._anchor_status_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._anchor_labels = []
        for sv in shot.videos:
            lbl = QLabel(f"{sv.camera_label}: —")
            lbl.setStyleSheet("font-size: 11px;")
            self._anchor_status_layout.addWidget(lbl)
            self._anchor_labels.append(lbl)
        self._anchor_status_layout.addStretch()

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
        self._set_anchor_btn.setEnabled(False)
        self._clear_anchors_btn.setEnabled(False)
        self._apply_rough_btn.setEnabled(False)
        self._led_sync_btn.setEnabled(False)
        self._rough_status_label.setText("")
        self._rough_status_label.setStyleSheet("color: grey; font-size: 11px;")
        self._sync_config_combo.currentIndexChanged.disconnect(self._on_sync_config_selected)
        self._sync_config_combo.clear()
        self._sync_config_combo.currentIndexChanged.connect(self._on_sync_config_selected)

    def _show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._error_label.setVisible(True)
        self._scrubber = None
