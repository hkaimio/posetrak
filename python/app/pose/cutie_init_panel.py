"""cutie_init_panel.py — Interactive Cutie segmentation initialisation panel.

Phase 1: video scrubber, camera selector, mask overlay from stored seg_masks.
Phase 2: click-to-SAM2 interaction — PersonSelector buttons, ClickController.
Phase 3: Cutie worker thread, Track/Stop buttons, mask persistence.
Phase 4: correction workflow, RTMPose post-step.
"""
from __future__ import annotations

import datetime
import logging
import sqlite3

log = logging.getLogger(__name__)

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.pose.frame_cache import FrameCache
from app.pose.video_canvas import VideoCanvas, label_to_color
from posetrak.db.db import generate_id


class RangeBar(QWidget):
    """Horizontal timeline bar combining three layers of information:

    1. Mask coverage (lower 5 px) — teal segments where frames have stored masks.
    2. Selected tracking range (full height) — steel-blue fill between Mark Start/End.
    3. Mark boundary ticks — bright blue.
    4. Current frame position — white tick.
    """

    HEIGHT = 14

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self._first = 0
        self._last  = 1
        self._sel_start = 0
        self._sel_end   = 1
        self._pos = 0
        # List of (first_frame, last_frame) contiguous covered runs.
        self._coverage_runs: list[tuple[int, int]] = []

    def set_range(self, first: int, last: int) -> None:
        self._first = first
        self._last  = max(last, first + 1)
        self._sel_start = first
        self._sel_end   = self._last
        self._pos = first
        self._coverage_runs = []
        self.update()

    def set_selection(self, start: int, end: int) -> None:
        self._sel_start = start
        self._sel_end   = end
        self.update()

    def set_position(self, frame: int) -> None:
        self._pos = frame
        self.update()

    def set_covered_frames(self, frame_indices: list[int]) -> None:
        """Compute contiguous runs from *frame_indices* and repaint."""
        if not frame_indices:
            self._coverage_runs = []
            self.update()
            return
        import numpy as np
        arr = np.array(sorted(frame_indices), dtype=np.int32)
        breaks = np.where(np.diff(arr) > 1)[0]
        starts = np.concatenate([[0], breaks + 1])
        ends   = np.concatenate([breaks + 1, [len(arr)]])
        self._coverage_runs = [
            (int(arr[s]), int(arr[e - 1])) for s, e in zip(starts, ends)
        ]
        self.update()

    def _to_x(self, frame: int) -> int:
        w = self.width()
        span = self._last - self._first
        if span <= 0:
            return 0
        return int((frame - self._first) / span * w)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        w, h = self.width(), self.height()
        cov_h = 5   # pixels reserved for coverage band at bottom

        # Background — full track range
        p.fillRect(0, 0, w, h, QColor(55, 55, 55))

        # Mask coverage — teal band in lower cov_h px
        cov_color = QColor(56, 168, 120)
        for run_first, run_last in self._coverage_runs:
            x1 = self._to_x(run_first)
            x2 = self._to_x(run_last) + 1
            p.fillRect(x1, h - cov_h, max(1, x2 - x1), cov_h, cov_color)

        # Selected range — steel blue, upper portion only
        sel_h = h - cov_h
        x1 = self._to_x(self._sel_start)
        x2 = self._to_x(self._sel_end)
        if x2 > x1:
            p.fillRect(x1, 0, x2 - x1, sel_h, QColor(70, 130, 180))

        # Start / end tick marks — brighter blue, full height
        for xm in (x1, x2):
            p.fillRect(max(0, xm - 1), 0, 2, h, QColor(120, 180, 240))

        # Current position — white tick, full height
        xp = self._to_x(self._pos)
        p.fillRect(xp, 0, 2, h, QColor(255, 255, 255))

        p.end()


class CutieInitPanel(QWidget):
    """Video scrubber + click-based SAM2 segmentation init panel.

    Camera selector → frame scrubber → VideoCanvas with mask overlay.
    PersonSelector buttons let the user choose which person label to assign
    to each click.  Left-click = positive (foreground), right-click = negative.

    SAM2 encoding is lazy: the image encoder runs on the first click on a frame
    (or immediately when a person button is selected and the video is accessible).
    """

    closed = Signal()

    def __init__(
        self,
        conn: sqlite3.Connection,
        detection_run_id: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._conn = conn
        self._run_id = detection_run_id
        self._frame_cache = FrameCache(max_frames=300, max_dim=1920)

        # Populated by _load_run()
        self._cameras: list[dict] = []
        self._seg_run_id: str | None = None
        self._persons: list[str] = []       # person names, index+1 = SAM label

        # Click interaction state
        self._controller = None             # ClickController; created lazily
        self._selected_label: int = 0       # 0 = no person selected
        self._encoded_frame_idx: int = -1   # frame that _controller has encoded
        self._encoded_svid: str = ""        # camera id for encoded frame

        # Debounce timer: encode image 300 ms after last scrub event when
        # a person button is active (pre-warm the encoder for fast first click).
        self._encode_timer = QTimer(self)
        self._encode_timer.setSingleShot(True)
        self._encode_timer.timeout.connect(self._encode_current_frame)

        # Tracking range (frame indices; set via Mark Start / Mark End buttons)
        self._mark_start: int = 0
        self._mark_end: int   = 0

        # Tracking state
        self._worker = None                 # CutieWorker QThread
        self._seg_init_run_id: str | None = None   # seg_quality_run created for this session
        self._init_frame_idx: int = -1      # frame used as Cutie seed
        self._db_flush_buffer: list[tuple] = []    # buffered (svid, frame_idx, blob) rows
        self._DB_FLUSH_EVERY = 50           # write to DB every N frames
        self._canvas_update_counter: int = 0       # throttle live canvas redraws

        self._build_ui()
        self._load_run()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self._encode_timer.stop()
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        self._frame_cache.close()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # --- Top bar: camera selector + frame info ---
        top = QHBoxLayout()
        top.addWidget(QLabel("Camera:"))
        self._cam_combo = QComboBox()
        self._cam_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._cam_combo.currentIndexChanged.connect(self._on_camera_changed)
        top.addWidget(self._cam_combo)
        top.addStretch()
        self._frame_label = QLabel("Frame —")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self._frame_label)
        root.addLayout(top)

        # --- Video canvas ---
        self._canvas = VideoCanvas()
        self._canvas.left_clicked.connect(self._on_left_click)
        self._canvas.right_clicked.connect(self._on_right_click)
        root.addWidget(self._canvas, stretch=1)

        # --- Scrubber ---
        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setMinimum(0)
        self._scrubber.setMaximum(0)
        self._scrubber.valueChanged.connect(self._on_frame_changed)
        root.addLayout(self._make_scrubber_row())

        # --- Person selector (populated after _load_run) ---
        self._person_group = QGroupBox("Person  (click to select, then click on canvas)")
        self._person_layout = QHBoxLayout(self._person_group)
        self._person_layout.setContentsMargins(4, 2, 4, 2)
        self._person_layout.setSpacing(4)
        self._person_btn_group = QButtonGroup(self)
        self._person_btn_group.setExclusive(False)
        root.addWidget(self._person_group)

        # --- Edit action buttons ---
        edit_row = QHBoxLayout()
        self._clear_person_btn = QPushButton("Clear person")
        self._clear_person_btn.setToolTip("Remove all clicks for the selected person")
        self._clear_person_btn.clicked.connect(self._on_clear_person)
        self._clear_person_btn.setEnabled(False)
        edit_row.addWidget(self._clear_person_btn)

        self._clear_all_btn = QPushButton("Clear all")
        self._clear_all_btn.setToolTip("Remove all clicks for all persons")
        self._clear_all_btn.clicked.connect(self._on_clear_all)
        self._clear_all_btn.setEnabled(False)
        edit_row.addWidget(self._clear_all_btn)
        edit_row.addStretch()

        self._sam_status_label = QLabel("")
        self._sam_status_label.setStyleSheet("font-size: 10px; color: #666;")
        edit_row.addWidget(self._sam_status_label)
        root.addLayout(edit_row)

        # --- Tracking controls ---
        track_group = QGroupBox("Tracking  (seed from current mask, then propagate)")
        track_layout = QHBoxLayout(track_group)
        track_layout.setContentsMargins(4, 2, 4, 2)

        self._track_bwd_btn = QPushButton("◀ Track Backward")
        self._track_bwd_btn.setToolTip(
            "Propagate Cutie from current frame backward to start of track range"
        )
        self._track_bwd_btn.clicked.connect(self._on_track_backward)
        self._track_bwd_btn.setEnabled(False)
        track_layout.addWidget(self._track_bwd_btn)

        self._track_fwd_btn = QPushButton("▶ Track Forward")
        self._track_fwd_btn.setToolTip(
            "Propagate Cutie from current frame to end of track range"
        )
        self._track_fwd_btn.clicked.connect(self._on_track_forward)
        self._track_fwd_btn.setEnabled(False)
        track_layout.addWidget(self._track_fwd_btn)

        self._stop_btn = QPushButton("■ Stop")
        self._stop_btn.setToolTip("Stop tracking after the current frame")
        self._stop_btn.clicked.connect(self._on_stop_tracking)
        self._stop_btn.setEnabled(False)
        track_layout.addWidget(self._stop_btn)

        track_layout.addStretch()

        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(160)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setVisible(False)
        track_layout.addWidget(self._progress_bar)

        root.addWidget(track_group)

        # --- Status bar ---
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("font-size: 10px; color: #555;")
        root.addWidget(self._status_label)

    def _make_scrubber_row(self) -> QVBoxLayout:
        vbox = QVBoxLayout()
        vbox.setSpacing(2)
        vbox.setContentsMargins(0, 0, 0, 0)

        slider_row = QHBoxLayout()
        slider_row.addWidget(self._scrubber, 1)
        vbox.addLayout(slider_row)

        self._range_bar = RangeBar()
        vbox.addWidget(self._range_bar)

        mark_row = QHBoxLayout()
        mark_row.setSpacing(4)

        self._mark_start_btn = QPushButton("Mark Start")
        self._mark_start_btn.setMaximumWidth(90)
        self._mark_start_btn.setToolTip("Set track-from to current frame")
        self._mark_start_btn.clicked.connect(self._on_mark_start)
        self._mark_start_label = QLabel("Start: —")

        self._mark_end_btn = QPushButton("Mark End")
        self._mark_end_btn.setMaximumWidth(90)
        self._mark_end_btn.setToolTip("Set track-to to current frame")
        self._mark_end_btn.clicked.connect(self._on_mark_end)
        self._mark_end_label = QLabel("End: —")

        mark_row.addWidget(self._mark_start_btn)
        mark_row.addWidget(self._mark_start_label)
        mark_row.addStretch()
        mark_row.addWidget(self._mark_end_label)
        mark_row.addWidget(self._mark_end_btn)
        vbox.addLayout(mark_row)

        return vbox

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_run(self) -> None:
        run_row = self._conn.execute(
            "SELECT shot_id FROM detection_runs WHERE id = ?",
            (self._run_id,),
        ).fetchone()
        if run_row is None:
            self._set_status("Detection run not found.")
            return

        shot_id = run_row["shot_id"]
        cam_rows = self._conn.execute(
            "SELECT cv.id, cv.file_path, cv.first_video_frame, cv.last_video_frame, "
            "       cv.actual_fps, "
            "       COALESCE(ci.label, cv.camera_instance_id) AS label, "
            "       MIN(pt.first_frame) AS track_first, "
            "       MAX(pt.last_frame)  AS track_last "
            "FROM capture_videos cv "
            "LEFT JOIN camera_instances ci ON ci.id = cv.camera_instance_id "
            "LEFT JOIN person_tracks pt "
            "       ON pt.shot_video_id = cv.id AND pt.detection_run_id = ? "
            "WHERE cv.shot_id = ? "
            "GROUP BY cv.id "
            "ORDER BY label",
            (self._run_id, shot_id),
        ).fetchall()

        self._cameras = [
            {
                "id": r["id"],
                "label": r["label"],
                "file_path": r["file_path"],
                "first": int(r["first_video_frame"]),
                "last": int(r["last_video_frame"]),
                "fps": float(r["actual_fps"] or 30.0),
                "track_first": int(r["track_first"]) if r["track_first"] is not None else int(r["first_video_frame"]),
                "track_last":  int(r["track_last"])  if r["track_last"]  is not None else int(r["last_video_frame"]),
            }
            for r in cam_rows
        ]

        seg_row = self._conn.execute(
            "SELECT id FROM seg_quality_runs "
            "WHERE detection_run_id = ? ORDER BY created_at DESC LIMIT 1",
            (self._run_id,),
        ).fetchone()
        if seg_row:
            self._seg_run_id = seg_row["id"]

        person_rows = self._conn.execute(
            "SELECT DISTINCT person_name FROM detection_track_assignments "
            "WHERE detection_run_id = ? ORDER BY person_name",
            (self._run_id,),
        ).fetchall()
        self._persons = [r["person_name"] for r in person_rows]

        self._rebuild_camera_combo()
        self._rebuild_person_selector()
        self._init_controller()

    def _rebuild_camera_combo(self) -> None:
        self._cam_combo.blockSignals(True)
        self._cam_combo.clear()
        for cam in self._cameras:
            self._cam_combo.addItem(cam["label"], cam)
        self._cam_combo.blockSignals(False)
        if self._cameras:
            self._on_camera_changed(0)

    def _rebuild_person_selector(self) -> None:
        # Clear old buttons
        for btn in self._person_btn_group.buttons():
            self._person_btn_group.removeButton(btn)
            btn.deleteLater()
        while self._person_layout.count():
            item = self._person_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._persons:
            self._person_group.setVisible(False)
            return

        self._person_group.setVisible(True)
        for i, name in enumerate(self._persons):
            label = i + 1  # 1-based SAM label
            r, g, b = label_to_color(label)
            luma = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "#000" if luma > 140 else "#fff"
            btn = QPushButton(f"{label}: {name}")
            btn.setCheckable(True)
            btn.setProperty("person_label", label)
            btn.setStyleSheet(
                f"QPushButton {{ background-color: rgb({r},{g},{b}); color: {text_color};"
                f" border: 2px solid transparent; padding: 3px 8px; }}"
                f"QPushButton:checked {{ border: 2px solid #000; font-weight: bold; }}"
            )
            # Keyboard shortcut 1..9
            if label <= 9:
                QShortcut(QKeySequence(str(label)), self).activated.connect(
                    lambda checked=False, b=btn: self._toggle_person_btn(b)
                )
            btn.clicked.connect(lambda checked, b=btn: self._on_person_btn_clicked(b))
            self._person_btn_group.addButton(btn, label)
            self._person_layout.addWidget(btn)

        self._person_layout.addStretch()

    def _init_controller(self) -> None:
        """Create the ClickController lazily (imports SAM2 if available)."""
        try:
            from app.pose.cutie_click_controller import ClickController
            self._controller = ClickController()
            if self._controller.available:
                self._sam_status_label.setText("SAM2 ready")
                self._sam_status_label.setStyleSheet("font-size: 10px; color: #080;")
            else:
                self._sam_status_label.setText("SAM2 not available — install ultralytics")
                self._sam_status_label.setStyleSheet("font-size: 10px; color: #c60;")
        except Exception as e:
            self._sam_status_label.setText(f"SAM2 error: {e}")

        # Enable track buttons if there are persons to track
        can_track = bool(self._persons)
        self._track_fwd_btn.setEnabled(can_track)
        self._track_bwd_btn.setEnabled(can_track)

    # ------------------------------------------------------------------
    # Interaction — person selector
    # ------------------------------------------------------------------

    def _toggle_person_btn(self, btn: QPushButton) -> None:
        btn.setChecked(not btn.isChecked())
        self._on_person_btn_clicked(btn)

    def _on_person_btn_clicked(self, clicked_btn: QPushButton) -> None:
        label = clicked_btn.property("person_label")
        if clicked_btn.isChecked():
            # Deselect all others (manual exclusivity so clicking same btn toggles off)
            for btn in self._person_btn_group.buttons():
                if btn is not clicked_btn:
                    btn.setChecked(False)
            self._selected_label = label
            self._canvas.setCursor(Qt.CursorShape.CrossCursor)
            # Pre-warm encoder when person is selected and frame is accessible
            self._schedule_encode()
        else:
            self._selected_label = 0
            self._canvas.setCursor(Qt.CursorShape.ArrowCursor)
            self._encode_timer.stop()

        self._update_edit_buttons()

    def _update_edit_buttons(self) -> None:
        has_controller = self._controller is not None
        has_selection = self._selected_label > 0
        self._clear_person_btn.setEnabled(has_controller and has_selection)
        self._clear_all_btn.setEnabled(has_controller)

    # ------------------------------------------------------------------
    # Interaction — canvas clicks
    # ------------------------------------------------------------------

    def _on_left_click(self, x: int, y: int) -> None:
        self._handle_click(x, y, positive=True)

    def _on_right_click(self, x: int, y: int) -> None:
        self._handle_click(x, y, positive=False)

    def _handle_click(self, x: int, y: int, positive: bool) -> None:
        if self._selected_label == 0:
            self._set_status("Select a person button first, then click on the frame.")
            return
        if self._controller is None or not self._controller.available:
            self._set_status("SAM2 not available.")
            return

        cam = self._cam_combo.currentData()
        frame_idx = self._scrubber.value()
        if cam is None:
            return

        # Encode if needed (lazy, also handles first click after scrub)
        self._ensure_encoded(cam, frame_idx)

        self._set_status("Running SAM2…")
        mask = self._controller.push_point(self._selected_label, x, y, positive)
        self._set_status(
            f"Person {self._selected_label}: "
            f"{self._controller.click_count(self._selected_label)} click(s)"
        )
        self._refresh_overlay(cam, frame_idx, mask)

    def _on_clear_person(self) -> None:
        """Remove live SAM2 clicks for the selected person on the current frame.

        Does NOT touch stored DB masks — after clearing, the display falls back
        to the stored mask (if any) so the user can re-draw just that person.
        """
        if self._controller is None or self._selected_label == 0:
            return
        self._controller.clear_person(self._selected_label)
        self._set_status(f"Cleared live clicks for person {self._selected_label}")
        self._show_frame(self._scrubber.value())

    def _on_clear_all(self) -> None:
        """Remove all live SAM2 clicks for the current frame.

        Does NOT touch stored DB masks.
        """
        if self._controller is None:
            return
        self._controller.clear_all()
        self._set_status("Cleared all live clicks")
        self._show_frame(self._scrubber.value())

    # ------------------------------------------------------------------
    # Interaction — scrubbing
    # ------------------------------------------------------------------

    def _on_camera_changed(self, index: int) -> None:
        cam = self._cam_combo.itemData(index)
        if cam is None:
            return
        # Camera switch invalidates the encoded frame
        self._encoded_frame_idx = -1
        self._encoded_svid = ""
        if self._controller:
            self._controller.clear_all()

        self._scrubber.blockSignals(True)
        self._scrubber.setMinimum(cam["track_first"])
        self._scrubber.setMaximum(cam["track_last"])
        self._scrubber.setValue(cam["track_first"])
        self._scrubber.blockSignals(False)

        # Reset marks to full track range for the new camera
        self._mark_start = cam["track_first"]
        self._mark_end   = cam["track_last"]
        self._range_bar.set_range(cam["track_first"], cam["track_last"])
        self._update_mark_labels(cam)
        self._refresh_coverage_bar(cam)

        self._show_frame(cam["track_first"])

    def _on_frame_changed(self, frame_idx: int) -> None:
        # Frame changed: clear click state, show new frame.
        if self._controller:
            self._controller.clear_all()
        self._encoded_frame_idx = -1
        self._range_bar.set_position(frame_idx)
        self._show_frame(frame_idx)
        # If a person is selected, pre-warm encoder after scrubbing stops.
        if self._selected_label > 0:
            self._schedule_encode()

    def _refresh_coverage_bar(self, cam: dict) -> None:
        """Query seg_masks for covered frames and update the range bar."""
        if self._seg_run_id is None:
            self._range_bar.set_covered_frames([])
            return
        rows = self._conn.execute(
            "SELECT frame_idx FROM seg_masks "
            "WHERE seg_quality_run_id=? AND shot_video_id=? ORDER BY frame_idx",
            (self._seg_run_id, cam["id"]),
        ).fetchall()
        self._range_bar.set_covered_frames([r["frame_idx"] for r in rows])

    def _on_mark_start(self) -> None:
        cam = self._cam_combo.currentData()
        if cam is None:
            return
        self._mark_start = self._scrubber.value()
        if self._mark_start > self._mark_end:
            self._mark_end = self._mark_start
        self._range_bar.set_selection(self._mark_start, self._mark_end)
        self._update_mark_labels(cam)

    def _on_mark_end(self) -> None:
        cam = self._cam_combo.currentData()
        if cam is None:
            return
        self._mark_end = self._scrubber.value()
        if self._mark_end < self._mark_start:
            self._mark_start = self._mark_end
        self._range_bar.set_selection(self._mark_start, self._mark_end)
        self._update_mark_labels(cam)

    def _update_mark_labels(self, cam: dict) -> None:
        fps = cam.get("fps", 1) or 1
        first = cam["track_first"]
        ts = (self._mark_start - first) / fps
        te = (self._mark_end   - first) / fps
        ms_mm, ms_ss = divmod(ts, 60)
        me_mm, me_ss = divmod(te, 60)
        self._mark_start_label.setText(
            f"Start: {self._mark_start} ({int(ms_mm):02d}:{ms_ss:05.2f})"
        )
        self._mark_end_label.setText(
            f"End: {self._mark_end} ({int(me_mm):02d}:{me_ss:05.2f})"
        )

    def _schedule_encode(self) -> None:
        """Start/restart the debounce timer to encode the current frame."""
        self._encode_timer.start(300)

    def _encode_current_frame(self) -> None:
        """Called by debounce timer: encode current frame if accessible."""
        cam = self._cam_combo.currentData()
        frame_idx = self._scrubber.value()
        if cam is not None:
            self._ensure_encoded(cam, frame_idx)

    def _ensure_encoded(self, cam: dict, frame_idx: int) -> None:
        """Encode the frame for SAM2 if not already done for this frame."""
        if (
            self._controller is None
            or not self._controller.available
            or (self._encoded_frame_idx == frame_idx and self._encoded_svid == cam["id"])
        ):
            return

        frame = self._frame_cache.get_frame(cam["file_path"], frame_idx)
        if frame is None:
            return

        self._set_status("Encoding frame for SAM2…")
        self._controller.set_image(frame)
        # Load stored mask as base so other persons are preserved during editing.
        self._controller.set_base_mask(self._load_stored_mask(cam["id"], frame_idx))
        self._encoded_frame_idx = frame_idx
        self._encoded_svid = cam["id"]
        self._set_status("Ready — click to segment")

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _show_frame(self, frame_idx: int) -> None:
        cam = self._cam_combo.currentData()
        if cam is None:
            return

        frame = self._frame_cache.get_frame(cam["file_path"], frame_idx)
        # Prefer live controller mask; fall back to stored DB mask.
        mask = None
        if self._controller and np.any(self._controller.get_mask()):
            mask = self._controller.get_mask()
        else:
            mask = self._load_stored_mask(cam["id"], frame_idx)

        if frame is None:
            import os
            msg = (
                f"Video not accessible:\n{cam['file_path']}"
                if not os.path.exists(cam["file_path"])
                else f"Could not decode frame {frame_idx}"
            )
            self._canvas.display(None, message=msg)
        else:
            clicks = self._controller.get_all_clicks() if self._controller else {}
            display_frame = _draw_click_markers(frame, clicks) if clicks else frame
            self._canvas.display(display_frame, mask)

        t = (frame_idx - cam["track_first"]) / cam["fps"]
        mm, ss = divmod(t, 60)
        seg_indicator = " [mask]" if mask is not None else ""
        self._frame_label.setText(
            f"Frame {frame_idx}  ({int(mm):02d}:{ss:05.2f}){seg_indicator}"
        )

    def _refresh_overlay(
        self, cam: dict | None, frame_idx: int, mask: np.ndarray
    ) -> None:
        """Redraw the canvas with *mask* and click markers."""
        if cam is None:
            return
        frame = self._frame_cache.get_frame(cam["file_path"], frame_idx)
        if frame is not None:
            frame = _draw_click_markers(
                frame, self._controller.get_all_clicks() if self._controller else {}
            )
        self._canvas.display(frame, mask if np.any(mask) else None)

    def _load_stored_mask(
        self, shot_video_id: str, frame_idx: int
    ) -> np.ndarray | None:
        """Load a previously saved seg_mask blob from the DB."""
        if self._seg_run_id is None:
            return None
        row = self._conn.execute(
            "SELECT mask_blob FROM seg_masks "
            "WHERE seg_quality_run_id = ? AND shot_video_id = ? AND frame_idx = ?",
            (self._seg_run_id, shot_video_id, frame_idx),
        ).fetchone()
        if row is None:
            return None
        buf = np.frombuffer(bytes(row["mask_blob"]), dtype=np.uint8)
        return _decode_mask_png(buf)

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def _on_track_forward(self) -> None:
        self._start_tracking("forward")

    def _on_track_backward(self) -> None:
        self._start_tracking("backward")

    def _start_tracking(self, direction: str) -> None:
        cam = self._cam_combo.currentData()
        if cam is None or not self._persons:
            return

        # The current mask (from ClickController or stored) is the seed.
        seed_mask = None
        if self._controller and np.any(self._controller.get_mask()):
            seed_mask = self._controller.get_mask().copy()
        else:
            seed_mask = self._load_stored_mask(cam["id"], self._scrubber.value())

        if seed_mask is None or not np.any(seed_mask):
            self._set_status("No mask on current frame — click to create one first.")
            return

        init_frame_idx = self._scrubber.value()
        self._init_frame_idx = init_frame_idx
        self._ensure_seg_run()

        from app.pose.cutie_worker import CutieWorker
        self._worker = CutieWorker(
            video_path=cam["file_path"],
            init_frame=init_frame_idx,
            init_mask=seed_mask,
            persons_ordered=self._persons,
            first_frame=self._mark_start,
            last_frame=self._mark_end,
            direction=direction,
            max_dim=self._frame_cache._max_dim,
        )
        self._worker.mask_ready.connect(
            lambda fi, m, svid=cam["id"]: self._on_mask_ready(svid, fi, m)
        )
        self._worker.progress.connect(self._on_track_progress)
        self._worker.finished.connect(self._on_tracking_finished)
        self._worker.error.connect(self._on_tracking_error)

        self._set_tracking_ui(True)
        self._set_status(
            f"Tracking {direction} from frame {init_frame_idx}…"
        )
        self._worker.start()

    def _on_stop_tracking(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._set_status("Stopping…")

    def _on_mask_ready(self, svid: str, frame_idx: int, mask: np.ndarray) -> None:
        """Slot called in UI thread for each tracked frame."""
        # output_prob_to_mask returns int64; PNG encoder requires uint8.
        mask_u8 = mask.astype(np.uint8)
        ok, buf = cv2.imencode(".png", mask_u8)
        if ok:
            self._db_flush_buffer.append((svid, frame_idx, buf.tobytes()))
        if len(self._db_flush_buffer) >= self._DB_FLUSH_EVERY:
            self._flush_masks()

        # Live canvas update: advance scrubber; throttle redraws to every 10 frames
        # so that queued signals don't cause excessive repaints.
        cam = self._cam_combo.currentData()
        if cam and svid == cam["id"]:
            self._scrubber.blockSignals(True)
            self._scrubber.setValue(frame_idx)
            self._scrubber.blockSignals(False)
            self._range_bar.set_position(frame_idx)
            self._canvas_update_counter += 1
            if self._canvas_update_counter % 10 == 0:
                frame = self._frame_cache.get_frame(cam["file_path"], frame_idx)
                self._canvas.display(frame, mask_u8 if np.any(mask_u8) else None)

    def _on_track_progress(self, done: int, total: int) -> None:
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(done)

    def _on_tracking_finished(self) -> None:
        self._flush_masks()
        self._conn.commit()
        self._set_tracking_ui(False)
        # Clear stale SAM2 click state so _show_frame uses stored DB masks.
        if self._controller:
            self._controller.clear_all()
        self._encoded_frame_idx = -1
        n = self._conn.execute(
            "SELECT COUNT(*) FROM seg_masks WHERE seg_quality_run_id=?",
            (self._seg_init_run_id,),
        ).fetchone()[0]
        self._set_status(f"Tracking complete — {n} masks saved.")
        cam = self._cam_combo.currentData()
        if cam:
            self._refresh_coverage_bar(cam)
            self._show_frame(self._scrubber.value())

    def _on_tracking_error(self, message: str) -> None:
        self._flush_masks()
        self._set_tracking_ui(False)
        self._set_status(f"Error: {message}")

    def _set_tracking_ui(self, tracking: bool) -> None:
        """Enable/disable controls appropriately during tracking."""
        self._track_fwd_btn.setEnabled(not tracking)
        self._track_bwd_btn.setEnabled(not tracking)
        self._stop_btn.setEnabled(tracking)
        self._cam_combo.setEnabled(not tracking)
        self._scrubber.setEnabled(not tracking)
        for btn in self._person_btn_group.buttons():
            btn.setEnabled(not tracking)
        self._clear_person_btn.setEnabled(not tracking)
        self._clear_all_btn.setEnabled(not tracking)
        self._progress_bar.setVisible(tracking)
        if not tracking:
            self._progress_bar.setValue(0)

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _ensure_seg_run(self) -> None:
        """Create a seg_quality_run for this session if not already done."""
        if self._seg_init_run_id is not None:
            return
        run_id = generate_id()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO seg_quality_runs "
            "(id, detection_run_id, created_at, quality_source, erosion_px) "
            "VALUES (?, ?, ?, 'cutie-interactive', 5)",
            (run_id, self._run_id, now),
        )
        self._conn.commit()
        self._seg_init_run_id = run_id
        # Switch mask reads to the new run so scrubbing shows new masks.
        self._seg_run_id = run_id
        log.debug("Created seg_quality_run %s for interactive init", run_id)
        # Coverage bar now shows the new (empty) run; will fill in as tracking runs.
        cam = self._cam_combo.currentData()
        if cam:
            self._refresh_coverage_bar(cam)

    def _flush_masks(self) -> None:
        """Write buffered mask blobs to the seg_masks table."""
        if not self._db_flush_buffer or self._seg_init_run_id is None:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO seg_masks "
            "(seg_quality_run_id, shot_video_id, frame_idx, mask_blob) "
            "VALUES (?, ?, ?, ?)",
            [
                (self._seg_init_run_id, svid, fi, blob)
                for svid, fi, blob in self._db_flush_buffer
            ],
        )
        self._db_flush_buffer.clear()

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)


def _draw_click_markers(
    frame_bgr: np.ndarray,
    clicks: dict[int, list[tuple[int, int, bool]]],
) -> np.ndarray:
    """Return a copy of *frame_bgr* with click point markers drawn.

    Positive clicks: filled circle in person colour with white border.
    Negative clicks: filled circle in red with white border.
    """
    from app.pose.video_canvas import label_to_color
    out = frame_bgr.copy()
    for label, pts in clicks.items():
        r, g, b = label_to_color(label)
        person_bgr = (b, g, r)
        neg_bgr = (40, 40, 220)   # red-ish for negative
        for x, y, positive in pts:
            fill = person_bgr if positive else neg_bgr
            cv2.circle(out, (x, y), 7, (255, 255, 255), -1)   # white border
            cv2.circle(out, (x, y), 5, fill, -1)               # person colour
            # Cross for negative points
            if not positive:
                cv2.line(out, (x - 4, y - 4), (x + 4, y + 4), (255, 255, 255), 1)
                cv2.line(out, (x + 4, y - 4), (x - 4, y + 4), (255, 255, 255), 1)
    return out


def _decode_mask_png(buf: np.ndarray) -> np.ndarray | None:
    """Decode an indexed PNG mask blob to a (H, W) uint8 label array."""
    import cv2
    decoded = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        return None
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded
