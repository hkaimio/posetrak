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
import time

log = logging.getLogger(__name__)

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
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
        shot_id: str,
        parent: QWidget | None = None,
        trial_id: str | None = None,
    ) -> None:
        """*shot_id* is the capture this segmentation belongs to (see
        docs/roadmap/features/segmentation-reuse/segmentation-reuse-design.md's
        terminology note: ``capture_videos.shot_id`` etc. reference
        ``captures``, not a separate shots table). No detection run needs
        to exist yet -- segmentation is capture-scoped and independent of
        any specific detection run; one gets created lazily, on demand,
        when pose extraction is actually queued (``_resolve_or_create_detection_run``).
        *trial_id* is optional provenance (which trial this panel happened
        to be opened from) threaded onto any detection run created here;
        not required for anything to function."""
        super().__init__(parent)
        self._conn = conn
        self._shot_id = shot_id
        self._trial_id = trial_id
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

        # DB path for PoseWorker (needs its own connection for thread-safe writes)
        _db_path = ""
        for row in self._conn.execute("PRAGMA database_list"):
            if row[1] == "main":
                _db_path = row[2]
                break

        # Tracking state
        from app.pose.job_queue_runner import JobQueueRunner
        self._runner = JobQueueRunner(db_path=_db_path, parent=self)

        # Pose overlay
        self._pose_detection_run_id: str | None = None
        self._skeleton_overlay = None
        try:
            from app.pose.frame_view import SkeletonDetectionOverlay
            self._skeleton_overlay = SkeletonDetectionOverlay()
        except Exception:
            pass
        self._runner.mask_ready.connect(self._on_batch_ready)
        self._runner.progress.connect(self._on_track_progress)
        self._runner.job_started.connect(self._on_job_started)
        self._runner.job_finished.connect(self._on_job_finished)
        self._runner.job_failed.connect(self._on_job_failed)
        self._runner.queue_done.connect(self._on_queue_done)

        self._seg_init_run_id: str | None = None   # seg_quality_run created for this session
        self._db_flush_buffer: list[tuple] = []    # buffered (svid, frame_idx, blob) rows
        self._DB_FLUSH_EVERY = 50           # flush to DB every N frames
        self._batch_recv_count: int = 0     # batches received for current job (for logging)
        self._t_job_recv_start: float = 0.0

        self._build_ui()
        if self._skeleton_overlay is not None:
            self._canvas.set_skeleton_overlay(self._skeleton_overlay)
        self._load_run()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self._encode_timer.stop()
        self._runner.shutdown()
        self._frame_cache.close()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # Left: video + controls
        left_widget = QWidget()
        root = QVBoxLayout(left_widget)
        root.setContentsMargins(0, 0, 0, 0)
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
        track_group = QGroupBox("Tracking  (seed from current mask, add to queue)")
        track_vbox = QVBoxLayout(track_group)
        track_vbox.setContentsMargins(4, 2, 4, 4)
        track_vbox.setSpacing(4)
        track_layout = QHBoxLayout()
        track_vbox.addLayout(track_layout)
        track_layout.setContentsMargins(0, 0, 0, 0)

        self._track_bwd_btn = QPushButton("◀ Queue Backward")
        self._track_bwd_btn.setToolTip(
            "Add a backward tracking job to the queue (current frame → mark start)"
        )
        self._track_bwd_btn.clicked.connect(self._on_track_backward)
        self._track_bwd_btn.setEnabled(False)
        track_layout.addWidget(self._track_bwd_btn)

        self._track_fwd_btn = QPushButton("▶ Queue Forward")
        self._track_fwd_btn.setToolTip(
            "Add a forward tracking job to the queue (current frame → mark end)"
        )
        self._track_fwd_btn.clicked.connect(self._on_track_forward)
        self._track_fwd_btn.setEnabled(False)
        track_layout.addWidget(self._track_fwd_btn)

        self._stop_btn = QPushButton("■ Stop Current")
        self._stop_btn.setToolTip(
            "Stop the running job after the current frame; queue continues"
        )
        self._stop_btn.clicked.connect(self._on_stop_tracking)
        self._stop_btn.setEnabled(False)
        track_layout.addWidget(self._stop_btn)

        track_layout.addStretch()

        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(160)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setVisible(False)
        track_layout.addWidget(self._progress_bar)

        # --- Pose extraction row ---
        pose_row = QHBoxLayout()
        pose_row.setContentsMargins(0, 0, 0, 0)

        pose_row.addWidget(QLabel("Pose model:"))
        self._pose_model_combo = QComboBox()
        self._pose_model_combo.addItem("RTMPose-L 133kp", "rtmpose-l-133kp")
        self._pose_model_combo.addItem("VITpose-L 133kp", "vitpose-l-133kp")
        self._pose_model_combo.setToolTip("Pose estimator model for queued pose extraction")
        pose_row.addWidget(self._pose_model_combo)

        self._refine_hands_check = QCheckBox("Refine hands")
        self._refine_hands_check.setChecked(True)
        self._refine_hands_check.setToolTip(
            "After pose extraction, re-detect each tracked wrist's hand in a "
            "tight crop (rtmlib.Hand) and patch in the refined finger keypoints. "
            "Only has an effect for 133-keypoint pose models."
        )
        pose_row.addWidget(self._refine_hands_check)

        self._queue_pose_btn = QPushButton("🎯 Queue Pose")
        self._queue_pose_btn.setToolTip(
            "Queue pose extraction for the current camera using stored seg masks"
        )
        self._queue_pose_btn.clicked.connect(self._on_queue_pose_current)
        self._queue_pose_btn.setEnabled(False)
        pose_row.addWidget(self._queue_pose_btn)

        self._queue_pose_all_btn = QPushButton("🎯 Queue Pose — All Cameras")
        self._queue_pose_all_btn.setToolTip(
            "Queue pose extraction for all cameras using their stored seg masks"
        )
        self._queue_pose_all_btn.clicked.connect(self._on_queue_pose_all)
        self._queue_pose_all_btn.setEnabled(False)
        pose_row.addWidget(self._queue_pose_all_btn)

        pose_row.addStretch()
        track_vbox.addLayout(pose_row)    # second row inside the group box

        root.addWidget(track_group)

        # --- Status bar ---
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("font-size: 10px; color: #555;")
        root.addWidget(self._status_label)

        # --- Assemble outer layout ---
        outer.addWidget(left_widget, stretch=1)
        outer.addWidget(self._build_queue_panel(), stretch=0)

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

    def _build_queue_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(240)
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(2, 4, 2, 4)
        vbox.setSpacing(4)

        vbox.addWidget(QLabel("<b>Job Queue</b>"))

        self._now_running_label = QLabel("Idle")
        self._now_running_label.setStyleSheet(
            "font-size: 10px; color: #888; padding: 0px 0px 2px 0px;"
        )
        self._now_running_label.setWordWrap(True)
        vbox.addWidget(self._now_running_label)

        self._job_list = QListWidget()
        self._job_list.setAlternatingRowColors(True)
        self._job_list.setSelectionMode(
            QListWidget.SelectionMode.SingleSelection
        )
        vbox.addWidget(self._job_list, stretch=1)

        self._queue_status_label = QLabel("")
        self._queue_status_label.setStyleSheet("font-size: 10px; color: #666;")
        vbox.addWidget(self._queue_status_label)

        self._run_queue_btn = QPushButton("▶  Run Queue")
        self._run_queue_btn.setToolTip("Start executing all pending jobs")
        self._run_queue_btn.clicked.connect(self._on_run_queue)
        vbox.addWidget(self._run_queue_btn)

        btn_row = QHBoxLayout()
        remove_btn = QPushButton("Remove")
        remove_btn.setToolTip("Remove selected pending job from queue")
        remove_btn.clicked.connect(self._on_remove_job)
        btn_row.addWidget(remove_btn)

        cancel_all_btn = QPushButton("Cancel All")
        cancel_all_btn.setToolTip(
            "Stop current job and cancel all pending jobs"
        )
        cancel_all_btn.clicked.connect(self._on_cancel_all)
        btn_row.addWidget(cancel_all_btn)
        vbox.addLayout(btn_row)

        return w

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_run(self) -> None:
        shot_id = self._shot_id
        # track_first/track_last: narrow the scrubber to whichever range any
        # prior detection run on this capture already covers (falls back to
        # the full video range below when none exists) -- scoped across
        # every detection run for the capture now, not one specific run,
        # since segmentation is no longer tied to a single detection run.
        cam_rows = self._conn.execute(
            "SELECT cv.id, cv.file_path, cv.first_video_frame, cv.last_video_frame, "
            "       cv.actual_fps, "
            "       COALESCE(ci.label, cv.camera_instance_id) AS label, "
            "       MIN(pt.first_frame) AS track_first, "
            "       MAX(pt.last_frame)  AS track_last "
            "FROM capture_videos cv "
            "LEFT JOIN camera_instances ci ON ci.id = cv.camera_instance_id "
            "LEFT JOIN person_tracks pt "
            "       ON pt.shot_video_id = cv.id "
            "       AND pt.detection_run_id IN (SELECT id FROM detection_runs WHERE shot_id = ?) "
            "WHERE cv.shot_id = ? "
            "GROUP BY cv.id "
            "ORDER BY label",
            (shot_id, shot_id),
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
            "WHERE shot_id = ? ORDER BY created_at DESC LIMIT 1",
            (shot_id,),
        ).fetchone()
        if seg_row:
            self._seg_run_id = seg_row["id"]

        # Persons: capture-level definitions (can exist before any detection
        # has run -- the real prerequisite for segmentation, not "detection
        # already ran") union'd with any names already assigned to tracks
        # from a prior detection run on this capture, so captures that went
        # through the old detect-first flow keep working unchanged. See
        # docs/roadmap/features/segmentation-reuse/segmentation-reuse-design.md.
        from posetrak.db.manage_person import list_persons
        person_names = {r["name"] for r in list_persons(self._conn, shot_id)}
        person_names |= {
            r["person_name"] for r in self._conn.execute(
                "SELECT DISTINCT dta.person_name FROM detection_track_assignments dta "
                "JOIN detection_runs dr ON dr.id = dta.detection_run_id "
                "WHERE dr.shot_id = ?",
                (shot_id,),
            ).fetchall()
        }
        self._persons = sorted(person_names)

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

        # Enable track / pose buttons if there are persons to track
        can_track = bool(self._persons)
        self._track_fwd_btn.setEnabled(can_track)
        self._track_bwd_btn.setEnabled(can_track)
        self._queue_pose_btn.setEnabled(can_track)
        self._queue_pose_all_btn.setEnabled(can_track)

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

        # Probe native video resolution so the skeleton overlay scales keypoints
        # (stored at 4K) correctly over the 1920p FrameCache display frames.
        try:
            _cap = cv2.VideoCapture(cam["file_path"])
            _w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            _h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _cap.release()
            if _w > 0 and _h > 0:
                self._canvas.set_keypoint_resolution(_w, _h)
        except Exception:
            pass

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
        """Query seg_masks across all relevant runs and update the range bar."""
        run_ids = self._read_run_ids()
        if not run_ids:
            self._range_bar.set_covered_frames([])
            return
        placeholders = ",".join("?" * len(run_ids))
        rows = self._conn.execute(
            f"SELECT DISTINCT frame_idx FROM seg_masks "
            f"WHERE seg_quality_run_id IN ({placeholders}) AND shot_video_id=? "
            f"ORDER BY frame_idx",
            [*run_ids, cam["id"]],
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

        # Update skeleton overlay before display so it paints in the same render call.
        self._update_skeleton_overlay(cam, frame_idx)

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
        """Load a previously saved seg_mask blob from the DB.

        Tries the interactive run first (most recent edits), then falls back
        to the original batch/prior run so frames outside the tracked range
        remain visible.
        """
        run_ids = self._read_run_ids()
        for run_id in run_ids:
            row = self._conn.execute(
                "SELECT mask_blob FROM seg_masks "
                "WHERE seg_quality_run_id = ? AND shot_video_id = ? AND frame_idx = ?",
                (run_id, shot_video_id, frame_idx),
            ).fetchone()
            if row is not None:
                buf = np.frombuffer(bytes(row["mask_blob"]), dtype=np.uint8)
                return _decode_mask_png(buf)
        return None

    def _update_skeleton_overlay(self, cam: dict, frame_idx: int) -> None:
        """Load pose keypoints from DB and update the canvas skeleton overlay."""
        if self._skeleton_overlay is None or self._pose_detection_run_id is None:
            if self._skeleton_overlay is not None:
                self._skeleton_overlay.clear()
            return

        run_id = self._pose_detection_run_id
        svid   = cam["id"]

        dets = self._conn.execute(
            "SELECT track_id, bbox_x, bbox_y, bbox_w, bbox_h "
            "FROM person_detections "
            "WHERE detection_run_id=? AND shot_video_id=? AND video_frame=? "
            "AND region_type='full_body' ORDER BY track_id",
            (run_id, svid, frame_idx),
        ).fetchall()

        kp_dict: dict[int, np.ndarray] = {}
        for det in dets:
            row = self._conn.execute(
                "SELECT keypoints FROM detection_keypoints "
                "WHERE detection_run_id=? AND shot_video_id=? "
                "AND video_frame=? AND track_id=? AND region_type='full_body'",
                (run_id, svid, frame_idx, det["track_id"]),
            ).fetchone()
            if row:
                kp_bytes = bytes(row["keypoints"])
                n = len(kp_bytes) // (3 * 4)
                kp_dict[det["track_id"]] = np.frombuffer(
                    kp_bytes, dtype=np.float32
                ).reshape(n, 3)

        assignments = {i + 1: name for i, name in enumerate(self._persons)}
        self._skeleton_overlay.set_detections([dict(d) for d in dets], kp_dict)
        self._skeleton_overlay.set_assignments(assignments)

    def _read_run_ids(self) -> list[str]:
        """Ordered list of seg_quality_run IDs to try when reading masks.

        Interactive run (if started) takes priority; original batch run is
        the fallback so untracked frames remain visible.
        """
        ids: list[str] = []
        if self._seg_init_run_id is not None:
            ids.append(self._seg_init_run_id)
        if self._seg_run_id is not None and self._seg_run_id != self._seg_init_run_id:
            ids.append(self._seg_run_id)
        return ids

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def _on_track_forward(self) -> None:
        self._queue_tracking("forward")

    def _on_track_backward(self) -> None:
        self._queue_tracking("backward")

    def _queue_tracking(self, direction: str) -> None:
        """Create a TrackingJob from the current UI state and enqueue it."""
        cam = self._cam_combo.currentData()
        if cam is None or not self._persons:
            return

        # Seed mask: live SAM2 result or stored DB mask.
        seed_mask = None
        if self._controller and np.any(self._controller.get_mask()):
            seed_mask = self._controller.get_mask().copy()
        else:
            seed_mask = self._load_stored_mask(cam["id"], self._scrubber.value())

        if seed_mask is None or not np.any(seed_mask):
            self._set_status("No mask on current frame — click to create one first.")
            return

        self._ensure_seg_run()

        # PNG-encode the seed mask so the job is fully self-contained.
        ok, buf = cv2.imencode(".png", seed_mask.astype(np.uint8))
        if not ok:
            self._set_status("Failed to encode seed mask.")
            return

        from app.pose.job_queue_runner import TrackingJob
        import uuid
        job = TrackingJob(
            job_id=str(uuid.uuid4())[:8],
            camera_label=cam["label"],
            shot_video_id=cam["id"],
            video_path=cam["file_path"],
            init_frame=self._scrubber.value(),
            init_mask_png=buf.tobytes(),
            persons_ordered=list(self._persons),
            first_frame=self._mark_start,
            last_frame=self._mark_end,
            direction=direction,
            max_dim=self._frame_cache._max_dim,
        )
        self._runner.enqueue(job)
        self._refresh_queue_list()
        self._set_status(f"Queued {direction} job for {cam['label']}.")

    def _on_stop_tracking(self) -> None:
        self._runner.stop_current()
        self._set_status("Stopping current job…")

    def _on_run_queue(self) -> None:
        self._runner.start()
        self._refresh_queue_list()

    def _on_remove_job(self) -> None:
        item = self._job_list.currentItem()
        if item is None:
            return
        job_id = item.data(Qt.ItemDataRole.UserRole)
        if self._runner.remove_pending(job_id):
            self._refresh_queue_list()

    def _on_cancel_all(self) -> None:
        self._runner.cancel_all()
        self._refresh_queue_list()
        self._set_status("Queue cancelled.")

    def _on_queue_pose_current(self) -> None:
        cam = self._cam_combo.currentData()
        if cam is None:
            return
        self._queue_pose_jobs([cam])

    def _on_queue_pose_all(self) -> None:
        self._queue_pose_jobs(self._cameras)

    def _queue_pose_jobs(self, cameras: list[dict]) -> None:
        """Create and enqueue PoseExtractionJobs for the given cameras."""
        pose_model = self._pose_model_combo.currentData()
        refine_hands = self._refine_hands_check.isChecked()
        detection_run_id = self._resolve_or_create_detection_run(pose_model)
        if detection_run_id is None:
            return  # user cancelled

        from app.pose.pose_worker import PoseExtractionJob
        import uuid

        queued = 0
        for cam in cameras:
            run_ids = self._read_run_ids()
            if not run_ids:
                continue
            # Check masks exist for this camera in any relevant run.
            placeholders = ",".join("?" * len(run_ids))
            count = self._conn.execute(
                f"SELECT COUNT(*) FROM seg_masks "
                f"WHERE seg_quality_run_id IN ({placeholders}) AND shot_video_id=?",
                [*run_ids, cam["id"]],
            ).fetchone()[0]
            if count == 0:
                continue

            # Prefer the interactive run; fall back to original batch run.
            seg_run_id = run_ids[0]

            job = PoseExtractionJob(
                job_id=str(uuid.uuid4())[:8],
                camera_label=cam["label"],
                shot_video_id=cam["id"],
                video_path=cam["file_path"],
                detection_run_id=detection_run_id,
                seg_quality_run_id=seg_run_id,
                persons_ordered=list(self._persons),
                first_frame=cam["track_first"],
                last_frame=cam["track_last"],
                pose_model=pose_model,
                overwrite_range=True,
                refine_hands=refine_hands,
            )
            self._runner.enqueue(job)
            queued += 1

        if queued > 0:
            self._refresh_queue_list()
            self._set_status(f"Queued {queued} pose job(s) — {pose_model}.")
        else:
            self._set_status("No seg masks found for any camera. Run segmentation first.")

    def _resolve_or_create_detection_run(self, pose_model: str) -> str | None:
        """Return a detection_run_id to write pose results into, or None if cancelled.

        Creates a new run silently if none exists for this capture yet;
        asks the user what to do if a run with the same pose_model already
        exists. No longer resolves shot_id/sync_config_id from a pre-
        existing "parent" detection run -- the panel is capture-scoped
        (self._shot_id) now, so this is the one place that still needs a
        sync config to exist before it can create a fresh detection run.
        """
        from app.pose.db_cache import create_detection_run
        from PySide6.QtWidgets import QMessageBox

        shot_id = self._shot_id
        sync_row = self._conn.execute(
            "SELECT id FROM sync_configs WHERE shot_id = ? ORDER BY rowid DESC LIMIT 1",
            (shot_id,),
        ).fetchone()
        if sync_row is None:
            self._set_status("No sync config for this capture yet — set one up first.")
            return None
        sync_cfg_id = sync_row["id"]
        trial_id = self._trial_id

        existing = self._conn.execute(
            "SELECT id, created_at FROM detection_runs "
            "WHERE shot_id=? AND pose_model=? AND status != 'failed' "
            "ORDER BY created_at DESC",
            (shot_id, pose_model),
        ).fetchall()

        try:
            import rtmlib
            pose_ver = getattr(rtmlib, "__version__", "")
        except ImportError:
            pose_ver = ""
        from app.pose.backends_rtmpose import _KNOWN_MODELS as _PM
        _pm_hw = _PM.get(pose_model, (None, (0, 0), None, 1.0))[1]  # (H, W)
        pose_w, pose_h = _pm_hw[1], _pm_hw[0]

        if not existing:
            # time_start_s/time_end_s are provenance metadata on
            # detection_runs, not the actual per-camera frame gating (that's
            # PoseExtractionJob.first_frame/last_frame, resolved per camera
            # from cam["track_first"]/["track_last"] below) -- no single
            # meaningful range to record without a parent run to inherit
            # one from, so 0.0/0.0.
            return create_detection_run(
                self._conn, shot_id, sync_cfg_id,
                0.0, 0.0,
                detector_model="cutie-interactive",
                pose_model=pose_model,
                trial_id=trial_id,
                pose_version=pose_ver,
                pose_input_width=pose_w,
                pose_input_height=pose_h,
            )

        # Ask user: update existing or create new
        run = existing[0]
        created = str(run["created_at"])[:19].replace("T", " ")
        msg = QMessageBox(self)
        msg.setWindowTitle("Pose Extraction")
        msg.setText(
            f"A <b>{pose_model}</b> detection run already exists<br>"
            f"(created {created}).<br><br>"
            f"<b>Update existing</b> — overwrites keypoints for the queued cameras/frames<br>"
            f"<b>Create new</b> — adds a separate detection run for this capture"
        )
        update_btn = msg.addButton("Update existing", QMessageBox.ButtonRole.AcceptRole)
        new_btn    = msg.addButton("Create new run",  QMessageBox.ButtonRole.ActionRole)
        cancel_btn = msg.addButton("Cancel",          QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked is cancel_btn:
            return None
        if clicked is update_btn:
            return run["id"]

        return create_detection_run(
            self._conn, shot_id, sync_cfg_id,
            0.0, 0.0,
            detector_model="cutie-interactive",
            pose_model=pose_model,
            trial_id=trial_id,
            pose_version=pose_ver,
            pose_input_width=pose_w,
            pose_input_height=pose_h,
        )

    # ------------------------------------------------------------------
    # Runner signal handlers
    # ------------------------------------------------------------------

    def _on_job_started(self, job_id: str) -> None:
        self._stop_btn.setEnabled(True)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._batch_recv_count = 0
        self._t_job_recv_start = time.monotonic()
        log.info("Panel: job_started  job_id=%s  t=%.3f", job_id, self._t_job_recv_start)
        for job in self._runner.jobs:
            if job.job_id == job_id:
                from app.pose.pose_worker import PoseExtractionJob as _PoseJob
                if isinstance(job, _PoseJob):
                    summary = f"🎯 Pose  {job.camera_label}  {job.first_frame}–{job.last_frame}"
                    status_txt = f"🎯 Pose extraction {job.camera_label}…"
                else:
                    arrow = "▶" if job.direction == "forward" else "◀"
                    summary = (f"{arrow} {job.direction.capitalize()}  {job.camera_label}  "
                               f"frames {job.first_frame}–{job.last_frame}")
                    status_txt = f"{arrow} Tracking {job.camera_label}…"
                self._now_running_label.setText(f"Running: {summary}")
                self._now_running_label.setStyleSheet(
                    "font-size: 10px; color: #4af; padding: 0px 0px 2px 0px;"
                )
                self._set_status(status_txt)
                break
        self._refresh_queue_list()

    def _on_batch_ready(self, svid: str, batch: list) -> None:
        """Process a batch of tracked frames from the worker.

        The worker accumulates masks in batches of ~50 frames before emitting,
        so at most ~60 signals are queued for a 3000-frame job — regardless of
        GIL contention between the worker and main threads.  Each signal carries
        PNG-encoded masks (~50–150 KB each) to keep per-signal memory small.
        """
        self._batch_recv_count += 1
        t = time.monotonic()
        if self._batch_recv_count == 1:
            self._t_job_recv_start = t
            log.info("Panel: first batch received  svid=%s  frames=%d  t=%.3f", svid, len(batch), t)
        elif self._batch_recv_count % 10 == 0:
            log.debug("Panel: batch %d received  svid=%s  t=%.3f  (+%.3fs since first)",
                      self._batch_recv_count, svid, t, t - self._t_job_recv_start)

        for frame_idx, mask_png in batch:
            self._db_flush_buffer.append((svid, frame_idx, mask_png))
        if len(self._db_flush_buffer) >= self._DB_FLUSH_EVERY:
            self._flush_masks()

        # Advance scrubber to the last frame in the batch.
        cam = self._cam_combo.currentData()
        if cam and svid == cam["id"] and batch:
            last_fi = batch[-1][0]
            self._scrubber.blockSignals(True)
            self._scrubber.setValue(last_fi)
            self._scrubber.blockSignals(False)
            self._range_bar.set_position(last_fi)

    def _on_track_progress(self, done: int, total: int) -> None:
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(done)

    def _on_job_finished(self, job_id: str, units_written: int) -> None:
        t = time.monotonic()
        log.info("Panel: _on_job_finished  job_id=%s  units_written=%d  "
                 "batches_received=%d  buffer_pending=%d  t=%.3f",
                 job_id, units_written, self._batch_recv_count,
                 len(self._db_flush_buffer), t)

        # Determine whether this was a pose job or segmentation job.
        from app.pose.pose_worker import PoseExtractionJob as _PoseJob
        is_pose = any(
            isinstance(j, _PoseJob) and j.job_id == job_id
            for j in self._runner.jobs
        )

        if is_pose:
            # Find the detection_run_id so we can show the overlay.
            for j in self._runner.jobs:
                if j.job_id == job_id:
                    self._pose_detection_run_id = j.detection_run_id
                    break
            self._refresh_queue_list()
            cam = self._cam_combo.currentData()
            if cam:
                self._show_frame(self._scrubber.value())
            self._set_status(f"Pose done — {units_written} frames with keypoints.")
        else:
            self._flush_masks()
            self._conn.commit()  # no-op if _flush_masks already committed everything
            if self._controller:
                self._controller.clear_all()
            self._encoded_frame_idx = -1
            cam = self._cam_combo.currentData()
            if cam:
                self._refresh_coverage_bar(cam)
                self._show_frame(self._scrubber.value())
            self._refresh_queue_list()
            self._set_status(f"Segmentation done — {units_written} masks written.")

    def _on_job_failed(self, job_id: str, error: str) -> None:
        self._flush_masks()
        self._conn.commit()
        self._refresh_queue_list()
        if error.startswith("CUTIE_SETUP_ERROR:"):
            detail = error[len("CUTIE_SETUP_ERROR:"):]
            msg = QMessageBox(self)
            msg.setWindowTitle("Cutie not found")
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setText("Cutie segmentation library is not installed.")
            msg.setInformativeText(detail)
            msg.exec()
            self._set_status("Cutie not found — see dialog for setup instructions.")
        else:
            self._set_status(f"Job failed: {error}")

    def _on_queue_done(self) -> None:
        self._stop_btn.setEnabled(False)
        self._progress_bar.setVisible(False)
        self._progress_bar.setValue(0)
        self._now_running_label.setText("Idle")
        self._now_running_label.setStyleSheet(
            "font-size: 10px; color: #888; padding: 0px 0px 2px 0px;"
        )
        self._refresh_queue_list()

    # ------------------------------------------------------------------
    # Queue list widget helpers
    # ------------------------------------------------------------------

    def _refresh_queue_list(self) -> None:
        from PySide6.QtGui import QFont
        self._job_list.clear()
        status_icons = {
            "pending":   "⏳",
            "running":   "▶",
            "done":      "✓",
            "failed":    "✗",
            "cancelled": "—",
        }
        from app.pose.pose_worker import PoseExtractionJob as _PoseJob
        pending = running = done = 0
        running_item = None
        for job in self._runner.jobs:
            icon = status_icons.get(job.status, "?")
            if isinstance(job, _PoseJob):
                model_short = job.pose_model.split("-")[0].upper()
                extra = f"  ({job.keypoints_written} kp)" if job.status == "done" else ""
                if job.status == "failed":
                    extra = f"  {job.error[:24]}"
                text = (f"{icon} 🎯 {job.camera_label}  "
                        f"{job.first_frame}–{job.last_frame}  [{model_short}]{extra}")
            else:
                arrow = "▶" if job.direction == "forward" else "◀"
                extra = f"  ({job.masks_written} masks)" if job.status == "done" else ""
                if job.status == "failed":
                    extra = f"  {job.error[:24]}"
                text = f"{icon} {arrow} {job.camera_label}  {job.first_frame}–{job.last_frame}{extra}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, job.job_id)
            if job.status == "running":
                item.setBackground(QColor(50, 100, 180))
                item.setForeground(QColor(220, 240, 255))
                font = QFont()
                font.setBold(True)
                item.setFont(font)
                running_item = item
            elif job.status == "done":
                item.setForeground(QColor(100, 200, 100))
            elif job.status == "pending":
                item.setForeground(QColor(200, 200, 200))
            elif job.status in ("failed", "cancelled"):
                item.setForeground(QColor(120, 120, 120))
            self._job_list.addItem(item)
            if job.status == "pending":   pending  += 1
            elif job.status == "running": running  += 1
            elif job.status == "done":    done     += 1

        if running_item is not None:
            self._job_list.scrollToItem(running_item)

        self._queue_status_label.setText(
            f"{pending} pending  {running} running  {done} done"
        )
        # Run Queue enabled only when there are pending jobs and queue is idle.
        has_pending = pending > 0
        self._run_queue_btn.setEnabled(has_pending and not self._runner.is_running)

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _ensure_seg_run(self) -> None:
        """Create a seg_quality_run for this capture if not already done.

        time_start_s/time_end_s: an interactively-created segmentation
        isn't scoped to a specific sub-range of the capture today (masks
        are stored per-frame regardless of range) -- 0.0/a large sentinel
        means "covers the whole capture," trivially satisfying any trial's
        future containment check. Narrowing this to the actually-segmented
        range is a real future refinement, not done here. See
        docs/roadmap/features/segmentation-reuse/segmentation-reuse-design.md.
        """
        if self._seg_init_run_id is not None:
            return
        run_id = generate_id()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO seg_quality_runs "
            "(id, shot_id, trial_id, time_start_s, time_end_s, created_at, "
            " quality_source, erosion_px) "
            "VALUES (?, ?, ?, 0.0, 1e9, ?, 'cutie-interactive', 5)",
            (run_id, self._shot_id, self._trial_id, now),
        )
        self._conn.commit()
        self._seg_init_run_id = run_id
        # _seg_run_id intentionally left unchanged — it is the original batch/prior
        # run and serves as a fallback in _load_stored_mask for frames not yet
        # covered by the new interactive run.
        log.debug("Created seg_quality_run %s for interactive init", run_id)
        # Coverage bar now shows both runs; will fill in as tracking runs.
        cam = self._cam_combo.currentData()
        if cam:
            self._refresh_coverage_bar(cam)

    def _flush_masks(self) -> None:
        """Write buffered mask blobs to the seg_masks table and commit immediately.

        Committing here (not only in _on_job_finished) ensures masks are durable
        even if tracking_done is somehow processed before all batch signals, or if
        _on_job_finished is called with an already-empty buffer.
        """
        if not self._db_flush_buffer or self._seg_init_run_id is None:
            log.debug("Panel: _flush_masks skipped  buffer=%d  run_id=%s",
                      len(self._db_flush_buffer), self._seg_init_run_id)
            return
        n = len(self._db_flush_buffer)
        t0 = time.monotonic()
        self._conn.executemany(
            "INSERT OR REPLACE INTO seg_masks "
            "(seg_quality_run_id, shot_video_id, frame_idx, mask_blob) "
            "VALUES (?, ?, ?, ?)",
            [
                (self._seg_init_run_id, svid, fi, blob)
                for svid, fi, blob in self._db_flush_buffer
            ],
        )
        self._conn.commit()
        self._db_flush_buffer.clear()
        log.debug("Panel: flushed %d masks to DB  %.1fms", n, (time.monotonic() - t0) * 1000)

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
