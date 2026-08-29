# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""cutie_init_panel.py — Interactive Cutie segmentation initialisation panel.

Phase 1: video scrubber, camera selector, mask overlay from stored seg_masks.
Phase 2: click-to-SAM2 interaction — PersonSelector buttons, ClickController.
Phase 3: Cutie worker thread, Track/Stop buttons, mask persistence.
Phase 4: correction workflow, RTMPose post-step.
"""
from __future__ import annotations

import datetime
import json
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
    """Horizontal timeline bar combining six layers of information:

    1. Queued-job ranges (lower 5 px, drawn first) — gold segments where a
       pending/running tracking job covers frames not yet actually masked.
       See set_queued_ranges().
    2. Mask coverage (lower 5 px, drawn over queued) — teal segments where
       frames have stored masks.
    3. Selected tracking range (full height) — steel-blue fill between Mark Start/End.
    4. Trial range (top 3 px) — amber band showing the trial this panel was opened
       from, if any -- purely informational, independent of the Mark Start/End
       selection (see docs/roadmap/features/segmentation-reuse/status.md's
       2026-08-16 note: segmentation is capture-scoped, so the selected range is
       deliberately free to differ from the trial's own bounds -- redo just part
       of a trial, or run wider than one trial on purpose).
    5. Split points — magenta ticks (brighter red when the current selection
       spans one), see set_split_points().
    6. Mark boundary ticks — bright blue.
    7. Current frame position — white tick.

    The bar's own coordinate space is whatever unit the caller uses when
    calling set_range/set_selection/set_position/set_trial_range -- global-
    time scrubber units when a sync table is available (see
    CutieInitPanel._local_frame_for), or raw per-camera video frame indices
    as a fallback when it isn't. The bar itself has no opinion on which.
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
        self._trial_start: int | None = None
        self._trial_end: int | None = None
        # List of (first_frame, last_frame) contiguous covered runs.
        self._coverage_runs: list[tuple[int, int]] = []
        self._split_points: list[int] = []
        self._queued_ranges: list[tuple[int, int]] = []

    def set_range(self, first: int, last: int) -> None:
        self._first = first
        self._last  = max(last, first + 1)
        self._sel_start = first
        self._sel_end   = self._last
        self._pos = first
        self._trial_start = None
        self._trial_end = None
        self._coverage_runs = []
        self.update()

    def set_selection(self, start: int, end: int) -> None:
        self._sel_start = start
        self._sel_end   = end
        self.update()

    def set_trial_range(self, start: int | None, end: int | None) -> None:
        self._trial_start = start
        self._trial_end = end
        self.update()

    def set_position(self, frame: int) -> None:
        self._pos = frame
        self.update()

    def set_split_points(self, points: list[int]) -> None:
        self._split_points = sorted(points)
        self.update()

    def set_queued_ranges(self, ranges: list[tuple[int, int]]) -> None:
        self._queued_ranges = ranges
        self.update()

    def set_covered_frames(self, frame_indices: list[int], gap_threshold: int = 1) -> None:
        """Compute contiguous runs from *frame_indices* and repaint.

        *gap_threshold*: max gap between consecutive values still counted as
        one contiguous run -- 1 for raw per-camera frame indices (adjacent
        frames differ by exactly 1); wider when *frame_indices* are in
        global-time units instead, where adjacent video frames are spaced
        by roughly (scale / fps) units, not 1.
        """
        if not frame_indices:
            self._coverage_runs = []
            self.update()
            return
        import numpy as np
        arr = np.array(sorted(frame_indices), dtype=np.int64)
        breaks = np.where(np.diff(arr) > gap_threshold)[0]
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

        # Queued-job ranges — gold band in lower cov_h px, drawn *before*
        # mask coverage so already-completed frames (teal) paint over the
        # gold as a job actually runs, reading as a fill-in-progress
        # effect rather than the two bands needing separate space.
        queued_color = QColor(200, 160, 40)
        for run_first, run_last in self._queued_ranges:
            x1 = self._to_x(run_first)
            x2 = self._to_x(run_last) + 1
            p.fillRect(x1, h - cov_h, max(1, x2 - x1), cov_h, queued_color)

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

        # Trial range — thin amber band at the very top, drawn over the
        # selection so it stays visible regardless of overlap. Purely
        # informational (see class docstring); independent of _sel_start/
        # _sel_end.
        if self._trial_start is not None and self._trial_end is not None:
            trial_h = 3
            tx1 = self._to_x(self._trial_start)
            tx2 = self._to_x(self._trial_end)
            if tx2 > tx1:
                p.fillRect(tx1, 0, tx2 - tx1, trial_h, QColor(230, 160, 40))

        # Split points — magenta, full height; brighter red where the
        # current selection spans one (a planned boundary the queued job
        # would silently cross), distinct from both the trial band's
        # amber and the mark ticks' blue so it never reads as either.
        for sp in self._split_points:
            crossed = self._sel_start < sp < self._sel_end
            color = QColor(255, 60, 60) if crossed else QColor(200, 60, 200)
            xs = self._to_x(sp)
            p.fillRect(max(0, xs - 1), 0, 2, h, color)

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

    #: Scrubber/RangeBar units per second of global time, when a sync
    #: table is available (see _local_frame_for/_global_units_for_local).
    #: 100 = centisecond precision -- comfortably finer than any real
    #: frame spacing (30-120fps is ~3.3-0.8 units/frame) without needing
    #: float slider values, which QSlider doesn't support.
    _TIME_SCALE = 100

    def __init__(
        self,
        conn: sqlite3.Connection,
        shot_id: str,
        parent: QWidget | None = None,
        trial_id: str | None = None,
        seg_init_run_id: str | None = None,
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
        not required for anything to function.

        *seg_init_run_id*, when given, is an existing ``seg_quality_runs``
        row (from the session tree's "Open/Continue" action) to extend
        instead of creating a new one on first edit -- closes the gap
        where every panel reopen silently fragmented one capture's
        segmentation work across several rows (segmentation-ui-
        improvements design doc, Issue 2). Omit to start a fresh
        segmentation, same as before this parameter existed."""
        super().__init__(parent)
        self._conn = conn
        self._shot_id = shot_id
        self._trial_id = trial_id
        self._frame_cache = FrameCache(max_frames=300, max_dim=1920)

        # Populated by _load_run()
        self._cameras: list[dict] = []
        self._seg_run_id: str | None = None
        self._persons: list[str] = []       # person names, index+1 = SAM label
        # Global-time scrubbing (see _local_frame_for docstring): None
        # means no sync config for this capture, falling back to the
        # legacy per-camera-frame domain where each camera has its own
        # scrubber range and switching cameras resets marks.
        self._sync_config_id: str | None = None
        self._sync_table = None
        # (start_s, end_s) of self._trial_id, if it has a time range set --
        # purely informational (RangeBar's amber band) plus the default
        # Mark Start/End on load; never a constraint on the actual
        # selection (see RangeBar's docstring).
        self._trial_range_s: tuple[float, float] | None = None

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

        # Split points for the currently-selected camera (scrubber units;
        # see _load_split_points) -- camera-specific (a hard-transition
        # moment, e.g. two people crossing, is usually camera-angle-
        # dependent), but still needs a sync table for a shared scrubber
        # coordinate space to place it in.
        self._split_points: list[int] = []

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

        # seg_quality_run new masks are written to -- either a fresh run
        # created lazily by _ensure_seg_run() on first edit, or (when the
        # caller passed seg_init_run_id) an existing run being continued.
        self._seg_init_run_id: str | None = seg_init_run_id
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

        self._track_range_btn = QPushButton("⏩ Segment Range from Seed")
        self._track_range_btn.setToolTip(
            "Seed at the current frame, propagate to both Mark Start and Mark "
            "End (queues one or two jobs as needed) -- the recommended way to "
            "cover a range from a middle seed frame"
        )
        self._track_range_btn.clicked.connect(self._on_track_range)
        self._track_range_btn.setEnabled(False)
        track_layout.addWidget(self._track_range_btn)

        track_layout.addSpacing(8)

        self._track_bwd_btn = QPushButton("◀ Backward only")
        self._track_bwd_btn.setToolTip(
            "Add a backward tracking job to the queue (current frame → mark "
            "start) without also queuing forward -- for resuming just one "
            "direction, e.g. after a failed or cancelled job"
        )
        self._track_bwd_btn.clicked.connect(self._on_track_backward)
        self._track_bwd_btn.setEnabled(False)
        track_layout.addWidget(self._track_bwd_btn)

        self._track_fwd_btn = QPushButton("▶ Forward only")
        self._track_fwd_btn.setToolTip(
            "Add a forward tracking job to the queue (current frame → mark "
            "end) without also queuing backward -- for resuming just one "
            "direction, e.g. after a failed or cancelled job"
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

        self._finalise_btn = QPushButton("✓ Finalise")
        self._finalise_btn.setToolTip(
            "Build pose observation sequences directly from the segmentation's own "
            "person labels -- no manual track-to-person stitching needed, since a "
            "segmentation mask's labels are already stable per-person identities. "
            "Open the Stitcher instead if a track needs correcting first."
        )
        self._finalise_btn.clicked.connect(self._on_finalise)
        self._finalise_btn.setEnabled(False)
        pose_row.addWidget(self._finalise_btn)

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

        # Set once in _load_run() if self._trial_id has a time range --
        # the amber band on the RangeBar is easy to miss on its own, so
        # this spells it out in text too.
        self._trial_range_label = QLabel("")
        self._trial_range_label.setStyleSheet("font-size: 10px; color: #c98a28;")
        vbox.addWidget(self._trial_range_label)

        mark_row = QHBoxLayout()
        mark_row.setSpacing(4)

        self._mark_start_btn = QPushButton("Mark Start")
        self._mark_start_btn.setMaximumWidth(90)
        self._mark_start_btn.setToolTip(
            "Set the segmentation range's start to the current position. "
            "Defaults to the trial's own start (if opened from a trial) but "
            "is freely adjustable -- e.g. to redo only part of a trial, or "
            "to cover a wider range than one trial on purpose."
        )
        self._mark_start_btn.clicked.connect(self._on_mark_start)
        self._mark_start_label = QLabel("Start: —")

        self._mark_end_btn = QPushButton("Mark End")
        self._mark_end_btn.setMaximumWidth(90)
        self._mark_end_btn.setToolTip(
            "Set the segmentation range's end to the current position. "
            "Defaults to the trial's own end (if opened from a trial) but "
            "is freely adjustable, same as Mark Start."
        )
        self._mark_end_btn.clicked.connect(self._on_mark_end)
        self._mark_end_label = QLabel("End: —")

        mark_row.addWidget(self._mark_start_btn)
        mark_row.addWidget(self._mark_start_label)
        mark_row.addStretch()
        mark_row.addWidget(self._mark_end_label)
        mark_row.addWidget(self._mark_end_btn)
        vbox.addLayout(mark_row)

        # Split points -- planning hard-transition boundaries (e.g. two
        # people crossing paths, usually camera-angle-dependent -- a
        # crossing that occludes in one camera's view may be clearly
        # separated in another's) so segmentation gets seeded
        # independently on each side instead of one continuous pass
        # diverging there. Magenta ticks on the RangeBar above,
        # per-camera (see _load_split_points); only meaningful with a
        # sync table, in which case the buttons just no-op with a status
        # message rather than being disabled outright.
        split_row = QHBoxLayout()
        split_row.setSpacing(4)

        self._mark_split_btn = QPushButton("✂ Mark Split")
        self._mark_split_btn.setToolTip(
            "Mark the current position, for the current camera only, as a "
            "split point -- a place where segmentation should be seeded "
            "independently on each side rather than propagated through in "
            "one pass"
        )
        self._mark_split_btn.clicked.connect(self._on_mark_split_point)
        split_row.addWidget(self._mark_split_btn)

        self._remove_split_btn = QPushButton("Remove Nearest")
        self._remove_split_btn.setToolTip(
            "Remove the split point nearest the current position"
        )
        self._remove_split_btn.clicked.connect(self._on_remove_nearest_split_point)
        split_row.addWidget(self._remove_split_btn)

        self._snap_marks_btn = QPushButton("⇤ Snap Marks to Segment")
        self._snap_marks_btn.setToolTip(
            "Set Mark Start/Mark End to the split points enclosing the "
            "current position -- pre-fills the range from the plan, still "
            "freely adjustable afterwards"
        )
        self._snap_marks_btn.clicked.connect(self._on_snap_marks_to_segment)
        split_row.addWidget(self._snap_marks_btn)

        split_row.addStretch()
        vbox.addLayout(split_row)

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

        # self._sync_config_id: set whenever any sync_configs row exists
        # for the capture (most recent by rowid) -- this alone is enough
        # to create a detection run (the FK it needs), independent of
        # whether sync has actually been solved yet.
        # self._sync_table: additionally set only if that config has real
        # sync_points -- gates global-time scrubbing specifically. A sync
        # config can exist with nothing solved yet, in which case this
        # stays None and everywhere below falls back to the legacy
        # per-camera-frame domain.
        self._sync_config_id = None
        self._sync_table = None
        sync_row = self._conn.execute(
            "SELECT id FROM sync_configs WHERE shot_id = ? ORDER BY rowid DESC LIMIT 1",
            (shot_id,),
        ).fetchone()
        if sync_row is not None:
            self._sync_config_id = sync_row["id"]
            self._sync_table = self._build_sync_table(sync_row["id"])

        self._trial_range_s = None
        if self._trial_id is not None:
            trial_row = self._conn.execute(
                "SELECT time_start_s, time_end_s FROM trials WHERE id = ?",
                (self._trial_id,),
            ).fetchone()
            if (
                trial_row is not None
                and trial_row["time_start_s"] is not None
                and trial_row["time_end_s"] is not None
            ):
                self._trial_range_s = (
                    float(trial_row["time_start_s"]), float(trial_row["time_end_s"]),
                )

        self._setup_scrubber_range()
        self._rebuild_camera_combo()  # triggers _on_camera_changed(0), loading split points
        self._rebuild_person_selector()
        self._init_controller()

    def _build_sync_table(self, sync_config_id: str):
        """Return a SyncTable for *sync_config_id*, or None if it has no
        sync_points (a sync_configs row can exist with nothing solved yet)."""
        from app.setup.db_context import SyncPoint, SyncTable

        rows = self._conn.execute(
            "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, sv.actual_fps "
            "FROM sync_points sp JOIN capture_videos sv ON sv.id = sp.shot_video_id "
            "WHERE sp.sync_config_id = ?",
            (sync_config_id,),
        ).fetchall()
        if not rows:
            return None
        points = [
            SyncPoint(camera_instance_id="", shot_video_id=r["shot_video_id"],
                      video_frame=r["video_frame"], timestamp_s=r["timestamp_s"])
            for r in rows
        ]
        fps_by_video = {r["shot_video_id"]: float(r["actual_fps"]) for r in rows}
        return SyncTable(points, fps_by_video)

    def _setup_scrubber_range(self) -> None:
        """Set the scrubber's min/max/value and the RangeBar's range/trial
        band/selection once, up front -- not per camera-switch, unlike the
        legacy per-camera-frame domain this falls back to when no sync
        table is available (handled instead in _on_camera_changed, the one
        place that still needs a per-camera reset).
        """
        if self._sync_table is None or not self._cameras:
            return  # legacy per-camera domain; _on_camera_changed sets it up

        t_min: float | None = None
        t_max: float | None = None
        for cam in self._cameras:
            t0 = self._sync_table.frame_to_global_time(cam["track_first"], cam["id"])
            t1 = self._sync_table.frame_to_global_time(cam["track_last"], cam["id"])
            if t0 is None or t1 is None:
                continue
            t_min = t0 if t_min is None else min(t_min, t0)
            t_max = t1 if t_max is None else max(t_max, t1)
        if t_min is None:
            # No camera actually has sync data despite the table existing --
            # fall back to the legacy per-camera domain.
            self._sync_table = None
            return

        g_min, g_max = self._to_units(t_min), self._to_units(t_max)
        self._scrubber.blockSignals(True)
        self._scrubber.setMinimum(g_min)
        self._scrubber.setMaximum(g_max)
        self._scrubber.blockSignals(False)
        self._range_bar.set_range(g_min, g_max)

        if self._trial_range_s is not None:
            t0_units = max(g_min, self._to_units(self._trial_range_s[0]))
            t1_units = min(g_max, self._to_units(self._trial_range_s[1]))
            self._range_bar.set_trial_range(t0_units, t1_units)
            self._mark_start, self._mark_end = t0_units, t1_units
            s0, s1 = self._trial_range_s
            self._trial_range_label.setText(
                f"Trial range: {self._fmt_mmss(s0)}–{self._fmt_mmss(s1)}"
            )
        else:
            self._mark_start, self._mark_end = g_min, g_max
            self._trial_range_label.setText("")

        self._scrubber.blockSignals(True)
        self._scrubber.setValue(self._mark_start)
        self._scrubber.blockSignals(False)
        self._range_bar.set_selection(self._mark_start, self._mark_end)
        self._range_bar.set_position(self._mark_start)

    # ------------------------------------------------------------------
    # Global-time <-> per-camera-local-frame conversion
    # ------------------------------------------------------------------

    def _to_units(self, seconds: float) -> int:
        return round(seconds * self._TIME_SCALE)

    def _to_seconds(self, units: int) -> float:
        return units / self._TIME_SCALE

    @staticmethod
    def _fmt_mmss(seconds: float) -> str:
        mm, ss = divmod(max(0.0, seconds), 60)
        return f"{int(mm):02d}:{ss:05.2f}"

    def _local_frame_for(self, cam: dict, global_units: int | None = None) -> int:
        """Convert a scrubber position to *cam*'s own local video frame index.

        Without a sync table (legacy fallback), the scrubber's domain IS
        already the current camera's local frame index, so this is the
        identity function. With one, *global_units* (default: the
        scrubber's current value) is global-time scrubber units, converted
        via SyncTable.lookup(); if *cam* has no footage at that instant
        (outside its own range), clamps to whichever edge is closer rather
        than raising.
        """
        if global_units is None:
            global_units = self._scrubber.value()
        if self._sync_table is None:
            return global_units
        local = self._sync_table.lookup(self._to_seconds(global_units), cam["id"])
        if local is not None:
            return local
        t0 = self._sync_table.frame_to_global_time(cam["first"], cam["id"])
        t1 = self._sync_table.frame_to_global_time(cam["last"], cam["id"])
        t = self._to_seconds(global_units)
        if t1 is not None and t > t1:
            return cam["last"]
        return cam["first"]

    def _global_units_for_local(self, cam: dict, local_frame: int) -> int:
        """Inverse of _local_frame_for: cam's own local frame -> scrubber
        units. Falls back to the current scrubber position (best-effort,
        no-op) if the sync table has no data for this specific frame."""
        if self._sync_table is None:
            return local_frame
        t = self._sync_table.frame_to_global_time(local_frame, cam["id"])
        if t is None:
            return self._scrubber.value()
        return self._to_units(t)

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
                self._sam_status_label.setText("SAM2 not available — install sam2")
                self._sam_status_label.setStyleSheet("font-size: 10px; color: #c60;")
        except Exception as e:
            self._sam_status_label.setText(f"SAM2 error: {e}")

        # Enable track / pose buttons if there are persons to track
        can_track = bool(self._persons)
        self._track_range_btn.setEnabled(can_track)
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
        if cam is None:
            return
        frame_idx = self._local_frame_for(cam)

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
        cam = self._cam_combo.currentData()
        if cam is not None:
            self._show_frame(self._local_frame_for(cam))

    def _on_clear_all(self) -> None:
        """Remove all live SAM2 clicks for the current frame.

        Does NOT touch stored DB masks.
        """
        if self._controller is None:
            return
        self._controller.clear_all()
        self._set_status("Cleared all live clicks")
        cam = self._cam_combo.currentData()
        if cam is not None:
            self._show_frame(self._local_frame_for(cam))

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

        if self._sync_table is None:
            # No sync data for this capture -- legacy per-camera-frame
            # domain: each camera has its own scrubber range, and
            # switching resets it (there's no meaningful shared "same
            # instant" without sync data to convert through).
            self._scrubber.blockSignals(True)
            self._scrubber.setMinimum(cam["track_first"])
            self._scrubber.setMaximum(cam["track_last"])
            self._scrubber.setValue(cam["track_first"])
            self._scrubber.blockSignals(False)
            self._mark_start = cam["track_first"]
            self._mark_end   = cam["track_last"]
            self._range_bar.set_range(cam["track_first"], cam["track_last"])
            self._update_mark_labels(cam)
            self._refresh_coverage_bar(cam)
            self._refresh_queued_bar(cam)
            self._load_split_points()
            self._show_frame(cam["track_first"])
            return

        # Global-time domain: scrubber range/position and marks are shared
        # across cameras (set once in _setup_scrubber_range) -- switching
        # only changes which camera's local frame the current global
        # position maps to, not the position/marks themselves.
        self._update_mark_labels(cam)
        self._refresh_coverage_bar(cam)
        self._refresh_queued_bar(cam)
        self._load_split_points()  # split points are camera-specific
        self._show_frame(self._local_frame_for(cam))

    def _on_frame_changed(self, frame_idx: int) -> None:
        # Frame changed: clear click state, show new frame.
        if self._controller:
            self._controller.clear_all()
        self._encoded_frame_idx = -1
        self._range_bar.set_position(frame_idx)
        cam = self._cam_combo.currentData()
        if cam is not None:
            self._show_frame(self._local_frame_for(cam, frame_idx))
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
        local_frames = [r["frame_idx"] for r in rows]
        if self._sync_table is None:
            self._range_bar.set_covered_frames(local_frames)
            return
        # Local frame indices are ~1 apart; converted to global-time units
        # they're spaced by roughly (scale / fps) instead -- a generous
        # fixed 0.2s gap threshold comfortably covers any realistic fps
        # without needing this camera's own fps for the math.
        global_frames = []
        for lf in local_frames:
            t = self._sync_table.frame_to_global_time(lf, cam["id"])
            if t is not None:
                global_frames.append(self._to_units(t))
        self._range_bar.set_covered_frames(global_frames, gap_threshold=self._TIME_SCALE // 5)

    def _refresh_queued_bar(self, cam: dict) -> None:
        """Show this camera's pending/running Cutie tracking jobs on the
        range bar, distinct from already-written masks (teal) -- lets a
        queued-but-not-yet-run segment be seen at a glance instead of
        only showing up once the job actually starts producing masks.
        Only TrackingJob (Cutie mask propagation), not PoseExtractionJob
        (pose estimation, a separate later stage) -- this band pairs with
        the mask-coverage band above it, not pose extraction's own
        progress.
        """
        from app.pose.job_queue_runner import TrackingJob as _TrackingJob
        ranges: list[tuple[int, int]] = []
        for job in self._runner.jobs:
            if not isinstance(job, _TrackingJob):
                continue
            if job.shot_video_id != cam["id"] or job.status not in ("pending", "running"):
                continue
            if self._sync_table is None:
                ranges.append((job.first_frame, job.last_frame))
                continue
            t0 = self._sync_table.frame_to_global_time(job.first_frame, cam["id"])
            t1 = self._sync_table.frame_to_global_time(job.last_frame, cam["id"])
            if t0 is not None and t1 is not None:
                ranges.append((self._to_units(t0), self._to_units(t1)))
        self._range_bar.set_queued_ranges(ranges)

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

    # ------------------------------------------------------------------
    # Split points (segmentation-ui-improvements design doc, Issue 4)
    # ------------------------------------------------------------------

    def _load_split_points(self) -> None:
        """Load this capture's split points and push them to the RangeBar.

        Only meaningful with a sync table -- a split point is one
        specific camera's own hard-transition moment (2026-08-29: originally
        modeled as capture-wide/shared-across-cameras, corrected once real
        use showed the whole point of a split is usually camera-angle-
        dependent -- e.g. two people occlude each other from one camera's
        viewpoint at a moment they're clearly separated in another's
        parallax), so still needs global time to place on the shared
        scrubber, but is loaded per the *currently displayed* camera, not
        the capture as a whole.
        """
        self._split_points = []
        self._split_point_ids = []
        cam = self._cam_combo.currentData()
        if self._sync_table is None or cam is None:
            self._range_bar.set_split_points([])
            return
        rows = self._conn.execute(
            "SELECT id, time_s FROM capture_segmentation_hints "
            "WHERE shot_video_id = ? ORDER BY time_s",
            (cam["id"],),
        ).fetchall()
        self._split_points = [self._to_units(r["time_s"]) for r in rows]
        self._split_point_ids = [r["id"] for r in rows]
        self._range_bar.set_split_points(self._split_points)

    def _on_mark_split_point(self) -> None:
        cam = self._cam_combo.currentData()
        if self._sync_table is None or cam is None:
            self._set_status(
                "Split points need solved sync (one global time shared across "
                "cameras) -- not available for this capture yet."
            )
            return
        time_s = self._to_seconds(self._scrubber.value())
        self._conn.execute(
            "INSERT INTO capture_segmentation_hints "
            "(id, capture_id, shot_video_id, time_s, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                generate_id(), self._shot_id, cam["id"], time_s,
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        self._load_split_points()
        self._set_status(f"Split point marked at {self._fmt_mmss(time_s)} for {cam['label']}.")

    def _on_remove_nearest_split_point(self) -> None:
        if not self._split_points:
            self._set_status("No split points to remove.")
            return
        pos = self._scrubber.value()
        idx = min(range(len(self._split_points)), key=lambda i: abs(self._split_points[i] - pos))
        sp_id = self._split_point_ids[idx]
        time_s = self._to_seconds(self._split_points[idx])
        self._conn.execute("DELETE FROM capture_segmentation_hints WHERE id = ?", (sp_id,))
        self._conn.commit()
        self._load_split_points()
        self._set_status(f"Removed split point at {self._fmt_mmss(time_s)}.")

    def _on_snap_marks_to_segment(self) -> None:
        """Set Mark Start/Mark End to the split points enclosing the
        current scrubber position, falling back to the trial's own range
        on whichever side has no split point (matching what marks default
        to on load) rather than the full capture range -- "pre-fill the
        marks from the plan, still freely overridable" from the design
        doc."""
        cam = self._cam_combo.currentData()
        if cam is None:
            return
        if self._trial_range_s is not None:
            default_start = max(self._scrubber.minimum(), self._to_units(self._trial_range_s[0]))
            default_end = min(self._scrubber.maximum(), self._to_units(self._trial_range_s[1]))
        else:
            default_start = self._scrubber.minimum()
            default_end = self._scrubber.maximum()
        pos = self._scrubber.value()
        before = [sp for sp in self._split_points if sp <= pos]
        after = [sp for sp in self._split_points if sp > pos]
        self._mark_start = max(before) if before else default_start
        self._mark_end = min(after) if after else default_end
        self._range_bar.set_selection(self._mark_start, self._mark_end)
        self._update_mark_labels(cam)

    def _confirm_crossing_split_points(self, crossed: list[int]) -> bool:
        times = ", ".join(self._fmt_mmss(self._to_seconds(sp)) for sp in crossed)
        return QMessageBox.question(
            self, "Crosses a planned split point",
            f"This job crosses {len(crossed)} planned split point(s) at {times} -- "
            "propagation may diverge there. Queue anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes

    def _update_mark_labels(self, cam: dict) -> None:
        if self._sync_table is not None:
            self._mark_start_label.setText(
                f"Start: {self._fmt_mmss(self._to_seconds(self._mark_start))}"
            )
            self._mark_end_label.setText(
                f"End: {self._fmt_mmss(self._to_seconds(self._mark_end))}"
            )
            return
        fps = cam.get("fps", 1) or 1
        first = cam["track_first"]
        ts = (self._mark_start - first) / fps
        te = (self._mark_end   - first) / fps
        self._mark_start_label.setText(f"Start: {self._mark_start} ({self._fmt_mmss(ts)})")
        self._mark_end_label.setText(f"End: {self._mark_end} ({self._fmt_mmss(te)})")

    def _schedule_encode(self) -> None:
        """Start/restart the debounce timer to encode the current frame."""
        self._encode_timer.start(300)

    def _encode_current_frame(self) -> None:
        """Called by debounce timer: encode current frame if accessible."""
        cam = self._cam_combo.currentData()
        if cam is not None:
            self._ensure_encoded(cam, self._local_frame_for(cam))

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

    def _on_track_range(self) -> None:
        """Seed at the current frame, queue whichever of backward/forward
        actually covers new ground toward Mark Start/Mark End.

        Replaces having to remember to press both "Forward" and "Backward"
        separately for a middle seed frame (segmentation-ui-improvements
        design doc, Issue 3) -- same two _queue_tracking() calls the
        existing per-direction buttons already make, just wrapped behind
        one action with the degenerate (seed already at an edge) case
        skipped rather than queuing a zero-length job.
        """
        cam = self._cam_combo.currentData()
        if cam is None:
            return
        seed = self._local_frame_for(cam)
        mark_start_frame = self._local_frame_for(cam, self._mark_start)
        mark_end_frame = self._local_frame_for(cam, self._mark_end)

        queued_any = False
        if seed > mark_start_frame:
            self._queue_tracking("backward")
            queued_any = True
        if seed < mark_end_frame:
            self._queue_tracking("forward")
            queued_any = True

        if not queued_any:
            self._set_status(
                "Seed frame is the only frame in the marked range — nothing to propagate."
            )

    def _queue_tracking(self, direction: str) -> None:
        """Create a TrackingJob from the current UI state and enqueue it."""
        cam = self._cam_combo.currentData()
        if cam is None or not self._persons:
            return

        local_frame = self._local_frame_for(cam)

        # first_frame/last_frame bound the job's own propagation range for
        # its direction -- CutieWorker._run_forward only ever reads
        # last_frame (init_frame -> last_frame) and _run_backward only
        # ever reads first_frame (first_frame -> init_frame), so the bound
        # the *other* direction would have used is irrelevant to actual
        # propagation. Setting it to the current frame here (rather than
        # always passing the full mark_start/mark_end range regardless of
        # direction) makes TrackingJob.summary's "first-last" display
        # correctly show which range *this* job actually covers instead of
        # showing the same range for both a forward and a backward job
        # queued from the same position.
        if direction == "forward":
            first_frame, last_frame = local_frame, self._local_frame_for(cam, self._mark_end)
            range_start, range_end = self._scrubber.value(), self._mark_end
        else:
            first_frame, last_frame = self._local_frame_for(cam, self._mark_start), local_frame
            range_start, range_end = self._mark_start, self._scrubber.value()

        # Warn-but-allow on crossing a planned split point (segmentation-
        # ui-improvements design doc, Issue 4) -- not a hard block, since a
        # planned split can turn out to be unnecessary on reflection, but
        # never silent either.
        crossed = [sp for sp in self._split_points if range_start < sp < range_end]
        if crossed and not self._confirm_crossing_split_points(crossed):
            self._set_status("Cancelled — job would have crossed a planned split point.")
            return

        # Seed mask: live SAM2 result or stored DB mask.
        seed_mask = None
        if self._controller and np.any(self._controller.get_mask()):
            seed_mask = self._controller.get_mask().copy()
        else:
            seed_mask = self._load_stored_mask(cam["id"], local_frame)

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
            init_frame=local_frame,
            init_mask_png=buf.tobytes(),
            persons_ordered=list(self._persons),
            first_frame=first_frame,
            last_frame=last_frame,
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

            # Pose extraction covers the marked range now (converted to
            # this camera's own local frames), same range segmentation
            # itself was queued for -- not the camera's full track range
            # unconditionally, so narrowing/widening the marks (Mark
            # Start/Mark End) actually controls what gets extracted.
            job = PoseExtractionJob(
                job_id=str(uuid.uuid4())[:8],
                camera_label=cam["label"],
                shot_video_id=cam["id"],
                video_path=cam["file_path"],
                detection_run_id=detection_run_id,
                seg_quality_run_id=seg_run_id,
                persons_ordered=list(self._persons),
                first_frame=self._local_frame_for(cam, self._mark_start),
                last_frame=self._local_frame_for(cam, self._mark_end),
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

    def _on_finalise(self) -> None:
        """Build pose observation sequences directly from the segmentation's
        own person labels, no manual stitcher pass required -- see
        docs/roadmap/features/segmentation-reuse/segmentation-reuse-design.md,
        "Auto-assignment". Uses self._pose_detection_run_id, set to whichever
        detection run the most recently finished pose job wrote into
        (_on_job_finished); the queue mixes runs across pose models cleanly
        in practice since each _resolve_or_create_detection_run call picks
        one run per pose_model, so this is "the run for the pose model most
        recently queued/completed," same run the live overlay already shows.
        """
        if self._pose_detection_run_id is None:
            self._set_status("No pose extraction yet — queue pose extraction first.")
            return
        run_row = self._conn.execute(
            "SELECT shot_id, sync_config_id, pose_model FROM detection_runs WHERE id=?",
            (self._pose_detection_run_id,),
        ).fetchone()
        if run_row is None:
            self._set_status("Detection run not found.")
            return

        from app.pose.finalise import auto_assign_and_finalise, conf_scale_for_model
        from posetrak.db.manage_person import persons_ordered_for_seg_run

        # Read the ordinal->name mapping from whichever seg run's masks
        # were actually used (persisted at mask-creation time), not the
        # live self._persons -- keeps this consistent with gap 2's
        # RunDetectionDialog path, which has no in-memory self._persons to
        # fall back on at all. Falls back to self._persons only if no seg
        # run exists yet here (shouldn't happen -- pose extraction needs
        # masks -- but avoids a hard crash over a defensive fallback).
        run_ids = self._read_run_ids()
        persons_ordered = (
            persons_ordered_for_seg_run(self._conn, run_ids[0])
            if run_ids else list(self._persons)
        )

        try:
            seq_ids = auto_assign_and_finalise(
                session=self._conn,
                detection_run_id=self._pose_detection_run_id,
                shot_id=run_row["shot_id"],
                sync_config_id=run_row["sync_config_id"],
                persons_ordered=persons_ordered,
                pose_model=run_row["pose_model"],
                confidence_scale=conf_scale_for_model(run_row["pose_model"]),
            )
        except Exception as exc:  # noqa: BLE001
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Finalise Error", str(exc))
            return
        self._set_status(f"Finalised {len(seq_ids)} person sequence(s).")

    def _resolve_or_create_detection_run(self, pose_model: str) -> str | None:
        """Return a detection_run_id to write pose results into, or None if cancelled.

        Creates a new run silently if none exists for this capture yet;
        asks the user what to do if a run with the same pose_model already
        exists. No longer resolves shot_id/sync_config_id from a pre-
        existing "parent" detection run -- the panel is capture-scoped
        (self._shot_id) now, so this needs a sync config to exist before
        it can create a fresh detection run (reuses self._sync_config_id,
        resolved once in _load_run rather than re-queried here).
        """
        from app.pose.db_cache import create_detection_run
        from PySide6.QtWidgets import QMessageBox

        shot_id = self._shot_id
        if self._sync_config_id is None:
            self._set_status("No sync config for this capture yet — set one up first.")
            return None
        sync_cfg_id = self._sync_config_id
        trial_id = self._trial_id
        # time_start_s/time_end_s: provenance metadata on detection_runs,
        # not the actual per-camera frame gating (that's
        # PoseExtractionJob.first_frame/last_frame, resolved per camera
        # from the marked range in _queue_pose_jobs) -- but now that the
        # marks are real global-time values when a sync table exists, this
        # can record the actual requested range instead of always 0.0/0.0.
        if self._sync_table is not None:
            time_start_s = self._to_seconds(self._mark_start)
            time_end_s = self._to_seconds(self._mark_end)
        else:
            time_start_s, time_end_s = 0.0, 0.0

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
            return create_detection_run(
                self._conn, shot_id, sync_cfg_id,
                time_start_s, time_end_s,
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
            time_start_s, time_end_s,
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

        # Advance scrubber to the last frame in the batch (local frame index
        # from the worker -> scrubber's own global-time units, or the
        # identity conversion in the legacy per-camera-frame fallback).
        cam = self._cam_combo.currentData()
        if cam and svid == cam["id"] and batch:
            last_fi = batch[-1][0]
            pos = self._global_units_for_local(cam, last_fi)
            self._scrubber.blockSignals(True)
            self._scrubber.setValue(pos)
            self._scrubber.blockSignals(False)
            self._range_bar.set_position(pos)

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
            self._finalise_btn.setEnabled(True)
            self._refresh_queue_list()
            cam = self._cam_combo.currentData()
            if cam:
                self._show_frame(self._local_frame_for(cam))
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
                self._show_frame(self._local_frame_for(cam))
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

        # Job membership/status changed -- refresh the range bar's queued
        # band too. Covers every call site (enqueue, start, finish, fail,
        # cancel, remove, run queue) since they all already call this.
        cam = self._cam_combo.currentData()
        if cam is not None:
            self._refresh_queued_bar(cam)

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _ensure_seg_run(self) -> None:
        """Create a seg_quality_run for this capture if not already done.

        time_start_s/time_end_s: the current Mark Start/End range (global
        time), converted to seconds -- reflects what the user actually
        selected to segment (defaults to the trial's own bounds, but is
        freely adjustable; see RangeBar's docstring). Falls back to
        0.0/a large sentinel ("covers the whole capture," trivially
        satisfying any trial's containment check) when there's no sync
        table for this capture, since marks aren't real global-time values
        in that legacy per-camera-frame fallback.

        persons_json snapshots self._persons -- the ordinal->name mapping
        (index i = mask label i+1) actually in effect right now, the same
        list _queue_pose_jobs threads through as persons_ordered. Lets a
        *different* caller reuse this segmentation later without having to
        assume today's capture_persons order still matches (gap 2,
        RunDetectionDialog's "use existing segmentation" bbox source). See
        docs/roadmap/features/segmentation-reuse/segmentation-reuse-design.md.
        """
        if self._seg_init_run_id is not None:
            return
        run_id = generate_id()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if self._sync_table is not None:
            time_start_s = self._to_seconds(self._mark_start)
            time_end_s = self._to_seconds(self._mark_end)
        else:
            time_start_s, time_end_s = 0.0, 1e9
        self._conn.execute(
            "INSERT INTO seg_quality_runs "
            "(id, shot_id, trial_id, time_start_s, time_end_s, created_at, "
            " quality_source, erosion_px, persons_json) "
            "VALUES (?, ?, ?, ?, ?, ?, 'cutie-interactive', 5, ?)",
            (run_id, self._shot_id, self._trial_id, time_start_s, time_end_s,
             now, json.dumps(list(self._persons))),
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
