"""main.py — PoseExtractionWindow: main GUI for the pose extraction pipeline."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.pose.assignment import find_assignment_conflicts
from app.pose.person_preview import PersonPreviewWidget
from app.pose.db_cache import list_detection_runs
from app.pose.finalise import TrackAssignment, finalise_to_db
from app.pose.frame_view import CameraInfo, FrameViewWidget
from app.pose.stitcher import StitcherWidget
from app.setup.db_context import SyncPoint, SyncTable
from app.setup.job_runner import BackgroundJob


def _load_sync_table(session: sqlite3.Connection, sync_config_id: str) -> SyncTable:
    """Build a SyncTable from all sync_points for a given sync_config_id."""
    rows = session.execute(
        "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, sv.actual_fps "
        "FROM sync_points sp "
        "JOIN capture_videos sv ON sv.id = sp.shot_video_id "
        "WHERE sp.sync_config_id = ?",
        (sync_config_id,),
    ).fetchall()
    sync_points = [
        SyncPoint(
            camera_instance_id=r["shot_video_id"],
            shot_video_id=r["shot_video_id"],
            video_frame=r["video_frame"],
            timestamp_s=r["timestamp_s"],
        )
        for r in rows
    ]
    fps_by_video: dict[str, float] = {}
    for r in rows:
        fps_by_video.setdefault(r["shot_video_id"], float(r["actual_fps"] or 30.0))
    return SyncTable(sync_points, fps_by_video)


class _ComboBox(QComboBox):
    """QComboBox that reliably closes its popup on item selection.

    On some platforms (XWayland / WSL2) the popup item view does not receive
    the mouse-release event, so the default hidePopup() is never triggered.
    Connecting to ``activated`` and calling hidePopup() explicitly is the
    reliable cross-platform fix.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.activated.connect(lambda _: self.hidePopup())


# ---------------------------------------------------------------------------
# Background job
# ---------------------------------------------------------------------------


class DetectionJob(BackgroundJob):
    camera_progress = Signal(int, int)   # cameras_done, cameras_total

    def __init__(
        self,
        session_path: str,
        shot_id: str,
        sync_config_id: str,
        time_start_s: float,
        time_end_s: float,
        detector_name: str,
        pose_model_name: str,
        detector_conf: float,
    ):
        super().__init__()
        self._session_path = session_path
        self._shot_id = shot_id
        self._sync_config_id = sync_config_id
        self._time_start_s = time_start_s
        self._time_end_s = time_end_s
        self._detector_name = detector_name
        self._pose_model_name = pose_model_name
        self._detector_conf = detector_conf

    def run(self):
        from posetrak.db.db import open_session
        from app.pose.backends_yolo import YOLOv11Detector
        from app.pose.backends_rtmpose import RTMPoseEstimator
        from app.pose.detection_pipeline import DetectionPipeline

        session = open_session(Path(self._session_path))

        det = YOLOv11Detector(
            model_name=f"{self._detector_name}.pt",
            device="cuda",
            conf=self._detector_conf,
        )
        est = RTMPoseEstimator(model_name=self._pose_model_name, device="cuda")

        def on_progress(done: int, total: int, cam_id: str) -> None:
            pct = int(done / max(total, 1) * 100)
            self.progress.emit(pct, f"{cam_id}  {done}/{total} frames")

        def on_camera_done(done: int, total: int) -> None:
            self.camera_progress.emit(done, total)

        pipeline = DetectionPipeline(
            session=session,
            shot_id=self._shot_id,
            sync_config_id=self._sync_config_id,
            time_start_s=self._time_start_s,
            time_end_s=self._time_end_s,
            detector=det,
            estimator=est,
        )
        result = pipeline.run(on_progress=on_progress, on_camera_done=on_camera_done)
        self.finished.emit(result.detection_run_id)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class PoseExtractionWindow(QMainWindow):
    data_changed = Signal()  # emitted after successful finalization

    def __init__(self, session_db: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pose Extraction")
        self.resize(1400, 900)

        self._session: sqlite3.Connection | None = None
        self._session_path: str | None = None
        self._session_id: str | None = None
        self._shot_id: str | None = None
        self._sync_config_id: str | None = None
        self._current_run_id: str | None = None
        self._current_svid: str | None = None
        self._current_track_id: int | None = None

        # Detection time range — None until the user marks both endpoints
        self._time_start_s: float | None = None
        self._time_end_s: float | None = None

        # Segment assignments: (svid, tid, seg_first) → person_name.  A detection
        # track can be split into multiple segments; each segment is assigned
        # independently.  Not persisted until Finalise.
        self._assignments: dict[tuple[str, int, int], str] = {}

        # Currently selected segment
        self._current_seg_first: int | None = None

        # Most recent video frame index displayed (used to resolve which segment
        # applies to the current frame when syncing the frame view overlay)
        self._current_frame_idx: int = 0

        self._job: DetectionJob | None = None

        self._build_ui()

        if session_db:
            self._load_session(session_db)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ---- Top controls ----
        top = QGroupBox("Session")
        top_layout = QVBoxLayout(top)

        # Row 1: session path
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Session DB:"))
        self._session_path_edit = QLineEdit()
        self._session_path_edit.setReadOnly(True)
        row1.addWidget(self._session_path_edit, 1)
        self._open_btn = QPushButton("Open...")
        self._open_btn.clicked.connect(self._on_open_session)
        row1.addWidget(self._open_btn)
        self._session_actions_btn = QPushButton("Session…")
        self._session_actions_btn.setEnabled(False)
        self._session_actions_btn.clicked.connect(self._show_session_menu)
        row1.addWidget(self._session_actions_btn)
        top_layout.addLayout(row1)

        # Row 2: shot + sync
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Shot:"))
        self._shot_combo = _ComboBox()
        self._shot_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._shot_combo.currentIndexChanged.connect(self._on_shot_changed)
        row2.addWidget(self._shot_combo)
        row2.addWidget(QLabel("Sync:"))
        self._sync_combo = _ComboBox()
        self._sync_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._sync_combo.currentIndexChanged.connect(self._on_sync_changed)
        row2.addWidget(self._sync_combo)
        reload_sync_btn = QPushButton("⟳")
        reload_sync_btn.setFixedWidth(28)
        reload_sync_btn.setToolTip("Reload sync configs from DB (use after re-solving sync)")
        reload_sync_btn.clicked.connect(self._populate_syncs)
        row2.addWidget(reload_sync_btn)
        row2.addStretch()
        top_layout.addLayout(row2)

        # Row 3: time range (mark buttons) + detector + pose model + conf
        row3 = QHBoxLayout()
        self._mark_start_btn = QPushButton("Mark start")
        self._mark_start_btn.setToolTip("Set detection start to the currently displayed frame")
        self._mark_start_btn.clicked.connect(self._on_mark_start)
        row3.addWidget(self._mark_start_btn)
        self._start_label = QLabel("–")
        self._start_label.setStyleSheet("font-family: monospace; min-width: 70px;")
        row3.addWidget(self._start_label)
        self._mark_end_btn = QPushButton("Mark end")
        self._mark_end_btn.setToolTip("Set detection end to the currently displayed frame")
        self._mark_end_btn.clicked.connect(self._on_mark_end)
        row3.addWidget(self._mark_end_btn)
        self._end_label = QLabel("–")
        self._end_label.setStyleSheet("font-family: monospace; min-width: 70px;")
        row3.addWidget(self._end_label)
        row3.addSpacing(12)
        row3.addWidget(QLabel("Detector:"))
        self._detector_combo = _ComboBox()
        self._detector_combo.addItems(["yolo11x", "yolo11l", "yolo11m"])
        row3.addWidget(self._detector_combo)
        row3.addWidget(QLabel("Pose model:"))
        self._pose_combo = _ComboBox()
        self._pose_combo.addItems(["rtmpose-l-133kp", "rtmpose-l-17kp", "rtmpose-m-17kp"])
        row3.addWidget(self._pose_combo)
        row3.addWidget(QLabel("Conf:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.01, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.3)
        row3.addWidget(self._conf_spin)
        row3.addStretch()
        top_layout.addLayout(row3)

        # Row 4: detection run combo + run button
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Detection run:"))
        self._run_combo = _ComboBox()
        self._run_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._run_combo.setMinimumWidth(300)
        self._run_combo.currentIndexChanged.connect(self._on_run_selected)
        row4.addWidget(self._run_combo, 1)
        self._run_btn = QPushButton("Run Detection")
        self._run_btn.clicked.connect(self._on_run_detection)
        row4.addWidget(self._run_btn)
        top_layout.addLayout(row4)

        root.addWidget(top)

        # ---- Progress bars ----
        self._cam_progress_bar = QProgressBar()
        self._cam_progress_bar.setRange(0, 100)
        self._cam_progress_bar.setValue(0)
        self._cam_progress_label = QLabel("")
        cam_prog_row = QHBoxLayout()
        cam_prog_row.addWidget(QLabel("Frame:"))
        cam_prog_row.addWidget(self._cam_progress_bar, 1)
        cam_prog_row.addWidget(self._cam_progress_label)
        root.addLayout(cam_prog_row)

        self._total_progress_bar = QProgressBar()
        self._total_progress_bar.setRange(0, 100)
        self._total_progress_bar.setValue(0)
        self._total_progress_label = QLabel("")
        total_prog_row = QHBoxLayout()
        total_prog_row.addWidget(QLabel("Cameras:"))
        total_prog_row.addWidget(self._total_progress_bar, 1)
        total_prog_row.addWidget(self._total_progress_label)
        root.addLayout(total_prog_row)

        # ---- Main splitter: frame view | right panel ----
        splitter = QSplitter(Qt.Horizontal)

        self._frame_view = FrameViewWidget()
        self._frame_view.frame_changed.connect(self._on_frame_changed)
        splitter.addWidget(self._frame_view)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # ---- Stitcher row: timeline + person preview side by side ----
        stitcher_row = QSplitter(Qt.Horizontal)

        self._stitcher = StitcherWidget()
        self._stitcher.setMinimumHeight(150)
        self._stitcher.segment_clicked.connect(self._on_segment_clicked)
        self._stitcher.assignment_changed.connect(self._on_assignment_changed)
        self._stitcher.split_requested.connect(self._on_split_requested)
        self._stitcher.time_clicked.connect(self._frame_view.seek_global_time)
        stitcher_row.addWidget(self._stitcher)

        self._preview = PersonPreviewWidget()
        self._preview.setMinimumWidth(180)
        self._preview.setMaximumWidth(280)
        stitcher_row.addWidget(self._preview)

        stitcher_row.setStretchFactor(0, 1)   # stitcher takes all extra space
        stitcher_row.setStretchFactor(1, 0)   # preview stays at preferred width
        right_layout.addWidget(stitcher_row, 1)

        # Connect frame view → preview (raw frame + detections forwarded each seek)
        self._frame_view.frame_data_ready.connect(self._preview.update_frame)

        # ---- Assignment panel ----
        assign_group = QGroupBox("Selected track")
        assign_layout = QVBoxLayout(assign_group)

        self._selected_label = QLabel("None selected")
        assign_layout.addWidget(self._selected_label)

        person_row = QHBoxLayout()
        person_row.addWidget(QLabel("Person:"))
        self._person_combo = _ComboBox()
        self._person_combo.setEditable(True)
        self._person_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._person_combo.setMinimumWidth(120)
        person_row.addWidget(self._person_combo, 1)
        self._add_person_btn = QPushButton("+Add")
        self._add_person_btn.clicked.connect(self._on_add_person)
        person_row.addWidget(self._add_person_btn)
        assign_layout.addLayout(person_row)

        btn_row = QHBoxLayout()
        self._assign_btn = QPushButton("Assign")
        self._assign_btn.clicked.connect(self._on_assign)
        btn_row.addWidget(self._assign_btn)
        self._finalise_btn = QPushButton("Finalise")
        self._finalise_btn.clicked.connect(self._on_finalise)
        btn_row.addWidget(self._finalise_btn)
        assign_layout.addLayout(btn_row)

        right_layout.addWidget(assign_group)
        splitter.addWidget(right)
        splitter.setSizes([700, 700])

        root.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    # Session loading
    # ------------------------------------------------------------------

    def _on_open_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Session DB", "", "SQLite DB (*.db);;All files (*)"
        )
        if path:
            self._load_session(path)

    def _show_session_menu(self) -> None:
        menu = QMenu(self)
        act_extrinsics = menu.addAction("Import Extrinsics…")
        act_extrinsics.triggered.connect(self._open_extrinsics_dialog)
        act_skeletons = menu.addAction("Manage Skeletons…")
        act_skeletons.triggered.connect(self._open_skeleton_dialog)
        menu.addSeparator()
        act_run = menu.addAction("Run Tracker…")
        act_run.triggered.connect(self._open_run_tracker_dialog)
        menu.exec(self._session_actions_btn.mapToGlobal(
            self._session_actions_btn.rect().bottomLeft()
        ))

    def _open_extrinsics_dialog(self) -> None:
        if self._session is None or self._session_id is None:
            return
        from app.setup.page_extrinsics import ExtrinsicsImportDialog
        shot_ids = [self._shot_id] if self._shot_id else []
        dlg = ExtrinsicsImportDialog(
            self._session, self._session_id, shot_ids=shot_ids, parent=self
        )
        dlg.exec()

    def _open_skeleton_dialog(self) -> None:
        if self._session is None or self._session_id is None:
            return
        from app.setup.page_skeleton import SkeletonSetupDialog
        dlg = SkeletonSetupDialog(self._session, self._session_id, parent=self)
        dlg.exec()

    def _open_run_tracker_dialog(self) -> None:
        if self._session is None or self._session_path is None:
            return
        from app.pose.run_tracker import RunTrackerDialog
        dlg = RunTrackerDialog(self._session, self._session_path, parent=self)
        dlg.exec()

    def _load_session(self, path: str) -> None:
        from posetrak.db.db import open_session
        try:
            conn = open_session(Path(path))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open session:\n{e}")
            return

        self._session = conn
        self._session_path = path
        self._session_path_edit.setText(path)

        # Get session_id
        row = conn.execute("SELECT id FROM mocap_sessions LIMIT 1").fetchone()
        if row is None:
            QMessageBox.warning(self, "Warning", "No mocap session found in this DB.")
            return
        self._session_id = row["id"]
        self._session_actions_btn.setEnabled(True)

        self._populate_shots()

    def _populate_shots(self) -> None:
        self._shot_combo.blockSignals(True)
        self._shot_combo.clear()
        if self._session is None or self._session_id is None:
            self._shot_combo.blockSignals(False)
            return

        rows = self._session.execute(
            "SELECT id, label, capture_number FROM captures WHERE session_id=? ORDER BY capture_number",
            (self._session_id,),
        ).fetchall()

        for r in rows:
            label = r["label"] or f"Capture {r['capture_number']}"
            self._shot_combo.addItem(f"{r['capture_number']}: {label}", r["id"])

        self._shot_combo.blockSignals(False)
        if self._shot_combo.count() > 0:
            self._on_shot_changed(0)

    def _on_shot_changed(self, index: int) -> None:
        if self._session is None or index < 0:
            return
        self._shot_id = self._shot_combo.itemData(index)
        QTimer.singleShot(0, self._populate_syncs)
        QTimer.singleShot(0, self._populate_runs)
        QTimer.singleShot(0, self._load_cameras_for_shot)

    def _populate_syncs(self) -> None:
        self._sync_combo.blockSignals(True)
        self._sync_combo.clear()
        if self._session is not None and self._shot_id is not None:
            rows = self._session.execute(
                "SELECT id, created_by, notes FROM sync_configs WHERE shot_id=? ORDER BY rowid DESC",
                (self._shot_id,),
            ).fetchall()
            total = len(rows)
            for i, r in enumerate(rows):
                # Show newest-first; number as #N counting from oldest so #1 = first ever.
                n = total - i
                kind = r["created_by"] or r["id"][:8]
                label = f"{kind} #{n}"
                if r["notes"]:
                    label += f"  {r['notes']}"
                self._sync_combo.addItem(label, r["id"])
        self._sync_combo.blockSignals(False)
        self._on_sync_changed(0)

    def _on_sync_changed(self, index: int) -> None:
        self._sync_config_id = self._sync_combo.itemData(index) if index >= 0 else None
        QTimer.singleShot(0, self._load_cameras_for_shot)

    def _populate_runs(self) -> None:
        self._run_combo.blockSignals(True)
        self._run_combo.clear()
        if self._session is not None and self._shot_id is not None:
            runs = list_detection_runs(self._session, self._shot_id)
            for r in runs:
                label = (
                    f"{r['id'][:8]}  {r['status']}  "
                    f"{r['detector_model']}+{r['pose_model']}  "
                    f"[{r['time_start_s']:.1f}–{r['time_end_s']:.1f}s]  "
                    f"{r['created_at'][:16]}"
                )
                self._run_combo.addItem(label, r["id"])
        self._run_combo.blockSignals(False)
        self._on_run_selected(0)

    def _on_run_selected(self, index: int) -> None:
        if self._session is None or index < 0:
            self._current_run_id = None
            return
        run_id = self._run_combo.itemData(index)
        if run_id is None:
            self._current_run_id = None
            return
        self._current_run_id = run_id
        self._assignments.clear()
        self._current_seg_first = None

        # Auto-select the sync config that was used when this run was created,
        # so the stitcher timeline and the frame view are always in the same
        # time domain.  (Using different configs causes seek_global_time to
        # land at a completely wrong frame.)
        run_row = self._session.execute(
            "SELECT sync_config_id FROM detection_runs WHERE id = ?", (run_id,)
        ).fetchone()
        run_sync_id = run_row["sync_config_id"] if run_row else None

        if run_sync_id:
            combo_idx = self._sync_combo.findData(run_sync_id)
            if combo_idx >= 0 and combo_idx != self._sync_combo.currentIndex():
                self._sync_combo.setCurrentIndex(combo_idx)
                # _on_sync_changed will fire and reload cameras + set SyncTable

        # Build SyncTable from the run's own sync config (not the UI selection)
        # so the stitcher uses the correct sync regardless of the combo state.
        run_sync_table = (
            _load_sync_table(self._session, run_sync_id) if run_sync_id else None
        )
        self._stitcher.set_sync_table(run_sync_table)
        self._frame_view.set_sync_table(run_sync_table)

        self._stitcher.load_run(self._session, run_id)
        self._restore_finalised_assignments(run_id)

    def _restore_finalised_assignments(self, run_id: str) -> None:
        """Restore track→person assignments saved at finalise time.

        Reads from detection_track_assignments, which is written by
        finalise_to_db.  Each row records one contiguous segment of a track
        assigned to a named person, so splits and assignments are restored
        exactly as the user left them.
        """
        if self._session is None:
            return

        rows = self._session.execute(
            "SELECT shot_video_id, track_id, person_name, first_frame, last_frame"
            " FROM detection_track_assignments"
            " WHERE detection_run_id = ?"
            " ORDER BY shot_video_id, track_id, first_frame",
            (run_id,),
        ).fetchall()

        if not rows:
            return

        from collections import defaultdict
        by_track: dict[tuple[str, int], list] = defaultdict(list)
        for r in rows:
            by_track[(r["shot_video_id"], r["track_id"])].append(r)

        spans = self._stitcher.get_spans()
        for (svid, tid), track_rows in by_track.items():
            seg_first = next(
                (sf for (s, t, sf) in spans if s == svid and t == tid), None
            )
            if seg_first is None:
                continue

            cur_seg_first = seg_first
            for i, r in enumerate(track_rows):
                if i < len(track_rows) - 1:
                    split_frame = track_rows[i + 1]["first_frame"]
                    self._stitcher.split_segment(svid, tid, cur_seg_first, split_frame)
                    self._apply_assignment(svid, tid, cur_seg_first, r["person_name"])
                    cur_seg_first = split_frame
                else:
                    self._apply_assignment(svid, tid, cur_seg_first, r["person_name"])

    # ------------------------------------------------------------------
    # Detection job
    # ------------------------------------------------------------------

    def _on_run_detection(self) -> None:
        if (self._session is None or self._shot_id is None
                or self._sync_config_id is None
                or self._time_start_s is None or self._time_end_s is None):
            QMessageBox.warning(self, "Warning", "Please open a session, select a shot, and mark a time range.")
            return

        self._set_controls_enabled(False)
        self._cam_progress_bar.setValue(0)
        self._cam_progress_label.setText("")
        self._total_progress_bar.setValue(0)
        self._total_progress_label.setText("Starting...")

        self._job = DetectionJob(
            session_path=self._session_path,
            shot_id=self._shot_id,
            sync_config_id=self._sync_config_id,
            time_start_s=self._time_start_s,
            time_end_s=self._time_end_s,
            detector_name=self._detector_combo.currentText(),
            pose_model_name=self._pose_combo.currentText(),
            detector_conf=self._conf_spin.value(),
        )
        self._job.progress.connect(self._on_job_progress)
        self._job.camera_progress.connect(self._on_camera_progress)
        self._job.finished.connect(self._on_job_finished)
        self._job.error.connect(self._on_job_error)
        self._job.start()

    def _on_job_progress(self, pct: int, msg: str) -> None:
        self._cam_progress_bar.setValue(pct)
        self._cam_progress_label.setText(msg)

    def _on_camera_progress(self, done: int, total: int) -> None:
        self._total_progress_bar.setValue(int(done / max(total, 1) * 100))
        self._total_progress_label.setText(f"{done}/{total} cameras")

    def _on_job_finished(self, run_id: str) -> None:
        self._set_controls_enabled(True)
        self._cam_progress_bar.setValue(100)
        self._total_progress_bar.setValue(100)
        self._total_progress_label.setText("Done")
        self._populate_runs()
        # Select the new run
        idx = self._run_combo.findData(run_id)
        if idx >= 0:
            self._run_combo.setCurrentIndex(idx)

    def _on_job_error(self, msg: str) -> None:
        self._set_controls_enabled(True)
        self._total_progress_label.setText("Error")
        QMessageBox.critical(self, "Detection Error", msg)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for w in [
            self._open_btn, self._shot_combo, self._sync_combo,
            self._mark_start_btn, self._mark_end_btn,
            self._detector_combo, self._pose_combo, self._conf_spin,
            self._run_combo,
        ]:
            w.setEnabled(enabled)
        # Run button also requires both time marks to be set
        if enabled:
            self._update_run_btn()
        else:
            self._run_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Camera loading
    # ------------------------------------------------------------------

    def _load_cameras_for_shot(self) -> None:
        """Populate FrameViewWidget from shot_videos using the current sync config."""
        if self._session is None or self._shot_id is None:
            return

        rows = self._session.execute(
            "SELECT sv.id, sv.file_path, sv.actual_fps, ci.label "
            "FROM capture_videos sv "
            "JOIN camera_instances ci ON ci.id = sv.camera_instance_id "
            "WHERE sv.shot_id = ? ORDER BY ci.label",
            (self._shot_id,),
        ).fetchall()
        if not rows:
            return

        # Build per-camera sync anchors from the current sync config (if any)
        anchors: dict[str, tuple[int, float]] = {}
        if self._sync_config_id:
            sp_rows = self._session.execute(
                "SELECT shot_video_id, video_frame, timestamp_s "
                "FROM sync_points WHERE sync_config_id = ? "
                "ORDER BY shot_video_id, video_frame",
                (self._sync_config_id,),
            ).fetchall()
            for sp in sp_rows:
                svid = sp["shot_video_id"]
                if svid not in anchors:
                    anchors[svid] = (int(sp["video_frame"]), float(sp["timestamp_s"]))

        cameras: list[CameraInfo] = []
        for r in rows:
            ref_frame, ref_ts = anchors.get(r["id"], (0, 0.0))
            cameras.append(CameraInfo(
                shot_video_id=r["id"],
                file_path=r["file_path"] or "",
                camera_instance_id=r["id"],
                label=r["label"] or r["id"][:8],
                fps=float(r["actual_fps"] or 30.0),
                ref_frame=ref_frame,
                ref_timestamp_s=ref_ts,
            ))

        sync_table = (
            _load_sync_table(self._session, self._sync_config_id)
            if self._sync_config_id else None
        )
        self._frame_view.set_sync_table(sync_table)
        self._frame_view.load_cameras(cameras)

    # ------------------------------------------------------------------
    # Mark start / end
    # ------------------------------------------------------------------

    def _on_mark_start(self) -> None:
        t = self._frame_view.current_global_time()
        self._time_start_s = t
        mm, ss = int(t // 60), t % 60
        self._start_label.setText(f"{mm:02d}:{ss:05.2f}")
        self._update_run_btn()

    def _on_mark_end(self) -> None:
        t = self._frame_view.current_global_time()
        self._time_end_s = t
        mm, ss = int(t // 60), t % 60
        self._end_label.setText(f"{mm:02d}:{ss:05.2f}")
        self._update_run_btn()

    def _update_run_btn(self) -> None:
        ready = (
            self._session is not None
            and self._shot_id is not None
            and self._sync_config_id is not None
            and self._time_start_s is not None
            and self._time_end_s is not None
            and self._time_end_s > self._time_start_s
        )
        self._run_btn.setEnabled(ready)

    def _on_frame_changed(self, frame_idx: int, global_s: float) -> None:
        self._current_frame_idx = frame_idx
        self._stitcher.set_current_time(global_s)
        self._update_run_btn()
        self._sync_frame_view_assignments()

    # ------------------------------------------------------------------
    # Stitcher / frame view integration
    # ------------------------------------------------------------------

    def _on_segment_clicked(
        self,
        shot_video_id: str,
        track_id: int,
        first_frame: int,
        last_frame: int,
    ) -> None:
        """Handle a click on a segment bar in the stitcher.

        Stores the selected segment, updates the sidebar label, and loads
        the video and pose data for the frame view.

        Args:
            shot_video_id: Camera / shot-video ID of the clicked segment.
            track_id: Detection track ID.
            first_frame: First frame of the clicked segment (also its identifier).
            last_frame: Last frame of the clicked segment.
        """
        if self._session is None or self._current_run_id is None:
            return

        self._current_svid = shot_video_id
        self._current_track_id = track_id
        self._current_seg_first = first_frame
        self._selected_label.setText(
            f"video: {shot_video_id[:8]}  track: {track_id}  "
            f"frames {first_frame}–{last_frame}"
        )
        person_name = self._assignments.get((shot_video_id, track_id, first_frame))
        self._preview.set_track(track_id, person_name)

        # Load file path and camera_instance_id
        row = self._session.execute(
            "SELECT file_path, camera_instance_id, actual_fps FROM capture_videos WHERE id=?",
            (shot_video_id,),
        ).fetchone()
        if row is None:
            return

        # Switch the frame view to this camera.  The SyncTable was already set
        # in _on_run_selected, so global-time ↔ frame conversion is handled there.
        self._frame_view.load_camera(
            shot_video_id, row["file_path"], row["camera_instance_id"],
            fps=float(row["actual_fps"] or 30.0),
        )
        self._frame_view.set_pose_data(self._session, self._current_run_id, track_id)
        self._frame_view.set_selected_track(track_id)
        self._sync_frame_view_assignments()

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def _on_add_person(self) -> None:
        name, ok = QInputDialog.getText(self, "Add Person", "Person name:")
        if ok and name.strip():
            name = name.strip()
            if self._person_combo.findText(name) < 0:
                self._person_combo.addItem(name)
            self._person_combo.setCurrentText(name)
            persons = [self._person_combo.itemText(i) for i in range(self._person_combo.count())]
            self._stitcher.set_known_persons(persons)

    def _on_assign(self) -> None:
        """Handle the Assign button.

        Reads the person name from the combo box and assigns the currently
        selected segment.
        """
        if (self._current_svid is None
                or self._current_track_id is None
                or self._current_seg_first is None):
            QMessageBox.information(self, "No track selected", "Click a track segment first.")
            return
        name = self._person_combo.currentText().strip()
        if not name:
            QMessageBox.warning(self, "No name", "Enter a person name.")
            return
        self._do_assign(self._current_svid, self._current_track_id,
                        self._current_seg_first, name)

    def _on_assignment_changed(self, svid: str, tid: int, seg_first: int, person_name: object) -> None:
        """Handle an assignment or detach from the stitcher context menu.

        Args:
            svid: Shot video ID of the clicked segment.
            tid: Track ID.
            seg_first: First frame of the segment (its identifier).
            person_name: Person name string, or None for a detach operation.
        """
        if person_name:
            self._do_assign(svid, tid, seg_first, str(person_name))
        else:
            self._detach(svid, tid, seg_first)

    def _on_split_requested(self, svid: str, tid: int, seg_first: int, split_frame: int) -> None:
        """Handle a split request from the stitcher context menu.

        Splits the segment at *split_frame*.  The left half keeps its existing
        assignment; the right half starts unassigned.

        Args:
            svid: Shot video ID.
            tid: Track ID.
            seg_first: First frame of the segment to split.
            split_frame: Frame at which to split (first frame of the right half).
        """
        # Remove any assignment for the right half (it has no assignment yet, but
        # be defensive in case of a future double-split scenario)
        right_key = (svid, tid, split_frame)
        self._assignments.pop(right_key, None)

        self._stitcher.split_segment(svid, tid, seg_first, split_frame)
        self._sync_frame_view_assignments()

    def _do_assign(self, svid: str, tid: int, seg_first: int, name: str) -> None:
        """Check for conflicts, show a dialog if needed, then apply the segment assignment.

        Args:
            svid: Camera to assign within.
            tid: Track ID.
            seg_first: First frame of the segment to assign.
            name: Person name to assign.
        """
        seg_key = (svid, tid, seg_first)
        conflicts = find_assignment_conflicts(
            svid, [seg_key], name,
            self._stitcher.get_spans(), self._assignments,
        )
        if conflicts and not self._resolve_conflicts(conflicts, name):
            return
        for conflict_key in conflicts:
            self._detach(*conflict_key)
        self._apply_assignment(svid, tid, seg_first, name)

    def _resolve_conflicts(
        self,
        conflicts: list[tuple[str, int, int]],
        person_name: str,
    ) -> bool:
        """Show a conflict resolution dialog and return whether to proceed.

        Lists the conflicting segments with their frame ranges and offers two
        choices: detach all conflicting segments (returns True) or cancel
        (returns False).

        Args:
            conflicts: List of (svid, tid, seg_first) keys that overlap the proposed assignment.
            person_name: The person being assigned, shown in the dialog text.
        """
        msg = QMessageBox(self)
        msg.setWindowTitle("Assignment conflict")
        msg.setText(
            f"'{person_name}' is already assigned to {len(conflicts)} "
            f"overlapping segment(s) in this camera:"
        )
        lines = []
        spans = self._stitcher.get_spans()
        for key in conflicts:
            s, t, sf = key
            ff, lf = spans.get(key, (sf, sf))
            lines.append(f"  track {t}  frames {ff}–{lf}")
        msg.setInformativeText("\n".join(lines))
        detach_btn: QAbstractButton = msg.addButton("Detach conflicting segments", QMessageBox.AcceptRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        msg.exec()
        return msg.clickedButton() is detach_btn

    def _apply_assignment(self, svid: str, tid: int, seg_first: int, name: str) -> None:
        """Record a segment-to-person assignment and sync all dependent UI state.

        Updates the internal assignments dict, the stitcher segment colour and
        name label, the person combo box, and the frame view overlay colours.

        Args:
            svid: Shot video ID.
            tid: Track ID.
            seg_first: First frame of the segment to assign.
            name: Person name to assign.
        """
        self._assignments[(svid, tid, seg_first)] = name
        self._stitcher.set_segment_assignment(svid, tid, seg_first, name)
        if self._person_combo.findText(name) < 0:
            self._person_combo.addItem(name)
        persons = [self._person_combo.itemText(i) for i in range(self._person_combo.count())]
        self._stitcher.set_known_persons(persons)
        # Refresh preview name if the assignment is for the currently selected segment
        if (svid, tid, seg_first) == (self._current_svid, self._current_track_id,
                                       self._current_seg_first):
            self._preview.set_track(tid, name)
        self._sync_frame_view_assignments()

    def _detach(self, svid: str, tid: int, seg_first: int) -> None:
        """Remove the person assignment from one segment and sync all dependent UI state.

        Args:
            svid: Shot video ID.
            tid: Track ID.
            seg_first: First frame of the segment to detach.
        """
        self._assignments.pop((svid, tid, seg_first), None)
        self._stitcher.set_segment_assignment(svid, tid, seg_first, None)
        if (svid, tid, seg_first) == (self._current_svid, self._current_track_id,
                                       self._current_seg_first):
            self._preview.set_track(tid, None)
        self._sync_frame_view_assignments()

    def _sync_frame_view_assignments(self) -> None:
        """Push the current camera's track→person assignments to the frame view overlay.

        For each track, the assignment that applies at the current frame is
        used (a split track can have different assignments in different frame
        ranges).  Calls ``set_track_assignments`` on the frame view so bbox
        colours stay in sync with the stitcher.
        """
        if self._current_svid is None:
            return
        spans = self._stitcher.get_spans()
        f = self._current_frame_idx
        tid_to_person: dict[int, str] = {}
        for (svid, tid, sf), person in self._assignments.items():
            if svid != self._current_svid:
                continue
            seg_range = spans.get((svid, tid, sf))
            if seg_range and seg_range[0] <= f <= seg_range[1]:
                tid_to_person[tid] = person
        self._frame_view.set_track_assignments(tid_to_person)

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------

    def _on_finalise(self) -> None:
        if self._session is None or self._current_run_id is None:
            QMessageBox.warning(self, "Warning", "No detection run loaded.")
            return
        if not self._assignments:
            QMessageBox.warning(self, "Warning", "No track assignments defined.")
            return
        if self._shot_id is None or self._sync_config_id is None:
            QMessageBox.warning(self, "Warning", "Shot or sync config not selected.")
            return

        # Build one TrackAssignment per segment that has a person assigned.
        # The stitcher's get_spans() returns {(svid, tid, seg_first): (first, last)}.
        spans = self._stitcher.get_spans()
        assignment_list = []
        for (svid, tid, seg_first), person_name in self._assignments.items():
            seg_range = spans.get((svid, tid, seg_first))
            if seg_range is None:
                continue
            first_frame, last_frame = seg_range
            assignment_list.append(TrackAssignment(
                shot_video_id=svid,
                track_id=tid,
                person_name=person_name,
                first_frame=first_frame,
                last_frame=last_frame,
            ))

        if not assignment_list:
            QMessageBox.warning(self, "Warning", "No valid assignments found.")
            return

        # Get pose_model from the run
        run_row = self._session.execute(
            "SELECT pose_model FROM detection_runs WHERE id=?",
            (self._current_run_id,),
        ).fetchone()
        pose_model = run_row["pose_model"] if run_row else ""

        try:
            seq_ids = finalise_to_db(
                session=self._session,
                detection_run_id=self._current_run_id,
                shot_id=self._shot_id,
                sync_config_id=self._sync_config_id,
                assignments=assignment_list,
                pose_model=pose_model,
            )
        except Exception as e:
            QMessageBox.critical(self, "Finalise Error", str(e))
            return

        self.data_changed.emit()
        QMessageBox.information(
            self,
            "Finalised",
            f"Created {len(seq_ids)} person sequence(s):\n" + "\n".join(seq_ids),
        )
