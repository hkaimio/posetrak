"""main.py — PoseExtractionWindow: main GUI for the pose extraction pipeline."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.pose.db_cache import list_detection_runs
from app.pose.finalise import TrackAssignment, finalise_to_db
from app.pose.frame_view import FrameViewWidget
from app.pose.stitcher import StitcherWidget
from app.setup.job_runner import BackgroundJob


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

        # track -> person assignment (not persisted until Finalise)
        self._assignments: dict[tuple[str, int], str] = {}

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
        top_layout.addLayout(row1)

        # Row 2: shot + sync
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Shot:"))
        self._shot_combo = QComboBox()
        self._shot_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._shot_combo.currentIndexChanged.connect(self._on_shot_changed)
        row2.addWidget(self._shot_combo)
        row2.addWidget(QLabel("Sync:"))
        self._sync_combo = QComboBox()
        self._sync_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._sync_combo.currentIndexChanged.connect(self._on_sync_changed)
        row2.addWidget(self._sync_combo)
        row2.addStretch()
        top_layout.addLayout(row2)

        # Row 3: time range + detector + pose model + conf
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Start (s):"))
        self._start_spin = QDoubleSpinBox()
        self._start_spin.setRange(0, 99999)
        self._start_spin.setValue(0)
        self._start_spin.setSingleStep(1)
        row3.addWidget(self._start_spin)
        row3.addWidget(QLabel("End (s):"))
        self._end_spin = QDoubleSpinBox()
        self._end_spin.setRange(0, 99999)
        self._end_spin.setValue(60)
        self._end_spin.setSingleStep(1)
        row3.addWidget(self._end_spin)
        row3.addWidget(QLabel("Detector:"))
        self._detector_combo = QComboBox()
        self._detector_combo.addItems(["yolo11x", "yolo11l", "yolo11m"])
        row3.addWidget(self._detector_combo)
        row3.addWidget(QLabel("Pose model:"))
        self._pose_combo = QComboBox()
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
        self._run_combo = QComboBox()
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
        splitter.addWidget(self._frame_view)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._stitcher = StitcherWidget()
        self._stitcher.setMinimumHeight(150)
        self._stitcher.segment_clicked.connect(self._on_segment_clicked)
        right_layout.addWidget(self._stitcher, 1)

        # Assignment panel
        assign_group = QGroupBox("Selected track")
        assign_layout = QVBoxLayout(assign_group)

        self._selected_label = QLabel("None selected")
        assign_layout.addWidget(self._selected_label)

        person_row = QHBoxLayout()
        person_row.addWidget(QLabel("Person:"))
        self._person_combo = QComboBox()
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

        self._populate_shots()

    def _populate_shots(self) -> None:
        self._shot_combo.blockSignals(True)
        self._shot_combo.clear()
        if self._session is None or self._session_id is None:
            self._shot_combo.blockSignals(False)
            return

        rows = self._session.execute(
            "SELECT id, label, shot_number FROM shots WHERE session_id=? ORDER BY shot_number",
            (self._session_id,),
        ).fetchall()

        for r in rows:
            label = r["label"] or f"Shot {r['shot_number']}"
            self._shot_combo.addItem(f"{r['shot_number']}: {label}", r["id"])

        self._shot_combo.blockSignals(False)
        if self._shot_combo.count() > 0:
            self._on_shot_changed(0)

    def _on_shot_changed(self, index: int) -> None:
        if self._session is None or index < 0:
            return
        self._shot_id = self._shot_combo.itemData(index)
        QTimer.singleShot(0, self._populate_syncs)
        QTimer.singleShot(0, self._populate_runs)

    def _populate_syncs(self) -> None:
        self._sync_combo.currentIndexChanged.disconnect(self._on_sync_changed)
        self._sync_combo.clear()
        if self._session is not None and self._shot_id is not None:
            rows = self._session.execute(
                "SELECT id, created_by, notes FROM sync_configs WHERE shot_id=? ORDER BY rowid",
                (self._shot_id,),
            ).fetchall()
            for r in rows:
                label = r["created_by"] or r["id"][:8]
                if r["notes"]:
                    label += f" ({r['notes']})"
                self._sync_combo.addItem(label, r["id"])
        self._sync_combo.currentIndexChanged.connect(self._on_sync_changed)
        self._sync_config_id = self._sync_combo.itemData(0) if self._sync_combo.count() > 0 else None

    def _on_sync_changed(self, index: int) -> None:
        if index < 0:
            return
        self._sync_config_id = self._sync_combo.itemData(index)

    def _populate_runs(self) -> None:
        self._run_combo.currentIndexChanged.disconnect(self._on_run_selected)
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
        self._run_combo.currentIndexChanged.connect(self._on_run_selected)
        if self._run_combo.count() > 0:
            self._on_run_selected(0)

    def _on_run_selected(self, index: int) -> None:
        if self._session is None or index < 0:
            return
        run_id = self._run_combo.itemData(index)
        if run_id is None:
            return
        self._current_run_id = run_id
        self._assignments.clear()
        self._stitcher.load_run(self._session, run_id)

    # ------------------------------------------------------------------
    # Detection job
    # ------------------------------------------------------------------

    def _on_run_detection(self) -> None:
        if self._session is None or self._shot_id is None or self._sync_config_id is None:
            QMessageBox.warning(self, "Warning", "Please open a session and select a shot.")
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
            time_start_s=self._start_spin.value(),
            time_end_s=self._end_spin.value(),
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
            self._start_spin, self._end_spin, self._detector_combo,
            self._pose_combo, self._conf_spin, self._run_btn, self._run_combo,
        ]:
            w.setEnabled(enabled)

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
        if self._session is None or self._current_run_id is None:
            return

        self._current_svid = shot_video_id
        self._current_track_id = track_id
        self._selected_label.setText(
            f"video: {shot_video_id[:8]}  track: {track_id}  "
            f"frames {first_frame}–{last_frame}"
        )

        # Load file path and camera_instance_id
        row = self._session.execute(
            "SELECT file_path, camera_instance_id, actual_fps FROM shot_videos WHERE id=?",
            (shot_video_id,),
        ).fetchone()
        if row is None:
            return

        # Load sync anchor for global-time display in frame view
        run_row = self._session.execute(
            "SELECT sync_config_id FROM detection_runs WHERE id=?",
            (self._current_run_id,),
        ).fetchone()
        ref_frame, ref_ts = 0, 0.0
        if run_row:
            sp = self._session.execute(
                "SELECT video_frame, timestamp_s FROM sync_points "
                "WHERE shot_video_id=? AND sync_config_id=? "
                "ORDER BY video_frame ASC LIMIT 1",
                (shot_video_id, run_row["sync_config_id"]),
            ).fetchone()
            if sp:
                ref_frame, ref_ts = int(sp["video_frame"]), float(sp["timestamp_s"])

        self._frame_view.load_camera(
            shot_video_id, row["file_path"], row["camera_instance_id"],
            fps=float(row["actual_fps"] or 30.0),
            ref_frame=ref_frame,
            ref_timestamp_s=ref_ts,
        )
        self._frame_view.seek_frame(first_frame)
        self._frame_view.set_pose_data(self._session, self._current_run_id, track_id)

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

    def _on_assign(self) -> None:
        if self._current_svid is None or self._current_track_id is None:
            QMessageBox.information(self, "No track selected", "Click a track segment first.")
            return

        name = self._person_combo.currentText().strip()
        if not name:
            QMessageBox.warning(self, "No name", "Enter a person name.")
            return

        key = (self._current_svid, self._current_track_id)
        self._assignments[key] = name
        self._stitcher.set_assignment(self._current_svid, self._current_track_id, name)

        # Add to combo if not present
        if self._person_combo.findText(name) < 0:
            self._person_combo.addItem(name)

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

        # Get detection_run span data to build assignments
        # Load all track span info
        assignment_list = []
        for (svid, tid), person_name in self._assignments.items():
            row = self._session.execute(
                "SELECT first_frame, last_frame FROM person_tracks "
                "WHERE detection_run_id=? AND shot_video_id=? AND track_id=?",
                (self._current_run_id, svid, tid),
            ).fetchone()
            if row is None:
                continue
            assignment_list.append(TrackAssignment(
                shot_video_id=svid,
                track_id=tid,
                person_name=person_name,
                first_frame=row["first_frame"],
                last_frame=row["last_frame"],
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
            seq_id = finalise_to_db(
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

        QMessageBox.information(
            self,
            "Finalised",
            f"pose_observation_sequence created:\n{seq_id}",
        )
