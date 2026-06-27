"""run_detection_dialog.py — Modal dialog to configure and launch a detection run.

Opens from CapturePanel.  On success it creates a trials row, links the new
detection_run to it, and emits detection_finished(trial_id, run_id) so the
caller can refresh the session tree.
"""
from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QVBoxLayout,
)

from posetrak.db.db import generate_id


class RunDetectionDialog(QDialog):
    """Configure model, time range, and trial name; run detection in background."""

    detection_finished = Signal(str, str)  # trial_id, run_id

    def __init__(
        self,
        conn: sqlite3.Connection,
        session_path: Path,
        capture_id: str,
        time_start_s: float | None = None,
        time_end_s: float | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Detect Pose")
        self.setMinimumWidth(520)
        self._conn = conn
        self._session_path = session_path
        self._capture_id = capture_id
        self._job = None
        self._build_ui(time_start_s, time_end_s)

    # ------------------------------------------------------------------

    def _build_ui(self, time_start_s: float | None, time_end_s: float | None) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Trial name
        trial_count = self._conn.execute(
            "SELECT COUNT(*) FROM trials WHERE capture_id = ?", (self._capture_id,)
        ).fetchone()[0]
        self._trial_name = QLineEdit(f"Trial {trial_count + 1}")
        form.addRow("Trial name:", self._trial_name)

        # Sync config
        syncs = self._conn.execute(
            "SELECT id, created_by, notes FROM sync_configs WHERE shot_id = ? ORDER BY rowid",
            (self._capture_id,),
        ).fetchall()
        self._sync_combo = QComboBox()
        for s in syncs:
            label = s["created_by"] or "sync"
            if s["notes"]:
                label += f" — {s['notes']}"
            self._sync_combo.addItem(label, s["id"])
        form.addRow("Sync config:", self._sync_combo)

        # Time range
        self._start_spin = QDoubleSpinBox()
        self._start_spin.setRange(0.0, 100_000.0)
        self._start_spin.setDecimals(2)
        self._start_spin.setSuffix(" s")
        self._start_spin.setValue(time_start_s if time_start_s is not None else 0.0)

        self._end_spin = QDoubleSpinBox()
        self._end_spin.setRange(0.0, 100_000.0)
        self._end_spin.setDecimals(2)
        self._end_spin.setSuffix(" s")
        self._end_spin.setValue(time_end_s if time_end_s is not None else 0.0)

        time_row = QHBoxLayout()
        time_row.addWidget(self._start_spin)
        time_row.addWidget(QLabel("to"))
        time_row.addWidget(self._end_spin)
        time_widget = self._make_row_widget(time_row)
        form.addRow("Time range:", time_widget)

        # Model selection
        self._detector_combo = QComboBox()
        self._detector_combo.addItems(["yolo11x", "yolo11l", "yolo11m"])
        form.addRow("Detector:", self._detector_combo)

        self._pose_combo = QComboBox()
        self._pose_combo.addItems(["rtmpose-l-133kp", "vitpose-l-133kp"])
        form.addRow("Pose model:", self._pose_combo)

        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.01, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.3)
        form.addRow("Confidence:", self._conf_spin)

        layout.addLayout(form)

        # Progress
        self._frame_bar = QProgressBar()
        self._frame_bar.setRange(0, 100)
        self._frame_label = QLabel("")
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frames:"))
        frame_row.addWidget(self._frame_bar, 1)
        frame_row.addWidget(self._frame_label)
        layout.addLayout(frame_row)

        self._cam_bar = QProgressBar()
        self._cam_bar.setRange(0, 100)
        self._cam_label = QLabel("")
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Cameras:"))
        cam_row.addWidget(self._cam_bar, 1)
        cam_row.addWidget(self._cam_label)
        layout.addLayout(cam_row)

        # Buttons
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Detection")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)
        self._close_btn = QPushButton("Cancel")
        self._close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._close_btn)
        layout.addLayout(btn_row)

        if not syncs:
            self._run_btn.setEnabled(False)
            self._run_btn.setToolTip("No sync config — set one up first")

    @staticmethod
    def _make_row_widget(layout: QHBoxLayout):
        from PySide6.QtWidgets import QWidget
        w = QWidget()
        w.setLayout(layout)
        return w

    # ------------------------------------------------------------------

    def _controls_enabled(self, enabled: bool) -> None:
        for w in [
            self._trial_name, self._sync_combo,
            self._start_spin, self._end_spin,
            self._detector_combo, self._pose_combo, self._conf_spin,
            self._run_btn, self._close_btn,
        ]:
            w.setEnabled(enabled)

    def _on_run(self) -> None:
        sync_id = self._sync_combo.currentData()
        if not sync_id:
            QMessageBox.warning(self, "Missing sync", "No sync config selected.")
            return
        start_s = self._start_spin.value()
        end_s = self._end_spin.value()
        if end_s <= start_s:
            QMessageBox.warning(self, "Invalid range", "End time must be after start time.")
            return

        self._controls_enabled(False)
        self._frame_bar.setValue(0)
        self._cam_bar.setValue(0)
        self._cam_label.setText("Starting…")

        from app.pose.main import DetectionJob
        self._job = DetectionJob(
            session_path=str(self._session_path),
            shot_id=self._capture_id,
            sync_config_id=sync_id,
            time_start_s=start_s,
            time_end_s=end_s,
            detector_name=self._detector_combo.currentText(),
            pose_model_name=self._pose_combo.currentText(),
            detector_conf=self._conf_spin.value(),
        )
        self._job.progress.connect(self._on_progress)
        self._job.camera_progress.connect(self._on_camera_progress)
        self._job.finished.connect(self._on_finished)
        self._job.error.connect(self._on_error)
        self._job.start()

    def _on_progress(self, pct: int, msg: str) -> None:
        self._frame_bar.setValue(pct)
        self._frame_label.setText(msg)

    def _on_camera_progress(self, done: int, total: int) -> None:
        self._cam_bar.setValue(int(done / max(total, 1) * 100))
        self._cam_label.setText(f"{done}/{total}")

    def _on_finished(self, run_id: str) -> None:
        self._frame_bar.setValue(100)
        self._cam_bar.setValue(100)
        self._cam_label.setText("Done")

        # Create trial and link detection run to it
        trial_id = generate_id()
        name = self._trial_name.text().strip() or "Trial"
        self._conn.execute(
            "INSERT INTO trials (id, capture_id, name, time_start_s, time_end_s) "
            "VALUES (?, ?, ?, ?, ?)",
            (trial_id, self._capture_id, name,
             self._start_spin.value(), self._end_spin.value()),
        )
        self._conn.execute(
            "UPDATE detection_runs SET trial_id = ? WHERE id = ?",
            (trial_id, run_id),
        )
        self._conn.commit()

        self.detection_finished.emit(trial_id, run_id)

        self._close_btn.setEnabled(True)
        self._close_btn.setText("Close")
        self._close_btn.clicked.disconnect()
        self._close_btn.clicked.connect(self.accept)

    def _on_error(self, msg: str) -> None:
        self._controls_enabled(True)
        self._cam_label.setText("Error")
        QMessageBox.critical(self, "Detection Error", msg)
