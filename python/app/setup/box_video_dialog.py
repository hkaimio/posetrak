# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Dialog "Calibrate from box video": the GUI front end of ``app.setup.box_video``.

Runs the same code as ``posetrak extrinsics box-video``, shows the per-camera
error table, and offers to save only a calibration that passed.
"""

from __future__ import annotations

import sqlite3

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from app.setup.box_video import (
    BoxVideoError,
    BoxVideoParams,
    BoxVideoResult,
    calibrate_from_box_video,
    format_report,
)
from app.setup.extrinsics_solver import write_extrinsics_to_db
from app.setup.fiducial_markers import load_marker_body_yaml
from posetrak.db.db import set_capture_extrinsics


def _db_path(conn: sqlite3.Connection) -> str:
    return next(r[2] for r in conn.execute("PRAGMA database_list") if r[1] == "main")


class _Cancelled(Exception):
    pass


class _Worker(QThread):
    progress = Signal(str)
    finished_ok = Signal(object)   # BoxVideoResult
    failed = Signal(str)

    def __init__(self, db_path: str, capture_id: str, yaml_content: str, body_id: str,
                 params: BoxVideoParams, parent=None) -> None:
        super().__init__(parent)
        self._args = (db_path, capture_id, yaml_content, body_id, params)
        self._cancelled = False

    def cancel(self) -> None:
        """Stop at the next progress report (the final solve itself runs to its end)."""
        self._cancelled = True

    def _report(self, message: str) -> None:
        if self._cancelled:
            raise _Cancelled
        self.progress.emit(message)

    def run(self) -> None:
        db_path, capture_id, yaml_content, body_id, params = self._args
        # SQLite connections stay in the thread that made them: read with a connection of our own.
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            config = load_marker_body_yaml(yaml_content, rig_id=body_id)
            self.finished_ok.emit(
                calibrate_from_box_video(conn, capture_id, config, params, self._report))
        except _Cancelled:
            pass
        except BoxVideoError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 -- shown to the user, not swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            conn.close()


class BoxVideoDialog(QDialog):
    imported = Signal(str)  # extrinsic_calibration_id

    def __init__(self, conn: sqlite3.Connection, session_id: str, capture_id: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibrate from box video")
        self.setMinimumSize(720, 560)
        self._conn, self._session_id, self._capture_id = conn, session_id, capture_id
        self._result: BoxVideoResult | None = None
        self._worker: _Worker | None = None

        self._body = QComboBox()
        for body_id, name in conn.execute("SELECT id, name FROM marker_body_definitions ORDER BY name"):
            self._body.addItem(name or body_id[:8], body_id)
        self._camera = QComboBox()
        for (label,) in conn.execute(
            "SELECT ci.label FROM capture_videos cv JOIN camera_instances ci ON ci.id = cv.camera_instance_id "
            "WHERE cv.shot_id = ? ORDER BY ci.label", (capture_id,)):
            self._camera.addItem(label)
        self._start, self._end = self._spin(0, 100000, 6.0, " s"), self._spin(0, 100000, 33.0, " s")
        self._z_offset = self._spin(-10, 10, 0.0, " m", decimals=3)
        self._max_err = self._spin(0.1, 1000, 5.0, " px")
        self._exclude = QLineEdit()
        self._exclude.setPlaceholderText("window times to drop, seconds, comma separated")

        form = QFormLayout()
        form.addRow("Calibration box", self._body)
        form.addRow("Reference camera", self._camera)
        form.addRow("Scan from", self._start)
        form.addRow("Scan to (the box must be set down before)", self._end)
        form.addRow("World Z offset", self._z_offset)
        form.addRow("Fail above median error", self._max_err)
        form.addRow("Exclude windows", self._exclude)

        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(self._run)
        self._save_btn = QPushButton("Save calibration")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        self._status = QLabel("")
        self._report = QPlainTextEdit()
        self._report.setReadOnly(True)
        self._report.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        buttons = QHBoxLayout()
        buttons.addWidget(self._run_btn)
        buttons.addWidget(self._save_btn)
        buttons.addStretch()
        buttons.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._status)
        layout.addWidget(self._report, 1)
        layout.addLayout(buttons)

    @staticmethod
    def _spin(lo: float, hi: float, value: float, suffix: str, decimals: int = 1) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setValue(value)
        s.setSuffix(suffix)
        return s

    def _params(self) -> BoxVideoParams:
        exclude = [float(t) for t in self._exclude.text().split(",") if t.strip()]
        return BoxVideoParams(
            reference_camera=self._camera.currentText(), time_start_s=self._start.value(),
            time_end_s=self._end.value(), world_z_offset=self._z_offset.value(),
            max_median_error_px=self._max_err.value(), exclude_window_ts=exclude,
        )

    def _run(self) -> None:
        if self._body.currentData() is None or not self._camera.count():
            QMessageBox.warning(self, "Calibrate from box video",
                                "The capture needs videos and a calibration box (a marker body definition).")
            return
        try:
            params = self._params()
        except ValueError:
            QMessageBox.warning(self, "Calibrate from box video", "Exclude windows must be numbers.")
            return
        body_id = self._body.currentData()
        yaml_content = self._conn.execute(
            "SELECT yaml_content FROM marker_body_definitions WHERE id = ?", (body_id,)).fetchone()[0]
        self._result = None
        self._save_btn.setEnabled(False)
        self._run_btn.setEnabled(False)
        self._report.clear()
        self._worker = _Worker(_db_path(self._conn), self._capture_id, yaml_content, body_id, params, self)
        self._worker.progress.connect(self._status.setText)
        self._worker.finished_ok.connect(self._on_result)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_failed(self, message: str) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText("")
        self._report.setPlainText(f"Could not calibrate: {message}")

    def _on_result(self, res: BoxVideoResult) -> None:
        self._result = res
        self._run_btn.setEnabled(True)
        self._status.setText("Calibration OK" if res.ok else "Calibration FAILED: not saved")
        self._report.setPlainText(format_report(res))
        self._save_btn.setEnabled(res.ok)

    def _save(self) -> None:
        # A failed calibration never gets here: the button is only enabled for res.ok.
        if self._result is None or not self._result.ok:
            return
        labels = {r[1]: r[0] for r in self._conn.execute("SELECT id, label FROM camera_instances")}
        calib_id = write_extrinsics_to_db(
            self._result.calibration, self._conn, self._session_id, labels, method="moving-box-anchor")
        set_capture_extrinsics(self._conn, self._capture_id, calib_id)
        self.imported.emit(calib_id)
        self.accept()

    def done(self, code: int) -> None:  # noqa: D102 -- Qt override
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()
        super().done(code)
