"""run_tracker.py — Widget and dialog for running the posetrak tracker binary."""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from posetrak.tracker.runner import TrackerResult, default_binary_path
from posetrak.tracker.runner import run_tracker as _run_tracker

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOOLS_DIR = _REPO_ROOT / "python" / "tools"
_DEFAULT_BINARY = default_binary_path()


# ---------------------------------------------------------------------------
# Background thread
# ---------------------------------------------------------------------------


class _TrackerThread(QThread):
    """Runs run_tracker() in a background thread and emits Qt signals."""

    line_output = Signal(str)
    # exit_code as `object`, not `int`: on Windows, a process killed before
    # main() runs (e.g. a missing DLL dependency) reports its NTSTATUS code
    # as the return code, which is always > INT32_MAX -- marshalling that
    # into a C++ `int` signal argument overflows (a real crash reported as
    # "libshiboken: Overflow" once, traced to exactly this).
    tracking_finished = Signal(object, str)  # exit_code, run_id (empty str if None)

    def __init__(
        self,
        *,
        session_path: str,
        sequence_id: str,
        skeleton_id: str,
        config_id: str,
        output_dir: Path,
        binary_path: Path,
        person_id: int,
        start_time: float,
        end_time: float,
        smooth: bool,
    ) -> None:
        super().__init__()
        self._kwargs = dict(
            session_path=Path(session_path),
            sequence_id=sequence_id,
            skeleton_id=skeleton_id,
            config_id=config_id,
            output_dir=output_dir,
            binary_path=binary_path,
            person_id=person_id,
            start_time=start_time,
            end_time=end_time,
            smooth=smooth,
        )

    def run(self) -> None:
        result: TrackerResult = _run_tracker(**self._kwargs, on_progress=self.line_output.emit)
        self.tracking_finished.emit(result.exit_code, result.run_id or "")


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


class RunTrackerWidget(QWidget):
    """Configure and run the posetrak tracker against an open session database."""

    run_finished = Signal(str)  # emits tracking_run_id on successful completion

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._conn: sqlite3.Connection | None = None
        self._session_path: str | None = None
        self._run_id: str | None = None
        self._person_id: int = 0
        self._thread: _TrackerThread | None = None
        self._bvh_process = None
        self._sequence_cameras: list[str] = []
        self._velocity_cam_indices: set[int] = set()

        # ---- Configuration group ----------------------------------------
        self._skeleton_combo = QComboBox()

        self._person_id_spin = QSpinBox()
        self._person_id_spin.setRange(0, 99)
        self._person_id_spin.setValue(0)
        self._person_id_spin.setToolTip(
            "Index of the person to track (0 for single-person sessions)"
        )

        self._proc_noise_std  = _float_spin(0.1,   0.0, 1000.0, 4)
        self._proc_vel_noise  = _float_spin(0.5,   0.0, 1000.0, 4)
        self._vel_half_life   = _float_spin(0.25,  0.0,   10.0, 4)
        self._pose_noise      = _float_spin(0.0,   0.0, 1.0e6,  2)
        self._calib_noise     = _float_spin(60.0,  0.0, 1.0e6,  2)
        self._outlier_thresh  = _float_spin(4.0,   0.1,   50.0, 2)
        self._tracker_fps     = _float_spin(120.0, 1.0,  500.0, 1)

        self._vel_cam_label = QLabel("None")
        vel_cam_edit_btn = QPushButton("Edit…")
        vel_cam_edit_btn.setFixedWidth(60)
        vel_cam_edit_btn.clicked.connect(self._edit_velocity_cameras)
        vel_cam_row = QHBoxLayout()
        vel_cam_row.addWidget(self._vel_cam_label, 1)
        vel_cam_row.addWidget(vel_cam_edit_btn)

        self._use_relative = QCheckBox()
        self._use_relative.setChecked(False)
        self._use_relative.setToolTip(
            "Emit child-minus-parent pixel observations alongside absolute positions.\n"
            "Calibration error cancels in the difference; requires pose_noise_std > 0."
        )
        self._relative_min_conf = _float_spin(0.5, 0.0, 1.0, 2)
        self._relative_min_conf.setToolTip(
            "Minimum keypoint confidence for both child and parent to form a relative pair."
        )
        self._use_relative.toggled.connect(self._relative_min_conf.setEnabled)
        self._relative_min_conf.setEnabled(False)

        self._cross_pair_max_px = _float_spin(0.0, 0.0, 9999.0, 1)
        self._cross_pair_max_px.setToolTip(
            "Pixel radius for spatial cross-pair relative observations.\n"
            "Pairs of visible markers within this distance and > 2 skeleton hops apart\n"
            "emit an additional RELATIVE observation. 0 = disabled (Phase 4)."
        )
        self._cross_pair_max_n = QSpinBox()
        self._cross_pair_max_n.setRange(1, 999)
        self._cross_pair_max_n.setValue(10)
        self._cross_pair_max_n.setToolTip(
            "Maximum spatial cross-pairs per frame per camera (closest pairs kept)."
        )
        self._cross_pair_max_px.valueChanged.connect(
            lambda v: self._cross_pair_max_n.setEnabled(v > 0.0)
        )
        self._cross_pair_max_n.setEnabled(False)

        config_form = QFormLayout()
        config_form.addRow("Skeleton:", self._skeleton_combo)
        config_form.addRow("Person ID:", self._person_id_spin)
        config_form.addRow("Process noise std:", self._proc_noise_std)
        config_form.addRow("Velocity noise std:", self._proc_vel_noise)
        config_form.addRow("Velocity half-life (s):", self._vel_half_life)
        config_form.addRow("Pose noise std (px in model):", self._pose_noise)
        config_form.addRow("Calib noise std (px in video):", self._calib_noise)
        config_form.addRow("Outlier threshold:", self._outlier_thresh)
        config_form.addRow("Tracker FPS:", self._tracker_fps)
        config_form.addRow("Velocity cameras:", vel_cam_row)
        config_form.addRow("Relative observations:", self._use_relative)
        config_form.addRow("Relative min confidence:", self._relative_min_conf)
        config_form.addRow("Cross-pair radius (px):", self._cross_pair_max_px)
        config_form.addRow("Cross-pair max count:", self._cross_pair_max_n)

        config_box = QGroupBox("Tracker configuration")
        config_box.setLayout(config_form)

        # ---- Run group --------------------------------------------------
        self._sequence_combo = QComboBox()

        self._out_dir_edit = QLineEdit()
        self._out_dir_edit.setPlaceholderText(
            "Leave empty for <session-dir>/posetrak_results/<shot>/<skeleton>/"
        )
        out_browse_btn = QPushButton("Browse…")
        out_browse_btn.clicked.connect(self._browse_out_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_dir_edit, 1)
        out_row.addWidget(out_browse_btn)

        self._binary_edit = QLineEdit(str(_DEFAULT_BINARY))
        bin_browse_btn = QPushButton("Browse…")
        bin_browse_btn.clicked.connect(self._browse_binary)
        bin_row = QHBoxLayout()
        bin_row.addWidget(self._binary_edit, 1)
        bin_row.addWidget(bin_browse_btn)

        self._sequence_combo.currentIndexChanged.connect(self._on_sequence_changed)

        run_form = QFormLayout()
        run_form.addRow("Pose sequence:", self._sequence_combo)
        run_form.addRow("Output directory:", out_row)
        run_form.addRow("Tracker binary:", bin_row)

        run_box = QGroupBox("Run")
        run_box.setLayout(run_form)

        # ---- Run button -------------------------------------------------
        self._run_btn = QPushButton("Run Tracker")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._start_tracking)

        # ---- Progress group (hidden until run starts) -------------------
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._status_label = QLabel("")
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self._log.setFont(mono)

        prog_layout = QVBoxLayout()
        prog_layout.addWidget(self._progress_bar)
        prog_layout.addWidget(self._status_label)
        prog_layout.addWidget(self._log)
        self._prog_box = QGroupBox("Progress")
        self._prog_box.setLayout(prog_layout)
        self._prog_box.setVisible(False)

        # ---- Results group (hidden until run completes) ----------------
        self._results_label = QLabel("")
        self._results_label.setWordWrap(True)
        self._export_bvh_btn = QPushButton("Export BVH…")
        self._export_bvh_btn.clicked.connect(self._export_bvh)

        results_layout = QVBoxLayout()
        results_layout.addWidget(self._results_label)
        results_layout.addWidget(self._export_bvh_btn)
        self._results_box = QGroupBox("Results")
        self._results_box.setLayout(results_layout)
        self._results_box.setVisible(False)

        # ---- Root layout ------------------------------------------------
        root = QVBoxLayout(self)
        root.addWidget(config_box)
        root.addWidget(run_box)
        root.addWidget(self._run_btn)
        root.addWidget(self._prog_box)
        root.addWidget(self._results_box)
        root.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_session(self, conn: sqlite3.Connection, session_path: str) -> None:
        """Supply open session connection and the path to its .db file."""
        self._conn = conn
        self._session_path = session_path
        self._refresh_skeletons()
        self._refresh_sequences()
        self._update_run_btn()

    def preselect_sequence(self, seq_id: str) -> None:
        """Pre-select and lock the sequence combo to *seq_id*.

        Call after set_session().  The combo is disabled so the user cannot
        change the sequence when this widget is embedded in a PersonPanel.
        """
        for i in range(self._sequence_combo.count()):
            item_seq_id, _, _ = self._sequence_combo.itemData(i)
            if item_seq_id == seq_id:
                self._sequence_combo.setCurrentIndex(i)
                break
        self._sequence_combo.setEnabled(False)
        self._update_run_btn()

    def set_person_id(self, person_id: int) -> None:
        """Pre-fill the person ID spinner."""
        self._person_id_spin.setValue(person_id)

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    def _refresh_skeletons(self) -> None:
        self._skeleton_combo.clear()
        if self._conn is None:
            return
        rows = self._conn.execute(
            "SELECT id, name FROM skeletons ORDER BY name"
        ).fetchall()
        for r in rows:
            self._skeleton_combo.addItem(r["name"] or r["id"][:12], r["id"])

    def _refresh_sequences(self) -> None:
        self._sequence_combo.clear()
        if self._conn is None:
            return
        rows = self._conn.execute(
            "SELECT pos.id AS seq_id, sh.label AS shot_label, sh.capture_number,"
            "       pos.time_start_s, pos.time_end_s,"
            "       sh.extrinsic_calibration_id, pos.sync_config_id"
            " FROM pose_observation_sequences pos"
            " JOIN captures sh ON sh.id = pos.shot_id"
            " ORDER BY sh.capture_number, pos.time_start_s"
        ).fetchall()
        for r in rows:
            shot = r["shot_label"] or f"capture{r['capture_number']:03d}"
            duration = r["time_end_s"] - r["time_start_s"]
            label = f"{shot}  [{r['time_start_s']:.1f}–{r['time_end_s']:.1f}s, {duration:.1f}s]"
            missing = []
            if not r["sync_config_id"]:
                missing.append("no sync")
            if not r["extrinsic_calibration_id"]:
                missing.append("no extrinsics")
            if missing:
                label += f"  ⚠ {', '.join(missing)}"
            self._sequence_combo.addItem(
                label, (r["seq_id"], r["time_start_s"], r["time_end_s"])
            )

    def _update_run_btn(self) -> None:
        ok = (
            self._skeleton_combo.count() > 0
            and self._sequence_combo.count() > 0
        )
        self._run_btn.setEnabled(ok)

    def _on_sequence_changed(self) -> None:
        data = self._sequence_combo.currentData()
        if data is None or self._conn is None:
            self._sequence_cameras = []
        else:
            seq_id, _, _ = data
            self._sequence_cameras = self._cameras_for_sequence(seq_id)
        self._velocity_cam_indices = set()
        self._update_velocity_cam_label()

    def _cameras_for_sequence(self, seq_id: str) -> list[str]:
        row = self._conn.execute(
            "SELECT pos.sync_config_id FROM pose_observation_sequences pos WHERE pos.id = ?",
            (seq_id,),
        ).fetchone()
        if not row or not row["sync_config_id"]:
            return []
        sync_id = row["sync_config_id"]
        rows = self._conn.execute(
            "SELECT ci.label"
            " FROM capture_videos sv"
            " JOIN captures sh ON sh.id = sv.shot_id"
            " JOIN sync_configs scfg ON scfg.shot_id = sh.id"
            " JOIN camera_instances ci ON ci.id = sv.camera_instance_id"
            " WHERE scfg.id = ?"
            " ORDER BY ci.label ASC",
            (sync_id,),
        ).fetchall()
        return [r["label"] for r in rows]

    def _update_velocity_cam_label(self) -> None:
        if not self._velocity_cam_indices or not self._sequence_cameras:
            self._vel_cam_label.setText("None")
        else:
            names = [
                self._sequence_cameras[i]
                for i in sorted(self._velocity_cam_indices)
                if i < len(self._sequence_cameras)
            ]
            self._vel_cam_label.setText(", ".join(names) if names else "None")

    def _edit_velocity_cameras(self) -> None:
        if not self._sequence_cameras:
            QMessageBox.information(self, "No cameras", "Select a sequence with cameras first.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Velocity mode cameras")
        layout = QVBoxLayout(dlg)
        label = QLabel(
            "Cameras in velocity mode use keypoint displacement between frames as the "
            "measurement instead of absolute position. Select cameras with poor or "
            "uncertain absolute calibration."
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        checkboxes: list[QCheckBox] = []
        for i, cam_label in enumerate(self._sequence_cameras):
            cb = QCheckBox(cam_label)
            cb.setChecked(i in self._velocity_cam_indices)
            checkboxes.append(cb)
            layout.addWidget(cb)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._velocity_cam_indices = {i for i, cb in enumerate(checkboxes) if cb.isChecked()}
            self._update_velocity_cam_label()

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self._out_dir_edit.setText(path)

    def _browse_binary(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select posetrak binary", "", "All files (*)")
        if path:
            self._binary_edit.setText(path)

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def _start_tracking(self) -> None:
        if self._conn is None or self._session_path is None:
            return

        binary = Path(self._binary_edit.text())
        if not binary.exists():
            QMessageBox.critical(
                self,
                "Binary not found",
                f"Cannot find tracker binary:\n{binary}\n\n"
                "Build the optimised release first:\n"
                "  meson setup optbuild --buildtype=release\n"
                "  meson compile -C optbuild",
            )
            return

        seq_id, time_start_s, time_end_s = self._sequence_combo.currentData()
        skel_id = self._skeleton_combo.currentData()
        person_id = self._person_id_spin.value()

        err = self._check_sequence_ready(seq_id)
        if err:
            QMessageBox.critical(self, "Cannot run tracker", err)
            return

        out_dir = self._resolve_out_dir(seq_id, skel_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        config_id = self._create_config()
        self._person_id = person_id
        self._run_id = None

        self._progress_bar.setValue(0)
        self._status_label.setText("Starting…")
        self._log.clear()
        self._prog_box.setVisible(True)
        self._results_box.setVisible(False)
        self._run_btn.setEnabled(False)

        thread = _TrackerThread(
            session_path=self._session_path,
            sequence_id=seq_id,
            skeleton_id=skel_id,
            config_id=config_id,
            output_dir=out_dir,
            binary_path=binary,
            person_id=person_id,
            start_time=time_start_s,
            end_time=time_end_s,
            smooth=True,
        )
        thread.line_output.connect(self._on_output)
        thread.tracking_finished.connect(self._on_finished)
        thread.start()
        self._thread = thread

    def _check_sequence_ready(self, seq_id: str) -> str | None:
        """Return an error message if the sequence is missing sync or extrinsics, else None."""
        row = self._conn.execute(
            "SELECT s.label, s.extrinsic_calibration_id, pos.sync_config_id"
            " FROM pose_observation_sequences pos"
            " JOIN captures s ON s.id = pos.shot_id"
            " WHERE pos.id = ?",
            (seq_id,),
        ).fetchone()
        if row is None:
            return f"Sequence '{seq_id}' not found in the database."
        shot = row["label"] or seq_id[:12]
        if not row["sync_config_id"]:
            return (
                f"Capture \"{shot}\" has no sync configuration.\n\n"
                "Run the setup wizard and complete the Camera Synchronisation step "
                "before tracking."
            )
        if not row["extrinsic_calibration_id"]:
            return (
                f"Capture \"{shot}\" has no extrinsic calibration.\n\n"
                "Run the setup wizard and complete the Extrinsics step before tracking."
            )
        return None

    def _resolve_out_dir(self, seq_id: str, skel_id: str) -> Path:
        explicit = self._out_dir_edit.text().strip()
        if explicit:
            return Path(explicit)
        db_dir = Path(self._session_path).parent
        seq_row = self._conn.execute(
            "SELECT sh.label, sh.capture_number"
            " FROM pose_observation_sequences pos"
            " JOIN captures sh ON sh.id = pos.shot_id"
            " WHERE pos.id = ?",
            (seq_id,),
        ).fetchone()
        shot = (
            seq_row["label"] if seq_row and seq_row["label"]
            else f"capture{seq_row['capture_number']:03d}" if seq_row
            else "capture"
        )
        skel_name = (self._skeleton_combo.currentText() or "skeleton").replace(" ", "_")
        return db_dir / "posetrak_results" / shot / skel_name / "tracking"

    def _create_config(self) -> str:
        import datetime as dt
        import json
        from posetrak.db.db import generate_id
        config_id = generate_id()
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        vel_ids = sorted(self._velocity_cam_indices) if self._velocity_cam_indices else None
        vel_ids_json = json.dumps(vel_ids) if vel_ids is not None else None
        use_rel = 1 if self._use_relative.isChecked() else 0
        rel_min_conf = self._relative_min_conf.value() if use_rel else None
        cross_px = self._cross_pair_max_px.value()
        cross_n = self._cross_pair_max_n.value() if cross_px > 0.0 else None
        cross_px_val = cross_px if cross_px > 0.0 else None
        with self._conn:
            self._conn.execute(
                "INSERT INTO tracker_configs"
                " (id, name, parent_id, created_at,"
                "  process_noise_std, process_noise_vel_std, velocity_half_life_s,"
                "  measurement_noise_std, pose_noise_std, outlier_threshold, tracker_fps,"
                "  velocity_mode_camera_ids,"
                "  use_relative_observations, relative_min_confidence,"
                "  cross_pair_max_px, cross_pair_max_n)"
                " VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    config_id, "ui-run", now,
                    self._proc_noise_std.value(),
                    self._proc_vel_noise.value(),
                    self._vel_half_life.value(),
                    self._calib_noise.value(),   # stored in legacy column for compat
                    self._pose_noise.value(),
                    self._outlier_thresh.value(),
                    self._tracker_fps.value(),
                    vel_ids_json,
                    use_rel,
                    rel_min_conf,
                    cross_px_val,
                    cross_n,
                ),
            )
        return config_id

    def _on_output(self, line: str) -> None:
        m = re.match(r"\s*Progress:\s*(\d+)/(\d+)\s*\(([0-9.]+)%\)", line)
        if m:
            self._progress_bar.setValue(int(float(m.group(3))))
            self._status_label.setText(line)
        else:
            self._log.appendPlainText(line)

    def _on_finished(self, exit_code: int, run_id: str) -> None:
        self._run_id = run_id or None
        self._thread = None
        self._run_btn.setEnabled(True)

        if exit_code != 0:
            self._progress_bar.setValue(0)
            self._status_label.setText(f"Tracker exited with code {exit_code}.")
            detail = _describe_windows_exit_code(exit_code)
            if detail:
                QMessageBox.critical(self, "Tracker failed to start", detail)
            return

        self._progress_bar.setValue(100)
        self._status_label.setText("Tracking complete.")
        self._show_results()
        if self._run_id:
            self.run_finished.emit(self._run_id)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _show_results(self) -> None:
        if self._conn is None or self._run_id is None:
            return
        row = self._conn.execute(
            "SELECT COUNT(*) AS total,"
            "       SUM(CASE WHEN tracking_lost = 0 THEN 1 ELSE 0 END) AS tracked,"
            "       AVG(COALESCE(n_inlier_observations, 0)) AS avg_inliers"
            " FROM tracking_results"
            " WHERE run_id = ? AND person_id = ? AND is_smoothed = 0",
            (self._run_id, self._person_id),
        ).fetchone()
        if row and row["total"]:
            total = row["total"]
            tracked = row["tracked"] or 0
            pct = 100.0 * tracked / total
            avg = row["avg_inliers"] or 0.0
            text = (
                f"Run: {self._run_id[:16]}…\n"
                f"Tracked: {tracked}/{total} steps ({pct:.1f}%)\n"
                f"Average inliers per step: {avg:.1f}"
            )
        else:
            text = f"Run: {self._run_id[:16]}…\n(No per-frame stats available.)"
        self._results_label.setText(text)
        self._results_box.setVisible(True)

    # ------------------------------------------------------------------
    # BVH export
    # ------------------------------------------------------------------

    def _export_bvh(self) -> None:
        if self._run_id is None or self._session_path is None:
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save BVH file", "", "BVH files (*.bvh)")
        if not out_path:
            return

        export_script = _TOOLS_DIR / "export_bvh.py"
        self._status_label.setText("Exporting BVH…")
        self._export_bvh_btn.setEnabled(False)

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._bvh_process = proc

        def _done(code: int, _status) -> None:
            self._bvh_process = None
            self._export_bvh_btn.setEnabled(True)
            if code == 0:
                self._status_label.setText(f"BVH exported: {Path(out_path).name}")
                QMessageBox.information(self, "Export complete",
                                        f"BVH file written to:\n{out_path}")
            else:
                output = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
                self._status_label.setText("BVH export failed.")
                QMessageBox.critical(
                    self, "Export failed",
                    f"export_bvh.py exited with code {code}.\n\n{output[-800:]}",
                )

        proc.finished.connect(_done)
        proc.start(
            sys.executable,
            [
                str(export_script),
                "--session-db", self._session_path,
                "--run-id",     self._run_id,
                "--person-id",  str(self._person_id),
                "--smoothed",
                "--output",     out_path,
            ],
        )


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class RunTrackerDialog(QDialog):
    """Standalone dialog for running the tracker (accessible from pose window)."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        session_path: str,
        sequence_id: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run Tracker")
        self.setMinimumWidth(640)
        self.setMinimumHeight(500)

        self._widget = RunTrackerWidget()
        self._widget.set_session(conn, session_path)
        if sequence_id is not None:
            self._widget.preselect_sequence(sequence_id)
            self._widget.set_person_id(0)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._widget, 1)
        layout.addWidget(buttons)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _float_spin(default: float, mn: float, mx: float, decimals: int) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(mn, mx)
    spin.setDecimals(decimals)
    spin.setValue(default)
    return spin


# Windows NTSTATUS codes with the Error severity bits set (top two bits) are
# always > 2**31 -- a normal program's own exit code never reaches that range,
# so seeing one here means Windows killed the process before main() ran
# (missing DLL, bad image, etc.), not that the tracker itself failed.
_STATUS_DLL_NOT_FOUND = 0xC0000135


def _describe_windows_exit_code(exit_code: int) -> str | None:
    """Return a human-readable explanation for a Windows process-launch
    failure exit code, or None if *exit_code* looks like an ordinary exit
    code the tracker itself returned (nothing further to explain here).
    """
    if exit_code < 0x8000_0000:
        return None
    if exit_code == _STATUS_DLL_NOT_FOUND:
        return (
            "The tracker binary could not load a required DLL "
            "(boost_serialization.dll and/or yaml-cpp.dll).\n\n"
            "See CONTRIBUTING.md's \"Windows (native, MSVC)\" section — "
            "these need to be copied next to posetrak-tracker.exe, not just "
            "added to PATH (re-running setup-windows.ps1 does this)."
        )
    return (
        f"Windows terminated the tracker process before it could run "
        f"(NTSTATUS 0x{exit_code:08X}), rather than the tracker exiting "
        f"with an error of its own. This usually means a missing or "
        f"mismatched runtime dependency -- see CONTRIBUTING.md's "
        f"\"Windows (native, MSVC)\" section."
    )
