"""intrinsics_calib_dialog.py — Intrinsics calibration from video / image directory.

Opens a dialog that lets the user:
  1. Select a video file or image directory.
  2. Choose a calibration pattern (checkerboard or ChArUco board).
  3. Configure pattern parameters and distortion model.
  4. Run corner detection + calibration in a background thread.
  5. Review the RMS reprojection error and camera matrix.
  6. Save the result as a new intrinsics_calibrations row for a camera mode.

Public class
------------
IntrinsicsCalibDialog(QDialog)
    Pass ``conn``, ``mode_id``, and ``mode_label`` from the caller.
    Emits ``calibration_saved(str)`` with the new intrinsics_calibration_id.
"""
from __future__ import annotations

import datetime
import struct
import sys
import zlib
import sqlite3
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
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
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pipeline" / "calibration"))
from calibrate_intrinsics import (
    CalibrationResult,
    UndistortionMaps,
    _aruco_dicts,
    charuco_available,
    collect_sharp_frames,
    run_intrinsics_pipeline,
)

from posetrak.db.db import generate_id


# ---------------------------------------------------------------------------
# Background calibration thread
# ---------------------------------------------------------------------------


class _CalibThread(QThread):
    log_line = Signal(str)
    frames_collected = Signal(list)   # emitted after video scan, before detection
    succeeded = Signal(object, object)   # (CalibrationResult, UndistortionMaps)
    failed = Signal(str)

    def __init__(self, config: dict, preloaded_frames=None, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._preloaded_frames = preloaded_frames

    def run(self) -> None:
        def log(msg: str) -> None:
            self.log_line.emit(msg)

        try:
            config = self._config
            input_path = config["input_path"]
            frames = self._preloaded_frames

            if frames is None and input_path.is_file():
                frames = collect_sharp_frames(
                    input_path,
                    window=config["window"],
                    threshold=config["threshold"],
                    skip=config["skip"],
                    use_global_metric=config["use_global_metric"],
                    log_fn=log,
                )
                self.frames_collected.emit(list(frames))

            result, maps = run_intrinsics_pipeline(**config, preloaded_frames=frames, log_fn=log)
            self.succeeded.emit(result, maps)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Undistortion preview dialog
# ---------------------------------------------------------------------------


class _UndistortPreviewDialog(QDialog):
    """Scrub through undistorted vs distorted video frames to check calibration quality."""

    def __init__(self, video_path: str, maps: UndistortionMaps, parent=None) -> None:
        super().__init__(parent)
        self._video_path = video_path
        self._maps = maps
        self._cap = None
        self._n_frames = 0
        self._show_undistorted = True

        self.setWindowTitle(f"Undistortion Preview — {Path(video_path).name}")
        self.resize(1280, 820)
        self._build_ui()
        QTimer.singleShot(0, self._open_video)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self._frame_label = QLabel("Loading…")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setStyleSheet("background: black; color: #888;")
        self._frame_label.setMinimumHeight(400)
        root.addWidget(self._frame_label, 1)

        slider_row = QHBoxLayout()
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._frame_info = QLabel("Frame 0 / 0")
        self._frame_info.setFixedWidth(110)
        slider_row.addWidget(self._slider, 1)
        slider_row.addWidget(self._frame_info)
        root.addLayout(slider_row)

        ctrl_row = QHBoxLayout()
        self._mode_btn = QPushButton("View: Undistorted")
        self._mode_btn.clicked.connect(self._toggle_mode)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        ctrl_row.addWidget(self._mode_btn)
        ctrl_row.addStretch()
        ctrl_row.addWidget(close_btn)
        root.addLayout(ctrl_row)

        self._load_timer = QTimer(self)
        self._load_timer.setSingleShot(True)
        self._load_timer.setInterval(80)
        self._load_timer.timeout.connect(self._load_current_frame)

    def _open_video(self) -> None:
        import cv2
        self._cap = cv2.VideoCapture(self._video_path)
        if not self._cap.isOpened():
            self._frame_label.setText(f"Cannot open video:\n{self._video_path}")
            return
        self._n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._slider.setRange(0, max(0, self._n_frames - 1))
        self._frame_info.setText(f"Frame 0 / {self._n_frames - 1}")
        self._load_frame(0)

    def _on_slider_changed(self, idx: int) -> None:
        self._frame_info.setText(f"Frame {idx} / {self._n_frames - 1}")
        self._load_timer.start()

    def _toggle_mode(self) -> None:
        self._show_undistorted = not self._show_undistorted
        self._mode_btn.setText(
            "View: Undistorted" if self._show_undistorted else "View: Distorted"
        )
        self._load_current_frame()

    def _load_current_frame(self) -> None:
        self._load_frame(self._slider.value())

    def _load_frame(self, idx: int) -> None:
        import cv2
        if self._cap is None or not self._cap.isOpened():
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        if not ok:
            return

        if self._show_undistorted:
            frame = cv2.remap(frame, self._maps.mapx, self._maps.mapy, cv2.INTER_LINEAR)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        label_sz = self._frame_label.size()
        if label_sz.width() > 0 and label_sz.height() > 0:
            pixmap = pixmap.scaled(
                label_sz,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._frame_label.setPixmap(pixmap)

    def closeEvent(self, event) -> None:
        self._load_timer.stop()
        if self._cap is not None:
            import cv2
            self._cap.release()
            self._cap = None
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------


class IntrinsicsCalibDialog(QDialog):
    """Calibrate camera intrinsics from a video or image directory."""

    calibration_saved = Signal(str)   # intrinsics_calibration_id

    def __init__(
        self,
        conn: sqlite3.Connection,
        mode_id: str,
        mode_label: str,
        session_conn: sqlite3.Connection | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._conn = conn
        self._session_conn = session_conn if session_conn is not conn else None
        self._mode_id = mode_id
        self._result: CalibrationResult | None = None
        self._maps: UndistortionMaps | None = None
        self._thread: _CalibThread | None = None
        self._frame_cache: dict[str, list] = {}

        self.setWindowTitle(f"Calibrate Intrinsics — {mode_label}")
        self.setMinimumWidth(560)
        self.setMinimumHeight(580)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # ---- Input file ----
        self._input_path = QLineEdit()
        self._input_path.setReadOnly(True)
        self._input_path.setPlaceholderText("Select a video file or image directory…")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_input)

        file_row = QHBoxLayout()
        file_row.addWidget(self._input_path, 1)
        file_row.addWidget(browse_btn)

        input_box = QGroupBox("Input")
        input_layout = QFormLayout(input_box)
        input_layout.addRow("Video / directory:", file_row)

        root.addWidget(input_box)

        # ---- Pattern settings ----
        pattern_box = QGroupBox("Calibration pattern")
        pattern_form = QFormLayout(pattern_box)

        self._pattern_combo = QComboBox()
        self._pattern_combo.addItem("Checkerboard")
        if charuco_available():
            self._pattern_combo.addItem("ChArUco board")
        else:
            self._pattern_combo.addItem("ChArUco board (cv2.aruco not available)", userData="disabled")
        self._pattern_combo.currentIndexChanged.connect(self._on_pattern_changed)

        self._rows = QSpinBox(); self._rows.setRange(3, 30); self._rows.setValue(7)
        self._cols = QSpinBox(); self._cols.setRange(3, 30); self._cols.setValue(10)
        rc_row = QHBoxLayout()
        rc_row.addWidget(QLabel("Rows:")); rc_row.addWidget(self._rows)
        rc_row.addWidget(QLabel("Cols:")); rc_row.addWidget(self._cols)
        rc_row.addStretch()

        self._square_size = QDoubleSpinBox()
        self._square_size.setRange(0.001, 10.0); self._square_size.setValue(0.025)
        self._square_size.setDecimals(4); self._square_size.setSingleStep(0.005)
        self._square_size_unit = QLabel("m")

        sq_row = QHBoxLayout()
        sq_row.addWidget(self._square_size)
        sq_row.addWidget(self._square_size_unit)
        sq_row.addStretch()

        # ChArUco-specific
        self._marker_ratio = QDoubleSpinBox()
        self._marker_ratio.setRange(0.1, 0.99); self._marker_ratio.setValue(0.75)
        self._marker_ratio.setDecimals(2); self._marker_ratio.setSingleStep(0.05)
        self._marker_ratio.setToolTip("Marker side as a fraction of square size.")

        self._aruco_dict_combo = QComboBox()
        for name in _aruco_dicts():
            self._aruco_dict_combo.addItem(name)
        # DICT_4X4_100 is the default: larger markers are easier to detect at distance
        # and the 4×4 family matches the most common printed calibration boards.
        default_idx = self._aruco_dict_combo.findText("DICT_4X4_100")
        if default_idx < 0:
            default_idx = self._aruco_dict_combo.findText("DICT_4X4_50")
        if default_idx >= 0:
            self._aruco_dict_combo.setCurrentIndex(default_idx)

        self._charuco_widget = QWidget()
        charuco_form = QFormLayout(self._charuco_widget)
        charuco_form.setContentsMargins(0, 0, 0, 0)
        charuco_form.addRow("Marker size (fraction of square):", self._marker_ratio)
        charuco_form.addRow("ArUco dictionary:", self._aruco_dict_combo)

        pattern_form.addRow("Pattern type:", self._pattern_combo)
        pattern_form.addRow("Grid (rows × cols):", rc_row)
        pattern_form.addRow("Square size:", sq_row)
        pattern_form.addRow("", self._charuco_widget)

        root.addWidget(pattern_box)

        # ---- Camera model ----
        model_box = QGroupBox("Camera model")
        model_layout = QFormLayout(model_box)

        self._fisheye_cb = QCheckBox("Fisheye (equidistant) distortion model")
        self._fisheye_cb.setToolTip("Use cv2.fisheye instead of the standard radial-tangential model.")
        model_layout.addRow(self._fisheye_cb)

        self._notes = QLineEdit()
        self._notes.setPlaceholderText("Optional notes stored with the calibration")
        model_layout.addRow("Notes:", self._notes)

        root.addWidget(model_box)

        # ---- Advanced ----
        adv_box = QGroupBox("Frame selection (video only)")
        adv_form = QFormLayout(adv_box)

        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.0, 1e6); self._threshold.setValue(0.8)
        self._threshold.setDecimals(1); self._threshold.setSingleStep(1.0)
        self._threshold.setToolTip("Laplacian variance threshold for sharp-frame detection.")

        self._window = QSpinBox()
        self._window.setRange(1, 100); self._window.setValue(10)

        self._skip = QSpinBox()
        self._skip.setRange(1, 60); self._skip.setValue(1)
        self._skip.setToolTip("Analyse every Nth frame (1 = all frames).")

        adv_form.addRow("Sharpness threshold:", self._threshold)
        adv_form.addRow("Window size:", self._window)
        adv_form.addRow("Frame skip:", self._skip)

        root.addWidget(adv_box)

        # ---- Progress / log ----
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.setFixedHeight(140)
        self._log.setPlaceholderText("Calibration log will appear here…")

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # indeterminate
        self._progress_bar.setVisible(False)

        self._run_btn = QPushButton("▶  Run Calibration")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run)

        run_row = QHBoxLayout()
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._progress_bar, 1)

        root.addLayout(run_row)

        cache_row = QHBoxLayout()
        self._cache_label = QLabel()
        self._cache_label.setVisible(False)
        self._clear_cache_btn = QPushButton("Clear")
        self._clear_cache_btn.setVisible(False)
        self._clear_cache_btn.setFlat(True)
        self._clear_cache_btn.clicked.connect(self._on_clear_cache)
        cache_row.addWidget(self._cache_label)
        cache_row.addWidget(self._clear_cache_btn)
        cache_row.addStretch()
        root.addLayout(cache_row)

        root.addWidget(self._log)

        # ---- Result summary ----
        self._result_label = QLabel()
        self._result_label.setWordWrap(True)
        self._result_label.setStyleSheet("font-family: monospace;")
        self._result_label.setVisible(False)
        root.addWidget(self._result_label)

        self._preview_btn = QPushButton("Preview undistortion…")
        self._preview_btn.setEnabled(False)
        self._preview_btn.clicked.connect(self._on_preview)
        preview_row = QHBoxLayout()
        preview_row.addWidget(self._preview_btn)
        preview_row.addStretch()
        root.addLayout(preview_row)

        # ---- Dialog buttons ----
        self._btn_box = QDialogButtonBox()
        self._save_btn = self._btn_box.addButton("Save calibration", QDialogButtonBox.ButtonRole.AcceptRole)
        self._btn_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        self._save_btn.setEnabled(False)
        self._btn_box.accepted.connect(self._on_save)
        self._btn_box.rejected.connect(self.reject)

        root.addWidget(self._btn_box)

        # Initial state
        self._on_pattern_changed()
        self._input_path.textChanged.connect(self._update_run_enabled)
        self._input_path.textChanged.connect(self._update_cache_ui)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse_input(self) -> None:
        # Try video first
        path, _ = QFileDialog.getOpenFileName(
            self, "Select calibration video", "",
            "Video files (*.mp4 *.mov *.avi *.mkv *.MP4 *.MOV);;All files (*)"
        )
        if not path:
            # Fall back to directory
            path = QFileDialog.getExistingDirectory(self, "Select image directory")
        if path:
            self._input_path.setText(path)

    def _on_pattern_changed(self) -> None:
        is_charuco = self._pattern_combo.currentText().startswith("ChArUco")
        self._charuco_widget.setVisible(is_charuco)
        self._square_size_unit.setText("m" if is_charuco else "")
        if is_charuco:
            enabled = charuco_available()
            self._charuco_widget.setEnabled(enabled)
        self._fisheye_cb.setEnabled(True)
        self._fisheye_cb.setToolTip("Use cv2.fisheye instead of the standard radial-tangential model.")
        self._update_run_enabled()

    def _update_run_enabled(self) -> None:
        has_input = bool(self._input_path.text().strip())
        charuco_ok = not self._pattern_combo.currentText().startswith("ChArUco") or charuco_available()
        self._run_btn.setEnabled(has_input and charuco_ok)

    def _on_run(self) -> None:
        if self._thread and self._thread.isRunning():
            return

        self._log.clear()
        self._result_label.setVisible(False)
        self._save_btn.setEnabled(False)
        self._run_btn.setEnabled(False)
        self._progress_bar.setVisible(True)

        is_charuco = self._pattern_combo.currentText().startswith("ChArUco")
        config = {
            "input_path": Path(self._input_path.text().strip()),
            "rows": self._rows.value(),
            "cols": self._cols.value(),
            "pattern": "charuco" if is_charuco else "checkerboard",
            "square_size": self._square_size.value(),
            "marker_size_ratio": self._marker_ratio.value(),
            "aruco_dict_name": self._aruco_dict_combo.currentText(),
            "use_fisheye": self._fisheye_cb.isChecked(),
            "window": self._window.value(),
            "threshold": self._threshold.value(),
            "skip": self._skip.value(),
            "use_global_metric": False,
        }

        video_path = self._input_path.text().strip()
        preloaded: list | None = None
        if video_path and Path(video_path).is_file():
            key = str(Path(video_path).resolve())
            preloaded = self._frame_cache.get(key)
            if preloaded:
                self._append_log(f"Using {len(preloaded)} cached frames (skipping video scan)…")

        self._thread = _CalibThread(config, preloaded_frames=preloaded, parent=self)
        self._thread.log_line.connect(self._append_log)
        self._thread.frames_collected.connect(self._on_frames_collected)
        self._thread.succeeded.connect(self._on_succeeded)
        self._thread.failed.connect(self._on_failed)
        self._thread.start()

    def _append_log(self, msg: str) -> None:
        self._log.appendPlainText(msg)

    def _on_frames_collected(self, frames: list) -> None:
        video_path = self._input_path.text().strip()
        if video_path and Path(video_path).is_file():
            self._frame_cache[str(Path(video_path).resolve())] = frames
            self._update_cache_ui()

    def _update_cache_ui(self) -> None:
        video_path = self._input_path.text().strip()
        if not video_path or not Path(video_path).is_file():
            self._cache_label.setVisible(False)
            self._clear_cache_btn.setVisible(False)
            return
        key = str(Path(video_path).resolve())
        cached = self._frame_cache.get(key)
        if cached:
            self._cache_label.setText(f"{len(cached)} frames cached")
            self._cache_label.setVisible(True)
            self._clear_cache_btn.setVisible(True)
        else:
            self._cache_label.setVisible(False)
            self._clear_cache_btn.setVisible(False)

    def _on_clear_cache(self) -> None:
        video_path = self._input_path.text().strip()
        if video_path and Path(video_path).is_file():
            self._frame_cache.pop(str(Path(video_path).resolve()), None)
        self._update_cache_ui()

    def _on_succeeded(self, result: CalibrationResult, maps: UndistortionMaps) -> None:
        self._result = result
        self._maps = maps
        self._progress_bar.setVisible(False)
        self._run_btn.setEnabled(True)
        self._save_btn.setEnabled(True)

        video_path = self._input_path.text().strip()
        self._preview_btn.setEnabled(bool(video_path) and Path(video_path).is_file())

        K = result.matrix_undistorted
        d = result.distortion.flatten()
        w, h = result.size
        lines = [
            f"<b>RMS error: {result.error:.4f} px</b>  ({result.model_type})",
            f"Resolution: {w} × {h}",
            f"fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}",
            f"dist: {' '.join(f'{v:.4f}' for v in d)}",
        ]
        self._result_label.setText("<br>".join(lines))
        self._result_label.setVisible(True)

    def _on_preview(self) -> None:
        if self._maps is None:
            return
        video_path = self._input_path.text().strip()
        if not video_path or not Path(video_path).is_file():
            QMessageBox.warning(
                self, "No video", "Preview requires a video file (not an image directory)."
            )
            return
        dlg = _UndistortPreviewDialog(video_path, self._maps, parent=self)
        dlg.exec()

    def _on_failed(self, msg: str) -> None:
        self._progress_bar.setVisible(False)
        self._run_btn.setEnabled(True)
        self._append_log(f"\nERROR: {msg}")
        QMessageBox.critical(self, "Calibration failed", msg)

    def _on_save(self) -> None:
        if self._result is None:
            return
        try:
            calib_id = self._save_to_db()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.calibration_saved.emit(calib_id)
        self.accept()

    # ------------------------------------------------------------------
    # DB write
    # ------------------------------------------------------------------

    def _save_to_db(self) -> str:
        result = self._result
        maps = self._maps

        K_new = result.matrix_undistorted
        fx, fy = float(K_new[0, 0]), float(K_new[1, 1])
        cx, cy = float(K_new[0, 2]), float(K_new[1, 2])

        K_orig_blob = struct.pack("<9d", *result.matrix.flatten())
        dist_flat = result.distortion.flatten()
        dist_blob = struct.pack(f"<{len(dist_flat)}d", *dist_flat)

        mapx_blob = mapy_blob = None
        if maps is not None:
            mapx_blob = zlib.compress(maps.mapx.astype(np.float32).tobytes(), level=6)
            mapy_blob = zlib.compress(maps.mapy.astype(np.float32).tobytes(), level=6)

        is_charuco = self._pattern_combo.currentText().startswith("ChArUco")
        distortion_model = "fisheye" if result.model_type == "fisheye" else "radtan"
        tool = "auto-charuco" if is_charuco else "auto-checkerboard"
        notes = self._notes.text().strip() or None
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        w, h = result.size
        calib_id = generate_id()

        ic_args = (
            calib_id, self._mode_id, now, tool, distortion_model,
            fx, fy, cx, cy, dist_blob, float(result.error), notes,
            w, h, K_orig_blob, mapx_blob, mapy_blob,
        )
        ic_sql = (
            "INSERT INTO intrinsics_calibrations "
            "(id, camera_mode_id, calibrated_at, calibration_tool, distortion_model, "
            "fx, fy, cx, cy, dist_coeffs, rms_error, notes, "
            "image_width, image_height, matrix_original, undistort_mapx, undistort_mapy) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        )

        self._conn.execute(ic_sql, ic_args)
        self._conn.commit()

        # Auto-set as default if the mode has none yet
        row = self._conn.execute(
            "SELECT default_intrinsics_calibration_id FROM camera_modes WHERE id=?",
            (self._mode_id,),
        ).fetchone()
        set_as_default = row and row[0] is None
        if set_as_default:
            self._conn.execute(
                "UPDATE camera_modes SET default_intrinsics_calibration_id=? WHERE id=?",
                (calib_id, self._mode_id),
            )
            self._conn.commit()

        # Mirror to session DB so it stays self-contained when a separate registry is used.
        if self._session_conn is not None:
            self._mirror_to_session(calib_id, ic_sql, ic_args, set_as_default)

        return calib_id

    def _mirror_to_session(
        self,
        calib_id: str,
        ic_sql: str,
        ic_args: tuple,
        set_as_default: bool,
    ) -> None:
        sess = self._session_conn
        # Ensure the camera_modes row exists in session DB (copied from primary conn).
        mode_row = self._conn.execute(
            "SELECT id, camera_model_id, width_px, height_px, nominal_fps, codec, notes "
            "FROM camera_modes WHERE id=?",
            (self._mode_id,),
        ).fetchone()
        if mode_row:
            sess.execute(
                "INSERT OR IGNORE INTO camera_modes "
                "(id, camera_model_id, width_px, height_px, nominal_fps, codec, notes) "
                "VALUES (?,?,?,?,?,?,?)",
                tuple(mode_row),
            )

        # Write the intrinsics row (use INSERT OR IGNORE in case already present).
        sess.execute(ic_sql.replace("INSERT INTO", "INSERT OR IGNORE INTO"), ic_args)

        # Update default calibration pointer in session DB.
        if set_as_default:
            sess.execute(
                "UPDATE camera_modes SET default_intrinsics_calibration_id=? WHERE id=?",
                (calib_id, self._mode_id),
            )
        sess.commit()
