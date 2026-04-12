"""camera_registry.py — Camera model / mode / instance management dialog.

Opened from the setup wizard via the "Manage Cameras…" button.  Provides full
CRUD for camera_models, camera_modes, intrinsics_calibrations, and
camera_instances in a registry (or session-local) SQLite database.

Classes
-------
CameraRegistryWidget
    Main dialog: model/mode tree on the left, instance table on the right.
ModelDialog
    Add / edit a camera_models row.
ModeDialog
    Add / edit a camera_modes row; lists calibrations and manages the default.
CalibrationImportDialog
    Import an HDF5 calibration or enter calibration values manually.
InstanceDialog
    Add a camera_instances row.
"""

from __future__ import annotations

import datetime
import struct
import zlib
import sqlite3
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from posetrak.db.db import generate_id

# ---------------------------------------------------------------------------
# Tree item roles
# ---------------------------------------------------------------------------

_ID_ROLE = Qt.ItemDataRole.UserRole
_MODEL_ID_ROLE = Qt.ItemDataRole.UserRole + 1


# ---------------------------------------------------------------------------
# ModelDialog
# ---------------------------------------------------------------------------


class ModelDialog(QDialog):
    """Add or edit a camera_models row."""

    def __init__(self, conn: sqlite3.Connection, model_id: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn
        self._model_id = model_id

        self.setWindowTitle("Edit Camera Model" if model_id else "Add Camera Model")
        self.setMinimumWidth(380)

        self._manufacturer = QLineEdit()
        self._manufacturer.setPlaceholderText("e.g. GoPro")
        self._model_name = QLineEdit()
        self._model_name.setPlaceholderText("e.g. Hero 12 Black  (required)")
        self._sensor_size = QLineEdit()
        self._sensor_size.setPlaceholderText("e.g. 1/1.9\"  (optional)")

        self._error = QLabel()
        self._error.setStyleSheet("color: red;")
        self._error.setVisible(False)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("Manufacturer:", self._manufacturer)
        form.addRow("Model name *:", self._model_name)
        form.addRow("Sensor size:", self._sensor_size)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._error)
        layout.addWidget(buttons)

        if model_id:
            self._load(model_id)

    def _load(self, model_id: str) -> None:
        row = self._conn.execute(
            "SELECT manufacturer, model_name, sensor_size FROM camera_models WHERE id = ?",
            (model_id,),
        ).fetchone()
        if row:
            self._manufacturer.setText(row["manufacturer"] or "")
            self._model_name.setText(row["model_name"] or "")
            self._sensor_size.setText(row["sensor_size"] or "")

    def _accept(self) -> None:
        name = self._model_name.text().strip()
        if not name:
            self._error.setText("Model name is required.")
            self._error.setVisible(True)
            return
        manufacturer = self._manufacturer.text().strip() or None
        sensor = self._sensor_size.text().strip() or None
        try:
            if self._model_id:
                self._conn.execute(
                    "UPDATE camera_models SET manufacturer=?, model_name=?, sensor_size=? WHERE id=?",
                    (manufacturer, name, sensor, self._model_id),
                )
            else:
                self._model_id = generate_id()
                self._conn.execute(
                    "INSERT INTO camera_models (id, manufacturer, model_name, sensor_size) VALUES (?,?,?,?)",
                    (self._model_id, manufacturer, name, sensor),
                )
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001
            self._error.setText(str(exc))
            self._error.setVisible(True)
            return
        self.accept()

    def saved_model_id(self) -> str | None:
        return self._model_id


# ---------------------------------------------------------------------------
# CalibrationImportDialog
# ---------------------------------------------------------------------------


class CalibrationImportDialog(QDialog):
    """Import a calibration from an HDF5 file or enter values manually."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        mode_id: str,
        mode_label: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._conn = conn
        self._mode_id = mode_id
        self._saved_id: str | None = None

        self.setWindowTitle(f"Import Calibration — {mode_label}")
        self.setMinimumWidth(480)

        self._tabs = QTabWidget()

        # ---- Tab 1: HDF5 import ----
        h5_widget = QWidget()
        h5_layout = QVBoxLayout(h5_widget)

        self._h5_path = QLineEdit()
        self._h5_path.setPlaceholderText("Path to .h5 calibration file")
        self._h5_path.setReadOnly(True)
        h5_browse = QPushButton("Browse…")
        h5_browse.clicked.connect(self._browse_h5)

        path_row = QHBoxLayout()
        path_row.addWidget(self._h5_path)
        path_row.addWidget(h5_browse)

        self._h5_notes = QLineEdit()
        self._h5_notes.setPlaceholderText("Optional notes")

        h5_form = QFormLayout()
        h5_form.addRow("File:", path_row)
        h5_form.addRow("Notes:", self._h5_notes)

        h5_layout.addLayout(h5_form)
        h5_layout.addStretch()

        # ---- Tab 2: Manual entry ----
        manual_widget = QWidget()
        manual_form = QFormLayout(manual_widget)

        self._fx = QLineEdit(); self._fx.setPlaceholderText("e.g. 2800.0")
        self._fy = QLineEdit(); self._fy.setPlaceholderText("e.g. 2800.0")
        self._cx = QLineEdit(); self._cx.setPlaceholderText("e.g. 1920.0")
        self._cy = QLineEdit(); self._cy.setPlaceholderText("e.g. 1080.0")
        self._dist = QLineEdit(); self._dist.setPlaceholderText("k1 k2 p1 p2  (space-separated)")
        self._img_w = QLineEdit(); self._img_w.setPlaceholderText("e.g. 3840")
        self._img_h = QLineEdit(); self._img_h.setPlaceholderText("e.g. 2160")
        self._manual_notes = QLineEdit(); self._manual_notes.setPlaceholderText("Optional notes")

        manual_form.addRow("fx:", self._fx)
        manual_form.addRow("fy:", self._fy)
        manual_form.addRow("cx:", self._cx)
        manual_form.addRow("cy:", self._cy)
        manual_form.addRow("Dist coeffs:", self._dist)
        manual_form.addRow("Image width:", self._img_w)
        manual_form.addRow("Image height:", self._img_h)
        manual_form.addRow("Notes:", self._manual_notes)

        try:
            import h5py  # noqa: F401
            self._tabs.addTab(h5_widget, "From HDF5 file")
        except ImportError:
            h5_widget_disabled = QWidget()
            dl = QVBoxLayout(h5_widget_disabled)
            dl.addWidget(QLabel("h5py is not installed — HDF5 import unavailable.\nInstall with: pip install h5py"))
            self._tabs.addTab(h5_widget_disabled, "From HDF5 file")

        self._tabs.addTab(manual_widget, "Enter manually")

        self._error = QLabel()
        self._error.setStyleSheet("color: red;")
        self._error.setWordWrap(True)
        self._error.setVisible(False)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._tabs)
        layout.addWidget(self._error)
        layout.addWidget(buttons)

    def _browse_h5(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open calibration file", "", "HDF5 files (*.h5 *.hdf5);;All files (*)"
        )
        if path:
            self._h5_path.setText(path)

    def _accept(self) -> None:
        self._error.setVisible(False)
        if self._tabs.currentIndex() == 0:
            self._import_h5()
        else:
            self._import_manual()

    def _import_h5(self) -> None:
        path = self._h5_path.text().strip()
        if not path:
            self._error.setText("Select an HDF5 file first.")
            self._error.setVisible(True)
            return
        try:
            from posetrak.db.import_calib_h5 import import_calib_h5
            result = import_calib_h5(
                self._conn,
                Path(path),
                self._mode_id,
                notes=self._h5_notes.text().strip(),
            )
            self._saved_id = result.intrinsics_id
            self._maybe_set_default(self._saved_id)
            self.accept()
        except Exception as exc:  # noqa: BLE001
            self._error.setText(f"Import failed: {exc}")
            self._error.setVisible(True)

    def _import_manual(self) -> None:
        try:
            fx = float(self._fx.text())
            fy = float(self._fy.text())
            cx = float(self._cx.text())
            cy = float(self._cy.text())
            dist_parts = self._dist.text().strip().split()
            dist_coeffs = [float(v) for v in dist_parts] if dist_parts else []
            img_w = int(self._img_w.text()) if self._img_w.text().strip() else None
            img_h = int(self._img_h.text()) if self._img_h.text().strip() else None
        except ValueError as exc:
            self._error.setText(f"Invalid number: {exc}")
            self._error.setVisible(True)
            return

        dist_blob = struct.pack(f"{len(dist_coeffs)}d", *dist_coeffs) if dist_coeffs else None
        mapx_blob = mapy_blob = None

        if img_w and img_h and dist_coeffs:
            try:
                import cv2
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
                dist = np.array(dist_coeffs, dtype=np.float64)
                K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (img_w, img_h), 1)
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, dist, None, K_new, (img_w, img_h), cv2.CV_32FC1
                )
                mapx_blob = zlib.compress(mapx.astype(np.float32).tobytes(), level=6)
                mapy_blob = zlib.compress(mapy.astype(np.float32).tobytes(), level=6)
            except ImportError:
                pass  # cv2 not available; skip map computation

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        calib_id = generate_id()
        notes = self._manual_notes.text().strip() or None
        try:
            self._conn.execute(
                "INSERT INTO intrinsics_calibrations "
                "(id, camera_mode_id, calibrated_at, calibration_tool, distortion_model, "
                "fx, fy, cx, cy, dist_coeffs, image_width, image_height, "
                "undistort_mapx, undistort_mapy) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    calib_id, self._mode_id, now, "manual", "radtan",
                    fx, fy, cx, cy, dist_blob, img_w, img_h,
                    mapx_blob, mapy_blob,
                ),
            )
            if notes:
                self._conn.execute(
                    "UPDATE intrinsics_calibrations SET notes=? WHERE id=?", (notes, calib_id)
                )
            self._conn.commit()
            self._saved_id = calib_id
            self._maybe_set_default(calib_id)
            self.accept()
        except Exception as exc:  # noqa: BLE001
            self._error.setText(f"Database error: {exc}")
            self._error.setVisible(True)

    def _maybe_set_default(self, calib_id: str) -> None:
        """Auto-set as default if the mode currently has none."""
        row = self._conn.execute(
            "SELECT default_intrinsics_calibration_id FROM camera_modes WHERE id=?",
            (self._mode_id,),
        ).fetchone()
        if row and row[0] is None:
            self._conn.execute(
                "UPDATE camera_modes SET default_intrinsics_calibration_id=? WHERE id=?",
                (calib_id, self._mode_id),
            )
            self._conn.commit()

    def saved_calibration_id(self) -> str | None:
        return self._saved_id


# ---------------------------------------------------------------------------
# ModeDialog
# ---------------------------------------------------------------------------


class ModeDialog(QDialog):
    """Add / edit a camera mode and manage its calibrations."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        model_id: str,
        model_name: str,
        mode_id: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._conn = conn
        self._model_id = model_id
        self._mode_id = mode_id

        self.setWindowTitle(
            f"Edit Mode — {model_name}" if mode_id else f"Add Mode — {model_name}"
        )
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        # ---- Fields ----
        self._width = QLineEdit(); self._width.setPlaceholderText("e.g. 3840")
        self._height = QLineEdit(); self._height.setPlaceholderText("e.g. 2160")
        self._fps = QLineEdit(); self._fps.setPlaceholderText("e.g. 120.0")
        self._codec = QLineEdit(); self._codec.setPlaceholderText("e.g. h265  (optional)")
        self._notes = QLineEdit(); self._notes.setPlaceholderText("e.g. 4K Linear 120fps  (optional)")

        fields_form = QFormLayout()
        wh_row = QHBoxLayout()
        wh_row.addWidget(self._width)
        wh_row.addWidget(QLabel("×"))
        wh_row.addWidget(self._height)
        fields_form.addRow("Resolution:", wh_row)
        fields_form.addRow("Nominal fps:", self._fps)
        fields_form.addRow("Codec:", self._codec)
        fields_form.addRow("Notes:", self._notes)

        # ---- Calibrations list ----
        self._calib_list = QListWidget()
        self._calib_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

        calib_btns = QHBoxLayout()
        self._import_btn = QPushButton("Import calibration…")
        self._import_btn.clicked.connect(self._import_calib)
        self._set_default_btn = QPushButton("Set as default")
        self._set_default_btn.clicked.connect(self._set_default)
        self._set_default_btn.setEnabled(False)
        calib_btns.addWidget(self._import_btn)
        calib_btns.addWidget(self._set_default_btn)
        calib_btns.addStretch()

        calib_box = QGroupBox("Intrinsics calibrations")
        calib_layout = QVBoxLayout(calib_box)
        calib_layout.addWidget(self._calib_list)
        calib_layout.addLayout(calib_btns)

        self._calib_list.currentItemChanged.connect(
            lambda cur, _: self._set_default_btn.setEnabled(cur is not None)
        )

        # ---- Error + buttons ----
        self._error = QLabel()
        self._error.setStyleSheet("color: red;")
        self._error.setVisible(False)

        dialog_btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        dialog_btns.accepted.connect(self._accept)
        dialog_btns.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(fields_form)
        layout.addWidget(calib_box)
        layout.addWidget(self._error)
        layout.addWidget(dialog_btns)

        if mode_id:
            self._load(mode_id)

        self._import_btn.setEnabled(mode_id is not None)

    def _load(self, mode_id: str) -> None:
        row = self._conn.execute(
            "SELECT width_px, height_px, nominal_fps, codec, notes, "
            "default_intrinsics_calibration_id FROM camera_modes WHERE id=?",
            (mode_id,),
        ).fetchone()
        if row:
            self._width.setText(str(row["width_px"]))
            self._height.setText(str(row["height_px"]))
            self._fps.setText(str(row["nominal_fps"]))
            self._codec.setText(row["codec"] or "")
            self._notes.setText(row["notes"] or "")
            self._default_calib_id = row["default_intrinsics_calibration_id"]
        else:
            self._default_calib_id = None
        self._reload_calibrations()

    def _reload_calibrations(self) -> None:
        if not self._mode_id:
            return
        self._calib_list.clear()
        rows = self._conn.execute(
            "SELECT id, calibrated_at, rms_error, notes FROM intrinsics_calibrations "
            "WHERE camera_mode_id=? ORDER BY calibrated_at DESC",
            (self._mode_id,),
        ).fetchall()
        default_id = self._conn.execute(
            "SELECT default_intrinsics_calibration_id FROM camera_modes WHERE id=?",
            (self._mode_id,),
        ).fetchone()
        default_id = default_id[0] if default_id else None

        for row in rows:
            rms = f"rms {row['rms_error']:.3f}" if row["rms_error"] else "rms —"
            date = (row["calibrated_at"] or "")[:10]
            marker = " ●" if row["id"] == default_id else ""
            notes_part = f"  {row['notes']}" if row["notes"] else ""
            label = f"{date}  {rms}{notes_part}{marker}"
            item = QListWidgetItem(label)
            item.setData(_ID_ROLE, row["id"])
            self._calib_list.addItem(item)

    def _import_calib(self) -> None:
        mode_label = self._notes.text().strip() or f"{self._width.text()}×{self._height.text()}"
        dlg = CalibrationImportDialog(self._conn, self._mode_id, mode_label, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._reload_calibrations()

    def _set_default(self) -> None:
        item = self._calib_list.currentItem()
        if not item:
            return
        calib_id = item.data(_ID_ROLE)
        self._conn.execute(
            "UPDATE camera_modes SET default_intrinsics_calibration_id=? WHERE id=?",
            (calib_id, self._mode_id),
        )
        self._conn.commit()
        self._reload_calibrations()

    def _accept(self) -> None:
        self._error.setVisible(False)
        try:
            w = int(self._width.text())
            h = int(self._height.text())
            fps = float(self._fps.text())
        except ValueError:
            self._error.setText("Width, height, and fps must be numbers.")
            self._error.setVisible(True)
            return

        codec = self._codec.text().strip() or None
        notes = self._notes.text().strip() or None

        try:
            if self._mode_id:
                self._conn.execute(
                    "UPDATE camera_modes SET width_px=?, height_px=?, nominal_fps=?, "
                    "codec=?, notes=? WHERE id=?",
                    (w, h, fps, codec, notes, self._mode_id),
                )
            else:
                self._mode_id = generate_id()
                self._conn.execute(
                    "INSERT INTO camera_modes (id, camera_model_id, width_px, height_px, "
                    "nominal_fps, codec, notes) VALUES (?,?,?,?,?,?,?)",
                    (self._mode_id, self._model_id, w, h, fps, codec, notes),
                )
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001
            self._error.setText(str(exc))
            self._error.setVisible(True)
            return

        # Enable calibration import now that mode is saved
        self._import_btn.setEnabled(True)
        self.accept()

    def saved_mode_id(self) -> str | None:
        return self._mode_id


# ---------------------------------------------------------------------------
# InstanceDialog
# ---------------------------------------------------------------------------


class InstanceDialog(QDialog):
    """Add a camera_instances row."""

    def __init__(self, conn: sqlite3.Connection, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn
        self._saved_id: str | None = None

        self.setWindowTitle("Register Physical Camera")
        self.setMinimumWidth(380)

        self._label = QLineEdit()
        self._label.setPlaceholderText("e.g. cam1  (required)")

        self._model_combo = _ModelCombo(conn)

        self._serial = QLineEdit()
        self._serial.setPlaceholderText("Serial number (optional)")

        self._error = QLabel()
        self._error.setStyleSheet("color: red;")
        self._error.setVisible(False)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Register")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("Label *:", self._label)
        form.addRow("Camera model *:", self._model_combo)
        form.addRow("Serial number:", self._serial)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._error)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        label = self._label.text().strip()
        model_id = self._model_combo.current_model_id()
        if not label:
            self._show_error("Label is required.")
            return
        if not model_id:
            self._show_error("Select a camera model.")
            return
        serial = self._serial.text().strip() or None

        if serial:
            existing = self._conn.execute(
                "SELECT id FROM camera_instances WHERE serial_number=?", (serial,)
            ).fetchone()
            if existing:
                self._show_error(f"Serial number '{serial}' is already registered.")
                return

        self._saved_id = generate_id()
        try:
            self._conn.execute(
                "INSERT INTO camera_instances (id, camera_model_id, serial_number, label) "
                "VALUES (?,?,?,?)",
                (self._saved_id, model_id, serial, label),
            )
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001
            self._saved_id = None
            self._show_error(str(exc))
            return
        self.accept()

    def _show_error(self, msg: str) -> None:
        self._error.setText(msg)
        self._error.setVisible(True)

    def saved_instance_id(self) -> str | None:
        return self._saved_id


# ---------------------------------------------------------------------------
# _ModelCombo — reusable combo for picking a camera model
# ---------------------------------------------------------------------------


class _ModelCombo(QWidget):
    """Compact model-picker: combo box populated from camera_models."""

    def __init__(self, conn: sqlite3.Connection, parent=None) -> None:
        super().__init__(parent)
        from PySide6.QtWidgets import QComboBox
        self._combo = QComboBox()
        self._combo.setMinimumWidth(220)
        self._ids: list[str] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._combo)
        self._conn = conn
        self.refresh()

    def refresh(self) -> None:
        from PySide6.QtWidgets import QComboBox
        self._combo.clear()
        self._ids = []
        self._combo.addItem("(select model…)", None)
        rows = self._conn.execute(
            "SELECT id, manufacturer, model_name FROM camera_models ORDER BY model_name"
        ).fetchall()
        for row in rows:
            label = row["model_name"]
            if row["manufacturer"]:
                label = f"{row['manufacturer']} {label}"
            self._combo.addItem(label, row["id"])
            self._ids.append(row["id"])

    def current_model_id(self) -> str | None:
        return self._combo.currentData()

    def set_model_id(self, model_id: str) -> None:
        for i in range(self._combo.count()):
            if self._combo.itemData(i) == model_id:
                self._combo.setCurrentIndex(i)
                return


# ---------------------------------------------------------------------------
# CameraRegistryWidget
# ---------------------------------------------------------------------------


class CameraRegistryWidget(QDialog):
    """Camera model / mode / instance management dialog.

    Parameters
    ----------
    conn:
        Open SQLite connection to a registry or session database.  The widget
        writes directly to this connection; callers must ensure it stays open
        for the lifetime of the dialog.
    """

    def __init__(self, conn: sqlite3.Connection, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn

        self.setWindowTitle("Camera Registry")
        self.setMinimumSize(820, 560)

        # ---- Left pane: model / mode tree ----
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Model / Mode", "Resolution", "FPS", "Calibration"])
        self._tree.setColumnWidth(0, 240)
        self._tree.setColumnWidth(1, 110)
        self._tree.setColumnWidth(2, 60)
        self._tree.setAlternatingRowColors(True)
        self._tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.itemDoubleClicked.connect(self._on_double_click)

        self._add_model_btn = QPushButton("+ Add Model")
        self._add_model_btn.clicked.connect(self._add_model)
        self._add_mode_btn = QPushButton("+ Add Mode")
        self._add_mode_btn.clicked.connect(self._add_mode)
        self._add_mode_btn.setEnabled(False)
        self._edit_btn = QPushButton("Edit…")
        self._edit_btn.clicked.connect(self._edit_selected)
        self._edit_btn.setEnabled(False)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._delete_selected)
        self._delete_btn.setEnabled(False)
        self._import_calib_btn = QPushButton("Import calibration…")
        self._import_calib_btn.clicked.connect(self._import_calib_for_selected_mode)
        self._import_calib_btn.setEnabled(False)

        tree_btns = QHBoxLayout()
        tree_btns.addWidget(self._add_model_btn)
        tree_btns.addWidget(self._add_mode_btn)
        tree_btns.addWidget(self._edit_btn)
        tree_btns.addWidget(self._delete_btn)
        tree_btns.addWidget(self._import_calib_btn)
        tree_btns.addStretch()

        left_layout = QVBoxLayout()
        left_layout.addLayout(tree_btns)
        left_layout.addWidget(self._tree)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # ---- Right pane: camera instances ----
        self._inst_table = QTableWidget(0, 3)
        self._inst_table.setHorizontalHeaderLabels(["Label", "Model", "Serial #"])
        self._inst_table.horizontalHeader().setStretchLastSection(True)
        self._inst_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._inst_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._inst_table.setAlternatingRowColors(True)
        self._inst_table.itemSelectionChanged.connect(self._on_instance_selection_changed)

        self._add_cam_btn = QPushButton("+ Add Camera")
        self._add_cam_btn.clicked.connect(self._add_instance)
        self._del_cam_btn = QPushButton("Delete")
        self._del_cam_btn.clicked.connect(self._delete_instance)
        self._del_cam_btn.setEnabled(False)

        inst_btns = QHBoxLayout()
        inst_btns.addWidget(self._add_cam_btn)
        inst_btns.addWidget(self._del_cam_btn)
        inst_btns.addStretch()

        right_layout = QVBoxLayout()
        right_layout.addLayout(inst_btns)
        right_layout.addWidget(self._inst_table)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([500, 320])

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_row.addWidget(close_btn)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(splitter)
        main_layout.addLayout(close_row)

        self._reload()

    # ------------------------------------------------------------------
    # Reload / refresh
    # ------------------------------------------------------------------

    def _reload(self) -> None:
        self._reload_tree()
        self._reload_instances()

    def _reload_tree(self) -> None:
        self._tree.clear()
        models = self._conn.execute(
            "SELECT id, manufacturer, model_name FROM camera_models ORDER BY model_name"
        ).fetchall()
        for model in models:
            label = model["model_name"]
            if model["manufacturer"]:
                label = f"{model['manufacturer']} {label}"
            model_item = QTreeWidgetItem([label, "", "", ""])
            model_item.setData(0, _ID_ROLE, model["id"])
            model_item.setData(0, _MODEL_ID_ROLE, None)
            font = model_item.font(0)
            font.setBold(True)
            model_item.setFont(0, font)

            modes = self._conn.execute(
                "SELECT id, width_px, height_px, nominal_fps, notes, "
                "default_intrinsics_calibration_id FROM camera_modes "
                "WHERE camera_model_id=? ORDER BY width_px DESC, nominal_fps DESC",
                (model["id"],),
            ).fetchall()
            for mode in modes:
                fps_str = (
                    f"{mode['nominal_fps']:.0f}"
                    if mode["nominal_fps"] == int(mode["nominal_fps"])
                    else f"{mode['nominal_fps']}"
                )
                res = f"{mode['width_px']}×{mode['height_px']}"
                has_default = mode["default_intrinsics_calibration_id"] is not None
                calib_str = "✓" if has_default else "—"
                mode_label = mode["notes"] or res
                mode_item = QTreeWidgetItem([mode_label, res, fps_str, calib_str])
                mode_item.setData(0, _ID_ROLE, mode["id"])
                mode_item.setData(0, _MODEL_ID_ROLE, model["id"])
                if not has_default:
                    mode_item.setForeground(3, Qt.GlobalColor.darkRed)
                model_item.addChild(mode_item)

            self._tree.addTopLevelItem(model_item)
            model_item.setExpanded(True)

    def _reload_instances(self) -> None:
        self._inst_table.setRowCount(0)
        rows = self._conn.execute(
            "SELECT ci.id, ci.label, cm.model_name, cm.manufacturer, ci.serial_number "
            "FROM camera_instances ci "
            "JOIN camera_models cm ON cm.id = ci.camera_model_id "
            "ORDER BY ci.label",
        ).fetchall()
        for row in rows:
            r = self._inst_table.rowCount()
            self._inst_table.insertRow(r)
            label_item = QTableWidgetItem(row["label"])
            label_item.setData(_ID_ROLE, row["id"])
            model_label = row["model_name"]
            if row["manufacturer"]:
                model_label = f"{row['manufacturer']} {model_label}"
            self._inst_table.setItem(r, 0, label_item)
            self._inst_table.setItem(r, 1, QTableWidgetItem(model_label))
            self._inst_table.setItem(r, 2, QTableWidgetItem(row["serial_number"] or ""))

    # ------------------------------------------------------------------
    # Selection handlers
    # ------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        item = self._tree.currentItem()
        if item is None:
            self._add_mode_btn.setEnabled(False)
            self._edit_btn.setEnabled(False)
            self._delete_btn.setEnabled(False)
            self._import_calib_btn.setEnabled(False)
            return
        is_mode = item.data(0, _MODEL_ID_ROLE) is not None
        self._add_mode_btn.setEnabled(not is_mode)
        self._edit_btn.setEnabled(True)
        self._delete_btn.setEnabled(True)
        self._import_calib_btn.setEnabled(is_mode)

    def _on_instance_selection_changed(self) -> None:
        self._del_cam_btn.setEnabled(bool(self._inst_table.selectedItems()))

    def _on_double_click(self, item: QTreeWidgetItem, _col: int) -> None:
        self._edit_selected()

    # ------------------------------------------------------------------
    # Actions — models & modes
    # ------------------------------------------------------------------

    def _add_model(self) -> None:
        dlg = ModelDialog(self._conn, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._reload_tree()

    def _add_mode(self) -> None:
        item = self._tree.currentItem()
        if item is None:
            return
        # If a mode is selected, add to its parent model
        model_id = item.data(0, _MODEL_ID_ROLE) or item.data(0, _ID_ROLE)
        model_name = (item.parent() or item).text(0)
        dlg = ModeDialog(self._conn, model_id, model_name, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._reload_tree()

    def _edit_selected(self) -> None:
        item = self._tree.currentItem()
        if item is None:
            return
        is_mode = item.data(0, _MODEL_ID_ROLE) is not None
        row_id = item.data(0, _ID_ROLE)
        if is_mode:
            model_id = item.data(0, _MODEL_ID_ROLE)
            model_name = item.parent().text(0) if item.parent() else ""
            dlg = ModeDialog(self._conn, model_id, model_name, mode_id=row_id, parent=self)
        else:
            dlg = ModelDialog(self._conn, model_id=row_id, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._reload_tree()

    def _delete_selected(self) -> None:
        item = self._tree.currentItem()
        if item is None:
            return
        is_mode = item.data(0, _MODEL_ID_ROLE) is not None
        row_id = item.data(0, _ID_ROLE)
        name = item.text(0)

        if is_mode:
            n_calibs = self._conn.execute(
                "SELECT COUNT(*) FROM intrinsics_calibrations WHERE camera_mode_id=?",
                (row_id,),
            ).fetchone()[0]
            msg = f"Delete mode '{name}'?"
            if n_calibs:
                msg += f"\n\nThis will also delete {n_calibs} associated calibration(s)."
        else:
            n_modes = self._conn.execute(
                "SELECT COUNT(*) FROM camera_modes WHERE camera_model_id=?", (row_id,)
            ).fetchone()[0]
            n_instances = self._conn.execute(
                "SELECT COUNT(*) FROM camera_instances WHERE camera_model_id=?", (row_id,)
            ).fetchone()[0]
            msg = f"Delete model '{name}'?"
            if n_modes or n_instances:
                msg += f"\n\nThis will also delete {n_modes} mode(s) and {n_instances} instance(s)."

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            if is_mode:
                self._conn.execute(
                    "DELETE FROM intrinsics_calibrations WHERE camera_mode_id=?", (row_id,)
                )
                self._conn.execute("DELETE FROM camera_modes WHERE id=?", (row_id,))
            else:
                # cascade: calibrations → modes → instances → model
                mode_ids = [
                    r[0] for r in self._conn.execute(
                        "SELECT id FROM camera_modes WHERE camera_model_id=?", (row_id,)
                    ).fetchall()
                ]
                for mid in mode_ids:
                    self._conn.execute(
                        "DELETE FROM intrinsics_calibrations WHERE camera_mode_id=?", (mid,)
                    )
                self._conn.execute(
                    "DELETE FROM camera_modes WHERE camera_model_id=?", (row_id,)
                )
                self._conn.execute(
                    "DELETE FROM camera_instances WHERE camera_model_id=?", (row_id,)
                )
                self._conn.execute("DELETE FROM camera_models WHERE id=?", (row_id,))
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Delete Failed", str(exc))
            return

        self._reload()

    def _import_calib_for_selected_mode(self) -> None:
        item = self._tree.currentItem()
        if item is None or item.data(0, _MODEL_ID_ROLE) is None:
            return
        mode_id = item.data(0, _ID_ROLE)
        mode_label = item.text(0)
        dlg = CalibrationImportDialog(self._conn, mode_id, mode_label, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._reload_tree()

    # ------------------------------------------------------------------
    # Actions — camera instances
    # ------------------------------------------------------------------

    def _add_instance(self) -> None:
        dlg = InstanceDialog(self._conn, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._reload_instances()

    def _delete_instance(self) -> None:
        rows = self._inst_table.selectedItems()
        if not rows:
            return
        row_idx = self._inst_table.currentRow()
        label_item = self._inst_table.item(row_idx, 0)
        instance_id = label_item.data(_ID_ROLE)
        label = label_item.text()

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete camera instance '{label}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            self._conn.execute("DELETE FROM camera_instances WHERE id=?", (instance_id,))
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Delete Failed", str(exc))
            return
        self._reload_instances()
