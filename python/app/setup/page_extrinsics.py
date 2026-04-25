"""page_extrinsics.py — Extrinsic calibration import.

Three public classes:

ExtrinsicsImportWidget
    Reusable core widget.  Call ``set_session(conn, session_id)`` before
    showing.  Emits ``imported(str)`` with the new calibration ID on success.

ExtrinsicsPage
    QWizardPage hosting ExtrinsicsImportWidget.  Reads conn / session_id from
    ``wizard.session_conn`` / ``wizard.session_id`` on initializePage().
    Always completable — extrinsics are optional at wizard time.

ExtrinsicsImportDialog
    Standalone QDialog wrapping ExtrinsicsImportWidget, for use from the pose
    extraction window (or any other context where the wizard is not running).
"""
from __future__ import annotations

import sqlite3
import tomllib
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from posetrak.db.import_extrinsics import import_extrinsics


# ---------------------------------------------------------------------------
# Core widget
# ---------------------------------------------------------------------------


class ExtrinsicsImportWidget(QWidget):
    """File picker + camera matching table + import button."""

    imported = Signal(str)  # extrinsic_calibration_id

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._conn: sqlite3.Connection | None = None
        self._session_id: str | None = None
        self._cam_keys: list[str] = []
        self._toml_names: dict[str, str] = {}
        self._toml_path: Path | None = None
        self._instances: list[sqlite3.Row] = []

        # ---- File row ----
        self._path_label = QLabel("No file selected.")
        self._path_label.setStyleSheet("color: grey;")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("TOML file:"))
        file_row.addWidget(self._path_label, 1)
        file_row.addWidget(browse_btn)

        # ---- Existing calibrations info ----
        self._existing_label = QLabel()
        self._existing_label.setStyleSheet("color: #555; font-size: 11px;")
        self._existing_label.setVisible(False)

        # ---- Matching table ----
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["TOML entry", "Name in file", "Session camera"])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(100)

        match_box = QGroupBox("Camera assignment")
        match_layout = QVBoxLayout(match_box)
        match_layout.addWidget(
            QLabel("Assign each TOML camera entry to a camera instance in the session.")
        )
        match_layout.addWidget(self._table)

        # ---- Status / error ----
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setVisible(False)

        # ---- Import button ----
        self._import_btn = QPushButton("Import")
        self._import_btn.setEnabled(False)
        self._import_btn.clicked.connect(self._do_import)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self._import_btn)

        # ---- Layout ----
        root = QVBoxLayout(self)
        root.addLayout(file_row)
        root.addWidget(self._existing_label)
        root.addWidget(match_box)
        root.addWidget(self._status_label)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_session(self, conn: sqlite3.Connection, session_id: str) -> None:
        """Supply session connection and ID.  Safe to call multiple times."""
        self._conn = conn
        self._session_id = session_id
        self._instances = self._load_instances()
        self._refresh_existing_label()
        self._rebuild_table()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Pose2Sim calibration TOML", "", "TOML files (*.toml);;All files (*)"
        )
        if path:
            self._load_toml(Path(path))

    def _do_import(self) -> None:
        self._set_status(None)
        if self._conn is None or self._session_id is None:
            self._set_status("No session open.", error=True)
            return

        assignment: dict[str, str] = {}
        for row_idx in range(self._table.rowCount()):
            cam_key = self._table.item(row_idx, 0).text()
            combo: QComboBox = self._table.cellWidget(row_idx, 2)
            instance_id = combo.currentData()
            if instance_id is not None:
                assignment[cam_key] = instance_id

        if not assignment:
            self._set_status("Assign at least one camera before importing.", error=True)
            return

        dupes = [iid for iid in assignment.values() if list(assignment.values()).count(iid) > 1]
        if dupes:
            self._set_status("Each session camera can only be assigned to one TOML entry.", error=True)
            return

        try:
            result = import_extrinsics(
                self._conn,
                self._session_id,
                self._toml_path,
                assignment,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Import failed: {exc}", error=True)
            return

        n = len(result.camera_instance_ids)
        msg = f"Imported {n} camera{'s' if n != 1 else ''}."
        if result.skipped:
            msg += f"  Skipped: {', '.join(sorted(result.skipped))}."

        self._refresh_existing_label()
        QMessageBox.information(self, "Import successful", msg)
        self.imported.emit(result.extrinsic_calibration_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_instances(self) -> list[sqlite3.Row]:
        if self._conn is None:
            return []
        return self._conn.execute(
            "SELECT ci.id, ci.label, ci.serial_number,"
            "       COALESCE(cm.model_name, '') AS model_name"
            " FROM camera_instances ci"
            " LEFT JOIN camera_models cm ON cm.id = ci.camera_model_id"
            " ORDER BY ci.label"
        ).fetchall()

    def _refresh_existing_label(self) -> None:
        if self._conn is None:
            self._existing_label.setVisible(False)
            return
        rows = self._conn.execute(
            "SELECT id, calibrated_at, method FROM extrinsic_calibrations"
            " ORDER BY calibrated_at DESC"
        ).fetchall()
        if not rows:
            self._existing_label.setVisible(False)
            return
        parts = [f"{r['calibrated_at']}  [{r['method'] or '?'}]  {r['id'][:8]}…" for r in rows]
        self._existing_label.setText("Existing calibrations: " + " | ".join(parts))
        self._existing_label.setVisible(True)

    def _instance_display(self, row: sqlite3.Row) -> str:
        parts: list[str] = []
        if row["model_name"]:
            parts.append(row["model_name"])
        if row["label"]:
            parts.append(row["label"])
        if row["serial_number"]:
            parts.append(f"S/N {row['serial_number']}")
        return "  —  ".join(parts) if parts else row["id"][:8]

    def _load_toml(self, path: Path) -> None:
        self._set_status(None)
        try:
            with path.open("rb") as fh:
                raw = tomllib.load(fh)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Cannot read TOML: {exc}", error=True)
            return

        cam_keys = sorted(
            (k for k in raw if k.startswith("cam") and k != "metadata"),
            key=lambda k: int(k[3:]) if k[3:].isdigit() else float("inf"),
        )
        if not cam_keys:
            self._set_status("No camera sections (cam1, cam2, …) found in TOML.", error=True)
            return

        self._toml_path = path
        self._cam_keys = cam_keys
        self._toml_names = {k: str(raw[k].get("name", "")) for k in cam_keys}

        self._path_label.setText(path.name)
        self._path_label.setToolTip(str(path))
        self._path_label.setStyleSheet("")

        self._rebuild_table()
        self._import_btn.setEnabled(True)

    def _rebuild_table(self) -> None:
        self._table.setRowCount(0)
        if not self._cam_keys:
            return

        used_ids: set[str] = set()
        for row_idx, cam_key in enumerate(self._cam_keys):
            self._table.insertRow(row_idx)

            key_item = QTableWidgetItem(cam_key)
            key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row_idx, 0, key_item)

            toml_name = self._toml_names.get(cam_key, "")
            name_item = QTableWidgetItem(toml_name if toml_name else "—")
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row_idx, 1, name_item)

            combo = QComboBox()
            combo.addItem("(unassigned)", None)
            for inst in self._instances:
                combo.addItem(self._instance_display(inst), inst["id"])

            best = self._auto_match(cam_key, toml_name, used_ids)
            if best is not None:
                idx = combo.findData(best)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                    used_ids.add(best)

            self._table.setCellWidget(row_idx, 2, combo)

    def _auto_match(self, cam_key: str, toml_name: str, used_ids: set[str]) -> str | None:
        """Return instance id that best matches this TOML entry, or None."""
        candidates = [i for i in self._instances if i["id"] not in used_ids]
        if not candidates:
            return None

        # Match by name substring (bidirectional)
        if toml_name:
            name_lower = toml_name.lower()
            matches = [
                i for i in candidates
                if name_lower in i["label"].lower() or i["label"].lower() in name_lower
            ]
            if len(matches) == 1:
                return matches[0]["id"]

        # Match cam1→first instance by position only when counts are equal
        if len(self._cam_keys) == len(self._instances):
            pos = self._cam_keys.index(cam_key)
            if pos < len(candidates):
                return candidates[pos]["id"]

        return None

    def _set_status(self, msg: str | None, *, error: bool = False) -> None:
        if msg is None:
            self._status_label.setVisible(False)
            return
        self._status_label.setText(msg)
        self._status_label.setStyleSheet("color: red;" if error else "color: green;")
        self._status_label.setVisible(True)


# ---------------------------------------------------------------------------
# Wizard page
# ---------------------------------------------------------------------------


class ExtrinsicsPage(QWizardPage):
    """Wizard page 4 — import extrinsic calibration (optional step)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTitle("Extrinsic Calibration")
        self.setSubTitle(
            "Import a Pose2Sim cameras.toml to add camera positions to the session. "
            "This step is optional — you can import extrinsics later from the pose window."
        )
        self._widget = ExtrinsicsImportWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(self._widget)

    def initializePage(self) -> None:  # noqa: N802
        wiz = self.wizard()
        conn = getattr(wiz, "session_conn", None)
        sid = getattr(wiz, "session_id", None)
        if conn is not None and sid is not None:
            self._widget.set_session(conn, sid)

    def isComplete(self) -> bool:  # noqa: N802
        return True  # extrinsics are optional; never block progression


# ---------------------------------------------------------------------------
# Standalone dialog
# ---------------------------------------------------------------------------


class ExtrinsicsImportDialog(QDialog):
    """Dialog for importing extrinsics outside the wizard (e.g. pose window)."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Extrinsic Calibration")
        self.setMinimumWidth(560)

        self._widget = ExtrinsicsImportWidget()
        self._widget.set_session(conn, session_id)
        self._widget.imported.connect(self._on_imported)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._widget)
        layout.addWidget(buttons)

    def _on_imported(self, _calib_id: str) -> None:
        pass  # widget already shows a success dialog; nothing extra needed here
