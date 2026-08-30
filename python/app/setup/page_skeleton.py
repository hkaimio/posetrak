# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""page_skeleton.py — Skeleton import and rough pre-run scaling.

Three public classes:

SkeletonSetupWidget
    Reusable core: skeleton list for the session, import from YAML, copy from
    registry, and optional per-segment scaling panel.  Call
    ``set_session(conn, session_id, registry_conn)`` before showing.

SkeletonPage
    QWizardPage hosting SkeletonSetupWidget.  Always completable.  No
    longer wired into either capture wizard (replaced by
    app.setup.page_persons.PersonsPage as of the 2026-08-22 e2e-testing
    follow-up -- a wizard's last page showing an empty session-skeleton
    list had nothing to actually do at wizard time).  Kept as a reusable,
    currently-unused page in case session-level skeleton setup earns a
    wizard step again later.

SkeletonSetupDialog
    Standalone QDialog for use from the pose extraction window's Session menu.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import yaml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from posetrak.db.manage_skeleton import (
    copy_skeleton_to_session,
    import_skeleton_str,
    list_skeletons,
    skeletons_with_newer_version,
)
from posetrak.db.scale_skeleton import scale_skeleton_yaml, template_measurements

# Human-readable labels for the segments scale_skeleton.py knows about.
# Keys that are not in this mapping are displayed title-cased with underscores
# replaced by spaces, so future skeleton architectures render sensibly.
_SEG_LABELS: dict[str, str] = {
    "femur":          "Femur (hip → knee)",
    "shin":           "Shin (knee → ankle)",
    "upper_arm":      "Upper arm (shoulder → elbow)",
    "lower_arm":      "Lower arm (elbow → wrist)",
    "torso_height":   "Torso height",
    "shoulder_width": "Shoulder width",
}


def _seg_label(key: str) -> str:
    return _SEG_LABELS.get(key, key.replace("_", " ").title())


def _joint_count(yaml_content: str) -> int:
    try:
        return len(yaml.safe_load(yaml_content).get("joints", []))
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------
# Registry skeleton picker (used by "Copy from registry…" button)
# ---------------------------------------------------------------------------


class _RegistryPickerDialog(QDialog):
    """Simple picker listing skeletons from a registry DB."""

    def __init__(self, registry: sqlite3.Connection, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pick Skeleton from Registry")
        self.setMinimumSize(560, 320)
        self._selected_id: str | None = None
        self._selected_yaml: str | None = None

        rows = list_skeletons(registry)

        self._table = QTableWidget(len(rows), 4)
        self._table.setHorizontalHeaderLabels(["Name", "Person label", "Joints", "Created"])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        self._rows = rows
        # Created is already its own column here, so unlike the flat combo
        # pickers elsewhere this dialog doesn't need date-based disambiguation
        # -- but it can still miss that a same-named, newer version of a
        # skeleton exists (the real bug that motivated this whole feature),
        # so flag that explicitly on the older row's Name cell.
        newer_exists = skeletons_with_newer_version(rows)
        for i, r in enumerate(rows):
            name = r["name"] or ""
            if r["id"] in newer_exists:
                name += "  (newer version exists)"
            self._table.setItem(i, 0, QTableWidgetItem(name))
            self._table.setItem(i, 1, QTableWidgetItem(r["person_label"] or ""))
            self._table.setItem(i, 2, QTableWidgetItem(str(_joint_count(r["yaml_content"]))))
            self._table.setItem(i, 3, QTableWidgetItem((r["created_at"] or "")[:10]))

        self._table.doubleClicked.connect(self.accept)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._table)
        layout.addWidget(buttons)

    def accept(self) -> None:
        rows = self._table.selectedItems()
        if not rows:
            return
        row_idx = self._table.currentRow()
        self._selected_id = self._rows[row_idx]["id"]
        self._selected_yaml = self._rows[row_idx]["yaml_content"]
        super().accept()

    def selected_id(self) -> str | None:
        return self._selected_id

    def selected_yaml(self) -> str | None:
        return self._selected_yaml

    def selected_name(self) -> str | None:
        if self._selected_id is None:
            return None
        row_idx = self._table.currentRow()
        return self._rows[row_idx]["name"] if row_idx >= 0 else None


# ---------------------------------------------------------------------------
# Core widget
# ---------------------------------------------------------------------------


class SkeletonSetupWidget(QWidget):
    """Skeleton list + import + copy + scaling panel."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._conn: sqlite3.Connection | None = None
        self._session_id: str | None = None
        self._registry_conn: sqlite3.Connection | None = None
        self._selected_yaml: str | None = None
        self._selected_id: str | None = None
        self._seg_spins: dict[str, QDoubleSpinBox] = {}

        # ---- Skeleton list ----
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Name", "Person label", "Joints", "Created"])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setMinimumHeight(120)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)

        list_box = QGroupBox("Skeletons in this session")
        list_layout = QVBoxLayout(list_box)
        list_layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._import_btn = QPushButton("Import YAML…")
        self._import_btn.clicked.connect(self._import_yaml)
        self._copy_btn = QPushButton("Copy from registry…")
        self._copy_btn.setEnabled(False)
        self._copy_btn.clicked.connect(self._copy_from_registry)
        self._rename_btn = QPushButton("Rename…")
        self._rename_btn.setEnabled(False)
        self._rename_btn.clicked.connect(self._rename_skeleton)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._delete_skeleton)
        btn_row.addWidget(self._import_btn)
        btn_row.addWidget(self._copy_btn)
        btn_row.addWidget(self._rename_btn)
        btn_row.addWidget(self._delete_btn)
        btn_row.addStretch()

        # ---- Scaling panel ----
        self._scale_box = QGroupBox("Scale selected skeleton (optional)")
        self._scale_box.setVisible(False)
        scale_outer = QVBoxLayout(self._scale_box)

        self._scale_note = QLabel(
            "Pre-fill rough body measurements to improve the first tracking run.\n"
            "Only segments that exist in the selected skeleton are shown."
        )
        self._scale_note.setWordWrap(True)
        self._scale_note.setStyleSheet("color: #555; font-size: 11px;")
        scale_outer.addWidget(self._scale_note)

        self._scale_form_widget = QWidget()
        self._scale_form = QFormLayout(self._scale_form_widget)
        self._scale_form.setContentsMargins(0, 0, 0, 0)
        scale_outer.addWidget(self._scale_form_widget)

        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("New name:"))
        self._new_name_edit = QLineEdit()
        self._new_name_edit.setPlaceholderText("e.g. harri-scaled")
        save_row.addWidget(self._new_name_edit, 1)
        self._save_btn = QPushButton("Save as new skeleton")
        self._save_btn.clicked.connect(self._save_scaled)
        save_row.addWidget(self._save_btn)
        scale_outer.addLayout(save_row)

        # ---- Status ----
        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setVisible(False)

        # ---- Layout ----
        root = QVBoxLayout(self)
        root.addWidget(list_box)
        root.addLayout(btn_row)
        root.addWidget(self._scale_box)
        root.addWidget(self._status)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_session(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        registry_conn: sqlite3.Connection | None = None,
    ) -> None:
        """Supply session connection; safe to call multiple times."""
        self._conn = conn
        self._session_id = session_id
        self._registry_conn = registry_conn
        self._copy_btn.setEnabled(registry_conn is not None)
        self._refresh_list()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _import_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open skeleton YAML", "", "YAML files (*.yaml *.yml);;All files (*)"
        )
        if not path:
            return
        self._set_status(None)
        try:
            yaml_content = Path(path).read_text(encoding="utf-8")
            # Validate it parses and has joints
            parsed = yaml.safe_load(yaml_content)
            if not parsed or "joints" not in parsed:
                self._set_status("File does not contain a 'joints' key — not a valid skeleton.", error=True)
                return
            name = Path(path).stem
            import_skeleton_str(
                self._conn,
                yaml_content,
                name=name,
                source=str(path),
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Import failed: {exc}", error=True)
            return
        self._refresh_list()
        self._select_by_content(yaml_content)

    def _copy_from_registry(self) -> None:
        if self._registry_conn is None:
            return
        self._set_status(None)
        dlg = _RegistryPickerDialog(self._registry_conn, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        skeleton_id = dlg.selected_id()
        yaml_content = dlg.selected_yaml()
        if skeleton_id is None or yaml_content is None:
            return
        try:
            copy_skeleton_to_session(self._registry_conn, self._conn, skeleton_id)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Copy failed: {exc}", error=True)
            return
        self._refresh_list()
        self._select_by_content(yaml_content)

    def _on_selection_changed(self) -> None:
        rows = self._table.selectedItems()
        if not rows:
            self._selected_id = None
            self._selected_yaml = None
            self._scale_box.setVisible(False)
            self._rename_btn.setEnabled(False)
            self._delete_btn.setEnabled(False)
            return
        row_idx = self._table.currentRow()
        self._selected_id = self._table.item(row_idx, 0).data(Qt.ItemDataRole.UserRole)
        self._selected_yaml = self._table.item(row_idx, 0).data(Qt.ItemDataRole.UserRole + 1)
        self._rename_btn.setEnabled(True)
        self._delete_btn.setEnabled(True)
        self._rebuild_scale_panel()
        default_name = (self._table.item(row_idx, 0).text() or "") + "-scaled"
        self._new_name_edit.setText(default_name)

    def _rename_skeleton(self) -> None:
        if self._selected_id is None or self._conn is None:
            return
        row_idx = self._table.currentRow()
        current_name = self._table.item(row_idx, 0).text()
        new_name, ok = QInputDialog.getText(
            self, "Rename skeleton", "New name:", text=current_name
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        self._conn.execute("UPDATE skeletons SET name = ? WHERE id = ?", (new_name, self._selected_id))
        self._conn.commit()
        self._refresh_list()
        self._select_by_name(new_name)

    def _delete_skeleton(self) -> None:
        if self._selected_id is None or self._conn is None:
            return
        row_idx = self._table.currentRow()
        name = self._table.item(row_idx, 0).text()
        reply = QMessageBox.question(
            self, "Delete skeleton",
            f"Delete '{name}'?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            self._conn.execute("DELETE FROM skeletons WHERE id = ?", (self._selected_id,))
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Delete failed", str(exc))
            return
        self._selected_id = None
        self._selected_yaml = None
        self._scale_box.setVisible(False)
        self._rename_btn.setEnabled(False)
        self._delete_btn.setEnabled(False)
        self._refresh_list()

    def _save_scaled(self) -> None:
        self._set_status(None)
        if self._selected_yaml is None:
            return
        name = self._new_name_edit.text().strip()
        if not name:
            self._set_status("Enter a name for the new skeleton.", error=True)
            return

        measurements = {
            key: spin.value() / 100.0  # cm → m
            for key, spin in self._seg_spins.items()
        }
        try:
            scaled_yaml = scale_skeleton_yaml(self._selected_yaml, measurements)
            import_skeleton_str(
                self._conn,
                scaled_yaml,
                name=name,
                parent_id=self._selected_id,
                source="scaled",
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Scaling failed: {exc}", error=True)
            return

        self._set_status(f"Saved '{name}'.", error=False)
        self._refresh_list()
        self._select_by_name(name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_list(self) -> None:
        if self._conn is None:
            return
        try:
            rows = self._conn.execute(
                "SELECT id, name, person_label, yaml_content, created_at"
                " FROM skeletons ORDER BY created_at"
            ).fetchall()
        except Exception:  # noqa: BLE001
            return

        self._table.setRowCount(0)
        for r in rows:
            row_idx = self._table.rowCount()
            self._table.insertRow(row_idx)

            name_item = QTableWidgetItem(r["name"] or "")
            name_item.setData(Qt.ItemDataRole.UserRole, r["id"])
            name_item.setData(Qt.ItemDataRole.UserRole + 1, r["yaml_content"])
            self._table.setItem(row_idx, 0, name_item)
            self._table.setItem(row_idx, 1, QTableWidgetItem(r["person_label"] or ""))
            self._table.setItem(row_idx, 2, QTableWidgetItem(str(_joint_count(r["yaml_content"]))))
            self._table.setItem(row_idx, 3, QTableWidgetItem((r["created_at"] or "")[:10]))

    def _select_by_content(self, yaml_content: str) -> None:
        for row_idx in range(self._table.rowCount()):
            stored = self._table.item(row_idx, 0).data(Qt.ItemDataRole.UserRole + 1)
            if stored == yaml_content:
                self._table.selectRow(row_idx)
                return

    def _select_by_name(self, name: str) -> None:
        for row_idx in range(self._table.rowCount()):
            if self._table.item(row_idx, 0).text() == name:
                self._table.selectRow(row_idx)
                return

    def _rebuild_scale_panel(self) -> None:
        # Clear existing rows
        while self._scale_form.rowCount() > 0:
            self._scale_form.removeRow(0)
        self._seg_spins.clear()

        if self._selected_yaml is None:
            self._scale_box.setVisible(False)
            return

        try:
            joints = yaml.safe_load(self._selected_yaml).get("joints", [])
            meas = template_measurements(joints)
        except Exception:  # noqa: BLE001
            self._scale_box.setVisible(False)
            return

        visible_segs = {k: v for k, v in meas.items() if v > 0.0}
        if not visible_segs:
            self._scale_box.setVisible(False)
            return

        for key, value_m in visible_segs.items():
            spin = QDoubleSpinBox()
            spin.setRange(1.0, 200.0)
            spin.setSingleStep(0.5)
            spin.setDecimals(1)
            spin.setSuffix(" cm")
            spin.setValue(round(value_m * 100.0, 1))
            spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            spin.setFixedWidth(100)

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(spin)
            tmpl_label = QLabel(f"  (template: {value_m * 100:.1f} cm)")
            tmpl_label.setStyleSheet("color: #777; font-size: 11px;")
            row_layout.addWidget(tmpl_label)
            row_layout.addStretch()

            self._scale_form.addRow(_seg_label(key) + ":", row_widget)
            self._seg_spins[key] = spin

        self._scale_box.setVisible(True)

    def _set_status(self, msg: str | None, *, error: bool = False) -> None:
        if msg is None:
            self._status.setVisible(False)
            return
        self._status.setText(msg)
        self._status.setStyleSheet("color: red;" if error else "color: green;")
        self._status.setVisible(True)


# ---------------------------------------------------------------------------
# Wizard page
# ---------------------------------------------------------------------------


class SkeletonPage(QWizardPage):
    """Wizard page 5 — manage skeletons for this session (optional step)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTitle("Skeletons")
        self.setSubTitle(
            "Import a skeleton YAML or copy one from the registry. "
            "Optionally scale it to approximate body measurements before the first tracking run."
        )
        self._widget = SkeletonSetupWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(self._widget)

    def initializePage(self) -> None:  # noqa: N802
        wiz = self.wizard()
        conn = getattr(wiz, "session_conn", None)
        sid = getattr(wiz, "session_id", None)
        reg = getattr(wiz, "registry_conn", None)
        if conn is not None and sid is not None:
            self._widget.set_session(conn, sid, registry_conn=reg)

    def isComplete(self) -> bool:  # noqa: N802
        return True


# ---------------------------------------------------------------------------
# Standalone dialog
# ---------------------------------------------------------------------------


class SkeletonSetupDialog(QDialog):
    """Dialog for managing skeletons outside the wizard."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        registry_conn: sqlite3.Connection | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage Skeletons")
        self.setMinimumSize(600, 480)

        self._widget = SkeletonSetupWidget()
        self._widget.set_session(conn, session_id, registry_conn=registry_conn)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._widget)
        layout.addWidget(buttons)
