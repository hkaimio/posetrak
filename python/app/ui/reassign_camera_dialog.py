# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Dialog for fixing which camera recorded each video of a capture.

Opened from the capture panel's "Cameras…" button. The rows are the capture's
videos (by file name); choosing another camera for a row and exchanging the
cameras of two rows is done here, never by swapping file paths, so that the
sync points stay with their videos (see ``posetrak.db.reassign_camera``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from posetrak.db.reassign_camera import (
    Assignment,
    CameraReassignError,
    reassign_cameras,
    runs_referencing_capture,
)


class ReassignCameraDialog(QDialog):
    def __init__(self, conn: sqlite3.Connection, capture_id: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reassign cameras")
        self.setMinimumWidth(560)
        self._conn = conn
        self._capture_id = capture_id
        self.changed = False

        cameras = conn.execute("SELECT id, label FROM camera_instances ORDER BY label").fetchall()
        self._videos = conn.execute(
            "SELECT id, file_path, camera_instance_id FROM capture_videos "
            "WHERE shot_id = ? ORDER BY file_path", (capture_id,)).fetchall()

        self._table = QTableWidget(len(self._videos), 2)
        self._table.setHorizontalHeaderLabels(["Video", "Camera"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._combos: list[QComboBox] = []
        for row, (_, path, cam_id) in enumerate(self._videos):
            item = QTableWidgetItem(Path(path).name)
            item.setToolTip(path)
            self._table.setItem(row, 0, item)
            combo = QComboBox()
            for cid, label in cameras:
                combo.addItem(label, cid)
            combo.setCurrentIndex(combo.findData(cam_id))
            self._table.setCellWidget(row, 1, combo)
            self._combos.append(combo)
        self._table.resizeColumnToContents(0)

        hint = QLabel(
            "Change the camera of a video, or exchange the cameras of two videos. File paths and "
            "sync points stay with their videos. Intrinsics follow the camera mode of the new camera."
        )
        hint.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(lambda: self._apply())
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(self._table, 1)
        layout.addWidget(buttons)

    def assignments(self) -> list[Assignment]:
        """The rows whose camera was changed."""
        return [
            Assignment(vid, combo.currentData())
            for (vid, _, cam_id), combo in zip(self._videos, self._combos)
            if combo.currentData() != cam_id
        ]

    def _apply(self, *, confirm_recompute: bool | None = None) -> None:
        todo = self.assignments()
        if not todo:
            return
        if confirm_recompute is None:
            existing = {k: n for k, n in runs_referencing_capture(self._conn, self._capture_id).items() if n}
            confirm_recompute = False
            if existing:
                what = ", ".join(f"{n} {k}" for k, n in existing.items())
                confirm_recompute = QMessageBox.question(
                    self, "Results exist",
                    f"This capture has {what}, which keep the old camera ids.\n\n"
                    "Reassign the cameras anyway and recompute them afterwards?",
                ) == QMessageBox.StandardButton.Yes
                if not confirm_recompute:
                    return
        try:
            warnings = reassign_cameras(self._conn, self._capture_id, todo, confirm_recompute=confirm_recompute)
        except CameraReassignError as exc:
            QMessageBox.warning(self, "Cannot reassign cameras", str(exc))
            return
        self.changed = True
        if warnings:
            QMessageBox.information(self, "Cameras reassigned", "\n\n".join(warnings))
        self.accept()
