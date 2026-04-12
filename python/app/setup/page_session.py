"""page_session.py — Wizard page 1: open or create a session database.

The user either picks an existing ``.db`` file or provides a path for a new
one.  On *Next*, the page opens/creates the database, inserts a
``mocap_sessions`` row when creating, and stores the open connection +
``DBContext`` on the wizard for downstream pages to access.

Wizard field names written by this page
----------------------------------------
``session_db_path``  — str, absolute path to the session file
``session_id``       — str, UUID of the ``mocap_sessions`` row
``session_conn``     — sqlite3.Connection (set in validatePage; not a Qt
                        field — stored directly on the wizard object)
``db_context``       — ``DBContext`` (same caveat as above)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWizardPage,
)

from app.setup.db_context import DBContext
from posetrak.db.db import (
    REGISTRY_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION,
    create_session,
    generate_id,
    get_schema_version,
    open_registry,
    open_session,
)


class SessionPage(QWizardPage):
    """Wizard page 1 — choose / create session database."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTitle("Session")
        self.setSubTitle(
            "Open an existing session database or create a new one."
        )

        # --- mode radio buttons ---
        self._rb_open   = QRadioButton("Open existing session")
        self._rb_create = QRadioButton("Create new session")
        self._rb_open.setChecked(True)

        # --- file path ---
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Path to session .db file")
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.clicked.connect(self._browse)

        path_row = QHBoxLayout()
        path_row.addWidget(self._path_edit)
        path_row.addWidget(self._browse_btn)

        # --- new-session metadata (visible only in create mode) ---
        self._location_edit = QLineEdit()
        self._location_edit.setPlaceholderText("Studio / gym / outdoor (optional)")
        self._notes_edit = QTextEdit()
        self._notes_edit.setPlaceholderText("Optional notes about this session")
        self._notes_edit.setFixedHeight(60)

        self._meta_box = QGroupBox("New session details")
        meta_form = QFormLayout(self._meta_box)
        meta_form.addRow("Location:", self._location_edit)
        meta_form.addRow("Notes:", self._notes_edit)

        # --- summary panel (shown when opening an existing session) ---
        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        self._summary_label.setTextFormat(Qt.TextFormat.RichText)
        self._summary_box = QGroupBox("Session summary")
        summary_layout = QVBoxLayout(self._summary_box)
        summary_layout.addWidget(self._summary_label)
        self._summary_box.setVisible(False)

        # --- error label ---
        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: red;")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)

        # --- registry DB section ---
        self._registry_path_edit = QLineEdit()
        self._registry_path_edit.setPlaceholderText(
            "Path to registry .db file (optional — shared camera catalog)"
        )
        self._registry_browse_btn = QPushButton("Browse…")
        self._registry_browse_btn.clicked.connect(self._browse_registry)
        self._registry_create_btn = QPushButton("Create new")
        self._registry_create_btn.clicked.connect(self._create_registry)

        reg_path_row = QHBoxLayout()
        reg_path_row.addWidget(self._registry_path_edit)
        reg_path_row.addWidget(self._registry_browse_btn)
        reg_path_row.addWidget(self._registry_create_btn)

        self._registry_status = QLabel()
        self._registry_status.setWordWrap(True)

        reg_box = QGroupBox("Registry database (optional)")
        reg_layout = QVBoxLayout(reg_box)
        reg_layout.addLayout(reg_path_row)
        reg_layout.addWidget(self._registry_status)

        # --- register Qt wizard field for path (drives isComplete) ---
        self.registerField("session_db_path*", self._path_edit)

        # --- layout ---
        layout = QVBoxLayout(self)
        layout.addWidget(self._rb_open)
        layout.addWidget(self._rb_create)
        layout.addLayout(path_row)
        layout.addWidget(self._meta_box)
        layout.addWidget(self._summary_box)
        layout.addWidget(reg_box)
        layout.addWidget(self._error_label)
        layout.addStretch()

        # --- connect signals ---
        self._rb_open.toggled.connect(self._update_mode)
        self._path_edit.textChanged.connect(self._on_path_changed)
        self._registry_path_edit.textChanged.connect(self._on_registry_path_changed)
        self._update_mode()

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def validatePage(self) -> bool:  # noqa: N802
        """Open / create the session DB; store conn and DBContext on wizard."""
        self._error_label.setVisible(False)
        path = Path(self._path_edit.text().strip())

        try:
            if self._rb_open.isChecked():
                conn, session_id = self._open_existing(path)
            else:
                conn, session_id = self._create_new(path)
        except Exception as exc:  # noqa: BLE001
            self._show_error(str(exc))
            return False

        # Open registry DB if a path was provided.
        registry_conn = None
        reg_path_text = self._registry_path_edit.text().strip()
        if reg_path_text:
            reg_path = Path(reg_path_text)
            try:
                registry_conn = open_registry(reg_path)
            except Exception as exc:  # noqa: BLE001
                self._show_error(f"Cannot open registry: {exc}")
                conn.close()
                return False

        # Store on the wizard object so subsequent pages can access them.
        wiz = self.wizard()
        wiz.session_conn  = conn
        wiz.session_id    = session_id
        wiz.db_context    = DBContext(conn, session_id)
        wiz.registry_conn = registry_conn
        self.setField("session_db_path", str(path))
        return True

    def cleanupPage(self) -> None:  # noqa: N802
        """Close DB connections if the user goes back."""
        wiz = self.wizard()
        conn = getattr(wiz, "session_conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
            wiz.session_conn = None
            wiz.session_id   = None
            wiz.db_context   = None
        reg_conn = getattr(wiz, "registry_conn", None)
        if reg_conn is not None:
            try:
                reg_conn.close()
            except Exception:  # noqa: BLE001
                pass
            wiz.registry_conn = None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        if self._rb_open.isChecked():
            path, _ = QFileDialog.getOpenFileName(
                self, "Open session database", "", "SQLite databases (*.db)"
            )
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "New session database", "", "SQLite databases (*.db)"
            )
        if path:
            self._path_edit.setText(path)

    def _browse_registry(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open registry database", "", "SQLite databases (*.db)"
        )
        if path:
            self._registry_path_edit.setText(path)

    def _create_registry(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Create registry database", "", "SQLite databases (*.db)"
        )
        if not path:
            return
        p = Path(path)
        if p.exists():
            QMessageBox.warning(
                self,
                "File exists",
                f"{path} already exists. Choose a different path or open it instead.",
            )
            return
        try:
            conn = open_registry(p)  # open_registry creates the file with schema
            conn.close()
            self._registry_path_edit.setText(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error creating registry", str(exc))

    def _update_mode(self) -> None:
        is_create = self._rb_create.isChecked()
        self._meta_box.setVisible(is_create)
        self._summary_box.setVisible(False)
        self._error_label.setVisible(False)
        if not is_create and self._path_edit.text().strip():
            self._on_path_changed(self._path_edit.text())

    def _on_path_changed(self, text: str) -> None:
        """Preview session summary when user types a valid existing path."""
        self._error_label.setVisible(False)
        self._summary_box.setVisible(False)
        if not self._rb_open.isChecked():
            return
        path = Path(text.strip())
        if not path.exists():
            return
        try:
            conn = sqlite3.connect(str(path))
            conn.row_factory = sqlite3.Row
            self._populate_summary(conn)
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    def _on_registry_path_changed(self, text: str) -> None:
        """Show registry version info when user types a valid registry path."""
        self._registry_status.setText("")
        path = Path(text.strip())
        if not path.exists():
            return
        try:
            conn = sqlite3.connect(str(path))
            conn.row_factory = sqlite3.Row
            ver = get_schema_version(conn)
            conn.close()
            if ver == REGISTRY_SCHEMA_VERSION:
                self._registry_status.setText(
                    f"Registry v{ver} — OK"
                )
                self._registry_status.setStyleSheet("color: green;")
            else:
                self._registry_status.setText(
                    f"Registry schema v{ver} (will be migrated to v{REGISTRY_SCHEMA_VERSION})"
                )
                self._registry_status.setStyleSheet("color: orange;")
        except Exception:  # noqa: BLE001
            self._registry_status.setText("Not a valid registry database.")
            self._registry_status.setStyleSheet("color: red;")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _open_existing(self, path: Path) -> tuple[sqlite3.Connection, str]:
        conn = open_session(path)
        # Read the session row (take the first one)
        row = conn.execute(
            "SELECT id FROM mocap_sessions LIMIT 1"
        ).fetchone()
        if row is None:
            conn.close()
            raise ValueError("Session database contains no sessions.")
        return conn, row["id"]

    def _create_new(self, path: Path) -> tuple[sqlite3.Connection, str]:
        if path.exists():
            raise FileExistsError(
                f"File already exists: {path}\n"
                "Choose a new path or use 'Open existing session'."
            )
        conn = create_session(path)
        session_id = generate_id()
        now = datetime.now(timezone.utc).isoformat()
        location = self._location_edit.text().strip() or None
        notes = self._notes_edit.toPlainText().strip() or None
        conn.execute(
            "INSERT INTO mocap_sessions (id, recorded_at, location, notes) "
            "VALUES (?, ?, ?, ?)",
            (session_id, now, location, notes),
        )
        conn.commit()
        return conn, session_id

    def _populate_summary(self, conn: sqlite3.Connection) -> None:
        try:
            ver = get_schema_version(conn)
            if ver != SESSION_SCHEMA_VERSION:
                self._show_error(
                    f"Incompatible schema version {ver} "
                    f"(expected {SESSION_SCHEMA_VERSION})."
                )
                return
            row = conn.execute(
                "SELECT recorded_at, location, notes FROM mocap_sessions LIMIT 1"
            ).fetchone()
            if row is None:
                return
            shot_count = conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
            parts = [
                f"<b>Recorded:</b> {row['recorded_at']}",
                f"<b>Location:</b> {row['location'] or '—'}",
                f"<b>Shots:</b> {shot_count}",
            ]
            if row["notes"]:
                parts.append(f"<b>Notes:</b> {row['notes']}")
            self._summary_label.setText("<br>".join(parts))
            self._summary_box.setVisible(True)
        except Exception:  # noqa: BLE001
            pass

    def _show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._error_label.setVisible(True)
