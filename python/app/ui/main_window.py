"""main_window.py — Shell window for the unified posetrak-ui application."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSettings
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QStatusBar,
    QWidget,
)

from posetrak.db.db import (
    DEFAULT_REGISTRY_PATH,
    create_registry,
    create_session,
    open_registry,
    open_session,
)

_MAX_RECENT = 8
_SETTINGS_ORG = "posetrak"
_SETTINGS_APP = "posetrak-ui"


class MainWindow(QMainWindow):
    """Top-level application window.

    Owns the registry connection (always open) and the current session
    connection (None until the user opens or creates a session DB).
    The central widget is a placeholder that T3.3 will replace with the
    session tree + content panel.
    """

    def __init__(self, registry_conn: sqlite3.Connection) -> None:
        super().__init__()
        self.setWindowTitle("posetrak")
        self.resize(1200, 800)

        self._registry_conn: sqlite3.Connection = registry_conn
        self._session_conn: Optional[sqlite3.Connection] = None
        self._session_path: Optional[Path] = None

        self._settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._recent: list[str] = self._load_recent()

        self._build_menu()
        self._build_status_bar()

        placeholder = QLabel("Open a session database via File → Open session database…")
        placeholder.setAlignment(placeholder.alignment())
        self.setCentralWidget(placeholder)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def registry_conn(self) -> sqlite3.Connection:
        return self._registry_conn

    @property
    def session_conn(self) -> Optional[sqlite3.Connection]:
        return self._session_conn

    @property
    def session_path(self) -> Optional[Path]:
        return self._session_path

    def open_session_file(self, path: Path) -> bool:
        """Open *path* as the current session DB. Returns True on success."""
        if not path.exists():
            QMessageBox.warning(
                self,
                "File not found",
                f"Session database not found:\n{path}",
            )
            self._remove_recent(str(path))
            return False

        try:
            conn = open_session(path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Cannot open session",
                f"Failed to open session database:\n{exc}",
            )
            return False

        self._set_session(conn, path)
        return True

    # ------------------------------------------------------------------
    # Private — menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        # ---- File menu ----
        file_menu = menu_bar.addMenu("&File")

        new_act = QAction("&New session database…", self)
        new_act.setShortcut(QKeySequence.StandardKey.New)
        new_act.triggered.connect(self._on_new_session)
        file_menu.addAction(new_act)

        open_act = QAction("&Open session database…", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_open_session)
        file_menu.addAction(open_act)

        file_menu.addSeparator()

        self._recent_menu = QMenu("Open &recent", self)
        file_menu.addMenu(self._recent_menu)
        self._rebuild_recent_menu()

        file_menu.addSeparator()

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

    def _build_status_bar(self) -> None:
        bar = QStatusBar(self)
        self.setStatusBar(bar)
        self._status_label = QLabel("No session open")
        bar.addWidget(self._status_label)

    # ------------------------------------------------------------------
    # Private — session management
    # ------------------------------------------------------------------

    def _on_new_session(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "New session database",
            str(Path.home()),
            "SQLite databases (*.db);;All files (*)",
        )
        if not path:
            return
        p = Path(path)
        if p.exists():
            if QMessageBox.question(
                self,
                "File exists",
                f"{p.name} already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            ) != QMessageBox.StandardButton.Yes:
                return
            p.unlink()
        try:
            conn = create_session(p)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot create session", str(exc))
            return
        self._set_session(conn, p)

    def _on_open_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open session database",
            str(Path.home()),
            "SQLite databases (*.db);;All files (*)",
        )
        if path:
            self.open_session_file(Path(path))

    def _set_session(self, conn: sqlite3.Connection, path: Path) -> None:
        if self._session_conn is not None:
            self._session_conn.close()
        self._session_conn = conn
        self._session_path = path
        self._status_label.setText(f"Session: {path}")
        self.setWindowTitle(f"posetrak — {path.name}")
        self._add_recent(str(path))

    # ------------------------------------------------------------------
    # Private — recent files
    # ------------------------------------------------------------------

    def _load_recent(self) -> list[str]:
        val = self._settings.value("recent_sessions", [])
        if isinstance(val, str):
            val = [val]
        return list(val)

    def _save_recent(self) -> None:
        self._settings.setValue("recent_sessions", self._recent)

    def _add_recent(self, path: str) -> None:
        if path in self._recent:
            self._recent.remove(path)
        self._recent.insert(0, path)
        self._recent = self._recent[:_MAX_RECENT]
        self._save_recent()
        self._rebuild_recent_menu()

    def _remove_recent(self, path: str) -> None:
        if path in self._recent:
            self._recent.remove(path)
            self._save_recent()
            self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.clear()
        if not self._recent:
            no_act = QAction("(none)", self)
            no_act.setEnabled(False)
            self._recent_menu.addAction(no_act)
            return
        for path_str in self._recent:
            p = Path(path_str)
            act = QAction(p.name, self)
            act.setToolTip(path_str)
            act.triggered.connect(lambda checked=False, ps=path_str: self.open_session_file(Path(ps)))
            self._recent_menu.addAction(act)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._session_conn is not None:
            self._session_conn.close()
            self._session_conn = None
        if self._registry_conn is not None:
            self._registry_conn.close()
            self._registry_conn = None
        super().closeEvent(event)


def open_or_create_registry(path: Path) -> sqlite3.Connection:
    """Open the registry at *path*, creating it (and parent dirs) if absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return open_registry(path)
    return create_registry(path)
