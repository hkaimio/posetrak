"""main_window.py — Shell window for the unified posetrak-ui application."""

from __future__ import annotations

import hashlib
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
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolBar,
)
from PySide6.QtCore import Qt

from posetrak.db.db import (
    DEFAULT_REGISTRY_PATH,
    create_mocap_session,
    create_registry,
    create_session,
    open_registry,
    open_session,
)
from app.ui.session_tree import SessionTreeWidget

_MAX_RECENT = 8
_SETTINGS_ORG = "posetrak"
_SETTINGS_APP = "posetrak-ui"


class MainWindow(QMainWindow):
    """Top-level application window.

    Owns the registry connection (always open) and the current session
    connection (None until the user opens or creates a session DB).
    Left pane: SessionTreeWidget. Right pane: QStackedWidget (filled by T3.4).
    """

    def __init__(self, registry_conn: sqlite3.Connection) -> None:
        super().__init__()
        self.setWindowTitle("posetrak")
        self.resize(1200, 800)

        self._registry_conn: sqlite3.Connection = registry_conn
        self._session_conn: Optional[sqlite3.Connection] = None
        self._session_path: Optional[Path] = None
        self._session_id: Optional[str] = None

        self._settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._recent: list[str] = self._load_recent()

        self._build_central()
        self._build_menu()
        self._build_toolbar()
        self._build_status_bar()

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
    # Private — layout
    # ------------------------------------------------------------------

    def _build_central(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._tree = SessionTreeWidget()
        self._tree.capture_selected.connect(self._show_capture)
        self._tree.trial_selected.connect(self._show_trial)
        self._tree.detection_run_selected.connect(self._show_detection_run)
        self._tree.person_track_selected.connect(self._show_person_track)
        self._tree.tracking_run_selected.connect(self._show_tracking_run)
        self._tree.selection_changed.connect(self._save_tree_selection)
        splitter.addWidget(self._tree)

        self._content = QStackedWidget()
        placeholder = QLabel("Open a session database to see its contents.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._content.addWidget(placeholder)
        splitter.addWidget(self._content)

        splitter.setSizes([260, 940])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

    # ------------------------------------------------------------------
    # Private — menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        bar = self.menuBar()

        # ---- File menu ----
        file_menu = bar.addMenu("&File")

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

        # ---- Cameras menu ----
        cameras_menu = bar.addMenu("&Cameras")
        manage_cam_act = QAction("Manage cameras…", self)
        manage_cam_act.triggered.connect(self._on_manage_cameras)
        cameras_menu.addAction(manage_cam_act)

        # ---- Session menu ----
        session_menu = bar.addMenu("&Session")

        self._new_capture_act = QAction("New &Capture…", self)
        self._new_capture_act.setEnabled(False)
        self._new_capture_act.triggered.connect(self._launch_capture_wizard)
        session_menu.addAction(self._new_capture_act)

        session_menu.addSeparator()

        reload_act = QAction("&Reload tree", self)
        reload_act.setShortcut("F5")
        reload_act.triggered.connect(self._tree.reload)
        session_menu.addAction(reload_act)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        open_act = QAction("Open session…", self)
        open_act.triggered.connect(self._on_open_session)
        tb.addAction(open_act)

        self._reload_act = QAction("Reload", self)
        self._reload_act.setEnabled(False)
        self._reload_act.triggered.connect(self._tree.reload)
        tb.addAction(self._reload_act)

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
            create_mocap_session(conn)
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
        row = conn.execute(
            "SELECT id FROM mocap_sessions ORDER BY recorded_at LIMIT 1"
        ).fetchone()
        self._session_id = row["id"] if row else None
        self._status_label.setText(f"Session: {path}")
        self.setWindowTitle(f"posetrak — {path.name}")
        self._add_recent(str(path))
        self._tree.load(conn)
        self._restore_tree_selection(path)
        self._reload_act.setEnabled(True)
        self._new_capture_act.setEnabled(True)
        self._show_placeholder()

    def auto_open_last_session(self) -> None:
        """Open the most-recently-used session DB on startup if one exists."""
        if self._recent:
            self.open_session_file(Path(self._recent[0]))

    def _save_tree_selection(self, _kind: str, item_id: str) -> None:
        if self._session_path is not None:
            self._settings.setValue(
                f"selection/{_path_key(self._session_path)}", item_id
            )

    def _restore_tree_selection(self, path: Path) -> None:
        item_id = self._settings.value(f"selection/{_path_key(path)}")
        if item_id:
            self._tree.restore_selection(item_id)

    def _show_placeholder(self) -> None:
        self._content.setCurrentIndex(0)

    def _on_manage_cameras(self) -> None:
        from app.setup.camera_registry import CameraRegistryWidget
        conn = self._registry_conn
        dlg = CameraRegistryWidget(conn, parent=self)
        dlg.exec()

    def _launch_capture_wizard(self) -> None:
        if self._session_conn is None or self._session_id is None:
            return
        from PySide6.QtWidgets import QWizard
        from app.setup.camera_registry import CameraRegistryWidget
        from app.setup.db_context import DBContext
        from app.setup.page_extrinsics import ExtrinsicsPage
        from app.setup.page_shots import ShotsPage
        from app.setup.page_skeleton import SkeletonPage
        from app.setup.page_sync import SyncPage

        wizard = QWizard(self)
        wizard.setWindowTitle("New Capture")
        wizard.resize(1000, 700)

        wizard.session_conn = self._session_conn
        wizard.session_id   = self._session_id
        wizard.db_context   = DBContext(
            self._session_conn, self._session_id, self._registry_conn
        )
        wizard.registry_conn = self._registry_conn
        wizard.new_shot_ids  = []

        wizard.setOption(QWizard.WizardOption.HaveCustomButton1, True)
        wizard.setButtonText(QWizard.WizardButton.CustomButton1, "Manage Cameras…")

        shots_page = ShotsPage()

        def _open_camera_registry() -> None:
            conn = wizard.registry_conn or wizard.session_conn
            dlg = CameraRegistryWidget(conn, parent=wizard)
            dlg.cameras_changed.connect(shots_page.refresh_camera_combos)
            dlg.exec()

        wizard.customButtonClicked.connect(lambda _btn: _open_camera_registry())

        wizard.addPage(shots_page)
        wizard.addPage(SyncPage())
        wizard.addPage(ExtrinsicsPage())
        wizard.addPage(SkeletonPage())

        if wizard.exec() == QWizard.DialogCode.Accepted:
            self._tree.reload()

    def _show_capture(self, capture_id: str) -> None:
        from app.ui.content_panels import CapturePanel
        panel = CapturePanel(self._session_conn, capture_id, self._session_path)
        self._swap_content(panel)

    def _show_trial(self, trial_id: str) -> None:
        from app.ui.content_panels import TrialPanel
        panel = TrialPanel(self._session_conn, trial_id)
        self._swap_content(panel)

    def _show_detection_run(self, run_id: str) -> None:
        from app.ui.content_panels import DetectionRunPanel
        panel = DetectionRunPanel(self._session_conn, run_id, self._session_path)
        self._swap_content(panel)

    def _show_person_track(self, sequence_id: str) -> None:
        from app.ui.content_panels import PersonTrackPanel
        panel = PersonTrackPanel(self._session_conn, sequence_id, self._session_path)
        self._swap_content(panel)

    def _show_tracking_run(self, run_id: str) -> None:
        from app.ui.content_panels import TrackingRunPanel
        panel = TrackingRunPanel(self._session_conn, run_id)
        self._swap_content(panel)

    def _swap_content(self, widget: QWidget) -> None:
        """Replace the content panel, removing the previous non-placeholder widget."""
        # Index 0 is always the permanent placeholder; everything else is transient.
        while self._content.count() > 1:
            old = self._content.widget(1)
            self._content.removeWidget(old)
            old.deleteLater()
        self._content.addWidget(widget)
        self._content.setCurrentIndex(1)

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
            act.triggered.connect(
                lambda checked=False, ps=path_str: self.open_session_file(Path(ps))
            )
            self._recent_menu.addAction(act)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._tree.unload()
        if self._session_conn is not None:
            self._session_conn.close()
            self._session_conn = None
        if self._registry_conn is not None:
            self._registry_conn.close()
            self._registry_conn = None
        super().closeEvent(event)


def _path_key(path: Path) -> str:
    """Stable, slash-free QSettings key fragment for a filesystem path."""
    return hashlib.sha1(str(path).encode()).hexdigest()[:16]


def open_or_create_registry(path: Path) -> sqlite3.Connection:
    """Open the registry at *path*, creating it (and parent dirs) if absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return open_registry(path)
    return create_registry(path)
