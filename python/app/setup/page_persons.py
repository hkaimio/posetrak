# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""page_persons.py — Wizard page: add persons to the capture(s) just created.

Public classes
--------------
PersonsPage
    QWizardPage (the wizard's last page as of the 2026-08-22 e2e-testing
    follow-up) hosting one CapturePersonsSection per capture ShotsPage
    created (usually just one; a tab per capture when there's more than
    one). Replaces the previous last page, SkeletonPage, which showed an
    empty session-skeleton list with nothing to actually do at wizard time
    -- adding this capture's people is the thing a new session actually
    needs before it's usable, and reuses the exact widget CapturePanel
    already uses for the same job in the main window. Session-level
    skeleton import/management (SkeletonSetupWidget/SkeletonSetupDialog in
    page_skeleton.py) is still reachable from the main window afterward;
    it isn't required before adding a person, since a default skeleton is
    optional (CapturePersonsSection's own "(none)" choice).
"""
from __future__ import annotations

import sqlite3

from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget, QWizardPage

from app.ui.content_panels import CapturePersonsSection


class PersonsPage(QWizardPage):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTitle("Persons")
        self.setSubTitle(
            "Add the people tracked in this capture. Each can have a default "
            "skeleton (optional) to pre-fill later tracking setup. You can "
            "also add persons later from the main window."
        )
        self._layout = QVBoxLayout(self)
        self._placeholder = QLabel("No capture was created yet.")
        self._layout.addWidget(self._placeholder)
        self._content: QWidget | None = None
        self._built_for: list[str] = []

    def initializePage(self) -> None:  # noqa: N802
        wiz = self.wizard()
        conn: sqlite3.Connection | None = getattr(wiz, "session_conn", None)
        shot_ids: list[str] = getattr(wiz, "new_shot_ids", [])
        if conn is None or not shot_ids:
            return
        if shot_ids == self._built_for:
            return  # unchanged since last shown (e.g. Back then Next) -- keep it as-is

        if self._content is not None:
            self._layout.removeWidget(self._content)
            self._content.deleteLater()
            self._content = None
        self._built_for = list(shot_ids)
        self._placeholder.setVisible(False)

        if len(shot_ids) == 1:
            self._content = CapturePersonsSection(conn, shot_ids[0])
        else:
            tabs = QTabWidget()
            for i, shot_id in enumerate(shot_ids, start=1):
                row = conn.execute(
                    "SELECT label FROM captures WHERE id = ?", (shot_id,)
                ).fetchone()
                title = row[0] if row and row[0] else f"Capture {i}"
                tabs.addTab(CapturePersonsSection(conn, shot_id), title)
            self._content = tabs
        self._layout.addWidget(self._content)

    def isComplete(self) -> bool:  # noqa: N802
        return True
