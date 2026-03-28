"""Shared fixtures for the app test suite."""

from __future__ import annotations

import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def qt_offscreen():
    """Force Qt to use the offscreen platform so tests run without a display."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp(qt_offscreen):
    """Session-scoped QApplication; re-uses an existing instance if present."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
