"""Tests for app.setup.page_session (SessionPage wizard page)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.setup.page_session import SessionPage
from posetrak.db.db import SESSION_SCHEMA_VERSION, create_session, generate_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_db(tmp_path: Path) -> Path:
    """Create a minimal valid session database and return its path."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)
    session_id = generate_id()
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at, location, notes) "
        "VALUES (?, ?, ?, ?)",
        (session_id, "2026-03-01T10:00:00+00:00", "Studio", None),
    )
    conn.commit()
    conn.close()
    return db_path


def _make_wizard_mock(page: SessionPage):
    """Attach a minimal wizard mock to *page* and return it."""
    wiz = MagicMock()
    wiz.session_conn = None
    wiz.session_id   = None
    wiz.db_context   = None
    page.wizard = MagicMock(return_value=wiz)
    return wiz


# ---------------------------------------------------------------------------
# Widget construction
# ---------------------------------------------------------------------------


def test_page_constructs(qapp) -> None:
    page = SessionPage()
    assert page.title() == "Session"


def test_default_mode_is_open(qapp) -> None:
    page = SessionPage()
    assert page._rb_open.isChecked()
    assert not page._rb_create.isChecked()


def test_meta_box_hidden_in_open_mode(qapp) -> None:
    page = SessionPage()
    assert page._meta_box.isHidden()


def test_meta_box_visible_in_create_mode(qapp) -> None:
    page = SessionPage()
    page._rb_create.setChecked(True)
    assert not page._meta_box.isHidden()


# ---------------------------------------------------------------------------
# validatePage — open existing
# ---------------------------------------------------------------------------


def test_validate_opens_existing_session(qapp, tmp_path) -> None:
    db_path = _make_session_db(tmp_path)
    page = SessionPage()
    wiz = _make_wizard_mock(page)

    page._rb_open.setChecked(True)
    page._path_edit.setText(str(db_path))

    result = page.validatePage()

    assert result is True
    assert wiz.session_conn is not None
    assert wiz.session_id is not None
    assert wiz.db_context is not None
    wiz.session_conn.close()


def test_validate_fails_on_nonexistent_path(qapp, tmp_path) -> None:
    page = SessionPage()
    _make_wizard_mock(page)
    page._rb_open.setChecked(True)
    page._path_edit.setText(str(tmp_path / "missing.db"))

    result = page.validatePage()
    assert result is False
    assert not page._error_label.isHidden()


def test_validate_fails_on_wrong_schema_version(qapp, tmp_path) -> None:
    db_path = tmp_path / "bad.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    page = SessionPage()
    _make_wizard_mock(page)
    page._rb_open.setChecked(True)
    page._path_edit.setText(str(db_path))

    result = page.validatePage()
    assert result is False


# ---------------------------------------------------------------------------
# validatePage — create new
# ---------------------------------------------------------------------------


def test_validate_creates_new_session(qapp, tmp_path) -> None:
    db_path = tmp_path / "new_session.db"
    page = SessionPage()
    wiz = _make_wizard_mock(page)

    page._rb_create.setChecked(True)
    page._path_edit.setText(str(db_path))
    page._location_edit.setText("Studio A")

    result = page.validatePage()

    assert result is True
    assert db_path.exists()
    assert wiz.session_id is not None

    # Verify the session row was written
    conn = wiz.session_conn
    row = conn.execute("SELECT location FROM mocap_sessions").fetchone()
    assert row["location"] == "Studio A"
    conn.close()


def test_validate_create_fails_if_file_exists(qapp, tmp_path) -> None:
    db_path = _make_session_db(tmp_path)
    page = SessionPage()
    _make_wizard_mock(page)

    page._rb_create.setChecked(True)
    page._path_edit.setText(str(db_path))

    result = page.validatePage()
    assert result is False
    assert not page._error_label.isHidden()


# ---------------------------------------------------------------------------
# cleanupPage
# ---------------------------------------------------------------------------


def test_cleanup_closes_connection(qapp, tmp_path) -> None:
    db_path = _make_session_db(tmp_path)
    page = SessionPage()
    wiz = _make_wizard_mock(page)
    page._rb_open.setChecked(True)
    page._path_edit.setText(str(db_path))
    page.validatePage()

    conn = wiz.session_conn
    page.cleanupPage()

    # After cleanup, the wizard should have no conn
    assert wiz.session_conn is None
    # The connection should be closed (executing on it raises ProgrammingError)
    with pytest.raises(Exception):
        conn.execute("SELECT 1")


# ---------------------------------------------------------------------------
# Summary panel
# ---------------------------------------------------------------------------


def test_summary_populates_on_valid_path(qapp, tmp_path) -> None:
    db_path = _make_session_db(tmp_path)
    page = SessionPage()

    page._rb_open.setChecked(True)
    page._on_path_changed(str(db_path))

    assert not page._summary_box.isHidden()
    text = page._summary_label.text()
    assert "Studio" in text
