# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for app.setup.page_persons (PersonsPage wizard page).

Replaces the previous wizard's last page (SkeletonPage, an empty
session-skeleton list at wizard time) with this capture's persons roster,
reusing CapturePersonsSection -- see docs/roadmap findings from the
2026-08-22 e2e-testing follow-up.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.setup.page_persons import PersonsPage
from app.ui.content_panels import CapturePersonsSection
from posetrak.db.db import create_capture, create_mocap_session, create_session
from posetrak.db.manage_person import create_person


def _make_wizard_mock(page: PersonsPage, conn, shot_ids: list[str]):
    wiz = MagicMock()
    wiz.session_conn = conn
    wiz.new_shot_ids = shot_ids
    page.wizard = MagicMock(return_value=wiz)
    return wiz


@pytest.fixture()
def session_with_capture(tmp_path: Path):
    conn = create_session(tmp_path / "session.db")
    session_id = create_mocap_session(conn)
    capture_id = create_capture(conn, session_id, label="cap1")
    yield conn, capture_id
    conn.close()


def test_no_shot_ids_shows_placeholder(qapp) -> None:
    page = PersonsPage()
    _make_wizard_mock(page, conn=None, shot_ids=[])
    page.initializePage()
    assert page._content is None
    assert not page._placeholder.isHidden()


def test_single_capture_shows_capture_persons_section(qapp, session_with_capture) -> None:
    conn, capture_id = session_with_capture
    page = PersonsPage()
    _make_wizard_mock(page, conn, [capture_id])
    page.initializePage()
    assert isinstance(page._content, CapturePersonsSection)
    assert page._placeholder.isHidden()


def test_single_capture_section_shows_existing_persons(qapp, session_with_capture) -> None:
    conn, capture_id = session_with_capture
    create_person(conn, capture_id, "Alice")
    page = PersonsPage()
    _make_wizard_mock(page, conn, [capture_id])
    page.initializePage()
    assert page._content._list.count() == 1


def test_multiple_captures_shows_one_tab_each(qapp, session_with_capture) -> None:
    from PySide6.QtWidgets import QTabWidget

    conn, capture_id = session_with_capture
    session_id = conn.execute("SELECT session_id FROM captures WHERE id = ?", (capture_id,)).fetchone()[0]
    capture_id2 = create_capture(conn, session_id, label="cap2")

    page = PersonsPage()
    _make_wizard_mock(page, conn, [capture_id, capture_id2])
    page.initializePage()

    assert isinstance(page._content, QTabWidget)
    assert page._content.count() == 2
    assert page._content.tabText(0) == "cap1"
    assert page._content.tabText(1) == "cap2"


def test_revisiting_with_same_shot_ids_does_not_rebuild(qapp, session_with_capture) -> None:
    """Back then Next again shouldn't discard whatever the user already
    added -- only rebuild when the actual capture set changes."""
    conn, capture_id = session_with_capture
    page = PersonsPage()
    _make_wizard_mock(page, conn, [capture_id])
    page.initializePage()
    first_content = page._content

    page.initializePage()
    assert page._content is first_content


def test_is_complete_always_true(qapp) -> None:
    page = PersonsPage()
    assert page.isComplete() is True
