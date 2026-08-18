# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for _CapturePersonsSection (config-improvements design doc, phase 5,
D3): CapturePanel's "Persons" list -- add/rename/set-default-skeleton/remove
against a real session DB.

Qt dialogs (QInputDialog/QMessageBox) can't be driven headlessly, so these
tests call the private _on_*() handlers directly rather than going through
button clicks + modal dialogs -- the same approach test_run_tracker*.py uses
for anything behind a QInputDialog. That covers the actual DB-mutating logic;
the "click Add… and see the text prompt" wiring itself is a one-line
`.clicked.connect(self._on_add)` not worth a headless test.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from posetrak.db.db import create_session
from posetrak.db.manage_person import create_person, get_person, list_persons


@pytest.fixture()
def session_db(tmp_path: Path):
    conn = create_session(tmp_path / "session.db")
    yield conn
    conn.close()


def _make_capture(conn: sqlite3.Connection, capture_id: str = "cap1") -> str:
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES (?, 'sess1', 1)",
        (capture_id,),
    )
    conn.commit()
    return capture_id


def _make_skeleton(conn: sqlite3.Connection, skel_id: str, name: str) -> None:
    conn.execute(
        "INSERT INTO skeletons (id, name, yaml_content, created_at) "
        "VALUES (?, ?, 'name: x', '2026-01-01')",
        (skel_id, name),
    )
    conn.commit()


def test_empty_capture_shows_no_rows(qapp, session_db) -> None:
    from app.ui.content_panels import _CapturePersonsSection

    capture_id = _make_capture(session_db)
    section = _CapturePersonsSection(session_db, capture_id)
    assert section._list.count() == 0
    assert section._rename_btn.isEnabled() is False
    assert section._remove_btn.isEnabled() is False


def test_refresh_lists_existing_persons_sorted_by_name(qapp, session_db) -> None:
    from app.ui.content_panels import _CapturePersonsSection

    capture_id = _make_capture(session_db)
    create_person(session_db, capture_id, "Zoe")
    create_person(session_db, capture_id, "Alice")

    section = _CapturePersonsSection(session_db, capture_id)
    labels = [section._list.item(i).text() for i in range(section._list.count())]
    assert labels[0].startswith("Alice")
    assert labels[1].startswith("Zoe")
    assert "no default skeleton" in labels[0]


def test_refresh_shows_default_skeleton_name(qapp, session_db) -> None:
    from app.ui.content_panels import _CapturePersonsSection

    capture_id = _make_capture(session_db)
    _make_skeleton(session_db, "skel-a", "Skeleton A")
    create_person(session_db, capture_id, "Alice", default_skeleton_id="skel-a")

    section = _CapturePersonsSection(session_db, capture_id)
    assert section._list.item(0).text() == "Alice  —  Skeleton A"


def test_on_add_creates_person(qapp, session_db, monkeypatch) -> None:
    from app.ui.content_panels import _CapturePersonsSection
    from PySide6.QtWidgets import QInputDialog

    capture_id = _make_capture(session_db)
    section = _CapturePersonsSection(session_db, capture_id)

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Bob", True)))
    section._on_add()

    names = [r["name"] for r in list_persons(session_db, capture_id)]
    assert names == ["Bob"]
    assert section._list.count() == 1


def test_on_rename_updates_person(qapp, session_db, monkeypatch) -> None:
    from app.ui.content_panels import _CapturePersonsSection
    from PySide6.QtWidgets import QInputDialog

    capture_id = _make_capture(session_db)
    person_id = create_person(session_db, capture_id, "Alice")
    section = _CapturePersonsSection(session_db, capture_id)
    section._list.setCurrentRow(0)

    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("Alicia", True))
    )
    section._on_rename()

    assert get_person(session_db, person_id)["name"] == "Alicia"
    assert section._list.item(0).text().startswith("Alicia")


def test_on_set_default_skeleton_updates_person(qapp, session_db, monkeypatch) -> None:
    from app.ui.content_panels import _CapturePersonsSection
    from PySide6.QtWidgets import QInputDialog

    capture_id = _make_capture(session_db)
    _make_skeleton(session_db, "skel-a", "Skeleton A")
    person_id = create_person(session_db, capture_id, "Alice")
    section = _CapturePersonsSection(session_db, capture_id)
    section._list.setCurrentRow(0)

    monkeypatch.setattr(
        QInputDialog, "getItem", staticmethod(lambda *a, **k: ("Skeleton A", True))
    )
    section._on_set_default_skeleton()

    assert get_person(session_db, person_id)["default_skeleton_id"] == "skel-a"


def test_on_remove_deletes_unreferenced_person(qapp, session_db, monkeypatch) -> None:
    from app.ui.content_panels import _CapturePersonsSection
    from PySide6.QtWidgets import QMessageBox

    capture_id = _make_capture(session_db)
    person_id = create_person(session_db, capture_id, "Alice")
    section = _CapturePersonsSection(session_db, capture_id)
    section._list.setCurrentRow(0)

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    section._on_remove()

    assert get_person(session_db, person_id) is None
    assert section._list.count() == 0


def test_on_remove_shows_error_and_keeps_person_when_referenced(
    qapp, session_db, monkeypatch
) -> None:
    from app.ui.content_panels import _CapturePersonsSection
    from PySide6.QtWidgets import QMessageBox

    capture_id = _make_capture(session_db)
    person_id = create_person(session_db, capture_id, "Alice")
    session_db.execute(
        "INSERT INTO sync_configs (id, shot_id) VALUES ('sync1', ?)", (capture_id,)
    )
    session_db.execute(
        "INSERT INTO pose_observation_sequences "
        "(id, shot_id, sync_config_id, time_start_s, time_end_s) "
        "VALUES ('seq1', ?, 'sync1', 0.0, 1.0)",
        (capture_id,),
    )
    session_db.execute(
        "INSERT INTO sequence_persons (sequence_id, person_id, person_name, capture_person_id) "
        "VALUES ('seq1', 0, 'Alice', ?)",
        (person_id,),
    )
    session_db.commit()

    section = _CapturePersonsSection(session_db, capture_id)
    section._list.setCurrentRow(0)

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    critical_calls = []
    monkeypatch.setattr(
        QMessageBox, "critical",
        staticmethod(lambda *a, **k: critical_calls.append(a) or QMessageBox.StandardButton.Ok),
    )
    section._on_remove()

    assert len(critical_calls) == 1
    assert get_person(session_db, person_id) is not None
    assert section._list.count() == 1
