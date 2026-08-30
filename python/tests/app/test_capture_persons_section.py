# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for CapturePersonsSection (config-improvements design doc, phase 5,
D3): CapturePanel's "Persons" list -- add/rename/set-default-skeleton/remove
against a real session DB.

Qt's own static dialogs (QInputDialog/QMessageBox) can't be driven
headlessly, so tests for the handlers behind those (_on_rename,
_on_set_default_skeleton, _on_remove) call the private _on_*() methods
directly rather than going through button clicks + modal dialogs -- the
same approach test_run_tracker*.py uses. _on_add uses a real custom QDialog
(_AddPersonDialog) instead, which *can* be driven headlessly (construct it,
set its widgets, read back .name()/.default_skeleton_id()) -- see
TestAddPersonDialog below -- so its own tests monkeypatch just
_AddPersonDialog.exec to skip the modal event loop while still exercising
the real widget-to-value wiring, rather than bypassing the dialog entirely.
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
    from app.ui.content_panels import CapturePersonsSection

    capture_id = _make_capture(session_db)
    section = CapturePersonsSection(session_db, capture_id)
    assert section._list.count() == 0
    assert section._rename_btn.isEnabled() is False
    assert section._remove_btn.isEnabled() is False


def test_refresh_lists_existing_persons_sorted_by_name(qapp, session_db) -> None:
    from app.ui.content_panels import CapturePersonsSection

    capture_id = _make_capture(session_db)
    create_person(session_db, capture_id, "Zoe")
    create_person(session_db, capture_id, "Alice")

    section = CapturePersonsSection(session_db, capture_id)
    labels = [section._list.item(i).text() for i in range(section._list.count())]
    assert labels[0].startswith("Alice")
    assert labels[1].startswith("Zoe")
    assert "no default skeleton" in labels[0]


def test_refresh_shows_default_skeleton_name(qapp, session_db) -> None:
    from app.ui.content_panels import CapturePersonsSection

    capture_id = _make_capture(session_db)
    _make_skeleton(session_db, "skel-a", "Skeleton A")
    create_person(session_db, capture_id, "Alice", default_skeleton_id="skel-a")

    section = CapturePersonsSection(session_db, capture_id)
    assert section._list.item(0).text() == "Alice  —  Skeleton A"


def test_on_add_creates_person(qapp, session_db, monkeypatch) -> None:
    from app.ui.content_panels import _AddPersonDialog, CapturePersonsSection
    from PySide6.QtWidgets import QDialog

    capture_id = _make_capture(session_db)
    section = CapturePersonsSection(session_db, capture_id)

    def fake_exec(self) -> int:
        self._name_edit.setText("Bob")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(_AddPersonDialog, "exec", fake_exec)
    section._on_add()

    names = [r["name"] for r in list_persons(session_db, capture_id)]
    assert names == ["Bob"]
    assert section._list.count() == 1


def test_on_add_cancelled_creates_nothing(qapp, session_db, monkeypatch) -> None:
    from app.ui.content_panels import _AddPersonDialog, CapturePersonsSection
    from PySide6.QtWidgets import QDialog

    capture_id = _make_capture(session_db)
    section = CapturePersonsSection(session_db, capture_id)

    monkeypatch.setattr(
        _AddPersonDialog, "exec", lambda self: QDialog.DialogCode.Rejected
    )
    section._on_add()

    assert list_persons(session_db, capture_id) == []


def test_on_add_sets_default_skeleton_chosen_in_dialog(qapp, session_db, monkeypatch) -> None:
    """The Add… dialog lets the skeleton be picked in the same step as the
    name, instead of requiring a separate "Default skeleton…" click
    afterward (2026-08-22 e2e testing)."""
    from app.ui.content_panels import _AddPersonDialog, CapturePersonsSection
    from PySide6.QtWidgets import QDialog

    capture_id = _make_capture(session_db)
    _make_skeleton(session_db, "skel-a", "Skeleton A")
    section = CapturePersonsSection(session_db, capture_id)

    def fake_exec(self) -> int:
        self._name_edit.setText("Bob")
        idx = self._skeleton_combo.findData("skel-a")
        self._skeleton_combo.setCurrentIndex(idx)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(_AddPersonDialog, "exec", fake_exec)
    section._on_add()

    persons = list_persons(session_db, capture_id)
    assert persons[0]["default_skeleton_id"] == "skel-a"


def test_on_rename_updates_person(qapp, session_db, monkeypatch) -> None:
    from app.ui.content_panels import CapturePersonsSection
    from PySide6.QtWidgets import QInputDialog

    capture_id = _make_capture(session_db)
    person_id = create_person(session_db, capture_id, "Alice")
    section = CapturePersonsSection(session_db, capture_id)
    section._list.setCurrentRow(0)

    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("Alicia", True))
    )
    section._on_rename()

    assert get_person(session_db, person_id)["name"] == "Alicia"
    assert section._list.item(0).text().startswith("Alicia")


def test_on_set_default_skeleton_updates_person(qapp, session_db, monkeypatch) -> None:
    from app.ui.content_panels import CapturePersonsSection
    from PySide6.QtWidgets import QInputDialog

    capture_id = _make_capture(session_db)
    _make_skeleton(session_db, "skel-a", "Skeleton A")
    person_id = create_person(session_db, capture_id, "Alice")
    section = CapturePersonsSection(session_db, capture_id)
    section._list.setCurrentRow(0)

    monkeypatch.setattr(
        QInputDialog, "getItem", staticmethod(lambda *a, **k: ("Skeleton A", True))
    )
    section._on_set_default_skeleton()

    assert get_person(session_db, person_id)["default_skeleton_id"] == "skel-a"


def test_on_remove_deletes_unreferenced_person(qapp, session_db, monkeypatch) -> None:
    from app.ui.content_panels import CapturePersonsSection
    from PySide6.QtWidgets import QMessageBox

    capture_id = _make_capture(session_db)
    person_id = create_person(session_db, capture_id, "Alice")
    section = CapturePersonsSection(session_db, capture_id)
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
    from app.ui.content_panels import CapturePersonsSection
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

    section = CapturePersonsSection(session_db, capture_id)
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


class TestAddPersonDialog:
    def test_skeleton_combo_lists_none_plus_every_skeleton(self, qapp) -> None:
        from app.ui.content_panels import _AddPersonDialog

        dlg = _AddPersonDialog({"skel-a": "Skeleton A", "skel-b": "Skeleton B"})
        labels = [dlg._skeleton_combo.itemText(i) for i in range(dlg._skeleton_combo.count())]
        assert labels == ["(none)", "Skeleton A", "Skeleton B"]
        assert dlg._skeleton_combo.currentData() is None  # "(none)" selected by default

    def test_name_and_skeleton_readable_after_selection(self, qapp) -> None:
        from app.ui.content_panels import _AddPersonDialog

        dlg = _AddPersonDialog({"skel-a": "Skeleton A"})
        dlg._name_edit.setText("  Bob  ")
        dlg._skeleton_combo.setCurrentIndex(dlg._skeleton_combo.findData("skel-a"))
        assert dlg.name() == "Bob"  # stripped
        assert dlg.default_skeleton_id() == "skel-a"

    def test_no_skeleton_selected_reads_as_none(self, qapp) -> None:
        from app.ui.content_panels import _AddPersonDialog

        dlg = _AddPersonDialog({"skel-a": "Skeleton A"})
        dlg._name_edit.setText("Bob")
        assert dlg.default_skeleton_id() is None

    def test_empty_name_rejected_without_accepting(self, qapp, monkeypatch) -> None:
        from app.ui.content_panels import _AddPersonDialog
        from PySide6.QtWidgets import QMessageBox

        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok),
        )
        dlg = _AddPersonDialog({})
        dlg._name_edit.setText("   ")
        dlg._on_accept()
        assert dlg.result() != dlg.DialogCode.Accepted
