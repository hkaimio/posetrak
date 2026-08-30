# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for CaptureObjectsSection (marker-based-mocap design doc §7.1
sub-phase 1c): CapturePanel's "Objects" list -- add/rename/remove against a
real session DB. Mirrors test_capture_persons_section.py's approach --
private _on_*() handlers are called directly for the Qt-native-dialog
cases (QInputDialog/QMessageBox can't be driven headlessly), while _on_add
goes through the real _AddObjectDialog with only .exec() monkeypatched.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from posetrak.db.db import create_session
from posetrak.db.manage_capture_object import create_capture_object, get_capture_object, list_capture_objects
from posetrak.db.manage_marker_body import import_marker_body_str

_MARKER_BODY_YAML = """\
name: test-bokken
units: meters
markers:
  - name: hilt
    type: aruco
    dictionary: DICT_4X4_50
    id: "3"
    size: 0.05
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
"""


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


def _make_marker_body(conn: sqlite3.Connection, name: str = "Test Bokken") -> str:
    return import_marker_body_str(conn, _MARKER_BODY_YAML, name=name)


def test_empty_capture_shows_no_rows(qapp, session_db) -> None:
    from app.ui.content_panels import CaptureObjectsSection

    capture_id = _make_capture(session_db)
    section = CaptureObjectsSection(session_db, capture_id)
    assert section._list.count() == 0
    assert section._rename_btn.isEnabled() is False
    assert section._remove_btn.isEnabled() is False


def test_refresh_lists_existing_objects_sorted_by_name(qapp, session_db) -> None:
    from app.ui.content_panels import CaptureObjectsSection

    capture_id = _make_capture(session_db)
    body_id = _make_marker_body(session_db)
    create_capture_object(session_db, capture_id, "Zoe-prop", body_id)
    create_capture_object(session_db, capture_id, "Alice-prop", body_id)

    section = CaptureObjectsSection(session_db, capture_id)
    labels = [section._list.item(i).text() for i in range(section._list.count())]
    assert labels[0].startswith("Alice-prop")
    assert labels[1].startswith("Zoe-prop")


def test_refresh_shows_marker_body_name(qapp, session_db) -> None:
    from app.ui.content_panels import CaptureObjectsSection

    capture_id = _make_capture(session_db)
    body_id = _make_marker_body(session_db, name="Bokken Body")
    create_capture_object(session_db, capture_id, "bokken-A", body_id)

    section = CaptureObjectsSection(session_db, capture_id)
    assert section._list.item(0).text() == "bokken-A  —  Bokken Body"


def test_on_add_creates_object(qapp, session_db, monkeypatch) -> None:
    from app.ui.content_panels import _AddObjectDialog, CaptureObjectsSection
    from PySide6.QtWidgets import QDialog

    capture_id = _make_capture(session_db)
    body_id = _make_marker_body(session_db)
    section = CaptureObjectsSection(session_db, capture_id)

    def fake_exec(self) -> int:
        self._name_edit.setText("bokken-A")
        idx = self._body_combo.findData(body_id)
        self._body_combo.setCurrentIndex(idx)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(_AddObjectDialog, "exec", fake_exec)
    section._on_add()

    objects = list_capture_objects(session_db, capture_id)
    assert [r["name"] for r in objects] == ["bokken-A"]
    assert objects[0]["marker_body_definition_id"] == body_id
    assert section._list.count() == 1


def test_on_add_cancelled_creates_nothing(qapp, session_db, monkeypatch) -> None:
    from app.ui.content_panels import _AddObjectDialog, CaptureObjectsSection
    from PySide6.QtWidgets import QDialog

    capture_id = _make_capture(session_db)
    _make_marker_body(session_db)
    section = CaptureObjectsSection(session_db, capture_id)

    monkeypatch.setattr(
        _AddObjectDialog, "exec", lambda self: QDialog.DialogCode.Rejected
    )
    section._on_add()

    assert list_capture_objects(session_db, capture_id) == []


def test_on_rename_updates_object(qapp, session_db, monkeypatch) -> None:
    from app.ui.content_panels import CaptureObjectsSection
    from PySide6.QtWidgets import QInputDialog

    capture_id = _make_capture(session_db)
    body_id = _make_marker_body(session_db)
    object_id = create_capture_object(session_db, capture_id, "bokken-A", body_id)
    section = CaptureObjectsSection(session_db, capture_id)
    section._list.setCurrentRow(0)

    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("bokken-renamed", True))
    )
    section._on_rename()

    assert get_capture_object(session_db, object_id)["name"] == "bokken-renamed"
    assert section._list.item(0).text().startswith("bokken-renamed")


def test_on_remove_deletes_unreferenced_object(qapp, session_db, monkeypatch) -> None:
    from app.ui.content_panels import CaptureObjectsSection
    from PySide6.QtWidgets import QMessageBox

    capture_id = _make_capture(session_db)
    body_id = _make_marker_body(session_db)
    object_id = create_capture_object(session_db, capture_id, "bokken-A", body_id)
    section = CaptureObjectsSection(session_db, capture_id)
    section._list.setCurrentRow(0)

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    section._on_remove()

    assert get_capture_object(session_db, object_id) is None
    assert section._list.count() == 0


def test_on_remove_shows_error_and_keeps_object_when_referenced(
    qapp, session_db, monkeypatch
) -> None:
    from app.ui.content_panels import CaptureObjectsSection
    from PySide6.QtWidgets import QMessageBox
    from posetrak.db.db import generate_id

    capture_id = _make_capture(session_db)
    body_id = _make_marker_body(session_db)
    object_id = create_capture_object(session_db, capture_id, "bokken-A", body_id)

    sync_id = generate_id()
    session_db.execute(
        "INSERT INTO sync_configs (id, shot_id, created_by) VALUES (?, ?, 'test')",
        (sync_id, capture_id),
    )
    run_id = generate_id()
    session_db.execute(
        "INSERT INTO detection_runs "
        "(id, shot_id, sync_config_id, time_start_s, time_end_s, detector_model, "
        " pose_model, status, created_at, capture_object_id) "
        "VALUES (?, ?, ?, 0.0, 1.0, 'aruco:DICT_4X4_50', '', 'complete', '2026-01-01', ?)",
        (run_id, capture_id, sync_id, object_id),
    )
    session_db.commit()

    section = CaptureObjectsSection(session_db, capture_id)
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
    assert get_capture_object(session_db, object_id) is not None
    assert section._list.count() == 1


class TestAddObjectDialog:
    def test_body_combo_lists_every_marker_body(self, qapp) -> None:
        from app.ui.content_panels import _AddObjectDialog

        dlg = _AddObjectDialog({"body-a": "Body A", "body-b": "Body B"})
        labels = [dlg._body_combo.itemText(i) for i in range(dlg._body_combo.count())]
        assert labels == ["Body A", "Body B"]

    def test_name_and_body_readable_after_selection(self, qapp) -> None:
        from app.ui.content_panels import _AddObjectDialog

        dlg = _AddObjectDialog({"body-a": "Body A"})
        dlg._name_edit.setText("  bokken-A  ")
        dlg._body_combo.setCurrentIndex(dlg._body_combo.findData("body-a"))
        assert dlg.name() == "bokken-A"  # stripped
        assert dlg.marker_body_definition_id() == "body-a"

    def test_empty_name_rejected_without_accepting(self, qapp, monkeypatch) -> None:
        from app.ui.content_panels import _AddObjectDialog
        from PySide6.QtWidgets import QMessageBox

        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok),
        )
        dlg = _AddObjectDialog({"body-a": "Body A"})
        dlg._name_edit.setText("   ")
        dlg._on_accept()
        assert dlg.result() != dlg.DialogCode.Accepted

    def test_no_marker_bodies_disables_ok(self, qapp) -> None:
        from app.ui.content_panels import _AddObjectDialog

        dlg = _AddObjectDialog({})
        assert dlg._body_combo.isEnabled() is False
