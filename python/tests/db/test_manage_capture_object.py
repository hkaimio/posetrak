# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for posetrak/db/manage_capture_object.py (design phase 1c)."""

from __future__ import annotations

import sqlite3

import pytest

from posetrak.db.manage_capture_object import (
    create_capture_object,
    delete_capture_object,
    get_capture_object,
    list_capture_objects,
    rename_capture_object,
)
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


def _make_marker_body(conn: sqlite3.Connection, yaml_content: str = _MARKER_BODY_YAML) -> str:
    return import_marker_body_str(conn, yaml_content, name="Test Bokken")


def test_create_capture_object_and_get(session_db) -> None:
    capture_id = _make_capture(session_db)
    body_id = _make_marker_body(session_db)
    object_id = create_capture_object(session_db, capture_id, "bokken-A", body_id, notes="main prop")

    row = get_capture_object(session_db, object_id)
    assert row["name"] == "bokken-A"
    assert row["capture_id"] == capture_id
    assert row["marker_body_definition_id"] == body_id
    assert row["notes"] == "main prop"


def test_create_capture_object_requires_real_marker_body(session_db) -> None:
    capture_id = _make_capture(session_db)
    with pytest.raises(sqlite3.IntegrityError):
        create_capture_object(session_db, capture_id, "bokken-A", "does-not-exist")


def test_get_capture_object_missing_returns_none(session_db) -> None:
    assert get_capture_object(session_db, "does-not-exist") is None


def test_list_capture_objects_ordered_by_name(session_db) -> None:
    capture_id = _make_capture(session_db)
    body_id = _make_marker_body(session_db)
    create_capture_object(session_db, capture_id, "Zoe-prop", body_id)
    create_capture_object(session_db, capture_id, "Alice-prop", body_id)

    names = [r["name"] for r in list_capture_objects(session_db, capture_id)]
    assert names == ["Alice-prop", "Zoe-prop"]


def test_list_capture_objects_scoped_to_capture(session_db) -> None:
    cap1 = _make_capture(session_db, "cap1")
    session_db.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES ('cap2', 'sess1', 2)"
    )
    session_db.commit()
    body_id = _make_marker_body(session_db)
    create_capture_object(session_db, cap1, "bokken-A", body_id)
    create_capture_object(session_db, "cap2", "jo-B", body_id)

    assert [r["name"] for r in list_capture_objects(session_db, cap1)] == ["bokken-A"]
    assert [r["name"] for r in list_capture_objects(session_db, "cap2")] == ["jo-B"]


def test_rename_capture_object(session_db) -> None:
    capture_id = _make_capture(session_db)
    body_id = _make_marker_body(session_db)
    object_id = create_capture_object(session_db, capture_id, "bokken-A", body_id)
    rename_capture_object(session_db, object_id, "bokken-renamed")
    assert get_capture_object(session_db, object_id)["name"] == "bokken-renamed"


def test_rename_capture_object_missing_raises(session_db) -> None:
    with pytest.raises(ValueError):
        rename_capture_object(session_db, "does-not-exist", "new-name")


def test_delete_capture_object(session_db) -> None:
    capture_id = _make_capture(session_db)
    body_id = _make_marker_body(session_db)
    object_id = create_capture_object(session_db, capture_id, "bokken-A", body_id)
    delete_capture_object(session_db, object_id)
    assert get_capture_object(session_db, object_id) is None


def test_delete_capture_object_missing_raises(session_db) -> None:
    with pytest.raises(ValueError):
        delete_capture_object(session_db, "does-not-exist")


def test_delete_capture_object_refuses_if_referenced_by_detection_run(session_db) -> None:
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

    with pytest.raises(ValueError, match="still referenced"):
        delete_capture_object(session_db, object_id)
