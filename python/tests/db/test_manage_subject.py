# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for posetrak/db/manage_subject.py and the capture_subjects registry
triggers (docs/roadmap/features/subject-identity/subject-identity-design.md)."""

from __future__ import annotations

import sqlite3

import pytest

from posetrak.db.db import SESSION_SCHEMA_VERSION, create_session, open_session
from posetrak.db.manage_capture_object import (
    create_capture_object,
    delete_capture_object,
    rename_capture_object,
)
from posetrak.db.manage_person import (
    create_person,
    delete_person,
    rename_person,
    resolve_seg_mask_label,
)
from posetrak.db.manage_subject import (
    get_subject_by_name,
    label_for_subject_in_run,
    list_subjects,
    set_run_label_subject,
    subject_for_label_in_run,
)


def _capture(conn: sqlite3.Connection, capture_id: str = "cap1") -> str:
    conn.execute("INSERT OR IGNORE INTO mocap_sessions (id, recorded_at) VALUES ('s', '2026-01-01')")
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES (?, 's', 1)", (capture_id,)
    )
    conn.commit()
    return capture_id


def _seg_run(conn: sqlite3.Connection, capture_id: str, persons_json: str | None = None) -> str:
    conn.execute(
        "INSERT INTO seg_quality_runs (id, shot_id, time_start_s, time_end_s, created_at, persons_json) "
        "VALUES ('seg1', ?, 0, 1, '2026-01-01', ?)",
        (capture_id, persons_json),
    )
    conn.commit()
    return "seg1"


def _object(conn: sqlite3.Connection, capture_id: str, name: str) -> str:
    conn.execute(
        "INSERT OR IGNORE INTO marker_body_definitions (id, name, yaml_content, created_at) "
        "VALUES ('mb', 'mb', 'x', '2026-01-01')"
    )
    return create_capture_object(conn, capture_id, name, "mb")


def test_persons_and_objects_get_distinct_labels_in_creation_order(session_db) -> None:
    cap = _capture(session_db)
    a = create_person(session_db, cap, "Harri")
    b = _object(session_db, cap, "sword")
    c = create_person(session_db, cap, "Kim")
    subjects = list_subjects(session_db, cap)
    assert [(s["id"], s["kind"], s["label"]) for s in subjects] == [
        (a, "person", 1), (b, "object", 2), (c, "person", 3),
    ]


def test_labels_are_per_capture(session_db) -> None:
    cap1 = _capture(session_db, "cap1")
    session_db.execute("INSERT INTO captures (id, session_id, capture_number) VALUES ('cap2', 's', 2)")
    create_person(session_db, cap1, "A")
    create_person(session_db, "cap2", "B")
    assert list_subjects(session_db, "cap2")[0]["label"] == 1


def test_raw_insert_also_gets_a_label(session_db) -> None:
    cap = _capture(session_db)
    session_db.execute(
        "INSERT INTO capture_persons (id, capture_id, name, created_at) "
        "VALUES ('p1', ?, 'Raw', '2026-01-01')", (cap,),
    )
    assert get_subject_by_name(session_db, cap, "Raw")["label"] == 1


def test_a_person_and_an_object_cannot_share_a_name(session_db) -> None:
    cap = _capture(session_db)
    create_person(session_db, cap, "Sword")
    with pytest.raises(sqlite3.IntegrityError):
        _object(session_db, cap, "Sword")


def test_rename_follows_and_delete_retires_the_label(session_db) -> None:
    cap = _capture(session_db)
    person = create_person(session_db, cap, "Old")
    rename_person(session_db, person, "New")
    assert get_subject_by_name(session_db, cap, "New")["label"] == 1
    assert get_subject_by_name(session_db, cap, "Old") is None

    delete_person(session_db, person)
    assert list_subjects(session_db, cap) == []
    # The name is free again, but label 1 is never reused.
    create_person(session_db, cap, "New")
    assert get_subject_by_name(session_db, cap, "New")["label"] == 2
    assert [s["label"] for s in list_subjects(session_db, cap, include_deleted=True)] == [1, 2]


def test_object_rename_and_delete_are_mirrored(session_db) -> None:
    cap = _capture(session_db)
    obj = _object(session_db, cap, "ball")
    rename_capture_object(session_db, obj, "ball2")
    assert get_subject_by_name(session_db, cap, "ball2") is not None
    delete_capture_object(session_db, obj)
    assert list_subjects(session_db, cap) == []


def test_label_limit_is_255(session_db) -> None:
    cap = _capture(session_db)
    for i in range(255):
        create_person(session_db, cap, f"p{i}")
    with pytest.raises(sqlite3.IntegrityError):
        create_person(session_db, cap, "one too many")


def test_identity_is_the_default_in_a_run(session_db) -> None:
    cap = _capture(session_db)
    harri = create_person(session_db, cap, "Harri")
    sword = _object(session_db, cap, "sword")
    run = _seg_run(session_db, cap)
    assert label_for_subject_in_run(session_db, run, sword) == 2
    assert subject_for_label_in_run(session_db, run, 1)["id"] == harri
    assert subject_for_label_in_run(session_db, run, 9) is None
    assert resolve_seg_mask_label(session_db, run, "sword") == 2


def test_exception_row_remaps_and_frees_the_registry_label(session_db) -> None:
    cap = _capture(session_db)
    harri = create_person(session_db, cap, "Harri")
    sword = _object(session_db, cap, "sword")
    run = _seg_run(session_db, cap)

    # In this segmentation the sword was painted as label 3, not its registry label 2.
    set_run_label_subject(session_db, run, 3, sword)
    assert label_for_subject_in_run(session_db, run, sword) == 3
    assert subject_for_label_in_run(session_db, run, 3)["id"] == sword
    # Label 2 no longer means the sword; nobody else has it.
    assert subject_for_label_in_run(session_db, run, 2) is None
    assert resolve_seg_mask_label(session_db, run, "sword") == 3
    assert resolve_seg_mask_label(session_db, run, "Harri") == 1

    # Moving Harri to label 2 (the sword's registry label) is honoured, and
    # label 1 no longer means Harri.
    set_run_label_subject(session_db, run, 2, harri)
    assert label_for_subject_in_run(session_db, run, harri) == 2
    assert subject_for_label_in_run(session_db, run, 1) is None


def test_assigning_the_registry_label_stores_no_row(session_db) -> None:
    cap = _capture(session_db)
    harri = create_person(session_db, cap, "Harri")
    run = _seg_run(session_db, cap)
    set_run_label_subject(session_db, run, 5, harri)
    set_run_label_subject(session_db, run, 1, harri)  # back to identity
    assert session_db.execute("SELECT COUNT(*) FROM seg_quality_run_subjects").fetchone()[0] == 0


def test_explicitly_unassigned_label(session_db) -> None:
    cap = _capture(session_db)
    create_person(session_db, cap, "Harri")
    run = _seg_run(session_db, cap)
    set_run_label_subject(session_db, run, 1, None)
    assert subject_for_label_in_run(session_db, run, 1) is None


def test_subject_from_another_capture_is_rejected(session_db) -> None:
    cap = _capture(session_db)
    session_db.execute("INSERT INTO captures (id, session_id, capture_number) VALUES ('cap2', 's', 2)")
    other = create_person(session_db, "cap2", "Other")
    run = _seg_run(session_db, cap)
    with pytest.raises(ValueError):
        set_run_label_subject(session_db, run, 1, other)


def test_name_fallback_to_persons_json_for_a_subject_not_in_the_registry(session_db) -> None:
    cap = _capture(session_db)
    run = _seg_run(session_db, cap, '["Ghost"]')
    assert resolve_seg_mask_label(session_db, run, "Ghost") == 1
    assert resolve_seg_mask_label(session_db, run, "Nobody") is None


def test_migration_backfills_labels_in_sorted_name_order_and_exception_rows(tmp_path) -> None:
    """A v57 database with persons/objects and an old segmentation whose
    persons_json disagrees with the sorted-name order."""
    path = tmp_path / "s.db"
    conn = create_session(path)
    _capture(conn)
    for table in ("capture_subjects", "seg_quality_run_subjects"):
        conn.execute(f"DROP TABLE {table}")
    for trg in [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'trg_capture_%subject%'")]:
        conn.execute(f"DROP TRIGGER {trg}")
    conn.execute("INSERT INTO capture_persons (id, capture_id, name, created_at) VALUES ('p1','cap1','Harri','x')")
    conn.execute("INSERT INTO capture_persons (id, capture_id, name, created_at) VALUES ('p2','cap1','Alice','x')")
    conn.execute("INSERT OR IGNORE INTO marker_body_definitions (id, name, yaml_content, created_at) VALUES ('mb','mb','x','x')")
    conn.execute("INSERT INTO capture_objects (id, capture_id, name, marker_body_definition_id, created_at) VALUES ('o1','cap1','sword','mb','x')")
    # Old segmentation numbered Harri=1, Alice=2 (creation order, not sorted).
    conn.execute(
        "INSERT INTO seg_quality_runs (id, shot_id, time_start_s, time_end_s, created_at, persons_json) "
        "VALUES ('seg1','cap1',0,1,'x','[\"Harri\",\"Alice\"]')"
    )
    conn.execute("PRAGMA user_version = 57")
    conn.commit()
    conn.close()

    conn = open_session(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SESSION_SCHEMA_VERSION
    # Sorted-name order: Alice=1, Harri=2, sword=3 ("A" < "H" < "s").
    assert {s["name"]: s["label"] for s in list_subjects(conn, "cap1")} == {
        "Alice": 1, "Harri": 2, "sword": 3,
    }
    # The old run disagrees for Harri and Alice, so it gets exception rows.
    assert label_for_subject_in_run(conn, "seg1", "p1") == 1
    assert label_for_subject_in_run(conn, "seg1", "p2") == 2
    # The sword's label 3 needs no exception row: identity.
    assert label_for_subject_in_run(conn, "seg1", "o1") == 3
    assert resolve_seg_mask_label(conn, "seg1", "sword") == 3
