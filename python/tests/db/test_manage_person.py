# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for posetrak/db/manage_person.py."""

from __future__ import annotations

import sqlite3

import pytest

from posetrak.db.manage_person import (
    create_person,
    delete_person,
    find_or_create_person,
    get_person,
    list_persons,
    persons_ordered_for_seg_run,
    rename_person,
    set_default_skeleton,
)


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


def test_create_person_and_get(session_db) -> None:
    capture_id = _make_capture(session_db)
    person_id = create_person(session_db, capture_id, "Alice", notes="lead performer")

    row = get_person(session_db, person_id)
    assert row["name"] == "Alice"
    assert row["capture_id"] == capture_id
    assert row["default_skeleton_id"] is None
    assert row["notes"] == "lead performer"


def test_create_person_with_default_skeleton(session_db) -> None:
    capture_id = _make_capture(session_db)
    person_id = create_person(session_db, capture_id, "Bob", default_skeleton_id="skel1")
    row = get_person(session_db, person_id)
    assert row["default_skeleton_id"] == "skel1"


def test_get_person_missing_returns_none(session_db) -> None:
    assert get_person(session_db, "does-not-exist") is None


def test_list_persons_ordered_by_name(session_db) -> None:
    capture_id = _make_capture(session_db)
    create_person(session_db, capture_id, "Zoe")
    create_person(session_db, capture_id, "Alice")
    create_person(session_db, capture_id, "Mallory")

    names = [r["name"] for r in list_persons(session_db, capture_id)]
    assert names == ["Alice", "Mallory", "Zoe"]


def test_list_persons_scoped_to_capture(session_db) -> None:
    cap1 = _make_capture(session_db, "cap1")
    session_db.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES ('cap2', 'sess1', 2)"
    )
    session_db.commit()
    create_person(session_db, cap1, "Alice")
    create_person(session_db, "cap2", "Bob")

    assert [r["name"] for r in list_persons(session_db, cap1)] == ["Alice"]
    assert [r["name"] for r in list_persons(session_db, "cap2")] == ["Bob"]


def test_find_or_create_person_creates_new(session_db) -> None:
    capture_id = _make_capture(session_db)
    person_id = find_or_create_person(session_db, capture_id, "Alice")
    assert get_person(session_db, person_id)["name"] == "Alice"


def test_find_or_create_person_reuses_existing(session_db) -> None:
    capture_id = _make_capture(session_db)
    first_id = create_person(session_db, capture_id, "Alice")
    second_id = find_or_create_person(session_db, capture_id, "Alice")
    assert second_id == first_id
    assert len(list_persons(session_db, capture_id)) == 1


def test_find_or_create_person_scoped_per_capture(session_db) -> None:
    """Same name in a different capture is a distinct person, not a reuse."""
    cap1 = _make_capture(session_db, "cap1")
    session_db.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES ('cap2', 'sess1', 2)"
    )
    session_db.commit()
    id1 = find_or_create_person(session_db, cap1, "Alice")
    id2 = find_or_create_person(session_db, "cap2", "Alice")
    assert id1 != id2


def test_rename_person(session_db) -> None:
    capture_id = _make_capture(session_db)
    person_id = create_person(session_db, capture_id, "Alice")
    rename_person(session_db, person_id, "Alicia")
    assert get_person(session_db, person_id)["name"] == "Alicia"


def test_rename_person_invalid_id_raises(session_db) -> None:
    with pytest.raises(ValueError, match="not found"):
        rename_person(session_db, "does-not-exist", "Alicia")


def test_set_default_skeleton(session_db) -> None:
    capture_id = _make_capture(session_db)
    person_id = create_person(session_db, capture_id, "Alice")
    set_default_skeleton(session_db, person_id, "skel1")
    assert get_person(session_db, person_id)["default_skeleton_id"] == "skel1"

    set_default_skeleton(session_db, person_id, None)
    assert get_person(session_db, person_id)["default_skeleton_id"] is None


def test_set_default_skeleton_invalid_id_raises(session_db) -> None:
    with pytest.raises(ValueError, match="not found"):
        set_default_skeleton(session_db, "does-not-exist", "skel1")


def test_delete_person_removes_row(session_db) -> None:
    capture_id = _make_capture(session_db)
    person_id = create_person(session_db, capture_id, "Alice")
    delete_person(session_db, person_id)
    assert get_person(session_db, person_id) is None


def test_delete_person_invalid_id_raises(session_db) -> None:
    with pytest.raises(ValueError, match="not found"):
        delete_person(session_db, "does-not-exist")


def test_delete_person_refuses_when_referenced_by_sequence_persons(session_db) -> None:
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

    with pytest.raises(ValueError, match="still referenced"):
        delete_person(session_db, person_id)
    assert get_person(session_db, person_id) is not None


def test_delete_person_refuses_when_referenced_by_detection_track_assignments(
    session_db,
) -> None:
    capture_id = _make_capture(session_db)
    person_id = create_person(session_db, capture_id, "Alice")
    session_db.execute(
        "INSERT INTO sync_configs (id, shot_id) VALUES ('sync1', ?)", (capture_id,)
    )
    session_db.execute(
        "INSERT INTO detection_runs "
        "(id, shot_id, sync_config_id, time_start_s, time_end_s, "
        " detector_model, pose_model, created_at) "
        "VALUES ('dr1', ?, 'sync1', 0.0, 1.0, 'yolo', 'rtmpose', '2026-01-01')",
        (capture_id,),
    )
    session_db.execute(
        "INSERT INTO detection_track_assignments "
        "(detection_run_id, shot_video_id, track_id, person_name, capture_person_id, "
        " first_frame, last_frame) "
        "VALUES ('dr1', 'vid1', 0, 'Alice', ?, 0, 100)",
        (person_id,),
    )
    session_db.commit()

    with pytest.raises(ValueError, match="still referenced"):
        delete_person(session_db, person_id)
    assert get_person(session_db, person_id) is not None


# ---------------------------------------------------------------------------
# persons_ordered_for_seg_run (segmentation-reuse gap 2)
# ---------------------------------------------------------------------------


def test_persons_ordered_for_seg_run_reads_persisted_snapshot(session_db) -> None:
    """The persisted persons_json wins even if capture_persons has since
    changed -- the whole point: don't assume today's order still matches
    what the masks were actually labeled with."""
    capture_id = _make_capture(session_db)
    create_person(session_db, capture_id, "Alice")
    session_db.execute(
        "INSERT INTO seg_quality_runs "
        "(id, shot_id, time_start_s, time_end_s, created_at, persons_json) "
        "VALUES ('seg1', ?, 0.0, 1e9, '2026-01-01', '[\"Bob\", \"Alice\"]')",
        (capture_id,),
    )
    session_db.commit()
    # A person added after the segmentation was created must not affect
    # the persisted order.
    create_person(session_db, capture_id, "Carol")

    assert persons_ordered_for_seg_run(session_db, "seg1") == ["Bob", "Alice"]


def test_persons_ordered_for_seg_run_falls_back_to_capture_persons(session_db) -> None:
    """No persisted snapshot (older row, or the offline add_seg_quality.py
    tool) -- falls back to today's capture_persons order, best-effort."""
    capture_id = _make_capture(session_db)
    create_person(session_db, capture_id, "Zoe")
    create_person(session_db, capture_id, "Alice")
    session_db.execute(
        "INSERT INTO seg_quality_runs "
        "(id, shot_id, time_start_s, time_end_s, created_at) "
        "VALUES ('seg1', ?, 0.0, 1e9, '2026-01-01')",
        (capture_id,),
    )
    session_db.commit()

    assert persons_ordered_for_seg_run(session_db, "seg1") == ["Alice", "Zoe"]


def test_persons_ordered_for_seg_run_missing_row_raises(session_db) -> None:
    with pytest.raises(ValueError, match="not found"):
        persons_ordered_for_seg_run(session_db, "does-not-exist")
