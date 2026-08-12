"""Tests for posetrak/db/manage_marker_body.py."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from posetrak.db.manage_marker_body import (
    copy_marker_body_to_session,
    delete_scene_marker_body,
    import_marker_body,
    import_marker_body_str,
    list_marker_bodies,
    list_scene_marker_bodies,
    list_scene_marker_bodies_by_group,
    list_scene_marker_group_names,
    read_scene_marker_body_pose,
    upsert_scene_marker_body,
)


def _write_yaml(tmp_path: Path, name: str = "test_rig.yaml", content: str = "name: test-rig\nmarkers: []\n") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# import_marker_body
# ---------------------------------------------------------------------------


def test_import_marker_body_returns_sha256_id(registry_db: sqlite3.Connection, tmp_path: Path) -> None:
    """import_marker_body() should return the SHA-256 hex digest of the YAML content."""
    content = "name: test-rig\nmarkers: []\n"
    yaml_path = _write_yaml(tmp_path, content=content)
    body_id = import_marker_body(registry_db, yaml_path)
    assert body_id == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert len(body_id) == 64


def test_import_marker_body_idempotent(registry_db: sqlite3.Connection, tmp_path: Path) -> None:
    """Second import of the same YAML returns the same ID without creating a duplicate row."""
    yaml_path = _write_yaml(tmp_path, content="name: box\nmarkers: []\n")
    count_before = registry_db.execute("SELECT COUNT(*) FROM marker_body_definitions").fetchone()[0]
    id1 = import_marker_body(registry_db, yaml_path)
    id2 = import_marker_body(registry_db, yaml_path)
    assert id1 == id2
    count_after = registry_db.execute("SELECT COUNT(*) FROM marker_body_definitions").fetchone()[0]
    assert count_after == count_before + 1


def test_import_marker_body_default_name_from_filename(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """When name is None, the name defaults to the stem of the YAML path."""
    yaml_path = _write_yaml(tmp_path, name="aikido-calib-box-v1.yaml")
    body_id = import_marker_body(registry_db, yaml_path)
    row = registry_db.execute(
        "SELECT name FROM marker_body_definitions WHERE id = ?", (body_id,)
    ).fetchone()
    assert row["name"] == "aikido-calib-box-v1"


def test_import_marker_body_custom_name_and_source(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    yaml_path = _write_yaml(tmp_path)
    body_id = import_marker_body(registry_db, yaml_path, name="My Rig", source="hand-measured")
    row = registry_db.execute(
        "SELECT name, source FROM marker_body_definitions WHERE id = ?", (body_id,)
    ).fetchone()
    assert row["name"] == "My Rig"
    assert row["source"] == "hand-measured"


def test_import_marker_body_stores_yaml_content(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    content = "name: test-rig\nmarkers:\n  - name: top\n    type: aruco\n"
    yaml_path = _write_yaml(tmp_path, content=content)
    body_id = import_marker_body(registry_db, yaml_path)
    row = registry_db.execute(
        "SELECT yaml_content FROM marker_body_definitions WHERE id = ?", (body_id,)
    ).fetchone()
    assert row["yaml_content"] == content


def test_import_marker_body_missing_file_raises(registry_db: sqlite3.Connection, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        import_marker_body(registry_db, tmp_path / "nonexistent.yaml")


def test_import_marker_body_str_idempotent(registry_db: sqlite3.Connection) -> None:
    content = "name: box\nmarkers: []\n"
    id1 = import_marker_body_str(registry_db, content, name="box")
    id2 = import_marker_body_str(registry_db, content, name="box")
    assert id1 == id2


# ---------------------------------------------------------------------------
# copy_marker_body_to_session / list_marker_bodies
# ---------------------------------------------------------------------------


def test_copy_marker_body_to_session(
    registry_db: sqlite3.Connection, session_db: sqlite3.Connection, tmp_path: Path
) -> None:
    yaml_path = _write_yaml(tmp_path, content="name: box\nmarkers: []\n")
    body_id = import_marker_body(registry_db, yaml_path, name="box")

    copy_marker_body_to_session(registry_db, session_db, body_id)

    row = session_db.execute(
        "SELECT name, yaml_content FROM marker_body_definitions WHERE id = ?", (body_id,)
    ).fetchone()
    assert row is not None
    assert row["name"] == "box"


def test_copy_marker_body_to_session_missing_raises(session_db: sqlite3.Connection) -> None:
    registry = sqlite3.connect(":memory:")
    registry.row_factory = sqlite3.Row
    registry.execute(
        "CREATE TABLE marker_body_definitions (id TEXT PRIMARY KEY, name TEXT, "
        "yaml_content TEXT, source TEXT, created_at TEXT, notes TEXT)"
    )
    with pytest.raises(ValueError):
        copy_marker_body_to_session(registry, session_db, "nonexistent")


def test_copy_marker_body_to_session_idempotent(
    registry_db: sqlite3.Connection, session_db: sqlite3.Connection, tmp_path: Path
) -> None:
    yaml_path = _write_yaml(tmp_path)
    body_id = import_marker_body(registry_db, yaml_path)
    copy_marker_body_to_session(registry_db, session_db, body_id)
    copy_marker_body_to_session(registry_db, session_db, body_id)  # no error, no duplicate
    count = session_db.execute(
        "SELECT COUNT(*) FROM marker_body_definitions WHERE id = ?", (body_id,)
    ).fetchone()[0]
    assert count == 1


def test_list_marker_bodies_empty(registry_db: sqlite3.Connection) -> None:
    assert list_marker_bodies(registry_db) == []


def test_list_marker_bodies_returns_all(registry_db: sqlite3.Connection, tmp_path: Path) -> None:
    import_marker_body(registry_db, _write_yaml(tmp_path, name="a.yaml", content="name: a\nmarkers: []\n"))
    import_marker_body(registry_db, _write_yaml(tmp_path, name="b.yaml", content="name: b\nmarkers: []\n"))
    names = {r["name"] for r in list_marker_bodies(registry_db)}
    assert names == {"a", "b"}


# ---------------------------------------------------------------------------
# upsert_scene_marker_body / list_scene_marker_bodies / read pose
# ---------------------------------------------------------------------------


def _insert_session(conn: sqlite3.Connection, session_id: str = "sess1") -> None:
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES (?, '2026-01-01')", (session_id,)
    )
    conn.commit()


def test_upsert_scene_marker_body_inserts_new_row(session_db: sqlite3.Connection) -> None:
    _insert_session(session_db)
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.0])
    row_id = upsert_scene_marker_body(
        session_db, "sess1", "calib-box", R, t, is_primary_anchor=True,
    )
    row = session_db.execute(
        "SELECT * FROM scene_marker_bodies WHERE id = ?", (row_id,)
    ).fetchone()
    assert row["session_id"] == "sess1"
    assert row["label"] == "calib-box"
    assert row["is_primary_anchor"] == 1
    R_read, t_read = read_scene_marker_body_pose(row)
    np.testing.assert_allclose(R_read, R)
    np.testing.assert_allclose(t_read, t)


def test_upsert_scene_marker_body_lone_tag_inline_fields(session_db: sqlite3.Connection) -> None:
    _insert_session(session_db)
    row_id = upsert_scene_marker_body(
        session_db, "sess1", "wall-tag-north",
        R=np.eye(3), t=np.array([1.0, 2.0, 3.0]),
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="3", marker_size=0.1,
    )
    row = session_db.execute(
        "SELECT marker_body_definition_id, dictionary, marker_id, marker_size "
        "FROM scene_marker_bodies WHERE id = ?", (row_id,)
    ).fetchone()
    assert row["marker_body_definition_id"] is None
    assert row["dictionary"] == "DICT_5X5_50"
    assert row["marker_id"] == "3"
    assert row["marker_size"] == pytest.approx(0.1)


def test_upsert_scene_marker_body_overwrites_same_label(session_db: sqlite3.Connection) -> None:
    """Re-solving the same body under the same label updates the existing row
    in place, not a new one (this table is 'current believed pose', not history)."""
    _insert_session(session_db)
    id1 = upsert_scene_marker_body(session_db, "sess1", "calib-box", np.eye(3), np.zeros(3))
    id2 = upsert_scene_marker_body(
        session_db, "sess1", "calib-box", np.eye(3), np.array([1.0, 2.0, 3.0]),
    )
    assert id1 == id2
    count = session_db.execute(
        "SELECT COUNT(*) FROM scene_marker_bodies WHERE session_id = 'sess1' AND label = 'calib-box'"
    ).fetchone()[0]
    assert count == 1
    row = session_db.execute("SELECT * FROM scene_marker_bodies WHERE id = ?", (id1,)).fetchone()
    _, t_read = read_scene_marker_body_pose(row)
    np.testing.assert_allclose(t_read, [1.0, 2.0, 3.0])


def test_upsert_scene_marker_body_different_labels_different_rows(session_db: sqlite3.Connection) -> None:
    _insert_session(session_db)
    id1 = upsert_scene_marker_body(session_db, "sess1", "tag-a", np.eye(3), np.zeros(3))
    id2 = upsert_scene_marker_body(session_db, "sess1", "tag-b", np.eye(3), np.zeros(3))
    assert id1 != id2


def test_list_scene_marker_bodies_ordered_by_label(session_db: sqlite3.Connection) -> None:
    _insert_session(session_db)
    upsert_scene_marker_body(session_db, "sess1", "zzz-tag", np.eye(3), np.zeros(3))
    upsert_scene_marker_body(session_db, "sess1", "aaa-tag", np.eye(3), np.zeros(3))
    rows = list_scene_marker_bodies(session_db, "sess1")
    assert [r["label"] for r in rows] == ["aaa-tag", "zzz-tag"]


def test_list_scene_marker_bodies_scoped_to_session(session_db: sqlite3.Connection) -> None:
    _insert_session(session_db, "sess1")
    _insert_session(session_db, "sess2")
    upsert_scene_marker_body(session_db, "sess1", "tag-a", np.eye(3), np.zeros(3))
    upsert_scene_marker_body(session_db, "sess2", "tag-b", np.eye(3), np.zeros(3))
    rows = list_scene_marker_bodies(session_db, "sess1")
    assert [r["label"] for r in rows] == ["tag-a"]


def test_upsert_scene_marker_body_with_real_definition(
    session_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """A rig-anchor row referencing a real marker_body_definitions row
    (already embedded in the session DB, or imported straight into it)."""
    _insert_session(session_db)
    body_id = import_marker_body_str(session_db, "name: box\nmarkers: []\n", name="box")
    row_id = upsert_scene_marker_body(
        session_db, "sess1", "calib-box", np.eye(3), np.zeros(3),
        marker_body_definition_id=body_id, is_primary_anchor=True,
    )
    row = session_db.execute(
        "SELECT marker_body_definition_id FROM scene_marker_bodies WHERE id = ?", (row_id,)
    ).fetchone()
    assert row["marker_body_definition_id"] == body_id


def test_read_scene_marker_body_pose_nonidentity(session_db: sqlite3.Connection) -> None:
    _insert_session(session_db)
    R_true = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t_true = np.array([1.5, -2.5, 0.3])
    row_id = upsert_scene_marker_body(session_db, "sess1", "cam-check", R_true, t_true)
    row = session_db.execute("SELECT * FROM scene_marker_bodies WHERE id = ?", (row_id,)).fetchone()
    R_read, t_read = read_scene_marker_body_pose(row)
    np.testing.assert_allclose(R_read, R_true)
    np.testing.assert_allclose(t_read, t_true)


# ---------------------------------------------------------------------------
# delete_scene_marker_body
# ---------------------------------------------------------------------------


def test_delete_scene_marker_body_removes_row(session_db: sqlite3.Connection) -> None:
    _insert_session(session_db)
    upsert_scene_marker_body(session_db, "sess1", "stale-tag", np.eye(3), np.zeros(3))
    assert delete_scene_marker_body(session_db, "sess1", "stale-tag") is True
    rows = list_scene_marker_bodies(session_db, "sess1")
    assert rows == []


def test_delete_scene_marker_body_missing_returns_false(session_db: sqlite3.Connection) -> None:
    _insert_session(session_db)
    assert delete_scene_marker_body(session_db, "sess1", "nonexistent") is False


def test_delete_scene_marker_body_scoped_to_session(session_db: sqlite3.Connection) -> None:
    _insert_session(session_db, "sess1")
    _insert_session(session_db, "sess2")
    upsert_scene_marker_body(session_db, "sess1", "tag-a", np.eye(3), np.zeros(3))
    upsert_scene_marker_body(session_db, "sess2", "tag-a", np.eye(3), np.zeros(3))
    delete_scene_marker_body(session_db, "sess1", "tag-a")
    assert list_scene_marker_bodies(session_db, "sess1") == []
    assert len(list_scene_marker_bodies(session_db, "sess2")) == 1


def test_delete_scene_marker_body_leaves_other_labels(session_db: sqlite3.Connection) -> None:
    _insert_session(session_db)
    upsert_scene_marker_body(session_db, "sess1", "tag-a", np.eye(3), np.zeros(3))
    upsert_scene_marker_body(session_db, "sess1", "tag-b", np.eye(3), np.zeros(3))
    delete_scene_marker_body(session_db, "sess1", "tag-a")
    rows = list_scene_marker_bodies(session_db, "sess1")
    assert [r["label"] for r in rows] == ["tag-b"]


# ---------------------------------------------------------------------------
# group_name (2026-08-12) -- named groups of scene markers, e.g. one per
# physical room, so a later capture can pick a specific room's markers
# instead of every stored marker in the session loading together.
# ---------------------------------------------------------------------------


def test_upsert_scene_marker_body_defaults_to_ungrouped(session_db: sqlite3.Connection) -> None:
    _insert_session(session_db)
    row_id = upsert_scene_marker_body(session_db, "sess1", "tag-a", np.eye(3), np.zeros(3))
    row = session_db.execute(
        "SELECT group_name FROM scene_marker_bodies WHERE id = ?", (row_id,)
    ).fetchone()
    assert row["group_name"] == ""


def test_upsert_scene_marker_body_stores_group_name(session_db: sqlite3.Connection) -> None:
    _insert_session(session_db)
    row_id = upsert_scene_marker_body(
        session_db, "sess1", "tag:3", np.eye(3), np.zeros(3), group_name="room7",
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="3", marker_size=0.1,
    )
    row = session_db.execute(
        "SELECT group_name FROM scene_marker_bodies WHERE id = ?", (row_id,)
    ).fetchone()
    assert row["group_name"] == "room7"


def test_upsert_scene_marker_body_different_groups_same_label_no_collision(
    session_db: sqlite3.Connection,
) -> None:
    """Two different rooms may reuse the same tag id/label without one
    overwriting the other -- the whole point of group_name existing."""
    _insert_session(session_db)
    id_a = upsert_scene_marker_body(
        session_db, "sess1", "tag:3", np.eye(3), np.array([1.0, 0.0, 0.0]), group_name="room7",
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="3", marker_size=0.1,
    )
    id_b = upsert_scene_marker_body(
        session_db, "sess1", "tag:3", np.eye(3), np.array([9.0, 0.0, 0.0]), group_name="room8",
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="3", marker_size=0.1,
    )
    assert id_a != id_b
    row_a = session_db.execute("SELECT * FROM scene_marker_bodies WHERE id = ?", (id_a,)).fetchone()
    row_b = session_db.execute("SELECT * FROM scene_marker_bodies WHERE id = ?", (id_b,)).fetchone()
    _, t_a = read_scene_marker_body_pose(row_a)
    _, t_b = read_scene_marker_body_pose(row_b)
    np.testing.assert_allclose(t_a, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(t_b, [9.0, 0.0, 0.0])


def test_upsert_scene_marker_body_same_group_same_label_overwrites(
    session_db: sqlite3.Connection,
) -> None:
    _insert_session(session_db)
    id1 = upsert_scene_marker_body(
        session_db, "sess1", "tag:3", np.eye(3), np.array([1.0, 0.0, 0.0]), group_name="room7",
    )
    id2 = upsert_scene_marker_body(
        session_db, "sess1", "tag:3", np.eye(3), np.array([2.0, 0.0, 0.0]), group_name="room7",
    )
    assert id1 == id2
    row = session_db.execute("SELECT * FROM scene_marker_bodies WHERE id = ?", (id1,)).fetchone()
    _, t = read_scene_marker_body_pose(row)
    np.testing.assert_allclose(t, [2.0, 0.0, 0.0])


def test_list_scene_marker_group_names_excludes_ungrouped(session_db: sqlite3.Connection) -> None:
    _insert_session(session_db)
    upsert_scene_marker_body(
        session_db, "sess1", "tag:1", np.eye(3), np.zeros(3),
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="1", marker_size=0.1,
    )  # ungrouped
    upsert_scene_marker_body(
        session_db, "sess1", "tag:2", np.eye(3), np.zeros(3), group_name="room7",
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="2", marker_size=0.1,
    )
    groups = list_scene_marker_group_names(session_db, "sess1")
    assert [g["group_name"] for g in groups] == ["room7"]
    assert groups[0]["n_markers"] == 1


def test_list_scene_marker_group_names_excludes_rig_anchor_rows(
    session_db: sqlite3.Connection,
) -> None:
    _insert_session(session_db)
    body_id = import_marker_body_str(session_db, "name: box\nmarkers: []\n", name="box")
    upsert_scene_marker_body(
        session_db, "sess1", "rig:box", np.eye(3), np.zeros(3), group_name="room7",
        marker_body_definition_id=body_id, is_primary_anchor=True,
    )
    upsert_scene_marker_body(
        session_db, "sess1", "tag:2", np.eye(3), np.zeros(3), group_name="room7",
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="2", marker_size=0.1,
    )
    groups = list_scene_marker_group_names(session_db, "sess1")
    assert groups[0]["n_markers"] == 1  # the rig anchor row doesn't count


def test_list_scene_marker_bodies_by_group_filters_correctly(session_db: sqlite3.Connection) -> None:
    _insert_session(session_db)
    upsert_scene_marker_body(
        session_db, "sess1", "tag:1", np.eye(3), np.zeros(3), group_name="room7",
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="1", marker_size=0.1,
    )
    upsert_scene_marker_body(
        session_db, "sess1", "tag:2", np.eye(3), np.zeros(3), group_name="room8",
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="2", marker_size=0.1,
    )
    rows = list_scene_marker_bodies_by_group(session_db, "sess1", "room7")
    assert [r["label"] for r in rows] == ["tag:1"]


def test_list_scene_marker_bodies_by_group_none_means_ungrouped(
    session_db: sqlite3.Connection,
) -> None:
    _insert_session(session_db)
    upsert_scene_marker_body(
        session_db, "sess1", "tag:1", np.eye(3), np.zeros(3),
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="1", marker_size=0.1,
    )
    upsert_scene_marker_body(
        session_db, "sess1", "tag:2", np.eye(3), np.zeros(3), group_name="room7",
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="2", marker_size=0.1,
    )
    rows = list_scene_marker_bodies_by_group(session_db, "sess1", None)
    assert [r["label"] for r in rows] == ["tag:1"]


def test_delete_scene_marker_body_scoped_to_group(session_db: sqlite3.Connection) -> None:
    _insert_session(session_db)
    upsert_scene_marker_body(
        session_db, "sess1", "tag:3", np.eye(3), np.zeros(3), group_name="room7",
    )
    upsert_scene_marker_body(
        session_db, "sess1", "tag:3", np.eye(3), np.zeros(3), group_name="room8",
    )
    delete_scene_marker_body(session_db, "sess1", "tag:3", group_name="room7")

    remaining = list_scene_marker_bodies(session_db, "sess1")
    assert [(r["label"], r["group_name"]) for r in remaining] == [("tag:3", "room8")]
