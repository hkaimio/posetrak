"""Tests for scripts/db/manage_skeleton.py."""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[3]))  # project root

from scripts.db.manage_skeleton import import_skeleton, list_skeletons


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, name: str = "test_skeleton.yaml", content: str = "joints: []\n") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# import_skeleton
# ---------------------------------------------------------------------------


def test_import_skeleton_returns_sha256_id(registry_db: sqlite3.Connection, tmp_path: Path) -> None:
    """import_skeleton() should return the SHA-256 hex digest of the YAML content."""
    yaml_path = _write_yaml(tmp_path, content="joints: []\n")
    skeleton_id = import_skeleton(registry_db, yaml_path)
    expected = hashlib.sha256("joints: []\n".encode("utf-8")).hexdigest()
    assert skeleton_id == expected
    assert len(skeleton_id) == 64


def test_import_skeleton_idempotent(registry_db: sqlite3.Connection, tmp_path: Path) -> None:
    """Second import of the same YAML returns the same ID without creating a duplicate row."""
    yaml_path = _write_yaml(tmp_path, content="joints: [hip, knee]\n")
    id1 = import_skeleton(registry_db, yaml_path)
    id2 = import_skeleton(registry_db, yaml_path)
    assert id1 == id2
    count = registry_db.execute("SELECT COUNT(*) FROM skeletons").fetchone()[0]
    assert count == 1


def test_import_skeleton_default_name_from_filename(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """When name is None, the skeleton name defaults to the stem of the YAML path."""
    yaml_path = _write_yaml(tmp_path, name="my_skeleton.yaml")
    skeleton_id = import_skeleton(registry_db, yaml_path)
    row = registry_db.execute(
        "SELECT name FROM skeletons WHERE id = ?", (skeleton_id,)
    ).fetchone()
    assert row["name"] == "my_skeleton"


def test_import_skeleton_custom_name(registry_db: sqlite3.Connection, tmp_path: Path) -> None:
    """A custom name is stored instead of the filename stem."""
    yaml_path = _write_yaml(tmp_path)
    skeleton_id = import_skeleton(registry_db, yaml_path, name="SubjectA Skeleton")
    row = registry_db.execute(
        "SELECT name FROM skeletons WHERE id = ?", (skeleton_id,)
    ).fetchone()
    assert row["name"] == "SubjectA Skeleton"


def test_import_skeleton_stores_yaml_content(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """The raw YAML content is stored verbatim in the skeletons row."""
    content = "joints:\n  - hip\n  - knee\n"
    yaml_path = _write_yaml(tmp_path, content=content)
    skeleton_id = import_skeleton(registry_db, yaml_path)
    row = registry_db.execute(
        "SELECT yaml_content FROM skeletons WHERE id = ?", (skeleton_id,)
    ).fetchone()
    assert row["yaml_content"] == content


def test_import_skeleton_parent_id(registry_db: sqlite3.Connection, tmp_path: Path) -> None:
    """parent_id is correctly stored, enabling skeleton lineage tracking."""
    parent_yaml = _write_yaml(tmp_path, name="parent.yaml", content="joints: []\n")
    parent_id = import_skeleton(registry_db, parent_yaml)

    child_yaml = _write_yaml(tmp_path, name="child.yaml", content="joints: [hip]\n")
    child_id = import_skeleton(registry_db, child_yaml, parent_id=parent_id)

    row = registry_db.execute(
        "SELECT parent_id FROM skeletons WHERE id = ?", (child_id,)
    ).fetchone()
    assert row["parent_id"] == parent_id


def test_import_skeleton_person_label_and_source(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """person_label and source are stored when provided."""
    yaml_path = _write_yaml(tmp_path, content="joints: []\n")
    skeleton_id = import_skeleton(
        registry_db, yaml_path, person_label="subject_01", source="measurements_2025"
    )
    row = registry_db.execute(
        "SELECT person_label, source FROM skeletons WHERE id = ?", (skeleton_id,)
    ).fetchone()
    assert row["person_label"] == "subject_01"
    assert row["source"] == "measurements_2025"


# ---------------------------------------------------------------------------
# list_skeletons
# ---------------------------------------------------------------------------


def test_list_skeletons_empty(registry_db: sqlite3.Connection) -> None:
    """list_skeletons() returns an empty list when no skeletons are registered."""
    assert list_skeletons(registry_db) == []


def test_list_skeletons_returns_all(registry_db: sqlite3.Connection, tmp_path: Path) -> None:
    """list_skeletons() returns one row per imported skeleton."""
    yaml1 = _write_yaml(tmp_path, name="skel_a.yaml", content="joints: [a]\n")
    yaml2 = _write_yaml(tmp_path, name="skel_b.yaml", content="joints: [b]\n")
    import_skeleton(registry_db, yaml1)
    import_skeleton(registry_db, yaml2)
    rows = list_skeletons(registry_db)
    assert len(rows) == 2
    names = {row["name"] for row in rows}
    assert names == {"skel_a", "skel_b"}
