# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""manage_skeleton.py — Registry CRUD for skeleton definitions.

Skeletons are stored in the registry database. The primary key is the
SHA-256 hex digest of the YAML content, which makes import idempotent:
re-importing the same YAML file returns the existing ID without creating
a duplicate row.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.resources
import sqlite3
from pathlib import Path


def import_skeleton(
    registry: sqlite3.Connection,
    yaml_path: Path,
    *,
    name: str | None = None,
    person_label: str | None = None,
    source: str | None = None,
    parent_id: str | None = None,
    notes: str | None = None,
) -> str:
    """Import a skeleton YAML file into the registry.

    The skeleton ID is the SHA-256 hex digest of the YAML file content.
    If a skeleton with the same ID already exists, the function returns the
    existing ID without reinserting any rows.

    Parameters
    ----------
    registry:
        Open connection to a posetrak registry database.
    yaml_path:
        Path to the skeleton YAML file to import.
    name:
        Human-readable name for the skeleton. Defaults to ``yaml_path.stem``
        if ``None``.
    person_label:
        Optional label identifying the person this skeleton belongs to
        (e.g. ``"subject_01"``).
    source:
        Optional provenance string (e.g. path or description of origin).
    parent_id:
        Optional ID of a parent skeleton (for recording lineage, e.g. scaled
        versions).
    notes:
        Optional free-text notes stored with the skeleton row.

    Returns
    -------
    str
        SHA-256 hex ID (64 characters) of the skeleton row — either the newly
        created row or the pre-existing one.

    Raises
    ------
    FileNotFoundError
        If *yaml_path* does not exist.
    """
    yaml_content = yaml_path.read_text(encoding="utf-8")
    skeleton_id = hashlib.sha256(yaml_content.encode("utf-8")).hexdigest()

    # Idempotency: return early if already imported.
    existing = registry.execute(
        "SELECT id FROM skeletons WHERE id = ?", (skeleton_id,)
    ).fetchone()
    if existing is not None:
        return skeleton_id

    resolved_name = name if name is not None else yaml_path.stem
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    with registry:
        registry.execute(
            "INSERT INTO skeletons "
            "(id, name, parent_id, person_label, source, yaml_content, created_at, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                skeleton_id,
                resolved_name,
                parent_id,
                person_label,
                source,
                yaml_content,
                created_at,
                notes,
            ),
        )

    return skeleton_id


def import_skeleton_str(
    db: sqlite3.Connection,
    yaml_content: str,
    *,
    name: str | None = None,
    parent_id: str | None = None,
    person_label: str | None = None,
    source: str | None = None,
    notes: str | None = None,
) -> str:
    """Import a skeleton from a YAML string into *db* (registry or session).

    Identical to :func:`import_skeleton` but accepts YAML content directly
    instead of a file path.  The skeleton ID is the SHA-256 hex digest of the
    content, so this is idempotent.
    """
    skeleton_id = hashlib.sha256(yaml_content.encode("utf-8")).hexdigest()

    existing = db.execute(
        "SELECT id FROM skeletons WHERE id = ?", (skeleton_id,)
    ).fetchone()
    if existing is not None:
        return skeleton_id

    resolved_name = name or skeleton_id[:12]
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    with db:
        db.execute(
            "INSERT INTO skeletons "
            "(id, name, parent_id, person_label, source, yaml_content, created_at, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (skeleton_id, resolved_name, parent_id, person_label,
             source, yaml_content, created_at, notes),
        )

    return skeleton_id


def copy_skeleton_to_session(
    registry: sqlite3.Connection,
    session: sqlite3.Connection,
    skeleton_id: str,
) -> None:
    """Copy a skeleton row from registry into a session DB.

    Uses INSERT OR IGNORE so calling this function multiple times with the
    same *skeleton_id* is safe.

    Parameters
    ----------
    registry:
        Open connection to the posetrak registry database (source).
    session:
        Open connection to a posetrak session database (destination).
    skeleton_id:
        ``skeletons.id`` (SHA-256 hex) to copy.

    Raises
    ------
    ValueError
        If *skeleton_id* does not exist in *registry*.
    """
    from posetrak.db.db import _copy_rows_if_missing
    _copy_rows_if_missing(registry, session, "skeletons", [skeleton_id])


def copy_skeleton(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    skeleton_id: str,
) -> None:
    """Copy a skeleton row from *src* to *dst* (INSERT OR IGNORE, idempotent).

    Parameters
    ----------
    src:
        Source database connection.
    dst:
        Destination database connection.
    skeleton_id:
        ``skeletons.id`` (SHA-256 hex) to copy.

    Raises
    ------
    ValueError
        If *skeleton_id* does not exist in *src*.
    """
    from posetrak.db.db import _copy_rows_if_missing
    _copy_rows_if_missing(src, dst, "skeletons", [skeleton_id])


_DEFAULT_SKELETONS: list[tuple[str, str]] = [
    ("Default male", "default_male.yaml"),
    ("Default female", "default_female.yaml"),
]


def seed_default_skeletons(conn: sqlite3.Connection) -> list[str]:
    """Insert bundled default skeletons into *conn* if not already present.

    Idempotent: re-seeding a DB that already has the defaults is a no-op
    (SHA-256 primary key means the INSERT silently skips duplicates).
    Returns the list of skeleton IDs (one per default, whether new or existing).
    """
    pkg = importlib.resources.files("posetrak.data.skeletons")
    ids: list[str] = []
    for name, filename in _DEFAULT_SKELETONS:
        yaml_content = pkg.joinpath(filename).read_text(encoding="utf-8")
        ids.append(
            import_skeleton_str(conn, yaml_content, name=name, source="bundled-default")
        )
    return ids


def list_skeletons(registry: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all skeleton rows from the registry, ordered by creation time.

    Parameters
    ----------
    registry:
        Open connection to a posetrak registry database.

    Returns
    -------
    list[sqlite3.Row]
        All rows from the ``skeletons`` table, ordered by ``created_at``
        ascending.
    """
    return registry.execute(
        "SELECT * FROM skeletons ORDER BY created_at"
    ).fetchall()


def skeletons_with_newer_version(rows) -> set[str]:
    """Return the ids of skeletons that have a descendant (any number of
    parent_id hops, not just a direct child) sharing their own name.

    Found via a real bug, 2026-08-24: a hierarchical-tracking run silently
    used a skeleton whose HandL/HandR groups lacked the
    freeflyer_joint/ref_marker metadata upgrade_skeleton_hand_groups.py adds
    -- because a *corrected* version of that exact skeleton already existed
    (same name, `parent_id` pointing at the original), and every picker
    listed both under the identical name with no way to tell them apart.
    This flags the older, easy-to-mistake-for-current row.

    Parameters
    ----------
    rows:
        Skeleton rows as returned by list_skeletons() (or any iterable of
        rows/mappings with id, name, parent_id columns).
    """
    by_id = {r["id"]: r for r in rows}
    children_by_parent: dict[str, list[str]] = {}
    for r in rows:
        if r["parent_id"]:
            children_by_parent.setdefault(r["parent_id"], []).append(r["id"])

    def has_same_name_descendant(skel_id: str, name: str) -> bool:
        for child_id in children_by_parent.get(skel_id, []):
            child = by_id.get(child_id)
            if child is None:
                continue
            if child["name"] == name or has_same_name_descendant(child_id, name):
                return True
        return False

    return {r["id"] for r in rows if has_same_name_descendant(r["id"], r["name"])}


def skeleton_picker_labels(rows) -> dict[str, str]:
    """Build a {skeleton_id: display_label} map for pickers, disambiguating
    skeletons that share a name and flagging one that has a same-named,
    newer descendant. See skeletons_with_newer_version() for the bug this
    guards against.

    Parameters
    ----------
    rows:
        Skeleton rows as returned by list_skeletons() (or any iterable of
        rows/mappings with id, name, parent_id, created_at columns).

    Returns
    -------
    dict[str, str]
        - A name that's unique among *rows* passes through unchanged.
        - A duplicated name gets its creation date appended, e.g.
          "Harri fingers fixed (2026-07-21)" -- and, in the (expected to be
          rare) case where the date alone still doesn't disambiguate, an id
          prefix too.
        - A skeleton flagged by skeletons_with_newer_version() gets
          " -- newer version exists" appended, so the older, easy-to-mistake
          duplicate is the one visibly flagged rather than the one to use.
    """
    names = [r["name"] for r in rows]
    name_counts = {n: names.count(n) for n in set(names)}
    newer_exists = skeletons_with_newer_version(rows)

    # For duplicated names, precompute date-based labels and detect any
    # residual collision (same name AND same date) needing an id suffix too.
    date_label_counts: dict[tuple[str, str], int] = {}
    for r in rows:
        if name_counts[r["name"]] > 1:
            date = (r["created_at"] or "")[:10]
            key = (r["name"], date)
            date_label_counts[key] = date_label_counts.get(key, 0) + 1

    labels: dict[str, str] = {}
    for r in rows:
        name = r["name"]
        label = name
        if name_counts[name] > 1:
            date = (r["created_at"] or "")[:10]
            label = f"{name} ({date})" if date else name
            if date_label_counts.get((name, date), 0) > 1:
                label += f" [{r['id'][:8]}]"
        if r["id"] in newer_exists:
            label += " -- newer version exists"
        labels[r["id"]] = label
    return labels
