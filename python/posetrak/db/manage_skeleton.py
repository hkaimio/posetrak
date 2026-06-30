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
