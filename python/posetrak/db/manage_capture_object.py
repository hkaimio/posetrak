# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""manage_capture_object.py — Session CRUD for capture-scoped tracked objects.

``capture_objects`` is the object analog of ``capture_persons``
(manage_person.py) -- see
docs/roadmap/features/marker-based-mocap/marker-mocap-design.md §4.2 and
§7.1 sub-phase 1c. Deliberately mirrors that module's shape (create, list,
get, rename, delete-with-in-use-guard) rather than inventing a different
pattern for what is structurally the same kind of row.
"""

from __future__ import annotations

import datetime
import sqlite3

from posetrak.db.db import generate_id


def create_capture_object(
    session: sqlite3.Connection,
    capture_id: str,
    name: str,
    marker_body_definition_id: str,
    *,
    notes: str | None = None,
) -> str:
    """Create a new tracked object for *capture_id*.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    capture_id:
        The capture this object belongs to.
    name:
        Human-readable instance name (e.g. "bokken-A").
    marker_body_definition_id:
        The registry ``marker_body_definitions`` row (already copied into
        this session, see ``manage_marker_body.copy_marker_body_to_session``)
        describing this object's physical marker geometry. Required --
        unlike a person's ``default_skeleton_id``, an object with no marker
        body has no way to ever be detected or tracked.
    notes:
        Optional free-text notes.

    Returns
    -------
    str
        UUID of the newly created ``capture_objects`` row.
    """
    object_id = generate_id()
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with session:
        session.execute(
            "INSERT INTO capture_objects "
            "(id, capture_id, name, marker_body_definition_id, notes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (object_id, capture_id, name, marker_body_definition_id, notes, created_at),
        )
    return object_id


def list_capture_objects(session: sqlite3.Connection, capture_id: str) -> list[sqlite3.Row]:
    """Return *capture_id*'s tracked objects, ordered by name."""
    return session.execute(
        "SELECT * FROM capture_objects WHERE capture_id = ? ORDER BY name",
        (capture_id,),
    ).fetchall()


def get_capture_object(session: sqlite3.Connection, capture_object_id: str) -> sqlite3.Row | None:
    """Return one ``capture_objects`` row by id, or ``None`` if not found."""
    return session.execute(
        "SELECT * FROM capture_objects WHERE id = ?", (capture_object_id,)
    ).fetchone()


def rename_capture_object(session: sqlite3.Connection, capture_object_id: str, name: str) -> None:
    """Rename an existing tracked object, in place.

    Raises
    ------
    ValueError
        If *capture_object_id* does not refer to an existing row.
    """
    with session:
        cur = session.execute(
            "UPDATE capture_objects SET name = ? WHERE id = ?", (name, capture_object_id)
        )
        if cur.rowcount == 0:
            raise ValueError(f"capture_objects row not found: {capture_object_id!r}")


def delete_capture_object(session: sqlite3.Connection, capture_object_id: str) -> None:
    """Delete a tracked object, refusing if any detection/tracking data
    still references it (mirrors ``manage_person.delete_person``'s
    detection-run-immutability guard).

    Raises
    ------
    ValueError
        If *capture_object_id* does not refer to an existing row, or if it
        is still referenced by ``detection_runs``/``tracking_run_persons``.
    """
    if get_capture_object(session, capture_object_id) is None:
        raise ValueError(f"capture_objects row not found: {capture_object_id!r}")
    in_use = session.execute(
        "SELECT 1 FROM detection_runs WHERE capture_object_id = ? "
        "UNION SELECT 1 FROM tracking_run_persons WHERE capture_object_id = ? "
        "LIMIT 1",
        (capture_object_id, capture_object_id),
    ).fetchone()
    if in_use is not None:
        raise ValueError(
            f"Cannot delete object {capture_object_id!r}: still referenced by existing "
            "detection or tracking data. Rename it instead if it was created by mistake."
        )
    with session:
        session.execute("DELETE FROM capture_objects WHERE id = ?", (capture_object_id,))
