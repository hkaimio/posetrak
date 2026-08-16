"""manage_person.py — Session CRUD for capture-scoped named performers.

``capture_persons`` defines a performer once per capture (trials within one
capture are near-certain to share the same physical performers -- see
docs/roadmap/features/configuration-improvements/config-improvements-design.md,
"Person model: promote identity to capture level"), replacing the previous
model where identity only existed as a free-text ``person_name`` scoped to
one detection run. Unlike ``manage_config``/``manage_skeleton``, this module
operates on *session* databases, not the registry -- persons are inherently
capture-specific, and captures live in session DBs.
"""

from __future__ import annotations

import datetime
import json
import sqlite3

from posetrak.db.db import generate_id


def create_person(
    session: sqlite3.Connection,
    capture_id: str,
    name: str,
    *,
    default_skeleton_id: str | None = None,
    notes: str | None = None,
) -> str:
    """Create a new named performer for *capture_id*.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    capture_id:
        The capture this person belongs to.
    name:
        Human-readable performer name (e.g. "Alice").
    default_skeleton_id:
        Optional registry ``skeletons.id`` this person is usually tracked
        with -- pre-fills, but doesn't lock, the skeleton choice at
        tracking-run time.
    notes:
        Optional free-text notes.

    Returns
    -------
    str
        UUID of the newly created ``capture_persons`` row.
    """
    person_id = generate_id()
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with session:
        session.execute(
            "INSERT INTO capture_persons "
            "(id, capture_id, name, default_skeleton_id, notes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (person_id, capture_id, name, default_skeleton_id, notes, created_at),
        )
    return person_id


def find_or_create_person(
    session: sqlite3.Connection,
    capture_id: str,
    name: str,
    *,
    default_skeleton_id: str | None = None,
) -> str:
    """Return the id of *capture_id*'s existing person named *name*
    (exact, case-sensitive match), creating one if none exists yet.

    The quick-create path behind "+ New person..." pickers -- callers that
    already resolved a name to an id should use that id directly rather
    than calling this on every reference.
    """
    row = session.execute(
        "SELECT id FROM capture_persons WHERE capture_id = ? AND name = ?",
        (capture_id, name),
    ).fetchone()
    if row is not None:
        return row["id"]
    return create_person(session, capture_id, name, default_skeleton_id=default_skeleton_id)


def list_persons(session: sqlite3.Connection, capture_id: str) -> list[sqlite3.Row]:
    """Return *capture_id*'s persons, ordered by name."""
    return session.execute(
        "SELECT * FROM capture_persons WHERE capture_id = ? ORDER BY name",
        (capture_id,),
    ).fetchall()


def persons_ordered_for_seg_run(session: sqlite3.Connection, seg_quality_run_id: str) -> list[str]:
    """Return the ordinal->name mapping (index i = mask label i+1) a
    segmentation's masks were labeled with.

    Reads ``seg_quality_runs.persons_json`` -- the snapshot taken at mask-
    creation time (see ``CutieInitPanel._ensure_seg_run``) -- so a caller
    reusing this segmentation later doesn't have to assume today's
    ``capture_persons`` order still matches whatever order was in effect
    when the masks were made (see docs/roadmap/features/segmentation-reuse/
    segmentation-reuse-design.md, gap 2). Falls back to today's
    ``list_persons`` order for a segmentation created before this column
    existed, or via the offline ``add_seg_quality.py`` tool (no
    interactive person labeling there) -- best-effort, not guaranteed
    correct for those older rows.

    Raises
    ------
    ValueError
        If *seg_quality_run_id* does not refer to an existing row.
    """
    row = session.execute(
        "SELECT shot_id, persons_json FROM seg_quality_runs WHERE id = ?",
        (seg_quality_run_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"seg_quality_runs row not found: {seg_quality_run_id!r}")
    if row["persons_json"]:
        return json.loads(row["persons_json"])
    return [p["name"] for p in list_persons(session, row["shot_id"])]


def get_person(session: sqlite3.Connection, person_id: str) -> sqlite3.Row | None:
    """Return one ``capture_persons`` row by id, or ``None`` if not found."""
    return session.execute(
        "SELECT * FROM capture_persons WHERE id = ?", (person_id,)
    ).fetchone()


def rename_person(session: sqlite3.Connection, person_id: str, name: str) -> None:
    """Rename an existing person, in place.

    Raises
    ------
    ValueError
        If *person_id* does not refer to an existing row.
    """
    with session:
        cur = session.execute(
            "UPDATE capture_persons SET name = ? WHERE id = ?", (name, person_id)
        )
        if cur.rowcount == 0:
            raise ValueError(f"capture_persons row not found: {person_id!r}")


def set_default_skeleton(
    session: sqlite3.Connection, person_id: str, skeleton_id: str | None
) -> None:
    """Set (or clear, with ``skeleton_id=None``) a person's default skeleton.

    Raises
    ------
    ValueError
        If *person_id* does not refer to an existing row.
    """
    with session:
        cur = session.execute(
            "UPDATE capture_persons SET default_skeleton_id = ? WHERE id = ?",
            (skeleton_id, person_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"capture_persons row not found: {person_id!r}")


def delete_person(session: sqlite3.Connection, person_id: str) -> None:
    """Delete a person, refusing if any detection/tracking data still
    references it (mirrors the project's detection-run-immutability
    principle -- a person that's already been used shouldn't quietly
    disappear out from under that data's provenance).

    Raises
    ------
    ValueError
        If *person_id* does not refer to an existing row, or if it is
        still referenced by ``sequence_persons``/``detection_track_assignments``.
    """
    if get_person(session, person_id) is None:
        raise ValueError(f"capture_persons row not found: {person_id!r}")
    in_use = session.execute(
        "SELECT 1 FROM sequence_persons WHERE capture_person_id = ? "
        "UNION SELECT 1 FROM detection_track_assignments WHERE capture_person_id = ? "
        "LIMIT 1",
        (person_id, person_id),
    ).fetchone()
    if in_use is not None:
        raise ValueError(
            f"Cannot delete person {person_id!r}: still referenced by existing "
            "detection or tracking data. Rename it instead if it was created "
            "by mistake."
        )
    with session:
        session.execute("DELETE FROM capture_persons WHERE id = ?", (person_id,))
