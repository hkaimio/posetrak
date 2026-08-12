"""manage_marker_body.py — Registry CRUD for marker body definitions, and
session-scoped read/write helpers for solved marker-body poses.

See docs/roadmap/features/extrinsics-improvements/
extrinsics-improvements-design.md, section 10 ("Marker body definitions:
format and storage").

``marker_body_definitions`` follows manage_skeleton.py's exact pattern:
the primary key is the SHA-256 hex digest of the YAML content, which
makes import idempotent (re-importing the same YAML returns the existing
ID without creating a duplicate row) -- not a new convention, deliberately
the same one.
"""

from __future__ import annotations

import datetime
import hashlib
import sqlite3
import struct
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# marker_body_definitions (registry, embedded into every session DB)
# ---------------------------------------------------------------------------


def import_marker_body(
    registry: sqlite3.Connection,
    yaml_path: Path,
    *,
    name: str | None = None,
    source: str | None = None,
    notes: str | None = None,
) -> str:
    """Import a marker body definition YAML file into the registry.

    The body ID is the SHA-256 hex digest of the YAML file content. If a
    body with the same ID already exists, the function returns the
    existing ID without reinserting any rows.

    Parameters
    ----------
    registry:
        Open connection to a posetrak registry database.
    yaml_path:
        Path to the marker body definition YAML file to import.
    name:
        Human-readable name. Defaults to ``yaml_path.stem`` if ``None``.
    source:
        Optional provenance string (e.g. ``"orbit_video_self_calibration"``
        or a path/description of origin).
    notes:
        Optional free-text notes stored with the row.

    Returns
    -------
    str
        SHA-256 hex ID (64 characters) of the row — either the newly
        created row or the pre-existing one.

    Raises
    ------
    FileNotFoundError
        If *yaml_path* does not exist.
    """
    yaml_content = yaml_path.read_text(encoding="utf-8")
    resolved_name = name if name is not None else yaml_path.stem
    return import_marker_body_str(registry, yaml_content, name=resolved_name, source=source, notes=notes)


def import_marker_body_str(
    db: sqlite3.Connection,
    yaml_content: str,
    *,
    name: str | None = None,
    source: str | None = None,
    notes: str | None = None,
) -> str:
    """Import a marker body definition from a YAML string into *db*
    (registry or session).

    Identical to :func:`import_marker_body` but accepts YAML content
    directly instead of a file path. Idempotent (content-addressed id).
    """
    body_id = hashlib.sha256(yaml_content.encode("utf-8")).hexdigest()

    existing = db.execute(
        "SELECT id FROM marker_body_definitions WHERE id = ?", (body_id,)
    ).fetchone()
    if existing is not None:
        return body_id

    resolved_name = name or body_id[:12]
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    with db:
        db.execute(
            "INSERT INTO marker_body_definitions "
            "(id, name, yaml_content, source, created_at, notes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (body_id, resolved_name, yaml_content, source, created_at, notes),
        )

    return body_id


def copy_marker_body_to_session(
    registry: sqlite3.Connection,
    session: sqlite3.Connection,
    marker_body_definition_id: str,
) -> None:
    """Copy a ``marker_body_definitions`` row from registry into a session DB.

    Uses INSERT OR IGNORE (via the shared ``_copy_rows_if_missing`` helper),
    so calling this multiple times with the same id is safe. Mirrors
    ``manage_skeleton.copy_skeleton_to_session``.

    Raises
    ------
    ValueError
        If *marker_body_definition_id* does not exist in *registry*.
    """
    from posetrak.db.db import _copy_rows_if_missing
    _copy_rows_if_missing(registry, session, "marker_body_definitions", [marker_body_definition_id])


def list_marker_bodies(registry: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all ``marker_body_definitions`` rows, ordered by creation time."""
    return registry.execute(
        "SELECT * FROM marker_body_definitions ORDER BY created_at"
    ).fetchall()


# ---------------------------------------------------------------------------
# scene_marker_bodies (session-scoped solved poses)
# ---------------------------------------------------------------------------


def upsert_scene_marker_body(
    session: sqlite3.Connection,
    session_id: str,
    label: str,
    R: np.ndarray,
    t: np.ndarray,
    *,
    group_name: str | None = None,
    marker_body_definition_id: str | None = None,
    marker_type: str | None = None,
    dictionary: str | None = None,
    marker_id: str | None = None,
    marker_size: float | None = None,
    is_primary_anchor: bool = False,
    source_extrinsic_calibration_id: str | None = None,
) -> str:
    """Insert or update a solved marker-body pose for this session.

    Exactly one of *marker_body_definition_id* (a real multi-marker rig)
    or the ``(marker_type, dictionary, marker_id)`` trio (a lone scattered
    tag — no bespoke YAML needed for a single marker's trivial geometry)
    should be given; this is not enforced here, matching how
    ``scene_marker_bodies``' schema itself leaves both nullable rather
    than modelling them as a SQL-level exclusive-or.

    Upserts by ``(session_id, group_name, label)``: re-solving the same
    body under the same group and label overwrites its pose in place.
    This table represents "current believed pose," not a history — the
    same reasoning already established for the design doc's original
    ``scene_fiducial_markers`` sketch (section 7's recalibration workflow:
    "existing marker → overwrite pose unconditionally").

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    session_id:
        ``mocap_sessions.id`` this pose belongs to.
    label:
        User-facing label for this body instance, unique within
        ``(session_id, group_name)`` (e.g. ``"calib-box"``,
        ``"wall-tag-north"``).
    R, t:
        Body-local → world rotation (3×3) and translation (3,), any
        array-like accepted by ``np.asarray``.
    group_name:
        User-chosen name grouping every marker anchored together in one
        physical space (e.g. ``"room7"``) — lets a later capture in the
        same room select just that group instead of every stored marker
        in the session (design doc section 9). ``None``/omitted means
        ungrouped, stored as ``''`` (not NULL — see session_schema.sql's
        column comment for why this matters for the uniqueness
        constraint). Two different groups may reuse the same *label*
        without colliding.
    is_primary_anchor:
        True for the instrument that defined this session's world frame
        (``R = I, t = 0`` for that one row, by construction).

    Returns
    -------
    str
        The row's id — a freshly generated one for a new (group, label)
        pair, or the existing row's id if that pair already had one for
        this session.
    """
    from posetrak.db.db import generate_id

    group_name = group_name or ""
    R_blob = struct.pack("<9d", *np.asarray(R, dtype=np.float64).flatten())
    t_blob = struct.pack("<3d", *np.asarray(t, dtype=np.float64).flatten())
    updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    existing = session.execute(
        "SELECT id FROM scene_marker_bodies WHERE session_id = ? AND group_name = ? AND label = ?",
        (session_id, group_name, label),
    ).fetchone()

    with session:
        if existing is not None:
            row_id = existing["id"]
            session.execute(
                "UPDATE scene_marker_bodies SET "
                "marker_body_definition_id = ?, marker_type = ?, dictionary = ?, "
                "marker_id = ?, marker_size = ?, R = ?, t = ?, is_primary_anchor = ?, "
                "source_extrinsic_calibration_id = ?, updated_at = ? "
                "WHERE id = ?",
                (
                    marker_body_definition_id, marker_type, dictionary, marker_id, marker_size,
                    R_blob, t_blob, int(is_primary_anchor), source_extrinsic_calibration_id,
                    updated_at, row_id,
                ),
            )
        else:
            row_id = generate_id()
            session.execute(
                "INSERT INTO scene_marker_bodies "
                "(id, session_id, label, group_name, marker_body_definition_id, marker_type, "
                " dictionary, marker_id, marker_size, R, t, is_primary_anchor, "
                " source_extrinsic_calibration_id, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row_id, session_id, label, group_name, marker_body_definition_id, marker_type,
                    dictionary, marker_id, marker_size, R_blob, t_blob, int(is_primary_anchor),
                    source_extrinsic_calibration_id, updated_at,
                ),
            )

    return row_id


def list_scene_marker_bodies(session: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    """Return all ``scene_marker_bodies`` rows for *session_id*, ordered by
    group name then label -- across every group, e.g. for the manager
    dialog's "everything, for troubleshooting" view."""
    return session.execute(
        "SELECT * FROM scene_marker_bodies WHERE session_id = ? ORDER BY group_name, label",
        (session_id,),
    ).fetchall()


def list_scene_marker_group_names(session: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    """Return every named group (``group_name != ''``) this session has
    loadable scene markers under, with a marker count and the most recent
    update time -- for populating a "pick which room's markers to load"
    UI (design doc section 9; the GUI's ``_SceneMarkerGroupPickerDialog``
    and the CLI's `extrinsics scene-marker groups`).

    Only counts loadable scattered-tag rows (``marker_body_definition_id
    IS NULL`` and dictionary/marker_id/marker_size all set), matching
    ``list_scene_marker_bodies_by_group``'s own filter -- a rig's own
    anchor row wouldn't be loaded from this group anyway, so counting it
    here would overstate how many tags actually come back.

    Rows still stuck at the ungrouped default (``group_name == ''``,
    written before this feature existed or via a caller that didn't
    bother naming a group) are deliberately excluded -- there's no name
    to show for them; ``list_scene_marker_bodies_by_group(..., None)``
    is how to reach them.
    """
    return session.execute(
        "SELECT group_name, COUNT(*) AS n_markers, MAX(updated_at) AS last_updated "
        "FROM scene_marker_bodies "
        "WHERE session_id = ? AND group_name != '' AND marker_body_definition_id IS NULL "
        "AND dictionary IS NOT NULL AND marker_id IS NOT NULL AND marker_size IS NOT NULL "
        "GROUP BY group_name ORDER BY last_updated DESC",
        (session_id,),
    ).fetchall()


def list_scene_marker_bodies_by_group(
    session: sqlite3.Connection, session_id: str, group_name: str | None,
) -> list[sqlite3.Row]:
    """Return the loadable scattered-tag rows for one named group (or the
    ungrouped default when *group_name* is ``None``/``''``) -- the
    filtered counterpart to ``list_scene_marker_bodies``, used by "From
    Scene Markers…"/`reanchor` once a group has been chosen.

    Same loadable-row filter as ``list_scene_marker_group_names`` uses to
    count: a rig's own anchor row is never returned here (design doc
    section 9 -- see status.md's 2026-08-12 "rig markers leaking into
    scene markers" entry for why that matters).
    """
    return session.execute(
        "SELECT * FROM scene_marker_bodies "
        "WHERE session_id = ? AND group_name = ? AND marker_body_definition_id IS NULL "
        "AND dictionary IS NOT NULL AND marker_id IS NOT NULL AND marker_size IS NOT NULL "
        "ORDER BY label",
        (session_id, group_name or ""),
    ).fetchall()


def delete_scene_marker_body(
    session: sqlite3.Connection, session_id: str, label: str, group_name: str | None = None,
) -> bool:
    """Delete one ``scene_marker_bodies`` row by ``(session_id, group_name,
    label)`` -- *group_name* defaults to the ungrouped ``''``, so callers
    that never adopted named groups keep working unchanged.

    For pruning stale/wrong entries -- e.g. a portable calibration rig's
    own anchor row, or a scattered tag whose physical position has moved
    -- rather than mutating a row in place, since there's no legitimate
    "correct" R/t to replace a stale one with from this function's own
    knowledge (see design doc section 9; ``upsert_scene_marker_body``
    already covers the "resolve it again, same label" case).

    Returns
    -------
    bool
        True if a row was actually deleted, False if no row matched.
    """
    with session:
        cur = session.execute(
            "DELETE FROM scene_marker_bodies WHERE session_id = ? AND group_name = ? AND label = ?",
            (session_id, group_name or "", label),
        )
    return cur.rowcount > 0


def read_scene_marker_body_pose(row: sqlite3.Row) -> tuple[np.ndarray, np.ndarray]:
    """Decode a ``scene_marker_bodies`` row's R/t BLOBs into numpy arrays.

    Same little-endian float64 encoding ``import_extrinsics.py`` already
    uses for ``extrinsic_entries.R``/``.t`` (row-major 3×3 for R).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(R, t)`` — ``R`` shape (3, 3), ``t`` shape (3,).
    """
    R = np.array(struct.unpack("<9d", bytes(row["R"])), dtype=np.float64).reshape(3, 3)
    t = np.array(struct.unpack("<3d", bytes(row["t"])), dtype=np.float64)
    return R, t
