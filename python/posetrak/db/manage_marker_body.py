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

    Upserts by ``(session_id, label)``: re-solving the same body under the
    same label overwrites its pose in place. This table represents
    "current believed pose," not a history — the same reasoning already
    established for the design doc's original ``scene_fiducial_markers``
    sketch (section 7's recalibration workflow: "existing marker → overwrite
    pose unconditionally").

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    session_id:
        ``mocap_sessions.id`` this pose belongs to.
    label:
        User-facing label for this body instance, unique within the
        session (e.g. ``"calib-box"``, ``"wall-tag-north"``).
    R, t:
        Body-local → world rotation (3×3) and translation (3,), any
        array-like accepted by ``np.asarray``.
    is_primary_anchor:
        True for the instrument that defined this session's world frame
        (``R = I, t = 0`` for that one row, by construction).

    Returns
    -------
    str
        The row's id — a freshly generated one for a new label, or the
        existing row's id if *label* already had one for this session.
    """
    from posetrak.db.db import generate_id

    R_blob = struct.pack("<9d", *np.asarray(R, dtype=np.float64).flatten())
    t_blob = struct.pack("<3d", *np.asarray(t, dtype=np.float64).flatten())
    updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    existing = session.execute(
        "SELECT id FROM scene_marker_bodies WHERE session_id = ? AND label = ?",
        (session_id, label),
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
                "(id, session_id, label, marker_body_definition_id, marker_type, dictionary, "
                " marker_id, marker_size, R, t, is_primary_anchor, "
                " source_extrinsic_calibration_id, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row_id, session_id, label, marker_body_definition_id, marker_type, dictionary,
                    marker_id, marker_size, R_blob, t_blob, int(is_primary_anchor),
                    source_extrinsic_calibration_id, updated_at,
                ),
            )

    return row_id


def list_scene_marker_bodies(session: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    """Return all ``scene_marker_bodies`` rows for *session_id*, ordered by label."""
    return session.execute(
        "SELECT * FROM scene_marker_bodies WHERE session_id = ? ORDER BY label",
        (session_id,),
    ).fetchall()


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
