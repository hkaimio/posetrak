# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""manage_config.py — Registry CRUD for tracker configuration snapshots.

Tracker configurations capture the UKF and initialization parameters used
for a tracking run. Each configuration is identified by a UUID. Editing a
configuration creates a new row with ``parent_id`` pointing to the original,
preserving the full history of changes.

Column coverage for ``edit_config()``/``create_config_from_toml()`` is
derived at runtime from ``PRAGMA table_info(tracker_configs)`` rather than a
hardcoded parameter list, so a future schema migration that adds a column
needs no change here to stay correct -- see
docs/roadmap/features/configuration-improvements/config-improvements-design.md,
"Prerequisite fix" for the bug this replaced (a hardcoded column list that
had gone stale across migrations v22-v37 and silently dropped ~35 columns,
plus every tracker_config_stages row, on every edit).
"""

from __future__ import annotations

import datetime
import json
import sqlite3
import tomllib
from pathlib import Path

from posetrak.db.db import generate_id

# Columns tracker_configs carries that are not part of the generic
# copy/override machinery -- identity, lineage, and provenance, not tuning.
# is_named is excluded too: unlike a tuning value, it must NOT silently carry
# forward from the source row on edit_config() -- see that function's own
# handling below for why (config-improvements design doc, "editing a named
# config, and name collisions").
_NON_TUNING_COLUMNS = {"id", "name", "parent_id", "created_at", "is_named"}


def _tuning_columns(conn: sqlite3.Connection) -> list[str]:
    """Return tracker_configs' tuning-parameter column names, schema order.

    Excludes id/name/parent_id/created_at. Reads the live schema via
    PRAGMA rather than a hardcoded list, so a new migration column is
    picked up automatically.
    """
    rows = conn.execute("PRAGMA table_info(tracker_configs)").fetchall()
    return [r[1] for r in rows if r[1] not in _NON_TUNING_COLUMNS]


def _stage_columns(conn: sqlite3.Connection) -> list[str]:
    """Return tracker_config_stages' own columns, schema order, excluding
    the (tracker_config_id, group_name) key."""
    rows = conn.execute("PRAGMA table_info(tracker_config_stages)").fetchall()
    return [r[1] for r in rows if r[1] not in ("tracker_config_id", "group_name")]


def _encode(value: object) -> object:
    """JSON-encode list/dict override values; pass everything else through.

    tracker_configs stores several columns as JSON text (e.g.
    velocity_mode_camera_ids, pose_reg_joint_names, process_noise_vel_scopes).
    Rather than hardcode which columns need this, encode generically by
    Python type -- a caller passing a list/dict for any column gets the
    same treatment, present or future.
    """
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def create_config_from_toml(
    registry: sqlite3.Connection,
    name: str,
    toml_path: Path,
    *,
    parent_id: str | None = None,
    notes: str | None = None,
    is_named: bool = True,
) -> str:
    """Create a tracker_configs row populated from a posetrak TOML config file.

    Reads ``[tracking]``, ``[tracking.ukf]``, ``[tracking.initialization]``,
    and ``[processing]`` sections, flattened into one namespace (their key
    sets don't overlap in practice). Any tracker_configs tuning column not
    present under one of those sections is stored as ``NULL``. List/dict
    values (e.g. ``velocity_mode_camera_ids``) are JSON-encoded automatically.

    Parameters
    ----------
    registry:
        Open connection to a posetrak registry database.
    name:
        Human-readable name for this configuration snapshot.
    toml_path:
        Path to the posetrak ``.toml`` configuration file.
    parent_id:
        Optional ID of a parent ``tracker_configs`` row (for lineage tracking).
    notes:
        Optional free-text notes stored with the row; takes priority over
        any (currently nonexistent) ``notes`` key under the TOML sections
        read above.
    is_named:
        Whether this row should appear in the named-config picker (default
        ``True`` -- every caller of this function supplies an explicit,
        meaningful *name* already, unlike the auto-generated per-run
        snapshots ``edit_config()`` produces by default).

    Returns
    -------
    str
        UUID of the newly created ``tracker_configs`` row.
    """
    with toml_path.open("rb") as fh:
        raw = tomllib.load(fh)

    tracking = raw.get("tracking", {})
    flat: dict[str, object] = {
        **tracking,
        **tracking.get("ukf", {}),
        **tracking.get("initialization", {}),
        **raw.get("processing", {}),
    }
    flat.pop("ukf", None)
    flat.pop("initialization", None)
    if notes is not None:
        flat["notes"] = notes

    config_id = generate_id()
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    columns = _tuning_columns(registry)
    values = [_encode(flat.get(col)) for col in columns]

    with registry:
        registry.execute(
            "INSERT INTO tracker_configs (id, name, parent_id, created_at, is_named, "
            + ", ".join(columns) + ") VALUES ("
            + ", ".join(["?"] * (5 + len(columns))) + ")",
            (config_id, name, parent_id, created_at, 1 if is_named else 0, *values),
        )

    return config_id


def edit_config(
    registry: sqlite3.Connection,
    config_id: str,
    **overrides: object,
) -> str:
    """Create a new tracker_configs row that overrides selected fields of an existing one.

    Copies every tuning column from the existing *config_id* row, then
    overrides any column named in *overrides* with a non-``None`` value
    (``None`` means "keep the source row's value", matching this function's
    long-standing convention -- there is no way to explicitly clear a column
    to NULL this way). Also copies every ``tracker_config_stages`` row
    belonging to *config_id*, unchanged, so a hierarchical config's per-stage
    overrides survive an edit of its base row. The new row's ``parent_id``
    is set to *config_id*.

    ``name`` and ``is_named`` are deliberately **not** carried forward from
    the source row like the tuning columns are -- every edit defaults to the
    source's own name and an unnamed, auto-generated per-run snapshot
    (``is_named=0``) regardless of whether the source was itself a named
    template, matching the design doc's "editing a named config produces an
    unnamed working copy" rule. Pass ``is_named=True`` explicitly (e.g. from
    a "Save"/"Save as..." action) to produce a named row instead, and
    ``name=...`` to give it a different name than the source ("Save as...").

    Parameters
    ----------
    registry:
        Open connection to a posetrak registry database.
    config_id:
        ID of the existing ``tracker_configs`` row to derive from.
    **overrides:
        Any ``tracker_configs`` tuning-column name to override, e.g.
        ``alpha=0.5``, ``pose_reg_joint_names=["spine1", "spine2"]``, plus
        the two special, non-tuning keys ``is_named`` (bool) and ``name``
        (str, defaults to the source row's own name) described above. List/
        dict override values are JSON-encoded automatically. Unknown column
        names raise ``sqlite3.OperationalError`` when the INSERT runs (not
        validated ahead of time -- keeps this function from needing its own
        copy of the column list to validate against).

    Returns
    -------
    str
        UUID of the newly created row.

    Raises
    ------
    ValueError
        If *config_id* does not refer to an existing ``tracker_configs`` row.
    """
    row = registry.execute(
        "SELECT * FROM tracker_configs WHERE id = ?", (config_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"tracker_configs row not found: {config_id!r}")

    new_id = generate_id()
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    is_named = 1 if overrides.pop("is_named", False) else 0
    name = overrides.pop("name", None) or row["name"]

    columns = _tuning_columns(registry)
    values = [
        _encode(overrides[col]) if overrides.get(col) is not None else row[col]
        for col in columns
    ]

    with registry:
        registry.execute(
            "INSERT INTO tracker_configs (id, name, parent_id, created_at, is_named, "
            + ", ".join(columns) + ") VALUES ("
            + ", ".join(["?"] * (5 + len(columns))) + ")",
            (new_id, name, config_id, created_at, is_named, *values),
        )

        stage_columns = _stage_columns(registry)
        stage_rows = registry.execute(
            "SELECT * FROM tracker_config_stages WHERE tracker_config_id = ?",
            (config_id,),
        ).fetchall()
        for stage_row in stage_rows:
            registry.execute(
                "INSERT INTO tracker_config_stages (tracker_config_id, group_name, "
                + ", ".join(stage_columns) + ") VALUES ("
                + ", ".join(["?"] * (2 + len(stage_columns))) + ")",
                (new_id, stage_row["group_name"],
                 *[stage_row[c] for c in stage_columns]),
            )

    return new_id


def copy_config_to_session(
    registry: sqlite3.Connection,
    session: sqlite3.Connection,
    config_id: str,
) -> None:
    """Copy a tracker_config row from registry into a session DB.

    Uses INSERT OR IGNORE so calling this function multiple times with the
    same *config_id* is safe.

    Parameters
    ----------
    registry:
        Open connection to the posetrak registry database (source).
    session:
        Open connection to a posetrak session database (destination).
    config_id:
        ``tracker_configs.id`` UUID to copy.

    Raises
    ------
    ValueError
        If *config_id* does not exist in *registry*.
    """
    from posetrak.db.db import _copy_rows_if_missing
    _copy_rows_if_missing(registry, session, "tracker_configs", [config_id])


#: Fixed, well-known ID for the checked-in baseline config the
#: session/capture/trial default-config resolution chain terminates in when
#: no scope in the chain has set its own default. A literal, readable ID
#: (not a random UUID) so it's easy to recognize in logs/DB browsing and so
#: seeding is idempotent via INSERT OR IGNORE keyed on it.
BASELINE_CONFIG_ID = "factory-defaults"
BASELINE_CONFIG_NAME = "(factory defaults)"

#: Known-good tuning values the baseline config is seeded with, copied from
#: config id 3d0dd7fc-195d-4997-8218-1d17b5179e5d ("ukemit - tommi et al" in
#: a real, working 16-capture session recorded 2026-08) -- replaces an
#: earlier all-NULL baseline, which showed a wall of empty/zero fields the
#: first time a new user opened the tracker config editor rather than a
#: usable starting point. Deliberately excludes that config's
#: velocity_mode_camera_ids ([2]): it names a specific camera by index
#: within one particular capture's camera set, which has no general meaning
#: for a fresh install with a different (or no) camera layout. Columns not
#: listed here stay NULL, same as before this fix: SessionReader's own
#: hardcoded fallback constants apply (see load_tracker_config in
#: cpp/src/db/session_reader.cpp).
BASELINE_CONFIG_VALUES: dict[str, object] = {
    "process_noise_std": 0.3,
    "process_noise_vel_std": 1.0,
    "velocity_half_life_s": 0.25,
    "measurement_noise_std": 25.0,
    "outlier_threshold": 6.0,
    "tracker_fps": 120.0,
    "pose_noise_std": 13.0,
    "use_relative_observations": 1,
    "relative_min_confidence": 0.5,
    "process_noise_vel_gain_joint": 0.0,
    "process_noise_vel_ref_joint": 2.0,
    "process_noise_vel_gain_root": 0.0,
    "process_noise_vel_ref_root": 1.0,
    "pose_reg_joint_names": ["spine1", "spine2"],
    "pose_reg_equal_split_noise_std": 0.03,
    "pose_reg_rest_pose_noise_std": 0.15,
    "nis_feedback_window": 8,
    "nis_feedback_threshold": 1.5,
    "nis_feedback_max_multiplier": 3.0,
    "soft_limit_joint_names": ["upper_arm.L", "upper_arm.R"],
    "soft_limit_margin_rad": 0.1222,
    "soft_limit_noise_std": 0.03,
    "edited_kp_noise_std": 28.178,
    "cross_person_max_world_mm": 400.0,
    "cross_person_min_confidence": 0.5,
    "cross_person_max_n": 30,
}


def seed_baseline_tracker_config(conn: sqlite3.Connection) -> str:
    """Insert the checked-in baseline tracker_configs row if not already present.

    Idempotent (INSERT OR IGNORE keyed on the fixed BASELINE_CONFIG_ID), like
    manage_skeleton.seed_default_skeletons() which this mirrors. Tuning
    columns are populated from BASELINE_CONFIG_VALUES; any column not listed
    there is left NULL, so SessionReader's existing hardcoded fallback
    constants apply to it -- this row's job is to give a new user a sane,
    working starting point in the config editor, not to be the single
    source of truth for every tuning constant. is_named=1 so it appears in
    the named-config picker.

    Parameters
    ----------
    conn:
        Open connection to a posetrak registry or session database.

    Returns
    -------
    str
        BASELINE_CONFIG_ID, whether the row was just created or already existed.
    """
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    columns = _tuning_columns(conn)
    values = [_encode(BASELINE_CONFIG_VALUES.get(col)) for col in columns]
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO tracker_configs (id, name, parent_id, created_at, is_named, "
            + ", ".join(columns) + ") VALUES ("
            + ", ".join(["?"] * (5 + len(columns))) + ")",
            (BASELINE_CONFIG_ID, BASELINE_CONFIG_NAME, None, created_at, 1, *values),
        )
    return BASELINE_CONFIG_ID


def refresh_baseline_tracker_config(conn: sqlite3.Connection) -> None:
    """Update an already-seeded baseline row's tuning values in place.

    seed_baseline_tracker_config() is INSERT OR IGNORE, so an existing
    registry/session created before BASELINE_CONFIG_VALUES was populated
    keeps its all-NULL row forever -- this backfills it. Only touches the
    row matching BASELINE_CONFIG_ID; safe to call unconditionally (no-op if
    the row doesn't exist yet, matching seed's own idempotency).
    """
    columns = _tuning_columns(conn)
    values = [_encode(BASELINE_CONFIG_VALUES.get(col)) for col in columns]
    with conn:
        conn.execute(
            "UPDATE tracker_configs SET " + ", ".join(f"{c} = ?" for c in columns)
            + " WHERE id = ?",
            (*values, BASELINE_CONFIG_ID),
        )


def name_existing_config(conn: sqlite3.Connection, config_id: str, name: str) -> None:
    """Give an already-existing tracker_configs row a name, in place.

    Deliberately **not** a copy-on-write like edit_config() -- this is the
    "Save config" action on a tracking run's info pane, which names the
    *exact* row that run's tracker_config_id already points to (so its
    provenance link is unchanged, not superseded by a new row) rather than
    deriving a new one. No tuning value changes, so the immutability
    concern edit_config()'s copy-on-write exists for doesn't apply here.

    Parameters
    ----------
    conn:
        Open connection to a posetrak registry or session database.
    config_id:
        ID of the existing ``tracker_configs`` row to name.
    name:
        The name to give it (overwrites any existing name).

    Raises
    ------
    ValueError
        If *config_id* does not refer to an existing ``tracker_configs`` row.
    """
    with conn:
        cur = conn.execute(
            "UPDATE tracker_configs SET name = ?, is_named = 1 WHERE id = ?",
            (name, config_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"tracker_configs row not found: {config_id!r}")


def list_configs(
    registry: sqlite3.Connection,
    *,
    name: str | None = None,
) -> list[sqlite3.Row]:
    """Return tracker_configs rows from the registry.

    Parameters
    ----------
    registry:
        Open connection to a posetrak registry database.
    name:
        If provided, filter rows to those whose ``name`` column matches
        exactly.

    Returns
    -------
    list[sqlite3.Row]
        Matching rows ordered by ``created_at`` ascending.
    """
    if name is not None:
        return registry.execute(
            "SELECT * FROM tracker_configs WHERE name = ? ORDER BY created_at",
            (name,),
        ).fetchall()
    return registry.execute(
        "SELECT * FROM tracker_configs ORDER BY created_at"
    ).fetchall()


# ---------------------------------------------------------------------------
# Default-config resolution chain (config-improvements design doc, phase 3)
# ---------------------------------------------------------------------------


def resolve_default_tracker_config(
    session: sqlite3.Connection,
    *,
    trial_id: str | None = None,
    capture_id: str | None = None,
) -> str:
    """Resolve the effective default ``tracker_configs`` id for a trial or capture.

    Resolution order: the trial's own ``default_tracker_config_id`` (if
    *trial_id* given), else its capture's, else the checked-in baseline
    config (:data:`BASELINE_CONFIG_ID`) -- seeded into *session* on demand if
    not already present, so this always resolves to a row that actually
    exists in *session* even for a session DB the baseline was never
    otherwise copied into (see the design doc's self-containment note).

    Parameters
    ----------
    session:
        Open connection to a posetrak session (or registry) database.
    trial_id:
        Trial to resolve for. Falls through to its own capture if the trial
        has no default set.
    capture_id:
        Capture to resolve for directly. Ignored if *trial_id* is given and
        that trial has its own default set; otherwise used as the trial's
        fallback if *trial_id* was given, or resolved directly if not.

    Returns
    -------
    str
        A ``tracker_configs.id`` guaranteed to exist in *session*.

    Raises
    ------
    ValueError
        If *trial_id* is given but no such trial exists, or if neither
        *trial_id* nor *capture_id* is given.
    """
    if trial_id is not None:
        row = session.execute(
            "SELECT default_tracker_config_id, capture_id FROM trials WHERE id = ?",
            (trial_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"trial not found: {trial_id!r}")
        if row["default_tracker_config_id"]:
            return row["default_tracker_config_id"]
        capture_id = row["capture_id"]
    elif capture_id is None:
        raise ValueError("resolve_default_tracker_config: must supply trial_id or capture_id")

    if capture_id is not None:
        crow = session.execute(
            "SELECT default_tracker_config_id FROM captures WHERE id = ?",
            (capture_id,),
        ).fetchone()
        if crow is not None and crow["default_tracker_config_id"]:
            return crow["default_tracker_config_id"]

    seed_baseline_tracker_config(session)
    return BASELINE_CONFIG_ID


def set_default_tracker_config(
    session: sqlite3.Connection,
    config_id: str,
    *,
    trial_id: str | None = None,
    capture_id: str | None = None,
) -> None:
    """Repoint a trial's or capture's ``default_tracker_config_id``.

    Exactly one of *trial_id*/*capture_id* should be given -- this never
    touches the other level, matching the design doc's "editing a
    capture-level default never silently changes what a trial-level
    override resolves to, and vice versa."

    Parameters
    ----------
    session:
        Open connection to a posetrak session (or registry) database.
    config_id:
        The ``tracker_configs.id`` to set as the default.
    trial_id:
        Trial to update, if repointing a trial-level default.
    capture_id:
        Capture to update, if repointing a capture-level default.

    Raises
    ------
    ValueError
        If neither or both of *trial_id*/*capture_id* are given.
    """
    if (trial_id is None) == (capture_id is None):
        raise ValueError(
            "set_default_tracker_config: supply exactly one of trial_id/capture_id"
        )
    with session:
        if trial_id is not None:
            session.execute(
                "UPDATE trials SET default_tracker_config_id = ? WHERE id = ?",
                (config_id, trial_id),
            )
        else:
            session.execute(
                "UPDATE captures SET default_tracker_config_id = ? WHERE id = ?",
                (config_id, capture_id),
            )
