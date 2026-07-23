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


def seed_baseline_tracker_config(conn: sqlite3.Connection) -> str:
    """Insert the checked-in baseline tracker_configs row if not already present.

    Idempotent (INSERT OR IGNORE keyed on the fixed BASELINE_CONFIG_ID), like
    manage_skeleton.seed_default_skeletons() which this mirrors. Every tuning
    column is left NULL, so SessionReader's existing hardcoded fallback
    constants apply -- this row is not a *value* to reach for, it's just
    something real for the default-config resolution chain (session ->
    capture -> trial) to terminate in rather than an empty dialog. is_named=1
    so it appears in the named-config picker.

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
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO tracker_configs (id, name, parent_id, created_at, is_named) "
            "VALUES (?, ?, NULL, ?, 1)",
            (BASELINE_CONFIG_ID, BASELINE_CONFIG_NAME, created_at),
        )
    return BASELINE_CONFIG_ID


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
