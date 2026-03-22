"""manage_config.py — Registry CRUD for tracker configuration snapshots.

Tracker configurations capture the UKF and initialization parameters used
for a tracking run. Each configuration is identified by a UUID. Editing a
configuration creates a new row with ``parent_id`` pointing to the original,
preserving the full history of changes.
"""

from __future__ import annotations

import datetime
import sqlite3
import tomllib
from pathlib import Path

from posetrak.db.db import generate_id


def create_config_from_toml(
    registry: sqlite3.Connection,
    name: str,
    toml_path: Path,
    *,
    parent_id: str | None = None,
    notes: str | None = None,
) -> str:
    """Create a tracker_configs row populated from a posetrak TOML config file.

    Reads ``[tracking]``, ``[tracking.ukf]``, ``[tracking.initialization]``,
    and ``[processing]`` sections. Any parameter absent from the TOML is stored
    as ``NULL``.

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
        Optional free-text notes stored with the row.

    Returns
    -------
    str
        UUID of the newly created ``tracker_configs`` row.
    """
    with toml_path.open("rb") as fh:
        raw = tomllib.load(fh)

    tracking = raw.get("tracking", {})
    ukf = tracking.get("ukf", {})
    init = tracking.get("initialization", {})
    processing = raw.get("processing", {})

    config_id = generate_id()
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    with registry:
        registry.execute(
            "INSERT INTO tracker_configs "
            "(id, name, parent_id, created_at, "
            "alpha, beta, kappa, "
            "process_noise_std, measurement_noise_std, outlier_threshold, "
            "tracker_fps, "
            "ik_max_iterations, ik_tolerance, "
            "init_position_std, init_orientation_std, init_joint_std, init_velocity_std, "
            "min_cameras_for_init, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                config_id,
                name,
                parent_id,
                created_at,
                ukf.get("alpha"),
                ukf.get("beta"),
                ukf.get("kappa"),
                tracking.get("process_noise_std"),
                tracking.get("measurement_noise_std"),
                tracking.get("outlier_threshold"),
                processing.get("tracker_fps"),
                init.get("ik_max_iterations"),
                init.get("ik_tolerance"),
                init.get("init_position_std"),
                init.get("init_orientation_std"),
                init.get("init_joint_std"),
                init.get("init_velocity_std"),
                init.get("min_cameras_for_init"),
                notes,
            ),
        )

    return config_id


def edit_config(
    registry: sqlite3.Connection,
    config_id: str,
    *,
    alpha: float | None = None,
    beta: float | None = None,
    kappa: float | None = None,
    process_noise_std: float | None = None,
    measurement_noise_std: float | None = None,
    outlier_threshold: float | None = None,
    tracker_fps: float | None = None,
    ik_max_iterations: int | None = None,
    ik_tolerance: float | None = None,
    init_position_std: float | None = None,
    init_orientation_std: float | None = None,
    init_joint_std: float | None = None,
    init_velocity_std: float | None = None,
    min_cameras_for_init: int | None = None,
    notes: str | None = None,
) -> str:
    """Create a new tracker_configs row that overrides selected fields of an existing one.

    Copies all fields from the existing *config_id* row, then overrides any
    supplied non-``None`` keyword arguments. The new row's ``parent_id`` is
    set to *config_id*.

    Parameters
    ----------
    registry:
        Open connection to a posetrak registry database.
    config_id:
        ID of the existing ``tracker_configs`` row to derive from.
    alpha:
        UKF alpha parameter (overrides existing if provided).
    beta:
        UKF beta parameter (overrides existing if provided).
    kappa:
        UKF kappa parameter (overrides existing if provided).
    process_noise_std:
        Process noise standard deviation (overrides existing if provided).
    measurement_noise_std:
        Measurement noise standard deviation (overrides existing if provided).
    outlier_threshold:
        Mahalanobis outlier rejection threshold (overrides existing if provided).
    tracker_fps:
        Target tracker frame rate (overrides existing if provided).
    ik_max_iterations:
        Max IK solver iterations (overrides existing if provided).
    ik_tolerance:
        IK solver convergence tolerance (overrides existing if provided).
    init_position_std:
        Initial position uncertainty std (overrides existing if provided).
    init_orientation_std:
        Initial orientation uncertainty std (overrides existing if provided).
    init_joint_std:
        Initial joint angle uncertainty std (overrides existing if provided).
    init_velocity_std:
        Initial velocity uncertainty std (overrides existing if provided).
    min_cameras_for_init:
        Minimum cameras required for initialization (overrides existing if provided).
    notes:
        Notes for the new row (overrides existing if provided).

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
        raise ValueError(
            f"tracker_configs row not found: {config_id!r}"
        )

    def _pick(kwarg_val, col_name):
        """Return kwarg_val if not None, otherwise the existing row value."""
        if kwarg_val is not None:
            return kwarg_val
        return row[col_name]

    new_id = generate_id()
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    with registry:
        registry.execute(
            "INSERT INTO tracker_configs "
            "(id, name, parent_id, created_at, "
            "alpha, beta, kappa, "
            "process_noise_std, measurement_noise_std, outlier_threshold, "
            "tracker_fps, "
            "ik_max_iterations, ik_tolerance, "
            "init_position_std, init_orientation_std, init_joint_std, init_velocity_std, "
            "min_cameras_for_init, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new_id,
                row["name"],
                config_id,  # parent_id points to the source row
                created_at,
                _pick(alpha, "alpha"),
                _pick(beta, "beta"),
                _pick(kappa, "kappa"),
                _pick(process_noise_std, "process_noise_std"),
                _pick(measurement_noise_std, "measurement_noise_std"),
                _pick(outlier_threshold, "outlier_threshold"),
                _pick(tracker_fps, "tracker_fps"),
                _pick(ik_max_iterations, "ik_max_iterations"),
                _pick(ik_tolerance, "ik_tolerance"),
                _pick(init_position_std, "init_position_std"),
                _pick(init_orientation_std, "init_orientation_std"),
                _pick(init_joint_std, "init_joint_std"),
                _pick(init_velocity_std, "init_velocity_std"),
                _pick(min_cameras_for_init, "min_cameras_for_init"),
                _pick(notes, "notes"),
            ),
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
