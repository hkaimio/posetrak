# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Config commands: create, list, show, edit."""

from __future__ import annotations

from pathlib import Path

import click

from posetrak.db.db import open_registry, open_session
from posetrak.db.manage_config import (
    create_config_from_toml,
    edit_config,
    list_configs,
)

from posetrak.cli._output import print_table, print_record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_registry(obj: dict):
    path = obj["registry"]
    try:
        return open_registry(Path(path))
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(f"Error opening registry: {exc}") from exc


def _open_session(path: str):
    try:
        return open_session(Path(path))
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(f"Error opening session DB: {exc}") from exc


def _open_conn(obj: dict):
    """Return conn using session DB if provided, else registry."""
    session_path = obj.get("session")
    if session_path:
        return _open_session(session_path)
    return _open_registry(obj)


# ---------------------------------------------------------------------------
# config group
# ---------------------------------------------------------------------------


@click.group("config")
def config_group() -> None:
    """Manage tracker configurations."""


@config_group.command("create")
@click.option("--global", "global_registry", is_flag=True, default=False,
              help="Also write to global registry")
@click.option("--name", required=True, metavar="S",
              help="Name for this configuration snapshot")
@click.option("--from-toml", required=True, metavar="TOML_PATH",
              help="Path to the posetrak TOML config file")
@click.option("--notes", default="", metavar="S")
@click.pass_obj
def config_create(
    obj: dict, global_registry: bool, name: str, from_toml: str, notes: str
) -> None:
    """Create a tracker config snapshot from a TOML file."""
    session_path = obj.get("session")

    if session_path is None and not global_registry:
        raise click.UsageError("Specify --session, --global, or both.")

    config_id = None

    if global_registry:
        registry = _open_registry(obj)
        try:
            config_id = create_config_from_toml(
                registry,
                name,
                Path(from_toml),
                notes=notes or None,
            )
        except Exception as exc:  # noqa: BLE001
            registry.close()
            raise click.ClickException(f"Error creating config: {exc}") from exc
        finally:
            registry.close()

    if session_path is not None:
        session = _open_session(session_path)
        try:
            config_id = create_config_from_toml(
                session,
                name,
                Path(from_toml),
                notes=notes or None,
            )
        except Exception as exc:  # noqa: BLE001
            session.close()
            raise click.ClickException(f"Error creating config in session: {exc}") from exc
        finally:
            session.close()

    click.echo(f"tracker_config_id: {config_id}")


@config_group.command("list")
@click.option("--name", default="", metavar="S", help="Filter by name")
@click.pass_obj
def config_list(obj: dict, name: str) -> None:
    """List tracker configs from session DB (if provided) or registry."""
    conn = _open_conn(obj)
    try:
        rows = list_configs(conn, name=name or None)
    finally:
        conn.close()

    if not rows:
        if not obj["json_mode"]:
            click.echo("No tracker configs registered.")
        return

    print_table(
        [dict(r) for r in rows],
        columns=["id", "name", "created_at", "parent_id"],
        json_mode=obj["json_mode"],
    )


@config_group.command("show")
@click.argument("config_id", metavar="ID_OR_PREFIX")
@click.pass_obj
def config_show(obj: dict, config_id: str) -> None:
    """Show details for a tracker config."""
    from posetrak.db.db import resolve_id_prefix
    conn = _open_conn(obj)
    try:
        resolved = resolve_id_prefix(conn, "tracker_configs", config_id)
        row = conn.execute(
            "SELECT id, name, created_at, parent_id, notes "
            "FROM tracker_configs WHERE id = ?",
            (resolved,),
        ).fetchone()
    except ValueError as exc:
        conn.close()
        raise click.ClickException(str(exc)) from exc
    finally:
        conn.close()

    if row is None:
        raise click.ClickException(f"Config not found: {config_id}")

    print_record(
        {
            "id": row[0],
            "name": row[1] or "",
            "created_at": row[2] or "",
            "parent_id": row[3] or "",
            "notes": row[4] or "",
        },
        json_mode=obj["json_mode"],
    )


@config_group.command("edit")
@click.option("--global", "global_registry", is_flag=True, default=False)
@click.option("--id", "config_id", required=True, metavar="UUID",
              help="ID of the existing tracker_configs row to derive from")
@click.option("--alpha", type=float, default=None)
@click.option("--beta", type=float, default=None)
@click.option("--kappa", type=float, default=None)
@click.option("--process-noise-std", type=float, default=None)
@click.option("--process-noise-vel-std", type=float, default=None)
@click.option("--velocity-half-life-s", type=float, default=None)
@click.option("--measurement-noise-std", type=float, default=None)
@click.option("--outlier-threshold", type=float, default=None)
@click.option("--tracker-fps", type=float, default=None)
@click.option("--notes", default="", metavar="S")
@click.pass_obj
def config_edit(
    obj: dict,
    global_registry: bool,
    config_id: str,
    alpha: float | None,
    beta: float | None,
    kappa: float | None,
    process_noise_std: float | None,
    process_noise_vel_std: float | None,
    velocity_half_life_s: float | None,
    measurement_noise_std: float | None,
    outlier_threshold: float | None,
    tracker_fps: float | None,
    notes: str,
) -> None:
    """Derive a new tracker config by overriding selected fields."""
    session_path = obj.get("session")

    if session_path is None and not global_registry:
        raise click.UsageError("Specify --session, --global, or both.")

    new_id = None

    kwargs = dict(
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        process_noise_std=process_noise_std,
        process_noise_vel_std=process_noise_vel_std,
        velocity_half_life_s=velocity_half_life_s,
        measurement_noise_std=measurement_noise_std,
        outlier_threshold=outlier_threshold,
        tracker_fps=tracker_fps,
        notes=notes or None,
    )

    if global_registry:
        registry = _open_registry(obj)
        try:
            new_id = edit_config(registry, config_id, **kwargs)
        except ValueError as exc:
            registry.close()
            raise click.ClickException(str(exc)) from exc
        finally:
            registry.close()

    if session_path is not None:
        session = _open_session(session_path)
        try:
            new_id = edit_config(session, config_id, **kwargs)
        except ValueError as exc:
            session.close()
            raise click.ClickException(str(exc)) from exc
        finally:
            session.close()

    click.echo(f"new tracker_config_id: {new_id}")
