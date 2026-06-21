"""Pose commands: import, list."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import click

from posetrak.db.db import open_session, resolve_id_prefix
from posetrak.db.import_pose_json import import_pose_json

from posetrak.cli._output import print_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_session_required(obj: dict) -> sqlite3.Connection:
    path = obj.get("session")
    if not path:
        raise click.UsageError(
            "A session DB path is required. Use --session PATH or set $POSETRAK_SESSION_DB."
        )
    try:
        return open_session(Path(path))
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(f"Error opening session DB: {exc}") from exc


def _resolve(conn, table: str, prefix: str | None) -> str | None:
    if prefix is None:
        return None
    try:
        return resolve_id_prefix(conn, table, prefix)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _parse_camera_instances(values: tuple[str, ...] | None) -> str | dict[str, str] | None:
    if not values:
        return None
    values = list(values)
    if len(values) == 1 and "=" not in values[0]:
        return values[0]
    result: dict[str, str] = {}
    for v in values:
        if "=" not in v:
            raise click.UsageError(f"--camera-instance value {v!r} is ambiguous.")
        key, _, uuid = v.partition("=")
        if not key or not uuid:
            raise click.UsageError(f"Malformed --camera-instance value: {v!r}")
        result[key] = uuid
    return result


# ---------------------------------------------------------------------------
# pose group
# ---------------------------------------------------------------------------


@click.group("pose")
def pose_group() -> None:
    """Manage pose observations."""


@pose_group.command("list")
@click.option("--shot", default=None, metavar="UUID",
              help="Filter by captures.id")
@click.pass_obj
def pose_list(obj: dict, shot: str | None) -> None:
    """List pose observation sequences in a session database."""
    conn = _open_session_required(obj)
    try:
        q = (
            "SELECT id, shot_id, sync_config_id, time_start_s, time_end_s, pose_model"
            " FROM pose_observation_sequences"
        )
        params: list = []
        if shot:
            q += " WHERE shot_id = ?"
            params.append(shot)
        q += " ORDER BY time_start_s"
        rows = conn.execute(q, params).fetchall()
    except sqlite3.Error as exc:
        conn.close()
        raise click.ClickException(str(exc)) from exc
    finally:
        conn.close()

    if not rows:
        if not obj["json_mode"]:
            click.echo("(no pose sequences)")
        return

    print_table(
        [
            {
                "id": r[0],
                "shot_id": r[1],
                "sync_config_id": r[2],
                "time_start_s": f"{r[3]:.2f}" if r[3] is not None else "",
                "time_end_s": f"{r[4]:.2f}" if r[4] is not None else "",
                "pose_model": r[5] or "",
            }
            for r in rows
        ],
        columns=["id", "shot_id", "sync_config_id", "time_start_s", "time_end_s", "pose_model"],
        json_mode=obj["json_mode"],
    )


@pose_group.command("import")
@click.option("--shot", required=True, metavar="UUID", help="captures.id")
@click.option("--sync-config", required=True, metavar="UUID")
@click.option("--pose-dir", required=True, metavar="DIR")
@click.option("--camera-instance", multiple=True, metavar="SPEC")
@click.option("--person-id", "person_ids", type=int, multiple=True, metavar="N",
              help="Import only this person ID (repeatable). Default: import all persons.")
@click.option("--time-start", type=float, default=None)
@click.option("--time-end", type=float, default=None)
@click.option("--pose-model", default="", metavar="S")
@click.pass_obj
def pose_import(
    obj: dict,
    shot: str,
    sync_config: str,
    pose_dir: str,
    camera_instance: tuple[str, ...],
    person_ids: tuple[int, ...],
    time_start: float | None,
    time_end: float | None,
    pose_model: str,
) -> None:
    """Import 2-D pose observations from a pose directory."""
    cam_inst = _parse_camera_instances(camera_instance)
    if cam_inst is None:
        raise click.UsageError("At least one --camera-instance is required.")

    conn = _open_session_required(obj)
    try:
        shot_id = _resolve(conn, "captures", shot)
        sync_config_id = _resolve(conn, "sync_configs", sync_config)
        result = import_pose_json(
            conn,
            shot_id,
            sync_config_id,
            Path(pose_dir),
            cam_inst,
            person_ids=list(person_ids) if person_ids else None,
            time_start=time_start,
            time_end=time_end,
            pose_model=pose_model or "",
        )
    except Exception as exc:  # noqa: BLE001
        conn.close()
        raise click.ClickException(f"Error importing poses: {exc}") from exc
    finally:
        conn.close()

    click.echo(f"sequence_id: {result.sequence_id}")
    click.echo(f"n_observations: {result.n_observations}")
    if result.skipped_cameras:
        click.echo(f"skipped cameras: {', '.join(sorted(result.skipped_cameras))}")
