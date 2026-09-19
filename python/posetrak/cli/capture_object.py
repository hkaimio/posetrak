# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""capture object commands: add, list, rename, rm.

A capture object is a rigid prop of one capture, the counterpart of a capture
person. It names a marker body definition, which fixes the prop's geometry.
Adds an ``object`` group to the existing ``capture`` group (defined in
``posetrak.cli.session``).
"""

from __future__ import annotations

from pathlib import Path

import click

from posetrak.cli._output import print_table
from posetrak.cli.session import _open_session_required, _resolve, capture_group
from posetrak.db.db import open_registry, resolve_id_prefix
from posetrak.db.manage_capture_object import (
    create_capture_object,
    delete_capture_object,
    list_capture_objects,
    rename_capture_object,
)
from posetrak.db.manage_marker_body import copy_marker_body_to_session


@capture_group.group("object")
def object_group() -> None:
    """Manage the tracked objects (rigid props) of a capture."""


def _marker_body_id(obj: dict, conn, prefix: str) -> str:
    """Resolve a marker body of this session by ID prefix, else copy it in from the registry."""
    rows = conn.execute(
        "SELECT id FROM marker_body_definitions WHERE id LIKE ? || '%'", (prefix,)
    ).fetchall()
    if len(rows) == 1:
        return rows[0][0]
    if len(rows) > 1:
        raise click.ClickException(f"Marker body prefix '{prefix}' is ambiguous in this session.")
    try:
        registry = open_registry(Path(obj["registry"]))
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(
            f"No marker body '{prefix}' in this session, and the registry could not be opened: {exc}"
        ) from exc
    try:
        body_id = resolve_id_prefix(registry, "marker_body_definitions", prefix)
        copy_marker_body_to_session(registry, conn, body_id)
    except ValueError as exc:
        raise click.ClickException(f"Marker body '{prefix}': {exc}") from exc
    finally:
        registry.close()
    return body_id


def _require_free_name(conn, capture_id: str, name: str) -> None:
    """`detect run --object` and `track run-persons --object` find an object by name."""
    if any(o["name"] == name for o in list_capture_objects(conn, capture_id)):
        raise click.ClickException(f"This capture already has an object named '{name}'.")


@object_group.command("add")
@click.option("--capture", required=True, metavar="UUID", help="captures.id (prefix accepted).")
@click.option("--name", required=True, metavar="S", help="Name of this prop, e.g. bokken-A.")
@click.option(
    "--marker-body", required=True, metavar="ID",
    help="Marker body definition of the prop (prefix accepted). Taken from the session, "
         "else copied in from the registry.",
)
@click.option("--notes", default=None, metavar="S")
@click.pass_obj
def object_add(obj: dict, capture: str, name: str, marker_body: str, notes: str | None) -> None:
    """Add a tracked object to a capture."""
    conn = _open_session_required(obj)
    try:
        capture_id = _resolve(conn, "captures", capture)
        _require_free_name(conn, capture_id, name)
        body_id = _marker_body_id(obj, conn, marker_body)
        object_id = create_capture_object(conn, capture_id, name, body_id, notes=notes)
    finally:
        conn.close()
    click.echo(f"capture_object_id: {object_id}")


@object_group.command("list")
@click.option("--capture", default=None, metavar="UUID", help="Only this capture's objects.")
@click.pass_obj
def object_list(obj: dict, capture: str | None) -> None:
    """List tracked objects."""
    conn = _open_session_required(obj)
    try:
        if capture is not None:
            rows = list_capture_objects(conn, _resolve(conn, "captures", capture))
        else:
            rows = conn.execute("SELECT * FROM capture_objects ORDER BY capture_id, name").fetchall()
        records = [dict(r) for r in rows]
    finally:
        conn.close()
    print_table(
        records, ["id", "capture_id", "name", "marker_body_definition_id", "notes"],
        json_mode=obj.get("json_mode", False),
    )


@object_group.command("rename")
@click.argument("object_id", metavar="ID_OR_PREFIX")
@click.argument("name")
@click.pass_obj
def object_rename(obj: dict, object_id: str, name: str) -> None:
    """Rename a tracked object."""
    conn = _open_session_required(obj)
    try:
        resolved = _resolve(conn, "capture_objects", object_id)
        capture_id = conn.execute(
            "SELECT capture_id FROM capture_objects WHERE id = ?", (resolved,)
        ).fetchone()[0]
        _require_free_name(conn, capture_id, name)
        rename_capture_object(conn, resolved, name)
    finally:
        conn.close()
    click.echo(f"Renamed {resolved} to '{name}'.")


@object_group.command("rm")
@click.argument("object_id", metavar="ID_OR_PREFIX")
@click.pass_obj
def object_rm(obj: dict, object_id: str) -> None:
    """Delete a tracked object that no detection or tracking run refers to."""
    conn = _open_session_required(obj)
    try:
        resolved = _resolve(conn, "capture_objects", object_id)
        try:
            delete_capture_object(conn, resolved)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    finally:
        conn.close()
    click.echo(f"Deleted {resolved}.")
