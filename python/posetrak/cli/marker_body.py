"""Marker body commands: import, list, show, export.

See docs/roadmap/features/extrinsics-improvements/
extrinsics-improvements-design.md, section 10 ("Marker body definitions:
format and storage"). Mirrors posetrak/cli/skeleton.py's structure and
command shape directly -- marker_body_definitions follows the exact same
storage convention as skeletons (SHA-256-of-content id, YAML blob), so
there is no reason for its CLI to look any different.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from posetrak.db.db import open_registry, open_session, resolve_id_prefix
from posetrak.db.manage_marker_body import import_marker_body, list_marker_bodies

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
    """Return (conn, label) using session DB if provided, else registry."""
    session_path = obj.get("session")
    if session_path:
        return _open_session(session_path), "session"
    return _open_registry(obj), "registry"


def _resolve_prefix(conn, table: str, prefix: str | None) -> str | None:
    if prefix is None:
        return None
    try:
        return resolve_id_prefix(conn, table, prefix)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


# ---------------------------------------------------------------------------
# marker-body group
# ---------------------------------------------------------------------------


@click.group("marker-body")
def marker_body_group() -> None:
    """Manage marker body definitions (portable calibration rigs, reusable
    marker-cluster objects -- see extrinsics-improvements-design.md section 10)."""


@marker_body_group.command("import")
@click.option("--file", "yaml_file", required=True, metavar="YAML_PATH",
              help="Path to the marker body definition YAML file")
@click.option("--global", "global_registry", is_flag=True, default=False,
              help="Also write to global registry")
@click.option("--name", default="", metavar="S", help="Human-readable name")
@click.option("--source", default="", metavar="S")
@click.option("--notes", default="", metavar="S")
@click.pass_obj
def marker_body_import(
    obj: dict,
    yaml_file: str,
    global_registry: bool,
    name: str,
    source: str,
    notes: str,
) -> None:
    """Import a marker body definition YAML file into the registry and/or a session DB."""
    yaml_path = Path(yaml_file)
    session_path = obj.get("session")

    if session_path is None and not global_registry:
        raise click.UsageError("Specify --session, --global, or both.")

    registry = None
    session = None
    body_id = None

    try:
        if global_registry:
            registry = _open_registry(obj)
        if session_path:
            session = _open_session(session_path)

        if registry is not None:
            try:
                body_id = import_marker_body(
                    registry, yaml_path, name=name or None, source=source or None,
                    notes=notes or None,
                )
            except Exception as exc:  # noqa: BLE001
                raise click.ClickException(f"Error importing marker body: {exc}") from exc

        if session is not None:
            try:
                body_id = import_marker_body(
                    session, yaml_path, name=name or None, source=source or None,
                    notes=notes or None,
                )
            except Exception as exc:  # noqa: BLE001
                raise click.ClickException(
                    f"Error importing marker body to session: {exc}"
                ) from exc

    finally:
        if registry is not None:
            registry.close()
        if session is not None:
            session.close()

    click.echo(f"marker_body_id: {body_id}")


@marker_body_group.command("list")
@click.pass_obj
def marker_body_list(obj: dict) -> None:
    """List marker body definitions -- from session DB if provided, otherwise from registry."""
    conn, _ = _open_conn(obj)
    try:
        rows = list_marker_bodies(conn)
    finally:
        conn.close()

    if not rows:
        if not obj["json_mode"]:
            click.echo("No marker body definitions registered.")
        return

    print_table(
        [dict(r) for r in rows],
        columns=["id", "name", "source", "created_at"],
        json_mode=obj["json_mode"],
    )


@marker_body_group.command("show")
@click.argument("marker_body_id", metavar="ID_OR_PREFIX")
@click.pass_obj
def marker_body_show(obj: dict, marker_body_id: str) -> None:
    """Show metadata for a marker body definition."""
    conn, _ = _open_conn(obj)
    try:
        resolved = _resolve_prefix(conn, "marker_body_definitions", marker_body_id)
        row = conn.execute(
            "SELECT id, name, source, created_at, notes "
            "FROM marker_body_definitions WHERE id = ?",
            (resolved,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise click.ClickException(f"Marker body not found: {marker_body_id}")

    print_record(
        {
            "id": row[0],
            "name": row[1] or "",
            "source": row[2] or "",
            "created_at": row[3] or "",
            "notes": row[4] or "",
        },
        json_mode=obj["json_mode"],
    )


@marker_body_group.command("export")
@click.argument("marker_body_id", metavar="ID_OR_PREFIX")
@click.option("--output", default="", metavar="PATH",
              help="Output file path (- for stdout, default: stdout)")
@click.pass_obj
def marker_body_export(obj: dict, marker_body_id: str, output: str) -> None:
    """Export a marker body definition's YAML to a file or stdout."""
    conn, _ = _open_conn(obj)
    try:
        resolved = _resolve_prefix(conn, "marker_body_definitions", marker_body_id)
        row = conn.execute(
            "SELECT yaml_content FROM marker_body_definitions WHERE id = ?", (resolved,)
        ).fetchone()
    finally:
        conn.close()

    if row is None or row[0] is None:
        raise click.ClickException(f"Marker body '{marker_body_id}' has no YAML content")

    yaml_content: str = row[0]

    if output and output != "-":
        Path(output).write_text(yaml_content, encoding="utf-8")
        click.echo(f"Exported marker body to {output}")
    else:
        sys.stdout.write(yaml_content)
