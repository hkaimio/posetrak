"""Skeleton commands: import, list, show, export, scale."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from posetrak.db.db import (
    open_registry,
    open_session,
    resolve_id_prefix,
)
from posetrak.db.manage_skeleton import (
    copy_skeleton,
    import_skeleton,
    import_skeleton_str,
    list_skeletons,
)
from posetrak.db.scale_skeleton import scale_skeleton_yaml, scaling_summary

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


def _open_session_required(obj: dict):
    path = obj.get("session")
    if not path:
        raise click.UsageError(
            "A session DB path is required. Use --session PATH or set $POSETRAK_SESSION_DB."
        )
    return _open_session(path)


def _resolve_prefix(conn, table: str, prefix: str | None) -> str | None:
    if prefix is None:
        return None
    try:
        return resolve_id_prefix(conn, table, prefix)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _open_conn(obj: dict):
    """Return (conn, label) using session DB if provided, else registry."""
    session_path = obj.get("session")
    if session_path:
        return _open_session(session_path), "session"
    else:
        return _open_registry(obj), "registry"


# ---------------------------------------------------------------------------
# skeleton group
# ---------------------------------------------------------------------------


@click.group("skeleton")
def skeleton_group() -> None:
    """Manage skeleton definitions."""


@skeleton_group.command("import")
@click.option("--file", "yaml_file", required=True, metavar="YAML_PATH",
              help="Path to the skeleton YAML file")
@click.option("--global", "global_registry", is_flag=True, default=False,
              help="Also write to global registry")
@click.option("--name", default="", metavar="S", help="Human-readable name")
@click.option("--person-label", default="", metavar="S")
@click.option("--source", default="", metavar="S")
@click.option("--parent-id", default="", metavar="SHA256",
              help="Parent skeleton ID (for lineage)")
@click.option("--notes", default="", metavar="S")
@click.pass_obj
def skeleton_import(
    obj: dict,
    yaml_file: str,
    global_registry: bool,
    name: str,
    person_label: str,
    source: str,
    parent_id: str,
    notes: str,
) -> None:
    """Import a skeleton YAML file into the registry and/or a session DB."""
    yaml_path = Path(yaml_file)
    session_path = obj.get("session")

    if session_path is None and not global_registry:
        raise click.UsageError("Specify --session, --global, or both.")

    registry = None
    session = None
    skeleton_id = None
    pid = parent_id or None

    try:
        if global_registry:
            registry = _open_registry(obj)

        if session_path:
            session = _open_session(session_path)

        # Ensure parent exists in both targets (copy across if needed).
        if pid is not None:
            for target, other, target_label in [
                (registry, session, "registry"),
                (session, registry, "session db"),
            ]:
                if target is None:
                    continue
                exists = target.execute(
                    "SELECT 1 FROM skeletons WHERE id = ?", (pid,)
                ).fetchone()
                if exists is None:
                    if other is None:
                        raise click.ClickException(
                            f"Parent skeleton '{pid[:12]}...' not found in {target_label}"
                        )
                    try:
                        copy_skeleton(other, target, pid)
                    except ValueError:
                        raise click.ClickException(
                            f"Parent skeleton '{pid[:12]}...' not found in either db"
                        )

        if registry is not None:
            try:
                skeleton_id = import_skeleton(
                    registry,
                    yaml_path,
                    name=name or None,
                    person_label=person_label or None,
                    source=source or None,
                    parent_id=pid,
                    notes=notes or None,
                )
            except Exception as exc:  # noqa: BLE001
                raise click.ClickException(f"Error importing skeleton: {exc}") from exc

        if session is not None:
            try:
                skeleton_id = import_skeleton(
                    session,
                    yaml_path,
                    name=name or None,
                    person_label=person_label or None,
                    source=source or None,
                    parent_id=pid,
                    notes=notes or None,
                )
            except Exception as exc:  # noqa: BLE001
                raise click.ClickException(
                    f"Error importing skeleton to session: {exc}"
                ) from exc

    finally:
        if registry is not None:
            registry.close()
        if session is not None:
            session.close()

    click.echo(f"skeleton_id: {skeleton_id}")


@skeleton_group.command("list")
@click.pass_obj
def skeleton_list(obj: dict) -> None:
    """List skeletons — from session DB if provided, otherwise from registry."""
    conn, _ = _open_conn(obj)
    try:
        rows = list_skeletons(conn)
    finally:
        conn.close()

    if not rows:
        if not obj["json_mode"]:
            click.echo("No skeletons registered.")
        return

    print_table(
        [dict(r) for r in rows],
        columns=["id", "name", "created_at"],
        json_mode=obj["json_mode"],
    )


@skeleton_group.command("show")
@click.argument("skeleton_id", metavar="ID_OR_PREFIX")
@click.pass_obj
def skeleton_show(obj: dict, skeleton_id: str) -> None:
    """Show metadata for a skeleton."""
    conn, _ = _open_conn(obj)
    try:
        resolved = _resolve_prefix(conn, "skeletons", skeleton_id)
        row = conn.execute(
            "SELECT id, name, person_label, source, parent_id, created_at, notes "
            "FROM skeletons WHERE id = ?",
            (resolved,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise click.ClickException(f"Skeleton not found: {skeleton_id}")

    print_record(
        {
            "id": row[0],
            "name": row[1] or "",
            "person_label": row[2] or "",
            "source": row[3] or "",
            "parent_id": row[4] or "",
            "created_at": row[5] or "",
            "notes": row[6] or "",
        },
        json_mode=obj["json_mode"],
    )


@skeleton_group.command("export")
@click.argument("skeleton_id", metavar="ID_OR_PREFIX")
@click.option("--output", default="", metavar="PATH",
              help="Output file path (- for stdout, default: stdout)")
@click.pass_obj
def skeleton_export(obj: dict, skeleton_id: str, output: str) -> None:
    """Export a skeleton YAML to a file or stdout."""
    conn, _ = _open_conn(obj)
    try:
        resolved = _resolve_prefix(conn, "skeletons", skeleton_id)
        row = conn.execute(
            "SELECT yaml_content FROM skeletons WHERE id = ?", (resolved,)
        ).fetchone()
    finally:
        conn.close()

    if row is None or row[0] is None:
        raise click.ClickException(f"Skeleton '{skeleton_id}' has no YAML content")

    yaml_content: str = row[0]

    if output and output != "-":
        Path(output).write_text(yaml_content, encoding="utf-8")
        click.echo(f"Exported skeleton to {output}")
    else:
        sys.stdout.write(yaml_content)


@skeleton_group.command("scale")
@click.argument("skeleton_id", metavar="ID")
@click.option("--femur", type=float, default=None, metavar="M")
@click.option("--shin", type=float, default=None, metavar="M")
@click.option("--upper-arm", type=float, default=None, metavar="M")
@click.option("--lower-arm", type=float, default=None, metavar="M")
@click.option("--torso-height", type=float, default=None, metavar="M")
@click.option("--shoulder-width", type=float, default=None, metavar="M")
@click.option("--name", default=None, metavar="STR",
              help="Save scaled skeleton to DB with this name")
@click.option("--output", default=None, metavar="PATH",
              help="Write YAML to this file (- for stdout)")
@click.option("--notes", default="", metavar="S")
@click.pass_obj
def skeleton_scale(
    obj: dict,
    skeleton_id: str,
    femur: float | None,
    shin: float | None,
    upper_arm: float | None,
    lower_arm: float | None,
    torso_height: float | None,
    shoulder_width: float | None,
    name: str | None,
    output: str | None,
    notes: str,
) -> None:
    """Create a scaled skeleton from body measurements.

    Exactly one of --name (save to DB) or --output (write YAML file) must be given.
    Use --output - to write to stdout.
    """
    if name is None and output is None:
        raise click.UsageError("Specify exactly one of --name or --output.")
    if name is not None and output is not None:
        raise click.UsageError("Specify only one of --name or --output, not both.")

    # Build measurements dict from CLI options.
    measurements: dict[str, float] = {}
    if femur is not None:
        measurements["femur"] = femur
    if shin is not None:
        measurements["shin"] = shin
    if upper_arm is not None:
        measurements["upper_arm"] = upper_arm
    if lower_arm is not None:
        measurements["lower_arm"] = lower_arm
    if torso_height is not None:
        measurements["torso_height"] = torso_height
    if shoulder_width is not None:
        measurements["shoulder_width"] = shoulder_width

    conn, _ = _open_conn(obj)
    try:
        resolved = _resolve_prefix(conn, "skeletons", skeleton_id)
        row = conn.execute(
            "SELECT id, yaml_content, name FROM skeletons WHERE id = ?",
            (resolved,),
        ).fetchone()
        if row is None:
            raise click.ClickException(f"Skeleton not found: {skeleton_id}")

        parent_id: str = row[0]
        original_yaml: str = row[1]
        parent_name: str = row[2] or parent_id[:12]

        # Apply scaling.
        try:
            scaled_yaml = scale_skeleton_yaml(original_yaml, measurements)
        except Exception as exc:  # noqa: BLE001
            raise click.ClickException(f"Error scaling skeleton: {exc}") from exc

        # Print scaling summary to stderr so that --output - writes only YAML to stdout.
        click.echo(scaling_summary(original_yaml, scaled_yaml, measurements), err=True)
        click.echo("", err=True)

        if output is not None:
            # Write to file or stdout.
            if output == "-":
                sys.stdout.write(scaled_yaml)
            else:
                Path(output).write_text(scaled_yaml, encoding="utf-8")
                click.echo(f"Exported scaled skeleton to {output}")
            return

        # --name: save to DB.
        skeleton_name = name or f"{parent_name}-scaled"
        try:
            new_id = import_skeleton_str(
                conn,
                scaled_yaml,
                name=skeleton_name,
                parent_id=parent_id,
                source=f"scaled from {parent_id}",
                notes=notes or None,
            )
        except Exception as exc:  # noqa: BLE001
            raise click.ClickException(f"Error storing skeleton: {exc}") from exc

        click.echo(f"skeleton_id: {new_id}")
        click.echo(f"name:        {skeleton_name}")

    finally:
        conn.close()
