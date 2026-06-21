"""track.py — CLI commands for listing and exporting tracking runs."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from posetrak.cli._output import fail, print_record, print_table
from posetrak.db.db import open_session, resolve_id_prefix
from posetrak.export.bvh import export_bvh
from posetrak.export.gltf import export_gltf
from posetrak.export.usd import export_usd


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------


@click.group("track")
def track_group() -> None:
    """List and export tracking runs."""


# ---------------------------------------------------------------------------
# track list
# ---------------------------------------------------------------------------


@track_group.command("list")
@click.option("--sequence", default=None, help="Filter by observation_sequence_id (prefix accepted).")
@click.pass_context
def cmd_list(ctx: click.Context, sequence: str | None) -> None:
    """List tracking runs in the session."""
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'track list'.")

    try:
        conn = open_session(Path(session_path))
    except (FileNotFoundError, ValueError) as exc:
        fail(str(exc))

    sequence_id: str | None = None
    if sequence is not None:
        try:
            sequence_id = resolve_id_prefix(conn, "observation_sequences", sequence)
        except ValueError as exc:
            fail(str(exc))

    query = (
        "SELECT id, skeleton_id, ran_at, posetrak_version "
        "FROM tracking_runs"
    )
    params: list = []
    if sequence_id is not None:
        query += " WHERE observation_sequence_id = ?"
        params.append(sequence_id)
    query += " ORDER BY ran_at DESC"

    rows = [dict(r) for r in conn.execute(query, params).fetchall()]
    conn.close()

    if not rows:
        sys.exit(0)

    columns = ["id", "skeleton_id", "ran_at", "posetrak_version"]
    print_table(rows, columns, json_mode=ctx.obj["json_mode"])


# ---------------------------------------------------------------------------
# track show
# ---------------------------------------------------------------------------


@track_group.command("show")
@click.argument("run_id")
@click.pass_context
def cmd_show(ctx: click.Context, run_id: str) -> None:
    """Show details of a single tracking run."""
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'track show'.")

    try:
        conn = open_session(Path(session_path))
    except (FileNotFoundError, ValueError) as exc:
        fail(str(exc))

    try:
        full_id = resolve_id_prefix(conn, "tracking_runs", run_id)
    except ValueError as exc:
        fail(str(exc))

    row = conn.execute(
        "SELECT id, observation_sequence_id, tracker_config_id, skeleton_id, "
        "       extrinsic_calibration_id, sync_config_id, ran_at, "
        "       posetrak_version, active_camera_ids, marker_names, notes "
        "FROM tracking_runs WHERE id = ?",
        (full_id,),
    ).fetchone()
    conn.close()

    if row is None:
        fail(f"Tracking run '{full_id}' not found.")

    print_record(dict(row), json_mode=ctx.obj["json_mode"])


# ---------------------------------------------------------------------------
# track export
# ---------------------------------------------------------------------------


@track_group.group("export")
def export_group() -> None:
    """Export tracking run data to an animation format."""


def _resolve_run(session_path: str, run_id: str) -> str:
    """Resolve a run_id prefix to a full ID."""
    conn = open_session(Path(session_path))
    try:
        return resolve_id_prefix(conn, "tracking_runs", run_id)
    except ValueError as exc:
        fail(str(exc))
    finally:
        conn.close()


_EXPORT_OPTIONS = [
    click.option("--smoothed", is_flag=True, default=False, help="Use RTS-smoothed output."),
    click.option("--fps", default=None, type=float, help="Override output frame rate."),
    click.option(
        "--units",
        default="m",
        show_default=True,
        type=click.Choice(["m", "cm"]),
        help="Output length units.",
    ),
    click.option(
        "--coord",
        default="yup",
        show_default=True,
        type=click.Choice(["yup", "zup"]),
        help="Output coordinate convention.",
    ),
    click.option("--person-id", default=0, show_default=True, type=int, help="Person index."),
]


def _add_export_options(cmd):
    for opt in reversed(_EXPORT_OPTIONS):
        cmd = opt(cmd)
    return cmd


@export_group.command("bvh")
@click.argument("run_id")
@click.argument("output")
@_add_export_options
@click.pass_context
def cmd_export_bvh(
    ctx: click.Context,
    run_id: str,
    output: str,
    smoothed: bool,
    fps: float | None,
    units: str,
    coord: str,
    person_id: int,
) -> None:
    """Export a tracking run to BVH format."""
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'track export'.")

    full_id = _resolve_run(session_path, run_id)

    export_bvh(
        output,
        session_db=session_path,
        run_id=full_id,
        person_id=person_id,
        fps=fps,
        units=units,
        coord=coord,
        smoothed=smoothed,
    )
    click.echo(f"Wrote {output}", err=True)


@export_group.command("gltf")
@click.argument("run_id")
@click.argument("output")
@_add_export_options
@click.pass_context
def cmd_export_gltf(
    ctx: click.Context,
    run_id: str,
    output: str,
    smoothed: bool,
    fps: float | None,
    units: str,
    coord: str,
    person_id: int,
) -> None:
    """Export a tracking run to glTF format."""
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'track export'.")

    full_id = _resolve_run(session_path, run_id)

    export_gltf(
        output,
        session_db=session_path,
        run_id=full_id,
        person_id=person_id,
        fps=fps,
        units=units,
        coord=coord,
        smoothed=smoothed,
    )
    click.echo(f"Wrote {output}", err=True)


@export_group.command("usd")
@click.argument("run_id")
@click.argument("output")
@_add_export_options
@click.pass_context
def cmd_export_usd(
    ctx: click.Context,
    run_id: str,
    output: str,
    smoothed: bool,
    fps: float | None,
    units: str,
    coord: str,
    person_id: int,
) -> None:
    """Export a tracking run to USD format."""
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'track export'.")

    full_id = _resolve_run(session_path, run_id)

    try:
        export_usd(
            output,
            session_db=session_path,
            run_id=full_id,
            person_id=person_id,
            fps=fps,
            units=units,
            coord=coord,
            smoothed=smoothed,
        )
    except ImportError:
        fail("usd-core package is required. Install with: uv sync --group usd")

    click.echo(f"Wrote {output}", err=True)
