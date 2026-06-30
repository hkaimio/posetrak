"""trial.py — Trial management, export, and import commands.

Commands
--------
trial list          List trials in the session DB (group with one sub-command).
export OUTPUT.db    Export data from the current session to a new DB.
import SOURCE.db    Import data from an exported DB into the current session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from posetrak.cli._output import fail, print_table
from posetrak.db.db import create_session, generate_id, open_session, resolve_id_prefix
from posetrak.db.trial_export import (
    AnchorSpec,
    ExportScope,
    export_trials,
    import_trials,
    open_source_readonly,
)


# ---------------------------------------------------------------------------
# trial list
# ---------------------------------------------------------------------------


@click.group("trial")
def trial_group() -> None:
    """Manage trials."""


@trial_group.command("list")
@click.pass_context
def cmd_list(ctx: click.Context) -> None:
    """List trials in the session database.

    Falls back to listing captures when no trials are defined.

    Example:

        posetrak --session session.db trial list
    """
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'trial list'.")

    json_mode: bool = ctx.obj.get("json_mode", False)

    try:
        conn = open_session(Path(session_path))
    except Exception as exc:
        fail(str(exc))

    rows = conn.execute("""
        SELECT
            t.id,
            t.name,
            t.time_start_s,
            t.time_end_s,
            c.label  AS capture_label,
            (SELECT COUNT(*) FROM detection_runs dr WHERE dr.trial_id = t.id)
                AS n_detections,
            (SELECT COUNT(*) FROM tracking_runs WHERE trial_id = t.id)
                AS n_tracking_runs
        FROM trials t
        JOIN captures c ON c.id = t.capture_id
        ORDER BY c.label, t.time_start_s
    """).fetchall()

    if not rows:
        cap_rows = conn.execute("""
            SELECT
                c.id,
                c.label,
                c.capture_number,
                (SELECT COUNT(*) FROM trials      t  WHERE t.capture_id  = c.id) AS n_trials,
                (SELECT COUNT(*) FROM detection_runs dr WHERE dr.shot_id  = c.id) AS n_detections,
                (SELECT COUNT(*) FROM tracking_runs tr
                 JOIN pose_observation_sequences s ON s.id = tr.observation_sequence_id
                 WHERE s.shot_id = c.id) AS n_tracking_runs
            FROM captures c
            ORDER BY c.capture_number
        """).fetchall()

        if not cap_rows:
            click.echo("No captures or trials found.")
            return

        click.echo("No trials defined. Showing captures:\n", err=True)
        print_table(
            [dict(r) for r in cap_rows],
            columns=["id", "label", "capture_number", "n_trials", "n_detections", "n_tracking_runs"],
            json_mode=json_mode,
        )
        return

    print_table(
        [dict(r) for r in rows],
        columns=["id", "name", "capture_label", "time_start_s", "time_end_s", "n_detections", "n_tracking_runs"],
        json_mode=json_mode,
    )


# ---------------------------------------------------------------------------
# trial create
# ---------------------------------------------------------------------------


@trial_group.command("create")
@click.option("--capture", "capture_id", required=True, metavar="ID",
              help="Capture ID or unique prefix.")
@click.option("--name", "name", required=True, metavar="NAME",
              help="Trial name.")
@click.option("--start", "time_start_s", type=float, default=None, metavar="S",
              help="Trial start time in seconds.")
@click.option("--end", "time_end_s", type=float, default=None, metavar="S",
              help="Trial end time in seconds.")
@click.pass_context
def cmd_create(
    ctx: click.Context,
    capture_id: str,
    name: str,
    time_start_s: float | None,
    time_end_s: float | None,
) -> None:
    """Create a new trial within a capture.

    Example:

        posetrak --session session.db trial create --capture 13af67f5 --name "Sprint 1" --start 10.0 --end 30.0
    """
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'trial create'.")

    try:
        conn = open_session(Path(session_path))
    except Exception as exc:
        fail(str(exc))

    try:
        capture_id = resolve_id_prefix(conn, "captures", capture_id)
    except ValueError as exc:
        fail(str(exc))

    trial_id = generate_id()
    conn.execute(
        "INSERT INTO trials (id, capture_id, name, time_start_s, time_end_s) "
        "VALUES (?, ?, ?, ?, ?)",
        (trial_id, capture_id, name, time_start_s, time_end_s),
    )
    conn.commit()
    conn.close()

    click.echo(f"trial_id: {trial_id}")


# ---------------------------------------------------------------------------
# trial show
# ---------------------------------------------------------------------------


@trial_group.command("show")
@click.argument("trial_id", metavar="TRIAL_ID")
@click.pass_context
def cmd_show(ctx: click.Context, trial_id: str) -> None:
    """Show details of a trial including its detection runs and tracking runs.

    TRIAL_ID accepts a unique prefix (first 8 characters suffice).

    Example:

        posetrak --session session.db trial show abc12345
    """
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'trial show'.")

    json_mode: bool = ctx.obj.get("json_mode", False)

    try:
        conn = open_session(Path(session_path))
    except Exception as exc:
        fail(str(exc))

    try:
        trial_id = resolve_id_prefix(conn, "trials", trial_id)
    except ValueError as exc:
        fail(str(exc))

    trial = conn.execute(
        "SELECT t.id, t.name, t.time_start_s, t.time_end_s, c.label AS capture_label "
        "FROM trials t JOIN captures c ON c.id = t.capture_id "
        "WHERE t.id = ?",
        (trial_id,),
    ).fetchone()

    if trial is None:
        fail(f"Trial not found: {trial_id}")

    detection_runs = conn.execute(
        "SELECT id, detector_model, pose_model, status, created_at "
        "FROM detection_runs WHERE trial_id = ? ORDER BY created_at",
        (trial_id,),
    ).fetchall()

    tracking_runs = conn.execute(
        "SELECT tr.id, COALESCE(sp.person_name, 'unnamed') AS person_name, "
        "       tr.status, tr.created_at "
        "FROM tracking_runs tr "
        "LEFT JOIN pose_observation_sequences pos ON pos.id = tr.observation_sequence_id "
        "LEFT JOIN sequence_persons sp ON sp.sequence_id = pos.id "
        "WHERE tr.trial_id = ? "
        "ORDER BY tr.created_at",
        (trial_id,),
    ).fetchall()

    if json_mode:
        import json
        click.echo(json.dumps({
            "id": trial["id"],
            "name": trial["name"],
            "capture_label": trial["capture_label"],
            "time_start_s": trial["time_start_s"],
            "time_end_s": trial["time_end_s"],
            "detection_runs": [dict(r) for r in detection_runs],
            "tracking_runs": [dict(r) for r in tracking_runs],
        }))
        return

    click.echo(f"Trial:    {trial['name']} ({trial['id'][:8]}…)")
    click.echo(f"Capture:  {trial['capture_label']}")
    start = f"{trial['time_start_s']:.3f} s" if trial["time_start_s"] is not None else "—"
    end = f"{trial['time_end_s']:.3f} s" if trial["time_end_s"] is not None else "—"
    click.echo(f"Range:    {start} → {end}")
    click.echo()

    if detection_runs:
        click.echo(f"Detection runs ({len(detection_runs)}):")
        print_table(
            [dict(r) for r in detection_runs],
            columns=["id", "detector_model", "pose_model", "status", "created_at"],
            json_mode=False,
        )
    else:
        click.echo("Detection runs: none")

    click.echo()

    if tracking_runs:
        click.echo(f"Tracking runs ({len(tracking_runs)}):")
        print_table(
            [dict(r) for r in tracking_runs],
            columns=["id", "person_name", "status", "created_at"],
            json_mode=False,
        )
    else:
        click.echo("Tracking runs: none")

    conn.close()


# ---------------------------------------------------------------------------
# Shared helpers for export / import
# ---------------------------------------------------------------------------


def _default_scope(anchor: AnchorSpec) -> ExportScope:
    if anchor.tracking_run_ids:
        return ExportScope.FULL
    if anchor.detection_ids or anchor.trial_ids:
        return ExportScope.DETECTION_ONLY
    return ExportScope.CAPTURE_ONLY


def _resolve_anchors(
    conn,
    trial_ids: tuple[str, ...],
    capture_ids: tuple[str, ...],
    detection_ids: tuple[str, ...],
    tracking_run_ids: tuple[str, ...],
) -> AnchorSpec:
    """Resolve ID prefixes to full UUIDs; fail loudly if ambiguous or missing."""

    def resolve(table: str, ids: tuple[str, ...]) -> list[str]:
        result = []
        for id_ in ids:
            try:
                result.append(resolve_id_prefix(conn, table, id_))
            except ValueError as exc:
                fail(str(exc))
        return result

    return AnchorSpec(
        trial_ids=resolve("trials", trial_ids),
        capture_ids=resolve("captures", capture_ids),
        detection_ids=resolve("detection_runs", detection_ids),
        tracking_run_ids=resolve("tracking_runs", tracking_run_ids),
    )


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@click.command("export")
@click.argument("output", metavar="OUTPUT.db")
@click.option("--trial",        "trial_ids",        multiple=True, metavar="ID", help="Trial ID or prefix (repeatable).")
@click.option("--capture",      "capture_ids",      multiple=True, metavar="ID", help="Capture ID or prefix (repeatable).")
@click.option("--detection",    "detection_ids",    multiple=True, metavar="ID", help="Detection run ID or prefix (repeatable).")
@click.option("--tracking-run", "tracking_run_ids", multiple=True, metavar="ID", help="Tracking run ID or prefix (repeatable).")
@click.option(
    "--scope",
    type=click.Choice([s.value for s in ExportScope], case_sensitive=False),
    default=None,
    help=(
        "Data scope: capture-only | trial-only | detection-only | full. "
        "Default depends on anchor type."
    ),
)
@click.option("--include-cache", is_flag=True, default=False,
              help="Include frame_cache_entries (large; excluded by default).")
@click.option("--skip-tables", default="", metavar="TABLE,...",
              help="Comma-separated table names to skip (useful for known-corrupted tables).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Count rows that would be copied without writing anything.")
@click.pass_context
def cmd_export(
    ctx: click.Context,
    output: str,
    trial_ids: tuple[str, ...],
    capture_ids: tuple[str, ...],
    detection_ids: tuple[str, ...],
    tracking_run_ids: tuple[str, ...],
    scope: str | None,
    include_cache: bool,
    skip_tables: str,
    dry_run: bool,
) -> None:
    """Export session data to a new portable database.

    At least one anchor flag (--trial, --capture, --detection, --tracking-run)
    selects what to export. Without any anchor, all captures are exported.
    Anchor IDs accept a unique prefix (first 8 characters suffice).

    Scope controls how much data is included:

    \b
      capture-only    — camera, session, capture infrastructure
      trial-only      — above + trial metadata
      detection-only  — above + detection runs, observations, edits
      full            — above + tracking runs, configs, skeletons

    Default scope: capture → capture-only; trial/detection → detection-only;
    tracking-run → full.

    Examples:

        posetrak -s session.db export backup.db --capture 13af67f5 --scope full
        posetrak -s session.db export backup.db --trial <id> --dry-run
        posetrak -s session.db export backup.db --skip-tables pose_observation_edits
    """
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'export'.")

    output_path = Path(output)
    if output_path.exists() and not dry_run:
        fail(f"Output already exists: {output_path}  (delete it first or use --dry-run)")

    skip_set = {t.strip() for t in skip_tables.split(",") if t.strip()}

    try:
        src = open_source_readonly(Path(session_path))
    except Exception as exc:
        fail(f"Cannot open source session: {exc}")

    # Resolve prefixes against the source DB
    anchor = _resolve_anchors(src, trial_ids, capture_ids, detection_ids, tracking_run_ids)

    resolved_scope = ExportScope(scope) if scope else _default_scope(anchor)

    click.echo(f"Source:  {session_path}", err=True)
    click.echo(f"Output:  {output_path}", err=True)
    click.echo(f"Scope:   {resolved_scope.value}", err=True)
    if dry_run:
        click.echo("Mode:    dry-run (no files written)", err=True)
    if skip_set:
        click.echo(f"Skipped: {', '.join(sorted(skip_set))}", err=True)
    click.echo("", err=True)

    dst = None
    if not dry_run:
        try:
            dst = create_session(output_path)
        except FileExistsError as exc:
            fail(str(exc))

    result = export_trials(
        src, dst, anchor,
        scope=resolved_scope,
        include_cache=include_cache,
        skip_tables=skip_set,
        on_progress=lambda msg: click.echo(msg, err=True),
        dry_run=dry_run,
    )

    src.close()
    if dst is not None:
        dst.close()

    click.echo("", err=True)
    errors = [t for t in result.tables if t.error]
    if errors:
        click.echo(f"Completed with {len(errors)} warning(s):", err=True)
        for t in errors:
            click.echo(f"  {t.table}: {t.error}", err=True)
    else:
        click.echo(f"Done. {result.total_rows} rows {'would be ' if dry_run else ''}copied.", err=True)

    sys.exit(0 if result.success else 1)


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------


@click.command("import")
@click.argument("source", metavar="SOURCE.db")
@click.option("--trial",     "trial_ids",     multiple=True, metavar="ID", help="Trial ID or prefix to import (default: all).")
@click.option("--capture",   "capture_ids",   multiple=True, metavar="ID", help="Capture ID or prefix to import (default: all).")
@click.option("--detection", "detection_ids", multiple=True, metavar="ID", help="Detection run ID or prefix to import (default: all).")
@click.option("--sync-registry", is_flag=True, default=False,
              help="Also copy camera/skeleton/config data to --registry.")
@click.option("--skip-tables", default="", metavar="TABLE,...",
              help="Comma-separated table names to skip.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Count rows that would be imported without writing anything.")
@click.pass_context
def cmd_import(
    ctx: click.Context,
    source: str,
    trial_ids: tuple[str, ...],
    capture_ids: tuple[str, ...],
    detection_ids: tuple[str, ...],
    sync_registry: bool,
    skip_tables: str,
    dry_run: bool,
) -> None:
    """Import data from SOURCE.db into the current session.

    SOURCE.db is typically a file produced by 'export'. All data in SOURCE.db
    is imported (full scope) unless anchor flags narrow the selection.
    Anchor IDs accept a unique prefix (first 8 characters suffice).

    Examples:

        posetrak -s session.db import backup.db --dry-run
        posetrak -s session.db import backup.db --trial <id>
        posetrak -s session.db import backup.db --skip-tables pose_observation_edits
    """
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'import'.")

    registry_path: str = ctx.obj["registry"]
    skip_set = {t.strip() for t in skip_tables.split(",") if t.strip()}

    try:
        src = open_source_readonly(Path(source))
    except Exception as exc:
        fail(f"Cannot open source: {exc}")

    # Resolve prefixes against the source DB
    anchor = _resolve_anchors(src, trial_ids, capture_ids, detection_ids, ())

    try:
        dst_session = open_session(Path(session_path))
    except Exception as exc:
        fail(f"Cannot open session: {exc}")

    dst_registry = None
    if sync_registry:
        from posetrak.db.db import open_registry
        try:
            dst_registry = open_registry(Path(registry_path))
        except Exception as exc:
            fail(f"Cannot open registry: {exc}")

    click.echo(f"Source:  {source}", err=True)
    click.echo(f"Session: {session_path}", err=True)
    if sync_registry:
        click.echo(f"Registry: {registry_path}", err=True)
    if dry_run:
        click.echo("Mode:    dry-run (no files written)", err=True)
    if skip_set:
        click.echo(f"Skipped: {', '.join(sorted(skip_set))}", err=True)
    click.echo("", err=True)

    result = import_trials(
        src, dst_session, dst_registry, anchor,
        skip_tables=skip_set,
        dry_run=dry_run,
        on_progress=lambda msg: click.echo(msg, err=True),
    )

    src.close()
    dst_session.close()
    if dst_registry is not None:
        dst_registry.close()

    click.echo("", err=True)
    errors = [t for t in result.tables if t.error]
    if errors:
        click.echo(f"Completed with {len(errors)} warning(s):", err=True)
        for t in errors:
            click.echo(f"  {t.table}: {t.error}", err=True)
    else:
        click.echo(f"Done. {result.total_rows} rows {'would be ' if dry_run else ''}imported.", err=True)

    sys.exit(0 if result.success else 1)
