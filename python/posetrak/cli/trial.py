"""trial.py — CLI commands for trial export, import, and listing."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from posetrak.cli._output import fail, print_table
from posetrak.db.db import create_session, open_session
from posetrak.db.trial_export import (
    AnchorSpec,
    ExportScope,
    export_trials,
    import_trials,
    open_source_readonly,
)


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------


@click.group("trial")
def trial_group() -> None:
    """List, export, and import trials."""


# ---------------------------------------------------------------------------
# trial list
# ---------------------------------------------------------------------------


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
    except (FileNotFoundError, ValueError, Exception) as exc:
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
            (SELECT COUNT(*) FROM tracking_runs tr
             JOIN pose_observation_sequences s ON s.id = tr.observation_sequence_id
             WHERE s.shot_id = t.capture_id
               AND (t.time_start_s IS NULL OR s.time_start_s >= t.time_start_s - 0.5)
               AND (t.time_end_s   IS NULL OR s.time_end_s   <= t.time_end_s   + 0.5))
                AS n_tracking_runs
        FROM trials t
        JOIN captures c ON c.id = t.capture_id
        ORDER BY c.label, t.time_start_s
    """).fetchall()

    if not rows:
        # Fall back to capture listing when no trials are defined
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
# trial export
# ---------------------------------------------------------------------------


def _default_scope(anchor: AnchorSpec) -> ExportScope:
    if anchor.tracking_run_ids:
        return ExportScope.FULL
    if anchor.detection_ids or anchor.trial_ids:
        return ExportScope.DETECTION_ONLY
    return ExportScope.CAPTURE_ONLY


@trial_group.command("export")
@click.argument("output", metavar="OUTPUT.db")
@click.option("--trial",        "trial_ids",        multiple=True, metavar="ID", help="Trial ID to export (repeatable).")
@click.option("--capture",      "capture_ids",      multiple=True, metavar="ID", help="Capture ID to export (repeatable).")
@click.option("--detection",    "detection_ids",    multiple=True, metavar="ID", help="Detection run ID to export (repeatable).")
@click.option("--tracking-run", "tracking_run_ids", multiple=True, metavar="ID", help="Tracking run ID to export (repeatable).")
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
    """Export trials and their dependencies to a new session database.

    At least one anchor (--trial, --capture, --detection, --tracking-run) is
    required; if none are given, all captures in the session are exported.

    Scope controls how much data is included:

    \b
      capture-only    — camera, session, capture infrastructure
      trial-only      — above + trial metadata
      detection-only  — above + detection runs, observations, edits
      full            — above + tracking runs, configs, skeletons

    Default scope: capture → capture-only; trial/detection → detection-only;
    tracking-run → full.

    Example:

        posetrak --session session.db trial export backup.db --trial <id> --dry-run
    """
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'trial export'.")

    output_path = Path(output)
    if output_path.exists() and not dry_run:
        fail(f"Output already exists: {output_path}  (delete it first or use --dry-run)")

    anchor = AnchorSpec(
        trial_ids=list(trial_ids),
        capture_ids=list(capture_ids),
        detection_ids=list(detection_ids),
        tracking_run_ids=list(tracking_run_ids),
    )

    resolved_scope = (
        ExportScope(scope) if scope else _default_scope(anchor)
    )

    skip_set = {t.strip() for t in skip_tables.split(",") if t.strip()}

    try:
        src = open_source_readonly(Path(session_path))
    except Exception as exc:
        fail(f"Cannot open source session: {exc}")

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
# trial import
# ---------------------------------------------------------------------------


@trial_group.command("import")
@click.argument("source", metavar="SOURCE.db")
@click.option("--trial",        "trial_ids",     multiple=True, metavar="ID", help="Trial ID to import (repeatable; default: all).")
@click.option("--capture",      "capture_ids",   multiple=True, metavar="ID", help="Capture ID to import (repeatable; default: all).")
@click.option("--detection",    "detection_ids", multiple=True, metavar="ID", help="Detection run ID to import (repeatable; default: all).")
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
    """Import trials from SOURCE.db into the current session.

    SOURCE.db is typically a file produced by 'trial export'.  All data in
    SOURCE.db is imported (full scope) unless anchor flags narrow the selection.

    Example:

        posetrak --session session.db trial import backup.db --dry-run
        posetrak --session session.db trial import backup.db --trial <id>
    """
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'trial import'.")

    registry_path: str = ctx.obj["registry"]
    anchor = AnchorSpec(
        trial_ids=list(trial_ids),
        capture_ids=list(capture_ids),
        detection_ids=list(detection_ids),
    )
    skip_set = {t.strip() for t in skip_tables.split(",") if t.strip()}

    try:
        src = open_source_readonly(Path(source))
    except Exception as exc:
        fail(f"Cannot open source: {exc}")

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
