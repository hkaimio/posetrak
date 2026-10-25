# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""seg.py — CLI commands for segmentation mask revisions."""
from __future__ import annotations

from pathlib import Path

import click

from posetrak.cli._output import fail
from posetrak.db.db import open_session, resolve_id_prefix
from posetrak.db.seg_revisions import prune


@click.group("seg")
def seg_group() -> None:
    """Segmentation mask revisions."""


@seg_group.command("prune")
@click.argument("seg_run", required=False)
@click.option("--dry-run", is_flag=True, default=False, help="Report what would be removed and change nothing.")
@click.pass_context
def cmd_prune(ctx: click.Context, seg_run: str | None, dry_run: bool) -> None:
    """Remove mask revisions nothing needs.

    Keeps the newest saved revision of each segmentation and every revision that a detection run or a
    tracking run recorded (whatever its status, running included); an open revision is never touched.
    Without SEG_RUN (prefix accepted) every segmentation of the session is pruned. Never automatic.
    """
    session_path = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'seg prune'.")
    try:
        session = open_session(Path(session_path))
        run_ids = ([resolve_id_prefix(session, "seg_quality_runs", seg_run)] if seg_run
                   else [r[0] for r in session.execute("SELECT id FROM seg_quality_runs ORDER BY created_at")])
    except (FileNotFoundError, ValueError) as exc:
        fail(str(exc))
    verb = "would remove" if dry_run else "removed"
    for run_id in run_ids:
        result = prune(session, run_id, dry_run=dry_run)
        if result.revisions_removed:
            click.echo(f"{run_id}: {verb} revisions {result.revisions_removed}: {result.rows_removed} masks deleted, "
                       f"{result.rows_moved} carried to the next kept revision")
    session.close()
