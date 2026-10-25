# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""runs.py — status of tracking and detection runs.

Commands
--------
runs list      List runs that are not complete (running, interrupted, failed).
runs cleanup   Mark interrupted runs failed.
"""

from __future__ import annotations

from pathlib import Path

import click

from posetrak.cli._output import fail, print_table
from posetrak.db.db import open_session
from posetrak.db.run_status import STALE_AFTER_S, cleanup_interrupted, display_status


def _session_path(ctx: click.Context, command: str) -> Path:
    path = ctx.obj.get("session")
    if path is None:
        fail(f"--session / POSETRAK_SESSION_DB is required for '{command}'.")
    return Path(path)


@click.group("runs")
def runs_group() -> None:
    """Inspect and clean up tracking and detection runs."""


@runs_group.command("list")
@click.pass_context
def cmd_list(ctx: click.Context) -> None:
    """List runs that are running, interrupted or failed.

    Example:

        posetrak --session session.db runs list
    """
    conn = open_session(_session_path(ctx, "runs list"))
    rows = []
    for table, kind in (("tracking_runs", "tracking"), ("detection_runs", "detection")):
        for r in conn.execute(f"SELECT id, status FROM {table} WHERE status != 'complete'"):
            rows.append(
                {"kind": kind, "id": r["id"], "status": display_status(conn, table, r["id"], r["status"])}
            )
    print_table(rows, ["kind", "id", "status"], json_mode=ctx.obj.get("json_mode", False))


@runs_group.command("cleanup")
@click.option(
    "--stale-after",
    default=STALE_AFTER_S,
    show_default=True,
    help="Seconds without progress after which a running run counts as interrupted.",
)
@click.pass_context
def cmd_cleanup(ctx: click.Context, stale_after: float) -> None:
    """Mark interrupted runs (running, no progress for 10 minutes) as failed.

    Example:

        posetrak --session session.db runs cleanup
    """
    conn = open_session(_session_path(ctx, "runs cleanup"))
    rows = [
        {"kind": table.removesuffix("_runs"), "id": run_id, "status": "failed"}
        for table, run_id in cleanup_interrupted(conn, stale_after_s=stale_after)
    ]
    print_table(rows, ["kind", "id", "status"], json_mode=ctx.obj.get("json_mode", False))
    click.echo(f"{len(rows)} interrupted run(s) marked failed.", err=True)
