# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Status of long-running jobs (tracking and detection runs) in a session database.

A run row is 'running' while its process works. A process that is killed cannot
update its row, so a 'running' row counts as *interrupted* when it has made no
progress for :data:`STALE_AFTER_S`. Progress is dated by ``run_progress``, which
writers touch with every batch of results; a run that has not reported yet is
dated by its start time.
"""

from __future__ import annotations

import datetime
import sqlite3
from typing import Final

#: Seconds without progress after which a 'running' run is interrupted.
STALE_AFTER_S: Final[float] = 600.0

#: Run tables with a status column, and the column holding the start time.
_RUN_TABLES: Final[dict[str, str]] = {"tracking_runs": "ran_at", "detection_runs": "created_at"}

INTERRUPTED: Final[str] = "interrupted"


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _parse(ts: str) -> datetime.datetime:
    """Parse an ISO 8601 UTC timestamp, with a trailing Z or an offset."""
    t = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc)


def touch_progress(conn: sqlite3.Connection, run_id: str) -> None:
    """Record that run *run_id* is alive now. The caller commits."""
    conn.execute(
        "INSERT INTO run_progress (run_id, updated_at) VALUES (?, ?)"
        " ON CONFLICT(run_id) DO UPDATE SET updated_at = excluded.updated_at",
        (run_id, _utc_now().isoformat()),
    )


def _last_activity(conn: sqlite3.Connection, table: str, run_id: str) -> datetime.datetime | None:
    start_col = _RUN_TABLES[table]
    row = conn.execute(
        f"SELECT COALESCE(p.updated_at, r.{start_col}) FROM {table} r"
        " LEFT JOIN run_progress p ON p.run_id = r.id WHERE r.id = ?",
        (run_id,),
    ).fetchone()
    return _parse(row[0]) if row and row[0] else None


def display_status(
    conn: sqlite3.Connection,
    table: str,
    run_id: str,
    status: str,
    *,
    now: datetime.datetime | None = None,
    stale_after_s: float = STALE_AFTER_S,
) -> str:
    """Status to show for a run: its stored status, or 'interrupted' for a stale 'running'."""
    if status != "running":
        return status
    last = _last_activity(conn, table, run_id)
    if last is None:
        return status
    if ((now or _utc_now()) - last).total_seconds() > stale_after_s:
        return INTERRUPTED
    return status


def interrupted_runs(
    conn: sqlite3.Connection, *, stale_after_s: float = STALE_AFTER_S
) -> list[tuple[str, str]]:
    """(table, run id) of every interrupted run in the session database."""
    now = _utc_now()
    found = []
    for table in _RUN_TABLES:
        for (run_id,) in conn.execute(f"SELECT id FROM {table} WHERE status = 'running'").fetchall():
            if display_status(conn, table, run_id, "running", now=now, stale_after_s=stale_after_s) == INTERRUPTED:
                found.append((table, run_id))
    return found


def cleanup_interrupted(
    conn: sqlite3.Connection, *, stale_after_s: float = STALE_AFTER_S
) -> list[tuple[str, str]]:
    """Mark every interrupted run 'failed'; return the (table, run id) pairs changed."""
    found = interrupted_runs(conn, stale_after_s=stale_after_s)
    now = _utc_now().isoformat()
    for table, run_id in found:
        if table == "tracking_runs":
            conn.execute(
                "UPDATE tracking_runs SET status='failed', completed_at=?, failure=? WHERE id=?",
                (now, "interrupted: the process stopped without finishing", run_id),
            )
        else:
            conn.execute(
                "UPDATE detection_runs SET status='failed', completed_at=? WHERE id=?", (now, run_id)
            )
        conn.execute("DELETE FROM run_progress WHERE run_id = ?", (run_id,))
    conn.commit()
    return found
