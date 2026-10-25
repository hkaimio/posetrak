# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Interrupted-run detection, `runs cleanup`, and lock tolerance of session connections."""

from __future__ import annotations

import datetime
import multiprocessing
import sqlite3
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from posetrak.cli.main import main
from posetrak.db.db import _connect, create_session, open_session
from posetrak.db.run_status import (
    INTERRUPTED,
    cleanup_interrupted,
    display_status,
    interrupted_runs,
    touch_progress,
)


def _ago(seconds: float) -> str:
    t = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=seconds)
    return t.isoformat()


@pytest.fixture
def conn(session_db: sqlite3.Connection) -> sqlite3.Connection:
    """Session with the tables the status logic reads, no foreign-key chain behind them."""
    session_db.execute("PRAGMA foreign_keys = OFF")
    return session_db


def _tracking_run(conn, run_id: str, status: str, started_s_ago: float) -> None:
    conn.execute(
        "INSERT INTO tracking_runs (id, observation_sequence_id, tracker_config_id, skeleton_id,"
        " extrinsic_calibration_id, sync_config_id, ran_at, posetrak_version, active_camera_ids,"
        " marker_names, status) VALUES (?, 's', 'c', 'k', 'e', 'y', ?, 'v', '[]', '[]', ?)",
        (run_id, _ago(started_s_ago), status),
    )


def _detection_run(conn, run_id: str, status: str, started_s_ago: float) -> None:
    conn.execute(
        "INSERT INTO detection_runs (id, shot_id, sync_config_id, time_start_s, time_end_s,"
        " detector_model, pose_model, status, created_at) VALUES (?, 'cap', 'y', 0, 1, 'd', 'p', ?, ?)",
        (run_id, status, _ago(started_s_ago)),
    )


def test_running_row_without_recent_progress_is_interrupted(conn):
    _tracking_run(conn, "alive", "running", 3600)
    touch_progress(conn, "alive")
    _tracking_run(conn, "dead", "running", 3600)
    _tracking_run(conn, "young", "running", 5)
    _tracking_run(conn, "done", "complete", 3600)
    _detection_run(conn, "det-dead", "running", 3600)
    assert display_status(conn, "tracking_runs", "alive", "running") == "running"
    assert display_status(conn, "tracking_runs", "young", "running") == "running"
    assert display_status(conn, "tracking_runs", "dead", "running") == INTERRUPTED
    assert display_status(conn, "tracking_runs", "done", "complete") == "complete"
    assert set(interrupted_runs(conn)) == {("tracking_runs", "dead"), ("detection_runs", "det-dead")}


def test_cleanup_marks_only_interrupted_runs_failed(conn):
    _tracking_run(conn, "dead", "running", 3600)
    _tracking_run(conn, "alive", "running", 5)
    _detection_run(conn, "det-dead", "running", 3600)
    cleanup_interrupted(conn)
    rows = {r[0]: r[1:] for r in conn.execute("SELECT id, status, completed_at, failure FROM tracking_runs")}
    assert rows["dead"][0] == "failed" and rows["dead"][1] and "interrupted" in rows["dead"][2]
    assert rows["alive"] == ("running", None, None)
    assert conn.execute("SELECT status FROM detection_runs").fetchone()[0] == "failed"
    assert interrupted_runs(conn) == []


def test_runs_cleanup_command(tmp_path: Path):
    path = tmp_path / "s.db"
    create_session(path).close()
    conn = open_session(path)
    conn.execute("PRAGMA foreign_keys = OFF")
    _tracking_run(conn, "dead", "running", 3600)
    conn.commit()
    conn.close()
    result = CliRunner().invoke(main, ["--session", str(path), "runs", "cleanup"])
    assert result.exit_code == 0, result.output
    assert "dead" in result.output
    assert open_session(path).execute("SELECT status FROM tracking_runs").fetchone()[0] == "failed"


def _hold_write_lock(path: str, seconds: float, ready) -> None:
    c = sqlite3.connect(path, isolation_level=None)
    c.execute("BEGIN IMMEDIATE")
    ready.set()
    time.sleep(seconds)
    c.execute("COMMIT")
    c.close()


def test_connection_waits_for_another_writer_instead_of_failing(tmp_path: Path):
    path = tmp_path / "s.db"
    create_session(path).close()
    ready = multiprocessing.Event()
    holder = multiprocessing.Process(target=_hold_write_lock, args=(str(path), 3.0, ready))
    holder.start()
    try:
        assert ready.wait(30)
        conn = _connect(path)
        t0 = time.monotonic()
        conn.execute("INSERT INTO run_progress (run_id, updated_at) VALUES ('x', 'y')")
        conn.commit()
        assert time.monotonic() - t0 >= 1.0  # it did wait for the holder
    finally:
        holder.join()
