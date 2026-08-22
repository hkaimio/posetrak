# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for `posetrak config refresh-baseline`."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from posetrak.cli.main import main
from posetrak.db.manage_config import BASELINE_CONFIG_ID, seed_baseline_tracker_config


def _seed_and_null_out_baseline(db_path: Path) -> None:
    """Put a session/registry into the state a pre-fix install would be in:
    a baseline row present (as if copied down at session-creation time),
    but with stale/NULL tuning values -- create_session() itself doesn't
    seed one, unlike create_registry(), so tests must put one there first."""
    conn = sqlite3.connect(db_path)
    seed_baseline_tracker_config(conn)
    conn.execute(
        "UPDATE tracker_configs SET process_noise_std = NULL, tracker_fps = NULL "
        "WHERE id = ?",
        (BASELINE_CONFIG_ID,),
    )
    conn.commit()
    conn.close()


def _baseline_values(db_path: Path) -> tuple:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT process_noise_std, tracker_fps FROM tracker_configs WHERE id = ?",
        (BASELINE_CONFIG_ID,),
    ).fetchone()
    conn.close()
    return row


class TestConfigRefreshBaseline:
    def test_refreshes_session_baseline(
        self, cli_runner: CliRunner, session_db_path: Path
    ) -> None:
        _seed_and_null_out_baseline(session_db_path)
        assert _baseline_values(session_db_path) == (None, None)

        result = cli_runner.invoke(main, [
            "--session", str(session_db_path),
            "config", "refresh-baseline",
        ])
        assert result.exit_code == 0, result.output
        assert "baseline config refreshed" in result.output
        assert _baseline_values(session_db_path) == pytest.approx((0.3, 120.0))

    def test_refreshes_registry_baseline(
        self, cli_runner: CliRunner, registry_db_path: Path
    ) -> None:
        _seed_and_null_out_baseline(registry_db_path)

        result = cli_runner.invoke(main, [
            "--registry", str(registry_db_path),
            "config", "refresh-baseline", "--global",
        ])
        assert result.exit_code == 0, result.output
        assert "baseline config refreshed" in result.output
        assert _baseline_values(registry_db_path) == pytest.approx((0.3, 120.0))

    def test_reports_missing_row_without_error(
        self, cli_runner: CliRunner, session_db_path: Path
    ) -> None:
        conn = sqlite3.connect(session_db_path)
        conn.execute("DELETE FROM tracker_configs WHERE id = ?", (BASELINE_CONFIG_ID,))
        conn.commit()
        conn.close()

        result = cli_runner.invoke(main, [
            "--session", str(session_db_path),
            "config", "refresh-baseline",
        ])
        assert result.exit_code == 0, result.output
        assert "no baseline config row found" in result.output

    def test_requires_session_or_global(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["config", "refresh-baseline"])
        assert result.exit_code != 0
        assert "--session" in result.output
