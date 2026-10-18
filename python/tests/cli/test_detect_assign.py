# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the posetrak detect assign command."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from posetrak.cli.main import main
from tests.detection.synthetic_people import build_session

STARTS = [(-2.0, -0.6), (2.0, 0.6)]
VELOCITIES = [(1.0, 0.0), (-1.0, 0.0)]


def _invoke(db: Path, *args: str):
    return CliRunner().invoke(main, ["--session", str(db), "detect", "assign", *args])


def test_the_command_writes_rows_and_the_report_file(tmp_path: Path) -> None:
    conn, run_id, _ = build_session(tmp_path / "s.db", STARTS, VELOCITIES, 90, swap_at=45)
    conn.close()

    result = _invoke(tmp_path / "s.db", run_id[:8], "--persons", "2")

    assert result.exit_code == 0, result.output
    assert "Assigned" in result.output and "cam1" in result.output
    report = tmp_path / f"s.assign-{run_id[:8]}.json"
    assert json.loads(report.read_text())["run"] == run_id
    check = sqlite3.connect(tmp_path / "s.db")
    assert check.execute("SELECT COUNT(*) FROM detection_track_assignments").fetchone()[0] > 0
    check.close()


def test_the_command_refuses_a_run_with_rows_and_replace_overrides(tmp_path: Path) -> None:
    conn, run_id, videos = build_session(tmp_path / "s.db", STARTS, VELOCITIES, 90)
    conn.execute("INSERT INTO detection_track_assignments (detection_run_id, shot_video_id, track_id, person_name, "
                 "first_frame, last_frame) VALUES (?, ?, 0, 'Alice', 0, 89)", (run_id, videos[0]))
    conn.commit()
    conn.close()

    refused = _invoke(tmp_path / "s.db", run_id, "--persons", "2")
    both = _invoke(tmp_path / "s.db", run_id, "--fill-only", "--replace")
    replaced = _invoke(tmp_path / "s.db", run_id, "--persons", "2", "--replace")

    assert refused.exit_code != 0 and "--fill-only" in refused.output
    assert both.exit_code != 0 and "exclude each other" in both.output
    assert replaced.exit_code == 0, replaced.output


def test_dry_run_prints_the_report_and_writes_nothing(tmp_path: Path) -> None:
    conn, run_id, _ = build_session(tmp_path / "s.db", STARTS, VELOCITIES, 90)
    conn.close()

    result = _invoke(tmp_path / "s.db", run_id, "--persons", "2", "--dry-run")

    assert result.exit_code == 0 and "Assigned" in result.output
    assert not list(tmp_path.glob("*.json"))
    check = sqlite3.connect(tmp_path / "s.db")
    assert check.execute("SELECT COUNT(*) FROM detection_track_assignments").fetchone()[0] == 0
    check.close()


def test_finalise_builds_one_sequence_per_person(tmp_path: Path) -> None:
    conn, run_id, _ = build_session(tmp_path / "s.db", STARTS, VELOCITIES, 90, swap_at=45)
    conn.close()

    result = _invoke(tmp_path / "s.db", run_id, "--persons", "2", "--finalise")

    assert result.exit_code == 0, result.output
    check = sqlite3.connect(tmp_path / "s.db")
    assert check.execute("SELECT COUNT(*) FROM pose_observation_sequences WHERE detection_run_id = ?",
                         (run_id,)).fetchone()[0] == 2
    check.close()
