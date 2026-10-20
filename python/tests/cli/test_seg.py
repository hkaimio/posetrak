# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``posetrak seg prune`` and ``posetrak detect stale``."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from click.testing import CliRunner

from posetrak.cli.main import main
from posetrak.db import seg_revisions as sr


def _invoke(args: list[str], session_path: Path):
    return CliRunner().invoke(main, ["--session", str(session_path), *args], catch_exceptions=False)


def _segmentation(session_path: Path, capture_id: str) -> None:
    conn = sqlite3.connect(session_path)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        "INSERT INTO seg_quality_runs (id, shot_id, time_start_s, time_end_s, created_at) VALUES ('seg1', ?, 0, 1, 't')",
        (capture_id,))
    for frame in range(3):
        rev = sr.open_revision(conn, "seg1")
        sr.write_mask(conn, "seg1", "vid", frame, rev, b"m%d" % frame)
        sr.close_revision(conn, "seg1", rev)
    conn.close()


def test_seg_prune_reports_and_removes(seeded_session_db_path: Path, capture_id: str) -> None:
    _segmentation(seeded_session_db_path, capture_id)
    dry = _invoke(["seg", "prune", "--dry-run"], seeded_session_db_path)
    assert dry.exit_code == 0 and "would remove revisions [0, 1, 2]" in dry.output
    done = _invoke(["seg", "prune", "seg1"], seeded_session_db_path)
    assert done.exit_code == 0 and "removed revisions [0, 1, 2]" in done.output
    conn = sqlite3.connect(seeded_session_db_path)
    assert [r[0] for r in conn.execute("SELECT revision FROM seg_mask_revisions")] == [3]
    assert conn.execute("SELECT COUNT(*) FROM seg_masks").fetchone()[0] == 3     # all three frames still have a mask


def test_detect_stale_refuses_a_run_not_made_from_masks(seeded_session_db_path: Path, capture_id: str, sync_id: str) -> None:
    conn = sqlite3.connect(seeded_session_db_path)
    conn.execute(
        "INSERT INTO detection_runs (id, shot_id, sync_config_id, time_start_s, time_end_s, detector_model, pose_model,"
        " status, created_at) VALUES ('run1', ?, ?, 0, 1, 'yolox-x', 'm', 'complete', 't')", (capture_id, sync_id))
    conn.commit()
    conn.close()
    result = _invoke(["detect", "stale", "run1"], seeded_session_db_path)
    assert result.exit_code != 0 and "not made from segmentation" in result.output
