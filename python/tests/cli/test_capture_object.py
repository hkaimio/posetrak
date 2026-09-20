# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``capture object`` CLI commands."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from posetrak.cli.main import main
from posetrak.db.db import open_registry
from posetrak.db.manage_marker_body import import_marker_body_str

_BODY_YAML = """\
name: test-bokken
units: meters
markers:
  - name: hilt
    type: aruco
    dictionary: DICT_4X4_50
    id: "3"
    size: 0.05
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
"""


def _invoke(args: list[str], session_path: Path, registry_path: Path | None = None):
    base = ["--session", str(session_path)]
    if registry_path is not None:
        base = ["--registry", str(registry_path), *base]
    return CliRunner().invoke(main, base + args, catch_exceptions=False)


def _session_body(session_path: Path) -> str:
    conn = sqlite3.connect(str(session_path))
    try:
        return import_marker_body_str(conn, _BODY_YAML, name="Test Bokken")
    finally:
        conn.close()


def _objects(session_path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(session_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM capture_objects ORDER BY name").fetchall()
    conn.close()
    return rows


class TestCaptureObject:
    def test_add_list_rename_rm(self, seeded_session_db_path: Path, capture_id: str) -> None:
        body_id = _session_body(seeded_session_db_path)

        added = _invoke(
            ["capture", "object", "add", "--capture", capture_id[:8], "--name", "bokken-A",
             "--marker-body", body_id[:8], "--notes", "first"],
            seeded_session_db_path,
        )
        assert added.exit_code == 0, added.output
        (row,) = _objects(seeded_session_db_path)
        assert added.output.strip() == f"capture_object_id: {row['id']}"
        assert (row["capture_id"], row["name"], row["marker_body_definition_id"], row["notes"]) == (
            capture_id, "bokken-A", body_id, "first")

        listed = _invoke(["--json", "capture", "object", "list", "--capture", capture_id],
                         seeded_session_db_path)
        assert [json.loads(line)["name"] for line in listed.output.splitlines()] == ["bokken-A"]

        renamed = _invoke(["capture", "object", "rename", row["id"][:8], "bokken-B"], seeded_session_db_path)
        assert renamed.exit_code == 0, renamed.output
        assert _objects(seeded_session_db_path)[0]["name"] == "bokken-B"

        removed = _invoke(["capture", "object", "rm", row["id"][:8]], seeded_session_db_path)
        assert removed.exit_code == 0, removed.output
        assert _objects(seeded_session_db_path) == []

    def test_add_copies_the_marker_body_from_the_registry(
        self, seeded_session_db_path: Path, registry_db_path: Path, capture_id: str
    ) -> None:
        registry = open_registry(registry_db_path)
        body_id = import_marker_body_str(registry, _BODY_YAML, name="Test Bokken")
        registry.close()

        result = _invoke(
            ["capture", "object", "add", "--capture", capture_id, "--name", "bokken-A",
             "--marker-body", body_id[:8]],
            seeded_session_db_path, registry_db_path,
        )

        assert result.exit_code == 0, result.output
        assert _objects(seeded_session_db_path)[0]["marker_body_definition_id"] == body_id

    def test_add_rejects_an_unknown_marker_body(
        self, seeded_session_db_path: Path, registry_db_path: Path, capture_id: str
    ) -> None:
        result = _invoke(
            ["capture", "object", "add", "--capture", capture_id, "--name", "x", "--marker-body", "nope"],
            seeded_session_db_path, registry_db_path,
        )
        assert result.exit_code != 0
        assert "Marker body 'nope'" in result.output
        assert _objects(seeded_session_db_path) == []

    def test_a_name_is_unique_within_a_capture(self, seeded_session_db_path: Path, capture_id: str) -> None:
        body_id = _session_body(seeded_session_db_path)
        add = ["capture", "object", "add", "--capture", capture_id, "--marker-body", body_id]
        assert _invoke([*add, "--name", "a"], seeded_session_db_path).exit_code == 0
        assert _invoke([*add, "--name", "b"], seeded_session_db_path).exit_code == 0

        again = _invoke([*add, "--name", "a"], seeded_session_db_path)
        clash = _invoke(["capture", "object", "rename", _objects(seeded_session_db_path)[1]["id"], "a"],
                        seeded_session_db_path)

        assert again.exit_code != 0 and "already has an object named 'a'" in again.output
        assert clash.exit_code != 0 and "already has an object named 'a'" in clash.output
        assert [o["name"] for o in _objects(seeded_session_db_path)] == ["a", "b"]

    def test_rm_refuses_an_object_a_detection_run_uses(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        body_id = _session_body(seeded_session_db_path)
        _invoke(["capture", "object", "add", "--capture", capture_id, "--name", "a", "--marker-body", body_id],
                seeded_session_db_path)
        (obj,) = _objects(seeded_session_db_path)
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.execute(
            "INSERT INTO detection_runs (id, shot_id, sync_config_id, time_start_s, time_end_s, "
            "detector_model, pose_model, created_at, capture_object_id) "
            "VALUES ('r1', ?, ?, 0, 1, 'aruco:DICT_4X4_50', '', '2026-01-01', ?)",
            (capture_id, sync_id, obj["id"]),
        )
        conn.commit()
        conn.close()

        result = _invoke(["capture", "object", "rm", obj["id"]], seeded_session_db_path)

        assert result.exit_code != 0
        assert "still referenced" in result.output
        assert len(_objects(seeded_session_db_path)) == 1

    def test_without_a_session_it_says_so(self) -> None:
        result = CliRunner().invoke(main, ["capture", "object", "list"], catch_exceptions=False)
        assert result.exit_code != 0
        assert "session DB path is required" in result.output
