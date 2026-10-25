# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests of named camera modes with capture settings, and their schema migration."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from posetrak.cli.main import main
from posetrak.db.db import (
    REGISTRY_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION,
    create_camera_mode,
    create_camera_model,
    create_registry,
    create_session,
    get_schema_version,
    list_camera_modes,
    open_registry,
    open_session,
    update_camera_mode,
)


@pytest.fixture
def registry(tmp_path: Path):
    reg = create_registry(tmp_path / "registry.db")
    yield reg
    reg.close()


def test_a_mode_keeps_its_name_and_settings(registry) -> None:
    model = create_camera_model(registry, manufacturer="GoPro", model_name="Hero13")
    mode = create_camera_mode(
        registry, model, width_px=3840, height_px=2160, nominal_fps=120, name="4K 120 linear",
        capture_settings={"resolution": "4K", "fps": 120, "lens": "linear"},
    )
    plain = create_camera_mode(registry, model, width_px=1920, height_px=1080, nominal_fps=30)
    rows = {r["id"]: r for r in list_camera_modes(registry, model)}
    assert rows[mode]["name"] == "4K 120 linear"
    assert json.loads(rows[mode]["capture_settings"]) == {"resolution": "4K", "fps": 120, "lens": "linear"}
    assert rows[plain]["name"] is None and rows[plain]["capture_settings"] is None


def test_names_are_unique_within_a_model_only(registry) -> None:
    a = create_camera_model(registry, manufacturer="GoPro", model_name="Hero13")
    b = create_camera_model(registry, manufacturer="GoPro", model_name="Hero11")
    create_camera_mode(registry, a, name="wide")
    create_camera_mode(registry, b, name="wide")
    with pytest.raises(sqlite3.IntegrityError):
        create_camera_mode(registry, a, name="wide")
    create_camera_mode(registry, a)
    create_camera_mode(registry, a)  # unnamed modes are not constrained


def test_update_changes_only_what_is_given(registry) -> None:
    model = create_camera_model(registry, manufacturer="GoPro", model_name="Hero13")
    mode = create_camera_mode(registry, model, name="old", capture_settings={"fps": 60})
    update_camera_mode(registry, mode, name="new")
    row = list_camera_modes(registry, model)[0]
    assert (row["name"], json.loads(row["capture_settings"])) == ("new", {"fps": 60})
    update_camera_mode(registry, mode, capture_settings={})
    assert list_camera_modes(registry, model)[0]["capture_settings"] is None
    with pytest.raises(KeyError):
        update_camera_mode(registry, "missing", name="x")


def test_older_databases_gain_the_columns(tmp_path: Path) -> None:
    reg_path = tmp_path / "registry.db"
    reg = create_registry(reg_path)
    model = create_camera_model(reg, manufacturer="GoPro", model_name="Hero13")
    create_camera_mode(reg, model, width_px=1920, height_px=1080, nominal_fps=30)
    reg.execute("DROP INDEX idx_camera_modes_model_name")
    reg.execute("ALTER TABLE camera_modes DROP COLUMN name")
    reg.execute("ALTER TABLE camera_modes DROP COLUMN capture_settings")
    reg.execute("PRAGMA user_version = 21")
    reg.commit()
    reg.close()
    reg = open_registry(reg_path)
    assert get_schema_version(reg) == REGISTRY_SCHEMA_VERSION
    row = list_camera_modes(reg)[0]
    assert (row["width_px"], row["name"], row["capture_settings"]) == (1920, None, None)
    reg.close()

    sess_path = tmp_path / "session.db"
    sess = create_session(sess_path)
    sess.execute("DROP INDEX idx_camera_modes_model_name")
    for table, column in (
        ("camera_modes", "name"),
        ("camera_modes", "capture_settings"),
        ("capture_events", "camera_mode_id"),
        ("capture_events", "settings_json"),
    ):
        sess.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    sess.execute("PRAGMA user_version = 72")
    sess.commit()
    sess.close()
    sess = open_session(sess_path)
    assert get_schema_version(sess) == SESSION_SCHEMA_VERSION
    columns = {r[1] for r in sess.execute("PRAGMA table_info(capture_events)")}
    assert {"camera_mode_id", "settings_json"} <= columns
    sess.close()


def test_cli_adds_and_edits_modes(tmp_path: Path) -> None:
    path = tmp_path / "registry.db"
    reg = create_registry(path)
    model = create_camera_model(reg, manufacturer="GoPro", model_name="Hero13")
    reg.close()

    def run(*args: str):
        return CliRunner().invoke(main, ["--registry", str(path), *args])

    added = run(
        "camera-mode", "add", "--model-id", model, "--width", "3840", "--height", "2160", "--fps", "120",
        "--name", "4K 120 linear", "--settings", '{"lens": "linear"}',
    )
    assert added.exit_code == 0, added.output
    mode_id = added.output.split("camera_mode_id:")[1].split()[0]
    listed = run("camera-mode", "list")
    assert "4K 120 linear" in listed.output and "linear" in listed.output

    assert run("camera-mode", "set", mode_id, "--settings", '{"lens": "wide"}').exit_code == 0
    assert run("camera-mode", "set", mode_id, "--name", "4K 120 wide").exit_code == 0
    reg = open_registry(path)
    row = list_camera_modes(reg)[0]
    assert (row["name"], json.loads(row["capture_settings"])) == ("4K 120 wide", {"lens": "wide"})
    reg.close()

    assert run("camera-mode", "add", "--model-id", model, "--settings", "[1]").exit_code != 0
    assert run("camera-mode", "add", "--model-id", model, "--settings", "{oops").exit_code != 0
    duplicate = run("camera-mode", "add", "--model-id", model, "--name", "4K 120 wide")
    assert duplicate.exit_code != 0
    assert run("camera-mode", "set", mode_id).exit_code != 0
