# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure-Python pieces of posetrak.cli.extrinsics_rig.

The commands themselves (anchor-rig, reanchor) are I/O-heavy -- real video
decode, real multi-camera solving -- and were instead validated end-to-end
against real capture footage and a real registry (see
docs/roadmap/features/extrinsics-improvements/status.md, 2026-08-12
entry), the same "validate against real data" preference this whole
feature has followed throughout. What's covered here is the parsing/
resolution logic that doesn't need real video or a full session: camera
spec parsing and intrinsics-mode resolution against a synthetic session DB.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import click
import numpy as np
import pytest
from click.testing import CliRunner

from posetrak.cli.extrinsics_rig import (
    _CameraSpec,
    _label_to_instance_id,
    _parse_camera_spec,
    _resolve_intrinsics,
)
from posetrak.cli.main import main
from posetrak.db.db import create_mocap_session, create_session, open_session
from posetrak.db.manage_marker_body import upsert_scene_marker_body


# ---------------------------------------------------------------------------
# _parse_camera_spec
# ---------------------------------------------------------------------------


def test_parse_camera_spec_basic():
    spec = _parse_camera_spec("insta_ace2_pro|4K 120 fps linear|D:/videos/x.mp4|2069")
    assert spec == _CameraSpec(
        label="insta_ace2_pro", camera_mode="4K 120 fps linear",
        video_path="D:/videos/x.mp4", frame_idx=2069,
    )


def test_parse_camera_spec_windows_drive_letter_path():
    """A literal ':' in a Windows path must not break the '|'-delimited parse."""
    spec = _parse_camera_spec("cam1||D:/mocap/videos/test-1.mp4|100")
    assert spec.video_path == "D:/mocap/videos/test-1.mp4"


def test_parse_camera_spec_empty_mode_is_none():
    spec = _parse_camera_spec("cam1||video.mp4|0")
    assert spec.camera_mode is None


def test_parse_camera_spec_wrong_field_count_raises():
    with pytest.raises(click.UsageError):
        _parse_camera_spec("cam1|mode|video.mp4")


def test_parse_camera_spec_non_integer_frame_raises():
    with pytest.raises(click.UsageError):
        _parse_camera_spec("cam1|mode|video.mp4|not-a-number")


# ---------------------------------------------------------------------------
# _resolve_intrinsics
# ---------------------------------------------------------------------------


@pytest.fixture()
def session_with_two_modes(tmp_path):
    conn = create_session(tmp_path / "session.db")
    conn.execute(
        "INSERT INTO camera_models (id, manufacturer, model_name) VALUES ('model1', 'Insta360', 'ACE2 Pro')"
    )
    conn.execute(
        "INSERT INTO camera_instances (id, camera_model_id, label) "
        "VALUES ('inst1', 'model1', 'insta_ace2_pro')"
    )
    conn.execute(
        "INSERT INTO camera_modes (id, camera_model_id, width_px, height_px, nominal_fps, notes) "
        "VALUES ('mode_mega', 'model1', 3840, 2160, 119.88, 'MEGA mode 4K 120 fps')"
    )
    # camera_modes.default_intrinsics_calibration_id and
    # intrinsics_calibrations.camera_mode_id reference each other -- insert
    # the mode row first (no default yet), then the calibration, then
    # backfill the default, same order create_registry's own seed data uses.
    conn.execute(
        "INSERT INTO camera_modes (id, camera_model_id, width_px, height_px, nominal_fps, notes) "
        "VALUES ('mode_linear', 'model1', 3840, 2160, 120.0, '4K 120 fps linear')"
    )
    conn.execute(
        "INSERT INTO intrinsics_calibrations "
        "(id, camera_mode_id, calibrated_at, fx, fy, cx, cy) "
        "VALUES ('calib_linear', 'mode_linear', '2026-01-01', 1000.0, 1000.0, 960.0, 540.0)"
    )
    conn.execute(
        "UPDATE camera_modes SET default_intrinsics_calibration_id = 'calib_linear' "
        "WHERE id = 'mode_linear'"
    )
    conn.commit()
    yield conn
    conn.close()


def test_resolve_intrinsics_ambiguous_without_mode_raises(session_with_two_modes):
    with pytest.raises(click.ClickException, match="2 matching camera_modes"):
        _resolve_intrinsics(session_with_two_modes, "insta_ace2_pro", None)


def test_resolve_intrinsics_disambiguated_by_mode_substring(session_with_two_modes):
    intr = _resolve_intrinsics(session_with_two_modes, "insta_ace2_pro", "linear")
    assert intr["K"][0, 0] == pytest.approx(1000.0)
    assert intr["fisheye"] is False


def test_resolve_intrinsics_unknown_camera_raises(session_with_two_modes):
    with pytest.raises(click.ClickException, match="No camera_modes found"):
        _resolve_intrinsics(session_with_two_modes, "nonexistent_camera", None)


def test_resolve_intrinsics_mode_with_no_calibration_raises(tmp_path):
    conn = create_session(tmp_path / "session2.db")
    conn.execute(
        "INSERT INTO camera_models (id, manufacturer, model_name) VALUES ('model1', 'X', 'Y')"
    )
    conn.execute(
        "INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('inst1', 'model1', 'cam1')"
    )
    conn.execute(
        "INSERT INTO camera_modes (id, camera_model_id, width_px, height_px, nominal_fps) "
        "VALUES ('mode1', 'model1', 1920, 1080, 30.0)"
    )
    conn.commit()
    with pytest.raises(click.ClickException, match="no intrinsics_calibrations"):
        _resolve_intrinsics(conn, "cam1", None)
    conn.close()


# ---------------------------------------------------------------------------
# _label_to_instance_id
# ---------------------------------------------------------------------------


def test_label_to_instance_id(session_with_two_modes):
    mapping = _label_to_instance_id(session_with_two_modes)
    assert mapping == {"insta_ace2_pro": "inst1"}


# ---------------------------------------------------------------------------
# scene-marker list / delete
# ---------------------------------------------------------------------------


@pytest.fixture()
def session_with_scene_markers(tmp_path: Path):
    """A session DB path with one mocap_sessions row and two scene markers
    (a rig anchor + a scattered tag) already stored. Returns
    (db_path, session_id)."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)
    session_id = create_mocap_session(conn, location="lab")
    upsert_scene_marker_body(
        conn, session_id, label="rig:calib-box", R=np.eye(3), t=np.zeros(3),
        marker_body_definition_id="def1", is_primary_anchor=True,
    )
    upsert_scene_marker_body(
        conn, session_id, label="tag:7", R=np.eye(3), t=np.array([1.0, 2.0, 3.0]),
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="7", marker_size=0.1,
    )
    conn.commit()
    conn.close()
    return db_path, session_id


def test_scene_marker_list_empty_session(cli_runner: CliRunner, session_db_path: Path) -> None:
    conn = open_session(session_db_path)
    session_id = create_mocap_session(conn, location="lab")
    conn.commit()
    conn.close()

    result = cli_runner.invoke(
        main,
        ["--session", str(session_db_path), "extrinsics", "scene-marker", "list",
         "--session", session_id],
    )
    assert result.exit_code == 0, result.output
    assert "No scene markers" in result.output


def test_scene_marker_list_shows_both(
    cli_runner: CliRunner, session_with_scene_markers,
) -> None:
    db_path, session_id = session_with_scene_markers
    result = cli_runner.invoke(
        main,
        ["--session", str(db_path), "extrinsics", "scene-marker", "list", "--session", session_id],
    )
    assert result.exit_code == 0, result.output
    assert "rig:calib-box" in result.output
    assert "tag:7" in result.output


def test_scene_marker_delete_removes_row(
    cli_runner: CliRunner, session_with_scene_markers,
) -> None:
    db_path, session_id = session_with_scene_markers
    result = cli_runner.invoke(
        main,
        ["--session", str(db_path), "extrinsics", "scene-marker", "delete",
         "--session", session_id, "rig:calib-box"],
    )
    assert result.exit_code == 0, result.output
    assert "Deleted" in result.output

    result = cli_runner.invoke(
        main,
        ["--session", str(db_path), "extrinsics", "scene-marker", "list", "--session", session_id],
    )
    assert "rig:calib-box" not in result.output
    assert "tag:7" in result.output


def test_scene_marker_delete_missing_label_errors(
    cli_runner: CliRunner, session_with_scene_markers,
) -> None:
    db_path, session_id = session_with_scene_markers
    result = cli_runner.invoke(
        main,
        ["--session", str(db_path), "extrinsics", "scene-marker", "delete",
         "--session", session_id, "nonexistent"],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# scene-marker groups / delete --group (group_name, 2026-08-12)
# ---------------------------------------------------------------------------


@pytest.fixture()
def session_with_named_groups(tmp_path: Path):
    """Two named groups ("room7", "room8") each with one tag reusing the
    same marker id, plus one ungrouped tag. Returns (db_path, session_id)."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)
    session_id = create_mocap_session(conn, location="lab")
    upsert_scene_marker_body(
        conn, session_id, label="tag:3", R=np.eye(3), t=np.zeros(3), group_name="room7",
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="3", marker_size=0.1,
    )
    upsert_scene_marker_body(
        conn, session_id, label="tag:3", R=np.eye(3), t=np.array([9.0, 0.0, 0.0]),
        group_name="room8",
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="3", marker_size=0.1,
    )
    upsert_scene_marker_body(
        conn, session_id, label="tag:9", R=np.eye(3), t=np.zeros(3),
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="9", marker_size=0.1,
    )
    conn.commit()
    conn.close()
    return db_path, session_id


def test_scene_marker_groups_lists_named_groups_only(
    cli_runner: CliRunner, session_with_named_groups,
) -> None:
    db_path, session_id = session_with_named_groups
    result = cli_runner.invoke(
        main,
        ["--session", str(db_path), "extrinsics", "scene-marker", "groups",
         "--session", session_id],
    )
    assert result.exit_code == 0, result.output
    assert "room7" in result.output
    assert "room8" in result.output


def test_scene_marker_groups_empty_when_none_named(
    cli_runner: CliRunner, session_with_scene_markers,
) -> None:
    db_path, session_id = session_with_scene_markers
    result = cli_runner.invoke(
        main,
        ["--session", str(db_path), "extrinsics", "scene-marker", "groups",
         "--session", session_id],
    )
    assert result.exit_code == 0, result.output
    assert "No named scene-marker groups" in result.output


def test_scene_marker_delete_scoped_to_group(
    cli_runner: CliRunner, session_with_named_groups,
) -> None:
    db_path, session_id = session_with_named_groups
    result = cli_runner.invoke(
        main,
        ["--session", str(db_path), "extrinsics", "scene-marker", "delete",
         "--session", session_id, "--group", "room7", "tag:3"],
    )
    assert result.exit_code == 0, result.output

    result = cli_runner.invoke(
        main,
        ["--session", str(db_path), "extrinsics", "scene-marker", "list", "--session", session_id],
    )
    # room8's tag:3 (same label, different group) must survive.
    assert result.output.count("tag:3") == 1
