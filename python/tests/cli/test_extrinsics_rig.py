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

import click
import pytest

from posetrak.cli.extrinsics_rig import (
    _CameraSpec,
    _label_to_instance_id,
    _parse_camera_spec,
    _resolve_intrinsics,
)
from posetrak.db.db import create_session


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
