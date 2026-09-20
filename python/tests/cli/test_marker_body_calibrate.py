# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``marker-body calibrate``: option handling and where the result goes.

The calibration itself is covered in tests/calibration; here it is replaced by a stub.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from posetrak.calibration.rigid_marker_body import CalibrationResult
from posetrak.cli.main import main

_BODY_YAML = """\
name: calibrated-rigid-body
units: meters
markers:
  - name: aruco_2
    type: aruco
    dictionary: DICT_4X4_50
    id: "2"
    size: 0.05
    corners:
      - [-0.025, 0.025, 0.0]
      - [0.025, 0.025, 0.0]
      - [0.025, -0.025, 0.0]
      - [-0.025, -0.025, 0.0]
"""
_ARGS = ["--time-start", "10", "--time-end", "20", "--marker-size", "0.05", "--marker-ids", "2, 3",
         "--reference-id", "2"]


def _invoke(session_path: Path, *args: str):
    return CliRunner().invoke(main, ["--session", str(session_path), "marker-body", "calibrate", *args],
                              catch_exceptions=False)


def _stub(**changes):
    return patch(
        "posetrak.cli.marker_body.calibrate_rigid_marker_body",
        return_value=CalibrationResult(yaml=_BODY_YAML, reference_solved=3, marker_samples={"3": 2},
                                       marker_corner_std={"3": np.zeros(3)}),
        **changes,
    )


def test_the_result_is_written_and_the_options_reach_the_calibration(
    seeded_session_db_path: Path, capture_id: str, tmp_path: Path
) -> None:
    out = tmp_path / "body.yaml"

    with _stub() as calibrate:
        result = _invoke(seeded_session_db_path, "--capture", capture_id[:8], *_ARGS, "--output", str(out),
                         "--camera", "cam1", "--detect-dots", "--dot-threshold", "180", "--stride", "3", "--name", "bokken")

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == _BODY_YAML
    _, shot_id, time_start, time_end, options = calibrate.call_args.args
    assert (shot_id, time_start, time_end) == (capture_id, 10.0, 20.0)
    assert (options.marker_ids, options.reference_id, options.marker_size) == (["2", "3"], "2", 0.05)
    assert (options.camera_labels, options.detect_dots, options.dot_threshold, options.stride, options.name) == (
        ["cam1"], True, 180, 3, "bokken")


def test_the_result_can_be_imported_into_the_session(
    seeded_session_db_path: Path, capture_id: str
) -> None:
    with _stub():
        result = _invoke(seeded_session_db_path, "--capture", capture_id, *_ARGS, "--import", "--name", "bokken")

    assert result.exit_code == 0, result.output
    conn = sqlite3.connect(str(seeded_session_db_path))
    rows = conn.execute("SELECT id, name, source FROM marker_body_definitions").fetchall()
    conn.close()
    assert [(r[1], r[2]) for r in rows] == [("bokken", f"calibrated from capture {capture_id}")]
    assert result.output.strip().splitlines()[-1] == f"marker_body_definition_id: {rows[0][0]}"


def test_the_option_defaults_are_the_library_defaults() -> None:
    import dataclasses

    from posetrak.calibration.rigid_marker_body import CalibrationOptions
    from posetrak.cli.marker_body import marker_body_calibrate

    defaults = {f.name: getattr(CalibrationOptions(marker_ids=[], reference_id="", marker_size=1.0), f.name)
                for f in dataclasses.fields(CalibrationOptions)}
    checked = [p.name for p in marker_body_calibrate.params if p.name in defaults and not p.required and p.default is not None]

    assert {"stride", "min_cameras", "dictionary", "dot_threshold", "dot_gate_radius_mult"} <= set(checked)
    for name in checked:
        assert next(p for p in marker_body_calibrate.params if p.name == name).default == defaults[name], name


def test_a_result_with_nowhere_to_go_is_rejected_before_any_work(seeded_session_db_path: Path, capture_id: str) -> None:
    with _stub() as calibrate:
        result = _invoke(seeded_session_db_path, "--capture", capture_id, *_ARGS)

    assert result.exit_code != 0 and "nowhere to put the result" in result.output
    calibrate.assert_not_called()


def test_a_reference_that_is_not_a_marker_is_rejected(seeded_session_db_path: Path, capture_id: str, tmp_path: Path) -> None:
    args = [a if a != "2" else "9" for a in _ARGS]                   # --reference-id 9, markers 2, 3

    with _stub() as calibrate:
        result = _invoke(seeded_session_db_path, "--capture", capture_id, *args, "--output", str(tmp_path / "b.yaml"))

    assert result.exit_code != 0 and "reference id must be one of the marker ids" in result.output
    calibrate.assert_not_called()


def test_a_calibration_error_is_reported_and_nothing_is_written(
    seeded_session_db_path: Path, capture_id: str, tmp_path: Path
) -> None:
    out = tmp_path / "body.yaml"

    with patch("posetrak.cli.marker_body.calibrate_rigid_marker_body",
               side_effect=ValueError("capture has no solved extrinsic_calibration_id")):
        result = _invoke(seeded_session_db_path, "--capture", capture_id, *_ARGS, "--output", str(out))

    assert result.exit_code != 0 and "no solved extrinsic_calibration_id" in result.output
    assert not out.exists()


def test_an_unknown_capture_is_reported(seeded_session_db_path: Path, tmp_path: Path) -> None:
    result = _invoke(seeded_session_db_path, "--capture", "nope", *_ARGS, "--output", str(tmp_path / "b.yaml"))
    assert result.exit_code != 0


def test_the_session_is_only_read_unless_import_is_given(
    seeded_session_db_path: Path, capture_id: str, tmp_path: Path
) -> None:
    before = seeded_session_db_path.read_bytes()

    with _stub():
        _invoke(seeded_session_db_path, "--capture", capture_id, *_ARGS, "--output", str(tmp_path / "b.yaml"))

    assert seeded_session_db_path.read_bytes() == before


# ------------------------------------------------------------------ calibrate-video

_INTRINSICS_ID = "abcd1234-0000-4000-8000-000000000001"


def _add_intrinsics(db_path: Path, calibration_id: str = _INTRINSICS_ID, *, width: int = 3840, height: int = 2160) -> None:
    """An intrinsics calibration row (foreign keys are not enforced on this plain connection)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO intrinsics_calibrations (id, camera_mode_id, calibrated_at, distortion_model, fx, fy, cx, cy, "
        "image_width, image_height) VALUES (?, 'mode', '2026-09-06', 'radtan', 1898.0, 1895.0, 1920.0, 1080.0, ?, ?)",
        (calibration_id, width, height),
    )
    conn.commit()
    conn.close()


@pytest.fixture()
def session_with_intrinsics(seeded_session_db_path: Path) -> Path:
    _add_intrinsics(seeded_session_db_path)
    return seeded_session_db_path


def _video_stub():
    from posetrak.calibration.video_marker_body import VideoCalibrationResult

    return patch(
        "posetrak.cli.marker_body.calibrate_marker_body_from_video",
        return_value=VideoCalibrationResult(yaml=_BODY_YAML, frames_used=5, reprojection_rms_px=0.4,
                                            marker_stats={}, unconnected=[]),
    )


def _video_args(tmp_path: Path, *extra: str) -> list[str]:
    video = tmp_path / "orbit.mp4"
    video.write_bytes(b"")
    return ["--video", str(video), "--intrinsics", _INTRINSICS_ID[:8],
            "--body-markers", "2,3:0.06", "--body-marker-size", "0.095", "--reference-id", "2",
            "--anchor-markers", "0,1", "--anchor-marker-size", "0.19", *extra]


def _invoke_video(session_path: Path, *args: str, registry_path: Path | None = None):
    base = ["--session", str(session_path)]
    if registry_path is not None:
        base = ["--registry", str(registry_path), *base]
    return CliRunner().invoke(main, [*base, "marker-body", "calibrate-video", *args], catch_exceptions=False)


def test_a_video_calibration_reaches_the_library_with_the_intrinsics_and_marker_sizes(
    session_with_intrinsics: Path, tmp_path: Path
) -> None:
    out = tmp_path / "body.yaml"

    with _video_stub() as calibrate:
        result = _invoke_video(session_with_intrinsics, *_video_args(
            tmp_path, "--output", str(out), "--detect-dots", "--stride", "3"))

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == _BODY_YAML
    video, intrinsics, options = calibrate.call_args.args
    assert video.endswith("orbit.mp4")
    assert (intrinsics.calibration_id, intrinsics.image_width, intrinsics.image_height) == (_INTRINSICS_ID, 3840, 2160)
    assert intrinsics.state.K[0, 0] == pytest.approx(1898.0)
    assert options.body_markers == {"2": 0.095, "3": 0.06}
    assert options.anchor_markers == {"0": 0.19, "1": 0.19}
    assert (options.reference_id, options.detect_dots, options.stride) == ("2", True, 3)


def test_the_intrinsics_are_found_in_the_registry_when_the_session_lacks_them(
    seeded_session_db_path: Path, registry_db_path: Path, tmp_path: Path
) -> None:
    _add_intrinsics(registry_db_path)

    with _video_stub() as calibrate:
        result = _invoke_video(seeded_session_db_path, *_video_args(tmp_path, "--output", str(tmp_path / "b.yaml")),
                               registry_path=registry_db_path)

    assert result.exit_code == 0, result.output
    assert calibrate.call_args.args[1].calibration_id == _INTRINSICS_ID


def test_unknown_intrinsics_are_reported_before_any_work(
    seeded_session_db_path: Path, registry_db_path: Path, tmp_path: Path
) -> None:
    with _video_stub() as calibrate:
        result = _invoke_video(seeded_session_db_path, *_video_args(tmp_path, "--output", str(tmp_path / "b.yaml")),
                               registry_path=registry_db_path)

    assert result.exit_code != 0
    assert "no intrinsics calibration 'abcd1234'" in result.output and "posetrak calib list" in result.output
    calibrate.assert_not_called()


def test_an_ambiguous_intrinsics_prefix_is_reported(session_with_intrinsics: Path, tmp_path: Path) -> None:
    _add_intrinsics(session_with_intrinsics, "abcd1234-0000-4000-8000-000000000002")

    with _video_stub() as calibrate:
        result = _invoke_video(session_with_intrinsics, *_video_args(tmp_path, "--output", str(tmp_path / "b.yaml")))

    assert result.exit_code != 0 and "ambiguous" in result.output
    calibrate.assert_not_called()


def test_a_video_calibration_can_be_imported_into_the_session(session_with_intrinsics: Path, tmp_path: Path) -> None:
    with _video_stub():
        result = _invoke_video(session_with_intrinsics, *_video_args(tmp_path, "--import", "--name", "bokken"))

    assert result.exit_code == 0, result.output
    conn = sqlite3.connect(str(session_with_intrinsics))
    rows = conn.execute("SELECT name, source FROM marker_body_definitions").fetchall()
    conn.close()
    assert rows == [("bokken", "calibrated from video orbit.mp4")]


def test_a_prop_without_anchors_needs_no_anchor_options(session_with_intrinsics: Path, tmp_path: Path) -> None:
    video = tmp_path / "box.mp4"
    video.write_bytes(b"")

    with _video_stub() as calibrate:
        result = _invoke_video(
            session_with_intrinsics, "--video", str(video), "--intrinsics", _INTRINSICS_ID,
            "--body-markers", "2:0.1,3:0.1,4:0.1", "--reference-id", "2", "--output", str(tmp_path / "b.yaml"),
        )

    assert result.exit_code == 0, result.output
    assert calibrate.call_args.args[2].anchor_markers == {}


def test_bad_video_options_are_rejected_before_any_work(session_with_intrinsics: Path, tmp_path: Path) -> None:
    cases = [
        (["--body-markers", "2,3", "--body-marker-size", "0"], "marker sizes must be positive"),
        (["--body-markers", "2,3"], "no size"),
        (["--reference-id", "9"], "reference id must be one of the body markers"),
        (["--anchor-markers", "2:0.2"], "cannot be both body and anchor markers"),
    ]
    for change, message in cases:
        args = _video_args(tmp_path, "--output", str(tmp_path / "b.yaml"))
        for flag, value in zip(change[::2], change[1::2]):
            args[args.index(flag) + 1] = value
        if "--body-markers" in change and "--body-marker-size" not in change:
            del args[args.index("--body-marker-size"):args.index("--body-marker-size") + 2]

        with _video_stub() as calibrate:
            result = _invoke_video(session_with_intrinsics, *args)

        assert result.exit_code != 0 and message in result.output, (change, result.output)
        calibrate.assert_not_called()


def test_a_video_result_with_nowhere_to_go_is_rejected(session_with_intrinsics: Path, tmp_path: Path) -> None:
    with _video_stub() as calibrate:
        result = _invoke_video(session_with_intrinsics, *_video_args(tmp_path))

    assert result.exit_code != 0 and "nowhere to put the result" in result.output
    calibrate.assert_not_called()


def test_the_video_option_defaults_are_the_library_defaults() -> None:
    import dataclasses

    from posetrak.calibration.video_marker_body import VideoCalibrationOptions
    from posetrak.cli.marker_body import marker_body_calibrate_video

    defaults = {f.name: getattr(VideoCalibrationOptions(body_markers={"0": 1.0}, reference_id="0"), f.name)
                for f in dataclasses.fields(VideoCalibrationOptions)}
    checked = [p.name for p in marker_body_calibrate_video.params
               if p.name in defaults and not p.required and p.default is not None and p.name != "anchor_markers"]

    assert {"stride", "dictionary", "dot_max_distance_m", "dot_min_views", "dot_match_gate_px"} <= set(checked)
    for name in checked:
        assert next(p for p in marker_body_calibrate_video.params if p.name == name).default == defaults[name], name
