# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for app.setup.video_probe."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.setup.video_probe import (
    VideoProbeResult,
    _build_mode_hint,
    _get,
    _infer_make,
    _parse_focal_length,
    _strip_groups,
    exiftool_available,
    probe_video,
)


# ---------------------------------------------------------------------------
# _strip_groups
# ---------------------------------------------------------------------------


def test_strip_groups_removes_prefix() -> None:
    raw = {"QuickTime:Model": "GoPro", "EXIF:Make": "Canon"}
    result = _strip_groups(raw)
    assert result["Model"] == "GoPro"
    assert result["Make"] == "Canon"


def test_strip_groups_first_value_wins_on_collision() -> None:
    # Two keys map to the same stripped name; first one should win.
    raw = {"QuickTime:VideoFrameRate": 30.0, "Track1:VideoFrameRate": 120.0}
    result = _strip_groups(raw)
    assert result["VideoFrameRate"] == 30.0


def test_strip_groups_no_prefix_passthrough() -> None:
    raw = {"Model": "HERO11 Mini", "Make": "GoPro"}
    assert _strip_groups(raw) == raw


# ---------------------------------------------------------------------------
# _get
# ---------------------------------------------------------------------------


def test_get_returns_first_non_empty() -> None:
    tags = {"Make": "Canon", "AndroidManufacturer": "Google"}
    assert _get(tags, "Make", "AndroidManufacturer") == "Canon"


def test_get_skips_missing_keys() -> None:
    tags = {"AndroidManufacturer": "Google"}
    assert _get(tags, "Make", "AndroidManufacturer") == "Google"


def test_get_returns_none_when_all_missing() -> None:
    assert _get({}, "Make", "Model") is None


def test_get_skips_empty_string() -> None:
    tags = {"Make": "", "Model": "HERO11"}
    assert _get(tags, "Make", "Model") == "HERO11"


def test_get_strips_whitespace() -> None:
    tags = {"Model": "  HERO11 Mini  "}
    assert _get(tags, "Model") == "HERO11 Mini"


# ---------------------------------------------------------------------------
# _parse_focal_length
# ---------------------------------------------------------------------------


def test_parse_focal_length_numeric_string() -> None:
    assert _parse_focal_length("17.0 mm") == 17.0


def test_parse_focal_length_integer_string() -> None:
    assert _parse_focal_length("40mm") == 40.0


def test_parse_focal_length_none() -> None:
    assert _parse_focal_length(None) is None


def test_parse_focal_length_no_digits() -> None:
    assert _parse_focal_length("Unknown") is None


# ---------------------------------------------------------------------------
# _infer_make
# ---------------------------------------------------------------------------


def test_infer_make_gopro_from_compressor_name() -> None:
    tags = {"CompressorName": "GoPro H.265 encoder"}
    assert _infer_make(tags, None) == "GoPro"


def test_infer_make_gopro_from_model_hero_prefix() -> None:
    assert _infer_make({}, "HERO11 Mini") == "GoPro"


def test_infer_make_insta360_from_model() -> None:
    assert _infer_make({}, "Insta360 Ace Pro 2") == "Insta360"


def test_infer_make_google_from_model_pixel() -> None:
    assert _infer_make({}, "Pixel 9 Pro") == "Google"


def test_infer_make_returns_none_for_unknown() -> None:
    assert _infer_make({}, "Unknown Camera") is None


# ---------------------------------------------------------------------------
# _build_mode_hint
# ---------------------------------------------------------------------------


def _make_result(**kwargs) -> VideoProbeResult:
    defaults = dict(width=3840, height=2160, container_fps=30.0, frame_count=300)
    defaults.update(kwargs)
    return VideoProbeResult(**defaults)


def test_mode_hint_gopro_with_fov() -> None:
    r = _make_result(make="GoPro", capture_fps=120.0, field_of_view="Linear")
    hint = _build_mode_hint({}, r)
    assert "Linear" in hint
    assert "120fps" in hint


def test_mode_hint_gopro_no_fov() -> None:
    r = _make_result(make="GoPro", capture_fps=30.0)
    hint = _build_mode_hint({}, r)
    assert "fps" in hint


def test_mode_hint_canon_with_lens() -> None:
    r = _make_result(
        make="Canon",
        capture_fps=59.94,
        lens_model="EF17-40mm f/4L USM",
        focal_length_mm=17.0,
    )
    hint = _build_mode_hint({}, r)
    assert "EF17-40mm" in hint
    assert "17mm" in hint
    assert "59.94fps" in hint


def test_mode_hint_google_slow_mo() -> None:
    r = _make_result(
        make="Google",
        container_fps=29.996,
        capture_fps=120.0,
        width=1920,
        height=1080,
    )
    hint = _build_mode_hint({}, r)
    assert "slow-mo" in hint
    assert "120fps" in hint


def test_mode_hint_google_not_slow_mo_when_fps_equal() -> None:
    r = _make_result(make="Google", container_fps=30.0, capture_fps=30.0)
    hint = _build_mode_hint({}, r)
    # Falls back to generic since capture == container fps
    assert "slow-mo" not in hint


def test_mode_hint_generic_fallback() -> None:
    r = _make_result(make="SomeBrand", capture_fps=60.0)
    hint = _build_mode_hint({}, r)
    assert "×" in hint or "fps" in hint


# ---------------------------------------------------------------------------
# probe_video — cv2 path (no exiftool)
# ---------------------------------------------------------------------------


def _make_fake_cap(width=1920, height=1080, fps=30.0, frame_count=300):
    cap = MagicMock()
    def get_side_effect(prop):
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:  return float(width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT: return float(height)
        if prop == cv2.CAP_PROP_FPS:          return float(fps)
        if prop == cv2.CAP_PROP_FRAME_COUNT:  return float(frame_count)
        return 0.0
    cap.get.side_effect = get_side_effect
    return cap


@patch("cv2.VideoCapture")
def test_probe_video_cv2_only(mock_vc) -> None:
    mock_vc.return_value = _make_fake_cap(1920, 1080, 60.0, 600)
    result = probe_video(Path("/fake/video.mp4"))
    assert result.width == 1920
    assert result.height == 1080
    assert result.container_fps == 60.0
    assert result.frame_count == 600
    assert result.make is None
    assert not result.exiftool_available


# ---------------------------------------------------------------------------
# probe_video — exiftool enrichment (mocked)
# ---------------------------------------------------------------------------


def _make_exiftool_mock(tags: dict):
    """Build a context-manager mock for ExifToolHelper that returns *tags*."""
    et_instance = MagicMock()
    et_instance.__enter__ = MagicMock(return_value=et_instance)
    et_instance.__exit__ = MagicMock(return_value=False)
    et_instance.get_metadata.return_value = [tags]

    et_module = MagicMock()
    et_module.ExifToolHelper.return_value = et_instance
    return et_module


@patch("cv2.VideoCapture")
def test_probe_video_gopro_enriched(mock_vc) -> None:
    mock_vc.return_value = _make_fake_cap(2704, 2028, 119.88, 1800)
    gopro_tags = {
        "QuickTime:Model": "HERO11 Mini",
        "QuickTime:CameraSerialNumber": "C3461324988885",
        "QuickTime:FieldOfView": "Linear",
        "QuickTime:FirmwareVersion": "H22.01.01.10.00",
        "QuickTime:CompressorName": "GoPro H.265 encoder",
        "QuickTime:VideoFrameRate": 119.88,
    }
    with patch.dict("sys.modules", {"exiftool": _make_exiftool_mock(gopro_tags)}):
        result = probe_video(Path("/fake/gopro.mp4"))

    assert result.make == "GoPro"
    assert result.model == "HERO11 Mini"
    assert result.serial_number == "C3461324988885"
    assert result.field_of_view == "Linear"
    assert result.firmware == "H22.01.01.10.00"
    assert result.capture_fps == pytest.approx(119.88)
    assert result.exiftool_available


@patch("cv2.VideoCapture")
def test_probe_video_canon_enriched(mock_vc) -> None:
    mock_vc.return_value = _make_fake_cap(3840, 2160, 59.94, 1000)
    canon_tags = {
        "EXIF:Make": "Canon",
        "EXIF:Model": "Canon EOS R5",
        "EXIF:SerialNumber": "083021000890",
        "Canon:LensModel": "EF17-40mm f/4L USM",
        "Canon:FocalLength": "17.0 mm",
        "Canon:CanonFirmwareVersion": "Firmware Version 1.10.0",
        "QuickTime:VideoFrameRate": 59.94,
    }
    with patch.dict("sys.modules", {"exiftool": _make_exiftool_mock(canon_tags)}):
        result = probe_video(Path("/fake/canon.mp4"))

    assert result.make == "Canon"
    assert result.model == "Canon EOS R5"
    assert result.serial_number == "083021000890"
    assert result.focal_length_mm == pytest.approx(17.0)
    assert result.firmware == "1.10.0"   # "Firmware Version " prefix stripped
    assert result.capture_fps == pytest.approx(59.94)


@patch("cv2.VideoCapture")
def test_probe_video_android_slow_mo(mock_vc) -> None:
    mock_vc.return_value = _make_fake_cap(1920, 1080, 29.996, 300)
    android_tags = {
        "QuickTime:AndroidManufacturer": "Google",
        "QuickTime:AndroidModel": "Pixel 9 Pro",
        "QuickTime:AndroidCaptureFPS": 120.0,
        "QuickTime:VideoFrameRate": 29.996,
    }
    with patch.dict("sys.modules", {"exiftool": _make_exiftool_mock(android_tags)}):
        result = probe_video(Path("/fake/pixel.mp4"))

    assert result.make == "Google"
    assert result.model == "Pixel 9 Pro"
    assert result.capture_fps == pytest.approx(120.0)   # AndroidCaptureFPS wins
    assert result.container_fps == pytest.approx(29.996)
    assert result.mode_hint is not None
    assert "slow-mo" in result.mode_hint


@patch("cv2.VideoCapture")
def test_probe_video_insta360_enriched(mock_vc) -> None:
    mock_vc.return_value = _make_fake_cap(3840, 2160, 30.0, 900)
    insta_tags = {
        "QuickTime:Model": "Insta360 Ace Pro 2",
        "QuickTime:SerialNumber": "IBGLA2412JHVRQ",
        "QuickTime:FirmwareVersion": "2.0.5",
        "QuickTime:VideoFrameRate": 30.0,
    }
    with patch.dict("sys.modules", {"exiftool": _make_exiftool_mock(insta_tags)}):
        result = probe_video(Path("/fake/insta360.mp4"))

    assert result.make == "Insta360"
    assert result.model == "Insta360 Ace Pro 2"
    assert result.serial_number == "IBGLA2412JHVRQ"
    assert result.firmware == "2.0.5"


@patch("cv2.VideoCapture")
def test_probe_video_exiftool_import_error_graceful(mock_vc) -> None:
    """If pyexiftool is not installed, probe_video should still return cv2 data."""
    mock_vc.return_value = _make_fake_cap()
    # Remove exiftool from sys.modules so the import fails
    with patch.dict("sys.modules", {"exiftool": None}):
        result = probe_video(Path("/fake/video.mp4"))
    assert result.width == 1920
    assert not result.exiftool_available


@patch("cv2.VideoCapture")
def test_probe_video_exiftool_runtime_error_graceful(mock_vc) -> None:
    """If exiftool subprocess fails, result should still contain cv2 data."""
    mock_vc.return_value = _make_fake_cap()

    et_instance = MagicMock()
    et_instance.__enter__ = MagicMock(return_value=et_instance)
    et_instance.__exit__ = MagicMock(return_value=False)
    et_instance.get_metadata.side_effect = RuntimeError("exiftool crashed")

    et_module = MagicMock()
    et_module.ExifToolHelper.return_value = et_instance

    with patch.dict("sys.modules", {"exiftool": et_module}):
        result = probe_video(Path("/fake/video.mp4"))

    assert result.width == 1920
    assert not result.exiftool_available


# ---------------------------------------------------------------------------
# exiftool_available
# ---------------------------------------------------------------------------


def test_exiftool_available_false_when_import_fails() -> None:
    with patch.dict("sys.modules", {"exiftool": None}):
        assert not exiftool_available()


def test_exiftool_available_false_when_binary_missing() -> None:
    with patch.dict("sys.modules", {"exiftool": MagicMock()}):
        with patch("shutil.which", return_value=None):
            assert not exiftool_available()


def test_exiftool_available_true() -> None:
    with patch.dict("sys.modules", {"exiftool": MagicMock()}):
        with patch("shutil.which", return_value="/usr/bin/exiftool"):
            assert exiftool_available()
