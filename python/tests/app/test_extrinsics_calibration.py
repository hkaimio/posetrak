"""Tests for extrinsics calibration filename matching and label normalisation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))
from calibrate_from_exports import _label_from_filename, _match_label, _normalise_label


# ---------------------------------------------------------------------------
# _label_from_filename
# ---------------------------------------------------------------------------

class TestLabelFromFilename:
    def test_standard_export_format(self):
        assert _label_from_filename("frame_00_22_109_gopro_11_mini_01_005193.png") == "gopro 11 mini 01"

    def test_custom_prefix(self):
        assert _label_from_filename("calib_01_30_500_insta_ace2_pro_006886.png") == "insta ace2 pro"

    def test_simple_label(self):
        assert _label_from_filename("frame_00_00_000_pixel9_002653.png") == "pixel9"

    def test_label_with_hyphen_sanitised(self):
        # The export function sanitises hyphens to underscores, so gopro-11 → gopro_11
        assert _label_from_filename("frame_00_22_109_gopro_11_mini_02_013973.png") == "gopro 11 mini 02"

    def test_case_insensitive(self):
        assert _label_from_filename("FRAME_00_22_109_CAM_A_005193.PNG") == "CAM A"

    def test_no_match_missing_timestamp(self):
        assert _label_from_filename("gopro_11_mini_01_005193.png") is None

    def test_no_match_wrong_extension(self):
        assert _label_from_filename("frame_00_22_109_gopro_11_mini_01_005193.jpg") is None

    def test_no_match_plain_name(self):
        assert _label_from_filename("image.png") is None

    def test_frame_number_not_captured(self):
        # The 6-digit frame number must not be included in the label
        label = _label_from_filename("frame_00_22_109_cam_a_123456.png")
        assert label == "cam a"
        assert "123456" not in label

    def test_label_with_numbers(self):
        # Camera names like "cam01" should not be confused with the frame number
        label = _label_from_filename("frame_00_00_000_cam01_000100.png")
        assert label == "cam01"


# ---------------------------------------------------------------------------
# _normalise_label
# ---------------------------------------------------------------------------

class TestNormaliseLabel:
    def test_underscore_to_space(self):
        assert _normalise_label("gopro_11_mini_01") == "gopro 11 mini 01"

    def test_hyphen_to_space(self):
        assert _normalise_label("gopro-11-mini-01") == "gopro 11 mini 01"

    def test_mixed_separators(self):
        assert _normalise_label("gopro-11_mini.01") == "gopro 11 mini 01"

    def test_case_insensitive(self):
        assert _normalise_label("GoPro_Mini") == "gopro mini"

    def test_leading_trailing_stripped(self):
        assert _normalise_label("_cam_") == "cam"

    def test_consecutive_separators_collapsed(self):
        assert _normalise_label("cam__01") == "cam 01"


# ---------------------------------------------------------------------------
# _match_label
# ---------------------------------------------------------------------------

class TestMatchLabel:
    DB = ["gopro-11_mini_01", "gopro-11_mini_02", "insta_ace2_pro", "pixel7", "pixel9"]

    def test_exact_normalised_match(self):
        # Filename sanitises hyphens to underscores; normalisation makes them equal
        assert _match_label("gopro 11 mini 01", self.DB) == "gopro-11_mini_01"

    def test_underscore_vs_hyphen(self):
        assert _match_label("gopro_11_mini_02", self.DB) == "gopro-11_mini_02"

    def test_simple_label_no_separator(self):
        assert _match_label("pixel9", self.DB) == "pixel9"

    def test_pixel7_does_not_match_pixel9(self):
        # The old prefix fallback would have matched "pixel7" against "pixel9"
        # because "pixel" is a common prefix. Exact normalised match avoids this.
        assert _match_label("pixel7", self.DB) == "pixel7"
        assert _match_label("pixel9", self.DB) == "pixel9"

    def test_no_match_returns_none(self):
        assert _match_label("unknown_camera", self.DB) is None

    def test_no_match_partial_prefix(self):
        # "pixel" must NOT match either pixel7 or pixel9 — no prefix matching
        assert _match_label("pixel", self.DB) is None

    def test_case_insensitive(self):
        assert _match_label("PIXEL9", self.DB) == "pixel9"

    def test_empty_labels(self):
        assert _match_label("cam_a", []) is None

    def test_match_against_all_labels_not_just_intrinsics(self):
        # Regression: pixel9 must be findable even if it has no intrinsics.
        # The match function takes a list — caller controls which labels are included.
        all_labels = ["pixel9", "pixel7"]  # both present, regardless of intrinsics
        assert _match_label("pixel9", all_labels) == "pixel9"

    def test_no_false_positive_when_no_intrinsics(self):
        # When pixel9 has no intrinsics, the caller must check *after* matching.
        # Simulate: all_labels includes pixel9, intrinsics_by_label does not.
        all_labels = ["pixel9", "gopro-11_mini_01"]
        intrinsics_by_label = {"gopro-11_mini_01": {}}
        db_label = _match_label("pixel9", all_labels)
        assert db_label == "pixel9"
        assert db_label not in intrinsics_by_label  # caller skips with correct message
