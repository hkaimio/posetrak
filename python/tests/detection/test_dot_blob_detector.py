# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for anonymous reflective-dot blob detection (see
docs/roadmap/features/marker-based-mocap/reflective-dot-detection-design.md).

Synthetic frames only -- the detector's default threshold/area/compactness
values are validated against real footage separately (that design doc's
§2.1); these tests exercise the detection logic itself (threshold, area
gate, compactness gate, centroid accuracy) in isolation.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from posetrak.detection.dot_blob_detector import detect_blobs


def _blank_frame(size: int = 200, fill: int = 20) -> np.ndarray:
    return np.full((size, size), fill, dtype=np.uint8)


def _draw_dot(frame: np.ndarray, cx: int, cy: int, radius: int, value: int = 250) -> None:
    cv2.circle(frame, (cx, cy), radius, value, thickness=-1)


def test_detect_blobs_finds_a_bright_round_dot() -> None:
    frame = _blank_frame()
    _draw_dot(frame, 100, 80, radius=6)

    blobs = detect_blobs(frame)

    assert len(blobs) == 1
    assert blobs[0].cx == pytest.approx(100.0, abs=1.0)
    assert blobs[0].cy == pytest.approx(80.0, abs=1.0)
    assert blobs[0].compactness > 0.8  # a filled circle is close to 1.0


def test_detect_blobs_finds_several_dots_independently() -> None:
    frame = _blank_frame()
    _draw_dot(frame, 40, 40, radius=5)
    _draw_dot(frame, 150, 60, radius=5)
    _draw_dot(frame, 90, 160, radius=5)

    blobs = detect_blobs(frame)

    centers = sorted((round(b.cx), round(b.cy)) for b in blobs)
    assert centers == [(40, 40), (90, 160), (150, 60)]


def test_detect_blobs_rejects_dim_spots_below_threshold() -> None:
    frame = _blank_frame()
    _draw_dot(frame, 100, 100, radius=6, value=150)  # bright, but below default threshold=235

    assert detect_blobs(frame) == []


def test_detect_blobs_rejects_area_outside_range() -> None:
    frame = _blank_frame(size=400)
    _draw_dot(frame, 100, 100, radius=1)   # area well under min_area=4.0
    _draw_dot(frame, 300, 300, radius=40)  # area well over max_area=400.0

    assert detect_blobs(frame) == []


def test_detect_blobs_rejects_elongated_glare_streak() -> None:
    """A shape filter (compactness), not just brightness+area, is needed to
    reject a glare streak (e.g. a shiny edge) that happens to fall inside
    the area range but is nothing like a round dot."""
    frame = _blank_frame()
    cv2.rectangle(frame, (60, 98), (140, 102), 250, thickness=-1)  # 80x4 streak

    assert detect_blobs(frame) == []


def test_detect_blobs_thresholds_and_area_are_overridable() -> None:
    frame = _blank_frame()
    _draw_dot(frame, 100, 100, radius=6, value=150)

    assert detect_blobs(frame, threshold=100) != []
    assert detect_blobs(frame, threshold=100, min_area=10000.0) == []


def test_detect_blobs_reports_axes_close_to_diameter_for_a_round_dot() -> None:
    frame = _blank_frame()
    _draw_dot(frame, 100, 80, radius=6)  # diameter ~12

    blobs = detect_blobs(frame)

    assert len(blobs) == 1
    assert blobs[0].major_axis_px == pytest.approx(12.0, abs=2.0)
    assert blobs[0].minor_axis_px == pytest.approx(12.0, abs=2.0)


def test_detect_blobs_accepts_a_short_motion_blur_streak_with_dot_like_width() -> None:
    """A real dot's width doesn't change under motion blur, only its length
    grows -- a short streak (dot-width, well under max_streak_length_px)
    should be accepted and reported with its real length/width, not
    rejected the way an arbitrarily long glare streak is (see the test
    below, unchanged)."""
    frame = _blank_frame()
    cv2.rectangle(frame, (60, 97), (85, 103), 250, thickness=-1)  # 25x6 streak, compactness ~0.49

    blobs = detect_blobs(frame)

    assert len(blobs) == 1
    assert blobs[0].major_axis_px == pytest.approx(25.0, abs=2.0)
    assert blobs[0].minor_axis_px == pytest.approx(6.0, abs=2.0)


def test_detect_blobs_rejects_a_streak_too_long_to_be_realistic_blur() -> None:
    """A streak whose width falls in the round-dot diameter range must
    still be rejected once it's longer than any realistic motion blur --
    the original glare-streak test above (80x4) already covers this at a
    narrower width; this checks it holds at a dot-like width too."""
    frame = _blank_frame()
    cv2.rectangle(frame, (20, 96), (180, 104), 250, thickness=-1)  # 160x8, dot-width but too long

    assert detect_blobs(frame) == []
