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

from posetrak.detection.dot_blob_detector import compute_background, detect_blobs


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


def test_detect_blobs_reports_no_direction_for_a_round_dot() -> None:
    frame = _blank_frame()
    _draw_dot(frame, 100, 80, radius=6)

    blobs = detect_blobs(frame)

    assert len(blobs) == 1
    assert blobs[0].dir_x == 0.0
    assert blobs[0].dir_y == 0.0


def test_detect_blobs_reports_a_streaks_own_axis_direction() -> None:
    """A horizontal streak's direction should come back as (approximately)
    a horizontal unit vector, canonicalized to the dy>=0 half-plane (here
    dy==0, so dx>=0) -- see dot_blob_detector.py's own "Streak direction"
    docstring section for why the sign can't be resolved from the blob
    alone."""
    frame = _blank_frame()
    cv2.rectangle(frame, (60, 97), (85, 103), 250, thickness=-1)  # horizontal streak

    blobs = detect_blobs(frame)

    assert len(blobs) == 1
    assert blobs[0].dir_x == pytest.approx(1.0, abs=0.05)
    assert blobs[0].dir_y == pytest.approx(0.0, abs=0.05)


def test_detect_blobs_canonicalizes_a_vertical_streaks_direction_sign() -> None:
    """A vertical streak's direction should land in the canonical dy>=0
    half-plane regardless of which way the rectangle happens to be drawn."""
    frame = _blank_frame()
    cv2.rectangle(frame, (97, 60), (103, 85), 250, thickness=-1)  # vertical streak

    blobs = detect_blobs(frame)

    assert len(blobs) == 1
    assert blobs[0].dir_x == pytest.approx(0.0, abs=0.05)
    assert blobs[0].dir_y == pytest.approx(1.0, abs=0.05)


def test_detect_blobs_rejects_a_streak_too_long_to_be_realistic_blur() -> None:
    """A streak whose width falls in the round-dot diameter range must
    still be rejected once it's longer than any realistic motion blur --
    the original glare-streak test above (80x4) already covers this at a
    narrower width; this checks it holds at a dot-like width too."""
    frame = _blank_frame()
    cv2.rectangle(frame, (20, 96), (180, 104), 250, thickness=-1)  # 160x8, dot-width but too long

    assert detect_blobs(frame) == []


def test_detect_blobs_accepts_a_wide_area_streak_that_shape_alone_would_pass() -> None:
    """Shape is classified before `area`: a legitimately dot-width streak whose raw pixel count happens to exceed
    max_area=400 (its area scales with length, not just width) was
    discarded before its shape was ever considered. This one is 55x8 --
    area=440, over max_area, but a real streak by every shape criterion
    (width in the round-dot range, length under max_streak_length_px)."""
    frame = _blank_frame()
    cv2.rectangle(frame, (60, 96), (115, 104), 250, thickness=-1)  # 55x8, area=440

    blobs = detect_blobs(frame)

    assert len(blobs) == 1
    assert blobs[0].major_axis_px == pytest.approx(55.0, abs=2.0)
    assert blobs[0].minor_axis_px == pytest.approx(8.0, abs=2.0)


def test_detect_blobs_background_subtraction_finds_a_dim_moving_highlight() -> None:
    """A streak dimmed by motion blur can fall below any fixed absolute
    brightness threshold that also has to stay high enough to reject the
    room's own bright static texture -- background subtraction thresholds
    the *residual* (this frame minus the normal per-pixel background)
    instead, which stays near zero for anything static regardless of how
    low the threshold is set, so a real but dim highlight can be found."""
    background = _blank_frame(fill=20)
    frame = background.copy()
    _draw_dot(frame, 100, 100, radius=6, value=90)  # dim: rejected by any threshold near 235

    assert detect_blobs(frame, threshold=235) == []
    assert detect_blobs(frame, threshold=235, background=background) == []  # residual ~70, still gated

    blobs = detect_blobs(frame, threshold=50, background=background)
    assert len(blobs) == 1
    assert blobs[0].cx == pytest.approx(100.0, abs=1.0)


def test_detect_blobs_background_subtraction_ignores_static_texture() -> None:
    """The room's own static bright texture must NOT reappear as a false
    detection just because background subtraction allows a much lower
    threshold -- a pixel identical to its own background has ~zero
    residual regardless of its absolute brightness."""
    background = _blank_frame(fill=20)
    background[40:60, 40:60] = 200  # a bright, but permanently-there, patch
    frame = background.copy()  # nothing changed from the background this frame

    assert detect_blobs(frame, threshold=10, background=background) == []


def test_compute_background_is_the_per_pixel_median_of_the_samples() -> None:
    frames = [_blank_frame(fill=v) for v in (10, 12, 200)]  # outlier shouldn't win
    bg = compute_background(frames)
    assert bg[0, 0] == 12


def test_detect_blobs_max_saturation_rejects_a_skin_toned_highlight() -> None:
    """A real reflective dot is white/near-neutral (low saturation); skin
    in motion can also produce a bright, otherwise-dot-shaped residual --
    max_saturation (with the original color frame) tells them apart by
    color even when brightness/shape alone can't."""
    bgr = np.full((200, 200, 3), 20, dtype=np.uint8)
    skin_bgr = (90, 140, 220)  # a warm, saturated (skin-like) BGR color
    white_bgr = (245, 245, 245)  # a near-neutral bright color
    cv2.circle(bgr, (60, 60), 6, skin_bgr, thickness=-1)
    cv2.circle(bgr, (140, 140), 6, white_bgr, thickness=-1)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Both clear a plain brightness threshold with no chroma check.
    both = detect_blobs(gray, threshold=100)
    assert len(both) == 2

    filtered = detect_blobs(gray, threshold=100, bgr=bgr, max_saturation=60.0)
    assert len(filtered) == 1
    assert filtered[0].cx == pytest.approx(140.0, abs=1.0)


def test_detect_blobs_blacklist_mode_recovers_a_marker_subtract_mode_fuses_away() -> None:
    """A marker sitting on a subject occupying a
    pose the background model's samples didn't cover reads as one large,
    connected, non-round blob under 'subtract' mode (the subject's own
    limb + the marker fused together, both crossing the same residual
    threshold) -- shape-rejected as a whole. 'blacklist' mode thresholds
    the live frame directly, so the marker's own small round contour is
    never fused with the surrounding limb in the first place."""
    background = _blank_frame(fill=20)
    frame = background.copy()
    # A bright limb-sized region appears (a subject in an atypical pose) --
    # itself well below `threshold`, but at exactly the level 'subtract'
    # mode's residual would need to explain the marker's own presence too.
    cv2.rectangle(frame, (40, 40), (160, 160), 90, thickness=-1)
    _draw_dot(frame, 100, 100, radius=6, value=250)

    subtract_result = detect_blobs(frame, threshold=60, background=background, background_mode="subtract")
    assert subtract_result == []  # fused into one large non-round blob, correctly shape-rejected as a whole

    blacklist_result = detect_blobs(frame, threshold=200, background=background, background_mode="blacklist")
    assert len(blacklist_result) == 1
    assert blacklist_result[0].cx == pytest.approx(100.0, abs=1.0)


def test_detect_blobs_blacklist_mode_rejects_a_spot_thats_always_bright() -> None:
    """A fixed light/glare source -- already nearly as bright in the
    background with no subject present at all -- should still be vetoed
    under 'blacklist' mode, the thing background subtraction was actually
    trying to suppress in the first place."""
    background = _blank_frame(fill=20)
    _draw_dot(background, 60, 60, radius=6, value=240)  # a fixed bright spot, no subject involved
    frame = background.copy()
    _draw_dot(frame, 150, 150, radius=6, value=250)  # a real, new highlight elsewhere

    result = detect_blobs(frame, threshold=200, background=background, background_mode="blacklist")

    assert len(result) == 1
    assert result[0].cx == pytest.approx(150.0, abs=1.0)


def test_detect_blobs_max_saturation_survives_a_saturated_backdrops_edge_bleed() -> None:
    """A genuinely white/near-neutral marker's own
    contour mask includes its anti-aliased boundary pixels, which blend
    with whatever is directly behind it -- on a highly saturated backdrop
    (e.g. a patterned fabric a person-worn marker is sewn onto) that alone
    can push the *mean* saturation over max_saturation even though the
    marker's own core color is neutral. Anti-aliased drawing (LINE_AA)
    reproduces the same blended-boundary effect a real marker's video
    compression does. A same-size, fully-saturated blob with no neutral
    core must still be rejected -- this isn't just raising the cutoff."""
    dim_backdrop = (0, 20, 90)  # dim and saturated, like a patterned legging in shadow
    bgr = np.full((200, 200, 3), dim_backdrop, dtype=np.uint8)
    white_bgr = (245, 245, 245)
    cv2.circle(bgr, (60, 60), 6, white_bgr, thickness=-1, lineType=cv2.LINE_AA)
    # A same-size, bright but fully-saturated orange highlight -- no neutral
    # core at all, unlike the marker above -- must still be rejected.
    bright_saturated = (0, 80, 250)
    cv2.circle(bgr, (140, 140), 6, bright_saturated, thickness=-1)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    filtered = detect_blobs(gray, threshold=100, bgr=bgr, max_saturation=45.0)

    assert len(filtered) == 1
    assert filtered[0].cx == pytest.approx(60.0, abs=1.0)
