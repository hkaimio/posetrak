# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for mask_treatment.py's suppress_others() -- the "feather2"
treatment validated by python/tools/segmentation_mask_steering_experiment.py
(see docs/roadmap/features/segmentation-pose-treatment/) before being
wired into pose_worker.py's real segmentation-driven pose extraction.
"""
from __future__ import annotations

import numpy as np

from posetrak.detection.mask_treatment import (
    CONTRAST_FACTOR,
    FEATHER_PX,
    FILL_GRAY,
    suppress_others,
)


def _solid_frame(value: int, h: int = 80, w: int = 80) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_pixels_deep_inside_target_are_unchanged():
    # A vertical split: label 1 on the left half, label 2 on the right --
    # deep interior pixels, far from the boundary, should pass through
    # untouched for their own label.
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[:, :40] = 1
    mask[:, 40:] = 2
    frame = _solid_frame(200)
    frame[:, 40:] = 50  # make the two halves visually distinct

    treated = suppress_others(frame, mask, target_label=1)
    # Deep inside label 1 (far left, away from the boundary at x=40)
    np.testing.assert_array_equal(treated[40, 5], frame[40, 5])


def test_pixels_far_outside_target_are_fully_treated():
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[:, :40] = 1
    frame = _solid_frame(10)
    frame[:, :40] = 200  # label-1 region is bright; background is dark

    treated = suppress_others(frame, mask, target_label=1)
    # Far outside the mask (label-1 region ends at x=40; this is well
    # past FEATHER_PX=10 pixels away) -- alpha should be ~1, so the
    # result should be close to the flat gray/blur blend, not the
    # original dark background value of 10.
    far_pixel = treated[40, 79]
    assert far_pixel[0] != 10
    # A solid-color background blurs to itself, so the only change here
    # is the contrast reduction toward FILL_GRAY.
    expected = round(10 * CONTRAST_FACTOR + FILL_GRAY * (1 - CONTRAST_FACTOR))
    assert abs(int(far_pixel[0]) - expected) <= 2


def test_alpha_ramps_smoothly_across_the_feather_band():
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[:, :40] = 1
    frame = _solid_frame(10)
    frame[:, :40] = 200

    treated = suppress_others(frame, mask, target_label=1)
    row = 40
    # Distance from the boundary (x=40) increases moving right; the
    # treated value should move gradually from "near original" (10)
    # toward the fully-treated blend as we cross the feather band --
    # blur and alpha compound non-linearly, so exact monotonicity isn't
    # guaranteed pixel-to-pixel, but no single step should jump most of
    # the way there at once (that would be a hard cutoff, the opposite
    # of what this treatment is for).
    values = [int(treated[row, 40 + dx][0]) for dx in range(0, FEATHER_PX + 5)]
    total_span = abs(values[-1] - values[0])
    max_step = max(abs(b - a) for a, b in zip(values, values[1:]))
    assert total_span > 20, "treatment should meaningfully change the pixel across the band"
    assert max_step < 0.6 * total_span, "no single step should be a hard cutoff"


def test_other_person_is_treated_same_as_background():
    """The whole point: another tracked person, not just empty background,
    gets suppressed too."""
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[:, :40] = 1
    mask[:, 40:] = 2  # a second person, not background
    frame = _solid_frame(10)
    frame[:, :40] = 200
    frame[:, 40:] = 200  # person 2 looks just as bright as person 1

    treated = suppress_others(frame, mask, target_label=1)
    # Far into person 2's region, suppression should still have fully
    # kicked in despite those pixels belonging to a real tracked person,
    # not background.
    far_pixel = treated[40, 79]
    assert far_pixel[0] != 200
