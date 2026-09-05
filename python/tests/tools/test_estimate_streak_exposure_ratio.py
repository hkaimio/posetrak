# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for estimate_streak_exposure_ratio.py's pure accumulator logic
(_CamAccum). The rest of the script (DB loading, camera-state/sync-table
joins, redistortion matching) is exercised only by running it against real
data (streak-velocity-design.md §3), same "standalone, not yet a unit-
tested DB path" precedent as calibrate_rigid_marker_body.py's own tests.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "estimate_streak_exposure_ratio.py"
)
_spec = importlib.util.spec_from_file_location("estimate_streak_exposure_ratio", _MODULE_PATH)
estimate_streak_exposure_ratio = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = estimate_streak_exposure_ratio
_spec.loader.exec_module(estimate_streak_exposure_ratio)

_CamAccum = estimate_streak_exposure_ratio._CamAccum


def test_k_is_ratio_of_sums_not_mean_of_ratios():
    """The whole point of summing before dividing (streak-velocity-design.md
    §3): a single sample with a tiny, noise-dominated displacement must not
    be able to dominate the estimate the way it would under a mean of
    per-sample ratios."""
    acc = _CamAccum()
    acc.add(timestamp=0.0, streak_px=4.0, disp_px=20.0)   # per-sample ratio 0.20
    acc.add(timestamp=1.0, streak_px=0.5, disp_px=0.1)    # per-sample ratio 5.0 -- noisy outlier

    mean_of_ratios = (4.0 / 20.0 + 0.5 / 0.1) / 2.0
    k = acc.k()

    assert k == pytest.approx((4.0 + 0.5) / (20.0 + 0.1))
    assert k < mean_of_ratios / 2.0  # ratio-of-sums stays anchored near the well-conditioned sample


def test_k_is_none_with_no_samples():
    assert _CamAccum().k() is None


def test_k_matches_a_constant_underlying_ratio_regardless_of_sample_speed():
    """k = exposure_time/frame_time should come out the same whether the
    underlying motion was slow or fast (streak-velocity-design.md §1: v
    cancels out of the ratio) -- simulated here as several samples sharing
    one true k=0.3 at different speeds."""
    true_k = 0.3
    acc = _CamAccum()
    for i, disp in enumerate([5.0, 12.0, 40.0, 3.0]):
        acc.add(timestamp=float(i), streak_px=disp * true_k, disp_px=disp)

    assert acc.k() == pytest.approx(true_k)


def test_half_split_k_is_stable_when_the_true_ratio_is_constant():
    true_k = 0.25
    acc = _CamAccum()
    for i, disp in enumerate([4.0, 10.0, 6.0, 20.0, 8.0, 15.0]):
        acc.add(timestamp=float(i), streak_px=disp * true_k, disp_px=disp)

    k1, k2 = acc.half_split_k()
    assert k1 == pytest.approx(true_k)
    assert k2 == pytest.approx(true_k)


def test_half_split_k_orders_by_timestamp_not_insertion_order():
    true_k = 0.4
    acc = _CamAccum()
    # Inserted out of chronological order -- half_split_k must sort by
    # timestamp before splitting, not just bisect insertion order.
    for i, disp in [(2, 10.0), (0, 5.0), (1, 8.0), (3, 12.0)]:
        acc.add(timestamp=float(i), streak_px=disp * true_k, disp_px=disp)

    k1, k2 = acc.half_split_k()
    assert k1 == pytest.approx(true_k)
    assert k2 == pytest.approx(true_k)


def test_half_split_k_is_none_with_too_few_samples():
    acc = _CamAccum()
    acc.add(timestamp=0.0, streak_px=1.0, disp_px=4.0)
    assert acc.half_split_k() == (None, None)


def test_direction_cos_averages_across_samples():
    acc = _CamAccum()
    acc.add_direction_cos(1.0)
    acc.add_direction_cos(0.5)
    assert acc.dir_cos_sum / acc.n_dir_samples == pytest.approx(0.75)
