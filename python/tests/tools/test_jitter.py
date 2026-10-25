# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""jitter: the noise metric returns a known noise, and episodes start and end where the series crosses the threshold."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "gt"))
import jitter  # noqa: E402


def test_the_residual_returns_the_noise_added_to_a_smooth_motion():
    r = jitter.synthetic_check(sigma=0.004)
    assert abs(r["measured"] / r["noise_3d_rms"] - 1) < 0.05
    assert r["smooth_residual"] < 0.1 * r["noise_3d_rms"]
    assert abs(r["jerk_measured"] / r["jerk_expected"] - 1) < 0.1


def test_the_series_of_a_burst_of_noise_crosses_the_threshold_where_the_burst_is():
    fs = jitter.FS
    t = np.arange(0, 20, 1 / fs)
    rng = np.random.default_rng(1)
    x = np.column_stack([np.sin(t), np.cos(t), 0 * t])
    x[600:660] += rng.normal(0, 0.03, (60, 3))
    r = jitter.residual(x)
    series = jitter.sliding_rms(np.sum(r ** 2, axis=1))
    (a, b), = jitter.find_episodes(t, series, 0.02)
    assert 600 - 30 <= a <= 600 and 660 <= b <= 660 + 30
