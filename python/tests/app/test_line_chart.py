# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for app.ui.content_panels._LineChart's log-scale Y-axis mapping.

Covers _y_space() -- the linear-vs-log10 value transform used both for the
plotted data and the reference line. Actual QPainter drawing follows the
project's usual manual-validation convention and is not covered here.
"""
from __future__ import annotations

import math

import pytest


@pytest.fixture()
def chart_linear(qapp):
    from app.ui.content_panels import _LineChart
    return _LineChart("test")


@pytest.fixture()
def chart_log(qapp):
    from app.ui.content_panels import _LineChart
    return _LineChart("test", log_y=True)


def test_y_space_linear_is_identity(chart_linear) -> None:
    assert chart_linear._y_space(5.0) == 5.0
    assert chart_linear._y_space(-3.0) == -3.0
    assert chart_linear._y_space(0.0) == 0.0


def test_y_space_log_transforms_positive_values(chart_log) -> None:
    assert chart_log._y_space(1_000_000.0) == pytest.approx(6.0)
    assert chart_log._y_space(1.0) == pytest.approx(0.0)


def test_y_space_log_rejects_non_positive_values(chart_log) -> None:
    assert chart_log._y_space(0.0) is None
    assert chart_log._y_space(-5.0) is None
