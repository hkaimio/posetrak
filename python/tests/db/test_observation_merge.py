"""Tests for posetrak.db.observation_merge."""

from __future__ import annotations

import numpy as np

from posetrak.db.observation_merge import merge_observation_sources

N_KP = 133


def _kp(fill: float, n: int = N_KP) -> np.ndarray:
    kp = np.zeros((n, 3), dtype=np.float32)
    kp[:] = [fill, fill, fill]
    return kp


def test_body_only_returns_body_unchanged():
    body = _kp(1.0)
    merged = merge_observation_sources([("body", body)])
    np.testing.assert_array_equal(merged, body)
    assert merged is not body  # must not alias the caller's array


def test_hand_l_overwrites_only_its_index_range():
    body = _kp(1.0)
    hand = _kp(9.0, n=21)
    merged = merge_observation_sources([("body", body), ("hand_l", hand)])

    np.testing.assert_array_equal(merged[91:112], hand)
    # Everything outside the hand_l range is untouched.
    np.testing.assert_array_equal(merged[:91], body[:91])
    np.testing.assert_array_equal(merged[112:], body[112:])


def test_hand_r_overwrites_only_its_index_range():
    body = _kp(1.0)
    hand = _kp(9.0, n=21)
    merged = merge_observation_sources([("body", body), ("hand_r", hand)])

    np.testing.assert_array_equal(merged[112:133], hand)
    np.testing.assert_array_equal(merged[:112], body[:112])


def test_both_hands_and_body_merge_independently():
    body = _kp(1.0)
    hand_l = _kp(2.0, n=21)
    hand_r = _kp(3.0, n=21)
    merged = merge_observation_sources(
        [("hand_r", hand_r), ("body", body), ("hand_l", hand_l)]
    )

    np.testing.assert_array_equal(merged[91:112], hand_l)
    np.testing.assert_array_equal(merged[112:133], hand_r)
    np.testing.assert_array_equal(merged[:91], body[:91])


def test_no_body_row_returns_none():
    hand = _kp(9.0, n=21)
    assert merge_observation_sources([("hand_l", hand)]) is None


def test_unknown_source_is_ignored():
    body = _kp(1.0)
    weird = _kp(9.0, n=21)
    merged = merge_observation_sources([("body", body), ("face", weird)])
    np.testing.assert_array_equal(merged, body)


def test_empty_rows_returns_none():
    assert merge_observation_sources([]) is None
