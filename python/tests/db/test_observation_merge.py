"""Tests for posetrak.db.observation_merge."""

from __future__ import annotations

import numpy as np

from posetrak.db.observation_merge import (
    infer_body_width,
    merge_observation_sources,
    refined_indices,
)

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


def test_refined_overrides_its_base_source():
    body = _kp(1.0)
    hand_l = _kp(2.0, n=21)
    hand_l_refined = _kp(5.0, n=21)
    merged = merge_observation_sources(
        [("body", body), ("hand_l", hand_l), ("hand_l.refined", hand_l_refined)]
    )
    np.testing.assert_array_equal(merged[91:112], hand_l_refined)
    np.testing.assert_array_equal(merged[:91], body[:91])


def test_refined_wins_regardless_of_row_order():
    """The DB has no ORDER BY on source, so the merge must not rely on
    '.refined' rows happening to arrive after their base row."""
    body = _kp(1.0)
    hand_l = _kp(2.0, n=21)
    hand_l_refined = _kp(5.0, n=21)
    merged = merge_observation_sources(
        [("hand_l.refined", hand_l_refined), ("body", body), ("hand_l", hand_l)]
    )
    np.testing.assert_array_equal(merged[91:112], hand_l_refined)


def test_refined_alone_without_base_still_applies():
    body = _kp(1.0)
    hand_r_refined = _kp(7.0, n=21)
    merged = merge_observation_sources([("body", body), ("hand_r.refined", hand_r_refined)])
    np.testing.assert_array_equal(merged[112:133], hand_r_refined)


def test_refined_on_one_side_does_not_affect_the_other():
    body = _kp(1.0)
    hand_l = _kp(2.0, n=21)
    hand_r = _kp(3.0, n=21)
    hand_l_refined = _kp(9.0, n=21)
    merged = merge_observation_sources(
        [("body", body), ("hand_l", hand_l), ("hand_r", hand_r), ("hand_l.refined", hand_l_refined)]
    )
    np.testing.assert_array_equal(merged[91:112], hand_l_refined)
    np.testing.assert_array_equal(merged[112:133], hand_r)


def test_refined_indices_empty_when_no_refined_rows():
    body = _kp(1.0)
    hand_l = _kp(2.0, n=21)
    assert refined_indices([("body", body), ("hand_l", hand_l)]) == frozenset()


def test_refined_indices_reports_the_refined_sides_range():
    body = _kp(1.0)
    hand_l_refined = _kp(5.0, n=21)
    assert refined_indices([("body", body), ("hand_l.refined", hand_l_refined)]) == frozenset(
        range(91, 112)
    )


def test_refined_indices_ignores_unrecognised_source():
    body = _kp(1.0)
    weird = _kp(9.0, n=21)
    assert refined_indices([("body", body), ("face.refined", weird)]) == frozenset()


def test_no_body_row_with_default_width_synthesizes_zero_body():
    """A ghost frame's auto-redetected hand (no 'body' row of its own) must
    still merge to the camera's full width, not the bare 21-kp overlay --
    otherwise downstream code that assumes one width per camera breaks."""
    hand_l = _kp(9.0, n=21)
    merged = merge_observation_sources([("hand_l", hand_l)], default_width=N_KP)
    assert merged.shape == (N_KP, 3)
    np.testing.assert_array_equal(merged[91:112], hand_l)
    np.testing.assert_array_equal(merged[:91], np.zeros((91, 3), dtype=np.float32))
    np.testing.assert_array_equal(merged[112:], np.zeros((21, 3), dtype=np.float32))


def test_no_body_row_without_default_width_still_returns_none():
    hand = _kp(9.0, n=21)
    assert merge_observation_sources([("hand_l", hand)], default_width=None) is None


def test_infer_body_width_finds_body_row_in_any_frame():
    body = _kp(1.0)
    hand = _kp(9.0, n=21)
    rows_by_frame = [
        [("hand_l", hand)],  # ghost frame: no body row here
        [("body", body), ("hand_l", hand)],
    ]
    assert infer_body_width(rows_by_frame) == N_KP


def test_infer_body_width_returns_none_when_no_frame_has_a_body_row():
    hand = _kp(9.0, n=21)
    assert infer_body_width([[("hand_l", hand)]]) is None
