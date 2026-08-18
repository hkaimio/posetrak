# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for pipeline.pose.segmentation — CutieSegmentor and helpers.

These tests cover the pure-Python logic (score computation, encode/decode,
mask arithmetic) and do NOT require Cutie, YOLO, SAM, or a GPU.  The Cutie
integration is covered by the manual test scripts (cutie_rtmpose_test.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.pose.segmentation import (
    SCORE_INSIDE,
    SCORE_BOUNDARY,
    SCORE_OUTSIDE,
    SCORE_UNAVAILABLE,
    N_KEYPOINTS,
    encode_scores,
    decode_scores,
    _score_keypoints,
    CutieSegmentor,
)


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------

class TestEncodeDecodeScores:
    def test_roundtrip_all_values(self):
        original = np.array(
            [SCORE_INSIDE, SCORE_BOUNDARY, SCORE_OUTSIDE, SCORE_UNAVAILABLE],
            dtype=np.float32,
        )
        blob = encode_scores(original)
        decoded = decode_scores(blob, n=4)
        np.testing.assert_array_equal(original, decoded)

    def test_blob_length(self):
        arr = np.zeros(N_KEYPOINTS, dtype=np.float32)
        assert len(encode_scores(arr)) == N_KEYPOINTS * 4

    def test_little_endian(self):
        arr = np.array([1.0], dtype=np.float32)
        blob = encode_scores(arr)
        # 1.0 as little-endian IEEE 754 = 0x3F800000
        assert blob == b"\x00\x00\x80\x3f"

    def test_decode_is_writable_copy(self):
        blob = encode_scores(np.ones(4, dtype=np.float32))
        decoded = decode_scores(blob, n=4)
        decoded[0] = 99.0   # must not raise (frombuffer without copy raises)


# ---------------------------------------------------------------------------
# _score_keypoints
# ---------------------------------------------------------------------------

class TestScoreKeypoints:
    """Tests for the erosion-based scoring core."""

    def _solid_mask(self, h=100, w=100) -> np.ndarray:
        """All-True mask."""
        return np.ones((h, w), dtype=bool)

    def _ring_mask(self, h=100, w=100, outer=40, inner=30) -> np.ndarray:
        """Annular mask: ring between inner and outer radius from centre."""
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r2 = (y - cy) ** 2 + (x - cx) ** 2
        return (r2 <= outer**2) & (r2 > inner**2)

    def _circle_mask(self, h=100, w=100, r=30) -> np.ndarray:
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        return (y - cy) ** 2 + (x - cx) ** 2 <= r**2

    # -- inside / outside / boundary ---

    def test_centre_point_scores_inside(self):
        mask = self._circle_mask(r=30)
        kpts = np.array([[50.0, 50.0]])   # dead centre
        scores = _score_keypoints(mask, kpts, erosion_px=5)
        assert scores[0] == SCORE_INSIDE

    def test_outside_point_scores_zero(self):
        mask = self._circle_mask(r=20)
        kpts = np.array([[5.0, 5.0]])     # top-left corner, well outside
        scores = _score_keypoints(mask, kpts, erosion_px=5)
        assert scores[0] == SCORE_OUTSIDE

    def test_boundary_point_scores_half(self):
        # Build a mask with a very thin ring so the point is inside the
        # raw mask but outside the eroded mask.
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:55, 45:55] = True   # 10×10 block; erosion_px=6 will eat it
        kpts = np.array([[50.0, 50.0]])
        scores_with_erosion = _score_keypoints(mask, kpts, erosion_px=6)
        # After 6px erosion the 10×10 block is fully eroded → boundary
        assert scores_with_erosion[0] == SCORE_BOUNDARY

    def test_zero_erosion_boundary_becomes_inside(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:55, 45:55] = True
        kpts = np.array([[50.0, 50.0]])
        scores = _score_keypoints(mask, kpts, erosion_px=0)
        assert scores[0] == SCORE_INSIDE

    # -- out-of-bounds ---

    def test_oob_keypoint_scores_unavailable(self):
        mask = self._solid_mask()
        kpts = np.array([[-1.0, 50.0], [50.0, 200.0]])
        scores = _score_keypoints(mask, kpts, erosion_px=5)
        assert all(s == SCORE_UNAVAILABLE for s in scores)

    # -- float coordinates rounded ---

    def test_float_coords_rounded(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 50] = True   # single pixel inside
        kpts = np.array([[50.4, 49.6]])   # rounds to (50, 50)
        scores = _score_keypoints(mask, kpts, erosion_px=0)
        assert scores[0] == SCORE_INSIDE

    # -- empty mask ---

    def test_all_outside_on_empty_mask(self):
        mask = np.zeros((50, 50), dtype=bool)
        kpts = np.array([[10.0, 10.0], [20.0, 20.0]])
        scores = _score_keypoints(mask, kpts, erosion_px=5)
        assert all(s == SCORE_OUTSIDE for s in scores)

    # -- dtype tolerance ---

    def test_accepts_uint8_mask(self):
        mask = np.ones((50, 50), dtype=np.uint8)
        kpts = np.array([[25.0, 25.0]])
        scores = _score_keypoints(mask, kpts, erosion_px=0)
        assert scores[0] == SCORE_INSIDE


# ---------------------------------------------------------------------------
# CutieSegmentor — unit-testable methods (no GPU / Cutie required)
# ---------------------------------------------------------------------------

class TestCutieSegmentorQueryInterface:
    """Tests for get_mask / get_keypoint_scores without running process_video."""

    def _make_segmentor(self) -> CutieSegmentor:
        seg = CutieSegmentor.__new__(CutieSegmentor)
        seg._device = "cpu"
        seg._max_internal_size = 480
        seg._erosion_px = 5
        seg._masks = {}
        seg._person_ids = []
        return seg

    def _inject_mask(
        self,
        seg: CutieSegmentor,
        frame_idx: int,
        person_id: str,
        mask: np.ndarray,
    ) -> None:
        seg._masks.setdefault(frame_idx, {})[person_id] = mask

    # -- get_mask ---

    def test_get_mask_returns_none_for_missing_frame(self):
        seg = self._make_segmentor()
        assert seg.get_mask(99, "Harri") is None

    def test_get_mask_returns_none_for_missing_person(self):
        seg = self._make_segmentor()
        self._inject_mask(seg, 10, "Harri", np.ones((50, 50), bool))
        assert seg.get_mask(10, "Tommi") is None

    def test_get_mask_returns_array(self):
        seg = self._make_segmentor()
        m = np.zeros((50, 50), bool)
        m[10:20, 10:20] = True
        self._inject_mask(seg, 5, "Harri", m)
        result = seg.get_mask(5, "Harri")
        np.testing.assert_array_equal(result, m)

    # -- get_keypoint_scores ---

    def test_scores_unavailable_when_no_mask(self):
        seg = self._make_segmentor()
        kpts = np.zeros((10, 2), dtype=float)
        scores = seg.get_keypoint_scores(0, "Harri", kpts)
        assert (scores == SCORE_UNAVAILABLE).all()
        assert scores.dtype == np.float32

    def test_scores_correct_with_injected_mask(self):
        seg = self._make_segmentor()
        mask = np.zeros((100, 100), bool)
        mask[20:80, 20:80] = True
        self._inject_mask(seg, 42, "Harri", mask)

        kpts = np.array([
            [50.0, 50.0],   # deep inside → INSIDE
            [1.0,  1.0],    # outside → OUTSIDE
        ])
        scores = seg.get_keypoint_scores(42, "Harri", kpts, erosion_px=5)
        assert scores[0] == SCORE_INSIDE
        assert scores[1] == SCORE_OUTSIDE

    def test_get_all_scores_for_frame(self):
        seg = self._make_segmentor()
        m = np.zeros((100, 100), bool)
        m[10:90, 10:90] = True
        self._inject_mask(seg, 0, "p0", m)
        self._inject_mask(seg, 0, "p1", m)

        kpts = {"p0": np.array([[50.0, 50.0]]), "p1": np.array([[50.0, 50.0]])}
        result = seg.get_all_scores_for_frame(0, kpts)
        assert set(result.keys()) == {"p0", "p1"}
        assert result["p0"][0] == SCORE_INSIDE

    # -- erosion_px override ---

    def test_erosion_px_override(self):
        """Passing erosion_px=0 should suppress the boundary zone."""
        seg = self._make_segmentor()   # default erosion_px=5
        mask = np.zeros((20, 20), bool)
        mask[8:12, 8:12] = True   # 4×4 block — fully eroded by 5px
        self._inject_mask(seg, 0, "p0", mask)

        kpts = np.array([[10.0, 10.0]])
        scores_default = seg.get_keypoint_scores(0, "p0", kpts)
        scores_no_erosion = seg.get_keypoint_scores(0, "p0", kpts, erosion_px=0)

        assert scores_default[0] == SCORE_BOUNDARY  # eroded away
        assert scores_no_erosion[0] == SCORE_INSIDE   # no erosion
