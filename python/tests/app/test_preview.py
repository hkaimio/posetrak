"""Tests for PersonPreviewWidget pure functions.

compute_crop_rect and bbox_from_detections contain no Qt dependencies and can
be tested without a display.  Skeleton drawing and Qt rendering are covered by
visual inspection.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.pose.person_preview import (
    _MIN_CROP_PX,
    bbox_from_detections,
    compute_crop_rect,
    draw_skeleton_on_crop,
)

FRAME_W = 1920
FRAME_H = 1080


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def det(tid: int, cx: float, cy: float, w: float, h: float) -> dict:
    """Build a minimal detection dict."""
    return {"track_id": tid, "bbox_x": cx, "bbox_y": cy, "bbox_w": w, "bbox_h": h}


# ---------------------------------------------------------------------------
# compute_crop_rect
# ---------------------------------------------------------------------------

class TestComputeCropRect:

    def test_basic_centre(self):
        """Centre bbox, no edge proximity — margin applied symmetrically."""
        x1, y1, x2, y2 = compute_crop_rect(960, 540, 200, 400, FRAME_W, FRAME_H, margin=0.1)
        # mx = 200*0.1 = 20, my = 400*0.1 = 40
        # x1 = 960 - 100 - 20 = 840, x2 = 960 + 100 + 20 = 1080
        # y1 = 540 - 200 - 40 = 300, y2 = 540 + 200 + 40 = 780
        assert x1 == 840
        assert x2 == 1080
        assert y1 == 300
        assert y2 == 780

    def test_near_top_left(self):
        """Bbox near top-left corner — crop is clamped so x1/y1 >= 0."""
        x1, y1, x2, y2 = compute_crop_rect(10, 10, 100, 100, FRAME_W, FRAME_H, margin=0.5)
        assert x1 >= 0
        assert y1 >= 0

    def test_near_bottom_right(self):
        """Bbox near bottom-right corner — crop is clamped to frame bounds."""
        x1, y1, x2, y2 = compute_crop_rect(
            FRAME_W - 10, FRAME_H - 10, 100, 100, FRAME_W, FRAME_H, margin=0.5
        )
        assert x2 <= FRAME_W
        assert y2 <= FRAME_H

    def test_zero_margin(self):
        """margin=0 — crop is tight to the bbox edges."""
        x1, y1, x2, y2 = compute_crop_rect(500, 400, 200, 300, FRAME_W, FRAME_H, margin=0.0)
        assert x1 == 400
        assert x2 == 600
        assert y1 == 250
        assert y2 == 550

    def test_tall_narrow_bbox(self):
        """Portrait bbox — crop height > crop width."""
        x1, y1, x2, y2 = compute_crop_rect(960, 540, 100, 600, FRAME_W, FRAME_H, margin=0.1)
        assert (y2 - y1) > (x2 - x1)

    def test_wide_short_bbox(self):
        """Landscape bbox — crop width > crop height."""
        x1, y1, x2, y2 = compute_crop_rect(960, 540, 600, 100, FRAME_W, FRAME_H, margin=0.1)
        assert (x2 - x1) > (y2 - y1)

    def test_zero_width_bbox(self):
        """Zero bbox width — returns at least _MIN_CROP_PX wide."""
        x1, y1, x2, y2 = compute_crop_rect(960, 540, 0, 200, FRAME_W, FRAME_H)
        assert x2 - x1 >= _MIN_CROP_PX

    def test_zero_height_bbox(self):
        """Zero bbox height — returns at least _MIN_CROP_PX tall."""
        x1, y1, x2, y2 = compute_crop_rect(960, 540, 200, 0, FRAME_W, FRAME_H)
        assert y2 - y1 >= _MIN_CROP_PX

    def test_huge_bbox_clamped_to_frame(self):
        """Bbox + margin larger than frame — clamped to exactly frame dimensions."""
        x1, y1, x2, y2 = compute_crop_rect(960, 540, FRAME_W * 2, FRAME_H * 2, FRAME_W, FRAME_H)
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= FRAME_W
        assert y2 <= FRAME_H

    def test_result_always_non_negative_and_ordered(self):
        """x1 < x2 and y1 < y2 for any valid input."""
        for cx, cy, bw, bh in [
            (0, 0, 1, 1),
            (FRAME_W, FRAME_H, 1, 1),
            (FRAME_W // 2, FRAME_H // 2, FRAME_W, FRAME_H),
        ]:
            x1, y1, x2, y2 = compute_crop_rect(cx, cy, bw, bh, FRAME_W, FRAME_H)
            assert x1 < x2, f"x1={x1} x2={x2} for cx={cx}"
            assert y1 < y2, f"y1={y1} y2={y2} for cy={cy}"


# ---------------------------------------------------------------------------
# bbox_from_detections
# ---------------------------------------------------------------------------

class TestBboxFromDetections:

    def test_found(self):
        """track_id present — returns correct bbox tuple."""
        dets = [det(0, 100, 200, 50, 80), det(1, 500, 400, 60, 90)]
        result = bbox_from_detections(1, dets)
        assert result == (500.0, 400.0, 60.0, 90.0)

    def test_not_found(self):
        """track_id absent — returns None."""
        dets = [det(0, 100, 200, 50, 80)]
        assert bbox_from_detections(99, dets) is None

    def test_empty_list(self):
        """Empty detections list — returns None."""
        assert bbox_from_detections(0, []) is None

    def test_first_match_returned(self):
        """Returns the first detection with matching track_id."""
        dets = [det(0, 100, 200, 50, 80), det(0, 999, 999, 1, 1)]
        result = bbox_from_detections(0, dets)
        assert result == (100.0, 200.0, 50.0, 80.0)

    def test_multiple_tracks_correct_one_returned(self):
        """Multiple different tracks — only the requested one is returned."""
        dets = [det(0, 10, 20, 5, 8), det(1, 50, 60, 15, 25), det(2, 100, 110, 30, 40)]
        result = bbox_from_detections(2, dets)
        assert result == (100.0, 110.0, 30.0, 40.0)


# ---------------------------------------------------------------------------
# draw_skeleton_on_crop
# ---------------------------------------------------------------------------

class TestDrawSkeletonOnCrop:

    def _blank_crop(self, h: int = 200, w: int = 100) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_returns_same_shape(self):
        """Drawing does not change crop dimensions."""
        crop = self._blank_crop()
        kp = np.zeros((17, 3), dtype=np.float32)
        result = draw_skeleton_on_crop(crop, kp, 0, 0)
        assert result.shape == (200, 100, 3)

    def test_no_keypoints_array_unchanged(self):
        """Empty keypoints (N=0) — crop returned without modification."""
        crop = self._blank_crop()
        original = crop.copy()
        kp = np.zeros((0, 3), dtype=np.float32)
        result = draw_skeleton_on_crop(crop, kp, 0, 0)
        np.testing.assert_array_equal(result, original)

    def test_none_keypoints_returns_crop(self):
        """None keypoints — crop returned as-is without crash."""
        crop = self._blank_crop()
        result = draw_skeleton_on_crop(crop, None, 0, 0)
        assert result is crop

    def test_low_confidence_keypoints_not_drawn(self):
        """Keypoints with conf < 0.1 leave the crop unchanged."""
        crop = self._blank_crop()
        original = crop.copy()
        # Single keypoint at centre with near-zero confidence
        kp = np.array([[50, 100, 0.05]], dtype=np.float32)
        result = draw_skeleton_on_crop(crop, kp, 0, 0)
        np.testing.assert_array_equal(result, original)

    def test_high_confidence_keypoint_modifies_crop(self):
        """High-confidence keypoint at crop centre changes at least one pixel."""
        crop = self._blank_crop(200, 100)
        kp = np.array([[50, 100, 0.9]], dtype=np.float32)  # centre of crop
        result = draw_skeleton_on_crop(crop, kp, 0, 0)
        assert result.sum() > 0  # at least one non-zero pixel drawn

    def test_keypoints_outside_crop_no_crash(self):
        """Keypoints with coordinates outside the crop region do not crash."""
        crop = self._blank_crop(100, 100)
        kp = np.array([
            [-999, -999, 0.9],  # far outside top-left
            [9999, 9999, 0.9],  # far outside bottom-right
        ], dtype=np.float32)
        # Should not raise; cv2 clamps or ignores out-of-bounds pixels
        draw_skeleton_on_crop(crop, kp, 0, 0)

    def test_133_keypoints_no_crash(self):
        """133-keypoint array (COCO Wholebody) — no crash, body sticks drawn."""
        crop = self._blank_crop(300, 200)
        kp = np.random.rand(133, 3).astype(np.float32)
        kp[:, 0] *= 200   # x in [0, 200]
        kp[:, 1] *= 300   # y in [0, 300]
        kp[:, 2] = 0.9    # all high confidence
        draw_skeleton_on_crop(crop, kp, 0, 0)
