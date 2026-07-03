"""Tests for the wide-crop cluster-cache algorithms (geometry, clustering,
gap-handling) in app.pose.wide_crop_cache.

Covers the pure algorithmic pieces described in "Background wide-crop frame
cache" (keypoint-editing-design.md): padding, overlap clustering with the
merge-area guard, and the per-track raw-rect gap search that extends past
the epoch boundary for long undetected stretches. Worker-thread mechanics
(sequential decode, priority queue, QThread lifecycle) follow the same
manual-validation convention already used for CropBackfillWorker and are not
covered here.
"""
from __future__ import annotations

import numpy as np

from app.pose.wide_crop_cache import (
    _encode_rect,
    _pad_rect,
    _rect_area,
    _rects_overlap,
    _TrackWindow,
    _union_rect,
    cluster_rects,
)
from app.ui.content_panels import (
    _expand_rect_to_aspect,
    _kp_overlay_bbox,
    _nearest_segment_track_id,
    _tracked_overlay_bbox,
)


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def test_pad_rect_expands_by_fraction_of_each_dimension():
    # 100x50 rect padded 20% -> +/-20 in x, +/-10 in y
    padded = _pad_rect(0, 0, 100, 50, 0.20)
    assert padded == (-20, -10, 120, 60)


def test_rects_overlap_true_for_intersecting_rects():
    assert _rects_overlap((0, 0, 10, 10), (5, 5, 15, 15))


def test_rects_overlap_false_for_disjoint_rects():
    assert not _rects_overlap((0, 0, 10, 10), (20, 20, 30, 30))


def test_rects_overlap_false_for_merely_touching_rects():
    # Sharing only an edge (no area of intersection) should not count as overlap.
    assert not _rects_overlap((0, 0, 10, 10), (10, 0, 20, 10))


def test_union_rect_and_area():
    u = _union_rect((0, 0, 10, 10), (5, 5, 20, 20))
    assert u == (0, 0, 20, 20)
    assert _rect_area(u) == 400


# ---------------------------------------------------------------------------
# Clustering with the merge-area guard
# ---------------------------------------------------------------------------

def test_cluster_rects_merges_overlapping_pair():
    rects = {1: (0, 0, 10, 10), 2: (5, 5, 15, 15)}
    clusters = cluster_rects(rects, guard=1.3)
    assert len(clusters) == 1
    ids, rect = clusters[0]
    assert set(ids) == {1, 2}
    assert rect == (0, 0, 15, 15)


def test_cluster_rects_keeps_disjoint_rects_separate():
    rects = {1: (0, 0, 10, 10), 2: (100, 100, 110, 110)}
    clusters = cluster_rects(rects, guard=1.3)
    assert len(clusters) == 2
    assert {frozenset(ids) for ids, _ in clusters} == {frozenset({1}), frozenset({2})}


def test_cluster_rects_transitively_merges_a_chain():
    # 1 overlaps 2, 2 overlaps 3, but 1 and 3 don't directly touch.
    rects = {1: (0, 0, 10, 10), 2: (8, 0, 18, 10), 3: (16, 0, 26, 10)}
    clusters = cluster_rects(rects, guard=10.0)  # generous guard: focus on transitivity
    assert len(clusters) == 1
    ids, rect = clusters[0]
    assert set(ids) == {1, 2, 3}
    assert rect == (0, 0, 26, 10)


def test_cluster_rects_merge_guard_rejects_corner_graze():
    # Two large rects overlapping only in a tiny corner: the union is much
    # bigger than the sum of the individual areas, so the guard should keep
    # them separate rather than caching one mostly-empty shared crop.
    a = (0, 0, 100, 100)      # area 10,000
    b = (99, 99, 199, 199)    # area 10,000, overlaps a in a 1x1 corner
    clusters = cluster_rects({1: a, 2: b}, guard=1.3)
    assert len(clusters) == 2


def test_cluster_rects_merge_guard_allows_mostly_overlapping_pair():
    a = (0, 0, 100, 100)
    b = (0, 0, 100, 110)  # nearly identical, union only slightly bigger
    clusters = cluster_rects({1: a, 2: b}, guard=1.3)
    assert len(clusters) == 1


# ---------------------------------------------------------------------------
# _TrackWindow: per-epoch raw rect with gap handling
# ---------------------------------------------------------------------------

def _bbox(cx, cy, w, h):
    return (cx, cy, w, h)


def test_raw_rect_unions_detections_strictly_inside_epoch():
    det = {10: _bbox(100, 100, 20, 20), 11: _bbox(110, 100, 20, 20)}
    win = _TrackWindow(det)
    rect = win.raw_rect(epoch_start=0, epoch_end=20, margin=0, gap_radius=5)
    # union of (90,90,110,110) and (100,90,120,110) -> (90,90,120,110)
    assert rect == (90, 90, 120, 110)


def test_raw_rect_reaches_past_epoch_boundary_via_margin():
    # Detection sits just outside the epoch, but within the +/-margin widening.
    det = {24: _bbox(100, 100, 20, 20)}
    win = _TrackWindow(det)
    rect = win.raw_rect(epoch_start=0, epoch_end=20, margin=10, gap_radius=5)
    assert rect is not None


def test_raw_rect_none_when_nothing_in_widened_window_or_gap_radius():
    det = {1000: _bbox(100, 100, 20, 20)}
    win = _TrackWindow(det)
    rect = win.raw_rect(epoch_start=0, epoch_end=20, margin=5, gap_radius=5)
    assert rect is None


def test_raw_rect_gap_search_unions_anchors_on_both_sides():
    # No detections anywhere near [0, 20), but real detections just outside
    # the gap-search radius boundary on each side -- simulates a person fully
    # undetected for longer than one epoch (e.g. occluded during a throw).
    det = {-5: _bbox(0, 0, 10, 10), 25: _bbox(100, 100, 10, 10)}
    win = _TrackWindow(det)
    rect = win.raw_rect(epoch_start=0, epoch_end=20, margin=0, gap_radius=10)
    # before=-5 (epoch_start - before = 5 <= 10), after=25 (after - 19 = 6 <= 10)
    assert rect == (-5, -5, 105, 105)


def test_raw_rect_gap_search_uses_only_the_in_range_anchor():
    # "before" anchor is too far away (exceeds gap_radius); only "after" counts.
    det = {-1000: _bbox(0, 0, 10, 10), 25: _bbox(100, 100, 10, 10)}
    win = _TrackWindow(det)
    rect = win.raw_rect(epoch_start=0, epoch_end=20, margin=0, gap_radius=10)
    assert rect == (95, 95, 105, 105)


def test_raw_rect_gap_search_returns_none_when_both_anchors_out_of_range():
    det = {-1000: _bbox(0, 0, 10, 10), 1000: _bbox(100, 100, 10, 10)}
    win = _TrackWindow(det)
    rect = win.raw_rect(epoch_start=0, epoch_end=20, margin=0, gap_radius=10)
    assert rect is None


# ---------------------------------------------------------------------------
# _encode_rect
# ---------------------------------------------------------------------------

def test_encode_rect_crops_and_reports_src_geometry():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    result = _encode_rect(img, (10, 20, 110, 120))
    assert result is not None
    jpeg, wpx, hpx, src_x, src_y, src_w, src_h = result
    assert isinstance(jpeg, bytes) and len(jpeg) > 0
    assert (src_x, src_y, src_w, src_h) == (10, 20, 100, 100)
    assert wpx == 100 and hpx == 100


def test_encode_rect_clips_to_image_bounds():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    result = _encode_rect(img, (-20, -20, 30, 30))
    assert result is not None
    _, wpx, hpx, src_x, src_y, src_w, src_h = result
    assert (src_x, src_y) == (0, 0)
    assert (src_w, src_h) == (30, 30)


def test_encode_rect_none_when_rect_outside_image():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    assert _encode_rect(img, (100, 100, 200, 200)) is None


def test_encode_rect_downscales_past_max_long_edge():
    import app.pose.wide_crop_cache as wc

    img = np.zeros((2000, 2000, 3), dtype=np.uint8)
    result = _encode_rect(img, (0, 0, 2000, 2000))
    assert result is not None
    _, wpx, hpx, _src_x, _src_y, src_w, src_h = result
    assert (src_w, src_h) == (2000, 2000)
    assert max(wpx, hpx) == wc.MAX_LONG_EDGE


# ---------------------------------------------------------------------------
# _kp_overlay_bbox -- what the wide-crop display sub-crop must cover
# ---------------------------------------------------------------------------

def _kp_array(*points):
    """points: list of (x, y, conf)."""
    return np.array(points, dtype=np.float32)


def test_kp_overlay_bbox_none_for_no_keypoints():
    assert _kp_overlay_bbox(None, frozenset()) is None


def test_kp_overlay_bbox_ignores_low_confidence_points():
    # Matches paintEvent's conf < 0.1 cutoff -- these aren't really drawn.
    kp = _kp_array((10, 10, 0.05), (500, 500, 0.09))
    assert _kp_overlay_bbox(kp, frozenset()) is None


def test_kp_overlay_bbox_ignores_hidden_indices():
    kp = _kp_array((10, 10, 0.9), (500, 500, 0.9))
    assert _kp_overlay_bbox(kp, frozenset({1})) == (10, 10, 10, 10)


def test_kp_overlay_bbox_covers_all_visible_points():
    kp = _kp_array((50, 100, 0.9), (10, 400, 0.5), (200, 20, 0.2))
    assert _kp_overlay_bbox(kp, frozenset()) == (10, 20, 200, 400)


# ---------------------------------------------------------------------------
# _tracked_overlay_bbox -- tracked-skeleton coverage for the same sub-crop
# ---------------------------------------------------------------------------

def test_tracked_overlay_bbox_none_when_nothing_available():
    assert _tracked_overlay_bbox(None, None) is None


def test_tracked_overlay_bbox_covers_joints_and_markers():
    joint_xy = {"hip": [50.0, 60.0], "knee": [10.0, 400.0]}
    marker_xy = np.array([[200.0, 20.0], [30.0, 30.0]], dtype=np.float32)
    assert _tracked_overlay_bbox(joint_xy, marker_xy) == (10.0, 20.0, 200.0, 400.0)


def test_tracked_overlay_bbox_ignores_nan_joints():
    joint_xy = {"hip": [50.0, 60.0], "behind_camera": [float("nan"), float("nan")]}
    assert _tracked_overlay_bbox(joint_xy, None) == (50.0, 60.0, 50.0, 60.0)


def test_tracked_overlay_bbox_ignores_nan_markers():
    marker_xy = np.array([[float("nan"), float("nan")], [30.0, 40.0]], dtype=np.float32)
    assert _tracked_overlay_bbox(None, marker_xy) == (30.0, 40.0, 30.0, 40.0)


# ---------------------------------------------------------------------------
# _expand_rect_to_aspect -- fill the cell instead of letterboxing
# ---------------------------------------------------------------------------

def test_expand_rect_to_aspect_grows_width_when_too_narrow():
    # 100x200 rect (ar=0.5) expanded to a 1:1 cell -> grow width to 200x200.
    rect = _expand_rect_to_aspect((0, 0, 100, 200), target_ar=1.0)
    assert rect == (-50, 0, 150, 200)


def test_expand_rect_to_aspect_grows_height_when_too_short():
    # 200x100 rect (ar=2.0) expanded to a 1:1 cell -> grow height to 200x200.
    rect = _expand_rect_to_aspect((0, 0, 200, 100), target_ar=1.0)
    assert rect == (0, -50, 200, 150)


def test_expand_rect_to_aspect_noop_when_already_matching():
    rect = _expand_rect_to_aspect((0, 0, 160, 90), target_ar=16 / 9)
    assert rect == (0, 0, 160, 90)


# ---------------------------------------------------------------------------
# _nearest_segment_track_id -- resolving a wide-crop lookup across a true
# gap between two of a person's own assigned track segments
# ---------------------------------------------------------------------------

def test_nearest_segment_track_id_returns_covering_segment_directly():
    segs = [(1, 0, 99), (2, 200, 299)]
    assert _nearest_segment_track_id(segs, 50) == 1
    assert _nearest_segment_track_id(segs, 250) == 2


def test_nearest_segment_track_id_picks_closer_side_of_a_gap():
    segs = [(1, 0, 99), (2, 200, 299)]
    assert _nearest_segment_track_id(segs, 110) == 1   # 11 frames from seg 1's end
    assert _nearest_segment_track_id(segs, 190) == 2   # 10 frames from seg 2's start
    assert _nearest_segment_track_id(segs, 149) == 1   # 50 vs 51 frames -- seg 1 closer


def test_nearest_segment_track_id_none_when_no_segments():
    assert _nearest_segment_track_id([], 50) is None
