# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

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
    _composite_black_fill,
    _expand_rect_to_aspect,
    _kp_overlay_bbox,
    _MAX_CANVAS_DIM_PX,
    _nearest_segment_track_id,
    _sane_bbox,
    _tracked_overlay_bbox,
    _windowed_kp_bbox,
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
# _windowed_kp_bbox -- gap-aware +/-N real-observation-frame window
# ---------------------------------------------------------------------------

def test_windowed_kp_bbox_none_when_no_observations():
    assert _windowed_kp_bbox({}, 100, frozenset()) is None


def test_windowed_kp_bbox_skips_over_a_long_gap():
    # Real observations only at frames 60-74 and 451-465; nothing 75-450.
    # Target frame 100 (itself has no data) must reach back to 65-74 and
    # forward to 451-460 -- the nearest 10 real frames on each side -- not
    # just a fixed +/-10 index window around 100.
    obs = {}
    for f in range(60, 75):
        obs[f] = _kp_array((float(f), 0.0, 0.9))
    for f in range(451, 466):
        obs[f] = _kp_array((float(f), 0.0, 0.9))
    bbox = _windowed_kp_bbox(obs, 100, frozenset(), n_frames=10)
    assert bbox == (65.0, 0.0, 460.0, 0.0)


def test_windowed_kp_bbox_includes_target_frame_itself():
    obs = {100: _kp_array((999.0, 0.0, 0.9))}
    assert _windowed_kp_bbox(obs, 100, frozenset(), n_frames=10) == (999.0, 0.0, 999.0, 0.0)


def test_windowed_kp_bbox_uses_fewer_frames_near_sequence_start():
    obs = {f: _kp_array((float(f), 0.0, 0.9)) for f in range(0, 5)}
    # Only 5 frames with data exist at all, less than n_frames=10 on either side.
    assert _windowed_kp_bbox(obs, 2, frozenset(), n_frames=10) == (0.0, 0.0, 4.0, 0.0)


def test_windowed_kp_bbox_respects_hidden_indices_and_confidence():
    obs = {
        98: _kp_array((10.0, 10.0, 0.9), (500.0, 500.0, 0.9)),
        102: _kp_array((20.0, 20.0, 0.9), (0.0, 0.0, 0.05)),
    }
    bbox = _windowed_kp_bbox(obs, 100, frozenset({1}), n_frames=10)
    assert bbox == (10.0, 10.0, 20.0, 20.0)


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


def test_tracked_overlay_bbox_drops_points_outside_video_dims():
    # "knee" projects well outside a 640x480 frame (e.g. person occluded /
    # out of this camera's view) -- must not drag the bbox out to it.
    joint_xy = {"hip": [50.0, 60.0], "knee": [-500.0, 2000.0]}
    marker_xy = np.array([[600.0, 400.0], [700.0, 100.0]], dtype=np.float32)
    assert _tracked_overlay_bbox(joint_xy, marker_xy, (640, 480)) == (50.0, 60.0, 600.0, 400.0)


def test_tracked_overlay_bbox_none_when_entirely_out_of_view():
    joint_xy = {"hip": [-50.0, -60.0]}
    marker_xy = np.array([[5000.0, 5000.0]], dtype=np.float32)
    assert _tracked_overlay_bbox(joint_xy, marker_xy, (640, 480)) is None


def test_tracked_overlay_bbox_no_video_dims_keeps_old_behavior():
    joint_xy = {"hip": [-500.0, 2000.0]}
    assert _tracked_overlay_bbox(joint_xy, None) == (-500.0, 2000.0, -500.0, 2000.0)


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
# _composite_black_fill -- black-fill the part of target_rect not decoded
# ---------------------------------------------------------------------------

def test_composite_black_fill_fully_covered_by_decoded_crop():
    # 10x10 decoded crop at full-frame origin (0, 0), scale 1:1; requesting
    # exactly that same window back should reproduce it untouched.
    crop = np.full((10, 10, 3), 200, dtype=np.uint8)
    canvas, x1, y1, black_filled = _composite_black_fill(crop, 0.0, 0.0, 1.0, (0.0, 0.0, 10.0, 10.0))
    assert (x1, y1) == (0.0, 0.0)
    assert canvas.shape == (10, 10, 3)
    assert (canvas == 200).all()
    assert black_filled is False


def test_composite_black_fill_pads_black_where_target_exceeds_decoded():
    # Decoded crop only covers (0,0)-(10,10); requesting a (0,0)-(20,10)
    # window must keep the real pixels on the left and black-fill the right.
    crop = np.full((10, 10, 3), 200, dtype=np.uint8)
    canvas, x1, y1, black_filled = _composite_black_fill(crop, 0.0, 0.0, 1.0, (0.0, 0.0, 20.0, 10.0))
    assert (x1, y1) == (0.0, 0.0)
    assert canvas.shape == (10, 20, 3)
    assert (canvas[:, :10] == 200).all()
    assert (canvas[:, 10:] == 0).all()
    assert black_filled is True


def test_composite_black_fill_places_decoded_patch_at_correct_offset():
    # Decoded crop's full-frame origin is (5, 5); requesting a window
    # starting at (0, 0) must place the real pixels at the right offset
    # inside the canvas, not at the canvas's own origin.
    crop = np.full((10, 10, 3), 200, dtype=np.uint8)
    canvas, x1, y1, black_filled = _composite_black_fill(crop, 5.0, 5.0, 1.0, (0.0, 0.0, 15.0, 15.0))
    assert (x1, y1) == (0.0, 0.0)
    assert canvas.shape == (15, 15, 3)
    assert (canvas[:5, :] == 0).all()
    assert (canvas[:, :5] == 0).all()
    assert (canvas[5:15, 5:15] == 200).all()
    assert black_filled is True


def test_composite_black_fill_entirely_uncovered_is_all_black():
    crop = np.full((10, 10, 3), 200, dtype=np.uint8)
    canvas, x1, y1, black_filled = _composite_black_fill(crop, 0.0, 0.0, 1.0, (100.0, 100.0, 110.0, 110.0))
    assert (x1, y1) == (100.0, 100.0)
    assert canvas.shape == (10, 10, 3)
    assert (canvas == 0).all()
    assert black_filled is True


def test_composite_black_fill_survives_fractional_scale_rounding_drift():
    # Regression test for a real crash: independently rounding the overlap's
    # source-side edge ((ox1 - x1) * scale) and the canvas width
    # ((tx1 - tx0) * scale) can disagree by a pixel at certain fractional
    # scales/offsets, producing a source patch one pixel wider/taller than
    # the canvas slot it's copied into -- "could not broadcast input array
    # from shape (h, w+1, 3) into shape (h, w, 3)". Sweep scales and
    # fractional edge offsets prone to exactly this disagreement.
    crop = np.full((300, 300, 3), 200, dtype=np.uint8)
    x1, y1 = 10.0, 5.0
    for scale in (0.6165, 0.7139, 1.333, 2.71, 0.999, 1.001):
        for edge_frac in (0.1, 0.3, 0.5, 0.7, 0.9):
            target = (
                x1 - 20.0, y1 - 20.0,
                x1 + 300 / scale - edge_frac, y1 + 300 / scale - edge_frac,
            )
            canvas, tx0, ty0, _black_filled = _composite_black_fill(crop, x1, y1, scale, target)
            assert (tx0, ty0) == (target[0], target[1])
            assert canvas.shape[0] > 0 and canvas.shape[1] > 0


def test_composite_black_fill_clamps_implausible_target_rect():
    # Regression test for a real crash: an unbounded target_rect (e.g. from
    # a diverged tracked-marker projection) tried to allocate a multi-
    # terabyte canvas. _composite_black_fill must clamp rather than crash,
    # even if a caller failed to sanity-check target_rect first.
    crop = np.full((10, 10, 3), 200, dtype=np.uint8)
    canvas, tx0, ty0, black_filled = _composite_black_fill(
        crop, 0.0, 0.0, 1.0, (0.0, 0.0, 5_000_000.0, 1_000_000.0)
    )
    assert (tx0, ty0) == (0.0, 0.0)
    assert canvas.shape[0] <= _MAX_CANVAS_DIM_PX
    assert canvas.shape[1] <= _MAX_CANVAS_DIM_PX
    assert black_filled is True


# ---------------------------------------------------------------------------
# _sane_bbox -- rejects a bbox with a non-finite or implausibly large extent
# ---------------------------------------------------------------------------

def test_sane_bbox_passes_through_a_normal_bbox():
    assert _sane_bbox((10.0, 20.0, 100.0, 200.0)) == (10.0, 20.0, 100.0, 200.0)


def test_sane_bbox_none_stays_none():
    assert _sane_bbox(None) is None


def test_sane_bbox_rejects_non_finite_coordinates():
    assert _sane_bbox((0.0, 0.0, float("inf"), 100.0)) is None
    assert _sane_bbox((0.0, 0.0, float("nan"), 100.0)) is None


def test_sane_bbox_rejects_implausibly_large_extent():
    # e.g. a diverged tracked-marker projection landing hundreds of
    # thousands of pixels away -- must be rejected, not unioned in.
    assert _sane_bbox((0.0, 0.0, 571_414.0, 100.0)) is None


def test_sane_bbox_accepts_extent_at_the_cap():
    assert _sane_bbox((0.0, 0.0, 20_000.0, 20_000.0)) == (0.0, 0.0, 20_000.0, 20_000.0)


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
