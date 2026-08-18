# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for app.setup.sync_solver — no Qt required."""

from __future__ import annotations

import pytest

from app.setup.db_context import CaptureVideoInfo, SyncAnchorObservation, SyncTable
from app.setup.sync_solver import SolveResult, check_connectivity, solve_sync_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _video(vid_id: str, fps: float = 30.0) -> CaptureVideoInfo:
    return CaptureVideoInfo(
        id=vid_id,
        shot_id="shot-1",
        camera_instance_id=vid_id,
        file_path=f"/{vid_id}.mp4",
        actual_fps=fps,
        first_video_frame=0,
        last_video_frame=899,
    )


def _obs(anchor_id: str, video_id: str, frame: int, subframe: float = 0.0) -> SyncAnchorObservation:
    return SyncAnchorObservation(
        id=f"obs-{anchor_id}-{video_id}",
        sync_anchor_id=anchor_id,
        shot_video_id=video_id,
        video_frame=frame,
        subframe=subframe,
    )


def _anchor(anchor_id: str, *obs: SyncAnchorObservation) -> tuple[str, list[SyncAnchorObservation]]:
    return (anchor_id, list(obs))


# ---------------------------------------------------------------------------
# check_connectivity
# ---------------------------------------------------------------------------


def test_check_connectivity_single_video() -> None:
    assert check_connectivity([], ["vid-a"]) == (True, [])


def test_check_connectivity_two_connected() -> None:
    anchors = [_anchor("a1", _obs("a1", "vid-a", 100), _obs("a1", "vid-b", 50))]
    ok, isolated = check_connectivity(anchors, ["vid-a", "vid-b"])
    assert ok is True
    assert isolated == []


def test_check_connectivity_chain() -> None:
    anchors = [
        _anchor("a1", _obs("a1", "v1", 0), _obs("a1", "v2", 10)),
        _anchor("a2", _obs("a2", "v2", 20), _obs("a2", "v3", 30)),
    ]
    ok, isolated = check_connectivity(anchors, ["v1", "v2", "v3"])
    assert ok is True
    assert isolated == []


def test_check_connectivity_isolated_video() -> None:
    anchors = [_anchor("a1", _obs("a1", "v1", 0), _obs("a1", "v2", 10))]
    ok, isolated = check_connectivity(anchors, ["v1", "v2", "v3"])
    assert ok is False
    assert "v3" in isolated


def test_check_connectivity_no_anchors() -> None:
    ok, isolated = check_connectivity([], ["v1", "v2"])
    assert ok is False
    assert set(isolated) == {"v2"}  # v1 is the start node, v2 is unreachable


def test_check_connectivity_empty_video_list() -> None:
    assert check_connectivity([], []) == (True, [])


# ---------------------------------------------------------------------------
# solve_sync_graph — basic cases
# ---------------------------------------------------------------------------


def test_solve_single_pair() -> None:
    """Two cameras, one anchor: offset is computed correctly."""
    videos = [_video("v1", 30.0), _video("v2", 30.0)]
    # v1 frame 300 is simultaneous with v2 frame 150
    anchors = [_anchor("a1", _obs("a1", "v1", 300), _obs("a1", "v2", 150))]

    result = solve_sync_graph(anchors, videos, reference_video_id="v1")

    assert isinstance(result, SolveResult)
    assert {"v1", "v2"} == result.connected_video_ids
    assert result.isolated_video_ids == set()

    pts = {sp.shot_video_id: sp for sp in result.sync_points}
    assert "v1" in pts and "v2" in pts

    # Both observations should map to the same global time
    t_v1 = pts["v1"].timestamp_s
    t_v2 = pts["v2"].timestamp_s
    assert t_v1 == pytest.approx(t_v2, abs=1e-9)

    # Reference has offset 0: t = frame/fps = 300/30 = 10.0
    assert t_v1 == pytest.approx(10.0)


def test_solve_linear_chain() -> None:
    """A→B→C chain: timestamps propagate transitively."""
    videos = [_video("v1", 30.0), _video("v2", 30.0), _video("v3", 30.0)]
    anchors = [
        _anchor("a1", _obs("a1", "v1", 300), _obs("a1", "v2", 150)),
        _anchor("a2", _obs("a2", "v2", 600), _obs("a2", "v3", 450)),
    ]

    result = solve_sync_graph(anchors, videos, reference_video_id="v1")

    assert result.connected_video_ids == {"v1", "v2", "v3"}
    assert result.isolated_video_ids == set()

    pts = {sp.shot_video_id: sp for sp in result.sync_points}
    # a1: v1@300 (t=10s) ↔ v2@150 (t=5s from start), so v2 offset = 10-5 = +5s
    # a2: v2@600 (t=600/30+5=25s) ↔ v3@450 (t=450/30=15s from start), so v3 offset=25-15=+10s
    # Both observations in each anchor must share the same global time
    a1_t_v1 = pts["v1"].timestamp_s   # = 10.0
    # find a2 point for v2
    a2_pts = [sp for sp in result.sync_points if sp.shot_video_id == "v2" and sp.video_frame == 600]
    a2_pts_v3 = [sp for sp in result.sync_points if sp.shot_video_id == "v3"]
    assert len(a2_pts) == 1
    assert len(a2_pts_v3) == 1
    assert a2_pts[0].timestamp_s == pytest.approx(a2_pts_v3[0].timestamp_s, abs=1e-9)


def test_solve_star_topology() -> None:
    """One reference camera linked to three others via separate anchors."""
    videos = [_video(f"v{i}", 30.0) for i in range(4)]
    anchors = [
        _anchor("a1", _obs("a1", "v0", 300), _obs("a1", "v1", 100)),
        _anchor("a2", _obs("a2", "v0", 600), _obs("a2", "v2", 200)),
        _anchor("a3", _obs("a3", "v0", 900), _obs("a3", "v3", 300)),
    ]

    result = solve_sync_graph(anchors, videos, reference_video_id="v0")

    assert result.connected_video_ids == {"v0", "v1", "v2", "v3"}
    assert result.isolated_video_ids == set()
    assert len(result.sync_points) == 6  # 2 obs per anchor × 3 anchors


def test_solve_isolated_video_excluded_from_sync_points() -> None:
    """A video with no anchors appears in isolated_video_ids and not in sync_points."""
    videos = [_video("v1", 30.0), _video("v2", 30.0), _video("v3", 30.0)]
    anchors = [_anchor("a1", _obs("a1", "v1", 100), _obs("a1", "v2", 50))]

    result = solve_sync_graph(anchors, videos, reference_video_id="v1")

    assert "v3" in result.isolated_video_ids
    video_ids_in_points = {sp.shot_video_id for sp in result.sync_points}
    assert "v3" not in video_ids_in_points


def test_solve_no_videos_returns_empty() -> None:
    result = solve_sync_graph([], [], reference_video_id=None)
    assert result.sync_points == []
    assert result.connected_video_ids == set()
    assert result.isolated_video_ids == set()


def test_solve_auto_selects_reference() -> None:
    """With reference_video_id=None the most-connected video is chosen."""
    # v2 appears in 2 anchors; v1 and v3 appear in 1 each
    videos = [_video("v1", 30.0), _video("v2", 30.0), _video("v3", 30.0)]
    anchors = [
        _anchor("a1", _obs("a1", "v1", 300), _obs("a1", "v2", 100)),
        _anchor("a2", _obs("a2", "v2", 600), _obs("a2", "v3", 200)),
    ]

    result = solve_sync_graph(anchors, videos)

    # All three should be reachable regardless of which is chosen as ref
    assert result.connected_video_ids == {"v1", "v2", "v3"}


# ---------------------------------------------------------------------------
# solve_sync_graph — subframe precision
# ---------------------------------------------------------------------------


def test_solve_subframe_shifts_timestamp() -> None:
    """subframe=0.5 shifts the global time by half a frame period."""
    videos = [_video("v1", 30.0), _video("v2", 30.0)]
    anchors = [
        _anchor(
            "a1",
            _obs("a1", "v1", 300, subframe=0.0),
            _obs("a1", "v2", 150, subframe=0.5),
        )
    ]

    result = solve_sync_graph(anchors, videos, reference_video_id="v1")

    pts = {sp.shot_video_id: sp for sp in result.sync_points}
    t_v1 = pts["v1"].timestamp_s   # = 300/30 = 10.0
    t_v2 = pts["v2"].timestamp_s
    assert t_v1 == pytest.approx(10.0)
    assert t_v2 == pytest.approx(10.0, abs=1e-9)

    # v2 offset should be 10.0 - 150.5/30 = 10.0 - 5.0167 = 4.9833...
    # v2 timestamp_s = (150 + 0.5)/30 + offset_v2 = 5.0167 + 4.9833 = 10.0 ✓


def test_solve_different_fps() -> None:
    """Cameras with different fps: offsets computed correctly."""
    videos = [_video("ref", 30.0), _video("slow", 25.0)]
    # ref frame 300 (t=10s) ↔ slow frame 250 (t=10s at 25fps)
    anchors = [_anchor("a1", _obs("a1", "ref", 300), _obs("a1", "slow", 250))]

    result = solve_sync_graph(anchors, videos, reference_video_id="ref")

    pts = {sp.shot_video_id: sp for sp in result.sync_points}
    assert pts["ref"].timestamp_s == pytest.approx(10.0)
    assert pts["slow"].timestamp_s == pytest.approx(10.0, abs=1e-9)


# ---------------------------------------------------------------------------
# solve_sync_graph — multiple anchors per camera pair (drift)
# ---------------------------------------------------------------------------


def test_solve_multiple_anchors_per_pair() -> None:
    """Two anchors between the same pair produce two SyncPoints per video."""
    videos = [_video("v1", 30.0), _video("v2", 30.0)]
    anchors = [
        _anchor("a1", _obs("a1", "v1", 300), _obs("a1", "v2", 150)),
        _anchor("a2", _obs("a2", "v1", 900), _obs("a2", "v2", 750)),
    ]

    result = solve_sync_graph(anchors, videos, reference_video_id="v1")

    v1_pts = [sp for sp in result.sync_points if sp.shot_video_id == "v1"]
    v2_pts = [sp for sp in result.sync_points if sp.shot_video_id == "v2"]
    assert len(v1_pts) == 2
    assert len(v2_pts) == 2


# ---------------------------------------------------------------------------
# solve_sync_graph — camera_instance_id collision avoidance
# ---------------------------------------------------------------------------


def test_solve_uses_shot_video_id_as_camera_instance_id() -> None:
    """SyncPoints use shot_video_id for camera_instance_id to avoid PK collisions."""
    videos = [_video("v1"), _video("v2")]
    # Give both videos the same camera_instance_id (the __unassigned__ case)
    videos[0] = videos[0]._replace(camera_instance_id="__unassigned__")
    videos[1] = videos[1]._replace(camera_instance_id="__unassigned__")

    anchors = [_anchor("a1", _obs("a1", "v1", 100), _obs("a1", "v2", 50))]
    result = solve_sync_graph(anchors, videos, reference_video_id="v1")

    for sp in result.sync_points:
        assert sp.camera_instance_id == sp.shot_video_id


# ---------------------------------------------------------------------------
# Helpers for SyncTable round-trip tests
# ---------------------------------------------------------------------------


def _build_sync_table(
    anchors: list[tuple[str, list[SyncAnchorObservation]]],
    videos: list[CaptureVideoInfo],
    reference_video_id: str | None = None,
) -> tuple[SyncTable, dict[str, float]]:
    """Solve anchor graph and return (SyncTable, fps_by_video)."""
    result = solve_sync_graph(anchors, videos, reference_video_id=reference_video_id)
    fps_by_video = {v.id: (v.actual_fps or 30.0) for v in videos}
    pts_by_vid: dict[str, list] = {}
    for sp in result.sync_points:
        pts_by_vid.setdefault(sp.shot_video_id, []).append(sp)
    table = SyncTable(result.sync_points, fps_by_video)
    return table, fps_by_video


# ---------------------------------------------------------------------------
# SyncTable round-trip — basic correctness
# ---------------------------------------------------------------------------


def test_synctable_anchor_frame_maps_to_anchor_timestamp() -> None:
    """lookup(anchor_timestamp, vid) must return the anchor frame itself."""
    fps = 120.0
    videos = [_video("gopro", fps), _video("pixel", fps)]
    anchors = [_anchor("a1", _obs("a1", "gopro", 80442), _obs("a1", "pixel", 77965))]

    table, _ = _build_sync_table(anchors, videos, reference_video_id="pixel")

    # Find the timestamp for the anchor
    pts = {sp.shot_video_id: sp for sp in solve_sync_graph(
        anchors, videos, reference_video_id="pixel"
    ).sync_points}
    t_anchor = pts["pixel"].timestamp_s  # global time of the anchor event

    # Both cameras should look up to their respective anchor frames at t_anchor
    assert table.lookup(t_anchor, "pixel") == 77965
    assert table.lookup(t_anchor, "gopro") == 80442


def test_synctable_anchor_frames_share_same_global_timestamp() -> None:
    """frame_to_global_time(anchor_frame) must be equal for both cameras."""
    fps = 120.0
    videos = [_video("gopro", fps), _video("pixel", fps)]
    anchors = [_anchor("a1", _obs("a1", "gopro", 80442), _obs("a1", "pixel", 77965))]

    table, _ = _build_sync_table(anchors, videos, reference_video_id="pixel")

    t_gopro = table.frame_to_global_time(80442, "gopro")
    t_pixel = table.frame_to_global_time(77965, "pixel")

    assert t_gopro is not None
    assert t_pixel is not None
    assert t_gopro == pytest.approx(t_pixel, abs=1e-6)


def test_synctable_same_fps_constant_frame_offset() -> None:
    """With same fps cameras the frame offset equals anchor-frame difference at any time."""
    fps = 120.0
    gopro_anchor, pixel_anchor = 80442, 77965
    expected_offset = gopro_anchor - pixel_anchor  # +2477

    videos = [_video("gopro", fps), _video("pixel", fps)]
    anchors = [_anchor("a1", _obs("a1", "gopro", gopro_anchor), _obs("a1", "pixel", pixel_anchor))]

    table, _ = _build_sync_table(anchors, videos, reference_video_id="pixel")

    # Test at several different global times
    for pixel_frame in [77965, 150000, 231076, 50000]:
        t = table.frame_to_global_time(pixel_frame, "pixel")
        assert t is not None
        gopro_frame = table.lookup(t, "gopro")
        assert gopro_frame is not None
        assert gopro_frame == pytest.approx(pixel_frame + expected_offset, abs=1)


def test_synctable_switch_camera_preserves_global_time() -> None:
    """Simulates the pose-extraction camera-switch: seeking camera B to the
    global time of camera A's current frame must land at the expected frame.

    This reproduces the exact scenario that was producing wrong frames in the UI:
    pixel_9 at frame 231076 → switch to gopro_mini → should be at ~233553, not 8886.
    """
    fps = 120.0
    gopro_anchor, pixel_anchor = 80442, 77965

    videos = [_video("gopro", fps), _video("pixel", fps)]
    anchors = [_anchor("a1", _obs("a1", "gopro", gopro_anchor), _obs("a1", "pixel", pixel_anchor))]

    table, _ = _build_sync_table(anchors, videos, reference_video_id="pixel")

    # User is viewing pixel at frame 231076
    pixel_frame = 231076
    global_s = table.frame_to_global_time(pixel_frame, "pixel")
    assert global_s is not None

    # Switch to gopro: frame must be anchor + same delta
    gopro_frame = table.lookup(global_s, "gopro")
    assert gopro_frame is not None

    expected = pixel_frame + (gopro_anchor - pixel_anchor)  # 231076 + 2477 = 233553
    assert gopro_frame == pytest.approx(expected, abs=1), (
        f"Expected gopro frame ~{expected} but got {gopro_frame}. "
        f"Sync is {gopro_anchor - pixel_anchor} frames off."
    )


# ---------------------------------------------------------------------------
# SyncTable round-trip — multiple anchors
# ---------------------------------------------------------------------------


def test_synctable_two_consistent_anchors_both_correct() -> None:
    """Two anchors that imply the SAME offset both round-trip correctly.

    Consistent anchors: same fps, same frame-offset (2477) at two different times.
    """
    fps = 120.0
    # Both anchors imply the same offset (gopro is always 2477 frames ahead of pixel).
    # anchor1: gopro@80442 ↔ pixel@77965
    # anchor2: gopro@88500 ↔ pixel@86023  (80442+8058 ↔ 77965+8058, same offset)
    videos = [_video("gopro", fps), _video("pixel", fps)]
    anchors = [
        _anchor("a1", _obs("a1", "gopro", 80442), _obs("a1", "pixel", 77965)),
        _anchor("a2", _obs("a2", "gopro", 88500), _obs("a2", "pixel", 86023)),
    ]

    table, _ = _build_sync_table(anchors, videos, reference_video_id="pixel")

    for gopro_f, pixel_f in [(80442, 77965), (88500, 86023)]:
        t_g = table.frame_to_global_time(gopro_f, "gopro")
        t_p = table.frame_to_global_time(pixel_f, "pixel")
        assert t_g is not None and t_p is not None
        assert t_g == pytest.approx(t_p, abs=1 / fps), (
            f"Anchor ({gopro_f},{pixel_f}): gopro t={t_g:.4f}s, pixel t={t_p:.4f}s"
        )


def test_synctable_two_anchors_ols_perfect_fit() -> None:
    """With 2 anchor pairs between the same cameras the OLS solver fits a
    line through both points exactly (2 points always determine a line).

    This means the solver finds an 'effective fps' that makes both anchors
    consistent, even when the anchors imply different single-anchor offsets.
    Residual at each anchor is 0 regardless of whether the two anchors are
    genuinely consistent or come from a drifting/wrong-fps camera.

    The old single-anchor BFS would have produced ~2077 frame residual.
    The OLS solver eliminates that by using all available evidence.
    """
    fps = 120.0
    # Anchor A: gopro@1200  ↔ pixel@800   → single-anchor offset −3.33 s
    # Anchor B: gopro@80442 ↔ pixel@77965 → single-anchor offset −20.64 s
    # OLS through these two points finds an effective fps ≈ 123.2 and a
    # consistent offset — residual at each anchor is 0.
    videos = [_video("gopro", fps), _video("pixel", fps)]
    anchors = [
        _anchor("old", _obs("old", "gopro", 1200), _obs("old", "pixel", 800)),
        _anchor("new", _obs("new", "gopro", 80442), _obs("new", "pixel", 77965)),
    ]

    result = solve_sync_graph(anchors, videos, reference_video_id="pixel")
    table = SyncTable(result.sync_points, result.effective_fps)

    # Both anchor timestamps are exact in the OLS fit.
    for pixel_f, gopro_f in [(800, 1200), (77965, 80442)]:
        t_pixel = table.frame_to_global_time(pixel_f, "pixel")
        gopro_at_t = table.lookup(t_pixel, "gopro")
        assert gopro_at_t == pytest.approx(gopro_f, abs=1), (
            f"Anchor residual too large at pixel={pixel_f}: got gopro={gopro_at_t}, expected {gopro_f}"
        )

    # Effective fps deviates from nominal (two points with different implied offsets
    # at the same nominal fps means the solver infers a different fps).
    assert result.effective_fps["gopro"] == pytest.approx(123.2, abs=1.0)
