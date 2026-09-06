# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for DotTrackletLinker (see dot_tracklet.py's own module docstring
for the Phase B investigation this exists to act on)."""
from __future__ import annotations

from posetrak.detection.dot_blob_detector import BlobCandidate
from posetrak.detection.dot_tracklet import DotTrackletLinker


def _cand(cx: float, cy: float) -> BlobCandidate:
    return BlobCandidate(cx=cx, cy=cy, area=10.0, compactness=0.9, bbox=(0, 0, 1, 1),
                         major_axis_px=3.0, minor_axis_px=3.0)


def test_a_moving_candidate_keeps_the_same_id_across_frames() -> None:
    linker = DotTrackletLinker(max_link_px=20.0)
    frames = [[_cand(100.0, 100.0)], [_cand(105.0, 100.0)], [_cand(110.0, 100.0)]]
    for f in frames:
        linker.link_frame(f)

    ids = [f[0].tracklet_id for f in frames]
    assert ids[0] == ids[1] == ids[2]
    assert ids[0] != -1  # a real id was assigned, not left at the unlinked default


def test_two_independent_candidates_get_different_ids_and_stay_linked() -> None:
    linker = DotTrackletLinker(max_link_px=20.0)
    frames = [
        [_cand(100.0, 100.0), _cand(400.0, 400.0)],
        [_cand(105.0, 100.0), _cand(405.0, 400.0)],
    ]
    for f in frames:
        linker.link_frame(f)

    assert frames[0][0].tracklet_id == frames[1][0].tracklet_id
    assert frames[0][1].tracklet_id == frames[1][1].tracklet_id
    assert frames[0][0].tracklet_id != frames[0][1].tracklet_id


def test_every_candidate_gets_some_id_even_a_lone_singleton() -> None:
    """The linker assigns identity, not validity -- it never drops a
    candidate, unlike the prototype's own build_tracklets() filter."""
    linker = DotTrackletLinker()
    frame = [_cand(100.0, 100.0)]
    linker.link_frame(frame)
    assert frame[0].tracklet_id != -1


def test_a_candidate_too_far_away_starts_a_new_tracklet() -> None:
    linker = DotTrackletLinker(max_link_px=20.0)
    linker.link_frame([_cand(100.0, 100.0)])
    frame2 = [_cand(500.0, 500.0)]  # far outside max_link_px
    linker.link_frame(frame2)

    assert frame2[0].tracklet_id not in {0}  # didn't reuse the first tracklet's id
    linker.link_frame([_cand(505.0, 500.0)])  # confirm it's its own live tracklet


def test_a_tracklet_survives_a_brief_gap_within_max_missed() -> None:
    linker = DotTrackletLinker(max_link_px=20.0, max_missed=2)
    f1 = [_cand(100.0, 100.0)]
    linker.link_frame(f1)
    linker.link_frame([])  # missed frame 1
    linker.link_frame([])  # missed frame 2 -- still within max_missed
    f4 = [_cand(105.0, 100.0)]
    linker.link_frame(f4)

    assert f4[0].tracklet_id == f1[0].tracklet_id


def test_a_tracklet_does_not_survive_beyond_max_missed() -> None:
    linker = DotTrackletLinker(max_link_px=20.0, max_missed=1)
    f1 = [_cand(100.0, 100.0)]
    linker.link_frame(f1)
    linker.link_frame([])
    linker.link_frame([])  # exceeds max_missed=1 -- tracklet should have aged out
    f4 = [_cand(105.0, 100.0)]
    linker.link_frame(f4)

    assert f4[0].tracklet_id != f1[0].tracklet_id


def test_two_candidates_prefer_the_nearer_tracklet_match() -> None:
    linker = DotTrackletLinker(max_link_px=50.0)
    linker.link_frame([_cand(100.0, 100.0), _cand(300.0, 300.0)])
    id_a, id_b = linker._open[0].tracklet_id, linker._open[1].tracklet_id

    # A new frame with two candidates, each clearly closer to one of the two
    # open tracklets than the other.
    f2 = [_cand(110.0, 100.0), _cand(310.0, 300.0)]
    linker.link_frame(f2)

    assert f2[0].tracklet_id == id_a
    assert f2[1].tracklet_id == id_b
