# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for MotionGatedLinker (see dot_tracklet.py's own module docstring
for the Phase B investigation this exists to act on, and status.md's
2026-09-11 entries for the real-data validation that shaped its gates)."""
from __future__ import annotations

from posetrak.detection.dot_blob_detector import BlobCandidate
from posetrak.detection.dot_tracklet import MotionGatedLinker


def _cand(cx: float, cy: float) -> BlobCandidate:
    return BlobCandidate(cx=cx, cy=cy, area=10.0, compactness=0.9, bbox=(0, 0, 1, 1),
                         major_axis_px=3.0, minor_axis_px=3.0)


def _link_all(linker: MotionGatedLinker, frames: list[list[BlobCandidate]]) -> None:
    for vf, f in enumerate(frames):
        linker.link_frame(vf, f)


def test_a_moving_candidate_keeps_the_same_id_across_frames() -> None:
    linker = MotionGatedLinker()
    frames = [[_cand(100.0, 100.0)], [_cand(105.0, 100.0)], [_cand(110.0, 100.0)]]
    _link_all(linker, frames)

    ids = [f[0].tracklet_id for f in frames]
    assert ids[0] == ids[1] == ids[2]
    assert ids[0] != -1  # a real id was assigned, not left at the unlinked default


def test_two_independent_candidates_get_different_ids_and_stay_linked() -> None:
    linker = MotionGatedLinker()
    frames = [
        [_cand(100.0, 100.0), _cand(400.0, 400.0)],
        [_cand(105.0, 100.0), _cand(405.0, 400.0)],
        [_cand(110.0, 100.0), _cand(410.0, 400.0)],
    ]
    _link_all(linker, frames)

    assert frames[0][0].tracklet_id == frames[1][0].tracklet_id == frames[2][0].tracklet_id
    assert frames[0][1].tracklet_id == frames[1][1].tracklet_id == frames[2][1].tracklet_id
    assert frames[0][0].tracklet_id != frames[0][1].tracklet_id


def test_every_candidate_gets_some_id_even_a_lone_singleton() -> None:
    """The linker assigns identity, not validity -- it never drops a
    candidate, unlike the prototype's own build_tracklets() filter."""
    linker = MotionGatedLinker()
    frame = [_cand(100.0, 100.0)]
    linker.link_frame(0, frame)
    assert frame[0].tracklet_id != -1


def test_a_candidate_too_far_away_starts_a_new_tracklet() -> None:
    linker = MotionGatedLinker(birth_gate_px=20.0)
    linker.link_frame(0, [_cand(100.0, 100.0)])
    frame2 = [_cand(500.0, 500.0)]  # far outside birth_gate_px
    linker.link_frame(1, frame2)

    assert frame2[0].tracklet_id != 0  # didn't reuse the first tracklet's id


def test_a_tracklet_survives_a_brief_gap_within_max_missed() -> None:
    linker = MotionGatedLinker(birth_gate_px=20.0, max_missed=2)
    f1 = [_cand(100.0, 100.0)]
    linker.link_frame(0, f1)
    linker.link_frame(1, [_cand(105.0, 100.0)])  # establish a velocity estimate
    linker.link_frame(2, [])  # missed frame
    linker.link_frame(3, [])  # missed frame -- still within max_missed
    f4 = [_cand(115.0, 100.0)]  # consistent with the established ~5px/frame drift
    linker.link_frame(4, f4)

    assert f4[0].tracklet_id == f1[0].tracklet_id


def test_a_tracklet_does_not_survive_beyond_max_missed() -> None:
    linker = MotionGatedLinker(birth_gate_px=20.0, max_missed=1)
    f1 = [_cand(100.0, 100.0)]
    linker.link_frame(0, f1)
    linker.link_frame(1, [])
    linker.link_frame(2, [])  # exceeds max_missed=1 -- tracklet should have aged out
    f4 = [_cand(105.0, 100.0)]
    linker.link_frame(3, f4)

    assert f4[0].tracklet_id != f1[0].tracklet_id


def test_two_candidates_prefer_the_nearer_tracklet_match() -> None:
    linker = MotionGatedLinker(birth_gate_px=50.0)
    linker.link_frame(0, [_cand(100.0, 100.0), _cand(300.0, 300.0)])
    id_a, id_b = linker._open[0].tracklet_id, linker._open[1].tracklet_id

    # A new frame with two candidates, each clearly closer to one of the two
    # open tracklets than the other.
    f2 = [_cand(110.0, 100.0), _cand(310.0, 300.0)]
    linker.link_frame(1, f2)

    assert f2[0].tracklet_id == id_a
    assert f2[1].tracklet_id == id_b


def test_an_occluded_track_coasts_past_a_stale_distractor() -> None:
    """The real bug this linker exists to fix (status.md 2026-09-11): a
    tracklet moving steadily, then occluded for a couple of frames, must
    NOT jump onto a different, stationary candidate that happens to sit
    near its *last seen* pixel -- it should keep predicting forward along
    its own established velocity and pick up its own continuation
    instead, even though the stale distractor is closer to the last-seen
    point than the real continuation is."""
    linker = MotionGatedLinker(birth_gate_px=20.0, max_missed=3)
    # establish a steady ~10px/frame rightward drift
    linker.link_frame(0, [_cand(100.0, 100.0)])
    real = linker.link_frame(1, [_cand(110.0, 100.0)])
    tid = linker._open[0].tracklet_id

    # occluded for two frames -- a distractor sits right where the track
    # was last actually seen (120, 100), not where its own motion predicts
    # it should now be (~140, 100)
    linker.link_frame(2, [])
    linker.link_frame(3, [])
    distractor = _cand(122.0, 100.0)
    continuation = _cand(140.0, 100.0)
    frame4 = [distractor, continuation]
    linker.link_frame(4, frame4)

    assert continuation.tracklet_id == tid
    assert distractor.tracklet_id != tid


def test_coasting_does_not_compound_across_multiple_missed_frames() -> None:
    """Regression for a real bug found on full-capture data (status.md,
    2026-09-11): `last_frame` must advance on every coast step, not only
    on a real match -- otherwise the next predict() computes dt as time-
    since-last-*match* and reapplies it on top of state that was already
    advanced by the previous coast, compounding quadratically instead of
    linearly. A steady ~10px/frame mover occluded for 5 frames must
    predict close to its true constant-velocity position (~160,100) with
    a correspondingly tight gate -- not run far past it (~310+,100) with
    a gate so wide it would accept nearly any nearby candidate, including
    a distant, unrelated real marker (exactly the "big jump" failure
    Harri found reviewing the full capture). The earlier 2-frame-gap test
    above didn't run long enough to expose this -- the compounding is
    small until several consecutive misses accumulate."""
    linker = MotionGatedLinker(birth_gate_px=20.0, max_missed=10)
    linker.link_frame(0, [_cand(100.0, 100.0)])
    linker.link_frame(1, [_cand(110.0, 100.0)])  # establishes ~10px/frame velocity
    tid = linker._open[0].tracklet_id

    for vf in range(2, 7):
        linker.link_frame(vf, [])  # 5 consecutive missed frames

    continuation = _cand(160.0, 100.0)  # true constant-velocity position
    far_distractor = _cand(280.0, 100.0)  # only reachable via the buggy runaway
    frame7 = [far_distractor, continuation]
    linker.link_frame(7, frame7)

    assert continuation.tracklet_id == tid
    assert far_distractor.tracklet_id != tid


def test_two_close_simultaneous_tracks_do_not_swap_identity() -> None:
    """The second failure mode found validating this linker (status.md
    2026-09-11, gopro13_02#283): two simultaneously visible, closely-
    spaced real markers moving in parallel must not have their ids
    ping-pong between frames just because a Hungarian solver's tie-break
    happens to flip. Both should keep their own id across an ambiguous
    step rather than have identities cross."""
    linker = MotionGatedLinker(birth_gate_px=20.0)
    f0 = [_cand(100.0, 100.0), _cand(112.0, 100.0)]
    linker.link_frame(0, f0)
    id_a, id_b = f0[0].tracklet_id, f0[1].tracklet_id
    assert id_a != id_b

    # both drift in parallel, staying close together
    for vf, (xa, xb) in enumerate([(105.0, 117.0), (110.0, 122.0), (115.0, 127.0)], start=1):
        f = [_cand(xa, 100.0), _cand(xb, 100.0)]
        linker.link_frame(vf, f)
        assert f[0].tracklet_id == id_a
        assert f[1].tracklet_id == id_b
