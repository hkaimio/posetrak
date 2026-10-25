# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the geometric association of detector fragments to persons."""
from __future__ import annotations

from collections import Counter

import numpy as np

from posetrak.detection import track_assignment as ta
from tests.detection import synthetic_people as sp

TRUNK = [5, 6, 11, 12]


def _scene(grounds: list[tuple[float, float]], n_cams: int = 3):
    """Per camera: one Obs per person (trunk of a standing person), and the projection matrices."""
    cams = sp.cameras(n_cams)
    P = {f"cam{k}": sp.K @ np.hstack([R, t[:, None]]) for k, (R, t) in enumerate(cams)}
    obs = []
    for k, (R, t) in enumerate(cams):
        for p, g in enumerate(grounds):
            xy = sp.project(R, t, sp.person_points(np.array(g)))
            obs.append(ta.Obs(f"cam{k}", p, 0, xy[TRUNK], np.ones(4, bool), float(np.ptp(xy[:, 1]))))
    return obs, P


def test_boxes_of_one_person_agree_and_boxes_of_two_persons_do_not() -> None:
    obs, P = _scene([(0.0, 0.0), (1.5, -1.0)], n_cams=2)
    cam0_a, cam0_b, cam1_a, cam1_b = obs

    same = ta.pair_error(cam0_a, cam1_a, P["cam0"], P["cam1"], 3)
    other = ta.pair_error(cam0_a, cam1_b, P["cam0"], P["cam1"], 3)

    assert same < 0.001
    assert other > 0.05


def test_an_instant_clusters_into_one_hypothesis_per_person_at_its_place() -> None:
    obs, P = _scene([(0.0, 0.0), (2.0, -1.5)])

    hyps = ta.cluster_instant(obs, P, ta.AssignParams())

    assert sorted(len(h.members) for h in hyps) == [3, 3]
    assert sorted(h.frags for h in hyps) == [frozenset({("cam0", 0), ("cam1", 0), ("cam2", 0)}),
                                              frozenset({("cam0", 1), ("cam1", 1), ("cam2", 1)})]
    places = sorted(tuple(np.round(h.pos[:2], 2)) for h in hyps)
    assert places[0] == (0.0, 0.0) or places[1] == (0.0, 0.0)           # the trunk centre is above the ground point


def test_pieces_follow_the_skeleton_not_the_detector_id() -> None:
    kp_a = np.array([[100.0 + 10 * j, 200.0, 0.9] for j in range(17)])
    kp_b = kp_a + [300.0, 0.0, 0.0]                                      # another person, 300 px away
    frames = {}
    for f in range(10):
        a = kp_a + [f, 0.0, 0.0]
        b = kp_b - [f, 0.0, 0.0]
        # the detector swaps its ids from frame 5 on
        frames[f] = [(0, a, 800.0), (1, b, 800.0)] if f < 5 else [(1, a, 800.0), (0, b, 800.0)]

    piece = ta.make_pieces(frames, ta.AssignParams())

    assert piece[(0, 0)] == piece[(9, 1)]            # person A: id 0, then id 1
    assert piece[(0, 1)] == piece[(9, 0)]            # person B: id 1, then id 0
    assert piece[(0, 0)] != piece[(0, 1)]


def test_a_box_without_comparable_keypoints_continues_its_detector_track() -> None:
    kp = np.array([[100.0 + 10 * j, 200.0, 0.9] for j in range(17)])
    blind = kp.copy()
    blind[:, 2] = 0.0
    frames = {0: [(7, kp, 800.0)], 1: [(7, blind, 800.0)], 2: [(7, kp, 800.0)]}

    piece = ta.make_pieces(frames, ta.AssignParams())

    assert len(set(piece.values())) == 1


def test_a_fragment_that_changes_person_is_split_between_the_boxes_that_say_so() -> None:
    frames = list(range(100))
    votes = {f: 0 for f in range(0, 40)} | {f: 1 for f in range(60, 100)}

    segments = ta.label_fragment(frames, votes, min_run=4)

    assert [(s.first, s.last, s.person) for s in segments] == [(0, 49, 0), (50, 99, 1)]
    assert segments[0].blind == [(40, 49)] and segments[1].blind == [(50, 59)]


def test_a_flicker_inside_a_fragment_does_not_split_it() -> None:
    frames = list(range(50))
    votes = {f: 0 for f in frames} | {20: 1, 21: 1}

    segments = ta.label_fragment(frames, votes, min_run=4)

    assert [(s.first, s.last, s.person) for s in segments] == [(0, 49, 0)]


def test_a_fragment_without_any_person_stays_unassigned() -> None:
    assert [(s.person) for s in ta.label_fragment([3, 4, 5], {}, min_run=4)] == [None]


def _track(i: int, first: int, last: int, at: tuple[float, float]) -> ta.Track:
    return ta.Track(i, {k: np.array([at[0], at[1], 1.0]) for k in range(first, last + 1)})


def test_tracks_at_the_same_time_in_different_places_are_two_persons() -> None:
    tracks = [_track(0, 0, 100, (0.0, 0.0)), _track(1, 0, 100, (3.0, 0.0))]
    ends = {t.id: (Counter(), Counter()) for t in tracks}

    label = ta.label_tracks(tracks, ends, 0.01, 2, ta.AssignParams())

    assert label[0] != label[1] and None not in label.values()


def test_a_track_after_a_contact_takes_the_person_of_the_fragments_it_shares() -> None:
    # Persons X and Y meet: before the contact X is track 0 and Y track 1, after it X is track 2 and Y track 3.
    # The positions favour the wrong pairing a little; the fragments that span the contact say who is who.
    tracks = [_track(0, 0, 100, (0.0, 0.0)), _track(1, 0, 100, (3.0, 0.0)),
              _track(2, 200, 300, (1.25, 0.0)), _track(3, 200, 300, (1.75, 0.0))]
    ends = {0: (Counter(), Counter({("c", 5): 100})), 1: (Counter(), Counter({("c", 6): 100})),
            2: (Counter({("c", 6): 100}), Counter()), 3: (Counter({("c", 5): 100}), Counter())}

    label = ta.label_tracks(tracks, ends, 0.01, 2, ta.AssignParams())

    assert label[0] == label[3] and label[1] == label[2] and label[0] != label[1]


def test_a_seed_fixes_the_person_of_a_track() -> None:
    tracks = [_track(0, 0, 100, (0.0, 0.0)), _track(1, 0, 100, (3.0, 0.0))]
    ends = {t.id: (Counter(), Counter()) for t in tracks}

    label = ta.label_tracks(tracks, ends, 0.01, 2, ta.AssignParams(), anchors={0: 1})

    assert label == {0: 1, 1: 0}


def test_a_bystander_does_not_join_the_only_person() -> None:
    tracks = [_track(0, 0, 100, (0.0, 0.0)), _track(1, 20, 80, (4.0, 0.0))]
    ends = {t.id: (Counter(), Counter()) for t in tracks}

    label = ta.label_tracks(tracks, ends, 0.01, 1, ta.AssignParams())

    assert label[0] == 0 and label[1] is None


def test_the_number_of_persons_is_the_number_seen_together_for_a_fifth_of_the_time() -> None:
    tracks = [_track(0, 0, 1000, (0, 0)), _track(1, 0, 400, (3, 0)), _track(2, 100, 140, (0, 3))]

    assert ta.count_persons(tracks, ta.AssignParams()) == 2
