"""Tests for track assignment conflict detection and 'from here onwards' expansion."""
from __future__ import annotations

import pytest

from app.pose.assignment import find_assignment_conflicts, tracks_from_here_onwards

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CAM = "cam-a"
CAM2 = "cam-b"


def spans(*args: tuple[int, int, int]) -> dict:
    """Build a spans dict from (tid, first, last) triples, all in CAM."""
    return {(CAM, tid): (first, last) for tid, first, last in args}


def assignments(**kwargs: str) -> dict:
    """Build assignments dict from keyword args: t0='harri' → {(CAM, 0): 'harri'}."""
    return {(CAM, int(k[1:])): v for k, v in kwargs.items()}


# ---------------------------------------------------------------------------
# find_assignment_conflicts
# ---------------------------------------------------------------------------

class TestFindConflicts:

    def test_no_existing_assignments(self):
        sp = spans((0, 0, 100), (1, 50, 150))
        assert find_assignment_conflicts(CAM, [1], "harri", sp, {}) == []

    def test_no_overlap_different_persons(self):
        sp = spans((0, 0, 100), (1, 50, 150))
        asn = assignments(t0="timo")
        assert find_assignment_conflicts(CAM, [1], "harri", sp, asn) == []

    def test_no_overlap_adjacent_tracks(self):
        # track 0 ends at frame 100; track 1 starts at frame 100 — adjacent, not a conflict
        sp = spans((0, 0, 100), (1, 100, 200))
        asn = assignments(t0="harri")
        assert find_assignment_conflicts(CAM, [1], "harri", sp, asn) == []

    def test_partial_overlap_from_left(self):
        # existing: 0–100; new: 50–150 → overlap at 50–100
        sp = spans((0, 0, 100), (1, 50, 150))
        asn = assignments(t0="harri")
        result = find_assignment_conflicts(CAM, [1], "harri", sp, asn)
        assert result == [(CAM, 0)]

    def test_partial_overlap_from_right(self):
        # existing: 100–200; new: 50–120 → overlap at 100–120
        sp = spans((0, 100, 200), (1, 50, 120))
        asn = assignments(t0="harri")
        result = find_assignment_conflicts(CAM, [1], "harri", sp, asn)
        assert result == [(CAM, 0)]

    def test_new_contains_existing(self):
        # existing: 30–70; new: 0–100 — existing fully inside new
        sp = spans((0, 30, 70), (1, 0, 100))
        asn = assignments(t0="harri")
        result = find_assignment_conflicts(CAM, [1], "harri", sp, asn)
        assert result == [(CAM, 0)]

    def test_existing_contains_new(self):
        # existing: 0–100; new: 30–70 — new fully inside existing
        sp = spans((0, 0, 100), (1, 30, 70))
        asn = assignments(t0="harri")
        result = find_assignment_conflicts(CAM, [1], "harri", sp, asn)
        assert result == [(CAM, 0)]

    def test_multiple_conflicts(self):
        sp = spans((0, 0, 100), (1, 50, 150), (2, 80, 200))
        asn = {(CAM, 0): "harri", (CAM, 1): "timo", (CAM, 2): "harri"}
        result = find_assignment_conflicts(CAM, [1], "harri", sp, asn)
        assert sorted(result) == sorted([(CAM, 0), (CAM, 2)])

    def test_no_conflict_for_tracks_in_tids(self):
        # track 0 is in the tids list — should not be reported as a conflict with itself
        sp = spans((0, 0, 100), (1, 0, 100))
        asn = assignments(t0="harri")
        # assigning [0, 1] to harri; t0 is already harri but is in tids, so no conflict
        result = find_assignment_conflicts(CAM, [0, 1], "harri", sp, asn)
        assert result == []

    def test_different_camera_not_a_conflict(self):
        sp = {(CAM, 0): (0, 100), (CAM, 1): (50, 150), (CAM2, 0): (0, 200)}
        asn = {(CAM2, 0): "harri"}  # same person but different camera
        result = find_assignment_conflicts(CAM, [1], "harri", sp, asn)
        assert result == []

    def test_batch_tids_multiple_conflicts(self):
        # Assigning [1, 2] to harri; track 0 overlaps track 1; track 3 overlaps track 2
        sp = spans((0, 0, 80), (1, 60, 150), (2, 200, 300), (3, 250, 400))
        asn = {(CAM, 0): "harri", (CAM, 3): "harri"}
        result = find_assignment_conflicts(CAM, [1, 2], "harri", sp, asn)
        assert sorted(result) == sorted([(CAM, 0), (CAM, 3)])

    def test_empty_tids(self):
        sp = spans((0, 0, 100))
        asn = assignments(t0="harri")
        assert find_assignment_conflicts(CAM, [], "harri", sp, asn) == []


# ---------------------------------------------------------------------------
# tracks_from_here_onwards
# ---------------------------------------------------------------------------

class TestTracksFromHereOnwards:

    def test_single_track_no_others(self):
        sp = spans((0, 0, 100))
        assert tracks_from_here_onwards(CAM, 0, sp, {}) == [0]

    def test_all_subsequent_unassigned(self):
        sp = spans((0, 0, 100), (1, 100, 200), (2, 200, 300))
        result = tracks_from_here_onwards(CAM, 0, sp, {})
        assert result == [0, 1, 2]

    def test_earlier_tracks_excluded(self):
        sp = spans((0, 0, 100), (1, 50, 150), (2, 200, 300))
        # selecting track 1 (starts at 50): track 0 starts at 0 < 50, excluded
        result = tracks_from_here_onwards(CAM, 1, sp, {})
        assert 0 not in result
        assert 1 in result
        assert 2 in result

    def test_already_assigned_tracks_excluded(self):
        sp = spans((0, 0, 100), (1, 100, 200), (2, 200, 300))
        asn = assignments(t1="timo")  # track 1 already assigned
        result = tracks_from_here_onwards(CAM, 0, sp, asn)
        assert result == [0, 2]  # track 1 skipped

    def test_primary_tid_always_included(self):
        # Primary track is already assigned to someone — still included
        sp = spans((0, 0, 100), (1, 100, 200))
        asn = assignments(t0="timo")
        result = tracks_from_here_onwards(CAM, 0, sp, asn)
        assert 0 in result

    def test_different_camera_excluded(self):
        sp = {(CAM, 0): (0, 100), (CAM2, 1): (100, 200)}
        result = tracks_from_here_onwards(CAM, 0, sp, {})
        assert (CAM2, 1) not in [(CAM, t) for t in result]
        assert result == [0]

    def test_result_ordered_by_first_frame(self):
        # tracks with different start times should come in time order
        sp = spans((0, 0, 50), (1, 300, 400), (2, 100, 200), (3, 200, 300))
        result = tracks_from_here_onwards(CAM, 0, sp, {})
        assert result == [0, 2, 3, 1]

    def test_same_start_frame_not_included(self):
        # Track 1 starts at the same frame as the selected track — excluded from expansion.
        # Two tracks starting simultaneously represent different people; auto-assigning
        # both to the same person would be wrong.
        sp = spans((0, 100, 200), (1, 100, 300))
        result = tracks_from_here_onwards(CAM, 0, sp, {})
        assert 1 not in result

    def test_all_tracks_start_at_zero(self):
        # Regression: when every track starts at frame 0, "from here onwards" should
        # return only the selected track (nothing starts strictly after it).
        sp = spans((0, 0, 100), (1, 0, 80), (2, 0, 120))
        result = tracks_from_here_onwards(CAM, 0, sp, {})
        assert result == [0]

    def test_typical_fragmented_track_scenario(self):
        # Two people from frame 0 (tracks 0 and 1); person A reappears as track 2 at frame 3000.
        # From-here-onwards on track 0 should pick up track 2 but NOT track 1.
        sp = spans((0, 0, 5000), (1, 0, 4000), (2, 3000, 6000))
        result = tracks_from_here_onwards(CAM, 0, sp, {})
        assert 0 in result
        assert 2 in result
        assert 1 not in result
