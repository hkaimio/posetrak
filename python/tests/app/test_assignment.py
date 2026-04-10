"""Tests for track assignment conflict detection and 'from here onwards' expansion."""
from __future__ import annotations

from app.pose.assignment import find_assignment_conflicts, tracks_from_here_onwards

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CAM = "cam-a"
CAM2 = "cam-b"


def spans(*args: tuple[int, int, int]) -> dict:
    """Build a frame-based spans dict from (tid, first, last) triples, all in CAM."""
    return {(CAM, tid): (first, last) for tid, first, last in args}


def time_spans(*args: tuple[int, float, float]) -> dict:
    """Build a time-based spans dict from (tid, t0_s, t1_s) triples, all in CAM."""
    return {(CAM, tid): (t0, t1) for tid, t0, t1 in args}


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
    """tracks_from_here_onwards uses global seconds (min_time_s) as the reference."""

    def test_single_track_no_others(self):
        """Only the selected track exists — result is just [tid]."""
        ts = time_spans((0, 0.0, 10.0))
        assert tracks_from_here_onwards(CAM, 0, ts, {}, min_time_s=0.0) == [0]

    def test_all_subsequent_unassigned(self):
        """Three sequential tracks; clicking at t=0 includes tracks starting after t=0."""
        ts = time_spans((0, 0.0, 10.0), (1, 10.0, 20.0), (2, 20.0, 30.0))
        result = tracks_from_here_onwards(CAM, 0, ts, {}, min_time_s=0.0)
        assert result == [0, 1, 2]

    def test_click_midpoint_excludes_earlier_tracks(self):
        """Clicking at t=15 should exclude track 0 (starts at t=0) and include track 2 (t=20)."""
        ts = time_spans((0, 0.0, 10.0), (1, 5.0, 15.0), (2, 20.0, 30.0))
        result = tracks_from_here_onwards(CAM, 1, ts, {}, min_time_s=15.0)
        assert 0 not in result
        assert 1 in result
        assert 2 in result

    def test_already_assigned_tracks_excluded(self):
        """Tracks already assigned to any person are skipped in the expansion."""
        ts = time_spans((0, 0.0, 10.0), (1, 10.0, 20.0), (2, 20.0, 30.0))
        asn = assignments(t1="timo")  # track 1 already assigned
        result = tracks_from_here_onwards(CAM, 0, ts, asn, min_time_s=0.0)
        assert result == [0, 2]  # track 1 skipped

    def test_primary_tid_always_included(self):
        """Primary track is always included even if already assigned to someone."""
        ts = time_spans((0, 0.0, 10.0), (1, 10.0, 20.0))
        asn = assignments(t0="timo")
        result = tracks_from_here_onwards(CAM, 0, ts, asn, min_time_s=0.0)
        assert 0 in result

    def test_different_camera_excluded(self):
        """Tracks in other cameras are never included."""
        ts = {(CAM, 0): (0.0, 10.0), (CAM2, 1): (10.0, 20.0)}
        result = tracks_from_here_onwards(CAM, 0, ts, {}, min_time_s=0.0)
        assert result == [0]

    def test_result_ordered_by_start_time(self):
        """Expansion tracks are returned in ascending start-time order."""
        ts = time_spans((0, 0.0, 5.0), (1, 30.0, 40.0), (2, 10.0, 20.0), (3, 20.0, 30.0))
        result = tracks_from_here_onwards(CAM, 0, ts, {}, min_time_s=0.0)
        assert result == [0, 2, 3, 1]

    def test_same_start_time_not_included(self):
        """Tracks starting at exactly min_time_s are excluded (strict >).

        Two tracks starting simultaneously represent different people; including
        both in the expansion would assign them to the same person.
        """
        ts = time_spans((0, 10.0, 20.0), (1, 10.0, 30.0))
        result = tracks_from_here_onwards(CAM, 0, ts, {}, min_time_s=10.0)
        assert 1 not in result

    def test_all_tracks_start_at_zero(self):
        """Regression: when every track starts at t=0, clicking at t=0 returns only the
        selected track — nothing starts strictly after t=0."""
        ts = time_spans((0, 0.0, 10.0), (1, 0.0, 8.0), (2, 0.0, 12.0))
        result = tracks_from_here_onwards(CAM, 0, ts, {}, min_time_s=0.0)
        assert result == [0]

    def test_click_in_middle_of_bar_uses_clicked_time(self):
        """Clicking at t=5 on a bar that starts at t=0 should only expand to tracks
        starting after t=5, not to all tracks starting after t=0."""
        # bar 0: t=0–30, bar 1: t=0–15 (simultaneous), bar 2: t=8–20 (starts after click)
        ts = time_spans((0, 0.0, 30.0), (1, 0.0, 15.0), (2, 8.0, 20.0))
        result = tracks_from_here_onwards(CAM, 0, ts, {}, min_time_s=5.0)
        assert 1 not in result   # starts at 0, not after 5
        assert 2 in result       # starts at 8 > 5

    def test_typical_fragmented_track_scenario(self):
        """Two people from t=0; person A reappears as track 2 at t=25.
        From-here-onwards on track 0 (clicked at t=0) picks up track 2 but not track 1."""
        ts = time_spans((0, 0.0, 50.0), (1, 0.0, 40.0), (2, 25.0, 60.0))
        result = tracks_from_here_onwards(CAM, 0, ts, {}, min_time_s=0.0)
        assert 0 in result
        assert 2 in result
        assert 1 not in result
