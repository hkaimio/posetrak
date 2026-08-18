# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for track segment assignment conflict detection.

Each detection track can be split into segments identified by a 3-tuple key
(svid, tid, seg_first).  These tests cover the pure conflict-detection logic
in assignment.py using that key model.
"""
from __future__ import annotations

from app.pose.assignment import find_assignment_conflicts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CAM = "cam-a"
CAM2 = "cam-b"


def seg_spans(*args: tuple[int, int, int]) -> dict:
    """Build a segment spans dict from (tid, seg_first, seg_last) triples, all in CAM.

    Returns {(CAM, tid, seg_first): (seg_first, seg_last)}.
    """
    return {(CAM, tid, sf): (sf, sl) for tid, sf, sl in args}


def seg_assign(*args: tuple[int, int, str]) -> dict:
    """Build a segment assignment dict from (tid, seg_first, person) triples, all in CAM.

    Returns {(CAM, tid, seg_first): person}.
    """
    return {(CAM, tid, sf): p for tid, sf, p in args}


def key(tid: int, sf: int) -> tuple:
    """Shorthand for (CAM, tid, seg_first)."""
    return (CAM, tid, sf)


# ---------------------------------------------------------------------------
# find_assignment_conflicts
# ---------------------------------------------------------------------------

class TestFindConflicts:

    def test_no_existing_assignments(self):
        """No existing assignments → no conflicts."""
        sp = seg_spans((0, 0, 100), (1, 50, 150))
        assert find_assignment_conflicts(CAM, [key(1, 50)], "harri", sp, {}) == []

    def test_no_overlap_different_persons(self):
        """Overlap in frames but different person → not a conflict."""
        sp = seg_spans((0, 0, 100), (1, 50, 150))
        asn = seg_assign((0, 0, "timo"))
        assert find_assignment_conflicts(CAM, [key(1, 50)], "harri", sp, asn) == []

    def test_no_overlap_adjacent_segments(self):
        """Adjacent segments (one ends where the other begins) are not a conflict."""
        sp = seg_spans((0, 0, 100), (1, 100, 200))
        asn = seg_assign((0, 0, "harri"))
        assert find_assignment_conflicts(CAM, [key(1, 100)], "harri", sp, asn) == []

    def test_partial_overlap_from_left(self):
        """Existing ends inside new → overlap at frames 50–100."""
        sp = seg_spans((0, 0, 100), (1, 50, 150))
        asn = seg_assign((0, 0, "harri"))
        result = find_assignment_conflicts(CAM, [key(1, 50)], "harri", sp, asn)
        assert result == [key(0, 0)]

    def test_partial_overlap_from_right(self):
        """Existing starts inside new → overlap at frames 100–120."""
        sp = seg_spans((0, 100, 200), (1, 50, 120))
        asn = seg_assign((0, 100, "harri"))
        result = find_assignment_conflicts(CAM, [key(1, 50)], "harri", sp, asn)
        assert result == [key(0, 100)]

    def test_new_contains_existing(self):
        """Existing fully inside new → overlap."""
        sp = seg_spans((0, 30, 70), (1, 0, 100))
        asn = seg_assign((0, 30, "harri"))
        result = find_assignment_conflicts(CAM, [key(1, 0)], "harri", sp, asn)
        assert result == [key(0, 30)]

    def test_existing_contains_new(self):
        """New fully inside existing → overlap."""
        sp = seg_spans((0, 0, 100), (1, 30, 70))
        asn = seg_assign((0, 0, "harri"))
        result = find_assignment_conflicts(CAM, [key(1, 30)], "harri", sp, asn)
        assert result == [key(0, 0)]

    def test_multiple_conflicts(self):
        """Multiple existing segments of same person overlap the new segment."""
        sp = seg_spans((0, 0, 100), (1, 50, 150), (2, 80, 200))
        asn = {key(0, 0): "harri", key(1, 50): "timo", key(2, 80): "harri"}
        result = find_assignment_conflicts(CAM, [key(1, 50)], "harri", sp, asn)
        assert sorted(result) == sorted([key(0, 0), key(2, 80)])

    def test_no_conflict_for_segments_in_new_list(self):
        """Segments already in new_segments are excluded from conflict reporting."""
        sp = seg_spans((0, 0, 100), (1, 0, 100))
        asn = seg_assign((0, 0, "harri"))
        # Assigning both segments to harri; seg (0,0) overlaps but is in new_segments
        result = find_assignment_conflicts(CAM, [key(0, 0), key(1, 0)], "harri", sp, asn)
        assert result == []

    def test_different_camera_not_a_conflict(self):
        """Segments in other cameras are never reported as conflicts."""
        sp = {key(0, 0): (0, 100), key(1, 50): (50, 150), (CAM2, 0, 0): (0, 200)}
        asn = {(CAM2, 0, 0): "harri"}
        result = find_assignment_conflicts(CAM, [key(1, 50)], "harri", sp, asn)
        assert result == []

    def test_empty_new_segments(self):
        """Empty new_segments list → no conflicts."""
        sp = seg_spans((0, 0, 100))
        asn = seg_assign((0, 0, "harri"))
        assert find_assignment_conflicts(CAM, [], "harri", sp, asn) == []

    # ---------------------------------------------------------------------------
    # Split-specific scenarios
    # ---------------------------------------------------------------------------

    def test_split_halves_do_not_conflict_with_each_other(self):
        """After splitting, the two halves are adjacent and must not conflict."""
        # Track 0 split at frame 50: left [0,49], right [50,100]
        sp = seg_spans((0, 0, 49), (0, 50, 100))
        asn = seg_assign((0, 0, "harri"))  # left half assigned to harri
        # Assigning the right half to harri too → no conflict (adjacent, not overlapping)
        result = find_assignment_conflicts(CAM, [key(0, 50)], "harri", sp, asn)
        assert result == []

    def test_split_halves_can_have_different_persons(self):
        """Right half assigned to a different person → no conflict at all."""
        sp = seg_spans((0, 0, 49), (0, 50, 100))
        asn = seg_assign((0, 0, "harri"))
        result = find_assignment_conflicts(CAM, [key(0, 50)], "timo", sp, asn)
        assert result == []

    def test_split_right_half_conflicts_with_other_detection(self):
        """Assigning the right half to person P conflicts with another detection of P that overlaps."""
        # Detection 0 split: right half is [50, 100].  Detection 1 covers [30, 70] and is P's.
        sp = {key(0, 50): (50, 100), key(1, 30): (30, 70)}
        asn = seg_assign((1, 30, "harri"))
        result = find_assignment_conflicts(CAM, [key(0, 50)], "harri", sp, asn)
        assert result == [key(1, 30)]

    def test_multi_split_no_cross_contamination(self):
        """Three segments of the same track can each be assigned to a different person
        without triggering conflicts with each other."""
        # Track 0 split into three parts: [0,29], [30,59], [60,100]
        sp = seg_spans((0, 0, 29), (0, 30, 59), (0, 60, 100))
        asn = {key(0, 0): "alice", key(0, 30): "bob"}
        # Assigning third segment to alice — only conflict could be seg [0,29] (alice, adjacent)
        result = find_assignment_conflicts(CAM, [key(0, 60)], "alice", sp, asn)
        # [0,29] ends at 29, [60,100] starts at 60 — no overlap
        assert result == []
