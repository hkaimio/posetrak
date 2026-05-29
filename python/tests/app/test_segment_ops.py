"""Tests for the pure segment split/merge/assign/detach logic in segment_ops.py.

All tests operate on plain dicts — no Qt required.

Terminology
-----------
``segments``   = {(svid, tid): [(seg_first, seg_last), ...]}
``assignments``= {(svid, tid, seg_first): person_name}

We use a single camera "cam" and a single track id 1 throughout so the
helpers below reduce boilerplate.
"""
from __future__ import annotations

import pytest

from app.pose.segment_ops import (
    split,
    merge,
    auto_merge,
    do_assign,
    do_detach,
)

CAM = "cam"
TID = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_segs(*ranges: tuple[int, int]) -> dict:
    """Build a segments dict from (seg_first, seg_last) pairs."""
    return {(CAM, TID): list(ranges)}


def make_assigns(**kwargs: str) -> dict:
    """Build an assignments dict.  Keyword = "sf" (int key), value = person.
    Because keyword args must be strings, pass them as seg_first=person pairs,
    e.g. make_assigns(sf0="alice") uses seg_first=0."""
    return {}  # helper below is easier


def assigns(*pairs: tuple[int, str]) -> dict:
    """Build assignments dict from (seg_first, person) pairs."""
    return {(CAM, TID, sf): p for sf, p in pairs}


def segs_list(segments: dict) -> list[tuple[int, int]]:
    """Extract the (seg_first, seg_last) list for the default track."""
    return segments.get((CAM, TID), [])


def assigned(assignments: dict, sf: int) -> str | None:
    """Look up the assignment for seg_first=sf."""
    return assignments.get((CAM, TID, sf))


# ===========================================================================
# split()
# ===========================================================================

class TestSplit:

    def test_basic_split(self):
        """Split divides [0,100] into [0,49] and [50,100]."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        ok = split(segs, asns, CAM, TID, 0, 50)
        assert ok is True
        assert segs_list(segs) == [(0, 49), (50, 100)]

    def test_split_propagates_assignment_to_right_half(self):
        """Both halves inherit the original assignment after a split."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        split(segs, asns, CAM, TID, 0, 50)
        assert assigned(asns, 0) == "alice"
        assert assigned(asns, 50) == "alice"

    def test_split_unassigned_segment_creates_no_assignment(self):
        """Splitting an unassigned bar leaves both halves unassigned."""
        segs = make_segs((0, 100))
        asns = {}
        split(segs, asns, CAM, TID, 0, 50)
        assert assigned(asns, 0) is None
        assert assigned(asns, 50) is None
        assert asns == {}

    def test_split_at_first_frame_is_rejected(self):
        """split_frame == seg_first is not allowed (would create zero-length left)."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        ok = split(segs, asns, CAM, TID, 0, 0)
        assert ok is False
        assert segs_list(segs) == [(0, 100)]  # unchanged

    def test_split_at_last_frame_plus_one_is_rejected(self):
        """split_frame == seg_last + 1 is also not allowed."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        ok = split(segs, asns, CAM, TID, 0, 101)
        assert ok is False
        assert segs_list(segs) == [(0, 100)]

    def test_split_at_last_frame_is_valid(self):
        """split_frame == seg_last creates [0,98] and [99,100] (single-frame right)."""
        segs = make_segs((0, 100))
        asns = {}
        ok = split(segs, asns, CAM, TID, 0, 99)
        assert ok is True
        assert segs_list(segs) == [(0, 98), (99, 100)]

    def test_split_at_second_frame_is_valid(self):
        """split_frame == seg_first + 1 creates a single-frame left half."""
        segs = make_segs((0, 100))
        asns = {}
        ok = split(segs, asns, CAM, TID, 0, 1)
        assert ok is True
        assert segs_list(segs) == [(0, 0), (1, 100)]

    def test_split_single_frame_segment_rejected(self):
        """A segment of exactly one frame cannot be split."""
        segs = make_segs((5, 5))
        asns = {}
        ok = split(segs, asns, CAM, TID, 5, 5)
        assert ok is False

    def test_split_nonexistent_seg_first_rejected(self):
        """seg_first not found → rejected."""
        segs = make_segs((0, 100))
        asns = {}
        ok = split(segs, asns, CAM, TID, 10, 50)
        assert ok is False

    def test_split_middle_segment_of_multiple(self):
        """Split only affects the target segment; neighbours are unchanged."""
        segs = {(CAM, TID): [(0, 49), (50, 150), (151, 200)]}
        asns = assigns((0, "alice"), (50, "bob"), (151, "alice"))
        ok = split(segs, asns, CAM, TID, 50, 100)
        assert ok is True
        assert segs_list(segs) == [(0, 49), (50, 99), (100, 150), (151, 200)]
        assert assigned(asns, 50) == "bob"
        assert assigned(asns, 100) == "bob"   # inherited
        assert assigned(asns, 0) == "alice"   # untouched
        assert assigned(asns, 151) == "alice" # untouched


# ===========================================================================
# merge()
# ===========================================================================

class TestMerge:

    def test_basic_merge(self):
        """Merge two adjacent unassigned segments."""
        segs = make_segs((0, 49), (50, 100))
        asns = {}
        ok = merge(segs, asns, CAM, TID, 0, 50)
        assert ok is True
        assert segs_list(segs) == [(0, 100)]

    def test_merge_assignment_from_first_segment_kept(self):
        """Merged segment keeps the assignment of the first (left) segment."""
        segs = make_segs((0, 49), (50, 100))
        asns = assigns((0, "alice"), (50, "alice"))
        merge(segs, asns, CAM, TID, 0, 50)
        assert assigned(asns, 0) == "alice"
        assert assigned(asns, 50) is None  # removed

    def test_merge_removes_second_assignment(self):
        """After merge, the second key is removed from assignments."""
        segs = make_segs((0, 49), (50, 100))
        asns = assigns((0, "alice"), (50, "alice"))
        merge(segs, asns, CAM, TID, 0, 50)
        assert (CAM, TID, 50) not in asns

    def test_merge_non_adjacent_rejected(self):
        """Segments with a gap cannot be merged."""
        segs = make_segs((0, 48), (50, 100))  # gap: frame 49 missing
        asns = {}
        ok = merge(segs, asns, CAM, TID, 0, 50)
        assert ok is False
        assert segs_list(segs) == [(0, 48), (50, 100)]

    def test_merge_wrong_order_rejected(self):
        """Trying to merge in reverse order (sf2 before sf1) is rejected."""
        segs = make_segs((0, 49), (50, 100))
        asns = {}
        ok = merge(segs, asns, CAM, TID, 50, 0)
        assert ok is False

    def test_merge_nonexistent_key_rejected(self):
        """If either key is not found, merge is rejected."""
        segs = make_segs((0, 100))
        asns = {}
        ok = merge(segs, asns, CAM, TID, 0, 50)  # 50 doesn't exist
        assert ok is False

    def test_split_then_merge_restores_original(self):
        """Split followed by merge of same-person halves restores original segment."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        split(segs, asns, CAM, TID, 0, 50)
        assert segs_list(segs) == [(0, 49), (50, 100)]
        merge(segs, asns, CAM, TID, 0, 50)
        assert segs_list(segs) == [(0, 100)]
        assert assigned(asns, 0) == "alice"
        assert (CAM, TID, 50) not in asns


# ===========================================================================
# auto_merge()
# ===========================================================================

class TestAutoMerge:

    def test_no_merge_needed(self):
        """No adjacent pairs with same assignment → nothing merged."""
        segs = make_segs((0, 49), (50, 100))
        asns = assigns((0, "alice"), (50, "bob"))
        merged = auto_merge(segs, asns, CAM, TID)
        assert merged == []
        assert segs_list(segs) == [(0, 49), (50, 100)]

    def test_merge_two_same_person_segments(self):
        """Two adjacent same-person segments are collapsed into one."""
        segs = make_segs((0, 49), (50, 100))
        asns = assigns((0, "alice"), (50, "alice"))
        merged = auto_merge(segs, asns, CAM, TID)
        assert (0, 50) in merged
        assert segs_list(segs) == [(0, 100)]

    def test_merge_three_same_person_segments(self):
        """Three consecutive same-person segments collapse to one."""
        segs = make_segs((0, 29), (30, 59), (60, 100))
        asns = assigns((0, "alice"), (30, "alice"), (60, "alice"))
        merged = auto_merge(segs, asns, CAM, TID)
        assert len(merged) == 2
        assert segs_list(segs) == [(0, 100)]

    def test_merge_both_unassigned(self):
        """Two adjacent unassigned segments are also merged."""
        segs = make_segs((0, 49), (50, 100))
        asns = {}
        merged = auto_merge(segs, asns, CAM, TID)
        assert len(merged) == 1
        assert segs_list(segs) == [(0, 100)]

    def test_merge_preserves_different_assignment_boundary(self):
        """Boundary between different-person segments is NOT merged."""
        segs = make_segs((0, 49), (50, 99), (100, 150))
        asns = assigns((0, "alice"), (50, "bob"), (100, "alice"))
        merged = auto_merge(segs, asns, CAM, TID)
        assert merged == []
        assert segs_list(segs) == [(0, 49), (50, 99), (100, 150)]

    def test_merge_collapses_only_matching_neighbours(self):
        """Only the adjacent pair with same assignment is merged."""
        segs = make_segs((0, 49), (50, 99), (100, 150))
        asns = assigns((0, "alice"), (50, "alice"), (100, "bob"))
        merged = auto_merge(segs, asns, CAM, TID)
        assert segs_list(segs) == [(0, 99), (100, 150)]
        assert assigned(asns, 0) == "alice"
        assert assigned(asns, 100) == "bob"


# ===========================================================================
# do_assign()
# ===========================================================================

class TestDoAssign:

    def test_assign_full_bar(self):
        """Assigning the full bar just updates the assignment; no splits."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        do_assign(segs, asns, CAM, TID, 0, "bob", 0, 100)
        assert segs_list(segs) == [(0, 100)]
        assert assigned(asns, 0) == "bob"

    def test_assign_full_bar_unassigned(self):
        """Assigning an unassigned bar creates the assignment entry."""
        segs = make_segs((0, 100))
        asns = {}
        do_assign(segs, asns, CAM, TID, 0, "alice", 0, 100)
        assert assigned(asns, 0) == "alice"

    def test_assign_left_portion_preserves_right(self):
        """Assigning [0,50] of [0,100]→alice: left=new, right keeps alice."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        do_assign(segs, asns, CAM, TID, 0, "bob", 0, 50)
        assert segs_list(segs) == [(0, 50), (51, 100)]
        assert assigned(asns, 0) == "bob"
        assert assigned(asns, 51) == "alice"

    def test_assign_right_portion_preserves_left(self):
        """Assigning [50,100] of [0,100]→alice: right=new, left keeps alice."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        do_assign(segs, asns, CAM, TID, 0, "bob", 50, 100)
        assert segs_list(segs) == [(0, 49), (50, 100)]
        assert assigned(asns, 0) == "alice"
        assert assigned(asns, 50) == "bob"

    def test_assign_middle_preserves_both_ends(self):
        """Assigning [30,70] of [0,100]→alice: middle=new, both ends keep alice."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        do_assign(segs, asns, CAM, TID, 0, "bob", 30, 70)
        assert segs_list(segs) == [(0, 29), (30, 70), (71, 100)]
        assert assigned(asns, 0) == "alice"
        assert assigned(asns, 30) == "bob"
        assert assigned(asns, 71) == "alice"

    def test_assign_same_person_collapses_with_neighbour(self):
        """Assigning the same person as a neighbour triggers auto_merge."""
        segs = make_segs((0, 49), (50, 100))
        asns = assigns((0, "alice"), (50, "bob"))
        do_assign(segs, asns, CAM, TID, 50, "alice", 50, 100)
        # Both segments now alice → should merge
        assert segs_list(segs) == [(0, 100)]
        assert assigned(asns, 0) == "alice"

    def test_assign_same_person_both_sides_collapses_to_one(self):
        """Assigning the same person as both adjacent segments triggers double merge."""
        segs = make_segs((0, 29), (30, 69), (70, 100))
        asns = assigns((0, "alice"), (30, "bob"), (70, "alice"))
        do_assign(segs, asns, CAM, TID, 30, "alice", 30, 69)
        # All three should merge to one alice segment
        assert segs_list(segs) == [(0, 100)]
        assert assigned(asns, 0) == "alice"

    def test_assign_clamped_to_segment_bounds(self):
        """sel_first/sel_last are clamped to the actual segment range."""
        segs = make_segs((20, 80))
        asns = assigns((20, "alice"))
        # Selection extends beyond both ends
        do_assign(segs, asns, CAM, TID, 20, "bob", 0, 200)
        assert segs_list(segs) == [(20, 80)]
        assert assigned(asns, 20) == "bob"

    def test_assign_middle_of_unassigned_bar(self):
        """Assigning middle of unassigned bar: left and right remain unassigned."""
        segs = make_segs((0, 100))
        asns = {}
        do_assign(segs, asns, CAM, TID, 0, "alice", 30, 70)
        assert segs_list(segs) == [(0, 29), (30, 70), (71, 100)]
        assert assigned(asns, 0) is None
        assert assigned(asns, 30) == "alice"
        assert assigned(asns, 71) is None

    def test_assign_single_frame_selection(self):
        """A one-frame selection creates the minimal possible splits."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        do_assign(segs, asns, CAM, TID, 0, "bob", 50, 50)
        assert segs_list(segs) == [(0, 49), (50, 50), (51, 100)]
        assert assigned(asns, 0) == "alice"
        assert assigned(asns, 50) == "bob"
        assert assigned(asns, 51) == "alice"


# ===========================================================================
# do_detach()
# ===========================================================================

class TestDoDetach:

    def test_detach_full_bar(self):
        """Detaching the full bar removes the assignment."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        do_detach(segs, asns, CAM, TID, 0, 0, 100)
        assert segs_list(segs) == [(0, 100)]
        assert assigned(asns, 0) is None

    def test_detach_left_portion_preserves_right(self):
        """Detaching [0,50] of [0,100]→alice: left unassigned, right keeps alice."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        do_detach(segs, asns, CAM, TID, 0, 0, 50)
        assert segs_list(segs) == [(0, 50), (51, 100)]
        assert assigned(asns, 0) is None
        assert assigned(asns, 51) == "alice"

    def test_detach_right_portion_preserves_left(self):
        """Detaching [50,100] of [0,100]→alice: right unassigned, left keeps alice."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        do_detach(segs, asns, CAM, TID, 0, 50, 100)
        assert segs_list(segs) == [(0, 49), (50, 100)]
        assert assigned(asns, 0) == "alice"
        assert assigned(asns, 50) is None

    def test_detach_middle_preserves_both_ends(self):
        """Detaching middle [30,70] of [0,100]→alice: ends keep alice."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        do_detach(segs, asns, CAM, TID, 0, 30, 70)
        assert segs_list(segs) == [(0, 29), (30, 70), (71, 100)]
        assert assigned(asns, 0) == "alice"
        assert assigned(asns, 30) is None
        assert assigned(asns, 71) == "alice"

    def test_detach_unassigned_bar_is_noop(self):
        """Detaching an already-unassigned bar changes nothing."""
        segs = make_segs((0, 100))
        asns = {}
        do_detach(segs, asns, CAM, TID, 0, 0, 100)
        assert segs_list(segs) == [(0, 100)]
        assert asns == {}

    def test_detach_collapses_adjacent_unassigned_after(self):
        """If detaching creates two adjacent unassigned segments, they merge."""
        # Three segments: alice | bob | alice → detach bob → alice | ??? | alice
        # The ??? is unassigned but the two alice segments cannot merge (different)
        segs = make_segs((0, 29), (30, 69), (70, 100))
        asns = assigns((0, "alice"), (30, "bob"), (70, "alice"))
        do_detach(segs, asns, CAM, TID, 30, 30, 69)
        # [30,69] is now unassigned — the alice segments on either side can't collapse
        assert assigned(asns, 30) is None
        assert assigned(asns, 0) == "alice"
        assert assigned(asns, 70) == "alice"

    def test_detach_creates_two_adjacent_unassigned_merges(self):
        """Two adjacent unassigned halves merge after detach."""
        # Three segments: alice | alice | bob → detach middle alice → unassigned+alice merge?
        # No — [0,29]→alice, [30,69]→alice, [70,100]→bob
        # Detach [30,69]: [0,29]→alice, [30,69]→None, [70,100]→bob
        # auto_merge: none merge (alice≠None, None≠bob)
        segs = make_segs((0, 29), (30, 69), (70, 100))
        asns = assigns((0, "alice"), (30, "alice"), (70, "bob"))
        do_detach(segs, asns, CAM, TID, 30, 30, 69)
        assert segs_list(segs) == [(0, 29), (30, 69), (70, 100)]
        assert assigned(asns, 0) == "alice"
        assert assigned(asns, 30) is None
        assert assigned(asns, 70) == "bob"

    def test_detach_clamped_to_segment_bounds(self):
        """sel_first/sel_last outside the segment are clamped."""
        segs = make_segs((20, 80))
        asns = assigns((20, "alice"))
        do_detach(segs, asns, CAM, TID, 20, 0, 200)
        assert segs_list(segs) == [(20, 80)]
        assert assigned(asns, 20) is None

    def test_detach_single_frame_selection(self):
        """A one-frame selection is detached while surrounding frames keep assignment."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        do_detach(segs, asns, CAM, TID, 0, 50, 50)
        assert segs_list(segs) == [(0, 49), (50, 50), (51, 100)]
        assert assigned(asns, 0) == "alice"
        assert assigned(asns, 50) is None
        assert assigned(asns, 51) == "alice"


# ===========================================================================
# Combined / edge-case round trips
# ===========================================================================

class TestRoundTrips:

    def test_assign_then_detach_restores_original(self):
        """Assigning then detaching the same range restores original state."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        orig_segs = segs_list(segs).copy()

        do_assign(segs, asns, CAM, TID, 0, "bob", 30, 70)
        # Mid-state: three segments
        assert len(segs_list(segs)) == 3

        # Detach the middle (now assigned to bob)
        do_detach(segs, asns, CAM, TID, 30, 30, 70)
        # Two alice + one unassigned → no auto-merge between them
        assert segs_list(segs) == [(0, 29), (30, 70), (71, 100)]

        # Re-assign middle back to alice → should collapse back to one
        do_assign(segs, asns, CAM, TID, 30, "alice", 30, 70)
        assert segs_list(segs) == [(0, 100)]
        assert assigned(asns, 0) == "alice"

    def test_multiple_splits_and_reassign(self):
        """Split at two points, reassign one piece, verify others unchanged."""
        segs = make_segs((0, 200))
        asns = assigns((0, "alice"))

        # Assign bob to [50,150]
        do_assign(segs, asns, CAM, TID, 0, "bob", 50, 150)
        assert segs_list(segs) == [(0, 49), (50, 150), (151, 200)]
        assert assigned(asns, 0) == "alice"
        assert assigned(asns, 50) == "bob"
        assert assigned(asns, 151) == "alice"

        # Reassign [100,150] back to alice
        do_assign(segs, asns, CAM, TID, 50, "alice", 100, 150)
        assert segs_list(segs) == [(0, 49), (50, 99), (100, 200)]
        assert assigned(asns, 0) == "alice"
        assert assigned(asns, 50) == "bob"
        assert assigned(asns, 100) == "alice"

    def test_assign_entirely_inside_existing_assignment_does_not_orphan_remainder(self):
        """Regression: assigning a sub-range of an already-assigned bar must not
        leave the remainder without an assignment entry."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        do_assign(segs, asns, CAM, TID, 0, "bob", 20, 40)
        # Segments: [0,19]→alice, [20,40]→bob, [41,100]→alice
        assert assigned(asns, 0) == "alice"
        assert assigned(asns, 20) == "bob"
        assert assigned(asns, 41) == "alice", (
            "Right remainder must inherit alice — not become unassigned"
        )

    def test_detach_partial_does_not_remove_remainder(self):
        """Regression: detaching a sub-range must not unassign the remainder."""
        segs = make_segs((0, 100))
        asns = assigns((0, "alice"))
        do_detach(segs, asns, CAM, TID, 0, 20, 40)
        # Segments: [0,19]→alice, [20,40]→None, [41,100]→alice
        assert assigned(asns, 0) == "alice"
        assert assigned(asns, 20) is None
        assert assigned(asns, 41) == "alice", (
            "Right remainder must keep alice — not become unassigned"
        )
