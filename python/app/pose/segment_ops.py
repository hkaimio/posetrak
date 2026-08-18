# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""segment_ops.py — Pure-logic operations on segment/assignment dicts.

All functions operate on plain Python dicts so they can be tested without Qt.
``FilmstripStitcherWidget`` calls these to update its internal state and then
drives the QGraphics scene separately.  ``StitcherPanel`` also passes its own
``_assignments`` dict through these functions, keeping both dicts in sync.

Type aliases
------------
Segments   = dict[(svid, tid)         → [(seg_first, seg_last), ...]]
Assignments = dict[(svid, tid, seg_first) → person_name]
"""
from __future__ import annotations

# Type aliases (not enforced at runtime; here for documentation only)
Segments = dict   # {(svid, tid): [(seg_first, seg_last), ...]}
Assignments = dict  # {(svid, tid, seg_first): str}


def split(
    segments: Segments,
    assignments: Assignments,
    svid: str,
    tid: int,
    seg_first: int,
    split_frame: int,
) -> bool:
    """Split segment [seg_first, seg_last] at *split_frame*.

    Creates two new segments:
    - Left  [seg_first,      split_frame - 1]
    - Right [split_frame,    seg_last]

    **Both halves inherit the original assignment.**  The caller only needs to
    overwrite the target half with the new assignment; the other half keeps its
    correct value automatically.

    Returns True if the split was performed, False if rejected (split_frame is
    not strictly inside [seg_first+1, seg_last]).
    """
    key = (svid, tid)
    segs = segments.get(key)
    if segs is None:
        return False

    for i, (sf, sl) in enumerate(segs):
        if sf != seg_first:
            continue
        if not (sf < split_frame <= sl):
            return False  # split_frame not strictly inside the segment

        orig = assignments.get((svid, tid, sf))

        # Update segment list in-place
        segs[i] = (sf, split_frame - 1)
        segs.insert(i + 1, (split_frame, sl))

        # Right half inherits the original assignment
        if orig is not None:
            assignments[(svid, tid, split_frame)] = orig

        return True

    return False  # seg_first not found


def merge(
    segments: Segments,
    assignments: Assignments,
    svid: str,
    tid: int,
    sf1: int,
    sf2: int,
) -> bool:
    """Merge two adjacent segments (sf2 must equal sl1 + 1).

    The merged segment [sf1, sl2] keeps *sf1*'s assignment.
    The *sf2* assignment entry is removed from *assignments*.

    Returns True if merged, False if rejected.
    """
    key = (svid, tid)
    segs = segments.get(key)
    if segs is None:
        return False

    idx1 = next((i for i, (sf, _) in enumerate(segs) if sf == sf1), None)
    idx2 = next((i for i, (sf, _) in enumerate(segs) if sf == sf2), None)

    if idx1 is None or idx2 is None or idx2 != idx1 + 1:
        return False

    _, sl1 = segs[idx1]
    _, sl2 = segs[idx2]

    if sl1 + 1 != sf2:
        return False  # not adjacent

    segs[idx1] = (sf1, sl2)
    del segs[idx2]
    assignments.pop((svid, tid, sf2), None)
    return True


def auto_merge(
    segments: Segments,
    assignments: Assignments,
    svid: str,
    tid: int,
) -> list[tuple[int, int]]:
    """Collapse all adjacent segment pairs that share the same assignment.

    Mutates *segments* and *assignments* in-place.  Returns a list of
    (sf1, sf2) pairs that were merged, in the order they were processed.
    The caller uses this to update the visual layer.
    """
    merged: list[tuple[int, int]] = []
    changed = True
    while changed:
        changed = False
        segs = segments.get((svid, tid), [])
        for i in range(len(segs) - 1):
            sf1, sl1 = segs[i]
            sf2, _   = segs[i + 1]
            if sf2 != sl1 + 1:
                continue
            a1 = assignments.get((svid, tid, sf1))
            a2 = assignments.get((svid, tid, sf2))
            if a1 == a2:
                merge(segments, assignments, svid, tid, sf1, sf2)
                merged.append((sf1, sf2))
                changed = True
                break
    return merged


def do_assign(
    segments: Segments,
    assignments: Assignments,
    svid: str,
    tid: int,
    seg_first: int,
    person: str,
    sel_first: int,
    sel_last: int,
) -> list[tuple[int, int]]:
    """Assign *person* to the frame range [sel_first, sel_last].

    The range is clamped to [seg_first, seg_last].  Portions outside the
    selection keep their original assignment.

    Performs at most two splits (at sel_first and sel_last+1), assigns the
    middle segment, then runs auto_merge.

    Returns the list of (sf1, sf2) pairs that were auto-merged, in order.
    """
    segs = segments.get((svid, tid), [])
    seg_range = next((r for r in segs if r[0] == seg_first), None)
    if seg_range is None:
        return []
    cur_first, cur_last = seg_range

    sel_first = max(sel_first, cur_first)
    sel_last  = min(sel_last,  cur_last)

    # Left split: carve off [cur_first, sel_first-1]
    target_sf = seg_first
    if sel_first > cur_first:
        split(segments, assignments, svid, tid, cur_first, sel_first)
        target_sf = sel_first

    # Right split: carve off [sel_last+1, cur_last]
    segs_now = segments.get((svid, tid), [])
    target_range = next((r for r in segs_now if r[0] == target_sf), None)
    if target_range is not None and sel_last < target_range[1]:
        split(segments, assignments, svid, tid, target_sf, sel_last + 1)

    # Assign the target range
    assignments[(svid, tid, target_sf)] = person

    return auto_merge(segments, assignments, svid, tid)


def do_detach(
    segments: Segments,
    assignments: Assignments,
    svid: str,
    tid: int,
    seg_first: int,
    sel_first: int,
    sel_last: int,
) -> list[tuple[int, int]]:
    """Remove the assignment for the frame range [sel_first, sel_last].

    Portions outside the selection keep their original assignment.
    Runs auto_merge after detaching.

    Returns the list of (sf1, sf2) pairs that were auto-merged.
    """
    segs = segments.get((svid, tid), [])
    seg_range = next((r for r in segs if r[0] == seg_first), None)
    if seg_range is None:
        return []
    cur_first, cur_last = seg_range

    sel_first = max(sel_first, cur_first)
    sel_last  = min(sel_last,  cur_last)

    # Left split
    target_sf = seg_first
    if sel_first > cur_first:
        split(segments, assignments, svid, tid, cur_first, sel_first)
        target_sf = sel_first

    # Right split
    segs_now = segments.get((svid, tid), [])
    target_range = next((r for r in segs_now if r[0] == target_sf), None)
    if target_range is not None and sel_last < target_range[1]:
        split(segments, assignments, svid, tid, target_sf, sel_last + 1)

    # Remove the assignment for the target range
    assignments.pop((svid, tid, target_sf), None)

    return auto_merge(segments, assignments, svid, tid)
