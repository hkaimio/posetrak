# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""assignment.py — Pure functions for track-to-person assignment logic.

Kept free of Qt imports so they can be unit-tested without a display.

Type aliases
------------
SegKey   : (svid, tid, seg_first_frame)  — uniquely identifies one segment
SegSpans : {SegKey: (first_frame, last_frame)}  — integer video frames
SegAssign: {SegKey: person_name}
"""
from __future__ import annotations

SegKey = tuple[str, int, int]
SegSpans = dict[SegKey, tuple[int, int]]
SegAssign = dict[SegKey, str]


def find_assignment_conflicts(
    svid: str,
    new_segments: list[SegKey],
    person_name: str,
    spans: SegSpans,
    assignments: SegAssign,
) -> list[SegKey]:
    """Return segments already assigned to *person_name* that frame-overlap any segment in *new_segments*.

    Overlap is ``max(start_a, start_b) < min(end_a, end_b)`` on integer frame
    numbers.  Adjacent segments (one ends exactly where the other begins) are
    **not** considered a conflict.  Only segments in the same camera (*svid*)
    are checked.

    Parameters
    ----------
    svid:
        Camera to check within.
    new_segments:
        ``(svid, tid, seg_first)`` keys for the segments being assigned.
        These are excluded from the conflict search.
    person_name:
        The person being assigned.
    spans:
        Segment-keyed frame data: ``{(svid, tid, seg_first): (first, last)}``.
    assignments:
        Current segment assignment state ``{(svid, tid, seg_first): person_name}``.

    Returns
    -------
    list[SegKey]
        Keys for every already-assigned segment that overlaps.
    """
    new_ranges = [spans[k] for k in new_segments if k in spans]
    if not new_ranges:
        return []

    new_set = set(new_segments)
    conflicts: list[SegKey] = []
    for key, name in assignments.items():
        s, _t, _sf = key
        if s != svid or key in new_set or name != person_name:
            continue
        other = spans.get(key)
        if other is None:
            continue
        other_first, other_last = other
        for new_first, new_last in new_ranges:
            if max(new_first, other_first) < min(new_last, other_last):
                conflicts.append(key)
                break
    return conflicts
