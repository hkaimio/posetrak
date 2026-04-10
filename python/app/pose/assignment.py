"""assignment.py — Pure functions for track-to-person assignment logic.

Kept free of Qt imports so they can be unit-tested without a display.
"""
from __future__ import annotations

# Type aliases
Spans = dict[tuple[str, int], tuple[int, int]]        # (svid, tid) -> (first, last)
Assignments = dict[tuple[str, int], str]               # (svid, tid) -> person_name


def tracks_from_here_onwards(
    svid: str,
    tid: int,
    spans: Spans,
    assignments: Assignments,
) -> list[int]:
    """Return [tid] + subsequent unassigned tracks in the same camera.

    "Subsequent" means the track's first_frame >= the selected track's first_frame.
    Already-assigned tracks (to any person) are excluded from the expansion.
    The primary *tid* is always included first, regardless of its assignment state.
    """
    first_frame = spans.get((svid, tid), (0, 0))[0]
    result = [tid]
    for (s, t), (ff, _lf) in sorted(spans.items(), key=lambda x: x[1][0]):
        # Use strict > so that tracks starting at the same frame as the selected track
        # (e.g. two simultaneous people both tracked from frame 0) are not included.
        if s == svid and t != tid and ff > first_frame and (svid, t) not in assignments:
            result.append(t)
    return result


def find_assignment_conflicts(
    svid: str,
    tids: list[int],
    person_name: str,
    spans: Spans,
    assignments: Assignments,
) -> list[tuple[str, int]]:
    """Return tracks already assigned to person_name that time-overlap any track in tids.

    Overlap condition: max(start_a, start_b) < min(end_a, end_b).
    Adjacent tracks (one ends exactly where the other begins) are NOT a conflict.
    Only tracks in the same camera (svid) are considered.
    """
    new_ranges = [spans[(svid, t)] for t in tids if (svid, t) in spans]
    if not new_ranges:
        return []

    conflicts: list[tuple[str, int]] = []
    for (s, t), name in assignments.items():
        if s != svid or t in tids or name != person_name:
            continue
        other = spans.get((s, t))
        if other is None:
            continue
        other_first, other_last = other
        for new_first, new_last in new_ranges:
            if max(new_first, other_first) < min(new_last, other_last):
                conflicts.append((s, t))
                break
    return conflicts
