"""assignment.py — Pure functions for track-to-person assignment logic.

Kept free of Qt imports so they can be unit-tested without a display.

Type aliases
------------
Spans       : {(svid, tid): (first_frame, last_frame)}  — integer video frames
TimeSpans   : {(svid, tid): (t0_s, t1_s)}               — global seconds
Assignments : {(svid, tid): person_name}
"""
from __future__ import annotations

# Type aliases (plain dicts; no runtime overhead)
Spans = dict[tuple[str, int], tuple[int, int]]
TimeSpans = dict[tuple[str, int], tuple[float, float]]
Assignments = dict[tuple[str, int], str]


def tracks_from_here_onwards(
    svid: str,
    tid: int,
    time_spans: TimeSpans,
    assignments: Assignments,
    min_time_s: float,
) -> list[int]:
    """Return the selected track plus unassigned tracks that start after *min_time_s*.

    Parameters
    ----------
    svid:
        Camera (shot_video_id) to search within.
    tid:
        The selected track ID; always included first in the result regardless of
        its own start time or assignment state.
    time_spans:
        Mapping of (svid, tid) to (t0_s, t1_s) in global seconds.  Typically
        obtained from ``StitcherWidget.get_time_spans()``.
    assignments:
        Current assignment state.  Tracks already assigned to *any* person are
        excluded from the expansion (the primary *tid* is still included).
    min_time_s:
        Reference timestamp in global seconds.  Only tracks whose start time is
        **strictly greater** than this value are included in the expansion.
        Tracks starting at the same time are excluded to avoid auto-assigning
        simultaneous tracks (different people) to the same person.

    Returns
    -------
    list[int]
        Track IDs to assign, starting with *tid* and followed by any expansion
        tracks in ascending start-time order.
    """
    result = [tid]
    for (s, t), (t0, _t1) in sorted(time_spans.items(), key=lambda x: x[1][0]):
        if s == svid and t != tid and t0 > min_time_s and (svid, t) not in assignments:
            result.append(t)
    return result


def find_assignment_conflicts(
    svid: str,
    tids: list[int],
    person_name: str,
    spans: Spans,
    assignments: Assignments,
) -> list[tuple[str, int]]:
    """Return tracks already assigned to *person_name* that time-overlap any track in *tids*.

    Overlap is defined as ``max(start_a, start_b) < min(end_a, end_b)`` on
    integer frame numbers within the same camera.  Adjacent tracks (one ends
    exactly where the other begins) are **not** considered a conflict.  Only
    tracks in the same camera (*svid*) are checked.

    Parameters
    ----------
    svid:
        Camera to check within.
    tids:
        The tracks that are about to be assigned to *person_name*.  These are
        excluded from the conflict search (a track cannot conflict with itself).
    person_name:
        The person being assigned.
    spans:
        Frame-based span data; typically from ``StitcherWidget.get_spans()``.
    assignments:
        Current assignment state.

    Returns
    -------
    list[tuple[str, int]]
        ``(svid, tid)`` pairs for every already-assigned track that overlaps.
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
