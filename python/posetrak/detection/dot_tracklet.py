# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""dot_tracklet.py — per-camera frame-to-frame dot-candidate identity
linking (Phase B, see docs/roadmap/features/marker-based-mocap/status.md's
2026-09-06 entry for the investigation this exists to act on).

Checking the production Phase A run (background subtraction + chroma
filter) against real data found the actual bottleneck has moved: raw
detection recall improved hugely, but only ~4-6% of raw candidates during
a fast swing survive `resolve_dot_assignment()`'s gate to become a used
observation, with zero rejected at the later UKF outlier-check stage --
the attrition is entirely at assignment. `DotTrackletLinker` assigns each
candidate a per-camera `tracklet_id` so `resolve_dot_assignment()`
(dot_assignment.cpp) can relax its own gate for a candidate continuing an
already-established track, rather than judging every frame from a cold
start.

This linker assigns *identity*, not *validity* -- every candidate gets
some tracklet_id, including a length-1 "singleton" tracklet; whether a
tracklet is trustworthy is the tracker's own call to make (it has real
predicted positions and gate costs to judge with), not this module's.
That is a deliberate difference from the prototype's own
`build_tracklets()` (python/tools/prototype_streak_detector.py), which
returns a keep/reject boolean per candidate -- a judgment that genuinely
needs to see a tracklet's own future (it only trusts a tracklet once it's
survived several later frames and moved enough), so that tool works over
a whole camera's already-detected candidate sequence at once. Plain
identity linking has no such lookahead requirement: each frame only needs
the previous frames' currently-open tracklets, so `DotTrackletLinker` runs
frame-by-frame inside the existing per-camera detection loop
(`marker_pipeline.MarkerDetectionPipeline._process_camera()`) with no
buffering/restructuring of that loop needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from posetrak.detection.dot_blob_detector import BlobCandidate


@dataclass
class _OpenTracklet:
    tracklet_id: int
    last_cx: float
    last_cy: float
    missed: int = 0


class DotTrackletLinker:
    """Assigns a stable per-camera `tracklet_id` to each frame's
    `BlobCandidate` list, in place, as frames arrive in order -- one
    instance per camera, fed one frame at a time via `link_frame()`.

    Greedy nearest-neighbor matching against the currently-open tracklets
    (same idea as any simple online multi-object tracker, and the same
    algorithm the prototype's `build_tracklets()` uses for its own linking
    step, just without that tool's separate survival judgment on top). Not
    Hungarian-optimal across a whole frame's matches at once -- with only a
    handful of open tracklets and candidates per camera per step, greedy is
    adequate, and this needs to be cheap since it runs on every frame of
    every dot-enabled camera.
    """

    def __init__(self, max_link_px: float = 60.0, max_missed: int = 2) -> None:
        self._max_link_px = max_link_px
        self._max_missed = max_missed
        self._open: list[_OpenTracklet] = []
        self._next_id = 0

    def link_frame(self, candidates: list[BlobCandidate]) -> None:
        """Assign `tracklet_id` to every candidate in *candidates*
        (mutated in place) and advance this linker's own state by one
        frame. Call once per frame, in timestamp order, for one camera."""
        matched: set[int] = set()
        unmatched = set(range(len(candidates)))
        for tr in self._open:
            if not unmatched:
                break
            best_i, best_d = None, self._max_link_px
            for i in unmatched:
                d = math.hypot(candidates[i].cx - tr.last_cx, candidates[i].cy - tr.last_cy)
                if d <= best_d:
                    best_i, best_d = i, d
            if best_i is not None:
                candidates[best_i].tracklet_id = tr.tracklet_id
                tr.last_cx, tr.last_cy = candidates[best_i].cx, candidates[best_i].cy
                tr.missed = 0
                matched.add(id(tr))
                unmatched.discard(best_i)

        for tr in self._open:
            if id(tr) not in matched:
                tr.missed += 1
        self._open = [tr for tr in self._open if tr.missed <= self._max_missed]

        for i in unmatched:
            candidates[i].tracklet_id = self._next_id
            self._open.append(_OpenTracklet(self._next_id, candidates[i].cx, candidates[i].cy))
            self._next_id += 1
