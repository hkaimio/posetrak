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
the attrition is entirely at assignment. `MotionGatedLinker` assigns each
candidate a per-camera `tracklet_id` so `resolve_dot_assignment()`
(dot_assignment.cpp) can relax its own gate for a candidate continuing an
already-established track, rather than judging every frame from a cold
start.

This linker assigns *identity*, not *validity* -- every candidate gets
some tracklet_id, including a length-1 "singleton" tracklet; whether a
tracklet is trustworthy is the tracker's own call to make (it has real
predicted positions and gate costs to judge with), not this module's.
That is a deliberate difference from the prototype's own
`build_tracklets()` (python/tools/prototypes/prototype_streak_detector.py), which
returns a keep/reject boolean per candidate -- a judgment that genuinely
needs to see a tracklet's own future (it only trusts a tracklet once it's
survived several later frames and moved enough), so that tool works over
a whole camera's already-detected candidate sequence at once. Plain
identity linking has no such lookahead requirement: each frame only needs
the previous frames' currently-open tracklets, so `MotionGatedLinker` runs
frame-by-frame inside the existing per-camera detection loop
(`marker_pipeline.MarkerDetectionPipeline._process_camera()`) with no
buffering/restructuring of that loop needed.

Replaces (2026-09-11) an earlier `DotTrackletLinker`, which linked purely
by a fixed 60px nearest-neighbor gate to each open tracklet's *last seen
pixel*, with no motion-consistency or identity check at all. Reviewing a
real 42-48s/5-camera B2/B3 test window by hand (label_tracklet_groups_gui.
py), Harri found that ~14/25 final tracklet groups needed a manual
mid-tracklet split -- most from the same mechanism: an occluded track's
last-seen pixel stays *frozen* while missed, so a different real marker
that later drifts near that stale point gets adopted as a continuation of
the wrong track. `MotionGatedLinker` instead keeps a per-track
constant-velocity Kalman filter that *coasts* forward along the track's
own last known motion during a gap, and gates a candidate on Mahalanobis
distance from that moving prediction rather than raw pixel distance to a
static point.

Validated (python/tools/prototypes/prototype_motion_gated_linker.py, status.md
2026-09-11) against 11 real human-determined identity-switch frames from
that same review (recorded as the 4-element sub-range members `label_
tracklet_groups_gui.py` writes when a group is manually split): 11/11
correctly separated, using only the raw already-detected candidate
positions. A per-frame disambiguation margin (below) was added after
that same validation surfaced a second, harder failure mode -- two
*simultaneously visible*, closely-spaced real markers (e.g. adjacent
knee/ankle markers) both landing inside one track's gate, with tiny cost
differences flipping which one "wins" frame to frame. That ambiguity is
inherent to single-camera 2D tracking (the same close-marker-projection
problem that motivated the whole marker-catalog-and-assignment redesign,
see marker-catalog-and-assignment-redesign.md §0) and isn't fully solved
by motion consistency alone -- B2's cross-camera reprojection test (or a
human, via the scrub-and-split tool) remains the backstop for whatever
this doesn't catch. Harri's explicit priority: tracklet construction
should strongly favor a low false-link (wrong-identity) rate over
completeness -- an unnecessary break is cheap to fix later; a false link
silently corrupts a slot's whole trajectory. Every gate below defaults
tight for that reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from posetrak.detection.dot_blob_detector import BlobCandidate

_H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
_I4 = np.eye(4)


@dataclass
class _Track:
    tracklet_id: int
    x: np.ndarray  # [px, py, vx, vy]
    P: np.ndarray  # 4x4 covariance
    last_frame: int
    n_obs: int = 1
    missed: int = 0


class MotionGatedLinker:
    """Assigns a stable per-camera `tracklet_id` to each frame's
    `BlobCandidate` list, in place, as frames arrive in order -- one
    instance per camera, fed one frame at a time via `link_frame()`.

    Each open tracklet carries a constant-velocity Kalman filter
    (state `[x, y, vx, vy]`); a candidate is accepted as a continuation
    only if its Mahalanobis distance from the filter's *predicted*
    position (not its last seen pixel) is within `mahal_gate`. A missed
    frame still predicts/coasts the filter forward along its last known
    velocity, so an occlusion doesn't leave the track's belief frozen at
    a stale point a different real marker could later drift into (see
    module docstring). A brand-new track (a single observation, no
    velocity estimate yet) falls back to a small fixed-radius gate for
    its second point only -- there's no motion history yet to be
    inconsistent with. Frame-to-candidate matching is the Hungarian
    optimum over the whole frame's gated cost matrix, not greedy --
    the candidate counts here (a handful of dots per camera per frame)
    make that cheap.
    """

    def __init__(
        self,
        birth_gate_px: float = 25.0,
        mahal_gate: float = 9.21,  # chi2, df=2, ~99% -- tight, per the low-false-link priority
        # Cut aggressively on occlusion (2026-09-11, Harri): most of the
        # remaining wrong-identity jumps happen when a marker is briefly
        # occluded by another body part while a *different* marker's own
        # trajectory crosses nearby -- coasting through the whole gap
        # risks the crossing marker looking like the better match by the
        # time real detections resume. Swept 1-6 against the real
        # full-capture human splits: 3 gives the best result on *both*
        # metrics at once (12/12 real cuts correctly separated, best
        # purity on the 40 non-split tracklets) -- not just "more
        # aggressive is better" but the actual measured optimum in this
        # sweep. A short gap is cheap to re-stitch later (B2's cross-
        # camera check, or a human); a wrong link isn't.
        max_missed: int = 3,
        meas_std_px: float = 3.0,
        # Lowered 10 -> 2 (2026-09-12, from Harri's review feedback: "many
        # otherwise-OK tracklets have a clearly outlier last frame").
        # Diagnosed: even with the 2026-09-11 coasting-bug fix (so dt is
        # correctly 1 per real elapsed frame, not compounding), Q's own
        # dt**4 growth still means the gate is *widest* on exactly the
        # last frame before max_missed forces a track to die -- three
        # consecutively missed frames alone, with the old accel_std_px=10,
        # already grow the position std to ~30px (95% radius ~60px) even
        # from a converged, confident steady state. A track that's about
        # to legitimately die (no real continuation nearby) is exactly
        # the one most likely to snap onto a stray candidate right at
        # that peak-width moment, becoming its erroneous last point.
        # Swept 1/2/3/5/7/10 against every mid-tracklet split Harri had
        # made reviewing the full capture (17 real cases by that point):
        # 10 (the old default) resolves 0/17 -- not a coincidence, these
        # are by definition cases the linker at its shipped settings
        # still got wrong -- falling monotonically to 2 at 17/17 (1 also
        # gets 17/17 but with worse purity elsewhere: 390/429 >= 0.90
        # vs. 2's 396/429). Also improves (not fully resolves) the
        # B2 cross-camera grouping precision issue from the max_missed=3
        # tuning (0.39 -> 0.44 on the 42-48s GT window) -- see
        # status.md's 2026-09-11 "B2/linker tension" entry for the larger,
        # still-open problem that remains.
        accel_std_px: float = 2.0,
        # General-purpose, not the fix for the above (2026-09-12): caps
        # the *uncertainty growth* (Q) at this many frames' worth of dt,
        # while `dt` itself still drives the mean prediction (F)
        # uncorrected. A no-op whenever consecutive link_frame() calls are
        # already 1 frame apart (true for every detection run so far,
        # frame_step=1) -- dt is then always <= 1 already, so any cap
        # >= 1 changes nothing; matters only for a frame_step > 1
        # detection run, where a genuine gap exists between calls with no
        # missed frame at all. None (default): no cap.
        max_noise_dt: float | None = None,
        disambiguation_margin: float = 2.0,
    ) -> None:
        self._birth_gate_px = birth_gate_px
        self._mahal_gate = mahal_gate
        self._max_missed = max_missed
        self._R = np.eye(2) * meas_std_px ** 2
        self._accel_var = accel_std_px ** 2
        self._disambiguation_margin = disambiguation_margin
        self._max_noise_dt = max_noise_dt
        self._open: list[_Track] = []
        self._next_id = 0

    def _predict(self, tr: _Track, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        q = self._accel_var
        q_dt = dt if self._max_noise_dt is None else min(dt, self._max_noise_dt)
        Q = q * np.array([
            [q_dt ** 4 / 4, 0, q_dt ** 3 / 2, 0],
            [0, q_dt ** 4 / 4, 0, q_dt ** 3 / 2],
            [q_dt ** 3 / 2, 0, q_dt ** 2, 0],
            [0, q_dt ** 3 / 2, 0, q_dt ** 2],
        ])
        return F @ tr.x, F @ tr.P @ F.T + Q

    def link_frame(self, video_frame: int, candidates: list[BlobCandidate]) -> None:
        """Assign `tracklet_id` to every candidate in *candidates*
        (mutated in place) and advance this linker's own state by one
        frame. Call once per frame, in video-frame order, for one
        camera -- `video_frame` must be the real frame index (not just a
        call counter) since it drives the Kalman prediction's `dt`, which
        must reflect any real gap (e.g. a frame_step > 1 detection run)
        rather than assuming every call is one frame apart."""
        n = len(candidates)
        assigned = [-1] * n
        preds = [(tr, *self._predict(tr, video_frame - tr.last_frame)) for tr in self._open]

        cost = np.full((len(preds), n), 1e9)
        feasible = np.zeros((len(preds), n), dtype=bool)
        for i, (tr, x_pred, P_pred) in enumerate(preds):
            S = _H @ P_pred @ _H.T + self._R
            Sinv = np.linalg.inv(S)
            for j, cand in enumerate(candidates):
                y = np.array([cand.cx, cand.cy]) - _H @ x_pred
                if tr.n_obs < 2:
                    d = math.hypot(y[0], y[1])
                    if d <= self._birth_gate_px:
                        cost[i, j] = d
                        feasible[i, j] = True
                else:
                    m2 = float(y @ Sinv @ y)
                    if m2 <= self._mahal_gate:
                        cost[i, j] = m2
                        feasible[i, j] = True

        # Per-track disambiguation: two closely-spaced *different* real
        # markers can both sit inside one track's gate every frame,
        # near-tied in cost -- committing to whichever the Hungarian
        # solver happens to prefer that frame makes the assignment
        # ping-pong between them, corrupting both real tracks. Refuse to
        # commit this frame (coast instead) unless the best candidate is
        # clearly better than the runner-up (B2's own "disambiguation
        # margin over the runner-up" idea, applied per frame). Only for
        # motion-established tracks (n_obs >= 2, Mahalanobis-gated) --
        # a birth-phase row's costs are raw pixel distances within a
        # small fixed radius, naturally close together for two markers
        # that simply start out near each other, with no motion evidence
        # yet to call genuinely ambiguous (found via this module's own
        # test suite: two adjacent tracks born a frame apart otherwise
        # spuriously lost their second observation to this check).
        for i, (tr, _, _) in enumerate(preds):
            if tr.n_obs < 2:
                continue
            row_costs = sorted(cost[i, j] for j in range(n) if feasible[i, j])
            if len(row_costs) >= 2 and row_costs[1] < row_costs[0] * self._disambiguation_margin:
                feasible[i, :] = False

        matched_tids: set[int] = set()
        if preds and n:
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                if not feasible[i, j]:
                    continue
                tr, x_pred, P_pred = preds[i]
                S = _H @ P_pred @ _H.T + self._R
                K = P_pred @ _H.T @ np.linalg.inv(S)
                z = np.array([candidates[j].cx, candidates[j].cy])
                tr.x = x_pred + K @ (z - _H @ x_pred)
                tr.P = (_I4 - K @ _H) @ P_pred
                tr.last_frame = video_frame
                tr.n_obs += 1
                tr.missed = 0
                candidates[j].tracklet_id = tr.tracklet_id
                assigned[j] = tr.tracklet_id
                matched_tids.add(tr.tracklet_id)

        still_open = []
        for tr, x_pred, P_pred in preds:
            if tr.tracklet_id in matched_tids:
                still_open.append(tr)
            else:
                tr.missed += 1
                if tr.missed <= self._max_missed:
                    tr.x, tr.P = x_pred, P_pred  # coast on last known motion
                    # BUG (found 2026-09-11, real full-capture data: tracks
                    # snapping onto unrelated, distant markers after only a
                    # handful of missed frames): `last_frame` must advance
                    # here too, not only on a real match. Leaving it at the
                    # last real observation means the *next* predict() computes
                    # dt as time-since-last-match and reapplies it on top of
                    # state that was already advanced by this step -- the
                    # coast compounds quadratically instead of linearly.
                    # Traced numerically: 6 missed frames at a steady 10px/
                    # frame produced a predicted position 150px past the
                    # correct one and a ~957px gate std-dev (should be ~85px)
                    # -- wide enough to accept nearly any nearby candidate,
                    # including a different real marker.
                    tr.last_frame = video_frame
                    still_open.append(tr)
        self._open = still_open

        for j in range(n):
            if assigned[j] == -1:
                cand = candidates[j]
                x0 = np.array([cand.cx, cand.cy, 0.0, 0.0])
                P0 = np.diag([self._R[0, 0], self._R[1, 1], 400.0, 400.0])
                tr = _Track(self._next_id, x0, P0, video_frame)
                self._open.append(tr)
                cand.tracklet_id = self._next_id
                self._next_id += 1
