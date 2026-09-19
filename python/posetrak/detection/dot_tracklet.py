# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""dot_tracklet.py — per-camera frame-to-frame dot-candidate identity
linking.

Raw detection recall on fast motion can be high while only a small fraction
of candidates survive `resolve_dot_assignment()`'s gate to become used
observations: the attrition is at assignment, not at the later UKF outlier
check, because every frame is judged from a cold start. `MotionGatedLinker`
assigns each candidate a per-camera `tracklet_id` so `resolve_dot_assignment()`
(dot_assignment.cpp) can relax its own gate for a candidate that continues an
already-established track.

This linker assigns *identity*, not *validity*. Every candidate gets some
tracklet_id, including a length-1 "singleton" tracklet; whether a tracklet is
trustworthy is the tracker's call (it has predicted positions and gate costs
to judge with). This differs from a whole-sequence tracklet builder that
returns a keep/reject verdict per candidate: such a verdict needs a
tracklet's own future (trust it only once it has survived several later
frames and moved enough), whereas identity linking needs only the previous
frames' open tracklets. So the linker runs frame by frame inside the
per-camera detection loop
(`marker_pipeline.MarkerDetectionPipeline._process_camera()`), with no
buffering of that loop.

Why motion-gated. Linking by nearest neighbour to each open tracklet's *last
seen pixel* has no motion or identity check, and an occluded track's last
seen pixel stays frozen while it is missed. A different real marker that
later drifts near that stale point is then adopted as a continuation of the
wrong track. Here each track keeps a constant-velocity Kalman filter that
*coasts* along the track's own last known motion through a gap, and a
candidate is gated on Mahalanobis distance from that moving prediction
instead. When validated against human-determined identity switches, this
separated all of them using only the raw candidate positions.

Ambiguity that motion alone does not resolve. Two *simultaneously visible*,
closely spaced markers (adjacent knee or ankle markers, say) can both fall
inside one track's gate, with tiny cost differences flipping which one wins
from frame to frame. That is inherent to single-camera 2D tracking (see
marker-catalog-and-assignment-redesign.md §0), so a per-frame disambiguation
margin makes the linker coast instead of guessing, and a cross-camera
reprojection test (tools/build_tracklet_groups.py) or a human reviewer is the
backstop for what remains.

Bias toward breaking. Tracklet construction favours a low false-link
(wrong-identity) rate over completeness: an unnecessary break is cheap to
re-stitch later, while a false link silently corrupts a slot's whole
trajectory. Every gate below defaults tight for that reason.
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
        # Cut aggressively on occlusion: most wrong-identity jumps happen when
        # a marker is briefly occluded by another body part while a *different*
        # marker's trajectory crosses nearby -- coasting through the whole gap
        # risks the crossing marker looking like the better match by the time
        # real detections resume. Swept over 1-6 against human-labeled identity
        # switches, 3 was the best on both measures (correct separation of the
        # labeled cuts, and purity of the tracklets that should not be split),
        # so this is a measured optimum, not just "more aggressive is better".
        # A short gap is cheap to re-stitch later (cross-camera check, or a
        # human); a wrong link isn't.
        max_missed: int = 3,
        meas_std_px: float = 3.0,
        # Kept small so the gate does not widen much while a track coasts. The
        # process-noise term Q grows as dt**4, so the gate is *widest* on
        # exactly the last frame before max_missed forces a track to die: with
        # accel_std_px=10, three missed frames alone grow the position std to
        # ~30 px (95% radius ~60 px) even from a converged track. A track about
        # to legitimately die (no real continuation nearby) is then the one
        # most likely to snap onto a stray candidate at that peak-width moment
        # and end with an erroneous last point. Swept over 1/2/3/5/7/10 against
        # human-labeled mid-tracklet splits: larger values fail them, 2 and 1
        # both fix them, and 2 keeps better purity on tracklets that should
        # not be split.
        accel_std_px: float = 2.0,
        # Caps the *uncertainty growth* (Q) at this many frames' worth of dt,
        # while `dt` itself still drives the mean prediction (F) uncorrected.
        # A no-op whenever consecutive link_frame() calls are 1 frame apart
        # (frame_step=1): dt is then always <= 1, so any cap >= 1 changes
        # nothing. It matters only for a frame_step > 1 detection run, where a
        # genuine gap exists between calls with no missed frame at all. None
        # (default): no cap.
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
        # clearly better than the runner-up (the "margin over the runner-up"
        # test used for cross-camera grouping, applied per frame). Only for
        # motion-established tracks (n_obs >= 2, Mahalanobis-gated): a
        # birth-phase row's costs are raw pixel distances within a small fixed
        # radius, naturally close together for two markers that simply start
        # out near each other, with no motion evidence yet to call them
        # ambiguous (two adjacent tracks born a frame apart would otherwise
        # spuriously lose their second observation to this check).
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
                    # `last_frame` must advance here too, not only on a real
                    # match. Leaving it at the last real observation makes the
                    # *next* predict() compute dt as time-since-last-match and
                    # apply it on top of state this step already advanced, so
                    # the coast compounds quadratically instead of linearly:
                    # after 6 missed frames at a steady 10 px/frame the
                    # prediction would be 150 px past the correct position with
                    # a ~957 px gate std-dev (correct: ~85 px) -- wide enough
                    # to accept nearly any nearby candidate, including a
                    # different real marker.
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
