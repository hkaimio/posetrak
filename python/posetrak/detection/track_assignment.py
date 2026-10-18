# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""track_assignment.py — assign the track fragments of a detection run to persons from cross-camera geometry.

A detector's tracker starts a new track id whenever a person is lost for a frame, so one person is a chain of
fragments in every camera, and a fragment can also change person where two persons overlap. Which fragments belong
to one person is decided here without segmentation:

1. At each instant of a common time grid the trunk keypoints of every box are triangulated across cameras. Boxes
   of different cameras that agree on one trunk (it reprojects within a fraction of the box height in each) form
   a cluster, at most one box per camera: a 3D person hypothesis.
2. Hypotheses are linked over time into *tracks*, only where a hypothesis is made of the boxes the track was made
   of and no other person is near. A track therefore ends at every contact between persons; it is pure.
3. The tracks are labelled with the persons. The cues across a contact are the detector fragments that have boxes
   at the end of one track and the start of another, the distance between the ends, and the rule that two tracks
   at the same time in different places are two persons. The cues conflict where persons overlap (a detector
   swaps identities there), so seeds from rows already in the database anchor the labelling.
4. Every box is matched to the persons by projecting their 3D positions onto it, and the persons are estimated
   again from the boxes they received. Boxes that stay ambiguous are left to the continuity of their fragment, and
   a fragment whose boxes switch person is split at the switch.

The module works on arrays and plain dataclasses; reading the session database and writing the rows is
``assign_run.py``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class AssignParams:
    """Tuning of the association. Distances are metres, times seconds, errors fractions of the box height."""

    # --- pieces (the detector's fragments cut where its identity swaps)
    # Largest median distance, in box heights, between the body keypoints of a box and of the piece's last box.
    piece_gate: float = 0.06
    # A piece not continued for more frames than this ends.
    piece_max_gap: int = 20
    # Keypoint confidence for the comparison.
    piece_conf: float = 0.3

    # --- hypotheses
    # Largest median trunk reprojection error for two boxes to be consistent.
    max_rel_err: float = 0.05
    # A cluster is cut down until every box reprojects within this error of the joint estimate.
    cluster_max_err: float = 0.04
    # Fewest trunk keypoints, confident in both boxes, a pair test needs.
    min_joints: int = 3
    # A pair is rejected when its triangulated trunk is not person-sized: the largest distance between two trunk
    # keypoints. Epipolar consistency alone also accepts two persons of similar height.
    trunk_extent_m: tuple[float, float] = (0.3, 0.85)

    # --- tracks
    # Fewest boxes of a hypothesis that is linked. Two boxes can also agree by chance for two persons of similar
    # height, which the requirement that a hypothesis continue the pieces of its track makes harmless.
    min_link_members: int = 2
    # A hypothesis with another person closer than this is not linked, so a track ends at a contact.
    link_clear_m: float = 0.8
    # A hypothesis continues a track only if this share of its fragments were in the track's last hypothesis.
    link_min_share: float = 0.5
    # Distance a hypothesis made of all the track's fragments is worth, in choosing among candidates.
    fragment_bonus_m: float = 0.6
    # A hypothesis joins a track within gate_base_m + gate_speed_mps * elapsed of its prediction, at most gate_max_m.
    gate_base_m: float = 0.4
    gate_speed_mps: float = 3.0
    gate_max_m: float = 2.5
    # A track not seen for longer than this ends.
    max_gap_s: float = 1.0

    # --- labelling the tracks
    # Tracks with fewer observed instants are not labelled by themselves.
    min_track_instants: int = 30
    # Fragments with boxes within this long of a track's ends tie it to the tracks before and after it.
    stitch_window_s: float = 0.5
    # Tracks that end and start within this long of each other can be one person.
    merge_gap_s: float = 3.0
    # Pull between such tracks, per instant-equivalent, scaled down by the distance between their ends.
    kinematic_weight: float = 200.0
    # Tracks at the same time within same_place_m are one person (two clusters of the same boxes); farther apart
    # they are separate persons and the pair costs separate_weight per instant.
    same_place_m: float = 0.35
    separate_weight: float = 30.0
    # Random starts of the labelling search.
    label_restarts: int = 200
    # Boxes of a seed row a track needs before the seed's person anchors it.
    anchor_min_boxes: int = 30

    # --- boxes
    # A box joins a person when its trunk centre is within this many box heights of the person's projection ...
    project_rel_dist: float = 0.25
    # ... and no other person projects this much (box heights) farther away; else the box is left to its fragment.
    project_margin: float = 0.05
    # A person is carried across a stretch without observations up to this long (linear interpolation).
    interp_gap_s: float = 4.0
    # Rounds of assigning boxes to persons and re-estimating the persons.
    refine_passes: int = 3
    # A run of fewer boxes inside a fragment is an identity flicker and is absorbed.
    min_run: int = 4

    # --- report
    # Two persons closer than this are in contact.
    contact_m: float = 0.6
    # Boxes this close in time to a contact or a split are listed for checking.
    check_margin_s: float = 0.5
    # A stretch of a fragment labelled by its neighbours alone, longer than this, is listed.
    max_unevidenced_s: float = 0.5


@dataclass
class Obs:
    """One box of one fragment at one video frame: its confident trunk keypoints, undistorted."""

    cam: str            # camera key (capture_videos.id)
    tid: int            # detector track id
    frame: int          # video frame
    pts: np.ndarray     # (k, 2) undistorted pixels of the k trunk keypoints
    ok: np.ndarray      # (k,) bool: keypoint confident
    height: float       # box height, pixels


@dataclass
class Hypothesis:
    """A 3D person at one instant: the boxes (indices into the instant's list) that agree on it."""

    members: list[int]
    pos: np.ndarray                  # trunk centre, world metres
    err: list[float]                 # per member: median trunk reprojection error / box height
    frags: frozenset = frozenset()   # (camera, detector track id) of the members


@dataclass
class Track:
    """A person observed over a stretch, between contacts: trunk centre per instant."""

    id: int
    pos: dict[int, np.ndarray] = field(default_factory=dict)

    def first(self) -> int:
        return min(self.pos)

    def last(self) -> int:
        return max(self.pos)


def _project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    h = P @ np.vstack([X.T, np.ones(len(X))])
    return (h[:2] / h[2]).T


# ---------------------------------------------------------------------------
# 0. pieces
# ---------------------------------------------------------------------------


def make_pieces(
    frames: dict[int, list[tuple[int, np.ndarray, float]]], prm: AssignParams,
) -> dict[tuple[int, int], int]:
    """Cut one camera's detector fragments into pieces that do not swap persons.

    The detector's tracker follows boxes by overlap, so where two persons overlap it swaps their ids and a
    fragment then carries one person before and another after. The body keypoints do not swap: from one frame to
    the next a person's skeleton moves a few hundredths of its height. *frames* maps a video frame to its boxes
    (detector track id, body keypoints (n, 3), box height); the boxes are matched from frame to frame by the
    median distance of the confident keypoints (Hungarian, gated), and a box that matches nothing starts a new
    piece. A box with too few keypoints to compare continues the piece of its detector track.

    Returns (frame, detector track id) -> piece id (unique within the camera).
    """
    out: dict[tuple[int, int], int] = {}
    alive: list[dict] = []
    n_pieces = 0
    for f in sorted(frames):
        boxes = frames[f]
        alive = [t for t in alive if f - t["f"] <= prm.piece_max_gap]
        match: dict[int, dict] = {}
        if alive and boxes:
            C = np.full((len(alive), len(boxes)), np.inf)
            for i, t in enumerate(alive):
                for j, (tid, kp, h) in enumerate(boxes):
                    both = (t["kp"][:, 2] >= prm.piece_conf) & (kp[:, 2] >= prm.piece_conf)
                    if both.sum() >= 4 and h > 0:
                        C[i, j] = np.median(np.linalg.norm(t["kp"][both, :2] - kp[both, :2], axis=1)) / h
                    elif t["tid"] == tid:
                        C[i, j] = prm.piece_gate / 2        # nothing to compare: the detector's word
            C[C > prm.piece_gate] = 1e6
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] < 1e6:
                    match[j] = alive[i]
        for j, (tid, kp, h) in enumerate(boxes):
            t = match.get(j)
            if t is None:
                t = {"id": n_pieces}
                n_pieces += 1
                alive.append(t)
            t.update(f=f, kp=kp, tid=tid)
            out[(f, tid)] = t["id"]
    return out


# ---------------------------------------------------------------------------
# 1. hypotheses
# ---------------------------------------------------------------------------


def pair_error(a: Obs, b: Obs, Pa: np.ndarray, Pb: np.ndarray, min_joints: int,
               extent_m: tuple[float, float] = (0.0, np.inf)) -> float:
    """Median trunk reprojection error of the pair, as a fraction of the mean box height; inf if not consistent."""
    both = a.ok & b.ok
    if both.sum() < min_joints:
        return np.inf
    xa, xb = a.pts[both], b.pts[both]
    X4 = cv2.triangulatePoints(Pa, Pb, xa.T, xb.T)
    w = X4[3]
    if np.any(np.abs(w) < 1e-9):
        return np.inf
    X = (X4[:3] / w).T
    for P in (Pa, Pb):
        if np.any((P @ np.vstack([X.T, np.ones(len(X))]))[2] <= 0):
            return np.inf
    if not extent_m[0] <= np.linalg.norm(X[:, None] - X[None], axis=2).max() <= extent_m[1]:
        return np.inf
    e = np.maximum(np.linalg.norm(_project(Pa, X) - xa, axis=1), np.linalg.norm(_project(Pb, X) - xb, axis=1))
    return float(np.median(e) / (0.5 * (a.height + b.height)))


def triangulate_multi(obs: list[Obs], Ps: list[np.ndarray]) -> tuple[np.ndarray, list[float]] | None:
    """Trunk centre of the boxes *obs* (DLT per keypoint seen by two or more) and each box's median relative error."""
    pts = []
    errs: list[list[float]] = [[] for _ in obs]
    for j in range(len(obs[0].ok)):
        rows = []
        for o, P in zip(obs, Ps):
            if o.ok[j]:
                x, y = o.pts[j]
                rows += [x * P[2] - P[0], y * P[2] - P[1]]
        if len(rows) < 4:
            continue
        _, _, vt = np.linalg.svd(np.array(rows))
        if abs(vt[-1, 3]) < 1e-12:
            continue
        X = vt[-1, :3] / vt[-1, 3]
        pts.append(X)
        for n, (o, P) in enumerate(zip(obs, Ps)):
            if o.ok[j]:
                errs[n].append(np.linalg.norm(_project(P, X[None])[0] - o.pts[j]) / o.height)
    if not pts:
        return None
    return np.mean(pts, axis=0), [float(np.median(e)) if e else np.inf for e in errs]


def cluster_instant(obs: list[Obs], P: dict[str, np.ndarray], prm: AssignParams) -> list[Hypothesis]:
    """Cluster the boxes of one instant into 3D person hypotheses (greedy by pair error, a box per camera)."""
    pair: dict[tuple[int, int], float] = {}
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            if obs[i].cam != obs[j].cam:
                e = pair_error(obs[i], obs[j], P[obs[i].cam], P[obs[j].cam], prm.min_joints, prm.trunk_extent_m)
                if e <= prm.max_rel_err:
                    pair[(i, j)] = e
    cl: dict[int, list[int]] = {}      # box -> its cluster (a list shared by the members)

    def agrees(box: int, members: list[int]) -> bool:
        return 2 * sum((min(box, m), max(box, m)) in pair for m in members) > len(members)

    for (i, j), _ in sorted(pair.items(), key=lambda kv: kv[1]):
        ci, cj = cl.get(i), cl.get(j)
        if ci is None and cj is None:
            cl[i] = cl[j] = [i, j]
        elif ci is None or cj is None:
            box, c = (i, cj) if ci is None else (j, ci)
            if all(obs[box].cam != obs[m].cam for m in c) and agrees(box, c):
                c.append(box)
                cl[box] = c
        elif ci is not cj:
            if not {obs[m].cam for m in ci} & {obs[m].cam for m in cj} \
                    and all(agrees(m, cj) for m in ci) and all(agrees(m, ci) for m in cj):
                ci.extend(cj)
                for m in cj:
                    cl[m] = ci
    out = []
    for c in {id(v): v for v in cl.values()}.values():
        c = list(c)
        while len(c) >= 2:
            r = triangulate_multi([obs[m] for m in c], [P[obs[m].cam] for m in c])
            if r is None:
                break
            worst = int(np.argmax(r[1]))
            if r[1][worst] <= prm.cluster_max_err:
                out.append(Hypothesis(sorted(c), r[0], r[1], frozenset((obs[m].cam, obs[m].tid) for m in c)))
                break
            del c[worst]        # the box that does not fit the others (a neighbour's keypoints)
    return out


# ---------------------------------------------------------------------------
# 2. tracks
# ---------------------------------------------------------------------------


def link_hypotheses(
    hyps: list[list[Hypothesis]], dt: float, prm: AssignParams,
) -> tuple[list[Track], list[list[int | None]]]:
    """Link the hypotheses of consecutive instants into tracks (constant-velocity prediction, Hungarian gate).

    Returns the tracks and, per instant, the track id of each hypothesis (None: not linked).
    """
    tracks: list[Track] = []
    vel: dict[int, np.ndarray] = {}
    last_frags: dict[int, frozenset] = {}
    alive: list[int] = []
    which: list[list[int | None]] = []
    for i, all_hs in enumerate(hyps):
        # A hypothesis with another person close by is not linked: its cluster may hold a neighbour's box, and a
        # distance gate cannot tell the two apart.
        usable = [len(h.members) >= prm.min_link_members
                  and not any(g is not h and np.linalg.norm(g.pos - h.pos) < prm.link_clear_m for g in all_hs)
                  for h in all_hs]
        hs = [h for h, ok in zip(all_hs, usable) if ok]
        alive = [t for t in alive if (i - tracks[t].last()) * dt <= prm.max_gap_s]
        pred, gate = [], []
        for t in alive:
            el = (i - tracks[t].last()) * dt
            pred.append(tracks[t].pos[tracks[t].last()] + vel[t] * el)
            gate.append(min(prm.gate_base_m + prm.gate_speed_mps * el, prm.gate_max_m))
        got: list[int | None] = [None] * len(hs)
        if alive and hs:
            dist = np.array([[np.linalg.norm(h.pos - p) for p in pred] for h in hs])
            # Frame to frame the detector keeps a person's fragment: a hypothesis made of other fragments is
            # another person, or the same person after a contact, which the labelling decides.
            share = np.array([[len(h.frags & last_frags[t]) / max(1, len(h.frags)) for t in alive] for h in hs])
            dist = np.where(share >= prm.link_min_share, dist, np.inf)
            cost = np.where(np.isfinite(dist), dist - prm.fragment_bonus_m * share, 1e6)
            for r, c in zip(*linear_sum_assignment(cost)):
                if dist[r, c] <= gate[c]:
                    got[r] = alive[c]
        for r, h in enumerate(hs):
            t = got[r]
            if t is None:
                t = len(tracks)
                tracks.append(Track(t))
                vel[t] = np.zeros(3)
                alive.append(t)
                got[r] = t
            else:
                tr = tracks[t]
                el = (i - tr.last()) * dt
                vel[t] = 0.5 * vel[t] + 0.5 * (h.pos - tr.pos[tr.last()]) / el   # damped: keypoints are noisy
            tracks[t].pos[i] = h.pos
            last_frags[t] = h.frags
        full: list[int | None] = [None] * len(all_hs)
        for slot, t in zip([n for n, ok in enumerate(usable) if ok], got):
            full[slot] = t
        which.append(full)
    return tracks, which


def interpolate(
    pos: dict[int, np.ndarray], i: int, dt: float, max_gap_s: float, keys: list[int] | None = None,
) -> np.ndarray | None:
    """Position at instant *i*: observed, or interpolated between observations less than max_gap_s apart."""
    if i in pos:
        return pos[i]
    keys = keys if keys is not None else sorted(pos)
    k = int(np.searchsorted(keys, i))
    if k == 0 or k == len(keys):
        return None
    a, b = keys[k - 1], keys[k]
    if (b - a) * dt > max_gap_s:
        return None
    f = (i - a) / (b - a)
    return (1 - f) * pos[a] + f * pos[b]


def count_persons(tracks: list[Track], prm: AssignParams) -> int:
    """Number of persons seen together for a fifth of the time that anyone is seen (long tracks only)."""
    long_ = [t for t in tracks if len(t.pos) >= prm.min_track_instants]
    if not long_:
        return 0
    cover = np.zeros(max(t.last() for t in long_) + 1, int)
    for t in long_:
        cover[t.first():t.last() + 1] += 1
    active = cover[cover > 0]
    return max(k for k in range(1, int(cover.max()) + 1) if (active >= k).mean() >= 0.2)


# ---------------------------------------------------------------------------
# 3. labelling the tracks
# ---------------------------------------------------------------------------


def label_tracks(
    tracks: list[Track], ends: dict[int, tuple[Counter, Counter]], dt: float, n_persons: int, prm: AssignParams,
    anchors: dict[int, int] | None = None,
) -> dict[int, int | None]:
    """Decide which person each long track belongs to: track id -> person index, None for no person.

    Pairs of tracks carry a weight, positive where they are likely one person and negative where they cannot be:

    * Tracks that do not overlap in time are tied by the detector fragments (*ends*: per track the fragments of
      its first and of its last stretch) that have boxes at the end of one and the start of the other, and by the
      distance between those ends when the gap is short.
    * Tracks that overlap in time are one person when in the same place (two clusters of the same boxes) and
      two persons otherwise.

    The labelling that keeps the most weight inside persons is found by iterated local search from random starts;
    a track may also take no person, so a bystander does not join the subject. *anchors* fixes the person of
    tracks the caller knows.
    """
    anchors = anchors or {}
    long_ = [t for t in tracks if len(t.pos) >= prm.min_track_instants]
    n = len(long_)
    if n == 0:
        return {}
    W = np.zeros((n, n))
    for a in range(n):
        A = long_[a]
        for b in range(a + 1, n):
            B = long_[b]
            lo, hi = max(A.first(), B.first()), min(A.last(), B.last())
            if lo < hi:
                ds = [np.linalg.norm(x - y) for i in range(lo, hi + 1, max(1, (hi - lo) // 20))
                      if (x := interpolate(A.pos, i, dt, prm.interp_gap_s)) is not None
                      and (y := interpolate(B.pos, i, dt, prm.interp_gap_s)) is not None]
                if ds:
                    W[a, b] = W[b, a] = (hi - lo) * (1.0 if np.mean(ds) < prm.same_place_m else -prm.separate_weight)
                continue
            first, second = (A, B) if B.first() > A.last() else (B, A)
            gap = (second.first() - first.last()) * dt
            if gap > prm.merge_gap_s:
                continue
            shared = sum(min(c, ends[second.id][0].get(f, 0)) for f, c in ends[first.id][1].items())
            d = float(np.linalg.norm(second.pos[second.first()] - first.pos[first.last()]))
            reach = min(prm.gate_base_m + prm.gate_speed_mps * gap, prm.gate_max_m)
            W[a, b] = W[b, a] = shared + (prm.kinematic_weight * (1.0 - d / reach) if d < reach else 0.0)

    none = n_persons                           # the label 'no person'
    fixed = np.array([anchors.get(t.id, -1) for t in long_])
    rng = np.random.default_rng(0)
    best, best_e = None, np.inf
    for _ in range(prm.label_restarts):
        lab = np.where(fixed >= 0, fixed, rng.integers(0, n_persons + 1, n))
        for _ in range(50):
            moved = False
            for a in rng.permutation(n):
                if fixed[a] >= 0:
                    continue
                # a track without evidence either way joins a person rather than staying out
                gain = np.array([W[a, lab == p].sum() for p in range(n_persons)] + [-1e-3])
                p = int(gain.argmax())
                if gain[p] > gain[lab[a]] + 1e-9:
                    lab[a], moved = p, True
            if not moved:
                break
        e = -sum(W[a, b] for a in range(n) for b in range(a + 1, n) if lab[a] == lab[b] and lab[a] != none)
        if e < best_e - 1e-9:
            best, best_e = lab.copy(), e
    return {t.id: (int(best[k]) if best[k] != none else None) for k, t in enumerate(long_)}


# ---------------------------------------------------------------------------
# 4. boxes
# ---------------------------------------------------------------------------


def refine_persons(
    obs_at: list[list[Obs]], seed: dict[int, dict[int, np.ndarray]], dt: float, P: dict[str, np.ndarray],
    prm: AssignParams,
) -> tuple[dict[int, dict[int, int]], dict[int, dict[int, np.ndarray]]]:
    """Assign every box to one of the persons by projecting the person's 3D position onto it, and re-estimate.

    *seed* holds each person's trunk centre at the instants where it was observed. Each pass interpolates the
    persons across gaps, matches the boxes of every camera to the persons (Hungarian; cost: distance between the
    box's trunk centre and the projected person, in box heights, gated, and rejected when another person is
    nearly as close) and triangulates each person again from the boxes it received. A box left out here takes the
    person of its fragment's other boxes (:func:`label_fragment`).

    Returns ``{instant: {box index: person}}`` and the final observed positions.
    """
    pos = {p: dict(v) for p, v in seed.items()}
    got: dict[int, dict[int, int]] = {}
    for _ in range(prm.refine_passes):
        got = {}
        new: dict[int, dict[int, np.ndarray]] = {p: {} for p in pos}
        keys = {p: sorted(v) for p, v in pos.items()}
        for i, obs in enumerate(obs_at):
            at = {p: x for p in pos if (x := interpolate(pos[p], i, dt, prm.interp_gap_s, keys[p])) is not None}
            if not at or not obs:
                continue
            people = list(at)
            assigned: dict[int, int] = {}
            for cam in {o.cam for o in obs}:
                boxes = [k for k, o in enumerate(obs) if o.cam == cam and o.ok.any()]
                if not boxes:
                    continue
                proj = {p: _project(P[cam], at[p][None])[0] for p in people}
                cost = np.array([[np.linalg.norm(proj[p] - obs[k].pts[obs[k].ok].mean(axis=0)) / obs[k].height
                                  for p in people] for k in boxes])
                for r, c in zip(*linear_sum_assignment(cost)):
                    others = np.delete(cost[r], c)
                    if cost[r, c] <= prm.project_rel_dist and (
                            not others.size or others.min() >= cost[r, c] + prm.project_margin):
                        assigned[boxes[r]] = people[c]
            got[i] = assigned
            for p in people:
                mem = [k for k, q in assigned.items() if q == p]
                if len(mem) >= 2:
                    r = triangulate_multi([obs[k] for k in mem], [P[obs[k].cam] for k in mem])
                    if r is not None:
                        new[p][i] = r[0]
        pos = new
    return got, pos


@dataclass
class Segment:
    first: int
    last: int
    person: int | None            # None: unassigned
    # (first, last) video frames of the stretches of boxes that carry no person of their own: the person is taken
    # from the neighbouring boxes of the fragment there
    blind: list[tuple[int, int]] = field(default_factory=list)


def label_fragment(frames: list[int], votes: dict[int, int | None], min_run: int) -> list[Segment]:
    """Split a fragment into segments, one per person, from the person of each of its boxes.

    *frames* are all boxes of the fragment, sorted; *votes* maps the frames that have a person to it (None for a
    box assigned to a person that is not kept). Boxes without a vote take the person of the nearest voted box, and
    a switch between persons falls in the middle of the unvoted stretch between them.
    """
    voted = [f for f in frames if f in votes]
    if not voted:
        return [Segment(frames[0], frames[-1], None)]
    runs: list[list] = []                               # [person, boxes]
    for f in voted:
        if runs and runs[-1][0] == votes[f]:
            runs[-1][1].append(f)
        else:
            runs.append([votes[f], [f]])
    while len(runs) > 1:                                # absorb flicker into the longer neighbour
        k = min(range(len(runs)), key=lambda r: len(runs[r][1]))
        if len(runs[k][1]) >= min_run:
            break
        t = max((r for r in (k - 1, k + 1) if 0 <= r < len(runs)), key=lambda r: len(runs[r][1]))
        runs[t][1] = sorted(runs[t][1] + runs[k][1])
        del runs[k]
        merged: list[list] = []
        for r in runs:
            if merged and merged[-1][0] == r[0]:
                merged[-1][1] = sorted(merged[-1][1] + r[1])
            else:
                merged.append(r)
        runs = merged
    cuts = []                                           # first frame of each run after the first
    for a, b in zip(runs, runs[1:]):
        lo, hi = max(a[1]), min(b[1])
        between = [f for f in frames if lo < f < hi]
        cuts.append(between[len(between) // 2] if between else hi)
    segs = []
    start = frames[0]
    for k, (person, _) in enumerate(runs):
        end = cuts[k] - 1 if k < len(cuts) else frames[-1]
        idx = [f for f in frames if start <= f <= end]
        if idx:
            blind: list[tuple[int, int]] = []
            prev_blind = False
            for f in idx:
                is_blind = votes.get(f) != person
                if is_blind:
                    blind[-1:] = [(blind[-1][0], f)] if prev_blind else [*blind[-1:], (f, f)]
                prev_blind = is_blind
            segs.append(Segment(idx[0], idx[-1], person, blind))
        start = end + 1
    return segs


def contacts(
    track_pos: dict[int, dict[int, np.ndarray]], n_instants: int, dt: float, prm: AssignParams,
) -> dict[int, set[int]]:
    """Instant -> the persons within contact_m of another person (positions interpolated across short gaps)."""
    out: dict[int, set[int]] = defaultdict(set)
    keys = {p: sorted(v) for p, v in track_pos.items()}
    for i in range(n_instants):
        at = {p: x for p in track_pos if (x := interpolate(track_pos[p], i, dt, prm.max_gap_s, keys[p])) is not None}
        for a in at:
            for b in at:
                if a < b and np.linalg.norm(at[a] - at[b]) < prm.contact_m:
                    out[i] |= {a, b}
    return out
