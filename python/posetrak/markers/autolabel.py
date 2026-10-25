# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""autolabel.py — assign the dot tracklets of a person to marker slots without running the tracker.

The tracker assigns dots to slots frame by frame inside the Kalman filter, which makes
every pass cost the filter. The assignment itself is geometry: each slot's place in the
image is the joint frame of a stored pose times the slot's offset, projected into the camera.
This module does it for whole tracklets, from the stored poses of any earlier tracking run:

1. every slot is predicted in the camera at a sample of a tracklet's frames, skipping the
   slots whose normal faces away from the camera;
2. the tracklet's cost for a slot is the median distance in pixels between the tracklet and the
   prediction, over the samples;
3. tracklets are given slots best first, a slot being taken by at most one tracklet at a
   time in a camera.

The stored poses are only as good as the run they come from; the confidence of an assignment
(the margin to the next slot) says how much it depends on that.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares

from posetrak.markers.bones import BodyPose, triangulate
from posetrak.markers.tracklets import Tracklets

MAX_COST_PX = 60.0         # a tracklet further than this from a slot (median) is not that slot
MIN_VISIBLE_FRACTION = 0.5  # the slot must face the camera in this share of the samples
VISIBLE_COSINE = -0.1       # a slot is hidden when the camera is this far behind its normal


@dataclass
class SlotPlacement:
    name: str
    joint: str
    offset: np.ndarray
    normal: np.ndarray | None


def load_placements(path: str) -> list[SlotPlacement]:
    """The markers of an attachment set (joint-frame offsets and normals)."""
    doc = yaml.safe_load(open(path, encoding="utf-8"))
    return [
        SlotPlacement(m["name"], m["parent_joint"], np.asarray(m["offset"], float),
                      np.asarray(m["normal"], float) if m.get("normal") else None)
        for m in doc["markers"]
    ]


class Predictor:
    """Where each slot appears in a camera at a time, from a stored pose.

    *corrections* maps (joint, window index) to a 4x4 transform in the joint's own frame
    that adjusts the stored pose in that window (see :func:`refit_bones`); windows are
    *window_s* seconds long from *t_origin*.
    """

    def __init__(self, pose: BodyPose, placements: list[SlotPlacement], states: dict,
                 t_origin: float = 0.0, window_s: float = 4.0) -> None:
        self.pose, self.placements, self.states = pose, placements, states
        self.names = [p.name for p in placements]
        self._joints = [p.joint for p in placements]
        self.t_origin, self.window_s = t_origin, window_s
        self.corrections: dict[tuple[str, int], np.ndarray] = {}

    def window(self, t: float) -> int:
        return int((t - self.t_origin) // self.window_s)

    def joint_frame(self, t: float, joint: str, frames: dict) -> np.ndarray:
        """The joint's world frame at *t* with the window's correction, if any, applied."""
        delta = self.corrections.get((joint, self.window(t)))
        return frames[joint] if delta is None else frames[joint] @ delta

    def pixels(self, t: float, cam_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        """(distorted pixels (N, 2), visible (N,)) of the slots, or None without a pose at *t*."""
        frames = self.pose.joint_frames(t)
        if frames is None:
            return None
        placed = {j: self.joint_frame(t, j, frames) for j in set(self._joints)}
        points = np.array([(placed[j] @ np.append(p.offset, 1.0))[:3] for j, p in zip(self._joints, self.placements)])
        state = self.states[cam_id]
        centre = -state.R.T @ state.t.ravel()
        visible = np.ones(len(points), bool)
        for i, (j, p) in enumerate(zip(self._joints, self.placements)):
            if p.normal is not None:
                n = placed[j][:3, :3] @ p.normal
                to_camera = centre - points[i]
                visible[i] = (n @ to_camera) / (np.linalg.norm(n) * np.linalg.norm(to_camera)) >= VISIBLE_COSINE
        rvec = cv2.Rodrigues(state.R)[0]
        project = cv2.fisheye.projectPoints if state.fisheye else cv2.projectPoints
        px, _ = project(points.reshape(-1, 1, 3), rvec, state.t.reshape(3, 1), state.K_orig, state.dist)
        return px.reshape(-1, 2), visible


@dataclass
class TrackletScore:
    cam_id: str
    tracklet_id: int
    first_frame: int
    last_frame: int
    cost: np.ndarray                     # median pixel distance per slot, inf if not visible / no pose
    slot: int | None = None              # index of the assigned slot
    margin: float = 0.0                  # extra cost of the next best slot
    support: str = "alone"               # see check_consistency
    confidence: str = "none"
    frames: dict = field(default_factory=dict, repr=False)


def score_tracklets(
    tracklets: Tracklets, cam_id: str, svid: str, predictor: Predictor, sync_table, n_samples: int = 12,
) -> list[TrackletScore]:
    """The cost of every slot for every tracklet of one camera."""
    out = []
    n_slots = len(predictor.names)
    for tracklet_id, frames in tracklets.items():
        order = sorted(frames)
        sample = order[:: max(1, len(order) // n_samples)]
        per_slot: list[list[float]] = [[] for _ in range(n_slots)]
        hidden = np.zeros(n_slots)
        seen = 0
        for frame in sample:
            t = sync_table.frame_to_global_time(frame, svid)
            predicted = None if t is None else predictor.pixels(t, cam_id)
            if predicted is None:
                continue
            px, visible = predicted
            seen += 1
            distance = np.linalg.norm(px - np.array(frames[frame]), axis=1)
            for s in range(n_slots):
                if visible[s]:
                    per_slot[s].append(float(distance[s]))
                else:
                    hidden[s] += 1
        cost = np.full(n_slots, np.inf)
        if seen:
            for s in range(n_slots):
                if per_slot[s] and len(per_slot[s]) >= MIN_VISIBLE_FRACTION * seen:
                    cost[s] = float(np.median(per_slot[s]))
        out.append(TrackletScore(cam_id, tracklet_id, order[0], order[-1], cost, frames=frames))
    return out


def assign(scores: list[TrackletScore], max_cost: float = MAX_COST_PX) -> None:
    """Give each tracklet its best slot that is still free during its lifetime in its camera,
    most confident first; sets ``slot`` and ``margin`` (the next best slot's extra cost)."""
    taken: dict[tuple[str, int], list[tuple[int, int]]] = {}
    ranked = sorted(scores, key=lambda s: _best_two(s.cost)[0])
    for score in ranked:
        candidates = np.argsort(score.cost)
        for position, slot in enumerate(candidates):
            if not np.isfinite(score.cost[slot]) or score.cost[slot] > max_cost:
                break
            spans = taken.setdefault((score.cam_id, int(slot)), [])
            if any(a <= score.last_frame and score.first_frame <= b for a, b in spans):
                continue
            spans.append((score.first_frame, score.last_frame))
            score.slot = int(slot)
            rest = [c for c in score.cost[candidates[position + 1:]] if np.isfinite(c)]
            score.margin = float(min(rest) - score.cost[slot]) if rest else 99.0
            break


def _best_two(cost: np.ndarray) -> tuple[float, float]:
    finite = np.sort(cost[np.isfinite(cost)])
    if len(finite) == 0:
        return float("inf"), float("inf")
    return float(finite[0]), float(finite[1]) if len(finite) > 1 else float("inf")


# --- refitting the bone poses from confident assignments ----------------------------------

ROTATION_PRIOR_RAD = 0.1      # how far a window's correction of a bone's pose is expected to be
TRANSLATION_PRIOR_M = 0.04
MIN_OBSERVATIONS = 12
MIN_TRACKLETS = 2


def _correction(x: np.ndarray) -> np.ndarray:
    delta = np.eye(4)
    delta[:3, :3] = cv2.Rodrigues(x[:3])[0]
    delta[:3, 3] = x[3:]
    return delta


def refit_bones(
    scores: list[TrackletScore], predictor: Predictor, svid_by_cam: dict, sync_table, *,
    max_cost: float = 45.0, min_margin: float = 10.0, per_tracklet: int = 8,
) -> dict:
    """Refit each bone's pose in each window from the tracklets assigned with confidence.

    A bone's correction is a small rigid transform in its own frame, chosen to bring the
    confidently assigned dots of its slots onto their predictions in all cameras (robust
    loss, pulled towards no correction). It replaces ``predictor.corrections`` and its
    summary is returned: per (joint, window), observations, tracklets and the median
    residual in pixels before and after.
    """
    groups: dict[tuple[str, int], list] = defaultdict(list)
    for score in scores:
        if score.slot is None or score.cost[score.slot] > max_cost or score.margin < min_margin:
            continue
        slot = predictor.placements[score.slot]
        order = sorted(score.frames)
        for frame in order[:: max(1, len(order) // per_tracklet)]:
            t = sync_table.frame_to_global_time(frame, svid_by_cam[score.cam_id])
            if t is not None:
                groups[(slot.joint, predictor.window(t))].append(
                    (score.cam_id, score.tracklet_id, t, np.array(score.frames[frame]), slot.offset))
    corrections, summary = {}, {}
    for (joint, window), items in groups.items():
        if len(items) < MIN_OBSERVATIONS or len({(c, t) for c, t, *_ in items}) < MIN_TRACKLETS:
            continue
        frames = [predictor.pose.joint_frames(t) for _, _, t, _, _ in items]
        keep = [i for i, f in enumerate(frames) if f is not None]
        items, frames = [items[i] for i in keep], [frames[i] for i in keep]
        base = [f[joint] for f in frames]
        offsets = np.array([np.append(o, 1.0) for *_, o in items])
        pixels = np.array([px for *_, px, _ in items])
        cams = [c for c, *_ in items]

        def residual(x, apply=True):
            delta = _correction(x) if apply else np.eye(4)
            world = np.array([(b @ delta @ o)[:3] for b, o in zip(base, offsets)])
            out = np.zeros((len(items), 2))
            for cam in set(cams):
                rows = [i for i, c in enumerate(cams) if c == cam]
                state = predictor.states[cam]
                project = cv2.fisheye.projectPoints if state.fisheye else cv2.projectPoints
                px, _ = project(world[rows].reshape(-1, 1, 3), cv2.Rodrigues(state.R)[0], state.t.reshape(3, 1),
                                state.K_orig, state.dist)
                out[rows] = px.reshape(-1, 2) - pixels[rows]
            return out.ravel()

        def objective(x):
            return np.concatenate([residual(x), x[:3] / ROTATION_PRIOR_RAD * 10.0, x[3:] / TRANSLATION_PRIOR_M * 10.0])

        fit = least_squares(objective, np.zeros(6), loss="soft_l1", f_scale=15.0)
        corrections[(joint, window)] = _correction(fit.x)
        before = np.median(np.hypot(*residual(np.zeros(6), False).reshape(-1, 2).T))
        after = np.median(np.hypot(*residual(fit.x).reshape(-1, 2).T))
        summary[(joint, window)] = (len(items), len({(c, t) for c, t, *_ in items}), float(before), float(after))
    predictor.corrections = corrections
    return summary


# --- cross-camera consistency and confidence ----------------------------------------------

AGREE_PX = 8.0             # two cameras' dots of one marker reproject this close to their triangulation
CONFLICT_PX = 25.0         # ... and this far means they are not the same physical dot
MIN_COMMON_SAMPLES = 4
MAX_SKEW_S = 0.012        # frames of two cameras this close in time show the same moment
CONFIDENT_COST_PX = 45.0
CONFIDENT_MARGIN_PX = 10.0


def check_consistency(scores: list[TrackletScore], predictor: Predictor, svid_by_cam: dict, sync_table) -> None:
    """Set ``support`` (``corroborated``, ``conflict`` or ``alone``) and ``confidence`` of the assigned tracklets.

    Tracklets given the same slot in two cameras while overlapping in time are the same
    physical dot: their rays must meet. They are *corroborated* when they do and in
    *conflict* when they do not (one of them is the wrong slot, or clutter). A tracklet with
    no partner is *alone*. Confidence is ``high`` when the tracklet is near its slot with a
    clear margin and is not in conflict, ``review`` when it is in conflict or ambiguous,
    ``medium`` otherwise.
    """
    by_slot: dict[int, list[TrackletScore]] = defaultdict(list)
    for score in scores:
        score.support = "alone"
        if score.slot is not None:
            by_slot[score.slot].append(score)
    for members in by_slot.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if a.cam_id == b.cam_id:
                    continue
                verdict = _pair_verdict(a, b, predictor, svid_by_cam, sync_table)
                if verdict is None:
                    continue
                for x in (a, b):
                    if verdict == "conflict" or x.support == "alone":
                        x.support = verdict
    for score in scores:
        if score.slot is None:
            score.confidence = "none"
        elif score.support == "conflict" or score.margin < 0.5 * CONFIDENT_MARGIN_PX:
            score.confidence = "review"
        elif (score.cost[score.slot] <= CONFIDENT_COST_PX and score.margin >= CONFIDENT_MARGIN_PX) or score.support == "corroborated":
            score.confidence = "high"
        else:
            score.confidence = "medium"


def _pair_verdict(a: TrackletScore, b: TrackletScore, predictor: Predictor, svid_by_cam: dict, sync_table) -> str | None:
    """corroborated / conflict from the frames both tracklets are alive in, None without enough common time."""
    stamped = [(t, f) for f in sorted(a.frames) if (t := sync_table.frame_to_global_time(f, svid_by_cam[a.cam_id])) is not None]
    if not stamped:
        return None
    times_a = np.array([t for t, _ in stamped])
    errors = []
    for frame_b in sorted(b.frames)[::3]:
        t = sync_table.frame_to_global_time(frame_b, svid_by_cam[b.cam_id])
        if t is None:
            continue
        k = int(np.argmin(np.abs(times_a - t)))
        if abs(times_a[k] - t) > MAX_SKEW_S:
            continue
        frame_a = stamped[k][1]
        pair = [(a.cam_id, a.frames[frame_a]), (b.cam_id, b.frames[frame_b])]
        X = triangulate(pair, predictor.states)
        if X is None:
            continue
        err = 0.0
        for cam, xy in pair:
            state = predictor.states[cam]
            project = cv2.fisheye.projectPoints if state.fisheye else cv2.projectPoints
            px, _ = project(X.reshape(1, 1, 3), cv2.Rodrigues(state.R)[0], state.t.reshape(3, 1), state.K_orig, state.dist)
            err = max(err, float(np.linalg.norm(px.ravel() - np.array(xy))))
        errors.append(err)
    if len(errors) < MIN_COMMON_SAMPLES:
        return None
    median = float(np.median(errors))
    return "corroborated" if median <= AGREE_PX else "conflict" if median >= CONFLICT_PX else None
