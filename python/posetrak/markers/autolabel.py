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

from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml

from posetrak.markers.bones import BodyPose
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
    """Where each slot appears in a camera at a time, from a stored pose."""

    def __init__(self, pose: BodyPose, placements: list[SlotPlacement], states: dict) -> None:
        self.pose, self.placements, self.states = pose, placements, states
        self.names = [p.name for p in placements]
        self._joints = [p.joint for p in placements]

    def pixels(self, t: float, cam_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        """(distorted pixels (N, 2), visible (N,)) of the slots, or None without a pose at *t*."""
        frames = self.pose.joint_frames(t)
        if frames is None:
            return None
        points = np.array([(frames[j] @ np.append(p.offset, 1.0))[:3] for j, p in zip(self._joints, self.placements)])
        state = self.states[cam_id]
        centre = -state.R.T @ state.t.ravel()
        visible = np.ones(len(points), bool)
        for i, (j, p) in enumerate(zip(self._joints, self.placements)):
            if p.normal is not None:
                n = frames[j][:3, :3] @ p.normal
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
