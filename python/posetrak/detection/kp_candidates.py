# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""kp_candidates.py — several heatmap maxima per keypoint, their storage format and their online linking.

A heatmap pose model has more than one local maximum per keypoint; the strongest is the ordinary pick, the others
are stored as *candidates* (``detection_keypoints.region_type = 'kp_candidates'``) so that the tracker can choose
among them with its prediction, the other cameras and the person masks. Design:
docs/roadmap/features/keypoint-candidates/keypoint-candidates-design.md.

Blob of one (run, video, frame, track) row, little endian: ``int32 n``, ``int32 n_fields`` (6), then
``float32[n, n_fields]`` with the fields :data:`KP_CANDIDATE_FIELDS`. The C++ reader
(``decode_kp_candidates`` in ``cpp/include/posetrak/io/kp_candidates_blob.hpp``) reads the same bytes.

Linking is causal: a candidate's tracklet is decided from earlier frames only, so a row is written once and never
changed. The per-person constant-velocity linker below is not :class:`~posetrak.detection.dot_tracklet.MotionGatedLinker`:
that one gates in absolute pixels (a fixed measurement standard deviation and a fixed 25 px birth gate) and refuses to
commit when two candidates are nearly tied, which is the normal situation for the neighbouring maxima of one
keypoint; here the gate is a share of the person's box so that it follows the person's apparent size, and the
nearest-in-cost match is always taken.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

KP_CANDIDATE_FIELDS = ("keypoint_idx", "x", "y", "peak", "rank", "tracklet_id")
_UNREACHABLE = 1e6


@dataclass(frozen=True)
class CandidateOptions:
    """The ``config_json.candidates`` object of a detection run (design §4.1)."""

    max_per_keypoint: int = 6
    min_peak: float = 0.05
    nms_cells: int = 5
    max_gap_frames: int = 3
    gate_box_frac: float = 0.04

    @classmethod
    def from_config(cls, config: dict | None) -> CandidateOptions | None:
        """The options of a run's ``config_json`` (None when the run has no candidates)."""
        block = (config or {}).get("candidates")
        if block is None:
            return None
        link = block.get("link", {})
        defaults = cls()
        return cls(
            max_per_keypoint=int(block.get("max_per_keypoint", defaults.max_per_keypoint)),
            min_peak=float(block.get("min_peak", defaults.min_peak)),
            nms_cells=int(block.get("nms_cells", defaults.nms_cells)),
            max_gap_frames=int(link.get("max_gap_frames", defaults.max_gap_frames)),
            gate_box_frac=float(link.get("gate_box_frac", defaults.gate_box_frac)),
        )

    def to_config(self) -> dict:
        d = asdict(self)
        return {"max_per_keypoint": d["max_per_keypoint"], "min_peak": d["min_peak"], "nms_cells": d["nms_cells"],
                "link": {"max_gap_frames": d["max_gap_frames"], "gate_box_frac": d["gate_box_frac"]}}


# -- blob -------------------------------------------------------------------

def encode_kp_candidates_blob(keypoint_idx, x, y, peak, rank, tracklet_id) -> bytes:
    """Pack the candidates of one (video, frame, track) into the ``kp_candidates`` blob."""
    table = np.column_stack([keypoint_idx, x, y, peak, rank, tracklet_id]).astype("<f4")
    return np.array([len(table), len(KP_CANDIDATE_FIELDS)], "<i4").tobytes() + table.tobytes()


def decode_kp_candidates_blob(blob: bytes) -> np.ndarray:
    """Unpack a ``kp_candidates`` blob to ``float32[n, n_fields]``.

    The field count is read from the header, so a blob with more fields than this module knows still decodes; the
    first six columns are always :data:`KP_CANDIDATE_FIELDS`.
    """
    n, n_fields = np.frombuffer(blob, "<i4", 2)
    if n_fields < len(KP_CANDIDATE_FIELDS):
        raise ValueError(f"kp_candidates blob has {n_fields} fields, expected at least {len(KP_CANDIDATE_FIELDS)}")
    return np.frombuffer(blob, "<f4", n * n_fields, offset=8).reshape(n, n_fields)


# -- extraction -------------------------------------------------------------

def heatmap_maxima(heat: np.ndarray, options: CandidateOptions) -> list[tuple[int, int, float, float, float]]:
    """Local maxima of heatmaps [K, H, W] as ``(keypoint, rank, cell_x, cell_y, peak)``.

    A maximum is a cell equal to the largest in its ``nms_cells`` square and at least ``min_peak``; at most
    ``max_per_keypoint`` per keypoint, strongest first (rank 0). The position is the cell index refined by a parabola
    through the logarithm of the peak and its neighbours on each axis (as ``backends_soma77.decode`` does), in cells
    with the cell index at the cell's left edge.
    """
    k, h, w = heat.shape
    is_max = (heat == maximum_filter(heat, size=(1, options.nms_cells, options.nms_cells))) & (heat >= options.min_peak)
    logs = np.log(np.maximum(heat, 1e-4))
    out = []
    for kp in range(k):
        ys, xs = np.nonzero(is_max[kp])
        order = np.argsort(-heat[kp, ys, xs], kind="stable")[:options.max_per_keypoint]
        for rank, i in enumerate(order):
            x, y = int(xs[i]), int(ys[i])
            fx, fy = float(x), float(y)
            if 0 < x < w - 1:
                before, peak, after = logs[kp, y, x - 1], logs[kp, y, x], logs[kp, y, x + 1]
                curvature = before - 2 * peak + after
                if curvature < 0:
                    fx += float(np.clip(0.5 * (before - after) / curvature, -0.5, 0.5))
            if 0 < y < h - 1:
                before, peak, after = logs[kp, y - 1, x], logs[kp, y, x], logs[kp, y + 1, x]
                curvature = before - 2 * peak + after
                if curvature < 0:
                    fy += float(np.clip(0.5 * (before - after) / curvature, -0.5, 0.5))
            out.append((kp, rank, fx, fy, float(heat[kp, y, x])))
    return out


def extract_candidates(
    heat: np.ndarray, options: CandidateOptions, cells_to_frame: Callable
) -> np.ndarray:
    """float32 [n, 5] ``keypoint_idx, x, y, peak, rank`` of the maxima of *heat*, in frame pixels.

    *cells_to_frame* maps (cell_x, cell_y) arrays to frame pixels (``backends_soma77.cells_to_frame`` with the
    crop bound).
    """
    maxima = heatmap_maxima(heat, options)
    if not maxima:
        return np.empty((0, 5), np.float32)
    kp, rank, fx, fy, peak = (np.array(c) for c in zip(*maxima))
    x, y = cells_to_frame(fx, fy)
    return np.column_stack([kp, x, y, peak, rank]).astype(np.float32)


# -- online linking ---------------------------------------------------------

@dataclass
class TrackletIds:
    """Source of tracklet ids, unique within a (detection run, camera): one per camera, shared by all persons."""

    next_id: int = 0

    def take(self) -> int:
        self.next_id += 1
        return self.next_id - 1


@dataclass
class _Open:
    id: int
    last: int
    pos: np.ndarray
    vel: np.ndarray


@dataclass
class PersonLinker:
    """Constant-velocity linking of one person's candidates in one camera, per keypoint, frame by frame.

    Each keypoint has its own open tracklets. A candidate continues a tracklet when it lies within the gate
    ``gate_box_frac * box_size`` of the tracklet's predicted position; the gate grows by half of itself for each
    frame of gap, and a tracklet unseen for more than ``max_gap_frames`` frames is closed. Matching is the Hungarian
    optimum over the keypoint's candidates and open tracklets; an unmatched candidate opens a tracklet.
    """

    ids: TrackletIds
    max_gap_frames: int = 3
    gate_box_frac: float = 0.04
    _open: dict[int, list[_Open]] = field(default_factory=dict)

    def link(self, frame: int, table: np.ndarray, box_size: float) -> np.ndarray:
        """Tracklet ids [n] for the candidates *table* (columns ``keypoint_idx, x, y, ...``) of *frame*.

        Frames must arrive in increasing order. *box_size* is the longer side of the person's box in pixels.
        """
        out = np.full(len(table), -1, np.int64)
        if not len(table):
            return out
        gate = self.gate_box_frac * box_size
        keypoints = table[:, 0].astype(int)
        for kp in np.unique(keypoints):
            idx = np.nonzero(keypoints == kp)[0]
            pts = table[idx, 1:3].astype(np.float64)
            active = [a for a in self._open.get(int(kp), []) if frame - a.last <= self.max_gap_frames + 1]
            free = list(range(len(idx)))
            if active:
                gaps = np.array([frame - a.last for a in active])[:, None]
                pred = np.array([a.pos + a.vel * (frame - a.last) for a in active])
                dist = np.hypot(pred[:, None, 0] - pts[None, :, 0], pred[:, None, 1] - pts[None, :, 1])
                cost = np.where(dist < gate * (1 + (gaps - 1) * 0.5), dist, _UNREACHABLE)
                matched = set()
                for a_i, c_i in zip(*linear_sum_assignment(cost)):
                    if cost[a_i, c_i] < _UNREACHABLE:
                        a = active[a_i]
                        dt = frame - a.last
                        a.vel = 0.5 * a.vel + 0.5 * (pts[c_i] - a.pos) / dt
                        a.pos, a.last = pts[c_i], frame
                        out[idx[c_i]] = a.id
                        matched.add(c_i)
                free = [c for c in range(len(idx)) if c not in matched]
            for c in free:
                new = _Open(self.ids.take(), frame, pts[c], np.zeros(2))
                active.append(new)
                out[idx[c]] = new.id
            self._open[int(kp)] = active
        return out
