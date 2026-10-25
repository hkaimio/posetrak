# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""tracklets.py — the dot tracklets of a detection run, narrowed to a person's body.

A detection run stores every reflective-dot candidate of every frame with the
id of the tracklet (a dot followed from frame to frame in one camera) it belongs
to. The scene holds many more bright dots than the body markers: plants, glints,
other props. :func:`filter_tracklets` keeps the ones that stay near the person:
inside the person's segmentation mask, and near the tracked skeleton.
"""
from __future__ import annotations

import sqlite3
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from app.pose.db_cache import decode_dot_candidates
from posetrak.db.seg_revisions import mask_at, newest_closed_revision

Tracklets = dict[int, dict[int, tuple[float, float]]]  # {tracklet_id: {video_frame: (x, y)}}


def collect_tracklets(
    conn: sqlite3.Connection, detection_run_id: str, shot_video_id: str, frame_lo: int, frame_hi: int,
) -> Tracklets:
    """The dot tracklets of one video in [frame_lo, frame_hi], in distorted pixels."""
    out: Tracklets = defaultdict(dict)
    rows = conn.execute(
        "SELECT video_frame, keypoints FROM detection_keypoints WHERE detection_run_id = ? "
        "AND shot_video_id = ? AND video_frame BETWEEN ? AND ? AND region_type = 'dots'",
        (detection_run_id, shot_video_id, frame_lo, frame_hi),
    ).fetchall()
    for row in rows:
        for cand in decode_dot_candidates(bytes(row["keypoints"])):
            tracklet_id = int(cand[8])
            if tracklet_id >= 0:
                out[tracklet_id][row["video_frame"]] = (float(cand[0]), float(cand[1]))
    return dict(out)


class SubjectMasks:
    """The mask of one subject label, per video frame, grown by a margin.

    Masks are stored at a fraction of the video's size; positions are given in
    video pixels (the intrinsics calibration's size, *native_size*). For speed the
    mask is reduced to *work_width* pixels across before it is grown.
    """

    def __init__(self, conn: sqlite3.Connection, seg_run_id: str, label: int, margin_px: float,
                 native_size: dict[str, tuple[int, int]], work_width: int = 480,
                 cache_size: int = 64, revision: int | None = None) -> None:
        self._conn, self._run, self._label, self._margin = conn, seg_run_id, label, margin_px
        # Pinned at construction (or given), so a save made meanwhile does not change the masks read.
        self.revision = newest_closed_revision(conn, seg_run_id) if revision is None else revision
        self._native = native_size
        self._work_width = work_width
        self._cache: OrderedDict[tuple[str, int], np.ndarray | None] = OrderedDict()
        self._cache_size = cache_size

    def _grown(self, shot_video_id: str, frame: int) -> np.ndarray | None:
        key = (shot_video_id, frame)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        blob = mask_at(self._conn, self._run, shot_video_id, frame, self.revision)
        grown = None
        if blob is not None:
            mask = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED)
            if mask is not None:
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                width, height = self._native[shot_video_id]
                size = (self._work_width, max(1, round(self._work_width * height / width)))
                small = cv2.resize((mask == self._label).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
                radius = max(1, round(self._margin * self._work_width / width))
                grown = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
        self._cache[key] = grown
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return grown

    def contains(self, shot_video_id: str, frame: int, x: float, y: float) -> bool | None:
        """Whether the point is on the subject; None if there is no mask for the frame."""
        grown = self._grown(shot_video_id, frame)
        if grown is None:
            return None
        width, height = self._native[shot_video_id]
        mx = min(grown.shape[1] - 1, max(0, int(x * grown.shape[1] / width)))
        my = min(grown.shape[0] - 1, max(0, int(y * grown.shape[0] / height)))
        return bool(grown[my, mx])


@dataclass
class FilterStats:
    raw: int = 0
    long_enough: int = 0
    on_subject: int = 0


def filter_tracklets(
    tracklets: Tracklets, shot_video_id: str, masks: SubjectMasks, *,
    min_frames: int = 15, min_on_subject: float = 0.7, frame_step: int = 6,
) -> tuple[Tracklets, FilterStats]:
    """Keep tracklets that last *min_frames* frames and lie on the subject's mask in
    at least *min_on_subject* of the sampled frames that have a mask.

    Every *frame_step*-th frame is looked at: a tracklet barely moves between
    neighbouring frames, so sampling keeps this cheap, and each mask is decoded once.
    """
    stats = FilterStats(raw=len(tracklets))
    long = {tid: frames for tid, frames in tracklets.items() if len(frames) >= min_frames}
    stats.long_enough = len(long)
    by_frame: dict[int, list[tuple[int, tuple[float, float]]]] = defaultdict(list)
    for tid, frames in long.items():
        for frame, xy in frames.items():
            if frame % frame_step == 0:
                by_frame[frame].append((tid, xy))
    hits: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for frame in sorted(by_frame):
        for tid, (x, y) in by_frame[frame]:
            inside = masks.contains(shot_video_id, frame, x, y)
            if inside is not None:
                hits[tid][0] += int(inside)
                hits[tid][1] += 1
    kept = {tid: long[tid] for tid, (hit, total) in hits.items() if total and hit / total >= min_on_subject}
    stats.on_subject = len(kept)
    return kept, stats
