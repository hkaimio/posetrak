# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""kp_gt.py — keypoint-candidate ground-truth metrics.

Library behind ``eval_kp_candidates.py``. Scores the keypoint a pipeline picked per
(camera, frame, keypoint) against operator labels, and tracking runs against a label-fed
reference run. Candidates come through one function, :func:`read_candidates`, which reads
either the ``kp_candidates`` rows of a detection run or a candidate file in the layout of the
prototype labeller, so the metrics do not depend on where the candidates are stored.

Candidate blob (``detection_keypoints.region_type = 'kp_candidates'``, one row per
(run, video, frame, track)): ``int32 n``, ``int32 n_fields`` (6), then ``float32[n, n_fields]``
with the fields :data:`KP_CANDIDATE_FIELDS`. ``x, y`` are distorted pixels; ``rank`` 0 is the
strongest maximum of its keypoint; ``tracklet_id`` is -1 when unlinked. See
docs/roadmap/features/keypoint-candidates/keypoint-candidates-design.md §4.1.

Label export (``labels_*.npz``, columns :data:`LABEL_COLUMNS`): the resolved pick of every
(camera, keypoint, frame) after the operator's edits. A label holds a candidate when ``row`` is
non-negative, and then ``x, y`` is that candidate's position; a keypoint marked not visible or
without a candidate has no position. Only the (range, keypoint) units listed as done in the units
file were reviewed; other rows are automatic proposals and are not ground truth.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from posetrak.detection.kp_candidates import (  # noqa: F401  (the one codec, re-exported for callers)
    KP_CANDIDATE_FIELDS, decode_kp_candidates_blob, encode_kp_candidates_blob,
)

LABEL_COLUMNS = ("cam", "slot", "frame", "state", "row", "x", "y", "tracklet", "source_person")
# Label states of the export that hold a visible keypoint: proposal (agreeing), proposal (uncertain),
# edited: tracklet, proposal next to the neighbouring keypoints.
VISIBLE_STATES = (0, 1, 2, 5)
# The prototype's candidate files store the crop side (1.6 x the person box height) in the frame table.
_CROP_PER_BOX_HEIGHT = 1.6
# Joint names of the SOMA skeleton that belong to the hands' fingers; excluded from body-joint distances.
_FINGER_JOINT_MARKS = ("f_", "thumb")


@dataclass
class Candidates:
    """Struct of arrays, one entry per candidate; ``camera`` indexes ``cameras``."""

    cameras: list[str]
    camera: np.ndarray
    frame: np.ndarray
    keypoint: np.ndarray
    x: np.ndarray
    y: np.ndarray
    peak: np.ndarray
    rank: np.ndarray
    tracklet: np.ndarray
    # (camera, frame) -> person box height in pixels
    box_height: dict[tuple[int, int], float] = field(default_factory=dict)
    # camera -> (frames ascending, global time of each)
    frame_times: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.x)

    def time_of(self, camera: int, frame) -> np.ndarray:
        frames, times = self.frame_times[camera]
        return np.interp(frame, frames, times)


def _from_prototype_file(path) -> Candidates:
    z = np.load(path, allow_pickle=True)
    rows, frames = z["rows"], z["frames"]
    cand = Candidates(
        cameras=[str(c) for c in z["cams"]],
        camera=rows[:, 0].astype(np.int32), frame=rows[:, 1].astype(np.int64), keypoint=rows[:, 2].astype(np.int32),
        x=rows[:, 4], y=rows[:, 5], peak=rows[:, 6], rank=rows[:, 3].astype(np.int32),
        tracklet=z["tracklet"].astype(np.int64) if "tracklet" in z.files else np.full(len(rows), -1, np.int64),
    )
    for cam in np.unique(frames[:, 0]).astype(int):
        sel = frames[frames[:, 0] == cam]
        sel = sel[np.argsort(sel[:, 1])]
        cand.frame_times[cam] = (sel[:, 1], sel[:, 2])
        cand.box_height.update({(cam, int(f)): float(s) / _CROP_PER_BOX_HEIGHT for f, s in zip(sel[:, 1], sel[:, 5])})
    return cand


def _from_database(conn: sqlite3.Connection, run_id: str, track_ids, sync) -> Candidates:
    where, args = "detection_run_id = ? AND region_type = 'kp_candidates'", [run_id]
    if track_ids is not None:
        where += f" AND track_id IN ({','.join('?' * len(track_ids))})"
        args += list(track_ids)
    videos = sorted(r[0] for r in conn.execute(
        f"SELECT DISTINCT shot_video_id FROM detection_keypoints WHERE {where}", args))
    index = {v: i for i, v in enumerate(videos)}
    cols: list[list[np.ndarray]] = [[] for _ in range(8)]
    frames_by_cam: dict[int, set[int]] = {}
    for video, frame, blob in conn.execute(
            f"SELECT shot_video_id, video_frame, keypoints FROM detection_keypoints WHERE {where}", args):
        t = decode_kp_candidates_blob(bytes(blob))
        cam = index[video]
        frames_by_cam.setdefault(cam, set()).add(frame)
        for i, column in enumerate([np.full(len(t), cam), np.full(len(t), frame), t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 4], t[:, 5]]):
            cols[i].append(column)
    cat = [np.concatenate(c) if c else np.empty(0) for c in cols]
    cand = Candidates(videos, cat[0].astype(np.int32), cat[1].astype(np.int64), cat[2].astype(np.int32),
                      cat[3], cat[4], cat[5], cat[6].astype(np.int32), cat[7].astype(np.int64))
    for video, frame, h in conn.execute(
            f"SELECT shot_video_id, video_frame, MAX(bbox_h) FROM person_detections WHERE detection_run_id = ? "
            f"AND region_type = 'full_body' GROUP BY shot_video_id, video_frame", [run_id]):
        if video in index:
            cand.box_height[(index[video], frame)] = float(h)
    if sync is not None:
        for cam, frames in frames_by_cam.items():
            fr = np.array(sorted(frames))
            cand.frame_times[cam] = (fr, np.array([sync.frame_to_global_time(int(f), videos[cam]) for f in fr]))
    return cand


def reproduction(reference: Candidates, other: Candidates, camera_map: dict[int, int], tol_px: float, *,
                 frames: set[tuple[int, int]] | None = None) -> dict[str, float]:
    """Share of the *reference* candidates that *other* reproduces within *tol_px*, and the reverse share.

    A candidate is reproduced when *other* has a candidate of the same keypoint, camera and frame that close.
    ``camera_map`` maps a reference camera index to the other's; only frames present in both are compared
    (``frames`` restricts them further to (reference camera, frame) pairs). Returns the two shares and the
    counts compared.
    """
    from scipy.spatial import cKDTree

    other_cam = {v: k for k, v in camera_map.items()}
    ref_frames = {(int(c), int(f)) for c, f in zip(reference.camera, reference.frame)}
    oth_frames = {(other_cam[int(c)], int(f)) for c, f in zip(other.camera, other.frame) if int(c) in other_cam}
    common = ref_frames & oth_frames if frames is None else ref_frames & oth_frames & frames

    def points(c: Candidates, cams: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keep = np.array([(int(a), int(b)) in common for a, b in zip(cams, c.frame)], bool)
        key = (cams[keep].astype(np.int64) * 1_000_000 + c.frame[keep]) * 100 + c.keypoint[keep]
        return np.column_stack([key * 1000.0, c.x[keep], c.y[keep]]), keep

    ref_pts, ref_keep = points(reference, reference.camera.astype(np.int64))
    oth_cams = np.array([other_cam.get(int(c), -1) for c in other.camera], np.int64)
    oth_pts, _ = points(other, oth_cams)
    if not len(ref_pts) or not len(oth_pts):
        return {"reference_reproduced": float("nan"), "other_reproduced": float("nan"), "n_reference": 0, "n_other": 0}
    d_ref = cKDTree(oth_pts).query(ref_pts)[0]
    d_oth = cKDTree(ref_pts).query(oth_pts)[0]
    return {"reference_reproduced": float((d_ref <= tol_px).mean()), "other_reproduced": float((d_oth <= tol_px).mean()),
            "n_reference": int(len(ref_pts)), "n_other": int(len(oth_pts))}


def read_candidates(source, *, run_id: str | None = None, track_ids=None, sync=None) -> Candidates:
    """Read keypoint candidates from one source.

    ``source`` is a path to a prototype candidate file (``.npz``), or an open session database
    connection together with ``run_id`` (the detection run whose ``kp_candidates`` rows to read;
    ``track_ids`` restricts the persons). ``sync`` is an optional object with
    ``frame_to_global_time(frame, video_id)`` that fills the frame times of database candidates.
    """
    if isinstance(source, sqlite3.Connection):
        if run_id is None:
            raise ValueError("reading from a database needs run_id")
        return _from_database(source, run_id, track_ids, sync)
    return _from_prototype_file(Path(source))


@dataclass
class Labels:
    """Resolved operator picks of the labelled units."""

    camera: np.ndarray
    keypoint: np.ndarray
    frame: np.ndarray
    visible: np.ndarray  # the keypoint has a labelled position
    holds_candidate: np.ndarray  # the position is that of a candidate (a tracklet edit may place it elsewhere)
    x: np.ndarray
    y: np.ndarray


def read_labels(resolved_npz, units_json, unit_key: str) -> Labels:
    """Rows of the label export that belong to the units marked done for ``unit_key`` (``"43.5-47.5"``)."""
    z = np.load(resolved_npz, allow_pickle=True)
    r = z["rows"]
    done = {int(d.split("|")[1]) for d in json.loads(Path(units_json).read_text())["done"] if d.split("|")[0] == unit_key}
    keep = np.isin(r[:, 1].astype(int), sorted(done))
    r = r[keep]
    visible = np.isin(r[:, 3].astype(int), VISIBLE_STATES) & np.isfinite(r[:, 5]) & np.isfinite(r[:, 6])
    return Labels(r[:, 0].astype(np.int32), r[:, 1].astype(np.int32), r[:, 2].astype(np.int64), visible, r[:, 4] >= 0, r[:, 5], r[:, 6])


@dataclass
class Picks:
    """One picked position per (camera, frame, keypoint)."""

    camera: np.ndarray
    frame: np.ndarray
    keypoint: np.ndarray
    x: np.ndarray
    y: np.ndarray


def strongest_picks(cand: Candidates) -> Picks:
    """The strongest maximum of every keypoint, the choice a pipeline makes without candidates."""
    s = cand.rank == 0
    return Picks(cand.camera[s], cand.frame[s], cand.keypoint[s], cand.x[s], cand.y[s])


def read_sequence_picks(conn: sqlite3.Connection, sequence_id: str, person_id: int, cameras: list[str], source: str = "body") -> Picks:
    """Picks stored in a pose observation sequence (what the tracker is fed).

    ``cameras`` lists the camera instance ids in the order of ``Candidates.camera``; rows of other cameras are skipped.
    """
    index = {c: i for i, c in enumerate(cameras)}
    cam, frame, kp, x, y = [], [], [], [], []
    for camera_id, video_frame, blob in conn.execute(
            "SELECT camera_instance_id, video_frame, kp_blob FROM pose_observations "
            "WHERE sequence_id = ? AND person_id = ? AND source = ?", (sequence_id, person_id, source)):
        if camera_id not in index:
            continue
        t = np.frombuffer(bytes(blob), "<f4").reshape(-1, 3)
        ok = np.isfinite(t[:, 0]) & np.isfinite(t[:, 1])
        n = int(ok.sum())
        cam.append(np.full(n, index[camera_id]))
        frame.append(np.full(n, video_frame))
        kp.append(np.flatnonzero(ok))
        x.append(t[ok, 0])
        y.append(t[ok, 1])
    cat = lambda parts, dtype: np.concatenate(parts).astype(dtype) if parts else np.empty(0, dtype)
    return Picks(cat(cam, np.int32), cat(frame, np.int64), cat(kp, np.int32), cat(x, np.float32), cat(y, np.float32))


def _key(camera, frame, keypoint) -> np.ndarray:
    return (np.asarray(camera, np.int64) * 10_000_000 + np.asarray(frame, np.int64)) * 128 + np.asarray(keypoint, np.int64)


def pick_metrics(
    picks: Picks, labels: Labels, cand: Candidates, *, t0: float, t1: float,
    tolerance: float = 0.03, same_px: float = 0.5, skip_cameras=(), windows=None,
) -> dict[str, float]:
    """Accuracy of ``picks`` against ``labels`` on the labels with a time in [t0, t1].

    - ``within_tol``: share of visible labels whose pick lies within ``tolerance`` x box height of the label.
    - ``same_candidate``: share of visible labels whose pick is the candidate the label holds (position within
      ``same_px`` pixels of the label's candidate).
    - ``no_pick``: share of visible labels without a pick.
    - ``false_pick``: share of not-visible labels that still have a pick.
    ``windows`` is an optional list of (t0, t1); with it the result covers only labels inside those windows.
    """
    sel = ~np.isin(labels.camera, list(skip_cameras))
    t = np.full(len(labels.frame), np.nan)
    for cam in np.unique(labels.camera[sel]):
        m = sel & (labels.camera == cam)
        t[m] = cand.time_of(int(cam), labels.frame[m])
    inside = (t >= t0) & (t <= t1)
    if windows is not None:
        inside &= np.any([(t >= a) & (t <= b) for a, b in windows], axis=0)
    sel &= inside
    order = np.argsort(_key(picks.camera, picks.frame, picks.keypoint))
    pk = _key(picks.camera, picks.frame, picks.keypoint)[order]
    lk = _key(labels.camera, labels.frame, labels.keypoint)
    pos = np.clip(np.searchsorted(pk, lk), 0, max(len(pk) - 1, 0))
    has_pick = (pk[pos] == lk) if len(pk) else np.zeros(len(lk), bool)
    px, py = picks.x[order][pos] if len(pk) else lk * np.nan, picks.y[order][pos] if len(pk) else lk * np.nan
    height = np.array([cand.box_height.get((int(c), int(f)), np.nan) for c, f in zip(labels.camera, labels.frame)])
    dist = np.hypot(px - labels.x, py - labels.y)
    vis, invis = sel & labels.visible, sel & ~labels.visible
    n_vis, n_invis = max(int(vis.sum()), 1), max(int(invis.sum()), 1)
    return {
        "n_visible": int(vis.sum()),
        "within_tol": float((vis & has_pick & (dist <= tolerance * height)).sum() / n_vis),
        "same_candidate": float((vis & labels.holds_candidate & has_pick & (dist <= same_px)).sum() / n_vis),
        "no_pick": float((vis & ~has_pick).sum() / n_vis),
        "n_invisible": int(invis.sum()),
        "false_pick": float((invis & has_pick).sum() / n_invis),
    }


class _Poses:
    """Joint positions of a tracking run; wraps ``posetrak.markers.bones.BodyPose`` (imported lazily)."""

    def __init__(self, conn: sqlite3.Connection, run_prefix: str) -> None:
        from posetrak.markers.bones import BodyPose

        row = conn.execute("SELECT id FROM tracking_runs WHERE id LIKE ?", (run_prefix + "%",)).fetchone()
        if row is None:
            raise KeyError(f"no tracking run starting with {run_prefix!r}")
        self.pose = BodyPose(conn, row[0])

    def joints(self, t: float, parts: tuple[str, ...], body_only: bool) -> dict[str, np.ndarray]:
        frames = self.pose.joint_frames(float(t))
        if frames is None:
            return {}
        return {n: f[:3, 3] for n, f in frames.items()
                if any(p in n for p in parts) and not (body_only and any(m in n for m in _FINGER_JOINT_MARKS))}


def joint_distances(
    conn: sqlite3.Connection, run: str, reference: str, times, *, parts: tuple[str, ...] = ("",), body_only: bool = True,
) -> np.ndarray:
    """Distance (cm) between the same joint of two tracking runs, every joint and time, pooled; NaN-free.

    ``run`` and ``reference`` are tracking run ids or id prefixes. ``parts`` keeps joints whose name contains
    one of the strings; ``body_only`` drops finger joints. Times where either run has no state are skipped.
    """
    a, b = _Poses(conn, run), _Poses(conn, reference)
    out = []
    for t in times:
        ja, jb = a.joints(t, parts, body_only), b.joints(t, parts, body_only)
        out += [100 * float(np.linalg.norm(ja[n] - jb[n])) for n in ja if n in jb]
    return np.array(out)


def tracker_comparison(conn, run, reference, windows, *, step_s: float = 0.05) -> dict[str, float]:
    """Mean, median and 90th percentile body-joint distance (cm) to the label-fed reference over ``windows``."""
    times = np.concatenate([np.arange(a, b, step_s) for a, b in windows])
    d = joint_distances(conn, run, reference, times)
    if len(d) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {"n": len(d), "mean": float(d.mean()), "median": float(np.median(d)), "p90": float(np.percentile(d, 90))}


EVENT_PARTS = {"legs": ("thigh", "shin", "foot", "toe"), "arms": ("shoulder", "upper_arm", "forearm", "hand", "f_", "thumb"), "whole": ("",)}


def event_distance(conn, run, other, event: dict, *, step_s: float = 0.05) -> float:
    """Largest joint distance (cm) of the event's body part between two tracking runs inside the event window."""
    times = np.arange(event["t0"], event["t1"], step_s)
    d = joint_distances(conn, run, other, times, parts=EVENT_PARTS[event["part"]], body_only=False)
    return float(d.max()) if len(d) else float("nan")


def events_flagged(distances: dict[str, float], threshold_cm: float = 40.0) -> list[str]:
    """Ids of the events whose largest distance reaches the failure-flag threshold."""
    return [k for k, v in distances.items() if v >= threshold_cm]
