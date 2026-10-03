# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""refit.py — refit marker offsets on their bones from human-confirmed dot tracklets.

Each frame of a tracklet labelled as a slot puts the marker on a ray through a camera's
centre. With the bone moving, the offset in the bone's frame is the point closest to the rays
of all cameras and times; the fit pulls towards the slot's earlier offset with a small weight
and drops observations that are far from the fit (a stored pose that is off at that moment).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from posetrak.markers.bones import BodyPose, _unit_rays

MIN_OBSERVATIONS = 30
MAX_DROP_PX = 15.0           # an observation this far or 3 x the median from the fit is left out
ITERATIONS = 4


@dataclass
class Refit:
    offset: np.ndarray       # bone frame, metres
    n: int                   # observations kept
    rms_px: float            # of the kept observations
    moved_m: float           # distance from the prior offset


def sample_frames(frames: list[int], per_segment: int) -> list[int]:
    return frames[:: max(1, len(frames) // per_segment)]


def refit_offset(obs: list[tuple[float, str, tuple[float, float]]], pose: BodyPose, joint: str, states: dict,
                 prior: np.ndarray, prior_weight: float = 1.0) -> Refit | None:
    """The offset of one marker in *joint*'s frame from observations (time, camera id, distorted pixel)."""
    rows = []
    for t, cam, xy in obs:
        F = pose.joint_frames(t)
        if F is None or joint not in F:
            continue
        state = states[cam]
        ray = _unit_rays(np.array([xy], float), state)[0]
        rows.append((F[joint], -state.R.T @ state.t.ravel(), ray, float(state.K_orig[0, 0])))
    if len(rows) < MIN_OBSERVATIONS:
        return None
    keep = np.ones(len(rows), bool)
    offset = prior.copy()
    for _ in range(ITERATIONS):
        A = prior_weight * 1e-2 * np.eye(3)
        b = prior_weight * 1e-2 * prior
        for k, (F, centre, ray, _) in zip(keep, rows):
            if k:
                R, c = F[:3, :3], F[:3, 3]
                P = R.T @ (np.eye(3) - np.outer(ray, ray)) @ R
                A += P
                b += R.T @ (np.eye(3) - np.outer(ray, ray)) @ (centre - c)
        offset = np.linalg.solve(A, b)
        err = np.array([_error_px(F, centre, ray, f, offset) for F, centre, ray, f in rows])
        limit = max(3.0 * float(np.median(err[keep])), MAX_DROP_PX)
        keep = err <= limit
        if keep.sum() < MIN_OBSERVATIONS:
            return None
    return Refit(offset, int(keep.sum()), float(np.sqrt(np.mean(err[keep] ** 2))), float(np.linalg.norm(offset - prior)))


def _error_px(F, centre, ray, focal_px, offset) -> float:
    v = (F @ np.append(offset, 1.0))[:3] - centre
    angle = np.arccos(np.clip(v @ ray / np.linalg.norm(v), -1, 1))
    return float(focal_px * np.tan(angle))


def refit_attachment(conn, shot_id: str, detection_run: str, pose_run: str, truth_path: str, attachment_in: str,
                     attachment_out: str, *, max_move_cm: float = 9.0, max_rms_px: float = 30.0,
                     before_s: float | None = None, per_segment: int = 40) -> dict:
    """Write *attachment_out*: *attachment_in* with the offsets refitted from the frozen ground truth.

    A refit that moves a marker more than *max_move_cm* from its earlier offset, or fits worse than
    *max_rms_px*, is not trusted (the stored pose is then probably what is off) and the marker keeps its
    offset. *before_s* limits the fit to tracklets starting before that time, to keep the rest as a
    held-out test. Returns {marker: Refit} of the markers that changed.
    """
    import collections

    import yaml

    from posetrak.calibration.session_cameras import load_camera_states, load_sync_table
    from posetrak.markers.attachment import bone_length
    from posetrak.markers.evaluate import load_frozen
    from posetrak.markers.review import NOT_A_MARKER, OTHER_SUBJECT
    from posetrak.markers.tracklets import collect_tracklets

    states = load_camera_states(conn, shot_id)
    sync, svid_by_cam = load_sync_table(conn, shot_id)
    pose = BodyPose(conn, pose_run)
    doc = yaml.safe_load(open(attachment_in, encoding="utf-8"))
    markers = {m["name"]: m for m in doc["markers"]}
    truth = load_frozen(truth_path)
    tracklets = {}
    for cam in {t[0] for t in truth}:
        lo, hi = (f(t[i] for t in truth if t[0] == cam) for f, i in ((min, 2), (max, 3)))
        tracklets[cam] = collect_tracklets(conn, detection_run, svid_by_cam[cam], lo, hi)
    obs = collections.defaultdict(list)
    for cam, tid, first, last, label in truth:
        if label in (NOT_A_MARKER, OTHER_SUBJECT) or label not in markers:
            continue
        for f in sample_frames(sorted(f for f in tracklets[cam].get(tid, {}) if first <= f <= last), per_segment):
            t = sync.frame_to_global_time(f, svid_by_cam[cam])
            if t is not None and (before_s is None or t < before_s):
                obs[label].append((t, cam, tracklets[cam][tid][f]))
    changed = {}
    for m in doc["markers"]:
        fit = refit_offset(obs[m["name"]], pose, m["parent_joint"], states, np.array(m["offset"])) if obs.get(m["name"]) else None
        if fit is None or fit.moved_m > max_move_cm / 100 or fit.rms_px > max_rms_px:
            continue
        length = bone_length(pose, m["parent_joint"])
        m.update({"offset": [round(float(v), 4) for v in fit.offset], "along": round(float(fit.offset[1] / length), 4),
                  "lateral": round(float(fit.offset[0]), 4), "anterior": round(float(fit.offset[2]), 4),
                  "refit": {"n_observations": fit.n, "rms_px": round(fit.rms_px, 1), "moved_cm": round(100 * fit.moved_m, 1)}})
        changed[m["name"]] = fit
    doc.setdefault("calibration", {})["ground_truth_refit"] = {
        "truth": str(truth_path), "pose_run": pose_run, "max_move_cm": max_move_cm, "max_rms_px": max_rms_px,
        "fitted_before_s": before_s,
        "method": "offsets refitted by least squares on camera rays of human-labelled tracklets in the bone's frame; "
                  "markers whose refit moved too far or fitted poorly keep their earlier offset",
    }
    with open(attachment_out, "w", encoding="utf-8") as f:
        f.write("# SPDX-FileCopyrightText: 2026 Harri Kaimio\n#\n# SPDX-License-Identifier: Apache-2.0\n\n")
        yaml.safe_dump(doc, f, sort_keys=False)
    return changed


def median_attachment(conn, shot_id: str, detection_run: str, pose_run: str, truth_path: str, attachment_in: str,
                      attachment_out: str, *, min_samples: int = 30, before_s: float | None = None) -> dict:
    """Write *attachment_out*: *attachment_in* with offsets set to where the labelled dots sit.

    The dots of a slot that two or more cameras saw are triangulated; the offset becomes their median
    position in the slot's bone frame *of the given pose run*. That is the offset that predicts the dots
    from that pose, whatever the pose's own bias or the marker's attachment asymmetry; slots with fewer than
    *min_samples* triangulated samples keep their offset. Returns {marker: (offset, samples, spread_m)}.
    """
    import collections

    import yaml

    from posetrak.calibration.session_cameras import load_camera_states, load_sync_table
    from posetrak.markers.attachment import bone_length
    from posetrak.markers.bones import marker_observations, marker_track
    from posetrak.markers.evaluate import load_frozen
    from posetrak.markers.tracklets import collect_tracklets

    states = load_camera_states(conn, shot_id)
    sync, svid_by_cam = load_sync_table(conn, shot_id)
    pose = BodyPose(conn, pose_run)
    doc = yaml.safe_load(open(attachment_in, encoding="utf-8"))
    truth = [r for r in load_frozen(truth_path)
             if before_s is None or sync.frame_to_global_time(r[2], svid_by_cam[r[0]]) < before_s]
    tracklets = {cam: collect_tracklets(conn, detection_run, svid_by_cam[cam], min(r[2] for r in truth if r[0] == cam),
                                        max(r[3] for r in truth if r[0] == cam)) for cam in {r[0] for r in truth}}
    changed = {}
    for m in doc["markers"]:
        rows = [r for r in truth if r[4] == m["name"]]
        members = [(r[0], r[1]) for r in rows]
        sub = {cam: {tid: {f: xy for f, xy in tracklets[cam][tid].items()
                           if any(f0 <= f <= f1 for c2, t2, f0, f1, _ in rows if c2 == cam and t2 == tid)}
                     for c2, tid in members if c2 == cam} for cam in tracklets}
        times, points = marker_track(marker_observations(members, sub, sync, svid_by_cam), states) if rows else ([], [])
        local = []
        for t, X in zip(times, points):
            frames = pose.joint_frames(t)
            if frames is not None and m["parent_joint"] in frames:
                local.append((np.linalg.inv(frames[m["parent_joint"]]) @ np.append(X, 1.0))[:3])
        if len(local) < min_samples:
            continue
        local = np.array(local)
        offset, spread = np.median(local, 0), float(np.linalg.norm(local.std(0)))
        length = bone_length(pose, m["parent_joint"])
        m.update({"offset": [round(float(v), 4) for v in offset], "along": round(float(offset[1] / length), 4),
                  "lateral": round(float(offset[0]), 4), "anterior": round(float(offset[2]), 4),
                  "median_fit": {"n_samples": len(local), "spread_m": round(spread, 3)}})
        changed[m["name"]] = (offset, len(local), spread)
    doc.setdefault("calibration", {})["ground_truth_median_fit"] = {
        "truth": str(truth_path), "pose_run": pose_run, "min_samples": min_samples, "fitted_before_s": before_s,
        "method": "median position of the triangulated labelled dots in the bone frame of the pose run; "
                  "offsets are expressed in that pose's frames",
    }
    with open(attachment_out, "w", encoding="utf-8") as f:
        f.write("# SPDX-FileCopyrightText: 2026 Harri Kaimio\n#\n# SPDX-License-Identifier: Apache-2.0\n\n")
        yaml.safe_dump(doc, f, sort_keys=False)
    return changed
