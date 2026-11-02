# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Measurement B: is a keypoint hidden behind a Body Pose surface, and does that predict bad labels and picks?

Surfaces are capsules around the Body Pose skeleton (the run stored joints and body parameters, not meshes): one
capsule per bone with a radius by body part, plus a pelvis bar and a shoulder bar. A keypoint is hidden when, along
the camera ray to its joint, a capsule of any subject of the frame (own body: only bones not incident to the
joint) lies in front of the joint. The continuous score is the deepest coverage 1 - d / r over the capsules in front.
Targets: the label marks the keypoint not visible; the windowed choice's pick lies more than 0.10 box heights from
the label. The comparison predictor is the heatmap peak of the pick (of the strongest candidate for not-visible).

    python bodypose_occlusion.py --db session.db --root kp_eval_set_v1 --bodypose out --out occlusion.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bodypose_common as bc  # noqa: E402
import bodypose_cue as cue  # noqa: E402
from bodypose_common import kp_gt  # noqa: E402

T0, T1 = 51.7, 56.8
PARENTS = json.load(open("D:/mocap/models/gem-x/soma77_joints.json"))["parents"]
FRONT_MARGIN = 0.5   # the capsule's closest approach must lie this many radii in front of the joint


def capsule_radii():
    r = np.full(77, 0.04)
    r[[1, 2, 3]] = 0.14
    r[[4, 5, 6]] = 0.05
    r[7] = 0.10
    r[[8, 9, 10]] = 0.04
    for left in (0, 28):
        r[11 + left] = 0.06     # clavicle
        r[12 + left] = 0.05     # upper arm (shoulder -> elbow)
        r[13 + left] = 0.04     # forearm
        r[14 + left] = 0.04     # hand
        r[15 + left:39 + left - (0)] = 0.012  # fingers (overwritten for the other side below)
    r[11:15] = [0.06, 0.05, 0.04, 0.04]
    r[39:43] = [0.06, 0.05, 0.04, 0.04]
    r[15:39] = 0.012
    r[43:67] = 0.012
    for first in (67, 72):
        r[first:first + 5] = [0.08, 0.075, 0.05, 0.045, 0.03]
    return r


RADII = capsule_radii()
INCIDENT = [{j} | {c for c in range(1, 77) if PARENTS[c] == j} for j in range(77)]   # bones touching joint j


def segment_distance(P, A, B):
    """Closest distance and ray parameter between segments [0, P] (n x 3) and [A, B] (m x 3) -> (n, m) arrays."""
    d1 = P[:, None, :]
    d2 = (B - A)[None, :, :]
    r = -A[None, :, :]
    a = (d1 * d1).sum(-1)
    e = (d2 * d2).sum(-1)
    f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1)
    b = (d1 * d2).sum(-1)
    denom = a * e - b * b
    s = np.where(denom > 1e-9, np.clip((b * f - c * e) / np.where(denom > 1e-9, denom, 1), 0, 1), 0.0)
    t = (b * s + f) / np.where(e > 1e-12, e, 1)
    low, high = t < 0, t > 1
    t = np.clip(t, 0, 1)
    s = np.where(low, np.clip(-c / np.where(a > 1e-12, a, 1), 0, 1), s)
    s = np.where(high, np.clip((b - c) / np.where(a > 1e-12, a, 1), 0, 1), s)
    p1 = s[..., None] * d1
    p2 = A[None] + t[..., None] * d2
    return np.linalg.norm(p1 - p2, axis=-1), s


def hidden_scores(joints):
    """joints: [S, 77, 3] camera coordinates of the subjects of one frame -> [S, 77] occlusion score in [0, 1]."""
    S = len(joints)
    A, B, R, owner, child, kind = [], [], [], [], [], []
    for s in range(S):
        for c in range(1, 77):
            A.append(joints[s, PARENTS[c]]); B.append(joints[s, c]); R.append(RADII[c]); owner.append(s); child.append(c); kind.append(0)
        A.append(joints[s, 67]); B.append(joints[s, 72]); R.append(0.10); owner.append(s); child.append(-1); kind.append(1)
        A.append(joints[s, 11]); B.append(joints[s, 39]); R.append(0.06); owner.append(s); child.append(-1); kind.append(2)
    A, B, R, owner, child, kind = (np.array(v) for v in (A, B, R, owner, child, kind))
    P = joints.reshape(-1, 3)
    d, s_ray = segment_distance(P, A, B)
    norm = np.linalg.norm(P, axis=1)[:, None]
    in_front = s_ray * norm < norm - FRONT_MARGIN * R[None]
    cover = np.clip(1 - d / R[None], 0, 1) * in_front
    # own-body exclusions
    joint_owner = np.repeat(np.arange(S), 77)
    joint_idx = np.tile(np.arange(77), S)
    same = joint_owner[:, None] == owner[None]
    incident = np.zeros_like(cover, bool)
    for k in range(77):
        rows = joint_idx == k
        incident[rows] = np.isin(child, list(INCIDENT[k]))[None]
    bar_hips = np.isin(joint_idx, [0, 67, 72])[:, None] & (kind == 1)[None]
    bar_sh = np.isin(joint_idx, [3, 11, 12, 39, 40])[:, None] & (kind == 2)[None]
    cover[same & (incident | bar_hips | bar_sh)] = 0
    return cover.max(1).reshape(S, 77)


def auc(score, y):
    """Mann-Whitney AUC of score for y (ties averaged)."""
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    sorted_s = score[order]
    # average ranks for ties
    uniq, inv, cnt = np.unique(sorted_s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = csum - (cnt - 1) / 2
    ranks[order] = avg[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--bodypose", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    conn = bc.open_db(a.db)
    bp = bc.BodyPose(a.bodypose)
    cache = {}

    def frame_scores(c, f):
        k = (c, f)
        if k not in cache:
            idx = bp.find(np.full(3, c), np.full(3, f), np.array([1, 2, 3]))
            have = idx >= 0
            sc = np.full((3, 77), np.nan)
            if have.any():
                sc[have] = hidden_scores(bp.fields["joints_camera"][idx[have]].astype(np.float64))
            cache[k] = sc
        return cache[k]

    rows = []
    for subject in (1, 2):
        cand, labels = bc.load_subject(a.root, subject)
        window = cue.window_positions(conn, cue.BASIS_RUNS[subject], cand.cameras, cand)
        t = np.full(len(labels.frame), np.nan)
        for c in np.unique(labels.camera):
            m = labels.camera == c
            t[m] = cand.time_of(int(c), labels.frame[m])
        sel = np.flatnonzero((~np.isin(labels.camera, [3])) & (t >= T0) & (t <= T1))
        ckey = kp_gt._key(cand.camera, cand.frame, cand.keypoint)
        order = np.argsort(ckey, kind="stable")
        cks = ckey[order]
        for i in sel:
            c, f, kp = int(labels.camera[i]), int(labels.frame[i]), int(labels.keypoint[i])
            sc = frame_scores(c, f)[subject - 1, kp]
            if not np.isfinite(sc):
                continue
            h = cand.box_height.get((c, f), np.nan)
            wp = window.get((c, f, kp))
            off = np.nan
            peak = np.nan
            if wp is not None and labels.visible[i]:
                off = np.hypot(wp[0] - labels.x[i], wp[1] - labels.y[i]) / h
                peak = wp[2]
            lo, hi = np.searchsorted(cks, kp_gt._key(c, f, kp)), np.searchsorted(cks, kp_gt._key(c, f, kp), side="right")
            peak_strongest = float(cand.peak[order[lo:hi]].max()) if hi > lo else np.nan
            rows.append((subject, c, kp, t[i], sc, float(not labels.visible[i]), off, peak, peak_strongest))
        print("subject", subject, len(rows), flush=True)
    R = np.array(rows)
    np.save(Path(a.out).with_suffix(".npy"), R)
    D = dict(zip(("subject", "cam", "kp", "t", "score", "not_visible", "off", "peak", "peak_strongest"), R.T))
    group = bc.GROUP_OF[D["kp"].astype(int)]
    result = {}

    def evaluate(mask, target, peak):
        y = target[mask]
        out = {"n": int(mask.sum()), "positives": int(y.sum())}
        if out["positives"] and out["positives"] < out["n"]:
            hid = D["score"][mask] > 0
            out["auc_occlusion_score"] = auc(D["score"][mask], y)
            out["auc_heatmap_peak"] = auc(-peak[mask], y)
            tp, fp = int((hid & (y > 0)).sum()), int((hid & (y == 0)).sum())
            fn, tn = int((~hid & (y > 0)).sum()), int((~hid & (y == 0)).sum())
            out["confusion_hidden_vs_target"] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
            out["hidden_share"] = float(hid.mean())
        return out

    nv = D["not_visible"] > 0
    # not-visible labels are scored on keypoint-frames that have candidates, where a heatmap peak exists
    sets = {"label_not_visible": (np.isfinite(D["peak_strongest"]), D["not_visible"], D["peak_strongest"]),
            "window_pick_off_0.10": (np.isfinite(D["off"]), (D["off"] > 0.10).astype(float), D["peak"])}
    for name, (base, target, peak) in sets.items():
        result[name] = {"pooled": evaluate(base, target, peak)}
        for s in (1, 2):
            result[name][f"subject{s}"] = evaluate(base & (D["subject"] == s), target, peak)
        for gi, g in enumerate(bc.GROUP_NAMES):
            result[name][g] = evaluate(base & (group == gi), target, peak)
        result[name]["body_no_fingers"] = evaluate(base & np.isin(D["kp"].astype(int), bc.BODY), target, peak)
    Path(a.out).write_text(json.dumps(result, indent=1))
    for name in sets:
        for g in ("pooled", "body_no_fingers"):
            print(name, g, json.dumps(result[name][g]))


if __name__ == "__main__":
    main()
