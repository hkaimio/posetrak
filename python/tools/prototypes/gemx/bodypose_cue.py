# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Measurement A: does the nearest candidate to the Body Pose keypoint pick the labelled candidate?

For each visible label of range 3 that holds a candidate (the reviewed units), the correct candidate is the one at
the label's position (the eval set's same-candidate definition). Rankers compared on the same keypoint-frames:
the strongest peak, the windowed choice of the basis run (nearest used candidate), the candidate nearest to the
tracked skeleton's projected marker (prior), and the candidate nearest to the Body Pose 2D keypoint (cue).

    python bodypose_cue.py --db session.db --root kp_eval_set_v1 --bodypose out --out cue.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gt"))
import bodypose_common as bc  # noqa: E402
from bodypose_common import kp_gt  # noqa: E402

T0, T1 = 51.7, 56.8
HARD = [(53.0, 54.5), (54.9, 56.4)]
BASIS_RUNS = {1: "36afbdaf", 2: "de1e686b"}   # 'window', ace2 outside the choice, body-only strict filter
SAME_PX = 0.5


def tracker_prior(conn, run_prefix, cand, states):
    """Projected position (undistorted pixels) of each marker's keypoint in the run's forward filter.

    Returns (step times, array [steps, cameras, 77, 2]) with NaN where the run has no prediction.
    """
    import json as _json

    row = conn.execute("SELECT id, observation_sequence_id, active_camera_ids, marker_names FROM tracking_runs "
                       "WHERE id LIKE ?", (run_prefix + "%",)).fetchone()
    run_id, seq = row["id"], row["observation_sequence_id"]
    labels = _json.loads(row["active_camera_ids"])
    markers = _json.loads(row["marker_names"])
    camera_of = {r["label"]: r["id"] for r in conn.execute("SELECT id, label FROM camera_instances")}
    names = {r["name"]: r["keypoint_idx"] for r in conn.execute(
        "SELECT keypoint_idx, name FROM pose_sequence_keypoints WHERE sequence_id = ? AND obs_source = 'kp_candidates'", (seq,))}
    kp_of = [names.get(f"soma77:{m}", -1) for m in markers]
    steps = {r[0]: r[1] for r in conn.execute(
        "SELECT tracker_step, timestamp_s FROM tracking_results WHERE run_id = ? AND person_id = 0 AND is_smoothed = 0", (run_id,))}
    order = sorted(steps)
    times = np.array([steps[s] for s in order])
    row_of = {s: i for i, s in enumerate(order)}
    pred = np.full((len(order), len(cand.cameras), 77, 2), np.nan)
    cam_idx = [cand.cameras.index(camera_of[lab]) if camera_of[lab] in cand.cameras else -1 for lab in labels]
    for step, blob in conn.execute("SELECT tracker_step, obs_blob FROM tracking_obs_results WHERE run_id = ? AND person_id = 0", (run_id,)):
        if step not in row_of:
            continue
        obs = np.frombuffer(bytes(blob), "<f4").reshape(len(labels), len(markers), 8)
        for ci, c in enumerate(cam_idx):
            if c < 0:
                continue
            for mi, kp in enumerate(kp_of):
                if kp >= 0 and obs[ci, mi, 7] == 0:
                    pred[row_of[step], c, kp] = obs[ci, mi, 2:4]
    return times, pred


def window_positions(conn, run_prefix, cams, cand):
    """Position of the candidate the windowed choice of a run took, by (camera, frame, keypoint), from the run's
    ``tracking_kp_picks`` (tracklet, rank) and the candidate rows of its observation sequence."""
    import eval_kp_candidates as ev

    row = conn.execute("SELECT id, observation_sequence_id FROM tracking_runs WHERE id LIKE ?", (run_prefix + "%",)).fetchone()
    picks = ev.read_window_picks(conn, row["id"])
    rows = kp_gt.read_sequence_candidates(conn, row["observation_sequence_id"], 0)
    out = {}
    for cam_id, (frames, tracklets, ranks) in picks.items():
        if cam_id not in cams:
            continue
        c = cams.index(cam_id)
        by_frame = {frame: table for frame, _, table in rows.get(cam_id, [])}
        for i, frame in enumerate(frames):
            table = by_frame.get(int(frame))
            if table is None:
                continue
            for kp in range(tracklets.shape[1]):
                trk, rank = int(tracklets[i, kp]), int(ranks[i, kp])
                if trk == -1:
                    continue
                m = (table[:, 0] == kp) & (table[:, 4] == rank)
                if trk != -2:
                    m &= table[:, 5] == trk
                hit = np.flatnonzero(m)
                if len(hit):
                    # the product decodes with cell centres, the frozen candidate files with left edges: half a cell
                    side = 1.6 * cand.box_height.get((c, int(frame)), np.nan)
                    out[(c, int(frame), kp)] = (float(table[hit[0], 1]) - 0.5 * 0.75 * side / 48,
                                                float(table[hit[0], 2]) - 0.5 * side / 64, float(table[hit[0], 3]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--bodypose", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    conn = bc.open_db(a.db)
    states = bc.camera_states(conn)
    bp = bc.BodyPose(a.bodypose)
    result, rows_all = {}, []
    for subject in (1, 2):
        cand, labels = bc.load_subject(a.root, subject)
        cams = cand.cameras
        skip = [3]
        # candidate rows indexed by (camera, frame, keypoint)
        key = kp_gt._key(cand.camera, cand.frame, cand.keypoint)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        # strongest picks and the basis run's picks
        window_pick = window_positions(conn, BASIS_RUNS[subject], cams, cand)
        ptimes, ppred = tracker_prior(conn, BASIS_RUNS[subject], cand, states)
        # label selection
        t = np.full(len(labels.frame), np.nan)
        for c in np.unique(labels.camera):
            m = labels.camera == c
            t[m] = cand.time_of(int(c), labels.frame[m])
        sel = labels.visible & labels.holds_candidate & ~np.isin(labels.camera, skip) & (t >= T0) & (t <= T1)
        idx = np.flatnonzero(sel)
        und_cache = {}
        n_total = 0
        for i in idx:
            c, f, kp = int(labels.camera[i]), int(labels.frame[i]), int(labels.keypoint[i])
            lo, hi = np.searchsorted(ks, kp_gt._key(c, f, kp)), np.searchsorted(ks, kp_gt._key(c, f, kp), side="right")
            if lo == hi:
                continue
            r = order[lo:hi]
            d = np.hypot(cand.x[r] - labels.x[i], cand.y[r] - labels.y[i])
            correct = np.flatnonzero(d <= SAME_PX)
            if not len(correct):
                continue
            b = bp.find(c, f, subject)
            if b < 0:
                continue
            st = states[cams[c]]
            u = bc.undistort(np.c_[cand.x[r], cand.y[r]], st)
            ref = bp.fields["keypoints_2d"][b, kp]
            box_h = cand.box_height.get((c, f), np.nan)
            dist_bp = np.hypot(u[:, 0] - ref[0], u[:, 1] - ref[1]) / box_h
            right = int(correct[0])
            strongest = int(np.argmin(cand.rank[r]))
            # windowed pick: the candidate the basis run used, if any
            win = None
            wp = window_pick.get((c, f, kp))
            if wp is not None:
                dd = np.hypot(cand.x[r] - wp[0], cand.y[r] - wp[1])
                win = int(np.argmin(dd)) if dd.min() <= SAME_PX else None
            # prior: the run's predicted marker position nearest in time
            prior = None
            si = int(np.argmin(np.abs(ptimes - t[i]))) if len(ptimes) else -1
            if si >= 0 and abs(ptimes[si] - t[i]) < 0.01 and np.isfinite(ppred[si, c, kp, 0]):
                pp = ppred[si, c, kp]
                prior = int(np.argmin(np.hypot(u[:, 0] - pp[0], u[:, 1] - pp[1])))
            rows_all.append((subject, c, kp, t[i], len(r), right, strongest, -1 if win is None else win,
                             -1 if prior is None else prior, int(np.argmin(dist_bp)), float(dist_bp[right]),
                             float(np.sort(dist_bp)[0])))
            n_total += 1
        print("subject", subject, "samples", n_total, flush=True)
    R = np.array(rows_all)
    np.save(Path(a.out).with_suffix(".npy"), R)
    cols = ("subject", "cam", "kp", "t", "n_cand", "right", "strongest", "window", "prior", "bp", "d_right", "d_min")
    D = dict(zip(cols, R.T))
    in_hard = np.any([(D["t"] >= lo) & (D["t"] <= hi) for lo, hi in HARD], axis=0)

    def shares(m):
        n = int(m.sum())
        if not n:
            return {"n": 0}
        out = {"n": n, "strongest_correct": float((D["strongest"][m] == D["right"][m]).mean()),
               "bodypose_correct": float((D["bp"][m] == D["right"][m]).mean())}
        w = m & (D["window"] >= 0)
        out["window_n"] = int(w.sum())
        out["window_correct"] = float((D["window"][w] == D["right"][w]).mean()) if w.any() else None
        out["bodypose_on_window_subset"] = float((D["bp"][w] == D["right"][w]).mean()) if w.any() else None
        p = m & (D["prior"] >= 0)
        out["prior_n"] = int(p.sum())
        out["prior_correct"] = float((D["prior"][p] == D["right"][p]).mean()) if p.any() else None
        out["bodypose_on_prior_subset"] = float((D["bp"][p] == D["right"][p]).mean()) if p.any() else None
        out["d_correct_median"] = float(np.median(D["d_right"][m]))
        out["d_correct_p90"] = float(np.percentile(D["d_right"][m], 90))
        return out

    multi = D["n_cand"] >= 2
    wrong_strong = D["strongest"] != D["right"]
    wrong_window = (D["window"] >= 0) & (D["window"] != D["right"])
    sets = {"all": np.ones(len(R), bool), "two_or_more_candidates": multi, "strongest_wrong": wrong_strong,
            "window_wrong": wrong_window, "strongest_wrong_in_hard_windows": wrong_strong & in_hard}
    for name, base in sets.items():
        result[name] = {"pooled": shares(base)}
        for s in (1, 2):
            result[name][f"subject{s}"] = shares(base & (D["subject"] == s))
        for gi, g in enumerate(bc.GROUP_NAMES):
            result[name][g] = shares(base & (bc.GROUP_OF[D["kp"].astype(int)] == gi))
        body = np.isin(D["kp"].astype(int), bc.BODY)
        result[name]["body_no_fingers"] = shares(base & body)
    Path(a.out).write_text(json.dumps(result, indent=1))
    for name in sets:
        p = result[name]["pooled"]
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()})


if __name__ == "__main__":
    main()
