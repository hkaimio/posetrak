# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Decoder-convention bias of the 2D SOMA-77 model, measured without a decode-dependent reference.

The stored candidates of the frozen range-3 files were decoded with the left-edge convention, i / W of the model
window (the product's decoder adds half a cell to each coordinate). Cell coordinates u are recovered from the stored pixels and re-mapped to the other conventions
(left edge i / W, UDP edge i / (W - 1)); the mapping is affine in u, so no model run is needed.

Reference: for each labelled visible keypoint held by a candidate, triangulate it from the other labelled cameras
(same convention in all cameras) and take the signed difference between the keypoint and the reprojection in the
target camera's distorted pixels, in heatmap cells. A decoding bias in the target camera is uncorrelated with the
other cameras' errors, so it shows as a trend of the difference against the keypoint's position in the window.

    python decoder_convention_2d.py --db session.db --root kp_eval_set_v1 --out result.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))            # python/
sys.path.insert(0, str(HERE.parents[2] / "gt"))     # python/tools/gt
import kp_gt  # noqa: E402
from posetrak.calibration.session_cameras import load_camera_states  # noqa: E402

W, H = 48, 64
FPS = 120.0
CONVENTIONS = {
    "cell_centre": lambda i, n: (i + 0.5) / n,
    "left_edge": lambda i, n: i / n,
    "udp_edge": lambda i, n: i / (n - 1),
}
GROUPS = {
    "trunk": list(range(0, 4)), "head": list(range(4, 11)),
    "arm": [11, 12, 13, 39, 40, 41], "wrist": [14, 42],
    "fingers": list(range(15, 39)) + list(range(43, 67)), "leg": list(range(67, 77)),
}
GROUP_OF = np.full(77, -1)
for gi, (_, ks) in enumerate(GROUPS.items()):
    GROUP_OF[ks] = gi


def cell_coords(x, y, cx, cy, s):
    ww, wh = 0.75 * s, s
    # the frozen candidate files were decoded with the left-edge convention: stored pixel = window edge + u / W * width
    return (x - cx + ww / 2) / ww * W, (y - cy + wh / 2) / wh * H


def to_pixels(u, v, cx, cy, s, conv):
    ww, wh = 0.75 * s, s
    f = CONVENTIONS[conv]
    return cx - ww / 2 + f(u, W) * ww, cy - wh / 2 + f(v, H) * wh


def undistort(p, st):
    p = np.asarray(p, np.float32).reshape(-1, 1, 2)
    fn = cv2.fisheye.undistortPoints if st.fisheye else cv2.undistortPoints
    return fn(p, st.K_orig, st.dist, None, st.K).reshape(-1, 2).astype(np.float64)


def project(X, st):
    rvec, _ = cv2.Rodrigues(np.asarray(st.R, np.float64))
    t = np.asarray(st.t, np.float64).reshape(3, 1)
    X = X.reshape(-1, 1, 3).astype(np.float64)
    if st.fisheye:
        return cv2.fisheye.projectPoints(X, rvec, t, st.K_orig, st.dist.reshape(-1)[:4])[0].reshape(-1, 2)
    return cv2.projectPoints(X, rvec, t, st.K_orig, st.dist)[0].reshape(-1, 2)


def proj_matrix(st):
    return st.K @ np.hstack([np.asarray(st.R, np.float64), np.asarray(st.t, np.float64).reshape(3, 1)])


def load_subject(root: Path, subject: int, unit_key: str):
    d = root / "range3"
    cz = np.load(d / f"candidates_subject{subject}.npz", allow_pickle=True)
    frames = cz["frames"]
    lab = kp_gt.read_labels(d / f"labels_subject{subject}.npz", d / f"units_subject{subject}.json", unit_key)
    order = np.lexsort((frames[:, 1], frames[:, 0]))
    fr = frames[order]
    key = fr[:, 0].astype(np.int64) * 10_000_000 + fr[:, 1].astype(np.int64)
    return cz, lab, fr, key


def lookup(fr, key, cam, frame):
    q = np.asarray(cam, np.int64) * 10_000_000 + np.asarray(frame, np.int64)
    i = np.clip(np.searchsorted(key, q), 0, len(key) - 1)
    return fr[i], key[i] == q


def residuals(subject_data, states, cams, conv):
    """Per labelled sample: camera, keypoint, time bin, signed residual in cells (x, y), residual in box heights,
    window position (xi, eta) under the cell-centre convention."""
    cz, lab, fr, key = subject_data
    m = lab.visible & lab.holds_candidate
    rows, ok = lookup(fr, key, lab.camera[m], lab.frame[m])
    cam, kp, x, y = lab.camera[m][ok], lab.keypoint[m][ok], lab.x[m][ok], lab.y[m][ok]
    rows = rows[ok]
    t, cx, cy, s = rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5]
    u, v = cell_coords(x, y, cx, cy, s)
    xs, ys = to_pixels(u, v, cx, cy, s, conv)
    tb = np.round(t * FPS).astype(np.int64)
    bins, binx = np.unique(tb, return_inverse=True)
    n_cam = len(cams)
    obs = np.full((len(bins), 77, n_cam, 2), np.nan)
    obs[binx, kp, cam] = np.c_[xs, ys]
    sample_at = np.full((len(bins), 77, n_cam), -1, np.int64)
    sample_at[binx, kp, cam] = np.arange(len(kp))
    und = np.full_like(obs, np.nan)
    for c in range(n_cam):
        sel = np.isfinite(obs[:, :, c, 0])
        if sel.any():
            und[:, :, c][sel] = undistort(obs[:, :, c][sel], states[cams[c]])
    M = np.zeros((len(bins), 77, n_cam, 4, 4))
    for c in range(n_cam):
        P = proj_matrix(states[cams[c]])
        sel = np.isfinite(und[:, :, c, 0])
        if not sel.any():
            continue
        p = und[:, :, c][sel]
        a1 = p[:, :1] * P[2] - P[0]
        a2 = p[:, 1:] * P[2] - P[1]
        a1 /= np.linalg.norm(a1[:, :3], axis=1, keepdims=True)
        a2 /= np.linalg.norm(a2[:, :3], axis=1, keepdims=True)
        M[:, :, c][sel] = a1[:, :, None] * a1[:, None, :] + a2[:, :, None] * a2[:, None, :]
    out = {k: [] for k in ("cam", "kp", "tb", "rx", "ry", "rh", "xi", "eta")}
    box_h = s / 1.6
    for c in range(n_cam):
        have = np.isfinite(obs[:, :, c, 0])
        others = [j for j in range(n_cam) if j != c]
        cnt = np.isfinite(und[:, :, others, 0]).sum(-1)
        bi, ki = np.nonzero(have & (cnt >= 3))
        if not len(bi):
            continue
        w, V = np.linalg.eigh(M[bi, ki][:, others].sum(1))
        Xh = V[:, :, 0]
        good = Xh[:, 3] != 0
        X = np.where(good[:, None], Xh[:, :3] / np.where(good, Xh[:, 3], 1)[:, None], np.nan)
        R_, t_ = np.asarray(states[cams[c]].R), np.asarray(states[cams[c]].t).reshape(3)
        good &= np.isfinite(X).all(1) & ((X @ R_.T + t_)[:, 2] > 0)
        bi, ki, X = bi[good], ki[good], X[good]
        rep = project(X, states[cams[c]])
        d = obs[bi, ki, c] - rep
        idx = sample_at[bi, ki, c]
        ww = 0.75 * s[idx]
        out["cam"].append(np.full(len(bi), c))
        out["kp"].append(ki)
        out["tb"].append(bins[bi])
        out["rx"].append(d[:, 0] / (ww / W))
        out["ry"].append(d[:, 1] / (s[idx] / H))
        out["rh"].append(np.hypot(d[:, 0], d[:, 1]) / box_h[idx])
        out["xi"].append((x[idx] - cx[idx]) / ww)
        out["eta"].append((y[idx] - cy[idx]) / s[idx])
    return {k: np.concatenate(v) if v else np.empty(0) for k, v in out.items()}


def fit(r, mask, boot=100, seed=0):
    """OLS of rx on xi and ry on eta over the masked samples; bootstrap over 1 s blocks for the uncertainty."""
    rng = np.random.default_rng(seed)
    n = int(mask.sum())
    if n < 200:
        return {"n": n}
    sel = {k: v[mask] for k, v in r.items()}
    blocks = (sel["tb"] // int(FPS)).astype(int)
    ub = np.unique(blocks)
    by = {b: np.flatnonzero(blocks == b) for b in ub}
    res = {"n": n}
    for name, res_key, pos_key in (("x", "rx", "xi"), ("y", "ry", "eta")):
        def ols(idx):
            A = np.c_[np.ones(len(idx)), sel[pos_key][idx]]
            return np.linalg.lstsq(A, sel[res_key][idx], rcond=None)[0]
        beta = ols(np.arange(n))
        bs = np.array([ols(np.concatenate([by[b] for b in rng.choice(ub, len(ub))])) for _ in range(boot)])
        res[f"{name}_offset"] = [float(beta[0]), float(bs[:, 0].std())]   # cells at the window centre
        res[f"{name}_slope"] = [float(beta[1]), float(bs[:, 1].std())]    # cells per window width
    res["median_box_heights"] = float(np.median(sel["rh"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default="855e1e19", help="a tracking run of the trial; its capture gives the cameras")
    ap.add_argument("--trim", type=float, default=0.08, help="drop samples farther than this many box heights")
    a = ap.parse_args()
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    shot = conn.execute("SELECT s.shot_id FROM tracking_runs t JOIN pose_observation_sequences s "
                        "ON s.id = t.observation_sequence_id WHERE t.id LIKE ?", (a.run + "%",)).fetchone()[0]
    states = load_camera_states(conn, shot)
    root = Path(a.root)
    result = {}
    for subject in (1, 2):
        data = load_subject(root, subject, "51.7-56.8")
        cams = [str(c) for c in data[0]["cams"]]
        for conv in CONVENTIONS:
            r = residuals(data, states, cams, conv)
            ok = r["rh"] < a.trim
            entry = {"pooled": fit(r, ok), "untrimmed_median_box_heights": float(np.median(r["rh"])),
                     "trimmed_share": float(ok.mean()), "per_camera": {}, "per_group": {}}
            for c in range(len(cams)):
                entry["per_camera"][cams[c][:8]] = fit(r, ok & (r["cam"] == c))
            for gi, g in enumerate(GROUPS):
                entry["per_group"][g] = fit(r, ok & (GROUP_OF[r["kp"].astype(int)] == gi))
            result[f"subject{subject}/{conv}"] = entry
            print(subject, conv, entry["pooled"], flush=True)
    Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
