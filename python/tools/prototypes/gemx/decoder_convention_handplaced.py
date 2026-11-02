# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Cross-check of the decoder convention on the free-hand labels of range 3 (labels not held by a candidate).

Each free-hand label is compared with the nearest stored candidate of the same camera, frame and keypoint (within
--tol box heights), re-mapped to each convention. The label is the reference, so its position does not depend on
a decoder. The regressor is the label's own position in the window.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import decoder_convention_2d as dc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args()
    root = Path(a.root) / "range3"
    for subject in (1, 2):
        cz, lab, fr, key = dc.load_subject(Path(a.root), subject, "51.7-56.8")
        m = lab.visible & ~lab.holds_candidate
        lrows, ok = dc.lookup(fr, key, lab.camera[m], lab.frame[m])
        cam, kp, lx, ly = lab.camera[m][ok], lab.keypoint[m][ok], lab.x[m][ok], lab.y[m][ok]
        lrows = lrows[ok]
        rows = cz["rows"]
        ck = (rows[:, 0].astype(np.int64) * 10_000_000 + rows[:, 1].astype(np.int64)) * 100 + rows[:, 2].astype(np.int64)
        order = np.argsort(ck)
        cks = ck[order]
        print(f"subject {subject}: {int(m.sum())} free-hand labels, {len(cam)} with crop")
        pairs = []  # (label x, y, candidate u, v, crop) paired once, under the cell-centre convention
        for i in range(len(cam)):
            k = (int(cam[i]) * 10_000_000 + int(lab.frame[m][ok][i])) * 100 + int(kp[i])
            lo, hi = np.searchsorted(cks, k), np.searchsorted(cks, k, side="right")
            if lo == hi:
                continue
            c = rows[order[lo:hi]]
            cx, cy, s = lrows[i][3], lrows[i][4], lrows[i][5]
            d = np.hypot(c[:, 4] - lx[i], c[:, 5] - ly[i])
            j = int(np.argmin(d))
            if d[j] <= a.tol * s / 1.6:
                u, v = dc.cell_coords(c[j, 4], c[j, 5], cx, cy, s)
                pairs.append((lx[i], ly[i], u, v, cx, cy, s))
        P = np.array(pairs)
        for conv in dc.CONVENTIONS:
            lxx, lyy, u, v, cx, cy, s = P.T
            px, py = dc.to_pixels(u, v, cx, cy, s, conv)
            ww = 0.75 * s
            dx, dy = (lxx - px) / (ww / dc.W), (lyy - py) / (s / dc.H)
            xi, eta = (lxx - cx) / ww, (lyy - cy) / s
            out = {"n": len(dx)}
            for name, d_, p_ in (("x", dx, xi), ("y", dy, eta)):
                A = np.c_[np.ones(len(p_)), p_]
                beta = np.linalg.lstsq(A, d_, rcond=None)[0]
                sig = np.sqrt(np.sum((d_ - A @ beta) ** 2) / (len(d_) - 2))
                cov = sig**2 * np.linalg.inv(A.T @ A)
                out[name] = {"offset": [round(beta[0], 3), round(float(np.sqrt(cov[0, 0])), 3)],
                             "slope": [round(beta[1], 3), round(float(np.sqrt(cov[1, 1])), 3)], "resid_sd_cells": round(sig, 2)}
            print(" ", conv, json.dumps(out))


if __name__ == "__main__":
    main()
