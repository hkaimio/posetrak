# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""eval_fk_prediction.py — the real P-A metric: FK-predicted marker
position vs. the actual **slot-labelled ground truth** (not "nearest raw
detection", which prototype_fk_marker_prediction.py used as a proxy
before slot-labelled GT existed).

Also resolves the thing P-A's probe attachment set (`_DEFAULT_TRIAL` in
prototype_fk_marker_prediction.py -- unlabelled `k_Xp`/`k_Xm`/`k_Zp`/
`k_Zm` etc. probes around the knee/ankle) was left deliberately
open-ended for: which local direction is which real anatomical slot.
For each GT point, finds the *closest* predicted probe; the majority
vote across all occurrences of a given GT slot name is that slot's
empirical probe mapping, and the accompanying purity number says how
consistent that mapping is (a low purity means the trial catalog's
geometry doesn't cleanly separate that slot from its neighbours yet).

Usage:
    python tools/eval_fk_prediction.py \\
        --labels scratch/dot_ground_truth/pc_labels_refined.json \\
        --session nelli=/path/to/session.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --tracking-run 990fb01a-6c72-47bc-bdb7-4d31147d2ef7
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from posetrak.db.skeleton_layout import SkeletonLayout  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states, load_sync_table  # noqa: E402
from tools.prototype_fk_marker_prediction import _load_attachment_set, _project_raw  # noqa: E402

_UNLABELED = "unlabeled"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--session", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--tracking-run", required=True)
    ap.add_argument("--state-time-tolerance-s", type=float, default=0.05)
    ap.add_argument("--attachment-set", default=None,
                     help="A fit_calibrated_attachment_set.py (B4) output YAML, or "
                          "prototype_fk_marker_prediction.py's own attachment-set file shape. "
                          "Default: P-A's exploratory _DEFAULT_TRIAL probes.")
    args = ap.parse_args()
    attach = _load_attachment_set(args.attachment_set)

    session_paths = {}
    for entry in args.session:
        name, _, path = entry.partition("=")
        session_paths[name] = path
    conns = {n: sqlite3.connect(f"file:{p}?mode=ro", uri=True) for n, p in session_paths.items()}
    conn0 = next(iter(conns.values()))
    conn0.row_factory = sqlite3.Row

    states = load_camera_states(conn0, args.shot_id)
    sync_table, svid_by_cam = load_sync_table(conn0, args.shot_id)
    label_to_cam = {r["label"]: r["id"] for r in conn0.execute("SELECT id, label FROM camera_instances")}
    svid_to_cam = {v: k for k, v in svid_by_cam.items()}

    sk_id = conn0.execute("SELECT skeleton_id FROM tracking_runs WHERE id = ?", (args.tracking_run,)).fetchone()["skeleton_id"]
    yc = conn0.execute("SELECT yaml_content FROM skeletons WHERE id = ?", (sk_id,)).fetchone()["yaml_content"]
    layout = SkeletonLayout(yc)
    st_rows = conn0.execute(
        "SELECT timestamp_s, state FROM tracking_results WHERE run_id = ? AND is_smoothed = 1 ORDER BY tracker_step",
        (args.tracking_run,),
    ).fetchall()
    st_times = np.array([r["timestamp_s"] for r in st_rows])
    st_blobs = [bytes(r["state"]) for r in st_rows]

    def state_at(t: float):
        i = int(np.argmin(np.abs(st_times - t)))
        if abs(st_times[i] - t) > args.state_time_tolerance_s:
            return None
        return layout.decode_state_blob(st_blobs[i])

    frames = json.loads(Path(args.labels).read_text())

    # (gt_slot, probe_name) -> list of pixel distances; also per-camera breakdown
    votes: dict[str, Counter] = defaultdict(Counter)
    dists: dict[tuple[str, str], list[float]] = defaultdict(list)
    dists_by_cam: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    n_skipped_no_state = 0
    n_gt_points = 0

    for e in frames:
        pts = e.get("points", [])
        pm = e.get("point_meta", [])
        if not pts:
            continue
        cam_label = e["camera_label"]
        cam_id = label_to_cam.get(cam_label)
        if cam_id is None or cam_id not in states:
            continue
        svid = svid_by_cam.get(cam_id)
        if svid is None:
            continue
        t = sync_table.frame_to_global_time(e["video_frame"], svid)
        if t is None:
            continue
        dec = state_at(t)
        if dec is None:
            n_skipped_no_state += 1
            continue
        T = layout.compute_joint_transforms(dec)

        probe_world = {}
        for name, (parent, off, _nrm) in attach.items():
            Tp = T.get(parent)
            if Tp is not None:
                probe_world[name] = Tp[:3, :3] @ off + Tp[:3, 3]
        if not probe_world:
            continue
        names = list(probe_world)
        proj = _project_raw(np.array([probe_world[n] for n in names]), states[cam_id])

        for k, xy in enumerate(pts):
            slot = pm[k].get("slot") if k < len(pm) else _UNLABELED
            if not slot or slot == _UNLABELED:
                continue
            n_gt_points += 1
            d = np.hypot(proj[:, 0] - xy[0], proj[:, 1] - xy[1])
            j = int(np.argmin(d))
            probe_name, dist = names[j], float(d[j])
            votes[slot][probe_name] += 1
            dists[(slot, probe_name)].append(dist)
            dists_by_cam[(slot, probe_name, cam_label)].append(dist)

    if n_skipped_no_state:
        print(f"(skipped {n_skipped_no_state} labelled frames with no tracking state within "
              f"{args.state_time_tolerance_s}s)\n")

    print(f"{n_gt_points} GT slot points evaluated\n")
    print(f"{'GT slot':16s} {'-> probe':10s} {'purity':>7s} {'n':>5s} {'med_px':>7s} {'p90_px':>7s}")
    print("-" * 60)
    for slot in sorted(votes):
        total = sum(votes[slot].values())
        best_probe, best_n = votes[slot].most_common(1)[0]
        purity = best_n / total
        d = np.array(dists[(slot, best_probe)])
        print(f"{slot:16s} {best_probe:10s} {purity:7.2f} {total:5d} {np.median(d):7.1f} {np.percentile(d, 90):7.1f}")

    cams = sorted({c for (_s, _p, c) in dists_by_cam})
    print("\nPer-camera median error (px) for each slot's majority probe:")
    print(f"{'GT slot':16s}" + "".join(f"{c[:14]:>16s}" for c in cams))
    for slot in sorted(votes):
        best_probe, _n = votes[slot].most_common(1)[0]
        cells = []
        for cam in cams:
            d = dists_by_cam.get((slot, best_probe, cam))
            cells.append(f"{np.median(d):.1f} (n={len(d)})" if d else "-")
        print(f"{slot:16s}" + "".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    main()
