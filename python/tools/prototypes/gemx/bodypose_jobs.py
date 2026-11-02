# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Write one Body Pose job file per camera of the range-3 trial (Posetrak environment).

A job holds, for every video frame of the trial with person boxes, the subjects' square boxes mapped into the
undistorted image (Body Pose's [cx, cy, size] convention: 1.2 x max(height, width / 0.75), as in GEM's own
``square_crops``), plus the undistortion the run script applies. Boxes come from the product's mask box path: the
``person_detections`` of a detection run made with ``detect run --segmentation``.

    python bodypose_jobs.py --db session.db --registry registry.db --root kp_eval_set_v1 --out-dir jobs
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
from posetrak.calibration.session_cameras import load_camera_states, load_sync_table  # noqa: E402

ENLARGE = 1.2


def undistort_points(p, st):
    p = np.asarray(p, np.float32).reshape(-1, 1, 2)
    fn = cv2.fisheye.undistortPoints if st.fisheye else cv2.undistortPoints
    return fn(p, st.K_orig, st.dist, None, st.K).reshape(-1, 2)


def undistorted_square(box_cxcywh, st):
    cx, cy, w, h = box_cxcywh
    xs = np.linspace(cx - w / 2, cx + w / 2, 5)
    ys = np.linspace(cy - h / 2, cy + h / 2, 5)
    edge = np.array([(x, y) for x in xs for y in (ys[0], ys[-1])] + [(x, y) for y in ys for x in (xs[0], xs[-1])])
    u = undistort_points(edge, st)
    lo, hi = u.min(0), u.max(0)
    ww, hh = hi - lo
    return (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, max(hh, ww / 0.75) * ENLARGE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run", default="8e2147a0", help="detection run with the mask boxes of the three subjects")
    ap.add_argument("--tracking-run", default="855e1e19", help="a tracking run of the trial; its capture is used")
    ap.add_argument("--t0", type=float, default=43.5)
    ap.add_argument("--t1", type=float, default=66.44)
    a = ap.parse_args()
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    shot = conn.execute("SELECT s.shot_id FROM tracking_runs t JOIN pose_observation_sequences s ON "
                        "s.id = t.observation_sequence_id WHERE t.id LIKE ?", (a.tracking_run + "%",)).fetchone()[0]
    run_id = conn.execute("SELECT id FROM detection_runs WHERE id LIKE ?", (a.run + "%",)).fetchone()[0]
    states = load_camera_states(conn, shot)
    sync, svid_by_cam = load_sync_table(conn, shot)
    cams = [str(c) for c in np.load(Path(a.root) / "range3" / "candidates_subject1.npz", allow_pickle=True)["cams"]]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ci, cam in enumerate(cams):
        svid = svid_by_cam[cam]
        st = states[cam]
        path = conn.execute("SELECT file_path FROM capture_videos WHERE id = ?", (svid,)).fetchone()[0]
        rows = conn.execute(
            "SELECT video_frame, track_id, bbox_x, bbox_y, bbox_w, bbox_h FROM person_detections WHERE "
            "detection_run_id = ? AND shot_video_id = ? AND region_type = 'full_body' ORDER BY video_frame, track_id",
            (run_id, svid)).fetchall()
        frames = sorted({r["video_frame"] for r in rows})
        times = {f: sync.frame_to_global_time(f, svid) for f in frames}
        keep = [f for f in frames if a.t0 <= times[f] <= a.t1]
        index = {f: i for i, f in enumerate(keep)}
        boxes = np.full((len(keep), 3, 3), np.nan, np.float32)   # frame, track 1..3, (cx, cy, size) undistorted
        for r in rows:
            i = index.get(r["video_frame"])
            if i is None or not 1 <= r["track_id"] <= 3:
                continue
            boxes[i, r["track_id"] - 1] = undistorted_square((r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]), st)
        np.savez(out / f"job_cam{ci}.npz", camera=cam, camera_index=ci, video=path, frames=np.array(keep),
                 times=np.array([times[f] for f in keep]), boxes=boxes, K=st.K, K_orig=st.K_orig, dist=st.dist,
                 fisheye=st.fisheye, R=np.asarray(st.R), t=np.asarray(st.t).reshape(3))
        print(ci, cam[:8], path, len(keep), "frames", int(np.isfinite(boxes[..., 0]).sum()), "boxes", flush=True)


if __name__ == "__main__":
    main()
