# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Run GEM-X Body Pose on a camera job (GEM environment, not the Posetrak one).

Decodes the frames with Posetrak's frame reader, undistorts them with the calibration of the job, and
reconstructs all subjects of a frame together from the same undistorted image. Results: one file per camera.

    D:/mocap/models/gem-x-repo/.venv/Scripts/python.exe bodypose_run.py --jobs jobs --out out --cameras 0 1 2 3 4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "D:/mocap/wt-gemx/python")
from posetrak.detection.frame_source import iter_frames  # noqa: E402

from gem.pose.api import load_pose_predictor  # noqa: E402

ROOT = "D:/mocap/models/gem-x-repo/"
BATCH = 16


def undistort_maps(job, size):
    K, K0, dist = job["K"], job["K_orig"], job["dist"]
    if bool(job["fisheye"]):
        return cv2.fisheye.initUndistortRectifyMap(K0, dist.reshape(-1)[:4], np.eye(3), K, size, cv2.CV_16SC2)
    return cv2.initUndistortRectifyMap(K0, dist, None, K, size, cv2.CV_16SC2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many frames with boxes (timing runs)")
    ap.add_argument("--compare", type=int, default=0, help="also run fp32 on the first N crops and report the difference")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    release = json.load(open(ROOT + "weights/body_config.json"))
    pred = load_pose_predictor(ROOT + "weights/body_s240000.ckpt", release, soma_assets=ROOT + "assets/soma", device="cuda")
    torch.cuda.synchronize()
    print("model load s", round(time.time() - t0, 1), flush=True)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.precision == "bf16")

    for ci in a.cameras:
        job = np.load(Path(a.jobs) / f"job_cam{ci}.npz", allow_pickle=True)
        frames, boxes = job["frames"], job["boxes"]
        K = job["K"].astype(np.float32)
        maps = None
        pending, results = [], {k: [] for k in ("frame", "subject", "box", "joints_camera", "keypoints_2d", "valid",
                                                "camera_translation", "body_pose", "global_orient", "identity", "scale")}
        t_cam = time.time()
        n_crops = n_done = 0
        compared = 0

        def flush():
            nonlocal n_done, compared
            if not pending:
                return
            imgs = [p[0] for p in pending]
            bx = np.array([p[3] for p in pending], np.float32)
            with autocast:
                o = pred.predict(imgs, bx, K)
            if a.compare and compared < a.compare:
                o32 = pred.predict(imgs, bx, K)
                d = (o.joints_camera.float() - o32.joints_camera.float()).norm(dim=-1)
                print("bf16 vs fp32 joint distance m: median", float(d.median()), "p99", float(d.flatten().quantile(0.99)),
                      "max", float(d.max()), flush=True)
                compared += len(pending)
            results["frame"] += [p[1] for p in pending]
            results["subject"] += [p[2] for p in pending]
            results["box"].append(bx)
            results["joints_camera"].append(o.joints_camera.float().cpu().numpy())
            results["keypoints_2d"].append(o.keypoints_2d.float().cpu().numpy())
            results["valid"].append(o.projection_valid.cpu().numpy())
            results["camera_translation"].append(o.camera_translation.float().cpu().numpy())
            results["body_pose"].append(o.parameters["body_pose"].float().cpu().numpy().astype(np.float16))
            results["global_orient"].append(o.parameters["global_orient"].float().cpu().numpy())
            results["identity"].append(o.parameters["identity_coeffs"].float().cpu().numpy())
            results["scale"].append(o.parameters["scale_params"].float().cpu().numpy())
            n_done += len(pending)
            pending.clear()

        wanted = {int(f): i for i, f in enumerate(frames)}
        reader = iter_frames(str(job["video"]), int(frames[0]), int(frames[-1]) + 1)
        for frame_idx, bgr in reader:
            i = wanted.get(frame_idx)
            if i is None:
                continue
            if maps is None:
                h, w = bgr.shape[:2]
                maps = undistort_maps(job, (w, h))
            ok = np.isfinite(boxes[i, :, 0])
            if not ok.any():
                continue
            img = np.ascontiguousarray(cv2.remap(bgr, maps[0], maps[1], cv2.INTER_LINEAR)[:, :, ::-1])
            for s in np.flatnonzero(ok):
                pending.append((img, frame_idx, int(s) + 1, boxes[i, s]))
                n_crops += 1
            if len(pending) >= BATCH:
                flush()
            if a.limit and n_crops >= a.limit:
                break
        flush()
        torch.cuda.synchronize()
        dt = time.time() - t_cam
        print(f"camera {ci}: {n_done} crops in {dt:.0f} s, {n_done / dt:.1f} crops/s (decode, undistort, model)", flush=True)
        cat = {k: (np.concatenate(v) if isinstance(v[0], np.ndarray) else np.array(v)) for k, v in results.items()}
        fidx = {int(f): i for i, f in enumerate(frames)}
        np.savez(out_dir / f"bodypose_cam{ci}.npz", camera=str(job["camera"]), K=K,
                 time=job["times"][[fidx[int(f)] for f in cat["frame"]]], **cat)


if __name__ == "__main__":
    main()
