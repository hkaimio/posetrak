# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Load GEM-X Body Pose, time the load and the throughput per batch size. Run in the GEM environment.

    D:/mocap/models/gem-x-repo/.venv/Scripts/python.exe bodypose_smoke.py
"""
import json
import time

import numpy as np
import torch

from gem.pose.api import load_pose_predictor

ROOT = "D:/mocap/models/gem-x-repo/"
t0 = time.time()
release = json.load(open(ROOT + "weights/body_config.json"))
pred = load_pose_predictor(ROOT + "weights/body_s240000.ckpt", release, soma_assets=ROOT + "assets/soma", device="cuda")
torch.cuda.synchronize()
print("load s", round(time.time() - t0, 1), "params M", sum(p.numel() for p in pred.model.parameters()) / 1e6)

rng = np.random.default_rng(0)
img = rng.integers(0, 255, (2160, 3840, 3), dtype=np.uint8)
K = np.array([[2000, 0, 1920], [0, 2000, 1080], [0, 0, 1]], np.float32)
for dtype in (None, "bf16"):
    for n in (1, 3, 8, 16):
        boxes = np.array([[1900 + 10 * i, 1000, 1000] for i in range(n)], np.float32)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if dtype else torch.autocast("cuda", enabled=False)
        for rep in range(3):
            torch.cuda.synchronize()
            t = time.time()
            with ctx:
                out = pred.predict(img, boxes, K)
            torch.cuda.synchronize()
            dt = time.time() - t
        print(dtype, "batch", n, "s", round(dt, 3), "crops/s", round(n / dt, 1),
              "mem GB", round(torch.cuda.max_memory_allocated() / 1e9, 1))
print({k: tuple(v.shape) for k, v in out.parameters.items()})
print(out.joints_camera.shape, out.keypoints_2d.shape, out.projection_valid.shape, out.vertices_camera.shape)
