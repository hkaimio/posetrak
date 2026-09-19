# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prototype_ball_cutie_segmentation.py — check whether Cutie video-object
segmentation (already used in production for *person* tracking, see
pipeline/pose/segmentation.py) can localize the reflective ball directly via
its mask centroid, as an alternative/complement to the reflective-dot blob
detector this session's other ball-tracking prototypes used.

Background (2026-09-16, Harri): "What if we'd add support for segmenting
these other objects to the cutie segmentation pass? ... I expect that for
our ball case this could even be sufficient for tracking it (just calculate
centroid of the mask) ... I'd be interested in testing it just with the
ball" -- explicitly scoped down from the larger "arbitrary object by text
description" idea to just this one concrete test.

CutieSegmentor.process_video()'s automatic init path (_build_init_mask) is
person-detector-driven (YOLOX in "human" mode) -- useless for a ball, and
actively wrong here since Nelli herself is in every frame: it would find
her, not the ball, and happily use her bbox instead of failing loudly. So
this script builds its own SAM2 init mask from a manually-supplied box
prompt (a real, already-known-good ball detection from the reflective-dot
data used earlier -- see the same session's prototype_ball_tracking.py),
bypassing _build_init_mask() entirely, then calls process_video() with that
precomputed init_mask.

Single camera, single throw window, no triangulation/tracker wiring yet --
this only answers "does the mask centroid look like the ball, and does
Cutie hold onto it through the same bounce-blur/occlusion difficulty the
dot detector struggled with." Read-only against the session DB and the
source video; writes only a CSV + optional debug frames to the output dir
given on the command line.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import cv2
import numpy as np

CAMERA_LABEL = "gopro-11_mini_01"
VIDEO_PATH = "D:/mocap/2026-09-06-kare-tests/nelli-gopro11_01-4k-120fps.MP4"
SESSION_DB = "D:/mocap/2026-09-06-kare-tests/2026-09-06-kare-tests.db"

# A real, manually-confirmed-clean ball detection from the reflective-dot
# data (prototype_ball_tracking.py's own manual inspection of this camera,
# throw 1): frame 9618, (x=1804, y=905, area=220) -- isolated, well inside
# the frame, ball clearly separated from her body at this instant.
INIT_FRAME = 9618
INIT_CENTER = (1804.0, 905.0)
INIT_HALF_SIZE = 30.0  # ball diameter here is ~17px; generous but still tight

# Throw 1 window (same as prototype_ball_tracking.py): steps 2816-2902 at
# 120Hz tracker rate from t0=33.620s maps to real seconds, but Cutie/OpenCV
# work in this camera's own raw video_frame numbers instead -- convert via
# its own actual_fps (capture_videos.actual_fps=119.88, first_video_frame=0)
# rather than reusing the tracker's frame grid.
START_FRAME = 9500
END_FRAME = 9900


def build_sam2_init_mask(frame_bgr: np.ndarray, box_xyxy: np.ndarray, device: str) -> np.ndarray:
    """Single-box SAM2 prompt -> (H, W) uint8 mask, values {0, 1}.
    Deliberately bypasses CutieSegmentor._build_init_mask()'s YOLOX person
    detector entirely -- see module docstring for why that path is actively
    wrong for a non-person object when a person is also in frame."""
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    h, w = frame_bgr.shape[:2]
    sam_model = build_sam2_hf("facebook/sam2.1-hiera-base-plus", device=device)
    predictor = SAM2ImagePredictor(sam_model)
    predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    masks, scores, _logits = predictor.predict(box=box_xyxy.astype(np.float32), multimask_output=False)
    m = masks[0] > 0.5
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    print(f"SAM2 init mask: {m.sum()} px, score={scores[0]:.3f}")
    return m.astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--debug-frames", action="store_true", help="Save annotated frames to output-dir")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from pipeline.pose.segmentation import CutieSegmentor

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, INIT_FRAME)
    ok, init_img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read init frame {INIT_FRAME} from {VIDEO_PATH}")

    cx, cy = INIT_CENTER
    box = np.array([cx - INIT_HALF_SIZE, cy - INIT_HALF_SIZE, cx + INIT_HALF_SIZE, cy + INIT_HALF_SIZE])
    init_mask = build_sam2_init_mask(init_img, box, args.device)

    # SAM2's own model builder (just used, above) and Cutie's get_default_model()
    # (called inside process_video() below) both initialize Hydra's global config
    # state via a relative-path hydra.initialize() -- Hydra only tolerates one
    # such initialization per process without an explicit clear() in between.
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()

    if args.debug_frames:
        vis = init_img.copy()
        cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        overlay = vis.copy()
        overlay[init_mask.astype(bool)] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        cv2.imwrite(str(out_dir / "init_frame_debug.png"), vis)

    segmentor = CutieSegmentor(device=args.device, max_internal_size=480, erosion_px=3)
    segmentor.process_video(
        VIDEO_PATH,
        init_frame=INIT_FRAME,
        persons={"ball": box},  # key name only; init_mask= below skips detection entirely
        init_mask=init_mask,
        start_frame=START_FRAME,
        end_frame=END_FRAME,
        verbose=True,
    )

    rows = []
    for frame_idx in sorted(segmentor._masks.keys()):
        m = segmentor._masks[frame_idx].get("ball")
        if m is None or not m.any():
            continue
        ys, xs = np.nonzero(m)
        rows.append((frame_idx, float(xs.mean()), float(ys.mean()), int(m.sum())))

    print(f"{len(rows)}/{END_FRAME - START_FRAME} frames with a ball mask")
    out_path = out_dir / "cutie_ball_throw1_gopro11mini.csv"
    with open(out_path, "w") as f:
        f.write("video_frame,cx,cy,mask_area_px\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.2f},{r[2]:.2f},{r[3]}\n")
    print(f"wrote {out_path}")

    if args.debug_frames and rows:
        cap = cv2.VideoCapture(VIDEO_PATH)
        for frame_idx, cx, cy, area in rows[::10]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, img = cap.read()
            if not ok:
                continue
            m = segmentor._masks[frame_idx]["ball"]
            overlay = img.copy()
            overlay[m] = (0, 0, 255)
            vis = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
            cv2.circle(vis, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.png"), vis)
        cap.release()


if __name__ == "__main__":
    main()
