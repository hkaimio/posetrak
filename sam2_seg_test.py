#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""sam2_seg_test.py — Visualise SAM2 segmentation masks + keypoint quality scores.

For each sampled frame of test.mp4 this script saves a composite image:
  LEFT  Original frame with SAM2 mask overlays (one colour per person)
  RIGHT Per-person crops with keypoints colour-coded by mask quality score:
          green  = clearly inside mask (score 1.0)
          yellow = boundary zone      (score 0.5)
          red    = outside mask       (score 0.0)
          grey   = unavailable        (score -1.0)

Usage:
    python sam2_seg_test.py                    # defaults
    python sam2_seg_test.py --step 5           # denser sampling
    python sam2_seg_test.py --video other.mp4  # different clip

Models needed (auto-downloaded):
    yolo11x.pt     — person detection (initial bboxes for SAM2)
    sam2.1_b.pt    — SAM2 video segmentation (fast, ~150 MB)
    yolo11x-pose.pt — keypoint estimation for visualisation
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

VIDEO_PATH  = Path.home() / "projects/mocap_videos/test.mp4"
OUT_DIR     = Path.home() / "projects/mocap_videos/sam2_seg_results"
YOLO_MODEL  = "yolo11x.pt"
SAM2_MODEL  = "sam2.1_b.pt"
POSE_MODEL  = "yolo11x-pose.pt"
FRAME_STEP  = 15
CONF_YOLO   = 0.30
CONF_POSE   = 0.25
DEVICE      = "cuda"
EROSION_PX  = 5

# Colours per person (BGR)
PERSON_COLORS = [
    (0, 200, 255),   # cyan
    (0, 255, 100),   # green
    (200, 100, 255), # purple
    (0, 100, 255),   # orange
]

# COCO-17 skeleton for pose visualisation
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Score → circle colour (BGR)
SCORE_COLOR = {
    1.0:  (0, 220,  60),  # green  = inside
    0.5:  (0, 220, 220),  # yellow = boundary
    0.0:  (0,   0, 220),  # red    = outside
   -1.0:  (100, 100, 100), # grey  = unavailable
}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def mask_overlay(frame: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.35) -> np.ndarray:
    """Return a copy of frame with a coloured mask overlay."""
    out = frame.copy()
    overlay = np.zeros_like(out)
    overlay[mask] = color
    return cv2.addWeighted(out, 1.0, overlay, alpha, 0)


def draw_keypoints_quality(
    img: np.ndarray,
    kpts: np.ndarray,        # (17, 3)  x, y, conf  — full-frame coords
    scores: np.ndarray,      # (17,)    quality scores
    offset_x: int = 0,
    offset_y: int = 0,
    conf_thr: float = 0.15,
) -> None:
    """Draw keypoints colour-coded by mask quality score on img in-place."""
    n = min(17, kpts.shape[0])  # limit to body keypoints for vis
    for a, b in COCO_SKELETON:
        if a >= n or b >= n:
            continue
        if kpts[a, 2] < conf_thr or kpts[b, 2] < conf_thr:
            continue
        xa = int(kpts[a, 0]) - offset_x
        ya = int(kpts[a, 1]) - offset_y
        xb = int(kpts[b, 0]) - offset_x
        yb = int(kpts[b, 1]) - offset_y
        cv2.line(img, (xa, ya), (xb, yb), (200, 200, 200), 1, cv2.LINE_AA)
    for i in range(n):
        if kpts[i, 2] < conf_thr:
            continue
        x = int(kpts[i, 0]) - offset_x
        y = int(kpts[i, 1]) - offset_y
        s = float(scores[i]) if i < len(scores) else -1.0
        # find nearest score key
        nearest = min(SCORE_COLOR.keys(), key=lambda k: abs(k - s))
        color = SCORE_COLOR[nearest]
        cv2.circle(img, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)


def label_bar(width: int, text: str, height: int = 22) -> np.ndarray:
    bar = np.full((height, width, 3), 25, dtype=np.uint8)
    cv2.putText(bar, text, (6, height - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    return bar


def resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == 0 or img.shape[1] == 0:
        return np.zeros((h, max(h, 1), 3), dtype=np.uint8)
    scale = h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * scale)), h),
                      interpolation=cv2.INTER_AREA)


def column_header(img: np.ndarray, text: str) -> np.ndarray:
    h = 18
    hdr = np.full((h, img.shape[1], 3), 45, dtype=np.uint8)
    cv2.putText(hdr, text, (4, h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 200, 255), 1, cv2.LINE_AA)
    return np.concatenate([hdr, img], axis=0)


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def detect_persons_yolo(frame: np.ndarray, yolo, conf: float) -> np.ndarray:
    """Return (N, 4) xyxy bboxes for persons detected by YOLO."""
    res = yolo(frame, classes=[0], conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4))
    return res.boxes.xyxy.cpu().numpy()


def match_pose_to_box(
    pose_kpts: np.ndarray,    # (N_people, 17, 3)
    pose_boxes: np.ndarray,   # (N_people, 4) xyxy
    target_box: np.ndarray,   # (4,) xyxy
    min_iou: float = 0.2,
) -> np.ndarray | None:
    """Return keypoints for the pose detection best matching target_box."""
    if pose_kpts is None or len(pose_kpts) == 0:
        return None
    tx1, ty1, tx2, ty2 = target_box
    t_area = max(1.0, (tx2 - tx1) * (ty2 - ty1))
    best_iou, best_i = min_iou, None
    for i, pb in enumerate(pose_boxes):
        ix1, iy1 = max(tx1, pb[0]), max(ty1, pb[1])
        ix2, iy2 = min(tx2, pb[2]), min(ty2, pb[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            continue
        p_area = max(1.0, (pb[2] - pb[0]) * (pb[3] - pb[1]))
        iou = inter / (t_area + p_area - inter)
        if iou > best_iou:
            best_iou, best_i = iou, i
    return pose_kpts[best_i] if best_i is not None else None


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process(
    video_path: Path,
    out_dir: Path,
    step: int,
    max_persons: int,
    device: str,
) -> None:
    from ultralytics import YOLO
    # SAM2Segmentor lives in the pipeline package
    sys.path.insert(0, str(Path(__file__).parent / "python"))
    from pipeline.pose.segmentation import SAM2Segmentor, SCORE_UNAVAILABLE

    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO…")
    yolo  = YOLO(YOLO_MODEL)
    pose_model = YOLO(POSE_MODEL)

    # Step 1: detect persons on the first frame to get SAM2 init bboxes
    cap = cv2.VideoCapture(str(video_path))
    ok, first_frame = cap.read()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not ok:
        raise RuntimeError(f"Cannot read {video_path}")
    print(f"Video: {total} frames @ {fps:.2f} fps")

    init_boxes = detect_persons_yolo(first_frame, yolo, CONF_YOLO)
    n_persons = min(len(init_boxes), max_persons)
    if n_persons == 0:
        print("No persons detected on first frame — aborting.")
        return

    person_ids = [f"person_{i}" for i in range(n_persons)]
    persons = {
        pid: (0, init_boxes[i])
        for i, pid in enumerate(person_ids)
    }
    print(f"Initialising SAM2 with {n_persons} persons: {init_boxes[:n_persons]}")

    # Step 2: run SAM2 on the full video, store all masks
    seg = SAM2Segmentor(model_name=SAM2_MODEL, device=device, erosion_px=EROSION_PX)
    t0 = time.time()
    seg.process_video(video_path, persons, verbose=False)
    print(f"SAM2 done in {time.time() - t0:.0f}s")

    # Step 3: iterate frames, run pose, draw composites
    print(f"Rendering every {step}th frame…")
    cap   = cv2.VideoCapture(str(video_path))
    fh, fw = first_frame.shape[:2]
    saved  = 0
    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        # Run pose on full frame
        pose_res  = pose_model(frame, conf=CONF_POSE, verbose=False)[0]
        pose_kpts = (pose_res.keypoints.data.cpu().numpy()
                     if pose_res.keypoints is not None else np.zeros((0, 17, 3)))
        pose_boxes = (pose_res.boxes.xyxy.cpu().numpy()
                      if pose_res.boxes is not None else np.zeros((0, 4)))

        # Build left panel: overview with mask overlays
        overview = frame.copy()
        for i, pid in enumerate(person_ids):
            mask = seg.get_mask(frame_idx, pid)
            if mask is not None:
                color = PERSON_COLORS[i % len(PERSON_COLORS)]
                overview = mask_overlay(overview, mask, color)
        # Draw person labels
        for i, pid in enumerate(person_ids):
            box = persons[pid][1].astype(int)
            cv2.putText(overview, pid, (box[0], max(0, box[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        PERSON_COLORS[i % len(PERSON_COLORS)], 2)

        # Build right panel: per-person crops with quality-coded keypoints
        person_panels = []
        for i, pid in enumerate(person_ids):
            box = init_boxes[i].astype(int)
            # Use pose result box if available (person may have moved)
            if len(pose_boxes) > 0:
                kpts17 = match_pose_to_box(pose_kpts, pose_boxes, init_boxes[i])
            else:
                kpts17 = None

            # Crop from current frame around the init bbox (rough)
            pad = 20
            x1 = max(0, box[0] - pad)
            y1 = max(0, box[1] - pad)
            x2 = min(fw, box[2] + pad)
            y2 = min(fh, box[3] + pad)
            crop = frame[y1:y2, x1:x2].copy()

            if kpts17 is not None:
                # Get quality scores for the 17 body keypoints
                kpts_xy = kpts17[:, :2]  # (17, 2)
                scores17 = seg.get_keypoint_scores(frame_idx, pid, kpts_xy)
                draw_keypoints_quality(crop, kpts17, scores17,
                                       offset_x=x1, offset_y=y1)

                in_mask  = int((scores17 >= 0.5).sum())
                out_mask = int((scores17 == 0.0).sum())
                header_text = (f"{pid}  in:{in_mask}  out:{out_mask}")
            else:
                header_text = f"{pid}  (no pose)"

            crop = column_header(crop, header_text)
            person_panels.append(crop)

        # Combine panels
        if person_panels:
            target_h = max(p.shape[0] for p in person_panels)
            panels_resized = [resize_to_height(p, target_h) for p in person_panels]
            right = np.concatenate(panels_resized, axis=1)
        else:
            right = np.zeros((100, 200, 3), dtype=np.uint8)

        # Scale overview to match right panel height
        target_h = max(right.shape[0], 1)
        left = resize_to_height(overview, target_h)
        left = column_header(left, "SAM2 masks (overview)")
        # column_header added 18 px; resize right and sep to match left
        if right.shape[0] != left.shape[0]:
            right = resize_to_height(right, left.shape[0])
        sep  = np.full((left.shape[0], 3, 3), 50, dtype=np.uint8)
        comp = np.concatenate([left, sep, right], axis=1)

        bar_text = (
            f"f{frame_idx:05d}  {n_persons} persons  "
            f"SAM2 masks + quality-coded keypoints"
        )
        comp = np.concatenate([label_bar(comp.shape[1], bar_text), comp], axis=0)

        out_path = out_dir / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(out_path), comp, [cv2.IMWRITE_JPEG_QUALITY, 88])
        saved += 1

        if (frame_idx // step) % 20 == 0 and frame_idx > 0:
            elapsed = time.time() - t0
            pct = 100 * frame_idx / total
            print(f"  {frame_idx:5d}/{total}  ({pct:.0f}%)  "
                  f"{saved} saved  {elapsed:.0f}s")

        frame_idx += 1

    cap.release()
    print(f"\nDone. {saved} images → {out_dir}  ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video",       default=str(VIDEO_PATH))
    ap.add_argument("--out",         default=str(OUT_DIR))
    ap.add_argument("--step",        type=int, default=FRAME_STEP,
                    help="Process every Nth frame (default: %(default)s)")
    ap.add_argument("--max-persons", type=int, default=4,
                    help="Max persons to track (default: %(default)s)")
    ap.add_argument("--device",      default=DEVICE)
    args = ap.parse_args()

    process(
        video_path=Path(args.video),
        out_dir=Path(args.out),
        step=args.step,
        max_persons=args.max_persons,
        device=args.device,
    )
