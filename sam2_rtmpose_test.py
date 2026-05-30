#!/usr/bin/env python3
"""sam2_rtmpose_test.py — SAM2 segmentation + RTMPose-133 keypoint quality analysis.

Output: an MP4 video at the source frame rate showing every frame:

  TOP    Full frame with per-person SAM2 mask overlays.
         Frames where SAM2 likely switched person identity are highlighted:
           RED border    = potential switch (mask IoU vs previous frame < 0.30)
           YELLOW border = uncertain      (IoU 0.30–0.50)

  BOTTOM Per-person panels — one column per person, each column has 3 stacked
         crops derived from the SAM2 mask bounding box that frame:
           Row A  All 133 RTMPose keypoints, coloured by confidence
           Row B  Same keypoints coloured by mask quality score
                    green  = inside mask  (1.0)
                    yellow = boundary     (0.5)
                    red    = outside mask (0.0)
                    grey   = unavailable  (−1.0)
           Row C  Only keypoints with quality ≥ 0.5 shown (UKF would update
                  on these); suppressed keypoints drawn as small grey dots

Usage:
    python sam2_rtmpose_test.py
    python sam2_rtmpose_test.py --video other.mp4 --out result.mp4

Models (auto-downloaded):
    yolo11x.pt      person detection for SAM2 init bboxes (~100 MB)
    sam2.1_b.pt     SAM2 video tracker (~150 MB)
    RTMPose-l-133   wholebody keypoints via rtmlib (~140 MB)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Layout constants  (output: 1920 × 1080)
# ---------------------------------------------------------------------------

OUTPUT_W       = 1920
OVERVIEW_H     = 480          # top section height
PERSON_ROW_H   = 600          # bottom section height (3 stacked crops)
OUTPUT_H       = OVERVIEW_H + PERSON_ROW_H   # 1080
MAX_PERSONS    = 3
PERSON_COL_W   = OUTPUT_W // MAX_PERSONS      # 640 per person
CROP_COL_W     = PERSON_COL_W                 # full person column width
CROP_H_EACH    = (PERSON_ROW_H - 3 * 18) // 3  # height per crop row (−3 headers)
HEADER_H       = 18

# Switch detection thresholds (consecutive-frame mask IoU)
SWITCH_RED_THR    = 0.30
SWITCH_YELLOW_THR = 0.50

# Visualisation
PERSON_COLORS = [        # BGR, one per person
    (0, 200, 255),       # cyan
    (0, 255, 100),       # green
    (200, 100, 255),     # purple
]
MASK_ALPHA    = 0.32
CROP_PAD      = 25       # padding around mask bbox for person crop

# Quality-score → keypoint colour (BGR)
SCORE_COLORS = {
     1.0: (0, 220,  60),   # green  — inside mask
     0.5: (0, 220, 220),   # yellow — boundary
     0.0: (0,   0, 220),   # red    — outside mask
    -1.0: (80,  80,  80),  # grey   — unavailable
}
SUPPRESSED_COLOR = (60, 60, 60)  # tiny dot for filtered-out keypoints in row C

# COCO-17 body skeleton (subset drawn over all 133 RTMPose keypoints)
BODY_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
]

# RTMPose-l wholebody (133 kp)
RTM_URL        = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
    "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
)
RTM_INPUT_SIZE = (288, 384)   # (width, height)

YOLO_MODEL = "yolo11x.pt"
SAM2_MODEL = "sam2.1_b.pt"

# ---------------------------------------------------------------------------
# Helpers: mask geometry
# ---------------------------------------------------------------------------

def mask_tight_bbox(
    mask: np.ndarray, pad: int = CROP_PAD
) -> tuple[int, int, int, int] | None:
    """Return (x1,y1,x2,y2) bbox of nonzero region with padding, or None."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(w, int(xs.max()) + pad),
        min(h, int(ys.max()) + pad),
    )


def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = int((m1 & m2).sum())
    union = int((m1 | m2).sum())
    return inter / union if union > 0 else 0.0


def mask_overlay(frame: np.ndarray, mask: np.ndarray,
                 color: tuple, alpha: float = MASK_ALPHA) -> np.ndarray:
    out = frame.copy()
    overlay = np.zeros_like(out)
    overlay[mask] = color
    return cv2.addWeighted(out, 1.0, overlay, alpha, 0)


# ---------------------------------------------------------------------------
# Helpers: keypoint drawing
# ---------------------------------------------------------------------------

def _score_color(s: float) -> tuple:
    nearest = min(SCORE_COLORS, key=lambda k: abs(k - s))
    return SCORE_COLORS[nearest]


def draw_kpts(
    img: np.ndarray,
    kpts: np.ndarray,    # (K, 2)  pixel coords in frame space
    conf: np.ndarray,    # (K,)    confidence
    scores: np.ndarray,  # (K,)    quality scores (for colour)
    mode: str,           # "all" | "quality" | "filtered"
    offset_x: int, offset_y: int,
    conf_thr: float = 0.15,
) -> None:
    """Draw keypoints on img in-place.

    mode:
        "all"       — draw all kp, colour by confidence (green/yellow/red)
        "quality"   — draw all kp, colour by quality score
        "filtered"  — draw in-mask kp normally, suppressed kp as grey dots
    """
    n = kpts.shape[0]

    # skeleton (body only, COCO-17)
    for a, b in BODY_SKELETON:
        if a >= n or b >= n:
            continue
        if conf[a] < conf_thr or conf[b] < conf_thr:
            continue
        if mode == "filtered" and (scores[a] < 0.5 or scores[b] < 0.5):
            continue
        xa = int(kpts[a, 0]) - offset_x
        ya = int(kpts[a, 1]) - offset_y
        xb = int(kpts[b, 0]) - offset_x
        yb = int(kpts[b, 1]) - offset_y
        cv2.line(img, (xa, ya), (xb, yb), (180, 180, 180), 1, cv2.LINE_AA)

    for i in range(n):
        if conf[i] < conf_thr:
            continue
        x = int(kpts[i, 0]) - offset_x
        y = int(kpts[i, 1]) - offset_y
        s = float(scores[i])

        if mode == "all":
            c = conf[i]
            color = ((0,220,60) if c > 0.6 else (0,200,255) if c > 0.3
                     else (40,40,200))
            radius = 3
        elif mode == "quality":
            color = _score_color(s)
            radius = 3
        else:  # filtered
            if s >= 0.5:
                color = _score_color(s)
                radius = 3
            else:
                color = SUPPRESSED_COLOR
                radius = 2

        cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
        if mode != "filtered" or s >= 0.5:
            cv2.circle(img, (x, y), radius, (255,255,255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Helpers: frame assembly
# ---------------------------------------------------------------------------

def header_bar(w: int, text: str, h: int = HEADER_H,
               bg: tuple = (45,45,45)) -> np.ndarray:
    bar = np.full((h, w, 3), bg[0], dtype=np.uint8)
    bar[:] = bg
    cv2.putText(bar, text, (5, h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180,200,255), 1, cv2.LINE_AA)
    return bar


def fit_to_box(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Scale img to fit inside target_w × target_h with letterboxing (black)."""
    if img.shape[0] == 0 or img.shape[1] == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / img.shape[1], target_h / img.shape[0])
    new_w = max(1, int(img.shape[1] * scale))
    new_h = max(1, int(img.shape[0] * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def make_overview(
    frame: np.ndarray,
    masks: dict[str, np.ndarray | None],
    person_ids: list[str],
    bboxes_init: np.ndarray,
    switch_iou: dict[str, float],
) -> np.ndarray:
    """Build the top overview panel (OUTPUT_W × OVERVIEW_H)."""
    overview = frame.copy()

    # Mask overlays
    for i, pid in enumerate(person_ids):
        m = masks.get(pid)
        if m is not None:
            overview = mask_overlay(overview, m, PERSON_COLORS[i])

    # Switch borders per person (drawn on overview after scaling)
    # Scale first, then add labels
    scaled = cv2.resize(overview, (OUTPUT_W, OVERVIEW_H), interpolation=cv2.INTER_AREA)

    sx = OUTPUT_W / frame.shape[1]
    sy = OVERVIEW_H / frame.shape[0]

    for i, pid in enumerate(person_ids):
        m = masks.get(pid)
        iou = switch_iou.get(pid, 1.0)
        if iou < SWITCH_RED_THR:
            border_color = (0, 0, 220)   # red
            label = "SWITCH?"
        elif iou < SWITCH_YELLOW_THR:
            border_color = (0, 200, 220)  # yellow
            label = "DRIFT?"
        else:
            border_color = None
            label = ""

        if m is not None:
            bb = mask_tight_bbox(m, pad=0)
        else:
            bb = bboxes_init[i].astype(int) if i < len(bboxes_init) else None

        if bb is not None:
            x1s = int(bb[0] * sx); y1s = int(bb[1] * sy)
            x2s = int(bb[2] * sx); y2s = int(bb[3] * sy)
            color = PERSON_COLORS[i]
            cv2.rectangle(scaled, (x1s, y1s), (x2s, y2s), color, 2)
            cv2.putText(scaled, pid, (x1s+4, max(y1s+16, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            if border_color:
                cv2.rectangle(scaled, (x1s-2, y1s-2), (x2s+2, y2s+2),
                              border_color, 3)
                cv2.putText(scaled, label, (x1s+4, y1s+36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2,
                            cv2.LINE_AA)

    return scaled


def make_person_column(
    frame: np.ndarray,
    mask: np.ndarray | None,
    pid: str,
    kpts: np.ndarray | None,    # (133, 2) frame-space coords
    conf: np.ndarray | None,    # (133,)
    scores: np.ndarray,         # (133,) quality scores
    iou: float,
    color: tuple,
) -> np.ndarray:
    """Build one person column (PERSON_COL_W × PERSON_ROW_H) with 3 stacked crops."""

    # Derive crop bbox from mask (or fall back to full frame)
    if mask is not None:
        bb = mask_tight_bbox(mask)
    else:
        bb = None

    if bb is None:
        bb = (0, 0, frame.shape[1], frame.shape[0])

    x1, y1, x2, y2 = bb
    raw_crop = frame[y1:y2, x1:x2]

    rows = []
    mode_labels = [
        ("all",      f"{pid}  all kp"),
        ("quality",  f"quality  in:{int((scores>=0.5).sum())} out:{int((scores==0.0).sum())}"),
        ("filtered", f"filtered ≥0.5  ({int((scores>=0.5).sum())}/133)"),
    ]

    for mode, label in mode_labels:
        crop = raw_crop.copy()

        # Overlay semi-transparent mask on the crop
        if mask is not None:
            mask_crop = mask[y1:y2, x1:x2]
            overlay = np.zeros_like(crop)
            overlay[mask_crop] = color
            crop = cv2.addWeighted(crop, 1.0, overlay, 0.22, 0)

        if kpts is not None and conf is not None:
            draw_kpts(crop, kpts, conf, scores, mode,
                      offset_x=x1, offset_y=y1)

        # Scale to fit column
        scaled = fit_to_box(crop, CROP_COL_W, CROP_H_EACH)

        # Switch indicator — tinted border
        if iou < SWITCH_RED_THR:
            cv2.rectangle(scaled, (0,0),
                          (scaled.shape[1]-1, scaled.shape[0]-1),
                          (0,0,180), 3)
        elif iou < SWITCH_YELLOW_THR:
            cv2.rectangle(scaled, (0,0),
                          (scaled.shape[1]-1, scaled.shape[0]-1),
                          (0,180,200), 2)

        hdr = header_bar(CROP_COL_W, label)
        rows.append(np.concatenate([hdr, scaled], axis=0))

    column = np.concatenate(rows, axis=0)
    # Pad / crop to exact PERSON_ROW_H
    h_diff = PERSON_ROW_H - column.shape[0]
    if h_diff > 0:
        pad = np.zeros((h_diff, PERSON_COL_W, 3), dtype=np.uint8)
        column = np.concatenate([column, pad], axis=0)
    else:
        column = column[:PERSON_ROW_H]
    return column


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process(
    video_path: Path,
    out_path: Path,
    device: str,
) -> None:
    from ultralytics import YOLO
    from rtmlib.tools.pose_estimation import RTMPose
    sys.path.insert(0, str(Path(__file__).parent / "python"))
    from pipeline.pose.segmentation import SAM2Segmentor, SCORE_UNAVAILABLE

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading YOLO…")
    yolo = YOLO(YOLO_MODEL)

    print("Loading RTMPose-133…")
    pose_model = RTMPose(
        RTM_URL,
        model_input_size=RTM_INPUT_SIZE,
        to_openpose=False,
        backend="onnxruntime",
        device=device,
    )

    # ── Detect persons on first frame for SAM2 init ──────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    ok, first_frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read {video_path}")

    fh, fw = first_frame.shape[:2]
    res = yolo(first_frame, classes=[0], conf=0.3, verbose=False)[0]
    init_boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0,4))
    n_persons = min(len(init_boxes), MAX_PERSONS)
    if n_persons == 0:
        raise RuntimeError("No persons detected on first frame")

    person_ids = [f"p{i}" for i in range(n_persons)]
    persons = {pid: (0, init_boxes[i]) for i, pid in enumerate(person_ids)}
    print(f"Video: {total} frames @ {fps:.2f} fps  |  {n_persons} persons detected")
    print(f"Init bboxes:\n{init_boxes[:n_persons]}")

    # ── SAM2 pass ─────────────────────────────────────────────────────────────
    print("Running SAM2 (full video)…")
    seg = SAM2Segmentor(model_name=SAM2_MODEL, device=device, erosion_px=5)
    t0 = time.time()
    seg.process_video(video_path, persons, verbose=True)
    print(f"SAM2 done in {time.time()-t0:.0f}s  ({len(seg._masks)} frames with masks)")

    # ── Set up VideoWriter ────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (OUTPUT_W, OUTPUT_H))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {out_path}")
    print(f"Writing video → {out_path}  ({OUTPUT_W}×{OUTPUT_H} @ {fps:.2f} fps)")

    # ── Render loop ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    prev_masks: dict[str, np.ndarray | None] = {pid: None for pid in person_ids}
    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Retrieve SAM2 masks
        masks: dict[str, np.ndarray | None] = {}
        for pid in person_ids:
            masks[pid] = seg.get_mask(frame_idx, pid)

        # Switch detection: consecutive-frame mask IoU
        switch_iou: dict[str, float] = {}
        for pid in person_ids:
            m_prev = prev_masks[pid]
            m_curr = masks[pid]
            if m_prev is not None and m_curr is not None:
                switch_iou[pid] = mask_iou(m_prev, m_curr)
            else:
                switch_iou[pid] = 1.0  # no data → don't flag

        # RTMPose: collect valid bboxes (from mask or init)
        person_bboxes = []
        for i, pid in enumerate(person_ids):
            bb = mask_tight_bbox(masks[pid]) if masks[pid] is not None else None
            if bb is None:
                # fall back to init box
                bb = tuple(init_boxes[i].astype(int)) if i < len(init_boxes) else (0,0,fw,fh)
            person_bboxes.append(list(bb))

        # Run RTMPose with all person bboxes at once
        try:
            kpts_all, conf_all = pose_model(frame, bboxes=person_bboxes)
            # kpts_all: (N, 133, 2),  conf_all: (N, 133)
        except Exception as e:
            print(f"  RTMPose error frame {frame_idx}: {e}")
            kpts_all = None
            conf_all = None

        # Quality scores for each person
        quality_scores: dict[str, np.ndarray] = {}
        for i, pid in enumerate(person_ids):
            m = masks[pid]
            if kpts_all is not None and i < len(kpts_all):
                kp_xy = kpts_all[i]   # (133, 2)
                quality_scores[pid] = seg.get_keypoint_scores(
                    frame_idx, pid, kp_xy
                )
            else:
                quality_scores[pid] = np.full(133, SCORE_UNAVAILABLE, dtype=np.float32)

        # ── Assemble frame ────────────────────────────────────────────────
        # Top: overview
        overview = make_overview(frame, masks, person_ids,
                                 init_boxes, switch_iou)

        # Bottom: per-person columns
        columns = []
        for i, pid in enumerate(person_ids):
            kp  = kpts_all[i]   if kpts_all is not None and i < len(kpts_all) else None
            cf  = conf_all[i]   if conf_all is not None and i < len(conf_all) else None
            col = make_person_column(
                frame, masks[pid], pid, kp, cf,
                quality_scores[pid], switch_iou[pid],
                PERSON_COLORS[i],
            )
            columns.append(col)

        # Pad to MAX_PERSONS columns if fewer persons
        while len(columns) < MAX_PERSONS:
            columns.append(np.zeros((PERSON_ROW_H, PERSON_COL_W, 3), dtype=np.uint8))

        bottom = np.concatenate(columns[:MAX_PERSONS], axis=1)  # (PERSON_ROW_H, OUTPUT_W)

        # Frame counter + switch summary bar
        n_switches = sum(1 for v in switch_iou.values() if v < SWITCH_RED_THR)
        bar_color = (0,0,30) if n_switches == 0 else (0,0,60)
        bar_txt = (f"f{frame_idx:05d} / {total}   "
                   + "   ".join(f"{pid} iou={switch_iou.get(pid,1.):.2f}"
                                for pid in person_ids)
                   + (f"   *** {n_switches} SWITCH(ES) ***" if n_switches else ""))
        info_bar = np.full((22, OUTPUT_W, 3), bar_color[0], dtype=np.uint8)
        info_bar[:] = bar_color
        cv2.putText(info_bar, bar_txt, (6, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (0,60,255) if n_switches else (180,200,180), 1, cv2.LINE_AA)

        # Stack: overview + info_bar + bottom
        top_h = OVERVIEW_H - 22
        overview_trimmed = overview[:top_h]
        out_frame = np.concatenate([overview_trimmed, info_bar, bottom], axis=0)

        # Ensure exact output size
        if out_frame.shape != (OUTPUT_H, OUTPUT_W, 3):
            out_frame = cv2.resize(out_frame, (OUTPUT_W, OUTPUT_H))

        writer.write(out_frame)

        # Progress
        if frame_idx % 300 == 0 and frame_idx > 0:
            elapsed = time.time() - t0
            fps_render = frame_idx / elapsed
            eta = (total - frame_idx) / fps_render if fps_render > 0 else 0
            print(f"  {frame_idx:5d}/{total}  ({100*frame_idx/total:.0f}%)  "
                  f"{fps_render:.1f} fps  ETA {eta:.0f}s")

        prev_masks = {pid: masks[pid] for pid in person_ids}
        frame_idx += 1

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"\nDone. {frame_idx} frames → {out_path}  ({elapsed:.0f}s, "
          f"{frame_idx/elapsed:.1f} fps)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video",  default=str(Path.home() / "projects/mocap_videos/test.mp4"))
    ap.add_argument("--out",    default=str(Path.home() / "projects/mocap_videos/sam2_rtmpose.mp4"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    process(Path(args.video), Path(args.out), args.device)
