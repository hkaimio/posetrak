#!/home/harri/projects/tests/Cutie/venv/bin/python
"""cutie_rtmpose_test.py — Cutie video segmentation + RTMPose-133 quality scoring.

Replaces the per-frame SAM approach (sam2_yolo_test.py) with Cutie (XMem++),
which maintains object identity through occlusion and crossings via a
propagating memory store — no per-frame tracking prompts needed.

Pipeline
--------
  Init (configurable frame):
    1. YOLO person detection → N bboxes sorted by x-centre
    2. SAM single-frame → N initial masks
    3. Combine into a Cutie-format labeled init mask
       (pixel value 0=background, 1=p0, 2=p1, 3=p2)
    4. Cutie.step(image, init_mask, objects=[1..N])  ← memory seeding

  Per frame:
    5. Cutie.step(image)  → labeled mask (H×W uint8, 0=bg, 1..N=persons)
    6. Derive per-person bbox from mask
    7. RTMPose-133 → 133 wholebody keypoints from bbox crop
    8. Quality score (1.0/0.5/0.0/−1.0) via erosion check
    9. Render → MP4

Usage
-----
    ./cutie_rtmpose_test.py
    ./cutie_rtmpose_test.py --video path/to/video.mp4 --out result.mp4
    ./cutie_rtmpose_test.py --init-frame 120 --n-persons 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

# Add Cutie project to path (before any cutie imports)
CUTIE_DIR = Path(__file__).parent.parent / "tests" / "Cutie"
sys.path.insert(0, str(CUTIE_DIR))

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model


# ---------------------------------------------------------------------------
# Layout  (1920 × 1080)
# ---------------------------------------------------------------------------

OUTPUT_W      = 1920
OVERVIEW_H    = 480
PERSON_ROW_H  = 600
OUTPUT_H      = OVERVIEW_H + PERSON_ROW_H
MAX_PERSONS   = 3
PERSON_COL_W  = OUTPUT_W // MAX_PERSONS        # 640
CROP_H_EACH   = (PERSON_ROW_H - 3 * 20) // 3  # ≈160 per sub-row
HEADER_H      = 20
INFO_BAR_H    = 22

PERSON_COLORS = [(0, 200, 255), (0, 255, 100), (200, 100, 255)]  # BGR  p0/p1/p2

MASK_ALPHA     = 0.30
CROP_PAD       = 40

# Model / inference settings
YOLO_CONF             = 0.30
SAM_IMGSZ             = 512
EROSION_PX            = 5
CUTIE_MAX_INTERNAL    = 480   # shorter edge; reduce to 360 if VRAM is tight

RTM_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
    "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
)
RTM_INPUT_SIZE = (288, 384)

# Keypoint quality → circle colour (BGR)
SCORE_COLORS = {
     1.0: (0,  220,  60),   # green    inside eroded mask
     0.5: (0,  220, 220),   # cyan     boundary zone
     0.0: (0,    0, 220),   # red      outside mask
    -1.0: (80,  80,  80),   # dark     unavailable
}

# COCO-17 body skeleton subset for drawing
BODY_SKEL = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
]

DEFAULT_VIDEO = Path.home() / "projects/mocap_videos/test.mp4"
DEFAULT_OUT   = Path.home() / "projects/mocap_videos/cutie_rtmpose.mp4"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mask_tight_bbox(mask: np.ndarray, pad: int = 0) -> tuple[int,int,int,int] | None:
    """Return (x1,y1,x2,y2) tight bbox of a bool mask, with optional padding."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r1, r2 = np.where(rows)[0][[0, -1]]
    c1, c2 = np.where(cols)[0][[0, -1]]
    h, w = mask.shape
    return (
        max(0, c1 - pad), max(0, r1 - pad),
        min(w, c2 + pad), min(h, r2 + pad),
    )


def keypoint_quality_scores(
    mask: np.ndarray,
    kpts_xy: np.ndarray,
    erosion_px: int = EROSION_PX,
) -> np.ndarray:
    """Float32 quality score per keypoint: 1.0/0.5/0.0/−1.0."""
    h, w = mask.shape
    n = len(kpts_xy)
    scores = np.full(n, 0.0, dtype=np.float32)

    if erosion_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erosion_px + 1, 2 * erosion_px + 1)
        )
        mask_eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    else:
        mask_eroded = mask

    for i, (x, y) in enumerate(kpts_xy):
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            scores[i] = -1.0
        elif mask_eroded[yi, xi]:
            scores[i] = 1.0
        elif mask[yi, xi]:
            scores[i] = 0.5
        # else stays 0.0
    return scores


def draw_kpts(
    img: np.ndarray,
    kpts: np.ndarray,   # (K, 2) x,y in full-frame coords
    conf: np.ndarray,   # (K,)
    scores: np.ndarray, # (K,)
    mode: str,          # "all" | "quality" | "filtered"
    ox: int, oy: int,   # crop origin
) -> None:
    """Draw keypoints and skeleton on *img* (a crop starting at ox, oy)."""
    k = kpts - np.array([ox, oy])

    # Skeleton (COCO-17)
    for a, b in BODY_SKEL:
        if a >= len(kpts) or b >= len(kpts):
            continue
        if mode == "filtered" and (scores[a] < 0.5 or scores[b] < 0.5):
            continue
        xa, ya = k[a].astype(int)
        xb, yb = k[b].astype(int)
        cv2.line(img, (xa, ya), (xb, yb), (100, 100, 100), 1, cv2.LINE_AA)

    # Points
    for i, (x, y) in enumerate(k.astype(int)):
        s = scores[i]
        c = conf[i]
        if mode == "filtered" and s < 0.5:
            continue
        color = SCORE_COLORS.get(s, SCORE_COLORS[-1.0]) if mode != "all" else (
            (0, int(255 * c), int(255 * (1 - c)))
        )
        r = 2 if s < 0.5 else 3
        cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)


def header_bar(w: int, text: str, bg=(30, 30, 30)) -> np.ndarray:
    bar = np.full((HEADER_H, w, 3), 0, dtype=np.uint8)
    bar[:] = bg
    cv2.putText(bar, text, (4, HEADER_H - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 200, 255), 1, cv2.LINE_AA)
    return bar


def fit_to_box(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    if img.shape[0] == 0 or img.shape[1] == 0:
        return np.zeros((th, tw, 3), dtype=np.uint8)
    s = min(tw / img.shape[1], th / img.shape[0])
    nw = max(1, int(img.shape[1] * s))
    nh = max(1, int(img.shape[0] * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = r
    return canvas


# ---------------------------------------------------------------------------
# Init mask generation  (frame 0 → YOLO + SAM)
# ---------------------------------------------------------------------------

def build_init_mask(
    frame: np.ndarray,
    yolo,
    sam,
    n_persons: int,
) -> tuple[np.ndarray, list[tuple[int,int,int,int]]]:
    """Run YOLO detection + SAM on a single frame → (H,W) uint8 labeled mask.

    Persons are sorted left-to-right; label 1 = leftmost person.
    In overlap zones the higher-label (rightmost) person takes precedence.

    Returns
    -------
    init_mask : (H, W) uint8, values 0..n_persons
    sorted_bboxes : list of (x1,y1,x2,y2) in the same order as labels 1..N
    """
    h, w = frame.shape[:2]

    # YOLO detection (class 0 = person)
    det = yolo(frame, classes=[0], conf=YOLO_CONF, verbose=False)
    if not det or det[0].boxes is None or len(det[0].boxes) == 0:
        raise ValueError("No persons detected in init frame — try a different --init-frame")

    boxes_xyxy = det[0].boxes.xyxy.cpu().numpy()   # (N, 4)
    x_centers  = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
    sort_idx   = np.argsort(x_centers)[:n_persons]
    boxes_xyxy = boxes_xyxy[sort_idx]
    n_found    = len(boxes_xyxy)

    if n_found < n_persons:
        print(f"  [warn] only {n_found} persons detected (requested {n_persons})")

    # SAM single-frame masks from bboxes
    sam_res = sam(frame, bboxes=boxes_xyxy, imgsz=SAM_IMGSZ, verbose=False)

    init_mask = np.zeros((h, w), dtype=np.uint8)

    if sam_res and sam_res[0].masks is not None:
        masks_raw = sam_res[0].masks.data.cpu().numpy()  # (M, Hm, Wm) float or bool
        for j in range(min(n_found, len(masks_raw))):
            m = masks_raw[j] > 0.5
            if m.shape != (h, w):
                m = cv2.resize(
                    m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            # Later labels overwrite earlier ones in overlap zones
            init_mask[m] = j + 1

    sorted_bboxes = [tuple(boxes_xyxy[j].astype(int)) for j in range(n_found)]
    return init_mask, sorted_bboxes


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def make_overview(
    frame: np.ndarray,
    masks: dict[str, np.ndarray],
    bboxes: dict[str, tuple],
    person_labels: list[str],
    init_frame_idx: int,
    current_fi: int,
) -> np.ndarray:
    overview = frame.copy()
    for i, pid in enumerate(person_labels):
        m = masks.get(pid)
        if m is not None:
            ov = np.zeros_like(overview)
            ov[m] = PERSON_COLORS[i % len(PERSON_COLORS)]
            overview = cv2.addWeighted(overview, 1.0, ov, MASK_ALPHA, 0)

    scaled = cv2.resize(overview, (OUTPUT_W, OVERVIEW_H - INFO_BAR_H),
                        interpolation=cv2.INTER_AREA)
    sx = OUTPUT_W / frame.shape[1]
    sy = (OVERVIEW_H - INFO_BAR_H) / frame.shape[0]

    for i, pid in enumerate(person_labels):
        bb = bboxes.get(pid)
        if bb is None:
            continue
        col = PERSON_COLORS[i % len(PERSON_COLORS)]
        x1s = int(bb[0] * sx); y1s = int(bb[1] * sy)
        x2s = int(bb[2] * sx); y2s = int(bb[3] * sy)
        cv2.rectangle(scaled, (x1s, y1s), (x2s, y2s), col, 2)
        cv2.putText(scaled, pid, (x1s + 4, max(y1s + 16, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    # Info bar
    init_tag = " [INIT]" if current_fi == init_frame_idx else ""
    bar_text = f"frame {current_fi}{init_tag}  |  Cutie + RTMPose-133"
    bar = np.full((INFO_BAR_H, OUTPUT_W, 3), 20, dtype=np.uint8)
    cv2.putText(bar, bar_text, (8, INFO_BAR_H - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    return np.concatenate([scaled, bar], axis=0)


def make_person_column(
    frame: np.ndarray,
    mask: np.ndarray | None,
    pid: str,
    bbox: tuple | None,
    kpts: np.ndarray | None,
    conf: np.ndarray | None,
    scores: np.ndarray,
    color: tuple,
) -> np.ndarray:
    """Return one person column: PERSON_COL_W × PERSON_ROW_H."""
    # Determine crop region
    crop_bb = None
    if mask is not None:
        crop_bb = mask_tight_bbox(mask, pad=CROP_PAD)
    if crop_bb is None and bbox is not None:
        x1, y1, x2, y2 = bbox
        crop_bb = (
            max(0, x1 - CROP_PAD), max(0, y1 - CROP_PAD),
            min(frame.shape[1], x2 + CROP_PAD),
            min(frame.shape[0], y2 + CROP_PAD),
        )
    if crop_bb is None:
        crop_bb = (0, 0, frame.shape[1], frame.shape[0])

    x1, y1, x2, y2 = crop_bb
    raw_crop = frame[y1:y2, x1:x2].copy()

    n_in    = int((scores >= 0.5).sum())  if kpts is not None else 0
    n_out   = int((scores == 0.0).sum())  if kpts is not None else 0
    n_avail = int((scores >= 0.0).sum())  if kpts is not None else 0

    rows = []
    configs = [
        ("all",      f"{pid}  all kp  (conf-coloured)"),
        ("quality",  f"quality  in:{n_in}  out:{n_out}  unavail:{133 - n_avail}"),
        ("filtered", f"UKF view  pass:{n_in}/133"),
    ]
    for mode, label in configs:
        crop = raw_crop.copy()
        if mask is not None:
            mc = mask[y1:y2, x1:x2]
            ov = np.zeros_like(crop)
            ov[mc] = color
            crop = cv2.addWeighted(crop, 1.0, ov, 0.20, 0)
        if kpts is not None:
            draw_kpts(crop, kpts, conf, scores, mode, x1, y1)
        scaled = fit_to_box(crop, PERSON_COL_W, CROP_H_EACH)
        hdr = header_bar(PERSON_COL_W, label)
        rows.append(np.concatenate([hdr, scaled], axis=0))

    col = np.concatenate(rows, axis=0)
    diff = PERSON_ROW_H - col.shape[0]
    if diff > 0:
        col = np.concatenate(
            [col, np.zeros((diff, PERSON_COL_W, 3), np.uint8)], axis=0
        )
    return col[:PERSON_ROW_H]


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process(
    video_path: Path,
    out_path: Path,
    device: str,
    n_persons: int,
    init_frame_idx: int,
) -> None:
    from ultralytics import YOLO, SAM
    from rtmlib.tools.pose_estimation import RTMPose

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading YOLO (person detection for init)…")
    yolo = YOLO("yolo11x.pt")

    print("Loading SAM2 (single-frame init mask)…")
    sam = SAM("sam2.1_b.pt")

    print("Loading Cutie (video object segmentation)…")
    cutie_model = get_default_model()
    processor   = InferenceCore(cutie_model, cfg=cutie_model.cfg)
    processor.max_internal_size = CUTIE_MAX_INTERNAL

    print("Loading RTMPose-133…")
    pose = RTMPose(
        RTM_URL, model_input_size=RTM_INPUT_SIZE,
        to_openpose=False, backend="onnxruntime", device=device,
    )

    # ── Video info ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    print(f"Video: {total} frames @ {fps:.2f} fps  ({fw}×{fh})")

    # ── Build init mask ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame_idx)
    ret, init_frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read frame {init_frame_idx} from {video_path}")

    print(f"Building init mask from frame {init_frame_idx} (YOLO + SAM)…")
    init_mask_np, init_bboxes = build_init_mask(init_frame, yolo, sam, n_persons)
    n_found = len(init_bboxes)
    objects_list   = list(range(1, n_found + 1))
    person_labels  = [f"p{i}" for i in range(n_found)]
    print(f"  Initialised {n_found} persons: {init_bboxes}")

    # ── Set up VideoWriter ────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUTPUT_W, OUTPUT_H)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter → {out_path}")
    print(f"Writing → {out_path}  ({OUTPUT_W}×{OUTPUT_H} @ {fps:.2f} fps)")

    # ── Main loop ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    t0  = time.time()
    fi  = 0
    cutie_initialized = False

    with torch.inference_mode(), torch.amp.autocast("cuda"):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ── Pre-init frames: write bare overview, skip Cutie ───────────
            if fi < init_frame_idx:
                overview = make_overview(frame, {}, {}, person_labels,
                                         init_frame_idx, fi)
                blank_row = np.zeros((PERSON_ROW_H, OUTPUT_W, 3), np.uint8)
                cv2.putText(
                    blank_row,
                    f"waiting for init at frame {init_frame_idx}…",
                    (20, PERSON_ROW_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 1, cv2.LINE_AA,
                )
                writer.write(np.concatenate([overview, blank_row], axis=0))
                fi += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_tensor = to_tensor(Image.fromarray(rgb)).to(device).float()

            # ── Cutie step ──────────────────────────────────────────────────
            if fi == init_frame_idx and not cutie_initialized:
                mask_tensor = torch.from_numpy(init_mask_np).to(device)
                output_prob = processor.step(image_tensor, mask_tensor,
                                             objects=objects_list)
                cutie_initialized = True
            else:
                output_prob = processor.step(image_tensor)

            labeled_mask = processor.output_prob_to_mask(output_prob).cpu().numpy()
            # labeled_mask: (H, W) uint8; value 0=bg, i=person i-1

            # ── Extract per-person masks & bboxes ───────────────────────────
            person_masks:  dict[str, np.ndarray] = {}
            person_bboxes: dict[str, tuple]      = {}
            for i, label in enumerate(person_labels):
                m = (labeled_mask == i + 1)
                if m.any():
                    person_masks[label]  = m
                    bb = mask_tight_bbox(m, pad=CROP_PAD)
                    if bb is not None:
                        person_bboxes[label] = bb

            # ── RTMPose-133 ─────────────────────────────────────────────────
            rtm_labels  = [l for l in person_labels if l in person_bboxes]
            rtm_bboxes  = [person_bboxes[l] for l in rtm_labels]

            person_kpts:   dict[str, np.ndarray] = {}
            person_conf:   dict[str, np.ndarray] = {}
            person_scores: dict[str, np.ndarray] = {}

            if rtm_bboxes:
                kpts_all, conf_all = pose(frame, np.array(rtm_bboxes, dtype=float))
                for j, label in enumerate(rtm_labels):
                    kp = kpts_all[j]   # (133, 2)
                    cf = conf_all[j]   # (133,)
                    person_kpts[label]  = kp
                    person_conf[label]  = cf
                    m = person_masks.get(label)
                    person_scores[label] = (
                        keypoint_quality_scores(m, kp)
                        if m is not None
                        else np.full(133, -1.0, dtype=np.float32)
                    )

            # ── Render ──────────────────────────────────────────────────────
            overview = make_overview(
                frame, person_masks, person_bboxes, person_labels,
                init_frame_idx, fi,
            )

            person_cols = []
            for i, label in enumerate(person_labels):
                col = make_person_column(
                    frame,
                    person_masks.get(label),
                    label,
                    person_bboxes.get(label),
                    person_kpts.get(label),
                    person_conf.get(label),
                    person_scores.get(label, np.full(133, -1.0, dtype=np.float32)),
                    PERSON_COLORS[i % len(PERSON_COLORS)],
                )
                person_cols.append(col)

            # Pad to MAX_PERSONS columns
            while len(person_cols) < MAX_PERSONS:
                person_cols.append(
                    np.zeros((PERSON_ROW_H, PERSON_COL_W, 3), np.uint8)
                )

            person_row = np.concatenate(person_cols[:MAX_PERSONS], axis=1)
            out_frame  = np.concatenate([overview, person_row], axis=0)
            writer.write(out_frame)

            if fi % 300 == 0 and fi > 0:
                elapsed = time.time() - t0
                fps_cur = fi / elapsed
                eta     = (total - fi) / fps_cur
                print(f"  {fi:5d}/{total}  ({100*fi//total}%)  "
                      f"{fps_cur:.1f} fps  ETA {eta:.0f}s")

            fi += 1

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"\nDone. {fi} frames → {out_path}  ({elapsed:.0f}s, {fi/elapsed:.1f} fps)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video",       type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--out",         type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--n-persons",   type=int,  default=3)
    parser.add_argument("--init-frame",  type=int,  default=0,
                        help="Frame index to use for YOLO+SAM initialisation")
    args = parser.parse_args()
    process(args.video, args.out, args.device, args.n_persons, args.init_frame)


if __name__ == "__main__":
    main()
