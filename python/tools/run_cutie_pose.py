#!/home/harri/projects/tests/Cutie/venv/bin/python
"""run_cutie_pose.py — Cutie ROI + RTMPose detection pipeline.

For each camera in a source detection run:
  1. Initialise Cutie from the source run (SAM2 bbox prompt → labeled mask).
  2. Run Cutie bidirectionally to propagate per-person masks.
  3. At each frame: derive tight bbox from Cutie mask → smart-pad → RTMPose batch.
  4. Score resulting keypoints against the Cutie mask.
  5. Write a new detection_run + seg_quality_run to the DB.
  6. Optionally write a per-camera debug video.

Usage::

    python run_cutie_pose.py \\
        --db ~/projects/mocap_videos/ukemi-tommi-20260509.db \\
        --source-run-id 8bfded7f-8f42-46a6-9ae8-c51a4f0dbd2d \\
        [--max-padding 20]       # max px to add per side of Cutie bbox
        [--erosion-px 5]         # erosion radius for quality boundary zone
        [--max-dim 1920]         # downscale 4K to 1080p for Cutie + RTMPose
        [--init-offset 0.0]      # fraction into range for SAM2 init (0=start)
        [--device cuda]
        [--cameras CAM_ID ...]   # process only these shot_video_ids
        [--debug-video DIR]      # directory for debug videos (one .mp4 per camera)
        [--debug-every N]        # render every Nth frame to debug video (default 1)
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sqlite3
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PYTHON_DIR = _SCRIPT_DIR.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from pipeline.pose.segmentation import (
    _find_cutie_dir,
    _score_keypoints,
    encode_scores,
    SCORE_UNAVAILABLE,
)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# RTMPose model config
_RTMPOSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
    "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
)
_RTMPOSE_INPUT_HW = (288, 384)  # height, width

# Per-person BGR colors for debug overlays
_COLORS = [
    (0, 200, 0),    # green
    (200, 50, 50),  # blue-ish
    (50, 50, 200),  # red
    (0, 200, 200),  # yellow
    (200, 0, 200),  # magenta
]

_PANEL_H = 480        # debug video frame height
_PERSON_PANEL_W = 300  # fixed width per person panel


# ---------------------------------------------------------------------------
# DB helpers (no posetrak package dependency)
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS seg_quality_runs (
    id             TEXT PRIMARY KEY,
    shot_id        TEXT NOT NULL REFERENCES captures(id),
    trial_id       TEXT REFERENCES trials(id),
    time_start_s   REAL NOT NULL,
    time_end_s     REAL NOT NULL,
    created_at     TEXT NOT NULL,
    quality_source TEXT NOT NULL DEFAULT 'cutie',
    erosion_px     INTEGER NOT NULL DEFAULT 5,
    mask_dir       TEXT,
    notes          TEXT
);

CREATE TABLE IF NOT EXISTS keypoint_obs_quality (
    seg_run_id    TEXT    NOT NULL,
    shot_video_id TEXT    NOT NULL,
    video_frame   INTEGER NOT NULL,
    track_id      INTEGER NOT NULL,
    quality_blob  BLOB    NOT NULL,
    PRIMARY KEY (seg_run_id, shot_video_id, video_frame, track_id)
);

CREATE INDEX IF NOT EXISTS idx_keypoint_obs_quality_video_frame
    ON keypoint_obs_quality (shot_video_id, video_frame);
"""


def _open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn


def _create_detection_run(
    conn: sqlite3.Connection,
    shot_id: str,
    sync_config_id: str,
    time_start_s: float,
    time_end_s: float,
) -> str:
    run_id = str(uuid.uuid4())
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO detection_runs "
        "(id, shot_id, sync_config_id, time_start_s, time_end_s, "
        " detector_model, pose_model, detector_version, pose_version, "
        " detector_conf, pose_conf_threshold, "
        " pose_input_width, pose_input_height, status, created_at) "
        "VALUES (?,?,?,?,?,'cutie-sam2','rtmpose-l-133kp','','',0.0,0.3,?,?,'running',?)",
        (run_id, shot_id, sync_config_id, time_start_s, time_end_s,
         _RTMPOSE_INPUT_HW[1], _RTMPOSE_INPUT_HW[0], now),
    )
    conn.commit()
    return run_id


def _mark_run_complete(conn: sqlite3.Connection, run_id: str, status: str = "complete") -> None:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        "UPDATE detection_runs SET status=?, completed_at=? WHERE id=?",
        (status, now, run_id),
    )
    conn.commit()


def _load_video_list(conn: sqlite3.Connection, source_run_id: str, cameras: list[str] | None) -> list[dict]:
    rows = conn.execute(
        """
        SELECT dk.shot_video_id, cv.file_path, cv.actual_fps,
               MIN(dk.video_frame) AS first_frame, MAX(dk.video_frame) AS last_frame
        FROM detection_keypoints dk
        JOIN capture_videos cv ON cv.id = dk.shot_video_id
        WHERE dk.detection_run_id = ?
        GROUP BY dk.shot_video_id
        """,
        (source_run_id,),
    ).fetchall()
    result = [dict(r) for r in rows]
    if cameras:
        result = [r for r in result if r["shot_video_id"] in cameras]
    return result


def _load_assignments(conn: sqlite3.Connection, run_id: str, svid: str) -> list[tuple]:
    rows = conn.execute(
        "SELECT track_id, person_name, first_frame, last_frame "
        "FROM detection_track_assignments WHERE detection_run_id=? AND shot_video_id=? "
        "ORDER BY first_frame",
        (run_id, svid),
    ).fetchall()
    return [(r["track_id"], r["person_name"], r["first_frame"], r["last_frame"]) for r in rows]


def _load_bboxes_at_frame(
    conn: sqlite3.Connection, run_id: str, svid: str, frame: int, track_ids: list[int]
) -> dict[int, np.ndarray]:
    ph = ",".join("?" * len(track_ids))
    rows = conn.execute(
        f"SELECT track_id, bbox_x, bbox_y, bbox_w, bbox_h FROM person_detections "
        f"WHERE detection_run_id=? AND shot_video_id=? AND video_frame=? "
        f"  AND track_id IN ({ph}) AND region_type='full_body'",
        (run_id, svid, frame, *track_ids),
    ).fetchall()
    result = {}
    for r in rows:
        if r["bbox_x"] is None:
            continue
        cx, cy, w, h = r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]
        result[r["track_id"]] = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
    return result


# (reuse) find init frame logic from add_seg_quality.py
def _find_init_frame(
    conn: sqlite3.Connection,
    run_id: str,
    svid: str,
    first: int,
    last: int,
    assignments: list[tuple],
    init_offset: float,
) -> tuple[int, dict[str, np.ndarray]]:
    all_persons = sorted({name for _, name, _, _ in assignments})
    n_want = len(all_persons)
    candidate = first + int((last - first) * init_offset)
    candidate = max(first, min(last, candidate))

    def _try(fi: int) -> dict[str, np.ndarray] | None:
        p2t = {}
        for tid, name, f0, f1 in assignments:
            if f0 <= fi <= f1 and name not in p2t:
                p2t[name] = tid
        if len(p2t) < n_want:
            return None
        bbs = _load_bboxes_at_frame(conn, run_id, svid, fi, list(p2t.values()))
        pb = {n: bbs[t] for n, t in p2t.items() if t in bbs}
        return pb if len(pb) == n_want else None

    for delta in range(301):
        for sign in ([0] if delta == 0 else [1, -1]):
            fi = candidate + sign * delta
            if fi < first or fi > last:
                continue
            pb = _try(fi)
            if pb is not None:
                log.info("  Init frame: %d (Δ%+d from candidate)", fi, fi - candidate)
                return fi, pb

    log.warning("  No frame with all %d persons; using candidate %d", n_want, candidate)
    return candidate, {}


# ---------------------------------------------------------------------------
# SAM2 init mask (copy from add_seg_quality.py)
# ---------------------------------------------------------------------------

def _build_rect_init_mask(frame: np.ndarray, bboxes: dict[str, np.ndarray]) -> np.ndarray:
    h, w = frame.shape[:2]
    names = list(bboxes.keys())
    mask = np.zeros((h, w), dtype=np.uint8)
    for j, name in enumerate(names):
        x1, y1, x2, y2 = [int(v) for v in bboxes[name]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        mask[y1:y2, x1:x2] = j + 1
    return mask


def _build_init_mask(frame: np.ndarray, bboxes: dict[str, np.ndarray]) -> np.ndarray:
    try:
        from ultralytics import SAM
    except ImportError:
        return _build_rect_init_mask(frame, bboxes)
    h, w = frame.shape[:2]
    names = list(bboxes.keys())
    bbs = np.array([bboxes[n] for n in names], dtype=np.float32)
    try:
        sam = SAM("sam2.1_b.pt")
        result = sam(frame, bboxes=bbs, imgsz=512, verbose=False)
    except Exception as exc:
        log.warning("SAM failed (%s) — rectangle fallback", exc)
        return _build_rect_init_mask(frame, bboxes)
    mask = np.zeros((h, w), dtype=np.uint8)
    if result and result[0].masks is not None:
        raw = result[0].masks.data.cpu().numpy()
        for j in range(min(len(names), len(raw))):
            m = raw[j].astype(bool)
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            mask[m] = j + 1
    n_found = int(mask.max())
    if n_found < len(names):
        log.warning("SAM gave %d/%d masks — filling missing with rectangles", n_found, len(names))
        rect = _build_rect_init_mask(frame, bboxes)
        for j in range(len(names)):
            lbl = j + 1
            if not np.any(mask == lbl):
                mask[(rect == lbl) & (mask == 0)] = lbl
    return mask


# ---------------------------------------------------------------------------
# Smart padding
# ---------------------------------------------------------------------------

def _mask_to_tight_bbox(labeled: np.ndarray, label: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(labeled == label)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _smart_pad_bbox(
    labeled: np.ndarray, label: int, tight: tuple[int, int, int, int], max_pad: int
) -> tuple[int, int, int, int]:
    """Expand tight bbox up to max_pad px per side, stopping before other persons."""
    x1, y1, x2, y2 = tight
    h, w = labeled.shape
    other = (labeled != 0) & (labeled != label)

    # left
    strip = other[y1:y2, :x1]
    blk = np.where(strip.any(axis=0))[0]
    gap = (x1 - int(blk[-1]) - 1) if blk.size else x1
    lp = min(max_pad, max(0, gap))

    # right
    strip = other[y1:y2, x2:]
    blk = np.where(strip.any(axis=0))[0]
    gap = int(blk[0]) if blk.size else (w - x2)
    rp = min(max_pad, max(0, gap))

    # top
    strip = other[:y1, x1:x2]
    blk = np.where(strip.any(axis=1))[0]
    gap = (y1 - int(blk[-1]) - 1) if blk.size else y1
    tp = min(max_pad, max(0, gap))

    # bottom
    strip = other[y2:, x1:x2]
    blk = np.where(strip.any(axis=1))[0]
    gap = int(blk[0]) if blk.size else (h - y2)
    bp = min(max_pad, max(0, gap))

    return max(0, x1 - lp), max(0, y1 - tp), min(w, x2 + rp), min(h, y2 + bp)


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------

def _overview_panel(
    proc_frame: np.ndarray,
    labeled: np.ndarray,
    persons: list[str],
    padded: list[tuple | None],
) -> np.ndarray:
    ph, pw = proc_frame.shape[:2]
    s = _PANEL_H / ph
    panel_w = max(1, int(pw * s))
    img = cv2.resize(proc_frame, (panel_w, _PANEL_H))

    # mask overlay
    lbl_sm = cv2.resize(labeled, (panel_w, _PANEL_H), interpolation=cv2.INTER_NEAREST)
    overlay = img.copy()
    for i, name in enumerate(persons):
        region = lbl_sm == (i + 1)
        overlay[region] = _COLORS[i % len(_COLORS)]
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    # bboxes + labels
    for i, (name, bbox) in enumerate(zip(persons, padded)):
        if bbox is None:
            continue
        color = _COLORS[i % len(_COLORS)]
        x1, y1, x2, y2 = [int(v * s) for v in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, name, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return img


def _person_panel(
    proc_frame: np.ndarray,
    labeled: np.ndarray,
    label: int,
    padded_bbox: tuple | None,
    keypoints_proc: np.ndarray | None,  # (133, 3) x,y,conf in proc coords
    name: str,
    color: tuple,
) -> np.ndarray:
    canvas = np.zeros((_PANEL_H, _PERSON_PANEL_W, 3), dtype=np.uint8)
    cv2.putText(canvas, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    if padded_bbox is None:
        cv2.putText(canvas, "not visible", (6, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
        return canvas

    x1, y1, x2, y2 = padded_bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(proc_frame.shape[1], x2)
    y2 = min(proc_frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return canvas

    crop = proc_frame[y1:y2, x1:x2].copy()

    # mask contour
    if labeled is not None:
        mask_crop = (labeled[y1:y2, x1:x2] == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, contours, -1, color, 2)

    # keypoints / skeleton
    if keypoints_proc is not None:
        kp = keypoints_proc.copy()
        kp[:, 0] -= x1
        kp[:, 1] -= y1
        try:
            from rtmlib.visualization.draw import draw_skeleton
            # draw_skeleton expects (N, n_kp, 2) + (N, n_kp) — add batch dim
            crop = draw_skeleton(crop, kp[:, :2][None], kp[:, 2][None], kpt_thr=0.3, radius=3, line_width=2)
        except Exception:
            for pt in kp:
                if pt[2] >= 0.3:
                    cv2.circle(crop, (int(pt[0]), int(pt[1])), 3, color, -1)

    # scale to panel height
    ch, cw = crop.shape[:2]
    if ch > 0 and cw > 0:
        crop_s = _PANEL_H / ch
        new_w = min(_PERSON_PANEL_W, int(cw * crop_s))
        crop = cv2.resize(crop, (new_w, _PANEL_H))
        canvas[:, :new_w] = crop

    return canvas


def _make_debug_frame(
    proc_frame: np.ndarray,
    labeled: np.ndarray,
    persons: list[str],
    padded: list[tuple | None],
    kp_proc: list[np.ndarray | None],
) -> np.ndarray:
    panels = [_overview_panel(proc_frame, labeled, persons, padded)]
    for i, name in enumerate(persons):
        color = _COLORS[i % len(_COLORS)]
        panels.append(_person_panel(proc_frame, labeled, i + 1, padded[i], kp_proc[i], name, color))
    return np.hstack(panels)


# ---------------------------------------------------------------------------
# Core camera processing
# ---------------------------------------------------------------------------

def _load_cutie(device: str):
    cutie_dir = _find_cutie_dir()
    if str(cutie_dir) not in sys.path:
        sys.path.insert(0, str(cutie_dir))
    try:
        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
    except Exception:
        pass
    from cutie.utils.get_default_model import get_default_model
    return get_default_model()


def _proc_scale(src_w: int, src_h: int, max_dim: int) -> tuple[float, int, int]:
    scale = min(1.0, max_dim / max(src_w, src_h))
    return scale, int(src_w * scale + 0.5), int(src_h * scale + 0.5)


def _process_camera(
    conn: sqlite3.Connection,
    new_run_id: str,
    seg_run_id: str,
    source_run_id: str,
    shot_video_id: str,
    video_path: Path,
    first_frame: int,
    last_frame: int,
    init_frame: int,
    init_mask_full: np.ndarray,
    persons_ordered: list[str],
    rtmpose,
    cutie_model,
    max_pad: int,
    erosion_px: int,
    max_dim: int,
    device: str,
    debug_video_dir: Path | None,
    debug_every: int,
) -> int:
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    cutie_dir = _find_cutie_dir()
    if str(cutie_dir) not in sys.path:
        sys.path.insert(0, str(cutie_dir))
    from cutie.inference.inference_core import InferenceCore

    cap_probe = cv2.VideoCapture(str(video_path))
    src_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    cap_probe.release()

    scale, proc_w, proc_h = _proc_scale(src_w, src_h, max_dim)
    if scale < 1.0:
        log.info("  scaling %dx%d → %dx%d (%.2f×)", src_w, src_h, proc_w, proc_h, scale)

    init_mask_proc = (
        cv2.resize(init_mask_full, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
        if scale < 1.0 else init_mask_full
    )

    n_persons = len(persons_ordered)
    objects_list = list(range(1, n_persons + 1))
    tid_for = {name: i for i, name in enumerate(persons_ordered)}  # person → track_id (0-indexed)
    device_type = device.split(":")[0]

    # accumulate DB rows
    det_rows: list[tuple] = []
    kp_rows: list[tuple] = []
    crop_rows: list[tuple] = []
    quality_rows: list[tuple] = []
    track_spans: dict[int, tuple[int, int]] = {}  # track_id → (first, last)

    _CROP_H = 240
    _CROP_JPEG_Q = 75

    # debug frames: (frame_idx, img)
    debug_frames: list[tuple[int, np.ndarray]] = []

    def _resize_to_proc(bgr: np.ndarray) -> np.ndarray:
        return cv2.resize(bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr

    def _to_tensor(bgr: np.ndarray) -> torch.Tensor:
        proc = _resize_to_proc(bgr)
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        return to_tensor(Image.fromarray(rgb)).to(device).float()

    def _process_frame(fi: int, labeled: np.ndarray, proc_frame: np.ndarray) -> None:
        """Score + RTMPose for one frame; append to accumulation lists."""
        padded_bboxes: list[tuple] = []
        present_idx: list[int] = []

        for i in range(n_persons):
            tight = _mask_to_tight_bbox(labeled, i + 1)
            if tight is None:
                continue
            padded = _smart_pad_bbox(labeled, i + 1, tight, max_pad)
            padded_bboxes.append(padded)
            present_idx.append(i)

        if not padded_bboxes:
            return

        bboxes_arr = np.array(padded_bboxes, dtype=np.float32)  # (N, 4) xyxy proc coords
        kp_all, sc_all = rtmpose(proc_frame, bboxes_arr)
        # kp_all: (N, 133, 2) proc coords; sc_all: (N, 133)

        for j, pi in enumerate(present_idx):
            name = persons_ordered[pi]
            track_id = tid_for[name]
            px1, py1, px2, py2 = padded_bboxes[j]
            kp_proc = kp_all[j]   # (133, 2)
            sc = sc_all[j]        # (133,)

            # quality score against Cutie mask
            mask_i = (labeled == (pi + 1))
            quality = _score_keypoints(mask_i, kp_proc.astype(np.float32), erosion_px)
            quality_rows.append((seg_run_id, shot_video_id, fi, track_id, encode_scores(quality)))

            # scale back to full-frame coords for DB
            inv = 1.0 / scale if scale < 1.0 else 1.0
            kp_full = kp_proc * inv  # (133, 2)
            kp_with_conf = np.concatenate([kp_full, sc[:, None]], axis=1).astype(np.float32)

            # bbox cx,cy,w,h in full-frame coords
            cx = (px1 + px2) / 2 * inv
            cy = (py1 + py2) / 2 * inv
            bw = (px2 - px1) * inv
            bh = (py2 - py1) * inv

            noise_scale = bw / _RTMPOSE_INPUT_HW[1]

            det_rows.append((
                new_run_id, shot_video_id, fi, track_id,
                "full_body", "cutie-rtmpose",
                cx, cy, bw, bh, 1.0,
            ))
            kp_rows.append((
                new_run_id, shot_video_id, fi, track_id,
                "full_body", kp_with_conf.tobytes(), noise_scale,
            ))

            first, last_s = track_spans.get(track_id, (fi, fi))
            track_spans[track_id] = (min(first, fi), max(last_s, fi))

            # thumbnail crop from proc_frame
            cx1, cy1, cx2, cy2 = max(0, px1), max(0, py1), min(proc_frame.shape[1], px2), min(proc_frame.shape[0], py2)
            if cx2 > cx1 and cy2 > cy1:
                tile = proc_frame[cy1:cy2, cx1:cx2]
                if tile.shape[0] > _CROP_H:
                    ts = _CROP_H / tile.shape[0]
                    tile = cv2.resize(tile, (int(tile.shape[1] * ts), _CROP_H))
                ok, buf = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, _CROP_JPEG_Q])
                if ok:
                    crop_rows.append((
                        shot_video_id, fi, "person_crop", track_id, "full_body",
                        tile.shape[1], tile.shape[0], buf.tobytes(), new_run_id,
                        int(cx1 * inv), int(cy1 * inv),
                        int((cx2 - cx1) * inv), int((cy2 - cy1) * inv),
                    ))

        # debug frame
        if debug_video_dir is not None and fi % debug_every == 0:
            all_padded = [None] * n_persons
            all_kp: list[np.ndarray | None] = [None] * n_persons
            for j, pi in enumerate(present_idx):
                all_padded[pi] = padded_bboxes[j]
                kp_with_conf_proc = np.concatenate([kp_all[j], sc_all[j][:, None]], axis=1)
                all_kp[pi] = kp_with_conf_proc  # (133, 3) in proc coords
            dbg = _make_debug_frame(proc_frame, labeled, persons_ordered, all_padded, all_kp)
            debug_frames.append((fi, dbg))

    def _flush(batch_size: int = 500) -> None:
        if len(det_rows) >= batch_size:
            _flush_all()

    def _flush_all() -> None:
        if det_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO person_detections "
                "(detection_run_id, shot_video_id, video_frame, track_id, region_type, "
                " model_name, bbox_x, bbox_y, bbox_w, bbox_h, confidence) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                det_rows,
            )
            det_rows.clear()
        if kp_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO detection_keypoints "
                "(detection_run_id, shot_video_id, video_frame, track_id, region_type, "
                " keypoints, noise_scale) VALUES (?,?,?,?,?,?,?)",
                kp_rows,
            )
            kp_rows.clear()
        if crop_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO frame_cache_entries "
                "(shot_video_id, frame_idx, cache_type, track_id, region_type, "
                " width_px, height_px, image_data, detection_run_id, "
                " src_x, src_y, src_w, src_h) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                crop_rows,
            )
            crop_rows.clear()
        conn.commit()

    init_mask_t = torch.from_numpy(init_mask_proc).to(device)

    # ── Forward pass: [init_frame, last_frame] ──────────────────────────
    log.info("  Forward pass: [%d, %d]  init=%d", init_frame, last_frame, init_frame)
    fwd = InferenceCore(cutie_model, cfg=cutie_model.cfg)
    fwd.max_internal_size = 480
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
    seeded = False
    n_fwd = 0

    with torch.inference_mode(), torch.amp.autocast(device_type):
        for fi in range(init_frame, last_frame + 1):
            ret, bgr = cap.read()
            if not ret:
                break
            proc = _resize_to_proc(bgr)
            img_t = _to_tensor(bgr)
            if not seeded:
                out = fwd.step(img_t, init_mask_t, objects=objects_list)
                seeded = True
            else:
                out = fwd.step(img_t)
            labeled = fwd.output_prob_to_mask(out).cpu().numpy()
            _process_frame(fi, labeled, proc)
            _flush()
            n_fwd += 1
            if n_fwd % 300 == 0:
                log.info("    fwd %d / %d", fi, last_frame)

    cap.release()
    log.info("  Forward: %d frames", n_fwd)

    # ── Backward pass: [first_frame, init_frame) ────────────────────────
    n_bwd_target = init_frame - first_frame
    if n_bwd_target > 0:
        log.info("  Backward pass: reading %d frames [%d, %d)...", n_bwd_target, first_frame, init_frame)
        bwd_frames: list[tuple[int, np.ndarray]] = []
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        for fi in range(first_frame, init_frame):
            ret, bgr = cap.read()
            if not ret:
                break
            bwd_frames.append((fi, _resize_to_proc(bgr)))
        cap.release()

        # Re-read init frame for seeding
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
        ret, init_bgr = cap.read()
        cap.release()

        if ret and bwd_frames:
            bwd = InferenceCore(cutie_model, cfg=cutie_model.cfg)
            bwd.max_internal_size = 480
            init_t = _to_tensor(init_bgr)
            with torch.inference_mode(), torch.amp.autocast(device_type):
                bwd.step(init_t, init_mask_t, objects=objects_list)
                for fi, proc in reversed(bwd_frames):
                    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
                    img_t = to_tensor(Image.fromarray(rgb)).to(device).float()
                    out = bwd.step(img_t)
                    labeled = bwd.output_prob_to_mask(out).cpu().numpy()
                    _process_frame(fi, labeled, proc)
                    _flush()
            log.info("  Backward: %d frames", len(bwd_frames))

    _flush_all()

    # ── Write track spans + assignments ─────────────────────────────────
    track_rows = [
        (str(uuid.uuid4()), new_run_id, shot_video_id, tid, first, last)
        for tid, (first, last) in track_spans.items()
    ]
    if track_rows:
        conn.executemany(
            "INSERT OR REPLACE INTO person_tracks "
            "(id, detection_run_id, shot_video_id, track_id, first_frame, last_frame) "
            "VALUES (?,?,?,?,?,?)",
            track_rows,
        )

    assign_rows = [
        (new_run_id, shot_video_id, tid_for[name], name,
         track_spans.get(tid_for[name], (first_frame, last_frame))[0],
         track_spans.get(tid_for[name], (first_frame, last_frame))[1])
        for name in persons_ordered
        if tid_for[name] in track_spans
    ]
    if assign_rows:
        conn.executemany(
            "INSERT OR REPLACE INTO detection_track_assignments "
            "(detection_run_id, shot_video_id, track_id, person_name, first_frame, last_frame) "
            "VALUES (?,?,?,?,?,?)",
            assign_rows,
        )
    conn.commit()

    # ── Write quality rows ───────────────────────────────────────────────
    if quality_rows:
        conn.executemany(
            "INSERT OR REPLACE INTO keypoint_obs_quality "
            "(seg_run_id, shot_video_id, video_frame, track_id, quality_blob) "
            "VALUES (?,?,?,?,?)",
            quality_rows,
        )
        conn.commit()
        log.info("  Quality rows written: %d", len(quality_rows))

    # ── Debug video ──────────────────────────────────────────────────────
    if debug_video_dir is not None and debug_frames:
        debug_frames.sort(key=lambda x: x[0])
        _, sample = debug_frames[0]
        dh, dw = sample.shape[:2]
        cam_label = Path(video_path).stem
        out_path = debug_video_dir / f"{cam_label}.mp4"
        out_fps = video_fps / max(1, debug_every)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_path), fourcc, out_fps, (dw, dh))
        for _, img in debug_frames:
            vw.write(img)
        vw.release()
        log.info("  Debug video: %s  (%d frames, %.1f fps)", out_path, len(debug_frames), out_fps)

    return n_fwd + (len(bwd_frames) if n_bwd_target > 0 else 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cutie ROI + RTMPose detection pipeline."
    )
    parser.add_argument("--db", required=True, help="Path to session SQLite DB")
    parser.add_argument("--source-run-id", required=True,
                        help="detection_runs.id to initialise from (track assignments + init bboxes)")
    parser.add_argument("--max-padding", type=int, default=20,
                        help="Max px to add per side of Cutie mask bbox (default: 20)")
    parser.add_argument("--erosion-px", type=int, default=5,
                        help="Erosion radius for boundary zone scoring (default: 5)")
    parser.add_argument("--max-dim", type=int, default=1920,
                        help="Max processing dimension; 4K downscaled to 1080p (default: 1920)")
    parser.add_argument("--init-offset", type=float, default=0.0,
                        help="Fraction into detection range for SAM2 init (default: 0 = start)")
    parser.add_argument("--device", default="cuda",
                        help='Torch device (default: cuda)')
    parser.add_argument("--cameras", nargs="*", default=None,
                        help="Process only these shot_video_ids")
    parser.add_argument("--debug-video", default=None,
                        help="Directory for debug .mp4 files (one per camera)")
    parser.add_argument("--debug-every", type=int, default=1,
                        help="Render every Nth frame to debug video (default: 1)")
    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        log.error("DB not found: %s", db_path)
        sys.exit(1)

    conn = _open_db(str(db_path))

    # Load source run metadata
    src_row = conn.execute("SELECT * FROM detection_runs WHERE id=?", (args.source_run_id,)).fetchone()
    if src_row is None:
        log.error("Source detection run not found: %s", args.source_run_id)
        sys.exit(1)
    log.info("Source run: %s  model=%s  status=%s",
             args.source_run_id, src_row["pose_model"], src_row["status"])

    # Create new detection run
    new_run_id = _create_detection_run(
        conn,
        shot_id=src_row["shot_id"],
        sync_config_id=src_row["sync_config_id"],
        time_start_s=src_row["time_start_s"],
        time_end_s=src_row["time_end_s"],
    )
    log.info("New detection run: %s", new_run_id)

    # Create seg quality run -- time-range-scoped on the capture now, not
    # tied to new_run_id (see docs/roadmap/features/segmentation-reuse/
    # segmentation-reuse-design.md); backfilled from the source run's own
    # shot/time range, same as new_run_id itself inherits above.
    seg_run_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO seg_quality_runs "
        "(id, shot_id, trial_id, time_start_s, time_end_s, created_at, "
        " quality_source, erosion_px, notes) "
        "VALUES (?,?,?,?,?,?,'cutie',?,?)",
        (seg_run_id, src_row["shot_id"], src_row["trial_id"],
         src_row["time_start_s"], src_row["time_end_s"],
         datetime.datetime.now(datetime.timezone.utc).isoformat(),
         args.erosion_px,
         f"cutie-rtmpose run from source {args.source_run_id[:8]}"),
    )
    conn.commit()
    log.info("Seg quality run: %s", seg_run_id)

    debug_dir = Path(args.debug_video).expanduser().resolve() if args.debug_video else None

    # Load video list
    videos = _load_video_list(conn, args.source_run_id, args.cameras)
    if not videos:
        log.error("No videos found for source run (cameras filter: %s)", args.cameras)
        _mark_run_complete(conn, new_run_id, "failed")
        sys.exit(1)
    log.info("Processing %d camera(s)", len(videos))

    # Pre-load models once
    log.info("Loading Cutie model...")
    cutie = _load_cutie(args.device)
    log.info("Cutie loaded.")

    log.info("Loading RTMPose...")
    from rtmlib.tools.pose_estimation import RTMPose as _RTMPose
    rtmpose = _RTMPose(
        _RTMPOSE_URL,
        model_input_size=(_RTMPOSE_INPUT_HW[0], _RTMPOSE_INPUT_HW[1]),
        to_openpose=False,
        backend="onnxruntime",
        device=args.device,
    )
    log.info("RTMPose loaded.")

    total_frames = 0
    for vid in videos:
        svid = vid["shot_video_id"]
        vpath = Path(vid["file_path"])
        first, last = vid["first_frame"], vid["last_frame"]
        log.info("\n=== Camera: %s  frames [%d, %d] ===", vpath.name, first, last)

        if not vpath.exists():
            log.error("  Video not found: %s — skipping", vpath)
            continue

        assignments = _load_assignments(conn, args.source_run_id, svid)
        if not assignments:
            log.warning("  No track assignments for %s — skipping", vpath.name)
            continue

        all_persons = sorted({name for _, name, _, _ in assignments})
        log.info("  Persons: %s", ", ".join(all_persons))

        init_frame, person_bboxes = _find_init_frame(
            conn, args.source_run_id, svid, first, last, assignments, args.init_offset
        )
        if not person_bboxes:
            log.error("  No init bboxes found — skipping")
            continue

        persons_ordered = list(person_bboxes.keys())
        log.info("  Init frame: %d  persons: %s", init_frame, persons_ordered)

        # Read init frame
        cap = cv2.VideoCapture(str(vpath))
        cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
        ret, init_bgr = cap.read()
        cap.release()
        if not ret:
            log.error("  Cannot read init frame %d — skipping", init_frame)
            continue

        # Build SAM2 init mask
        log.info("  Building SAM2 init mask...")
        try:
            init_mask = _build_init_mask(init_bgr, person_bboxes)
        except Exception as exc:
            log.error("  Init mask failed: %s — skipping", exc)
            continue
        if int(init_mask.max()) == 0:
            log.error("  Empty init mask — skipping")
            continue

        try:
            n = _process_camera(
                conn=conn,
                new_run_id=new_run_id,
                seg_run_id=seg_run_id,
                source_run_id=args.source_run_id,
                shot_video_id=svid,
                video_path=vpath,
                first_frame=first,
                last_frame=last,
                init_frame=init_frame,
                init_mask_full=init_mask,
                persons_ordered=persons_ordered,
                rtmpose=rtmpose,
                cutie_model=cutie,
                max_pad=args.max_padding,
                erosion_px=args.erosion_px,
                max_dim=args.max_dim,
                device=args.device,
                debug_video_dir=debug_dir,
                debug_every=args.debug_every,
            )
            total_frames += n
            log.info("  Camera done: %d frames", n)
        except Exception as exc:
            log.error("  Camera failed: %s", exc, exc_info=True)

    _mark_run_complete(conn, new_run_id)
    log.info(
        "\nDone.  detection_run=%s  seg_run=%s  total_frames=%d",
        new_run_id, seg_run_id, total_frames,
    )
    conn.close()


if __name__ == "__main__":
    main()
