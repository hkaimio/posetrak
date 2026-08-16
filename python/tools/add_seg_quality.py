#!/home/harri/projects/tests/Cutie/venv/bin/python
"""add_seg_quality.py — Compute Cutie segmentation quality scores for a detection run.

Runs Cutie (XMem++) video segmentation on each camera video of a detection run
and writes per-keypoint quality scores into the ``keypoint_obs_quality`` table.

Quality scores are float32 values:
  1.0  — keypoint clearly inside the person mask (after erosion)
  0.5  — keypoint in the boundary zone (inside raw mask, outside eroded mask)
  0.0  — keypoint outside the person mask
 -1.0  — sentinel: no mask data for this frame/track

Scores are aligned with ``detection_keypoints``: one quality blob per
(seg_run_id, shot_video_id, video_frame, track_id).

Usage
-----
::

    python add_seg_quality.py \\
        --db ~/projects/mocap_videos/ukemi-tommi-20260509.db \\
        --detection-run-id 8bfded7f-8f42-46a6-9ae8-c51a4f0dbd2d \\
        [--init-offset 0]          # fractional position in detection range for init frame
        [--max-dim 1920]           # max processing dimension (scales 4K down to 1080p)
        [--erosion-px 5]
        [--debug-masks]            # save per-frame labeled masks to NPZ
        [--debug-every 30]         # save every Nth frame's mask (default 30)
        [--cameras CAM_ID ...]     # process only these shot_video_ids (default: all)

Requires: Cutie clone at CUTIE_DIR (or ../tests/Cutie), torch, ultralytics (SAM2).
This script must run inside the Cutie venv which has all these dependencies.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sqlite3
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure posetrak python/ is importable (for encode_scores / _score_keypoints)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PYTHON_DIR = _SCRIPT_DIR.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from pipeline.pose.segmentation import (
    _find_cutie_dir,
    _score_keypoints,
    encode_scores,
    SCORE_UNAVAILABLE,
    N_KEYPOINTS,
)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema migration
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


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _load_video_info(conn: sqlite3.Connection, detection_run_id: str) -> list[dict]:
    """Return per-video detection metadata for the run."""
    rows = conn.execute(
        """
        SELECT dk.shot_video_id,
               cv.file_path,
               cv.actual_fps,
               MIN(dk.video_frame) AS first_frame,
               MAX(dk.video_frame) AS last_frame,
               COUNT(DISTINCT dk.track_id) AS n_tracks
        FROM detection_keypoints dk
        JOIN capture_videos cv ON cv.id = dk.shot_video_id
        WHERE dk.detection_run_id = ?
        GROUP BY dk.shot_video_id
        """,
        (detection_run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _load_keypoints(
    conn: sqlite3.Connection,
    detection_run_id: str,
    shot_video_id: str,
) -> dict[int, dict[int, np.ndarray]]:
    """Return {video_frame: {track_id: (N, 2) xy array}}."""
    rows = conn.execute(
        """
        SELECT video_frame, track_id, keypoints
        FROM detection_keypoints
        WHERE detection_run_id = ? AND shot_video_id = ?
        ORDER BY video_frame, track_id
        """,
        (detection_run_id, shot_video_id),
    ).fetchall()
    result: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for row in rows:
        kp = np.frombuffer(bytes(row["keypoints"]), dtype="<f4").reshape(-1, 3)
        result[row["video_frame"]][row["track_id"]] = kp[:, :2].copy()
    return dict(result)


def _load_assignments(
    conn: sqlite3.Connection,
    detection_run_id: str,
    shot_video_id: str,
) -> list[tuple[int, str, int, int]]:
    """Return list of (track_id, person_name, first_frame, last_frame)."""
    rows = conn.execute(
        """
        SELECT track_id, person_name, first_frame, last_frame
        FROM detection_track_assignments
        WHERE detection_run_id = ? AND shot_video_id = ?
        ORDER BY first_frame
        """,
        (detection_run_id, shot_video_id),
    ).fetchall()
    return [(r["track_id"], r["person_name"], r["first_frame"], r["last_frame"]) for r in rows]


def _load_bboxes_at_frame(
    conn: sqlite3.Connection,
    detection_run_id: str,
    shot_video_id: str,
    video_frame: int,
    track_ids: list[int],
) -> dict[int, np.ndarray]:
    """Return {track_id: xyxy bbox} from person_detections for a specific frame.

    The schema stores YOLO-format (cx, cy, w, h) in the bbox_x/y/w/h columns,
    so we convert to xyxy here.
    """
    placeholders = ",".join("?" * len(track_ids))
    rows = conn.execute(
        f"""
        SELECT track_id, bbox_x, bbox_y, bbox_w, bbox_h
        FROM person_detections
        WHERE detection_run_id = ? AND shot_video_id = ?
          AND video_frame = ? AND track_id IN ({placeholders})
          AND region_type = 'full_body'
        """,
        (detection_run_id, shot_video_id, video_frame, *track_ids),
    ).fetchall()
    result = {}
    for row in rows:
        if row["bbox_x"] is None:
            continue
        # bbox_x/y are the bbox CENTRE (YOLO xywh format); convert to xyxy
        cx, cy = row["bbox_x"], row["bbox_y"]
        w, h = row["bbox_w"], row["bbox_h"]
        result[row["track_id"]] = np.array(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32
        )
    return result


def _person_at_frame(
    assignments: list[tuple[int, str, int, int]],
    track_id: int,
    frame: int,
) -> str | None:
    """Look up the person_name for a track_id at a given frame (temporal lookup)."""
    for tid, name, first, last in assignments:
        if tid == track_id and first <= frame <= last:
            return name
    return None


# ---------------------------------------------------------------------------
# Init mask building
# ---------------------------------------------------------------------------

def _build_rect_init_mask(
    frame_bgr: np.ndarray,
    bboxes_by_person: dict[str, np.ndarray],
) -> np.ndarray:
    """Build a labeled init mask from bounding-box rectangles (no SAM needed).

    Used as fallback when SAM fails.  Each person's bbox is filled with the
    person's label index.  Overlapping regions are overwritten by later persons.
    """
    h, w = frame_bgr.shape[:2]
    names = list(bboxes_by_person.keys())
    mask_out = np.zeros((h, w), dtype=np.uint8)
    for j, name in enumerate(names):
        x1, y1, x2, y2 = bboxes_by_person[name]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
        mask_out[y1:y2, x1:x2] = j + 1
    return mask_out


def _build_init_mask(
    frame_bgr: np.ndarray,
    bboxes_by_person: dict[str, np.ndarray],
) -> np.ndarray:
    """Run SAM2 on frame_bgr using bbox-centre point prompts to produce a labeled init mask.

    Uses centre-point prompts rather than bbox prompts because in ultralytics ≥ 8.4,
    passing multiple bboxes to SAM2 treats them as prompts for one object.  A flat
    list of N centre points returns N separate masks.

    Falls back to filled bounding-box rectangles if SAM fails or returns too few masks.

    Parameters
    ----------
    frame_bgr:
        Full-resolution BGR frame (BGR, as returned by cv2.VideoCapture.read()).
    bboxes_by_person:
        OrderedDict mapping person_name → xyxy bbox (in frame pixel coordinates).
        Order determines mask label (first → 1, second → 2, …).

    Returns
    -------
    ``(H, W)`` uint8 labeled mask.  0 = background; k = k-th person (1-indexed).
    """
    try:
        from ultralytics import SAM
    except ImportError:
        log.warning("ultralytics not available — using rectangle fallback for init mask")
        return _build_rect_init_mask(frame_bgr, bboxes_by_person)

    h, w = frame_bgr.shape[:2]
    names = list(bboxes_by_person.keys())
    bboxes = np.array([bboxes_by_person[n] for n in names], dtype=float)

    try:
        sam = SAM("sam2.1_b.pt")
        # Pass as float32 numpy array — ultralytics SAM returns N masks for N bboxes
        # when given a float32 array (float64 silently triggers a different code path).
        result = sam(frame_bgr, bboxes=bboxes.astype(np.float32), imgsz=512, verbose=False)
    except Exception as exc:
        log.warning("SAM failed (%s) — using rectangle fallback", exc)
        return _build_rect_init_mask(frame_bgr, bboxes_by_person)

    mask_out = np.zeros((h, w), dtype=np.uint8)
    n_sam = 0
    if result and result[0].masks is not None:
        masks_raw = result[0].masks.data.cpu().numpy()  # (N_masks, H, W) bool
        n_sam = len(masks_raw)
        for j in range(min(len(names), n_sam)):
            m = masks_raw[j].astype(bool)
            if m.shape != (h, w):
                m = cv2.resize(
                    m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            mask_out[m] = j + 1   # higher index wins in overlap zones

    n_found = int(mask_out.max())
    if n_found < len(names):
        log.warning(
            "SAM returned %d/%d masks — filling missing persons with rectangle fallback",
            n_sam, len(names),
        )
        # Use rectangles only for persons whose label is completely absent
        rect_mask = _build_rect_init_mask(frame_bgr, bboxes_by_person)
        for j in range(len(names)):
            label = j + 1
            if not np.any(mask_out == label):
                region = rect_mask == label
                mask_out[region & (mask_out == 0)] = label

    log.info("  Init mask: %d/%d persons labeled (SAM gave %d)", n_found, len(names), n_sam)
    return mask_out


# ---------------------------------------------------------------------------
# Core streaming processing
# ---------------------------------------------------------------------------

def _load_cutie_model(device: str = "cuda"):
    """Load and return the Cutie model (call once, reuse across cameras).

    Clears GlobalHydra state first if already initialized, so this can be
    called at the start of the program and the result shared.
    """
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
    model = get_default_model()
    return model


def _process_video_streaming(
    video_path: Path,
    init_frame: int,
    persons_ordered: list[str],
    init_mask_full: np.ndarray,
    keypoints_per_frame: dict[int, dict[int, np.ndarray]],
    assignments: list[tuple[int, str, int, int]],
    start_frame: int,
    end_frame: int,
    device: str,
    max_dim: int,
    erosion_px: int,
    debug_mask_every: int,
    cutie_model=None,
) -> tuple[dict[tuple[int, int], bytes], list[tuple[int, np.ndarray]]]:
    """Run Cutie bidirectionally, scoring keypoints on-the-fly.

    Parameters
    ----------
    video_path:
        Path to camera video file.
    init_frame:
        Frame index used to seed Cutie (absolute, 0-based).
    persons_ordered:
        Person names in order; label k in mask corresponds to persons_ordered[k-1].
    init_mask_full:
        ``(H, W)`` uint8 labeled mask at full video resolution.
    keypoints_per_frame:
        Pre-loaded keypoints: {frame: {track_id: (N, 2) xy array (full-res coords)}}.
    assignments:
        Track-to-person assignments: [(track_id, person_name, first_frame, last_frame)].
    start_frame / end_frame:
        Frame range to process (inclusive start, exclusive end).
    device:
        Torch device string.
    max_dim:
        Maximum dimension for Cutie processing (frames are downscaled if larger).
    erosion_px:
        Erosion kernel radius for boundary zone detection.
    debug_mask_every:
        Save a labeled mask every this many frames (0 = disabled).

    Returns
    -------
    quality_results:
        ``{(video_frame, track_id): quality_blob_bytes}``
    debug_masks:
        List of ``(video_frame, labeled_uint8_at_proc_res)`` for debug.
    """
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    cutie_dir = _find_cutie_dir()
    if str(cutie_dir) not in sys.path:
        sys.path.insert(0, str(cutie_dir))
    from cutie.inference.inference_core import InferenceCore

    n = len(persons_ordered)
    objects_list = list(range(1, n + 1))

    # Video source dimensions
    cap = cv2.VideoCapture(str(video_path))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Processing scale: downscale 4K → 1080p, leave ≤1080p untouched
    scale = min(1.0, max_dim / max(src_w, src_h))
    proc_w = int(src_w * scale + 0.5)
    proc_h = int(src_h * scale + 0.5)

    if scale < 1.0:
        log.info(
            "  %s: scaling %dx%d → %dx%d (%.2f×)",
            video_path.name, src_w, src_h, proc_w, proc_h, scale,
        )
        init_mask_proc = cv2.resize(
            init_mask_full, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST
        )
    else:
        init_mask_proc = init_mask_full

    # ── Helpers ──────────────────────────────────────────────────────────

    def to_tensor_resize(bgr: np.ndarray) -> torch.Tensor:
        if scale < 1.0:
            bgr = cv2.resize(bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return to_tensor(Image.fromarray(rgb)).to(device).float()

    def score_frame(fi: int, labeled: np.ndarray) -> dict[tuple[int, int], bytes]:
        """Score all keypoints in frame fi against the labeled mask."""
        results: dict[tuple[int, int], bytes] = {}
        kp_dict = keypoints_per_frame.get(fi, {})
        for track_id, kp_xy in kp_dict.items():
            person_name = _person_at_frame(assignments, track_id, fi)
            if person_name is None:
                scores = np.full(len(kp_xy), SCORE_UNAVAILABLE, dtype=np.float32)
            else:
                try:
                    pid_idx = persons_ordered.index(person_name)
                except ValueError:
                    scores = np.full(len(kp_xy), SCORE_UNAVAILABLE, dtype=np.float32)
                    results[(fi, track_id)] = encode_scores(scores)
                    continue
                # Person label in Cutie mask is 1-indexed
                person_mask = labeled == (pid_idx + 1)
                # Scale keypoints to processing resolution
                kp_proc = kp_xy * scale
                scores = _score_keypoints(person_mask, kp_proc, erosion_px)
            results[(fi, track_id)] = encode_scores(scores)
        return results

    # ── Use provided or load Cutie model ─────────────────────────────────
    if cutie_model is None:
        log.info("  Loading Cutie model…")
        cutie_model = _load_cutie_model(device)

    quality_results: dict[tuple[int, int], bytes] = {}
    debug_masks: list[tuple[int, np.ndarray]] = []

    init_mask_t = torch.from_numpy(init_mask_proc).to(device)

    # ── Forward pass: init_frame → end_frame ─────────────────────────────
    log.info(
        "  Forward pass: frames [%d, %d)  init=%d",
        init_frame, end_frame, init_frame,
    )
    fwd = InferenceCore(cutie_model, cfg=cutie_model.cfg)
    fwd.max_internal_size = 480

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
    seeded = False
    n_fwd = 0

    device_type = device.split(":")[0]  # "cuda:1" → "cuda"
    with torch.inference_mode(), torch.amp.autocast(device_type):
        for fi in range(init_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            img_t = to_tensor_resize(frame)

            if not seeded:
                out = fwd.step(img_t, init_mask_t, objects=objects_list)
                seeded = True
            else:
                out = fwd.step(img_t)

            labeled = fwd.output_prob_to_mask(out).cpu().numpy()
            quality_results.update(score_frame(fi, labeled))

            if debug_mask_every > 0 and fi % debug_mask_every == 0:
                debug_masks.append((fi, labeled.astype(np.uint8)))

            n_fwd += 1
            if n_fwd % 500 == 0:
                log.info("    fwd %d / %d (%.0f%%)", fi, end_frame,
                         100 * (fi - init_frame) / max(1, end_frame - init_frame))

    cap.release()
    log.info("  Forward pass done: %d frames", n_fwd)

    # ── Backward pass: (init_frame-1) → start_frame ──────────────────────
    n_bwd = init_frame - start_frame
    if n_bwd <= 0:
        log.info("  Backward pass: skipped (init_frame == start_frame)")
    else:
        log.info(
            "  Backward pass: reading %d frames [%d, %d)…",
            n_bwd, start_frame, init_frame,
        )
        # Read backward range into memory at processing resolution
        bwd_frames: list[tuple[int, np.ndarray]] = []
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for fi in range(start_frame, init_frame):
            ret, frame = cap.read()
            if not ret:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            bwd_frames.append((fi, frame))
        cap.release()

        # Re-read init frame for backward seeding
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
        ret, init_bgr = cap.read()
        cap.release()
        if not ret:
            log.error("  Cannot re-read init frame %d — skipping backward pass", init_frame)
        else:
            bwd = InferenceCore(cutie_model, cfg=cutie_model.cfg)
            bwd.max_internal_size = 480

            with torch.inference_mode(), torch.amp.autocast(device_type):
                # Seed backward processor with init frame
                img_t = to_tensor_resize(init_bgr)
                bwd.step(img_t, init_mask_t, objects=objects_list)

                for fi, frame_proc in reversed(bwd_frames):
                    img_t = to_tensor_resize(frame_proc)
                    out = bwd.step(img_t)
                    labeled = bwd.output_prob_to_mask(out).cpu().numpy()
                    quality_results.update(score_frame(fi, labeled))

                    if debug_mask_every > 0 and fi % debug_mask_every == 0:
                        debug_masks.append((fi, labeled.astype(np.uint8)))

            log.info("  Backward pass done: %d frames", len(bwd_frames))

    log.info(
        "  Video done: %d quality entries, %d debug masks",
        len(quality_results), len(debug_masks),
    )
    return quality_results, debug_masks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_init_frame_and_bboxes(
    conn: sqlite3.Connection,
    detection_run_id: str,
    shot_video_id: str,
    first_frame: int,
    last_frame: int,
    assignments: list[tuple[int, str, int, int]],
    init_offset: float,
) -> tuple[int, dict[str, np.ndarray]]:
    """Pick a good init frame and return bboxes for each person at that frame.

    Searches a ±300-frame window around the candidate (``first_frame +
    init_offset * range``) for the first frame where ALL persons have
    both a track assignment and a detection bbox.  Searches outward
    from the candidate so the chosen frame stays close to the requested
    position.

    Returns
    -------
    (init_frame, {person_name: xyxy_bbox})  — may have fewer persons than
    expected if no good frame is found.
    """
    all_persons = sorted({name for _, name, _, _ in assignments})
    n_want = len(all_persons)

    candidate = first_frame + int((last_frame - first_frame) * init_offset)
    candidate = max(first_frame, min(last_frame, candidate))

    def _try_frame(frame_idx: int) -> dict[str, np.ndarray] | None:
        """Return person→bbox dict if all persons visible, else None."""
        person_to_trackid: dict[str, int] = {}
        for tid, name, first, last in assignments:
            if first <= frame_idx <= last and name not in person_to_trackid:
                person_to_trackid[name] = tid
        if len(person_to_trackid) < n_want:
            return None
        bboxes = _load_bboxes_at_frame(
            conn, detection_run_id, shot_video_id, frame_idx,
            list(person_to_trackid.values()),
        )
        pb = {name: bboxes[tid]
              for name, tid in person_to_trackid.items()
              if tid in bboxes}
        return pb if len(pb) == n_want else None

    # Spiral outward from candidate: 0, +1, -1, +2, -2, …
    search_radius = 300
    for delta in range(search_radius + 1):
        for sign in ([0] if delta == 0 else [+1, -1]):
            fi = candidate + sign * delta
            if fi < first_frame or fi > last_frame:
                continue
            pb = _try_frame(fi)
            if pb is not None:
                log.info("  Init frame: %d (Δ%+d from candidate %d)", fi, fi - candidate, candidate)
                return fi, pb

    # Fallback: best available frame (most persons visible)
    log.warning(
        "  Could not find frame with all %d persons in ±%d window; "
        "trying best partial match",
        n_want, search_radius,
    )
    best_frame, best_bboxes = candidate, {}
    for delta in range(search_radius + 1):
        for sign in ([0] if delta == 0 else [+1, -1]):
            fi = candidate + sign * delta
            if fi < first_frame or fi > last_frame:
                continue
            person_to_trackid = {}
            for tid, name, frst, lst in assignments:
                if frst <= fi <= lst and name not in person_to_trackid:
                    person_to_trackid[name] = tid
            if not person_to_trackid:
                continue
            bboxes = _load_bboxes_at_frame(
                conn, detection_run_id, shot_video_id, fi,
                list(person_to_trackid.values()),
            )
            pb = {name: bboxes[tid]
                  for name, tid in person_to_trackid.items()
                  if tid in bboxes}
            if len(pb) > len(best_bboxes):
                best_frame, best_bboxes = fi, pb
                if len(pb) == n_want:
                    break
        if len(best_bboxes) == n_want:
            break

    log.warning(
        "  Using frame %d with %d/%d persons", best_frame, len(best_bboxes), n_want
    )
    return best_frame, best_bboxes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Cutie segmentation quality scores for a detection run."
    )
    parser.add_argument("--db", required=True, help="Path to session SQLite DB")
    parser.add_argument(
        "--detection-run-id", required=True, help="detection_runs.id to process"
    )
    parser.add_argument(
        "--init-offset", type=float, default=0.0,
        help="Fractional position within detection range for init frame (default: 0, i.e., first frame)",
    )
    parser.add_argument(
        "--max-dim", type=int, default=1920,
        help="Maximum processing dimension; 4K frames are scaled to 1080p (default: 1920)",
    )
    parser.add_argument(
        "--erosion-px", type=int, default=5,
        help="Erosion kernel radius for boundary zone scoring (default: 5)",
    )
    parser.add_argument(
        "--debug-masks", action="store_true",
        help="Save per-frame labeled masks to NPZ files for debugging",
    )
    parser.add_argument(
        "--debug-every", type=int, default=30,
        help="Save every Nth frame's mask when --debug-masks is set (default: 30)",
    )
    parser.add_argument(
        "--cameras", nargs="*", default=None,
        help="Process only these shot_video_ids (default: all cameras)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help='Torch device: "cuda" or "cpu" (default: cuda)',
    )
    parser.add_argument(
        "--notes", default=None, help="Free-text notes stored in seg_quality_runs"
    )
    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        log.error("DB not found: %s", db_path)
        sys.exit(1)

    conn = _open_db(str(db_path))

    # ── Verify detection run ─────────────────────────────────────────────
    run_row = conn.execute(
        "SELECT * FROM detection_runs WHERE id = ?",
        (args.detection_run_id,),
    ).fetchone()
    if run_row is None:
        log.error("No detection run found with id=%r", args.detection_run_id)
        sys.exit(1)
    log.info(
        "Detection run: %s  model=%s  status=%s",
        args.detection_run_id, run_row["pose_model"], run_row["status"],
    )

    # ── Create seg quality run record ───────────────────────────────────
    seg_run_id = str(uuid.uuid4())
    debug_dir: Path | None = None
    if args.debug_masks:
        debug_dir = db_path.parent / "seg_masks" / seg_run_id
        debug_dir.mkdir(parents=True, exist_ok=True)
        log.info("Debug masks will be saved to: %s", debug_dir)

    # Time-range-scoped on the capture now, not tied to this one detection
    # run (see docs/roadmap/features/segmentation-reuse/
    # segmentation-reuse-design.md) -- backfilled from the detection run
    # this tool was pointed at, same as the v41->v42 migration does for
    # pre-existing rows.
    conn.execute(
        "INSERT INTO seg_quality_runs "
        "(id, shot_id, trial_id, time_start_s, time_end_s, created_at, "
        " quality_source, erosion_px, mask_dir, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, 'cutie', ?, ?, ?)",
        (
            seg_run_id,
            run_row["shot_id"],
            run_row["trial_id"],
            run_row["time_start_s"],
            run_row["time_end_s"],
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            args.erosion_px,
            str(debug_dir) if debug_dir else None,
            args.notes,
        ),
    )
    conn.commit()
    log.info("Created seg_quality_run: %s", seg_run_id)

    # ── Load video list ──────────────────────────────────────────────────
    videos = _load_video_info(conn, args.detection_run_id)
    if args.cameras:
        videos = [v for v in videos if v["shot_video_id"] in args.cameras]
    log.info("Processing %d camera video(s)", len(videos))

    # ── Pre-load Cutie model once (avoid Hydra re-init across cameras) ────
    log.info("Loading Cutie model…")
    cutie_model = _load_cutie_model(args.device)
    log.info("Cutie model loaded.")

    # ── Process each camera ──────────────────────────────────────────────
    for vid in videos:
        vid_id = vid["shot_video_id"]
        video_path = Path(vid["file_path"])
        first_frame = vid["first_frame"]
        last_frame = vid["last_frame"]

        log.info(
            "\n=== Camera: %s  frames [%d, %d] ===",
            video_path.name, first_frame, last_frame,
        )

        if not video_path.exists():
            log.error("  Video not found: %s — skipping", video_path)
            continue

        # Load track assignments
        assignments = _load_assignments(conn, args.detection_run_id, vid_id)
        if not assignments:
            log.warning("  No track assignments for %s — skipping", video_path.name)
            continue

        all_persons = sorted({name for _, name, _, _ in assignments})
        log.info("  Persons: %s", ", ".join(all_persons))

        # Load all keypoints
        kp_per_frame = _load_keypoints(conn, args.detection_run_id, vid_id)
        log.info("  Loaded keypoints for %d frames", len(kp_per_frame))

        # Find init frame and bboxes
        init_frame, person_bboxes = _find_init_frame_and_bboxes(
            conn, args.detection_run_id, vid_id,
            first_frame, last_frame, assignments, args.init_offset,
        )
        if not person_bboxes:
            log.error("  No person bboxes found for any init frame — skipping")
            continue

        log.info("  Init frame: %d  persons: %s", init_frame, list(person_bboxes.keys()))

        # ── Read init frame from video ──────────────────────────────────
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
        ret, init_bgr = cap.read()
        cap.release()
        if not ret:
            log.error("  Cannot read init frame %d from %s — skipping", init_frame, video_path)
            continue

        # ── Build init mask via SAM ─────────────────────────────────────
        log.info("  Building init mask via SAM2 (with rectangle fallback)…")
        try:
            init_mask_full = _build_init_mask(init_bgr, person_bboxes)
        except Exception as exc:
            log.error("  Init mask build failed: %s — skipping", exc)
            continue

        if int(init_mask_full.max()) == 0:
            log.error("  Init mask is empty (all background) — skipping")
            continue

        persons_ordered = list(person_bboxes.keys())

        # ── Run streaming Cutie ─────────────────────────────────────────
        debug_every = args.debug_every if args.debug_masks else 0
        try:
            quality_results, debug_masks = _process_video_streaming(
                video_path=video_path,
                init_frame=init_frame,
                persons_ordered=persons_ordered,
                init_mask_full=init_mask_full,
                keypoints_per_frame=kp_per_frame,
                assignments=assignments,
                start_frame=first_frame,
                end_frame=last_frame + 1,
                device=args.device,
                max_dim=args.max_dim,
                erosion_px=args.erosion_px,
                debug_mask_every=debug_every,
                cutie_model=cutie_model,
            )
        except Exception as exc:
            log.error("  Processing failed: %s", exc, exc_info=True)
            continue

        # ── Write quality scores to DB ──────────────────────────────────
        log.info(
            "  Writing %d quality score entries to DB…", len(quality_results)
        )
        batch = [
            (seg_run_id, vid_id, frame, track_id, blob)
            for (frame, track_id), blob in quality_results.items()
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO keypoint_obs_quality "
            "(seg_run_id, shot_video_id, video_frame, track_id, quality_blob) "
            "VALUES (?, ?, ?, ?, ?)",
            batch,
        )
        conn.commit()
        log.info("  DB write done.")

        # ── Save debug masks ────────────────────────────────────────────
        if debug_masks and debug_dir is not None:
            npz_path = debug_dir / f"masks_{vid_id[:8]}.npz"
            frames_arr = np.array([f for f, _ in debug_masks], dtype=np.int32)
            masks_arr = np.stack([m for _, m in debug_masks], axis=0)  # (N, H, W)
            np.savez_compressed(
                str(npz_path),
                frames=frames_arr,
                masks=masks_arr,
                persons_ordered=np.array(persons_ordered),
                init_frame=np.array([init_frame]),
                scale=np.array([min(1.0, args.max_dim / max(init_bgr.shape[:2]))]),
            )
            log.info("  Debug masks saved: %s  (%d frames)", npz_path, len(debug_masks))

    # ── Summary ─────────────────────────────────────────────────────────
    total = conn.execute(
        "SELECT COUNT(*) FROM keypoint_obs_quality WHERE seg_run_id = ?",
        (seg_run_id,),
    ).fetchone()[0]
    log.info(
        "\nDone.  seg_run_id=%s  total quality rows: %d",
        seg_run_id, total,
    )
    conn.close()


if __name__ == "__main__":
    main()
