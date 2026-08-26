#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""segmentation_mask_steering_experiment.py — does masking out the "other
person" steer pose estimation onto the right body?

Prototype for the second idea from the segmentation-improvement study
(2026-08-25 conversation, no design doc yet -- this script *is* the design
doc's first data point): when two people are close enough that their
detection bboxes nearly coincide, a top-down pose model's crop shows both
people at once, and it's the model's own judgement -- not anything this
pipeline controls -- which one it actually estimates joints for. This
tries the direct fix: use a person-labeled segmentation mask to blur/
replace every pixel *not* belonging to the target person in each person's
own crop, before running pose estimation on it, and see whether that
measurably improves which body the keypoints land on.

Two mask sources, chosen per clip:

- ``--seg-clip capture_id:camera_label:seg_quality_run_id:start_s:end_s``
  reads already-curated seg_masks rows (from the interactive Cutie init
  panel). Preferred whenever they exist -- first version of this script
  re-derived a mask from scratch and wasted hand-corrected work that was
  already sitting in the DB.
- ``--fresh-clip capture_id:camera_label:source_run_id:start_s:end_s`` runs
  Cutie itself (forward+backward from whichever frame in range has both
  people's bboxes from *source_run_id*'s track assignments), for time
  windows nobody has curated yet. This is also useful on its own terms:
  automatic Cutie/SAM2 mask quality (or lack of it) during a grab is
  itself one of the things being investigated.

Also runs the existing hand-specific refinement pass
(posetrak.detection.hand_refinement.detect_hand_in_crop) on each
treatment's own masked crop, anchored on that treatment's own body-model
wrist/elbow estimate -- lets a real qualitative comparison of "does a
dedicated hand model do better than the wholebody model in the same
masked crop" ride along for free, addressing the hypothesis that the
in-mask-fraction metric itself is unreliable wherever the mask doesn't
fully cover fingers (a few outside-mask keypoints on a 21-point hand
model swing the metric much harder than on a 133-point wholebody one).

**Read-only against the session DB** -- nothing is written back. Output is
local files only: one debug .mp4 per (clip, model) showing the overview
once plus each treatment's per-person crop (wholebody skeleton +
hand-refinement overlay) side by side, and a summary.csv.

Usage::

    python segmentation_mask_steering_experiment.py \\
        --db E:\\mocap\\vanhaa\\ukemi-tommi-20260509.db \\
        --out-dir D:\\mocap\\segmentation-study\\trial2 \\
        --seg-clip b862ca88-...:pixel7:e862c9af-...:42:46 \\
        --fresh-clip 300f5172-...:pixel9:2c272a4e-...:44:47 \\
        [--frame-stride 4] [--treatments none hard blur feather]
        [--models rtmpose vitpose] [--max-padding 20] [--erosion-px 5]
        [--device cuda]
"""

from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PYTHON_DIR = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _PYTHON_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_cutie_pose as rcp  # tight/padded-bbox + panel-drawing + fresh-Cutie helpers
from pipeline.pose.segmentation import _score_keypoints, SCORE_INSIDE, SCORE_UNAVAILABLE
from posetrak.detection.hand_refinement import (
    _ELBOW_IDX, _HAND_BASE_IDX, _HAND_N_KP, _WRIST_IDX, detect_hand_in_crop,
)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# Hand-region keypoint indices into the 133-point COCO-wholebody layout,
# reusing hand_refinement.py's canonical mapping rather than re-deriving it.
_HAND_IDX = sorted(
    idx for side in ("left", "right") for idx in range(_HAND_BASE_IDX[side], _HAND_BASE_IDX[side] + _HAND_N_KP)
)
_WRIST_IDX_LIST = sorted(_WRIST_IDX.values())
_SIDES = ("left", "right")

_TREATMENTS = ("none", "hard", "blur", "feather", "feather2")
_MODELS = ("rtmpose", "vitpose")
_FILL_GRAY = 127
_BLUR_KSIZE = 45          # strong blur -- shape/motion survives, detail doesn't
_FEATHER_PX = 15          # half-width of the soft transition band at the mask edge

# feather2 (2026-08-25, Harri's request after trial1-3's forearm-hallucination
# finding): narrower transition band than feather (10px, not 15) plus a
# contrast reduction on top of the blur, on the theory that a flatter,
# lower-contrast "other person" region gives the pose model even less to
# latch onto/hallucinate from than a merely-blurred one, while still
# avoiding feather's hard-edge risk.
_FEATHER2_PX = 10
_CONTRAST_FACTOR = 0.4   # 0 = flat gray, 1 = no contrast reduction
_HAND_COLOR = (0, 215, 255)  # amber dots -- distinct from the wholebody skeleton colors

_CSV_FIELDNAMES = [
    "clip", "model", "treatment", "person", "frame",
    "in_mask_frac_all", "in_mask_frac_hand", "mean_conf_hand", "mean_conf_wrist",
    "n_hands_refined", "hand_refine_in_mask_frac", "hand_refine_conf",
]


# ---------------------------------------------------------------------------
# Treatments -- what each one does to the pixels NOT belonging to the target
# person (i.e. background AND any other tracked person), inside the crop.
# ---------------------------------------------------------------------------

def _apply_treatment(
    frame: np.ndarray, labeled: np.ndarray, target_label: int, treatment: str
) -> np.ndarray:
    """Return a copy of *frame* with every pixel outside *target_label*'s
    mask altered per *treatment*. "none" is the unmodified baseline
    (today's behaviour: the raw crop, whoever else is in it)."""
    if treatment == "none":
        return frame

    other = labeled != target_label  # True = background OR another person
    if treatment == "hard":
        out = frame.copy()
        out[other] = _FILL_GRAY
        return out

    blurred = cv2.GaussianBlur(frame, (_BLUR_KSIZE, _BLUR_KSIZE), 0)
    if treatment == "blur":
        out = frame.copy()
        out[other] = blurred[other]
        return out

    if treatment == "feather":
        target = (labeled == target_label).astype(np.uint8)
        dist = cv2.distanceTransform(1 - target, cv2.DIST_L2, 3)
        alpha = np.clip(dist / _FEATHER_PX, 0.0, 1.0)[..., None]
        return (frame.astype(np.float32) * (1 - alpha) + blurred.astype(np.float32) * alpha).astype(np.uint8)

    if treatment == "feather2":
        target = (labeled == target_label).astype(np.uint8)
        dist = cv2.distanceTransform(1 - target, cv2.DIST_L2, 3)
        alpha = np.clip(dist / _FEATHER2_PX, 0.0, 1.0)[..., None]
        low_contrast = blurred.astype(np.float32) * _CONTRAST_FACTOR + _FILL_GRAY * (1 - _CONTRAST_FACTOR)
        return (frame.astype(np.float32) * (1 - alpha) + low_contrast * alpha).astype(np.uint8)

    raise ValueError(f"Unknown treatment: {treatment!r}")


# ---------------------------------------------------------------------------
# Pose models
# ---------------------------------------------------------------------------

def _load_pose_models(models: list[str], device: str) -> dict[str, object]:
    from posetrak.detection.backends_rtmpose import _KNOWN_MODELS
    from rtmlib.tools.pose_estimation import RTMPose as _RTMPose
    from rtmlib.tools.pose_estimation.vitpose import ViTPose as _ViTPose

    name_by_key = {"rtmpose": "rtmpose-l-133kp", "vitpose": "vitpose-l-133kp"}
    cls_by_key = {"rtmpose": _RTMPose, "vitpose": _ViTPose}

    loaded = {}
    for key in models:
        model_name = name_by_key[key]
        url, input_size_hw, _backend_cls, _conf_scale = _KNOWN_MODELS[model_name]
        log.info("Loading %s (%s)...", key, model_name)
        loaded[key] = cls_by_key[key](
            url,
            model_input_size=(input_size_hw[0], input_size_hw[1]),
            to_openpose=False,
            backend="onnxruntime",
            device=device,
        )
    return loaded


def _load_hand_model(device: str):
    from rtmlib import Hand as _Hand
    log.info("Loading hand-refinement model...")
    return _Hand(to_openpose=False, backend="onnxruntime", device=device)


def _transcode_to_h264(raw_path: Path, final_path: Path) -> None:
    """Re-encode *raw_path* (mp4v, VLC-only -- see _raw_path()) to H.264 at
    *final_path* via the system ffmpeg binary, then delete the raw file.
    Falls back to just renaming raw_path if ffmpeg isn't on PATH, so a
    machine without it still gets a (VLC-only) video rather than nothing.
    """
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        log.warning("  ffmpeg not found on PATH -- leaving %s as mp4v (VLC-only)", final_path.name)
        raw_path.replace(final_path)
        return
    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(raw_path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "20", "-loglevel", "error", str(final_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.warning("  ffmpeg transcode failed (%s) -- leaving %s as mp4v", result.stderr.strip()[:200], final_path.name)
        raw_path.replace(final_path)
        return
    raw_path.unlink()


# ---------------------------------------------------------------------------
# seg_masks access (curated path)
# ---------------------------------------------------------------------------

def _decode_mask_png(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded


def _seg_run_persons(conn: sqlite3.Connection, seg_run_id: str) -> list[str] | None:
    import json
    row = conn.execute("SELECT persons_json FROM seg_quality_runs WHERE id=?", (seg_run_id,)).fetchone()
    if row is None or not row["persons_json"]:
        return None
    return json.loads(row["persons_json"])


def _load_curated_masks(
    conn: sqlite3.Connection, seg_run_id: str, svid: str, f0: int, f1: int
) -> dict[int, np.ndarray]:
    rows = conn.execute(
        "SELECT frame_idx, mask_blob FROM seg_masks "
        "WHERE seg_quality_run_id=? AND shot_video_id=? AND frame_idx BETWEEN ? AND ? "
        "ORDER BY frame_idx",
        (seg_run_id, svid, f0, f1),
    ).fetchall()
    return {r["frame_idx"]: _decode_mask_png(bytes(r["mask_blob"])) for r in rows}


# ---------------------------------------------------------------------------
# Fresh-Cutie path -- forward+backward from whichever frame in [f0, f1] has
# every person's bbox in source_run_id's track assignments. Mirrors
# run_cutie_pose.py's own forward/backward loop, scoped to a short clip
# instead of a whole detection run's range, and always skips SAM2 (see
# run_cutie_pose.py's _build_init_mask -- always fails here, Hydra config
# conflict between Cutie's and SAM2's own model loading in one process).
# ---------------------------------------------------------------------------

def _generate_fresh_masks(
    video_path: Path, source_run_id: str, conn: sqlite3.Connection, svid: str,
    f0: int, f1: int, cutie_model, device: str,
) -> tuple[dict[int, np.ndarray], list[str]]:
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    cutie_dir = rcp._find_cutie_dir()
    if str(cutie_dir) not in sys.path:
        sys.path.insert(0, str(cutie_dir))
    from cutie.inference.inference_core import InferenceCore

    assignments = rcp._load_assignments(conn, source_run_id, svid)
    init_frame, person_bboxes = rcp._find_init_frame(conn, source_run_id, svid, f0, f1, assignments, init_offset=0.0)
    if not person_bboxes:
        return {}, []
    persons_ordered = list(person_bboxes)
    n = len(persons_ordered)
    objects_list = list(range(1, n + 1))
    device_type = device.split(":")[0]

    def _tensor(bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return to_tensor(Image.fromarray(rgb)).to(device).float()

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
    ret, init_bgr = cap.read()
    cap.release()
    if not ret:
        return {}, []
    init_mask = rcp._build_rect_init_mask(init_bgr, person_bboxes)
    init_mask_t = torch.from_numpy(init_mask).to(device)

    masks: dict[int, np.ndarray] = {}

    log.info("    fresh Cutie: forward [%d, %d] init=%d", init_frame, f1, init_frame)
    fwd = InferenceCore(cutie_model, cfg=cutie_model.cfg)
    fwd.max_internal_size = 480
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
    seeded = False
    with torch.inference_mode(), torch.amp.autocast(device_type):
        for fi in range(init_frame, f1 + 1):
            ret, bgr = cap.read()
            if not ret:
                break
            out = fwd.step(_tensor(bgr), init_mask_t, objects=objects_list) if not seeded else fwd.step(_tensor(bgr))
            seeded = True
            masks[fi] = fwd.output_prob_to_mask(out).cpu().numpy()
    cap.release()

    if init_frame > f0:
        log.info("    fresh Cutie: backward [%d, %d)", f0, init_frame)
        bwd_frames = []
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        for fi in range(f0, init_frame):
            ret, bgr = cap.read()
            if not ret:
                break
            bwd_frames.append((fi, bgr))
        cap.release()
        bwd = InferenceCore(cutie_model, cfg=cutie_model.cfg)
        bwd.max_internal_size = 480
        with torch.inference_mode(), torch.amp.autocast(device_type):
            bwd.step(_tensor(init_bgr), init_mask_t, objects=objects_list)
            for fi, bgr in reversed(bwd_frames):
                out = bwd.step(_tensor(bgr))
                masks[fi] = bwd.output_prob_to_mask(out).cpu().numpy()

    return masks, persons_ordered


# ---------------------------------------------------------------------------
# Hand refinement -- reuses detect_hand_in_crop() (pure function, no DB
# dependency) with wrist/elbow anchored on *this treatment's own* body-model
# estimate, run against *this treatment's own* masked crop.
# ---------------------------------------------------------------------------

def _refine_hands(hand_model, treated_frame: np.ndarray, kp_with_conf: np.ndarray) -> dict[str, "object"]:
    out = {}
    for side in _SIDES:
        wrist = tuple(kp_with_conf[_WRIST_IDX[side], :2])
        elbow = tuple(kp_with_conf[_ELBOW_IDX[side], :2])
        cand = detect_hand_in_crop(hand_model, treated_frame, wrist, elbow)
        if cand is not None:
            out[side] = cand
    return out


# ---------------------------------------------------------------------------
# Debug frame
# ---------------------------------------------------------------------------

def _draw_hand_overlay(panel: np.ndarray, crop_origin: tuple[int, int], crop_h: int, hand_candidates: dict) -> None:
    """Draw hand-refinement keypoints (amber dots) onto *panel* in place.

    *panel* was built by run_cutie_pose.py's _person_panel(), which crops
    at *crop_origin* = (x1, y1) then uniformly scales by _PANEL_H / crop_h
    -- replicate that same mapping here rather than re-deriving a new one.
    """
    if not hand_candidates or crop_h <= 0:
        return
    x1, y1 = crop_origin
    scale = rcp._PANEL_H / crop_h
    for cand in hand_candidates.values():
        for x, y in cand.keypoints:
            px, py = int((x - x1) * scale), int((y - y1) * scale)
            if 0 <= px < panel.shape[1] and 0 <= py < panel.shape[0]:
                cv2.circle(panel, (px, py), 2, _HAND_COLOR, -1)


def _make_experiment_frame(
    proc_frame: np.ndarray,
    labeled: np.ndarray,
    persons: list[str],
    padded: list[tuple | None],
    treatments: list[str],
    kp_by_treatment: dict[str, list[np.ndarray | None]],
    hands_by_treatment: dict[str, list[dict]],
) -> np.ndarray:
    _HEADER_H = 24
    overview = rcp._overview_panel(proc_frame, labeled, persons, padded)
    overview_header = np.zeros((_HEADER_H, overview.shape[1], 3), dtype=np.uint8)
    cv2.putText(overview_header, "overview", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    panels = [np.vstack([overview_header, overview])]

    for treatment in treatments:
        treated = [_apply_treatment(proc_frame, labeled, i + 1, treatment) for i in range(len(persons))]
        header = np.zeros((_HEADER_H, rcp._PERSON_PANEL_W * len(persons), 3), dtype=np.uint8)
        cv2.putText(header, treatment, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        person_panels = []
        for i, name in enumerate(persons):
            panel = rcp._person_panel(
                treated[i], labeled, i + 1, padded[i],
                kp_by_treatment[treatment][i], name, rcp._COLORS[i % len(rcp._COLORS)],
            )
            bbox = padded[i]
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                _draw_hand_overlay(panel, (x1, y1), y2 - y1, hands_by_treatment[treatment][i])
            person_panels.append(panel)
        panels.append(np.vstack([header, np.hstack(person_panels)]))
    return np.hstack(panels)


# ---------------------------------------------------------------------------
# Core: process one resolved clip (masks dict already built, persons known).
# ---------------------------------------------------------------------------

def _process_clip(
    video_path: Path,
    clip_tag: str,
    masks: dict[int, np.ndarray],
    persons_ordered: list[str],
    pose_models: dict[str, object],
    hand_model,
    treatments: list[str],
    frame_stride: int,
    max_pad: int,
    erosion_px: int,
    out_dir: Path,
    csv_writer: csv.DictWriter,
    csv_file,
) -> None:
    all_frame_indices = sorted(masks)
    if not all_frame_indices:
        log.warning("  %s: no mask frames -- skipping", clip_tag)
        return
    frame_indices = all_frame_indices[::frame_stride]
    log.info("  %s: %d mask frames (stride %d) -> %d evaluated", clip_tag, len(all_frame_indices), frame_stride, len(frame_indices))

    n_persons = len(persons_ordered)
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    debug_writers: dict[str, cv2.VideoWriter] = {}

    def _debug_path(model_key: str) -> Path:
        return out_dir / f"{clip_tag}__{model_key}.mp4"

    def _raw_path(model_key: str) -> Path:
        # cv2.VideoWriter only has a working mp4v (MPEG-4 Part 2) encoder in
        # this environment -- avc1/h264/H264/X264 all "open" but the
        # underlying OpenH264 DLL fails to load, so they silently produce
        # nothing. mp4v plays fine in VLC but not Windows Media Player
        # (confirmed 2026-08-26); _transcode_to_h264 below fixes that up
        # afterward via the system ffmpeg binary instead.
        return out_dir / f"{clip_tag}__{model_key}.raw.mp4"

    for n_done, fi in enumerate(frame_indices):
        labeled = masks[fi]
        mh, mw = labeled.shape
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, bgr = cap.read()
        if not ret:
            continue
        proc_frame = cv2.resize(bgr, (mw, mh), interpolation=cv2.INTER_AREA) if bgr.shape[:2] != (mh, mw) else bgr

        padded_bboxes: list[tuple | None] = [None] * n_persons
        for i in range(n_persons):
            tight = rcp._mask_to_tight_bbox(labeled, i + 1)
            if tight is not None:
                padded_bboxes[i] = rcp._smart_pad_bbox(labeled, i + 1, tight, max_pad)
        if not any(padded_bboxes):
            continue

        for model_key, model in pose_models.items():
            kp_by_treatment: dict[str, list[np.ndarray | None]] = {}
            hands_by_treatment: dict[str, list[dict]] = {}
            for treatment in treatments:
                per_person_kp: list[np.ndarray | None] = [None] * n_persons
                per_person_hands: list[dict] = [{} for _ in range(n_persons)]
                for i in range(n_persons):
                    bbox = padded_bboxes[i]
                    if bbox is None:
                        continue
                    treated = _apply_treatment(proc_frame, labeled, i + 1, treatment)
                    bbox_arr = np.array([bbox], dtype=np.float32)
                    kp, sc = model(treated, bbox_arr)
                    kp_with_conf = np.concatenate([kp[0], sc[0][:, None]], axis=1).astype(np.float32)
                    per_person_kp[i] = kp_with_conf

                    hands = _refine_hands(hand_model, treated, kp_with_conf)
                    per_person_hands[i] = hands

                    mask_i = labeled == (i + 1)
                    quality = _score_keypoints(mask_i, kp_with_conf[:, :2], erosion_px)
                    valid = quality != SCORE_UNAVAILABLE
                    hand_quality = quality[_HAND_IDX]
                    hand_valid = hand_quality != SCORE_UNAVAILABLE

                    hand_refine_scores = []
                    hand_refine_confs = []
                    for cand in hands.values():
                        q = _score_keypoints(mask_i, cand.keypoints, erosion_px)
                        qv = q != SCORE_UNAVAILABLE
                        if qv.any():
                            hand_refine_scores.append(float(np.mean(q[qv] == SCORE_INSIDE)))
                        hand_refine_confs.append(float(np.mean(cand.scores)))

                    csv_writer.writerow({
                        "clip": clip_tag, "model": model_key, "treatment": treatment,
                        "person": persons_ordered[i], "frame": fi,
                        "in_mask_frac_all": float(np.mean(quality[valid] == SCORE_INSIDE)) if valid.any() else float("nan"),
                        "in_mask_frac_hand": float(np.mean(hand_quality[hand_valid] == SCORE_INSIDE)) if hand_valid.any() else float("nan"),
                        "mean_conf_hand": float(np.mean(kp_with_conf[_HAND_IDX, 2])),
                        "mean_conf_wrist": float(np.mean(kp_with_conf[_WRIST_IDX_LIST, 2])),
                        "n_hands_refined": len(hands),
                        "hand_refine_in_mask_frac": float(np.mean(hand_refine_scores)) if hand_refine_scores else float("nan"),
                        "hand_refine_conf": float(np.mean(hand_refine_confs)) if hand_refine_confs else float("nan"),
                    })
                kp_by_treatment[treatment] = per_person_kp
                hands_by_treatment[treatment] = per_person_hands

            dbg = _make_experiment_frame(proc_frame, labeled, persons_ordered, padded_bboxes, treatments, kp_by_treatment, hands_by_treatment)
            if model_key not in debug_writers:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_fps = max(1.0, video_fps / max(1, frame_stride))
                debug_writers[model_key] = cv2.VideoWriter(str(_raw_path(model_key)), fourcc, out_fps, (dbg.shape[1], dbg.shape[0]))
            debug_writers[model_key].write(dbg)

        if n_done % 25 == 0:
            log.info("    %d / %d", n_done, len(frame_indices))
            csv_file.flush()

    cap.release()
    for w in debug_writers.values():
        w.release()
    for model_key in debug_writers:
        _transcode_to_h264(_raw_path(model_key), _debug_path(model_key))
    log.info("  Wrote %d debug video(s) for %s", len(debug_writers), clip_tag)


# ---------------------------------------------------------------------------
# Clip resolution
# ---------------------------------------------------------------------------

def _resolve_svid(conn: sqlite3.Connection, capture_id: str, camera_label: str) -> str:
    row = conn.execute(
        "SELECT sv.id FROM capture_videos sv LEFT JOIN camera_instances ci ON ci.id = sv.camera_instance_id "
        "WHERE sv.shot_id=? AND ci.label=?",
        (capture_id, camera_label),
    ).fetchone()
    if row is None:
        raise ValueError(f"Camera {camera_label!r} not found for capture {capture_id}")
    return row["id"]


def _frame_bounds(conn: sqlite3.Connection, capture_id: str, svid: str, t0: float, t1: float) -> tuple[int, int]:
    from app.setup.db_context import SyncPoint, SyncTable

    sync_row = conn.execute("SELECT id FROM sync_configs WHERE shot_id=?", (capture_id,)).fetchone()
    if sync_row is None:
        raise ValueError(f"No sync_config for capture {capture_id}")
    rows = conn.execute(
        "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, sv.actual_fps "
        "FROM sync_points sp JOIN capture_videos sv ON sv.id = sp.shot_video_id "
        "WHERE sp.sync_config_id=?",
        (sync_row["id"],),
    ).fetchall()
    points = [
        SyncPoint(camera_instance_id=r["shot_video_id"], shot_video_id=r["shot_video_id"],
                   video_frame=int(r["video_frame"]), timestamp_s=float(r["timestamp_s"]))
        for r in rows
    ]
    fps_by_video = {r["shot_video_id"]: float(r["actual_fps"] or 30.0) for r in rows}
    table = SyncTable(points, fps_by_video)
    f0, f1 = table.lookup(t0, svid), table.lookup(t1, svid)
    if f0 is None or f1 is None:
        raise ValueError(f"No sync data for {svid} at [{t0},{t1}]s")
    return f0, f1


def _video_path(conn: sqlite3.Connection, svid: str) -> Path:
    row = conn.execute("SELECT file_path FROM capture_videos WHERE id=?", (svid,)).fetchone()
    path = Path(row["file_path"])
    if not path.exists() and len(path.drive) == 2:
        # Recorded path's drive has moved (confirmed 2026-08-25: capture
        # b862ca88's videos, at the same relative path, are now on E: not
        # D:) -- this script is read-only exploratory tooling, not the
        # place to rewrite capture_videos.file_path in the real DB, so
        # just try every other drive letter rather than fix the row.
        for letter in "CDEFGHIJ":
            alt = Path(f"{letter}:{str(path)[2:]}")
            if alt.exists():
                log.info("  %s not found; using %s instead", path, alt)
                return alt
    return path


def _parse_clip_arg(spec: str) -> tuple[str, str, str, float, float]:
    capture_id, camera_label, run_id, t0, t1 = spec.split(":")
    return capture_id, camera_label, run_id, float(t0), float(t1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seg-clip", action="append", default=[],
                        metavar="capture_id:camera_label:seg_quality_run_id:start_s:end_s",
                        help="Use curated seg_masks for this clip. Repeatable.")
    parser.add_argument("--fresh-clip", action="append", default=[],
                        metavar="capture_id:camera_label:source_detection_run_id:start_s:end_s",
                        help="Run Cutie fresh (rectangle init, no SAM2) for this clip, seeded from "
                             "source_detection_run_id's track assignments. Repeatable.")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--treatments", nargs="*", default=list(_TREATMENTS), choices=_TREATMENTS)
    parser.add_argument("--models", nargs="*", default=list(_MODELS), choices=_MODELS)
    parser.add_argument("--max-padding", type=int, default=20)
    parser.add_argument("--erosion-px", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.seg_clip and not args.fresh_clip:
        log.error("Need at least one --seg-clip or --fresh-clip")
        sys.exit(1)

    db_path = Path(args.db).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    pose_models = _load_pose_models(args.models, args.device)
    hand_model = _load_hand_model(args.device)

    cutie_model = None
    if args.fresh_clip:
        log.info("Loading Cutie (needed for --fresh-clip)...")
        cutie_model = rcp._load_cutie(args.device)

    # Written incrementally (flushed every 25 evaluated frames, per clip) --
    # NOT buffered in memory and written once at the end. A batch killed
    # partway through (confirmed happening, 2026-08-25: a ~45-minute
    # 4-clip run got killed on its last clip) must not lose every metric
    # row for the clips that *did* finish; only the debug videos survived
    # that first time around, since cv2.VideoWriter itself writes as it
    # goes.
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDNAMES)
        csv_writer.writeheader()
        csv_file.flush()

        for spec in args.seg_clip:
            capture_id, camera_label, seg_run_id, t0, t1 = _parse_clip_arg(spec)
            clip_tag = f"{camera_label}_{t0:g}-{t1:g}s"
            log.info("=== [seg] %s (capture %s) ===", clip_tag, capture_id[:8])
            try:
                svid = _resolve_svid(conn, capture_id, camera_label)
                f0, f1 = _frame_bounds(conn, capture_id, svid, t0, t1)
                masks = _load_curated_masks(conn, seg_run_id, svid, f0, f1)
                persons_ordered = _seg_run_persons(conn, seg_run_id)
                if not persons_ordered:
                    n_labels = max((int(m.max()) for m in masks.values()), default=0)
                    persons_ordered = [f"P{i + 1}" for i in range(n_labels)]
                    log.warning("  no persons_json on seg_quality_run -- using generic labels %s", persons_ordered)
                _process_clip(
                    _video_path(conn, svid), clip_tag, masks, persons_ordered, pose_models, hand_model,
                    args.treatments, args.frame_stride, args.max_padding, args.erosion_px, out_dir,
                    csv_writer, csv_file,
                )
                csv_file.flush()
            except Exception:
                log.error("  clip failed", exc_info=True)

        for spec in args.fresh_clip:
            capture_id, camera_label, source_run_id, t0, t1 = _parse_clip_arg(spec)
            clip_tag = f"{camera_label}_{t0:g}-{t1:g}s"
            log.info("=== [fresh] %s (capture %s) ===", clip_tag, capture_id[:8])
            try:
                svid = _resolve_svid(conn, capture_id, camera_label)
                f0, f1 = _frame_bounds(conn, capture_id, svid, t0, t1)
                vpath = _video_path(conn, svid)
                masks, persons_ordered = _generate_fresh_masks(vpath, source_run_id, conn, svid, f0, f1, cutie_model, args.device)
                _process_clip(
                    vpath, clip_tag, masks, persons_ordered, pose_models, hand_model,
                    args.treatments, args.frame_stride, args.max_padding, args.erosion_px, out_dir,
                    csv_writer, csv_file,
                )
                csv_file.flush()
            except Exception:
                log.error("  clip failed", exc_info=True)

    conn.close()
    log.info("Done. summary.csv -> %s", csv_path)


if __name__ == "__main__":
    main()
