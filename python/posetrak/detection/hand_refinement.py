# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""hand_refinement.py — Idea 2: hand-specific detection pass.

Refines wrist/finger keypoints for an existing full-body detection run by
cropping around each tracked wrist and running a dedicated hand model
(``rtmlib.Hand``) on the crop, then writing the refined 21 keypoints as a
separate ``detection_keypoints`` row (``region_type='hand_l'``/``'hand_r'``)
if the result passes a proximity gate against the tracked wrist.

The crop, candidate-selection, and gate formulas below are exactly the ones
validated against four rounds of offline stills testing (60+ crops, three
people, two trials) described in
``docs/roadmap/features/hand-detection-refinement/hand-detection-refinement-design.md``
("Idea 2"). Phase 1 wrote hand keypoints patched in place into the existing
133-point whole-body row, inheriting its ``noise_scale``; Phase 2 (here)
writes them as their own row with a noise_scale derived from the hand crop's
own size, so a tight, low-noise hand crop isn't diluted by the wider
whole-body bbox's noise.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from posetrak.detection import stage_timing as st
from posetrak.detection.frame_source import iter_frames
from posetrak.detection.pipeline import CameraInfo
from posetrak.detection.pose_models import COCO133, COCO133_HAND_BASE, HAND_OBS_SOURCE, spec_for_pose_model

_log = logging.getLogger(__name__)

try:
    # Importing backends_rtmpose first runs its Windows onnxruntime-gpu
    # DLL-path setup, which rtmlib.Hand also depends on internally.
    import posetrak.detection.backends_rtmpose as _rtmpose_backend  # noqa: F401
    from rtmlib import Hand as _Hand
    _RTMLIB_AVAILABLE = True
except ImportError:
    _RTMLIB_AVAILABLE = False

# Validated against 60+ curated stills across three people and two trials —
# see the design doc's "Idea 2" section for the derivation and results.
_FOREARM_MULT = 0.9
_HALF_FLOOR_PX = 60.0
_OFFSET_FRAC = 0.35
_GATE_FRAC_OF_FOREARM = 0.5
_GATE_FLOOR_PX = 40.0

# rtmlib.Hand's own per-keypoint confidence is a genuine 0-1 score, unlike
# the whole-body RTMPose model's raw SimCC logits (typically 3-8 for a
# clearly-visible joint, see backends_rtmpose.py's _KNOWN_MODELS comment).
# Written into the same kp_blob confidence column as those logits, an
# unscaled 0-1 value would make even a perfect hand detection look far less
# trustworthy than an average body keypoint to the tracker's
# noise/confidence formula. Reuse the project's existing convention for
# this exact problem (ViTPose's heatmap-peak-to-SimCC-range conf_scale=5.0
# in backends_rtmpose.py) as a placeholder pending real empirical tuning —
# not a considered, validated value, same caveat as Idea 1's interim
# edited_kp_noise_std.
_HAND_CONF_SCALE = 5.0

# The wrist and elbow slots come from the pose model's registry entry, so the pass works for every model that
# names them.
_HAND_N_KP = 21
_REGION_TYPE = HAND_OBS_SOURCE
# The COCO-133 slots, for the hand redetection of the editing UI, which works on COCO-133 sequences only.
_WRIST_IDX = {side: w for side, (w, _) in COCO133.arm_indices().items()}
_ELBOW_IDX = {side: e for side, (_, e) in COCO133.arm_indices().items()}
_HAND_BASE_IDX = COCO133_HAND_BASE

# rtmlib.Hand's RTMPose stage input size (square). Used to derive noise_scale
# from the hand crop's own pixel size the same way the whole-body pipeline
# derives it from the person bbox (bbox_w / pose_input_width) — a tighter
# crop implies a more precise detection, so a smaller crop_w_px/width ratio
# means lower measurement noise.
_HAND_POSE_INPUT_WIDTH = 256

ProgressCallback = Callable[[int, int, str], None]  # done, total, camera_id


@dataclass
class HandCandidate:
    keypoints: np.ndarray  # float32[21, 2], full-frame pixel coordinates
    scores: np.ndarray     # float32[21], the hand model's own 0-1 confidence
    root_dist_px: float    # winning candidate's root-to-wrist distance
    crop_w_px: float       # pixel width of the wrist-centred crop fed to hand_model


@dataclass
class HandCropPlan:
    """The wrist-centred crop of one hand and the numbers its gate needs."""

    x0: int
    y0: int
    x1: int
    y1: int
    wrist: tuple[float, float]
    forearm_len: float
    half: float


def plan_hand_crop(
    image_shape: tuple[int, ...],
    wrist: tuple[float, float],
    elbow: tuple[float, float] | None,
) -> HandCropPlan | None:
    """The crop around *wrist* (offset along the forearm towards the hand), or None if it is degenerate.

    *elbow* is ``None`` when it isn't confidently known, in which case the crop centres on the wrist with no
    offset and the gate falls back to its floor value.
    """
    wx, wy = wrist
    img_h, img_w = image_shape[:2]

    forearm_len = 0.0
    ex = ey = None
    if elbow is not None:
        ex, ey = elbow
        forearm_len = float(np.hypot(wx - ex, wy - ey))

    half = max(_FOREARM_MULT * forearm_len, _HALF_FLOOR_PX)
    cx, cy = wx, wy
    if forearm_len > 1e-3:
        dirx, diry = (wx - ex) / forearm_len, (wy - ey) / forearm_len
        cx = wx + _OFFSET_FRAC * forearm_len * dirx
        cy = wy + _OFFSET_FRAC * forearm_len * diry

    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(img_w, cx + half)), int(min(img_h, cy + half))
    if x1 <= x0 or y1 <= y0:
        _log.debug(
            "plan_hand_crop: degenerate crop wrist=%s elbow=%s -> (%d,%d,%d,%d), rejecting",
            wrist, elbow, x0, y0, x1, y1,
        )
        return None
    return HandCropPlan(x0, y0, x1, y1, (wx, wy), forearm_len, half)


def gate_hand_candidates(plan: HandCropPlan, keypoints, scores) -> HandCandidate | None:
    """Pick the candidate hand whose root keypoint lands closest to the wrist; None if none passes the gate.

    *keypoints* is ``float32[n_hands, 21, 2]`` (crop-local) and *scores* ``float32[n_hands, 21]`` as the hand
    model returns them for the crop of *plan*.
    """
    wx, wy = plan.wrist
    x0, y0, forearm_len = plan.x0, plan.y0, plan.forearm_len
    n_det = len(keypoints) if keypoints is not None else 0
    if n_det == 0:
        _log.debug(
            "gate_hand_candidates: no candidates  crop=(%d,%d,%d,%d)  forearm_len=%.1f",
            x0, y0, plan.x1, plan.y1, forearm_len,
        )
        return None

    gate_thr = max(_GATE_FRAC_OF_FOREARM * forearm_len, _GATE_FLOOR_PX)
    best_idx, best_dist = None, None
    for i in range(n_det):
        dist = float(np.hypot(keypoints[i][0][0] + x0 - wx, keypoints[i][0][1] + y0 - wy))
        if best_dist is None or dist < best_dist:
            best_dist, best_idx = dist, i

    if best_idx is None or best_dist > gate_thr:
        _log.debug(
            "gate_hand_candidates: gate REJECT  n_candidates=%d  best_dist=%.1fpx  gate_thr=%.1fpx",
            n_det, best_dist if best_dist is not None else -1.0, gate_thr,
        )
        return None

    _log.debug(
        "gate_hand_candidates: gate PASS  n_candidates=%d  best_dist=%.1fpx  gate_thr=%.1fpx  crop_w=%.0fpx",
        n_det, best_dist, gate_thr, 2.0 * plan.half,
    )
    kp_full = keypoints[best_idx].astype(np.float32).copy()
    kp_full[:, 0] += x0
    kp_full[:, 1] += y0
    return HandCandidate(
        keypoints=kp_full,
        scores=scores[best_idx].astype(np.float32),
        root_dist_px=best_dist,
        crop_w_px=2.0 * plan.half,
    )


def detect_hand_in_crop(
    hand_model,
    image: np.ndarray,
    wrist: tuple[float, float],
    elbow: tuple[float, float] | None,
) -> HandCandidate | None:
    """Crop around *wrist*, run *hand_model*, gate the result by proximity.

    *hand_model* is anything callable as ``keypoints, scores = hand_model(crop)``
    returning ``keypoints: float32[n_hands, 21, 2]`` (crop-local) and
    ``scores: float32[n_hands, 21]`` — the ``rtmlib.Hand`` call signature.

    *wrist* and *elbow* are full-frame pixel coordinates (see :func:`plan_hand_crop`). Returns ``None`` on no
    detection or a gate reject.
    """
    plan = plan_hand_crop(image.shape, wrist, elbow)
    if plan is None:
        return None
    keypoints, scores = hand_model(image[plan.y0:plan.y1, plan.x0:plan.x1])
    return gate_hand_candidates(plan, keypoints, scores)


def hand_rows_of(candidates: dict[str, HandCandidate]) -> list[tuple[str, np.ndarray, float]]:
    """``(region_type, hand_kp[21, 3], noise_scale)`` rows for the gated *candidates* by side."""
    rows = []
    for side, result in candidates.items():
        hand_kp = np.empty((_HAND_N_KP, 3), dtype=np.float32)
        hand_kp[:, 0] = result.keypoints[:, 0]
        hand_kp[:, 1] = result.keypoints[:, 1]
        hand_kp[:, 2] = result.scores * _HAND_CONF_SCALE
        rows.append((_REGION_TYPE[side], hand_kp, result.crop_w_px / _HAND_POSE_INPUT_WIDTH))
    return rows


def hand_wrists(
    kp: np.ndarray, arms: dict[str, tuple[int, int]],
) -> dict[str, tuple[tuple[float, float], tuple[float, float] | None]]:
    """``{side: (wrist, elbow or None)}`` of the sides of keypoints *kp* whose wrist is confident.

    *arms* is ``PoseModelSpec.arm_indices()`` of the model *kp* comes from.
    """
    out = {}
    for side, (wrist_idx, elbow_idx) in arms.items():
        wx, wy, wc = kp[wrist_idx]
        if wc <= 0.0:
            continue
        ex, ey, ec = kp[elbow_idx]
        out[side] = ((float(wx), float(wy)), (float(ex), float(ey)) if ec > 0.0 else None)
    return out


class HandBatchModel:
    """``rtmlib.Hand`` with the pose stage run once for the hands found in many crops.

    The hand detector takes one image per call (its ONNX graph has batch size 1) and runs per crop; the pose
    model takes any batch and runs on every detected hand of every crop together. A crop in which the detector
    finds nothing goes to the pose model whole, as ``rtmlib.Hand`` does.
    """

    def __init__(self, hand=None) -> None:
        if hand is None:
            device = _rtmpose_backend._auto_device()
            hand = _Hand(to_openpose=False, backend="onnxruntime", device=device)
        self._hand = hand

    def __call__(self, crops: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """``(keypoints[n, 21, 2], scores[n, 21])`` per crop."""
        with st.stage("hand_model"):
            boxes = [self._hand.det_model(crop) for crop in crops]
            counts = [max(len(found), 1) for found in boxes]
            jobs = [(crop, np.asarray(box, dtype=np.float32))
                    for crop, found in zip(crops, boxes)
                    for box in (found if len(found) else [[0, 0, crop.shape[1], crop.shape[0]]])]
            decoded = iter(_rtmpose_backend.run_rtmlib_batch(self._hand.pose_model, jobs))
            out = []
            for n in counts:
                pairs = [next(decoded) for _ in range(n)]
                out.append((np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])))
            return out


class HandPass:
    """Refined hand keypoints for the poses of a batch of frames, the hand crops batched.

    *pose_model* is the stored model name of the poses (COCO-133 when not given); it says which keypoints are
    the wrists and elbows.

    Used inside the detection loop on the frames already decoded; ``HandRefinementPipeline`` is the same
    refinement as a second pass over a stored run.
    """

    def __init__(self, model: HandBatchModel | None = None, pose_model: str | None = None) -> None:
        self._model = model
        spec = spec_for_pose_model(pose_model) if pose_model else COCO133
        self._arms = spec.arm_indices() if spec is not None else {}
        self._width = len(spec.keypoint_names) if spec is not None else 0

    @staticmethod
    def supports(pose_model: str | None) -> bool:
        """True if *pose_model* names a wrist and an elbow."""
        spec = spec_for_pose_model(pose_model)
        return spec is not None and bool(spec.arm_indices())

    @property
    def model(self) -> HandBatchModel:
        if self._model is None:
            self._model = HandBatchModel()
        return self._model

    def refine(self, items: list[tuple[np.ndarray, np.ndarray]]) -> list[list[tuple[str, np.ndarray, float]]]:
        """For each ``(frame, keypoints[N, 3])`` the ``(region_type, hand_kp, noise_scale)`` rows of its hands."""
        plans = []                                   # (item index, side, plan)
        for index, (frame, kp) in enumerate(items):
            if kp.shape[0] != self._width:
                continue
            for side, (wrist, elbow) in hand_wrists(kp, self._arms).items():
                plan = plan_hand_crop(frame.shape, wrist, elbow)
                if plan is not None:
                    plans.append((index, side, plan))
        found: list[dict[str, HandCandidate]] = [{} for _ in items]
        if plans:
            crops = [items[i][0][p.y0:p.y1, p.x0:p.x1] for i, _, p in plans]
            for (index, side, plan), (kps, scores) in zip(plans, self.model(crops)):
                candidate = gate_hand_candidates(plan, kps, scores)
                if candidate is not None:
                    found[index][side] = candidate
        return [hand_rows_of(candidates) for candidates in found]


class HandRefinementPipeline:
    """Idea 2: write refined hand keypoints for an existing detection run.

    Reads ``detection_keypoints`` rows (``region_type='full_body'``) for a
    completed run, and for each frame/track/side with a confidently known
    wrist, re-detects the hand in a tight crop; if the result passes the
    proximity gate, writes it as its own ``detection_keypoints`` row
    (``region_type='hand_l'``/``'hand_r'``, 21 keypoints) with a noise_scale
    derived from the hand crop's own size — a separate row rather than a
    patch into the whole-body row, so ``finalise_to_db`` can carry it into
    ``pose_observations`` as its own ``source`` with its own measurement
    noise instead of inheriting the whole-body bbox's noise_scale.
    """

    def __init__(
        self,
        session: sqlite3.Connection,
        stop_event: threading.Event | None = None,
    ) -> None:
        if not _RTMLIB_AVAILABLE:
            raise ImportError(
                "rtmlib is required for HandRefinementPipeline. "
                "Install from the rtmlib repository."
            )
        self._session = session
        self._stop_event = stop_event or threading.Event()
        self._hand_model = None
        self._arms: dict[str, tuple[int, int]] = {}
        self._width = 0

    def _get_hand_model(self):
        if self._hand_model is None:
            device = _rtmpose_backend._auto_device()
            self._hand_model = _Hand(to_openpose=False, backend="onnxruntime", device=device)
        return self._hand_model

    def run(
        self,
        run_id: str,
        cameras: list[CameraInfo],
        on_progress: ProgressCallback | None = None,
        on_camera_done: Callable[[int, int], None] | None = None,
    ) -> int:
        """Refine hands for every camera in *cameras* for run *run_id*.

        Returns the number of (frame, track, side) hand detections that
        passed the gate and were written. No-ops (returns 0) if the run's
        pose model names no wrist and elbow.

        *on_camera_done* (done, total), if given, fires after each camera
        finishes -- mirrors DetectionPipeline.run()'s own callback of the
        same name, so a caller driving one combined "N/M cameras" progress
        indicator across both the detection and hand-refinement passes
        doesn't need two different callback shapes.
        """
        row = self._session.execute(
            "SELECT pose_model FROM detection_runs WHERE id=?", (run_id,)
        ).fetchone()
        pose_model = row["pose_model"] if row else None
        if not HandPass.supports(pose_model):
            _log.info("run: pose model %r names no wrist and elbow, skipping", pose_model)
            return 0
        spec = spec_for_pose_model(pose_model)
        self._arms, self._width = spec.arm_indices(), len(spec.keypoint_names)

        total_refined = 0
        for i, cam in enumerate(cameras):
            if self._stop_event.is_set():
                break
            total_refined += self._process_camera(run_id, cam, on_progress)
            if on_camera_done:
                on_camera_done(i + 1, len(cameras))
        return total_refined

    def _process_camera(
        self,
        run_id: str,
        cam: CameraInfo,
        on_progress: ProgressCallback | None,
    ) -> int:
        rows = self._session.execute(
            "SELECT video_frame, track_id, keypoints FROM detection_keypoints"
            " WHERE detection_run_id=? AND shot_video_id=? AND region_type='full_body'"
            " ORDER BY video_frame",
            (run_id, cam.shot_video_id),
        ).fetchall()
        if not rows:
            return 0

        by_frame: dict[int, list[sqlite3.Row]] = {}
        for r in rows:
            by_frame.setdefault(r["video_frame"], []).append(r)
        first_frame, last_frame = min(by_frame), max(by_frame) + 1
        total = len(by_frame)

        hand_model = self._get_hand_model()
        n_refined = 0
        frames_done = 0
        updates: list[tuple] = []
        for video_frame, img in iter_frames(cam.file_path, first_frame, last_frame):
            if self._stop_event.is_set():
                break
            frame_rows = by_frame.get(video_frame)
            if frame_rows is not None:
                for row in frame_rows:
                    kp = np.frombuffer(bytes(row["keypoints"]), dtype=np.float32).reshape(-1, 3)
                    if kp.shape[0] != self._width:
                        continue
                    for region_type, hand_kp, noise_scale in self._refine_one(hand_model, kp, img):
                        updates.append((
                            run_id, cam.shot_video_id, video_frame, row["track_id"],
                            region_type, hand_kp.tobytes(), noise_scale,
                        ))
                        n_refined += 1
                frames_done += 1
                if on_progress:
                    on_progress(frames_done, total, cam.label or cam.camera_instance_id)
                if len(updates) >= 200:
                    self._flush(updates)
                    updates.clear()
        self._flush(updates)
        _log.info(
            "_process_camera: %s done — %d/%d frames, %d hands refined",
            cam.label or cam.camera_instance_id, frames_done, total, n_refined,
        )
        return n_refined

    def _refine_one(
        self, hand_model, kp: np.ndarray, img: np.ndarray,
    ) -> list[tuple[str, np.ndarray, float]]:
        """Detect and gate both hands' keypoints given whole-body *kp* + *img*.

        Read-only with respect to *kp*. Returns a list of (region_type,
        hand_kp[21,3], noise_scale) — one entry per side whose hand was found
        and passed the gate — ready to write as new detection_keypoints rows.
        """
        candidates = {}
        for side, (wrist, elbow) in hand_wrists(kp, self._arms).items():
            result = detect_hand_in_crop(hand_model, img, wrist, elbow)
            if result is not None:
                candidates[side] = result
        return hand_rows_of(candidates)

    def _flush(self, updates: list[tuple]) -> None:
        if not updates:
            return
        self._session.executemany(
            "INSERT OR REPLACE INTO detection_keypoints"
            " (detection_run_id, shot_video_id, video_frame, track_id,"
            "  region_type, keypoints, noise_scale)"
            " VALUES (?,?,?,?,?,?,?)",
            updates,
        )
        self._session.commit()
