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

from posetrak.detection.frame_source import iter_frames
from posetrak.detection.pipeline import CameraInfo

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

# COCO-133 keypoint indices (see app/pose/kp_models.py). Hand refinement only
# runs against the 133-point whole-body layout — a 17-point model has no
# hand keypoints to refine at all.
_N_KP_133 = 133
_WRIST_IDX = {"left": 9, "right": 10}
_ELBOW_IDX = {"left": 7, "right": 8}
# Canonical source of truth for the hand21 -> COCO-133 index mapping;
# posetrak.db.observation_merge and the C++ session loader reference these
# values (91/112/21) in their own comments rather than importing this module.
_HAND_BASE_IDX = {"left": 91, "right": 112}
_HAND_N_KP = 21
_REGION_TYPE = {"left": "hand_l", "right": "hand_r"}

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

    *wrist* and *elbow* are full-frame pixel coordinates; *elbow* is ``None``
    when it isn't confidently known, in which case the crop centres on the
    wrist with no offset and the gate falls back to its floor value.

    Picks the candidate hand whose own root keypoint (index 0) lands
    closest to *wrist*, then rejects it if that distance still exceeds the
    gate threshold. Returns ``None`` on no detection or a gate reject.
    """
    wx, wy = wrist
    img_h, img_w = image.shape[:2]

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
            "detect_hand_in_crop: degenerate crop wrist=%s elbow=%s -> (%d,%d,%d,%d), rejecting",
            wrist, elbow, x0, y0, x1, y1,
        )
        return None

    keypoints, scores = hand_model(image[y0:y1, x0:x1])
    n_det = len(keypoints) if keypoints is not None else 0
    if n_det == 0:
        _log.debug(
            "detect_hand_in_crop: no candidates  crop=(%d,%d,%d,%d)  forearm_len=%.1f",
            x0, y0, x1, y1, forearm_len,
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
            "detect_hand_in_crop: gate REJECT  n_candidates=%d  best_dist=%.1fpx  gate_thr=%.1fpx",
            n_det, best_dist if best_dist is not None else -1.0, gate_thr,
        )
        return None

    _log.debug(
        "detect_hand_in_crop: gate PASS  n_candidates=%d  best_dist=%.1fpx  gate_thr=%.1fpx  crop_w=%.0fpx",
        n_det, best_dist, gate_thr, 2.0 * half,
    )
    kp_full = keypoints[best_idx].astype(np.float32).copy()
    kp_full[:, 0] += x0
    kp_full[:, 1] += y0
    return HandCandidate(
        keypoints=kp_full,
        scores=scores[best_idx].astype(np.float32),
        root_dist_px=best_dist,
        crop_w_px=2.0 * half,
    )


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
        pose model isn't the 133-point whole-body layout.

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
        if not pose_model or "133" not in pose_model:
            _log.info("run: pose model %r has no hand keypoints, skipping", pose_model)
            return 0

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
                    if kp.shape[0] != _N_KP_133:
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
        results: list[tuple[str, np.ndarray, float]] = []
        for side in ("left", "right"):
            wx, wy, wc = kp[_WRIST_IDX[side]]
            if wc <= 0.0:
                continue
            ex, ey, ec = kp[_ELBOW_IDX[side]]
            elbow = (float(ex), float(ey)) if ec > 0.0 else None
            result = detect_hand_in_crop(hand_model, img, (float(wx), float(wy)), elbow)
            if result is None:
                continue
            hand_kp = np.empty((_HAND_N_KP, 3), dtype=np.float32)
            hand_kp[:, 0] = result.keypoints[:, 0]
            hand_kp[:, 1] = result.keypoints[:, 1]
            hand_kp[:, 2] = result.scores * _HAND_CONF_SCALE
            noise_scale = result.crop_w_px / _HAND_POSE_INPUT_WIDTH
            results.append((_REGION_TYPE[side], hand_kp, noise_scale))
        return results

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
