"""hand_refinement.py — Idea 2 Phase 1: hand-specific detection pass.

Refines wrist/finger keypoints for an existing full-body detection run by
cropping around each tracked wrist and running a dedicated hand model
(``rtmlib.Hand``) on the crop, then patching the refined 21 keypoints back
into the same ``detection_keypoints`` row (``region_type='full_body'``) if
the result passes a proximity gate against the tracked wrist.

Phase 1 is the interim, no-schema-change version described in
``docs/roadmap/features/hand-detection-refinement/hand-detection-refinement-design.md``
("Idea 2" and "Phasing"): hand keypoints are patched in place in the
existing 133-point ``kp_blob``/row rather than written as a separate
``source='hand.L'/'hand.R'`` row (Phase 2, not built), so they still
inherit the frame's whole-body ``noise_scale`` rather than getting their
own tighter, crop-derived value. The crop, candidate-selection, and gate
formulas below are exactly the ones validated there against four rounds of
offline stills testing (60+ crops, three people, two trials).
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

# COCO-133 keypoint indices (see app/pose/kp_models.py). Phase 1 only
# refines runs using the 133-point whole-body layout — a 17-point model has
# no hand keypoints to refine at all.
_N_KP_133 = 133
_WRIST_IDX = {"left": 9, "right": 10}
_ELBOW_IDX = {"left": 7, "right": 8}
_HAND_BASE_IDX = {"left": 91, "right": 112}
_HAND_N_KP = 21

ProgressCallback = Callable[[int, int, str], None]  # done, total, camera_id


@dataclass
class HandCandidate:
    keypoints: np.ndarray  # float32[21, 2], full-frame pixel coordinates
    scores: np.ndarray     # float32[21], the hand model's own 0-1 confidence
    root_dist_px: float    # winning candidate's root-to-wrist distance


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
        return None

    keypoints, scores = hand_model(image[y0:y1, x0:x1])
    n_det = len(keypoints) if keypoints is not None else 0
    if n_det == 0:
        return None

    gate_thr = max(_GATE_FRAC_OF_FOREARM * forearm_len, _GATE_FLOOR_PX)
    best_idx, best_dist = None, None
    for i in range(n_det):
        dist = float(np.hypot(keypoints[i][0][0] + x0 - wx, keypoints[i][0][1] + y0 - wy))
        if best_dist is None or dist < best_dist:
            best_dist, best_idx = dist, i

    if best_idx is None or best_dist > gate_thr:
        return None

    kp_full = keypoints[best_idx].astype(np.float32).copy()
    kp_full[:, 0] += x0
    kp_full[:, 1] += y0
    return HandCandidate(
        keypoints=kp_full,
        scores=scores[best_idx].astype(np.float32),
        root_dist_px=best_dist,
    )


class HandRefinementPipeline:
    """Idea 2 Phase 1: patch hand keypoints into an existing detection run.

    Reads ``detection_keypoints`` rows (``region_type='full_body'``) for a
    completed run, and for each frame/track/side with a confidently known
    wrist, re-detects the hand in a tight crop; if the result passes the
    proximity gate, overwrites that hand's 21 keypoint slots (indices
    91-111 left, 112-132 right of the 133-point layout) in place.
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
            self._hand_model = _Hand(to_openpose=False, backend="onnxruntime", device="cpu")
        return self._hand_model

    def run(
        self,
        run_id: str,
        cameras: list[CameraInfo],
        on_progress: ProgressCallback | None = None,
    ) -> int:
        """Refine hands for every camera in *cameras* for run *run_id*.

        Returns the number of (frame, track, side) hand detections that
        passed the gate and were written. No-ops (returns 0) if the run's
        pose model isn't the 133-point whole-body layout.
        """
        row = self._session.execute(
            "SELECT pose_model FROM detection_runs WHERE id=?", (run_id,)
        ).fetchone()
        pose_model = row["pose_model"] if row else None
        if not pose_model or "133" not in pose_model:
            _log.info("run: pose model %r has no hand keypoints, skipping", pose_model)
            return 0

        total_refined = 0
        for cam in cameras:
            if self._stop_event.is_set():
                break
            total_refined += self._process_camera(run_id, cam, on_progress)
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
                    kp = kp.copy()
                    if self._refine_one(hand_model, kp, img):
                        updates.append((kp.tobytes(), run_id, cam.shot_video_id, video_frame, row["track_id"]))
                        n_refined += 1
                frames_done += 1
                if on_progress:
                    on_progress(frames_done, total, cam.camera_instance_id)
                if len(updates) >= 200:
                    self._flush(updates)
                    updates.clear()
        self._flush(updates)
        _log.info(
            "_process_camera: %s done — %d/%d frames, %d hands refined",
            cam.camera_instance_id, frames_done, total, n_refined,
        )
        return n_refined

    def _refine_one(self, hand_model, kp: np.ndarray, img: np.ndarray) -> bool:
        """Refine both hands' keypoints in *kp* (float32[133,3]) in place."""
        changed = False
        for side in ("left", "right"):
            wx, wy, wc = kp[_WRIST_IDX[side]]
            if wc <= 0.0:
                continue
            ex, ey, ec = kp[_ELBOW_IDX[side]]
            elbow = (float(ex), float(ey)) if ec > 0.0 else None
            result = detect_hand_in_crop(hand_model, img, (float(wx), float(wy)), elbow)
            if result is None:
                continue
            base = _HAND_BASE_IDX[side]
            kp[base:base + _HAND_N_KP, 0] = result.keypoints[:, 0]
            kp[base:base + _HAND_N_KP, 1] = result.keypoints[:, 1]
            kp[base:base + _HAND_N_KP, 2] = result.scores * _HAND_CONF_SCALE
            changed = True
        return changed

    def _flush(self, updates: list[tuple]) -> None:
        if not updates:
            return
        self._session.executemany(
            "UPDATE detection_keypoints SET keypoints=?"
            " WHERE detection_run_id=? AND shot_video_id=? AND video_frame=?"
            " AND track_id=? AND region_type='full_body'",
            updates,
        )
        self._session.commit()
