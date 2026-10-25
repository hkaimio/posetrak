# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""backends_soma77.py — a 77-keypoint (SOMA-77) DINOv3 ViTPose model run through ONNX Runtime.

The model is the 2D pose network of NVIDIA's GEM-X release (``onnx/vitpose.onnx``): a DINOv3 ViT-H backbone with a
heatmap head, input 3 x 256 x 192, output 77 heatmaps of 64 x 48. The keypoints are the joint
centres of the SOMA skeleton (``pose_models.SOMA77_NAMES``), not surface landmarks.

Pre- and post-processing follow GEM-X's ``VitPoseExtractor``: the person box becomes a square of side
``s`` around its centre, warped to 256 x 256 of which the central 192 columns are the model input
(the model sees a window 0.75 s wide and s high); heatmaps of the image and of its mirror are
averaged (mirrored through ``FLIP_PAIRS``) and decoded by the peak cell with a parabolic sub-cell refinement
(GEM-X's own quarter-cell step is available as ``refine="sign"``); the confidence is the heatmap peak.

A cell index ``i`` stands for the centre of the cell, ``i + 0.5``; GEM-X's decoder uses ``i``, which puts every
keypoint half a cell (about 1.3% of the person's box height) left of and above where the multi-camera geometry
places it. The difference is the same in every camera and does not matter for a single camera, but it is a
systematic triangulation error with several. It is independent of the crop size and of the refinement.

A box much wider than tall is clipped by the 0.75 window; callers that track people in horizontal
poses need to account for that.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from posetrak.detection import stage_timing as st
from posetrak.detection.backends import PersonDetection, PoseResult, register_estimator
from posetrak.detection.kp_candidates import CandidateOptions, extract_candidates
from posetrak.detection.pose_models import SOMA77_FLIP_PAIRS as FLIP_PAIRS
from posetrak.detection.pose_models import SOMA77_NAMES, SOMA77_PARENTS  # noqa: F401  (re-exported)

INPUT_H, INPUT_W = 256, 192
HEATMAP_H, HEATMAP_W = 64, 48
BOX_ENLARGE = 1.6         # box side = BOX_ENLARGE * the longer side of the person box; at 1.2 the model
                          # sometimes returns the whole pose at half scale about the crop centre when the
                          # person fills the crop, which a larger crop avoids (spike notes)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_square(bbox_cxcywh: np.ndarray, enlarge: float = BOX_ENLARGE) -> tuple[float, float, float]:
    """(cx, cy, s): the square the model's input window is cut from, from a centre-format box."""
    cx, cy, w, h = (float(v) for v in bbox_cxcywh)
    return cx, cy, max(w, h) * enlarge


def preprocess(frame_bgr: np.ndarray, cx: float, cy: float, s: float) -> np.ndarray:
    """float32 [3, 256, 192] model input for the square of side *s* centred on (cx, cy)."""
    half = s / 2
    src = np.array([[cx - half, cy - half], [cx + half, cy - half], [cx, cy]], dtype=np.float32)
    dst = np.array([[0, 0], [255, 0], [127.5, 127.5]], dtype=np.float32)
    crop = cv2.warpAffine(frame_bgr, cv2.getAffineTransform(src, dst), (256, 256), flags=cv2.INTER_LINEAR)
    rgb = (crop[..., ::-1].astype(np.float32) / 255.0 - _MEAN) / _STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[:, :, 32:224])


def flip_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """Heatmaps of a mirrored image turned back to the image's own left and right: [K, H, W]."""
    out = heatmaps.copy()
    for left, right in FLIP_PAIRS:
        out[left], out[right] = heatmaps[right], heatmaps[left]
    return out[:, :, ::-1]


def cells_to_frame(px, py, cx: float, cy: float, s: float):
    """Frame pixels of heatmap positions (*px*, *py*), given in cells with the cell index at its left edge.

    The window the model sees is 0.75 s wide and s high around (cx, cy); a position is placed at the centre of its
    cell, so index ``i`` maps to ``i + 0.5``.
    """
    window_w, window_h = 0.75 * s, s
    return ((np.asarray(px) + 0.5) / HEATMAP_W * window_w + cx - window_w / 2,
            (np.asarray(py) + 0.5) / HEATMAP_H * window_h + cy - window_h / 2)


def decode(heatmaps: np.ndarray, cx: float, cy: float, s: float, refine: str = "parabolic") -> np.ndarray:
    """float32 [K, 3] (x, y, confidence) in frame pixels from heatmaps [K, H, W].

    The peak is the heatmap's maximum cell. *refine* places the keypoint inside the cell: ``"parabolic"`` fits a
    parabola through the logarithm of the peak and its two neighbours on each axis (the position of a Gaussian
    peak, to a fraction of a cell), ``"sign"`` is GEM-X's step of a quarter cell towards the higher neighbour,
    which leaves the keypoints on a grid a quarter of a cell apart.
    """
    k, h, w = heatmaps.shape
    flat = heatmaps.reshape(k, -1)
    index = flat.argmax(-1)
    px = (index % w).astype(np.float64)
    py = (index // w).astype(np.float64)
    logs = np.log(np.maximum(heatmaps, 1e-4)) if refine == "parabolic" else None
    for i in range(k):
        x, y = int(px[i]), int(py[i])
        if refine == "sign":
            if 1 < x < w - 1:
                px[i] += np.sign(heatmaps[i, y, x + 1] - heatmaps[i, y, x - 1]) * 0.25
            if 1 < y < h - 1:
                py[i] += np.sign(heatmaps[i, y + 1, x] - heatmaps[i, y - 1, x]) * 0.25
        elif refine == "parabolic":
            if 0 < x < w - 1:
                before, peak, after = logs[i, y, x - 1], logs[i, y, x], logs[i, y, x + 1]
                curvature = before - 2 * peak + after
                if curvature < 0:
                    px[i] += np.clip(0.5 * (before - after) / curvature, -0.5, 0.5)
            if 0 < y < h - 1:
                before, peak, after = logs[i, y - 1, x], logs[i, y, x], logs[i, y + 1, x]
                curvature = before - 2 * peak + after
                if curvature < 0:
                    py[i] += np.clip(0.5 * (before - after) / curvature, -0.5, 0.5)
        else:
            raise ValueError(f"unknown refinement {refine!r}")
    out = np.empty((k, 3), dtype=np.float32)
    out[:, 0], out[:, 1] = cells_to_frame(px, py, cx, cy, s)
    out[:, 2] = flat.max(-1)
    return out


def mask_cells(inside: np.ndarray, native_w: float, native_h: float, cx: float, cy: float, s: float) -> np.ndarray:
    """float32 [64, 48]: 1 where the centre of a heatmap cell of the crop lies on the mask, else 0.

    *inside* is the mask (non-zero on the person, grown by a margin) at any resolution covering the whole
    frame of native size *native_w* x *native_h*; cells outside the frame count as off the mask.
    """
    mask_h, mask_w = inside.shape
    xs = cx - 0.375 * s + (np.arange(HEATMAP_W) + 0.5) / HEATMAP_W * 0.75 * s
    ys = cy - 0.5 * s + (np.arange(HEATMAP_H) + 0.5) / HEATMAP_H * s
    gx = np.clip((xs / native_w * mask_w).astype(int), 0, mask_w - 1)
    gy = np.clip((ys / native_h * mask_h).astype(int), 0, mask_h - 1)
    cells = inside[np.ix_(gy, gx)].astype(np.float32)
    on_frame = (xs[None, :] >= 0) & (xs[None, :] < native_w) & (ys[:, None] >= 0) & (ys[:, None] < native_h)
    return np.where(on_frame, cells, 0.0).astype(np.float32)


def gate_heatmaps(heatmaps: np.ndarray, cells: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Heatmaps [K, H, W] multiplied by the person-mask cells [H, W]; *floor* keeps that share of off-mask cells."""
    return heatmaps * (floor + (1.0 - floor) * cells)[None]


class Soma77Onnx:
    """The model in an ONNX Runtime session."""

    def __init__(self, model_path: str | Path, providers: tuple[str, ...] | None = None) -> None:
        import onnxruntime as ort

        try:                         # torch brings the CUDA runtime DLLs onnxruntime-gpu needs on Windows
            import torch  # noqa: F401
        except ImportError:
            pass
        options = ort.SessionOptions()
        options.log_severity_level = 3   # the exported graph declares its output batch size as 1; it takes any
        self.session = ort.InferenceSession(
            str(model_path), options, providers=list(providers or ("CUDAExecutionProvider", "CPUExecutionProvider")))

    def heatmaps(self, batch: np.ndarray) -> np.ndarray:
        """float32 [B, 77, 64, 48] for a batch [B, 3, 256, 192]."""
        return self.session.run(None, {"imgs": np.ascontiguousarray(batch, dtype=np.float32)})[0]

    def keypoints(self, frame_bgr: np.ndarray, bbox_cxcywh: np.ndarray, flip_test: bool = True,
                  enlarge: float = BOX_ENLARGE) -> np.ndarray:
        """float32 [77, 3] keypoints of the person in *bbox_cxcywh* (frame pixels)."""
        cx, cy, s = crop_square(bbox_cxcywh, enlarge)
        image = preprocess(frame_bgr, cx, cy, s)
        if not flip_test:
            return decode(self.heatmaps(image[None])[0], cx, cy, s)
        both = self.heatmaps(np.stack([image, image[:, :, ::-1]]))
        return decode((both[0] + flip_heatmaps(both[1])) * 0.5, cx, cy, s)


class Soma77WeightsMissing(RuntimeError):
    """The GEM-X weights are not on this machine, or their licences have not been accepted."""


@register_estimator
class Soma77Estimator:
    """SOMA-77 as a pose estimator of the detection loop: batched, with the options the spike validated.

    The options (``options()``, recorded in ``detection_runs.config_json``): the flip test, the crop enlargement,
    the sub-cell refinement, decoding at cell centres, and the mask gate, which multiplies a person's heatmaps by
    the person's mask (``PersonDetection.mask``) when the box source provides one. *batch_crops* counts the
    crops of one model call including the flipped copies (16 is the measured 12 ms per crop).
    """

    name = "soma77-vitpose-h"
    version = "gem-x onnx"
    input_size = (INPUT_H, INPUT_W)

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
        *,
        flip_test: bool = True,
        enlarge: float = BOX_ENLARGE,
        refine: str = "parabolic",
        mask_gate: bool = True,
        mask_gate_floor: float = 0.0,
        batch_crops: int = 16,
        candidates: CandidateOptions | None = None,
        onnx: Soma77Onnx | None = None,
    ) -> None:
        self.candidates = candidates
        self.flip_test, self.enlarge, self.refine = flip_test, enlarge, refine
        self.mask_gate, self.mask_gate_floor, self.batch_crops = mask_gate, mask_gate_floor, batch_crops
        if onnx is None:
            from posetrak.detection.model_weights import soma77_model_path

            path = Path(model_path) if model_path else soma77_model_path()
            if path is None:
                raise Soma77WeightsMissing("The SOMA-77 weights are not downloaded or their licences not accepted.")
            providers = ("CPUExecutionProvider",) if device == "cpu" else None
            onnx = Soma77Onnx(path, providers)
            self._model_file = path.name
        else:
            self._model_file = "injected"
        session = getattr(onnx, "session", None)
        self.device = "cuda" if session is not None and "CUDAExecutionProvider" in session.get_providers() else "cpu"
        self._onnx = onnx

    def estimate(self, frame: np.ndarray, detections: list[PersonDetection]) -> list[PoseResult]:
        return self.estimate_batch([(frame, detections)])[0]

    def estimate_batch(self, items: Sequence[tuple[np.ndarray, list[PersonDetection]]]) -> list[list[PoseResult]]:
        jobs = []                                   # (item index, detection, frame, input, cx, cy, s)
        for index, (frame, detections) in enumerate(items):
            for det in detections:
                cx, cy, s = crop_square(det.bbox, self.enlarge)
                with st.stage("crop_preprocess"):
                    jobs.append((index, det, frame, preprocess(frame, cx, cy, s), cx, cy, s))
        results: list[list[PoseResult]] = [[] for _ in items]
        per_call = max(1, self.batch_crops // (2 if self.flip_test else 1))
        for lo in range(0, len(jobs), per_call):
            chunk = jobs[lo:lo + per_call]
            inputs = [x for job in chunk for x in ((job[3], job[3][:, :, ::-1]) if self.flip_test else (job[3],))]
            with st.stage("inference"):
                maps = self._onnx.heatmaps(np.stack(inputs))
            for i, (index, det, frame, _, cx, cy, s) in enumerate(chunk):
                with st.stage("heatmap_decode"):
                    heat = (maps[2 * i] + flip_heatmaps(maps[2 * i + 1])) * 0.5 if self.flip_test else maps[i]
                found = None
                if self.candidates is not None:
                    # The maxima come from the heatmap before the mask gate: the gate would remove exactly the
                    # off-person alternatives the tracker may need to choose among.
                    with st.stage("candidates"):
                        found = extract_candidates(heat, self.candidates, lambda fx, fy: cells_to_frame(fx, fy, cx, cy, s))
                if self.mask_gate and det.mask is not None:
                    with st.stage("mask_gate"):
                        cells = mask_cells(det.mask, frame.shape[1], frame.shape[0], cx, cy, s)
                        heat = gate_heatmaps(heat, cells, self.mask_gate_floor)
                with st.stage("heatmap_decode"):
                    kp = decode(heat, cx, cy, s, self.refine)
                results[index].append(PoseResult(det.track_id, kp, noise_scale=0.75 * s / INPUT_W, candidates=found))
        return results

    def options(self) -> dict:
        return {
            "model": self.name, "model_file": self._model_file, "flip_test": self.flip_test,
            "box_enlarge": self.enlarge, "refine": self.refine, "decoding": "cell-centre",
            "mask_gate": self.mask_gate, "mask_gate_floor": self.mask_gate_floor, "batch_crops": self.batch_crops,
        }
