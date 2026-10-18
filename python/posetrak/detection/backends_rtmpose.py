# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""backends_rtmpose.py — RTMPose estimator backend."""
from __future__ import annotations

import os
import sys
from typing import Sequence

import numpy as np

# On Windows, onnxruntime-gpu's CUDA provider DLL looks for CUDA libraries
# (cublasLt64_12.dll etc.) on the system PATH.  PyTorch bundles those DLLs
# in its own lib/ directory, so we register that directory before rtmlib
# (and thus onnxruntime) is first imported.
if sys.platform == "win32":
    try:
        import torch as _torch
        _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.isdir(_torch_lib):
            os.add_dll_directory(_torch_lib)
    except (ImportError, OSError, AttributeError):
        pass

    # Also on Windows: Python's ssl module doesn't get the automatic AIA
    # (Authority Information Access) chasing that browsers/WinINet-based
    # apps get for free from the Windows certificate store, so a machine
    # that's never made an HTTPS connection needing a given CA's
    # intermediate certificate before -- a fresh install, or a Windows
    # Sandbox test run, both confirmed by hand, 2026-08-23 -- can fail
    # rtmlib's model-checkpoint download with "CERTIFICATE_VERIFY_FAILED:
    # unable to get local issuer certificate" even though the same URL
    # opens fine in any browser on that same machine. Point Python's
    # default SSL context at certifi's independently-maintained root CA
    # bundle instead of relying solely on Windows' local store, which
    # ssl.create_default_context() (used internally by rtmlib's plain
    # urllib download) still honours alongside the OS store. See
    # backends_rtmdet.py.
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

try:
    from rtmlib.tools.pose_estimation import RTMPose as _RTMPose
    from rtmlib.tools.pose_estimation.vitpose import ViTPose as _ViTPose
    import rtmlib
    _RTMLIB_VERSION = getattr(rtmlib, "__version__", "unknown")
    _RTMLIB_AVAILABLE = True
except ImportError:
    _RTMLIB_AVAILABLE = False
    _RTMLIB_VERSION = ""

from posetrak.detection import stage_timing as st
from posetrak.detection.backends import (
    PersonDetection,
    PoseResult,
    construct_with_corrupt_checkpoint_retry,
    register_estimator,
)

# Crops per model call. The models take any batch size; this bounds GPU memory.
MAX_BATCH = 16


def run_rtmlib_batch(
    model, jobs: list[tuple[np.ndarray, np.ndarray]], max_batch: int = MAX_BATCH,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Keypoints and scores of an rtmlib pose model (RTMPose or ViTPose) on many crops in few calls.

    *jobs* are ``(image, bbox_xyxy)`` pairs, the images of any size. Returns ``(keypoints [K, 2], scores [K])``
    per job, identical to calling ``model(image, [bbox])`` per job: only the model call is batched, the crop
    preparation and the output decoding are the model's own.
    """
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for lo in range(0, len(jobs), max_batch):
        chunk = jobs[lo:lo + max_batch]
        with st.stage("crop_preprocess"):
            prepared = [model.preprocess(image, bbox) for image, bbox in chunk]
            batch = np.stack([np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) for img, _, _ in prepared])
        with st.stage("inference"):
            name = model.session.get_inputs()[0].name
            outputs = model.session.run([o.name for o in model.session.get_outputs()], {name: batch})
        with st.stage("heatmap_decode"):
            for i, (_, center, scale) in enumerate(prepared):
                kpts, scores = model.postprocess([o[i:i + 1] for o in outputs], center, scale)
                out.append((kpts[0], scores[0]))
    return out


def _auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    # Torch absent: onnxruntime.get_available_providers() only reflects
    # what onnxruntime-gpu was *compiled* with, not whether the CUDA
    # runtime is actually installed and loadable -- a machine with
    # onnxruntime-gpu (a core dependency here) but no NVIDIA GPU at all
    # (confirmed on Windows Sandbox, 2026-08-23 -- the installer
    # prototype's base install has no torch) reports "CUDAExecutionProvider"
    # as available regardless, and then fails loudly trying to actually use
    # it (onnxruntime still recovers by falling back to CPU internally, but
    # not before printing a scary "DLL missing"/"FAIL" stderr block that
    # looks like a crash). Without torch there's no reliable way to confirm
    # CUDA is actually usable, so default to CPU; callers who know better
    # can still pass device="cuda" explicitly. See backends_rtmdet.py.
    return "cpu"


# Known model configs: name → (url, input_size HxW, backend_class, conf_scale)
#
# conf_scale normalises per-keypoint confidence values before they are written to
# pose_observations, so the C++ UKF's measurement_noise_std = base_noise/confidence
# formula yields comparable effective noise regardless of model backend.
#
# RTMPose outputs SimCC logit scores (typically 3–8 for clearly-visible joints).
# ViTPose outputs normalised heatmap peak values (0–1).  A conf_scale of ~5 on
# ViTPose brings its effective pixel noise into the same ballpark as RTMPose.
_KNOWN_MODELS: dict[str, tuple[str, tuple[int, int], str, float]] = {
    "rtmpose-l-133kp": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
        "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
        (288, 384),
        "rtmpose",
        1.0,  # SimCC logits already in tracker-compatible range
    ),
    "vitpose-l-133kp": (
        "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/"
        "wholebody/vitpose-l-wholebody.onnx",
        (192, 256),
        "vitpose",
        5.0,  # heatmap peaks [0,1] → scale to match RTMPose logit range
    ),
}


@register_estimator
class RTMPoseEstimator:
    name = "rtmpose-l-133kp"

    def __init__(
        self,
        model_name: str = "rtmpose-l-133kp",
        device: str | None = None,
        backend: str = "onnxruntime",
    ) -> None:
        if not _RTMLIB_AVAILABLE:
            raise ImportError(
                "rtmlib is required for RTMPoseEstimator. "
                "Install from the rtmlib repository."
            )
        if model_name not in _KNOWN_MODELS:
            raise ValueError(
                f"Unknown model {model_name!r}. Known: {list(_KNOWN_MODELS)}"
            )
        url, input_size_hw, backend_cls, _conf_scale = _KNOWN_MODELS[model_name]
        self.name = model_name
        self.version = _RTMLIB_VERSION
        self.input_size = input_size_hw        # (height, width)
        cls = _ViTPose if backend_cls == "vitpose" else _RTMPose
        self.device = device or _auto_device()
        self._model = construct_with_corrupt_checkpoint_retry(
            lambda: cls(
                url,
                model_input_size=(input_size_hw[0], input_size_hw[1]),
                to_openpose=False,
                backend=backend,
                device=self.device,
            ),
            checkpoint_url=url,
        )

    def estimate(
        self,
        frame: np.ndarray,
        detections: list[PersonDetection],
    ) -> list[PoseResult]:
        return self.estimate_batch([(frame, detections)])[0]

    def estimate_batch(
        self,
        items: Sequence[tuple[np.ndarray, list[PersonDetection]]],
    ) -> list[list[PoseResult]]:
        jobs = [
            (frame, np.array([d.bbox[0] - d.bbox[2] / 2, d.bbox[1] - d.bbox[3] / 2,
                              d.bbox[0] + d.bbox[2] / 2, d.bbox[1] + d.bbox[3] / 2], dtype=np.float32))
            for frame, detections in items for d in detections
        ]
        decoded = iter(run_rtmlib_batch(self._model, jobs))
        results: list[list[PoseResult]] = []
        for _, detections in items:
            row = []
            for det in detections:
                kp, sc = next(decoded)
                row.append(PoseResult(track_id=det.track_id, keypoints=np.concatenate(
                    [kp.astype(np.float32), sc[:, np.newaxis].astype(np.float32)], axis=1)))
            results.append(row)
        return results

    def options(self) -> dict:
        return {"model": self.name, "device": self.device, "max_batch": MAX_BATCH}
