# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""backends_rtmdet.py — YOLOX person detector backend (rtmlib, Apache-2.0).

Replaces the old ultralytics-based YOLOv11Detector (see git history) as
part of removing the AGPL-3.0 ``ultralytics`` dependency -- see
docs/license-analysis.md. rtmlib's YOLOX wraps a Megvii YOLOX ONNX model
under Apache-2.0, unrelated to (and legally independent of) Ultralytics'
own YOLO family despite the similar name. The specific model URLs below
match rtmlib's own bundled "Body" solution tiers (rtmlib/tools/solution/
body.py), so weights are shared with anything else in the app that
already uses rtmlib's defaults.

rtmlib's detector classes return bounding boxes only -- no per-detection
confidence (see YOLOX.postprocess in rtmlib/tools/object_detection/
yolox.py: it computes scores internally for NMS filtering but never
returns them). PersonDetection.confidence is therefore reported as the
configured score threshold rather than a real per-box value. The only
current consumer (db_cache.py) uses it for display only -- the UKF's
actual noise-weighting comes from PoseResult's per-keypoint confidence
in backends_rtmpose.py, untouched by this -- so this is a display-only
simplification, not a tracking-accuracy regression.

rtmlib has no persistent multi-object tracker exposed publicly
(rtmlib.tools.solution.pose_tracker.PoseTracker has a greedy-IoU one,
but it's private to that class's own frame loop) -- detect_and_track()/
reset_tracker() below reimplement the same greedy IoU-matching approach
independently, at the same sophistication level. Note this is simpler
than the ByteTrack/BoT-SORT tracker ultralytics' YOLO.track() used to
provide: a person fully occluded for even one frame gets a new track ID
here rather than being re-identified. This matters less than it used to
now that the segmentation (Cutie) pipeline -- which doesn't depend on
this per-frame identity tracking at all -- is the primary path for
exactly the heavy-occlusion multi-person scenes where that would bite.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# On Windows, onnxruntime-gpu's CUDA provider DLL looks for CUDA libraries
# (cublasLt64_12.dll etc.) on the system PATH. PyTorch bundles those DLLs
# in its own lib/ directory, so we register that directory before rtmlib
# (and thus onnxruntime) is first imported. See backends_rtmpose.py.
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
    # urllib download) still honours alongside the OS store.
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

try:
    from rtmlib.tools.object_detection import YOLOX as _YOLOX
    import rtmlib
    _RTMLIB_VERSION = getattr(rtmlib, "__version__", "unknown")
    _RTMLIB_AVAILABLE = True
except ImportError:
    _RTMLIB_AVAILABLE = False
    _RTMLIB_VERSION = ""

from posetrak.detection.backends import (
    PersonDetection,
    construct_with_corrupt_checkpoint_retry,
    register_detector,
)


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
    # can still pass device="cuda" explicitly.
    return "cpu"


# name -> (onnx model url, model_input_size HxW). Matches rtmlib's own
# Body.MODE detector tiers (performance/balanced/lightweight).
_KNOWN_MODELS: dict[str, tuple[str, tuple[int, int]]] = {
    "yolox-x": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "yolox_x_8xb8-300e_humanart-a39d44ed.zip",
        (640, 640),
    ),
    "yolox-m": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        (640, 640),
    ),
    "yolox-tiny": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        (416, 416),
    ),
}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@register_detector
class YOLOXDetector:
    name = "yolox-x"

    def __init__(
        self,
        model_name: str = "yolox-x",
        device: str | None = None,
        conf: float = 0.3,
        iou_thr: float = 0.3,
        tracker_config: str | None = None,   # kept for call-site parity with
        # the old YOLOv11Detector's kwarg; unused -- rtmlib has no tracker configs.
    ) -> None:
        if not _RTMLIB_AVAILABLE:
            raise ImportError(
                "rtmlib is required for YOLOXDetector. "
                "Install with: pip install rtmlib"
            )
        if model_name not in _KNOWN_MODELS:
            raise ValueError(
                f"Unknown model {model_name!r}. Known: {list(_KNOWN_MODELS)}"
            )
        url, input_size = _KNOWN_MODELS[model_name]
        self.name = model_name
        self.version = _RTMLIB_VERSION
        self._conf = conf
        self._iou_thr = iou_thr
        self.device = device or _auto_device()
        self._model = construct_with_corrupt_checkpoint_retry(
            lambda: _YOLOX(
                url,
                model_input_size=input_size,
                mode="human",
                score_thr=conf,
                backend="onnxruntime",
                device=self.device,
            ),
            checkpoint_url=url,
        )
        self._next_id = 0
        self._prev_boxes: list[np.ndarray] = []
        self._prev_ids: list[int] = []

    def detect_and_track(
        self, frame: np.ndarray, frame_idx: int
    ) -> list[PersonDetection]:
        boxes_xyxy = self._model(frame)   # (N, 4) float xyxy, may be empty

        cur_boxes: list[np.ndarray] = []
        cur_ids: list[int] = []
        detections: list[PersonDetection] = []

        avail_prev = list(range(len(self._prev_boxes)))
        for box in boxes_xyxy:
            best_j, best_iou = -1, 0.0
            for j in avail_prev:
                score = _iou(box, self._prev_boxes[j])
                if score > best_iou:
                    best_iou, best_j = score, j
            if best_j >= 0 and best_iou >= self._iou_thr:
                track_id = self._prev_ids[best_j]
                avail_prev.remove(best_j)
            else:
                track_id = self._next_id
                self._next_id += 1

            cur_boxes.append(box)
            cur_ids.append(track_id)

            x1, y1, x2, y2 = box
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            detections.append(PersonDetection(
                track_id=int(track_id),
                bbox=np.array([cx, cy, w, h], dtype=np.float32),
                confidence=float(self._conf),
            ))

        self._prev_boxes = cur_boxes
        self._prev_ids = cur_ids
        return detections

    def reset_tracker(self) -> None:
        """Reset tracker state between cameras without reloading model weights."""
        self._next_id = 0
        self._prev_boxes = []
        self._prev_ids = []
