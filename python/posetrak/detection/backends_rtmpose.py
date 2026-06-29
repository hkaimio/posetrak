"""backends_rtmpose.py — RTMPose estimator backend."""
from __future__ import annotations

import os
import sys
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

try:
    from rtmlib.tools.pose_estimation import RTMPose as _RTMPose
    from rtmlib.tools.pose_estimation.vitpose import ViTPose as _ViTPose
    import rtmlib
    _RTMLIB_VERSION = getattr(rtmlib, "__version__", "unknown")
    _RTMLIB_AVAILABLE = True
except ImportError:
    _RTMLIB_AVAILABLE = False
    _RTMLIB_VERSION = ""

from posetrak.detection.backends import PersonDetection, PoseResult, register_estimator


def _auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    # Torch absent or CPU-only: check onnxruntime directly.
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "cuda"
    except Exception:
        pass
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
        self._model = cls(
            url,
            model_input_size=(input_size_hw[0], input_size_hw[1]),
            to_openpose=False,
            backend=backend,
            device=device or _auto_device(),
        )

    def estimate(
        self,
        frame: np.ndarray,
        detections: list[PersonDetection],
    ) -> list[PoseResult]:
        if not detections:
            return []

        # Convert xywh (centre) → xyxy for rtmlib
        bboxes_xyxy = np.array([
            [d.bbox[0] - d.bbox[2] / 2,
             d.bbox[1] - d.bbox[3] / 2,
             d.bbox[0] + d.bbox[2] / 2,
             d.bbox[1] + d.bbox[3] / 2]
            for d in detections
        ], dtype=np.float32)

        keypoints_all, scores_all = self._model(frame, bboxes_xyxy)
        # keypoints_all: [N, n_kp, 2],  scores_all: [N, n_kp]

        results: list[PoseResult] = []
        for det, kp, sc in zip(detections, keypoints_all, scores_all):
            kp_with_conf = np.concatenate(
                [kp.astype(np.float32), sc[:, np.newaxis].astype(np.float32)],
                axis=1,
            )  # float32[n_kp, 3]
            results.append(PoseResult(track_id=det.track_id, keypoints=kp_with_conf))
        return results
