"""backends_rtmpose.py — RTMPose estimator backend."""
from __future__ import annotations

import numpy as np

try:
    from rtmlib.tools.pose_estimation import RTMPose as _RTMPose
    from rtmlib.tools.pose_estimation.vitpose import ViTPose as _ViTPose
    import rtmlib
    _RTMLIB_VERSION = getattr(rtmlib, "__version__", "unknown")
    _RTMLIB_AVAILABLE = True
except ImportError:
    _RTMLIB_AVAILABLE = False
    _RTMLIB_VERSION = ""

from app.pose.backends import PersonDetection, PoseResult, register_estimator

# Known model configs: name → (url, input_size HxW, backend_class)
# RTMPose uses SimCC heads (two outputs: simcc_x, simcc_y).
# ViTPose uses heatmap heads (one output) — requires the ViTPose backend class.
_KNOWN_MODELS: dict[str, tuple[str, tuple[int, int], str]] = {
    "rtmpose-l-133kp": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
        "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
        (288, 384),
        "rtmpose",
    ),
    "vitpose-l-133kp": (
        "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/"
        "wholebody/vitpose-l-wholebody.onnx",
        (192, 256),
        "vitpose",
    ),
}


@register_estimator
class RTMPoseEstimator:
    name = "rtmpose-l-133kp"

    def __init__(
        self,
        model_name: str = "rtmpose-l-133kp",
        device: str = "cuda",
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
        url, input_size_hw, backend_cls = _KNOWN_MODELS[model_name]
        self.name = model_name
        self.version = _RTMLIB_VERSION
        self.input_size = input_size_hw        # (height, width)
        cls = _ViTPose if backend_cls == "vitpose" else _RTMPose
        self._model = cls(
            url,
            model_input_size=(input_size_hw[0], input_size_hw[1]),
            to_openpose=False,
            backend=backend,
            device=device,
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
