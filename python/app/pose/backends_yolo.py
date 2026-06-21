"""backends_yolo.py — YOLOv11 person detector backend."""
from __future__ import annotations

import os
import numpy as np

# Prevent ultralytics from auto-installing packages via pip, which can
# silently downgrade CUDA torch to the CPU build from PyPI.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

try:
    from ultralytics import YOLO as _YOLO
    _ULTRALYTICS_VERSION = __import__("ultralytics").__version__
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    _ULTRALYTICS_VERSION = ""

from app.pose.backends import PersonDetection, register_detector


def _auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@register_detector
class YOLOv11Detector:
    name = "yolo11x"

    def __init__(
        self,
        model_name: str = "yolo11x.pt",
        device: str | None = None,
        conf: float = 0.3,
        tracker_config: str | None = None,
    ) -> None:
        if not _YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is required for YOLOv11Detector. "
                "Install with: pip install ultralytics"
            )
        self.name = model_name.replace(".pt", "")
        self.version = _ULTRALYTICS_VERSION
        self._model_name = model_name
        self._device = device or _auto_device()
        self._conf = conf
        self._tracker_config = tracker_config
        self._model = _YOLO(model_name)

    def detect_and_track(
        self, frame: np.ndarray, frame_idx: int
    ) -> list[PersonDetection]:
        kwargs: dict = dict(
            source=frame,
            persist=True,
            classes=[0],   # person only
            conf=self._conf,
            verbose=False,
            device=self._device,
        )
        if self._tracker_config:
            kwargs["tracker"] = self._tracker_config

        results = self._model.track(**kwargs)
        detections: list[PersonDetection] = []

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return detections

        xywh = boxes.xywh.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)

        for track_id, box, conf in zip(ids, xywh, confs):
            detections.append(PersonDetection(
                track_id=int(track_id),
                bbox=box.astype(np.float32),  # x_centre, y_centre, w, h
                confidence=float(conf),
            ))
        return detections

    def reset_tracker(self) -> None:
        """Reset tracker state between cameras without reloading model weights."""
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            self._model.predictor = None
