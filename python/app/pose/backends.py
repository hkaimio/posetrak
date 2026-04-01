"""backends.py — Detector and pose estimator protocol + registry.

All pixel coordinates use original (distorted) video space.  Undistortion
is applied by the tracker at load time, not here.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import numpy as np


@dataclass
class PersonDetection:
    track_id: int
    bbox: np.ndarray   # float32[4]: x, y, w, h  centre-format, distorted px
    confidence: float


@dataclass
class PoseResult:
    track_id: int
    keypoints: np.ndarray  # float32[N, 3]: x, y, conf  distorted px


@runtime_checkable
class PersonDetector(Protocol):
    name: str     # e.g. "yolo11x"
    version: str

    def detect_and_track(
        self,
        frame: np.ndarray,   # uint8 HxWx3 BGR, original distorted frame
        frame_idx: int,
    ) -> list[PersonDetection]: ...

    def reset_tracker(self) -> None: ...


@runtime_checkable
class PoseEstimator(Protocol):
    name: str
    version: str
    input_size: tuple[int, int]   # (height, width) of model input

    def estimate(
        self,
        frame: np.ndarray,                  # original distorted full frame
        detections: list[PersonDetection],   # bboxes in distorted px
    ) -> list[PoseResult]: ...


_detector_registry: dict[str, type] = {}
_estimator_registry: dict[str, type] = {}


def register_detector(cls):
    _detector_registry[cls.name] = cls
    return cls


def register_estimator(cls):
    _estimator_registry[cls.name] = cls
    return cls


def available_detectors() -> list[str]:
    return list(_detector_registry)


def available_estimators() -> list[str]:
    return list(_estimator_registry)


def get_detector(name: str) -> type:
    if name not in _detector_registry:
        raise KeyError(f"Unknown detector {name!r}. Available: {available_detectors()}")
    return _detector_registry[name]


def get_estimator(name: str) -> type:
    if name not in _estimator_registry:
        raise KeyError(f"Unknown estimator {name!r}. Available: {available_estimators()}")
    return _estimator_registry[name]
