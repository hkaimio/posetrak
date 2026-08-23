# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""backends.py — Detector and pose estimator protocol + registry.

All pixel coordinates use original (distorted) video space.  Undistortion
is applied by the tracker at load time, not here.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, TypeVar, runtime_checkable
from urllib.parse import urlparse
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


def _rtmlib_checkpoint_cache_paths(checkpoint_url: str) -> list[Path]:
    """Return the on-disk path(s) rtmlib's download_checkpoint() uses for *url*.

    Mirrors rtmlib.tools.file.download_checkpoint()'s own filename derivation
    (dst_dir / basename(url), plus the .onnx name it extracts .zip sources
    to) so a corrupted/truncated cached file can be identified and removed
    precisely, without touching any other cached checkpoint. Returns an
    empty list if rtmlib isn't importable (caller then has nothing to clean
    up and should just let the original exception propagate).
    """
    try:
        from rtmlib.tools.file import _get_rtmhub_dir
    except ImportError:
        return []
    dst_dir = Path(_get_rtmhub_dir()) / "checkpoints"
    filename = os.path.basename(urlparse(checkpoint_url).path)
    cached_file = dst_dir / filename
    onnx_name = dst_dir / (filename.split(".")[0] + ".onnx")
    return [cached_file, onnx_name] if cached_file != onnx_name else [cached_file]


_T = TypeVar("_T")


def construct_with_corrupt_checkpoint_retry(
    factory: Callable[[], _T], checkpoint_url: str
) -> _T:
    """Call factory(); on failure, clear rtmlib's cached checkpoint(s) and retry once.

    rtmlib's download_checkpoint() (rtmlib/tools/file.py) treats "a file
    already exists at the cache path" as "already downloaded, skip
    re-fetching", and download_url_to_file() never verifies the number of
    bytes actually read matches the server's Content-Length before
    atomically renaming the result into place. A connection that drops
    mid-transfer -- observed on a Windows Sandbox's virtualized network,
    but not specific to it; any flaky connection can do this -- silently
    produces a truncated-but-"complete" cached file, which then fails
    onnxruntime with an InvalidProtobuf error on every single subsequent
    attempt, with no way to recover short of a user manually finding and
    deleting a cache file under their own ~/.cache/rtmlib/. Self-heal
    instead: on any failure constructing the model, delete the specific
    cached file(s) for this checkpoint's URL and retry the exact same
    factory once, so a fresh download replaces the corrupt one.
    """
    try:
        return factory()
    except Exception:
        stale = [p for p in _rtmlib_checkpoint_cache_paths(checkpoint_url) if p.exists()]
        if not stale:
            raise
        for path in stale:
            path.unlink()
        return factory()
