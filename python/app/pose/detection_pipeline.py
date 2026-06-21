"""detection_pipeline.py — Synchronous detection + pose estimation pipeline.

Implementation has moved to posetrak.detection.pipeline; this module
re-exports for backwards compatibility.
"""
from __future__ import annotations

from posetrak.detection.pipeline import (  # noqa: F401
    DetectionPipeline,
    PipelineResult,
    CameraInfo,
    ProgressCallback,
)
