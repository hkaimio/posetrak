"""backends.py — Detector and pose estimator protocol + registry.

All pixel coordinates use original (distorted) video space.  Undistortion
is applied by the tracker at load time, not here.

Implementations have moved to posetrak.detection.backends; this module
re-exports everything for backwards compatibility.
"""
from __future__ import annotations

from posetrak.detection.backends import (  # noqa: F401
    PersonDetection,
    PoseResult,
    PersonDetector,
    PoseEstimator,
    register_detector,
    register_estimator,
    available_detectors,
    available_estimators,
    get_detector,
    get_estimator,
    _detector_registry,
    _estimator_registry,
)
