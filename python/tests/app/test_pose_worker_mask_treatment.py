# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for PoseWorker's mask-treatment wiring (_estimate_per_person,
pose_worker.py) -- Phase 1 of
docs/roadmap/features/segmentation-pose-treatment/segmentation-pose-treatment-design.md.

Uses a fake estimator so no rtmlib checkpoint load is required, matching
test_pose_worker_hand_refinement.py's pattern for this same module.
"""
from __future__ import annotations

import numpy as np

from app.pose.pose_worker import _estimate_per_person
from posetrak.detection.backends import PersonDetection, PoseResult


class _FakeEstimator:
    """Records every (frame, bboxes) it was called with; returns one
    zero-keypoint PoseResult per detection passed in."""

    def __init__(self):
        self.calls: list[tuple[np.ndarray, list]] = []

    def estimate(self, frame, detections):
        self.calls.append((frame, list(detections)))
        return [
            PoseResult(track_id=det.track_id, keypoints=np.zeros((1, 3), dtype=np.float32))
            for det in detections
        ]


def _frame(h=60, w=60):
    return np.full((h, w, 3), 100, dtype=np.uint8)


def _mask_two_people(h=60, w=60):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, :30] = 1
    mask[:, 30:] = 2
    return mask


def test_treatment_on_calls_estimate_once_per_person_with_treated_frame():
    estimator = _FakeEstimator()
    frame = _frame()
    mask = _mask_two_people()
    detections = [
        PersonDetection(track_id=1, bbox=np.array([15, 30, 30, 60], dtype=np.float32), confidence=1.0),
        PersonDetection(track_id=2, bbox=np.array([45, 30, 30, 60], dtype=np.float32), confidence=1.0),
    ]

    results = _estimate_per_person(estimator, frame, mask, detections, apply_mask_treatment=True)

    assert len(estimator.calls) == 2  # one call per person, not one batched call
    for (_frame_arg, dets), det in zip(estimator.calls, detections):
        assert len(dets) == 1
        assert dets[0].track_id == det.track_id
    # Each call's frame must actually differ from the raw frame (something
    # was suppressed) and must differ from each other (each person sees a
    # different treatment target).
    frame1, frame2 = estimator.calls[0][0], estimator.calls[1][0]
    assert not np.array_equal(frame1, frame)
    assert not np.array_equal(frame2, frame)
    assert not np.array_equal(frame1, frame2)
    assert {r.track_id for r in results} == {1, 2}


def test_treatment_off_falls_back_to_one_batched_call():
    estimator = _FakeEstimator()
    frame = _frame()
    mask = _mask_two_people()
    detections = [
        PersonDetection(track_id=1, bbox=np.array([15, 30, 30, 60], dtype=np.float32), confidence=1.0),
        PersonDetection(track_id=2, bbox=np.array([45, 30, 30, 60], dtype=np.float32), confidence=1.0),
    ]

    results = _estimate_per_person(estimator, frame, mask, detections, apply_mask_treatment=False)

    assert len(estimator.calls) == 1  # today's pre-existing behaviour, unchanged
    assert np.array_equal(estimator.calls[0][0], frame)  # untreated
    assert len(estimator.calls[0][1]) == 2
    assert {r.track_id for r in results} == {1, 2}
