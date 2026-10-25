# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""kp_gt.reproduction: how much of one candidate set another reproduces within a tolerance."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "gt"))
import kp_gt  # noqa: E402


def candidates(cameras, rows):
    """rows: (camera, frame, keypoint, x, y)."""
    a = np.array(rows, float)
    return kp_gt.Candidates(
        cameras=cameras, camera=a[:, 0].astype(np.int32), frame=a[:, 1].astype(np.int64), keypoint=a[:, 2].astype(np.int32),
        x=a[:, 3], y=a[:, 4], peak=np.full(len(a), 0.5), rank=np.zeros(len(a), np.int32), tracklet=np.full(len(a), -1, np.int64))


def test_a_candidate_is_reproduced_only_for_the_same_camera_frame_and_keypoint_within_the_tolerance():
    reference = candidates(["a", "b"], [(0, 10, 5, 100.0, 200.0), (0, 10, 6, 300.0, 300.0), (1, 10, 5, 50.0, 60.0),
                                         (0, 11, 5, 100.0, 200.0), (0, 99, 5, 1.0, 1.0)])
    other = candidates(["B", "A"], [(1, 10, 5, 100.3, 200.0),      # reference camera 0 is this set's 1; 0.3 px away
                                     (1, 10, 6, 303.0, 300.0),      # 3 px away: not reproduced
                                     (0, 10, 5, 50.0, 60.0),        # reference camera 1 is this set's 0
                                     (1, 11, 6, 100.0, 200.0),      # another keypoint
                                     (1, 12, 5, 7.0, 7.0)])         # a frame the reference lacks
    r = kp_gt.reproduction(reference, other, {0: 1, 1: 0}, 0.5)
    # Frames 10 and 11 of both cameras are common; frame 99 and 12 are not compared.
    assert r["n_reference"] == 4 and r["n_other"] == 4
    assert r["reference_reproduced"] == 0.5          # the first and third reference candidates
    assert r["other_reproduced"] == 0.5
    only = kp_gt.reproduction(reference, other, {0: 1, 1: 0}, 0.5, frames={(1, 10)})
    assert only["n_reference"] == 1 and only["reference_reproduced"] == 1.0
