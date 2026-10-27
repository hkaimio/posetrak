# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Shared loading for the Body Pose scoring scripts (Posetrak environment)."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[2] / "gt"))
import kp_gt  # noqa: E402
from posetrak.calibration.session_cameras import load_camera_states  # noqa: E402

SHOT = "b862ca88-d7d7-43f5-b36a-1fa10c035503"
GROUPS = {
    "trunk": list(range(0, 4)), "head": list(range(4, 11)),
    "arm": [11, 12, 13, 39, 40, 41], "wrist": [14, 42],
    "fingers": list(range(15, 39)) + list(range(43, 67)), "leg": list(range(67, 77)),
}
GROUP_OF = np.full(77, -1)
for _gi, (_, _ks) in enumerate(GROUPS.items()):
    GROUP_OF[_ks] = _gi
GROUP_NAMES = list(GROUPS)
# body keypoints: everything but the fingers (the tracker's body-only strict filter class)
BODY = np.array([k for g, ks in GROUPS.items() if g != "fingers" for k in ks])


def open_db(path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def camera_states(conn):
    return load_camera_states(conn, SHOT)


def undistort(points, st):
    """Distorted video pixels -> pixels of the undistorted image (matrix st.K)."""
    p = np.asarray(points, np.float32).reshape(-1, 1, 2)
    if not len(p):
        return np.empty((0, 2))
    fn = cv2.fisheye.undistortPoints if st.fisheye else cv2.undistortPoints
    return fn(p, st.K_orig, st.dist, None, st.K).reshape(-1, 2).astype(np.float64)


class BodyPose:
    """Body Pose output of all cameras: arrays per crop, looked up by (camera index, video frame, subject)."""

    def __init__(self, out_dir):
        self.cam, self.frame, self.subject, self.time = [], [], [], []
        self.fields = {k: [] for k in ("joints_camera", "keypoints_2d", "valid", "box", "identity", "scale",
                                       "body_pose", "global_orient", "camera_translation")}
        for c in range(5):
            z = np.load(Path(out_dir) / f"bodypose_cam{c}.npz")
            n = len(z["frame"])
            self.cam.append(np.full(n, c))
            self.frame.append(z["frame"])
            self.subject.append(z["subject"])
            self.time.append(z["time"])
            for k in self.fields:
                self.fields[k].append(z[k])
        self.cam, self.frame, self.subject, self.time = (np.concatenate(v) for v in (self.cam, self.frame, self.subject, self.time))
        self.fields = {k: np.concatenate(v) for k, v in self.fields.items()}
        self._key = self.key(self.cam, self.frame, self.subject)
        self._order = np.argsort(self._key)

    @staticmethod
    def key(cam, frame, subject):
        return (np.asarray(cam, np.int64) * 10_000_000 + np.asarray(frame, np.int64)) * 8 + np.asarray(subject, np.int64)

    def find(self, cam, frame, subject):
        """Row index of each (camera, frame, subject), -1 when Body Pose has no crop there."""
        q = self.key(cam, frame, subject)
        ks = self._key[self._order]
        pos = np.clip(np.searchsorted(ks, q), 0, len(ks) - 1)
        return np.where(ks[pos] == q, self._order[pos], -1)


def load_subject(root, subject, range_name="range3"):
    cfg = kp_gt.__dict__  # noqa: F841 (kept for symmetry with the eval script)
    d = Path(root) / range_name
    cand = kp_gt.read_candidates(d / f"candidates_subject{subject}.npz")
    labels = kp_gt.read_labels(d / f"labels_subject{subject}.npz", d / f"units_subject{subject}.json", "51.7-56.8")
    return cand, labels
