# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""kp_gt.tracker_picks: tracing a tracking run's used observations back to the candidates they came from."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from posetrak.detection.kp_candidates import encode_kp_candidates_blob

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "gt"))
import kp_gt  # noqa: E402


def make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE tracking_runs (id TEXT, observation_sequence_id TEXT, active_camera_ids TEXT, marker_names TEXT);
        CREATE TABLE camera_instances (id TEXT, label TEXT);
        CREATE TABLE pose_observation_sequences (id TEXT, shot_id TEXT);
        CREATE TABLE pose_sequence_keypoints (sequence_id TEXT, obs_source TEXT, keypoint_idx INT, name TEXT);
        CREATE TABLE pose_observations (sequence_id TEXT, camera_instance_id TEXT, video_frame INT, timestamp_s REAL,
                                        person_id INT, source TEXT, kp_blob BLOB);
        CREATE TABLE tracking_results (run_id TEXT, person_id INT, tracker_step INT, is_smoothed INT, timestamp_s REAL);
        CREATE TABLE tracking_obs_results (run_id TEXT, person_id INT, tracker_step INT, obs_blob BLOB);
    """)
    conn.execute("INSERT INTO tracking_runs VALUES ('run1', 'seq1', ?, ?)", (json.dumps(["L1"]), json.dumps(["A", "B"])))
    conn.execute("INSERT INTO camera_instances VALUES ('cam1', 'L1')")
    conn.execute("INSERT INTO pose_observation_sequences VALUES ('seq1', 'shot1')")
    conn.executemany("INSERT INTO pose_sequence_keypoints VALUES ('seq1', 'kp_candidates', ?, ?)",
                     [(0, "soma77:A"), (1, "soma77:B")])
    # keypoint 0 has a strong maximum at x=100 and a weaker one at x=110; keypoint 1 has one
    blob = encode_kp_candidates_blob([0, 0, 1], [100, 110, 50], [100, 100, 50], [0.9, 0.5, 0.8], [0, 1, 0], [1, 2, 3])
    conn.execute("INSERT INTO pose_observations VALUES ('seq1', 'cam1', 10, 1.0, 0, 'kp_candidates', ?)", (blob,))
    conn.execute("INSERT INTO tracking_results VALUES ('run1', 0, 0, 0, 1.001)")
    # [actual_x, actual_y, pred_x, pred_y, mahal, used, is_outlier, mode] for markers A and B
    obs = np.array([[[110.0, 100.0, 108, 100, 1, 1, 0, 0], [50.0, 50.0, 50, 50, 1, 0, 0, 0]]], "<f4")
    conn.execute("INSERT INTO tracking_obs_results VALUES ('run1', 0, 0, ?)", (obs.tobytes(),))
    return conn


def test_the_used_observation_is_traced_to_its_candidate_and_compared_with_the_strongest(monkeypatch):
    state = SimpleNamespace(K=np.eye(3), K_orig=np.eye(3), dist=np.zeros((1, 4)), fisheye=False)
    monkeypatch.setattr("posetrak.calibration.session_cameras.load_camera_states", lambda conn, shot: {"cam1": state})
    tracker, strongest, counts = kp_gt.tracker_picks(make_db(), "run", ["cam1"])
    # marker B's observation is not used (an inlier flag of 0), so only keypoint 0 has a pick
    assert counts == {"used": 1, "matched": 1, "unmatched": 0}
    assert (tracker.camera.tolist(), tracker.frame.tolist(), tracker.keypoint.tolist()) == ([0], [10], [0])
    assert tracker.x.tolist() == [110.0]         # the weaker maximum the tracker took
    assert strongest.x.tolist() == [100.0]       # the strongest maximum of the same keypoint-frame
    assert strongest.keypoint.tolist() == [0]
