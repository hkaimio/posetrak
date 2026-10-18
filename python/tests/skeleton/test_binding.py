# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sqlite3

import pytest

from posetrak.detection.pose_models import HAND21, SOMA77
from posetrak.skeleton import compose
from posetrak.skeleton.binding import missing_pose_models


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE pose_sequence_keypoints (sequence_id TEXT, obs_source TEXT, keypoint_idx INT,"
              " name TEXT, source TEXT)")
    return c


def _manifest(conn, sequence_id, obs_source, rows):
    conn.executemany("INSERT INTO pose_sequence_keypoints VALUES (?, ?, ?, ?, ?)",
                     [(sequence_id, obs_source, i, n, s) for i, n, s in rows])


COCO_ONLY = compose.compose_default("male", ["coco133", "hand21"])
WITH_SOMA = compose.compose_default("male")


def test_a_soma_sequence_needs_the_soma_markers(conn):
    _manifest(conn, "s", "body", SOMA77.manifest_rows())
    assert missing_pose_models(conn, "s", COCO_ONLY) == ["soma77"]
    assert missing_pose_models(conn, "s", WITH_SOMA) is None


def test_a_bare_manifest_name_binds_the_qualified_landmark(conn):
    _manifest(conn, "s", "body", [(i, n, "soma77") for i, n in enumerate(SOMA77.keypoint_names)])
    assert missing_pose_models(conn, "s", WITH_SOMA) is None


def test_a_coco_sequence_binds_a_skeleton_with_keypoint_indices_only(conn):
    _manifest(conn, "s", "body", [(i, f"coco133:k{i}", "coco133") for i in range(133)])
    assert missing_pose_models(conn, "s", COCO_ONLY) is None
    assert missing_pose_models(conn, "s", compose.add_pose_model(COCO_ONLY, "soma77")[0]) is None


def test_hand_rows_bind_through_their_side(conn):
    _manifest(conn, "s", "hand_l", HAND21.manifest_rows("left"))
    assert missing_pose_models(conn, "s", COCO_ONLY) is None
    assert missing_pose_models(conn, "s", WITH_SOMA) is None


def test_a_sequence_without_a_manifest_is_left_to_the_tracker(conn):
    assert missing_pose_models(conn, "s", COCO_ONLY) is None
