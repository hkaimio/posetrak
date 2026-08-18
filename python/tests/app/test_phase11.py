# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Phase 11: timeline status data plumbing.

Covers `app.pose.timeline_status` (axis-1 edit-state classification +
cross-camera inlier counts), the `PoseModel.tree_groups` partition added to
`app.pose.kp_models`, and the `clear_single_keypoint_edit` DB helper needed
to un-mark a timeline keyframe.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.pose.timeline_status import (
    STATUS_BLUE,
    STATUS_GREEN,
    STATUS_GREY,
    STATUS_YELLOW,
    compute_inlier_camera_counts,
    read_timeline_status,
)

_N_KP = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _enc(rows: list[tuple[float, float, float]]) -> bytes:
    kp = np.array(rows, dtype=np.float32)
    return kp.tobytes()


@pytest.fixture()
def status_db(tmp_path):
    """Minimal session DB: one camera, a few frames covering each status case."""
    from posetrak.db.db import create_session

    conn = create_session(tmp_path / "status.db")
    conn.execute("PRAGMA foreign_keys = OFF")

    conn.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES ('s1', '2026-01-01T00:00:00Z')")
    conn.execute("INSERT INTO captures (id, session_id, capture_number) VALUES ('sh1', 's1', 1)")
    conn.execute("INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('ci1', 'cm1', 'A')")
    conn.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,"
        " first_video_frame, last_video_frame, actual_fps)"
        " VALUES ('sv1', 'sh1', 'ci1', '/dev/null', 0, 100, 30.0)"
    )
    conn.execute(
        "INSERT INTO pose_observation_sequences"
        " (id, shot_id, sync_config_id, time_start_s, time_end_s, detection_run_id, pixels_are_undistorted)"
        " VALUES ('seq1', 'sh1', 'sc1', 0.0, 2.0, 'run1', 0)"
    )
    conn.execute("INSERT INTO sequence_persons (sequence_id, person_id, person_name)"
                 " VALUES ('seq1', 0, 'Alice')")

    # Frame 1: all 4 kp inliers, no edits, no segmentation data → all GREEN.
    conn.execute(
        "INSERT INTO pose_observations"
        " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, kp_blob)"
        " VALUES ('seq1', 'ci1', 1, 0.033, 0, ?)",
        (_enc([(1, 1, 0.9)] * _N_KP),),
    )
    # Frame 2: kp 0 confident, kp 1 zero-confidence (no usable detection) → GREEN, GREY.
    conn.execute(
        "INSERT INTO pose_observations"
        " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, kp_blob)"
        " VALUES ('seq1', 'ci1', 2, 0.067, 0, ?)",
        (_enc([(2, 2, 0.9), (2, 2, 0.0), (2, 2, 0.9), (2, 2, 0.9)]),),
    )
    conn.commit()
    yield conn
    conn.close()


def _add_edit(conn, video_frame: int, kp_idx: int, is_outlier: bool, n_kp: int = _N_KP) -> None:
    from app.pose.db_cache import update_single_keypoint_edit
    update_single_keypoint_edit(
        conn, "seq1", "ci1", video_frame, kp_idx, 9.0, 9.0, is_outlier=is_outlier,
    )


# ---------------------------------------------------------------------------
# read_timeline_status
# ---------------------------------------------------------------------------

def test_all_inlier_frame_is_green(status_db):
    result = read_timeline_status(status_db, "seq1", "ci1")
    assert list(result[1]) == [STATUS_GREEN] * _N_KP


def test_zero_confidence_slot_is_grey(status_db):
    result = read_timeline_status(status_db, "seq1", "ci1")
    assert result[2][0] == STATUS_GREEN
    assert result[2][1] == STATUS_GREY


def test_disabled_edit_is_grey(status_db):
    _add_edit(status_db, 1, kp_idx=0, is_outlier=True)
    result = read_timeline_status(status_db, "seq1", "ci1")
    assert result[1][0] == STATUS_GREY
    # Untouched slots on the same frame are unaffected.
    assert result[1][1] == STATUS_GREEN


def test_moved_edit_is_blue(status_db):
    _add_edit(status_db, 1, kp_idx=0, is_outlier=False)
    result = read_timeline_status(status_db, "seq1", "ci1")
    assert result[1][0] == STATUS_BLUE
    assert result[1][1] == STATUS_GREEN


def test_ghost_frame_edit_only_still_reports_status(status_db):
    """An edit on a frame with no pose_observations row still appears in the result."""
    _add_edit(status_db, 50, kp_idx=2, is_outlier=False)
    result = read_timeline_status(status_db, "seq1", "ci1")
    assert 50 in result
    assert result[50][2] == STATUS_BLUE
    # Untouched slots on a ghost frame have no detection at all → grey.
    assert result[50][0] == STATUS_GREY


def test_no_observations_and_no_edits_returns_empty(tmp_path):
    from posetrak.db.db import create_session
    conn = create_session(tmp_path / "empty.db")
    conn.execute("PRAGMA foreign_keys = OFF")
    result = read_timeline_status(conn, "seq-none", "ci-none")
    assert result == {}
    conn.close()


# ---------------------------------------------------------------------------
# Segmentation (axis-1 yellow/green split)
# ---------------------------------------------------------------------------

@pytest.fixture()
def seg_db(status_db):
    """Adds a seg_quality_run + per-keypoint quality for frame 1: kp0 outside, kp1 inside."""
    status_db.execute(
        "INSERT INTO seg_quality_runs (id, shot_id, time_start_s, time_end_s, created_at)"
        " VALUES ('sq1', 'shot1', 0.0, 1e9, '2026-01-01T00:00:00Z')"
    )
    quality = np.array([0.0, 1.0, 0.5, -1.0], dtype=np.float32)
    status_db.execute(
        "INSERT INTO keypoint_obs_quality (seg_run_id, shot_video_id, video_frame, track_id, quality_blob)"
        " VALUES ('sq1', 'sv1', 1, 7, ?)",
        (quality.tobytes(),),
    )
    status_db.commit()
    return status_db


def test_segmentation_outside_marks_yellow(seg_db):
    result = read_timeline_status(
        seg_db, "seq1", "ci1",
        shot_video_id="sv1", seg_run_id="sq1", track_id_by_frame={1: 7, 2: 7},
    )
    assert result[1][0] == STATUS_YELLOW   # quality 0.0 → outside
    assert result[1][1] == STATUS_GREEN    # quality 1.0 → inside
    assert result[1][2] == STATUS_GREEN    # quality 0.5 → boundary, treated as inside
    assert result[1][3] == STATUS_GREEN    # quality -1.0 → unavailable, defaults to green


def test_segmentation_yellow_overridden_by_edit(seg_db):
    """Editing a keypoint takes precedence over its segmentation-derived color."""
    _add_edit(seg_db, 1, kp_idx=0, is_outlier=False)
    result = read_timeline_status(
        seg_db, "seq1", "ci1",
        shot_video_id="sv1", seg_run_id="sq1", track_id_by_frame={1: 7, 2: 7},
    )
    assert result[1][0] == STATUS_BLUE


def test_missing_seg_run_defaults_to_green(status_db):
    """Without seg_run_id/shot_video_id/track map, no yellow is ever produced."""
    result = read_timeline_status(status_db, "seq1", "ci1")
    assert STATUS_YELLOW not in result[1]


def test_frame_without_track_mapping_defaults_to_green(seg_db):
    """track_id_by_frame missing an entry for a frame → segmentation skipped for it."""
    result = read_timeline_status(
        seg_db, "seq1", "ci1",
        shot_video_id="sv1", seg_run_id="sq1", track_id_by_frame={2: 7},  # no entry for frame 1
    )
    assert STATUS_YELLOW not in result[1]


# ---------------------------------------------------------------------------
# compute_inlier_camera_counts
# ---------------------------------------------------------------------------

def test_inlier_counts_across_cameras():
    def kp(conf_list):
        return np.array([[0.0, 0.0, c] for c in conf_list], dtype=np.float32)

    obs_kp_by_camera = {
        "ci1": {10: kp([0.9, 0.0, 0.9])},
        "ci2": {10: kp([0.9, 0.9, 0.0])},
        "ci3": {10: kp([0.0, 0.9, 0.9])},
    }
    counts = compute_inlier_camera_counts(obs_kp_by_camera)
    assert list(counts[10]) == [2, 2, 2]


def test_inlier_counts_frame_only_in_one_camera():
    def kp(conf_list):
        return np.array([[0.0, 0.0, c] for c in conf_list], dtype=np.float32)

    obs_kp_by_camera = {
        "ci1": {5: kp([0.9, 0.9])},
        "ci2": {},
    }
    counts = compute_inlier_camera_counts(obs_kp_by_camera)
    assert list(counts[5]) == [1, 1]


def test_inlier_counts_empty_input():
    assert compute_inlier_camera_counts({}) == {}


# ---------------------------------------------------------------------------
# PoseModel.tree_groups (kp_models.py)
# ---------------------------------------------------------------------------

def test_coco17_tree_groups_partition_all_indices():
    from app.pose.kp_models import COCO17
    seen: set[int] = set()
    for name in COCO17.tree_groups:
        idx = COCO17.group_indices(name)
        assert not (seen & idx), f"{name} overlaps"
        seen |= idx
    assert seen == COCO17.all_indices


def test_coco133_tree_groups_partition_all_indices():
    from app.pose.kp_models import COCO133
    seen: set[int] = set()
    for name in COCO133.tree_groups:
        idx = COCO133.group_indices(name)
        assert not (seen & idx), f"{name} overlaps"
        seen |= idx
    assert seen == COCO133.all_indices


def test_default_pose_model_has_empty_tree_groups():
    from app.pose.kp_models import PoseModel
    m = PoseModel(model_id="x", names=("a",), groups={})
    assert m.tree_groups == ()


# ---------------------------------------------------------------------------
# clear_single_keypoint_edit
# ---------------------------------------------------------------------------

def test_clear_single_keypoint_edit_removes_bit(status_db):
    from app.pose.db_cache import clear_single_keypoint_edit
    _add_edit(status_db, 1, kp_idx=0, is_outlier=False)
    _add_edit(status_db, 1, kp_idx=1, is_outlier=False)

    clear_single_keypoint_edit(status_db, "seq1", "ci1", 1, 0)

    row = status_db.execute(
        "SELECT kp_mask FROM pose_observation_edits"
        " WHERE sequence_id='seq1' AND camera_instance_id='ci1' AND video_frame=1"
    ).fetchone()
    assert row is not None
    mask = bytes(row["kp_mask"])
    assert not ((mask[0] >> 0) & 1)  # kp 0 cleared
    assert (mask[0] >> 1) & 1        # kp 1 still set


def test_clear_single_keypoint_edit_deletes_row_when_empty(status_db):
    from app.pose.db_cache import clear_single_keypoint_edit
    _add_edit(status_db, 1, kp_idx=0, is_outlier=False)

    clear_single_keypoint_edit(status_db, "seq1", "ci1", 1, 0)

    row = status_db.execute(
        "SELECT * FROM pose_observation_edits"
        " WHERE sequence_id='seq1' AND camera_instance_id='ci1' AND video_frame=1"
    ).fetchone()
    assert row is None


def test_clear_single_keypoint_edit_noop_without_edit_row(status_db):
    from app.pose.db_cache import clear_single_keypoint_edit
    # Should not raise even though frame 1 has no edit row.
    clear_single_keypoint_edit(status_db, "seq1", "ci1", 1, 0)
    row = status_db.execute(
        "SELECT * FROM pose_observation_edits WHERE sequence_id='seq1'"
    ).fetchone()
    assert row is None
