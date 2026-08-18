# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Idea 3 (automated post-edit hand redetection), Phase 2:
db_cache.write_hand_refinement/revert_hand_refinement and the
STATUS_ORANGE ("came from a '.refined' source, not yet human-verified")
timeline status.

See docs/roadmap/features/hand-detection-refinement/hand-detection-refinement-design.md,
"Idea 3" section.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.pose.timeline_status import (
    STATUS_BLUE,
    STATUS_GREEN,
    STATUS_GREY,
    STATUS_ORANGE,
    read_timeline_status,
)

_N_KP = 133


def _enc(kp: np.ndarray) -> bytes:
    return kp.astype(np.float32).tobytes()


def _body_kp(fill: float = 0.9) -> np.ndarray:
    kp = np.zeros((_N_KP, 3), dtype=np.float32)
    kp[:, 2] = fill
    return kp


def _hand_kp(fill: float = 0.8) -> np.ndarray:
    kp = np.zeros((21, 3), dtype=np.float32)
    kp[:, 2] = fill
    return kp


@pytest.fixture()
def hand_db(tmp_path):
    """Minimal session DB: one camera, one sequence, a 133-wide body row per frame."""
    from posetrak.db.db import create_session

    conn = create_session(tmp_path / "hand_status.db")
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

    # Frame 1: 'body' row plus a batch 'hand_l' row -- no refinement yet.
    conn.execute(
        "INSERT INTO pose_observations"
        " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, source, kp_blob)"
        " VALUES ('seq1', 'ci1', 1, 0.033, 0, 'body', ?)",
        (_enc(_body_kp()),),
    )
    conn.execute(
        "INSERT INTO pose_observations"
        " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, source, kp_blob)"
        " VALUES ('seq1', 'ci1', 1, 0.033, 0, 'hand_l', ?)",
        (_enc(_hand_kp(0.6)),),
    )
    conn.commit()
    yield conn
    conn.close()


def test_write_hand_refinement_adds_a_refined_row_without_touching_the_batch_row(hand_db):
    from app.pose.db_cache import write_hand_refinement

    write_hand_refinement(
        hand_db, "seq1", "ci1", 1, 0, timestamp_s=0.033, side="left",
        kp=_hand_kp(0.9), noise_scale=0.2,
    )

    rows = hand_db.execute(
        "SELECT source FROM pose_observations WHERE sequence_id='seq1' AND video_frame=1"
        " ORDER BY source"
    ).fetchall()
    assert [r["source"] for r in rows] == ["body", "hand_l", "hand_l.refined"]


def test_write_hand_refinement_upserts_on_repeated_calls(hand_db):
    from app.pose.db_cache import write_hand_refinement

    write_hand_refinement(
        hand_db, "seq1", "ci1", 1, 0, timestamp_s=0.033, side="left",
        kp=_hand_kp(0.9), noise_scale=0.2,
    )
    write_hand_refinement(
        hand_db, "seq1", "ci1", 1, 0, timestamp_s=0.033, side="left",
        kp=_hand_kp(0.5), noise_scale=0.4,
    )

    rows = hand_db.execute(
        "SELECT kp_blob, noise_scale FROM pose_observations"
        " WHERE sequence_id='seq1' AND video_frame=1 AND source='hand_l.refined'"
    ).fetchall()
    assert len(rows) == 1  # overwrite, not accumulation
    kp = np.frombuffer(bytes(rows[0]["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    assert kp[0, 2] == pytest.approx(0.5)
    assert rows[0]["noise_scale"] == pytest.approx(0.4)


def test_revert_hand_refinement_deletes_only_the_refined_row(hand_db):
    from app.pose.db_cache import revert_hand_refinement, write_hand_refinement

    write_hand_refinement(
        hand_db, "seq1", "ci1", 1, 0, timestamp_s=0.033, side="left",
        kp=_hand_kp(0.9), noise_scale=0.2,
    )
    revert_hand_refinement(hand_db, "seq1", "ci1", 1, 0, side="left")

    rows = hand_db.execute(
        "SELECT source FROM pose_observations WHERE sequence_id='seq1' AND video_frame=1"
        " ORDER BY source"
    ).fetchall()
    assert [r["source"] for r in rows] == ["body", "hand_l"]


def test_revert_hand_refinement_is_a_noop_if_nothing_to_revert(hand_db):
    from app.pose.db_cache import revert_hand_refinement

    revert_hand_refinement(hand_db, "seq1", "ci1", 1, 0, side="right")  # no hand_r.refined row
    rows = hand_db.execute(
        "SELECT source FROM pose_observations WHERE sequence_id='seq1' AND video_frame=1"
    ).fetchall()
    assert len(rows) == 2  # unchanged


# ---------------------------------------------------------------------------
# STATUS_ORANGE via read_timeline_status
# ---------------------------------------------------------------------------

def test_refined_hand_marks_status_orange(hand_db):
    from app.pose.db_cache import write_hand_refinement

    write_hand_refinement(
        hand_db, "seq1", "ci1", 1, 0, timestamp_s=0.033, side="left",
        kp=_hand_kp(0.9), noise_scale=0.2,
    )
    status = read_timeline_status(hand_db, "seq1", "ci1")[1]

    assert all(status[91:112] == STATUS_ORANGE)
    # Body-range and hand_r-range slots are untouched (no hand_r row at all
    # here, so those slots are GREY -- not asserted; just confirm body stays GREEN).
    assert status[0] == STATUS_GREEN


def test_human_edit_still_wins_over_refined_status(hand_db):
    from app.pose.db_cache import update_single_keypoint_edit, write_hand_refinement

    write_hand_refinement(
        hand_db, "seq1", "ci1", 1, 0, timestamp_s=0.033, side="left",
        kp=_hand_kp(0.9), noise_scale=0.2,
    )
    update_single_keypoint_edit(hand_db, "seq1", "ci1", 1, 95, 10.0, 20.0)

    status = read_timeline_status(hand_db, "seq1", "ci1")[1]
    assert status[95] == STATUS_BLUE
    # Neighbouring hand slots, still un-edited, stay ORANGE.
    assert status[96] == STATUS_ORANGE


def test_refined_hand_on_a_ghost_frame_does_not_crash_timeline_status(hand_db):
    """Reproduces a crash seen in real use: placing a wrist/elbow via an edit
    on a frame with no original 'body' detection ("set limb" on a gap), then
    auto-redetection writes only a 'hand_l.refined' row for that frame.  With
    no 'body' row of its own, that frame used to merge to a bare 21-point
    array while every other frame in the camera was 133-wide, and
    read_timeline_status crashed comparing a 133-wide status array against a
    21-wide observation array."""
    from app.pose.db_cache import write_hand_refinement

    # Frame 2 has no 'body' row at all -- only the auto-redetected overlay.
    write_hand_refinement(
        hand_db, "seq1", "ci1", 2, 0, timestamp_s=0.066, side="left",
        kp=_hand_kp(0.9), noise_scale=0.2,
    )

    status = read_timeline_status(hand_db, "seq1", "ci1")[2]
    assert status.shape == (_N_KP,)
    assert all(status[91:112] == STATUS_ORANGE)
    assert status[0] == STATUS_GREY  # no body detection at all on this frame


def test_reverting_refinement_restores_prior_status(hand_db):
    from app.pose.db_cache import revert_hand_refinement, write_hand_refinement

    write_hand_refinement(
        hand_db, "seq1", "ci1", 1, 0, timestamp_s=0.033, side="left",
        kp=_hand_kp(0.9), noise_scale=0.2,
    )
    assert read_timeline_status(hand_db, "seq1", "ci1")[1][95] == STATUS_ORANGE

    revert_hand_refinement(hand_db, "seq1", "ci1", 1, 0, side="left")
    assert read_timeline_status(hand_db, "seq1", "ci1")[1][95] == STATUS_GREEN


# ---------------------------------------------------------------------------
# clear_disabled_hand_edits -- the "auto-detect" half of Idea 3's two-mode
# design: a fresh redetection clears stale disable-edits for that hand's
# fingers, but never touches a deliberate repositioning edit.
# ---------------------------------------------------------------------------

def test_clears_a_disabled_finger_within_the_hand_range(hand_db):
    from app.pose.db_cache import clear_disabled_hand_edits, update_single_keypoint_edit

    update_single_keypoint_edit(hand_db, "seq1", "ci1", 1, 95, 0.0, 0.0, is_outlier=True)
    assert read_timeline_status(hand_db, "seq1", "ci1")[1][95] == STATUS_GREY

    clear_disabled_hand_edits(hand_db, "seq1", "ci1", 1, side="left")

    status = read_timeline_status(hand_db, "seq1", "ci1")[1]
    assert status[95] == STATUS_GREEN  # back to the original (unedited) detection


def test_leaves_a_repositioning_edit_untouched(hand_db):
    from app.pose.db_cache import clear_disabled_hand_edits, update_single_keypoint_edit

    update_single_keypoint_edit(hand_db, "seq1", "ci1", 1, 95, 10.0, 20.0, is_outlier=False)
    clear_disabled_hand_edits(hand_db, "seq1", "ci1", 1, side="left")

    status = read_timeline_status(hand_db, "seq1", "ci1")[1]
    assert status[95] == STATUS_BLUE  # the deliberate placement still stands


def test_leaves_edits_outside_the_hand_range_untouched(hand_db):
    from app.pose.db_cache import clear_disabled_hand_edits, update_single_keypoint_edit

    update_single_keypoint_edit(hand_db, "seq1", "ci1", 1, 5, 0.0, 0.0, is_outlier=True)  # body-range
    clear_disabled_hand_edits(hand_db, "seq1", "ci1", 1, side="left")

    status = read_timeline_status(hand_db, "seq1", "ci1")[1]
    assert status[5] == STATUS_GREY  # still disabled -- outside hand_l's 91-111 range


def test_noop_when_no_edit_row_exists(hand_db):
    from app.pose.db_cache import clear_disabled_hand_edits

    clear_disabled_hand_edits(hand_db, "seq1", "ci1", 1, side="left")  # must not raise
    rows = hand_db.execute(
        "SELECT count(*) AS n FROM pose_observation_edits WHERE sequence_id='seq1'"
    ).fetchone()
    assert rows["n"] == 0


def test_deletes_the_edit_row_once_every_bit_is_cleared(hand_db):
    from app.pose.db_cache import clear_disabled_hand_edits, update_single_keypoint_edit

    update_single_keypoint_edit(hand_db, "seq1", "ci1", 1, 95, 0.0, 0.0, is_outlier=True)
    clear_disabled_hand_edits(hand_db, "seq1", "ci1", 1, side="left")

    rows = hand_db.execute(
        "SELECT count(*) AS n FROM pose_observation_edits"
        " WHERE sequence_id='seq1' AND camera_instance_id='ci1' AND video_frame=1"
    ).fetchone()
    assert rows["n"] == 0


def test_mixed_disabled_and_repositioned_fingers_only_clears_the_disabled_one(hand_db):
    from app.pose.db_cache import clear_disabled_hand_edits, update_single_keypoint_edit

    update_single_keypoint_edit(hand_db, "seq1", "ci1", 1, 95, 0.0, 0.0, is_outlier=True)
    update_single_keypoint_edit(hand_db, "seq1", "ci1", 1, 96, 30.0, 40.0, is_outlier=False)
    clear_disabled_hand_edits(hand_db, "seq1", "ci1", 1, side="left")

    status = read_timeline_status(hand_db, "seq1", "ci1")[1]
    assert status[95] == STATUS_GREEN  # disable was cleared
    assert status[96] == STATUS_BLUE   # reposition survives
