# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for pose_observation_edits read/write helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from posetrak.db.db import create_session

from app.pose.db_cache import (
    read_observations_with_edits,
    update_single_keypoint_edit,
    write_observation_edit,
)

N_KP = 4  # small keypoint count for tests


def _make_kp(x: float, y: float, conf: float) -> np.ndarray:
    kp = np.zeros((N_KP, 3), dtype=np.float32)
    kp[:] = [x, y, conf]
    return kp


def _encode_kp(kp: np.ndarray) -> bytes:
    return kp.astype(np.float32).tobytes()


def _build_mask(*indices: int) -> bytes:
    """Build a uint8 bitmask with bits set at the given keypoint slot indices."""
    n_bytes = math.ceil(N_KP / 8)
    mask = bytearray(n_bytes)
    for i in indices:
        mask[i // 8] |= 1 << (i % 8)
    return bytes(mask)


@pytest.fixture()
def obs_session(tmp_path):
    """Session DB with two observation frames for sequence seq1 / camera cam1.

    Uses create_session to get the full v21 schema, then disables FK enforcement
    so the test fixture only needs to insert the rows that matter.
    """
    conn = create_session(tmp_path / "obs_edits.db")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES ('shot1', 'sess1', 1)"
    )
    conn.execute(
        "INSERT INTO pose_observation_sequences"
        " (id, shot_id, sync_config_id, time_start_s, time_end_s, pixels_are_undistorted)"
        " VALUES ('seq1', 'shot1', 'sync1', 0.0, 1.0, 0)"
    )
    conn.execute(
        "INSERT INTO pose_observations"
        " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, kp_blob)"
        " VALUES ('seq1', 'cam1', 10, 0.0, 0, ?)",
        (_encode_kp(_make_kp(100.0, 200.0, 0.9)),),
    )
    conn.execute(
        "INSERT INTO pose_observations"
        " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, kp_blob)"
        " VALUES ('seq1', 'cam1', 11, 0.1, 0, ?)",
        (_encode_kp(_make_kp(110.0, 210.0, 0.8)),),
    )
    conn.commit()
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# No-edit passthrough
# ---------------------------------------------------------------------------

def test_no_edits_returns_original(obs_session):
    result = read_observations_with_edits(obs_session, "seq1", "cam1")
    assert set(result.keys()) == {10, 11}
    np.testing.assert_allclose(result[10][:, 0], 100.0)
    np.testing.assert_allclose(result[10][:, 1], 200.0)
    np.testing.assert_allclose(result[10][:, 2], 0.9, rtol=1e-5)
    np.testing.assert_allclose(result[11][:, 0], 110.0)


def test_empty_result_for_missing_camera(obs_session):
    result = read_observations_with_edits(obs_session, "seq1", "cam_nobody")
    assert result == {}


# ---------------------------------------------------------------------------
# Positional edit (is_outlier = 0)
# ---------------------------------------------------------------------------

def test_edit_moves_keypoint(obs_session):
    """Slots with is_outlier=0 get their x/y replaced and confidence set to 1."""
    edit_kp = _make_kp(50.0, 60.0, 0.0)  # is_outlier=0
    mask = _build_mask(0, 2)
    write_observation_edit(obs_session, "seq1", "cam1", 10, edit_kp, mask)

    result = read_observations_with_edits(obs_session, "seq1", "cam1")
    kp = result[10]
    assert kp[0, 0] == pytest.approx(50.0)
    assert kp[0, 1] == pytest.approx(60.0)
    assert kp[0, 2] == pytest.approx(1.0)
    assert kp[2, 0] == pytest.approx(50.0)
    # Untouched slots keep originals
    assert kp[1, 0] == pytest.approx(100.0)
    assert kp[3, 2] == pytest.approx(0.9, rel=1e-5)
    # Frame 11 unchanged
    np.testing.assert_allclose(result[11][:, 0], 110.0)


# ---------------------------------------------------------------------------
# Outlier edit (is_outlier != 0)
# ---------------------------------------------------------------------------

def test_edit_marks_outlier_zeroes_confidence(obs_session):
    """Slots with is_outlier!=0 get confidence zeroed; x/y are left as original."""
    edit_kp = _make_kp(0.0, 0.0, 1.0)  # is_outlier=1
    mask = _build_mask(1)
    write_observation_edit(obs_session, "seq1", "cam1", 10, edit_kp, mask)

    result = read_observations_with_edits(obs_session, "seq1", "cam1")
    kp = result[10]
    assert kp[1, 2] == pytest.approx(0.0)
    assert kp[1, 0] == pytest.approx(100.0)  # x unchanged
    # Other slots unaffected
    assert kp[0, 2] == pytest.approx(0.9, rel=1e-5)


# ---------------------------------------------------------------------------
# Upsert behaviour
# ---------------------------------------------------------------------------

def test_upsert_replaces_existing_edit(obs_session):
    """Calling write_observation_edit twice for the same frame overwrites the edit."""
    write_observation_edit(obs_session, "seq1", "cam1", 10, _make_kp(50.0, 60.0, 0.0), _build_mask(0))
    write_observation_edit(obs_session, "seq1", "cam1", 10, _make_kp(77.0, 88.0, 0.0), _build_mask(0))

    result = read_observations_with_edits(obs_session, "seq1", "cam1")
    assert result[10][0, 0] == pytest.approx(77.0)
    assert result[10][0, 1] == pytest.approx(88.0)


# ---------------------------------------------------------------------------
# Multi-source rows (Phase 2: body + hand_l/hand_r sharing one frame)
# ---------------------------------------------------------------------------

_N_KP_133 = 133


def _kp133(fill: float, n: int = _N_KP_133) -> np.ndarray:
    kp = np.zeros((n, 3), dtype=np.float32)
    kp[:] = [fill, fill, fill]
    return kp


@pytest.fixture()
def multi_source_session(tmp_path):
    """Session DB with 'body' + 'hand_l' rows sharing frame 10 of seq1/cam1."""
    conn = create_session(tmp_path / "obs_edits_multi.db")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES ('shot1', 'sess1', 1)"
    )
    conn.execute(
        "INSERT INTO pose_observation_sequences"
        " (id, shot_id, sync_config_id, time_start_s, time_end_s, pixels_are_undistorted)"
        " VALUES ('seq1', 'shot1', 'sync1', 0.0, 1.0, 0)"
    )
    conn.execute(
        "INSERT INTO pose_observations"
        " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, source, kp_blob)"
        " VALUES ('seq1', 'cam1', 10, 0.0, 0, 'body', ?)",
        (_kp133(1.0).tobytes(),),
    )
    conn.execute(
        "INSERT INTO pose_observations"
        " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, source, kp_blob)"
        " VALUES ('seq1', 'cam1', 10, 0.0, 0, 'hand_l', ?)",
        (_kp133(9.0, n=21).tobytes(),),
    )
    conn.commit()
    yield conn
    conn.close()


def test_read_observations_merges_body_and_hand_rows(multi_source_session):
    """A frame with both 'body' and 'hand_l' rows must merge, not overwrite.

    Regression test: read_observations_with_edits used to do an unconditional
    `result[frame] = kp` per row, so whichever row loaded last silently won —
    losing the other source's keypoints instead of merging them.
    """
    result = read_observations_with_edits(multi_source_session, "seq1", "cam1")
    kp = result[10]
    assert kp.shape[0] == 133
    np.testing.assert_allclose(kp[91:112, 0], 9.0)  # hand_l slots
    np.testing.assert_allclose(kp[0, 0], 1.0)  # body slot untouched


def test_ghost_frame_hand_only_row_merges_to_camera_width(multi_source_session):
    """A frame with *only* an overlay row (no 'body' row) must still merge to
    the camera's full width, using another frame's 'body' row as the width
    reference.

    Regression test: this is exactly Idea 3's auto-redetect-on-a-ghost-frame
    case -- a wrist/elbow is placed via an edit on a frame with no original
    detection, auto-redetection fires and writes only a 'hand_l.refined' row
    for that frame (no accompanying 'body' row), and the frame used to come
    back as a bare 21-point array while every other frame in the camera was
    133-wide, crashing any code that assumes one width per camera.
    """
    multi_source_session.execute(
        "INSERT INTO pose_observations"
        " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, source, kp_blob)"
        " VALUES ('seq1', 'cam1', 11, 0.033, 0, 'hand_l.refined', ?)",
        (_kp133(5.0, n=21).tobytes(),),
    )
    multi_source_session.commit()

    result = read_observations_with_edits(multi_source_session, "seq1", "cam1")
    kp = result[11]
    assert kp.shape[0] == 133
    np.testing.assert_allclose(kp[91:112, 0], 5.0)
    np.testing.assert_allclose(kp[:91, 2], 0.0)  # no body detection -- zero confidence
    np.testing.assert_allclose(kp[112:, 2], 0.0)


def test_update_single_keypoint_edit_uses_body_row_width(multi_source_session):
    """n_kp inference must use the 'body' row, not whichever row loads first.

    Regression test: update_single_keypoint_edit inferred n_kp from a bare
    fetchone() with no source filter; if it happened to bind to the 21-point
    'hand_l' row instead of the 133-point 'body' row, editing a body-range
    index would silently corrupt the edit blob width.
    """
    update_single_keypoint_edit(
        multi_source_session, "seq1", "cam1", 10, kp_idx=5,
        new_x=42.0, new_y=43.0, is_outlier=False,
    )
    result = read_observations_with_edits(multi_source_session, "seq1", "cam1")
    kp = result[10]
    assert kp[5, 0] == pytest.approx(42.0)
    assert kp[5, 1] == pytest.approx(43.0)
    # hand_l slots still present and untouched by the edit.
    np.testing.assert_allclose(kp[91:112, 0], 9.0)


# ---------------------------------------------------------------------------
# primary_source (marker-based-mocap design doc §7.1 sub-phase 1e): a
# sequence whose real source is never 'body' -- an object sequence's
# source='markers' -- needs its own primary_source passed through both
# read_observations_with_edits and update_single_keypoint_edit.
# ---------------------------------------------------------------------------


@pytest.fixture()
def markers_source_session(tmp_path):
    """Session DB with a 'markers'-sourced sequence (no 'body' row at all)."""
    conn = create_session(tmp_path / "obs_edits_markers.db")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES ('shot1', 'sess1', 1)"
    )
    conn.execute(
        "INSERT INTO pose_observation_sequences"
        " (id, shot_id, sync_config_id, time_start_s, time_end_s, pixels_are_undistorted)"
        " VALUES ('seq1', 'shot1', 'sync1', 0.0, 1.0, 0)"
    )
    conn.execute(
        "INSERT INTO pose_observations"
        " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, source, kp_blob)"
        " VALUES ('seq1', 'cam1', 0, 0.0, 0, 'markers', ?)",
        (_make_kp(100.0, 200.0, 1.0).tobytes(),),
    )
    conn.commit()
    yield conn
    conn.close()


def test_read_with_primary_source_returns_real_row_unedited(markers_source_session):
    result = read_observations_with_edits(
        markers_source_session, "seq1", "cam1", primary_source="markers",
    )
    np.testing.assert_allclose(result[0][:, 0], 100.0)


def test_edit_one_slot_leaves_others_at_their_real_values(markers_source_session):
    """Regression test for the exact bug found building ObjectPanel
    (status.md, 2026-08-30): once an edit exists anywhere in the camera,
    merge_observation_sources used to synthesize a same-width *zero* body
    for every frame whose real source wasn't literally 'body' -- silently
    discarding the real 'markers' row's untouched slots instead of merging
    onto them, the moment default_width happened to become known (here,
    from the edit's own shape).
    """
    update_single_keypoint_edit(
        markers_source_session, "seq1", "cam1", 0, kp_idx=0,
        new_x=999.0, new_y=888.0, source="markers",
    )
    result = read_observations_with_edits(
        markers_source_session, "seq1", "cam1", primary_source="markers",
    )
    kp = result[0]
    assert kp[0, 0] == pytest.approx(999.0)
    assert kp[0, 1] == pytest.approx(888.0)
    # Every other slot must still carry its real, untouched value.
    np.testing.assert_allclose(kp[1:, 0], 100.0)
    np.testing.assert_allclose(kp[1:, 1], 200.0)
    np.testing.assert_allclose(kp[1:, 2], 1.0)


def test_update_single_keypoint_edit_with_wrong_default_source_is_a_noop(markers_source_session):
    """Without passing source='markers', update_single_keypoint_edit looks
    for a 'body' row (its own default), finds none, and returns without
    writing anything -- documenting the failure mode the source parameter
    exists to avoid, not silently mis-writing to the wrong width."""
    update_single_keypoint_edit(
        markers_source_session, "seq1", "cam1", 0, kp_idx=0, new_x=1.0, new_y=2.0,
    )
    assert markers_source_session.execute(
        "SELECT COUNT(*) FROM pose_observation_edits"
    ).fetchone()[0] == 0
