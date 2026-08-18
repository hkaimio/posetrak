# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for PersonCropGridWidget data loading logic."""
from __future__ import annotations

import math

import numpy as np

from app.pose.db_cache import write_observation_edit
from tests.app.conftest import _SEQ_DB_N_KP as N_KP


def _encode_kp(x: float = 100.0, y: float = 200.0, conf: float = 0.9) -> bytes:
    kp = np.full((N_KP, 3), [x, y, conf], dtype=np.float32)
    return kp.tobytes()


def _make_mask(*indices: int) -> bytes:
    n_bytes = math.ceil(N_KP / 8)
    mask = bytearray(n_bytes)
    for i in indices:
        mask[i // 8] |= 1 << (i % 8)
    return bytes(mask)


# ---------------------------------------------------------------------------
# Tests (data-layer only, no Qt rendering)
# ---------------------------------------------------------------------------

def test_load_sequence_builds_frame_list(qapp, seq_db):
    from app.pose.crop_editor import PersonCropGridWidget
    w = PersonCropGridWidget()
    w.load_sequence(seq_db, "seq1")

    assert len(w._frames) == 2
    assert len(w._cameras) == 2
    assert w._cameras[0].label in ("cam_A", "cam_B")


def test_frame_slots_have_per_camera_video_frames(qapp, seq_db):
    from app.pose.crop_editor import PersonCropGridWidget
    w = PersonCropGridWidget()
    w.load_sequence(seq_db, "seq1")

    # Both frames should have entries for both cameras
    for fs in w._frames:
        assert "ci1" in fs.per_cam
        assert "ci2" in fs.per_cam


def test_track_id_lookup(qapp, seq_db):
    from app.pose.crop_editor import PersonCropGridWidget, _CameraSlot
    w = PersonCropGridWidget()
    w.load_sequence(seq_db, "seq1")

    cam_sv1 = next(c for c in w._cameras if c.shot_video_id == "sv1")
    assert w._track_id_for_frame(cam_sv1, 10) == 42
    assert w._track_id_for_frame(cam_sv1, 100) == 42
    assert w._track_id_for_frame(cam_sv1, 200) is None  # out of range


def test_load_crop_returns_jpeg(qapp, seq_db):
    from app.pose.crop_editor import PersonCropGridWidget
    w = PersonCropGridWidget()
    w.load_sequence(seq_db, "seq1")

    jpeg, src_x, src_y, src_w, src_h = w._load_crop("sv1", 10, 42)
    assert jpeg is not None and len(jpeg) > 0
    assert src_x == 50
    assert src_y == 30
    assert src_w == 200
    assert src_h == 150


def test_load_crop_returns_none_for_missing(qapp, seq_db):
    from app.pose.crop_editor import PersonCropGridWidget
    w = PersonCropGridWidget()
    w.load_sequence(seq_db, "seq1")

    jpeg, *_ = w._load_crop("sv1", 99, 42)  # frame 99 has no crop
    assert jpeg is None


def test_kp_by_frame_populated(qapp, seq_db):
    from app.pose.crop_editor import PersonCropGridWidget
    w = PersonCropGridWidget()
    w.load_sequence(seq_db, "seq1")

    cam_ci1 = next(c for c in w._cameras if c.camera_instance_id == "ci1")
    assert 10 in cam_ci1.kp_by_frame
    assert cam_ci1.kp_by_frame[10].shape == (N_KP, 3)


def test_edited_mask_detected(qapp, seq_db):
    from app.pose.crop_editor import PersonCropGridWidget

    # Write an edit for ci1 frame 10 before loading
    edit_kp = np.full((N_KP, 3), [55.0, 65.0, 0.0], dtype=np.float32)
    write_observation_edit(seq_db, "seq1", "ci1", 10, edit_kp, _make_mask(1))

    w = PersonCropGridWidget()
    w.load_sequence(seq_db, "seq1")

    mask = w._edited_mask("ci1", 10)
    assert mask is not None
    assert bool(mask[1]) is True
    assert bool(mask[0]) is False
