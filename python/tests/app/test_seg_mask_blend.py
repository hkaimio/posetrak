# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for _blend_seg_mask (segmentation mask overlay) and its wiring into
_display_crop_result.

Bug fixed here: the segmentation-mask blend only ever ran inside
_load_frame's per-track low-res path. In edit mode, _load_frame's wide-crop
cluster cache "preferred layer" renders through _display_crop_result instead
and `continue`s before reaching that code -- so once the cache caught up
(the common case), the checkbox had no effect. _blend_seg_mask is now a
shared helper both call.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QWidget

from app.ui.content_panels import PersonCropGridWidget


def _encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def dummy_db(tmp_path):
    from posetrak.db.db import create_session

    conn = create_session(tmp_path / "dummy.db")
    conn.execute("PRAGMA foreign_keys = OFF")
    yield conn
    conn.close()


def _insert_mask(conn, mask: np.ndarray, frame_idx: int = 0) -> None:
    conn.execute(
        "INSERT INTO seg_masks (seg_quality_run_id, shot_video_id, frame_idx, mask_blob) "
        "VALUES ('sqr1', 'sv1', ?, ?)",
        (frame_idx, _encode_png(mask)),
    )
    conn.commit()


def _make_widget(conn):
    w = PersonCropGridWidget.__new__(PersonCropGridWidget)
    QWidget.__init__(w)
    w._conn = conn
    w._seg_sources = {"sv1": "sqr1"}
    w._video_dims = {"sv1": (4, 4)}
    return w


_DAVIS_FIRST_BGR = (80, 80, 240)  # track_id=1 -> color_idx 0


# ---------------------------------------------------------------------------
# _blend_seg_mask
# ---------------------------------------------------------------------------

def test_blend_paints_masked_pixels_with_track_color(qapp, dummy_db):
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    _insert_mask(dummy_db, mask)
    w = _make_widget(dummy_db)
    crop = np.full((4, 4, 3), 10, dtype=np.uint8)

    out = w._blend_seg_mask(crop, "sv1", 0, 1, 0.0, 0.0, 4.0, 4.0)

    expected = np.array(
        [10 * 0.55 + c * 0.45 for c in _DAVIS_FIRST_BGR], dtype=np.float32
    ).astype(np.uint8)
    # The blend rounds to the nearest value; the exact rounding is not the point.
    assert np.allclose(out[0, 0], expected, atol=1)
    assert np.array_equal(out[3, 3], [10, 10, 10])  # outside the masked region: untouched


def test_blend_unchanged_when_no_seg_source_configured(qapp, dummy_db):
    w = _make_widget(dummy_db)
    w._seg_sources = {}
    crop = np.full((4, 4, 3), 10, dtype=np.uint8)
    out = w._blend_seg_mask(crop, "sv1", 0, 1, 0.0, 0.0, 4.0, 4.0)
    assert np.array_equal(out, crop)


def test_blend_unchanged_when_no_mask_row_for_frame(qapp, dummy_db):
    w = _make_widget(dummy_db)
    crop = np.full((4, 4, 3), 10, dtype=np.uint8)
    out = w._blend_seg_mask(crop, "sv1", 0, 1, 0.0, 0.0, 4.0, 4.0)
    assert np.array_equal(out, crop)


def test_blend_unchanged_when_track_not_present_in_mask(qapp, dummy_db):
    mask = np.full((4, 4), 2, dtype=np.uint8)  # only track 2 present
    _insert_mask(dummy_db, mask)
    w = _make_widget(dummy_db)
    crop = np.full((4, 4, 3), 10, dtype=np.uint8)
    out = w._blend_seg_mask(crop, "sv1", 0, 1, 0.0, 0.0, 4.0, 4.0)
    assert np.array_equal(out, crop)


# ---------------------------------------------------------------------------
# _display_crop_result: wiring (wide-crop cache has a track_id; backfill/ghost doesn't)
# ---------------------------------------------------------------------------

def _make_display_widget():
    w = PersonCropGridWidget.__new__(PersonCropGridWidget)
    QWidget.__init__(w)
    w._apply_overlay = MagicMock()
    w._blend_seg_mask = MagicMock(side_effect=lambda crop, *a, **k: crop)
    return w


def _fake_result():
    jpeg = _encode_png(np.full((4, 4, 3), 128, dtype=np.uint8))
    return (jpeg, 4, 4, 0.0, 0.0, 4.0, 4.0)


def test_display_crop_result_blends_when_track_id_and_show_seg_given(qapp):
    w = _make_display_widget()
    cell = MagicMock()

    w._display_crop_result(
        cell, "cam1", "sv1", 0, None, True, True, _fake_result(),
        track_id=3, show_seg=True,
    )

    w._blend_seg_mask.assert_called_once()
    args = w._blend_seg_mask.call_args.args
    assert args[1:4] == ("sv1", 0, 3)


def test_display_crop_result_blends_all_labels_without_track_id(qapp):
    """The backfill/ghost path (no detection at this frame, and every frame
    of an object sequence -> no track_id) blends every label present when
    show_seg is set."""
    w = _make_display_widget()
    cell = MagicMock()

    w._display_crop_result(
        cell, "cam1", "sv1", 0, None, True, True, _fake_result(),
        show_seg=True,
    )

    w._blend_seg_mask.assert_called_once()
    assert w._blend_seg_mask.call_args.args[3] is None


def test_display_crop_result_skips_blend_when_show_seg_false(qapp):
    w = _make_display_widget()
    cell = MagicMock()

    w._display_crop_result(
        cell, "cam1", "sv1", 0, None, True, True, _fake_result(),
        track_id=3, show_seg=False,
    )

    w._blend_seg_mask.assert_not_called()


def test_blend_without_track_id_paints_every_label(qapp, dummy_db):
    """An object sequence has no person track to pick a label by (and its
    labels are not in persons_json): every non-zero label is painted."""
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1
    mask[3, 3] = 3
    _insert_mask(dummy_db, mask)
    w = _make_widget(dummy_db)
    crop = np.full((4, 4, 3), 10, dtype=np.uint8)

    out = w._blend_seg_mask(crop, "sv1", 0, None, 0.0, 0.0, 4.0, 4.0)

    assert not np.array_equal(out[0, 0], [10, 10, 10])
    assert not np.array_equal(out[3, 3], [10, 10, 10])
    assert not np.array_equal(out[0, 0], out[3, 3])  # different colours per label
    assert np.array_equal(out[1, 1], [10, 10, 10])


def test_blend_uses_the_editors_label_palette(qapp, dummy_db):
    """Same colour per label as the segmentation editor (video_canvas.label_to_color)."""
    from app.pose.video_canvas import label_to_color

    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 3
    _insert_mask(dummy_db, mask)
    w = _make_widget(dummy_db)
    crop = np.full((4, 4, 3), 10, dtype=np.uint8)

    out = w._blend_seg_mask(crop, "sv1", 0, 3, 0.0, 0.0, 4.0, 4.0)

    r, g, b = label_to_color(3)
    expected = np.array([10 * 0.55 + c * 0.45 for c in (b, g, r)], dtype=np.float32).astype(np.uint8)
    assert np.allclose(out[0, 0], expected, atol=1)


def test_blend_draws_a_legend_with_subject_names_on_a_large_crop(qapp, dummy_db):
    from posetrak.db.manage_person import create_person

    dummy_db.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES ('s', '2026-01-01')")
    dummy_db.execute("INSERT INTO captures (id, session_id, capture_number) VALUES ('cap1', 's', 1)")
    create_person(dummy_db, "cap1", "Harri")   # registry label 1
    dummy_db.execute(
        "INSERT INTO seg_quality_runs (id, shot_id, time_start_s, time_end_s, created_at) "
        "VALUES ('sqr1', 'cap1', 0, 1, '2026-01-01')"
    )
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[150:190, 150:190] = 1   # far from the top-left legend area
    _insert_mask(dummy_db, mask)
    w = _make_widget(dummy_db)
    w._video_dims = {"sv1": (200, 200)}
    crop = np.full((200, 200, 3), 10, dtype=np.uint8)

    out = w._blend_seg_mask(crop, "sv1", 0, None, 0.0, 0.0, 200.0, 200.0)

    assert not np.array_equal(out[8:40, 8:120], crop[8:40, 8:120])   # legend drawn
    assert np.array_equal(out[100:140, 100:140], crop[100:140, 100:140])  # rest untouched
