# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for posetrak.detection.frame_source's rotation handling.

_parse_displaymatrix in particular had zero test coverage despite being
described elsewhere (status.md) as "already-tested" -- it wasn't, and
carried a sign-convention bug (matched FFmpeg's own av_display_rotation_get(),
which returns *counter*-clockwise degrees, then treated the result as
clockwise degrees without renegating) that silently rotated every
DISPLAYMATRIX-tagged portrait video by an extra 180°, i.e. upside-down
rather than sideways. Found 2026-08-15 against a real OnePlus 10 Pro
portrait capture (see docs/roadmap/features/extrinsics-improvements/
status.md); confirmed empirically by comparing frames read via PyAV
(the buggy path) against the same frames read via a backend that
auto-rotates correctly, not just by re-deriving the formula.
"""
import struct

import cv2
import numpy as np

from posetrak.detection.frame_source import (
    _apply_rotation,
    _parse_displaymatrix,
    _stream_rotation,
)


class _FakeStream:
    def __init__(self, metadata):
        self.metadata = metadata


def test_parse_displaymatrix_real_oneplus_10_pro_matrix():
    """Exact DISPLAYMATRIX bytes extracted from a real OnePlus 10 Pro
    4K120 portrait capture (20260810-extrinsics-test-1-oneplus10pro-4k-120-
    portrait.mp4, frame side data). Ground truth (90° clockwise) confirmed
    by comparing this file's frames against the same frames read via a
    backend that auto-rotates correctly (cv2.VideoCapture on this system) --
    the previously-shipped formula computed 270°, verified upside-down by
    direct visual comparison, not just re-derivation."""
    m = (0, 65536, 0, -65536, 0, 0, 0, 0, 1073741824)
    data = struct.pack("<9i", *m)
    assert _parse_displaymatrix(data) == 90


def test_parse_displaymatrix_identity_matrix_is_no_rotation():
    m = (65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824)
    data = struct.pack("<9i", *m)
    assert _parse_displaymatrix(data) == 0


def test_parse_displaymatrix_180_matrix():
    m = (-65536, 0, 0, 0, -65536, 0, 0, 0, 1073741824)
    data = struct.pack("<9i", *m)
    assert _parse_displaymatrix(data) == 180


def test_parse_displaymatrix_270_matrix():
    # The mirror image of the real OnePlus matrix above -- confirms the
    # function distinguishes the two 90°-rotation directions correctly,
    # not just "some 90° multiple".
    m = (0, -65536, 0, 65536, 0, 0, 0, 0, 1073741824)
    data = struct.pack("<9i", *m)
    assert _parse_displaymatrix(data) == 270


def test_parse_displaymatrix_short_data_returns_zero():
    assert _parse_displaymatrix(b"\x00" * 10) == 0


def test_stream_rotation_reads_plain_tag():
    assert _stream_rotation(_FakeStream({"rotate": "90"})) == 90


def test_stream_rotation_missing_tag_defaults_zero():
    assert _stream_rotation(_FakeStream({})) == 0
    assert _stream_rotation(_FakeStream(None)) == 0


def test_apply_rotation_90_matches_cv2_clockwise_convention():
    """An asymmetric marker image makes the rotation direction checkable,
    not just the output shape."""
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[0, 0] = (255, 0, 0)  # top-left corner, distinguishable
    rotated = _apply_rotation(img, 90)
    assert rotated.shape == (20, 10, 3)
    expected = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    assert np.array_equal(rotated, expected)


def test_apply_rotation_zero_is_identity():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    assert _apply_rotation(img, 0) is img
