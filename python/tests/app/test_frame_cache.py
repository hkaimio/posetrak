# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for app.setup.frame_cache (FrameCache, CacheKey, CacheType)."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from app.setup.frame_cache import CacheKey, CacheType, FrameCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VID = "vid-abc"
_PATH = "/fake/video.mp4"


def _make_frame(val: int = 42) -> np.ndarray:
    """Return a tiny 4×4 BGR frame filled with *val*."""
    return np.full((4, 4, 3), val, dtype=np.uint8)


def _full_key(frame_idx: int) -> CacheKey:
    return CacheKey(shot_video_id=_VID, frame_idx=frame_idx, cache_type=CacheType.FULL_FRAME)


def _thumb_key(frame_idx: int) -> CacheKey:
    return CacheKey(shot_video_id=_VID, frame_idx=frame_idx,
                    cache_type=CacheType.THUMB, width_px=2, height_px=2)


# ---------------------------------------------------------------------------
# LRU cache tests
# ---------------------------------------------------------------------------


def test_lru_hit_does_not_seek(monkeypatch) -> None:
    """A cached frame must be returned from LRU without touching VideoCapture."""
    cache = FrameCache(conn=None, lru_max=10)
    frame = _make_frame(7)
    key = _full_key(0)
    cache._lru_put(key, frame)

    with patch("cv2.VideoCapture") as mock_cap:
        result = cache.get(key, file_path=_PATH)

    mock_cap.assert_not_called()
    np.testing.assert_array_equal(result, frame)


def test_lru_eviction() -> None:
    """LRU should evict the oldest entry when the cap is exceeded."""
    cache = FrameCache(conn=None, lru_max=3)
    keys = [_full_key(i) for i in range(4)]
    frames = [_make_frame(i) for i in range(4)]

    for k, f in zip(keys[:3], frames[:3]):
        cache._lru_put(k, f)

    assert len(cache._lru) == 3
    cache._lru_put(keys[3], frames[3])
    # Oldest (keys[0]) should be evicted
    assert len(cache._lru) == 3
    assert keys[0] not in cache._lru
    assert keys[3] in cache._lru


def test_lru_access_refreshes_order() -> None:
    """Accessing a cached entry should prevent it from being evicted next."""
    cache = FrameCache(conn=None, lru_max=2)
    k0, k1, k2 = _full_key(0), _full_key(1), _full_key(2)
    f0, f1, f2 = _make_frame(0), _make_frame(1), _make_frame(2)

    cache._lru_put(k0, f0)
    cache._lru_put(k1, f1)

    # Touch k0 so it becomes most-recently-used
    cache._lru.move_to_end(k0)

    # Adding k2 should evict k1 (least recently used), not k0
    cache._lru_put(k2, f2)
    assert k1 not in cache._lru
    assert k0 in cache._lru


# ---------------------------------------------------------------------------
# Sequential vs random seek tests
# ---------------------------------------------------------------------------


def _make_mock_cap(frame: np.ndarray) -> MagicMock:
    """Return a mock cv2.VideoCapture that always reads *frame*."""
    cap = MagicMock()
    cap.read.return_value = (True, frame)
    return cap


def test_sequential_read_does_not_seek() -> None:
    """Reading frames 0, 1, 2 in order must not call CAP_PROP_POS_FRAMES."""
    import cv2
    cache = FrameCache(conn=None, lru_max=100)
    frame = _make_frame()
    mock_cap = _make_mock_cap(frame)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        cache.get(_full_key(0), file_path=_PATH)
        cache.get(_full_key(1), file_path=_PATH)
        cache.get(_full_key(2), file_path=_PATH)

    # cap.set should not have been called
    mock_cap.set.assert_not_called()
    assert mock_cap.read.call_count == 3


def test_random_seek_calls_cap_set() -> None:
    """Reading a non-sequential frame must call CAP_PROP_POS_FRAMES."""
    import cv2
    cache = FrameCache(conn=None, lru_max=100)
    frame = _make_frame()
    mock_cap = _make_mock_cap(frame)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        cache.get(_full_key(0), file_path=_PATH)   # sequential
        cache.get(_full_key(5), file_path=_PATH)   # jump → seek

    # set should have been called for frame 5
    mock_cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 5)


def test_sequential_after_random_no_extra_seek() -> None:
    """After a random seek to frame N, reading N+1 must not seek again."""
    import cv2
    cache = FrameCache(conn=None, lru_max=100)
    frame = _make_frame()
    mock_cap = _make_mock_cap(frame)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        cache.get(_full_key(10), file_path=_PATH)   # jump → seek
        cache.get(_full_key(11), file_path=_PATH)   # sequential → no seek

    mock_cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 10)


# ---------------------------------------------------------------------------
# Thumb resize test
# ---------------------------------------------------------------------------


def test_thumb_key_resizes_frame() -> None:
    """A THUMB key should produce an image of the requested dimensions."""
    import cv2
    cache = FrameCache(conn=None, lru_max=10)
    big_frame = np.zeros((100, 200, 3), dtype=np.uint8)
    mock_cap = _make_mock_cap(big_frame)

    key = _thumb_key(0)
    with patch("cv2.VideoCapture", return_value=mock_cap):
        result = cache.get(key, file_path=_PATH)

    assert result.shape == (2, 2, 3)
