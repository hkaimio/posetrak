# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Scrub proxy cache: sidecar database of every Nth video frame."""
from __future__ import annotations

import pytest

av = pytest.importorskip("av")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from posetrak.detection.proxy_cache import ProxyCache, session_db_path, sidecar_path  # noqa: E402

N, W, H = 100, 160, 96


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("v") / "v.mp4")
    with av.open(path, "w") as out:
        stream = out.add_stream("libx264", rate=30)
        stream.width, stream.height = W, H
        stream.pix_fmt = "yuv420p"
        stream.options = {"g": "20", "crf": "10", "preset": "ultrafast", "bf": "0"}
        for i in range(N):
            img = np.full((H, W, 3), i, dtype=np.uint8)   # frame i has grey level i
            for p in stream.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                out.mux(p)
        for p in stream.encode():
            out.mux(p)
    return path


@pytest.fixture()
def cache(tmp_path):
    c = ProxyCache(tmp_path / "s.cache.db")
    yield c
    c.close()


def grey(result) -> int:
    return int(cv2.imdecode(np.frombuffer(result[0], np.uint8), cv2.IMREAD_GRAYSCALE).mean())


def test_sidecar_sits_beside_the_session(tmp_path):
    assert sidecar_path(tmp_path / "x.db") == tmp_path / "x.cache.db"


def test_in_memory_session_has_no_path():
    import sqlite3
    assert session_db_path(sqlite3.connect(":memory:")) is None


def test_build_stores_every_nth_frame_of_the_range(cache, video):
    written = cache.build("sv", video, start=10, end=60, step=8, width=80)
    assert written == 6   # 16, 24, 32, 40, 48, 56
    result = cache.nearest("sv", 40, 0)
    assert grey(result) in range(38, 43) and result[1] == 80 and result[5:] == (W, H)


def test_nearest_picks_the_closest_within_the_distance_only(cache, video):
    cache.build("sv", video, start=0, end=99, step=8, width=80)
    assert grey(cache.nearest("sv", 43, 8)) in range(38, 43)   # frame 40, not 48
    assert grey(cache.nearest("sv", 45, 8)) in range(46, 51)   # frame 48
    assert cache.nearest("sv", 43, 1) is None
    assert cache.nearest("other", 40, 8) is None


def test_a_finished_range_is_not_decoded_again_but_a_cancelled_one_is(cache, video):
    assert cache.build("sv", video, start=0, end=40, step=8, width=80, cancelled=lambda: True) == 0
    assert not cache.is_covered("sv", 0, 40, 8, 80)
    assert cache.build("sv", video, start=0, end=40, step=8, width=80) == 6
    assert cache.is_covered("sv", 8, 32, 8, 80)
    assert cache.build("sv", video, start=0, end=40, step=8, width=80) == 0


def test_clear_removes_one_video_or_everything(cache, video):
    cache.build("a", video, start=0, end=40, step=8, width=80)
    cache.build("b", video, start=0, end=40, step=8, width=80)
    cache.clear("a")
    assert cache.nearest("a", 8, 0) is None and cache.nearest("b", 8, 0) is not None
    cache.clear()
    assert cache.stats() == []
