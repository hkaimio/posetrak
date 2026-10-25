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


def _wait(predicate, timeout=20.0) -> bool:
    import time
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_builder_starts_with_the_chunk_around_the_focus_and_finishes_everything(tmp_path, video):
    from posetrak.detection.proxy_cache import ProxyBuilder

    path = tmp_path / "b.cache.db"
    builder = ProxyBuilder(path, "sv", video, 0, 99, focus=90, width=80, chunk=20)
    order = []
    original = builder._nearest_pending

    def spy():
        chunk = original()
        if chunk is not None and (not order or order[-1] != chunk):
            order.append(chunk)
        return chunk

    builder._nearest_pending = spy
    assert builder.total == 5
    builder.start()
    assert _wait(lambda: builder.done == 5)
    assert order[0] == (80, 99)
    assert order[1] == (60, 79)
    reader = ProxyCache(path)
    assert reader.nearest("sv", 8, 0) is not None and reader.nearest("sv", 96, 0) is not None
    reader.close()


def test_builder_follows_the_focus_to_the_nearest_unfinished_chunk(tmp_path, video):
    from posetrak.detection.proxy_cache import ProxyBuilder

    builder = ProxyBuilder(tmp_path / "c.cache.db", "sv", video, 0, 99, focus=0, chunk=20)
    assert builder._nearest_pending() == (0, 19)
    builder.set_focus(70)
    assert builder._nearest_pending() == (60, 79)
    builder._pending.remove((60, 79))
    assert builder._nearest_pending() in ((40, 59), (80, 99))   # equidistant from 70 either way
    builder.cancel()


def test_first_frame_index_counts_a_late_first_frame_as_an_offset():
    from types import SimpleNamespace

    from posetrak.detection.proxy_cache import first_frame_index

    def stream(start, tb, fps):
        return SimpleNamespace(start_time=start, time_base=tb, average_rate=fps)

    assert first_frame_index(stream(0, 1 / 120000, 119.88)) == 0
    assert first_frame_index(stream(None, 1 / 90000, 120)) == 0
    assert first_frame_index(stream(963, 1 / 90000, 119.967)) == 1   # first frame ~1.3 frames in
