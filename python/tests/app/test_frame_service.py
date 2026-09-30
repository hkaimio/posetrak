# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for app/pose/frame_service.py (persistent PyAV decoders with
read-ahead; see docs/roadmap/features/preview-caching/preview-caching-design.md).

The test video is tiny and every frame paints its own index as ten black/white
blocks, so a decoded JPEG can be checked for *which* frame it is.
"""
from __future__ import annotations

import threading
import time

import pytest

av = pytest.importorskip("av")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.pose.frame_service import FrameDecoder, FrameService  # noqa: E402

N_FRAMES = 200
GOP = 30
W, H = 160, 96
BITS = 10


def _frame_image(idx: int) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for bit in range(BITS):
        if (idx >> bit) & 1:
            img[0:16, bit * 16:(bit + 1) * 16] = 255
    return img


def index_of(result) -> int:
    """The frame index painted into a decoded result."""
    img = cv2.imdecode(np.frombuffer(result[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    scale = img.shape[1] / W
    value = 0
    for bit in range(BITS):
        x = int((bit * 16 + 8) * scale)
        y = int(8 * scale)
        if img[y, x] > 128:
            value |= 1 << bit
    return value


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("video") / "frames.mp4")
    with av.open(path, "w") as out:
        stream = out.add_stream("libx264", rate=30)
        stream.width, stream.height = W, H
        stream.pix_fmt = "yuv420p"
        stream.options = {"g": str(GOP), "crf": "10", "preset": "ultrafast", "bf": "0"}
        for i in range(N_FRAMES):
            frame = av.VideoFrame.from_ndarray(_frame_image(i), format="bgr24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


def wait_until(predicate, timeout: float = 10.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class Recorder:
    def __init__(self) -> None:
        self.ready: list[int] = []
        self.lock = threading.Lock()

    def __call__(self, idx: int) -> None:
        with self.lock:
            self.ready.append(idx)

    def has(self, idx: int) -> bool:
        with self.lock:
            return idx in self.ready


@pytest.fixture()
def decoder(video):
    rec = Recorder()
    dec = FrameDecoder(video, rec, settle_s=0.02, thread_type="AUTO")
    dec.rec = rec
    yield dec
    dec.close()


def count_seeks(dec: FrameDecoder) -> list[int]:
    seeks: list[int] = []
    original = dec._seek

    def counting(target: int) -> None:
        seeks.append(target)
        original(target)

    dec._seek = counting
    return seeks


def request_and_wait(dec: FrameDecoder, idx: int) -> None:
    if dec.request(idx):   # already cached: no ready notification follows
        return
    assert wait_until(lambda: dec.rec.has(idx)), f"frame {idx} never became ready"


def test_a_requested_frame_is_the_right_frame(decoder):
    for idx in (0, 7, 45, 133, 199, 12):
        request_and_wait(decoder, idx)
        assert index_of(decoder.get(idx)) == idx


def test_result_covers_the_whole_frame(decoder):
    request_and_wait(decoder, 20)
    jpeg, wpx, hpx, sx, sy, sw, sh = decoder.get(20)
    assert (sx, sy, sw, sh) == (0, 0, W, H)
    assert (wpx, hpx) == (W, H)


def test_forward_stepping_reuses_the_open_decoder(decoder):
    seeks = count_seeks(decoder)
    request_and_wait(decoder, 10)
    for idx in range(11, 60):
        request_and_wait(decoder, idx)
        assert index_of(decoder.get(idx)) == idx
    assert len(seeks) == 1   # only the first request needed a seek


def test_forward_prefetch_fills_the_frames_ahead(decoder):
    request_and_wait(decoder, 40)
    assert wait_until(lambda: 40 + 60 in decoder.cached_frames())
    assert all(idx in decoder.cached_frames() for idx in range(40, 101))


def test_backward_step_keeps_the_whole_keyframe_interval(decoder):
    request_and_wait(decoder, 100)          # a jump: direction unknown
    request_and_wait(decoder, 99)           # first step back
    seeks = count_seeks(decoder)
    # The keyframe interval containing 99 (frames 90..99) is cached, so
    # stepping back through it needs no further decode.
    for idx in range(98, 89, -1):
        assert decoder.get(idx) is not None, f"frame {idx} should already be cached"
        assert index_of(decoder.get(idx)) == idx
    assert wait_until(lambda: 99 - 59 in decoder.cached_frames()), "previous interval not loaded"
    assert len(seeks) >= 1   # the earlier interval needed its own seek


def test_requesting_the_frame_already_shown_does_not_wake_the_worker(decoder):
    """A redraw asks for the frame it is showing; that must not be answered
    with another "ready", or every redraw would trigger the next."""
    request_and_wait(decoder, 25)
    time.sleep(0.4)
    before = list(decoder.rec.ready)
    for _ in range(20):
        decoder.request(25)
    time.sleep(0.4)
    assert decoder.rec.ready == before


def test_the_latest_request_wins(decoder):
    for idx in (10, 120, 30, 170, 60):
        decoder.request(idx)
    assert wait_until(lambda: decoder.rec.has(60))
    assert index_of(decoder.get(60)) == 60


def test_cache_stays_within_budget_and_keeps_the_target(video):
    rec = Recorder()
    dec = FrameDecoder(video, rec, budget_bytes=6 * 1024, settle_s=0.02)
    dec.rec = rec
    try:
        request_and_wait(dec, 80)
        assert wait_until(lambda: dec._cache_bytes <= 6 * 1024 + 2048)
        assert dec.get(80) is not None
        assert len(dec.cached_frames()) < 90   # not everything ahead was kept
    finally:
        dec.close()


def test_an_unreadable_file_reports_failure_and_closes(tmp_path):
    rec = Recorder()
    failed = threading.Event()
    dec = FrameDecoder(str(tmp_path / "missing.mp4"), rec, on_failed=failed.set)
    assert failed.wait(10)
    assert dec.failed
    dec.request(5)
    assert dec.get(5) is None
    assert dec.close()


def test_a_frame_past_the_end_is_reported_missing_without_hanging(video):
    rec = Recorder()
    missing: list[int] = []
    dec = FrameDecoder(video, rec, settle_s=0.02, on_missing=missing.append)
    dec.rec = rec
    try:
        dec.request(N_FRAMES + 500)
        assert wait_until(lambda: missing == [N_FRAMES + 500])
        assert dec.get(N_FRAMES + 500) is None
        assert dec.is_missing(N_FRAMES + 500) and not dec.is_missing(5)
        assert not rec.has(N_FRAMES + 500)
        request_and_wait(dec, 5)   # still usable afterwards
        assert index_of(dec.get(5)) == 5
    finally:
        dec.close()


def test_close_returns_promptly_even_during_prefetch(video):
    rec = Recorder()
    dec = FrameDecoder(video, rec, settle_s=0.0)
    dec.rec = rec
    request_and_wait(dec, 3)
    start = time.time()
    assert dec.close(timeout=5)
    assert time.time() - start < 5


def test_service_serves_and_reports_per_camera(video, tmp_path):
    from PySide6.QtCore import Qt

    service = FrameService(
        [
            {"shot_video_id": "a", "file_path": video},
            {"shot_video_id": "bad", "file_path": str(tmp_path / "nope.mp4")},
        ],
        settle_s=0.02,
    )
    ready: list[tuple[str, int]] = []
    missing: list[str] = []
    service.frame_ready.connect(lambda s, i: ready.append((s, i)), Qt.ConnectionType.DirectConnection)
    service.unavailable.connect(missing.append, Qt.ConnectionType.DirectConnection)
    try:
        service.request("a", 33)
        service.request("bad", 33)
        assert wait_until(lambda: ("a", 33) in ready)
        assert index_of(service.get("a", 33)) == 33
        assert wait_until(lambda: "bad" in missing)
        assert service.is_unavailable("bad") and not service.is_unavailable("a")
        assert service.get("unknown", 1) is None
        assert not service.is_missing("a", 33)
    finally:
        service.shutdown()
