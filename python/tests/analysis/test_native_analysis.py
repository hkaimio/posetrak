# SPDX-FileCopyrightText: 2026 Harri Kaimio
# SPDX-License-Identifier: Apache-2.0
from fractions import Fraction
import av
import cv2
import numpy as np
import pytest

from posetrak.analysis.native_video import VideoFrames
from posetrak.analysis.shared_pass import Consumer, SharedAruco, run_shared_pass


def video(tmp_path):
    path = tmp_path / "native.mkv"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=30)
        stream.width = stream.height = 160
        stream.pix_fmt = "bgr0"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for pts in [1000, 1100, 1250, 1400]:
            image = np.full((160, 160, 3), 255, dtype=np.uint8)
            marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000), 51, 100)
            image[30:130, 30:130] = marker[:, :, None]
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_native_vfr_ordinals_and_shared_markers_are_not_decoded_twice(tmp_path):
    path = video(tmp_path)
    frames = VideoFrames(path, "source")
    detector = SharedAruco("DICT_4X4_1000")
    left, right = [], []
    def consume(target):
        def callback(frame, markers):
            assert not frame.image.bgr().flags.writeable
            crop = frame.image.bgr((40, 40, 40, 40))
            assert crop.shape == (40, 40, 3)
            assert np.array_equal(frame.image.bgr((31, 31, 41, 41)), frame.image.bgr()[31:72, 31:72])
            target.append((frame.reference, markers))
        return callback
    run_shared_pass(frames, [Consumer(consume(left), markers=True), Consumer(consume(right), markers=True)], detector)
    assert frames.complete
    assert frames.metrics["decoded"] == 4 and frames.metrics["passes"] == 1
    assert detector.calls == 4
    assert [r.ordinal for r in frames.index] == [0, 1, 2, 3]
    assert [float(r.pts*r.time_base) for r in frames.index] == pytest.approx([1, 1.1, 1.25, 1.4])
    assert all(a[1] is b[1] for a, b in zip(left, right))
    assert all(row[1][0].marker_id == 51 for row in left)


def test_cancel_before_decode_has_no_partial_complete_index(tmp_path):
    frames = VideoFrames(video(tmp_path), "source", cancelled=lambda: True)
    with pytest.raises(InterruptedError):
        list(frames)
    assert not frames.complete and not frames.index


def test_cancel_during_fanout_stops_after_current_frame(tmp_path):
    state = {"cancel": False}
    frames = VideoFrames(video(tmp_path), "source", cancelled=lambda: state["cancel"])
    def stop(frame, markers):
        state["cancel"] = True
    with pytest.raises(InterruptedError):
        run_shared_pass(frames, [Consumer(stop)], SharedAruco("DICT_4X4_1000"), cancelled=lambda: state["cancel"])
    assert frames.metrics["decoded"] == 1


def test_native_pts_reset_is_preserved_as_index_segment(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    native = []
    for pts in [1000, 1100, 0, 100]:
        frame = av.VideoFrame(16, 16, "bgr24")
        frame.pts, frame.time_base = pts, Fraction(1, 1000)
        native.append(frame)
    @contextmanager
    def opened(path):
        stream = SimpleNamespace(codec_context=SimpleNamespace(thread_count=1), metadata={})
        yield SimpleNamespace(streams=SimpleNamespace(video=[stream]), decode=lambda stream: iter(native))
    monkeypatch.setattr(av, "open", opened)
    frames = VideoFrames("source", "video")
    packets = list(frames)
    assert [p.reference.ordinal for p in packets] == [0, 1, 2, 3]
    assert [p.reference.pts_segment for p in packets] == [0, 0, 1, 1]
    assert [p.discontinuity_before for p in packets] == [False, False, True, False]


def test_persistent_markers_reuse_detection_and_reject_stale_source(tmp_path):
    path = video(tmp_path)
    frames = VideoFrames(path, "source")
    detector = SharedAruco("DICT_4X4_1000")
    run_shared_pass(frames, [Consumer(lambda f, m: None, markers=True)], detector)
    cache = tmp_path / "markers.json"
    sources = {"source": {"sha256": "content", "mode_id": "mode"}}
    detector.save(cache, sources)
    restored = SharedAruco("DICT_4X4_1000")
    restored.load(cache, sources)
    run_shared_pass(VideoFrames(path, "source"), [Consumer(lambda f, m: None, markers=True)], restored)
    assert restored.calls == 0
    with pytest.raises(ValueError, match="provenance"):
        restored.load(cache, {"source": {"sha256": "changed", "mode_id": "mode"}})


def test_optical_adapter_accepts_host_owned_frames_without_reopening(tmp_path, monkeypatch):
    from posetrak.analysis.optical_wand import OpticalWandInput
    from sync_device.wand_profile import WandProfile
    packets = list(VideoFrames(video(tmp_path), "source"))
    def cannot_open(*args, **kwargs):
        raise AssertionError("Adapter reopened a host-owned source")
    monkeypatch.setattr(av, "open", cannot_open)
    inputs = OpticalWandInput(WandProfile("test", "DICT_4X4_1000", {51: 4}), native_frames=lambda _: iter(packets))
    delivered = list(inputs({"path": "unavailable", "video_id": "source"}))
    assert [p.reference.pts for p in delivered] == [1000, 1100, 1250, 1400]
    assert inputs.stats["passes"] == 0
