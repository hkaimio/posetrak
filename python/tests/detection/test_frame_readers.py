# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""The decode readers: the same frames on a thread and in a child process."""
from __future__ import annotations

import cv2
import numpy as np
import pytest


def test_process_reader_yields_the_frames_of_the_thread_reader(tmp_path):
    from posetrak.detection.frame_readers import get_reader

    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for i in range(8):
        writer.write(np.full((48, 64, 3), 30 * i, np.uint8))
    writer.release()
    threaded = list(get_reader("thread")(path, 2, 6))
    processed = list(get_reader("process")(path, 2, 6))
    assert [f for f, _ in processed] == [f for f, _ in threaded] == [2, 3, 4, 5]
    for (_, a), (_, b) in zip(threaded, processed):
        np.testing.assert_array_equal(a, b)


def test_process_reader_can_be_abandoned_early(tmp_path):
    from posetrak.detection.frame_readers import get_reader

    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for i in range(40):
        writer.write(np.full((48, 64, 3), i, np.uint8))
    writer.release()
    frames = get_reader("process")(path, 0, 40)
    assert next(frames)[0] == 0
    frames.close()                                    # releases the child; must return promptly


def test_unknown_decode_mode_is_rejected():
    from posetrak.detection.frame_readers import get_reader

    with pytest.raises(ValueError, match="unknown decode mode"):
        get_reader("gpu")


def test_process_reader_kills_a_stalled_decoder(tmp_path, monkeypatch):
    from posetrak.detection import frame_readers

    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for i in range(4):
        writer.write(np.full((48, 64, 3), i, np.uint8))
    writer.release()
    monkeypatch.setattr(frame_readers, "STALL_TIMEOUT_S", 0.5)
    monkeypatch.setattr(frame_readers, "_POLL_S", 0.1)
    # a child that never produces anything: the read of a file that does not exist fails fast, so stall it by
    # making the child's decode wait forever
    monkeypatch.setattr(frame_readers, "_decode_child", _sleeping_child)
    with pytest.raises(frame_readers.DecodeStalled):
        list(frame_readers.process_reader(path, 0, 4))


def _sleeping_child(path, first_frame, last_frame, ready, free, n_slots):
    import time

    time.sleep(60)
