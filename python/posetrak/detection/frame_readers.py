# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""frame_readers.py — decoding one camera's frames on a thread or in a child process.

Both readers have the signature of :func:`posetrak.detection.frame_source.iter_frames` and yield the same
``(video_frame, bgr_array)`` pairs, so the detection loop chooses between them with one setting
(:func:`get_reader`):

* ``"thread"`` decodes in the calling thread's generator (the loop runs one such generator per camera in its
  own thread; PyAV releases the GIL while it decodes);
* ``"process"`` decodes in a spawned child process that hands frames over through shared memory, so a decoder
  that hangs on open, decode or close can be killed instead of stranding a thread.

Which is the default is decided by the decode soak test (``python/tools/prototypes/decode_soak.py``) and recorded
in docs/roadmap/features/detection-pipeline/detection-pipeline-design.md.
"""
from __future__ import annotations

import multiprocessing
import queue
import traceback
from collections.abc import Callable, Iterator
from multiprocessing import shared_memory

import numpy as np

from posetrak.detection.frame_source import iter_frames

FrameReader = Callable[[str, int, int], Iterator[tuple[int, np.ndarray]]]

#: Frames in flight between the child and the loop. Each is a full decoded frame (25 MB at 4K).
PROCESS_SLOTS = 3
#: Seconds the loop waits for the child before it checks that the child is alive.
_POLL_S = 1.0


def thread_reader(path: str, first_frame: int, last_frame: int) -> Iterator[tuple[int, np.ndarray]]:
    return iter_frames(path, first_frame, last_frame)


def _decode_child(path, first_frame, last_frame, ready, free, n_slots) -> None:
    slots: list[shared_memory.SharedMemory] = []
    released = False
    try:
        for index, (frame, img) in enumerate(iter_frames(path, first_frame, last_frame)):
            if not slots:
                slots = [shared_memory.SharedMemory(create=True, size=img.nbytes) for _ in range(n_slots)]
                ready.put(("slots", [s.name for s in slots], img.shape))
                for i in range(n_slots):
                    free.put(i)
            slot = free.get()
            if slot is None:
                released = True
                return
            view = np.ndarray(img.shape, np.uint8, buffer=slots[slot].buf)
            view[:] = img
            ready.put(("frame", frame, slot))
        ready.put(("end",))
    except BaseException:
        ready.put(("error", traceback.format_exc()))
    finally:
        if not released:
            free.get()                   # the loop has attached to the slots until it sends its release
        for s in slots:
            s.close()


def process_reader(path: str, first_frame: int, last_frame: int) -> Iterator[tuple[int, np.ndarray]]:
    ctx = multiprocessing.get_context("spawn")
    ready, free = ctx.Queue(), ctx.Queue()
    child = ctx.Process(target=_decode_child, args=(path, first_frame, last_frame, ready, free, PROCESS_SLOTS),
                        daemon=True)
    child.start()
    attached: list[shared_memory.SharedMemory] = []
    shape = None

    def receive():
        while True:
            try:
                return ready.get(timeout=_POLL_S)
            except queue.Empty:
                if not child.is_alive():
                    raise RuntimeError(f"decode process for {path} died (exit code {child.exitcode})") from None

    try:
        while True:
            message = receive()
            if message[0] == "slots":
                attached = [shared_memory.SharedMemory(name=name) for name in message[1]]
                shape = message[2]
            elif message[0] == "frame":
                _, frame, slot = message
                img = np.ndarray(shape, np.uint8, buffer=attached[slot].buf).copy()
                free.put(slot)
                yield frame, img
            elif message[0] == "end":
                return
            else:
                raise RuntimeError(f"decode process for {path} failed:\n{message[1]}")
    finally:
        free.put(None)                   # release: the child may close its handles and exit
        child.join(5.0)
        if child.is_alive():
            child.kill()
            child.join()
        for s in attached:
            s.close()


READERS: dict[str, FrameReader] = {"thread": thread_reader, "process": process_reader}


def get_reader(mode: str) -> FrameReader:
    if mode not in READERS:
        raise ValueError(f"unknown decode mode {mode!r}; choose from {sorted(READERS)}")
    return READERS[mode]
