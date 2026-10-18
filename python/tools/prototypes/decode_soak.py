# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""decode_soak.py — does decoding several camera videos at once hang, on threads or in child processes?

Prototype for the detection pipeline's decode producers (docs/roadmap/features/detection-pipeline/
detection-pipeline-design.md, section 3). Closing a PyAV container before the end of the file once hung inside
FFmpeg, and the detection loop opens, decodes and closes several videos in parallel, so the choice between decode
threads and decode processes rests on how often that happens.

Every video gets one worker, all running at once. A worker opens its video at a random place with
``posetrak.detection.frame_readers`` in the chosen mode, decodes a window of ``--window-s`` seconds, closes the
reader, and starts over, until ``--minutes`` have passed. A watchdog looks every few seconds: a worker whose current
step (``open`` until the first frame, ``decode`` between frames, ``close``) has made no progress for ``--hang-s``
seconds is counted as one hang of that step. A hung decode thread cannot be killed, so it is abandoned (it stays
blocked, a leaked thread) and its video gets a new worker; in process mode the reader's own stall timeout
(``--hang-s``) kills the child, raises, and the worker carries on, which is counted as a recovered stall.

    python python/tools/prototypes/decode_soak.py --mode both --minutes 30 VIDEO1 VIDEO2 VIDEO3 VIDEO4 VIDEO5
    python python/tools/prototypes/decode_soak.py --session SESSION.db --capture CAPTURE_ID --mode thread --minutes 5

Prints one JSON object per mode: windows done, frames decoded, frames per second over all workers, hangs per step,
recovered stalls, errors, leaked threads. ``--mode both`` runs thread mode, then process mode.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from posetrak.detection import frame_readers  # noqa: E402


@dataclass
class Worker:
    path: str
    n_frames: int
    fps: float
    thread: threading.Thread | None = None
    step: str = "idle"
    progress_at: float = field(default_factory=time.monotonic)
    counted_hang: bool = False
    abandoned: bool = False


class Soak:
    def __init__(self, mode: str, paths: list[str], window_s: float, hang_s: float, seed: int) -> None:
        self.mode, self.window_s, self.hang_s = mode, window_s, hang_s
        self.rng = random.Random(seed)
        self.rng_lock = threading.Lock()
        self.reader = frame_readers.get_reader(mode)
        frame_readers.STALL_TIMEOUT_S = hang_s
        self.workers = [Worker(p, *probe(p)) for p in paths]
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.windows = self.frames = 0
        self.hangs = {"open": 0, "decode": 0, "close": 0}
        self.recovered_stalls = 0
        self.errors: list[str] = []
        self.leaked = 0

    def start_worker(self, w: Worker) -> None:
        w.abandoned, w.counted_hang = False, False
        w.progress_at = time.monotonic()
        w.thread = threading.Thread(target=self.work, args=(w,), daemon=True)
        w.thread.start()

    def mark(self, w: Worker, step: str) -> None:
        w.step, w.progress_at, w.counted_hang = step, time.monotonic(), False

    def work(self, w: Worker) -> None:
        window = int(self.window_s * w.fps)
        while not self.stop.is_set() and not w.abandoned:
            with self.rng_lock:
                first = self.rng.randrange(0, max(1, w.n_frames - window))
            frames = None
            try:
                self.mark(w, "open")
                frames = self.reader(w.path, first, first + window)
                n = 0
                for _ in frames:
                    if w.abandoned:
                        return
                    n += 1
                    self.mark(w, "decode")
                self.mark(w, "close")
                frames.close()
                with self.lock:
                    self.windows += 1
                    self.frames += n
            except frame_readers.DecodeStalled:
                with self.lock:
                    self.recovered_stalls += 1
            except BaseException as exc:  # noqa: BLE001 -- a failing decode is a finding, not a crash of the soak
                with self.lock:
                    self.errors.append(f"{Path(w.path).name}: {exc!r}")
                time.sleep(1.0)
            finally:
                if frames is not None and not w.abandoned:
                    try:
                        frames.close()
                    except BaseException:  # noqa: BLE001
                        pass
            self.mark(w, "idle")

    def watch(self) -> None:
        while not self.stop.wait(2.0):
            now = time.monotonic()
            for w in self.workers:
                if w.step != "idle" and not w.counted_hang and now - w.progress_at > self.hang_s:
                    with self.lock:
                        self.hangs[w.step] += 1
                    w.counted_hang = True
                    print(f"[{self.mode}] hang in {w.step}: {Path(w.path).name}", file=sys.stderr, flush=True)
                    if self.mode == "thread":
                        # a blocked thread cannot be stopped: leave it, give the video a new worker
                        w.abandoned = True
                        self.leaked += 1
                        fresh = Worker(w.path, w.n_frames, w.fps)
                        self.workers[self.workers.index(w)] = fresh
                        self.start_worker(fresh)

    def run(self, minutes: float) -> dict:
        started = time.monotonic()
        for w in self.workers:
            self.start_worker(w)
        watchdog = threading.Thread(target=self.watch, daemon=True)
        watchdog.start()
        while time.monotonic() - started < minutes * 60:
            time.sleep(1.0)
        self.stop.set()
        for w in self.workers:
            if w.thread is not None:
                w.thread.join(self.hang_s + 5)
        elapsed = time.monotonic() - started
        return {
            "mode": self.mode, "videos": [Path(w.path).name for w in self.workers], "minutes": round(elapsed / 60, 2),
            "windows": self.windows, "frames": self.frames, "frames_per_s": round(self.frames / elapsed, 1),
            "hangs": self.hangs, "recovered_stalls": self.recovered_stalls, "errors": self.errors[:20],
            "error_count": len(self.errors), "leaked_threads": self.leaked,
        }


def probe(path: str) -> tuple[int, float]:
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        n = stream.frames or int(float(stream.duration * stream.time_base) * fps)
    return n, fps


def session_videos(session: str, capture: str) -> list[str]:
    conn = sqlite3.connect(f"file:{session}?mode=ro", uri=True)
    rows = conn.execute("SELECT file_path FROM capture_videos WHERE shot_id LIKE ? ORDER BY camera_instance_id",
                        (capture + "%",)).fetchall()
    return [r[0] for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--session")
    ap.add_argument("--capture", help="capture id (prefix) whose videos are used with --session")
    ap.add_argument("--mode", choices=["thread", "process", "both"], default="both")
    ap.add_argument("--minutes", type=float, default=30.0, help="per mode")
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--hang-s", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    paths = a.videos or (session_videos(a.session, a.capture) if a.session and a.capture else [])
    if not paths:
        ap.error("give video files, or --session and --capture")
    for mode in (["thread", "process"] if a.mode == "both" else [a.mode]):
        print(json.dumps(Soak(mode, paths, a.window_s, a.hang_s, a.seed).run(a.minutes)), flush=True)


if __name__ == "__main__":
    main()
