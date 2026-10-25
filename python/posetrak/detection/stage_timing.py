# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""stage_timing.py — opt-in wall-time accounting for the stages of the detection loop.

Set ``POSETRAK_PROFILE_DETECTION`` before the process starts to switch it on: ``1`` prints a table at exit,
any other value is taken as the path of a JSON file that receives the same numbers. With the variable unset
nothing changes: :func:`stage` returns a shared no-op context manager, and :func:`timed` and
:func:`timed_iter` return their argument itself, so the decorated functions and iterators are the original
objects.

Times are exclusive: while a nested stage runs, its parent stops counting, so the totals add up to the time
spent in stages. :func:`section` prefixes the names of the stages entered inside it (for example the
``hand_pass`` section's decode is reported as ``hand_pass/decode``). Stage names used by the detection
loop: ``decode``, ``box_source``, ``crop_preprocess``, ``inference``, ``heatmap_decode``, ``mask_gate``,
``hand_model``, ``preview_encode``, ``db_write``.
"""
from __future__ import annotations

import atexit
import contextlib
import functools
import json
import os
import sys
import threading
import time

ENV_VAR = "POSETRAK_PROFILE_DETECTION"
_value = os.environ.get(ENV_VAR, "")
ENABLED = bool(_value) and _value != "0"

_NOOP = contextlib.nullcontext()
_lock = threading.Lock()
_totals: dict[str, list[float]] = {}  # name -> [seconds, calls]
_local = threading.local()
_first_use: float | None = None


def _stack() -> list:
    if not hasattr(_local, "stack"):
        _local.stack, _local.prefix = [], ""
    return _local.stack


def _record(name: str, seconds: float) -> None:
    with _lock:
        entry = _totals.setdefault(name, [0.0, 0])
        entry[0] += seconds
        entry[1] += 1


class _Stage:
    __slots__ = ("raw", "name", "started", "elapsed")

    def __init__(self, name: str) -> None:
        self.raw = name

    def __enter__(self):
        global _first_use
        now = time.perf_counter()
        _first_use = _first_use or now
        stack = _stack()
        if stack:
            parent = stack[-1]
            parent.elapsed += now - parent.started
        self.name = _local.prefix + self.raw
        self.started, self.elapsed = now, 0.0
        stack.append(self)
        return self

    def __exit__(self, *exc) -> None:
        now = time.perf_counter()
        stack = _stack()
        stack.pop()
        _record(self.name, self.elapsed + now - self.started)
        if stack:
            stack[-1].started = now


def stage(name: str):
    """Context manager that adds the time spent inside to stage *name* (no-op when profiling is off)."""
    return _Stage(name) if ENABLED else _NOOP


@contextlib.contextmanager
def _section(name: str):
    _stack()
    previous = _local.prefix
    _local.prefix = previous + name + "/"
    try:
        yield
    finally:
        _local.prefix = previous


def section(name: str):
    """Context manager that prefixes the stages entered inside it with ``name/`` (no-op when profiling is off)."""
    return _section(name) if ENABLED else _NOOP


def timed(name: str):
    """Decorator that times every call as stage *name*; returns the function itself when profiling is off."""

    def wrap(fn):
        if not ENABLED:
            return fn

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            with stage(name):
                return fn(*args, **kwargs)

        return inner

    return wrap


def timed_iter(name: str, iterator):
    """Iterator that times each step of *iterator* as stage *name*; *iterator* itself when profiling is off."""
    if not ENABLED:
        return iterator

    def gen():
        it = iter(iterator)
        while True:
            with stage(name):
                try:
                    item = next(it)
                except StopIteration:
                    return
            yield item

    return gen()


def instrument_rtmlib_model(model):
    """Time the three steps of an rtmlib pose model call as crop_preprocess, inference and heatmap_decode.

    Replaces the model instance's methods only when profiling is on.
    """
    if ENABLED:
        for method, name in (("preprocess", "crop_preprocess"), ("inference", "inference"), ("postprocess", "heatmap_decode")):
            setattr(model, method, timed(name)(getattr(model, method)))
    return model


def snapshot() -> dict:
    """``{stage: {"seconds", "calls"}}`` plus ``"_wall_s"``, the time since the first stage was entered."""
    with _lock:
        out = {k: {"seconds": v[0], "calls": int(v[1])} for k, v in sorted(_totals.items())}
    out["_wall_s"] = time.perf_counter() - _first_use if _first_use else 0.0
    return out


def reset() -> None:
    """Forget all accumulated times."""
    global _first_use
    with _lock:
        _totals.clear()
    _first_use = None


def format_table(snap: dict) -> str:
    wall = snap.get("_wall_s", 0.0)
    stages = {k: v for k, v in snap.items() if not k.startswith("_")}
    lines = [f"{'stage':28s} {'seconds':>9s} {'calls':>9s} {'ms/call':>9s} {'share':>7s}"]
    for name, v in sorted(stages.items(), key=lambda kv: -kv[1]["seconds"]):
        lines.append(f"{name:28s} {v['seconds']:9.2f} {v['calls']:9d} {1000 * v['seconds'] / max(v['calls'], 1):9.2f} "
                     f"{100 * v['seconds'] / wall if wall else 0:6.1f}%")
    accounted = sum(v["seconds"] for v in stages.values())
    lines.append(f"{'(outside stages)':28s} {max(wall - accounted, 0.0):9.2f}")
    lines.append(f"{'wall since first stage':28s} {wall:9.2f}")
    return "\n".join(lines)


def _report_at_exit() -> None:
    snap = snapshot()
    if len(snap) <= 1:
        return
    print("detection stage timing:\n" + format_table(snap), file=sys.stderr)
    if _value not in ("1", "true", "yes"):
        with open(_value, "w") as f:
            json.dump(snap, f, indent=1)


if ENABLED:
    atexit.register(_report_at_exit)
