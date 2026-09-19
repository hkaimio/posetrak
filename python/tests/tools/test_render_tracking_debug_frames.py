# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for render_tracking_debug_frames.py's _nearest_state_row() --
a standalone visualization tool, with no unit tests for its DB-loading parts
(same precedent as calibrate_rigid_marker_body.py), but one failure mode is
serious enough to cover directly: RTS-smoothed
tracking_results rows use a *different* tracker_step numbering than raw
ones (smoothing only ever covers steps that were actually tracked, so a
gap-heavy run's smoothed index runs ahead of the raw one by however many
steps were lost before it), but tracking_obs_results is only ever written
from the raw forward pass -- so reusing a smoothed row's tracker_step to
look up tracking_obs_results silently fetches a *different, real*
instant's observation data and overlays it on the wrong video frame.
Confirmed on a real run: raw tracker_step 1933 and smoothed tracker_step
1603 both carry timestamp_s=53.766 -- reusing 1603 to key into
tracking_obs_results actually fetches data from timestamp_s=50.466 instead.

_nearest_state_row() takes an explicit is_smoothed flag rather than always
returning the raw row (its predecessor, _nearest_tracker_step()'s, fix) --
the tool now defaults to showing the smoothed (delivered) state and only
falls back to raw via --raw, but tracking_obs_results is always looked up
via the raw row regardless (see render_tracking_debug_frames.py's own
_frame_data_for_time() for the two separate lookups this enables).
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "render_tracking_debug_frames.py"
)
_spec = importlib.util.spec_from_file_location("render_tracking_debug_frames", _MODULE_PATH)
render_tracking_debug_frames = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = render_tracking_debug_frames
_spec.loader.exec_module(render_tracking_debug_frames)

_nearest_state_row = render_tracking_debug_frames._nearest_state_row
_robust_bounds = render_tracking_debug_frames._robust_bounds
_sequential_frame_lookup = render_tracking_debug_frames._sequential_frame_lookup


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE tracking_results ("
        " run_id TEXT, person_id INTEGER, tracker_step INTEGER,"
        " is_smoothed INTEGER, timestamp_s REAL, state BLOB)"
    )
    return conn


def test_nearest_state_row_never_returns_the_other_flavors_step_number():
    """The real bug: a smoothed row's tracker_step (1603) is a different
    numbering than the raw row that actually shares its real timestamp
    (1933) -- is_smoothed must be an explicit, required choice, not an
    accident of which row a naive nearest-timestamp query happens to hit.
    """
    conn = _make_conn()
    rows = [
        ("run1", 0, 1603, 0, 50.466, b"raw-1603"),  # raw: step 1603 is an EARLIER instant
        ("run1", 0, 1603, 1, 53.766, b"smoothed-1603"),  # smoothed: step 1603 reused, later instant
        ("run1", 0, 1933, 0, 53.766, b"raw-1933"),  # raw: this is the real step for 53.766
        ("run1", 0, 1933, 1, 58.406, b"smoothed-1933"),
    ]
    conn.executemany("INSERT INTO tracking_results VALUES (?,?,?,?,?,?)", rows)

    raw_step, raw_ts, raw_state = _nearest_state_row(conn, "run1", 53.766, is_smoothed=False)
    smoothed_step, smoothed_ts, smoothed_state = _nearest_state_row(
        conn, "run1", 53.766, is_smoothed=True
    )

    assert (raw_step, raw_ts, raw_state) == (1933, 53.766, b"raw-1933")
    assert (smoothed_step, smoothed_ts, smoothed_state) == (1603, 53.766, b"smoothed-1603")


def test_nearest_state_row_picks_the_closest_row_of_the_requested_flavor():
    conn = _make_conn()
    rows = [
        ("run1", 0, 10, 0, 1.0, b"a"),
        ("run1", 0, 11, 0, 2.0, b"b"),
        ("run1", 0, 12, 0, 3.0, b"c"),
        ("run1", 0, 10, 1, 2.05, b"smoothed-close"),  # closer to 2.1, but wrong flavor
    ]
    conn.executemany("INSERT INTO tracking_results VALUES (?,?,?,?,?,?)", rows)

    step, _ts, state = _nearest_state_row(conn, "run1", 2.1, is_smoothed=False)

    assert step == 11
    assert state == b"b"


def test_nearest_state_row_returns_none_with_no_matching_run():
    conn = _make_conn()
    assert _nearest_state_row(conn, "no-such-run", 1.0, is_smoothed=False) is None
    assert _nearest_state_row(conn, "no-such-run", 1.0, is_smoothed=True) is None


# ---------------------------------------------------------------------------
# _robust_bounds() -- the crop-window fix: a strict min/max over
# a whole clip's points lets a single wild outlier (a false-positive raw-dot
# candidate anywhere in frame, or a predicted marker reprojected far from the
# subject during high state uncertainty) blow up the crop for the entire
# clip, exactly the moments this tool exists to zoom in on.
# ---------------------------------------------------------------------------

def test_robust_bounds_ignores_a_single_wild_outlier_with_enough_points():
    # A realistic point count for a real clip (hundreds of frames x several
    # markers) -- tightly clustered, plus one wild point (e.g. a false-
    # positive raw-dot candidate near a door, far from the real subject).
    values = [100.0 + i * 0.01 for i in range(200)]  # tightly clustered 100.0-101.99
    values.append(5000.0)

    lo, hi = _robust_bounds(values)

    assert 99.0 <= lo <= 101.0  # near the tight cluster, not pulled down by anything
    assert 100.0 <= hi <= 200.0  # nowhere near the 5000.0 outlier


def test_robust_bounds_falls_back_to_min_max_with_too_few_points():
    values = [10.0, 5000.0, 12.0]  # too few for percentiles to mean anything
    assert _robust_bounds(values) == (10.0, 5000.0)


# ---------------------------------------------------------------------------
# _sequential_frame_lookup() -- the grid-render performance fix:
# _grab_frame() reopens and reseeks the whole video container on every call,
# which makes a 6-camera grid render extremely slow (I/O-bound repeated
# seeking through external-drive 4K footage) despite near-zero CPU use.
# _sequential_frame_lookup() should decode each camera's needed range with
# exactly one pass, not one open-and-seek per requested timestamp.
# ---------------------------------------------------------------------------

class _FakeSyncTable:
    def __init__(self, mapping: dict[float, int | None]) -> None:
        self._mapping = mapping

    def lookup(self, t: float, svid: str) -> int | None:
        return self._mapping.get(t)


class _FakeCtx:
    def __init__(self, mapping: dict[float, int | None]) -> None:
        self.sync_table = _FakeSyncTable(mapping)
        self.svid = "svid"
        self.file_path = "unused.mp4"


def test_sequential_frame_lookup_decodes_the_needed_range_exactly_once(monkeypatch):
    available = [(10, "img10"), (11, "img11"), (12, "img12"), (13, "img13"), (14, "img14")]
    calls: list[tuple[int, int]] = []

    def fake_iter_frames(path, first, last):
        calls.append((first, last))
        for fi, img in available:
            if first <= fi < last:
                yield fi, img

    monkeypatch.setattr(render_tracking_debug_frames, "iter_frames", fake_iter_frames)

    ctx = _FakeCtx({1.0: 10, 2.0: 12, 3.0: 14})
    results = list(_sequential_frame_lookup(ctx, [1.0, 2.0, 3.0]))

    assert results == [(10, "img10"), (12, "img12"), (14, "img14")]
    assert calls == [(10, 15)]  # exactly one decode pass over [min, max], not one per frame


def test_sequential_frame_lookup_returns_none_for_a_timestamp_with_no_sync_data(monkeypatch):
    available = [(10, "img10"), (11, "img11")]
    monkeypatch.setattr(
        render_tracking_debug_frames, "iter_frames",
        lambda path, first, last: (f for f in available if first <= f[0] < last),
    )

    ctx = _FakeCtx({1.0: 10, 2.0: None, 3.0: 11})
    results = list(_sequential_frame_lookup(ctx, [1.0, 2.0, 3.0]))

    assert results == [(10, "img10"), (None, None), (11, "img11")]


def test_sequential_frame_lookup_never_opens_the_video_when_nothing_has_sync_data(monkeypatch):
    def fail_iter_frames(path, first, last):
        raise AssertionError("iter_frames should not be called with no valid timestamps")

    monkeypatch.setattr(render_tracking_debug_frames, "iter_frames", fail_iter_frames)

    ctx = _FakeCtx({1.0: None, 2.0: None})
    results = list(_sequential_frame_lookup(ctx, [1.0, 2.0]))
    assert results == [(None, None), (None, None)]
