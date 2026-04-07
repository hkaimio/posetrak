"""Regression tests for run_led_sync against real brightness data dumps.

These tests load the .npz files recorded from the two Timo/Harri sessions
(20260403) and verify that the LED sync algorithm produces correct per-camera
global-time mappings.

Ground-truth anchor
-------------------
Both sessions captured the same subject.  At one sync-flash moment the cameras
were at these frames:

    Camera          Frame    FPS
    ace2pro (ref)   2307     119.88   → T_sync = 2307 / 119.88 ≈ 19.244 s
    gopromini-01    5768     119.88   → rough_offset ≈ -28.87 s
    gopromini-02    8868     119.88   → rough_offset ≈ -54.73 s
    instax3         5476      59.94   → rough_offset ≈ -72.10 s
    pixel9           862     118.88   → rough_offset ≈ +11.99 s
    r5              2067      59.94   → rough_offset ≈ -15.24 s

Camera index order in the NPZ files: [ace2pro, gopromini-01, gopromini-02,
instax3, pixel9, r5] (index 0 = reference).

All rough-offset values are derived from the anchor frames above; the stored
rough_offsets in session 1 are **wrong** (only pixel9 was anchored when the
dump was recorded).  Session 2 has the correct stored values.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from app.setup.led_sync import load_brightness_dump, run_led_sync

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[3]
_TEST_DATA_DIR = _REPO_ROOT / "tests/data"
_NPZ_SESSION_1 = _TEST_DATA_DIR / "led_brightness_01.npz"
_NPZ_SESSION_2 = _TEST_DATA_DIR / "led_brightness_02.npz"

# ---------------------------------------------------------------------------
# Ground-truth anchor
# ---------------------------------------------------------------------------

# Camera order matches the NPZ cam_ids array (index 0 = ace2pro = reference).
_ANCHOR_FRAMES = [2307, 5768, 8868, 5476, 862, 2067]
_ANCHOR_FPS    = [119.88, 119.88, 119.88, 59.94, 118.88, 59.94]

# Global time at the anchor moment equals ace2pro's local time at frame 2307.
_T_SYNC = _ANCHOR_FRAMES[0] / _ANCHOR_FPS[0]  # ≈ 19.244 s

# Frame period for the slowest camera — used as reference tolerance budget.
_SLOW_FPS = 59.94
_FRAME_PERIOD_S = 1.0 / _SLOW_FPS  # ≈ 16.7 ms


def _correct_rough_offsets(fps_list: list[float]) -> list[float]:
    """Compute rough_offsets from ground-truth anchor frames.

    For the reference camera (index 0) the offset is exactly 0.  For every
    other camera k:

        rough_offset[k] = T_sync − anchor_frame[k] / fps_list[k]
    """
    offsets = []
    for k, fps in enumerate(fps_list):
        if k == 0:
            offsets.append(0.0)
        else:
            offsets.append(_T_SYNC - _ANCHOR_FRAMES[k] / fps)
    return offsets


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"NPZ data file not found: {path}")


@pytest.fixture(scope="module")
def session1_data():
    """Load session 1 NPZ and override rough_offsets with correct anchor values."""
    _skip_if_missing(_NPZ_SESSION_1)
    d = load_brightness_dump(str(_NPZ_SESSION_1))
    d["rough_offsets"] = _correct_rough_offsets(d["fps_list"])
    return d


@pytest.fixture(scope="module")
def session2_data():
    """Load session 2 NPZ; stored rough_offsets are already correct."""
    _skip_if_missing(_NPZ_SESSION_2)
    return load_brightness_dump(str(_NPZ_SESSION_2))


@pytest.fixture(scope="module")
def session1_result(session1_data):
    return run_led_sync(**session1_data)


@pytest.fixture(scope="module")
def session2_result(session2_data):
    return run_led_sync(**session2_data)


# ---------------------------------------------------------------------------
# Data-integrity tests (never require algorithm fixes)
# ---------------------------------------------------------------------------


def test_session1_npz_loads(session1_data) -> None:
    d = session1_data
    assert len(d["signals"]) == 6
    assert len(d["fps_list"]) == 6
    assert d["ref_cam"] == 0


def test_session2_npz_loads(session2_data) -> None:
    d = session2_data
    assert len(d["signals"]) == 6
    assert len(d["fps_list"]) == 6
    assert d["ref_cam"] == 0


def test_session2_stored_rough_offsets_match_anchor() -> None:
    """Session 2's stored rough_offsets should already be close to anchor-derived values."""
    _skip_if_missing(_NPZ_SESSION_2)
    d = load_brightness_dump(str(_NPZ_SESSION_2))
    stored = d["rough_offsets"]
    correct = _correct_rough_offsets(d["fps_list"])
    for k, (s, c) in enumerate(zip(stored, correct)):
        assert abs(s - c) < 0.5, (
            f"Camera {k}: stored rough_offset {s:.3f} s differs from "
            f"anchor-derived {c:.3f} s by {abs(s - c)*1000:.0f} ms"
        )


def test_session1_stored_rough_offsets_are_wrong() -> None:
    """Session 1 NPZ stores wrong rough_offsets (all cameras got pixel9's anchor)."""
    _skip_if_missing(_NPZ_SESSION_1)
    d = load_brightness_dump(str(_NPZ_SESSION_1))
    correct = _correct_rough_offsets(d["fps_list"])
    # At least two non-reference cameras must have large errors
    wrong_count = sum(
        1 for k in range(1, 6) if abs(d["rough_offsets"][k] - correct[k]) > 5.0
    )
    assert wrong_count >= 4, (
        f"Expected ≥4 wrong offsets in session 1, got {wrong_count}; "
        f"stored={d['rough_offsets']}, correct={correct}"
    )


# ---------------------------------------------------------------------------
# Algorithm output structure tests
# ---------------------------------------------------------------------------


def _check_result_structure(result, n_cameras: int) -> None:
    assert len(result.cameras) == n_cameras
    for k, cr in enumerate(result.cameras):
        assert len(cr.frame_times) > 0, f"cam {k}: empty frame_times"
        assert np.all(np.isfinite(cr.frame_times)), f"cam {k}: non-finite frame_times"
        assert len(cr.brightness) > 0, f"cam {k}: empty brightness"


def test_session1_result_structure(session1_result) -> None:
    _check_result_structure(session1_result, 6)


def test_session2_result_structure(session2_result) -> None:
    _check_result_structure(session2_result, 6)


def test_session1_frame_times_length(session1_data, session1_result) -> None:
    for k, (sig, cr) in enumerate(zip(session1_data["signals"], session1_result.cameras)):
        assert len(cr.frame_times) == len(sig), (
            f"cam {k}: frame_times length {len(cr.frame_times)} != signal length {len(sig)}"
        )


def test_session2_frame_times_length(session2_data, session2_result) -> None:
    for k, (sig, cr) in enumerate(zip(session2_data["signals"], session2_result.cameras)):
        assert len(cr.frame_times) == len(sig), (
            f"cam {k}: frame_times length {len(cr.frame_times)} != signal length {len(sig)}"
        )


@pytest.mark.xfail(
    reason=(
        "PCHIP interpolation can produce non-monotonic frame_times when "
        "RANSAC inlier set has a poor temporal spread.  "
        "Fix: enforce monotonicity in build_time_map, or add monotone post-processing."
    ),
    strict=False,
)
def test_session1_frame_times_monotonic(session1_result) -> None:
    for k, cr in enumerate(session1_result.cameras):
        diffs = np.diff(cr.frame_times)
        assert np.all(diffs >= 0), (
            f"cam {k}: frame_times not monotonic; "
            f"first non-monotonic index: {np.argmax(diffs < 0)}"
        )


@pytest.mark.xfail(
    reason=(
        "PCHIP interpolation can produce non-monotonic frame_times when "
        "RANSAC inlier set has a poor temporal spread.  "
        "Fix: enforce monotonicity in build_time_map, or add monotone post-processing."
    ),
    strict=False,
)
def test_session2_frame_times_monotonic(session2_result) -> None:
    for k, cr in enumerate(session2_result.cameras):
        diffs = np.diff(cr.frame_times)
        assert np.all(diffs >= 0), (
            f"cam {k}: frame_times not monotonic; "
            f"first non-monotonic index: {np.argmax(diffs < 0)}"
        )


def test_reference_camera_identity_map(session1_result) -> None:
    cr = session1_result.cameras[0]
    assert cr.map_type == "reference"
    fps = 119.88
    expected = np.arange(len(cr.frame_times)) / fps
    np.testing.assert_allclose(cr.frame_times, expected, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# Anchor-frame global-time agreement
# ---------------------------------------------------------------------------
#
# The core requirement: all cameras were aimed at the same LED flash, so
# frame_times[anchor_frame_k] must all equal the same global timestamp T_sync.
# We test this at two tolerance levels:
#
#   LOOSE  (1.0 s)  — verifiable with any plausible rough_offset-based sync
#   TIGHT  (1 frame period ≈ 16.7 ms at 60 fps)  — goal after algorithm fix


def _anchor_time_errors(result, fps_list: list[float]) -> list[float]:
    """Return |frame_times[anchor_frame_k] - T_sync| for each camera."""
    errors = []
    for k, cr in enumerate(result.cameras):
        frame_idx = _ANCHOR_FRAMES[k]
        if frame_idx >= len(cr.frame_times):
            errors.append(float("inf"))
            continue
        t_global = cr.frame_times[frame_idx]
        errors.append(abs(t_global - _T_SYNC))
    return errors


def test_session1_anchor_times_loose(session1_result, session1_data) -> None:
    """With correct rough_offsets, session 1 anchor frames should agree within 1 s."""
    errors = _anchor_time_errors(session1_result, session1_data["fps_list"])
    for k, err in enumerate(errors):
        assert err < 1.0, (
            f"cam {k} ({session1_data['cam_ids'][k]}): "
            f"anchor frame maps to {session1_result.cameras[k].frame_times[_ANCHOR_FRAMES[k]]:.3f} s, "
            f"expected {_T_SYNC:.3f} s, error = {err * 1000:.0f} ms"
        )


@pytest.mark.xfail(
    reason=(
        "Session 2's reference camera (ace2pro) has only ~56 detected LED events "
        "out of ~427 s of footage.  With so few reference anchors the NN matching "
        "produces only ~11 pairs and RANSAC produces a degenerate a≈0 model that "
        "collapses all camera 1 (gopromini-01) frame_times to a single constant.  "
        "Fix: detect LED period from the camera with the most events, or allow "
        "choosing a non-zero-indexed reference for matching."
    ),
    strict=False,
)
def test_session2_anchor_times_loose(session2_result, session2_data) -> None:
    """With correct rough_offsets, session 2 anchor frames should agree within 1 s."""
    errors = _anchor_time_errors(session2_result, session2_data["fps_list"])
    for k, err in enumerate(errors):
        assert err < 1.0, (
            f"cam {k} ({session2_data['cam_ids'][k]}): "
            f"anchor frame maps to {session2_result.cameras[k].frame_times[_ANCHOR_FRAMES[k]]:.3f} s, "
            f"expected {_T_SYNC:.3f} s, error = {err * 1000:.0f} ms"
        )


@pytest.mark.xfail(
    reason="Tight anchor-time agreement requires 2-pass iterative NN+RANSAC fix",
    strict=False,
)
def test_session1_anchor_times_tight(session1_result, session1_data) -> None:
    """All cameras' anchor frames must map to T_sync within one slow-camera frame period."""
    errors = _anchor_time_errors(session1_result, session1_data["fps_list"])
    for k, err in enumerate(errors):
        assert err < _FRAME_PERIOD_S, (
            f"cam {k} ({session1_data['cam_ids'][k]}): "
            f"error {err * 1000:.1f} ms > {_FRAME_PERIOD_S * 1000:.1f} ms"
        )


@pytest.mark.xfail(
    reason="Tight anchor-time agreement requires 2-pass iterative NN+RANSAC fix",
    strict=False,
)
def test_session2_anchor_times_tight(session2_result, session2_data) -> None:
    """All cameras' anchor frames must map to T_sync within one slow-camera frame period."""
    errors = _anchor_time_errors(session2_result, session2_data["fps_list"])
    for k, err in enumerate(errors):
        assert err < _FRAME_PERIOD_S, (
            f"cam {k} ({session2_data['cam_ids'][k]}): "
            f"error {err * 1000:.1f} ms > {_FRAME_PERIOD_S * 1000:.1f} ms"
        )


# ---------------------------------------------------------------------------
# Minimum inlier count
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "Session 2 cam1 (gopromini-01) gets only 7 inliers because the reference "
        "camera has only ~56 events, limiting NN matches to ~11 pairs.  "
        "Fix: use the camera with the most events as the matching reference, "
        "or implement 2-pass refinement so NN uses a tighter window after an "
        "initial RANSAC offset estimate."
    ),
    strict=False,
)
def test_session2_minimum_inliers(session2_result, session2_data) -> None:
    """Every non-reference camera must produce at least 10 RANSAC inliers."""
    for k, cr in enumerate(session2_result.cameras):
        if k == session2_data["ref_cam"]:
            continue
        assert cr.n_inliers >= 10, (
            f"cam {k} ({session2_data['cam_ids'][k]}): "
            f"only {cr.n_inliers} inliers — too few for reliable sync"
        )


@pytest.mark.xfail(
    reason=(
        "Session 1 cam3 (instax3) gets only 7 inliers due to poor NN matching "
        "with the 1-pass algorithm.  Fix: 2-pass iterative NN+RANSAC."
    ),
    strict=False,
)
def test_session1_minimum_inliers(session1_result, session1_data) -> None:
    """Every non-reference camera must produce at least 10 RANSAC inliers."""
    for k, cr in enumerate(session1_result.cameras):
        if k == session1_data["ref_cam"]:
            continue
        assert cr.n_inliers >= 10, (
            f"cam {k} ({session1_data['cam_ids'][k]}): "
            f"only {cr.n_inliers} inliers — too few for reliable sync"
        )


# ---------------------------------------------------------------------------
# Reference-camera event count
# ---------------------------------------------------------------------------


def test_session2_reference_has_few_events(session2_data) -> None:
    """Document that session 2's reference camera (ace2pro) has very few LED events.

    With only ~56 events in a 427 s recording, the NN matching to any other
    camera produces very few pairs, causing RANSAC to fit a degenerate a≈0
    model that collapses all frame_times to a single constant.

    Expected behaviour once fixed: the algorithm should either (a) pick a
    better reference, or (b) use a 2-pass approach to recover from sparse
    reference events.
    """
    from app.setup.led_sync import detect_events
    sig = session2_data["signals"][session2_data["ref_cam"]]
    fps = session2_data["fps_list"][session2_data["ref_cam"]]
    events = detect_events(sig, fps)
    duration_s = len(sig) / fps
    # Document the sparsity — this is expected to be a very low number
    events_per_minute = len(events) / (duration_s / 60.0)
    assert events_per_minute < 15, (
        f"Ref camera now has {events_per_minute:.1f} events/min — "
        f"much denser than expected; test may need updating"
    )
    # Hard minimum: fewer than 200 events across the whole recording
    assert len(events) < 200, (
        f"Ref camera has {len(events)} events in {duration_s:.0f} s — "
        f"denser than the known sparse case; check if signal changed"
    )


# ---------------------------------------------------------------------------
# Diagnostic: print anchor-time errors for both sessions (always runs)
# ---------------------------------------------------------------------------


def test_print_anchor_time_errors(session1_result, session1_data,
                                   session2_result, session2_data) -> None:
    """Non-failing diagnostic test — logs anchor-time errors to pytest output."""
    for label, result, data in [
        ("session1", session1_result, session1_data),
        ("session2", session2_result, session2_data),
    ]:
        errors = _anchor_time_errors(result, data["fps_list"])
        lines = [f"\n{label} anchor-time errors:"]
        for k, (err, cr) in enumerate(zip(errors, result.cameras)):
            lines.append(
                f"  cam{k} {data['cam_ids'][k]}: "
                f"error={err * 1000:6.1f} ms  "
                f"n_inliers={cr.n_inliers:4d}  "
                f"map_type={cr.map_type}"
            )
        _log.warning("\n".join(lines))  # always visible with -s or in captured output
