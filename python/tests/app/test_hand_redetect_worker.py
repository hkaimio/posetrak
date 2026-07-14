"""Tests for app.ui.hand_redetect_worker — Idea 3 (automated post-edit hand
redetection).

Only `redetect_hand`, the small pure function pulled out of
`HandRedetectWorker` specifically to be testable, is unit-tested here.
Per this codebase's established convention (see `test_wide_crop_cache.py`'s
own docstring), a QThread's actual decode/queue mechanics are validated
manually once wired into the editor (Phase 4/5 of the implementation plan),
not unit-tested -- the same split already used for `CropBackfillWorker`/
`WideCropExtractWorker`.
"""
from __future__ import annotations

import numpy as np
import pytest

from posetrak.detection.hand_refinement import HandCandidate, _HAND_CONF_SCALE, _HAND_N_KP, _HAND_POSE_INPUT_WIDTH

from app.ui.hand_redetect_worker import redetect_hand


def _uniform_hand_kp(xy: tuple[float, float]) -> np.ndarray:
    return np.tile(np.array(xy, dtype=np.float32), (21, 1))


def _fake_candidate(xy: tuple[float, float], conf: float = 0.8, crop_w_px: float = 120.0) -> HandCandidate:
    return HandCandidate(
        keypoints=_uniform_hand_kp(xy),
        scores=np.full(21, conf, dtype=np.float32),
        root_dist_px=1.0,
        crop_w_px=crop_w_px,
    )


def test_returns_hand_kp_and_noise_scale_on_gate_pass(monkeypatch):
    fake_result = _fake_candidate((201.0, 199.0), conf=0.8, crop_w_px=128.0)
    calls = []

    def fake_detect(hand_model, image, wrist, elbow):
        calls.append((hand_model, wrist, elbow))
        return fake_result

    monkeypatch.setattr("posetrak.detection.hand_refinement.detect_hand_in_crop", fake_detect)

    img = np.zeros((400, 400, 3), dtype=np.uint8)
    sentinel_model = object()
    result = redetect_hand(sentinel_model, img, (200.0, 200.0), (150.0, 200.0))

    assert calls == [(sentinel_model, (200.0, 200.0), (150.0, 200.0))]
    assert result is not None
    hand_kp, noise_scale = result
    assert hand_kp.shape == (_HAND_N_KP, 3)
    np.testing.assert_allclose(hand_kp[:, 0], 201.0)
    np.testing.assert_allclose(hand_kp[:, 1], 199.0)
    np.testing.assert_allclose(hand_kp[:, 2], 0.8 * _HAND_CONF_SCALE)
    assert noise_scale == pytest.approx(128.0 / _HAND_POSE_INPUT_WIDTH)


def test_returns_none_on_gate_reject(monkeypatch):
    monkeypatch.setattr(
        "posetrak.detection.hand_refinement.detect_hand_in_crop", lambda *a, **k: None,
    )
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    assert redetect_hand(object(), img, (200.0, 200.0), None) is None


def test_works_without_a_confident_elbow(monkeypatch):
    """elbow=None must reach detect_hand_in_crop unchanged -- its own
    wrist-centred-floor-crop fallback (already tested in
    test_hand_refinement.py) handles that case, not this wrapper."""
    calls = []

    def fake_detect(hand_model, image, wrist, elbow):
        calls.append(elbow)
        return None

    monkeypatch.setattr("posetrak.detection.hand_refinement.detect_hand_in_crop", fake_detect)
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    redetect_hand(object(), img, (200.0, 200.0), None)
    assert calls == [None]
