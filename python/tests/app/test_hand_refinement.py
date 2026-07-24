"""Tests for posetrak.detection.hand_refinement — Idea 2.

Covers the pure crop/candidate-selection/gate logic in detect_hand_in_crop
and the hand-row-building/pose-model-gating logic in HandRefinementPipeline,
using a fake hand model so no rtmlib checkpoint download is required. See
docs/roadmap/features/hand-detection-refinement/hand-detection-refinement-design.md
for the validated formulas these tests pin down.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from posetrak.db.db import create_session
from posetrak.detection.pipeline import CameraInfo
from app.pose.db_cache import create_detection_run
from posetrak.detection.hand_refinement import (
    HandCandidate,
    HandRefinementPipeline,
    _ELBOW_IDX,
    _HAND_CONF_SCALE,
    _HAND_N_KP,
    _HAND_POSE_INPUT_WIDTH,
    _WRIST_IDX,
    detect_hand_in_crop,
)


_SHOT_ID = "test-shot-id"
_SYNC_ID = "test-sync-id"
_SVID = "test-sv-id"
_CAM_ID = "test-cam-id"


@pytest.fixture
def session(tmp_path):
    db_path = tmp_path / "test.db"
    conn = create_session(db_path)
    conn.row_factory = sqlite3.Row
    from posetrak.db.db import generate_id
    session_id = generate_id()
    conn.executescript(f"""
        INSERT INTO mocap_sessions (id, recorded_at) VALUES ('{session_id}', '2026-01-01');
        INSERT INTO captures (id, session_id, capture_number, label)
            VALUES ('{_SHOT_ID}', '{session_id}', 1, 'test');
        INSERT INTO sync_configs (id, shot_id, created_by)
            VALUES ('{_SYNC_ID}', '{_SHOT_ID}', 'test');
        INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,
                                 first_video_frame, last_video_frame, actual_fps)
            VALUES ('{_SVID}', '{_SHOT_ID}', '{_CAM_ID}', '/fake/video.mp4', 0, 1000, 120.0);
        INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id,
                                 video_frame, timestamp_s)
            VALUES ('{_SYNC_ID}', '{_CAM_ID}', '{_SVID}', 0, 0.0);
    """)
    conn.commit()
    return conn


class _FakeHandModel:
    """Returns a fixed set of (root_local, kp21_local, scores21) candidates."""

    def __init__(self, candidates):
        self._candidates = candidates
        self.calls: list[tuple[int, int, int]] = []

    def __call__(self, crop):
        self.calls.append(crop.shape)
        if not self._candidates:
            return np.zeros((0, 21, 2), dtype=np.float32), np.zeros((0, 21), dtype=np.float32)
        keypoints = np.stack([c[1] for c in self._candidates]).astype(np.float32)
        scores = np.stack([c[2] for c in self._candidates]).astype(np.float32)
        return keypoints, scores


def _uniform_hand_kp(xy: tuple[float, float]) -> np.ndarray:
    return np.tile(np.array(xy, dtype=np.float32), (21, 1))


def _fake_candidate(xy: tuple[float, float], conf: float = 0.8, crop_w_px: float = 120.0) -> HandCandidate:
    return HandCandidate(
        keypoints=_uniform_hand_kp(xy),
        scores=np.full(21, conf, dtype=np.float32),
        root_dist_px=1.0,
        crop_w_px=crop_w_px,
    )


# ---------------------------------------------------------------------------
# detect_hand_in_crop
# ---------------------------------------------------------------------------

class TestDetectHandInCrop:
    _WRIST = (200.0, 200.0)
    _ELBOW = (150.0, 200.0)  # forearm_len = 50 -> half=max(0.9*50,60)=60, offset=17.5, gate=max(0.5*50,40)=40

    def _crop_origin(self):
        half = max(0.9 * 50.0, 60.0)
        cx = self._WRIST[0] + 0.35 * 50.0
        cy = self._WRIST[1]
        return int(max(0, cx - half)), int(max(0, cy - half))

    def test_no_detections_returns_none(self):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        model = _FakeHandModel([])
        result = detect_hand_in_crop(model, img, wrist=self._WRIST, elbow=self._ELBOW)
        assert result is None

    def test_close_candidate_passes_gate(self):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        x0, y0 = self._crop_origin()
        root_local = (self._WRIST[0] - x0, self._WRIST[1] - y0)
        model = _FakeHandModel([(root_local, _uniform_hand_kp(root_local), np.full(21, 0.9))])
        result = detect_hand_in_crop(model, img, wrist=self._WRIST, elbow=self._ELBOW)
        assert result is not None
        assert result.root_dist_px < 1.0
        np.testing.assert_allclose(result.keypoints[0], self._WRIST, atol=1.0)
        assert result.crop_w_px == pytest.approx(120.0)  # 2 * half(=60)

    def test_far_candidate_rejected_by_gate(self):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        x0, y0 = self._crop_origin()
        far_full = (self._WRIST[0] + 55.0, self._WRIST[1])  # 55px > 40px gate
        root_local = (far_full[0] - x0, far_full[1] - y0)
        model = _FakeHandModel([(root_local, _uniform_hand_kp(root_local), np.full(21, 0.9))])
        result = detect_hand_in_crop(model, img, wrist=self._WRIST, elbow=self._ELBOW)
        assert result is None

    def test_picks_nearest_of_multiple_candidates_not_highest_confidence(self):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        x0, y0 = self._crop_origin()
        near_full = (self._WRIST[0] + 2.0, self._WRIST[1])
        far_full = (self._WRIST[0] + 30.0, self._WRIST[1])
        near_local = (near_full[0] - x0, near_full[1] - y0)
        far_local = (far_full[0] - x0, far_full[1] - y0)
        candidates = [
            (far_local, _uniform_hand_kp(far_local), np.full(21, 0.95)),   # higher confidence, farther
            (near_local, _uniform_hand_kp(near_local), np.full(21, 0.5)),  # lower confidence, nearer
        ]
        model = _FakeHandModel(candidates)
        result = detect_hand_in_crop(model, img, wrist=self._WRIST, elbow=self._ELBOW)
        assert result is not None
        assert result.root_dist_px < 5.0

    def test_no_elbow_uses_wrist_centred_floor_crop(self):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        model = _FakeHandModel([])
        detect_hand_in_crop(model, img, wrist=(200.0, 200.0), elbow=None)
        assert model.calls == [(120, 120, 3)]  # 2*60px floor, no offset

    def test_crop_clamped_to_image_bounds(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        model = _FakeHandModel([])
        detect_hand_in_crop(model, img, wrist=(5.0, 5.0), elbow=None)
        assert model.calls == [(65, 65, 3)]  # clamped at top-left edge


# ---------------------------------------------------------------------------
# HandRefinementPipeline._refine_one
# ---------------------------------------------------------------------------

class TestRefineOne:
    def test_returns_only_the_confident_hand(self, monkeypatch):
        kp = np.zeros((133, 3), dtype=np.float32)
        kp[_WRIST_IDX["left"]] = [200.0, 200.0, 5.0]
        kp[_ELBOW_IDX["left"]] = [150.0, 200.0, 5.0]
        # right wrist left at (0,0,0) confidence -> should be skipped entirely
        img = np.zeros((400, 400, 3), dtype=np.uint8)

        fake_result = _fake_candidate((201.0, 199.0), conf=0.8, crop_w_px=128.0)
        calls = []

        def fake_detect(hand_model, image, wrist, elbow):
            calls.append((wrist, elbow))
            return fake_result

        monkeypatch.setattr("posetrak.detection.hand_refinement.detect_hand_in_crop", fake_detect)

        results = HandRefinementPipeline._refine_one(None, object(), kp, img)
        assert calls == [((200.0, 200.0), (150.0, 200.0))]  # only left, with elbow anchor

        assert len(results) == 1
        region_type, hand_kp, noise_scale = results[0]
        assert region_type == "hand_l"
        assert hand_kp.shape == (_HAND_N_KP, 3)
        np.testing.assert_allclose(hand_kp[:, 0], 201.0)
        np.testing.assert_allclose(hand_kp[:, 1], 199.0)
        np.testing.assert_allclose(hand_kp[:, 2], 0.8 * _HAND_CONF_SCALE)
        assert noise_scale == pytest.approx(128.0 / _HAND_POSE_INPUT_WIDTH)

        # kp itself is never mutated — hand rows are returned, not patched in.
        assert np.all(kp[91:133] == 0.0)

    def test_no_confident_wrist_skips_entirely(self, monkeypatch):
        kp = np.zeros((133, 3), dtype=np.float32)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        called = []
        monkeypatch.setattr(
            "posetrak.detection.hand_refinement.detect_hand_in_crop",
            lambda *a, **k: called.append(1),
        )
        results = HandRefinementPipeline._refine_one(None, object(), kp, img)
        assert results == []
        assert called == []

    def test_gate_reject_returns_nothing(self, monkeypatch):
        kp = np.zeros((133, 3), dtype=np.float32)
        kp[_WRIST_IDX["left"]] = [200.0, 200.0, 5.0]
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        monkeypatch.setattr(
            "posetrak.detection.hand_refinement.detect_hand_in_crop",
            lambda *a, **k: None,
        )
        results = HandRefinementPipeline._refine_one(None, object(), kp, img)
        assert results == []


# ---------------------------------------------------------------------------
# HandRefinementPipeline.run — pose-model gating and DB round-trip
# ---------------------------------------------------------------------------

class TestHandRefinementPipelineRun:
    def test_run_skips_non_133kp_pose_model(self, session):
        run_id = create_detection_run(
            session, shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
            time_start_s=0.0, time_end_s=1.0,
            detector_model="yolo11x", pose_model="rtmpose-l-17kp",
        )
        pipeline = HandRefinementPipeline(session)
        assert pipeline.run(run_id, cameras=[]) == 0

    def test_run_writes_gated_pass_as_separate_hand_row(self, session, monkeypatch):
        run_id = create_detection_run(
            session, shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
            time_start_s=0.0, time_end_s=1.0,
            detector_model="yolo11x", pose_model="rtmpose-l-133kp",
        )
        kp = np.zeros((133, 3), dtype=np.float32)
        kp[_WRIST_IDX["left"]] = [200.0, 200.0, 5.0]
        kp[_ELBOW_IDX["left"]] = [150.0, 200.0, 5.0]
        session.execute(
            "INSERT INTO detection_keypoints"
            " (detection_run_id, shot_video_id, video_frame, track_id, region_type, keypoints, noise_scale)"
            " VALUES (?,?,?,?, 'full_body', ?, ?)",
            (run_id, _SVID, 0, 1, kp.tobytes(), 0.5),
        )
        session.commit()

        fake_img = np.zeros((400, 400, 3), dtype=np.uint8)
        monkeypatch.setattr(
            "posetrak.detection.hand_refinement.iter_frames",
            lambda path, first, last: iter([(0, fake_img)]),
        )
        fake_result = _fake_candidate((201.0, 199.0), conf=0.8, crop_w_px=64.0)
        monkeypatch.setattr(
            "posetrak.detection.hand_refinement.detect_hand_in_crop",
            lambda *a, **k: fake_result,
        )
        monkeypatch.setattr(HandRefinementPipeline, "_get_hand_model", lambda self: object())

        cam = CameraInfo(
            shot_video_id=_SVID, camera_instance_id=_CAM_ID,
            file_path="/fake/video.mp4", actual_fps=30.0,
            ref_frame=0, ref_timestamp_s=0.0,
        )
        pipeline = HandRefinementPipeline(session)
        n = pipeline.run(run_id, cameras=[cam])
        assert n == 1

        # Original whole-body row is untouched.
        body_row = session.execute(
            "SELECT keypoints, noise_scale FROM detection_keypoints"
            " WHERE detection_run_id=? AND track_id=1 AND region_type='full_body'",
            (run_id,),
        ).fetchone()
        assert bytes(body_row["keypoints"]) == kp.tobytes()
        assert abs(body_row["noise_scale"] - 0.5) < 1e-6

        # A separate hand_l row is written, with its own crop-derived noise_scale.
        hand_row = session.execute(
            "SELECT keypoints, noise_scale FROM detection_keypoints"
            " WHERE detection_run_id=? AND track_id=1 AND region_type='hand_l'",
            (run_id,),
        ).fetchone()
        assert hand_row is not None
        hand_kp = np.frombuffer(bytes(hand_row["keypoints"]), dtype=np.float32).reshape(-1, 3)
        assert hand_kp.shape == (_HAND_N_KP, 3)
        np.testing.assert_allclose(hand_kp[:, 0], 201.0)
        np.testing.assert_allclose(hand_kp[:, 2], 0.8 * _HAND_CONF_SCALE)
        expected_noise = 64.0 / _HAND_POSE_INPUT_WIDTH
        assert hand_row["noise_scale"] == pytest.approx(expected_noise)
        # A tight hand crop (64px) implies lower noise than the whole-body row (0.5).
        assert hand_row["noise_scale"] < body_row["noise_scale"]

        # No hand_r row was written — right wrist had zero confidence.
        assert session.execute(
            "SELECT 1 FROM detection_keypoints"
            " WHERE detection_run_id=? AND track_id=1 AND region_type='hand_r'",
            (run_id,),
        ).fetchone() is None

    def test_run_no_gate_pass_writes_no_hand_row(self, session, monkeypatch):
        run_id = create_detection_run(
            session, shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
            time_start_s=0.0, time_end_s=1.0,
            detector_model="yolo11x", pose_model="rtmpose-l-133kp",
        )
        kp = np.zeros((133, 3), dtype=np.float32)
        kp[_WRIST_IDX["left"]] = [200.0, 200.0, 5.0]
        original_bytes = kp.tobytes()
        session.execute(
            "INSERT INTO detection_keypoints"
            " (detection_run_id, shot_video_id, video_frame, track_id, region_type, keypoints, noise_scale)"
            " VALUES (?,?,?,?, 'full_body', ?, ?)",
            (run_id, _SVID, 0, 1, original_bytes, 0.5),
        )
        session.commit()

        fake_img = np.zeros((400, 400, 3), dtype=np.uint8)
        monkeypatch.setattr(
            "posetrak.detection.hand_refinement.iter_frames",
            lambda path, first, last: iter([(0, fake_img)]),
        )
        monkeypatch.setattr(
            "posetrak.detection.hand_refinement.detect_hand_in_crop",
            lambda *a, **k: None,
        )
        monkeypatch.setattr(HandRefinementPipeline, "_get_hand_model", lambda self: object())

        cam = CameraInfo(
            shot_video_id=_SVID, camera_instance_id=_CAM_ID,
            file_path="/fake/video.mp4", actual_fps=30.0,
            ref_frame=0, ref_timestamp_s=0.0,
        )
        pipeline = HandRefinementPipeline(session)
        n = pipeline.run(run_id, cameras=[cam])
        assert n == 0

        rows = session.execute(
            "SELECT region_type, keypoints FROM detection_keypoints WHERE detection_run_id=? AND track_id=1",
            (run_id,),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["region_type"] == "full_body"
        assert bytes(rows[0]["keypoints"]) == original_bytes

    def test_run_calls_on_camera_done_once_per_camera(self, session, monkeypatch):
        """on_camera_done(done, total) should fire after each camera, so a
        caller can drive a combined "N/M cameras" progress indicator across
        this pass too, not just the initial detection pass (DetectionPipeline
        already has this via its own on_camera_done)."""
        run_id = create_detection_run(
            session, shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
            time_start_s=0.0, time_end_s=1.0,
            detector_model="yolo11x", pose_model="rtmpose-l-133kp",
        )
        monkeypatch.setattr(
            "posetrak.detection.hand_refinement.iter_frames",
            lambda path, first, last: iter([]),
        )
        monkeypatch.setattr(HandRefinementPipeline, "_get_hand_model", lambda self: object())

        # Two cameras, no detection_keypoints rows for either -- _process_camera()
        # returns 0 immediately for each, but on_camera_done should still fire twice.
        cams = [
            CameraInfo(
                shot_video_id=f"sv{i}", camera_instance_id=f"cam{i}",
                file_path="/fake/video.mp4", actual_fps=30.0,
                ref_frame=0, ref_timestamp_s=0.0,
            )
            for i in range(2)
        ]
        calls: list[tuple[int, int]] = []
        pipeline = HandRefinementPipeline(session)
        pipeline.run(run_id, cameras=cams, on_camera_done=lambda done, total: calls.append((done, total)))

        assert calls == [(1, 2), (2, 2)]
