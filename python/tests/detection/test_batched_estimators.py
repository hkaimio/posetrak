# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Batched estimators, the hand pass, and the model weights and licences."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from posetrak.detection import backends_soma77 as soma
from posetrak.detection import model_weights, pose_model_choice
from posetrak.detection.backends import PersonDetection
from posetrak.detection.backends_rtmpose import run_rtmlib_batch
from posetrak.detection.hand_refinement import HandPass, detect_hand_in_crop


# ---------------------------------------------------------------------------
# SOMA-77
# ---------------------------------------------------------------------------


class FakeOnnx:
    """Heatmaps with a Gaussian peak whose cell depends on the mean brightness of the crop."""

    def __init__(self):
        self.calls = []

    def heatmaps(self, batch):
        self.calls.append(len(batch))
        out = np.zeros((len(batch), 77, soma.HEATMAP_H, soma.HEATMAP_W), np.float32)
        for i, crop in enumerate(batch):
            x, y = 10 + int(crop.mean() * 3) % 20, 20 + int(crop.mean() * 3) % 20
            out[i, :, y, x] = 1.0
        return out


def frame(value):
    return np.full((300, 400, 3), value, np.uint8)


def person(track=1, cx=200.0, cy=150.0, w=60.0, h=140.0, mask=None):
    return PersonDetection(track, np.array([cx, cy, w, h], np.float32), 1.0, mask)


def test_batched_estimates_equal_single_estimates_and_calls_hold_batch_crops():
    onnx = FakeOnnx()
    estimator = soma.Soma77Estimator(onnx=onnx, batch_crops=6)
    items = [(frame(40), [person(1), person(2, cx=120)]), (frame(90), []), (frame(150), [person(3)])]
    batched = estimator.estimate_batch(items)
    assert [len(r) for r in batched] == [2, 0, 1]
    assert onnx.calls == [6]                         # three persons, each with its flipped copy
    for (img, dets), result in zip(items, batched):
        for det, got in zip(dets, result):
            (single,) = estimator.estimate(img, [det])
            assert got.track_id == det.track_id
            np.testing.assert_allclose(got.keypoints, single.keypoints, atol=1e-5)
    assert batched[0][0].keypoints.shape == (77, 3)


def test_soma_batches_split_by_batch_crops_and_noise_scale_follows_the_crop():
    onnx = FakeOnnx()
    estimator = soma.Soma77Estimator(onnx=onnx, batch_crops=4)
    result = estimator.estimate_batch([(frame(40), [person(i) for i in range(5)])])
    assert onnx.calls == [4, 4, 2]
    _, _, s = soma.crop_square(person().bbox)
    assert result[0][0].noise_scale == pytest.approx(0.75 * s / soma.INPUT_W)


def test_soma_options_record_every_setting():
    options = soma.Soma77Estimator(onnx=FakeOnnx(), flip_test=False, enlarge=1.4, mask_gate_floor=0.1).options()
    assert options["flip_test"] is False and options["box_enlarge"] == 1.4 and options["mask_gate_floor"] == 0.1
    assert options["decoding"] == "cell-centre" and options["refine"] == "parabolic" and options["mask_gate"] is True


def test_mask_gate_keeps_only_peaks_on_the_person_mask():
    class TwoPeaks(FakeOnnx):
        def heatmaps(self, batch):
            out = np.zeros((len(batch), 77, soma.HEATMAP_H, soma.HEATMAP_W), np.float32)
            out[:, :, 20, 10] = 0.9           # left of the window centre
            out[:, :, 20, 40] = 1.0           # right: the stronger peak
            return out

    det = person()
    cx, cy, s = soma.crop_square(det.bbox)
    left_x = cx - 0.375 * s + (10 + 0.5) / soma.HEATMAP_W * 0.75 * s
    mask = np.zeros((300, 400), np.uint8)
    mask[:, int(left_x) - 30:int(left_x) + 30] = 1       # the person is where the weaker peak is
    gated = soma.Soma77Estimator(onnx=TwoPeaks(), flip_test=False).estimate(frame(10), [person(mask=mask)])[0]
    free = soma.Soma77Estimator(onnx=TwoPeaks(), flip_test=False).estimate(frame(10), [person()])[0]
    ungated = soma.Soma77Estimator(onnx=TwoPeaks(), flip_test=False, mask_gate=False).estimate(
        frame(10), [person(mask=mask)])[0]
    assert abs(gated.keypoints[0, 0] - left_x) < 2 and gated.keypoints[0, 2] == pytest.approx(0.9)
    assert free.keypoints[0, 0] > gated.keypoints[0, 0] + 20
    np.testing.assert_array_equal(ungated.keypoints, free.keypoints)


def test_soma_without_weights_says_so(monkeypatch, tmp_path):
    monkeypatch.setenv("POSETRAK_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("POSETRAK_SOMA77_MODEL", raising=False)
    with pytest.raises(soma.Soma77WeightsMissing):
        soma.Soma77Estimator()


# ---------------------------------------------------------------------------
# rtmlib batching
# ---------------------------------------------------------------------------


class FakeRtmModel:
    """The three rtmlib model steps and a session whose output is the input's mean, per batch element."""

    def __init__(self):
        self.runs = []
        outer = self

        class Session:
            def get_inputs(self):
                return [type("I", (), {"name": "input"})]

            def get_outputs(self):
                return [type("O", (), {"name": "out"})]

            def run(self, names, feeds):
                batch = feeds["input"]
                outer.runs.append(len(batch))
                return [batch.mean(axis=(1, 2, 3)).reshape(-1, 1, 1)]

        self.session = Session()

    def preprocess(self, image, bbox):
        x0, y0, x1, y1 = (int(v) for v in bbox)
        crop = image[y0:y1, x0:x1].astype(np.float64)
        return crop[:4, :4], np.array([x0, y0]), np.array([4.0, 4.0])

    def postprocess(self, outputs, center, scale):
        return outputs[0][:, :, 0] + center, outputs[0][:, :, 0]


def test_run_rtmlib_batch_calls_the_model_once_per_chunk_and_keeps_order():
    model = FakeRtmModel()
    image = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)
    jobs = [(image, np.array([i, i, i + 8, i + 8], np.float32)) for i in range(5)]
    out = run_rtmlib_batch(model, jobs, max_batch=2)
    assert model.runs == [2, 2, 1]
    single = [run_rtmlib_batch(FakeRtmModel(), [job])[0] for job in jobs]
    for got, want in zip(out, single):
        np.testing.assert_allclose(got[0], want[0])
        np.testing.assert_allclose(got[1], want[1])


# ---------------------------------------------------------------------------
# Hand pass
# ---------------------------------------------------------------------------


class FakeHand:
    """A hand model that finds one hand with its root at the crop centre."""

    def __init__(self):
        self.crop_sizes = []

    def __call__(self, crops):
        out = []
        for crop in crops:
            self.crop_sizes.append(crop.shape[:2])
            h, w = crop.shape[:2]
            kps = np.zeros((1, 21, 2), np.float32)
            kps[0, :, 0], kps[0, :, 1] = w / 2, h / 2
            out.append((kps, np.full((1, 21), 0.8, np.float32)))
        return out


def body_pose(left_wrist=(200.0, 200.0), right_wrist=(300.0, 200.0)):
    kp = np.zeros((133, 3), np.float32)
    kp[7] = (150, 200, 1)           # left elbow
    kp[9] = (*left_wrist, 1)
    kp[10] = (*right_wrist, 1)      # right wrist, no right elbow -> the crop is centred on it
    return kp


def test_hand_pass_rows_equal_the_per_crop_refinement():
    image = np.zeros((500, 500, 3), np.uint8)
    kp = body_pose()
    rows = HandPass(model=FakeHand()).refine([(image, kp), (image, np.zeros((17, 3), np.float32))])
    assert rows[1] == []                                          # not a 133-point pose
    assert {r[0] for r in rows[0]} == {"hand_l", "hand_r"}
    per_crop = detect_hand_in_crop(lambda crop: FakeHand()([crop])[0], image, (200.0, 200.0), (150.0, 200.0))
    left = next(r for r in rows[0] if r[0] == "hand_l")
    np.testing.assert_allclose(left[1][:, :2], per_crop.keypoints)
    assert left[1][0, 2] == pytest.approx(0.8 * 5.0)             # the hand model's 0-1 score on the tracker's scale
    assert left[2] == pytest.approx(per_crop.crop_w_px / 256)


def test_hand_pass_runs_one_model_call_for_all_hands_of_the_batch():
    calls = []

    class Counting(FakeHand):
        def __call__(self, crops):
            calls.append(len(crops))
            return super().__call__(crops)

    image = np.zeros((500, 500, 3), np.uint8)
    HandPass(model=Counting()).refine([(image, body_pose()), (image, body_pose((220.0, 210.0)))])
    assert calls == [4]


# ---------------------------------------------------------------------------
# Weights and licences
# ---------------------------------------------------------------------------


@pytest.fixture
def model_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("POSETRAK_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("POSETRAK_SOMA77_MODEL", raising=False)
    return tmp_path


@pytest.fixture
def fake_hub(monkeypatch, model_dir):
    monkeypatch.setattr(model_weights, "fetch_licence_text", lambda licence: f"terms of {licence.id} " * 200)
    fetched = []

    def hf_hub_download(repo, filename, local_dir):
        fetched.append((repo, filename))
        path = model_dir / "gem-x" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", type("M", (), {"hf_hub_download": staticmethod(hf_hub_download)}))
    return fetched


def test_download_needs_both_licences_accepted_and_stores_their_hashes(fake_hub, model_dir):
    shown = []
    assert model_weights.soma77_model_path() is None
    path = model_weights.download_gemx(lambda texts: shown.extend(texts) or True)
    assert [lic.id for lic, _ in shown] == ["nvidia-open-model-license", "dinov3-license"]
    assert path == model_dir / "gem-x" / "onnx" / "vitpose.onnx" and model_weights.soma77_model_path() == path
    assert fake_hub == [("nvidia/GEM-X", "onnx/vitpose.onnx"), ("nvidia/GEM-X", "onnx/vitpose.onnx.data")]
    record = json.loads((model_dir / "gem-x" / "licence-acceptance.json").read_text())
    assert {e["id"] for e in record["licences"]} == {"nvidia-open-model-license", "dinov3-license"}
    assert all(len(e["text_sha256"]) == 64 for e in record["licences"])
    assert "Built with DINOv3" in record["notices"]


def test_declining_downloads_nothing(fake_hub, model_dir):
    with pytest.raises(model_weights.LicenceDeclined):
        model_weights.download_gemx(lambda texts: False)
    assert fake_hub == [] and model_weights.soma77_model_path() is None


def test_weights_without_a_recorded_acceptance_are_not_used(fake_hub, model_dir):
    model_weights.download_gemx(lambda texts: True)
    (model_dir / "gem-x" / "licence-acceptance.json").unlink()
    assert model_weights.soma77_model_path() is None


def test_licence_that_cannot_be_shown_cannot_be_accepted(monkeypatch, model_dir):
    def unreachable(licence):
        raise OSError("offline")

    monkeypatch.setattr(model_weights, "fetch_licence_text", unreachable)
    with pytest.raises(model_weights.LicenceDeclined, match="could not be fetched"):
        model_weights.download_gemx(lambda texts: True)


def test_declined_licences_fall_back_to_rtmpose_and_say_why(fake_hub):
    model, reason = pose_model_choice.resolve_pose_model(None, accept=lambda texts: False)
    assert model == "rtmpose-l-133kp" and "not accepted" in reason
    model, reason = pose_model_choice.resolve_pose_model(None, accept=None)
    assert model == "rtmpose-l-133kp" and "--accept-licences" in reason
    model, reason = pose_model_choice.resolve_pose_model("vitpose-l-133kp", accept=None)
    assert (model, reason) == ("vitpose-l-133kp", None)


def test_accepting_selects_soma_and_a_later_run_needs_no_prompt(fake_hub):
    assert pose_model_choice.resolve_pose_model(None, accept=lambda texts: True) == ("soma77-vitpose-h", None)
    assert pose_model_choice.resolve_pose_model("soma77-vitpose-h", accept=None) == ("soma77-vitpose-h", None)


def test_failed_download_falls_back_with_the_error(monkeypatch, fake_hub):
    def broken(repo, filename, local_dir):
        raise ConnectionError("no route to host")

    __import__("sys").modules["huggingface_hub"].hf_hub_download = staticmethod(broken)
    model, reason = pose_model_choice.resolve_pose_model(None, accept=lambda texts: True)
    assert model == "rtmpose-l-133kp" and "no route to host" in reason


def test_a_user_supplied_model_file_bypasses_the_download(monkeypatch, model_dir, tmp_path):
    own = tmp_path / "vitpose.onnx"
    own.write_bytes(b"x")
    monkeypatch.setenv("POSETRAK_SOMA77_MODEL", str(own))
    assert model_weights.soma77_model_path() == own
    assert pose_model_choice.resolve_pose_model(None) == ("soma77-vitpose-h", None)
