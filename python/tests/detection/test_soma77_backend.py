# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from posetrak.detection import backends_soma77 as soma


def test_flip_pairs_connect_left_and_right_joints():
    names = soma.SOMA77_NAMES
    for left, right in soma.FLIP_PAIRS:
        assert names[left].replace("Left", "Right") == names[right]
    assert len(soma.SOMA77_NAMES) == len(soma.SOMA77_PARENTS) == 77


def test_flipping_heatmaps_twice_restores_them():
    heatmaps = np.random.default_rng(0).random((77, 8, 6)).astype(np.float32)
    assert np.array_equal(soma.flip_heatmaps(soma.flip_heatmaps(heatmaps)), heatmaps)


def test_decode_maps_heatmap_cells_to_frame_pixels():
    """A peak at the cell just right of and below the middle of the heatmap is the centre of the box plus half a
    cell; the first cell is the corner of the 0.75 s wide, s high window the model sees plus half a cell."""
    h, w = soma.HEATMAP_H, soma.HEATMAP_W
    heatmaps = np.zeros((2, h, w), np.float32)
    heatmaps[0, h // 2, w // 2] = 1.0
    heatmaps[1, 0, 0] = 0.5
    kp = soma.decode(heatmaps, cx=1000.0, cy=500.0, s=640.0)
    cell = np.array([0.75 * 640 / w, 640.0 / h])
    assert np.allclose(kp[0, :2], np.array([1000.0, 500.0]) + 0.5 * cell, atol=1e-3)
    assert np.allclose(kp[1, :2], np.array([1000.0 - 0.375 * 640, 500.0 - 320.0]) + 0.5 * cell, atol=1e-3)
    assert kp[0, 2] == 1.0 and kp[1, 2] == 0.5


def test_preprocess_gives_the_model_input_shape():
    frame = np.zeros((600, 800, 3), np.uint8)
    cx, cy, s = soma.crop_square(np.array([400, 300, 100, 300], np.float32))
    assert (cx, cy, s) == (400.0, 300.0, 300 * soma.BOX_ENLARGE)
    assert soma.preprocess(frame, cx, cy, s).shape == (3, soma.INPUT_H, soma.INPUT_W)


def test_soma77_pose_model_is_registered_for_the_viewer():
    from app.pose.kp_models import get_pose_model

    model = get_pose_model("soma77-dinov3-vitpose")
    assert model.model_id == "soma-77" and model.names == soma.SOMA77_NAMES
    assert model.limb_chain_indices("Left leg") == [67, 68, 69, 70, 71]


def test_parabolic_refinement_recovers_a_position_inside_the_cell():
    h, w = soma.HEATMAP_H, soma.HEATMAP_W
    yy, xx = np.mgrid[0:h, 0:w]
    truth_x, truth_y = 20.45, 31.55
    heatmaps = np.exp(-((xx - truth_x) ** 2 + (yy - truth_y) ** 2) / (2 * 1.5**2))[None].astype(np.float32)
    window_w, window_h = 0.75 * 640, 640.0
    expected = np.array([(truth_x + 0.5) / w * window_w, (truth_y + 0.5) / h * window_h]) + np.array([1000.0 - window_w / 2, 500.0 - window_h / 2])
    cell = np.array([window_w / w, window_h / h])
    refined = soma.decode(heatmaps, 1000.0, 500.0, 640.0)[0, :2]
    stepped = soma.decode(heatmaps, 1000.0, 500.0, 640.0, refine="sign")[0, :2]
    assert np.all(np.abs(refined - expected) < 0.05 * cell)
    assert np.all(np.abs(stepped - expected) < 0.55 * cell) and np.any(np.abs(stepped - expected) > 0.1 * cell)


def test_mask_gating_removes_peaks_outside_the_person():
    inside = np.zeros((100, 100), np.uint8)
    inside[:, :50] = 1                                    # the person is the left half of a 100 x 100 frame
    cells = soma.mask_cells(inside, 100, 100, cx=50.0, cy=50.0, s=100.0)
    assert cells.shape == (soma.HEATMAP_H, soma.HEATMAP_W)
    heatmaps = np.zeros((1, soma.HEATMAP_H, soma.HEATMAP_W), np.float32)
    heatmaps[0, 30, 40] = 1.0                              # a strong peak on the right half, off the mask
    heatmaps[0, 30, 20] = 0.4                              # a weaker one on the person
    kp = soma.decode(soma.gate_heatmaps(heatmaps, cells), 50.0, 50.0, 100.0)
    assert kp[0, 0] < 50.0 and abs(kp[0, 2] - 0.4) < 1e-6
