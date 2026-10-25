# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from app.pose.kp_models import COCO133, SOMA77, get_pose_model
from posetrak.detection import pose_models as pm


def test_specs_have_the_documented_widths():
    assert len(pm.SOMA77.keypoint_names) == len(pm.SOMA77.parents) == 77
    assert len(pm.COCO133.keypoint_names) == 133
    assert len(pm.HAND21.keypoint_names) == 21


@pytest.mark.parametrize("spec", [pm.SOMA77, pm.COCO133])
def test_flip_pairs_connect_left_and_right_keypoints(spec):
    swap = {"Left": "Right", "Right": "Left", "left_": "right_", "right_": "left_"}
    assert spec.flip_pairs
    for a, b in spec.flip_pairs:
        name = spec.keypoint_names[a]
        for key, value in swap.items():
            if name.startswith(key):
                assert spec.keypoint_names[b] == value + name[len(key):]
                break
        else:
            assert name.replace("Left", "Right") == spec.keypoint_names[b]


def test_coco133_hand_slots_are_the_hand21_names_with_a_side():
    for side, base in pm.COCO133_HAND_BASE.items():
        for k, name in enumerate(pm.HAND21.keypoint_names):
            assert pm.COCO133.keypoint_names[base + k] == f"{side}_{name}"


def test_landmarks_are_model_qualified():
    assert pm.SOMA77.landmark(13) == "soma77:LeftForeArm"
    assert pm.COCO133.landmark(7) == "coco133:left_elbow"
    assert pm.HAND21.landmark(5, "left") == "hand21:left_index_1"


def test_manifest_rows_and_stored_model_names():
    assert pm.body_manifest("soma77-dinov3-vitpose")[14] == (14, "soma77:LeftHand", "soma77")
    assert pm.body_manifest("rtmpose-l-133kp")[0] == (0, "coco133:nose", "coco133")
    assert pm.body_manifest("rtmpose-l-17kp") == [] and pm.body_manifest(None) == []
    assert pm.hand_manifest("hand_r.refined")[0] == (0, "hand21:right_hand_root", "hand21")


def test_viewer_models_use_the_registry_names():
    assert COCO133.names == pm.COCO133_NAMES and SOMA77.names == pm.SOMA77_NAMES
    assert get_pose_model("soma77-dinov3-vitpose") is SOMA77


def test_trunk_and_body_keypoints_are_registered_for_the_body_models():
    assert [pm.COCO133.keypoint_names[i] for i in pm.COCO133.trunk_indices()] == [
        "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    assert pm.COCO133.body_indices() == list(range(17))
    assert [pm.SOMA77.keypoint_names[i] for i in pm.SOMA77.trunk_indices()] == [
        "Hips", "Spine1", "Spine2", "Neck1", "LeftArm", "RightArm"]
    assert set(pm.SOMA77.trunk) <= set(pm.SOMA77.body)
    assert not pm.HAND21.trunk and not pm.HAND21.body
