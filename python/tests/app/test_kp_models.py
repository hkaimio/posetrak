"""Tests for app.pose.kp_models: PoseModel.limb_chain_indices()."""
from __future__ import annotations

from app.pose.kp_models import COCO17, COCO133


def test_coco17_left_arm_chain_is_shoulder_elbow_wrist():
    names = [COCO17.name_of(i) for i in COCO17.limb_chain_indices("Left arm")]
    assert names == ["left_shoulder", "left_elbow", "left_wrist"]


def test_coco17_right_leg_chain_is_hip_knee_ankle():
    names = [COCO17.name_of(i) for i in COCO17.limb_chain_indices("Right leg")]
    assert names == ["right_hip", "right_knee", "right_ankle"]


def test_coco17_has_no_finger_or_toe_keypoints_to_extend_chains_with():
    # COCO-17 has no hand/foot detail -- chains stop at wrist/ankle.
    for limb in ("Left arm", "Right arm", "Left leg", "Right leg"):
        for name in ("index_1", "pinky_1", "heel", "big_toe", "small_toe"):
            assert not any(
                COCO17.name_of(i).endswith(name) for i in COCO17.limb_chain_indices(limb)
            )


def test_coco133_left_arm_chain_includes_hand_keypoints():
    names = [COCO133.name_of(i) for i in COCO133.limb_chain_indices("Left arm")]
    assert names == [
        "left_shoulder", "left_elbow", "left_wrist", "left_index_1", "left_pinky_1",
    ]


def test_coco133_right_leg_chain_includes_foot_keypoints():
    names = [COCO133.name_of(i) for i in COCO133.limb_chain_indices("Right leg")]
    assert names == [
        "right_hip", "right_knee", "right_ankle",
        "right_heel", "right_big_toe", "right_small_toe",
    ]


def test_limb_chain_indices_unknown_limb_is_empty():
    assert COCO133.limb_chain_indices("Tail") == []
