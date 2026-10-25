# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""pose_models.py — the registry of pose models: keypoint vocabularies in blob order.

A pose model is a keypoint vocabulary. Its :class:`PoseModelSpec` names every slot of the model's keypoint blob
(``float32[N, 3]`` of x, y, confidence), and carries the tag (``source``) that prefixes the landmark names binding
the slots to skeleton markers and that is written to ``pose_sequence_keypoints.source``. The estimators, the
viewer (``app.pose.kp_models``), finalisation and the skeleton composition read this module instead of keeping
copies of the names.

A skeleton marker is bound to a keypoint by its ``landmark``, ``"<source>:<keypoint name>"``
(``soma77:LeftForeArm``, ``coco133:left_elbow``, ``hand21:left_index_1``). A hand21 blob has no side of its own:
it is stored as the ``hand_l`` or ``hand_r`` observation source, and its landmarks carry the side as a prefix of
the keypoint name, matching the COCO-133 hand slots it replaces.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PoseModelSpec:
    """One pose model's keypoints, in blob order."""

    source: str                                  # tag of the landmarks and of the manifest rows
    keypoint_names: tuple[str, ...]
    flip_pairs: tuple[tuple[int, int], ...] = ()  # left/right keypoint index pairs
    parents: tuple[int, ...] = ()                # parent keypoint index per keypoint (-1 root), if the model is a tree
    trunk: tuple[str, ...] = ()                  # keypoints of the torso, for telling persons apart across cameras
    body: tuple[str, ...] = ()                   # keypoints of the whole body without face and fingers
    wrists: tuple[str, str] = ()                 # (left, right) wrist keypoint names; empty if the model has none
    elbows: tuple[str, str] = ()                 # (left, right) elbow keypoint names

    def arm_indices(self) -> dict[str, tuple[int, int]]:
        """``{side: (wrist index, elbow index)}`` of the sides this model names a wrist and an elbow for."""
        if not (self.wrists and self.elbows):
            return {}
        return {side: (self.keypoint_names.index(w), self.keypoint_names.index(e))
                for side, w, e in zip(("left", "right"), self.wrists, self.elbows)}

    def trunk_indices(self) -> list[int]:
        """Blob indices of the trunk keypoints."""
        return [self.keypoint_names.index(n) for n in self.trunk]

    def body_indices(self) -> list[int]:
        """Blob indices of the body keypoints (trunk and limbs)."""
        return [self.keypoint_names.index(n) for n in self.body]

    def landmark(self, index: int, side: str = "") -> str:
        """The landmark name of slot *index*; *side* ('left' / 'right') prefixes the name of a hand21 slot."""
        name = self.keypoint_names[index]
        return f"{self.source}:{side}_{name}" if side else f"{self.source}:{name}"

    def manifest_rows(self, side: str = "") -> list[tuple[int, str, str]]:
        """(keypoint_idx, name, source) rows of ``pose_sequence_keypoints`` for a blob of this model."""
        return [(i, self.landmark(i, side), self.source) for i in range(len(self.keypoint_names))]


# ---------------------------------------------------------------------------
# SOMA-77: joint centres of NVIDIA's SOMA skeleton, the 77 heatmaps of the GEM-X 2D model
# ---------------------------------------------------------------------------

SOMA77_NAMES = (
    "Hips", "Spine1", "Spine2", "Chest",
    "Neck1", "Neck2", "Head", "HeadEnd",
    "Jaw", "LeftEye", "RightEye", "LeftShoulder",
    "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumb1",
    "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd", "LeftHandIndex1",
    "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4",
    "LeftHandMiddleEnd", "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
    "LeftHandRing4", "LeftHandRingEnd", "LeftHandPinky1", "LeftHandPinky2",
    "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd", "RightShoulder",
    "RightArm", "RightForeArm", "RightHand", "RightHandThumb1",
    "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd", "RightHandIndex1",
    "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4",
    "RightHandMiddleEnd", "RightHandRing1", "RightHandRing2", "RightHandRing3",
    "RightHandRing4", "RightHandRingEnd", "RightHandPinky1", "RightHandPinky2",
    "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd", "LeftLeg",
    "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
    "RightToeEnd",
)

# Parent index of each joint (-1 for the root).
SOMA77_PARENTS = (
    -1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 3, 11, 12, 13, 14,
    15, 16, 17, 14, 19, 20, 21, 22, 14, 24, 25, 26, 27, 14, 29, 30,
    31, 32, 14, 34, 35, 36, 37, 3, 39, 40, 41, 42, 43, 44, 45, 42,
    47, 48, 49, 50, 42, 52, 53, 54, 55, 42, 57, 58, 59, 60, 42, 62,
    63, 64, 65, 0, 67, 68, 69, 70, 0, 72, 73, 74, 75,
)

# Left/right joint pairs, for averaging the heatmaps of an image and of its mirror.
SOMA77_FLIP_PAIRS = (
    (9, 10), *((i, i + 28) for i in range(11, 39)), *((i, i + 5) for i in range(67, 72)),
)

SOMA77_TRUNK = ("Hips", "Spine1", "Spine2", "Neck1", "LeftArm", "RightArm")   # LeftArm: the shoulder joint

SOMA77_BODY = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
)

SOMA77 = PoseModelSpec(
    "soma77", SOMA77_NAMES, SOMA77_FLIP_PAIRS, SOMA77_PARENTS, SOMA77_TRUNK, SOMA77_BODY,
    wrists=("LeftHand", "RightHand"), elbows=("LeftForeArm", "RightForeArm"),
)


# ---------------------------------------------------------------------------
# COCO-133 (RTMPose whole-body, MMPose convention)
#
#   0-16   body (COCO-17)
#   17-22  feet (left big toe, small toe, heel; right the same)
#   23-90  face (68 points in the dlib / COCO-WholeBody order)
#   91-111 left hand (21 points), 112-132 right hand
# ---------------------------------------------------------------------------

_HAND_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HAND21_NAMES = ("hand_root", *(f"{finger}_{i}" for finger in _HAND_FINGERS for i in range(1, 5)))

# First COCO-133 slot of each hand, and the observation source a refined hand pass is stored under.
COCO133_HAND_BASE = {"left": 91, "right": 112}
HAND_OBS_SOURCE = {"left": "hand_l", "right": "hand_r"}
HAND_SIDE = {"hand_l": "left", "hand_r": "right"}

COCO133_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    *[f"jaw_{i}" for i in range(17)],
    *[f"right_eyebrow_{i}" for i in range(5)],
    *[f"left_eyebrow_{i}" for i in range(5)],
    *[f"nose_bridge_{i}" for i in range(4)],
    *[f"nose_tip_{i}" for i in range(5)],
    *[f"right_eye_{i}" for i in range(6)],
    *[f"left_eye_{i}" for i in range(6)],
    *[f"outer_lip_{i}" for i in range(12)],
    *[f"inner_lip_{i}" for i in range(8)],
    *[f"left_{n}" for n in HAND21_NAMES],
    *[f"right_{n}" for n in HAND21_NAMES],
)
assert len(COCO133_NAMES) == 133


def _swap_side(name: str) -> str:
    return name.replace("left_", "right_", 1) if name.startswith("left_") else name.replace("right_", "left_", 1)


# Mirror pairs of the body, feet and hands (by name). The face landmarks are not listed: their left/right
# correspondence is not given by their names.
COCO133_FLIP_PAIRS = tuple(
    (i, COCO133_NAMES.index(_swap_side(COCO133_NAMES[i])))
    for i in (*range(1, 23), *range(91, 112))
    if COCO133_NAMES[i].startswith("left_")
)

COCO133_TRUNK = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")

COCO133 = PoseModelSpec(
    "coco133", COCO133_NAMES, COCO133_FLIP_PAIRS, trunk=COCO133_TRUNK, body=COCO133_NAMES[:17],
    wrists=("left_wrist", "right_wrist"), elbows=("left_elbow", "right_elbow"),
)


# ---------------------------------------------------------------------------
# hand21: the 21 keypoints of a hand-only crop (the refined hand passes)
# ---------------------------------------------------------------------------

HAND21 = PoseModelSpec("hand21", HAND21_NAMES)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SPECS: dict[str, PoseModelSpec] = {s.source: s for s in (SOMA77, COCO133, HAND21)}


def spec_for_pose_model(pose_model: str | None) -> PoseModelSpec | None:
    """The spec of a stored ``pose_model`` string ('soma77-dinov3-vitpose', 'rtmpose-l-133kp', ...), if registered."""
    if not pose_model:
        return None
    key = pose_model.lower().replace("_", "-")
    if "soma" in key and "77" in key:
        return SOMA77
    if "133" in key:
        return COCO133
    return None


def body_manifest(pose_model: str | None) -> list[tuple[int, str, str]]:
    """Manifest rows (keypoint_idx, name, source) of the body blob of a *pose_model* sequence; [] if unregistered."""
    spec = spec_for_pose_model(pose_model)
    return spec.manifest_rows() if spec is not None else []


def hand_manifest(obs_source: str) -> list[tuple[int, str, str]]:
    """Manifest rows of a hand blob stored as observation source ``hand_l`` / ``hand_r`` (a '.refined' suffix is ignored)."""
    return HAND21.manifest_rows(HAND_SIDE[obs_source.removesuffix(".refined")])
