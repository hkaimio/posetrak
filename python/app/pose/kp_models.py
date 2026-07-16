"""Pose keypoint model definitions: names, selection groups.

Each PoseModel maps a model string (as stored in
pose_observation_sequences.pose_model) to ordered keypoint names and named
selection groups useful for bulk editing.

Hierarchy is intentionally omitted for now: COCO-133 has no single-joint
hip or neck node, so a proper tree would require virtual root nodes.

Usage::

    from app.pose.kp_models import get_pose_model
    model = get_pose_model("rtmpose-l-133kp")
    print(model.name_of(0))         # "nose"
    print(model.group_indices("Left hand"))  # frozenset({91, 92, ...})
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PoseModel:
    """Names and selection groups for one pose keypoint schema."""

    model_id: str
    names: tuple[str, ...]
    groups: dict[str, frozenset[int]]
    # Ordered subset of `groups` names that partitions all keypoint indices with
    # no overlap — used for tree-structured views (e.g. the timeline dope sheet).
    # `groups` itself is not partition-safe: entries like "Upper body" or "Torso"
    # deliberately overlap "Left arm" etc. for the flexible group-selection menu.
    tree_groups: tuple[str, ...] = ()
    # limb name -> proximal-to-distal keypoint names, for the chain keypoint
    # placement tool (click the shoulder, then the elbow, then the wrist, ...).
    # Unlike `groups`, order matters here. Names not present in this model
    # (e.g. finger tips on COCO-17) are simply absent from the tuple --
    # limb_chain_indices() below drops names that don't resolve.
    limb_chains: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def name_of(self, idx: int) -> str:
        if 0 <= idx < len(self.names):
            return self.names[idx]
        return str(idx)

    def index_of(self, name: str) -> int | None:
        try:
            return self.names.index(name)
        except ValueError:
            return None

    def group_indices(self, group: str) -> frozenset[int]:
        return self.groups.get(group, frozenset())

    @property
    def all_indices(self) -> frozenset[int]:
        return frozenset(range(len(self.names)))

    @property
    def group_names(self) -> list[str]:
        return list(self.groups)

    def limb_chain_indices(self, limb: str) -> list[int]:
        """Ordered keypoint indices for *limb*, skipping names absent from this model."""
        indices = []
        for name in self.limb_chains.get(limb, ()):
            idx = self.index_of(name)
            if idx is not None:
                indices.append(idx)
        return indices


# ---------------------------------------------------------------------------
# COCO-17
# ---------------------------------------------------------------------------

_COCO17_NAMES: tuple[str, ...] = (
    "nose",                                          # 0
    "left_eye", "right_eye",                        # 1-2
    "left_ear", "right_ear",                        # 3-4
    "left_shoulder", "right_shoulder",              # 5-6
    "left_elbow", "right_elbow",                    # 7-8
    "left_wrist", "right_wrist",                    # 9-10
    "left_hip", "right_hip",                        # 11-12
    "left_knee", "right_knee",                      # 13-14
    "left_ankle", "right_ankle",                    # 15-16
)

_COCO17_GROUPS: dict[str, frozenset[int]] = {
    "Face":        frozenset({0, 1, 2, 3, 4}),
    "Left arm":    frozenset({5, 7, 9}),
    "Right arm":   frozenset({6, 8, 10}),
    "Torso":       frozenset({5, 6, 11, 12}),
    "Left leg":    frozenset({11, 13, 15}),
    "Right leg":   frozenset({12, 14, 16}),
    "Upper body":  frozenset(range(11)),
    "Lower body":  frozenset(range(11, 17)),
}

_COCO17_TREE_GROUPS: tuple[str, ...] = (
    "Face", "Left arm", "Right arm", "Left leg", "Right leg",
)

_COCO17_LIMB_CHAINS: dict[str, tuple[str, ...]] = {
    "Face":       ("nose", "left_ear", "right_ear"),
    "Left arm":   ("left_shoulder", "left_elbow", "left_wrist"),
    "Right arm":  ("right_shoulder", "right_elbow", "right_wrist"),
    "Left leg":   ("left_hip", "left_knee", "left_ankle"),
    "Right leg":  ("right_hip", "right_knee", "right_ankle"),
}

COCO17 = PoseModel(
    model_id="coco-17",
    names=_COCO17_NAMES,
    groups=_COCO17_GROUPS,
    tree_groups=_COCO17_TREE_GROUPS,
    limb_chains=_COCO17_LIMB_CHAINS,
)


# ---------------------------------------------------------------------------
# COCO-133  (RTMPose whole-body, MMPose convention)
#
# Layout:
#   0-16   body (same as COCO-17)
#   17-22  feet (left: big-toe 17, small-toe 18, heel 19;
#                right: big-toe 20, small-toe 21, heel 22)
#   23-90  face (68 pts following dlib / COCO-WholeBody ordering)
#     23-39  jaw contour (17 pts)
#     40-44  right eyebrow (5)
#     45-49  left eyebrow (5)
#     50-53  nose bridge (4)
#     54-58  nose tip (5)
#     59-64  right eye (6)
#     65-70  left eye (6)
#     71-82  outer lips (12)
#     83-90  inner lips (8)
#   91-111 left hand (21 pts: palm + 4 fingers × 4 + thumb × 4)
#  112-132 right hand (21 pts)
# ---------------------------------------------------------------------------

_COCO133_NAMES: tuple[str, ...] = (
    # 0-16: body
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    # 17-22: feet
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    # 23-39: jaw contour
    *[f"jaw_{i}" for i in range(17)],
    # 40-44: right eyebrow, 45-49: left eyebrow
    *[f"right_eyebrow_{i}" for i in range(5)],
    *[f"left_eyebrow_{i}" for i in range(5)],
    # 50-53: nose bridge, 54-58: nose tip
    *[f"nose_bridge_{i}" for i in range(4)],
    *[f"nose_tip_{i}" for i in range(5)],
    # 59-64: right eye, 65-70: left eye
    *[f"right_eye_{i}" for i in range(6)],
    *[f"left_eye_{i}" for i in range(6)],
    # 71-82: outer lips, 83-90: inner lips
    *[f"outer_lip_{i}" for i in range(12)],
    *[f"inner_lip_{i}" for i in range(8)],
    # 91-111: left hand
    "left_hand_root",
    "left_thumb_1", "left_thumb_2", "left_thumb_3", "left_thumb_4",
    "left_index_1", "left_index_2", "left_index_3", "left_index_4",
    "left_middle_1", "left_middle_2", "left_middle_3", "left_middle_4",
    "left_ring_1", "left_ring_2", "left_ring_3", "left_ring_4",
    "left_pinky_1", "left_pinky_2", "left_pinky_3", "left_pinky_4",
    # 112-132: right hand
    "right_hand_root",
    "right_thumb_1", "right_thumb_2", "right_thumb_3", "right_thumb_4",
    "right_index_1", "right_index_2", "right_index_3", "right_index_4",
    "right_middle_1", "right_middle_2", "right_middle_3", "right_middle_4",
    "right_ring_1", "right_ring_2", "right_ring_3", "right_ring_4",
    "right_pinky_1", "right_pinky_2", "right_pinky_3", "right_pinky_4",
)

assert len(_COCO133_NAMES) == 133, f"expected 133, got {len(_COCO133_NAMES)}"

_FACE_IDX     = frozenset(range(23, 91))
_LEFT_HAND_IDX  = frozenset(range(91, 112))
_RIGHT_HAND_IDX = frozenset(range(112, 133))
_LEFT_FOOT_IDX  = frozenset({17, 18, 19})
_RIGHT_FOOT_IDX = frozenset({20, 21, 22})
_BODY17_IDX     = frozenset(range(17))

# The default skeleton (tests/data/Harri_skeleton-regress-test.yaml) only
# attaches markers to nose + ears — eyes and the 68 detailed face landmarks
# aren't used by tracking. Splitting "Face" into that small, actually-useful
# subset plus everything else keeps "Select Face" meaningful for editing
# instead of pulling in ~70 densely-packed, rarely-needed points at once.
_FACE_MARKER_IDX = frozenset({0, 3, 4})              # nose, left_ear, right_ear
_FACE_DETAIL_IDX = frozenset({1, 2}) | _FACE_IDX     # eyes + jaw/brow/nose/lip landmarks

_COCO133_GROUPS: dict[str, frozenset[int]] = {
    "Face":          _FACE_MARKER_IDX,
    "Face (detail)": _FACE_DETAIL_IDX,
    "Left arm":    frozenset({5, 7, 9}),
    "Right arm":   frozenset({6, 8, 10}),
    "Left hand":   _LEFT_HAND_IDX,
    "Right hand":  _RIGHT_HAND_IDX,
    "Left leg":    frozenset({11, 13, 15}) | _LEFT_FOOT_IDX,
    "Right leg":   frozenset({12, 14, 16}) | _RIGHT_FOOT_IDX,
    "Left foot":   _LEFT_FOOT_IDX,
    "Right foot":  _RIGHT_FOOT_IDX,
    "Body":        _BODY17_IDX,
    "Upper body":  frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) | _FACE_IDX,
    "Lower body":  frozenset({11, 12, 13, 14, 15, 16}) | _LEFT_FOOT_IDX | _RIGHT_FOOT_IDX,
}

_COCO133_TREE_GROUPS: tuple[str, ...] = (
    "Face", "Face (detail)", "Left arm", "Right arm", "Left hand", "Right hand",
    "Left leg", "Right leg",
)

_COCO133_LIMB_CHAINS: dict[str, tuple[str, ...]] = {
    # Same 3-point subset as _FACE_MARKER_IDX above -- the only face points
    # the default skeleton actually rigs.
    "Face":       ("nose", "left_ear", "right_ear"),
    "Left arm":   ("left_shoulder", "left_elbow", "left_wrist", "left_index_1", "left_pinky_1"),
    "Right arm":  ("right_shoulder", "right_elbow", "right_wrist", "right_index_1", "right_pinky_1"),
    "Left leg":   ("left_hip", "left_knee", "left_ankle", "left_heel", "left_big_toe", "left_small_toe"),
    "Right leg":  ("right_hip", "right_knee", "right_ankle", "right_heel", "right_big_toe", "right_small_toe"),
}

COCO133 = PoseModel(
    model_id="coco-133",
    names=_COCO133_NAMES,
    groups=_COCO133_GROUPS,
    tree_groups=_COCO133_TREE_GROUPS,
    limb_chains=_COCO133_LIMB_CHAINS,
)


def _assert_tree_partitions(model: PoseModel) -> None:
    """Verify tree_groups covers every keypoint index exactly once."""
    seen: set[int] = set()
    for group in model.tree_groups:
        idx = model.groups[group]
        overlap = seen & idx
        assert not overlap, f"{model.model_id}: {group!r} overlaps prior tree groups at {overlap}"
        seen |= idx
    assert seen == model.all_indices, (
        f"{model.model_id}: tree_groups cover {sorted(seen)}, "
        f"missing {sorted(model.all_indices - seen)}"
    )


_assert_tree_partitions(COCO17)
_assert_tree_partitions(COCO133)


# ---------------------------------------------------------------------------
# Registry  (model_name as stored in the DB → PoseModel)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, PoseModel] = {
    # canonical names
    "coco-17":          COCO17,
    "coco-133":         COCO133,
    # legacy / alternate spellings stored in older DBs
    "COCO17":           COCO17,
    "COCO133":          COCO133,
    # RTMPose model strings (as stored by the detection pipeline)
    "rtmpose-l-17kp":   COCO17,
    "rtmpose-m-17kp":   COCO17,
    "rtmpose-l-133kp":  COCO133,
    "rtmpose-m-133kp":  COCO133,
}


def get_pose_model(model_name: str | None) -> PoseModel:
    """Return PoseModel for *model_name* (from DB), falling back to COCO-17."""
    if model_name:
        m = _REGISTRY.get(model_name)
        if m is not None:
            return m
        # Case-insensitive fallback: normalise to lowercase with dash
        key = model_name.lower().replace("_", "-")
        m = _REGISTRY.get(key)
        if m is not None:
            return m
        if "133" in model_name:
            return COCO133
    return COCO17
