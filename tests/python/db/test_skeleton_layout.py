"""Tests for scripts/db/skeleton_layout.py."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[3]))

from scripts.db.skeleton_layout import SkeletonLayout

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SIMPLE_HUMANOID_YAML = (Path(__file__).parents[3] / "tests/data/simple_humanoid.yaml").read_text()

MINIMAL_YAML = """\
name: minimal
joints:
  - name: root
    type: root
    parent: null
    offset: [0.0, 0.0, 0.0]
  - name: spine
    type: ball
    parent: root
    offset: [0.0, 0.1, 0.0]
    limits:
      x: [-0.5, 0.5]
      y: [-0.3, 0.3]
      z: [-0.2, 0.2]
  - name: head
    type: revolute
    parent: spine
    offset: [0.0, 0.15, 0.0]
    axis: [1.0, 0.0, 0.0]
    limits: [-0.5, 0.5]
markers:
  - name: nose
    parent: head
    offset: [0.0, 0.05, 0.0]
    openpose_keypoint: 0
"""

LOCKED_DOF_YAML = """\
name: locked_test
joints:
  - name: root
    type: root
    parent: null
    offset: [0.0, 0.0, 0.0]
  - name: hinge
    type: ball
    parent: root
    offset: [0.0, 0.1, 0.0]
    limits:
      x: [-0.5, 0.5]
      y: [0.0, 0.0]
      z: [0.0, 0.0]
markers: []
"""

SCALE_GROUP_YAML = """\
name: scale_test
joints:
  - name: root
    type: root
    parent: null
    offset: [0.0, 0.0, 0.0]
  - name: upper_arm
    type: ball
    parent: root
    offset: [0.0, 0.2, 0.0]
    limits:
      x: [-1.0, 1.0]
      y: [-1.0, 1.0]
      z: [-1.0, 1.0]
  - name: lower_arm
    type: ball
    parent: upper_arm
    offset: [0.0, 0.2, 0.0]
    limits:
      x: [-1.0, 1.0]
      y: [-1.0, 1.0]
      z: [-1.0, 1.0]
scale_groups:
  - name: arm_len
    joints: [upper_arm, lower_arm]
markers: []
"""


def _make_blob(n_dof: int, values: dict | None = None) -> bytes:
    """Build a minimal state blob with n_dof joint angles all zero, except overrides."""
    arr = np.zeros(12 + 2 * n_dof, dtype="<f8")
    if values:
        for idx, val in values.items():
            arr[idx] = val
    return arr.tobytes()


# ---------------------------------------------------------------------------
# SkeletonLayout construction
# ---------------------------------------------------------------------------


class TestSkeletonLayoutConstruction:
    def test_simple_humanoid_n_dof(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        assert layout.n_dof == 23

    def test_simple_humanoid_joint_count(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        # 9 non-root non-fixed joints (no scale followers)
        non_follower = [j for j in layout.joints if not j.is_scale_follower]
        assert len(non_follower) == 9

    def test_simple_humanoid_root_excluded(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        names = [j.name for j in layout.joints]
        assert "pelvis" not in names

    def test_spherical_storage_dof_always_3(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        for j in layout.joints:
            if j.joint_type == "spherical":
                assert j.storage_dof == 3

    def test_revolute_storage_dof_is_1(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        revolutes = [j for j in layout.joints if j.joint_type == "revolute"]
        assert len(revolutes) == 2
        for j in revolutes:
            assert j.storage_dof == 1

    def test_state_indices_sequential(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        expected_idx = 0
        for j in layout.joints:
            if j.is_scale_follower:
                continue
            assert j.state_index == expected_idx
            expected_idx += j.storage_dof

    def test_minimal_yaml_n_dof(self):
        layout = SkeletonLayout(MINIMAL_YAML)
        # spine (ball=3) + head (revolute=1) = 4
        assert layout.n_dof == 4

    def test_minimal_yaml_joint_order(self):
        layout = SkeletonLayout(MINIMAL_YAML)
        names = [j.name for j in layout.joints if not j.is_scale_follower]
        assert names == ["spine", "head"]

    def test_root_joint_name(self):
        layout = SkeletonLayout(MINIMAL_YAML)
        assert layout.root_joint_name() == "root"


# ---------------------------------------------------------------------------
# Locked DOF handling
# ---------------------------------------------------------------------------


class TestLockedDOF:
    def test_locked_axes_reduce_active_dof(self):
        layout = SkeletonLayout(LOCKED_DOF_YAML)
        hinge = next(j for j in layout.joints if j.name == "hinge")
        # x is free, y and z are locked (0,0)
        assert hinge.active_mask[0] is True
        assert hinge.active_mask[1] is False
        assert hinge.active_mask[2] is False

    def test_locked_dof_still_has_3_storage_slots(self):
        layout = SkeletonLayout(LOCKED_DOF_YAML)
        hinge = next(j for j in layout.joints if j.name == "hinge")
        assert hinge.storage_dof == 3

    def test_locked_dof_n_dof_still_3(self):
        layout = SkeletonLayout(LOCKED_DOF_YAML)
        assert layout.n_dof == 3


# ---------------------------------------------------------------------------
# Scale group handling
# ---------------------------------------------------------------------------


class TestScaleGroups:
    def test_prismatic_joints_inserted(self):
        layout = SkeletonLayout(SCALE_GROUP_YAML)
        names = [j.name for j in layout.joints]
        assert "prismatic_upper_arm" in names
        assert "prismatic_lower_arm" in names

    def test_leader_has_storage_dof_1(self):
        layout = SkeletonLayout(SCALE_GROUP_YAML)
        leader = next(j for j in layout.joints if j.name == "prismatic_upper_arm")
        assert leader.storage_dof == 1
        assert leader.is_scale_follower is False

    def test_follower_has_storage_dof_0(self):
        layout = SkeletonLayout(SCALE_GROUP_YAML)
        follower = next(j for j in layout.joints if j.name == "prismatic_lower_arm")
        assert follower.storage_dof == 0
        assert follower.is_scale_follower is True

    def test_follower_shares_leader_state_index(self):
        layout = SkeletonLayout(SCALE_GROUP_YAML)
        leader = next(j for j in layout.joints if j.name == "prismatic_upper_arm")
        follower = next(j for j in layout.joints if j.name == "prismatic_lower_arm")
        assert follower.state_index == leader.state_index

    def test_scale_group_adds_1_dof_not_2(self):
        layout = SkeletonLayout(SCALE_GROUP_YAML)
        # upper_arm(ball=3) + prismatic_leader(1) + lower_arm(ball=3) + prismatic_follower(0) = 7
        assert layout.n_dof == 7


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


class TestMarkers:
    def test_marker_count(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        assert len(layout.markers) == 8

    def test_marker_fields(self):
        layout = SkeletonLayout(MINIMAL_YAML)
        assert len(layout.markers) == 1
        m = layout.markers[0]
        name = m["name"] if isinstance(m, dict) else m.name
        assert name == "nose"

    def test_marker_coco_ids(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        coco_ids = set()
        for m in layout.markers:
            kid = m["openpose_keypoint"] if isinstance(m, dict) else m.openpose_keypoint
            coco_ids.add(kid)
        assert 8 in coco_ids  # pelvis_center
        assert 2 in coco_ids  # r_shoulder_marker


# ---------------------------------------------------------------------------
# State blob decoding
# ---------------------------------------------------------------------------


class TestDecodeStateBlob:
    def test_zero_blob_gives_identity_quat(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        blob = _make_blob(layout.n_dof)
        dec = layout.decode_state_blob(blob)
        np.testing.assert_allclose(dec["quat"], [1.0, 0.0, 0.0, 0.0], atol=1e-9)

    def test_position_decoded_correctly(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        blob = _make_blob(layout.n_dof, {0: 1.0, 1: 2.0, 2: 3.0})
        dec = layout.decode_state_blob(blob)
        np.testing.assert_allclose(dec["pos"], [1.0, 2.0, 3.0])

    def test_axis_angle_to_quat(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        # 90° rotation around Z axis → axis-angle [0, 0, π/2]
        angle = math.pi / 2.0
        blob = _make_blob(layout.n_dof, {3: 0.0, 4: 0.0, 5: angle})
        dec = layout.decode_state_blob(blob)
        expected_w = math.cos(angle / 2)
        expected_z = math.sin(angle / 2)
        np.testing.assert_allclose(dec["quat"][0], expected_w, atol=1e-9)
        np.testing.assert_allclose(dec["quat"][3], expected_z, atol=1e-9)

    def test_joint_angles_keys_match_non_follower_joints(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        blob = _make_blob(layout.n_dof)
        dec = layout.decode_state_blob(blob)
        expected = {j.name for j in layout.joints if not j.is_scale_follower}
        assert set(dec["joint_angles"].keys()) == expected

    def test_spherical_joint_angle_shape(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        blob = _make_blob(layout.n_dof)
        dec = layout.decode_state_blob(blob)
        assert dec["joint_angles"]["spine_lower"].shape == (3,)

    def test_revolute_joint_angle_has_zero_yz(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        blob = _make_blob(layout.n_dof)
        dec = layout.decode_state_blob(blob)
        aa = dec["joint_angles"]["r_elbow"]
        assert aa[1] == 0.0
        assert aa[2] == 0.0

    def test_joint_angle_values_correct(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        # spine_lower starts at state_index=0; set [0.1, 0.2, 0.3]
        spine_lower = next(j for j in layout.joints if j.name == "spine_lower")
        idx = spine_lower.state_index + 6  # offset past pos(3) + aa(3)
        blob = _make_blob(layout.n_dof, {idx: 0.1, idx + 1: 0.2, idx + 2: 0.3})
        dec = layout.decode_state_blob(blob)
        np.testing.assert_allclose(dec["joint_angles"]["spine_lower"], [0.1, 0.2, 0.3])

    def test_root_velocity_decoded(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        n = layout.n_dof
        blob = _make_blob(n, {6 + n: 1.0, 6 + n + 1: 2.0, 6 + n + 2: 3.0})
        dec = layout.decode_state_blob(blob)
        np.testing.assert_allclose(dec["root_vel"], [1.0, 2.0, 3.0])

    def test_wrong_blob_size_raises(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        with pytest.raises((ValueError, Exception)):
            layout.decode_state_blob(b"\x00" * 10)


# ---------------------------------------------------------------------------
# Forward kinematics — rest pose
# ---------------------------------------------------------------------------


class TestForwardKinematics:
    def test_root_at_origin_in_rest_pose(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        blob = _make_blob(layout.n_dof)
        dec = layout.decode_state_blob(blob)
        transforms = layout.compute_joint_transforms(dec)
        pelvis_pos = transforms["pelvis"][:3, 3]
        np.testing.assert_allclose(pelvis_pos, [0.0, 0.0, 0.0], atol=1e-9)

    def test_spine_lower_at_offset_in_rest_pose(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        blob = _make_blob(layout.n_dof)
        dec = layout.decode_state_blob(blob)
        transforms = layout.compute_joint_transforms(dec)
        # spine_lower has offset [0, 0.1, 0] from pelvis at origin
        pos = transforms["spine_lower"][:3, 3]
        np.testing.assert_allclose(pos, [0.0, 0.1, 0.0], atol=1e-9)

    def test_root_position_applied(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        blob = _make_blob(layout.n_dof, {0: 1.0, 1: 2.0, 2: 3.0})
        dec = layout.decode_state_blob(blob)
        transforms = layout.compute_joint_transforms(dec)
        pelvis_pos = transforms["pelvis"][:3, 3]
        np.testing.assert_allclose(pelvis_pos, [1.0, 2.0, 3.0], atol=1e-9)

    def test_pelvis_center_marker_at_origin_rest_pose(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        blob = _make_blob(layout.n_dof)
        dec = layout.decode_state_blob(blob)
        positions = layout.compute_marker_positions(dec)
        np.testing.assert_allclose(positions["pelvis_center"], [0.0, 0.0, 0.0], atol=1e-9)

    def test_spine_base_marker_above_pelvis(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        blob = _make_blob(layout.n_dof)
        dec = layout.decode_state_blob(blob)
        positions = layout.compute_marker_positions(dec)
        # spine_base is parented to spine_lower (offset [0, 0.1, 0] from pelvis)
        # with local offset [0, 0.05, 0] → world y = 0.15
        assert positions["spine_base"][1] > positions["pelvis_center"][1]

    def test_all_markers_computed(self):
        layout = SkeletonLayout(SIMPLE_HUMANOID_YAML)
        blob = _make_blob(layout.n_dof)
        dec = layout.decode_state_blob(blob)
        positions = layout.compute_marker_positions(dec)
        expected_names = {
            "pelvis_center", "spine_base", "r_shoulder_marker",
            "r_elbow_marker", "r_wrist_marker", "l_shoulder_marker",
            "l_elbow_marker", "l_wrist_marker",
        }
        assert set(positions.keys()) == expected_names
