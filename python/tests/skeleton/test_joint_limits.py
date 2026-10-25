# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""The anatomical joint-limit table of the default skeletons."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from posetrak.skeleton.compose import bundled_hierarchy, bundled_joint_limits

DATA = Path(__file__).parent / "data"

# A hinge bends about x only; y (twist) and z (side bend) stay locked.
HINGE_JOINT_PREFIXES = ("shin.", "forearm.", "f_index.02.", "f_index.03.", "f_middle.02.", "f_middle.03.",
                        "f_ring.02.", "f_ring.03.", "f_pinky.02.", "f_pinky.03.", "thumb.02.", "thumb.03.")
OFF_AXIS_TOLERANCE = 0.0   # radians


def hinge_violations(limits: dict[str, dict[str, list[float]]]) -> list[str]:
    """Hinge joints (knee, elbow, finger interphalangeal) with range on an axis other than x."""
    return [f"{joint} {axis}: {lo}..{hi}"
            for joint, axes in limits.items() if joint.startswith(HINGE_JOINT_PREFIXES)
            for axis in ("y", "z") for lo, hi in [axes[axis]]
            if lo < -OFF_AXIS_TOLERANCE or hi > OFF_AXIS_TOLERANCE]


def test_hinge_joints_have_no_off_axis_range():
    assert hinge_violations(bundled_joint_limits()) == []


def test_the_hinge_check_fails_on_the_earlier_limits():
    earlier = yaml.safe_load((DATA / "legacy_joint_limits.yaml").read_text(encoding="utf-8"))["joints"]
    violations = hinge_violations(earlier)
    assert any(v.startswith("shin.L") for v in violations)
    assert any(v.startswith("forearm.R") for v in violations)


def test_limits_name_only_joints_of_the_hierarchy_and_are_ordered():
    joints = {j["name"] for j in bundled_hierarchy()["joints"]}
    limits = bundled_joint_limits()
    assert set(limits) <= joints
    for joint, axes in limits.items():
        assert set(axes) == {"x", "y", "z"}, joint
        assert all(lo <= hi for lo, hi in axes.values()), joint


@pytest.mark.parametrize("side", ["L", "R"])
def test_forearm_twist_is_carried_by_the_wrist(side):
    limits = bundled_joint_limits()
    assert limits[f"forearm.{side}"]["y"] == [0.0, 0.0]
    lo, hi = limits[f"hand.{side}"]["y"]
    assert lo <= -1.4 and hi >= 1.4
