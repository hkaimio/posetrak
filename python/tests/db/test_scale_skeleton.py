# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for posetrak/db/scale_skeleton.py.

No prior test coverage existed for this module (found 2026-09-14, while
adding hip_to_ear-derived neck+head scaling) -- these cover the
pre-existing limb/torso/shoulder-width scaling plus the new one, on a
small synthetic skeleton rather than a real multi-hundred-joint one.
"""

from __future__ import annotations

import pytest
import yaml

from posetrak.db.scale_skeleton import scale_skeleton_yaml, template_measurements

# A minimal skeleton with just enough of the real joint chain (hips -> spine1
# -> spine2 -> {shoulder.L/R, neck1 -> neck2 -> head}, plus one leg and one
# arm) to exercise every scaling rule without a full body's worth of joints.
# All offsets are purely along Y (no X/Z component) so the expected results
# are exact, not just approximately-Y-dominant like a real skeleton.
_SKELETON_YAML = """
name: test-skeleton
units: meters
joints:
  - {name: hips,          type: free,      parent: null}
  - {name: spine1,        type: ball,      parent: hips,     offset: [0.0, 0.10, 0.0]}
  - {name: spine2,        type: ball,      parent: spine1,   offset: [0.0, 0.12, 0.0]}
  - {name: shoulder.L,    type: ball,      parent: spine2,   offset: [0.05, 0.20, 0.0]}
  - {name: shoulder.R,    type: ball,      parent: spine2,   offset: [-0.05, 0.20, 0.0]}
  - {name: upper_arm.L,   type: ball,      parent: shoulder.L, offset: [0.10, 0.0, 0.0]}
  - {name: upper_arm.R,   type: ball,      parent: shoulder.R, offset: [-0.10, 0.0, 0.0]}
  - {name: forearm.L,     type: revolute,  parent: upper_arm.L, offset: [0.0, -0.25, 0.0]}
  - {name: hand.L,        type: ball,      parent: forearm.L, offset: [0.0, -0.20, 0.0]}
  - {name: neck1,         type: ball,      parent: spine2,   offset: [0.0, 0.30, 0.0]}
  - {name: neck2,         type: ball,      parent: neck1,    offset: [0.0, 0.04, 0.0]}
  - {name: head,          type: ball,      parent: neck2,    offset: [0.0, 0.04, 0.0]}
  - {name: thigh.L,       type: ball,      parent: hips,     offset: [0.05, -0.02, 0.0]}
  - {name: thigh.R,       type: ball,      parent: hips,     offset: [-0.05, -0.02, 0.0]}
  - {name: shin.L,        type: revolute,  parent: thigh.L,  offset: [0.0, -0.40, 0.0]}
  - {name: foot.L,        type: ball,      parent: shin.L,   offset: [0.0, -0.40, 0.0]}
markers: []
"""


def _joints() -> list[dict]:
    return yaml.safe_load(_SKELETON_YAML)["joints"]


def test_template_measurements_head_is_neck_plus_head_chain() -> None:
    tmpl = template_measurements(_joints())
    # torso_height: hip(-0.02) -> shoulder midpoint(0.10+0.12+0.20=0.42) = 0.44
    assert tmpl["torso_height"] == pytest.approx(0.44)
    # head (derived): hip_to_head_joint - torso_height. head_joint world Y =
    # 0.10+0.12+0.30+0.04+0.04 = 0.60; hip Y = -0.02 -> hip_to_head = 0.62;
    # torso_height = 0.44 -> head = 0.62 - 0.44 = 0.18 (neck1+neck2+head chain).
    assert tmpl["head"] == pytest.approx(0.18)
    # hip_to_ear is head's raw (non-torso-subtracted) counterpart, i.e.
    # hip_to_head_joint itself -- the UI's "Orig" template value for the
    # hip_to_ear card is looked up under this exact key (regression test
    # for a real bug: this key was missing, so the card showed 0.0/no
    # dotted line instead of the skeleton's actual current value).
    assert tmpl["hip_to_ear"] == pytest.approx(0.62)
    # shoulder_width = dist(upper_arm.L, upper_arm.R); each is offset 0.10 in
    # X from its own shoulder.L/R (themselves at X=+-0.05) -> world X = +-0.15.
    assert tmpl["shoulder_width"] == pytest.approx(0.30)
    # upper_arm = (dist(upper_arm.L, forearm.L) + dist(upper_arm.R, forearm.R)) / 2.
    # This fixture only defines an L-side forearm/hand/shin/foot, so the R-side
    # dist() call falls back to 0.0 (missing joint) -- the true L-side length
    # (0.25) is halved by the average, giving 0.125, not 0.25.
    assert tmpl["upper_arm"] == pytest.approx(0.125)


def test_scale_skeleton_yaml_hip_to_ear_scales_neck_and_head_only() -> None:
    tmpl = template_measurements(_joints())
    measurements = {
        "torso_height": tmpl["torso_height"],  # unchanged
        # Ask for a neck+head chain twice as long: hip_to_ear = torso_height + 2*head.
        "hip_to_ear": tmpl["torso_height"] + 2 * tmpl["head"],
    }
    scaled_yaml = scale_skeleton_yaml(_SKELETON_YAML, measurements)
    scaled = {j["name"]: j for j in yaml.safe_load(scaled_yaml)["joints"]}

    assert scaled["neck1"]["offset"][1] == pytest.approx(0.30 * 2)
    assert scaled["neck2"]["offset"][1] == pytest.approx(0.04 * 2)
    assert scaled["head"]["offset"][1] == pytest.approx(0.04 * 2)
    # X/Z components of the scaled joints are untouched (all zero here, but
    # confirms the code only ever touches index 1).
    assert scaled["neck1"]["offset"][0] == pytest.approx(0.0)
    assert scaled["neck1"]["offset"][2] == pytest.approx(0.0)
    # Unrelated chains (torso itself, since torso_height ratio is 1.0; limbs,
    # since no limb measurement was given) are unchanged.
    assert scaled["spine1"]["offset"][1] == pytest.approx(0.10)
    assert scaled["upper_arm.L"]["offset"][0] == pytest.approx(0.10)
    assert scaled["forearm.L"]["offset"][1] == pytest.approx(-0.25)


def test_scale_skeleton_yaml_without_hip_to_ear_leaves_neck_head_unscaled() -> None:
    # Template shoulder_width is 0.30 (see above); ask for double that so a
    # non-1.0 ratio actually exercises the scaling path.
    scaled_yaml = scale_skeleton_yaml(_SKELETON_YAML, {"shoulder_width": 0.60})
    scaled = {j["name"]: j for j in yaml.safe_load(scaled_yaml)["joints"]}
    assert scaled["neck1"]["offset"][1] == pytest.approx(0.30)
    assert scaled["head"]["offset"][1] == pytest.approx(0.04)
    # shoulder_width did apply (2x ratio), to confirm this run wasn't a no-op.
    assert scaled["upper_arm.L"]["offset"][0] == pytest.approx(0.20)
