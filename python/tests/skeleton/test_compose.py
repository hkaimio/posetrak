# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Composition of a skeleton from its layers."""
from __future__ import annotations

import importlib.resources
from pathlib import Path

import pytest
import yaml

from posetrak.detection.pose_models import SOMA77_NAMES
from posetrak.skeleton import compose as c

DATA = Path(__file__).parent / "data"


def _load(name: str) -> dict:
    return yaml.safe_load((DATA / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("sex", c.SEXES)
def test_the_earlier_limits_with_the_coco_sets_reproduce_the_earlier_default(sex):
    """Hierarchy + dimensions + COCO-133 and hand21 markers + the earlier limits = the default skeleton as it
    was. The markers are listed set by set instead of interleaved, so they are compared by name."""
    earlier = _load(f"default_{sex}.legacy.yaml")
    composed = c.compose_skeleton(
        c.bundled_hierarchy(), _load("legacy_joint_limits.yaml")["joints"],
        [c.bundled_marker_set("coco133"), c.bundled_marker_set("hand21")], c.bundled_dimensions(sex),
        landmarks=False)
    assert set(composed) == set(earlier)
    assert composed["joints"] == earlier["joints"]
    assert composed["groups"] == earlier["groups"]
    assert (composed["name"], composed["units"]) == (earlier["name"], earlier["units"])
    by_name = lambda doc: sorted(doc["markers"], key=lambda m: m["name"])  # noqa: E731
    assert by_name(composed) == by_name(earlier)


@pytest.mark.parametrize("sex", c.SEXES)
def test_the_bundled_default_skeleton_is_its_composition(sex):
    bundled = importlib.resources.files("posetrak.data.skeletons").joinpath(f"default_{sex}.yaml")
    assert bundled.read_text(encoding="utf-8") == c.compose_default(sex)


def test_default_skeleton_carries_both_marker_sets_by_landmark():
    doc = yaml.safe_load(c.compose_default("male"))
    by_landmark = {m["landmark"]: m for m in doc["markers"]}
    assert len(by_landmark) == len(doc["markers"])
    soma = [m for m in doc["markers"] if m["landmark"].startswith("soma77:")]
    assert len(soma) == 33
    names = {m["name"] for m in soma}
    assert names >= {"LeftHandIndex1", "LeftHandPinky1", "RightHandIndex1", "RightHandPinky1", "Hips", "RightToeEnd"}
    assert not names & {n for n in SOMA77_NAMES if "Thumb" in n or "Middle" in n or "Ring" in n}
    main = next(g for g in doc["groups"] if g["name"] == "main")
    assert names <= set(main["markers"])
    for hand_group in ("HandL", "HandR"):
        assert not names & set(next(g for g in doc["groups"] if g["name"] == hand_group)["markers"])
    # the COCO markers keep their keypoint index and gain model-qualified landmarks
    assert by_landmark["coco133:left_elbow"]["openpose_keypoint"] == 7
    assert by_landmark["hand21:left_index_1"]["openpose_keypoint"] == 96
    assert by_landmark["hand21:right_pinky_4"]["openpose_keypoint"] == 132


def test_every_marker_sits_on_an_existing_joint_and_group_lists_name_existing_markers():
    doc = yaml.safe_load(c.compose_default("female"))
    joints = {j["name"] for j in doc["joints"]}
    assert all(m["parent"] in joints for m in doc["markers"])
    markers = {m["name"] for m in doc["markers"]}
    assert all(set(g["markers"]) <= markers for g in doc["groups"])


def test_add_pose_model_adds_the_soma_markers_to_a_skeleton_of_the_users_own():
    earlier = (DATA / "default_male.legacy.yaml").read_text(encoding="utf-8")
    text, skipped = c.add_pose_model(earlier, "soma77")
    doc = yaml.safe_load(text)
    assert skipped == []
    assert sum(m.get("landmark", "").startswith("soma77:") for m in doc["markers"]) == 33
    assert len(doc["markers"]) == len(yaml.safe_load(earlier)["markers"]) + 33
    assert [j["name"] for j in doc["joints"]] == [j["name"] for j in yaml.safe_load(earlier)["joints"]]
    with pytest.raises(c.SkeletonLayerError, match="already"):
        c.add_pose_model(text, "soma77")


def test_add_pose_model_skips_markers_whose_joint_the_skeleton_lacks():
    doc = yaml.safe_load((DATA / "default_male.legacy.yaml").read_text(encoding="utf-8"))
    doc["joints"] = [j for j in doc["joints"] if not j["name"].startswith(("toe.",))]
    doc["markers"] = [m for m in doc["markers"] if not m["parent"].startswith("toe.")]
    for g in doc["groups"]:
        g["joints"] = [j for j in g["joints"] if not j.startswith("toe.")]
    _, skipped = c.add_pose_model(yaml.safe_dump(doc), "soma77")
    assert sorted(skipped) == ["LeftToeBase", "LeftToeEnd", "RightToeBase", "RightToeEnd"]


def test_compose_rejects_limits_and_dimensions_that_do_not_match_the_hierarchy():
    hierarchy, dims = c.bundled_hierarchy(), c.bundled_dimensions("male")
    with pytest.raises(c.SkeletonLayerError, match="unknown joint"):
        c.compose_skeleton(hierarchy, {"no.such.joint": {}}, [], dims)
    short = {"joints": {k: v for k, v in dims["joints"].items() if k != "head"}}
    with pytest.raises(c.SkeletonLayerError, match="head"):
        c.compose_skeleton(hierarchy, {}, [], short)
