# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""build_dot_augmented_skeleton refuses an attachment set written for another skeleton."""
from __future__ import annotations

import pytest
import yaml

from posetrak.markers.catalog import MarkerSetError
from posetrak.markers.topology import skeleton_topology
from tools.build_dot_augmented_skeleton import build_dot_augmented_skeleton


def _base(name: str = "reallusion-no-waist") -> str:
    return yaml.safe_dump({
        "name": name,
        "joints": [
            {"name": "pelvis", "type": "root", "parent": None, "offset": [0, 0, 0]},
            {"name": "shin.R", "type": "ball", "parent": "pelvis", "offset": [0, -0.4, 0], "bone_tip_offset": [0, -0.4, 0]},
        ],
    })


def _attachment_set(**header) -> dict:
    return {
        "module": "leg",
        "markers": [{
            "name": "knee_lat_R", "parent_joint": "shin.R", "along": 0.06, "lateral": 0.045, "anterior": 0.005,
            "offset": [0.045, 0.06, 0.005], "normal": [0.0, 1.0, 0.1],
        }],
        **header,
    }


def test_a_matching_attachment_set_adds_its_markers_and_keeps_the_topology() -> None:
    base = _base()
    doc = _attachment_set(requires_topology="reallusion-no-waist", requires_topology_hash=skeleton_topology(base).hash)

    merged = build_dot_augmented_skeleton(base, doc)

    skeleton = yaml.safe_load(merged)
    assert [m["name"] for m in skeleton["markers"]] == ["knee_lat_R"]
    assert skeleton_topology(merged).hash == skeleton_topology(base).hash


def test_an_attachment_set_for_another_declared_topology_is_refused() -> None:
    base = yaml.safe_load(_base())
    base["topology"] = {"name": "mixamo", "version": 1}

    with pytest.raises(MarkerSetError, match="needs topology 'reallusion-no-waist' but the skeleton is 'mixamo'"):
        build_dot_augmented_skeleton(yaml.safe_dump(base), _attachment_set(requires_topology="reallusion-no-waist"))


def test_an_attachment_set_for_another_structure_is_refused() -> None:
    with pytest.raises(MarkerSetError, match="skeleton structure .* differs"):
        build_dot_augmented_skeleton(_base(), _attachment_set(requires_topology_hash="0" * 64))


def test_an_attachment_set_naming_joints_the_skeleton_lacks_is_refused() -> None:
    doc = _attachment_set()
    doc["markers"][0]["parent_joint"] = "shin.L"

    with pytest.raises(MarkerSetError, match=r"no joint shin\.L"):
        build_dot_augmented_skeleton(_base(), doc)
