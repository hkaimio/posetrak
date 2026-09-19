# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for marker modules, attachment sets and the topology check."""
from __future__ import annotations

import dataclasses

import pytest
import yaml

from posetrak.markers.catalog import (
    MarkerSetError,
    catalog_module,
    check_module_fits_skeleton,
    load_marker_module,
)
from posetrak.markers.topology import skeleton_topology

_LEG_JOINTS = [
    ("pelvis", None, "root", None),
    ("thigh.R", "pelvis", "ball", [0.0, -0.4, 0.0]),
    ("shin.R", "thigh.R", "ball", [0.0, -0.4, 0.0]),
    ("foot.R", "shin.R", "ball", [0.0, 0.0, 0.2]),
    ("thigh.L", "pelvis", "ball", [0.0, -0.4, 0.0]),
    ("shin.L", "thigh.L", "ball", [0.0, -0.4, 0.0]),
    ("foot.L", "shin.L", "ball", [0.0, 0.0, 0.2]),
]


def _skeleton(name: str = "reallusion-no-waist", *, scale: float = 1.0, joints=_LEG_JOINTS, extra: dict | None = None) -> str:
    doc = {
        "name": name,
        "joints": [
            {"name": n, "type": t, "parent": p, "offset": [0.0, 0.1 * scale, 0.0],
             **({"bone_tip_offset": [v * scale for v in tip]} if tip else {})}
            for n, p, t, tip in joints
        ],
    }
    doc.update(extra or {})
    return yaml.safe_dump(doc)


def _leg_for(skeleton_yaml: str):
    """The leg module, pinned to this skeleton's structure instead of the real rig's."""
    return dataclasses.replace(catalog_module("leg"), requires_topology_hash=skeleton_topology(skeleton_yaml).hash)


# What fit_calibrated_attachment_set.py held in code before the module file existed.
_OLD_SLOT_PARENT_BASE = {
    "hip": "thigh",
    "knee_lat": "shin", "knee_med": "shin", "knee_front": "shin",
    "ankle_lat": "shin", "ankle_med": "shin",
    "heel": "foot", "toe": "foot",
}
# The slot list label_tracklet_groups_gui.py held.
_OLD_SLOT_NAMES = [
    "hip_R", "hip_L", "knee_lat_R", "knee_lat_L", "knee_med_R", "knee_med_L",
    "knee_front_R", "knee_front_L", "ankle_lat_R", "ankle_lat_L", "ankle_med_R", "ankle_med_L",
    "heel_R", "heel_L", "toe_R", "toe_L",
]


class TestLegModule:
    def test_it_holds_the_slots_and_parent_joints_the_scripts_used_to_hardcode(self) -> None:
        leg = catalog_module("leg")

        assert leg.slot_names() == _OLD_SLOT_NAMES
        for name in _OLD_SLOT_NAMES:
            base, side = name.rsplit("_", 1)
            assert leg.parent_joint(name) == f"{_OLD_SLOT_PARENT_BASE[base]}.{side}"

    def test_it_is_pinned_to_the_structure_of_its_rig(self) -> None:
        leg = catalog_module("leg")

        assert leg.requires_topology == "reallusion-no-waist" and len(leg.requires_topology_hash) == 64
        with pytest.raises(MarkerSetError, match="skeleton structure .* differs"):
            check_module_fits_skeleton(leg, _skeleton())

    def test_it_fits_a_skeleton_of_its_structure(self) -> None:
        skeleton = _skeleton()
        check_module_fits_skeleton(_leg_for(skeleton), skeleton)

    def test_an_unknown_module_lists_the_available_ones(self) -> None:
        with pytest.raises(MarkerSetError, match=r"no catalog module 'arm'.*modules: leg"):
            catalog_module("arm")

    def test_an_unknown_slot_is_a_key_error(self) -> None:
        with pytest.raises(KeyError):
            catalog_module("leg").parent_joint("nose_R")


class TestLoading:
    def test_a_mirrored_marker_gets_a_left_twin_with_reflected_geometry(self) -> None:
        module = load_marker_module("""
module: m
markers:
  - {name: knee_R, parent_joint: shin.R, mirror: true, along: 0.06, lateral: 0.045, anterior: 0.005,
     normal: [0.0, 1.0, 0.1]}
  - {name: sternum, parent_joint: spine, along: 0.5, lateral: 0.0, anterior: 0.1}
""")

        assert module.slot_names() == ["knee_R", "knee_L", "sternum"]
        right, left, sternum = module.slots
        assert (left.parent_joint, left.along, left.lateral, left.anterior) == ("shin.L", 0.06, -0.045, 0.005)
        assert left.normal == (0.0, -1.0, 0.1) and right.normal == (0.0, 1.0, 0.1)
        assert sternum.normal is None

    def test_a_fitted_attachment_set_loads_with_both_sides_explicit(self) -> None:
        module = load_marker_module("""
module: leg
skeleton_topology: reallusion-no-waist
requires_joints: [foot, shin, thigh]
calibration: {session: s, method: fit}
markers:
  - {name: heel_R, parent_joint: foot.R, along: 0.9, lateral: 0.0, anterior: -0.05, offset: [0, 0.1, 0], n_samples: 12}
  - {name: heel_L, parent_joint: foot.L, along: 0.9, lateral: 0.0, anterior: -0.05, offset: [0, 0.1, 0], n_samples: 9}
""")

        assert module.slot_names() == ["heel_R", "heel_L"]
        assert module.requires_topology == "reallusion-no-waist"      # the older key is understood
        assert module.requires_joints == ("foot", "shin", "thigh")

    @pytest.mark.parametrize(
        ("document", "message"),
        [
            ("markers: [{name: a, parent_joint: j}]", "no 'module' name"),
            ("module: m\nmarkers: []", "no markers"),
            ("module: m\nmarkers: [{name: a}]", "needs a name and a parent_joint"),
            ("module: m\nmarkers: [{name: a, parent_joint: j}, {name: a, parent_joint: k}]", "duplicate marker names: a"),
            ("module: m\nmarkers: [{name: a, parent_joint: j, along: 0.1}]", "gives along but needs all of"),
            ("module: m\nmarkers: [{name: knee_L, parent_joint: shin.L, mirror: true}]", "must be written for the right side"),
            ("module: m\nmarkers: [{name: knee_R, parent_joint: shin, mirror: true}]", "must be written for the right side"),
            ("module: m\nmarkers: [{name: knee_R, parent_joint: shin.R, mirror: true}, {name: knee_L, parent_joint: x}]",
             "duplicate marker names: knee_L"),
        ],
    )
    def test_a_malformed_document_is_rejected_with_the_reason(self, document: str, message: str) -> None:
        with pytest.raises(MarkerSetError, match=message):
            load_marker_module(document)


class TestSkeletonFit:
    def test_a_scaled_skeleton_and_one_with_markers_added_share_the_topology(self) -> None:
        base = skeleton_topology(_skeleton())
        scaled = skeleton_topology(_skeleton(scale=1.15))
        augmented = skeleton_topology(_skeleton(extra={
            "input_tracks": [{"id": "dots", "type": "unlabeled_points"}],
            "markers": [{"name": "heel_R", "parent": "foot.R", "offset": [0, 0, 0]}],
        }))

        assert base.hash == scaled.hash == augmented.hash
        assert base.name == "reallusion-no-waist" and base.version is None

    def test_a_structural_difference_changes_the_hash(self) -> None:
        base = skeleton_topology(_skeleton()).hash
        reparented = [(n, "pelvis" if n == "foot.R" else p, t, tip) for n, p, t, tip in _LEG_JOINTS]
        retyped = [(n, p, "hinge" if n == "shin.R" else t, tip) for n, p, t, tip in _LEG_JOINTS]
        turned = [(n, p, t, [0.0, 0.4, 0.0] if n == "shin.R" else tip) for n, p, t, tip in _LEG_JOINTS]
        renamed = [("leg.R" if n == "thigh.R" else n, p, t, tip) for n, p, t, tip in _LEG_JOINTS]

        assert len({skeleton_topology(_skeleton(joints=j)).hash for j in (reparented, retyped, turned, renamed)} | {base}) == 5

    def test_the_hash_does_not_depend_on_joint_order(self) -> None:
        shuffled = list(reversed(_LEG_JOINTS))
        assert skeleton_topology(_skeleton(joints=shuffled)).hash == skeleton_topology(_skeleton()).hash

    def test_a_declared_topology_block_names_the_topology(self) -> None:
        topology = skeleton_topology(_skeleton(name="Scaled_Harri", extra={"topology": {"name": "reallusion-no-waist", "version": 2}}))
        assert (topology.name, topology.version) == ("reallusion-no-waist", 2)

    def test_a_module_for_another_declared_topology_is_refused(self) -> None:
        skeleton = _skeleton(extra={"topology": {"name": "mixamo", "version": 1}})

        with pytest.raises(MarkerSetError, match="needs topology 'reallusion-no-waist' but the skeleton is 'mixamo'"):
            check_module_fits_skeleton(_leg_for(skeleton), skeleton)

    def test_a_skeleton_named_after_a_person_is_not_refused_for_its_name(self) -> None:
        skeleton = _skeleton(name="Nelli scale attempt")

        check_module_fits_skeleton(_leg_for(skeleton), skeleton)

    def test_missing_joints_are_all_named(self) -> None:
        no_left_foot = [j for j in _LEG_JOINTS if j[0] != "foot.L"]

        with pytest.raises(MarkerSetError, match=r"the skeleton has no joint foot\.L"):
            skeleton = _skeleton(joints=no_left_foot)
            check_module_fits_skeleton(_leg_for(skeleton), skeleton)

        renamed = [(n.replace("shin", "tibia"), (p or "").replace("shin", "tibia") or None, t, tip) for n, p, t, tip in _LEG_JOINTS]
        with pytest.raises(MarkerSetError) as error:
            skeleton = _skeleton(joints=renamed)
            check_module_fits_skeleton(_leg_for(skeleton), skeleton)
        assert "shin.L, shin.R" in str(error.value) and "no joint named shin" in str(error.value)

    def test_a_required_structure_must_match_exactly(self) -> None:
        skeleton = _skeleton()
        good = {"module": "m", "requires_topology_hash": skeleton_topology(skeleton).hash,
                "markers": [{"name": "a", "parent_joint": "pelvis"}]}
        bad = {**good, "requires_topology_hash": "0" * 64}

        check_module_fits_skeleton(good, skeleton)
        with pytest.raises(MarkerSetError, match="skeleton structure .* differs"):
            check_module_fits_skeleton(bad, skeleton)

    def test_a_skeleton_without_joints_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no joints"):
            skeleton_topology("name: x\n")
