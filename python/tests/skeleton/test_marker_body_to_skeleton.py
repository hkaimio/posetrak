# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for posetrak.skeleton.marker_body_to_skeleton (design phase 1b).

See docs/roadmap/features/marker-based-mocap/marker-mocap-design.md §7.1.
Validated per that sub-phase's own criterion: the generated YAML is diffed
against a hand-verified expected structure, then loaded through the
existing SkeletonLayout parser (used elsewhere for FK/visualization) to
confirm it is a valid root-only, no-extra-joints skeleton with the right
marker count -- SkeletonLayout is the closest thing to "the existing
SkeletonLoader unit-test harness" available on the Python side; the C++
SkeletonLoader itself is exercised once sub-phase 1f binds input_tracks.
"""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from app.setup.fiducial_markers import load_marker_body_yaml
from posetrak.db.skeleton_layout import SkeletonLayout
from posetrak.skeleton.marker_body_to_skeleton import (
    generate_prop_skeleton,
    generate_prop_skeleton_yaml,
)

_TWO_MARKER_BODY_YAML = """\
name: test-bokken
units: meters
markers:
  - name: hilt
    type: aruco
    dictionary: DICT_4X4_50
    id: "3"
    size: 0.05
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
  - name: tip
    type: aruco
    dictionary: DICT_4X4_50
    id: "7"
    size: 0.03
    center: [0.0, 0.9, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
"""

_DOT_ONLY_BODY_YAML = """\
name: test-dot-prop
units: meters
markers:
  - name: end_a
    type: reflective_dot
    center: [0.0, 0.0, 0.0]
  - name: end_b
    type: reflective_dot
    center: [0.0, 1.2, 0.0]
"""

_MIXED_BODY_WITH_SYMMETRY_YAML = """\
name: test-jo
units: meters
symmetry_axis: [0.0, 1.0, 0.0]
markers:
  - name: band_top
    type: aruco
    dictionary: DICT_4X4_50
    id: "10"
    size: 0.04
    center: [0.0, 1.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
  - name: mid_dot
    type: reflective_dot
    center: [0.0, 0.5, 0.0]
"""

_EMPTY_BODY_YAML = "name: test-empty\nunits: meters\nmarkers: []\n"


def test_coded_only_body_produces_root_plus_two_markers_per_corner():
    config = load_marker_body_yaml(_TWO_MARKER_BODY_YAML)
    doc = generate_prop_skeleton(config)

    assert doc["name"] == "test-bokken"
    assert doc["units"] == "meters"
    assert "generated_from_marker_body" not in doc  # not given here

    assert len(doc["joints"]) == 1
    root = doc["joints"][0]
    assert root["name"] == "prop_root"
    assert root["type"] == "root"
    assert root["parent"] is None
    assert root["offset"] == [0.0, 0.0, 0.0]
    assert "locked_dofs" not in root  # no symmetry_axis in this body

    assert doc["input_tracks"] == [{"id": "prop_markers", "type": "labeled_points"}]

    # 2 coded markers * 4 corners = 8 markers, list-position-major order
    # per marker-mocap-design.md §4.1 -- hilt's corners before tip's.
    names = [m["name"] for m in doc["markers"]]
    assert names == [
        "hilt:c0", "hilt:c1", "hilt:c2", "hilt:c3",
        "tip:c0", "tip:c1", "tip:c2", "tip:c3",
    ]
    for m in doc["markers"]:
        assert m["parent"] == "prop_root"
        assert m["track"] == "prop_markers"
        assert m["landmark"] == m["name"]


def test_coded_marker_corner_offsets_match_config():
    config = load_marker_body_yaml(_TWO_MARKER_BODY_YAML)
    doc = generate_prop_skeleton(config)

    hilt_id = next(mid for mid, n in config.marker_names.items() if n == "hilt")
    expected_corners = config.marker_corners[hilt_id]
    hilt_markers = [m for m in doc["markers"] if m["name"].startswith("hilt:")]
    for i, m in enumerate(hilt_markers):
        assert np.allclose(m["offset"], expected_corners[i], atol=1e-6)


def test_dot_only_body_produces_one_marker_per_dot():
    config = load_marker_body_yaml(_DOT_ONLY_BODY_YAML)
    doc = generate_prop_skeleton(config)

    # A dot-only body has no coded markers at all -- no prop_markers track
    # emitted for something nothing references.
    assert doc["input_tracks"] == [{"id": "prop_dots", "type": "unlabeled_points"}]

    names = {m["name"] for m in doc["markers"]}
    assert names == {"end_a", "end_b"}
    for m in doc["markers"]:
        assert m["landmark"] == m["name"]
        # Dots get their own unlabeled_points track, never prop_markers
        # (dot-assignment-architecture-design.md §1) -- a dot has no
        # manifest slot to resolve against the way a coded marker's corner
        # does, so it cannot share that track.
        assert m["track"] == "prop_dots"
    end_b = next(m for m in doc["markers"] if m["name"] == "end_b")
    assert np.allclose(end_b["offset"], [0.0, 1.2, 0.0])


def test_mixed_body_with_symmetry_axis_locks_root_dof():
    config = load_marker_body_yaml(_MIXED_BODY_WITH_SYMMETRY_YAML)
    doc = generate_prop_skeleton(config)

    root = doc["joints"][0]
    assert root["locked_dofs"] == {"axis": [0.0, 1.0, 0.0]}

    names = {m["name"] for m in doc["markers"]}
    assert names == {"band_top:c0", "band_top:c1", "band_top:c2", "band_top:c3", "mid_dot"}


def test_mixed_body_routes_coded_and_dot_markers_to_separate_tracks():
    config = load_marker_body_yaml(_MIXED_BODY_WITH_SYMMETRY_YAML)
    doc = generate_prop_skeleton(config)

    assert doc["input_tracks"] == [
        {"id": "prop_markers", "type": "labeled_points"},
        {"id": "prop_dots", "type": "unlabeled_points"},
    ]
    by_name = {m["name"]: m for m in doc["markers"]}
    for corner in ("band_top:c0", "band_top:c1", "band_top:c2", "band_top:c3"):
        assert by_name[corner]["track"] == "prop_markers"
    assert by_name["mid_dot"]["track"] == "prop_dots"


def test_empty_body_raises():
    config = load_marker_body_yaml(_EMPTY_BODY_YAML)
    with pytest.raises(ValueError, match="no markers"):
        generate_prop_skeleton(config)


def test_marker_body_definition_id_recorded_when_given():
    config = load_marker_body_yaml(_TWO_MARKER_BODY_YAML)
    doc = generate_prop_skeleton(config, marker_body_definition_id="abc123")
    assert doc["generated_from_marker_body"] == "abc123"


def test_name_override():
    config = load_marker_body_yaml(_TWO_MARKER_BODY_YAML)
    doc = generate_prop_skeleton(config, name="custom-name")
    assert doc["name"] == "custom-name"


# ---------------------------------------------------------------------------
# YAML serialization + round-trip through the existing SkeletonLayout parser
# ---------------------------------------------------------------------------


def test_generated_yaml_round_trips_through_yaml_safe_load():
    config = load_marker_body_yaml(_TWO_MARKER_BODY_YAML)
    text = generate_prop_skeleton_yaml(config)
    parsed = yaml.safe_load(text)
    assert parsed["name"] == "test-bokken"
    assert len(parsed["markers"]) == 8
    assert parsed["joints"][0]["parent"] is None  # not the string "None"


def test_generated_yaml_loads_through_skeleton_layout():
    config = load_marker_body_yaml(_TWO_MARKER_BODY_YAML)
    text = generate_prop_skeleton_yaml(config)

    layout = SkeletonLayout(text)
    assert layout.root_joint_name() == "prop_root"
    assert layout.n_dof == 0  # root-only: no active joints beyond the free-flyer
    assert len(layout.markers) == 8
    assert all(m["parent"] == "prop_root" for m in layout.markers)


def test_generated_yaml_loads_through_skeleton_layout_dot_only():
    config = load_marker_body_yaml(_DOT_ONLY_BODY_YAML)
    text = generate_prop_skeleton_yaml(config)

    layout = SkeletonLayout(text)
    assert layout.n_dof == 0
    assert {m["name"] for m in layout.markers} == {"end_a", "end_b"}


def test_generated_yaml_symmetry_axis_survives_yaml_round_trip():
    config = load_marker_body_yaml(_MIXED_BODY_WITH_SYMMETRY_YAML)
    text = generate_prop_skeleton_yaml(config)
    parsed = yaml.safe_load(text)
    assert parsed["joints"][0]["locked_dofs"]["axis"] == [0.0, 1.0, 0.0]
    # SkeletonLayout doesn't know about locked_dofs yet (tracker-side
    # consumption is a later sub-phase) -- it must simply ignore the extra
    # key rather than error, same as any other unrecognized field.
    SkeletonLayout(text)
