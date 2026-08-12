"""Tests for load_marker_body_yaml / load_marker_body_yaml_file (design doc
section 10 -- "Marker body definitions: format and storage").
"""

from __future__ import annotations

import numpy as np
import pytest

from app.setup.fiducial_markers import (
    MarkerRigDetector,
    load_marker_body_yaml,
    load_marker_body_yaml_file,
    marker_local_corners,
)


# ---------------------------------------------------------------------------
# center/normal/up geometry resolution
# ---------------------------------------------------------------------------


def test_center_normal_up_matches_marker_local_corners_for_identity_pose():
    """A marker at the origin, facing +Z with +Y up, should resolve to
    exactly marker_local_corners()'s own convention (the pose is the
    identity transform of that local frame)."""
    yaml_content = """
name: test-body
markers:
  - name: front
    type: aruco
    dictionary: DICT_4X4_50
    id: "7"
    size: 0.1
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
"""
    config = load_marker_body_yaml(yaml_content)
    np.testing.assert_allclose(config.marker_corners["7"], marker_local_corners(0.1), atol=1e-12)


def test_center_offset_translates_corners():
    yaml_content = """
name: test-body
markers:
  - name: front
    type: aruco
    dictionary: DICT_4X4_50
    id: "7"
    size: 0.1
    center: [1.0, 2.0, 3.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
"""
    config = load_marker_body_yaml(yaml_content)
    expected = marker_local_corners(0.1) + np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(config.marker_corners["7"], expected, atol=1e-12)


def test_up_need_not_be_exactly_perpendicular_to_normal():
    """A slightly-off 'up' hint is Gram-Schmidt-corrected, not rejected."""
    yaml_content = """
name: test-body
markers:
  - name: front
    type: aruco
    dictionary: DICT_4X4_50
    id: "7"
    size: 0.1
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.05, 1.0, 0.0]
"""
    config = load_marker_body_yaml(yaml_content)
    corners = config.marker_corners["7"]
    # Still a valid, planar, right-sized square facing +Z.
    edges = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
    for e in edges:
        assert e == pytest.approx(0.1, abs=1e-9)
    normal_check = np.cross(corners[1] - corners[0], corners[3] - corners[0])
    normal_check /= np.linalg.norm(normal_check)
    np.testing.assert_allclose(np.abs(normal_check), [0, 0, 1], atol=1e-9)


def test_explicit_corners_used_as_is():
    raw_corners = [[0.1, 0.1, 0.0], [0.2, 0.1, 0.0], [0.2, 0.2, 0.0], [0.1, 0.2, 0.0]]
    yaml_content = f"""
name: test-body
markers:
  - name: solved
    type: aruco
    dictionary: DICT_4X4_50
    id: "3"
    corners: {raw_corners}
"""
    config = load_marker_body_yaml(yaml_content)
    np.testing.assert_allclose(config.marker_corners["3"], raw_corners)


# ---------------------------------------------------------------------------
# reflective dots
# ---------------------------------------------------------------------------


def test_reflective_dot_stored_separately_not_in_marker_corners():
    yaml_content = """
name: test-body
markers:
  - name: dot_a
    type: reflective_dot
    center: [0.04, 0.03, 0.0]
"""
    config = load_marker_body_yaml(yaml_content)
    assert config.marker_corners == {}
    np.testing.assert_allclose(config.reflective_dots["dot_a"], [0.04, 0.03, 0.0])


def test_reflective_dot_missing_center_raises():
    yaml_content = """
name: test-body
markers:
  - name: dot_a
    type: reflective_dot
"""
    with pytest.raises(ValueError, match="center"):
        load_marker_body_yaml(yaml_content)


def test_mixed_aruco_and_reflective_dot_body():
    yaml_content = """
name: test-body
markers:
  - name: front
    type: aruco
    dictionary: DICT_4X4_50
    id: "7"
    size: 0.1
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
  - name: dot_a
    type: reflective_dot
    center: [0.02, 0.02, 0.0]
"""
    config = load_marker_body_yaml(yaml_content)
    assert set(config.marker_corners) == {"7"}
    assert set(config.reflective_dots) == {"dot_a"}


# ---------------------------------------------------------------------------
# multi-dictionary bodies
# ---------------------------------------------------------------------------


def test_marker_dictionaries_populated_per_marker():
    yaml_content = """
name: test-body
markers:
  - name: a
    type: aruco
    dictionary: DICT_4X4_50
    id: "1"
    size: 0.1
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
  - name: b
    type: aruco
    dictionary: DICT_5X5_50
    id: "2"
    size: 0.1
    center: [1.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
"""
    config = load_marker_body_yaml(yaml_content)
    assert config.marker_dictionaries == {"1": "DICT_4X4_50", "2": "DICT_5X5_50"}


def test_marker_rig_detector_builds_one_detector_per_dictionary():
    yaml_content = """
name: test-body
markers:
  - name: a
    type: aruco
    dictionary: DICT_4X4_50
    id: "1"
    size: 0.1
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
  - name: b
    type: aruco
    dictionary: DICT_5X5_50
    id: "2"
    size: 0.1
    center: [1.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
"""
    config = load_marker_body_yaml(yaml_content)
    detector = MarkerRigDetector(config)
    assert set(detector._aruco_by_dict) == {"DICT_4X4_50", "DICT_5X5_50"}


# ---------------------------------------------------------------------------
# validation errors
# ---------------------------------------------------------------------------


def test_duplicate_name_raises():
    yaml_content = """
name: test-body
markers:
  - name: a
    type: aruco
    dictionary: DICT_4X4_50
    id: "1"
    size: 0.1
    center: [0, 0, 0]
    normal: [0, 0, 1]
    up: [0, 1, 0]
  - name: a
    type: aruco
    dictionary: DICT_4X4_50
    id: "2"
    size: 0.1
    center: [1, 0, 0]
    normal: [0, 0, 1]
    up: [0, 1, 0]
"""
    with pytest.raises(ValueError, match="duplicate marker name"):
        load_marker_body_yaml(yaml_content)


def test_duplicate_id_across_different_dictionaries_raises():
    """The bare-id-keyed MarkerRigConfig can't disambiguate the same id
    reused in two different dictionaries -- this must be caught at load
    time, not silently overwrite one marker's geometry with the other's."""
    yaml_content = """
name: test-body
markers:
  - name: a
    type: aruco
    dictionary: DICT_4X4_50
    id: "1"
    size: 0.1
    center: [0, 0, 0]
    normal: [0, 0, 1]
    up: [0, 1, 0]
  - name: b
    type: aruco
    dictionary: DICT_5X5_50
    id: "1"
    size: 0.1
    center: [1, 0, 0]
    normal: [0, 0, 1]
    up: [0, 1, 0]
"""
    with pytest.raises(ValueError, match="duplicate marker id"):
        load_marker_body_yaml(yaml_content)


def test_missing_name_raises():
    yaml_content = """
name: test-body
markers:
  - type: aruco
    dictionary: DICT_4X4_50
    id: "1"
    size: 0.1
    center: [0, 0, 0]
    normal: [0, 0, 1]
    up: [0, 1, 0]
"""
    with pytest.raises(ValueError, match="'name'"):
        load_marker_body_yaml(yaml_content)


def test_missing_type_raises():
    yaml_content = """
name: test-body
markers:
  - name: a
    dictionary: DICT_4X4_50
    id: "1"
"""
    with pytest.raises(ValueError, match="'type'"):
        load_marker_body_yaml(yaml_content)


def test_coded_marker_missing_dictionary_raises():
    yaml_content = """
name: test-body
markers:
  - name: a
    type: aruco
    id: "1"
    size: 0.1
    center: [0, 0, 0]
    normal: [0, 0, 1]
    up: [0, 1, 0]
"""
    with pytest.raises(ValueError, match="'dictionary'"):
        load_marker_body_yaml(yaml_content)


def test_coded_marker_missing_id_raises():
    yaml_content = """
name: test-body
markers:
  - name: a
    type: aruco
    dictionary: DICT_4X4_50
    size: 0.1
    center: [0, 0, 0]
    normal: [0, 0, 1]
    up: [0, 1, 0]
"""
    with pytest.raises(ValueError, match="'id'"):
        load_marker_body_yaml(yaml_content)


def test_coded_marker_missing_geometry_raises():
    yaml_content = """
name: test-body
markers:
  - name: a
    type: aruco
    dictionary: DICT_4X4_50
    id: "1"
"""
    with pytest.raises(ValueError, match="corners"):
        load_marker_body_yaml(yaml_content)


def test_slot_id_reference_raises_not_implemented():
    yaml_content = """
name: cube-template
slots: ["top"]
markers:
  - name: top
    type: aruco
    dictionary: DICT_4X4_50
    id:
      slot: top
    size: 0.1
    center: [0, 0, 0]
    normal: [0, 0, 1]
    up: [0, 1, 0]
"""
    with pytest.raises(NotImplementedError, match="slot"):
        load_marker_body_yaml(yaml_content)


def test_no_markers_key_loads_empty_config():
    config = load_marker_body_yaml("name: empty-body\n")
    assert config.marker_corners == {}
    assert config.reflective_dots == {}


# ---------------------------------------------------------------------------
# rig_id resolution
# ---------------------------------------------------------------------------


def test_rig_id_defaults_to_yaml_name():
    config = load_marker_body_yaml("name: my-rig\nmarkers: []\n")
    assert config.rig_id == "my-rig"


def test_rig_id_explicit_override_wins():
    config = load_marker_body_yaml("name: my-rig\nmarkers: []\n", rig_id="override")
    assert config.rig_id == "override"


def test_rig_id_falls_back_when_no_yaml_name():
    config = load_marker_body_yaml("markers: []\n")
    assert config.rig_id == "marker_body"


# ---------------------------------------------------------------------------
# load_marker_body_yaml_file
# ---------------------------------------------------------------------------


def test_load_marker_body_yaml_file_reads_content(tmp_path):
    content = "name: box\nmarkers: []\n"
    path = tmp_path / "box_rig.yaml"
    path.write_text(content, encoding="utf-8")
    config = load_marker_body_yaml_file(str(path))
    assert config.rig_id == "box"  # YAML's own name: wins over the filename


def test_load_marker_body_yaml_file_falls_back_to_filename_stem(tmp_path):
    path = tmp_path / "unnamed_rig.yaml"
    path.write_text("markers: []\n", encoding="utf-8")
    config = load_marker_body_yaml_file(str(path))
    assert config.rig_id == "unnamed_rig"
