# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for annotate_dots_manually.py's write_marker_body_yaml() -- the
one piece of this interactive tool that's a pure function (everything
else needs a real cv2 GUI window and real video, matching
calibrate_rigid_marker_body.py's own "standalone, not yet validated as a
CLI subcommand" status for the parts that need real footage).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "annotate_dots_manually.py"
)
_spec = importlib.util.spec_from_file_location("annotate_dots_manually", _MODULE_PATH)
annotate_dots_manually = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = annotate_dots_manually
_spec.loader.exec_module(annotate_dots_manually)

write_marker_body_yaml = annotate_dots_manually.write_marker_body_yaml


def test_write_marker_body_yaml_round_trips_aruco_and_dot_entries(tmp_path):
    markers = [
        {
            "name": "aruco_2", "type": "aruco", "dictionary": "DICT_4X4_50", "id": "2",
            "size": 0.095,
            "corners": [
                [-0.0475, 0.0475, 0.0], [0.0475, 0.0475, 0.0],
                [0.0475, -0.0475, 0.0], [-0.0475, -0.0475, 0.0],
            ],
        },
        {"name": "dot0", "type": "reflective_dot", "center": np.array([0.01, 0.02, 0.03])},
    ]
    out_path = tmp_path / "body.yaml"
    write_marker_body_yaml("test-body", markers, out_path)

    parsed = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert parsed["name"] == "test-body"
    assert parsed["units"] == "meters"
    assert len(parsed["markers"]) == 2

    aruco = parsed["markers"][0]
    assert aruco["name"] == "aruco_2"
    assert aruco["type"] == "aruco"
    assert aruco["id"] == "2"
    assert np.allclose(aruco["corners"][0], [-0.0475, 0.0475, 0.0])

    dot = parsed["markers"][1]
    assert dot["name"] == "dot0"
    assert dot["type"] == "reflective_dot"
    assert np.allclose(dot["center"], [0.01, 0.02, 0.03])


def test_write_marker_body_yaml_rejects_unknown_marker_type(tmp_path):
    markers = [{"name": "bad", "type": "not_a_real_type"}]
    try:
        write_marker_body_yaml("test-body", markers, tmp_path / "body.yaml")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "not_a_real_type" in str(e)
