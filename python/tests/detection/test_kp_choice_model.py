# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""The choice model file the tracker reads: neighbour limits and keypoint classes of SOMA-77."""
from __future__ import annotations

from posetrak.detection.kp_choice_model import choice_model_csv, soma77_keypoint_class, write_choice_model
from posetrak.detection.pose_models import SOMA77
from posetrak.detection.soma77_neighbour_limits import SOMA77_NEIGHBOUR_LIMITS


def test_every_limit_is_a_skeleton_neighbour_pair_with_a_plausible_value() -> None:
    names = SOMA77.keypoint_names
    neighbours = {(names[c], names[p]) for c, p in enumerate(SOMA77.parents) if p >= 0}
    assert SOMA77_NEIGHBOUR_LIMITS, "the table is empty"
    assert set(SOMA77_NEIGHBOUR_LIMITS) <= neighbours
    assert all(0.0 < v < 1.0 for v in SOMA77_NEIGHBOUR_LIMITS.values())


def test_classes_split_hands_feet_and_body() -> None:
    assert soma77_keypoint_class("LeftHand") == "hands"
    assert soma77_keypoint_class("RightHandIndex1") == "hands"
    assert soma77_keypoint_class("LeftFoot") == "feet"
    assert soma77_keypoint_class("RightToeEnd") == "feet"
    assert soma77_keypoint_class("LeftForeArm") == "body"
    assert soma77_keypoint_class("HeadEnd") == "body"


def test_csv_has_a_line_per_pair_and_per_keypoint() -> None:
    text = choice_model_csv("soma77-v1")
    assert text is not None
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    pairs = [line.split(",") for line in lines if line.startswith("pair,")]
    classes = [line.split(",") for line in lines if line.startswith("class,")]
    assert len(pairs) == len(SOMA77_NEIGHBOUR_LIMITS)
    assert len(classes) == len(SOMA77.keypoint_names)
    assert ["pair", "soma77:Spine1", "soma77:Hips", str(SOMA77_NEIGHBOUR_LIMITS[("Spine1", "Hips")])] in pairs
    assert ["class", "soma77:LeftHand", "hands"] in classes


def test_other_models_have_no_choice_model(tmp_path) -> None:
    assert choice_model_csv("rtmpose-x") is None
    assert choice_model_csv(None) is None
    assert write_choice_model("rtmpose-x", tmp_path / "model.csv") is None
    assert not (tmp_path / "model.csv").exists()
    written = write_choice_model("soma77", tmp_path / "sub" / "model.csv")
    assert written is not None and written.read_text(encoding="utf-8").startswith("# choice model")
