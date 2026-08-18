# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""scale_skeleton.py — Scale a skeleton YAML to match measured body dimensions.

Scaling rules
-------------
Limbs (femur, shin, upper_arm, lower_arm):
    Scale the full offset vector of the child joint by the ratio
    measured / template.  This preserves bone direction while adjusting length.

Torso height (spine1, spine2, shoulder.L/R):
    Scale the Y-component (index 1) of the offset — the principal spine
    direction in this skeleton's local frames — by torso_ratio.

Shoulder width (upper_arm.L/R):
    shoulder_width is measured between glenohumeral joints (upper_arm.L/R),
    matching the OpenPose shoulder keypoint.  shoulder.L/R are the
    sternoclavicular joints (clavicle origins); upper_arm.L/R are the
    far ends of the clavicles.  Scale the full offset vector of upper_arm.L/R
    by the ratio measured / template, preserving clavicle direction.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# FK at rest pose (all joint angles = 0)
# ---------------------------------------------------------------------------

def _fk_rest_pose(joints: list[dict]) -> dict[str, np.ndarray]:
    """Return {joint_name: world_position} at rest pose."""
    by_name: dict[str, dict] = {j["name"]: j for j in joints}
    ordered: list[str] = []
    visited: set[str] = set()

    def _visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        parent = by_name[name].get("parent")
        if parent:
            _visit(parent)
        ordered.append(name)

    for j in joints:
        _visit(j["name"])

    def _R_from_zyx(zyx: list[float] | None) -> np.ndarray:
        if not zyx:
            return np.eye(3)
        z, y, x = zyx[0], zyx[1], zyx[2]
        Rz = np.array([[math.cos(z), -math.sin(z), 0],
                       [math.sin(z),  math.cos(z), 0],
                       [0,            0,           1]])
        Ry = np.array([[ math.cos(y), 0, math.sin(y)],
                       [0,            1, 0           ],
                       [-math.sin(y), 0, math.cos(y)]])
        Rx = np.array([[1, 0,           0           ],
                       [0, math.cos(x), -math.sin(x)],
                       [0, math.sin(x),  math.cos(x)]])
        return Rx @ Ry @ Rz

    transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ordered:
        jnt = by_name[name]
        offset = np.array(jnt.get("offset") or [0.0, 0.0, 0.0], dtype=float)
        R = _R_from_zyx(jnt.get("orientation"))
        parent = jnt.get("parent")
        if parent is None:
            transforms[name] = (offset.copy(), R)
        else:
            p_pos, p_R = transforms[parent]
            transforms[name] = (p_pos + p_R @ offset, p_R @ R)

    return {name: pos for name, (pos, _) in transforms.items()}


# ---------------------------------------------------------------------------
# Template measurement computation
# ---------------------------------------------------------------------------

def template_measurements(joints: list[dict]) -> dict[str, float]:
    jp = _fk_rest_pose(joints)

    def dist(a: str, b: str) -> float:
        if a not in jp or b not in jp:
            return 0.0
        return float(np.linalg.norm(jp[a] - jp[b]))

    def mid(a: str, b: str) -> np.ndarray:
        return (jp[a] + jp[b]) / 2.0

    return {
        "femur": (dist("thigh.L", "shin.L") + dist("thigh.R", "shin.R")) / 2,
        "shin": (dist("shin.L", "foot.L") + dist("shin.R", "foot.R")) / 2,
        "upper_arm": (dist("upper_arm.L", "forearm.L") + dist("upper_arm.R", "forearm.R")) / 2,
        "lower_arm": (dist("forearm.L", "hand.L") + dist("forearm.R", "hand.R")) / 2,
        "torso_height": float(np.linalg.norm(
            mid("shoulder.L", "shoulder.R") - mid("thigh.L", "thigh.R")
        )),
        # shoulder_width = distance between glenohumeral joints (upper_arm.L/R),
        # matching the OpenPose shoulder keypoint positions used in body_measurements.py.
        # shoulder.L/R are the sternoclavicular joints (clavicle origins); upper_arm.L/R
        # are where the upper arm actually attaches.
        "shoulder_width": dist("upper_arm.L", "upper_arm.R"),
    }


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

# Maps child joint name → measurement key whose scale factor applies to its
# full offset vector (limb length scaling).
_LIMB_CHILD_TO_MEASURE: dict[str, str] = {
    "shin.L":      "femur",
    "shin.R":      "femur",
    "foot.L":      "shin",
    "foot.R":      "shin",
    "forearm.L":   "upper_arm",
    "forearm.R":   "upper_arm",
    "hand.L":      "lower_arm",
    "hand.R":      "lower_arm",
    # Scale the clavicle-end (glenohumeral) offset to achieve the target
    # shoulder_width.  shoulder.L/R are the sternoclavicular joints; upper_arm.L/R
    # are the glenohumeral joints at the far end of the clavicle.
    "upper_arm.L": "shoulder_width",
    "upper_arm.R": "shoulder_width",
}

# Joints whose Y-component (index 1) is scaled for torso height.
_TORSO_HEIGHT_JOINTS = ("spine1", "spine2", "shoulder.L", "shoulder.R")


def scale_skeleton_yaml(
    yaml_content: str,
    measurements: dict[str, float],
) -> str:
    """Return a new skeleton YAML with joint offsets scaled to match measurements.

    Parameters
    ----------
    yaml_content:
        Original skeleton YAML string.
    measurements:
        Dict mapping measurement key → value in metres.
        Recognised keys: femur, shin, upper_arm, lower_arm,
        torso_height, shoulder_width.
        Missing keys are silently ignored (that dimension is left unscaled).

    Returns
    -------
    str
        New YAML string with scaled offsets.
    """
    skel: dict[str, Any] = yaml.safe_load(yaml_content)
    joints: list[dict] = skel["joints"]
    by_name: dict[str, dict] = {j["name"]: j for j in joints}

    tmpl = template_measurements(joints)

    def _ratio(key: str) -> float | None:
        if key not in measurements or tmpl.get(key, 0.0) < 1e-9:
            return None
        return measurements[key] / tmpl[key]

    # --- Limbs: scale full offset vector ------------------------------------
    for jname, mkey in _LIMB_CHILD_TO_MEASURE.items():
        r = _ratio(mkey)
        if r is None or jname not in by_name:
            continue
        j = by_name[jname]
        off = list(j.get("offset") or [0.0, 0.0, 0.0])
        j["offset"] = [off[0] * r, off[1] * r, off[2] * r]

    # --- Torso height: scale Y-component (index 1) of spine/clavicle joints -
    torso_r = _ratio("torso_height")
    if torso_r is not None:
        for jname in _TORSO_HEIGHT_JOINTS:
            if jname not in by_name:
                continue
            j = by_name[jname]
            off = list(j.get("offset") or [0.0, 0.0, 0.0])
            j["offset"] = [off[0], off[1] * torso_r, off[2]]

    return _dump_yaml(skel)


# ---------------------------------------------------------------------------
# YAML output — preserve inline lists for offsets/orientations
# ---------------------------------------------------------------------------

class _InlineDumper(yaml.SafeDumper):
    pass


def _represent_float(dumper: yaml.SafeDumper, value: float) -> yaml.ScalarNode:
    # Limit to 9 significant figures; avoid ugly repr like 1.0000000000000002
    return dumper.represent_float(float(f"{value:.9g}"))


def _represent_list(dumper: yaml.SafeDumper, data: list) -> yaml.SequenceNode:
    # Keep short numeric lists inline (offsets, orientations, limits)
    if data and all(isinstance(x, (int, float)) for x in data) and len(data) <= 6:
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=True
        )
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=False
    )


_InlineDumper.add_representer(float, _represent_float)
_InlineDumper.add_representer(list, _represent_list)


def _dump_yaml(data: Any) -> str:
    return yaml.dump(
        data,
        Dumper=_InlineDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )


# ---------------------------------------------------------------------------
# Summary helper (for CLI output)
# ---------------------------------------------------------------------------

def scaling_summary(
    original_yaml: str,
    scaled_yaml: str,
    measurements: dict[str, float],
) -> str:
    """Return a human-readable table comparing template vs measured vs scaled."""
    orig_joints = yaml.safe_load(original_yaml)["joints"]
    scaled_joints = yaml.safe_load(scaled_yaml)["joints"]
    tmpl = template_measurements(orig_joints)
    scaled_m = template_measurements(scaled_joints)

    keys = [k for k in ("femur", "shin", "upper_arm", "lower_arm",
                        "torso_height", "shoulder_width") if k in tmpl]
    lines = [
        f"{'Measurement':<18} {'Template':>10} {'Target':>10} {'Result':>10}",
        "-" * 52,
    ]
    for k in keys:
        t = tmpl[k] * 100
        tgt = measurements.get(k, float("nan")) * 100
        res = scaled_m[k] * 100
        lines.append(f"{k:<18} {t:>9.1f}cm {tgt:>9.1f}cm {res:>9.1f}cm")
    return "\n".join(lines)
