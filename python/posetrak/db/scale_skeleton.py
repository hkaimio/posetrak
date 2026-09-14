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

Neck + head (neck1, neck2, head):
    Not measured directly -- shoulder position moves too much with arm
    motion for a stable time-averaged anchor (Harri, 2026-09-14: "shoulders
    move a lot with arm movements so the time range average might have lot
    of variation"). Measured instead as ``hip_to_ear`` (hips to the
    ear-midpoint, a rigid-to-the-skull landmark independent of arm pose),
    with the neck+head segment's own length derived as
    ``hip_to_ear - torso_height`` -- i.e. whatever's left over once the
    already-measured spine segment is accounted for. Scale the Y-component
    of neck1/neck2/head's offsets by that derived ratio, the same
    single-ratio-across-a-multi-joint-chain approximation torso_height
    already uses for spine1/spine2/shoulder.L/R.

    Known imprecision, real but not a bug: both torso_height and the
    template-side "head" length are Euclidean (hip-to-shoulder,
    hip-to-head-joint) distances, not pure Y-sums, because spine/neck
    offsets have real (if small) X/Z components alongside their dominant
    Y one -- only Y actually gets scaled. torso_height's own target-vs-
    result gap from this is small in absolute terms (well under 1cm on a
    real skeleton) but "head" is a *difference* of two such Euclidean
    distances, so the same absolute X/Z leakage lands on a much shorter
    segment (~19cm vs ~45cm for torso) and shows up as a proportionally
    bigger relative miss -- confirmed on a real skeleton, ~15% in a
    deliberately-exaggerated synthetic test. Accepted rather than chasing
    exact precision here, consistent with this module's existing Y-index-
    only scaling philosophy elsewhere; revisit only if real-world scaled
    skeletons show a head segment visibly off from its own target.
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

    torso_height = float(np.linalg.norm(
        mid("shoulder.L", "shoulder.R") - mid("thigh.L", "thigh.R")
    ))
    # hip_to_ear proxy: the "head" joint itself, not the MRK-ear.L/R markers --
    # _fk_rest_pose only resolves joints, and the markers' own fixed offset
    # from "head" isn't part of what's being scaled here anyway (same
    # approximation body_measurements.py's own template-side "head" measurement
    # already uses). Falls back to 0.0 if "head" or either thigh joint is
    # missing (a bare prop skeleton, say), same guard dist() uses everywhere
    # else in this function.
    if "head" in jp and "thigh.L" in jp and "thigh.R" in jp:
        hip_to_head_joint = float(np.linalg.norm(jp["head"] - mid("thigh.L", "thigh.R")))
    else:
        hip_to_head_joint = 0.0

    return {
        "femur": (dist("thigh.L", "shin.L") + dist("thigh.R", "shin.R")) / 2,
        "shin": (dist("shin.L", "foot.L") + dist("shin.R", "foot.R")) / 2,
        "upper_arm": (dist("upper_arm.L", "forearm.L") + dist("upper_arm.R", "forearm.R")) / 2,
        "lower_arm": (dist("forearm.L", "hand.L") + dist("forearm.R", "hand.R")) / 2,
        "torso_height": torso_height,
        # shoulder_width = distance between glenohumeral joints (upper_arm.L/R),
        # matching the OpenPose shoulder keypoint positions used in body_measurements.py.
        # shoulder.L/R are the sternoclavicular joints (clavicle origins); upper_arm.L/R
        # are where the upper arm actually attaches.
        "shoulder_width": dist("upper_arm.L", "upper_arm.R"),
        # Derived, not an independent measurement: see the module docstring's
        # "Neck + head" section. hip_to_head_joint approximates hip_to_ear;
        # subtracting torso_height leaves just the neck+head chain's own length.
        "head": hip_to_head_joint - torso_height,
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

# Joints whose Y-component (index 1) is scaled for the neck+head segment
# (see the module docstring's "Neck + head" section -- driven by the
# derived hip_to_ear - torso_height length, not measured directly).
_NECK_HEAD_JOINTS = ("neck1", "neck2", "head")


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
        torso_height, shoulder_width, hip_to_ear.
        Missing keys are silently ignored (that dimension is left unscaled).
        hip_to_ear is raw (hip-to-ear-midpoint), not the neck+head segment
        length itself -- this function subtracts torso_height from it
        internally (see the module docstring's "Neck + head" section), so
        pass the actually-measured hip_to_ear distance here, not a
        pre-subtracted value.

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

    # --- Neck + head: derive the segment length from hip_to_ear - torso_height
    # (not measured directly -- see module docstring), then scale Y-component
    # of neck1/neck2/head the same way torso_height scales its own chain.
    if "hip_to_ear" in measurements and tmpl.get("head", 0.0) > 1e-9:
        measured_torso = measurements.get("torso_height", tmpl["torso_height"])
        measured_head = measurements["hip_to_ear"] - measured_torso
        head_r = measured_head / tmpl["head"]
        for jname in _NECK_HEAD_JOINTS:
            if jname not in by_name:
                continue
            j = by_name[jname]
            off = list(j.get("offset") or [0.0, 0.0, 0.0])
            j["offset"] = [off[0], off[1] * head_r, off[2]]

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
                        "torso_height", "shoulder_width", "head") if k in tmpl]
    lines = [
        f"{'Measurement':<18} {'Template':>10} {'Target':>10} {'Result':>10}",
        "-" * 52,
    ]
    for k in keys:
        t = tmpl[k] * 100
        if k == "head":
            # "head" is derived (hip_to_ear - torso_height), not a direct
            # measurements[] entry -- see scale_skeleton_yaml()'s own docstring.
            if "hip_to_ear" in measurements:
                measured_torso = measurements.get("torso_height", tmpl["torso_height"])
                tgt = (measurements["hip_to_ear"] - measured_torso) * 100
            else:
                tgt = float("nan")
        else:
            tgt = measurements.get(k, float("nan")) * 100
        res = scaled_m[k] * 100
        lines.append(f"{k:<18} {t:>9.1f}cm {tgt:>9.1f}cm {res:>9.1f}cm")
    return "\n".join(lines)
