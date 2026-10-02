# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""attachment.py — write calibrated placements as an attachment set.

An attachment set is a catalog module (see catalog.py) with the values fitted from a
capture: each marker's parent bone and its offset in the bone's frame, how much data
the fit rests on, and a ``calibration:`` block saying where it came from. The dot-augmented
skeleton is built from it (tools/build_dot_augmented_skeleton.py).
"""
from __future__ import annotations

import numpy as np
import yaml

from posetrak.markers.calibrate import Placement, bone_length
from posetrak.markers.bones import BodyPose

# How much data a placement needs to be trusted without a look.
HIGH_SAMPLES = 500
MEDIUM_SAMPLES = 100
_MIN_NORMAL_OFFSET_M = 0.015


def confidence(placement: Placement) -> str:
    """'high', 'medium' or 'low' from the samples behind the placement and how clearly it beat
    the next slot for the same markers."""
    if placement.runner_up_margin < 0.3 and placement.runner_up is not None:
        return "low" if placement.samples < HIGH_SAMPLES else "medium"
    if placement.samples >= HIGH_SAMPLES:
        return "high"
    return "medium" if placement.samples >= MEDIUM_SAMPLES else "low"


def attachment_document(
    placements: list[Placement], pose: BodyPose, *, module: str, topology: str | None,
    topology_hash: str | None, calibration: dict,
) -> str:
    """The attachment set as YAML text."""
    markers = []
    for p in sorted(placements, key=lambda p: p.slot):
        offset = np.asarray(p.offset, float)
        length = bone_length(pose, p.joint)
        across = np.array([offset[0], 0.0, offset[2]])         # a bone runs along its local y
        entry = {
            "name": p.slot,
            "parent_joint": p.joint,
            "along": round(float(offset[1] / length), 4),
            "lateral": round(float(offset[0]), 4),
            "anterior": round(float(offset[2]), 4),
            "offset": [round(float(v), 4) for v in offset],
        }
        if np.linalg.norm(across) >= _MIN_NORMAL_OFFSET_M:
            entry["normal"] = [round(float(v), 3) for v in across / np.linalg.norm(across)]
        entry.update({
            "n_samples": int(p.samples),
            "n_windows": int(p.chains),
            "spread_mm": round(1000 * float(p.spread_m), 1),
            "confidence": confidence(p),
            "runner_up": p.runner_up,
        })
        markers.append(entry)
    doc = {
        "module": module,
        "skeleton_topology": topology,
        "requires_topology_hash": topology_hash,
        "requires_joints": sorted({m["parent_joint"] for m in markers}),
        "calibration": calibration,
        "markers": markers,
    }
    return yaml.safe_dump({k: v for k, v in doc.items() if v is not None}, sort_keys=False)
