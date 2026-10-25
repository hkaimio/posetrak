# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""naming.py — tell apart markers that sit at the same place along one bone.

The calibration places a marker on a bone by where it sits along it, which cannot
separate the lateral and medial knee markers or the two ankle markers. Their names say
which is which, by direction: lateral or medial, in front, left or right of the body.
This module reads that direction from a marker's name, measures a marker's direction in
the body at a moment when its bone is as upright as it gets in the trial (so a hanging
arm or a standing leg), and prices a disagreement, which the slot assignment adds to its
costs.
"""
from __future__ import annotations

import numpy as np
from posetrak.markers.bones import BodyPose
from posetrak.markers.catalog import Slot

def role_direction(slot: Slot) -> tuple[float, float] | None:
    """The direction the marker's name asks for, as (towards the body's right, towards its front),
    or None if the name gives none: wrist_rad/wrist_uln and index/pinky depend on how the hand is
    held, and are left to review.

    Lateral and medial are relative to the limb's own side (lateral is away from the midline);
    the pinky toe is lateral and the big toe medial. R markers are on the subject's right.
    """
    base = slot.name[:-2] if slot.name.endswith(("_R", "_L")) else slot.name
    body_side = -1.0 if slot.name.endswith("_L") else 1.0       # for markers named by body side
    limb_side = -1.0 if slot.parent_joint.endswith(".L") else 1.0
    if base.endswith("_lat") or base == "toe_pinky":
        return (limb_side, 0.0)
    if base.endswith("_med") or base == "toe":
        return (-limb_side, 0.0)
    if base.endswith("_front") or base == "forehead":
        return (0.0, 1.0)
    if base == "sacrum":
        return (0.0, -1.0)
    if base == "asis":
        return (body_side, 0.5)
    if base == "chest":
        return (body_side, 0.8)
    if base == "temple":
        return (body_side, 0.0)
    return None


def body_axes(frames: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, front, up) of the body from the joint frames of one instant."""
    up = np.array([0.0, 0.0, 1.0])
    right = frames["shoulder.R"][:3, 3] - frames["shoulder.L"][:3, 3]
    right = right - (right @ up) * up
    right /= np.linalg.norm(right)
    return right, np.cross(up, right), up


def upright_instant(pose: BodyPose, joint: str, step: int = 12) -> dict[str, np.ndarray] | None:
    """Joint frames at the sampled instant where the bone is closest to vertical."""
    best, best_frames = -1.0, None
    for t in pose.times[::step]:
        frames = pose.joint_frames(float(t))
        if frames is None or joint not in frames:
            continue
        axis = frames[joint][:3, 1]            # a bone runs along its local y
        if abs(axis[2]) > best:
            best, best_frames = abs(axis[2]), frames
    return best_frames


def direction_of(pose: BodyPose, joint: str, offset: np.ndarray) -> tuple[float, float] | None:
    """(right, front) direction of a marker from its bone's axis, at the bone's most upright instant."""
    frames = upright_instant(pose, joint)
    if frames is None:
        return None
    R = frames[joint][:3, :3]
    axis = R[:, 1]
    world = R @ offset
    across = world - (world @ axis) * axis
    right, front, _ = body_axes(frames)
    return float(across @ right), float(across @ front)


def direction_penalty(pose: BodyPose, slot: Slot, joint: str, offset: np.ndarray, weight: float = 3.0) -> float:
    """Extra cost for a marker whose measured direction disagrees with its name.

    0 when the name gives no direction; otherwise ``weight`` times (1 - cosine) between the
    direction the name asks for and the marker's measured direction from its bone's axis, at
    the bone's most upright instant. A marker that hugs the axis (under 15 mm off it) has no
    direction to speak of and is charged half the weight.
    """
    role = role_direction(slot)
    if role is None:
        return 0.0
    measured = direction_of(pose, joint, offset)
    if measured is None or np.hypot(*measured) < 0.015:
        return 0.5 * weight
    cosine = float(np.dot(measured, role) / (np.hypot(*measured) * np.hypot(*role)))
    return weight * (1.0 - cosine)
