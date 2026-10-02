# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""reference.py — marker placement on a reference body, exchanged with Blender.

Markers are placed on a reference model in Blender (tools/blender/blender_marker_reference.py),
where a marker snaps to the body surface and takes the surface normal. What is exchanged is
the placement in a frame that depends only on the skeleton's rest pose and not on how a
particular armature orients its bones, so it carries over to any skeleton with the same
topology (a scaled copy, another person):

- ``along``: where on the bone the marker is, as a fraction of the bone's length from its
  joint (0) to its far end (1).
- ``lateral``, ``anterior``: how far it is off the bone, in metres, along the two directions
  below.
- ``normal``: the surface normal at the marker, as (long, lateral, anterior) components.

The frame of a bone is made from its direction ``d`` (joint to far end) and the body's axes
(right, front, up): *anterior* is the front direction with its component along the bone removed
(for a bone that points forward, such as the foot, the up direction instead: dorsal), and
*lateral* is ``d x anterior``. Both ends of an exchange compute it the same way, so the
frame needs no bone roll. Mirroring a marker across the body negates ``lateral`` (and the lateral
part of the normal), which is what catalog.py does for ``mirror: true``.
"""
from __future__ import annotations

import re

import numpy as np
import yaml

_FORWARD_BONE_LIMIT = 0.3   # if the front direction is this close to the bone's, use up instead


def body_axes(
    right_shoulder: np.ndarray, left_shoulder: np.ndarray, hips_to_head: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, front, up) unit vectors from two shoulder positions and the direction from the
    hips to the head; up is taken as the component of that direction ⟂ the shoulder line."""
    right = np.asarray(right_shoulder, float) - np.asarray(left_shoulder, float)
    right = right / np.linalg.norm(right)
    up = np.asarray(hips_to_head, float)
    up = up - (up @ right) * right
    up = up / np.linalg.norm(up)
    return right, np.cross(up, right), up


def bone_frame(head: np.ndarray, tail: np.ndarray, front: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Columns (long, lateral, anterior) of the bone's frame, as a 3x3 matrix."""
    d = np.asarray(tail, float) - np.asarray(head, float)
    d = d / np.linalg.norm(d)
    anterior = front - (front @ d) * d
    if np.linalg.norm(anterior) < _FORWARD_BONE_LIMIT:
        anterior = up - (up @ d) * d
    anterior = anterior / np.linalg.norm(anterior)
    lateral = np.cross(d, anterior)
    return np.column_stack([d, lateral, anterior])


def to_reference(
    head: np.ndarray, tail: np.ndarray, front: np.ndarray, up: np.ndarray,
    position: np.ndarray, normal: np.ndarray | None = None,
) -> dict:
    """Placement of a world-space marker position (and surface normal) on the bone head -> tail."""
    frame = bone_frame(head, tail, front, up)
    length = float(np.linalg.norm(np.asarray(tail, float) - np.asarray(head, float)))
    local = frame.T @ (np.asarray(position, float) - np.asarray(head, float))
    out = {"along": float(local[0] / length), "lateral": float(local[1]), "anterior": float(local[2])}
    if normal is not None:
        n = frame.T @ np.asarray(normal, float)
        out["normal"] = [float(v) for v in n / np.linalg.norm(n)]
    return out


def from_reference(
    head: np.ndarray, tail: np.ndarray, front: np.ndarray, up: np.ndarray, placement: dict,
    distance_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """World position (and normal) of a placement on the bone head -> tail. *distance_scale*
    scales the lateral and anterior distances, for a body larger or smaller than the reference."""
    frame = bone_frame(head, tail, front, up)
    length = float(np.linalg.norm(np.asarray(tail, float) - np.asarray(head, float)))
    local = np.array([placement["along"] * length,
                      placement["lateral"] * distance_scale,
                      placement["anterior"] * distance_scale])
    position = np.asarray(head, float) + frame @ local
    normal = frame @ np.asarray(placement["normal"], float) if placement.get("normal") is not None else None
    return position, normal


def rest_bones(layout, decoded_state: dict) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    """The skeleton in its rest pose (all joint angles zero), turned to stand upright facing
    +y with its right side towards +x, its pelvis on the x = y = 0 axis and its lowest point on the floor: head and
    far-end positions of every bone, and every joint's 4x4 frame.

    *decoded_state* is any decoded state of the skeleton (``layout.decode_state_blob``); only
    its root pose is used, and it is undone, so the result and the placements made on it do
    not depend on it. In these coordinates the body axes are (1,0,0), (0,1,0) and (0,0,1).
    """
    import copy

    rest = copy.deepcopy(decoded_state)
    for name, angles in rest["joint_angles"].items():
        rest["joint_angles"][name] = np.zeros_like(angles)
    frames = layout.compute_joint_transforms(rest)
    tips = {j.name: j.bone_tip_offset for j in layout.joints if j.bone_tip_offset is not None}
    first_child = {}
    for j in layout.joints:
        first_child.setdefault(j.parent, j.name)
    raw = {}
    for name, frame in frames.items():
        head = frame[:3, 3]
        if name in tips:
            raw[name] = (head, (frame @ np.append(tips[name], 1.0))[:3])
        elif name in first_child:                      # the pelvis: its bone runs to its first child
            raw[name] = (head, frames[first_child[name]][:3, 3])
    right, front, up = body_axes(frames["shoulder.R"][:3, 3], frames["shoulder.L"][:3, 3],
                                 frames["head"][:3, 3] - frames["hips"][:3, 3])
    turn = np.array([right, front, up])                 # rows: world -> body coordinates
    lowest = min(float((turn @ p)[2]) for ends in raw.values() for p in ends)
    centre = turn @ frames["hips"][:3, 3]
    shift = np.array([-centre[0], -centre[1], -lowest])
    to_body = np.eye(4)
    to_body[:3, :3] = turn
    to_body[:3, 3] = shift
    bones = {n: (turn @ h + shift, turn @ t + shift) for n, (h, t) in raw.items()}
    return bones, {n: to_body @ f for n, f in frames.items()}


def reference_document(
    bones: dict[str, tuple[np.ndarray, np.ndarray]], parents: dict[str, str | None],
    markers: list[dict],
) -> dict:
    """The document the Blender scene is built from: the skeleton's rest-pose bones (so a scene
    can be made without an armature) and the markers' placements."""
    return {
        "bones": {n: {"head": [float(v) for v in h], "tail": [float(v) for v in t], "parent": parents.get(n)}
                  for n, (h, t) in bones.items()},
        "markers": markers,
    }


def placements_from_attachment_set(attachment: dict, bones: dict, rest_frames: dict) -> list[dict]:
    """Starting placements for the reference scene, from a calibrated attachment set: each
    marker's offset in its joint's frame, taken to the rest pose and expressed on its bone."""
    front, up = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    out = []
    for m in attachment["markers"]:
        joint = m["parent_joint"]
        frame = rest_frames[joint]
        position = (frame @ np.append(np.asarray(m["offset"], float), 1.0))[:3]
        normal = frame[:3, :3] @ np.asarray(m["normal"], float) if m.get("normal") else None
        head, tail = bones[joint]
        placement = to_reference(head, tail, front, up, position, normal)
        out.append({"name": m["name"], "parent_joint": joint, **{k: round(v, 5) if isinstance(v, float) else
                    [round(x, 4) for x in v] for k, v in placement.items()},
                    "confidence": m.get("confidence")})
    return out


_KEY_ORDER = ("name", "parent_joint", "along", "lateral", "anterior", "normal", "reach", "mirror")
_MARKER_LINE = re.compile(r"^(?P<indent>\s*)- \{(?P<body>.*)\}(?P<tail>\s*(#.*)?)$")


def apply_placements(module_text: str, placements: dict[str, dict]) -> tuple[str, list[str]]:
    """Write placements into a catalog module's text, line by line so its comments stay.

    *placements* maps a marker name to ``along``, ``lateral``, ``anterior`` and optionally
    ``normal``. A marker written once for both sides (``mirror: true``) takes the placement of
    its right-hand name. Markers without a placement are left as they are. Returns the new text
    and the names of the markers that were changed.
    """
    out, changed = [], []
    for line in module_text.split("\n"):
        m = _MARKER_LINE.match(line)
        if m is None:
            out.append(line)
            continue
        entry = yaml.safe_load("{" + m["body"] + "}")
        placement = placements.get(entry.get("name"))
        if placement is None:
            out.append(line)
            continue
        entry.update({k: round(float(placement[k]), 4) for k in ("along", "lateral", "anterior")})
        if placement.get("normal"):
            entry["normal"] = [round(float(v), 3) for v in placement["normal"]]
        body = ", ".join(
            yaml.safe_dump({k: entry[k]}, default_flow_style=True).strip()[1:-1]
            for k in _KEY_ORDER if k in entry
        )
        out.append(f"{m['indent']}- {{{body}}}{m['tail']}")
        changed.append(entry["name"])
    return "\n".join(out), changed
