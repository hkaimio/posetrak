# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""topology.py — the structural identity of a skeleton.

A marker module or attachment set names joints of a skeleton (``shin.R``) and is
only meaningful on a skeleton built the same way. Two skeletons have the same
*topology* when they agree on the structure a marker can attach to: the joints,
their parents and types, and the direction each bone points. Bone lengths and
markers are left out, so a scaled skeleton and a skeleton that has had marker
slots added share their base's topology.

A skeleton YAML may declare a ``topology: {name, version}`` block. Without one
(every skeleton so far) the top-level ``name`` stands in for the topology name.
The structural hash is always computed, so a module can require one exactly. A
top-level ``name`` is unreliable as a topology name: a skeleton scaled to a person
is named after the person (real sessions hold such skeletons whose structure equals
the base rig's), so a name is only compared when a ``topology`` block declares it.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import yaml

_DIRECTION_DECIMALS = 3


@dataclass(frozen=True)
class Topology:
    """The identity of a skeleton's structure."""

    name: str
    version: int | None
    hash: str
    joints: frozenset[str]
    declared: bool           # the YAML has a ``topology`` block, so ``name`` is meant as one


def skeleton_topology(skeleton_yaml: str) -> Topology:
    """Read a skeleton's topology from its YAML.

    Parameters
    ----------
    skeleton_yaml:
        Skeleton YAML text (the ``skeletons.yaml_content`` of a session or registry).

    Returns
    -------
    Topology
        The declared ``topology`` name and version, else the top-level ``name`` and
        no version; whether it was declared; the structural hash; and the set of joint
        names.

    Raises
    ------
    ValueError
        If the YAML has no joints.
    """
    doc = yaml.safe_load(skeleton_yaml) or {}
    joints = doc.get("joints") or []
    if not joints:
        raise ValueError("skeleton has no joints")
    declared = doc.get("topology") or {}
    return Topology(
        name=str(declared.get("name") or doc.get("name") or ""),
        version=declared.get("version"),
        hash=_structural_hash(joints),
        joints=frozenset(str(j["name"]) for j in joints),
        declared=bool(declared.get("name")),
    )


def _structural_hash(joints: list[dict]) -> str:
    """Hash of joint names, parents, types and bone directions, independent of joint order."""
    structure = sorted(
        (str(j["name"]), j.get("parent"), str(j.get("type")), _direction(j.get("bone_tip_offset")))
        for j in joints
    )
    return hashlib.sha256(json.dumps(structure, separators=(",", ":")).encode("utf-8")).hexdigest()


def _direction(vector) -> list[float] | None:
    """Unit direction of a bone tip offset, so that scaling a bone does not change it."""
    if not vector:
        return None
    length = math.sqrt(sum(float(v) ** 2 for v in vector))
    if length == 0.0:
        return None
    return [round(float(v) / length + 0.0, _DIRECTION_DECIMALS) for v in vector]
