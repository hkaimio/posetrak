# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""marker_body_to_skeleton.py — Generate a prop tracking skeleton from a
marker body definition (design phase 1b).

See docs/roadmap/features/marker-based-mocap/marker-mocap-design.md §5.3
("Prop skeletons are generated, not authored") and §7.1 sub-phase 1b.

A prop is a degenerate skeleton: one free-flyer root (named ``prop_root``,
matching the joint name the design doc's §5.4 "Later" splicing note
already anticipates), no other joints, markers only. This module is a
pure transform -- ``MarkerRigConfig`` (already-resolved marker-body
geometry, from ``app.setup.fiducial_markers.load_marker_body_yaml``) in,
skeleton YAML text out -- with no capture/detection/tracker-run
involvement, matching this sub-phase's deliberately narrow scope.

Deliberately not yet consumed by the C++ loader: ``input_tracks`` and a
marker's ``track``/``landmark`` fields are new syntax this design
introduces (design §5.1); wiring the C++ side to actually bind them is
sub-phase 1f. Until then, a skeleton this module generates is loadable by
every *existing* consumer (the C++ ``SkeletonLoader``, the Python
``SkeletonLayout``) exactly like a hand-authored one -- both simply do not
look at the extra keys, matching how a skeleton with no ``input_tracks``
already behaves "exactly as today" per §5.1.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import yaml

from app.setup.fiducial_markers import MarkerRigConfig

_ROOT_JOINT_NAME = "prop_root"
_INPUT_TRACK_ID = "prop_markers"


def _corner_landmark_name(marker_name: str, corner_index: int) -> str:
    return f"{marker_name}:c{corner_index}"


def generate_prop_skeleton(
    config: MarkerRigConfig,
    *,
    name: str | None = None,
    marker_body_definition_id: str | None = None,
) -> dict[str, Any]:
    """Build the skeleton document (as a plain dict, before YAML dumping)
    for the prop described by *config*.

    Parameters
    ----------
    config:
        A resolved marker body, e.g. from
        ``app.setup.fiducial_markers.load_marker_body_yaml``.
    name:
        Skeleton's own ``name:``. Defaults to ``config.rig_id``.
    marker_body_definition_id:
        If given, recorded as ``generated_from_marker_body:`` in the
        header (design §4.2) -- the provenance link back to the
        ``marker_body_definitions`` row this skeleton was compiled from.
        Omitted (not even as a null) when not given, since a skeleton
        generated directly from a YAML string with no known definition id
        (e.g. in a test) has nothing true to record here.

    Raises
    ------
    ValueError
        If *config* has no markers at all (coded or dot) -- a skeleton
        with zero markers can never be tracked, so this is almost always
        a mistake in the source marker body definition, not a valid
        degenerate case worth silently accepting.
    """
    if not config.marker_corners and not config.reflective_dots:
        raise ValueError(
            f"marker body {config.rig_id!r} has no markers (no coded markers, no "
            "reflective dots) -- cannot generate a trackable skeleton from it"
        )

    root_joint: dict[str, Any] = {
        "name": _ROOT_JOINT_NAME,
        "type": "root",
        "parent": None,
        "offset": [0.0, 0.0, 0.0],
    }
    if config.symmetry_axis is not None:
        # Generator-only annotation (design §5.3 option 1): the C++ side
        # does not consume this yet (that regularization mechanism is an
        # open decision, design doc §8 open question 3) -- recorded now so
        # the symmetry information survives from characterization through
        # to whenever the tracker-side mechanism lands, rather than being
        # silently dropped in the meantime.
        root_joint["locked_dofs"] = {"axis": [float(v) for v in config.symmetry_axis]}

    markers: list[dict[str, Any]] = []
    for marker_id, corners in config.marker_corners.items():
        marker_name = config.marker_names.get(marker_id, marker_id)
        for corner_index in range(4):
            landmark = _corner_landmark_name(marker_name, corner_index)
            markers.append({
                "name": landmark,
                "parent": _ROOT_JOINT_NAME,
                "offset": [float(v) for v in corners[corner_index]],
                "track": _INPUT_TRACK_ID,
                "landmark": landmark,
            })

    for dot_name, center in config.reflective_dots.items():
        markers.append({
            "name": dot_name,
            "parent": _ROOT_JOINT_NAME,
            "offset": [float(v) for v in np.asarray(center, dtype=np.float64)],
            "track": _INPUT_TRACK_ID,
            "landmark": dot_name,
        })

    doc: dict[str, Any] = {"name": name or config.rig_id, "units": "meters"}
    if marker_body_definition_id is not None:
        doc["generated_from_marker_body"] = marker_body_definition_id
    doc["joints"] = [root_joint]
    doc["input_tracks"] = [{"id": _INPUT_TRACK_ID, "type": "labeled_points"}]
    doc["markers"] = markers
    return doc


# ---------------------------------------------------------------------------
# YAML output -- same inline-numeric-list convention as
# posetrak.db.scale_skeleton._dump_yaml, replicated rather than imported
# since that helper is module-private there.
# ---------------------------------------------------------------------------

class _InlineDumper(yaml.SafeDumper):
    pass


def _represent_float(dumper: yaml.SafeDumper, value: float) -> yaml.ScalarNode:
    # Limit to 9 significant figures; avoid ugly repr like 1.0000000000000002
    return dumper.represent_float(float(f"{value:.9g}"))


def _represent_list(dumper: yaml.SafeDumper, data: list) -> yaml.SequenceNode:
    if data and all(isinstance(x, (int, float)) for x in data) and len(data) <= 6:
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)


_InlineDumper.add_representer(float, _represent_float)
_InlineDumper.add_representer(list, _represent_list)


def generate_prop_skeleton_yaml(
    config: MarkerRigConfig,
    *,
    name: str | None = None,
    marker_body_definition_id: str | None = None,
) -> str:
    """``generate_prop_skeleton`` + YAML serialization -- the function the
    CLI (``posetrak marker-body to-skeleton``) and the GUI's "add object to
    capture" flow (design §6.2 step 1) both call."""
    doc = generate_prop_skeleton(
        config, name=name, marker_body_definition_id=marker_body_definition_id
    )
    return yaml.dump(
        doc, Dumper=_InlineDumper, default_flow_style=False, allow_unicode=True, sort_keys=False,
    )
