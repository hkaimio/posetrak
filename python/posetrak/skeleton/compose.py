# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""compose.py — a skeleton made of layers: hierarchy, joint limits, marker sets and dimensions.

The tracker consumes one resolved skeleton YAML. It is composed from one layer of each kind:

* **hierarchy** — joints (type, parent, rest orientation) and the groups of joints that can be solved as
  stages. File: ``layers/hierarchy-body-hands.yaml``.
* **joint limits** — lower and upper limit per joint and axis, ``joint_limits.yaml``.
* **marker sets** — the markers of one pose model: the parent joint, the offset in that joint's frame and the
  keypoint the marker is bound to. Files: ``layers/marker-sets/coco133.yaml`` and ``hand21.yaml`` (bundled) and
  ``catalog/pose-models/soma77.marker-offsets.yaml`` (the catalog, with the offsets fitted to a reference run).
* **dimensions** — the offset and bone tip of every joint, i.e. the segment lengths. Files:
  ``layers/dimensions-male.yaml`` and ``-female.yaml``.

The bundled default skeletons are the composition of the bundled layers and the SOMA-77 set. The same marker
set files add a pose model to a skeleton of the user's own (:func:`add_pose_model`).

A marker set file::

    pose_model: soma77            # coco133 | hand21 | soma77
    group: main                   # optional: every marker of the set joins this group
    groups: {main: [names]}       # optional: explicit membership lists, in the order they appear in the group
    markers:
      - {name: Hips, parent: hips, offset: [x, y, z]}                      # soma77, coco133
      - {name: MRK-index.L, side: left, slot: 5, parent: ..., offset: ...}  # hand21: slot of the hand blob

A marker is bound to its keypoint by ``landmark`` (``soma77:Hips``, ``coco133:left_elbow``,
``hand21:left_index_1``). The COCO-133 and hand21 markers also keep their ``openpose_keypoint``, the COCO-133
index of the keypoint, which binds sequences that have no keypoint manifest.
"""
from __future__ import annotations

import copy
import importlib.resources
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from posetrak.db.scale_skeleton import _dump_yaml as dump_skeleton_yaml
from posetrak.detection.pose_models import COCO133, COCO133_HAND_BASE, HAND21, SOMA77

SEXES = ("male", "female")
_JOINT_KEYS = ("name", "type", "parent", "offset", "bone_tip_offset", "orientation", "limits")
_MARKER_KEYS = ("name", "parent", "offset", "openpose_keypoint", "openpose_name", "confidence_threshold", "landmark")
_GROUP_KEYS = ("name", "depends_on", "joints", "markers")


class SkeletonLayerError(ValueError):
    """A layer is malformed or does not fit the skeleton it is applied to."""


def _data_dir() -> Any:
    return importlib.resources.files("posetrak.data.skeletons")


def catalog_pose_models_dir() -> Path:
    """Directory of the pose-model catalog files: ``catalog/pose-models`` next to ``catalog/modules``."""
    from posetrak.markers.catalog import catalog_dir

    return catalog_dir().parent / "pose-models"


def _load(resource: Any) -> dict:
    return yaml.safe_load(resource.read_text(encoding="utf-8"))


def bundled_joint_limits() -> dict[str, dict[str, list[float]]]:
    """The joint-limit layer: {joint name: {axis: [lower, upper]}}."""
    return _load(_data_dir().joinpath("joint_limits.yaml"))["joints"]


def bundled_hierarchy() -> dict:
    return _load(_data_dir().joinpath("layers", "hierarchy-body-hands.yaml"))


def bundled_dimensions(sex: str) -> dict:
    return _load(_data_dir().joinpath("layers", f"dimensions-{sex}.yaml"))


def bundled_marker_set(pose_model: str) -> dict:
    """A marker set of the package (``coco133``, ``hand21``) or of the catalog (``soma77``)."""
    bundled = _data_dir().joinpath("layers", "marker-sets", f"{pose_model}.yaml")
    if bundled.is_file():
        return _load(bundled)
    path = catalog_pose_models_dir() / f"{pose_model}.marker-offsets.yaml"
    if not path.is_file():
        raise SkeletonLayerError(f"no marker set for pose model {pose_model!r} (looked for {path})")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _landmark(pose_model: str, marker: dict) -> str:
    if pose_model == "coco133":
        return COCO133.landmark(marker["openpose_keypoint"])
    if pose_model == "hand21":
        return HAND21.landmark(marker["slot"], marker["side"])
    if pose_model == "soma77":
        return SOMA77.landmark(SOMA77.keypoint_names.index(marker["name"]))
    raise SkeletonLayerError(f"unknown pose model {pose_model!r}")


def _marker_entries(marker_set: dict, *, landmarks: bool) -> list[dict]:
    """The skeleton marker entries of a marker set, keys in file order."""
    pose_model = marker_set["pose_model"]
    out = []
    for raw in marker_set["markers"]:
        entry = {k: copy.deepcopy(v) for k, v in raw.items() if k not in ("side", "slot")}
        if pose_model == "hand21":
            entry["openpose_keypoint"] = COCO133_HAND_BASE[raw["side"]] + raw["slot"]
        if landmarks:
            entry["landmark"] = _landmark(pose_model, {**raw, **entry})
        out.append({k: entry[k] for k in _MARKER_KEYS if k in entry})
    return out


def _set_groups(marker_set: dict) -> dict[str, list[str]]:
    groups = {g: list(names) for g, names in (marker_set.get("groups") or {}).items()}
    if marker_set.get("group"):
        groups.setdefault(marker_set["group"], []).extend(m["name"] for m in marker_set["markers"])
    return groups


def apply_marker_set(doc: dict, marker_set: dict, *, landmarks: bool = True) -> list[str]:
    """Add the markers of *marker_set* to the skeleton document *doc*; returns the markers skipped.

    A marker whose parent joint the skeleton lacks is skipped (a skeleton without fingers takes no finger
    markers). Raises if a marker of the same name exists, which means the set is already applied.
    """
    joints = {j["name"] for j in doc["joints"]}
    existing = {m["name"] for m in doc.setdefault("markers", [])}
    entries = _marker_entries(marker_set, landmarks=landmarks)
    clash = sorted(e["name"] for e in entries if e["name"] in existing and e["parent"] in joints)
    if clash:
        raise SkeletonLayerError(f"the skeleton already has marker(s) {', '.join(clash[:3])}: "
                                 f"the {marker_set['pose_model']} markers are applied already")
    skipped = [e["name"] for e in entries if e["parent"] not in joints]
    doc["markers"].extend(e for e in entries if e["parent"] in joints)
    placed = {e["name"] for e in entries if e["parent"] in joints}
    for group_name, names in _set_groups(marker_set).items():
        group = next((g for g in doc.get("groups", []) if g["name"] == group_name), None)
        if group is None:
            continue
        group["markers"] = [*group.get("markers", []), *(n for n in names if n in placed)]
    return skipped


def _ordered_group(group: dict) -> dict:
    """*group* with ``name``, ``depends_on``, ``joints`` and ``markers`` first, the other keys after them."""
    head = {k: group[k] for k in _GROUP_KEYS if k in group}
    return head | {k: v for k, v in group.items() if k not in head}


def compose_skeleton(
    hierarchy: dict,
    limits: dict[str, dict[str, list[float]]],
    marker_sets: Sequence[dict],
    dimensions: dict,
    *,
    landmarks: bool = True,
    layers: dict | None = None,
) -> dict:
    """The resolved skeleton document of one layer of each kind.

    Parameters
    ----------
    hierarchy:
        ``name``, ``units``, ``joints`` (name, type, parent, orientation) and ``groups`` without markers.
    limits:
        Per joint name, ``{axis: [lower, upper]}``. A joint absent from the table has no limits.
    marker_sets:
        Marker sets in the order their markers are listed.
    dimensions:
        ``joints: {name: {offset, bone_tip_offset}}``.
    landmarks:
        Whether the markers get their model-qualified ``landmark``. Without, a skeleton composed from the
        COCO-133 and hand21 sets carries ``openpose_keypoint`` bindings only, as skeletons did before landmarks.
    layers:
        Names of the layers, recorded under ``layers`` in the skeleton so that it says what it was composed from.
    """
    missing = [j["name"] for j in hierarchy["joints"] if j["name"] not in dimensions["joints"]]
    if missing:
        raise SkeletonLayerError(f"the dimensions have no entry for joint(s) {', '.join(missing[:5])}")
    unknown = sorted(set(limits) - {j["name"] for j in hierarchy["joints"]})
    if unknown:
        raise SkeletonLayerError(f"the joint limits name unknown joint(s) {', '.join(unknown[:5])}")
    joints = []
    for j in hierarchy["joints"]:
        entry = {**j, **dimensions["joints"][j["name"]]}
        if j["name"] in limits:
            entry["limits"] = copy.deepcopy(limits[j["name"]])
        joints.append({k: copy.deepcopy(entry[k]) for k in _JOINT_KEYS if k in entry})
    groups = []
    for g in hierarchy.get("groups", []):
        groups.append(copy.deepcopy(g))
    doc: dict = {"name": hierarchy["name"], "units": hierarchy["units"]}
    if layers:
        doc["layers"] = copy.deepcopy(layers)
    doc |= {"joints": joints, "markers": [], "groups": groups}
    for marker_set in marker_sets:
        apply_marker_set(doc, marker_set, landmarks=landmarks)
    doc["groups"] = [_ordered_group(g) for g in doc["groups"]]
    return doc


def compose_default(sex: str, pose_models: Iterable[str] = ("coco133", "hand21", "soma77")) -> str:
    """The YAML of the bundled default skeleton of *sex* ('male' or 'female') with the given marker sets."""
    if sex not in SEXES:
        raise SkeletonLayerError(f"sex must be one of {SEXES}, not {sex!r}")
    pose_models = list(pose_models)
    layers = {"hierarchy": "body-hands", "limits": "joint_limits", "marker_sets": pose_models, "dimensions": sex}
    doc = compose_skeleton(
        bundled_hierarchy(), bundled_joint_limits(), [bundled_marker_set(m) for m in pose_models],
        bundled_dimensions(sex), layers=layers)
    return dump_skeleton_yaml(doc)


def add_pose_model(skeleton_yaml: str, pose_model: str, marker_set: dict | None = None) -> tuple[str, list[str]]:
    """The YAML of a skeleton with the markers of *pose_model* added, and the markers it had no joint for.

    *marker_set* defaults to the model's bundled or catalog set. The skeleton's joints, limits and markers are
    kept; the markers join the groups the set names.
    """
    doc = yaml.safe_load(skeleton_yaml)
    skipped = apply_marker_set(doc, marker_set or bundled_marker_set(pose_model))
    if isinstance(doc.get("layers"), dict):
        doc["layers"].setdefault("marker_sets", []).append(pose_model)
    return dump_skeleton_yaml(doc), skipped


def write_default_skeletons(out_dir: Path | None = None) -> list[Path]:
    """Regenerate the bundled ``default_<sex>.yaml`` files from the layers (LF line ends, no header)."""
    out_dir = out_dir or Path(str(_data_dir()))
    paths = []
    for sex in SEXES:
        path = out_dir / f"default_{sex}.yaml"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(compose_default(sex))
        paths.append(path)
    return paths



if __name__ == "__main__":   # regenerate the bundled default skeletons after a change to a layer
    for written in write_default_skeletons():
        print(written)
