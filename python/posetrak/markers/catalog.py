# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""catalog.py — marker modules and attachment sets: which markers go where on a skeleton.

A *module* (``catalog/modules/<name>.marker-module.yaml``) names the markers a
group of body parts carries and the joint each rides on. An *attachment set* is
the same document with values fitted from a capture and a ``calibration:`` block;
:func:`load_marker_module` reads both.

File shape::

    module: leg
    requires_topology: reallusion-no-waist    # optional; a topology name, compared when the
                                              # skeleton declares one (see topology.py)
    requires_topology_hash: <sha256>          # optional; the exact structure
    requires_joints: [thigh, shin, foot]      # optional; joint names without the side
    markers:
      - name: knee_lat_R          # one instance ...
        parent_joint: shin.R
        mirror: true              # ... and its left twin knee_lat_L on shin.L
        along: 0.06               # optional geometry, in the marker's own local frame:
        lateral: 0.045            #   fraction of the bone, metres, metres
        anterior: 0.005
        normal: [0.0, 1.0, 0.1]   # (long, lateral, anterior)

A marker with ``mirror: true`` is written for the right side: its name ends in
``_R`` and its joint in ``.R``. The left twin swaps both suffixes and reflects
across the sagittal plane, which negates ``lateral`` and the lateral component
of ``normal``. Every other marker is taken as written.

The geometry fields are optional, but ``along``, ``lateral`` and ``anterior``
come together. A module authored without nominal geometry still fixes the slot
names, their order and their parent joints.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from posetrak.markers.topology import skeleton_topology

_MODULE_SUFFIX = ".marker-module.yaml"
_GEOMETRY = ("along", "lateral", "anterior")


class MarkerSetError(ValueError):
    """A module or attachment set is malformed, or does not fit a skeleton."""


@dataclass(frozen=True)
class Slot:
    """One marker of a module."""

    name: str
    parent_joint: str
    along: float | None = None
    lateral: float | None = None
    anterior: float | None = None
    normal: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class MarkerModule:
    """A module or attachment set, with mirrored markers expanded."""

    name: str
    slots: tuple[Slot, ...]
    requires_topology: str | None = None
    requires_topology_hash: str | None = None
    requires_joints: tuple[str, ...] = ()

    def slot_names(self) -> list[str]:
        """The marker names in file order, each right marker followed by its left twin."""
        return [s.name for s in self.slots]

    def parent_joint(self, slot_name: str) -> str:
        """The joint a marker rides on.

        Raises
        ------
        KeyError
            If the module has no such marker.
        """
        for slot in self.slots:
            if slot.name == slot_name:
                return slot.parent_joint
        raise KeyError(f"module {self.name!r} has no marker {slot_name!r}")


def catalog_dir() -> Path:
    """Directory of the catalog's module files.

    ``$POSETRAK_CATALOG_DIR`` when set, else ``catalog/modules`` in the repository.
    """
    override = os.environ.get("POSETRAK_CATALOG_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "catalog" / "modules"


def catalog_module(name: str) -> MarkerModule:
    """Load a module of the catalog by name (``leg`` for ``leg.marker-module.yaml``)."""
    path = catalog_dir() / f"{name}{_MODULE_SUFFIX}"
    if not path.is_file():
        available = sorted(p.name.removesuffix(_MODULE_SUFFIX) for p in catalog_dir().glob(f"*{_MODULE_SUFFIX}"))
        raise MarkerSetError(f"no catalog module {name!r} in {catalog_dir()} (modules: {', '.join(available) or 'none'})")
    return load_marker_module(path.read_text(encoding="utf-8"), source=str(path))


def load_marker_module(text: str | dict, *, source: str = "marker module") -> MarkerModule:
    """Parse a module or attachment set.

    Parameters
    ----------
    text:
        The YAML text, or the already parsed document.
    source:
        What to call the document in error messages.

    Returns
    -------
    MarkerModule
        With mirrored markers expanded.

    Raises
    ------
    MarkerSetError
        If the document is malformed: no ``module`` name or no markers, a marker without a
        name or joint, a duplicate name, a partial geometry, or a mirrored marker not written
        for the right side.
    """
    doc = yaml.safe_load(text) if isinstance(text, str) else text
    if not isinstance(doc, dict) or not doc.get("module"):
        raise MarkerSetError(f"{source}: no 'module' name")
    entries = doc.get("markers") or []
    if not entries:
        raise MarkerSetError(f"{source}: no markers")

    slots: list[Slot] = []
    for entry in entries:
        slot = _slot(entry, source)
        slots.append(slot)
        if entry.get("mirror"):
            slots.append(_mirrored(slot, source))
    names = [s.name for s in slots]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise MarkerSetError(f"{source}: duplicate marker names: {', '.join(duplicates)}")

    return MarkerModule(
        name=str(doc["module"]),
        slots=tuple(slots),
        # An attachment set written before topology was named calls it `skeleton_topology`.
        requires_topology=doc.get("requires_topology") or doc.get("skeleton_topology"),
        requires_topology_hash=doc.get("requires_topology_hash"),
        requires_joints=tuple(doc.get("requires_joints") or ()),
    )


def check_module_fits_skeleton(module: MarkerModule | dict | str, skeleton_yaml: str) -> None:
    """Refuse a module or attachment set on a skeleton it was not written for.

    Parameters
    ----------
    module:
        A loaded module, or the YAML text or parsed document of one.
    skeleton_yaml:
        The skeleton the markers would be attached to.

    Raises
    ------
    MarkerSetError
        Naming every mismatch: a different declared topology name, a different structural
        hash, or parent joints the skeleton does not have.
    """
    if not isinstance(module, MarkerModule):
        module = load_marker_module(module)
    topology = skeleton_topology(skeleton_yaml)
    problems = []
    if module.requires_topology and topology.declared and module.requires_topology != topology.name:
        problems.append(f"it needs topology {module.requires_topology!r} but the skeleton is {topology.name!r}")
    if module.requires_topology_hash and module.requires_topology_hash != topology.hash:
        problems.append(
            f"its skeleton structure ({module.requires_topology_hash[:12]}...) differs from this "
            f"skeleton's ({topology.hash[:12]}...)"
        )
    missing = sorted({s.parent_joint for s in module.slots} - topology.joints)
    if missing:
        problems.append(f"the skeleton has no joint {', '.join(missing)}")
    missing_required = [
        j for j in module.requires_joints
        if not topology.joints & {j, f"{j}.R", f"{j}.L"}
    ]
    if missing_required:
        problems.append(f"the skeleton has no joint named {', '.join(missing_required)}")
    if problems:
        raise MarkerSetError(f"marker module {module.name!r} does not fit this skeleton: " + "; ".join(problems))


def _slot(entry: dict, source: str) -> Slot:
    name, joint = entry.get("name"), entry.get("parent_joint")
    if not name or not joint:
        raise MarkerSetError(f"{source}: every marker needs a name and a parent_joint (got {entry})")
    given = [g for g in _GEOMETRY if entry.get(g) is not None]
    if given and len(given) != len(_GEOMETRY):
        raise MarkerSetError(f"{source}: marker {name!r} gives {', '.join(given)} but needs all of {', '.join(_GEOMETRY)}")
    normal = entry.get("normal")
    return Slot(
        name=str(name), parent_joint=str(joint),
        along=_float(entry.get("along")), lateral=_float(entry.get("lateral")), anterior=_float(entry.get("anterior")),
        normal=tuple(float(v) for v in normal) if normal else None,
    )


def _mirrored(slot: Slot, source: str) -> Slot:
    if not slot.name.endswith("_R") or not slot.parent_joint.endswith(".R"):
        raise MarkerSetError(
            f"{source}: mirrored marker {slot.name!r} must be written for the right side "
            f"(name ending _R, joint ending .R; joint is {slot.parent_joint!r})"
        )
    return Slot(
        name=slot.name[:-2] + "_L",
        parent_joint=slot.parent_joint[:-2] + ".L",
        along=slot.along,
        lateral=None if slot.lateral is None else -slot.lateral,
        anterior=slot.anterior,
        normal=None if slot.normal is None else (slot.normal[0], -slot.normal[1], slot.normal[2]),
    )


def _float(value) -> float | None:
    return None if value is None else float(value)
