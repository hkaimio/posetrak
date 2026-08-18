#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""
merge_skeleton.py — Combine a scaled skeleton's dimensions with a template skeleton's limits and markers.

Given skeleton A (carefully scaled for a performer) and skeleton B (updated template), produces a merged
skeleton where:

  - Joint structure (order, parent hierarchy) follows B.
  - Joint offsets, bone_tip_offsets, and orientations come from A (the scaled performer dimensions).
  - Joint type, limits, and axis come from B (the updated template).
  - Markers and groups come entirely from B.

Joints that exist in B but not in A are kept with B's dimensions (new template joints have no
performer-specific scaling yet).

Joints that exist in A but not in B are handled as follows:
  - Leaf joints (no children in A): silently omitted; a notice is printed.
  - Non-leaf joints (children in A that are also in B): structural conflict — printed as error, script exits.

If a joint exists in both A and B but its parent differs between them, that is also a structural conflict
and is reported as an error.

Usage
-----
    uv run python/tools/merge_skeleton.py \\
        --skeleton-a harri-scaled-kevin-2026-03-17.yaml \\
        --skeleton-b reallusion_skeleton_template.yaml \\
        --output merged-harri.yaml \\
        [--name "merged-harri"]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _joint_map(skel: dict) -> dict[str, dict]:
    return {j["name"]: j for j in skel.get("joints", [])}


def _children_map(joints: dict[str, dict]) -> dict[str, list[str]]:
    ch: dict[str, list[str]] = {n: [] for n in joints}
    for name, j in joints.items():
        p = j.get("parent")
        if p and p in ch:
            ch[p].append(name)
    return ch


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--skeleton-a", required=True, type=Path,
        help="Skeleton with performer-scaled dimensions (offsets, bone_tip_offsets, orientations).",
    )
    ap.add_argument(
        "--skeleton-b", required=True, type=Path,
        help="Template skeleton with updated limits, markers, and groups.",
    )
    ap.add_argument("--output", required=True, type=Path, help="Output merged skeleton YAML.")
    ap.add_argument(
        "--name", default=None,
        help="Name for the merged skeleton. Default: <A.name>-merged.",
    )
    args = ap.parse_args()

    skel_a = _load(args.skeleton_a)
    skel_b = _load(args.skeleton_b)

    joints_a = _joint_map(skel_a)
    joints_b = _joint_map(skel_b)
    children_a = _children_map(joints_a)

    errors: list[str] = []
    omitted: list[str] = []
    new_in_b: list[str] = []

    # Check parent consistency for joints in both skeletons
    for name in joints_b:
        if name not in joints_a:
            continue
        pa = joints_a[name].get("parent")
        pb = joints_b[name].get("parent")
        if pa != pb:
            errors.append(
                f"  {name!r}: parent in A is {pa!r}, in B is {pb!r}"
            )

    # Check for joints in A but not in B
    for name in joints_a:
        if name in joints_b:
            continue
        # Are any of A's children present in B? If so, this breaks the hierarchy.
        children_in_b = [c for c in children_a.get(name, []) if c in joints_b]
        if children_in_b:
            errors.append(
                f"  {name!r}: present in A (children: {children_a[name]}) but absent in B — "
                f"child joint(s) {children_in_b} are in B, which breaks the hierarchy"
            )
        else:
            omitted.append(name)

    if errors:
        print("ERROR: skeleton structural conflicts (fix before merging):", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        return 1

    if omitted:
        print(f"Omitting {len(omitted)} leaf joint(s) from A that are not in B:")
        for n in omitted:
            print(f"  {n}")

    # Joints in B that are not in A (new template joints — use B's dimensions)
    for name in joints_b:
        if name not in joints_a:
            new_in_b.append(name)
    if new_in_b:
        print(f"New joints in B (not in A) — using template dimensions for {len(new_in_b)}:")
        for n in new_in_b:
            print(f"  {n}")

    # Build merged joint list following B's order
    merged_joints: list[dict] = []
    for jb in skel_b.get("joints", []):
        name = jb["name"]
        ja = joints_a.get(name)

        j_out: dict = {"name": name}

        # Structure from B
        j_out["type"] = jb["type"]
        j_out["parent"] = jb.get("parent")

        if ja is not None:
            # Dimensions from A
            j_out["offset"] = ja["offset"]
            if "bone_tip_offset" in ja:
                j_out["bone_tip_offset"] = ja["bone_tip_offset"]
            elif "bone_tip_offset" in jb:
                j_out["bone_tip_offset"] = jb["bone_tip_offset"]
            if "orientation" in ja:
                j_out["orientation"] = ja["orientation"]
            elif "orientation" in jb:
                j_out["orientation"] = jb["orientation"]
        else:
            # New joint in B: use template dimensions
            j_out["offset"] = jb["offset"]
            if "bone_tip_offset" in jb:
                j_out["bone_tip_offset"] = jb["bone_tip_offset"]
            if "orientation" in jb:
                j_out["orientation"] = jb["orientation"]

        # Limits, axis, type details from B
        if "limits" in jb:
            j_out["limits"] = jb["limits"]
        if "axis" in jb:
            j_out["axis"] = jb["axis"]

        merged_joints.append(j_out)

    # Assemble output document
    out_name = args.name or f"{skel_a.get('name', 'A')}-merged"
    out: dict = {"name": out_name}
    units = skel_b.get("units") or skel_a.get("units")
    if units:
        out["units"] = units
    out["joints"] = merged_joints
    if "markers" in skel_b:
        out["markers"] = skel_b["markers"]
    if "groups" in skel_b:
        out["groups"] = skel_b["groups"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(out, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\nWritten: {args.output}")
    n_from_a = len(merged_joints) - len(new_in_b)
    print(f"  {len(merged_joints)} joints total  "
          f"({n_from_a} with A dimensions, {len(new_in_b)} from template only)")
    if "markers" in out:
        print(f"  {len(out['markers'])} markers  (from template)")
    if "groups" in out:
        print(f"  {len(out['groups'])} groups  (from template)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
