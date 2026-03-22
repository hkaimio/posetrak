#!/usr/bin/env python3
"""
print_skeleton_joints.py — Print global 3D joint positions of a skeleton at rest pose.

Useful for matching a 3D character rig to tracked person dimensions.

The skeleton is stored in Y-up convention.  Use --zup to convert to Z-up
(swaps Y↔Z axes) and --floor-z to set the Z coordinate of the lowest joint,
placing the skeleton at the desired height above the ground plane.

Usage:
    python3 scripts/print_skeleton_joints.py <skeleton.yaml> [options]

    python3 scripts/print_skeleton_joints.py tracking_tests/harri-scaled-skeleton-ri.yaml
    python3 scripts/print_skeleton_joints.py tracking_tests/harri-scaled-skeleton-ri.yaml --units cm --zup
    python3 scripts/print_skeleton_joints.py tracking_tests/harri-scaled-skeleton-ri.yaml --units cm --zup --floor-z 2.5
    python3 scripts/print_skeleton_joints.py tracking_tests/harri-scaled-skeleton-ri.yaml --group main --zup
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def orientation_to_matrix(zyx: Optional[list]) -> np.ndarray:
    """Convert skeleton orientation [z, y, x] Euler angles to rotation matrix.

    Convention from docs: R = Rx(x) * Ry(y) * Rz(z)
    """
    if not zyx:
        return np.eye(3)
    z_angle, y_angle, x_angle = zyx[0], zyx[1], zyx[2]
    return rx(x_angle) @ ry(y_angle) @ rz(z_angle)


# ---------------------------------------------------------------------------
# FK computation
# ---------------------------------------------------------------------------

def compute_joint_world_transforms(
    joints: list[dict],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {joint_name: (world_position, world_rotation_matrix)} at rest pose."""

    # Build lookup
    by_name = {j["name"]: j for j in joints}

    # Topological sort (parents before children)
    ordered: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        parent = by_name[name].get("parent")
        if parent:
            visit(parent)
        ordered.append(name)

    for j in joints:
        visit(j["name"])

    # Compute transforms
    transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # name → (pos, rot)

    for name in ordered:
        jnt = by_name[name]
        offset = np.array(jnt.get("offset") or [0.0, 0.0, 0.0], dtype=float)
        R_local = orientation_to_matrix(jnt.get("orientation"))

        parent_name = jnt.get("parent")
        if parent_name is None:
            # Root: offset is world position, orientation is world rotation
            world_pos = offset
            world_rot = R_local
        else:
            p_pos, p_rot = transforms[parent_name]
            world_pos = p_pos + p_rot @ offset
            world_rot = p_rot @ R_local

        transforms[name] = (world_pos, world_rot)

    return transforms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print global 3D joint positions of a skeleton at rest pose."
    )
    p.add_argument("skeleton", type=Path, help="Skeleton YAML file")
    p.add_argument(
        "--units", choices=["m", "cm", "mm"], default="m",
        help="Output unit (default: m, same as skeleton file)"
    )
    p.add_argument(
        "--group", metavar="NAME",
        help="Show only joints belonging to this group (default: all joints)"
    )
    p.add_argument(
        "--bone-lengths", action="store_true",
        help="Also print bone lengths (parent→child distances)"
    )
    p.add_argument(
        "--zup", action="store_true",
        help="Convert from Y-up (skeleton native) to Z-up: new_X=X, new_Y=old_Z, new_Z=old_Y"
    )
    p.add_argument(
        "--floor-z", type=float, default=0.0, metavar="VALUE",
        help="Z coordinate to assign to the lowest joint (default: 0). "
             "Translates the whole skeleton vertically. Only meaningful with --zup."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.skeleton) as f:
        data = yaml.safe_load(f)

    joints: list[dict] = data.get("joints", [])
    if not joints:
        print("No joints found in skeleton file.", file=sys.stderr)
        sys.exit(1)

    scale = {"m": 1.0, "cm": 100.0, "mm": 1000.0}[args.units]

    # Filter by group if requested
    if args.group:
        groups = {g["name"]: set(g.get("joints", [])) for g in data.get("groups", [])}
        if args.group not in groups:
            print(f"Group '{args.group}' not found. Available: {list(groups)}", file=sys.stderr)
            sys.exit(1)
        allowed = groups[args.group]
        joints_to_show = {j["name"] for j in joints if j["name"] in allowed}
    else:
        joints_to_show = {j["name"] for j in joints}

    transforms = compute_joint_world_transforms(joints)

    # Build parent lookup for bone-length output
    parent_of = {j["name"]: j.get("parent") for j in joints}

    # Collect positions, apply coord conversion and floor translation
    positions: dict[str, np.ndarray] = {}
    for name, (pos, _) in transforms.items():
        p = pos * scale
        if args.zup:
            # Y-up → Z-up: swap Y and Z axes
            p = np.array([p[0], p[2], p[1]])
        positions[name] = p

    if args.zup:
        visible_z = [positions[n][2] for n in joints_to_show if n in positions]
        z_offset = args.floor_z - min(visible_z)
        for name in positions:
            positions[name][2] += z_offset

    print(f"Skeleton : {data.get('name', args.skeleton.stem)}")
    print(f"Units    : {args.units}")
    print(f"Coord    : {'Z-up' if args.zup else 'Y-up (skeleton native)'}")
    if args.zup:
        print(f"Floor Z  : {args.floor_z}")
    if args.group:
        print(f"Group    : {args.group}")
    print(f"Joints   : {len(joints_to_show)}")
    print()

    # Column widths
    name_w = max(len(n) for n in joints_to_show)
    header = f"{'Joint':<{name_w}}   {'X':>10}  {'Y':>10}  {'Z':>10}"
    if args.bone_lengths:
        header += f"  {'BoneLen':>9}"
    print(header)
    print("-" * len(header))

    # Print in topological order
    for name in transforms:
        if name not in joints_to_show:
            continue
        x, y, z = positions[name]
        line = f"{name:<{name_w}}   {x:10.4f}  {y:10.4f}  {z:10.4f}"
        if args.bone_lengths:
            parent = parent_of.get(name)
            if parent and parent in transforms:
                bone_len = float(np.linalg.norm(positions[name] - positions[parent]))
                line += f"  {bone_len:9.4f}"
            else:
                line += f"  {'(root)':>9}"
        print(line)


if __name__ == "__main__":
    main()
