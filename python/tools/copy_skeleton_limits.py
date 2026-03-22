#!/usr/bin/env python3
"""
copy_skeleton_limits.py — Copy DOF limits (and joint type) from a source skeleton YAML
to a target skeleton YAML, matching joints by name.

Only the `limits` and `type` fields are copied.  All other fields in the target
(offsets, orientations, markers, groups, etc.) are left unchanged.

Joints present in the target but absent from the source are reported as warnings;
their existing limits (if any) are preserved.

Usage:
    python3 scripts/copy_skeleton_limits.py <source.yaml> <target.yaml> [--output <out.yaml>]

    # Overwrite in place (writes back to target.yaml):
    python3 scripts/copy_skeleton_limits.py source.yaml target.yaml

    # Write to a new file (leaves target.yaml untouched):
    python3 scripts/copy_skeleton_limits.py source.yaml target.yaml --output result.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _dump(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def copy_limits(source_path: Path, target_path: Path, output_path: Path) -> None:
    source = _load(source_path)
    target = _load(target_path)

    src_joints: dict[str, dict] = {j["name"]: j for j in source.get("joints", [])}
    tgt_joints: list[dict] = target.get("joints", [])

    copied = []
    missing = []

    for joint in tgt_joints:
        name = joint["name"]
        if name not in src_joints:
            missing.append(name)
            continue

        src = src_joints[name]

        # Copy type
        if "type" in src:
            joint["type"] = src["type"]

        # Copy limits (or remove if source has none, e.g. root)
        if "limits" in src:
            joint["limits"] = src["limits"]
        elif "limits" in joint:
            del joint["limits"]

        copied.append(name)

    _dump(target, output_path)

    print(f"Copied limits for {len(copied)} joints → {output_path}")
    if missing:
        print(f"\nWarning: {len(missing)} target joints not found in source (limits unchanged):")
        for name in missing:
            print(f"  {name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Copy DOF limits and joint type from source YAML to target YAML."
    )
    p.add_argument("source", type=Path, help="Skeleton YAML to copy limits FROM")
    p.add_argument("target", type=Path, help="Skeleton YAML to copy limits INTO")
    p.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output path (default: overwrite target in place)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source.exists():
        print(f"Error: source not found: {args.source}", file=sys.stderr)
        sys.exit(1)
    if not args.target.exists():
        print(f"Error: target not found: {args.target}", file=sys.stderr)
        sys.exit(1)

    output = args.output if args.output is not None else args.target
    copy_limits(args.source, args.target, output)


if __name__ == "__main__":
    main()
