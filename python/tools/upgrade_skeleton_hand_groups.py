#!/usr/bin/env python3
"""
upgrade_skeleton_hand_groups.py — Correct the HandL/HandR groups: entries in a
reallusion-style skeleton (the rig used throughout this codebase's production
data) so they work with the hierarchical solver. See
docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md
("Exact group definitions") and docs/skeleton-format.md ("Hierarchical
solver fields") for the reviewed, finalized definitions this script applies.

Two corrections, applied to any skeleton with main/HandL/HandR groups and a
hand.L/forearm.L joint, regardless of exact hand topology:

1. Every groups: joints:/markers: entry that names a joint or marker not
   actually present in the skeleton is removed (mirrors
   skeleton_loader.cpp's stale-reference warning -- these entries are
   always wrong, never a matter of topology or judgment).
2. HandL/HandR get freeflyer_joint: forearm.{L,R} and ref_marker:
   MRK-wrist.{L,R} -- the new optional groups: fields from
   docs/skeleton-format.md's "Hierarchical solver fields" section --
   PROVIDED the skeleton's fingers attach directly to hand.{L,R} (no real
   palm.0N.{L,R} joints), matching the "reallusion-no-waist" family the
   design doc's "Exact group definitions" section actually verified
   against. On that family this also adds hand.{L,R} to HandL/HandR's
   joints list and MRK-wrist.{L,R} to its markers list -- hand.{L,R}
   becomes a genuinely estimated DOF inside the hand group, per the design
   doc's "wrist ownership: solved twice, child wins" (it also stays in
   main's list -- both groups estimate it, deliberately).

A skeleton with real palm.0N.{L,R} joints (fingers attach to those instead)
is a different, unreviewed topology -- discovered mid-implementation when
this script's first version blindly stripped palm.* references from
tests/data/Harri_skeleton-regress-test.yaml and Harri_skeleton-shouldery-
rot.yaml, where they are real, load-bearing joints, not phantom references.
For that case this script only applies correction 1 (unambiguous, safe
regardless of topology) and reports the rest as skipped, rather than
guessing at group membership or a freeflyer/reference marker choice the
design doc never analyzed for that rig.

Idempotent: running it again on an already-upgraded skeleton reports no
changes.

List ordering inside a corrected groups: entry may differ from the design
doc's illustrative YAML (this script appends/prepends rather than
reproducing the doc's exact order) -- membership is what matters, not order.

Usage:
    # Single YAML file, in place:
    python3 python/tools/upgrade_skeleton_hand_groups.py --file skeleton.yaml

    # Single YAML file, write elsewhere (leaves the input untouched):
    python3 python/tools/upgrade_skeleton_hand_groups.py --file skeleton.yaml --output out.yaml

    # All matching skeletons in a registry/session DB. skeletons.id is a
    # SHA-256 content hash (see posetrak/db/manage_skeleton.py) -- existing
    # rows are immutable by convention, so this INSERTs a NEW row per match
    # instead of mutating yaml_content in place, with parent_id set to the
    # original skeleton's id to record the lineage. Existing tracking_runs
    # rows keep referencing the original, unmodified skeleton.
    python3 python/tools/upgrade_skeleton_hand_groups.py --db session.db
    python3 python/tools/upgrade_skeleton_hand_groups.py --db session.db --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import re
import sqlite3
import sys
from pathlib import Path

import yaml

SIDES = ["L", "R"]


def upgrade_groups(skeleton: dict) -> tuple[list[str], list[str]]:
    """Mutate skeleton['groups'] in place. Returns (changes, warnings).

    changes is empty if nothing needed changing (not this skeleton family, or
    already upgraded) -- callers should treat an empty changes list as "no
    write needed" even if warnings is non-empty, since a warning reports a
    pre-existing issue this script didn't create and can't fix (e.g. a
    groups: entry referencing a joint that was removed from the skeleton for
    an unrelated reason)."""
    groups = {g["name"]: g for g in skeleton.get("groups") or []}
    joint_names = {j["name"] for j in skeleton.get("joints") or []}
    marker_names = {m["name"] for m in skeleton.get("markers") or []}

    if "main" not in groups or "HandL" not in groups or "HandR" not in groups:
        return [], []
    if "hand.L" not in joint_names or "forearm.L" not in joint_names:
        return [], []  # not this skeleton family

    changes: list[str] = []
    warnings: list[str] = []

    # Correction 1 (always safe): drop any joints:/markers: entry that
    # doesn't name a real joint/marker, from every group, regardless of
    # topology -- an entry either resolves or it doesn't.
    for group in groups.values():
        joints = group.get("joints") or []
        kept_j = [j for j in joints if j in joint_names]
        if kept_j != joints:
            changes.append(f"{group['name']}: removed stale joints "
                           f"{sorted(set(joints) - set(kept_j))}")
            group["joints"] = kept_j

        markers = group.get("markers") or []
        kept_m = [m for m in markers if m in marker_names]
        if kept_m != markers:
            changes.append(f"{group['name']}: removed stale markers "
                           f"{sorted(set(markers) - set(kept_m))}")
            group["markers"] = kept_m

    # Correction 2 (topology-specific): only for the family the design doc
    # actually analyzed -- fingers attach directly to hand.{L,R}, no real
    # intermediate palm.0N.{L,R} joints.
    has_real_palm_joints = any(f"palm.0{n}.{side}" in joint_names
                               for n in range(1, 5) for side in SIDES)
    if has_real_palm_joints:
        warnings.append(
            "NOTE: this skeleton has real palm.0N.{L,R} joints (fingers don't attach "
            "directly to hand.{L,R}) -- a hand topology the hierarchical solver design "
            "doc's group definitions were never verified against. Left HandL/HandR's "
            "joint/marker membership and freeflyer_joint/ref_marker untouched; only "
            "removed unambiguously-stale references above, if any.")
    else:
        for side in SIDES:
            group = groups[f"Hand{side}"]
            hand_joint = f"hand.{side}"
            joints = group.get("joints") or []
            if hand_joint not in joints:
                group["joints"] = [hand_joint] + joints
                changes.append(f"Hand{side}: added '{hand_joint}' to joints")

            wrist_marker = f"MRK-wrist.{side}"
            markers = group.get("markers") or []
            if wrist_marker not in markers:
                group["markers"] = [wrist_marker] + markers
                changes.append(f"Hand{side}: added '{wrist_marker}' to markers")

            freeflyer = f"forearm.{side}"
            if group.get("freeflyer_joint") != freeflyer:
                group["freeflyer_joint"] = freeflyer
                changes.append(f"Hand{side}: freeflyer_joint = '{freeflyer}'")
            if group.get("ref_marker") != wrist_marker:
                group["ref_marker"] = wrist_marker
                changes.append(f"Hand{side}: ref_marker = '{wrist_marker}'")

    # Sanity check: every remaining joints:/markers: entry across every group
    # (not just the ones this script touched) must resolve to a real
    # joint/marker -- mirrors skeleton_loader.cpp's stale-reference warning,
    # surfaced here too so a bad correction doesn't ship silently. These are
    # diagnostic only -- kept separate from `changes` so a pre-existing,
    # unrelated stale reference doesn't force a write when this script itself
    # found nothing to correct.
    for group in groups.values():
        for j in group.get("joints") or []:
            if j not in joint_names:
                warnings.append(
                    f"WARNING: group '{group['name']}' still references unknown joint '{j}'")
        for m in group.get("markers") or []:
            if m not in marker_names:
                warnings.append(
                    f"WARNING: group '{group['name']}' still references unknown marker '{m}'")

    return changes, warnings


def _splice_groups_section(original_text: str, new_groups: list[dict]) -> str:
    """Replace only the top-level 'groups:' block with a freshly rendered one,
    leaving every other line (joints:, markers:, comments, formatting,
    float representations) byte-identical. A full yaml.safe_load + yaml.dump
    round-trip of the whole file was tried first and rejected -- PyYAML's
    dumper reformats every list in the file (e.g. flow-style `[x, y, z]`
    offsets become one-item-per-line) and risks subtly altering float
    representations, neither of which has anything to do with what this
    script is supposed to change.
    """
    lines = original_text.splitlines(keepends=True)
    start: int | None = None
    end = len(lines)
    for i, line in enumerate(lines):
        if start is None:
            if re.match(r"^groups:\s*$", line):
                start = i
            continue
        # Once inside the groups: block, it ends at the next top-level
        # (column-0) key -- or EOF, if groups: is the last section.
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*:", line):
            end = i
            break
    if start is None:
        raise ValueError("No top-level 'groups:' section found to replace")

    new_block = yaml.dump({"groups": new_groups}, default_flow_style=False, allow_unicode=True,
                          sort_keys=False)
    return "".join(lines[:start]) + new_block + "".join(lines[end:])


def upgrade_yaml_text(yaml_text: str) -> tuple[str, list[str], list[str]]:
    """Returns (possibly-updated YAML text, change messages, warning messages).
    If changes is empty, the returned text is identical to yaml_text (nothing
    to write) even if warnings is non-empty."""
    skeleton = yaml.safe_load(yaml_text)
    changes, warnings = upgrade_groups(skeleton)
    if not changes:
        return yaml_text, changes, warnings
    new_text = _splice_groups_section(yaml_text, skeleton["groups"])
    return new_text, changes, warnings


def upgrade_file(path: Path, output: Path | None) -> None:
    text = path.read_text(encoding="utf-8")
    new_text, changes, warnings = upgrade_yaml_text(text)
    for w in warnings:
        print(f"{path}: {w}")
    if not changes:
        print(f"{path}: no changes needed (not a matching skeleton, or already upgraded)")
        return
    for c in changes:
        print(f"{path}: {c}")
    out_path = output or path
    out_path.write_text(new_text, encoding="utf-8")
    print(f"{path}: wrote {out_path}")


def upgrade_db(db_path: Path, dry_run: bool) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, name, yaml_content FROM skeletons").fetchall()

    for row in rows:
        new_text, changes, warnings = upgrade_yaml_text(row["yaml_content"])
        if not changes and not warnings:
            continue

        print(f"skeleton {row['id'][:12]} ({row['name']}):")
        for w in warnings:
            print(f"  {w}")
        if not changes:
            continue
        for c in changes:
            print(f"  {c}")

        new_id = hashlib.sha256(new_text.encode("utf-8")).hexdigest()
        if new_id == row["id"]:
            print("  already up to date")
            continue
        existing = conn.execute("SELECT id FROM skeletons WHERE id = ?", (new_id,)).fetchone()
        if existing is not None:
            print(f"  corrected version already exists as {existing['id'][:12]}")
            continue
        if dry_run:
            print(f"  would insert new skeleton row {new_id[:12]} (parent_id={row['id'][:12]})")
            continue

        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO skeletons (id, name, parent_id, person_label, source, yaml_content, "
            "created_at, notes) "
            "SELECT ?, name, ?, person_label, source, ?, ?, "
            "'Upgraded by upgrade_skeleton_hand_groups.py: hierarchical-solver group fields' "
            "FROM skeletons WHERE id = ?",
            (new_id, row["id"], new_text, created_at, row["id"]),
        )
        conn.commit()
        print(f"  inserted new skeleton row {new_id[:12]} (parent_id={row['id'][:12]})")

    conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--file", type=Path, help="Skeleton YAML file to upgrade")
    target.add_argument("--db", type=Path,
                        help="Registry or session DB to upgrade all matching skeletons in")
    parser.add_argument("--output", type=Path,
                        help="--file mode only: write result here instead of in place")
    parser.add_argument("--dry-run", action="store_true",
                        help="--db mode only: report without writing")
    args = parser.parse_args()

    if args.file:
        if args.dry_run:
            print("--dry-run is only valid with --db", file=sys.stderr)
            return 2
        upgrade_file(args.file, args.output)
    else:
        if args.output:
            print("--output is only valid with --file", file=sys.stderr)
            return 2
        upgrade_db(args.db, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
