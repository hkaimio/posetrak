# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prepare_conflict_review.py -- converts audit_cross_slot_consistency.py's
output into a build_tracklet_groups.py-shaped groups file, so the flagged
conflicts can be reviewed with the *existing* label_tracklet_groups_gui.py
-- no new review UI needed, just a data-shape conversion.

Each conflict chain and each high-confidence simple mismatch becomes one
synthetic "group" (its own members list, already flagged_contradiction +
ambiguous, since that's exactly what triggered it being flagged). A
`member_hints` block records each member's *previous* slot assignment
(`"label#tid": "was: <slot>"`) so it's visible on the member checkbox
while reviewing, without needing cross_slot_conflicts.json open
separately -- see label_tracklet_groups_gui.py's own 2026-09-12 addition
for the display side of this.

Low-confidence mismatches (likely 2-camera coincidences, not real
conflicts -- see audit_cross_slot_consistency.py's own docstring) are
excluded by default; pass --include-low-confidence to review those too.

Review exactly like any other B3 pass: check the member(s) that are
actually correct for a given slot, "Assign checked as new group" (or
"Discard checked" for noise); the ones left over can then be
individually assigned too. Once done, reconcile_conflict_review.py
(TODO if built) or a manual pass folds the corrected slots back into
the main tracklet_group_assignments file.

Usage:
    python tools/prepare_conflict_review.py \\
        --conflicts scratch/dot_ground_truth/cross_slot_conflicts.json \\
        --groups scratch/dot_ground_truth/tracklet_groups_full_v2.json \\
        --output scratch/dot_ground_truth/tracklet_groups_conflicts.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _member_entry(m: dict) -> list:
    if m.get("frame_lo") is None and m.get("frame_hi") is None:
        return [m["camera_label"], m["tracklet_id"]]
    return [m["camera_label"], m["tracklet_id"], m.get("frame_lo"), m.get("frame_hi")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conflicts", required=True, help="audit_cross_slot_consistency.py's --output file.")
    ap.add_argument("--groups", required=True,
                     help="The original build_tracklet_groups.py output (only its meta block is used).")
    ap.add_argument("--include-low-confidence", action="store_true",
                     help="Also include likely-coincidental low-confidence pairs (off by default).")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    conflicts = json.loads(Path(args.conflicts).read_text())
    meta = json.loads(Path(args.groups).read_text())["meta"]

    groups: list[dict] = []
    member_hints: dict[str, str] = {}

    def add_group(members: list[dict]) -> None:
        gid = f"conflict_{len(groups)}"
        for m in members:
            member_hints[f"{m['camera_label']}#{m['tracklet_id']}"] = f"was: {m['slot']}"
        groups.append({
            "group_id": gid,
            "members": [_member_entry(m) for m in members],
            "flagged_contradiction": True,
            "ambiguous": True,
        })

    for chain in conflicts["conflict_chains"]:
        add_group(chain["members"])
    for entry in conflicts["high_confidence"]:
        add_group([entry["a"], entry["b"]])
    if args.include_low_confidence:
        for entry in conflicts["low_confidence"]:
            add_group([entry["a"], entry["b"]])

    out_doc = {"meta": meta, "groups": groups, "member_hints": member_hints}
    Path(args.output).write_text(json.dumps(out_doc, indent=2))
    note = f", {len(conflicts['low_confidence'])} low-confidence" if args.include_low_confidence else ""
    print(f"wrote {len(groups)} conflict groups ({len(conflicts['conflict_chains'])} chains, "
          f"{len(conflicts['high_confidence'])} high-confidence mismatches{note}) to {args.output}")


if __name__ == "__main__":
    main()
