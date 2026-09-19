# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""reconcile_conflict_review.py -- folds a conflict-review pass's
resolved assignments (see audit_cross_slot_consistency.py /
prepare_conflict_review.py) back into the main tracklet_group_
assignments file.

For every member that was actually resolved during the conflict review
(appears in some group of --conflict-assignments -- i.e. the reviewer
made a real decision about it, whether "keep the old slot", "reassign
it", or "reject it"), this:
  1. removes it from whatever group currently holds it in
     --main-assignments (deleting that group entirely if it becomes
     empty), then
  2. adds each resolved conflict-review group as its own new group in
     the main file, under a `reconciled_N` id (a namespace that can't
     collide with either file's own group_ids) -- preserving the
     *reviewer's own* grouping, slot, and rejection decision exactly,
     not re-deriving anything from geometry again.

A member that was part of the conflict-review *groups* file (i.e. it was
flagged for review) but was never actually acted on -- no entry
references it in --conflict-assignments -- is left untouched in the main
file: the review didn't reach a decision for it, so there is nothing to
correct. (This can happen if a chain was only partially worked through;
harmless, just means that member's original slot stands.)

A resolution that is a genuine no-op -- the conflict-review group's
member exactly matches (same camera/tracklet/frame-range *and* same
slot/rejection) what the main file already has -- is left completely
alone, not torn out and re-added as its own new group. This matters:
the conflict-review process only ever looks at one original B2
component's own conflicting members, so a member that also happens to
already be correctly grouped with something *outside* that component in
the main file (e.g. a real cross-camera pair fixed by hand before the
audit ever ran) would otherwise get pulled out of that good grouping and
turned into a needless singleton, just because it also showed up
(correctly) in an unrelated conflict elsewhere. Found exactly this
regression on real data: `gopro-11_mini_01#66` + `gopro13_02#79`
(manually merged into one ankle_lat_L group before the audit) were torn
apart because `#66` also appeared, correctly, in `group_108`'s conflict
chain and got confirmed there via "keep own slot" -- naively replacing
its grouping with that chain's own (narrower) view of it split the pair
back into two single-camera groups B4 couldn't triangulate.

Writes to a new file by default (never silently overwrites
--main-assignments) so the result can be inspected/diffed before
replacing the working file.

Usage:
    python tools/reconcile_conflict_review.py \\
        --main-assignments scratch/dot_ground_truth/tracklet_group_assignments_full_v2.json \\
        --conflict-assignments scratch/dot_ground_truth/tracklet_group_assignments_conflicts.json \\
        --output scratch/dot_ground_truth/tracklet_group_assignments_full_v2.reconciled.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _parse_member(entry: list) -> tuple[str, int, int | None, int | None]:
    """Same shape as label_tracklet_groups_gui.py's own helper."""
    if len(entry) == 2:
        label, tid = entry
        return label, tid, None, None
    label, tid, lo, hi = entry
    return label, tid, lo, hi


def _ranges_overlap(lo1: int | None, hi1: int | None, lo2: int | None, hi2: int | None) -> bool:
    """True if [lo1, hi1) and [lo2, hi2) overlap (None = unbounded on
    that side). Used instead of exact-key matching so a main-file entry
    covering a *whole* tracklet is correctly recognized as superseded
    when the conflict review split it into sub-ranges (2026-09-12,
    found via oneplus9pro-01#1235: whole in the main file, resolved as
    two split fragments during conflict review -- an exact-key match
    would have missed both)."""
    lo1_b = float("-inf") if lo1 is None else lo1
    hi1_b = float("inf") if hi1 is None else hi1
    lo2_b = float("-inf") if lo2 is None else lo2
    hi2_b = float("inf") if hi2 is None else hi2
    return lo1_b < hi2_b and lo2_b < hi1_b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-assignments", required=True, help="label_tracklet_groups_gui.py's main --output file.")
    ap.add_argument("--conflict-assignments", required=True,
                     help="label_tracklet_groups_gui.py's --output file from the conflict-review pass.")
    ap.add_argument("--output", required=True, help="Written fresh -- never overwrites --main-assignments in place.")
    args = ap.parse_args()

    main_assign: dict[str, dict] = json.loads(Path(args.main_assignments).read_text())
    conflict_assign: dict[str, dict] = json.loads(Path(args.conflict_assignments).read_text())

    # Current status of every EXACT (label, tid, lo, hi) key already in
    # the main file -- used to detect a genuine no-op resolution below.
    main_status_by_exact_key: dict[tuple, tuple] = {}
    for a in main_assign.values():
        for m in a.get("members", []):
            key = _parse_member(m)
            main_status_by_exact_key[key] = (a.get("slot"), a.get("rejected"), a.get("reject_reason"))

    # Split each conflict-review group's members into real corrections
    # (need to move in the main file) vs. genuine no-ops (exact key *and*
    # exact slot/rejection already match the main file -- leave them
    # completely alone, in whatever grouping they already have there;
    # see module docstring for the real regression this avoids).
    real_correction_groups: list[dict] = []
    n_noop = 0
    for a in conflict_assign.values():
        status = (a.get("slot"), a.get("rejected"), a.get("reject_reason"))
        real_members = []
        for m in a.get("members", []):
            key = _parse_member(m)
            if main_status_by_exact_key.get(key) == status:
                n_noop += 1
            else:
                real_members.append(m)
        if real_members:
            real_correction_groups.append({
                "slot": a.get("slot"), "rejected": a.get("rejected"),
                "reject_reason": a.get("reject_reason"), "members": real_members,
            })

    resolved_ranges_by_tid: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    n_resolved = 0
    for grp in real_correction_groups:
        for m in grp["members"]:
            key = _parse_member(m)
            resolved_ranges_by_tid[(key[0], key[1])].append((key[2], key[3]))
            n_resolved += 1

    # Step 1: remove every main-file member whose own range OVERLAPS any
    # *real-correction* range for the same (label, tid) -- not an
    # exact-key match, so a whole-tracklet main-file entry is correctly
    # recognized as superseded when the conflict review split it into
    # sub-ranges. No-op members were already excluded above, so they
    # can't trigger removal of their own (unchanged) entry here.
    n_removed = 0
    for gid in list(main_assign.keys()):
        a = main_assign[gid]
        kept = []
        for m in a.get("members", []):
            label, tid, lo, hi = _parse_member(m)
            ranges = resolved_ranges_by_tid.get((label, tid), [])
            superseded = any(_ranges_overlap(lo, hi, rlo, rhi) for rlo, rhi in ranges)
            if superseded:
                n_removed += 1
            else:
                kept.append(m)
        if len(kept) != len(a.get("members", [])):
            if kept:
                a["members"] = kept
            else:
                del main_assign[gid]

    # Step 2: add each real-correction group as its own new group in the
    # main file, preserving its slot/rejection/members exactly as the
    # reviewer left them (minus any no-op members, which stayed put).
    n_added = 0
    next_idx = 0
    for grp in real_correction_groups:
        while f"reconciled_{next_idx}" in main_assign:
            next_idx += 1
        entry = {"slot": grp["slot"], "rejected": grp["rejected"], "members": grp["members"]}
        if grp["reject_reason"] is not None:
            entry["reject_reason"] = grp["reject_reason"]
        main_assign[f"reconciled_{next_idx}"] = entry
        next_idx += 1
        n_added += 1

    print(f"{n_noop} members already exactly matched the main file (no-op, left untouched)")
    print(f"{n_resolved} members are real corrections")
    print(f"removed {n_removed} member-entries from their old main-file group "
          f"(dropping any group left empty)")
    print(f"added {n_added} reconciled groups from the conflict review")
    print(f"main assignments: {len(main_assign)} groups after reconciliation "
          f"(was {len(json.loads(Path(args.main_assignments).read_text()))})")

    Path(args.output).write_text(json.dumps(main_assign, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
