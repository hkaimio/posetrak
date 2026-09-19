# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""audit_cross_slot_consistency.py -- post-B3 audit: finds pairs of
*differently*-slotted tracklets, from the same original B2 component,
that actually triangulate consistently with each other -- meaning at
least one of the two slot assignments is wrong.

Motivation (status.md, 2026-09-12): Harri checked the raw video and
found `ankle_lat_L` had real multi-camera coverage the completed B4 fit
had missed entirely. Root cause: two fragments of the *same* physical
marker (`gopro-11_mini_01#66`, `gopro13_02#79`) came from one confusing,
flagged B2 component (12 members, [CONTRADICTION][ambiguous]) and were
assigned *different* slots (`ankle_lat_L`, `toe_L`) after being split
apart during manual review -- the split process, necessarily, loses the
"these were originally linked" information once pieces are pulled apart
(finding #3 from the same session: "very large groups were confusing --
hard to figure out what commonality caused tracklets to end up
together"). This formalizes the ad hoc investigation that found it into
a reusable tool, since it directly implements the cross-camera-
consistency half of that finding's own proposed fix, just applied as a
post-review audit rather than during B2 grouping itself.

Method: for every original B2 component, look at every pair of its
*final* assigned members (after any manual split) that ended up with
different slots. Run the real `pairwise_stats()` reprojection check
(same triangulate-and-reproject test B2 itself uses) between every
cross-camera pair. An accepted pair (median/p90 within the same
thresholds B2 uses) is strong evidence they're the same physical marker
-- contradicting their disagreeing slot labels.

Two outcomes, not conflated:
  - A **simple mismatch**: exactly two tracklets, two slots, one
    consistent link. Classified `high confidence` (an anatomically
    plausible mixup -- lat/med, a knee variant, heel/ankle/toe, or a
    same-slot left/right swap -- with enough shared frames and a tight
    enough fit) or `low confidence` (small sample, or an anatomically
    implausible pair like hip/toe -- most likely a 2-camera epipolar-
    line coincidence over a short window, not a real mixup). Only
    `high confidence` cases are meant to be acted on without a visual
    check.
  - A **conflict chain**: 3+ tracklets (or 3+ distinct slots) mutually
    linked by accepted edges -- geometry alone proves at least two of
    the labels are wrong but can't say *which* is right (e.g. lat vs.
    med, or which leg) -- always flagged for a human to look at the
    actual images, regardless of sample size.

Usage:
    python tools/audit_cross_slot_consistency.py \\
        --session /path/to/session.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --groups scratch/dot_ground_truth/tracklet_groups_full_v2.json \\
        --assignments scratch/dot_ground_truth/tracklet_group_assignments_full_v2.json \\
        --output scratch/dot_ground_truth/cross_slot_conflicts.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_tracklet_groups import UnionFind, collect_tracklets, pairwise_stats  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states, load_sync_table  # noqa: E402

# Anatomically-adjacent SAME-SIDE base-slot pairs -- a plausible visual
# mixup, unlike e.g. hip/toe (opposite ends of the leg, not adjacent).
_ADJACENT_BASES = {frozenset(p) for p in [
    ("ankle_lat", "ankle_med"), ("knee_lat", "knee_med"), ("knee_lat", "knee_front"),
    ("knee_med", "knee_front"), ("heel", "ankle_lat"), ("heel", "ankle_med"), ("heel", "toe"),
]}
_MIN_SHARED = 5
_MEDIAN_THRESH_PX = 3.0
_P90_THRESH_PX = 6.0
_HIGH_CONF_MIN_N = 30
_HIGH_CONF_MAX_MEDIAN_PX = 2.0


def _parse_member(entry: list) -> tuple[str, int, int | None, int | None]:
    """Same shape as label_tracklet_groups_gui.py's own helper."""
    if len(entry) == 2:
        label, tid = entry
        return label, tid, None, None
    label, tid, lo, hi = entry
    return label, tid, lo, hi


def _base_side(slot: str) -> tuple[str, str]:
    base, side = slot.rsplit("_", 1)
    return base, side


def _plausible_mixup(slot_a: str, slot_b: str) -> bool:
    base_a, side_a = _base_side(slot_a)
    base_b, side_b = _base_side(slot_b)
    lr_swap = base_a == base_b and side_a != side_b
    adjacent_same_side = side_a == side_b and frozenset([base_a, base_b]) in _ADJACENT_BASES
    return lr_swap or adjacent_same_side


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--groups", required=True, help="build_tracklet_groups.py's --output file.")
    ap.add_argument("--assignments", required=True, help="label_tracklet_groups_gui.py's --output file.")
    ap.add_argument("--output", default=None, help="Optional: write the full flagged list as JSON.")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    states = load_camera_states(conn, args.shot_id)
    sync_table, svid_by_cam = load_sync_table(conn, args.shot_id)
    label_to_cam = {r["label"]: r["id"] for r in conn.execute("SELECT id, label FROM camera_instances")}

    groups_doc = json.loads(Path(args.groups).read_text())
    meta = groups_doc["meta"]
    assign = json.loads(Path(args.assignments).read_text())

    # Original B2 component id, keyed by the *whole tracklet* (label, tid)
    # -- a manual split doesn't change which original component a
    # fragment came from, only how it's currently grouped/assigned.
    orig_component: dict[tuple[str, int], str] = {}
    for g in groups_doc["groups"]:
        for m in g["members"]:
            orig_component[(m[0], m[1])] = g["group_id"]

    # Final assigned members, grouped by their original component.
    by_component: dict[str, list[tuple[str, int, int | None, int | None, str, str]]] = defaultdict(list)
    for gid, a in assign.items():
        slot = a.get("slot")
        if not slot or a.get("rejected"):
            continue
        for m in a.get("members", []):
            label, tid, lo, hi = _parse_member(m)
            comp = orig_component.get((label, tid))
            if comp is not None:
                by_component[comp].append((label, tid, lo, hi, slot, gid))

    frames_cache: dict[tuple[str, int], dict[int, tuple[float, float]]] = {}

    def member_frames(label: str, tid: int, lo: int | None, hi: int | None) -> dict[int, tuple[float, float]]:
        key = (label, tid)
        if key not in frames_cache:
            cam_id = label_to_cam[label]
            svid = svid_by_cam[cam_id]
            flo = sync_table.lookup(meta["start_time"], svid)
            fhi = sync_table.lookup(meta["end_time"], svid)
            frames_cache[key] = collect_tracklets(conn, meta["detection_run"], svid, flo, fhi).get(tid, {})
        frames = frames_cache[key]
        if lo is not None or hi is not None:
            frames = {vf: xy for vf, xy in frames.items() if (lo is None or vf >= lo) and (hi is None or vf < hi)}
        return frames

    simple_mismatches = []  # (comp, node_a, node_b, stats)
    conflict_chains = []  # for JSON output, mirrors what's printed live below
    n_pairs_checked = 0

    for comp, members in by_component.items():
        if len(members) < 2:
            continue
        # only cross-camera, differently-slotted pairs are candidates
        candidates = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                ma, mb = members[i], members[j]
                if ma[0] == mb[0] or ma[4] == mb[4]:  # same camera, or already the same slot
                    continue
                candidates.append((ma, mb))
        if not candidates:
            continue

        accepted: dict[tuple, dict] = {}
        for ma, mb in candidates:
            n_pairs_checked += 1
            fa = member_frames(ma[0], ma[1], ma[2], ma[3])
            fb = member_frames(mb[0], mb[1], mb[2], mb[3])
            stats = pairwise_stats(
                fa, fb, sync_table, svid_by_cam[label_to_cam[ma[0]]], svid_by_cam[label_to_cam[mb[0]]],
                states[label_to_cam[ma[0]]], states[label_to_cam[mb[0]]], _MIN_SHARED,
            )
            if stats and stats["median"] <= _MEDIAN_THRESH_PX and stats["p90"] <= _P90_THRESH_PX:
                node_a, node_b = (ma[0], ma[1], ma[2], ma[3]), (mb[0], mb[1], mb[2], mb[3])
                accepted[(node_a, node_b)] = stats

        if not accepted:
            continue

        # connected components over accepted edges -- 2 members = a
        # simple mismatch; 3+ (or 3+ distinct slots) = a conflict chain.
        nodes_involved = {n for pair in accepted for n in pair}
        node_slot = {(m[0], m[1], m[2], m[3]): m[4] for m in members}
        uf = UnionFind(nodes_involved)
        for (a, b) in accepted:
            uf.union(a, b)
        clusters: dict = defaultdict(list)
        for n in nodes_involved:
            clusters[uf.find(n)].append(n)

        for cluster_nodes in clusters.values():
            if len(cluster_nodes) < 2:
                continue
            distinct_slots = {node_slot[n] for n in cluster_nodes}
            cluster_edges = {k: v for k, v in accepted.items() if k[0] in cluster_nodes and k[1] in cluster_nodes}
            if len(cluster_nodes) == 2 and len(distinct_slots) == 2:
                (a, b), stats = next(iter(cluster_edges.items()))
                simple_mismatches.append((comp, a, node_slot[a], b, node_slot[b], stats))
            else:
                print(f"=== CONFLICT CHAIN in {comp} -- {len(cluster_nodes)} tracklets, "
                      f"{len(distinct_slots)} distinct slots, needs visual judgment ===")
                for n in sorted(cluster_nodes):
                    label, tid, lo, hi = n
                    rng = f"[{lo or ''}:{hi or ''}]" if lo is not None or hi is not None else ""
                    print(f"    {label}#{tid}{rng} -> {node_slot[n]}")
                for (a, b), stats in cluster_edges.items():
                    print(f"      {a[0]}#{a[1]} <-> {b[0]}#{b[1]}: median={stats['median']:.2f}px "
                          f"p90={stats['p90']:.2f}px n={stats['n']}")
                print()
                conflict_chains.append({
                    "component": comp,
                    "members": [
                        {"camera_label": n[0], "tracklet_id": n[1], "frame_lo": n[2], "frame_hi": n[3],
                         "slot": node_slot[n]}
                        for n in sorted(cluster_nodes)
                    ],
                    "edges": [
                        {"a": {"camera_label": a[0], "tracklet_id": a[1]}, "b": {"camera_label": b[0], "tracklet_id": b[1]},
                         "median_px": round(stats["median"], 2), "p90_px": round(stats["p90"], 2), "n_shared": stats["n"]}
                        for (a, b), stats in cluster_edges.items()
                    ],
                })

    high_conf, low_conf = [], []
    for comp, a, slot_a, b, slot_b, stats in simple_mismatches:
        entry = {
            "component": comp,
            "a": {"camera_label": a[0], "tracklet_id": a[1], "frame_lo": a[2], "frame_hi": a[3], "slot": slot_a},
            "b": {"camera_label": b[0], "tracklet_id": b[1], "frame_lo": b[2], "frame_hi": b[3], "slot": slot_b},
            "median_px": round(stats["median"], 2), "p90_px": round(stats["p90"], 2), "n_shared": stats["n"],
        }
        if _plausible_mixup(slot_a, slot_b) and stats["n"] >= _HIGH_CONF_MIN_N and stats["median"] < _HIGH_CONF_MAX_MEDIAN_PX:
            high_conf.append(entry)
        else:
            low_conf.append(entry)

    print(f"{n_pairs_checked} cross-camera, differently-slotted pairs checked within their original B2 component")
    print(f"{len(conflict_chains)} conflict chains (3+ tracklets or 3+ distinct slots -- always need visual judgment)")
    print(f"{len(simple_mismatches)} simple (2-tracklet, 2-slot) mismatches found "
          f"({len(high_conf)} high-confidence, {len(low_conf)} low-confidence/likely-coincidental)\n")

    print("=== HIGH CONFIDENCE (anatomically plausible, n>={}, median<{}px -- actionable) ===".format(
        _HIGH_CONF_MIN_N, _HIGH_CONF_MAX_MEDIAN_PX))
    for e in sorted(high_conf, key=lambda e: -e["n_shared"]):
        a, b = e["a"], e["b"]
        print(f"  {a['camera_label']}#{a['tracklet_id']} -> {a['slot']}   vs   "
              f"{b['camera_label']}#{b['tracklet_id']} -> {b['slot']}   "
              f"(median={e['median_px']}px n={e['n_shared']})")

    print(f"\n({len(low_conf)} low-confidence pairs omitted from the printout -- see --output for the full list)")

    if args.output:
        Path(args.output).write_text(json.dumps(
            {"conflict_chains": conflict_chains, "high_confidence": high_conf, "low_confidence": low_conf},
            indent=2,
        ))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
