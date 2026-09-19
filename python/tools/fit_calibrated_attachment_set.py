# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""fit_calibrated_attachment_set.py -- P-B, step B4 (redesign doc §4):
fits a real, per-slot marker attachment set (offset + normal, in the
catalog's scale-robust along/lateral/anterior parametrization, §1.5-1.6)
from a completed B3 manual tracklet-group -> slot assignment.

Replaces P-A's exploratory `_DEFAULT_TRIAL` probe catalog -- which was
never meant to have one distinct entry per real slot (several real slots
share a probe, e.g. ankle_lat_R/ankle_med_R both -> a_ankle_R, a known,
documented limitation) -- with real offsets fitted separately for every
one of the 16 real slots, directly addressing the lat/med suggestion
ambiguity Harri hit reviewing B3 (status.md, 2026-09-11/12).

Per the design doc: for each assigned group and every frame in its
lifetime, the parent joint's FK transform `T(t)` and the group's fused 3D
position `p(t)` give one equation `offset_local ~= T(t)[:3,:3]^T @
(p(t) - T(t)[:3,3])` (constant since the attachment is rigid) --
aggregated via median (robust to the occasional bad fused point, e.g. a
last-frame outlier per status.md's 2026-09-12 entry) over every frame of
every group assigned to that slot, not just one group's own lifetime.
`normal` defaults to the offset's own perpendicular direction
(`normalize([0, lateral, anterior])`) per §1.6 -- deferred, not fitted
against observed visibility, per the design doc's own explicit scope cut.

Parent-joint attachment (§1.3) per real slot, matching the skeleton's own
joint names and the convention already established in `_RIGHT_TRIAL`
(prototype_fk_marker_prediction.py): hip -> thigh (rides the femur),
knee_lat/med/front and ankle_lat/med -> shin (both ride the tibia --
ankle markers sit near the distal shin/malleoli, not the foot bone),
heel/toe -> foot. Confirmed each joint's local +Y is down-bone (toward
the child) for both sides of this skeleton, matching §1.4's e_long
convention directly -- local X/Z read off as lateral/anterior with no
further axis-building needed.

Fits **both sides independently** from their own real data (not
mirror-generated) since a calibrated set, unlike the nominal catalog, can
and should reflect whatever real (a)symmetry this subject/session's data
shows -- e.g. ankle_lat_R had 24 assigned groups vs. ankle_lat_L's 9 in
this session, real, not a labeling accident.

Usage:
    python tools/fit_calibrated_attachment_set.py \\
        --session /path/to/session.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --tracking-run 990fb01a-6c72-47bc-bdb7-4d31147d2ef7 \\
        --groups scratch/dot_ground_truth/tracklet_groups_full_v2.json \\
        --assignments scratch/dot_ground_truth/tracklet_group_assignments_full_v2.json \\
        --output scratch/dot_ground_truth/leg.calibrated.yaml
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.setup.extrinsics_solver import _proj_matrix, _undistort_pts  # noqa: E402
from posetrak.db.skeleton_layout import SkeletonLayout  # noqa: E402
from tools.build_tracklet_groups import collect_tracklets  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states, load_sync_table  # noqa: E402
from tools.prototype_multi_camera_fusion import triangulate_multiview  # noqa: E402

# real slot base name -> parent joint base name (§1.3; see module docstring)
_SLOT_PARENT_BASE = {
    "hip": "thigh",
    "knee_lat": "shin", "knee_med": "shin", "knee_front": "shin",
    "ankle_lat": "shin", "ankle_med": "shin",
    "heel": "foot", "toe": "foot",
}


def _slot_parent_joint(slot: str) -> str:
    base, side = slot.rsplit("_", 1)
    return f"{_SLOT_PARENT_BASE[base]}.{side}"


def _parse_member(entry: list) -> tuple[str, int, int | None, int | None]:
    """Same shape as label_tracklet_groups_gui.py's own helper -- a group's
    [camera_label, tracklet_id] pair, or a manual mid-tracklet split's
    [camera_label, tracklet_id, frame_lo, frame_hi]. Duplicated (not
    imported) to avoid pulling a PySide6 GUI module into this headless
    fitting script for one two-line helper."""
    if len(entry) == 2:
        label, tid = entry
        return label, tid, None, None
    label, tid, lo, hi = entry
    return label, tid, lo, hi


class _TransformCache:
    """Per-global-time joint transform cache, shared across every group --
    same caching pattern as build_tracklet_groups.FKPredictor, but exposes
    raw joint transforms directly (not pre-defined probe predictions),
    since B4 needs the parent joint's own transform for an arbitrary
    joint name, not a fixed attachment set."""

    def __init__(self, conn, tracking_run_id: str, time_tol: float = 0.05):
        sk_id = conn.execute(
            "SELECT skeleton_id FROM tracking_runs WHERE id = ?", (tracking_run_id,)
        ).fetchone()["skeleton_id"]
        yc = conn.execute("SELECT yaml_content FROM skeletons WHERE id = ?", (sk_id,)).fetchone()["yaml_content"]
        self.layout = SkeletonLayout(yc)
        # The skeleton row's own `name` column is the per-person instance
        # label (e.g. "Default female"); the rig-topology identifier this
        # catalog's `parent_joint` names actually depend on is the YAML's
        # own top-level `name` field (e.g. "reallusion-no-waist") -- §1.9's
        # `skeleton_topology`.
        self.skeleton_topology = yaml.safe_load(yc)["name"]
        rows = conn.execute(
            "SELECT timestamp_s, state FROM tracking_results WHERE run_id = ? AND is_smoothed = 1 ORDER BY tracker_step",
            (tracking_run_id,),
        ).fetchall()
        if not rows:
            # A run tracked without --smooth (e.g. a quick validation run) has
            # no is_smoothed=1 rows at all -- fall back to the raw per-step
            # state rather than refusing outright. Only ever used for parent-
            # joint transforms here (not for the marker fit's own residuals),
            # so raw-vs-smoothed makes negligible difference to offset_local.
            rows = conn.execute(
                "SELECT timestamp_s, state FROM tracking_results WHERE run_id = ? AND is_smoothed = 0 ORDER BY tracker_step",
                (tracking_run_id,),
            ).fetchall()
        self.times = np.array([r["timestamp_s"] for r in rows])
        self.blobs = [bytes(r["state"]) for r in rows]
        self.time_tol = time_tol
        self._cache: dict[float, dict | None] = {}
        self._bone_len: dict[str, float] = {}

    def at(self, t: float) -> dict[str, np.ndarray] | None:
        key = round(t, 3)
        if key not in self._cache:
            i = int(np.argmin(np.abs(self.times - t)))
            if abs(self.times[i] - t) > self.time_tol:
                self._cache[key] = None
            else:
                dec = self.layout.decode_state_blob(self.blobs[i])
                self._cache[key] = self.layout.compute_joint_transforms(dec)
        return self._cache[key]

    def bone_length(self, joint_name: str) -> float:
        if joint_name not in self._bone_len:
            for j in self.layout.joints:
                if j.name == joint_name:
                    self._bone_len[joint_name] = float(np.linalg.norm(j.bone_tip_offset))
                    break
            else:
                raise KeyError(joint_name)
        return self._bone_len[joint_name]


def group_fused_positions(
    conn, group_members: list, meta: dict, sync_table, svid_by_cam: dict, states: dict, label_to_cam: dict,
) -> list[tuple[float, np.ndarray]]:
    """[(global_time, fused_xyz), ...] for one assigned group -- N-view
    triangulation at every instant where 2+ members have a detection,
    walking the member with the most frames as the anchor (the same
    "shared frames" logic as build_tracklet_groups.py / GroupAnalyzer)."""
    members = [(label_to_cam[label], tid, lo, hi) for label, tid, lo, hi in (_parse_member(e) for e in group_members)]
    frames_by_member = {}
    for cam_id, tid, lo, hi in members:
        svid = svid_by_cam[cam_id]
        flo = sync_table.lookup(meta["start_time"], svid)
        fhi = sync_table.lookup(meta["end_time"], svid)
        all_tracklets = collect_tracklets(conn, meta["detection_run"], svid, flo, fhi)
        frames = all_tracklets.get(tid, {})
        if lo is not None or hi is not None:
            frames = {vf: xy for vf, xy in frames.items() if (lo is None or vf >= lo) and (hi is None or vf < hi)}
        frames_by_member[(cam_id, tid, lo, hi)] = frames

    if not frames_by_member:
        return []
    anchor = max(frames_by_member, key=lambda m: len(frames_by_member[m]))
    anchor_svid = svid_by_cam[anchor[0]]

    out = []
    for vf in frames_by_member[anchor]:
        t = sync_table.frame_to_global_time(vf, anchor_svid)
        if t is None:
            continue
        views_P, views_pt = [], []
        for m, frames in frames_by_member.items():
            cam_id = m[0]
            if m == anchor:
                fidx = vf
            else:
                fidx = sync_table.lookup(t, svid_by_cam[cam_id])
                if fidx is None or fidx not in frames:
                    continue
            px, py = frames[fidx]
            pt = _undistort_pts(np.array([[px, py]]), states[cam_id])[0]
            views_P.append(_proj_matrix(states[cam_id]))
            views_pt.append((float(pt[0]), float(pt[1])))
        if len(views_P) >= 2:
            xyz = triangulate_multiview(views_P, views_pt)
            if xyz is not None:
                out.append((t, xyz))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--tracking-run", required=True)
    ap.add_argument("--groups", required=True, help="build_tracklet_groups.py's --output file (for meta).")
    ap.add_argument("--assignments", required=True, help="label_tracklet_groups_gui.py's --output file.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    states = load_camera_states(conn, args.shot_id)
    sync_table, svid_by_cam = load_sync_table(conn, args.shot_id)
    label_to_cam = {r["label"]: r["id"] for r in conn.execute("SELECT id, label FROM camera_instances")}

    groups_doc = json.loads(Path(args.groups).read_text())
    meta = groups_doc["meta"]
    assign = json.loads(Path(args.assignments).read_text())

    xform = _TransformCache(conn, args.tracking_run)

    # per slot: parallel lists of (fitted-basis) samples and the raw
    # (R, trans, xyz) needed to compute residuals against the fit after
    samples: dict[str, list[np.ndarray]] = defaultdict(list)
    raw_by_slot: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)

    n_groups_used = 0
    n_groups_by_slot: dict[str, int] = defaultdict(int)
    n_groups_no_fusion: dict[str, int] = defaultdict(int)
    for a in assign.values():
        slot = a.get("slot")
        if not slot or a.get("rejected"):
            continue
        n_groups_by_slot[slot] += 1
        parent_joint = _slot_parent_joint(slot)
        fused = group_fused_positions(conn, a["members"], meta, sync_table, svid_by_cam, states, label_to_cam)
        if not fused:
            # single-camera-only group (or every member from the same
            # camera) -- no simultaneous 2-different-camera view ever
            # exists to triangulate from. A real data gap, not a bug --
            # surfaced below, not silently dropped.
            n_groups_no_fusion[slot] += 1
            continue
        n_groups_used += 1
        for t, xyz in fused:
            T = xform.at(t)
            if T is None or parent_joint not in T:
                continue
            Tp = T[parent_joint]
            R, trans = Tp[:3, :3], Tp[:3, 3]
            offset_local = R.T @ (xyz - trans)
            samples[slot].append(offset_local)
            raw_by_slot[slot].append((R, trans, xyz))

    print(f"{n_groups_used} assigned groups contributed samples across {len(samples)} slots\n")

    all_assigned_slots = set(n_groups_by_slot)
    unfit_slots = all_assigned_slots - set(samples)
    if unfit_slots:
        print("WARNING -- slots with assigned groups but zero usable (cross-camera) samples:")
        for slot in sorted(unfit_slots):
            print(f"  {slot}: {n_groups_by_slot[slot]} assigned group(s), all single-camera-only "
                  f"(no simultaneous 2-different-camera view to triangulate from) -- a real data gap, "
                  f"not fit here; needs better multi-camera coverage of this slot in a future capture, "
                  f"or a single-camera-only fitting fallback (not built).")
        print()
    partial_slots = {s: n_groups_no_fusion[s] for s in samples if n_groups_no_fusion.get(s)}
    if partial_slots:
        print("Note -- some assigned groups per slot contributed nothing (single-camera-only), "
              "but the slot still got samples from its other groups:")
        for slot, n in sorted(partial_slots.items()):
            print(f"  {slot}: {n}/{n_groups_by_slot[slot]} assigned groups unusable")
        print()

    catalog_markers = []
    for slot in sorted(samples):
        offs = np.array(samples[slot])
        n = len(offs)
        median_off = np.median(offs, axis=0)
        parent_joint = _slot_parent_joint(slot)
        bone_len = xform.bone_length(parent_joint)

        lateral = float(median_off[0])
        along = float(median_off[1] / bone_len)
        anterior = float(median_off[2])
        normal_vec = np.array([0.0, lateral, anterior])
        norm = float(np.linalg.norm(normal_vec))
        normal = (normal_vec / norm).tolist() if norm > 1e-9 else [0.0, 1.0, 0.0]

        residuals = np.array([
            float(np.linalg.norm((R @ median_off + trans) - xyz)) for R, trans, xyz in raw_by_slot[slot]
        ])
        median_res, p90_res = float(np.median(residuals)), float(np.percentile(residuals, 90))

        catalog_markers.append({
            "name": slot, "parent_joint": parent_joint,
            "along": round(along, 4), "lateral": round(lateral, 4), "anterior": round(anterior, 4),
            # Resolved local [x,y,z] in metres (x=lateral, y=along-bone,
            # z=anterior) -- D5 in the redesign doc's own open-questions
            # list: store the named, scale-robust fields for authoring/
            # interpretability, but also emit the raw offset a loader can
            # use directly with no further resolution step (this is what
            # prototype_fk_marker_prediction.py's own _load_attachment_set()
            # expects, so this file is directly usable as an
            # --attachment-set with no adapter needed).
            "offset": [round(float(x), 4) for x in median_off],
            "normal": [round(float(x), 3) for x in normal],
            "n_samples": n, "residual_median_m": round(median_res, 4), "residual_p90_m": round(p90_res, 4),
        })
        print(f"  {slot:16s} parent={parent_joint:9s} n={n:5d}  along={along:.3f}  "
              f"lateral={lateral:+.3f}  anterior={anterior:+.3f}  "
              f"residual median={median_res * 100:.1f}cm p90={p90_res * 100:.1f}cm")

    # Right-side base joint names actually depended on (§1.9's
    # requires_joints) -- derived from what was actually used, not assumed.
    requires_joints = sorted({j.split(".")[0] for j in {_slot_parent_joint(s) for s in samples}})

    out_doc = {
        "module": "leg",
        "skeleton_topology": xform.skeleton_topology,
        "requires_joints": requires_joints,
        "calibration": {
            "session": args.session, "shot_id": args.shot_id, "tracking_run": args.tracking_run,
            "detection_run": meta["detection_run"],
            "method": "fit_calibrated_attachment_set.py -- per-frame offset_local, median over every "
                      "assigned tracklet group's own fused 3D trajectory",
            "date": date.today().isoformat(),
            "n_groups": n_groups_used,
            "unfit_slots": sorted(unfit_slots),
        },
        "markers": catalog_markers,
    }
    # REUSE-IgnoreStart -- builds the *output* file's own SPDX header as a
    # string; concatenated so `reuse lint` doesn't mistake it for a badly-
    # formed header on this .py file itself.
    spdx_tag = "SPDX" "-License-Identifier"
    spdx_header = f"# SPDX-FileCopyrightText: 2026 Harri Kaimio\n#\n# {spdx_tag}: Apache-2.0\n\n"
    # REUSE-IgnoreEnd
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(spdx_header + yaml.safe_dump(out_doc, sort_keys=False))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
