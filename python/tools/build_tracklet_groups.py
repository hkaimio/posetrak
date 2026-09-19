# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""build_tracklet_groups.py — P-B, step B2 (redesign doc §4): group
per-camera tracklets (`MotionGatedLinker`'s own tracklet_id, already in
every detection) into cross-camera **tracklet groups** believed to be one
physical marker, by testing 3D reprojection consistency over each pair's
*shared* frames (a sustained temporal agreement, not one lucky frame --
the real difference from per-frame fusion).

Algorithm, per the design doc:
    1. Pre-filter: drop tracklets shorter than --min-lifetime, and any
       tracklet that's never within --proximity-px of *any* attachment-
       set slot's FK-predicted projection (cheap way to drop prop/glare/
       other-person tracklets before the expensive pairwise step, without
       deciding *which* slot yet).
    2. Pairwise test: for every camera pair x every (pre-filtered)
       tracklet pair with overlapping time, find shared frames (via the
       sync table), triangulate each, and require the reprojection error
       to be low on (nearly) every shared frame -- median <=
       --median-thresh-px and p90 <= --p90-thresh-px, over
       >= --min-shared-frames frames.
    3. Disambiguate: if a tracklet has multiple accepted partners in the
       same other camera, keep only the clearly-best one (runner-up's
       median >= --disambiguation-margin x the best's); otherwise drop
       all of them and flag for manual review.
    4. Group: connected components over the accepted edges. A component
       containing a pair that was directly tested and *rejected* (A-B,
       B-C accepted but A-C rejected) is flagged, not silently kept.

--validate-against-gt derives real tracklet-group ground truth for free
from a slot-labelled GT file (matching each labelled point back to its
own real tracklet_id in the given detection run) and reports grouping
precision/recall against it -- use this to tune the thresholds above
before trusting them on a full run, the same discipline Stage A's
thresholds were tuned with.

Usage:
    python tools/build_tracklet_groups.py \\
        --session /path/to/session.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --tracking-run 990fb01a-6c72-47bc-bdb7-4d31147d2ef7 \\
        --detection-run 6abcba67-4e7a-4852-87c3-14134599c1e4 \\
        --start-time 40.0 --end-time 60.0 \\
        --validate-against-gt scratch/dot_ground_truth/pc_labels_refined.json \\
        --output scratch/dot_ground_truth/tracklet_groups.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pose.db_cache import decode_dot_candidates  # noqa: E402
from app.setup.extrinsics_solver import _proj_matrix, _undistort_pts  # noqa: E402
from posetrak.db.skeleton_layout import SkeletonLayout  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states, load_sync_table  # noqa: E402
from tools.prototype_fk_marker_prediction import _DEFAULT_TRIAL, _project_raw  # noqa: E402

_UNLABELED = "unlabeled"


class FKPredictor:
    """Caches FK joint transforms by (rounded) global time -- shared across
    every camera and every tracklet's own frame at that instant, since FK
    itself doesn't depend on the camera, only the projection does."""

    def __init__(self, conn, tracking_run_id: str, states: dict, attach: dict, time_tol: float = 0.05):
        sk_id = conn.execute("SELECT skeleton_id FROM tracking_runs WHERE id = ?", (tracking_run_id,)).fetchone()["skeleton_id"]
        yc = conn.execute("SELECT yaml_content FROM skeletons WHERE id = ?", (sk_id,)).fetchone()["yaml_content"]
        self.layout = SkeletonLayout(yc)
        rows = conn.execute(
            "SELECT timestamp_s, state FROM tracking_results WHERE run_id = ? AND is_smoothed = 1 ORDER BY tracker_step",
            (tracking_run_id,),
        ).fetchall()
        self.times = np.array([r["timestamp_s"] for r in rows])
        self.blobs = [bytes(r["state"]) for r in rows]
        self.states = states
        self.attach = attach
        self.time_tol = time_tol
        self._transforms: dict[float, dict | None] = {}
        self._world: dict[float, dict] = {}

    def _transforms_at(self, t: float):
        key = round(t, 3)
        if key not in self._transforms:
            i = int(np.argmin(np.abs(self.times - t)))
            if abs(self.times[i] - t) > self.time_tol:
                self._transforms[key] = None
            else:
                dec = self.layout.decode_state_blob(self.blobs[i])
                self._transforms[key] = self.layout.compute_joint_transforms(dec)
        return self._transforms[key]

    def predicted_world(self, t: float) -> dict[str, np.ndarray]:
        key = round(t, 3)
        if key not in self._world:
            T = self._transforms_at(t)
            world = {}
            if T is not None:
                for name, (parent, off, _nrm) in self.attach.items():
                    Tp = T.get(parent)
                    if Tp is not None:
                        world[name] = Tp[:3, :3] @ np.array(off) + Tp[:3, 3]
            self._world[key] = world
        return self._world[key]

    def predicted_in_camera(self, t: float, cam_id: str) -> dict[str, np.ndarray]:
        world = self.predicted_world(t)
        if not world or cam_id not in self.states:
            return {}
        names = list(world)
        proj = _project_raw(np.array([world[n] for n in names]), self.states[cam_id])
        return dict(zip(names, proj))


def collect_tracklets(conn, detection_run_id: str, svid: str, frame_lo: int, frame_hi: int) -> dict[int, dict[int, tuple[float, float]]]:
    """{tracklet_id: {video_frame: (px, py)}}"""
    out: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    rows = conn.execute(
        "SELECT video_frame, keypoints FROM detection_keypoints WHERE detection_run_id = ? AND shot_video_id = ? "
        "AND video_frame BETWEEN ? AND ? AND region_type = 'dots'",
        (detection_run_id, svid, frame_lo, frame_hi),
    ).fetchall()
    for row in rows:
        arr = decode_dot_candidates(bytes(row["keypoints"]))
        for cand in arr:
            tid = int(cand[8])
            if tid < 0:
                continue
            out[tid][row["video_frame"]] = (float(cand[0]), float(cand[1]))
    return out


def prefilter(
    tracklets_by_cam: dict[str, dict[int, dict[int, tuple[float, float]]]],
    sync_table, svid_by_cam: dict[str, str], fkp: FKPredictor,
    min_lifetime: int, proximity_px: float, min_proximity_fraction: float = 0.0,
) -> dict[str, dict[int, dict[int, tuple[float, float]]]]:
    """min_proximity_fraction=0.0 (default) keeps the original "at least one
    frame near some slot" gate. Raise it to require that fraction of the
    tracklet's *own* frames to be near a prediction -- a real, found-2026-
    09-11 gap in the single-frame version: a long-lived static glare
    source only needs to be near *any* moving slot's prediction on *one*
    of its many frames to pass, which a multi-second window of real body
    motion makes easy by chance (confirmed: gopro13_01's tracklets are
    individually stable, but including it in B2 with the any-frame gate
    made both recall and precision worse, adding zero genuine new links)."""
    kept: dict[str, dict[int, dict[int, tuple[float, float]]]] = {}
    for cam_id, tracklets in tracklets_by_cam.items():
        svid = svid_by_cam[cam_id]
        kept[cam_id] = {}
        for tid, frames in tracklets.items():
            if len(frames) < min_lifetime:
                continue
            n_near = 0
            for vf, (px, py) in frames.items():
                t = sync_table.frame_to_global_time(vf, svid)
                if t is None:
                    continue
                pred = fkp.predicted_in_camera(t, cam_id)
                for (sx, sy) in pred.values():
                    if (sx - px) ** 2 + (sy - py) ** 2 <= proximity_px ** 2:
                        n_near += 1
                        break
            if n_near >= max(1, min_proximity_fraction * len(frames)):
                kept[cam_id][tid] = frames
    return kept


def count_shared_frames(frames_a: dict, frames_b: dict, sync_table, svid_a: str, svid_b: str) -> int:
    n = 0
    for vf_a in frames_a:
        t = sync_table.frame_to_global_time(vf_a, svid_a)
        if t is None:
            continue
        vf_b = sync_table.lookup(t, svid_b)
        if vf_b is not None and vf_b in frames_b:
            n += 1
    return n


def pairwise_stats(
    frames_a: dict[int, tuple[float, float]], frames_b: dict[int, tuple[float, float]],
    sync_table, svid_a: str, svid_b: str, state_a, state_b, min_shared: int,
) -> dict | None:
    shared_a, shared_b = [], []
    for vf_a, xy in frames_a.items():
        t = sync_table.frame_to_global_time(vf_a, svid_a)
        if t is None:
            continue
        vf_b = sync_table.lookup(t, svid_b)
        if vf_b is None or vf_b not in frames_b:
            continue
        shared_a.append(xy)
        shared_b.append(frames_b[vf_b])
    if len(shared_a) < min_shared:
        return None
    pa = _undistort_pts(np.array(shared_a), state_a)
    pb = _undistort_pts(np.array(shared_b), state_b)
    P_a, P_b = _proj_matrix(state_a), _proj_matrix(state_b)
    pts4d = cv2.triangulatePoints(P_a, P_b, pa.T, pb.T)
    w = pts4d[3]
    valid = np.abs(w) > 1e-8
    pts3d = np.where(valid, pts4d[:3] / np.where(valid, w, 1), np.nan).T

    def _reproj(P, pts3d_, obs):
        homog = P @ np.hstack([pts3d_, np.ones((len(pts3d_), 1))]).T
        proj = (homog[:2] / homog[2]).T
        return np.linalg.norm(proj - obs, axis=1)

    err = np.maximum(_reproj(P_a, pts3d, pa), _reproj(P_b, pts3d, pb))
    err = err[np.isfinite(err)]
    if len(err) < min_shared:
        return None
    return {"n": int(len(err)), "median": float(np.median(err)), "p90": float(np.percentile(err, 90))}


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_groups(
    tracklets_by_cam: dict, states: dict, sync_table, svid_by_cam: dict,
    min_shared: int, median_thresh: float, p90_thresh: float, disambiguation_margin: float,
):
    """Returns (groups: list[dict], pair_stats: dict[(key_a,key_b), stats])."""
    cam_ids = list(tracklets_by_cam)
    pair_stats: dict[tuple, dict] = {}
    for i, cam_a in enumerate(cam_ids):
        for cam_b in cam_ids[i + 1:]:
            for tid_a, frames_a in tracklets_by_cam[cam_a].items():
                for tid_b, frames_b in tracklets_by_cam[cam_b].items():
                    stats = pairwise_stats(
                        frames_a, frames_b, sync_table, svid_by_cam[cam_a], svid_by_cam[cam_b],
                        states[cam_a], states[cam_b], min_shared,
                    )
                    if stats is not None:
                        key = tuple(sorted([(cam_a, tid_a), (cam_b, tid_b)]))
                        pair_stats[key] = stats

    accepted = {k: v for k, v in pair_stats.items() if v["median"] <= median_thresh and v["p90"] <= p90_thresh}

    # disambiguate: for each node, per neighboring camera, keep only a clear winner
    by_node: dict[tuple, list[tuple]] = defaultdict(list)
    for (a, b) in accepted:
        by_node[a].append(b)
        by_node[b].append(a)

    final_edges: set[tuple] = set()
    ambiguous_nodes: set[tuple] = set()
    for node, neighbors in by_node.items():
        by_cam: dict[str, list[tuple]] = defaultdict(list)
        for n in neighbors:
            by_cam[n[0]].append(n)
        for cam, cands in by_cam.items():
            if len(cands) == 1:
                edge = tuple(sorted([node, cands[0]]))
                final_edges.add(edge)
                continue
            scored = sorted(cands, key=lambda n: accepted[tuple(sorted([node, n]))]["median"])
            best, runner_up = scored[0], scored[1]
            best_med = accepted[tuple(sorted([node, best]))]["median"]
            runner_med = accepted[tuple(sorted([node, runner_up]))]["median"]
            if runner_med >= disambiguation_margin * best_med:
                final_edges.add(tuple(sorted([node, best])))
            else:
                ambiguous_nodes.add(node)
                for n in cands:
                    ambiguous_nodes.add(n)

    all_nodes = {n for cam, tds in tracklets_by_cam.items() for n in [(cam, tid) for tid in tds]}
    uf = UnionFind(all_nodes)
    for a, b in final_edges:
        uf.union(a, b)

    components: dict[tuple, list[tuple]] = defaultdict(list)
    for n in all_nodes:
        components[uf.find(n)].append(n)

    groups = []
    for members in components.values():
        if len(members) < 2:
            continue
        flagged = False
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                key = tuple(sorted([members[i], members[j]]))
                if key in pair_stats and key not in accepted:
                    flagged = True
        groups.append({
            "members": sorted(members),
            "flagged_contradiction": flagged,
            "ambiguous": any(m in ambiguous_nodes for m in members),
        })
    return groups, pair_stats


def derive_gt_tracklet_slots(
    gt_frames: list, conn, detection_run_id: str, label_to_cam: dict, svid_by_cam: dict, match_radius_px: float,
) -> dict[tuple, str]:
    """{(cam_id, tracklet_id): majority slot name} from a slot-labelled GT
    file, by matching each labelled point back to its real tracklet_id in
    *detection_run_id* -- free cross-camera tracklet-group ground truth,
    no extra labeling."""
    votes: dict[tuple, Counter] = defaultdict(Counter)
    for e in gt_frames:
        pts = e.get("points", [])
        pm = e.get("point_meta", [])
        if not pts:
            continue
        cam_id = label_to_cam.get(e["camera_label"])
        if cam_id is None:
            continue
        svid = svid_by_cam.get(cam_id)
        if svid is None:
            continue
        row = conn.execute(
            "SELECT keypoints FROM detection_keypoints WHERE detection_run_id = ? AND shot_video_id = ? "
            "AND video_frame = ? AND region_type = 'dots'",
            (detection_run_id, svid, e["video_frame"]),
        ).fetchone()
        if row is None:
            continue
        arr = decode_dot_candidates(bytes(row["keypoints"]))
        if arr.shape[0] == 0:
            continue
        for k, (x, y) in enumerate(pts):
            slot = pm[k].get("slot") if k < len(pm) else None
            if not slot or slot == _UNLABELED:
                continue
            d = np.hypot(arr[:, 0] - x, arr[:, 1] - y)
            j = int(np.argmin(d))
            if d[j] > match_radius_px:
                continue
            tid = int(arr[j, 8])
            if tid < 0:
                continue
            votes[(cam_id, tid)][slot] += 1
    return {k: c.most_common(1)[0][0] for k, c in votes.items()}


def validate(
    groups: list[dict], gt_slot_by_node: dict[tuple, str], tracklets_by_cam: dict,
    sync_table, svid_by_cam: dict, min_shared: int,
) -> None:
    node_to_group = {}
    for gi, g in enumerate(groups):
        for m in g["members"]:
            node_to_group[tuple(m)] = gi

    # Only nodes whose tracklet actually exists in the tested window are a fair
    # test -- the same physical slot gets a fresh tracklet_id after every
    # occlusion/re-acquisition, so a GT node from outside the window is simply
    # absent from B2's candidate pool, not a real miss.
    window_nodes = {(cam, tid) for cam, tracklets in tracklets_by_cam.items() for tid in tracklets}
    gt_nodes = [n for n in gt_slot_by_node if n in window_nodes]

    n_pos = n_pos_testable = n_pos_hit = n_neg_pairs_in_groups = 0
    for i in range(len(gt_nodes)):
        for j in range(i + 1, len(gt_nodes)):
            a, b = gt_nodes[i], gt_nodes[j]
            if a[0] == b[0]:
                continue  # same camera -- not this algorithm's job (see design doc note)
            same_slot = gt_slot_by_node[a] == gt_slot_by_node[b]
            same_group = node_to_group.get(a) is not None and node_to_group.get(a) == node_to_group.get(b)
            if same_slot:
                n_pos += 1
                # A same-slot pair with zero shared frames (consecutive tracklet
                # FRAGMENTS of one physical marker across an occlusion gap, not a
                # simultaneous sighting) genuinely can't be linked by a
                # simultaneous-reprojection test -- not counting it isn't a real
                # miss, it's asking the wrong mechanism to solve a different
                # problem (temporal fragment-stitching, out of B2's scope).
                n_shared = count_shared_frames(
                    tracklets_by_cam[a[0]][a[1]], tracklets_by_cam[b[0]][b[1]],
                    sync_table, svid_by_cam[a[0]], svid_by_cam[b[0]],
                )
                if n_shared >= min_shared:
                    n_pos_testable += 1
                    if same_group:
                        n_pos_hit += 1
            elif same_group:
                n_neg_pairs_in_groups += 1

    recall = n_pos_hit / n_pos_testable if n_pos_testable else float("nan")
    grouped_pairs_with_gt = n_pos_hit + n_neg_pairs_in_groups
    precision = n_pos_hit / grouped_pairs_with_gt if grouped_pairs_with_gt else float("nan")
    print(f"\n=== validation against GT-derived tracklet identity ===")
    print(f"GT cross-camera same-slot pairs: {n_pos} total, {n_pos_testable} actually share "
          f">={min_shared} frames (nodes with GT slot: {len(gt_nodes)})")
    print(f"  recall    (testable same-slot pairs grouped together): {recall:.2f}  ({n_pos_hit}/{n_pos_testable})")
    print(f"  precision (grouped pairs that really are same-slot)  : {precision:.2f}  "
          f"({n_pos_hit}/{grouped_pairs_with_gt}, {n_neg_pairs_in_groups} false)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--tracking-run", required=True)
    ap.add_argument("--detection-run", required=True)
    ap.add_argument("--start-time", type=float, required=True)
    ap.add_argument("--end-time", type=float, required=True)
    ap.add_argument("--min-lifetime", type=int, default=15)
    ap.add_argument("--proximity-px", type=float, default=80.0)
    ap.add_argument("--min-proximity-fraction", type=float, default=0.0,
                     help="Require this fraction of a tracklet's own frames to be near a "
                          "predicted slot, not just any one (see prefilter()'s docstring).")
    ap.add_argument("--min-shared-frames", type=int, default=5)
    ap.add_argument("--median-thresh-px", type=float, default=3.0)
    ap.add_argument("--p90-thresh-px", type=float, default=6.0)
    ap.add_argument("--disambiguation-margin", type=float, default=1.5)
    ap.add_argument("--camera-label", action="append", default=None,
                     help="Restrict to these camera labels (repeatable). Default: every calibrated camera.")
    ap.add_argument("--validate-against-gt", default=None, help="A label_marker_slots_gui.py labels file.")
    ap.add_argument("--gt-match-radius-px", type=float, default=5.0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    states = load_camera_states(conn, args.shot_id)
    sync_table, svid_by_cam = load_sync_table(conn, args.shot_id)
    label_to_cam = {r["label"]: r["id"] for r in conn.execute("SELECT id, label FROM camera_instances")}
    cam_to_label = {v: k for k, v in label_to_cam.items()}

    fkp = FKPredictor(conn, args.tracking_run, states, _DEFAULT_TRIAL)

    wanted_cams = None
    if args.camera_label:
        wanted_cams = {label_to_cam[label] for label in args.camera_label}

    tracklets_by_cam = {}
    for cam_id, svid in svid_by_cam.items():
        if cam_id not in states:
            continue
        if wanted_cams is not None and cam_id not in wanted_cams:
            continue
        lo = sync_table.lookup(args.start_time, svid)
        hi = sync_table.lookup(args.end_time, svid)
        if lo is None or hi is None:
            continue
        tracklets_by_cam[cam_id] = collect_tracklets(conn, args.detection_run, svid, lo, hi)
    print("raw tracklets per camera:", {cam_to_label[c]: len(t) for c, t in tracklets_by_cam.items()})

    filtered = prefilter(tracklets_by_cam, sync_table, svid_by_cam, fkp, args.min_lifetime, args.proximity_px,
                         args.min_proximity_fraction)
    print("pre-filtered tracklets per camera:", {cam_to_label[c]: len(t) for c, t in filtered.items()})

    groups, pair_stats = build_groups(
        filtered, states, sync_table, svid_by_cam,
        args.min_shared_frames, args.median_thresh_px, args.p90_thresh_px, args.disambiguation_margin,
    )
    n_flagged = sum(g["flagged_contradiction"] for g in groups)
    n_ambiguous = sum(g["ambiguous"] for g in groups)
    print(f"\n{len(groups)} tracklet groups (>=2 members); {n_flagged} flagged (internal contradiction), "
          f"{n_ambiguous} touch an ambiguous node")
    for g in sorted(groups, key=lambda g: -len(g["members"]))[:15]:
        members = ", ".join(f"{cam_to_label[c]}#{t}" for c, t in g["members"])
        flags = ("  [CONTRADICTION]" if g["flagged_contradiction"] else "") + ("  [ambiguous]" if g["ambiguous"] else "")
        print(f"  {members}{flags}")

    if args.validate_against_gt:
        gt_frames = json.loads(Path(args.validate_against_gt).read_text())
        gt_slots = derive_gt_tracklet_slots(gt_frames, conn, args.detection_run, label_to_cam, svid_by_cam, args.gt_match_radius_px)
        validate(groups, gt_slots, tracklets_by_cam, sync_table, svid_by_cam, args.min_shared_frames)

    if args.output:
        out = {
            "meta": {
                "session": args.session, "shot_id": args.shot_id, "tracking_run": args.tracking_run,
                "detection_run": args.detection_run, "start_time": args.start_time, "end_time": args.end_time,
            },
            "groups": [
                {"group_id": f"group_{i}", "members": [[cam_to_label[c], t] for c, t in g["members"]],
                 "flagged_contradiction": g["flagged_contradiction"], "ambiguous": g["ambiguous"]}
                for i, g in enumerate(groups)
            ],
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
