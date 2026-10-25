# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""grouping.py — group the dot tracklets of several cameras into physical markers.

A reflective marker seen by two cameras gives each one tracklet. Two tracklets
belong together when, over the frames both have, the rays through their points
meet: the triangulated point reprojects within a few pixels in both cameras. The
test is applied to a tracklet's whole shared lifetime, not to one lucky frame.

Pairs that pass are then disambiguated (a tracklet with several equally good
partners in one other camera is ambiguous and dropped) and connected into groups.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from app.setup.extrinsics_solver import _proj_matrix, _undistort_pts


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
