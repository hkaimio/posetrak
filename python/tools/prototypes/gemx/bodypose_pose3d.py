# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Measurements C and D: Body Pose joints in the world against the operator-edited runs, and body shape.

C. Body Pose joints (camera frame) are moved to the world with the extrinsics of the trial's calibration and compared
with the same joints of the edited tracking runs (skeleton joints mapped to the SOMA joints they correspond to).
Distances are per crop (one camera), and for the fused pose (per-joint median over the cameras of a frame time).
Because SOMA joints and the skeleton's joints are defined differently (the knee and ankle sit several cm apart), each
distance is also given after removing a constant per-subject, per-joint offset in the skeleton joint's own frame
(median over the trial, the role the tracker's calibrated marker offsets play).
D. Bone lengths of the Body Pose joints against the edited runs' skeleton bone lengths.

    python bodypose_pose3d.py --db session.db --root kp_eval_set_v1 --bodypose out --out pose3d.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bodypose_common as bc  # noqa: E402
from posetrak.markers.bones import BodyPose as RunPose  # noqa: E402

EDITED = {"80d7fab3": None, "47cfeb6c": None, "6972748b": None}   # run -> subject, found from the root positions
MAP = {0: "hips", 1: "spine1", 3: "spine2", 4: "neck1", 5: "neck2", 6: "head",
       11: "shoulder.L", 12: "upper_arm.L", 13: "forearm.L", 14: "hand.L",
       39: "shoulder.R", 40: "upper_arm.R", 41: "forearm.R", 42: "hand.R",
       67: "thigh.L", 68: "shin.L", 69: "foot.L", 70: "toe.L",
       72: "thigh.R", 73: "shin.R", 74: "foot.R", 75: "toe.R"}
JOINTS = list(MAP)
GROUP = {"trunk": [0, 1, 3], "head": [4, 5, 6], "arms": [11, 12, 13, 14, 39, 40, 41, 42], "legs": [67, 68, 69, 70, 72, 73, 74, 75]}
BONES = {"clavicle": (11, 12), "upper_arm": (12, 13), "forearm": (13, 14), "thigh": (67, 68), "shin": (68, 69),
         "foot": (69, 70), "pelvis_spine": (0, 1), "spine": (1, 3), "chest_neck": (3, 4), "neck": (4, 5), "head": (5, 6),
         "hip_width": (67, 72), "shoulder_width": (12, 40)}
BONE_SKEL = {"clavicle": ("shoulder", "upper_arm"), "upper_arm": ("upper_arm", "forearm"), "forearm": ("forearm", "hand"),
             "thigh": ("thigh", "shin"), "shin": ("shin", "foot"), "foot": ("foot", "toe"), "pelvis_spine": ("hips", "spine1"),
             "spine": ("spine1", "spine2"), "chest_neck": ("spine2", "neck1"), "neck": ("neck1", "neck2"), "head": ("neck2", "head")}
RANGES = {"whole": (43.5, 66.44), "range1": (43.5, 47.5), "range3": (51.7, 56.8)}


class Edited:
    """Mapped joint positions and frames of an edited run, cached per tracker state."""

    def __init__(self, conn, run):
        rid = conn.execute("SELECT id FROM tracking_runs WHERE id LIKE ?", (run + "%",)).fetchone()[0]
        self.pose = RunPose(conn, rid)
        self.smoothed = len(self.pose.times) > 0
        if not self.smoothed:   # a run without smoothed results: use the forward pass
            rows = conn.execute("SELECT timestamp_s, state FROM tracking_results WHERE run_id = ? AND is_smoothed = 0 "
                                "AND tracking_lost = 0 ORDER BY tracker_step", (rid,)).fetchall()
            self.pose.times = np.array([r["timestamp_s"] for r in rows])
            self.pose._blobs = [bytes(r["state"]) for r in rows]
        self.cache = {}

    def at(self, t):
        times = self.pose.times
        i = int(np.argmin(np.abs(times - t)))
        if abs(times[i] - t) > 0.02:
            return None
        if i not in self.cache:
            frames = self.pose._blobs[i]
            fr = self.pose.layout.compute_joint_transforms(self.pose.layout.decode_state_blob(frames))
            self.cache[i] = fr
        return self.cache[i]

    def joints(self, t):
        fr = self.at(t)
        if fr is None:
            return None
        names = [MAP[j] for j in JOINTS]
        return (np.array([fr[n][:3, 3] for n in names]), np.array([fr[n][:3, :3] for n in names]))


def to_world(joints_cam, state):
    R, t = np.asarray(state.R), np.asarray(state.t).reshape(3)
    return (joints_cam - t) @ R   # R^T (x - t), rows as points


def stats(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)), "p90": float(np.percentile(x, 90))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--bodypose", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    conn = bc.open_db(a.db)
    states = bc.camera_states(conn)
    cams = [str(c) for c in np.load(Path(a.root) / "range3" / "candidates_subject1.npz", allow_pickle=True)["cams"]]
    bp = bc.BodyPose(a.bodypose)
    J = bp.fields["joints_camera"].astype(np.float64)[:, JOINTS]
    world = np.empty_like(J)
    for c in range(5):
        m = bp.cam == c
        world[m] = to_world(J[m], states[cams[c]])
    events = json.load(open(Path(a.root) / "events.json"))

    edited = {r: Edited(conn, r) for r in EDITED}
    # subject of each edited run: the BP subject whose hips are closest to the run's hips at a few times
    subject_of = {}
    for r, e in edited.items():
        best = {}
        for t in np.arange(45.0, 65.0, 2.5):
            j = e.joints(t)
            if j is None:
                continue
            for s in (1, 2, 3):
                m = (bp.subject == s) & (np.abs(bp.time - t) < 0.01)
                if m.any():
                    best.setdefault(s, []).append(np.median(np.linalg.norm(world[m][:, 0] - j[0][0], axis=1)))
        subject_of[r] = min(best, key=lambda s: np.median(best[s]))
        print("edited run", r, "-> subject", subject_of[r], {s: round(float(np.median(v)), 2) for s, v in best.items()}, flush=True)
    assert len(set(subject_of.values())) == 3, subject_of

    # per crop: distance of each mapped joint to the edited run
    N = len(bp.time)
    sk_pos = np.full((N, len(JOINTS), 3), np.nan)
    sk_rot = np.full((N, len(JOINTS), 3, 3), np.nan)
    for r, e in edited.items():
        s = subject_of[r]
        for i in np.flatnonzero(bp.subject == s):
            j = e.joints(bp.time[i])
            if j is not None:
                sk_pos[i], sk_rot[i] = j
    ok = np.isfinite(sk_pos[:, 0, 0])
    print("crops with an edited state", int(ok.sum()), "of", N, flush=True)
    diff = world - sk_pos
    # constant per-subject, per-joint offset in the skeleton joint frame
    local = np.einsum("nkji,nkj->nki", sk_rot, diff)          # R^T diff
    offsets = {}
    calibrated = np.full_like(diff, np.nan)
    for s in (1, 2, 3):
        m = ok & (bp.subject == s)
        offsets[s] = np.median(local[m], axis=0)
        calibrated[m] = diff[m] - np.einsum("nkij,kj->nki", sk_rot[m], offsets[s])
    # root aligned: BP translated so that its hips coincide with the run's hips
    aligned = diff - diff[:, :1]
    aligned_cal = calibrated - calibrated[:, :1]

    def per_crop(d):
        return 100 * np.linalg.norm(d, axis=-1)            # [N, joints] cm

    def mean_body(d, idx):
        cols = [JOINTS.index(j) for j in idx]
        return per_crop(d)[:, cols].mean(1)

    all_idx = JOINTS
    kinds = {"absolute": diff, "absolute_offset_removed": calibrated, "root_aligned": aligned,
             "root_aligned_offset_removed": aligned_cal}
    t = bp.time

    def inside(subject_arr, time_arr):
        inn = np.zeros(len(time_arr), bool)
        for e in events:
            inn |= (subject_arr == e["subject"]) & (time_arr >= e["t0"]) & (time_arr <= e["t1"])
        return inn

    ev_in = inside(bp.subject, t)
    sel_sets = {}
    for rn, (lo, hi) in RANGES.items():
        r_ok = ok & (t >= lo) & (t <= hi)
        sel_sets[rn] = {"all": r_ok, "inside_events": r_ok & ev_in, "outside_events": r_ok & ~ev_in}
    result = {"edited_run_subject": subject_of, "edited_run_smoothed": {r: e.smoothed for r, e in edited.items()},
              "edited_run_steps": {r: int(len(e.pose.times)) for r, e in edited.items()}, "offsets_cm": {str(s): {MAP[j]: [round(100 * float(v), 1) for v in offsets[s][k]]
                                                                         for k, j in enumerate(JOINTS)} for s in offsets}}
    out_c = {}
    for kind, d in kinds.items():
        mb = mean_body(d, all_idx)
        out_c[kind] = {}
        for rn, sets in sel_sets.items():
            out_c[kind][rn] = {}
            for sn, m in sets.items():
                entry = {"per_crop_mean_joint_distance_cm": stats(mb[m]),
                         "per_camera_mean_cm": {str(c): stats(mb[m & (bp.cam == c)]).get("mean") for c in range(5)},
                         "per_subject_mean_cm": {str(s): stats(mb[m & (bp.subject == s)]).get("mean") for s in (1, 2, 3)},
                         "hips_only_cm": stats(per_crop(d)[:, 0][m]),
                         "per_part_cm": {g: stats(mean_body(d, idx)[m]).get("mean") for g, idx in GROUP.items()}}
                out_c[kind][rn][sn] = entry
    result["per_camera_crops"] = out_c

    # fused: per-joint median over the cameras of one frame time and subject
    tb = np.round(t * 120).astype(np.int64)
    fused = {}
    key = tb * 8 + bp.subject
    for k in np.unique(key[ok]):
        m = (key == k) & ok
        if len(np.unique(bp.cam[m])) < 3:
            continue
        fused[k] = (np.median(world[m], axis=0), sk_pos[m][0], sk_rot[m][0], int(bp.subject[m][0]), float(t[m].mean()))
    ks = sorted(fused)
    F_w = np.array([fused[k][0] for k in ks]); F_s = np.array([fused[k][1] for k in ks])
    F_r = np.array([fused[k][2] for k in ks]); F_sub = np.array([fused[k][3] for k in ks]); F_t = np.array([fused[k][4] for k in ks])
    Fd = F_w - F_s
    Fcal = Fd.copy()
    for s in (1, 2, 3):
        m = F_sub == s
        Fcal[m] = Fd[m] - np.einsum("nkij,kj->nki", F_r[m], offsets[s])
    f_in = inside(F_sub, F_t)
    out_f = {}
    for kind, d in {"absolute": Fd, "absolute_offset_removed": Fcal, "root_aligned": Fd - Fd[:, :1],
                    "root_aligned_offset_removed": Fcal - Fcal[:, :1]}.items():
        mb = (100 * np.linalg.norm(d, axis=-1)).mean(1)
        out_f[kind] = {}
        for rn, (lo, hi) in RANGES.items():
            r_ok = (F_t >= lo) & (F_t <= hi)
            out_f[kind][rn] = {"all": stats(mb[r_ok]), "inside_events": stats(mb[r_ok & f_in]), "outside_events": stats(mb[r_ok & ~f_in]),
                               "hips_only_cm": stats(100 * np.linalg.norm(d[:, 0], axis=-1)[r_ok]),
                               "per_subject": {str(s): stats(mb[r_ok & (F_sub == s)]).get("mean") for s in (1, 2, 3)}}
    result["fused_median_over_cameras"] = out_f

    # fused by triangulation: the Body Pose 2D keypoints (undistorted pixels) of >= 3 cameras, DLT per joint
    P = [st.K @ np.hstack([np.asarray(st.R, np.float64), np.asarray(st.t, np.float64).reshape(3, 1)]) for st in (states[c] for c in cams)]
    kp2 = bp.fields["keypoints_2d"].astype(np.float64)[:, JOINTS]
    tri_w, tri_s, tri_r, tri_sub, tri_t = [], [], [], [], []
    for k in ks:
        m = np.flatnonzero((key == k) & ok)
        if len(np.unique(bp.cam[m])) < 3:
            continue
        M = np.zeros((len(JOINTS), 4, 4))
        for i in m:
            Pc = P[bp.cam[i]]
            a1 = kp2[i][:, :1] * Pc[2] - Pc[0]
            a2 = kp2[i][:, 1:] * Pc[2] - Pc[1]
            a1 /= np.linalg.norm(a1[:, :3], axis=1, keepdims=True)
            a2 /= np.linalg.norm(a2[:, :3], axis=1, keepdims=True)
            M += a1[:, :, None] * a1[:, None, :] + a2[:, :, None] * a2[:, None, :]
        Xh = np.linalg.eigh(M)[1][:, :, 0]
        tri_w.append(Xh[:, :3] / Xh[:, 3:4]); tri_s.append(sk_pos[m[0]]); tri_r.append(sk_rot[m[0]])
        tri_sub.append(int(bp.subject[m[0]])); tri_t.append(float(t[m].mean()))
    T_w, T_s, T_r, T_sub, T_t = (np.array(v) for v in (tri_w, tri_s, tri_r, tri_sub, tri_t))
    Td = T_w - T_s
    Tcal = Td.copy()
    for s_ in (1, 2, 3):
        m = T_sub == s_
        Tcal[m] = Td[m] - np.einsum("nkij,kj->nki", T_r[m], offsets[s_])
    t_in = inside(T_sub, T_t)
    out_t = {}
    for kind, d in {"absolute": Td, "absolute_offset_removed": Tcal, "root_aligned": Td - Td[:, :1],
                    "root_aligned_offset_removed": Tcal - Tcal[:, :1]}.items():
        mb = (100 * np.linalg.norm(d, axis=-1)).mean(1)
        out_t[kind] = {}
        for rn, (lo, hi) in RANGES.items():
            r_ok = (T_t >= lo) & (T_t <= hi)
            out_t[kind][rn] = {"all": stats(mb[r_ok]), "inside_events": stats(mb[r_ok & t_in]), "outside_events": stats(mb[r_ok & ~t_in]),
                               "hips_only_cm": stats(100 * np.linalg.norm(d[:, 0], axis=-1)[r_ok]),
                               "per_part_cm": {g: float((100 * np.linalg.norm(d[:, [JOINTS.index(j) for j in idx]], axis=-1)).mean(1)[r_ok].mean()) for g, idx in GROUP.items()},
                               "share_within_5cm": float((mb[r_ok] < 5).mean()), "share_within_10cm": float((mb[r_ok] < 10).mean())}
    result["fused_triangulated_2d"] = out_t

    # D: body shape
    shape = {}
    for s in (1, 2, 3):
        m = bp.subject == s
        run = [r for r, ss in subject_of.items() if ss == s][0]
        e = edited[run]
        fr = e.at(55.0) or e.at(50.0)
        d = {}
        for name, (p, c) in BONES.items():
            length = 100 * np.linalg.norm(J[m][:, JOINTS.index(c)] - J[m][:, JOINTS.index(p)], axis=1)
            entry = {"bodypose_median_cm": float(np.median(length)), "iqr_cm": [float(np.percentile(length, 25)), float(np.percentile(length, 75))],
                     "per_camera_median_cm": [float(np.median(length[bp.cam[m] == k])) for k in range(5)]}
            if name in BONE_SKEL:
                a_, b_ = BONE_SKEL[name]
                entry["skeleton_cm"] = float(100 * np.linalg.norm(fr[b_ + ".L" if b_ + ".L" in fr else b_][:3, 3]
                                                                   - fr[a_ + ".L" if a_ + ".L" in fr else a_][:3, 3]))
            elif name == "hip_width":
                entry["skeleton_cm"] = float(100 * np.linalg.norm(fr["thigh.L"][:3, 3] - fr["thigh.R"][:3, 3]))
            elif name == "shoulder_width":
                entry["skeleton_cm"] = float(100 * np.linalg.norm(fr["upper_arm.L"][:3, 3] - fr["upper_arm.R"][:3, 3]))
            d[name] = entry
        shape[str(s)] = d
    result["bone_lengths"] = shape
    Path(a.out).write_text(json.dumps(result, indent=1))
    for kind in out_c:
        w = out_c[kind]["whole"]
        print(kind, "per-crop whole:", {k: round(v["per_crop_mean_joint_distance_cm"].get("mean", np.nan), 1) for k, v in w.items()},
              "| fused whole:", {k: round(v.get("mean", np.nan), 1) for k, v in out_f[kind]["whole"].items() if k in ("all", "inside_events", "outside_events")})


if __name__ == "__main__":
    main()
