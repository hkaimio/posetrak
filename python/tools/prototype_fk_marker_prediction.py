# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prototype_fk_marker_prediction.py — PROTOTYPE P-A of
marker-catalog-and-assignment-redesign.md §8.

The keystone experiment: does an FK-predicted marker position, from a
hand-authored *nominal* attachment set + an existing markerless tracking
result, land close enough to the actual detected dots to drive
clustering, and does the surface-normal visibility prior agree with
reality?

Pipeline, per output frame:
    1. tracker global time -> nearest smoothed `tracking_results.state`
    2. SkeletonLayout FK -> every joint's 4x4 world transform
    3. per trial marker: world pos = T_parent @ offset_local;
       world normal = R_parent @ normal_local
    4. per camera: project world pos -> raw distorted pixels
       (cv2.projectPoints); facing = dot(normal_world,
       unit(camera_center - marker_world))
    5. draw predicted markers (filled if facing the camera, hollow if
       back-facing) over the actual detected dot candidates

At the end it prints a per-(marker, camera) table: median / p90 pixel
distance from each predicted marker to the nearest detected dot, and the
facing-value distribution -- the actual read-outs P-A exists to produce.

The trial attachment set is hard-coded (`_DEFAULT_TRIAL`) as a
right-leg probe set: the real skeleton anchors plus a ring of probe
markers around the shin just below the knee and near the ankle, in the
parent joint's own local frame, so the video shows which local direction
corresponds to medial / lateral / anterior for this skeleton. Override
with --attachment-set <yaml> once that's known.

Usage:
    python tools/prototype_fk_marker_prediction.py \\
        --session /path/to/session.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --tracking-run 990fb01a-6c72-47bc-bdb7-4d31147d2ef7 \\
        --marker-detection-run 01c2e3c1-204b-49f8-9b14-6fd611b765a4 \\
        --camera-label gopro-11_mini_01 gopro13_01 gopro13_02 insta_ace2_pro oneplus9pro-01 pixel9 \\
        --start-time 40.0 --end-time 50.0 --slow-factor 4 \\
        --video-out scratch/dot_ground_truth/fk_prediction.mp4
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pose.db_cache import decode_dot_candidates  # noqa: E402
from posetrak.db.skeleton_layout import SkeletonLayout  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states, load_sync_table  # noqa: E402

_STAMP_FONT_SCALE = 0.55

# name -> (parent_joint, offset_local[x,y,z] m, normal_local[x,y,z] or None for a pose anchor)
# offset_local is in the parent joint's own local frame; local +Y is down-bone
# (toward the child) for this skeleton's leg joints. Right side only --
# _mirror_to_left() below generates the left-side entries.
_RIGHT_TRIAL: dict[str, tuple[str, list[float], list[float] | None]] = {
    "a_hip_R":   ("thigh.R", [0.0, 0.0, 0.0], None),
    "a_knee_R":  ("shin.R", [0.0, 0.0, 0.0], None),
    "a_ankle_R": ("foot.R", [0.0, 0.0, 0.0], None),
    "a_heel_R":  ("foot.R", [0.0152, 0.00207, 0.0669], None),
    "a_toe_R":   ("foot.R", [0.0312, 0.1691, -0.0063], None),
    # knee probes: ring around the shin ~3 cm below the knee
    "k_Xp_R": ("shin.R", [0.05, 0.03, 0.0], [1.0, 0.0, 0.0]),
    "k_Xm_R": ("shin.R", [-0.05, 0.03, 0.0], [-1.0, 0.0, 0.0]),
    "k_Zp_R": ("shin.R", [0.0, 0.03, 0.05], [0.0, 0.0, 1.0]),
    "k_Zm_R": ("shin.R", [0.0, 0.03, -0.05], [0.0, 0.0, -1.0]),
    # ankle probes: ring around the distal shin (~along 0.42 m)
    "n_Xp_R": ("shin.R", [0.045, 0.42, 0.0], [1.0, 0.0, 0.0]),
    "n_Xm_R": ("shin.R", [-0.045, 0.42, 0.0], [-1.0, 0.0, 0.0]),
    "n_Zp_R": ("shin.R", [0.0, 0.42, 0.045], [0.0, 0.0, 1.0]),
    "n_Zm_R": ("shin.R", [0.0, 0.42, -0.045], [0.0, 0.0, -1.0]),
}


def _mirror_to_left(
    right: dict[str, tuple[str, list[float], list[float] | None]],
) -> dict[str, tuple[str, list[float], list[float] | None]]:
    """Mirror a right-side attachment set to the left side.

    Confirmed empirically against this skeleton's own *rest pose* (zero
    joint angles -- true bind-pose symmetry, not confounded by the current
    animated pose putting each leg in a different position):
    thigh.L/shin.L/foot.L's local axes are related to thigh.R/shin.R/
    foot.R's by reflection across the sagittal (world X=0) plane, composed
    with a local-frame correction so each side's frame stays right-handed
    -- net effect, a local offset or normal vector (x, y, z) on the right
    mirrors to (-x, y, z) on the left (only the local-X, i.e.
    medial/lateral, component flips; local-Y "down-bone" and local-Z stay
    as they are). Real, found the hard way: an earlier version of this
    trial set had *no* left-side entries at all, silently comparing
    left-leg ground truth against right-leg predictions -- see
    eval_fk_prediction.py's status.md entry for the 60-200px errors that
    caught it, versus 11-40px once mirrored.
    """
    out = {}
    for name, (parent, off, nrm) in right.items():
        assert parent.endswith(".R") and name.endswith("_R"), f"{name}/{parent} isn't a right-side entry"
        mirrored_off = [-off[0], off[1], off[2]]
        mirrored_nrm = None if nrm is None else [-nrm[0], nrm[1], nrm[2]]
        out[name[:-2] + "_L"] = (parent[:-2] + ".L", mirrored_off, mirrored_nrm)
    return out


_DEFAULT_TRIAL: dict[str, tuple[str, list[float], list[float] | None]] = {
    **_RIGHT_TRIAL, **_mirror_to_left(_RIGHT_TRIAL),
}

_COLORS: dict[str, tuple[int, int, int]] = {}


def _color(name: str) -> tuple[int, int, int]:
    if name not in _COLORS:
        h = (len(_COLORS) * 0.61803398875) % 1.0
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, 0.9, 1.0)
        _COLORS[name] = (int(b * 255), int(g * 255), int(r * 255))
    return _COLORS[name]


def _load_attachment_set(path: str | None) -> dict[str, tuple[str, np.ndarray, np.ndarray | None]]:
    if path is None:
        raw = _DEFAULT_TRIAL
    else:
        doc = yaml.safe_load(Path(path).read_text())
        raw = {m["name"]: (m["parent_joint"], m["offset"], m.get("normal"))
               for m in doc["markers"]}
    out = {}
    for name, (parent, off, nrm) in raw.items():
        n = None if nrm is None else np.asarray(nrm, float) / (np.linalg.norm(nrm) or 1.0)
        out[name] = (parent, np.asarray(off, float), n)
    return out


def _project_raw(pts_world: np.ndarray, state) -> np.ndarray:
    """world Nx3 -> raw distorted pixels Nx2 for *state*."""
    rvec = cv2.Rodrigues(state.R)[0]
    tvec = state.t.reshape(3, 1)
    obj = pts_world.reshape(-1, 1, 3).astype(np.float64)
    if state.fisheye:
        proj, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, state.K_orig, state.dist)
    else:
        proj, _ = cv2.projectPoints(obj, rvec, tvec, state.K_orig, state.dist)
    return proj.reshape(-1, 2)


def _prepare_cell(img, window, tw, th):
    x0, y0, x1, y1 = window
    crop = img[y0:min(y1, img.shape[0]), x0:min(x1, img.shape[1])]
    ch, cw = crop.shape[:2]
    s = min(tw / cw, th / ch)
    nw, nh = max(1, round(cw * s)), max(1, round(ch * s))
    canvas = np.zeros((th, tw, 3), np.uint8)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(crop, (nw, nh))
    return canvas, (lambda x, y: ((x - x0) * s + ox, (y - y0) * s + oy))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--tracking-run", required=True)
    ap.add_argument("--marker-detection-run", required=True)
    ap.add_argument("--camera-label", required=True, nargs="+")
    ap.add_argument("--start-time", type=float, required=True)
    ap.add_argument("--end-time", type=float, required=True)
    ap.add_argument("--slow-factor", type=float, default=4.0)
    ap.add_argument("--attachment-set", default=None)
    ap.add_argument("--video-out", required=True)
    ap.add_argument("--facing-thresh-deg", type=float, default=100.0,
                     help="predicted marker drawn hollow when the angle between its normal and "
                          "the camera direction exceeds this")
    ap.add_argument("--cell-width", type=int, default=760)
    ap.add_argument("--cell-height", type=int, default=560)
    ap.add_argument("--full-frame", action="store_true",
                     help="Do not crop each cell to the leg region (show the whole frame).")
    ap.add_argument("--crop-pad", type=int, default=180, help="Padding (px) around the leg-region crop.")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    states = load_camera_states(conn, args.shot_id)
    sync_table, svid_by_cam = load_sync_table(conn, args.shot_id)

    sk_id = conn.execute("SELECT skeleton_id FROM tracking_runs WHERE id = ?", (args.tracking_run,)).fetchone()["skeleton_id"]
    yc = conn.execute("SELECT yaml_content FROM skeletons WHERE id = ?", (sk_id,)).fetchone()["yaml_content"]
    layout = SkeletonLayout(yc)

    st_rows = conn.execute(
        "SELECT timestamp_s, state FROM tracking_results WHERE run_id = ? AND is_smoothed = 1 ORDER BY tracker_step",
        (args.tracking_run,),
    ).fetchall()
    st_times = np.array([r["timestamp_s"] for r in st_rows])
    st_blobs = [bytes(r["state"]) for r in st_rows]

    def state_at(t: float):
        i = int(np.argmin(np.abs(st_times - t)))
        if abs(st_times[i] - t) > 0.05:
            return None
        return layout.decode_state_blob(st_blobs[i])

    attach = _load_attachment_set(args.attachment_set)
    cam_center = {c: (-s.R.T @ s.t.reshape(3)) for c, s in states.items()}
    facing_cos = np.cos(np.deg2rad(args.facing_thresh_deg))

    cam_ids, file_paths, native_fps, cam_range = [], {}, {}, {}
    for label in args.camera_label:
        cid = conn.execute("SELECT id FROM camera_instances WHERE label = ?", (label,)).fetchone()["id"]
        cam_ids.append(cid)
        svid = svid_by_cam[cid]
        row = conn.execute("SELECT file_path, actual_fps FROM capture_videos WHERE id = ?", (svid,)).fetchone()
        file_paths[cid] = row["file_path"]
        native_fps[cid] = float(row["actual_fps"])

    ref_cam = cam_ids[0]
    ref_svid = svid_by_cam[ref_cam]
    frame_lo = sync_table.lookup(args.start_time, ref_svid)
    frame_hi = sync_table.lookup(args.end_time, ref_svid)

    for cid in cam_ids:
        svid = svid_by_cam[cid]
        lo = sync_table.lookup(args.start_time, svid) or frame_lo
        hi = sync_table.lookup(args.end_time, svid) or frame_hi
        cam_range[cid] = (lo, hi)

    # Per-camera crop window (unless --full-frame): the padded pixel bbox of every
    # attachment marker's FK projection over the clip, so each cell shows the leg
    # region big enough to read the labels rather than a full 4K frame downscaled
    # to a small grid cell. Aspect ratio is preserved at draw time (_prepare_cell).
    windows = {}
    for cid in cam_ids:
        cap = cv2.VideoCapture(file_paths[cid])
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 3840
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 2160
        cap.release()
        if args.full_frame:
            windows[cid] = (0, 0, W, H)
            continue
        xs, ys = [], []
        for tt in np.linspace(args.start_time, args.end_time, 60):
            dec = state_at(float(tt))
            if dec is None:
                continue
            T = layout.compute_joint_transforms(dec)
            pw = [T[p][:3, :3] @ o + T[p][:3, 3] for p, o, _n in attach.values() if p in T]
            if not pw:
                continue
            proj = _project_raw(np.array(pw), states[cid])
            proj = proj[np.isfinite(proj).all(axis=1)]
            proj = proj[(proj[:, 0] > -W) & (proj[:, 0] < 2 * W) & (proj[:, 1] > -H) & (proj[:, 1] < 2 * H)]
            xs += list(proj[:, 0])
            ys += list(proj[:, 1])
        if xs:
            pad = args.crop_pad
            x0 = max(0, int(min(xs) - pad))
            y0 = max(0, int(min(ys) - pad))
            x1 = min(W, int(max(xs) + pad))
            y1 = min(H, int(max(ys) + pad))
            windows[cid] = (x0, y0, x1, y1)
        else:
            windows[cid] = (0, 0, W, H)

    cols = int(np.ceil(np.sqrt(len(cam_ids))))
    rows = int(np.ceil(len(cam_ids) / cols))
    cw, chh = args.cell_width, args.cell_height

    decoders = {c: iter_frames(file_paths[c], cam_range[c][0], cam_range[c][1] + 1) for c in cam_ids}
    nxt = {c: next(decoders[c]) for c in cam_ids}

    def advance(cid, target):
        fidx, img = nxt[cid]
        try:
            while fidx < target:
                fidx, img = next(decoders[cid])
        except StopIteration:
            pass
        nxt[cid] = (fidx, img)
        return img

    out_path = Path(args.video_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    out_fps = native_fps[ref_cam] / args.slow_factor
    n_written = 0
    # stats[(marker, cam_label)] = list of (nn_dist_px, facing)
    stats: dict[tuple[str, str], list[tuple[float, float]]] = {}

    for ref_frame in range(frame_lo, frame_hi):
        t = sync_table.frame_to_global_time(ref_frame, ref_svid)
        if t is None:
            continue
        dec = state_at(t)
        canvas = np.zeros((rows * chh, cols * cw, 3), np.uint8)
        world_pos, world_nrm = {}, {}
        if dec is not None:
            T = layout.compute_joint_transforms(dec)
            for name, (parent, off, nrm) in attach.items():
                Tp = T.get(parent)
                if Tp is None:
                    continue
                world_pos[name] = Tp[:3, :3] @ off + Tp[:3, 3]
                if nrm is not None:
                    world_nrm[name] = Tp[:3, :3] @ nrm

        for i, cid in enumerate(cam_ids):
            svid = svid_by_cam[cid]
            fidx = sync_table.lookup(t, svid)
            if fidx is None:
                continue
            img = advance(cid, fidx)

            dot_row = conn.execute(
                "SELECT keypoints FROM detection_keypoints WHERE detection_run_id = ? AND shot_video_id = ? "
                "AND video_frame = ? AND region_type = 'dots'",
                (args.marker_detection_run, svid, fidx),
            ).fetchone()
            dots = decode_dot_candidates(bytes(dot_row["keypoints"]))[:, :2] if dot_row else np.zeros((0, 2))

            cell, tf = _prepare_cell(img, windows[cid], cw, chh)
            for dx, dy in dots:
                px, py = tf(dx, dy)
                cv2.circle(cell, (int(px), int(py)), 5, (0, 255, 255), 1)

            if world_pos:
                names = list(world_pos)
                proj = _project_raw(np.array([world_pos[n] for n in names]), states[cid])
                for n, (ux, uy) in zip(names, proj):
                    col = _color(n)
                    px, py = tf(ux, uy)
                    p = (int(round(px)), int(round(py)))
                    is_anchor = attach[n][2] is None
                    facing = None
                    if n in world_nrm:
                        d = cam_center[cid] - world_pos[n]
                        d = d / (np.linalg.norm(d) or 1.0)
                        facing = float(np.dot(world_nrm[n], d))
                    if is_anchor:
                        cv2.drawMarker(cell, p, col, cv2.MARKER_TILTED_CROSS, 12, 2)
                    elif facing is None or facing >= facing_cos:
                        cv2.circle(cell, p, 6, col, -1)
                        cv2.circle(cell, p, 9, col, 1)
                    else:
                        cv2.circle(cell, p, 6, col, 1)  # back-facing -> hollow
                    cv2.putText(cell, n, (p[0] + 8, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, col, 1, cv2.LINE_AA)
                    # stats: nearest detected dot within a gate (so a camera that
                    # barely detects dots reports a low match-rate, not a huge median)
                    nn = np.inf
                    if dots.shape[0]:
                        nn = float(np.hypot(dots[:, 0] - ux, dots[:, 1] - uy).min())
                    stats.setdefault((n, args.camera_label[i]), []).append(
                        (nn, float(facing) if facing is not None else np.nan))

            label = f"{args.camera_label[i]}  f={fidx}  t={t:.3f}s" + ("  [NO STATE]" if dec is None else "")
            cv2.putText(cell, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, _STAMP_FONT_SCALE, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(cell, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, _STAMP_FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)

            r, c = divmod(i, cols)
            canvas[r * chh:(r + 1) * chh, c * cw:(c + 1) * cw] = cell

        if writer is None:
            h, w = canvas.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
        writer.write(canvas)
        n_written += 1

    if writer is not None:
        writer.release()
    print(f"wrote {n_written} frames -> {out_path}\n")

    GATE = 60.0  # px: a predicted marker is "matched" if a detection is within this
    print(f"{'marker':8s} {'camera':18s} {'n':>5s} {'match%':>7s} {'med_matched_px':>15s} {'facing med':>11s}")
    for (mk, cam), vals in sorted(stats.items()):
        d = np.array([v[0] for v in vals])
        f = np.array([v[1] for v in vals])
        matched = d[d <= GATE]
        mr = 100.0 * len(matched) / len(d)
        mm = np.median(matched) if len(matched) else float("nan")
        fm = np.nanmedian(f) if np.isfinite(f).any() else float("nan")
        print(f"{mk:8s} {cam:18s} {len(vals):5d} {mr:6.0f}% {mm:15.1f} {fm:11.2f}")


if __name__ == "__main__":
    main()
