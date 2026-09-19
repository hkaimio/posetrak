# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""cutie_segment_ball_all_cameras.py — extend prototype_ball_cutie_segmentation.py's
single-camera/single-throw proof of concept (2026-09-15, validated on
gopro-11_mini_01/throw 1: continuous mask coverage through the whole
fall-and-bounce, no dropout) to all 6 cameras x all 4 throws.

Unlike the reflective-dot detector, Cutie's segmentation has no ring-light
dependency, so this deliberately does NOT exclude gopro13_01/insta_ace2_pro
the way prototype_ball_tracking.py had to -- all 6 cameras get a real
attempt here.

Seeding: rather than manually eyeballing a clean frame per camera (six
times over, four throws), this reuses the already-validated constant-
velocity UKF ball trajectory (2026-09-15's --seed-position tracking runs,
python/tools/prototypes/prototype_ball_tracking.py's own earlier triangulation before
that) -- pick one real 3D ball position per throw, project it into each
camera's own pixel space, and use that as the SAM2 box prompt center. No
manual per-camera bbox needed.

Writes one CSV per (throw, camera): video_frame, timestamp_s, cx, cy,
mask_area_px. Read-only against the session DB and source videos; this is
still the "does the detection look good" stage, not the DB-writing
finalization (see finalize_ball_cutie_detection.py for that).
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from posetrak.db.load_session import load_cameras_from_session

SESSION_DB = "D:/mocap/2026-09-06-kare-tests/2026-09-06-kare-tests.db"
SESSION_ID = "662854b2-fdaf-4f37-ae8c-30e1458e8f1c"
EXTRINSIC_CALIBRATION_ID = "4815f5ac-116b-4ae6-b619-2d16bb954777"

TRACKER_FPS = 120.0
TRACKER_T0 = 33.620

THROWS = [
    ("throw1", 2816, 2902),
    ("throw2", 3637, 3735),
    ("throw3", 3989, 4074),
    ("throw4", 4304, 4401),
]
_PAD_S = 0.6
INIT_HALF_SIZE_PX = 45.0  # generous box around the projected point for SAM2


def step_to_time(step: int) -> float:
    return TRACKER_T0 + step / TRACKER_FPS


def load_ball_trajectory(csv_path: Path) -> list[tuple[float, np.ndarray]]:
    """[(timestamp_s, xyz)] from a smoothed_root_pose.csv, skipping empty rows."""
    import csv as csv_mod

    out = []
    with open(csv_path) as f:
        for row in csv_mod.DictReader(f):
            if row["pos_x"] == "" or row["pos_x"] is None:
                continue
            out.append((
                float(row["timestamp"]),
                np.array([float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])]),
            ))
    return out


def camera_frame_ref(conn: sqlite3.Connection, camera_instance_id: str) -> tuple[int, float, float]:
    """Return (ref_video_frame, ref_timestamp_s, actual_fps) for a camera --
    anchors a linear timestamp<->video_frame conversion, using this
    project's own recorded sync rather than assuming a fixed frame-0 offset."""
    row = conn.execute(
        "SELECT video_frame, timestamp_s FROM pose_observations"
        " WHERE camera_instance_id=? AND source='body' ORDER BY timestamp_s LIMIT 1",
        (camera_instance_id,),
    ).fetchone()
    fps_row = conn.execute(
        "SELECT actual_fps FROM capture_videos WHERE camera_instance_id=?", (camera_instance_id,)
    ).fetchone()
    return row[0], row[1], fps_row[0]


def time_to_frame(t: float, ref_frame: int, ref_t: float, fps: float) -> int:
    return int(round(ref_frame + (t - ref_t) * fps))


def frame_to_time(frame: int, ref_frame: int, ref_t: float, fps: float) -> float:
    return ref_t + (frame - ref_frame) / fps


def project_point(xyz: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, dist: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(xyz.reshape(1, 1, 3), rvec, t.reshape(3, 1), K, dist)
    return proj.reshape(2)


def build_sam2_init_mask(frame_bgr: np.ndarray, box_xyxy: np.ndarray, device: str, predictor=None):
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    h, w = frame_bgr.shape[:2]
    if predictor is None:
        sam_model = build_sam2_hf("facebook/sam2.1-hiera-base-plus", device=device)
        predictor = SAM2ImagePredictor(sam_model)
    predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    masks, scores, _ = predictor.predict(box=box_xyxy.astype(np.float32), multimask_output=False)
    m = masks[0] > 0.5
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return m.astype(np.uint8), float(scores[0]), predictor


def write_debug_video(
    video_path: str,
    masks: dict[int, dict[str, np.ndarray]],
    start_frame: int,
    end_frame: int,
    out_path: Path,
    fps: float,
    scale: float,
) -> None:
    """Re-reads the source video over [start_frame, end_frame) and writes an
    annotated .mp4: the ball's mask tinted red, its centroid marked, and the
    real video_frame/timestamp burned in -- so a run's segmentation quality
    can actually be eyeballed rather than judged from a CSV of numbers."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for frame_idx in range(start_frame, end_frame):
            ok, img = cap.read()
            if not ok:
                break
            m = masks.get(frame_idx, {}).get("ball")
            if m is not None and m.any():
                overlay = img.copy()
                overlay[m] = (0, 0, 255)
                img = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
                ys, xs = np.nonzero(m)
                cx, cy = int(xs.mean()), int(ys.mean())
                cv2.circle(img, (cx, cy), 8, (0, 255, 0), 2)
                status = "tracked"
            else:
                status = "no mask"
            img = cv2.resize(img, (w, h))
            cv2.putText(img, f"frame {frame_idx}  {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(img)
    finally:
        writer.release()
        cap.release()
    print(f"    wrote debug video -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--traj-dir", required=True,
                    help="Directory containing ball_throw{1,2,3,4}_output/smoothed_root_pose.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--throw", default=None)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--save-video-dir", default=None,
                    help="If given, write one annotated .mp4 per (throw, camera) here.")
    ap.add_argument("--video-scale", type=float, default=0.5,
                    help="Debug video resize factor (default 0.5 -- full 4K is slow to write "
                         "and not needed to see the mask overlay).")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(f"file:{SESSION_DB}?mode=ro", uri=True)
    cams = load_cameras_from_session(SESSION_DB, EXTRINSIC_CALIBRATION_ID, SESSION_ID)
    cam_by_label = {c["label"]: c for c in cams}

    # Hydra global-state clash between SAM2's and Cutie's own config loaders
    # (see prototype_ball_cutie_segmentation.py's own note) -- clear once,
    # up front, since this script's per-camera loop touches both repeatedly.
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    from pipeline.pose.segmentation import CutieSegmentor

    for throw_name, step0, step1 in THROWS:
        if args.throw and throw_name != args.throw:
            continue
        traj_path = Path(args.traj_dir) / f"ball_{throw_name}_output" / "smoothed_root_pose.csv"
        traj = load_ball_trajectory(traj_path)
        if not traj:
            print(f"[{throw_name}] no trajectory points in {traj_path}, skipping")
            continue
        # Seed from the highest point in the air within the *core* (unpadded)
        # throw window, not the trajectory's middle index (2026-09-16, Harri:
        # throw 1's re-tracked result stayed at a near-constant position for
        # the whole run). Root-caused: a trajectory covers far more settled/
        # held time than actual flight time, so its middle *index* usually
        # lands in a settled stretch, not mid-flight -- confirmed directly
        # against tracking_obs_results for the first attempt (throw 1: every
        # camera's real 2D observation moved <20px across the whole ~190-step
        # window). Max-z within the core window is a simple, robust proxy for
        # "clearly mid-flight, not held or resting on the ground."
        core_t0, core_t1 = step_to_time(step0), step_to_time(step1)
        core_traj = [(t, xyz) for t, xyz in traj if core_t0 <= t <= core_t1] or traj
        ref_t, ref_xyz = max(core_traj, key=lambda p: p[1][2])
        t0 = step_to_time(step0) - _PAD_S
        t1 = step_to_time(step1) + _PAD_S
        print(f"\n=== {throw_name}: ref t={ref_t:.3f} xyz={ref_xyz} (max-z in core window), "
              f"window [{t0:.2f},{t1:.2f}] ===")

        for cam_label, cam in cam_by_label.items():
            if args.camera and cam_label != args.camera:
                continue
            video_path = conn.execute(
                "SELECT file_path FROM capture_videos WHERE camera_instance_id=?",
                (cam["camera_instance_id"],),
            ).fetchone()[0]
            ref_frame, ref_frame_t, fps = camera_frame_ref(conn, cam["camera_instance_id"])
            init_frame = time_to_frame(ref_t, ref_frame, ref_frame_t, fps)
            start_frame = time_to_frame(t0, ref_frame, ref_frame_t, fps)
            end_frame = time_to_frame(t1, ref_frame, ref_frame_t, fps)

            proj = project_point(ref_xyz, cam["K"], cam["R"], cam["t"], cam["dist"])
            cx, cy = proj

            cap = cv2.VideoCapture(video_path)
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not (0 <= cx < frame_w and 0 <= cy < frame_h):
                print(f"  {cam_label:20s} projected point ({cx:.0f},{cy:.0f}) outside frame, skipping")
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
            ok, init_img = cap.read()
            cap.release()
            if not ok:
                print(f"  {cam_label:20s} could not read init frame {init_frame}, skipping")
                continue

            box = np.array([cx - INIT_HALF_SIZE_PX, cy - INIT_HALF_SIZE_PX,
                            cx + INIT_HALF_SIZE_PX, cy + INIT_HALF_SIZE_PX])
            # SAM2 and Cutie both drive Hydra's global config state, but
            # asymmetrically: sam2/__init__.py does
            # `if not GlobalHydra.instance().is_initialized(): initialize_config_module(...)`
            # -- a *guarded*, import-time-only init that never fires again once
            # the module is cached, so a plain clear() before this call leaves
            # SAM2 with no config path at all (confirmed: "GlobalHydra is not
            # initialized" from compose() on the 2nd camera). Cutie's own
            # get_default_model() instead calls hydra.initialize() unconditionally
            # every time, which raises if Hydra is already initialized. So: clear,
            # then explicitly redo SAM2's own (otherwise one-shot) init before
            # every SAM2 call; just clear (no reinit needed) before every Cutie
            # call, since Cutie always initializes itself fresh.
            GlobalHydra.instance().clear()
            initialize_config_module("sam2", version_base="1.2")
            init_mask, score, _ = build_sam2_init_mask(init_img, box, args.device)
            GlobalHydra.instance().clear()
            print(f"  {cam_label:20s} init_frame={init_frame} proj=({cx:.0f},{cy:.0f}) "
                  f"SAM2 score={score:.3f} mask_px={init_mask.sum()}")

            segmentor = CutieSegmentor(device=args.device, max_internal_size=480, erosion_px=3)
            segmentor.process_video(
                video_path,
                init_frame=init_frame,
                persons={"ball": box},
                init_mask=init_mask,
                start_frame=max(0, start_frame),
                end_frame=end_frame,
                verbose=False,
            )

            rows = []
            for frame_idx in sorted(segmentor._masks.keys()):
                m = segmentor._masks[frame_idx].get("ball")
                if m is None or not m.any():
                    continue
                ys, xs = np.nonzero(m)
                ts = frame_to_time(frame_idx, ref_frame, ref_frame_t, fps)
                rows.append((frame_idx, ts, float(xs.mean()), float(ys.mean()), int(m.sum())))

            out_path = out_dir / f"{throw_name}_{cam_label}.csv"
            with open(out_path, "w") as f:
                f.write("video_frame,timestamp_s,cx,cy,mask_area_px\n")
                for r in rows:
                    f.write(f"{r[0]},{r[1]:.6f},{r[2]:.2f},{r[3]:.2f},{r[4]}\n")
            print(f"  {cam_label:20s} {len(rows)}/{end_frame - start_frame} frames -> {out_path.name}")

            if args.save_video_dir:
                write_debug_video(
                    video_path, segmentor._masks, start_frame, end_frame,
                    Path(args.save_video_dir) / f"{throw_name}_{cam_label}.mp4",
                    fps=fps, scale=args.video_scale,
                )


if __name__ == "__main__":
    main()
