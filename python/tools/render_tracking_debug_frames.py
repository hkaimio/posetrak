# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""render_tracking_debug_frames.py — offline visual diagnostic for a
marker/dot tracking run: draws the raw video frame with the actual ArUco
corner detections, the raw reflective-dot candidates the detector saw, and
the tracker's own current pose estimate (re-projected via forward
kinematics, not just whatever obs_blob happened to record) all overlaid --
either as PNGs for a handful of chosen timestamps, or as a slow-motion
video over a time range.

Built to diagnose a real, quantitative finding: adding reflective dots to
the sword skeleton made the tracker's ArUco-corner reprojection error
systematically worse -- median ~1.5-2x higher even when restricted to
timestamps an ArUco-only control run also tracked (ruling out "it's just
attempting harder frames"). A first version of this tool (still frames
only, predicted positions taken straight from tracking_obs_results.obs_blob)
surfaced real, sharper findings on real footage:

- The two ArUco tags' calibrated relative pose is measurably wrong: their
  centers are ~5.6cm apart in-plane (see status.md/session notes) when the
  physical tags sit opposite each other on the blade at roughly the same
  point along its length -- they should differ mainly through the blade's
  ~2cm thickness, not by 5.6cm along it.
- obs_blob only ever carries a predicted projection for a marker that had
  an actual observation fed into that step's update (see
  ResultWriter::write_obs_results -- a marker absent from the observations
  vector gets no slot at all, not even a predicted-only one). That hid the
  tracker's own current pose estimate for every marker it *didn't* detect
  that step -- exactly the cases most worth seeing. This version instead
  decodes tracking_results.state directly and re-projects every marker via
  forward kinematics + the real camera calibration, so the tracker's whole
  current pose estimate is visible every step it was live, detected or not.
- A hollow ring drawn at a marker's exact pixel is hard to eyeball as
  "centered" once the ring's own radius is a meaningful fraction of the
  zoomed view -- confirmed no coordinate bug via a synthetic test (a 1px
  dot dead-center under a circle+cross, at scratch/tracking_debug/
  offset_test.png). Switched to a filled center dot plus a thin ring, so
  the exact point is unambiguous even zoomed in.

Reads:

- The raw video frame (via posetrak.detection.frame_source.iter_frames,
  same decoder the manual dot-calibration tool uses).
- tracking_results.state for the tracker step nearest a given timestamp --
  decoded via the same axis-angle exponential map State::to_error_vector()
  writes (position[0:3], axis-angle[3:6] -> scipy Rotation.from_rotvec),
  then forward-kinematics'd (this skeleton is a plain rigid body: every
  marker's world position is R @ local_offset + position, no joint chain
  to walk) and projected through the real camera calibration (fisheye or
  standard, matching CamCalibState.fisheye) into the same distorted-pixel
  space pose_observations/tracking_obs_results already use
  (pixels_are_undistorted=0). This is the tracker's own current best
  estimate for a marker whether or not it was actually observed that step.
  RTS-smoothed (is_smoothed=1) by default -- the actual delivered output, what a
  client application sees -- with --raw for the live/causal (is_smoothed=0)
  estimate instead, a different, narrower question (e.g. to match the
  tracked/lost percentages, always computed from raw tracking_results rows).
  Raw and smoothed rows use *different* tracker_step numbering for the same real
  time (smoothing only covers steps actually tracked, so a gap-heavy run's
  smoothed index runs ahead of the raw one) -- fixed 2026-09-05 after this tool
  briefly used the wrong one's step number to look up tracking_obs_results,
  silently showing a different, real instant's dot positions on the wrong frame
  (see _nearest_state_row()'s own doc comment for the confirmed example).
- tracking_obs_results.obs_blob for the RAW tracker step nearest that
  timestamp, regardless of the --raw flag above (this table is only ever
  written from the raw forward pass, so there is no smoothed version of it
  to read) -- actual_x/y for every marker that WAS fed into that step's
  update (ArUco corners and, for a dots-enabled run, dot0..dot6), decoded
  via app.mcp.db's existing helpers so the camera/marker index ordering is
  guaranteed to match what the C++ tracker actually wrote. Only slots tagged
  mode==POSITION are shown as a position -- see observation-results-
  semantics.md for why actual_x/y isn't always one.
- pose_observations (source='dots') for that camera's own video frame --
  every raw candidate the detector saw that frame, whether or not it got
  resolved to a marker slot that step. This is the one piece obs_blob
  can't show: an unresolved candidate never appears there at all.

Usage (a handful of still frames, individually cropped to their own content):
    python tools/render_tracking_debug_frames.py \\
        --session /path/to/session.db \\
        --run-id 88b86bc0-7267-4852-8007-8705c1da2945 \\
        --camera-label gopro-11_mini_02 \\
        --timestamps 40.0 53.7 54.2 54.7 55.2 \\
        --output-dir scratch/tracking_debug

Usage (a slow-motion video over a time range, one fixed crop window so the
subject can move within frame without the view jumping around):
    python tools/render_tracking_debug_frames.py \\
        --session /path/to/session.db \\
        --run-id 88b86bc0-7267-4852-8007-8705c1da2945 \\
        --camera-label gopro-11_mini_02 \\
        --start-time 53.0 --end-time 56.0 --slow-factor 4 \\
        --video-out scratch/tracking_debug/gap.mp4

Usage (every active camera side by side in a grid, each with its own crop --
lets a single-camera-specific issue be told apart from a genuinely shared
one; omit --camera-label to use every camera active in the run):
    python tools/render_tracking_debug_frames.py \\
        --session /path/to/session.db \\
        --run-id 88b86bc0-7267-4852-8007-8705c1da2945 \\
        --grid --start-time 53.0 --end-time 56.0 --slow-factor 4 \\
        --video-out scratch/tracking_debug/gap_grid.mp4

Legend drawn on each frame (filled dot at the exact point + thin ring):
    green   = actual ArUco corner, used as inlier
    orange  = actual ArUco corner, rejected as outlier
    cyan    = actual dot observation, used as inlier (labeled with name)
    magenta = actual dot observation, rejected as outlier (labeled)
    gray    = raw dot candidate never resolved to a marker this step
    red '+' = tracker's current pose estimate, this marker's projection
              (drawn for every marker every step the tracker was live,
              detected or not -- small/dim when there's no matching actual
              detection this step, normal size when there is)
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mcp.db import decode_obs_blob, get_run_cameras, get_run_markers  # noqa: E402
from app.pose.db_cache import decode_dot_candidates  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states, load_sync_table  # noqa: E402

_PAD_PX = 220  # crop padding around the region of interest, in source pixels


class _RunContext:
    """Everything needed to render frames for one (tracking run, camera) pair."""

    def __init__(
        self, conn: sqlite3.Connection, run_id: str, camera_label: str, use_smoothed: bool = True
    ) -> None:
        self.conn = conn
        self.run_id = run_id
        self.camera_label = camera_label
        # Smoothed (RTS) is the actual delivered output -- what a client application
        # sees -- so it's the right default for "does this look correct" review.
        # Raw is an explicit opt-in (--raw) for the different question "what did the
        # live, causal tracker see frame by frame" (e.g. matching the tracked/lost
        # percentages, which are always computed from raw tracking_results rows).
        self.use_smoothed = use_smoothed

        run = conn.execute(
            "SELECT observation_sequence_id, skeleton_id FROM tracking_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if run is None:
            raise SystemExit(f"tracking run not found: {run_id}")
        self.sequence_id = run["observation_sequence_id"]

        seq = conn.execute(
            "SELECT shot_id FROM pose_observation_sequences WHERE id = ?", (self.sequence_id,)
        ).fetchone()
        shot_id = seq["shot_id"]

        cam_row = conn.execute(
            "SELECT id FROM camera_instances WHERE label = ?", (camera_label,)
        ).fetchone()
        if cam_row is None:
            raise SystemExit(f"camera label not found: {camera_label}")
        self.camera_instance_id = cam_row["id"]

        video_row = conn.execute(
            "SELECT id, file_path, actual_fps FROM capture_videos "
            "WHERE shot_id = ? AND camera_instance_id = ?",
            (shot_id, self.camera_instance_id),
        ).fetchone()
        if video_row is None:
            raise SystemExit(f"no capture_videos row for {camera_label} in shot {shot_id}")
        self.svid = video_row["id"]
        self.file_path = video_row["file_path"]
        self.native_fps = float(video_row["actual_fps"])

        self.sync_table, _svid_by_cam = load_sync_table(conn, shot_id)
        self.marker_names = get_run_markers(conn, run_id)
        camera_ids, _names = get_run_cameras(conn, run_id)
        if self.camera_instance_id not in camera_ids:
            raise SystemExit(f"{camera_label} was not an active camera in this run: {camera_ids}")
        self.cam_idx = camera_ids.index(self.camera_instance_id)
        self.n_cam, self.n_mrk = len(camera_ids), len(self.marker_names)

        self.marker_offsets = _load_skeleton_markers(conn, run["skeleton_id"])
        cam_states = load_camera_states(conn, shot_id)
        if self.camera_instance_id not in cam_states:
            raise SystemExit(f"no solved extrinsics for {camera_label} in shot {shot_id}")
        self.cam_state = cam_states[self.camera_instance_id]


def _load_skeleton_markers(conn: sqlite3.Connection, skeleton_id: str) -> dict[str, np.ndarray]:
    """Return {marker_name: local_offset (3,)} for a rigid (root-only) skeleton.

    Assumes every marker parents directly to a zero-offset root joint --
    true for every marker-based-mocap prop skeleton generate_prop_skeleton_yaml()
    produces (see docs/roadmap/features/marker-based-mocap). A skeleton with
    real joint chains below the root would need a full FK walk instead;
    not needed here since every real capture so far is a rigid prop.
    """
    row = conn.execute("SELECT yaml_content FROM skeletons WHERE id = ?", (skeleton_id,)).fetchone()
    if row is None:
        raise SystemExit(f"skeleton not found: {skeleton_id}")
    doc = yaml.safe_load(row["yaml_content"])
    return {m["name"]: np.array(m["offset"], dtype=np.float64) for m in doc.get("markers", [])}


def _decode_pose(state_blob: bytes) -> tuple[np.ndarray, Rotation]:
    """Decode tracking_results.state (State::to_error_vector() layout) ->
    (root position (3,), root orientation as a scipy Rotation).

    state[0:3] = position (direct), state[3:6] = orientation as an
    axis-angle exponential-map vector (State::quaternion_to_axis_angle /
    axis_angle_to_quaternion in cpp/src/core/state.cpp) -- exactly what
    Rotation.from_rotvec expects. Works for any joint count since only the
    first 6 entries are position+orientation regardless of n_dof.
    """
    vec = np.frombuffer(bytes(state_blob), dtype=np.float64)
    pos = vec[0:3].copy()
    rotvec = vec[3:6].copy()
    return pos, Rotation.from_rotvec(rotvec)


def _redistort_pts(cam_state, pts_undistorted: np.ndarray) -> np.ndarray:
    """Convert Nx2 pixels in the camera's undistorted (K_new) space -- what
    tracking_obs_results.obs_blob's actual_x/y actually store, confirmed by
    round-tripping a real point back to bit-identical raw candidate pixels
    -- into the same raw/distorted pixel space the source video frame (and
    pose_observations' raw dot candidates) use.

    cpp/include/posetrak/core/camera.hpp's Camera::project_undistorted()
    ("for UKF") is why: the tracker's measurement model runs entirely in
    undistorted pixel space, so both actual_x/y and predicted_x/y in
    obs_blob are in that space. Drawing them straight onto the raw frame
    without this step is exactly the "circles land a few pixels off,
    worse near the frame edges where distortion is larger" bug -- a real
    rendering bug in the first version of this tool, not (as first
    guessed) a false alarm from eyeballing a hollow ring's center.

    Standard "redistort a pixel" trick: turn it into a normalized ray via
    the undistorted K, then run it back through the real (distorted)
    camera model at zero rotation/translation -- projectPoints' distortion
    step does the rest.
    """
    if pts_undistorted.shape[0] == 0:
        return pts_undistorted
    Kinv = np.linalg.inv(cam_state.K)
    rays = (Kinv @ np.hstack([pts_undistorted, np.ones((len(pts_undistorted), 1))]).T).T
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    obj = rays.reshape(-1, 1, 3).astype(np.float64)
    if cam_state.fisheye:
        out, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, cam_state.K_orig, cam_state.dist)
    else:
        out, _ = cv2.projectPoints(obj, rvec, tvec, cam_state.K_orig, cam_state.dist)
    return out.reshape(-1, 2)


def _project_points(cam_state, world_pts: np.ndarray) -> np.ndarray:
    """Project Nx3 world points into this camera's distorted pixel space --
    the same space pose_observations/tracking_obs_results actual_x/y use
    (pixels_are_undistorted=0), so overlays land directly on the raw frame."""
    if world_pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    rvec, _ = cv2.Rodrigues(cam_state.R)
    obj = world_pts.reshape(-1, 1, 3).astype(np.float64)
    if cam_state.fisheye:
        out, _ = cv2.fisheye.projectPoints(obj, rvec, cam_state.t, cam_state.K_orig, cam_state.dist)
    else:
        out, _ = cv2.projectPoints(obj, rvec, cam_state.t, cam_state.K_orig, cam_state.dist)
    return out.reshape(-1, 2)


def _fk_predict_all(ctx: _RunContext, pos: np.ndarray, rot: Rotation) -> dict[str, tuple[float, float]]:
    """Project every marker this skeleton defines through the current pose
    estimate -- available for any marker, detected this step or not."""
    names = list(ctx.marker_offsets)
    offsets = np.stack([ctx.marker_offsets[n] for n in names])
    world = rot.apply(offsets) + pos
    pix = _project_points(ctx.cam_state, world)
    return {n: (float(u), float(v)) for n, (u, v) in zip(names, pix)}


def _grab_frame(file_path: str, video_frame: int) -> np.ndarray | None:
    for _, img in iter_frames(file_path, video_frame, video_frame + 1):
        return img
    return None


def _nearest_state_row(
    conn: sqlite3.Connection, run_id: str, timestamp_s: float, is_smoothed: bool
) -> tuple[int, float, bytes] | None:
    """Find the tracking_results row of one flavor (raw or RTS-smoothed) nearest
    *timestamp_s*.

    Always pass the flavor you actually want, never reuse one flavor's
    tracker_step to key into something that assumes the other: raw and
    smoothed rows use *different* tracker_step numbering for the same real
    time (smoothing only ever covers steps that were actually tracked, so a
    gap-heavy run's smoothed index runs ahead of the raw one by however many
    steps were lost before it -- confirmed on a real run where raw
    tracker_step 1933 and smoothed tracker_step 1603 both carry
    timestamp_s=53.766, a >3s gap between what step "1603" means in each).
    tracking_obs_results is only ever written from the raw forward pass
    (ResultWriter::write_obs_results(), called from Tracker::update_step()),
    so it's raw-indexed only regardless of which flavor's state is being
    displayed -- see _frame_data_for_time()'s own two separate lookups.
    """
    row = conn.execute(
        "SELECT tracker_step, timestamp_s, state FROM tracking_results "
        "WHERE run_id = ? AND person_id = 0 AND is_smoothed = ? "
        "ORDER BY ABS(timestamp_s - ?) LIMIT 1",
        (run_id, 1 if is_smoothed else 0, timestamp_s),
    ).fetchone()
    return (row["tracker_step"], row["timestamp_s"], row["state"]) if row else None


def _raw_dot_candidates(
    conn: sqlite3.Connection, sequence_id: str, camera_instance_id: str, video_frame: int
) -> np.ndarray:
    """Return float32[N,9] (px, py, area, compactness, major_axis_px,
    minor_axis_px, dir_x, dir_y, tracklet_id) -- see
    db_cache.decode_dot_candidates() -- for this camera's own frame, or an
    empty array if this run has no dot source at all."""
    row = conn.execute(
        "SELECT kp_blob FROM pose_observations WHERE sequence_id = ? "
        "AND camera_instance_id = ? AND source = 'dots' AND video_frame = ?",
        (sequence_id, camera_instance_id, video_frame),
    ).fetchone()
    if row is None:
        return np.zeros((0, 9), dtype=np.float32)
    try:
        return decode_dot_candidates(bytes(row["kp_blob"]))
    except ValueError:
        # Older-format data (see db_cache.decode_dot_candidates' own
        # docstring) -- degrade to "no raw candidates shown" rather than
        # crash; everything else (actual/predicted marker overlays) is
        # unaffected since it doesn't touch this blob at all.
        return np.zeros((0, 8), dtype=np.float32)


class _FrameData:
    """Everything _draw_overlay needs for one (camera, timestamp)."""

    def __init__(self) -> None:
        self.actual: dict[str, tuple[float, float, bool]] = {}   # name -> (x, y, is_outlier)
        self.predicted: dict[str, tuple[float, float]] = {}       # name -> (x, y), FK-projected
        self.raw_dots: np.ndarray = np.zeros((0, 9), dtype=np.float32)
        self.status: str = ""


def _frame_data_for_time(ctx: _RunContext, timestamp_s: float, video_frame: int) -> _FrameData:
    fd = _FrameData()
    fd.raw_dots = _raw_dot_candidates(ctx.conn, ctx.sequence_id, ctx.camera_instance_id, video_frame)

    # The displayed pose (predicted-marker overlay) follows ctx.use_smoothed -- smoothed
    # is the actual delivered output (the default), raw is the live tracker's own causal
    # estimate (--raw), a different, narrower question.
    state_row = _nearest_state_row(ctx.conn, ctx.run_id, timestamp_s, ctx.use_smoothed)
    if state_row is None:
        fd.status = "no tracking_results near this time"
        return fd
    step, step_t, state_blob = state_row
    flavor = "smoothed" if ctx.use_smoothed else "raw"
    fd.status = f"step {step} @ {step_t:.3f}s ({flavor})"

    if state_blob is not None:
        pos, rot = _decode_pose(state_blob)
        fd.predicted = _fk_predict_all(ctx, pos, rot)

    # tracking_obs_results is only ever written from the raw forward pass
    # (ResultWriter::write_obs_results(), called from Tracker::update_step()) --
    # always look it up via the RAW step nearest this timestamp, regardless of which
    # flavor's state/predicted overlay is shown above. Reusing `step` here when
    # ctx.use_smoothed is true would be exactly the raw/smoothed step-numbering
    # collision fixed 2026-09-05 (see _nearest_state_row()'s own doc comment).
    raw_step_row = _nearest_state_row(ctx.conn, ctx.run_id, timestamp_s, is_smoothed=False)
    if raw_step_row is None:
        fd.status += "  (no raw step for obs lookup)"
        return fd
    raw_step, _raw_step_t, _raw_state = raw_step_row

    obs_row = ctx.conn.execute(
        "SELECT obs_blob FROM tracking_obs_results WHERE run_id = ? AND person_id = 0 AND tracker_step = ?",
        (ctx.run_id, raw_step),
    ).fetchone()
    if obs_row is None:
        fd.status += "  (tracking_lost, no obs_results)"
        return fd
    blob = decode_obs_blob(obs_row["obs_blob"], ctx.n_cam, ctx.n_mrk)
    cam_slice = blob[ctx.cam_idx]
    names, undist_pts, outliers = [], [], []
    for i, name in enumerate(ctx.marker_names):
        ax, ay, _px, _py, _mahal, _used, is_outlier, mode = cam_slice[i]
        # mode==0 is MeasurementMode::POSITION -- the only mode where actual_x/y is an
        # absolute pixel position (observation-results-semantics.md). A POSITION
        # observation always wins its (camera, marker) slot when more than one
        # Observation exists for it, so this should be rare in practice, but skip
        # rather than misinterpret a VELOCITY/PAIR_DIFF-only slot's non-positional value.
        if not np.isnan(ax) and mode == 0.0:
            names.append(name)
            undist_pts.append((ax, ay))
            outliers.append(is_outlier > 0.5)
    if names:
        dist_pts = _redistort_pts(ctx.cam_state, np.array(undist_pts, dtype=np.float64))
        for name, (dx, dy), is_out in zip(names, dist_pts, outliers):
            fd.actual[name] = (float(dx), float(dy), is_out)
    return fd


def _collect_points(fd: _FrameData) -> list[tuple[float, float]]:
    pts = [(float(cx), float(cy)) for cx, cy, *_rest in fd.raw_dots]
    pts.extend((x, y) for x, y in fd.predicted.values())
    pts.extend((x, y) for x, y, _out in fd.actual.values())
    return pts


def _dot(img: np.ndarray, x: float, y: float, color, r: int) -> None:
    """Filled center dot + thin ring -- unambiguous exact center even zoomed
    in, unlike a plain hollow ring whose true center is hard to judge once
    the ring's own radius is a meaningful fraction of the viewed area."""
    p = (int(round(x)), int(round(y)))
    cv2.circle(img, p, 2, color, -1)
    cv2.circle(img, p, r, color, 1)


def _draw_overlay(img: np.ndarray, marker_names: list[str], fd: _FrameData) -> np.ndarray:
    out = img.copy()

    for cx, cy, area, _compact, major, _minor, dir_x, dir_y, _tracklet_id in fd.raw_dots:
        # Plain ring sized to the candidate's equivalent circular diameter.
        # No filled center: a solid gray dot disappears against a similarly
        # gray/textured background (concrete wall, mat); a ring stays
        # visible while still reading as "detected, not yet assigned".
        r = max(3, int(np.sqrt(max(area, 1.0) / np.pi)))
        cv2.circle(out, (int(round(cx)), int(round(cy))), r, (160, 160, 160), 2)
        if dir_x != 0.0 or dir_y != 0.0:
            # Streak axis (dot_blob_detector.py's canonicalized, direction-
            # ambiguous dir_x/dir_y) drawn both ways from center -- it's an
            # axis, not an arrow, so there's no real "forward" end to show.
            half = major / 2.0
            p0 = (int(round(cx - dir_x * half)), int(round(cy - dir_y * half)))
            p1 = (int(round(cx + dir_x * half)), int(round(cy + dir_y * half)))
            cv2.line(out, p0, p1, (0, 200, 255), 1, cv2.LINE_AA)

    for name in marker_names:
        is_dot = name.startswith("dot")
        px, py = fd.predicted.get(name, (None, None))
        has_actual = name in fd.actual
        if px is not None:
            size = 14 if has_actual else 8
            thickness = 2 if has_actual else 1
            cv2.drawMarker(out, (int(round(px)), int(round(py))), (0, 0, 255),
                            markerType=cv2.MARKER_CROSS, markerSize=size, thickness=thickness)
        if has_actual:
            ax, ay, is_outlier = fd.actual[name]
            if is_dot:
                color = (255, 0, 255) if is_outlier else (255, 255, 0)  # magenta/cyan
                _dot(out, ax, ay, color, 8)
                cv2.putText(out, name, (int(ax) + 10, int(ay) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            else:
                color = (0, 165, 255) if is_outlier else (0, 255, 0)  # orange/green
                _dot(out, ax, ay, color, 6)

    return out


def _crop_to_points(img: np.ndarray, pts: list[tuple[float, float]], pad: int) -> np.ndarray:
    if not pts:
        return img
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = max(0, int(min(xs) - pad)), min(img.shape[1], int(max(xs) + pad))
    y0, y1 = max(0, int(min(ys) - pad)), min(img.shape[0], int(max(ys) + pad))
    if x1 > x0 and y1 > y0:
        return img[y0:y1, x0:x1]
    return img


def render_pngs(ctx: _RunContext, timestamps: list[float], output_dir: Path, camera_label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for t in timestamps:
        frame_idx = ctx.sync_table.lookup(t, ctx.svid)
        if frame_idx is None:
            print(f"t={t:.3f}: no sync data for this camera, skipping")
            continue
        img = _grab_frame(ctx.file_path, frame_idx)
        if img is None:
            print(f"t={t:.3f}: could not decode frame {frame_idx}, skipping")
            continue

        fd = _frame_data_for_time(ctx, t, frame_idx)
        print(f"t={t:.3f} (frame {frame_idx}, {fd.status}): "
              f"{len(fd.raw_dots)} raw dot candidates, {len(fd.actual)} resolved this step")

        rendered = _draw_overlay(img, ctx.marker_names, fd)
        rendered = _crop_to_points(rendered, _collect_points(fd), _PAD_PX)
        out_path = output_dir / f"{camera_label}_t{t:.3f}.png"
        cv2.imwrite(str(out_path), rendered)
        print(f"  -> {out_path}")


_CROP_TRIM_FRAC = 0.12  # percentile trim per side -- see _compute_crop_window's own doc comment.
# 12%, not a token 3%: confirmed on a real, genuinely unstable window (2026-09-05, Harri's own
# multi-camera review) that it's not just a rare single outlier point dragging the crop wide --
# a real fraction of frames can have a wrong prediction/false-positive candidate while the
# tracker is struggling, which is exactly the case this tool exists to zoom in on despite.


def _robust_bounds(values: list[float], trim_frac: float = _CROP_TRIM_FRAC) -> tuple[float, float]:
    """(trim_frac, 1-trim_frac)-percentile bounds rather than strict min/max.

    A single wild point -- a false-positive raw-dot candidate anywhere in
    frame (the detector has no notion of "near the subject"), or a predicted
    marker reprojected far away during genuinely high state uncertainty
    (exactly what this tool exists to surface) -- would otherwise blow up
    the crop for the *whole* clip on its own, even though it's one instant
    out of hundreds. Falls back to plain min/max when there are too few
    points for percentiles to mean much (avoids a degenerate, empty range).
    """
    if len(values) < 10:
        return min(values), max(values)
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, trim_frac * 100)), float(np.percentile(arr, (1 - trim_frac) * 100))


def _compute_crop_window(
    ctx: _RunContext, start_time: float, end_time: float
) -> tuple[int, int, int, int] | None:
    """One fixed crop window for a whole clip: a robust (trimmed) bound over
    the union of every frame's actual/predicted/raw-dot points over
    [start_time, end_time], generously padded. A per-frame crop would jump
    around and make drift/bias impossible to judge by eye, which is the
    whole point of this tool -- see _robust_bounds() for why min/max isn't
    used directly. Returns None if this camera has no sync data or no
    observations at all in the range (e.g. it doesn't cover this part of
    the capture).
    """
    frame_lo = ctx.sync_table.lookup(start_time, ctx.svid)
    frame_hi = ctx.sync_table.lookup(end_time, ctx.svid)
    if frame_lo is None or frame_hi is None:
        return None

    all_pts: list[tuple[float, float]] = []
    for frame_idx in range(frame_lo, frame_hi + 1):
        t = ctx.sync_table.frame_to_global_time(frame_idx, ctx.svid)
        if t is None:
            continue
        fd = _frame_data_for_time(ctx, t, frame_idx)
        all_pts.extend(_collect_points(fd))
    if not all_pts:
        return None
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x_lo, x_hi = _robust_bounds(xs)
    y_lo, y_hi = _robust_bounds(ys)
    pad = _PAD_PX * 2
    return (max(0, int(x_lo - pad)), max(0, int(y_lo - pad)),
            int(x_hi + pad), int(y_hi + pad))


def render_video(
    ctx: _RunContext, start_time: float, end_time: float, slow_factor: float, output_path: Path,
) -> None:
    frame_lo = ctx.sync_table.lookup(start_time, ctx.svid)
    frame_hi = ctx.sync_table.lookup(end_time, ctx.svid)
    if frame_lo is None or frame_hi is None:
        raise SystemExit("no sync data for this camera in the requested time range")

    crop = _compute_crop_window(ctx, start_time, end_time)
    if crop is None:
        raise SystemExit("no observations of any kind in this time range -- nothing to crop to")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    out_fps = ctx.native_fps / slow_factor
    n_written = 0
    for frame_idx, img in iter_frames(ctx.file_path, frame_lo, frame_hi + 1):
        t = ctx.sync_table.frame_to_global_time(frame_idx, ctx.svid) or start_time
        fd = _frame_data_for_time(ctx, t, frame_idx)
        rendered = _draw_overlay(img, ctx.marker_names, fd)
        x0, y0, x1, y1 = crop
        rendered = rendered[y0:min(y1, rendered.shape[0]), x0:min(x1, rendered.shape[1])]
        cv2.putText(rendered, f"t={t:.3f}s  {fd.status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if writer is None:
            h, w = rendered.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (w, h))
        writer.write(rendered)
        n_written += 1

    if writer is not None:
        writer.release()
    print(f"wrote {n_written} frames ({n_written / ctx.native_fps:.2f}s of real time) "
          f"at {out_fps:.1f}fps ({slow_factor}x slow-motion) -> {output_path}")


def _sequential_frame_lookup(ctx: _RunContext, target_times: list[float]):
    """Yield this camera's own (frame_idx, img) for each of *target_times*
    (assumed non-decreasing), decoding its video file with *one* sequential
    pass over the needed range rather than a fresh open-and-seek per frame.

    Calling _grab_frame() once per (camera, output frame) -- 1080 times for
    a 6-camera, 180-frame grid -- reopens and reseeks the video container on
    every single call (_iter_frames_av opens a fresh av.open() each time).
    Confirmed live: this made a grid render take far longer than 6 single-
    camera renders combined, with the Python process sitting near 0% CPU --
    the classic signature of an I/O-bound repeated-seek bottleneck, not
    actual decode work, especially reading 4K footage off an external drive.
    Sequentially decoding [min(needed), max(needed)] once and picking off
    the frames actually wanted, as render_video() already does for a single
    camera, avoids the reseeking entirely.

    Yields (None, None) for a target time this camera has no sync data for,
    and (frame_idx, None) if decoding reached that frame_idx but produced no
    image (a real decode failure, not just "camera doesn't cover this
    moment").
    """
    needed = [ctx.sync_table.lookup(t, ctx.svid) for t in target_times]
    valid = [f for f in needed if f is not None]
    if not valid:
        for _ in target_times:
            yield None, None
        return

    frame_iter = iter_frames(ctx.file_path, min(valid), max(valid) + 1)
    current_idx: int | None = None
    current_img: np.ndarray | None = None
    exhausted = False
    for want in needed:
        if want is None:
            yield None, None
            continue
        while not exhausted and (current_idx is None or current_idx < want):
            try:
                current_idx, current_img = next(frame_iter)
            except StopIteration:
                exhausted = True
        yield want, (current_img if current_idx == want else None)


def _render_camera_cell(
    ctx: _RunContext, timestamp_s: float, frame_idx: int | None, img: np.ndarray | None,
    window: tuple[int, int, int, int] | None, cell_w: int, cell_h: int,
) -> np.ndarray:
    """One camera's own rendered, cropped, letterboxed-to-(cell_w, cell_h)
    frame at *timestamp_s*, given its already-decoded *frame_idx*/*img*
    (see _sequential_frame_lookup() -- decoding happens once per camera for
    a whole grid render, not per cell) -- a black cell with a status label
    if this camera has no sync data, no frame, or no crop window (doesn't
    cover this part of the capture) at this instant, so a grid always has
    the same shape regardless of which cameras have data when.
    """
    cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

    def _label(text: str) -> None:
        cv2.putText(cell, f"{ctx.camera_label}: {text}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if window is None:
        _label("no data in range")
        return cell
    if frame_idx is None:
        _label("no sync at this time")
        return cell
    if img is None:
        _label("frame decode failed")
        return cell

    fd = _frame_data_for_time(ctx, timestamp_s, frame_idx)
    rendered = _draw_overlay(img, ctx.marker_names, fd)
    x0, y0, x1, y1 = window
    rendered = rendered[y0:min(y1, rendered.shape[0]), x0:min(x1, rendered.shape[1])]
    if rendered.shape[0] == 0 or rendered.shape[1] == 0:
        _label("empty crop")
        return cell

    h, w = rendered.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(rendered, (new_w, new_h))
    ox, oy = (cell_w - new_w) // 2, (cell_h - new_h) // 2
    cell[oy:oy + new_h, ox:ox + new_w] = resized
    cv2.putText(cell, f"{ctx.camera_label}  {fd.status}", (10, cell_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def render_grid_video(
    contexts: list[_RunContext], start_time: float, end_time: float, slow_factor: float,
    output_path: Path, cell_w: int = 900, cell_h: int = 600,
) -> None:
    """Render every camera in *contexts* (same run, one _RunContext each)
    side by side in a grid over [start_time, end_time], each cropped to its
    own union-of-observed/predicted-points window (same idea as
    render_video(), just computed per camera rather than once) -- lets a
    single-camera-specific issue (occlusion, glare, a bad calibration) be
    told apart from a genuinely shared one by seeing every camera's own view
    of the same real moments at once, instead of guessing from one.

    The first context in *contexts* drives the output frame cadence (its own
    sync table's frame indices over the range); every other camera's own
    frame at each resulting timestamp is looked up independently via its own
    sync table, since different cameras don't share frame numbering.
    """
    if not contexts:
        raise SystemExit("render_grid_video: no cameras given")

    windows = []
    for ctx in contexts:
        w = _compute_crop_window(ctx, start_time, end_time)
        if w is None:
            print(f"{ctx.camera_label}: no observations in this range -- shown blank in the grid")
        windows.append(w)
    if all(w is None for w in windows):
        raise SystemExit("no camera has any observations in this time range -- nothing to render")

    ref = contexts[0]
    frame_lo = ref.sync_table.lookup(start_time, ref.svid)
    frame_hi = ref.sync_table.lookup(end_time, ref.svid)
    if frame_lo is None or frame_hi is None:
        raise SystemExit(f"no sync data for reference camera {ref.camera_label} in this range")

    n = len(contexts)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # One target timestamp per output frame, computed once from the reference
    # camera -- every camera's own decoder (below) consumes this same list in
    # order, so every cell in a given output frame shows the same real instant.
    target_times = []
    for frame_idx in range(frame_lo, frame_hi + 1):
        t = ref.sync_table.frame_to_global_time(frame_idx, ref.svid)
        if t is not None:
            target_times.append(t)

    # One sequential decode pass per camera over the whole range (see
    # _sequential_frame_lookup()'s own doc comment for why this matters).
    streams = [_sequential_frame_lookup(ctx, target_times) for ctx in contexts]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    out_fps = ref.native_fps / slow_factor
    n_written = 0

    for t in target_times:
        canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        for i, (ctx, window, stream) in enumerate(zip(contexts, windows, streams)):
            frame_idx, img = next(stream)
            cell = _render_camera_cell(ctx, t, frame_idx, img, window, cell_w, cell_h)
            r, c = divmod(i, cols)
            canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cell
        cv2.putText(canvas, f"t={t:.3f}s", (10, canvas.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (cols * cell_w, rows * cell_h))
        writer.write(canvas)
        n_written += 1

    if writer is not None:
        writer.release()
    print(f"wrote {n_written} frames ({n_written / ref.native_fps:.2f}s of real time) "
          f"at {out_fps:.1f}fps ({slow_factor}x slow-motion), {cols}x{rows} grid -> {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--camera-label", nargs="*", default=None,
                     help="One camera label for --timestamps/--video-out. For --grid, zero or "
                          "more (default: every camera active in this run).")
    ap.add_argument("--timestamps", nargs="+", type=float, default=None,
                     help="Still-frame mode: render one PNG per timestamp, each cropped to its own content.")
    ap.add_argument("--output-dir", default=None, help="Required with --timestamps.")
    ap.add_argument("--start-time", type=float, default=None, help="Video mode: clip start time (s).")
    ap.add_argument("--end-time", type=float, default=None, help="Video mode: clip end time (s).")
    ap.add_argument("--slow-factor", type=float, default=4.0,
                     help="Video mode: output fps = native fps / this factor (default 4x slow-motion).")
    ap.add_argument("--video-out", default=None, help="Video mode: output .mp4 path.")
    ap.add_argument("--grid", action="store_true",
                     help="With --video-out: render every --camera-label (default: all active "
                          "cameras in this run) side by side in a grid, each cropped to its own "
                          "content, instead of a single camera's own video -- lets a single-"
                          "camera-specific issue be told apart from a genuinely shared one.")
    ap.add_argument("--cell-width", type=int, default=900, help="--grid only: per-camera cell width.")
    ap.add_argument("--cell-height", type=int, default=600, help="--grid only: per-camera cell height.")
    ap.add_argument("--raw", action="store_true",
                     help="Show the raw (is_smoothed=0), live/causal tracker state instead of "
                          "the default RTS-smoothed one. Smoothed is the actual delivered "
                          "output (what a client application sees); raw is the narrower "
                          "question of what the tracker's own per-frame decisions looked like "
                          "(e.g. to match the tracked/lost percentages, always computed from "
                          "raw tracking_results rows). tracking_obs_results-derived overlays "
                          "(raw dot candidates, actual/predicted marker positions) are always "
                          "looked up via the raw step regardless of this flag -- that table is "
                          "only ever written from the raw forward pass.")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    if args.grid:
        if args.video_out is None or args.start_time is None or args.end_time is None:
            raise SystemExit("--grid requires --video-out, --start-time, and --end-time")
        labels = args.camera_label
        if not labels:
            _camera_ids, names = get_run_cameras(conn, args.run_id)
            labels = list(names.values())
            print(f"--grid with no --camera-label: using every active camera in this run: {labels}")
        contexts = [_RunContext(conn, args.run_id, label, use_smoothed=not args.raw) for label in labels]
        render_grid_video(contexts, args.start_time, args.end_time, args.slow_factor,
                          Path(args.video_out), args.cell_width, args.cell_height)
        return

    if not args.camera_label or len(args.camera_label) != 1:
        raise SystemExit("exactly one --camera-label is required outside --grid mode")
    ctx = _RunContext(conn, args.run_id, args.camera_label[0], use_smoothed=not args.raw)

    if args.timestamps is not None:
        if args.output_dir is None:
            raise SystemExit("--output-dir is required with --timestamps")
        render_pngs(ctx, args.timestamps, Path(args.output_dir), args.camera_label[0])
    elif args.video_out is not None:
        if args.start_time is None or args.end_time is None:
            raise SystemExit("--start-time and --end-time are required with --video-out")
        render_video(ctx, args.start_time, args.end_time, args.slow_factor, Path(args.video_out))
    else:
        raise SystemExit("specify either --timestamps --output-dir, or --start-time --end-time --video-out")


if __name__ == "__main__":
    main()
