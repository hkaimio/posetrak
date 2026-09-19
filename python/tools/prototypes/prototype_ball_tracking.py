# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prototype_ball_tracking.py — check whether the fully-reflective ball from
the "3rd case" (Nelli throws a small reflective ball to the ground and
catches it on the bounce, 2026-09-06-kare-tests capture) can be positioned
in 3D using data that already exists in the session DB, with no new
detection pass at all.

Background (2026-09-15, Harri): the ball is small, covered entirely in
reflective material (so only position, never orientation, is meaningful),
and is thrown/caught four times, alternating hands:

    throw 1: ~steps 2816-2902 (right hand)
    throw 2: ~steps 3637-3735
    throw 3: ~steps 3989-4074 (left hand)
    throw 4: ~steps 4304-4401

(step ranges are approximate; tracker_fps=120, tracker time range starts at
33.620s -- see any full-trial log's "Time range:" line -- so
t = 33.620 + step/120). Harri also scoped the goal down explicitly: full
automatic hand-to-ball-to-hand linking is not worth building; it is enough
to recover the ball's 3D position for the frames where it is actually
visible (i.e. NOT the moments it's swallowed inside a closed fist), and
leave stitching that into the hand's motion as a manual/animation-software
postprocessing step.

Key finding: a `dots` detection run (background-subtraction +
saturation-filtered reflective-blob detector, `python/posetrak/detection/
dot_blob_detector.py`) already covers this capture's *entire* time range
(33.62-130.185s) across all 6 cameras -- built earlier for the leg/hand
markers, not the ball, but it has no idea what it's looking at and picks up
every sufficiently bright, sufficiently round/saturated-and-desaturated
reflective blob in frame, ball included. So the "can we get position data"
question reduces to a correspondence problem: for each frame, several
candidates per camera (ball, body markers, and -- found the hard way below
-- static clutter), find the one 3D point that's actually consistent across
multiple independent views.

Two things that did NOT work well enough on their own, kept here as a
record of what was tried:

1. Naive area threshold, one "biggest candidate" per camera per frame: the
   ball IS distinctly large in some cameras (gopro13_02: ~90-230px vs
   ~10-40px for body markers, tracing an obviously ball-like falling/
   bouncing arc) but NOT in others (gopro13_01 has 200+/207 frames with
   *something* above the same threshold -- a closer/higher-resolution view
   where body markers alone already exceed it). A single global area
   threshold cannot separate the ball from body markers consistently
   across cameras with different focal lengths/distances to the subject.

2. Adding per-camera static-position rejection (candidates recurring at a
   near-constant screen position across the window, most likely ring-light
   reflections off the floor/walls/other props -- the existing dot
   detector's own per-camera background blacklist evidently wasn't tuned
   against this specific ball) helped but still left several cameras with
   a candidate in nearly every frame, and "biggest per camera" still often
   picked the wrong one -- reprojection error stayed in the hundreds of
   pixels for the vast majority of frames.

What actually works: real multi-view RANSAC. Keep *every* above-threshold,
non-static candidate per camera per frame (usually 0-2, not just the
single biggest), try triangulating every candidate pair from two cameras,
and score each hypothesis by how many *other* cameras have some candidate
within a tight reprojection tolerance of it. A real ball position gets
independent corroboration from several geometrically-unrelated cameras;
clutter in one camera essentially never lines up with clutter in another
by chance. Requiring >=3 total agreeing cameras (not just the seed pair)
is what actually rejects the false correspondences from (1)/(2).

Not part of the production write path -- read-only against the session DB,
writes only a CSV per throw (and this script's own stdout summary) to the
output dir given on the command line. If this holds up, the natural next
step is porting this into something that writes actual
`pose_observations`/`capture_objects` rows the way
`setup_pen_pad_capture_objects.py` did for the pen/pad -- not attempted
here.
"""

from __future__ import annotations

import argparse
import sqlite3
import struct
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from posetrak.db.load_session import load_cameras_from_session

_DOT_CANDIDATE_FLOATS = 9  # cx, cy, area, compactness, major, minor, dir_x, dir_y, tracklet_id

SESSION_DB = "D:/mocap/2026-09-06-kare-tests/2026-09-06-kare-tests.db"
SESSION_ID = "662854b2-fdaf-4f37-ae8c-30e1458e8f1c"
EXTRINSIC_CALIBRATION_ID = "4815f5ac-116b-4ae6-b619-2d16bb954777"

TRACKER_FPS = 120.0
TRACKER_T0 = 33.620  # "Time range: [33.620, ...)" from any full-trial log

# (name, step_start, step_end), padded by _PAD_STEPS on each side so an
# "approximate" window doesn't clip the real arc.
THROWS = [
    ("throw1_right", 2816, 2902),
    ("throw2", 3637, 3735),
    ("throw3_left", 3989, 4074),
    ("throw4", 4304, 4401),
]
_PAD_STEPS = 60  # 0.5s

# Body markers ran <=40px in every camera checked; the ball ran 90-230px in
# the cameras where it stands out. Not sufficient by itself (see module
# docstring) but still worth keeping as a first-pass filter to cut down
# candidate volume before the real (RANSAC) correspondence step.
DEFAULT_MIN_AREA = 60.0

# Camera reliability for reflective-dot detection is already characterized
# in status.md (2026-09-0x recall/precision sweep against real GT labels):
# gopro-11_mini_01 and gopro13_02 are the genuine ring-lit cameras (recall
# 0.96-0.98, precision* ~1.0). gopro13_01's own markers are dim (~130
# brightness) and what it flags "large" at any usable threshold is mostly
# prop/glare, not real reflective response (confirmed independently here:
# 200+/207 frames have *something* above DEFAULT_MIN_AREA, ~15/frame after
# static-rejection -- nothing like a single ball). insta_ace2_pro's ring
# light was failing during this take. oneplus9pro-01/pixel9 recover to
# decent (~0.6-0.8) recall only with per-camera-tuned thresholds for their
# own *body* markers, which is a different, smaller, brighter target than
# this ball -- included as optional corroboration, never required.
_TRUSTED_CAMERA_LABELS = {"gopro-11_mini_01", "gopro13_02"}
_CORROBORATING_CAMERA_LABELS = {"oneplus9pro-01", "pixel9"}
_EXCLUDED_CAMERA_LABELS = {"gopro13_01", "insta_ace2_pro"}

# Static-background rejection (see module docstring point 2): bucket
# candidate positions to a coarse grid: drop any bucket that recurs in more
# than _STATIC_FRAC of frames.
_STATIC_GRID_PX = 20.0
_STATIC_FRAC = 0.4

# RANSAC: a real 3D point's reprojection into a supporting camera should
# land within a few pixels; this is deliberately tight so unrelated clutter
# essentially never lines up by chance across independent views.
# 22px landed empirically: requiring the tighter reprojection error this
# session's actual skeleton/marker reprojection numbers never hit either
# (typically 20-50px even for correctly-associated body markers -- see
# status.md's many reprojection-error entries) starved coverage for no
# real gain, while much looser tolerances let unrelated candidates start
# coincidentally lining up.
_INLIER_REPROJ_PX = 22.0
_MIN_SUPPORTING_CAMERAS = 2
# Seeding from any of the 4 non-excluded cameras, not just the trusted pair,
# matters because occlusion is real and per-camera: in throw 2's window,
# gopro-11_mini_01 loses the ball outright for ~1/5 of the window (behind
# the body) while gopro13_02 keeps it continuously -- restricting seeding
# to a fixed pair means BOTH must be simultaneously unoccluded, which
# happens less often than "some 2 of the 4 are". Requiring >=2 *agreeing*
# cameras (not just present) is still the real discriminator against false
# correspondences (see solve_step's docstring) -- this is orthogonal to
# which cameras are allowed to seed.
_SEED_FROM_ALL_CAMERAS = True


def step_to_time(step: int) -> float:
    return TRACKER_T0 + step / TRACKER_FPS


def time_to_step(t: float) -> int:
    return int(round((t - TRACKER_T0) * TRACKER_FPS))


def decode_dot_blob(blob: bytes) -> np.ndarray:
    (n,) = struct.unpack_from("<i", blob, 0)
    expected = 4 + n * _DOT_CANDIDATE_FLOATS * 4
    if n < 0 or len(blob) != expected:
        return np.zeros((0, _DOT_CANDIDATE_FLOATS), dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32, offset=4).reshape(n, _DOT_CANDIDATE_FLOATS)


def _find_static_buckets(frames: list[np.ndarray]) -> set[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    n_frames = len(frames)
    for arr in frames:
        seen_this_frame = {
            (int(cx // _STATIC_GRID_PX), int(cy // _STATIC_GRID_PX)) for cx, cy in arr[:, :2]
        }
        counts.update(seen_this_frame)
    return {b for b, c in counts.items() if n_frames > 0 and c / n_frames >= _STATIC_FRAC}


def load_ball_candidates_by_step(
    conn: sqlite3.Connection, camera_instance_id: str, t0: float, t1: float, min_area: float
) -> dict[int, np.ndarray]:
    """Return {step: ndarray[K, 2] of (cx, cy)} -- every above-threshold,
    non-static candidate for that camera at that tracker step (0, 1, or a
    handful; never pre-collapsed to "the" ball)."""
    rows = conn.execute(
        "SELECT timestamp_s, kp_blob FROM pose_observations"
        " WHERE source='dots' AND camera_instance_id=? AND timestamp_s BETWEEN ? AND ?"
        " ORDER BY timestamp_s",
        (camera_instance_id, t0, t1),
    ).fetchall()
    decoded = [(float(ts), decode_dot_blob(bytes(blob))) for ts, blob in rows]
    area_filtered = [arr[arr[:, 2] >= min_area] for _, arr in decoded]
    static_buckets = _find_static_buckets(area_filtered)

    out: dict[int, list[np.ndarray]] = {}
    for (ts, _), cand in zip(decoded, area_filtered):
        if cand.shape[0] == 0:
            continue
        is_static = np.array(
            [
                (int(cx // _STATIC_GRID_PX), int(cy // _STATIC_GRID_PX)) in static_buckets
                for cx, cy in cand[:, :2]
            ]
        )
        cand = cand[~is_static]
        if cand.shape[0] == 0:
            continue
        step = time_to_step(ts)
        out.setdefault(step, []).append(cand[:, :2])
    return {s: np.concatenate(v, axis=0) for s, v in out.items()}


# Motion-gated single-object continuation tracker, per camera (2026-09-15,
# Harri: "the ball is visible for the whole 2nd throw in gopro-11_mini_01
# [but load_ball_candidates_by_step's flat area threshold missed most of
# it]"). Root cause, confirmed by inspecting the raw (unfiltered)
# candidates through one such gap: the ball is detected in *every* frame,
# but during the fast bounce impact its blob fragments/blurs -- area drops
# from ~350-400px to as low as ~20-50px and compactness collapses (a
# streak, the same motion-blur phenomenon this project's sword-prop work
# already characterized) for a handful of frames, then recovers. A flat
# area cutoff can't follow that without also letting in real body-marker-
# sized clutter everywhere else. What actually identifies the ball through
# the blur is motion continuity: nothing else is where the ball's own
# recent trajectory predicts it to be.
#
# This deliberately does not reuse the production `MotionGatedLinker`
# (posetrak.detection.dot_tracklet) -- that class is mid-edit in a parallel,
# uncommitted change on this same branch (a different investigation), and
# its tuning (accel_std_px=2.0, mahal_gate=9.21) targets slow, near-
# constant-velocity body-marker motion; a thrown/bounced ball's frame-to-
# frame speed changes far more abruptly right at the moments that matter
# most here. A small self-contained greedy version, tuned to what this
# ball's own trajectory actually looks like, is simpler to get right for
# this one case than retuning a shared linker mid-flight under someone
# else's edit.
_TRACK_SEED_MIN_AREA = 80.0  # bootstrap only needs one unambiguous frame
_TRACK_MIN_CANDIDATE_AREA = 8.0  # floor against pure noise specks
_TRACK_MAX_MISSED = 6  # consecutive frames allowed to coast with no accepted observation
_TRACK_GATE_MIN_PX = 60.0
_TRACK_GATE_SPEED_MULT = 3.0  # gate = max(GATE_MIN_PX, this * |velocity| * dt)
_TRACK_BIRTH_GATE_PX = 220.0  # first step away from a seed, no velocity estimate yet


def _greedy_track_from_seed(
    frames: list[tuple[float, np.ndarray]], seed_idx: int, seed_pos: np.ndarray, forward: bool
) -> dict[int, np.ndarray]:
    """Walk frames away from a seed index, in one direction, accepting the
    candidate nearest a constant-velocity prediction each step (gated by
    recent speed), coasting through up to _TRACK_MAX_MISSED consecutive
    misses. Returns {frame_index_into `frames`: (cx, cy)}."""
    out: dict[int, np.ndarray] = {}
    pos = seed_pos
    vel = np.zeros(2)
    have_vel = False
    missed = 0
    t_prev = frames[seed_idx][0]

    idx_range = range(seed_idx + 1, len(frames)) if forward else range(seed_idx - 1, -1, -1)
    for idx in idx_range:
        t, arr = frames[idx]
        dt = abs(t - t_prev)
        if dt <= 0:
            continue
        predicted = pos + (vel * dt if have_vel else 0.0)
        cand = arr[arr[:, 2] >= _TRACK_MIN_CANDIDATE_AREA]
        if have_vel:
            speed = float(np.linalg.norm(vel))
            gate = max(_TRACK_GATE_MIN_PX, _TRACK_GATE_SPEED_MULT * speed * dt)
        else:
            # No velocity estimate yet (first step away from the seed) --
            # a hard bounce reversal can move the ball a genuinely large
            # distance in one 8ms frame (found empirically: ~200px at the
            # exact instant of ground impact), well past what a body-marker
            # -tuned gate like MotionGatedLinker's birth_gate_px=25 assumes.
            gate = _TRACK_BIRTH_GATE_PX

        accepted = None
        if cand.shape[0] > 0:
            dists = np.linalg.norm(cand[:, :2] - predicted, axis=1)
            k = int(np.argmin(dists))
            if dists[k] <= gate:
                accepted = cand[k, :2]

        if accepted is not None:
            new_vel = (accepted - pos) / dt
            pos, vel, have_vel, missed = accepted, new_vel, True, 0
            out[idx] = accepted
            t_prev = t
        else:
            missed += 1
            if missed > _TRACK_MAX_MISSED:
                break
            pos = predicted  # coast
            t_prev = t
    return out


# A wrong seed (a body marker or artifact that happens to exceed
# _TRACK_SEED_MIN_AREA once, e.g. in the padding before/after the throw
# proper) still produces a perfectly smooth, confidently-continued track --
# it's real motion, just not the ball's. Found by running the first version
# of this tracker: several camera/throw combinations ended up "tracking"
# almost every single frame in the window, which a ball that's supposedly
# swallowed by a fist part of the time cannot do. Two guards: seed only from
# the largest candidate within the *core* (unpadded, user-given) step range
# -- much less likely to be padding-region clutter -- and reject the whole
# track after the fact if its total screen-space excursion is too small to
# be a thrown-to-the-ground-and-back ball (a stuck lock onto a torso/hand
# marker stays comparatively local).
_TRACK_MIN_RANGE_PX = 150.0

# A single-frame area spike with no correlate in either adjacent frame is
# almost certainly a one-off contour-merge artifact, not the ball -- found
# by tracing exactly this failure: gopro-11_mini_01's biggest candidate in
# throw 2's core window (426px, plausible-looking, decently round) turns
# out to appear at that screen position in *no* other frame at all, right
# after (not during) the low-area motion-blur stretch that's the real ball.
# A genuine ball, even mid-bounce, has *some* candidate within a modest
# gate in the frame immediately before or after -- 8ms apart at 120Hz.
_SEED_ISOLATION_GATE_PX = 100.0


def _has_nearby_candidate(frames: list[tuple[float, np.ndarray]], idx: int, pos: np.ndarray) -> bool:
    for neighbor in (idx - 1, idx + 1):
        if not (0 <= neighbor < len(frames)):
            continue
        arr = frames[neighbor][1]
        if arr.shape[0] == 0:
            continue
        if np.linalg.norm(arr[:, :2] - pos, axis=1).min() <= _SEED_ISOLATION_GATE_PX:
            return True
    return False


def track_ball_in_camera(
    frames: list[tuple[float, np.ndarray]], core_t0: float, core_t1: float
) -> dict[int, np.ndarray]:
    """frames: [(timestamp_s, candidates[K,9]), ...] sorted by time, already
    static-bucket-filtered. Returns {frame_index: (cx, cy)} for every frame
    where continuation found an accepted candidate (the seed frame
    included) -- NOT every frame in `frames`, gaps mean genuinely lost."""
    seed_candidates = []  # (area, idx, pos)
    for i, (t, arr) in enumerate(frames):
        if not (core_t0 <= t <= core_t1) or arr.shape[0] == 0:
            continue
        big = arr[arr[:, 2] >= _TRACK_SEED_MIN_AREA]
        for row in big:
            seed_candidates.append((float(row[2]), i, row[:2]))
    seed_candidates.sort(key=lambda c: -c[0])

    seed_idx, seed_pos = None, None
    for _, i, pos in seed_candidates:
        if _has_nearby_candidate(frames, i, pos):
            seed_idx, seed_pos = i, pos
            break
    if seed_idx is None:
        return {}

    result = {seed_idx: seed_pos}
    result.update(_greedy_track_from_seed(frames, seed_idx, seed_pos, forward=True))
    result.update(_greedy_track_from_seed(frames, seed_idx, seed_pos, forward=False))

    positions = np.array(list(result.values()))
    excursion = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    if excursion < _TRACK_MIN_RANGE_PX:
        return {}
    return result


def load_ball_candidates_by_step_tracked(
    conn: sqlite3.Connection,
    camera_instance_id: str,
    t0: float,
    t1: float,
    min_area: float,
    core_t0: float,
    core_t1: float,
) -> dict[int, np.ndarray]:
    """Motion-gated-continuation version of load_ball_candidates_by_step --
    see track_ball_in_camera(). Deliberately does NOT run the flat
    per-camera static-bucket rejection load_ball_candidates_by_step uses:
    the ball spends a real fraction of this padded window sitting almost
    still (held in hand before/after the quick throw itself), which the
    coarse "recurs in more than _STATIC_FRAC of frames" rule cannot tell
    apart from actual static clutter -- confirmed by trying it first,
    which starved several camera/throw combinations of exactly the seed
    frames needed. track_ball_in_camera()'s own core-window seeding plus
    excursion sanity check are the real discriminators here instead."""
    rows = conn.execute(
        "SELECT timestamp_s, kp_blob FROM pose_observations"
        " WHERE source='dots' AND camera_instance_id=? AND timestamp_s BETWEEN ? AND ?"
        " ORDER BY timestamp_s",
        (camera_instance_id, t0, t1),
    ).fetchall()
    frames = [(float(ts), decode_dot_blob(bytes(blob))) for ts, blob in rows]

    tracked = track_ball_in_camera(frames, core_t0, core_t1)
    out = {}
    for idx, pos in tracked.items():
        step = time_to_step(frames[idx][0])
        out[step] = pos.reshape(1, 2)
    return out


def undistort_batch(pts: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """[N,2] raw pixels -> [N,2] normalized (undistorted, K-free) coords."""
    if pts.shape[0] == 0:
        return pts.reshape(0, 2)
    und = cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), K, dist)
    return und.reshape(-1, 2)


def triangulate_dlt(
    normalized_pts: list[np.ndarray], extrinsics: list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Multi-view DLT triangulation in normalized (K-free) camera coordinates."""
    A = []
    for (x, y), (R, t) in zip(normalized_pts, extrinsics):
        P = np.hstack([R, t.reshape(3, 1)])
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def reproj_errors(X: np.ndarray, P: np.ndarray, raw_pts: np.ndarray) -> np.ndarray:
    proj = P @ np.append(X, 1.0)
    proj = proj[:2] / proj[2]
    return np.hypot(raw_pts[:, 0] - proj[0], raw_pts[:, 1] - proj[1])


def solve_step(
    cams: list[dict],
    raw_by_cam: dict[str, np.ndarray],
    normalized_by_cam: dict[str, np.ndarray],
) -> tuple[np.ndarray, int, float] | None:
    """RANSAC seeded *only* from the two trusted (proven ring-lit) cameras --
    with just two views, any candidate pair trivially "agrees" with itself
    (zero reprojection error by construction), so a seed pair drawn from
    unreliable cameras provides no real discriminative power at all. Scores
    each seed hypothesis by how many *other* cameras (including the
    optional corroborating set) independently support it. Returns
    (X, n_supporting_cameras, mean_inlier_reproj_err) or None."""
    cams_with_data = [c for c in cams if raw_by_cam.get(c["label"], np.zeros((0, 2))).shape[0] > 0]
    seed_cams = (
        cams_with_data
        if _SEED_FROM_ALL_CAMERAS
        else [c for c in cams_with_data if c["label"] in _TRUSTED_CAMERA_LABELS]
    )
    if len(seed_cams) < 2:
        return None

    best = None  # (n_inliers, mean_err, X, inlier_cams)
    for i in range(len(seed_cams)):
        cam_i = seed_cams[i]
        for a in range(raw_by_cam[cam_i["label"]].shape[0]):
            for j in range(i + 1, len(seed_cams)):
                cam_j = seed_cams[j]
                for b in range(raw_by_cam[cam_j["label"]].shape[0]):
                    X = triangulate_dlt(
                        [normalized_by_cam[cam_i["label"]][a], normalized_by_cam[cam_j["label"]][b]],
                        [(cam_i["R"], cam_i["t"]), (cam_j["R"], cam_j["t"])],
                    )
                    inlier_cams, inlier_errs = [], []
                    for cam in cams_with_data:
                        errs = reproj_errors(X, cam["P"], raw_by_cam[cam["label"]])
                        k = int(np.argmin(errs))
                        if errs[k] < _INLIER_REPROJ_PX:
                            inlier_cams.append((cam, k))
                            inlier_errs.append(errs[k])
                    n = len(inlier_cams)
                    mean_err = float(np.mean(inlier_errs)) if inlier_errs else 1e9
                    if best is None or n > best[0] or (n == best[0] and mean_err < best[1]):
                        best = (n, mean_err, X, inlier_cams)

    if best is None or best[0] < _MIN_SUPPORTING_CAMERAS:
        return None
    n, mean_err, X, inlier_cams = best
    # Refine using every supporting camera, not just the RANSAC seed pair.
    normalized_inliers = [normalized_by_cam[cam["label"]][k] for cam, k in inlier_cams]
    extrinsics = [(cam["R"], cam["t"]) for cam, _ in inlier_cams]
    X_refined = triangulate_dlt(normalized_inliers, extrinsics)
    errs = [
        reproj_errors(X_refined, cam["P"], raw_by_cam[cam["label"]][k : k + 1])[0]
        for cam, k in inlier_cams
    ]
    return X_refined, n, float(np.mean(errs))


# Even with 2-camera-agreement RANSAC (solve_step), a small number of
# results still land on a wrong-but-self-consistent 2D pairing: found by
# eyeballing throw4's own output, where roughly one point in six landed
# repeatedly on a *second*, physically implausible cluster (~9m from the
# main trajectory) instead of jumping around randomly -- almost certainly
# two independent, intermittently-detected static clutter sources (one per
# camera) that happen to both appear often enough in the same frames to
# keep re-triangulating to nearly the same wrong point, while each stays
# below the per-camera _STATIC_FRAC cutoff on its own. A real thrown ball
# cannot teleport between two positions ~9m apart between consecutive
# available frames, so this is a temporal-continuity problem, not a
# per-frame one: build a "compatible" graph over every result pair whose
# implied speed is physically plausible, and keep only the largest
# connected component -- the real, continuously-connected trajectory swamps
# an isolated recurring artifact, which mostly only ever connects to itself.
_MAX_BALL_SPEED_MPS = 12.0
_CONTINUITY_LOOKAHEAD = 10  # results, not frames -- bridges real detection gaps


def filter_by_continuity(results: np.ndarray) -> np.ndarray:
    n = results.shape[0]
    if n < 2:
        return results
    order = np.argsort(results[:, 0])
    results = results[order]
    t, pos = results[:, 0], results[:, 1:4]

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, min(i + 1 + _CONTINUITY_LOOKAHEAD, n)):
            dt = t[j] - t[i]
            if dt <= 0:
                continue
            speed = float(np.linalg.norm(pos[j] - pos[i])) / dt
            if speed <= _MAX_BALL_SPEED_MPS:
                union(i, j)

    sizes = Counter(find(i) for i in range(n))
    best_root = max(sizes, key=lambda r: sizes[r])
    keep = [i for i in range(n) if find(i) == best_root]
    return results[keep]


def process_throw(
    conn: sqlite3.Connection,
    cams: list[dict],
    name: str,
    step0: int,
    step1: int,
    min_area: float,
    use_tracker: bool = True,
) -> np.ndarray:
    core_t0, core_t1 = step_to_time(step0), step_to_time(step1)
    step0 -= _PAD_STEPS
    step1 += _PAD_STEPS
    t0, t1 = step_to_time(step0), step_to_time(step1)

    by_cam: dict[str, dict[int, np.ndarray]] = {}
    for cam in cams:
        if use_tracker:
            by_cam[cam["label"]] = load_ball_candidates_by_step_tracked(
                conn, cam["camera_instance_id"], t0, t1, min_area, core_t0, core_t1
            )
        else:
            by_cam[cam["label"]] = load_ball_candidates_by_step(
                conn, cam["camera_instance_id"], t0, t1, min_area
            )
        n_frames = sum(v.shape[0] for v in by_cam[cam["label"]].values())
        print(
            f"  {cam['label']:20s} {len(by_cam[cam['label']]):4d} frames with a candidate, "
            f"{n_frames} candidates total"
        )

    results = []
    for step in range(step0, step1 + 1):
        raw_by_cam = {lbl: d.get(step, np.zeros((0, 2))) for lbl, d in by_cam.items()}
        normalized_by_cam = {
            cam["label"]: undistort_batch(raw_by_cam[cam["label"]], cam["K"], cam["dist"])
            for cam in cams
        }
        solved = solve_step(cams, raw_by_cam, normalized_by_cam)
        if solved is None:
            continue
        X, n_support, mean_err = solved
        results.append((step_to_time(step), *X, n_support, mean_err))

    results = np.array(results) if results else np.zeros((0, 6))
    total_steps = step1 - step0 + 1
    n_raw = results.shape[0]
    results = filter_by_continuity(results)
    n_dropped = n_raw - results.shape[0]
    print(
        f"  -> {n_raw}/{total_steps} steps positioned (>= {_MIN_SUPPORTING_CAMERAS} cameras agreeing), "
        f"{n_dropped} dropped as temporally-disconnected outliers -> {results.shape[0]} final"
    )
    if results.shape[0]:
        print(
            f"     mean reproj err: median {np.median(results[:,5]):.1f}px, "
            f"max {results[:,5].max():.1f}px; "
            f"n_cameras: median {np.median(results[:,4]):.0f}, max {results[:,4].max():.0f}"
        )
    return results


def main() -> None:
    global _INLIER_REPROJ_PX, _MIN_SUPPORTING_CAMERAS, _SEED_FROM_ALL_CAMERAS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA)
    ap.add_argument("--inlier-reproj-px", type=float, default=_INLIER_REPROJ_PX)
    ap.add_argument("--min-supporting-cameras", type=int, default=_MIN_SUPPORTING_CAMERAS)
    ap.add_argument(
        "--seed-from-all-cameras",
        dest="seed_from_all_cameras",
        action="store_true",
        default=_SEED_FROM_ALL_CAMERAS,
    )
    ap.add_argument(
        "--no-seed-from-all-cameras", dest="seed_from_all_cameras", action="store_false"
    )
    ap.add_argument("--throw", default=None, help="Only process this throw name (for quick iteration)")
    ap.add_argument(
        "--no-tracker",
        dest="use_tracker",
        action="store_false",
        default=True,
        help="Use the old flat-area-threshold candidate loader instead of the "
        "motion-gated continuation tracker (for comparison).",
    )
    args = ap.parse_args()

    _INLIER_REPROJ_PX = args.inlier_reproj_px
    _MIN_SUPPORTING_CAMERAS = args.min_supporting_cameras
    _SEED_FROM_ALL_CAMERAS = args.seed_from_all_cameras

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(f"file:{SESSION_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    all_cams = load_cameras_from_session(SESSION_DB, EXTRINSIC_CALIBRATION_ID, SESSION_ID)
    cams = [c for c in all_cams if c["label"] not in _EXCLUDED_CAMERA_LABELS]
    print(
        f"Loaded {len(all_cams)} cameras, using {len(cams)} "
        f"(excluded {sorted(_EXCLUDED_CAMERA_LABELS)}: known-unreliable for this ball, see status.md)"
    )

    for name, step0, step1 in THROWS:
        if args.throw and name != args.throw:
            continue
        print(f"\n=== {name} (steps {step0}-{step1}) ===")
        results = process_throw(conn, cams, name, step0, step1, args.min_area, args.use_tracker)
        out_path = out_dir / f"{name}.csv"
        header = "t_s,x,y,z,n_cameras,mean_reproj_err_px"
        np.savetxt(out_path, results, delimiter=",", header=header, comments="")
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
