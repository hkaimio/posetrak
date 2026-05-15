"""extrinsics_solver.py — Semi-automatic multi-camera extrinsics calibration.

Pipeline
--------
1. SIFT detect + match all camera pairs (essential matrix, RANSAC).
2. BFS spanning tree → global poses (scale-free, root-camera frame).
3. Triangulate inlier point pairs → initial 3D cloud.
4. Bundle adjustment (scipy) → refined poses + 3D points.
5. Optional similarity transform → physical coordinate system.

All state is kept in memory; only the final Pose2Sim TOML is persisted.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CamCalibState:
    video_id: str
    label: str
    K: np.ndarray        # K_new — undistorted camera matrix (3×3)
    K_orig: np.ndarray   # K_original — needed to undistort raw frames
    dist: np.ndarray     # original distortion coefficients
    fisheye: bool
    image: np.ndarray | None = None  # BGR full-res; None until loaded
    R: np.ndarray | None = None      # world→cam rotation (3×3)
    t: np.ndarray | None = None      # world→cam translation (3×1)


@dataclass
class ControlPoint:
    name: str
    # distorted pixel observations per camera; solver undistorts before use
    obs: dict[str, tuple[float, float]] = field(default_factory=dict)
    # if set, this point's 3D position is fixed in the BA
    world_xyz: np.ndarray | None = None


@dataclass
class PairMatch:
    vid_a: str
    vid_b: str
    R_rel: np.ndarray    # rotation from cam-A frame to cam-B frame (3×3)
    t_rel: np.ndarray    # unit translation in cam-A frame (3×1)
    pts_a: np.ndarray    # Nx2 undistorted pixels in cam A (after pose inlier filter)
    pts_b: np.ndarray    # Nx2 undistorted pixels in cam B
    n_inliers: int


@dataclass
class CalibResult:
    cameras: dict[str, CamCalibState]
    # list of (xyz_world, {video_id: (px_undist, py_undist)})
    points_3d: list[tuple[np.ndarray, dict[str, tuple[float, float]]]]
    reprojection_errors: dict[str, dict]   # video_id → {mean, std, max, n}
    unsolved: list[str]                    # video_ids with no pose


# ---------------------------------------------------------------------------
# Image undistortion
# ---------------------------------------------------------------------------


def undistort_image(img: np.ndarray, state: CamCalibState) -> np.ndarray:
    """Return undistorted version of *img* using the camera's calibration."""
    if state.fisheye:
        h, w = img.shape[:2]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            state.K_orig, state.dist, np.eye(3), state.K, (w, h), cv2.CV_32FC1
        )
        return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    return cv2.undistort(img, state.K_orig, state.dist, None, state.K)


def _undistort_pts(pts: np.ndarray, state: CamCalibState) -> np.ndarray:
    """Undistort Nx2 distorted pixel array → Nx2 undistorted pixels."""
    if pts.shape[0] == 0:
        return pts
    p = pts.reshape(-1, 1, 2).astype(np.float32)
    if state.fisheye:
        out = cv2.fisheye.undistortPoints(p, state.K_orig, state.dist, None, state.K)
    else:
        out = cv2.undistortPoints(p, state.K_orig, state.dist, None, state.K)
    return out.reshape(-1, 2)


def _normalise_pts(pts_undist: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert undistorted pixels → normalised pinhole rays (divide by K)."""
    if pts_undist.shape[0] == 0:
        return pts_undist
    p = pts_undist.reshape(-1, 1, 2).astype(np.float64)
    out = cv2.undistortPoints(p, K, np.zeros(4))  # zero dist → just normalise
    return out.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Stage 1 — SIFT matching + essential matrix
# ---------------------------------------------------------------------------


def _detect_sift(img_undist: np.ndarray) -> tuple[list, np.ndarray | None]:
    sift = cv2.SIFT_create(nfeatures=4000)
    gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY) if img_undist.ndim == 3 else img_undist
    kp, desc = sift.detectAndCompute(gray, None)
    return kp, desc


def match_pair(
    state_a: CamCalibState,
    state_b: CamCalibState,
    ratio: float = 0.75,
    min_inliers: int = 20,
    ransac_threshold: float = 0.001,
) -> PairMatch | None:
    """SIFT match two cameras → PairMatch or None if insufficient overlap.

    Images must be loaded (state.image is not None).
    """
    if state_a.image is None or state_b.image is None:
        raise ValueError("Images must be loaded before matching")

    img_a = undistort_image(state_a.image, state_a)
    img_b = undistort_image(state_b.image, state_b)

    kp_a, desc_a = _detect_sift(img_a)
    kp_b, desc_b = _detect_sift(img_b)
    _log.debug("match_pair %s↔%s: %d / %d keypoints", state_a.label, state_b.label, len(kp_a), len(kp_b))

    if desc_a is None or desc_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return None

    matcher = cv2.BFMatcher()
    raw = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    _log.debug("  ratio-test survivors: %d", len(good))

    if len(good) < 8:
        return None

    # Pixel coordinates (undistorted image space)
    pts_a_px = np.float64([kp_a[m.queryIdx].pt for m in good])
    pts_b_px = np.float64([kp_b[m.trainIdx].pt for m in good])

    # Normalise by K for essential matrix (heterogeneous cameras)
    pts_a_n = _normalise_pts(pts_a_px, state_a.K)
    pts_b_n = _normalise_pts(pts_b_px, state_b.K)

    E, mask_e = cv2.findEssentialMat(
        pts_a_n, pts_b_n, np.eye(3),
        method=cv2.RANSAC, prob=0.999, threshold=ransac_threshold,
    )
    if E is None or mask_e is None:
        return None

    inl_e = mask_e.ravel().astype(bool)
    if inl_e.sum() < min_inliers:
        _log.debug("  essential-matrix inliers %d < %d, skipping pair", inl_e.sum(), min_inliers)
        return None

    pts_a_in = pts_a_n[inl_e]
    pts_b_in = pts_b_n[inl_e]
    pts_a_px_in = pts_a_px[inl_e]
    pts_b_px_in = pts_b_px[inl_e]

    _, R_rel, t_rel, mask_p = cv2.recoverPose(E, pts_a_in, pts_b_in, np.eye(3))
    pose_inl = mask_p.ravel().astype(bool)
    n_inliers = int(pose_inl.sum())
    if n_inliers < min_inliers:
        _log.debug("  cheirality inliers %d < %d, skipping pair", n_inliers, min_inliers)
        return None

    _log.info("match_pair %s↔%s: %d inliers", state_a.label, state_b.label, n_inliers)
    return PairMatch(
        vid_a=state_a.video_id,
        vid_b=state_b.video_id,
        R_rel=R_rel,
        t_rel=t_rel,
        pts_a=pts_a_px_in[pose_inl],
        pts_b=pts_b_px_in[pose_inl],
        n_inliers=n_inliers,
    )


def match_all_pairs(
    states: list[CamCalibState],
    ratio: float = 0.75,
    min_inliers: int = 20,
    ransac_threshold: float = 0.001,
) -> dict[tuple[str, str], PairMatch]:
    """Match every unordered camera pair. Returns only successful pairs."""
    results: dict[tuple[str, str], PairMatch] = {}
    for i, sa in enumerate(states):
        for sb in states[i + 1:]:
            pm = match_pair(sa, sb, ratio=ratio, min_inliers=min_inliers,
                            ransac_threshold=ransac_threshold)
            if pm is not None:
                results[(sa.video_id, sb.video_id)] = pm
    return results


# ---------------------------------------------------------------------------
# Stage 2 — BFS spanning tree → global poses
# ---------------------------------------------------------------------------


def chain_poses_bfs(
    states: list[CamCalibState],
    pair_matches: dict[tuple[str, str], PairMatch],
) -> list[str]:
    """Set R, t on each reachable CamCalibState.  Returns list of unsolved IDs.

    The root camera (most edges) is placed at the origin (R=I, t=0).
    t values are unit-length (scale-free) — BA and similarity transform fix scale.
    """
    # Build adjacency: (neighbour_id, R_to_nb, t_to_nb)
    adj: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = defaultdict(list)
    for (va, vb), pm in pair_matches.items():
        adj[va].append((vb, pm.R_rel, pm.t_rel))
        # Inverse: B→A
        R_ba = pm.R_rel.T
        t_ba = -pm.R_rel.T @ pm.t_rel
        adj[vb].append((va, R_ba, t_ba))

    state_by_id = {s.video_id: s for s in states}

    # Root: camera with most matched neighbours
    all_ids = [s.video_id for s in states]
    root = max(all_ids, key=lambda v: len(adj[v]))

    state_by_id[root].R = np.eye(3)
    state_by_id[root].t = np.zeros((3, 1))

    queue: deque[str] = deque([root])
    while queue:
        curr = queue.popleft()
        R_c = state_by_id[curr].R
        t_c = state_by_id[curr].t
        for nb_id, R_rel, t_rel in adj[curr]:
            if state_by_id[nb_id].R is not None:
                continue
            state_by_id[nb_id].R = R_rel @ R_c
            state_by_id[nb_id].t = R_rel @ t_c + t_rel.reshape(3, 1)
            queue.append(nb_id)

    unsolved = [v for v in all_ids if state_by_id[v].R is None]
    solved = len(all_ids) - len(unsolved)
    _log.info("chain_poses_bfs: %d/%d cameras solved (root=%s)", solved, len(all_ids), root)
    if unsolved:
        _log.warning("unsolved cameras: %s", unsolved)
    return unsolved


# ---------------------------------------------------------------------------
# Stage 3 — Triangulation
# ---------------------------------------------------------------------------


def _proj_matrix(state: CamCalibState) -> np.ndarray:
    """Build 3×4 projection matrix K [R|t]."""
    Rt = np.hstack([state.R, state.t.reshape(3, 1)])
    return state.K @ Rt


def triangulate_pair(
    state_a: CamCalibState,
    state_b: CamCalibState,
    pm: PairMatch,
    max_reprojection_px: float = 10.0,
) -> list[tuple[np.ndarray, dict[str, tuple[float, float]]]]:
    """Triangulate inlier matches for one pair.  Returns list of (xyz, obs_dict)."""
    if state_a.R is None or state_b.R is None:
        return []

    P_a = _proj_matrix(state_a)
    P_b = _proj_matrix(state_b)

    pts4d = cv2.triangulatePoints(P_a, P_b, pm.pts_a.T, pm.pts_b.T)
    w = pts4d[3]
    valid_w = np.abs(w) > 1e-8
    pts3d = np.where(valid_w, pts4d[:3] / np.where(valid_w, w, 1), np.nan).T  # N×3

    # Depth filter: both cameras must see the point in front
    depth_a = (state_a.R @ pts3d.T + state_a.t.reshape(3, 1))[2]
    depth_b = (state_b.R @ pts3d.T + state_b.t.reshape(3, 1))[2]
    keep = (depth_a > 0) & (depth_b > 0) & valid_w

    # Reprojection filter
    def _reproj_err(P, pts3d_kept, obs):
        h = (P @ np.hstack([pts3d_kept, np.ones((len(pts3d_kept), 1))]).T)
        proj = (h[:2] / h[2]).T
        return np.linalg.norm(proj - obs, axis=1)

    pts3d_k = pts3d[keep]
    if len(pts3d_k) == 0:
        return []

    obs_a_k = pm.pts_a[keep]
    obs_b_k = pm.pts_b[keep]
    err_a = _reproj_err(P_a, pts3d_k, obs_a_k)
    err_b = _reproj_err(P_b, pts3d_k, obs_b_k)
    good = (err_a < max_reprojection_px) & (err_b < max_reprojection_px)

    points = []
    for xyz, pa, pb in zip(pts3d_k[good], obs_a_k[good], obs_b_k[good]):
        obs = {
            pm.vid_a: (float(pa[0]), float(pa[1])),
            pm.vid_b: (float(pb[0]), float(pb[1])),
        }
        points.append((xyz, obs))

    _log.debug("triangulate_pair %s↔%s: %d/%d points kept",
               pm.vid_a, pm.vid_b, len(points), keep.sum())
    return points


def triangulate_all_pairs(
    states: list[CamCalibState],
    pair_matches: dict[tuple[str, str], PairMatch],
    max_reprojection_px: float = 10.0,
) -> list[tuple[np.ndarray, dict[str, tuple[float, float]]]]:
    state_by_id = {s.video_id: s for s in states}
    all_points: list[tuple[np.ndarray, dict]] = []
    for (va, vb), pm in pair_matches.items():
        pts = triangulate_pair(state_by_id[va], state_by_id[vb], pm, max_reprojection_px)
        all_points.extend(pts)
    _log.info("triangulate_all_pairs: %d 3D points total", len(all_points))
    return all_points


# ---------------------------------------------------------------------------
# Stage 4 — Bundle adjustment
# ---------------------------------------------------------------------------


def _undistort_control_obs(
    cp: ControlPoint,
    state_by_id: dict[str, CamCalibState],
) -> dict[str, tuple[float, float]]:
    """Return undistorted pixel coords for a control point's observations."""
    result = {}
    for vid, (px, py) in cp.obs.items():
        if vid not in state_by_id:
            continue
        pts_u = _undistort_pts(np.array([[px, py]], dtype=np.float32), state_by_id[vid])
        result[vid] = (float(pts_u[0, 0]), float(pts_u[0, 1]))
    return result


def run_bundle_adjustment(
    states: list[CamCalibState],
    points_3d: list[tuple[np.ndarray, dict[str, tuple[float, float]]]],
    control_points: list[ControlPoint] | None = None,
    fixed_point_weight: float = 10.0,
    max_nfev: int = 2000,
) -> list[tuple[np.ndarray, dict[str, tuple[float, float]]]]:
    """Jointly optimise camera poses and free 3D points.

    Modifies state.R / state.t in place on each solved camera.
    Returns refined points_3d list.
    Control points with world_xyz are held fixed (heavy residual weight).
    """
    solved = [s for s in states if s.R is not None]
    if not solved:
        return points_3d

    state_by_id = {s.video_id: s for s in states}
    cam_ids = [s.video_id for s in solved]
    n_cams = len(cam_ids)
    cam_idx = {v: i for i, v in enumerate(cam_ids)}

    # --------------- build parameter vector ---------------
    # [rvec0(3), tvec0(3), ..., point0(3), point1(3), ..., fixed points NOT included]
    params: list[float] = []

    cam_param_start: dict[str, int] = {}
    for s in solved:
        cam_param_start[s.video_id] = len(params)
        rvec, _ = cv2.Rodrigues(s.R)
        params.extend(rvec.flatten())
        params.extend(s.t.flatten())

    free_point_start: list[int] = []
    for xyz, _ in points_3d:
        free_point_start.append(len(params))
        params.extend(xyz.tolist())

    params_arr = np.array(params, dtype=np.float64)

    # --------------- build observations ---------------
    obs_list: list[tuple[str, np.ndarray, str, bool]] = []
    # (video_id, obs_pixel (2,), point_id_or_idx, is_fixed)

    # Free triangulated points
    for pt_idx, (_, obs_dict) in enumerate(points_3d):
        for vid, (px, py) in obs_dict.items():
            if vid in cam_idx:
                obs_list.append((vid, np.array([px, py]), f"free_{pt_idx}", False))

    # Control points
    fixed_pts: dict[str, np.ndarray] = {}  # cp.name → world_xyz
    cp_obs_undist: list[tuple[str, dict]] = []
    if control_points:
        for cp in control_points:
            undist = _undistort_control_obs(cp, state_by_id)
            if cp.world_xyz is not None:
                fixed_pts[cp.name] = cp.world_xyz.astype(np.float64)
            cp_obs_undist.append((cp.name, undist))
            for vid, (px, py) in undist.items():
                if vid in cam_idx:
                    obs_list.append((vid, np.array([px, py]), f"cp_{cp.name}", cp.world_xyz is not None))

    n_obs = len(obs_list)
    _log.info("BA: %d cameras, %d free points, %d fixed points, %d observations",
              n_cams, len(points_3d), len(fixed_pts), n_obs)

    if n_obs < 6:
        _log.warning("BA: too few observations (%d), skipping", n_obs)
        return points_3d

    def _get_point_3d(pt_key: str, x: np.ndarray) -> np.ndarray:
        if pt_key.startswith("free_"):
            idx = int(pt_key[5:])
            start = free_point_start[idx]
            return x[start:start + 3]
        # control point (cp_<name>)
        name = pt_key[3:]
        if name in fixed_pts:
            return fixed_pts[name]
        # free control point — not yet supported (treat as free using first obs mean)
        return np.zeros(3)

    def residuals(x: np.ndarray) -> np.ndarray:
        res = np.empty(n_obs * 2)
        for i, (vid, obs_px, pt_key, is_fixed) in enumerate(obs_list):
            ci = cam_param_start[vid]
            rvec = x[ci:ci + 3]
            tvec = x[ci + 3:ci + 6]
            pt3d = _get_point_3d(pt_key, x)
            K = state_by_id[vid].K
            proj, _ = cv2.projectPoints(
                pt3d.reshape(1, 3), rvec, tvec, K, np.zeros(4)
            )
            r = (proj.reshape(2) - obs_px)
            weight = fixed_point_weight if is_fixed else 1.0
            res[2 * i: 2 * i + 2] = r * weight
        return res

    result = least_squares(
        residuals, params_arr,
        method="trf", loss="huber",
        ftol=1e-6, xtol=1e-6,
        max_nfev=max_nfev,
        verbose=0,
    )
    _log.info("BA done: cost %.4f → %.4f (%d iters)",
              0.5 * np.sum(residuals(params_arr) ** 2),
              result.cost, result.nfev)

    x = result.x

    # Update camera poses
    for s in solved:
        ci = cam_param_start[s.video_id]
        rvec = x[ci:ci + 3]
        tvec = x[ci + 3:ci + 6]
        R, _ = cv2.Rodrigues(rvec)
        s.R = R
        s.t = tvec.reshape(3, 1)

    # Update free points
    refined: list[tuple[np.ndarray, dict]] = []
    for idx, (_, obs_dict) in enumerate(points_3d):
        start = free_point_start[idx]
        xyz = x[start:start + 3].copy()
        refined.append((xyz, obs_dict))

    return refined


# ---------------------------------------------------------------------------
# Stage 5 — Similarity transform
# ---------------------------------------------------------------------------


def apply_similarity_transform(
    states: list[CamCalibState],
    points_3d: list[tuple[np.ndarray, dict]],
    scale: float,
    R_align: np.ndarray,
    t_align: np.ndarray,
) -> list[tuple[np.ndarray, dict]]:
    """Apply T(x) = scale * R_align @ x + t_align to 3D scene.

    Camera poses are updated in place.  Returns transformed points_3d.
    scale, R_align, t_align can be computed from helper functions below.
    """
    for s in states:
        if s.R is None:
            continue
        C = -s.R.T @ s.t.reshape(3, 1)               # camera centre, old world
        C_new = scale * R_align @ C + t_align.reshape(3, 1)
        s.R = s.R @ R_align.T
        s.t = -s.R @ C_new

    transformed = [
        (scale * R_align @ xyz + t_align.reshape(3,), obs)
        for xyz, obs in points_3d
    ]
    return transformed


def similarity_from_two_points(
    p1_world: np.ndarray,
    p2_world: np.ndarray,
    known_distance: float,
) -> float:
    """Return scale factor so that |T(p1) - T(p2)| == known_distance."""
    current = float(np.linalg.norm(p2_world - p1_world))
    if current < 1e-9:
        raise ValueError("Points are coincident — cannot determine scale")
    return known_distance / current


def similarity_from_floor_plane(
    floor_pts_world: list[np.ndarray],
    forward_pt_world: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate R_align, t_align so that the floor becomes Z=0, Z points up.

    floor_pts_world: ≥3 points on the floor in the current world frame.
    forward_pt_world: optional point in the +X direction from the centroid.

    Returns (R_align, t_align) for use in apply_similarity_transform (scale=1).
    The centroid of floor_pts becomes the new origin.
    """
    pts = np.array(floor_pts_world)
    centroid = pts.mean(axis=0)

    # Fit plane normal via SVD
    _, _, Vt = np.linalg.svd(pts - centroid)
    normal = Vt[-1]              # smallest singular value = plane normal
    if normal[2] < 0:            # ensure Z_new points up (away from camera side)
        normal = -normal

    # Build rotation: Z_new = normal, X_new = toward forward_pt (projected onto floor)
    z_new = normal / np.linalg.norm(normal)
    if forward_pt_world is not None:
        x_cand = forward_pt_world - centroid
        x_cand -= np.dot(x_cand, z_new) * z_new  # project onto floor plane
        norm_x = np.linalg.norm(x_cand)
        x_new = x_cand / norm_x if norm_x > 1e-9 else np.array([1.0, 0.0, 0.0])
    else:
        # Arbitrary +X: pick any vector perpendicular to z_new
        tmp = np.array([0.0, 0.0, 1.0]) if abs(z_new[0]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x_new = np.cross(tmp, z_new)
        x_new /= np.linalg.norm(x_new)

    y_new = np.cross(z_new, x_new)
    R_align = np.stack([x_new, y_new, z_new], axis=0)   # rows = new axes in old frame
    t_align = -R_align @ centroid
    return R_align, t_align


# ---------------------------------------------------------------------------
# Reprojection error reporting
# ---------------------------------------------------------------------------


def compute_reprojection_errors(
    states: list[CamCalibState],
    points_3d: list[tuple[np.ndarray, dict[str, tuple[float, float]]]],
) -> dict[str, dict]:
    state_by_id = {s.video_id: s for s in states}
    per_cam: dict[str, list[float]] = defaultdict(list)

    for xyz, obs_dict in points_3d:
        for vid, (px, py) in obs_dict.items():
            s = state_by_id.get(vid)
            if s is None or s.R is None:
                continue
            rvec, _ = cv2.Rodrigues(s.R)
            proj, _ = cv2.projectPoints(
                xyz.reshape(1, 3), rvec, s.t.reshape(3, 1), s.K, np.zeros(4)
            )
            err = float(np.linalg.norm(proj.reshape(2) - np.array([px, py])))
            per_cam[vid].append(err)

    result = {}
    for vid, errs in per_cam.items():
        e = np.array(errs)
        result[vid] = {"mean": float(e.mean()), "std": float(e.std()),
                       "max": float(e.max()), "n": len(e)}
    return result


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_calibration(
    states: list[CamCalibState],
    control_points: list[ControlPoint] | None = None,
    sift_ratio: float = 0.75,
    sift_min_inliers: int = 20,
    ba_max_nfev: int = 2000,
) -> CalibResult:
    """Run the full pipeline.  states[*].image must be loaded before calling."""
    pair_matches = match_all_pairs(states, ratio=sift_ratio, min_inliers=sift_min_inliers)
    unsolved = chain_poses_bfs(states, pair_matches)
    points_3d = triangulate_all_pairs(states, pair_matches)
    points_3d = run_bundle_adjustment(states, points_3d, control_points, max_nfev=ba_max_nfev)
    errors = compute_reprojection_errors(states, points_3d)
    return CalibResult(
        cameras={s.video_id: s for s in states},
        points_3d=points_3d,
        reprojection_errors=errors,
        unsolved=unsolved,
    )


# ---------------------------------------------------------------------------
# TOML output (Pose2Sim format)
# ---------------------------------------------------------------------------


def to_toml_string(result: CalibResult) -> str:
    """Render solved cameras as a Pose2Sim-compatible TOML string."""
    lines: list[str] = []
    for vid, s in result.cameras.items():
        if s.R is None:
            _log.warning("Camera %s (%s) has no pose, skipping", vid, s.label)
            continue
        rvec, _ = cv2.Rodrigues(s.R)
        K = s.K
        dist = np.zeros(4)
        sz = (s.image.shape[1], s.image.shape[0]) if s.image is not None else (0, 0)
        lines += [
            f"[{s.label}]",
            f'name = "{s.label}"',
            f"size = [ {sz[0]}, {sz[1]}]",
            f"matrix = [ [ {K[0,0]:.6f}, 0.0, {K[0,2]:.6f}], "
            f"[ 0.0, {K[1,1]:.6f}, {K[1,2]:.6f}], [ 0.0, 0.0, 1.0]]",
            f"distortions = [ {dist[0]}, {dist[1]}, {dist[2]}, {dist[3]}]",
            f"rotation = [ {rvec[0,0]:.8f}, {rvec[1,0]:.8f}, {rvec[2,0]:.8f}]",
            f"translation = [ {s.t[0,0]:.8f}, {s.t[1,0]:.8f}, {s.t[2,0]:.8f}]",
            "fisheye = false",
            "",
        ]
    lines += ["[metadata]", "adjusted = true", "error = 0.0", ""]
    return "\n".join(lines)
