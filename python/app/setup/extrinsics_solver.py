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
from scipy.sparse import lil_matrix

_log = logging.getLogger(__name__)


class _Cancelled(Exception):
    """Raised inside the solver when a cancel event fires; caught by callers."""


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
    reprojection_errors: dict[str, dict]      # video_id → {mean, std, max, n}
    unsolved: list[str]                       # video_ids with no pose
    pair_matches: dict[tuple[str, str], PairMatch]  # SIFT matches per camera pair
    cp_reprojection_errors: dict[str, dict] = field(default_factory=dict)  # CP residuals


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
    progress_cb=None,
    cancel_event=None,
) -> dict[tuple[str, str], PairMatch]:
    """Match every unordered camera pair. Returns only successful pairs."""
    results: dict[tuple[str, str], PairMatch] = {}
    n = len(states)
    total = n * (n - 1) // 2
    done = 0
    for i, sa in enumerate(states):
        for sb in states[i + 1:]:
            if cancel_event is not None and cancel_event.is_set():
                raise _Cancelled()
            done += 1
            if progress_cb:
                progress_cb(f"SIFT matching {sa.label} ↔ {sb.label} ({done}/{total})…")
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

    all_ids = [s.video_id for s in states]

    # Prefer a camera already initialised (e.g. via PnP) as BFS root so that
    # the SIFT chain starts from a world-frame anchor.
    pnp_ids = [v for v in all_ids if state_by_id[v].R is not None]
    if pnp_ids:
        # Pick PnP root with most SIFT edges so the BFS covers the most cameras.
        root = max(pnp_ids, key=lambda v: len(adj[v]))
        _log.info("chain_poses_bfs: using PnP-initialised root %s", root)
    else:
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


def filter_sift_by_cheirality(
    states: list[CamCalibState],
    pair_matches: dict[tuple[str, str], PairMatch],
    min_pos_ratio: float = 0.5,
) -> dict[tuple[str, str], PairMatch]:
    """Remove SIFT pairs where triangulated points mostly fail the cheirality test.

    When two cameras don't actually share a view, RANSAC may still find a
    geometrically-consistent essential matrix for visually-similar (but semantically
    wrong) matches.  After BFS gives approximate poses, we triangulate each pair's
    matches: if most points end up behind one of the cameras the pair is rejected.

    min_pos_ratio: fraction of triangulated points that must have positive depth in
                   BOTH cameras. Pairs below this threshold are removed.
    """
    state_by_id = {s.video_id: s for s in states}
    filtered: dict[tuple[str, str], PairMatch] = {}

    for (va, vb), pm in pair_matches.items():
        sa = state_by_id[va]
        sb = state_by_id[vb]
        if sa.R is None or sb.R is None:
            continue

        Pa = _proj_matrix(sa)
        Pb = _proj_matrix(sb)
        pts4d = cv2.triangulatePoints(Pa, Pb, pm.pts_a.T, pm.pts_b.T)
        w = pts4d[3]
        valid = np.abs(w) > 1e-8
        if not np.any(valid):
            _log.info("filter_sift: reject %s↔%s — all triangulations degenerate", va, vb)
            continue

        pts3d = (pts4d[:3] / w).T
        depth_a = (sa.R @ pts3d[valid].T + sa.t.reshape(3, 1))[2]
        depth_b = (sb.R @ pts3d[valid].T + sb.t.reshape(3, 1))[2]
        pos_ratio = float(((depth_a > 0) & (depth_b > 0)).sum()) / valid.sum()

        if pos_ratio >= min_pos_ratio:
            filtered[(va, vb)] = pm
        else:
            _log.info(
                "filter_sift: reject %s↔%s — only %.0f%% of triangulated pts in front "
                "(cameras likely non-overlapping)",
                va, vb, pos_ratio * 100,
            )

    _log.info("filter_sift_by_cheirality: kept %d/%d pairs", len(filtered), len(pair_matches))
    return filtered


def init_poses_pnp(
    states: list[CamCalibState],
    control_points: list[ControlPoint],
    min_cps: int = 4,
    pnp_ransac_px: float = 8.0,
) -> list[str]:
    """Initialise camera poses via PnP for cameras with world_xyz control points.

    Uses cv2.solvePnP on the undistorted CP observations.  Camera must have
    ≥ min_cps observations of CPs that have world_xyz set.
    Returns list of video_ids that were initialised (R, t set in-place).
    """
    initialised: list[str] = []

    for s in states:
        obj_pts: list[np.ndarray] = []
        img_pts: list[np.ndarray] = []
        cp_names: list[str] = []
        for cp in control_points:
            if cp.world_xyz is None or s.video_id not in cp.obs:
                continue
            px, py = cp.obs[s.video_id]
            undist = _undistort_pts(np.array([[px, py]], dtype=np.float32), s)
            _log.debug(
                "init_poses_pnp: %s / %s  world=(%.3f, %.3f, %.3f)  "
                "px_raw=(%.1f, %.1f)  px_undist=(%.1f, %.1f)  finite=%s",
                s.label, cp.name,
                cp.world_xyz[0], cp.world_xyz[1], cp.world_xyz[2],
                px, py,
                float(undist[0, 0]), float(undist[0, 1]),
                np.isfinite(undist).all(),
            )
            obj_pts.append(cp.world_xyz.astype(np.float64))
            img_pts.append(undist[0])
            cp_names.append(cp.name)

        if len(obj_pts) < min_cps:
            _log.debug(
                "init_poses_pnp: skip %s — only %d world CPs (need %d): observed=%s",
                s.label, len(obj_pts), min_cps, cp_names,
            )
            continue

        obj_arr = np.array(obj_pts, dtype=np.float64)
        img_arr = np.array(img_pts, dtype=np.float32)

        # Check for non-finite undistorted pixels (would silently corrupt PnP).
        if not np.isfinite(img_arr).all():
            _log.warning(
                "init_poses_pnp: %s — non-finite undistorted pixels for CPs %s; "
                "check intrinsics / distortion model",
                s.label, cp_names,
            )
            continue

        # Check world-point geometry: coplanar (sv[2]≈0) and co-linear (sv[1]≈0).
        centred = obj_arr - obj_arr.mean(axis=0)
        _, sv, _ = np.linalg.svd(centred)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 1e-9 else float("inf")
        cond12 = float(sv[0] / sv[1]) if sv[1] > 1e-9 else float("inf")
        if sv[1] < 1e-6 or cond12 > 1e4:
            _log.warning(
                "init_poses_pnp: %s — world CPs appear co-linear (σ[0]/σ[1]=%.1f); "
                "PnP is underdetermined — add CPs not on the same line",
                s.label, cond12,
            )
        elif cond > 1e6:
            _log.warning(
                "init_poses_pnp: %s — world CPs are coplanar (all z=0); "
                "PnP has two valid solutions (above/below floor)",
                s.label,
            )
        else:
            _log.debug("init_poses_pnp: %s — world CP σ ratios (%.1f, %.1f) — ok",
                       s.label, cond, cond12)

        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_arr, img_arr, s.K, np.zeros(4),
                iterationsCount=1000, reprojectionError=pnp_ransac_px,
            )
        except cv2.error as exc:
            _log.warning("init_poses_pnp: PnP raised cv2.error for %s: %s", s.label, exc)
            continue

        if not ok:
            # Diagnostic: try EPNP without RANSAC to get reprojection errors.
            try:
                _, rvec_d, tvec_d = cv2.solvePnP(
                    obj_arr, img_arr, s.K, np.zeros(4), flags=cv2.SOLVEPNP_EPNP,
                )
                proj, _ = cv2.projectPoints(obj_arr, rvec_d, tvec_d, s.K, np.zeros(4))
                proj = proj.reshape(-1, 2)
                errs = np.linalg.norm(proj - img_arr, axis=1)
                err_info = "  ".join(
                    f"{cp_names[i]}:{errs[i]:.1f}px" for i in range(len(cp_names))
                )
                _log.warning(
                    "init_poses_pnp: RANSAC failed for %s (EPNP reprojection errors: %s)",
                    s.label, err_info,
                )
            except cv2.error as exc2:
                _log.warning(
                    "init_poses_pnp: RANSAC failed for %s; EPNP diagnostic also failed: %s",
                    s.label, exc2,
                )
            continue

        n_inliers = len(inliers) if inliers is not None else len(obj_pts)
        R, _ = cv2.Rodrigues(rvec)

        # Validate rotation matrix: det≈-1 (reflection) or NaN tvec both indicate
        # a degenerate PnP result (common with coplanar/colinear points).
        det = float(np.linalg.det(R))
        if abs(det - 1.0) > 0.05:
            _log.warning(
                "init_poses_pnp: %s — RANSAC returned non-rotation (det=%.4f); skipping",
                s.label, det,
            )
            continue
        if not np.isfinite(tvec).all():
            _log.warning(
                "init_poses_pnp: %s — RANSAC returned non-finite tvec %s; skipping "
                "(likely co-linear world points — add CPs not on the same line)",
                s.label, tvec.flatten().tolist(),
            )
            continue

        # For coplanar configurations PnP has two valid solutions (camera above
        # and below the reference plane).  Prefer camera above (C_z > 0).
        # If RANSAC picked the below-floor solution, try IPPE to get the other.
        cam_center = -R.T @ tvec.flatten()
        if cam_center[2] < 0 and cond > 1e4:
            _log.debug(
                "init_poses_pnp: %s — camera below floor (C_z=%.3f); trying IPPE",
                s.label, float(cam_center[2]),
            )
            try:
                n_sol, rvecs_i, tvecs_i, _ = cv2.solvePnPGeneric(
                    obj_arr, img_arr, s.K, np.zeros(4),
                    flags=cv2.SOLVEPNP_IPPE,
                )
                _log.debug(
                    "init_poses_pnp: %s — IPPE returned %d solutions",
                    s.label, n_sol,
                )
                for k, (rv_i, tv_i) in enumerate(zip(rvecs_i, tvecs_i)):
                    R_i, _ = cv2.Rodrigues(rv_i)
                    det_i = float(np.linalg.det(R_i))
                    finite_i = np.isfinite(tv_i).all()
                    C_i = -R_i.T @ tv_i.flatten() if finite_i else np.full(3, float("nan"))
                    _log.debug(
                        "init_poses_pnp: %s — IPPE sol %d: C_z=%.3f det=%.4f finite=%s",
                        s.label, k, float(C_i[2]), det_i, finite_i,
                    )
                    if C_i[2] > 0 and abs(det_i - 1.0) < 0.05 and finite_i:
                        rvec, tvec = rv_i, tv_i
                        R = R_i
                        cam_center = C_i
                        _log.debug(
                            "init_poses_pnp: %s — switched to IPPE sol %d (C_z=%.3f)",
                            s.label, k, float(cam_center[2]),
                        )
                        break
                else:
                    _log.debug(
                        "init_poses_pnp: %s — no IPPE solution with C_z>0; keeping RANSAC result",
                        s.label,
                    )
            except cv2.error as exc_ippe:
                _log.debug("init_poses_pnp: %s — IPPE raised: %s", s.label, exc_ippe)

        _log.debug(
            "init_poses_pnp: %s — camera center (%.3f, %.3f, %.3f) det=%.4f",
            s.label, float(cam_center[0]), float(cam_center[1]), float(cam_center[2]), det,
        )

        s.R = R
        s.t = tvec.reshape(3, 1)
        initialised.append(s.video_id)
        _log.info(
            "init_poses_pnp: initialised %s from %d world CPs (%d RANSAC inliers, C_z=%.3f)",
            s.label, len(obj_pts), n_inliers, float(cam_center[2]),
        )

    return initialised


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
    fixed_cp_weight: float = 1000.0,
    unfixed_cp_weight: float = 100.0,
    max_nfev: int = 2000,
    max_sift_pts: int = 300,
    progress_cb=None,
    cancel_event=None,
) -> list[tuple[np.ndarray, dict[str, tuple[float, float]]]]:
    """Jointly optimise camera poses and free 3D points.

    Modifies state.R / state.t in place on each solved camera.
    Returns refined points_3d list.

    Fixed CPs (world_xyz set) are NOT part of the parameter vector — their
    world position is used directly as a constant with weight=fixed_cp_weight.
    Free CPs (no world_xyz) are triangulated from current camera estimates and
    added to the parameter vector with weight=unfixed_cp_weight.

    Uses loss='linear' (no robust kernel). Large CP weights (1000×/100×)
    relative to SIFT (1×) give CPs strong priority without fighting a robust
    loss that would down-weight them.
    """
    solved = [s for s in states if s.R is not None]
    if not solved:
        return points_3d

    if control_points and len(points_3d) > max_sift_pts:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points_3d), size=max_sift_pts, replace=False)
        points_3d = [points_3d[i] for i in sorted(idx)]
        _log.info("BA: subsampled SIFT points to %d", max_sift_pts)

    state_by_id = {s.video_id: s for s in states}
    cam_ids = [s.video_id for s in solved]
    n_cams = len(cam_ids)
    cam_idx = {v: i for i, v in enumerate(cam_ids)}

    # --- Camera params [rvec(3), tvec(3)] per camera ---
    params: list[float] = []
    cam_param_start: dict[str, int] = {}
    for s in solved:
        cam_param_start[s.video_id] = len(params)
        rvec, _ = cv2.Rodrigues(s.R)
        params.extend(rvec.flatten())
        params.extend(s.t.flatten())

    # --- Free SIFT 3D points ---
    free_point_start: list[int] = []
    for xyz, _ in points_3d:
        free_point_start.append(len(params))
        params.extend(xyz.tolist())

    # --- Control points ---
    # Fixed (world_xyz set): constant, not in param vector.
    # Free (no world_xyz): DLT-triangulate from current poses and add to param vector.
    fixed_pts: dict[str, np.ndarray] = {}   # cp.name → constant world position
    free_cp_start: dict[str, int] = {}       # cp.name → index into params
    cp_obs_undist: list[tuple[str, dict]] = []

    if control_points:
        for cp in control_points:
            undist = _undistort_control_obs(cp, state_by_id)
            cp_obs_undist.append((cp.name, undist))
            if cp.world_xyz is not None:
                fixed_pts[cp.name] = cp.world_xyz.astype(np.float64)
            else:
                solved_obs = [(v, pt) for v, pt in undist.items() if v in cam_idx]
                if len(solved_obs) >= 2:
                    A_rows = []
                    for v, (px, py) in solved_obs:
                        P = _proj_matrix(state_by_id[v])
                        A_rows.append(px * P[2] - P[0])
                        A_rows.append(py * P[2] - P[1])
                    A = np.array(A_rows, dtype=np.float64)
                    if not np.isfinite(A).all():
                        _log.warning("BA: free CP '%s' has non-finite DLT matrix — skipping", cp.name)
                        continue
                    _, _, Vt = np.linalg.svd(A)
                    h = Vt[-1]
                    if abs(h[3]) > 1e-10:
                        xyz_init = h[:3] / h[3]
                        free_cp_start[cp.name] = len(params)
                        params.extend(xyz_init.tolist())

    params_arr = np.array(params, dtype=np.float64)

    # --- Build observation list ---
    obs_list: list[tuple[str, np.ndarray, str, float]] = []

    for pt_idx, (_, obs_dict) in enumerate(points_3d):
        for vid, (px, py) in obs_dict.items():
            if vid in cam_idx:
                obs_list.append((vid, np.array([px, py]), f"free_{pt_idx}", 1.0))

    if control_points:
        for cp_name, undist in cp_obs_undist:
            if cp_name in fixed_pts:
                weight = fixed_cp_weight
            elif cp_name in free_cp_start:
                weight = unfixed_cp_weight
            else:
                continue  # couldn't triangulate — skip
            for vid, (px, py) in undist.items():
                if vid in cam_idx:
                    obs_list.append((vid, np.array([px, py]), f"cp_{cp_name}", weight))

    n_obs = len(obs_list)
    _log.info(
        "BA: %d cameras, %d SIFT pts, %d fixed CPs, %d free CPs, %d observations",
        n_cams, len(points_3d), len(fixed_pts), len(free_cp_start), n_obs,
    )
    if progress_cb:
        progress_cb(
            f"Bundle adjustment: {n_cams} cameras, {len(points_3d)} SIFT pts, "
            f"{len(fixed_pts)} fixed CPs, {len(free_cp_start)} free CPs, "
            f"{n_obs} observations…"
        )

    if cancel_event is not None and cancel_event.is_set():
        raise _Cancelled()

    if n_obs < 6:
        _log.warning("BA: too few observations (%d), skipping", n_obs)
        return points_3d

    # --- Precompute per-camera projection data (done once, reused every evaluation) ---
    # For each camera: fixed 3D positions (constants) and free 3D param start indices
    # (indices into x), plus observed pixels and weights as contiguous numpy arrays.
    # This lets residuals() do one matmul per camera instead of N cv2.projectPoints calls.
    _cam_obs: dict[str, list] = defaultdict(list)
    for i, (vid, obs_px, pt_key, weight) in enumerate(obs_list):
        _cam_obs[vid].append((i, obs_px, pt_key, weight))

    cam_proj_data: list[tuple] = []
    for vid, entries in _cam_obs.items():
        n_e = len(entries)
        obs_px_arr = np.array([e[1] for e in entries], dtype=np.float64)   # (N, 2)
        weights_arr = np.array([e[3] for e in entries], dtype=np.float64)  # (N,)
        res_ridx = np.array([e[0] for e in entries], dtype=int)            # (N,)

        fixed_mask = np.zeros(n_e, bool)
        fixed_xyz_arr = np.zeros((n_e, 3), dtype=np.float64)
        param_starts_arr = np.zeros(n_e, int)

        for k, (_, _, pt_key, _) in enumerate(entries):
            if pt_key.startswith("free_"):
                param_starts_arr[k] = free_point_start[int(pt_key[5:])]
            else:
                name = pt_key[3:]
                if name in fixed_pts:
                    fixed_mask[k] = True
                    fixed_xyz_arr[k] = fixed_pts[name]
                elif name in free_cp_start:
                    param_starts_arr[k] = free_cp_start[name]

        K = state_by_id[vid].K
        cam_proj_data.append((
            cam_param_start[vid],             # ci: start of [rvec, tvec] in x
            K[0, 0], K[1, 1], K[0, 2], K[1, 2],  # fx, fy, cx, cy
            obs_px_arr, weights_arr, res_ridx,
            fixed_mask, fixed_xyz_arr, param_starts_arr,
        ))

    def residuals(x: np.ndarray) -> np.ndarray:
        if cancel_event is not None and cancel_event.is_set():
            raise _Cancelled()
        res = np.empty(n_obs * 2)
        for (ci, fx, fy, cx, cy,
             obs_px, weights, res_ridx,
             fixed_mask, fixed_xyz, param_starts) in cam_proj_data:
            R, _ = cv2.Rodrigues(x[ci:ci + 3])
            tvec = x[ci + 3:ci + 6].reshape(3, 1)

            # Build Nx3 3D-point array using vectorised numpy indexing
            # Fixed CPs: constant positions already in fixed_xyz
            # Free SIFT/CP points: positions read from x via fancy indexing
            if fixed_mask.all():
                pts3d = fixed_xyz
            elif (~fixed_mask).all():
                # All free — vectorised slice: x[param_starts[k] : param_starts[k]+3] for all k
                pts3d = x[param_starts[:, None] + np.arange(3)]  # (N, 3)
            else:
                pts3d = fixed_xyz.copy()
                fs = param_starts[~fixed_mask]
                pts3d[~fixed_mask] = x[fs[:, None] + np.arange(3)]

            # Project all N points for this camera in one matmul
            pts_cam = R @ pts3d.T + tvec   # (3, N)
            z = pts_cam[2]
            proj_x = fx * pts_cam[0] / z + cx
            proj_y = fy * pts_cam[1] / z + cy

            rx = (proj_x - obs_px[:, 0]) * weights
            ry = (proj_y - obs_px[:, 1]) * weights
            res[2 * res_ridx]     = rx
            res[2 * res_ridx + 1] = ry
        return res

    # --- Jacobian sparsity pattern ---
    # Each residual row (obs_i, component 0 or 1) depends only on:
    #   • 6 camera params for that observation's camera
    #   • 3 point params for that observation's 3D point (if free; fixed CPs have none)
    # Without sparsity, scipy needs 849 evaluations per Jacobian; with it, ~n_cams+n_colors.
    n_params = len(params_arr)
    jac_sp = lil_matrix((n_obs * 2, n_params), dtype=np.int8)
    for i, (vid, _, pt_key, _) in enumerate(obs_list):
        ci = cam_param_start[vid]
        jac_sp[2 * i, ci:ci + 6] = 1
        jac_sp[2 * i + 1, ci:ci + 6] = 1
        if pt_key.startswith("free_"):
            ps = free_point_start[int(pt_key[5:])]
            jac_sp[2 * i, ps:ps + 3] = 1
            jac_sp[2 * i + 1, ps:ps + 3] = 1
        elif pt_key[3:] in free_cp_start:
            ps = free_cp_start[pt_key[3:]]
            jac_sp[2 * i, ps:ps + 3] = 1
            jac_sp[2 * i + 1, ps:ps + 3] = 1

    # Validate initial params — non-finite values (e.g. from a degenerate PnP
    # rotation matrix) cause scipy to raise "Initial guess is outside of bounds".
    non_finite = ~np.isfinite(params_arr)
    if non_finite.any():
        bad = np.where(non_finite)[0].tolist()
        for idx in bad[:10]:
            for vid, start in cam_param_start.items():
                if start <= idx < start + 6:
                    _log.error(
                        "BA: non-finite param at idx=%d (camera %s, offset %d, val=%g)",
                        idx, vid, idx - start, params_arr[idx],
                    )
        raise ValueError(
            f"BA: {non_finite.sum()} non-finite values in initial params — "
            "check camera poses (coplanar PnP may have returned a reflection matrix)"
        )

    try:
        result = least_squares(
            residuals, params_arr,
            method="trf", loss="linear",
            ftol=1e-6, xtol=1e-6,
            max_nfev=max_nfev,
            jac_sparsity=jac_sp.tocsc(),
            verbose=0,
        )
    except _Cancelled:
        _log.info("BA cancelled by user")
        raise

    _log.info("BA done: cost=%.4f (%d iters)", result.cost, result.nfev)
    x = result.x

    for s in solved:
        ci = cam_param_start[s.video_id]
        R, _ = cv2.Rodrigues(x[ci:ci + 3])
        s.R = R
        s.t = x[ci + 3:ci + 6].reshape(3, 1)

    refined: list[tuple[np.ndarray, dict]] = []
    for idx, (_, obs_dict) in enumerate(points_3d):
        start = free_point_start[idx]
        refined.append((x[start:start + 3].copy(), obs_dict))

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


def compute_cp_errors(
    states: list[CamCalibState],
    control_points: list[ControlPoint],
) -> dict[str, dict]:
    """Per-camera reprojection errors for control points.

    For each CP with ≥2 observations in solved cameras, the 3D position is either
    taken from world_xyz (if set) or triangulated via DLT from all observations.
    The projected position is then compared to the observed (undistorted) position.
    """
    state_by_id = {s.video_id: s for s in states}
    per_cam: dict[str, list[float]] = defaultdict(list)

    for cp in control_points:
        undist = _undistort_control_obs(cp, state_by_id)
        solved_obs = {v: pt for v, pt in undist.items()
                      if v in state_by_id and state_by_id[v].R is not None}
        if len(solved_obs) < 2:
            continue

        if cp.world_xyz is not None:
            xyz = cp.world_xyz.astype(np.float64)
        else:
            # DLT triangulate: stack two rows per observation: [px*P[2]-P[0]; py*P[2]-P[1]]
            A_rows = []
            for v, (px, py) in solved_obs.items():
                P = _proj_matrix(state_by_id[v])
                A_rows.append(px * P[2] - P[0])
                A_rows.append(py * P[2] - P[1])
            A = np.array(A_rows, dtype=np.float64)
            if not np.isfinite(A).all():
                _log.warning("CP error: free CP '%s' has non-finite DLT matrix — skipping", cp.name)
                continue
            _, _, Vt = np.linalg.svd(A)
            h = Vt[-1]
            if abs(h[3]) < 1e-10:
                continue
            xyz = (h[:3] / h[3]).astype(np.float64)

        for vid, (px, py) in undist.items():
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
    progress_cb=None,
    cancel_event=None,
    cp_only: bool = False,
    pnp_ransac_px: float = 8.0,
) -> CalibResult:
    """Run the full pipeline.  states[*].image must be loaded before calling.

    When cp_only=True, skip SIFT matching/BFS/triangulation entirely and rely
    solely on control-point observations.  Each camera must have ≥4 world-xyz
    CP observations to be PnP-initialised; cameras without them stay unsolved.

    Otherwise the full pipeline runs:
    1. PnP-initialise cameras that have ≥4 world-xyz control point observations.
    2. SIFT match all pairs; BFS from a PnP-initialised root where possible.
    3. Cheirality filter: remove SIFT pairs where most triangulated points lie
       behind a camera (indicates cameras without real scene overlap).
    4. Re-triangulate with filtered pairs.
    5. Bundle adjustment with high CP weights (1000× for world-xyz, 100× for free).
       SIFT points subsampled to ≤ 300 when CPs are present.

    Raises _Cancelled if cancel_event is set mid-run.
    """
    def _prog(msg: str) -> None:
        _log.info(msg)
        if progress_cb:
            progress_cb(msg)

    def _check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise _Cancelled()

    n = len(states)

    # --- Debug dump: CP and camera inventory ---
    if control_points and _log.isEnabledFor(logging.DEBUG):
        _log.debug("run_calibration: %d cameras: %s",
                   n, [s.label for s in states])
        for cp in control_points:
            xyz_str = (f"world=({cp.world_xyz[0]:.3f},{cp.world_xyz[1]:.3f},"
                       f"{cp.world_xyz[2]:.3f})" if cp.world_xyz is not None else "free")
            obs_str = ", ".join(
                f"{vid}=({px:.1f},{py:.1f})" for vid, (px, py) in cp.obs.items()
            )
            _log.debug("  CP %s [%s]  obs: %s", cp.name, xyz_str, obs_str or "(none)")

    # --- Stage 0: PnP initialisation from world-xyz CPs ---
    if control_points:
        _prog("Initialising cameras from control points (PnP)…")
        pnp_ids = init_poses_pnp(states, control_points, pnp_ransac_px=pnp_ransac_px)
        if pnp_ids:
            _log.info("PnP pre-initialised %d cameras: %s", len(pnp_ids), pnp_ids)

    _check_cancel()

    if cp_only:
        _prog("CP-only mode: skipping SIFT matching…")
        pair_matches_filtered: dict[tuple[str, str], PairMatch] = {}
        points_3d: list[tuple[np.ndarray, dict]] = []
        unsolved = [s.video_id for s in states if s.R is None]
        if unsolved:
            _log.warning(
                "CP-only: %d cameras unsolved (need ≥4 world-xyz CPs each): %s",
                len(unsolved), unsolved,
            )
    else:
        # --- Stage 1-2: SIFT matching + BFS pose chain ---
        n_pairs = n * (n - 1) // 2
        _prog(f"SIFT matching {n} cameras ({n_pairs} pairs)…")
        pair_matches = match_all_pairs(
            states, ratio=sift_ratio, min_inliers=sift_min_inliers,
            progress_cb=progress_cb, cancel_event=cancel_event,
        )
        _prog(f"Chaining poses from {len(pair_matches)} matched pairs…")
        unsolved = chain_poses_bfs(states, pair_matches)

        _check_cancel()

        # --- Stage 2b: Cheirality filter ---
        _prog("Filtering non-overlapping camera pairs (cheirality check)…")
        pair_matches_filtered = filter_sift_by_cheirality(states, pair_matches)
        removed = len(pair_matches) - len(pair_matches_filtered)
        if removed:
            _log.info("Cheirality filter removed %d spurious SIFT pairs", removed)

        _check_cancel()

        # --- Stage 3: Triangulate ---
        _prog(f"Triangulating points from {len(pair_matches_filtered)} verified pairs…")
        points_3d = triangulate_all_pairs(states, pair_matches_filtered)

        _check_cancel()

    # --- Stage 4: Bundle adjustment ---
    points_3d = run_bundle_adjustment(
        states, points_3d, control_points,
        max_nfev=ba_max_nfev,
        progress_cb=progress_cb,
        cancel_event=cancel_event,
    )

    _prog("Computing reprojection errors…")
    errors = compute_reprojection_errors(states, points_3d)
    cp_errors = compute_cp_errors(states, control_points) if control_points else {}

    # Log solved camera world positions and per-CP residuals for debugging.
    _log.info("Solved camera positions (world XYZ):")
    for s in states:
        if s.R is None:
            _log.info("  %-30s  unsolved", s.label)
        else:
            C = -s.R.T @ s.t.flatten()
            err = cp_errors.get(s.video_id)
            err_str = f"  CP err {err['mean']:.1f}±{err['std']:.1f}px (max {err['max']:.1f}px)" if err else ""
            _log.info("  %-30s  (%.3f, %.3f, %.3f)%s", s.label, C[0], C[1], C[2], err_str)

    if control_points and _log.isEnabledFor(logging.DEBUG):
        state_by_id = {s.video_id: s for s in states}
        _log.debug("Per-CP reprojection errors after BA:")
        for cp in control_points:
            parts = []
            for vid, (px, py) in cp.obs.items():
                s = state_by_id.get(vid)
                if s is None or s.R is None:
                    continue
                pts_u = _undistort_pts(np.array([[px, py]], dtype=np.float32), s)
                rvec, _ = cv2.Rodrigues(s.R)
                xyz = cp.world_xyz if cp.world_xyz is not None else None
                if xyz is None:
                    continue
                proj, _ = cv2.projectPoints(
                    xyz.reshape(1, 3), rvec, s.t.reshape(3, 1), s.K, np.zeros(4)
                )
                err = float(np.linalg.norm(proj.reshape(2) - pts_u[0]))
                parts.append(f"{s.label}:{err:.1f}px")
            if parts:
                _log.debug("  %s [%s]: %s", cp.name,
                           "fixed" if cp.world_xyz is not None else "free",
                           "  ".join(parts))

    return CalibResult(
        cameras={s.video_id: s for s in states},
        points_3d=points_3d,
        reprojection_errors=errors,
        unsolved=unsolved,
        pair_matches=pair_matches_filtered,
        cp_reprojection_errors=cp_errors,
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


# ---------------------------------------------------------------------------
# Control-point file I/O (JSON)
# ---------------------------------------------------------------------------

def save_control_points(
    control_points: list[ControlPoint],
    states: list[CamCalibState],
    path: str,
) -> None:
    """Write control points to a JSON file.

    Observations are stored by camera label (not video_id) so the file is
    portable across sessions.  video_id is stored alongside for reference.
    """
    import json

    label_by_id = {s.video_id: s.label for s in states}
    data = {
        "version": 1,
        "control_points": [
            {
                "name": cp.name,
                "world_xyz": cp.world_xyz.tolist() if cp.world_xyz is not None else None,
                "obs": {
                    label_by_id.get(vid, vid): [px, py]
                    for vid, (px, py) in cp.obs.items()
                },
            }
            for cp in control_points
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_control_points(
    path: str,
    states: list[CamCalibState],
) -> list[ControlPoint]:
    """Load control points from a JSON file saved by save_control_points.

    Observations are matched to current cameras by label; unmatched labels are
    silently skipped so files can be shared across sessions with different
    camera subsets.
    """
    import json

    id_by_label = {s.label: s.video_id for s in states}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    version = data.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported control-point file version: {version}")

    result: list[ControlPoint] = []
    for rec in data.get("control_points", []):
        cp = ControlPoint(name=rec["name"])
        if rec.get("world_xyz") is not None:
            cp.world_xyz = np.array(rec["world_xyz"], dtype=np.float64)
        for label, (px, py) in rec.get("obs", {}).items():
            vid = id_by_label.get(label)
            if vid is not None:
                cp.obs[vid] = (float(px), float(py))
        result.append(cp)
    return result
