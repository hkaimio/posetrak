#!/usr/bin/env python3
"""
calibrate_scale.py — Skeleton bone-length calibration via triangulated marker distances.

After a normal tracking run, this script estimates per-bone scale factors by:
  1. Loading model marker 3D positions from tracking_results.csv (FK at UKF posterior)
  2. Triangulating each marker from its Mahalanobis-inlier 2D observations
     (from marker_projections.csv, is_outlier == false)
  3. Computing the ratio |tri_dist| / |model_dist| for each defined marker pair
  4. Aggregating via median across all valid frames in a scale group
  5. Writing a calibrated skeleton YAML with updated joint offsets

See docs/triangulated-distance-calibration-design.md for full algorithm description.

Usage:
    uv run scripts/calibrate_scale.py \\
        --tracking-dir <dir> \\
        --cameras <Calib_scene.toml> \\
        --skeleton <skeleton.yaml> \\
        --output <calibrated.yaml> \\
        [--min-inlier-cameras 2] \\
        [--max-tri-cond 200] \\
        [--scale-min 0.5] \\
        [--scale-max 2.0] \\
        [--min-samples 240]
"""

from __future__ import annotations

import argparse
import math
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Constants: default marker pairs per joint name.
#
# Each entry maps a joint name to (proximal_marker, distal_marker).
# The proximal marker is on the parent joint; the distal marker is on the
# joint itself.  The ratio |tri_dist| / |model_dist| between these markers
# estimates the scale factor for the joint's offset vector.
#
# Joints absent from this table are NOT_OBSERVABLE by default.
# ---------------------------------------------------------------------------

JOINT_MARKER_PAIRS: dict[str, tuple[str, str]] = {
    # Upper limbs
    "upper_arm.L": ("MRK-shoulder.L", "MRK-elbow.L"),
    "upper_arm.R": ("MRK-shoulder.R", "MRK-elbow.R"),
    "forearm.L":   ("MRK-elbow.L",    "MRK-wrist.L"),
    "forearm.R":   ("MRK-elbow.R",    "MRK-wrist.R"),
    # Lower limbs
    "shin.L":      ("MRK-hip.L",      "MRK-knee.L"),
    "shin.R":      ("MRK-hip.R",      "MRK-knee.R"),
    "foot.L":      ("MRK-knee.L",     "MRK-Ankle.L"),
    "foot.R":      ("MRK-knee.R",     "MRK-Ankle.R"),
    # Hip sockets — distal reference is midpoint of both hip markers (see §2.4 note ¹)
    "thigh.L":     ("MRK-hip.L",      "_midpoint:MRK-hip.L:MRK-hip.R"),
    "thigh.R":     ("MRK-hip.R",      "_midpoint:MRK-hip.L:MRK-hip.R"),
}

# For groups whose joints form a connected chain, a single scale factor is
# estimated from chain-endpoint markers and applied uniformly to all joints.
# Key: group name → (proximal_spec, distal_spec).
CHAIN_GROUP_PAIRS: dict[str, tuple[str, str]] = {
    "spine": (
        "_midpoint:MRK-hip.L:MRK-hip.R",
        "_midpoint:MRK-shoulder.L:MRK-shoulder.R",
    ),
}

# Convergence thresholds (IQR of per-sample scale estimates)
IQR_CONVERGED = 0.02
IQR_UNCERTAIN = 0.10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Camera(NamedTuple):
    name: str
    K: np.ndarray       # 3×3 intrinsic matrix
    R: np.ndarray       # 3×3 rotation matrix (world → camera)
    t: np.ndarray       # 3-vector translation (world → camera)
    dist: np.ndarray    # [k1, k2, p1, p2] distortion coefficients
    P: np.ndarray       # 3×4 projection matrix K @ [R | t]


class ScaleGroup(NamedTuple):
    name: str
    joint_names: list[str]          # all joints in the group
    is_chain: bool                   # True → use chain endpoint pair for whole group
    pair_spec: tuple[str, str] | None  # (proximal_spec, distal_spec) if is_chain
    joint_pairs: dict[str, tuple[str, str]]  # joint_name → (prox_spec, dist_spec)


# ---------------------------------------------------------------------------
# Camera loading
# ---------------------------------------------------------------------------

def _rodrigues_to_matrix(rvec: list[float]) -> np.ndarray:
    """Convert Rodrigues rotation vector to 3×3 rotation matrix."""
    v = np.array(rvec, dtype=float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-10:
        return np.eye(3)
    axis = v / angle
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


def load_cameras(toml_path: Path) -> dict[int, Camera]:
    """Load cameras from a Pose2Sim TOML file.

    Camera IDs are the integer suffix of the section key (cam1 → 1, cam2 → 2, …).
    """
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    cameras: dict[int, Camera] = {}
    for key, vals in data.items():
        if not key.startswith("cam") or key == "metadata":
            continue
        try:
            cam_id = int(key[3:])
        except ValueError:
            continue

        K = np.array(vals["matrix"], dtype=float)
        R = _rodrigues_to_matrix(vals["rotation"])
        t = np.array(vals["translation"], dtype=float)
        dist = np.array(vals.get("distortions", [0.0, 0.0, 0.0, 0.0]), dtype=float)
        P = K @ np.hstack([R, t.reshape(3, 1)])
        cameras[cam_id] = Camera(name=vals.get("name", key), K=K, R=R, t=t, dist=dist, P=P)

    return cameras


# ---------------------------------------------------------------------------
# Distortion / undistortion
# ---------------------------------------------------------------------------

def undistort_point(
    px: float, py: float, K: np.ndarray, dist: np.ndarray,
) -> tuple[float, float]:
    """Return undistorted pixel coordinates for a single observed point.

    Uses iterative refinement (5 iterations) to invert the radial + tangential
    distortion model.  For cameras with dist ≈ 0 the result is identical to
    the input.
    """
    k1, k2, p1, p2 = dist[0], dist[1], dist[2], dist[3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Start in distorted normalised coordinates
    xn = (px - cx) / fx
    yn = (py - cy) / fy

    if abs(k1) < 1e-9 and abs(k2) < 1e-9 and abs(p1) < 1e-9 and abs(p2) < 1e-9:
        return px, py  # no distortion — fast path

    # Iterative undistortion
    x0, y0 = xn, yn
    for _ in range(5):
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        dx = 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        dy = p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        xn = (x0 - dx) / radial
        yn = (y0 - dy) / radial

    return xn * fx + cx, yn * fy + cy


# ---------------------------------------------------------------------------
# DLT triangulation
# ---------------------------------------------------------------------------

def triangulate_dlt(
    observations: list[tuple[float, float]],
    Ps: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    """Triangulate a 3D point from N ≥ 2 undistorted pixel observations.

    Each observation (u, v) with projection matrix P contributes two rows to
    the DLT system:
        u * P[2,:] - P[0,:]
        v * P[2,:] - P[1,:]

    Returns (position_3d, condition_number).  A large condition number (> 200)
    indicates degenerate viewing geometry.
    """
    rows = []
    for (u, v), P in zip(observations, Ps):
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])

    A = np.array(rows, dtype=float)
    _, s, Vt = np.linalg.svd(A)
    X_hom = Vt[-1]
    pos = X_hom[:3] / X_hom[3]

    cond = float(s[0] / s[-2]) if s[-2] > 1e-12 else float("inf")
    return pos, cond


# ---------------------------------------------------------------------------
# Skeleton YAML loading
# ---------------------------------------------------------------------------

class Joint(NamedTuple):
    name: str
    parent: str | None
    offset: np.ndarray      # local translation from parent (metres)
    orientation: np.ndarray  # ZYX Euler [z, y, x] radians → rest rotation


def _zyx_euler_to_matrix(z: float, y: float, x: float) -> np.ndarray:
    cz, sz = math.cos(z), math.sin(z)
    cy, sy = math.cos(y), math.sin(y)
    cx, sx = math.cos(x), math.sin(x)
    Rz = np.array([[cz, -sz, 0], [sz,  cz, 0], [0, 0, 1]])
    Ry = np.array([[cy,  0, sy], [0,    1,  0], [-sy, 0, cy]])
    Rx = np.array([[1,   0,  0], [0,   cx, -sx], [0, sx,  cx]])
    return Rx @ Ry @ Rz


def load_skeleton(yaml_path: Path) -> tuple[
    dict[str, Joint], dict[str, str], list[ScaleGroup]
]:
    """Parse skeleton YAML.

    Returns:
        joints:       dict[joint_name, Joint]
        joint_children: dict[parent_name, list[child_name]]  — unused but kept for FK
        scale_groups: list[ScaleGroup]
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    joints: dict[str, Joint] = {}
    children: dict[str, list[str]] = defaultdict(list)

    for jd in data.get("joints", []):
        name = jd["name"]
        parent = jd.get("parent") or None
        offset_raw = jd.get("offset") or [0.0, 0.0, 0.0]
        ori_raw = jd.get("orientation") or [0.0, 0.0, 0.0]
        joints[name] = Joint(
            name=name,
            parent=parent,
            offset=np.array(offset_raw, dtype=float),
            orientation=np.array(ori_raw, dtype=float),
        )
        if parent:
            children[parent].append(name)

    scale_groups = _parse_scale_groups(data.get("scale_groups", []), joints)
    return joints, dict(children), scale_groups


def _parse_scale_groups(
    raw: list[dict], joints: dict[str, Joint]
) -> list[ScaleGroup]:
    groups: list[ScaleGroup] = []

    for gd in raw:
        name = gd["name"]

        # Collect joint names — support both list-of-strings and list-of-dicts
        joint_entries = gd.get("joints") or []
        joint_names: list[str] = []
        explicit_pairs: dict[str, tuple[str, str]] = {}

        for entry in joint_entries:
            if isinstance(entry, str):
                joint_names.append(entry)
            elif isinstance(entry, dict):
                jname = entry["name"]
                joint_names.append(jname)
                if "marker_pair" in entry:
                    mp = entry["marker_pair"]
                    explicit_pairs[jname] = (mp[0], mp[1])

        # Chain groups: explicitly declared or looked up by group name
        chain_key = gd.get("chain") or (name if name in CHAIN_GROUP_PAIRS else None)
        if chain_key and name in CHAIN_GROUP_PAIRS:
            prox, dist = CHAIN_GROUP_PAIRS[name]
            groups.append(ScaleGroup(
                name=name,
                joint_names=joint_names,
                is_chain=True,
                pair_spec=(prox, dist),
                joint_pairs={},
            ))
            continue

        # Individual joints
        resolved_pairs: dict[str, tuple[str, str]] = {}
        for jname in joint_names:
            if jname in explicit_pairs:
                resolved_pairs[jname] = explicit_pairs[jname]
            elif jname in JOINT_MARKER_PAIRS:
                resolved_pairs[jname] = JOINT_MARKER_PAIRS[jname]
            # else: NOT_OBSERVABLE — no entry means it will be skipped

        groups.append(ScaleGroup(
            name=name,
            joint_names=joint_names,
            is_chain=False,
            pair_spec=None,
            joint_pairs=resolved_pairs,
        ))

    return groups


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_tracking_results(
    csv_path: Path,
) -> dict[int, dict[str, np.ndarray]]:
    """Load model marker 3D positions from tracking_results.csv.

    Returns dict[frame][marker_name] = np.array([x, y, z]).
    Only rows with is_visible == true are included.
    """
    result: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    with open(csv_path) as f:
        header = f.readline().rstrip().split(",")
        frame_col = header.index("frame")
        name_col = header.index("marker_name")
        x_col = header.index("x_3d")
        y_col = header.index("y_3d")
        z_col = header.index("z_3d")
        vis_col = header.index("is_visible")
        for line in f:
            parts = line.rstrip().split(",")
            if parts[vis_col].lower() != "true":
                continue
            frame = int(parts[frame_col])
            name = parts[name_col]
            result[frame][name] = np.array([
                float(parts[x_col]),
                float(parts[y_col]),
                float(parts[z_col]),
            ])
    return dict(result)


def load_inlier_observations(
    csv_path: Path,
) -> dict[int, dict[str, dict[int, tuple[float, float]]]]:
    """Load per-camera inlier 2D observations from marker_projections.csv.

    Returns dict[frame][marker_name][camera_id] = (obs_x, obs_y).
    Only rows with is_outlier == false are included.
    """
    result: dict[int, dict[str, dict[int, tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with open(csv_path) as f:
        header = f.readline().rstrip().split(",")
        frame_col = header.index("frame")
        name_col = header.index("marker_name")
        cam_col = header.index("camera_id")
        ox_col = header.index("obs_x")
        oy_col = header.index("obs_y")
        out_col = header.index("is_outlier")
        for line in f:
            parts = line.rstrip().split(",")
            if parts[out_col].lower() != "false":
                continue
            frame = int(parts[frame_col])
            name = parts[name_col]
            cam_id = int(parts[cam_col])
            result[frame][name][cam_id] = (float(parts[ox_col]), float(parts[oy_col]))

    return {f: dict(m) for f, m in result.items()}


# ---------------------------------------------------------------------------
# Triangulation helpers
# ---------------------------------------------------------------------------

def triangulate_marker(
    cam_obs: dict[int, tuple[float, float]],
    cameras: dict[int, Camera],
    min_inlier_cameras: int,
    max_tri_cond: float,
) -> np.ndarray | None:
    """Triangulate a single marker from inlier camera observations.

    Returns 3D position or None if quality criteria are not met.
    """
    available = [(cam_id, obs) for cam_id, obs in cam_obs.items()
                 if cam_id in cameras]
    if len(available) < min_inlier_cameras:
        return None

    undistorted = []
    Ps = []
    for cam_id, (u, v) in available:
        cam = cameras[cam_id]
        u_ud, v_ud = undistort_point(u, v, cam.K, cam.dist)
        undistorted.append((u_ud, v_ud))
        Ps.append(cam.P)

    pos, cond = triangulate_dlt(undistorted, Ps)
    if cond > max_tri_cond or not np.all(np.isfinite(pos)):
        return None
    return pos


# ---------------------------------------------------------------------------
# Marker spec resolution
# ---------------------------------------------------------------------------

def resolve_spec(
    spec: str,
    tri: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Resolve a marker spec to a 3D position.

    Supported spec formats:
      "MRK-foo"                     — single triangulated marker
      "_midpoint:MRK-a:MRK-b"      — midpoint of two triangulated markers
      (model-joint specs are not supported here; those groups are NOT_OBSERVABLE
       without FK re-evaluation — see §5 of the design doc for future extension)
    """
    if spec.startswith("_midpoint:"):
        parts = spec.split(":")
        a, b = parts[1], parts[2]
        if a not in tri or b not in tri:
            return None
        return (tri[a] + tri[b]) * 0.5
    else:
        return tri.get(spec)


# ---------------------------------------------------------------------------
# Per-frame scale estimation
# ---------------------------------------------------------------------------

def compute_frame_samples(
    frame: int,
    groups: list[ScaleGroup],
    model: dict[str, np.ndarray],           # marker_name → model 3D pos
    tri: dict[str, np.ndarray],              # marker_name → triangulated 3D pos
    scale_min: float,
    scale_max: float,
) -> dict[str, list[float]]:
    """Compute per-group scale samples for one frame.

    Returns dict[group_name, list[float]] of valid scale estimates.
    """
    samples: dict[str, list[float]] = defaultdict(list)

    for group in groups:
        if group.is_chain:
            prox_spec, dist_spec = group.pair_spec  # type: ignore[misc]
            p_tri = resolve_spec(prox_spec, tri)
            d_tri = resolve_spec(dist_spec, tri)
            p_mod = resolve_spec(prox_spec, model)
            d_mod = resolve_spec(dist_spec, model)

            if p_tri is None or d_tri is None or p_mod is None or d_mod is None:
                continue

            tri_dist = float(np.linalg.norm(d_tri - p_tri))
            mod_dist = float(np.linalg.norm(d_mod - p_mod))
            if mod_dist < 1e-6:
                continue

            s = tri_dist / mod_dist
            if scale_min <= s <= scale_max:
                samples[group.name].append(s)

        else:
            for jname, (prox_spec, dist_spec) in group.joint_pairs.items():
                p_tri = resolve_spec(prox_spec, tri)
                d_tri = resolve_spec(dist_spec, tri)
                p_mod = resolve_spec(prox_spec, model)
                d_mod = resolve_spec(dist_spec, model)

                if p_tri is None or d_tri is None or p_mod is None or d_mod is None:
                    continue

                tri_dist = float(np.linalg.norm(d_tri - p_tri))
                mod_dist = float(np.linalg.norm(d_mod - p_mod))
                if mod_dist < 1e-6:
                    continue

                s = tri_dist / mod_dist
                if scale_min <= s <= scale_max:
                    samples[group.name].append(s)

    return dict(samples)


# ---------------------------------------------------------------------------
# Aggregation and convergence
# ---------------------------------------------------------------------------

def aggregate_samples(
    all_samples: dict[str, list[float]],
    min_samples: int,
) -> dict[str, dict]:
    """Compute median, IQR, count, and convergence status per group."""
    results = {}
    for group_name, samples in all_samples.items():
        n = len(samples)
        if n == 0:
            results[group_name] = {
                "scale": 1.0, "iqr": float("inf"), "n": 0, "status": "NOT_OBSERVABLE",
            }
            continue
        arr = np.array(samples, dtype=float)
        median = float(np.median(arr))
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        iqr = q3 - q1

        if n >= min_samples and iqr < IQR_CONVERGED:
            status = "CONVERGED"
        elif n >= min_samples and iqr < IQR_UNCERTAIN:
            status = "UNCERTAIN"
        else:
            status = "NOT_OBSERVABLE"

        results[group_name] = {"scale": median, "iqr": iqr, "n": n, "status": status}
    return results


# ---------------------------------------------------------------------------
# Bilateral divergence check
# ---------------------------------------------------------------------------

def check_bilateral_divergence(
    groups: list[ScaleGroup],
    agg: dict[str, dict],
    threshold: float = 0.05,
) -> list[str]:
    """Return warning strings for bilateral pairs that diverge > threshold."""
    warnings = []
    # Find groups that share the same name prefix up to last dot
    # (e.g. "femur.L" / "femur.R" — though current YAML uses single groups
    #  containing both sides; check per-joint estimates within a non-chain group)
    for group in groups:
        if group.is_chain:
            continue
        # Collect joint-pair-level left/right estimates if group was split per-side
        left_joints = [j for j in group.joint_names if j.endswith(".L")]
        right_joints = [j for j in group.joint_names if j.endswith(".R")]
        if not left_joints or not right_joints:
            continue

        group_result = agg.get(group.name)
        if group_result is None or group_result["status"] == "NOT_OBSERVABLE":
            continue

        # For per-side divergence we need per-joint samples, not available here.
        # Emit a note asking user to split group if interested in asymmetry.
        # Full per-joint aggregation would require separate sample lists.
    return warnings


# ---------------------------------------------------------------------------
# YAML writing
# ---------------------------------------------------------------------------

def write_calibrated_yaml(
    input_yaml_path: Path,
    groups: list[ScaleGroup],
    agg: dict[str, dict],
    output_path: Path,
) -> None:
    """Write calibrated skeleton YAML with updated joint offsets.

    All fields not modified by calibration are copied verbatim.
    The scale_groups key is removed from the output.
    """
    with open(input_yaml_path) as f:
        data = yaml.safe_load(f)

    # Build joint_name → scale_factor mapping
    joint_scale: dict[str, float] = {}
    for group in groups:
        result = agg.get(group.name)
        if result is None or result["status"] == "NOT_OBSERVABLE":
            continue
        scale = result["scale"]
        for jname in group.joint_names:
            joint_scale[jname] = scale

    # Apply scale factors to offset vectors
    for jd in data.get("joints", []):
        jname = jd["name"]
        if jname not in joint_scale:
            continue
        s = joint_scale[jname]
        raw = jd.get("offset") or [0.0, 0.0, 0.0]
        jd["offset"] = [v * s for v in raw]

    # Remove scale_groups — not needed in a calibrated skeleton
    data.pop("scale_groups", None)

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\nCalibrated skeleton written to: {output_path}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_convergence_table(
    groups: list[ScaleGroup],
    agg: dict[str, dict],
) -> None:
    print()
    print(f"{'Group':<20} {'Joints':<40} {'Scale':>7} {'IQR':>6} {'N':>6}  Status")
    print("-" * 90)
    for group in groups:
        result = agg.get(group.name)
        if result is None:
            joints_str = ", ".join(group.joint_names)
            print(f"{group.name:<20} {joints_str:<40} {'—':>7} {'—':>6} {'0':>6}  NOT_OBSERVABLE")
            continue
        joints_str = ", ".join(group.joint_names)
        if len(joints_str) > 38:
            joints_str = joints_str[:35] + "..."
        scale_str = f"{result['scale']:.4f}"
        iqr_str = f"{result['iqr']:.4f}" if result["iqr"] != float("inf") else "∞"
        print(
            f"{group.name:<20} {joints_str:<40} {scale_str:>7} {iqr_str:>6}"
            f" {result['n']:>6}  {result['status']}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracking-dir", required=True, type=Path,
                   help="Directory containing tracking_results.csv and marker_projections.csv")
    p.add_argument("--cameras", required=True, type=Path,
                   help="Pose2Sim camera calibration TOML")
    p.add_argument("--skeleton", required=True, type=Path,
                   help="Skeleton YAML with scale_groups")
    p.add_argument("--output", required=True, type=Path,
                   help="Output path for calibrated skeleton YAML")
    p.add_argument("--min-inlier-cameras", type=int, default=2, metavar="N",
                   help="Minimum Mahalanobis-inlier cameras for triangulation (default: 2)")
    p.add_argument("--max-tri-cond", type=float, default=200.0, metavar="C",
                   help="Maximum DLT condition number to accept triangulation (default: 200)")
    p.add_argument("--scale-min", type=float, default=0.5,
                   help="Sanity clamp: discard samples below this scale (default: 0.5)")
    p.add_argument("--scale-max", type=float, default=2.0,
                   help="Sanity clamp: discard samples above this scale (default: 2.0)")
    p.add_argument("--min-samples", type=int, default=240, metavar="N",
                   help="Minimum valid samples for CONVERGED/UNCERTAIN status (default: 240)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tracking_dir = args.tracking_dir
    results_csv = tracking_dir / "tracking_results.csv"
    projections_csv = tracking_dir / "marker_projections.csv"

    for path in [results_csv, projections_csv, args.cameras, args.skeleton]:
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading cameras from {args.cameras} …")
    cameras = load_cameras(args.cameras)
    print(f"  {len(cameras)} cameras loaded: {sorted(cameras)}")

    print(f"Loading skeleton from {args.skeleton} …")
    joints, _, scale_groups = load_skeleton(args.skeleton)
    if not scale_groups:
        print("ERROR: skeleton YAML has no scale_groups defined.", file=sys.stderr)
        sys.exit(1)
    observable_groups = [
        g for g in scale_groups
        if g.is_chain or g.joint_pairs
    ]
    skipped = [g.name for g in scale_groups if g not in observable_groups]
    print(f"  {len(scale_groups)} groups; {len(observable_groups)} have marker pairs.")
    if skipped:
        print(f"  NOT_OBSERVABLE (no marker pairs): {', '.join(skipped)}")

    print(f"Loading model marker positions from {results_csv} …")
    model_by_frame = load_tracking_results(results_csv)
    print(f"  {len(model_by_frame)} frames.")

    print(f"Loading inlier observations from {projections_csv} …")
    obs_by_frame = load_inlier_observations(projections_csv)
    print(f"  {len(obs_by_frame)} frames with inlier observations.")

    # --- Per-frame loop -------------------------------------------------
    print("Computing triangulated distances …")
    all_samples: dict[str, list[float]] = defaultdict(list)

    frames = sorted(model_by_frame.keys())
    n_frames = len(frames)
    report_every = max(1, n_frames // 10)

    for i, frame in enumerate(frames):
        if i % report_every == 0:
            print(f"  frame {frame} ({i}/{n_frames})", end="\r", flush=True)

        model = model_by_frame.get(frame, {})
        cam_obs_by_marker = obs_by_frame.get(frame, {})

        # Triangulate all markers that have enough inlier cameras
        tri: dict[str, np.ndarray] = {}
        for marker_name, cam_obs in cam_obs_by_marker.items():
            pos = triangulate_marker(
                cam_obs, cameras, args.min_inlier_cameras, args.max_tri_cond
            )
            if pos is not None:
                tri[marker_name] = pos

        # Compute scale samples for this frame
        frame_samples = compute_frame_samples(
            frame, observable_groups, model, tri,
            args.scale_min, args.scale_max,
        )
        for group_name, vals in frame_samples.items():
            all_samples[group_name].extend(vals)

    print(f"\nProcessed {n_frames} frames.")

    # Mark groups with no pairs as NOT_OBSERVABLE
    for group in scale_groups:
        if group.name not in all_samples:
            all_samples[group.name] = []

    # --- Aggregation and reporting --------------------------------------
    agg = aggregate_samples(dict(all_samples), args.min_samples)
    print_convergence_table(scale_groups, agg)

    # --- Write output ---------------------------------------------------
    write_calibrated_yaml(args.skeleton, scale_groups, agg, args.output)


if __name__ == "__main__":
    main()
