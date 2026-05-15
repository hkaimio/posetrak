# Semi-automatic extrinsics calibration — design

**Status:** Implementation in progress (solver backend done; Qt UI pending)

---

## What the existing script provides (reused as-is or adapted)

`python/pipeline/calibration/calibrate_extrinsics.py` already has:

| Function | Reuse plan |
|---|---|
| `bundle_adjustment` | Adapted — same scipy/rvec/tvec structure, new data-model wrappers |
| `triangulate_point_multiview` | Used directly |
| `estimate_camera_pose_pnp` (IPPE coplanar) | Used for control points with known world_xyz |
| `undistort_points` (fisheye + radtan + map-based) | Logic copied into solver |
| `save_results_toml` | Adapted for DB import path |

The new code lives in `python/app/setup/extrinsics_solver.py` (pure computation, no Qt).
The existing script remains unchanged as a standalone CLI tool.

---

## Data model — all in-memory, no new DB tables

```python
@dataclass
class CamCalibState:
    video_id: str
    label: str
    K: np.ndarray        # K_new — undistorted camera matrix (3×3)
    K_orig: np.ndarray   # K_original — for undistorting raw images
    dist: np.ndarray     # original distortion coefficients
    fisheye: bool
    image: np.ndarray | None  # BGR full-res frame; None if not loaded
    R: np.ndarray | None = None   # world→cam rotation (3×3), None = unsolved
    t: np.ndarray | None = None   # world→cam translation (3×1), None = unsolved

@dataclass
class ControlPoint:
    name: str
    obs: dict[str, tuple[float, float]]  # video_id → (px, py) distorted pixel
    world_xyz: np.ndarray | None = None  # (3,) — if set, fixes the 3D position in BA

@dataclass
class PairMatch:
    vid_a: str
    vid_b: str
    R_rel: np.ndarray    # rotation from cam-A frame to cam-B frame (3×3)
    t_rel: np.ndarray    # unit translation in cam-A frame (3×1)
    pts_a: np.ndarray    # Nx2 undistorted pixels in cam A
    pts_b: np.ndarray    # Nx2 undistorted pixels in cam B
    n_inliers: int

@dataclass
class CalibResult:
    cameras: dict[str, CamCalibState]   # video_id → solved state
    points_3d: list[tuple[np.ndarray, dict[str, tuple[float, float]]]]
    #   each entry: (xyz_world (3,), {video_id: (px_undist, py_undist)})
    reprojection_errors: dict[str, dict]  # video_id → {mean, std, max, n}
    unsolved: list[str]                   # video_ids with no pose (no overlap path)
```

Control points are ephemeral: placed in the UI for this calibration run and discarded once the result is written to `extrinsic_calibrations`. If the user needs to redo it they start fresh (same as any SfM workflow).

---

## Pipeline

```
Images + intrinsics (from DB)
    │
    ▼
[1] SIFT detect + match all pairs        → dict[(vid_a, vid_b) → PairMatch | None]
    │
    ▼
[2] Essential matrix per pair             (done inside step 1)
    findEssentialMat on normalised coords → R_rel, t_rel (unit length)
    │
    ▼
[3] BFS spanning tree                    → R, t per camera in root-camera frame
    root = camera with most inlier edges
    unsolved cameras flagged for manual bridge
    │
    ▼
[4] Triangulate inlier pairs             → initial 3D point cloud
    cv2.triangulatePoints per pair
    depth filter (both cameras positive)
    │
    ▼
[5] Bundle adjustment                    → refined poses + 3D points
    Variables: rvec/tvec per solved camera + xyz per free point
    Fixed:     control points with world_xyz (strong residual weight)
    Residuals: reprojection error (2 values per observation)
    scipy.optimize.least_squares, method='trf', Huber loss
    │
    ▼
[6] Similarity transform (manual)        → physical coordinate system
    scale   — distance between two named points
    origin  — control point or clicked position
    axis    — 3-point floor plane + forward direction
    │
    ▼
[7] Write extrinsic_calibrations row     (Pose2Sim TOML → import path)
```

### Essential matrix with heterogeneous cameras

Both match sets are normalised before `findEssentialMat`:
```python
pts_norm = cv2.undistortPoints(pts.reshape(-1,1,2), K_undistorted, np.zeros(4))
E, mask = cv2.findEssentialMat(pts_a_norm, pts_b_norm, np.eye(3),
                                method=cv2.RANSAC, prob=0.999, threshold=0.001)
_, R, t, pmask = cv2.recoverPose(E, pts_a_norm, pts_b_norm, np.eye(3))
```

`K_undistorted` is already the optimal undistorted matrix stored in the DB.
`undistortPoints` with zero distortion just normalises by K, giving pinhole rays.
RANSAC threshold 0.001 ≈ 1 pixel at f=1000 in normalised coordinates.

### BA fixed-point handling

Control points with `world_xyz` are not in the parameter vector. Their observations
contribute the same reprojection residual as free points — the difference is just
that their 3D position is held constant. This naturally fixes scale and origin when
≥2 such points are provided with consistent world coordinates.

---

## Manual control points (UI — pending)

The user marks the same physical feature in any subset of camera images by clicking.
Each click records a distorted pixel coordinate; the solver undistorts it before use.

Two roles:

1. **Bridge disconnected cameras** — one shared point between a solved and an unsolved
   camera is enough to bring the unsolved camera into the BFS tree (via PnP if ≥4
   shared points, or via essential-matrix if paired with another camera).

2. **Fix the coordinate system** — assign known world_xyz to ≥2 points.
   Two points with a known distance fix scale; adding Z constrains the floor plane.

In the UI, control points and their world_xyz values are edited in a side panel.
The solver is re-run on "Solve" — no intermediate state needs to be persisted.

---

## Similarity transform

Applied after BA. Transforms all 3D points and camera poses consistently.

```
C_new  = s * R_align @ C_old + t_align        # camera centres
R_new  = R_old @ R_align.T                    # camera rotations unchanged in body frame
t_new  = -R_new @ C_new
```

UI controls (proposed — exact design TBD):
- **Scale:** pick two control points, enter known distance between them
- **Origin:** pick control point or floor-plane intersection → world (0,0,0)
- **Axis:** pick 3 floor points → XY plane; pick forward point → +X direction

---

## Module structure

```
python/app/setup/extrinsics_solver.py   ← pure computation (this PR)
python/app/setup/page_extrinsics.py     ← existing import dialog; new dialog added later
python/tools/calibrate_from_exports.py  ← CLI wrapper for testing (this PR)
```

The CLI script loads exported PNGs + intrinsics from the session DB and outputs
`cameras.toml`.  No Qt required.  This is the checkpoint.

---

## Checkpoint

**Checkpoint: solver produces importable cameras.toml from exported PNGs.**

Verified when:
- `calibrate_from_exports.py` runs without error on the real session's exported frames
- Per-camera reprojection error < 5 px (ideally < 2 px) after BA
- The imported TOML produces correct triangulation in the tracker (visual check)

After checkpoint: build the Qt dialog (`ExtrinsicsAutoCalibDialog`) on top of the
verified solver, adding the image viewer, manual control point placement, and
coordinate system alignment UI.

---

## Open questions / future

- **Wand calibration:** adds a "distance constraint between two points" residual type
  to the BA. Control point data model already supports this (two named points +
  enter distance instead of absolute xyz). The pipeline is otherwise identical.
- **Jacobian sparsity:** current BA builds a dense Jacobian. For >20 cameras and
  >2000 points, pass a `jac_sparsity` matrix to `least_squares` for 10–100× speedup.
- **FLANN matcher:** faster than BFMatcher for >1000 keypoints per image.
  Drop-in replacement; not needed for 4–8 cameras.
