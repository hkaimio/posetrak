# C++ tracker architecture

The C++ tracker (`cli/posetrak track`) reads a pose observation sequence from the session DB, estimates the 3-D skeletal pose using an Unscented Kalman Filter, and writes the results back to the DB.

A second command (`cli/posetrak scale`) post-processes a bone-length calibration run.  This command is not currently invoked by the Python app and is a candidate for deprecation — skeleton scaling is performed by the Python tooling instead.

---

## Source layout

```
src/
├── cli/
│   └── main.cpp                        posetrak track / scale entry points
├── core/
│   ├── skeleton.hpp/.cpp               kinematic tree: joints, markers, DOF types
│   ├── skeleton_layout.hpp/.cpp        DOF index table (single source of truth)
│   ├── state.hpp/.cpp                  error-state representation, quaternion ops
│   ├── observation.hpp                 Observation, ObservationSet, ObservationSequence
│   └── config.cpp                      TOML config loading
├── db/
│   ├── session_reader.hpp/.cpp         SQLite read-only access to session DB
│   ├── result_writer.hpp/.cpp          writes tracking_results + tracking_obs_results
│   └── blob_codec.hpp/.cpp             encode/decode float32/float64 blobs
├── filters/
│   ├── ukf.hpp/.cpp                    UnscentedKalmanFilter: error-state, Joseph form
│   └── subset_ukf.hpp/.cpp             child filter for hierarchical tracking (experimental)
├── io/
│   ├── observation_loader.hpp/.cpp     loads from JSON (legacy) or TOML
│   ├── tracking_export.hpp/.cpp        exports CSV results
│   └── statistics_tracker.hpp/.cpp     per-frame outlier / NIS statistics
├── kinematics/
│   ├── forward_kinematics.hpp/.cpp     Pinocchio wrapper: State → marker world positions
│   ├── inverse_kinematics.hpp/.cpp     damped least-squares IK for initialisation
│   └── triangulation.hpp/.cpp          DLT multi-view triangulation
└── tracking/
    └── tracker.hpp/.cpp                Tracker: initialize() + track_frame() orchestration
```

---

## Data flow

```
YAML skeleton + TOML cameras + session DB observations
    │
    ├── SkeletonLoader      → Skeleton
    ├── SessionReader       → cameras (intrinsics, extrinsics), sync, observations
    │       load_observations() applies pose_observation_edits as an overlay
    │
    └── Tracker::initialize()
            DLT triangulation of first good frame
            InverseKinematics (damped least-squares) → initial joint angles
            → UKF state + covariance
                    │
                    ▼ for each frame:
            Tracker::track_frame()
                UKF::predict()   constant-velocity process model
                UKF::update()    project sigma points through FK → camera
                                 Mahalanobis outlier rejection
                                 Joseph-form covariance update
                ResultWriter → tracking_results, tracking_obs_results
```

Optional **RTS smoothing**: enable before tracking, call `smooth()` after the loop.  Smoothed and unsmoothed results share `tracking_results`; `is_smoothed` is part of the primary key.

---

## Core abstractions

### `Skeleton`

Kinematic tree of `Joint` objects.  Joint types: `REVOLUTE` (1 DOF), `SPHERICAL` (3 DOF), `FIXED` (0 DOF), `PRISMATIC` (1 DOF).  Each joint can have attached `Marker` points used as observations.  Loaded from YAML via `SkeletonLoader`.

### `SkeletonLayout`

Immutable, precomputed DOF index table derived from a `Skeleton`.  Shared via `shared_ptr<const SkeletonLayout>` between the UKF, process model, and sigma point generator.

**All DOF index arithmetic must go through `SkeletonLayout`.**  Never recompute indices ad-hoc.

Two factory functions: `from_full_skeleton()` and `from_groups()` (for child filters in hierarchical tracking).

### `State`

Root pose (position + quaternion) + joint angles + velocities.  Uses **error-state formulation**: orientation updates happen on the quaternion manifold via axis-angle perturbations applied through `State::apply_error_update()`.

State dimension: ~58 for a typical full-body skeleton (3 position + 3 orientation in tangent space + joint DOFs, all doubled for velocities).

### `UnscentedKalmanFilter`

Error-state UKF.  Sigma points are generated in tangent space, propagated through the constant-velocity process model, then measurement-updated via FK + camera projection.

Uses **Joseph form** covariance update `P' = (I−KH)P(I−KH)^T + KRK^T` for numerical stability.

Per-observation Mahalanobis distance gating rejects outlier keypoints before the update.

### `ForwardKinematics`

Wraps Pinocchio.  Converts `State` → Pinocchio config vector → computes joint transforms → extracts marker world positions.

Must call both `forwardKinematics()` and `updateFramePlacements()` — omitting the second silently returns stale placements.

### `Tracker`

Orchestrates initialisation (triangulation → IK) and the per-frame predict/update cycle.  Owns the UKF, FK instance, triangulator, IK solver, and Pinocchio model/data.

---

## Key design invariants

**UKF alpha must be ≥ 0.5.**  For a ~58-dimensional state, alpha = 0.001 causes negative covariance weights on the central sigma point.  Alpha ≥ 0.5 is required for positive-definite weights throughout.

**SPHERICAL joints always occupy 3 state slots**, even when some DOFs are locked by equal limits.  Active/locked distinction is handled at the layout level.

**Pinocchio quaternion convention is `[x, y, z, w]`** (scalar-last), not `[w, x, y, z]`.  State storage uses `[w, x, y, z]` internally; conversion happens inside `ForwardKinematics`.

**Pinocchio is used header-only.**  `PINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION` is not defined; no linking against `libpinocchio_default.so`.  Pinocchio is installed system-wide at `/opt/openrobots/`.

---

## Invocation from the Python app

The Python app (`RunTrackerDialog` in `python/app/pose/run_tracker.py`) invokes the tracker by passing all parameters as command-line arguments — no TOML file is written to disk:

```
posetrak track
    --session-db  <path>
    --sequence    <sequence_id>
    --skeleton    <skeleton_id>
    --tracker-config <config_id>
    --person-id   <int>
    --start-time  <float>
    --end-time    <float>
    --output-dir  <path>
    --smooth
```

The tracker resolves all IDs against the session DB at startup.  The Python app creates the `tracker_config` row before launching the process.

## Config file format (CLI usage)

When invoked directly from the command line (not via the Python app), the tracker also accepts a TOML config file:

```bash
optbuild/cli/posetrak track config.toml
```

Key sections:

| Section | Contents |
|---|---|
| `[data]` | Paths to session DB, sequence ID, skeleton, calibration |
| `[tracking]` | `tracker_fps`, `min_cameras_for_update`, enable_smoothing |
| `[tracking.initialization]` | `ik_max_iterations`, `ik_tolerance`, `min_cameras_for_init` |
| `[tracking.ukf]` | `alpha`, `beta`, `kappa`, `process_noise_std`, `measurement_noise_std`, `outlier_threshold` |
| `[output]` | `output_dir` for CSV exports |
| `[processing]` | Time window (`start_s`, `end_s`) |
| `[hierarchical]` | Optional: per-group child filter configuration |

See [UKF algorithm](algorithms/ukf.md) for the meaning of the UKF noise parameters.

---

## Build

```bash
meson setup optbuild -Dbuildtype=release   # optimised build for tracking
meson compile -C optbuild

optbuild/cli/posetrak track config.toml
```

Performance between debug and optimised builds is substantial.  Use `optbuild/` for actual tracking runs; the debug build in `builddir/` is adequate for unit tests.
