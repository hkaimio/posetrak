# Analytical Joint Angle Seeding for Initialization

**Status:** Proposed — not yet implemented
**Context:** Poor IK initialization causes slow UKF convergence (200+ frames)

---

## Problem

The current `Tracker::initialize()` flow is:

1. Triangulate markers from frame-0 observations
2. Estimate root pose analytically (Procrustes from hip/shoulder markers)
3. Run IK with zero joint angles as starting point
4. Use IK result if RMS < 0.5 m (root drift guard added March 2026)

The IK is a Levenberg-Marquardt optimizer over the full joint space. Starting from zero
joint angles (rest pose) it has to close a large gap in a handful of iterations, which
leads to two failure modes:

- **Degenerate local minima**: IK finds a configuration that fits marker positions in 3D
  (RMS < tolerance) but is anatomically wrong — e.g. root displaced 1 m upward with
  joints compensating. The root drift guard (0.5 m threshold) catches severe cases but
  not subtle ones.
- **Slow convergence for non-trivial poses**: Even when IK finds the right basin it may
  need many iterations starting from rest pose, and with more DOF (complex skeletons)
  the problem is harder.

After the IK fix in March 2026 the initialization improved but still takes ~50–100 frames
for the UKF to fully converge on the reallusion skeleton.

---

## Proposed Solution: Direction Seeding

Before running IK, analytically estimate each joint's rotation from the triangulated
3D positions of nearby markers. This dramatically narrows the IK search space so it
only needs fine-tuning steps from a good starting point.

### Core algorithm per joint

For a non-root joint J with known parent world transform T_parent:

1. Find the **proximal 3D point** P and the **distal 3D point** D from triangulation
2. Compute observed bone direction in parent's local frame:
   `v_obs = T_parent.rotation.inverse() * (D − P).normalized()`
3. Compute rest-pose bone direction in parent's local frame (from FK at zero angles):
   `v_rest = (R_rest * bone_tip_offset).normalized()`
   where `R_rest = zyx_euler_to_matrix(orientation)`
4. Find the rotation R_seed that maps v_rest → v_obs:
   - **Revolute joint (1 DOF)**: project v_obs onto the joint's rotation plane, compute
     the scalar flexion angle via `atan2`
   - **Ball joint (3 DOF)**: use the axis-angle between v_rest and v_obs as a 3-vector.
     This seeds 2 of the 3 DOFs; the axial roll (rotation around the bone axis) remains
     at zero.

Walk joints in topological order (parent before child) and update the partial world
transform after seeding each joint, so children use the already-seeded parent pose.

### Open question: ball joint axial roll

For ball joints (shoulder, hip, wrist) the direction seeding leaves the axial roll
component at zero. This is usually fine because:

- The IK will correct small roll errors quickly from a good directional starting point
- Axial roll is hard to observe from body markers without additional geometric constraints

**Unknown**: in the case of thigh.R specifically, the earlier analysis showed the IK
was finding a wrong configuration with a large angular error (~52°). It is not yet
confirmed whether this error was primarily in the bone *direction* (which seeding would
fix) or in the *axial roll* (which seeding would not fix). This should be verified by
examining the world rotation matrix of thigh.R at frame 1 of a fresh run and comparing
its bone-tip direction to the expected hip→knee direction.

---

## Generalization Options

Three levels of generalization, from least to most general:

### Level 1: Hardcoded marker names
Use specific marker names (`MRK-hip.R`, `MRK-knee.R`, etc.) to seed specific joints.

- **Pro**: Simplest, ~20 lines
- **Con**: Breaks for any skeleton that doesn't use these exact marker names

### Level 2: Reuse `scale_groups` marker pairs (recommended starting point)

The `scale_groups` section already maps a marker pair to each calibrated joint.  The
same proximal/distal markers that measure bone *length* for calibration also measure
bone *direction* for initialization.

```yaml
scale_groups:
  - name: femur
    joints:
      - name: shin.L
        marker_pair: [MRK-hip.L, MRK-knee.L]   # prox = hip socket, dist = knee
      - name: shin.R
        marker_pair: [MRK-hip.R, MRK-knee.R]
```

**Semantic note**: the marker pair is listed under `shin.L` (which is the joint whose
*offset* is being calibrated — the thigh bone length) but the direction it describes
constrains `thigh.L`'s *rotation*.  The seeding algorithm must walk one level up the
joint tree: for joint J listed in a marker pair, the direction seeds J's *parent*
joint's rotation.

- **Pro**: Works for any skeleton with `scale_groups` defined; no new YAML needed;
  the calibrated skeleton workflow always produces scale_groups
- **Con**: Slight semantic mismatch (pair listed under child joint, seeds parent joint);
  only covers joints explicitly listed in scale_groups

### Level 3: Automatic from skeleton topology

Inspect `Skeleton::markers()` to find markers whose `local_pos` is close to a joint
origin (`|offset| < threshold`) or close to a joint's `bone_tip_offset`. Use those to
reconstruct each joint's bone direction automatically.

- **Pro**: Requires zero YAML changes; works for any skeleton; most general
- **Con**: Requires heuristics for "close enough"; harder to debug when it picks
  wrong markers

---

## Implementation Plan (when ready to implement)

### Files to change

- `src/tracking/tracker.cpp` — add `seed_joint_angles_from_markers()` helper called
  inside `initialize()` before the IK call, replacing the current zero-angle initial
  guess
- Optionally `include/posetrak/tracking/tracker.hpp` if the helper needs to be a
  method rather than a lambda

### Sketch

```cpp
// Inside initialize(), between estimate_analytic_state() and ik_solver_->solve():

State seeded_state = analytic_state;
seed_joint_angles_from_markers(seeded_state, marker_positions, *skeleton_,
                                *model_, *fk_);

auto ik_result = ik_solver_->solve(marker_positions, *skeleton_, seeded_state, ...);
```

`seed_joint_angles_from_markers` walks joints in topological order:

1. Maintain a running map of joint world transforms (starting from root)
2. For each joint, look up its marker pair from scale_groups (or auto-detect)
3. If both proximal and distal markers are triangulated, estimate the rotation and
   set the joint angle in `seeded_state`
4. Update the world transform for this joint (so children benefit)

For Level 2 the function needs access to the parsed scale_groups; the calibration
script already parses these in Python — the C++ skeleton loader would need to expose
them or the tracker would need to re-parse the YAML `scale_groups` section.

Alternatively: add a `std::optional<std::vector<SeedPair>>` to `Tracker::Config` that
the CLI populates from the YAML, keeping the tracker itself free of YAML parsing.

### Validation

- Run `kotegaeshi-timo-scaled-ri.toml` and check that root_z at frame 1 is within
  0.1 m of the frame-250 steady-state value
- Check NIS at frames 1–10; should be lower than the current 291,000
- Check that 0-inlier frames (currently frames 2–14) are eliminated or at least reduced

---

## Related Issues

- The UKF `alpha = 0.1` issue (causes degenerate covariance from frame 50) should be
  fixed independently — increase to 0.5+ as per CLAUDE.md recommendation. Better
  initialization will not fully fix the convergence problem without also fixing alpha.
- See `docs/triangulated-distance-calibration-design.md` for the scale_groups schema
  that Level 2 would reuse.
