# Swing-twist joint orientation & limits — design sketch

**Status**: Design sketch, not implemented. Written to capture a concrete
proposal while the underlying diagnosis is still fresh; expect revision
once the Karcher-mean instrumentation results (see
`docs/roadmap/features/tracking-crisis-debugging-log.md`) are in.

## Motivation

Two previously-separate-looking findings turned out to be the same
mechanism, viewed from different angles:

1. **The box-corner effect** (crisis B, t=58-61s): `upper_arm.L/R` land
   within a few degrees of their configured limits on 2-3 axes
   simultaneously, coinciding with a covariance-conditioning jump and
   cascading tracking failure.
2. **The frame-227/228 event** (t≈39.9s, a separate part of the trial):
   `upper_arm.L`'s stored rotation-vector *magnitude* (the axis-angle
   representation's norm, i.e. the total rotation angle from identity)
   climbs smoothly from 113° to 173.9° over 32 frames — getting within
   6° of the 180° (π radian) boundary — then discontinuously snaps to
   48.9° at the very next frame. A synthetic experiment (perturbing the
   174°-magnitude nominal by a modest 10° in random directions) showed
   the *raw stored vector* can deviate by up to 353° between two sigma
   points whose *actual represented rotations* differ by only ~16° —
   exactly the ">360°" spread symptom seen in earlier debugging sessions.

**The unifying cause**: `SPHERICAL` joints store orientation as a
3-vector rotation vector (direction = rotation axis, norm = rotation
angle in radians) — see `forward_kinematics.cpp`'s
`Eigen::AngleAxisd(angle, angles/angle)`. Configured joint limits are
`std::clamp()`'d directly on this vector's *individual x/y/z components*
in `enforce_joint_limits()`. Nothing stops the vector's *norm* — which is
what actually matters for the representation's own topology — from
climbing toward or past π even while every individual component stays
within its own configured bound. Checked concretely for `upper_arm`: the
box corner `(x=160°, y=45°, z=-150°)` has `‖(2.79, 0.785, -2.62)‖ ≈ 3.91
rad ≈ 224°` — past the boundary. And frame 227's actual (not even
cornered) state, `(2.36°, -0.43°, -1.86°)` in radians, already reaches
174° purely from x and z both being moderately large at once.

Near that boundary, the rotation-vector parameterization is a genuine
2-to-1 covering of SO(3) (vectors `v` and `-v` with `‖v‖=π` represent the
identical rotation), and the codebase's canonicalization convention
(`quaternion_to_axis_angle()` forces quaternion `w≥0`, i.e. angle
`∈[0,π]`) creates a discontinuity exactly there: two physically-close
rotations can canonicalize to wildly different raw vectors depending on
which side of the boundary they land on.

Traced sigma-point generation (`apply_error_to_state`), the weighted mean
(`compute_state_mean`), and the covariance/error computation
(`compute_state_error`) for `SPHERICAL` joints — all three already do
proper SO(3)-aware composition/relative-rotation math, not naive raw
vector arithmetic, so this isn't a simple "forgot manifold-aware code"
bug. The corruption path from "raw vector spreads wildly near π" to
"tracking actually breaks" isn't fully pinned down yet (see the open
Karcher-mean-convergence instrumentation task), but regardless of the
exact mechanism, the *representation itself* structurally invites this
problem: nothing about "reasonable per-axis limits" prevents their
combination from reaching an unreasonable *total* rotation.

## Proposed representation: swing-twist decomposition

Decompose a spherical joint's orientation as `R = R_swing · R_twist`,
where:

- **Twist**: rotation about the joint's own fixed "bone axis" `ê` (a unit
  vector in the joint's rest/local frame — e.g. the humerus's long axis
  for the shoulder). A single scalar angle (e.g. humeral
  internal/external rotation).
- **Swing**: rotation that tilts the bone axis away from its rest
  direction, with *no* twist component — by construction, the swing
  rotation's own axis is always perpendicular to `ê`. Naturally
  parameterized as a 2D vector `(swing_x, swing_y)` in the plane
  perpendicular to `ê`: direction gives the swing axis, magnitude gives
  the swing angle.

Storage footprint is unchanged (still 3 doubles per joint:
`swing_x, swing_y, twist`), but the *semantics* change fundamentally —
swing magnitude and twist angle are each bounded independently by
construction, rather than being 3 components of a vector whose combined
norm can exceed any individual axis's own sensible range.

**Standard decomposition formulas** (well-established, not novel — used
routinely in animation/robotics for exactly this reason):

```
q = q_swing * q_twist
q_twist = normalize(w=q.w, xyz=(q.xyz · ê) * ê)   // projection onto the twist axis
q_swing = q * q_twist⁻¹
twist_angle = 2 * atan2(q.xyz · ê, q.w)
swing_axis_angle = quaternion_to_axis_angle(q_swing)  // guaranteed ⟂ ê
(swing_x, swing_y) = project swing_axis_angle onto the plane ⟂ ê
```

**Why this avoids the π problem structurally**: an anatomically sane
shoulder has `max_swing_angle` on the order of 90-120° and a twist range
on the order of ±90°. Because swing is capped as its *own* independent
parameter (not a component summed with others via vector norm), and
twist is a separate, typically-small rotation about a different axis,
the total combined rotation angle of `R_swing · R_twist` stays
comfortably below π for any anatomically plausible limit configuration —
there's no coordinate combination that can inadvertently reach 220°.

This also happens to match how biomechanics actually describes
ball-joint range of motion (a cone of circumduction plus an independent
axial-rotation range), which is a genuine accuracy improvement over the
current per-axis box, not just a numerical workaround — ties into the
"not anatomical accuracy" caveat already flagged in the soft-joint-limits
design doc.

## What has to change

This is a representation change, not just a new tracker-config knob —
every place that currently treats
`state.joint_angles().segment<3>(state_index)` as a raw axis-angle vector
for a `SPHERICAL` joint needs to instead treat it as
`(swing_x, swing_y, twist)` and go through matching
reconstruct/decompose functions:

- **`forward_kinematics.cpp`** (FK): replace
  `AngleAxisd(angle, angles/angle)` with swing-twist reconstruction
  (`R = R_swing(swing_x, swing_y) · R_twist(twist)`).
- **`sigma_points.cpp`** (`apply_error_to_state`): currently composes
  `R_new = R_nominal · R_error` (correct, keep this part) then stores the
  result via `quaternion_to_axis_angle` (needs to become swing-twist
  *decomposition* instead).
- **`ukf.cpp`** (`compute_state_mean`, `compute_state_error`): both
  reconstruct rotation matrices from the stored 3-vector via
  `axis_angle_to_quaternion` today; need the swing-twist reconstruction
  instead, and the mean/error results need swing-twist decomposition
  back into storage, not raw axis-angle.
- **`enforce_joint_limits()`**: change from per-axis `std::clamp()` to
  clamping `‖(swing_x,swing_y)‖` against `max_swing_angle` (possibly
  elliptical/anisotropic — real shoulders don't have a circular swing
  cone) and `twist` against `[twist_lo, twist_hi]` independently.
- **Skeleton YAML format / `SkeletonLoader`**: needs a new limit
  representation for opted-in joints (`bone_axis`, `max_swing_deg` or an
  elliptical equivalent, `twist_lo/hi`) alongside or instead of the
  current per-axis `x/y/z` limits. Needs a per-joint flag or new type
  distinguishing "box-limited spherical" from "swing-twist-limited
  spherical," since a full-skeleton conversion isn't realistic to do
  (and validate) in one step.
- **`InverseKinematics`**: currently presumably treats the same 3-vector
  as raw axis-angle when solving; needs updating or at least verifying
  it doesn't silently misinterpret the new semantics for opted-in
  joints.
- **CSV export (`joint_angles.csv`)**: `angle_x/angle_y/angle_z` columns
  currently mean raw rotation-vector components. For a swing-twist joint
  these would mean something different (`swing_x, swing_y, twist`) —
  needs either a column-semantics flag per joint or a documented
  divergence, since existing analysis scripts (including several used
  throughout this debugging arc) assume raw-axis-angle semantics and
  would silently misinterpret the new columns otherwise.
- **`set_soft_joint_limits()` / `set_near_limit_damping()`**: both
  currently operate per-axis against `j->limits[axis]`; would need a
  parallel swing-magnitude-based formulation for opted-in joints (though
  given a swing-twist joint structurally can't reach the π boundary with
  sane limits, `set_near_limit_damping()` specifically may simply become
  unnecessary for those joints — its whole purpose was mitigating a
  problem this representation avoids by construction).

## Open questions

1. **Bone-axis convention per joint**: needs a documented, per-joint
   choice of which local axis is "the bone" (twist axis) — presumably
   derivable from each joint's existing `orientation`/rest-pose data in
   the skeleton YAML, but needs to be made explicit and validated.
2. **Circular vs. elliptical swing cap**: a real shoulder's swing range
   isn't a circular cone (very different limits in flexion/extension vs.
   abduction/adduction directions). A circular cap is a simpler first
   cut; an elliptical one is more anatomically accurate but needs a
   2-parameter limit instead of 1.
3. **Migrating existing box limits**: the current per-axis limits (in
   degrees on x/y/z) don't map cleanly onto swing/twist. Options: (a)
   approximate via the box's own geometry (e.g. use the smallest
   enclosing cone), which will be conservative/lossy, or (b) recalibrate
   limits from scratch using proper swing-twist conventions and updated
   calibration poses. (b) is more correct but a bigger undertaking,
   especially given the earlier finding that `shoulder.L/R`'s current
   scaled-skeleton limits may already be miscalibrated independent of
   this representation question.
4. **New JointType vs. a flag on SPHERICAL**: whether this is a new enum
   value or a per-joint config flag on the existing `SPHERICAL` type is
   an implementation-detail choice, but affects how much existing
   `switch`/`if` logic across the codebase needs touching either way.
5. **Does this fully replace Proposal 1 (near-limit damping) and
   Proposal 2 (parent-joint redistribution)?** Near-limit damping is
   very likely obsoleted for opted-in joints — the mechanism it was
   trying to mitigate (sigma spread near a representation boundary)
   shouldn't arise once swing is capped well below π. Proposal 2 (shift
   rotation to the parent/clavicle when the child joint is maxed) is
   orthogonal — still potentially useful for anatomical realism
   independent of which representation the child joint uses, but no
   longer strictly necessary to avoid the numerical pathology this
   design specifically targets.

## Phasing recommendation

Scope Phase 1 to `upper_arm.L`/`upper_arm.R` only, matching the pattern
established for soft joint-limits and near-limit damping — validate on
the two joints actually implicated in the traced failures before
considering a broader rollout. This is a materially bigger lift than
either of those two mechanisms (touches FK, sigma-point math, the
skeleton file format, and IK, not just a new pseudo-measurement or
process-noise scaling term), so it warrants its own scoped
implementation/validation pass rather than folding into the existing
adaptive-process-noise config surface.

## Relationship to other findings in this investigation arc

- Confirms and generalizes the "box corner" diagnosis from crisis B: the
  corner problem *is* the π-proximity problem for a box expressed on raw
  rotation-vector components, not a separate Euler-angle nonlinearity
  issue as originally hypothesized.
- Directly explains why Proposal 1 (near-limit process-noise damping)
  didn't touch the frame-227/228 jump: damping the sigma spread doesn't
  change where the *representation itself* has a topological
  discontinuity — the fix needs to move the joint's operating range away
  from that discontinuity, not shrink the noise around it.
- The still-open Karcher-mean-convergence question (does the hard-capped
  10-iteration mean-finding loop actually fail to converge robustly near
  the injectivity radius, or does the corruption enter some other way?)
  matters for understanding *how* the representation problem propagates
  into a tracking failure, but doesn't change the recommended fix here —
  regardless of the exact corruption pathway, keeping the joint's
  operating range away from π by construction removes the precondition
  for any of these downstream mechanisms to fire.
