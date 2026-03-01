# Skeleton Scaling Calibration — Design Review

## Problem Statement

Posetrak uses a hierarchical articulated skeleton to track human body pose from optical
marker data.  The skeleton is defined in a YAML file that encodes joint hierarchy, joint
types (revolute / spherical), and crucially **joint offsets** — the translation vectors from
each parent joint to its child, which determine bone lengths and joint positions.

A *default* skeleton is built from an average or reference body and its joint offsets are
fixed constants.  When tracking a subject whose body proportions differ from the reference,
the mismatch between the assumed and actual bone lengths produces **systematic, frame-level
biased errors** that are structurally impossible to compensate by adjusting joint angles alone.
Concretely:

- A bone that is 3 cm longer than assumed will cause the tracker to place all distal markers
  3 cm closer to the joint in model space at every frame, regardless of how good the UKF is.
- The residual marker error from this mismatch is correlated across the entire limb and
  across all frames — it is not zero-mean noise but a permanent offset.
- The tracker will partially absorb this via biased joint angle estimates (e.g., the elbow
  will appear slightly more bent than it really is), polluting the output in a way that is
  invisible to the user.

The only correct remedy is to calibrate the skeleton offsets to match each individual subject
before tracking begins.

---

## User Stories / Requirements

### US-1 — Per-subject calibration (primary)

> As a user who wants to track a specific person, I want to run a short calibration capture
> and have posetrak automatically produce a skeleton YAML whose bone lengths match that
> person's body, so that all subsequent tracking of that person is free of systematic
> bone-length bias.

**Acceptance criteria**:
- Given a calibration recording (≥ 30 s, free motion covering full range of joint angles),
  the tool produces a `<name>_calibrated.yaml` with updated `offset` values.
- Residual marker RMSE on the calibration sequence is lower with the calibrated skeleton
  than with the default skeleton.
- The calibrated YAML is a drop-in replacement: no changes to the normal tracking command
  other than `--skeleton calibrated.yaml`.
- The tool reports a per-group convergence status so the user knows which bone lengths are
  reliably estimated and which are not.

### US-2 — Scale group definition in YAML

> As a skeleton designer, I want to define named scale groups in the skeleton YAML that
> declare which joint offsets should be calibrated, so that calibration scope is explicit
> and controlled at the skeleton-definition level.

**Acceptance criteria**:
- An optional `scale_groups` key in the skeleton YAML lists named groups, each containing
  a list of joint names.
- Every joint in a group's `joints:` list receives **one independent** prismatic calibration
  DOF.  Grouping is for reporting and configuration only — it does not imply shared
  parameters.
- Joints not referenced in any group are not modified by calibration.
- The calibration tool validates that all referenced joint names exist in the skeleton.

### US-3 — Asymmetric body support

> As a user, I want left and right homologous bones to be calibrated independently so that
> natural body asymmetries are captured correctly without any special configuration.

**Acceptance criteria**:
- Every joint listed in a scale group always gets its own independent parameter, regardless
  of whether a bilateral counterpart is in the same group.
- After calibration the tool reports the estimated offset magnitude for each joint and emits
  a warning if left/right counterparts in the same group diverge by more than 5 %.
- Both values are written to the output YAML regardless; no forced symmetry is applied.

### US-4 — Convergence transparency

> As a user, I want to know whether the calibration has converged and which bone-length
> estimates are reliable, so I can decide whether to re-capture or accept the result.

**Acceptance criteria**:
- The tool prints a convergence table: one row per scale group with estimated length,
  uncertainty (posterior σ), and a CONVERGED / UNCERTAIN / NOT_OBSERVABLE status.
- CONVERGED: posterior σ < 5 mm and estimate stable over the last 2 s of the sequence.
- UNCERTAIN: posterior σ between 5–15 mm.
- NOT_OBSERVABLE: posterior σ > 15 mm or fewer than 240 frames with all group markers
  visible.
- The calibrated YAML is still written for UNCERTAIN groups; NOT_OBSERVABLE groups retain
  the default offset unchanged.

### US-5 — Non-destructive output

> As a user, I want the calibration tool to write a new YAML file rather than modifying
> the default skeleton in place, so that I can maintain one default skeleton and many
> per-subject calibrations.

**Acceptance criteria**:
- Output path defaults to `<tracking_dir>/skeleton_calibrated.yaml` if `--output` is not
  specified.
- The output YAML is a complete, valid skeleton file (not a patch/diff).
- All fields not modified by calibration (orientations, joint types, limits, markers,
  groups) are copied verbatim from the input YAML.

### Non-requirements (out of scope for this feature)

- Real-time online body-shape adaptation during normal tracking (left as a future extension).
- Marker attachment offset calibration (distinguished from bone-length calibration; a
  separate, harder problem requiring static poses).
- Calibration across multiple subjects simultaneously.

---

## Overall Assessment

The idea is sound and addresses a real need: a default skeleton will always have wrong bone
lengths for any particular person, and mis-calibrated lengths produce systematic biased errors
in every tracked frame (they cannot be corrected by the pose estimation).  The two-stage
approach — (1) run a calibration motion, (2) produce a scaled YAML — is the right shape.

The key technical question is *how* the scale parameters live in the system.  Two plausible
architectures are described below, with a recommendation.

---

## Resolved Design Questions

1. **Pinocchio model immutability** — resolved via `JointModelPrismaticUnaligned`.  Scale
   parameters enter through the configuration vector `q`; the Pinocchio model is built once
   and stays immutable.  See §Architecture Options.

2. **Symmetric vs. independent parameters** — resolved: every joint in a `scale_groups`
   entry always receives its own independent prismatic DOF.  The group name exists for
   display and configuration only.  Bilateral pairs (e.g., shin.L and shin.R) are independent
   by design; a divergence warning is emitted if they differ by more than 5 %.

3. **What scales?** — first version calibrates joint `offset` magnitudes only (bone lengths
   and joint socket positions).  Marker attachment offsets are a follow-on feature.

4. **Calibration motion** — free-form motion covering a broad range of joint angles is
   sufficient.  Guidance ("raise both arms, do a squat") improves conditioning but is not
   required.  Observability is reported per group after the run.

5. **Output destination** — the tool always writes a new file.  Default path:
   `<tracking_dir>/skeleton_calibrated.yaml`.  The input skeleton is never modified.

---

## Architecture Options

### Option A — Online state augmentation via prismatic joints (original idea, blocker resolved)

Scale parameters are modelled as **prismatic (sliding) joints** inserted between the parent
and child of each scaled bone.  Pinocchio has `JointModelPrismaticUnaligned(axis)` (confirmed
present in `/opt/openrobots`) which translates along an arbitrary axis in the parent frame.

The key insight is that the Pinocchio **model stays completely immutable** — scale enters
through the configuration vector `q`, exactly like joint angles.  No model rebuild, no FK
hack.

**Model transformation for a scaled bone**

```
Before:  parent ──(offset = L·ĉ, rest_rot = R)──► child_joint
After:   parent ──(offset = 0)──► prismatic_joint(axis=ĉ, q=L) ──(offset=0, rest_rot=R)──► child_joint
```

- `ĉ = normalize(original_offset)` — bone's longitudinal axis in parent frame
- Prismatic `q` initialized to `L = |original_offset|` (the nominal bone length)
- Child joint's `offset` becomes zero (prismatic now fully handles the translation)
- `q_prismatic` IS the bone length in metres; the scale factor is `s = q / L_nominal`

For UKF purposes the prismatic DOF is identical to a revolute joint: nq=1, nv=1.  The
process model is constant-value with near-zero process noise (`σ ≈ 0.1 mm/√s`).

**State vector impact**: each scale parameter adds 1 position DOF + 1 velocity DOF to the
state.  For K=8 scale groups (upper/lower arm ×2, upper/lower leg ×2, torso, hip_width),
this adds 16 to the state dimension — increasing sigma-point count from 2n+1 to 2(n+16)+1.
Marginal computational cost at 120 Hz.

**Calibration lifecycle**:
1. Load default skeleton YAML → insert prismatic joints for all scale-group bones → build
   Pinocchio model (one-time, immutable)
2. Run UKF + RTS smoother on calibration sequence; scale DOFs converge
3. Extract posterior `q_pris` for each prismatic joint (precision-weighted mean over smoothed
   frames; see §Representative value selection)
4. Write calibrated YAML: update `offset` of each child joint to `q_pris * ĉ`, strip
   prismatic joints from the joint list

The calibration YAML is then used for all subsequent normal tracking (no prismatic joints,
no extra state DOFs).

**Offset vector semantics**: the prismatic axis is `ĉ = normalize(original_offset)`, so the
prismatic DOF scales the *magnitude* of the full 3D offset vector while preserving its
direction.  This is biomechanically correct for all joint types:

- **End-of-bone joints** (forearm, shin, hand, foot): offset is nearly pure along the
  bone axis; the prismatic directly encodes bone length.
- **Socket/junction joints** (thigh, shoulder): offset is a diagonal 3D vector encoding
  both lateral displacement and height drop of the socket from the parent origin.  Scaling
  its magnitude moves the socket in the correct anatomical direction — away from the pelvis
  / top of spine — proportionally in all three components.

Crucially, anatomically distinct quantities (hip socket position vs. femur length) are
always in **separate scale groups with separate joints** and thus get fully independent
prismatic DOFs.  For example:

```
hips
 ├─ prismatic_thigh.L  → thigh.L    (hip socket, calibrates socket reach from pelvis)
 │     └─ prismatic_shin.L  → shin.L (femur length, fully independent of socket)
 └─ prismatic_thigh.R  → thigh.R    (independent of .L)
```

A woman with wider hips and shorter femurs simply converges to a larger `|thigh.L offset|`
and a smaller `|shin.L offset|` than the reference.  No coupling.  Helper bones are not
needed for the 1-DOF-per-joint model.  Independent orthogonal components of a single offset
(e.g., hip-width vs. hip-drop independently) could be addressed with helper bones in a
future extension, but the single-scalar model captures the dominant anatomical variation.

**Advantages**
- Pinocchio model immutable — no changes to FK, UKF sigma-point loop, RTS smoother, or
  any existing infrastructure.
- Scale uncertainty propagated through the full Bayesian filter; posterior covariance of
  each prismatic DOF gives per-group convergence diagnostics for free.
- RTS smoother retroactively refines scale estimates from all future observations.
- Online adaptation possible: leave prismatic DOFs active during normal tracking with very
  tight process noise, so the tracker can compensate for systematic clothing/equipment effects.

**Disadvantages**
- Skeleton YAML and model-builder changes required to support prismatic joint type.
- Calibration skeleton (with prismatics) is different from the tracking skeleton (without).
  The two-YAML workflow must be clearly communicated to users.
- Must verify all downstream code (SkeletonLayout, state export, joint angle CSVs) gracefully
  handles a new joint type.

---

### Option B — Alternating estimation (recommended)

Rather than online joint estimation, decouple scale calibration into a separate pass that
**alternates** between pose estimation and bone-length regression.

```
Loop until convergence:
  1. Run UKF + RTS smoother on the calibration sequence with current skeleton
  2. For each scale group g, solve a weighted least-squares problem:
       argmin_sᵍ Σ_t Σ_m ||y_{m,t} - ŷ_m(q_t, sᵍ)||²  (linear in sᵍ given q_t)
  3. Update skeleton offsets:  offset_j ← s_j * offset_j⁰
  4. Check convergence: ||Δs|| < ε
Output: calibrated YAML
```

**Why the inner problem is linear in `s`**: given fixed joint angles `q_t`, the world position
of marker `m` attached to joint `j` is:

```
p_m(q_t, s) = R_j(q_t) · (s_group(j) · offset_j) + ... = s_group(j) · [R_j(q_t) · offset_j] + rest
```

Each scale group `g` contributes linearly to the marker predictions of its joints.  So step 2
is a simple linear least-squares that can be solved in closed form with a small matrix
(number of equations = N_markers_in_group × N_frames; number of unknowns = 1 or 2 per group).

**Advantages**
- No changes to Pinocchio model, UKF, or the state vector at all.
- The inner LS problem is trivially solved; no UKF plumbing needed.
- Scales and pose errors are decoupled, which is actually *better* numerically because the
  UKF can converge on poses without being confused by scale uncertainty, then scale
  regression refines lengths.
- Convergence is easy to monitor: `||Δs_k - Δs_{k-1}||`.
- Works with the existing smoother; use smoothed states `x̂_{k|N}` as `q_t` in the LS step.

**Disadvantages**
- Multi-iteration; typically 3–10 iterations needed.
- If scale and pose are highly correlated for a given sequence, alternating can converge slowly
  or to a local minimum (e.g., "shorter arm, more extended elbow" is degenerate).

---

## Handling Scale Parameters in State (Option A — prismatic joints)

| Aspect | Recommendation |
|--------|---------------|
| State representation | `q_pris ∈ ℝ` = bone length in metres; scale factor `s = q/L₀` |
| Error-state | Additive `Δq_pris` (Euclidean — no manifold complications) |
| Process model | Constant value: `q_{k+1|k} = q_k`, `Q_pris = σ² · dt · I` with `σ ≈ 0.1 mm/√s` |
| Positivity | Clamp `q_pris > 0` in sigma-point generation (e.g., min = 10 mm) |
| Initial uncertainty | `P_pris^0 = (0.15 · L₀)²` (±15% of nominal bone length at 1-σ) |
| Initial value | `L₀ = \|offset\|` from default skeleton YAML |
| Velocity DOF | 1 (rate of bone length change, m/s) — initialised to 0 with small covariance |

---

## Representative Value Selection

After the calibration sequence, the question is which frame's estimate to use (or how to
aggregate).

**Recommended**: precision-weighted mean of the posterior means over the smoothed sequence,
restricted to a "well-observed" mask.

```
s̄ = (Σ_t w_t · ŝ_{t|N}) / (Σ_t w_t)
where
  w_t = 1 / P_{s,t|N}   (inverse posterior variance, scalar per group)
  well-observed: only include frames where w_t > w_threshold
```

This automatically down-weights frames where the marker configuration is degenerate for
estimating a particular bone (e.g., arm fully extended = zero cross-section area, hard to
observe length).

Alternative/simpler: **median** of posterior means — robust to outlier frames caused by
occlusion or marker swaps.

**Do not** use the minimum-variance frame alone; it may be a single outlier frame.

---

## Convergence Criteria

### For the online filter (Option A)

Declare the scale group `g` converged at time `t` if both:

1. **Posterior precision sufficient**: `P_{s_g, t|t} < σ²_converged` (e.g., `(0.005 m)²`)
   — posterior uncertainty < 5 mm (for a bone in the ~0.3 m range this is < 2%)
2. **Estimate stable**: running mean of `ŝ_{g,t}` over a 2-second window has standard
   deviation < threshold, e.g., `std(ŝ_g[t-240:t]) < 0.005`

Output a per-group convergence flag at the end of the calibration run with the last achieved
`P_{s_g}` so the user knows which groups are reliable.

### For the alternating estimator (Option B)

Outer loop converges when:
- `max_g |s_g^{(k)} - s_g^{(k-1)}| < 0.002` (< 2 mm change in scale factor applied to
  a 1 m bone)
- Or `k > k_max` (e.g., 10) — always terminate

---

## Observability and Potential Failure Modes

### Identifiability issue: scale vs. joint angle

For a two-link chain (e.g., shoulder → elbow → wrist), the marker at the wrist satisfies:

```
p_wrist = p_shoulder + s_upper · L_upper · d̂_upper + s_lower · L_lower · d̂_lower
```

If the arm is always in the same configuration, `d̂_upper` and `d̂_lower` are fixed and the
two scales cannot be separately identified from the wrist marker alone.  The elbow marker (if
present) breaks this degeneracy.

**Recommendation**: require that all markers in a scale group's chain of joints be visible for
at least `N_min` frames (e.g., 240 frames at 120 Hz = 2 seconds) to declare that group
observable.  Warn the user otherwise.

### Torso vs. root translation ambiguity

The "back" scale parameter (spine length) is correlated with root translation (the whole
skeleton can shift up without changing marker residuals if there are no ground-plane
constraints).  This is best handled by including a ground-contact or height-from-ground
constraint during calibration, or by fixing the root Z position for the regression step.

### Marker attachment offsets

If `marker_offset` (position of sensor relative to joint in local frame) is also wrong, the
scale regression will absorb some of that error into the bone length.  These are not
separately identifiable without rich motion sequences.  Best practice: run an initial
calibration with a static frame (T-pose or similar) to estimate marker offsets first, then
bone lengths from dynamic motion.  This is a follow-on feature.

---

## Scale Group Definition (YAML Extension)

Add an optional top-level `scale_groups` key to the skeleton YAML.  Each group lists joint
names; **each joint receives exactly one independent prismatic DOF**.  The group name is used
only for the convergence report and has no effect on estimation.

```yaml
scale_groups:
  # Hip socket position — encodes where each thigh joint sits relative to the pelvis.
  # Independent of femur length; a wider pelvis does not imply longer femurs.
  - name: hip_socket
    description: "position of hip joints relative to pelvis origin"
    joints: [thigh.L, thigh.R]      # 2 independent DOFs

  # Femur length — from hip socket to knee.
  - name: femur
    joints: [shin.L, shin.R]        # 2 independent DOFs

  # Tibia length — from knee to ankle.
  - name: tibia
    joints: [foot.L, foot.R]        # 2 independent DOFs

  # Clavicle / shoulder reach — position of shoulder joint relative to top of spine.
  - name: shoulder_reach
    description: "position of shoulder joints relative to top of spine"
    joints: [shoulder.L, shoulder.R]

  # Upper arm — from shoulder to elbow.
  - name: upper_arm
    joints: [upper_arm.L, upper_arm.R]

  # Forearm — from elbow to wrist.
  - name: forearm
    joints: [forearm.L, forearm.R]

  # Spine segments — each independent (lumbar ≠ thoracic length).
  - name: spine
    joints: [spine1, spine2]

  # Neck
  - name: neck
    joints: [neck1]
```

**Semantics**:
- No `symmetric` flag — bilateral symmetry is never enforced.  Left/right always converge
  independently.  A >5 % divergence between bilateral counterparts triggers a report warning.
- The `description` field is optional and informational only.
- Joints absent from all scale groups are not modified by calibration.
- The model builder inserts `JointModelPrismaticUnaligned(normalize(offset))` before each
  listed joint.  The prismatic `q` is initialized to `|offset|`.

**Known naming debt**: the current skeleton uses joint names that mix joint and bone naming
conventions (e.g., `shin` refers to the knee joint, `thigh` to the hip socket).  This will
be corrected in a future skeleton rename — scale group descriptions should be updated at
the same time.

---

## CLI Design — Subcommand Architecture

Calibration is exposed as a subcommand of the main `posetrak` binary:

```
posetrak scale \
  --skeleton default.yaml \
  --input-dir <recording_dir> \
  --output <calibrated.yaml> \
  [--convergence-tol 0.002] \
  [--min-observable-frames 240]

posetrak track \
  --skeleton calibrated.yaml \
  --input-dir <recording_dir> \
  ...
```

The subcommand structure is introduced simultaneously with this feature.  `main.cpp` becomes
a thin dispatcher:

```cpp
int main(int argc, char** argv) {
    if (argc < 2) { print_usage(); return 1; }
    std::string cmd = argv[1];
    if (cmd == "track")    return run_track(argc - 1, argv + 1);
    if (cmd == "scale")    return run_scale(argc - 1, argv + 1);
    if (cmd == "validate") return run_validate(argc - 1, argv + 1);
    print_usage(); return 1;
}
```

File layout:
```
cli/
  main.cpp        — subcommand dispatch only
  cmd_track.cpp   — current track.cpp content
  cmd_scale.cpp   — calibration subcommand
```

`posetrak scale` internally (Option A):

1. Load default skeleton YAML, parse `scale_groups`
2. For each listed joint, insert a prismatic DOF → build calibration Pinocchio model
3. Run UKF + RTS smoother on the calibration recording
4. Compute precision-weighted mean of smoothed posterior for each prismatic DOF
5. Print convergence table (CONVERGED / UNCERTAIN / NOT_OBSERVABLE per joint)
6. Write `calibrated.yaml`: set `offset ← q_pris_mean · normalize(offset₀)` for each
   calibrated joint, omit `scale_groups` key (calibrated skeleton needs none)

---

## Recommendation

**Use Option A (prismatic joints in state)** now that the Pinocchio blocker is resolved.

The prismatic joint approach is strictly superior to Option B when you want uncertainty
propagation and RTS smoothing on the scale parameters.  The implementation is incremental:

1. Add `PRISMATIC` joint type to `skeleton.hpp` and the YAML loader
2. Extend `PinocchioModelBuilder::add_joint_recursive` to emit a
   `JointModelPrismaticUnaligned(ĉ)` before the child joint when a `scale_group` is
   indicated
3. Extend `SkeletonLayout` to include prismatic DOFs (identical bookkeeping to REVOLUTE)
4. Add `scale_groups` parsing to the skeleton YAML loader
5. Write `calibrate` CLI subcommand that takes a calibration sequence, runs the tracker
   with a prismatic-augmented model, extracts posterior bone lengths, and writes a
   stripped calibrated YAML

Option B remains a viable simpler alternative if implementation time is a concern — it can
produce a calibrated YAML without any C++ changes (pure Python, ~300 lines).  The two
approaches can also be combined: Option B for a fast first-pass initialisation of bone
lengths, Option A for a final high-fidelity refinement with Bayesian uncertainty.

---

## Open Issues Not Addressed Here

- **Foot/ankle**: difficult to scale correctly without foot-marker ground constraints
- **Marker attachment offset calibration**: orthogonal problem; T-pose + linear solve
- **Multi-person**: each person needs their own calibration; `--subject-id` flag on CLI
- **Convergence visualization**: a simple per-group scale-vs-iteration plot would be useful
  for diagnosing poor-observability groups before committing to the output YAML
