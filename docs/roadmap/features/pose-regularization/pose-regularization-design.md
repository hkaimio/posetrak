# Pose regularization for kinematically redundant chains — design sketch

> **Status (2026-07-07)**: Sketch only, not implemented. Written up after
> tracing a real tracking divergence (see *Motivation*) back to root/spine
> redundancy, and after confirming the underlying pattern — one joint in a
> redundant chain silently absorbing rotation until it hits its limit —
> recurs across other sequences as visibly unnatural (but non-divergent)
> poses. Complementary to, not a replacement for,
> `docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md`
> — see *Relationship to adaptive process noise* below.

## Goals

Give the solver a way to prefer, among many joint-angle splits that fit the
observed markers about equally well, the one that distributes rotation
sensibly across a kinematically redundant chain — rather than letting
whichever joint the local linearization finds "cheapest" to move absorb a
disproportionate share, all the way to its own limit if the motion demands
enough of it.

Two distinct symptoms motivate this, at two different severities:

1. **Visually unnatural poses that don't cause divergence.** Reported as a
   recurring pattern across multiple sequences, independent of the specific
   failure below — one spine joint (or root vs. spine) takes on far more
   rotation than the others in the chain for no anatomical reason, producing
   an implausible-looking pose even when the tracker never loses lock.
2. **Actual tracking divergence when the absorbing joint saturates.** The
   specific case that surfaced this (see below): `spine1` alone absorbs a
   forward-bend rotation until it hits its own configured limit, at which
   point the fit keeps degrading (nothing left in `spine1` to give), the
   outlier gate starts rejecting real observations, and once enough reject
   simultaneously the filter loses its correcting signal and diverges for
   ~75 steps.

## Non-goals

* **Not anatomical accuracy.** This is a redundancy-resolution heuristic —
  "prefer equal distribution and small total rotation among equivalent
  solutions" — not a biomechanical model of how a real spine actually
  distributes flexion. It's very likely wrong in detail for any specific
  person; it's there to avoid the *worse* wrongness of one joint absorbing
  everything.
* **Not a replacement for joint limits.** Limits remain the hard physical
  backstop (`enforce_joint_limits()`); this changes what solution the
  *update* step converges to before a limit is ever reached.
* **Not a general learned pose prior.** No motion model, no pose plausibility
  network — a narrow, explicit regularization term over a small, explicitly
  configured set of redundant joint chains.
* **Not changing sigma-point generation or the UKF's core predict/update
  structure** (same non-goal as the adaptive-process-noise note, and for the
  same reason: both proposed mechanisms below are additive to the existing
  measurement/process model, not a rewrite of it).

---

## Motivation: the diverging case, traced

Run `5dff7e33-feba-4164-929e-cd629912a45a` (`ukemi-tommi-20260509.db`),
t≈59.06-60.3s — full trace in the adaptive-process-noise design note's Case
3, reproduced briefly here because it's the concrete evidence for *this*
mechanism specifically:

- `spine1`'s X-axis angle hits its configured lower limit (`-0.122173` rad)
  at t≈59.08s and stays pinned there.
- The root/pelvis *does* correctly pick up the slack — its tilt-from-upright
  decreases correctly (65.0°→59.6°) through t≈59.17s, actively correcting
  toward the right answer.
- But the overall fit keeps degrading anyway (`NIS` climbs 1600→8267 across
  the same window) because `spine1` being maxed out is a standing
  model-mismatch root rotation alone doesn't fully absorb fast enough.
- At t≈59.18s, ~40% of observations (≈200 of 482) reject in a 3-4 step
  cascade. `spine1`'s Y-axis *also* saturates its limit at the same moment.
- With its correcting signal gone, the filter free-runs on extrapolation;
  root tilt reverses and grows from 61° to 106° over the next ~0.2s, and the
  pose doesn't recover until the real motion's arc brings it back into range
  around t≈59.75s (~75 steps later).

The proximate trigger is `spine1` saturating alone. If the same total
rotation had been split between `spine1`, `spine2` (and the root taking its
share earlier, before any single joint neared its wall), the model would
likely have kept up with the real motion without ever needing the outlier
gate to reject anything. That's the mechanism this note targets — not
directly the divergence itself (adaptive process noise / Mechanism B target
that), but the *reason* the redundant chain let one joint run out of room in
the first place.

## Relationship to adaptive process noise (Mechanisms A/B)

Different points in the same causal chain, not competing fixes:

- **This note (regularization)** targets the *trigger* — why `spine1`
  saturated alone instead of the rotation being shared. Fixing this reduces
  how often Mechanisms A/B even need to engage.
- **Mechanism A/B** target the *consequence* — once residual is growing
  (for whatever reason, including a saturated joint), keep the outlier gate
  from starving the filter of the real data it needs to keep correcting.

Worth trying regardless of which is implemented first; they should compose
without conflict since one changes what solution `update()` converges to and
the other changes how much process noise is available while it converges.

**Second motivation, independent of the divergence case**: a redundant
chain resolving to one degenerate direction is a textbook cause of an
ill-conditioned estimation problem. Case 2 in the adaptive-process-noise
note (`cov_condition_number` up to 3.8×10⁷ during a fast leg-swing) may be
the same underlying phenomenon showing up as a different symptom. If so,
regularization could improve both failure families at once — not yet
verified, worth checking once either mechanism exists to test against.

---

## Proposed mechanisms

### Alternative 1 — pseudo-measurement fused into `update()` (preferred)

Add synthetic residuals to the same weighted update that already fuses real
camera observations, rather than a separate correction pass. This is the
standard way to add a soft prior to a Bayesian estimator: it degrades
gracefully with strong contradicting real evidence (a genuinely one-sided
spine curl, if the cameras clearly show it, should still win) and only
"speaks up" when the real data is ambiguous or sparse in exactly the
redundant direction — which is exactly the failure case above (torso markers
too sparse to disambiguate root vs. spine1 vs. spine2 on their own).

**Two residual types, per configured chain** (e.g. `["spine1", "spine2"]`):

- **Equal-split**: for each pair of joints in the chain, per axis,
  `residual = angle_i − angle_j`, target 0. Discourages any one joint
  absorbing a disproportionate share.
- **Small-total-rotation / rest-pose pull**: for each joint in the chain, per
  axis, `residual = angle_i − rest_angle_i`, target 0, where `rest_angle_i`
  is that joint's own configured `rest_orientation` from the skeleton YAML
  (`Skeleton::Joint::rest_orientation`, already loaded per joint) — not an
  assumed zero, since not every rig's neutral pose is literally (0,0,0).
  Gently discourages using more total rotation than the data actually
  requires.

Each residual gets its own configurable noise std (the "stiffness" of that
spring — small std = strong pull, large std = easily overridden by real
data), likely two knobs per chain (`equal_split_noise_std`,
`rest_pose_noise_std`) rather than one, since the two express different
intents and will likely need different stiffness.

**Where this differs architecturally from the existing `Observation`/
`MeasurementMode` machinery**: `POSITION`/`VELOCITY`/`PAIR_DIFF` are all
pixel-space, `h(x) = project(marker, x)` — camera projection is intrinsic to
all three. A joint-angle residual has no camera and needs no FK/projection
at all: `h(x) = x.joint_angles()[i] - x.joint_angles()[j]` is a direct,
linear read of the state. Rather than forcing this into `Observation` (which
is fundamentally a camera/marker pair), the cleanest fit is a small,
separate set of pseudo-residuals computed directly from each sigma point's
`joint_angles()` and appended to the assembled measurement vector /
innovation covariance in `update()`, alongside (not instead of) the real
camera-based block — same Kalman gain, same unscented-transform machinery,
just a second, non-camera source of measurement rows.

**Config**: a list of redundant chains (each a joint-name list, same idiom
as `process_noise_vel_joint_names`), each with its two noise-std knobs.
Empty list = feature off, matching every other opt-in mechanism added so
far.

### Alternative 2 — mean-reverting process model bias

Instead of a measurement-side correction, bias `predict()` itself: after
constant-velocity propagation (and after the existing `vel_half_life_s_`
velocity damping), nudge each redundant chain's joint angles toward the
equal-split / rest-pose target by a per-step fraction — the same
exponential-decay pattern already used for velocity damping
(`alpha = pow(0.5, dt / half_life_s)`), applied to angles instead of
velocities:

```
target_i = mean(angle_j for j in chain)          // equal-split target
angle_i  = target_i + alpha * (angle_i - target_i)
```

and similarly toward `rest_orientation` with its own half-life.

**Why this is simpler in isolation but weaker overall** (matches the stated
preference for Alternative 1): it's a smaller, more localized code change —
a second post-propagation pass next to the existing velocity-damping block,
no new residual/measurement plumbing. But it applies *unconditionally* every
step regardless of how strong or ambiguous the real data is, so:

- it can't naturally soften when real cameras clearly resolve the ambiguity
  (a genuine, correctly-observed asymmetric pose gets a small persistent
  drag against it, forever, not just when needed);
- it composes awkwardly with Mechanism A and `vel_half_life_s_`, which
  already modify the same `predict()` region — ordering and interaction
  between three separate per-step adjustments needs care;
- it's a bespoke special case rather than fitting the "one noise diagonal /
  one measurement vector" shape the rest of the filter uses, so it's harder
  to reason about alongside everything else.

Worth keeping as a fallback if Alternative 1 turns out insufficient (e.g. if
the measurement-side correction is too weak to act before a joint nears its
limit), but not the starting point.

---

## Open questions

1. **Which chains get this, beyond spine?** Start scoped to
   `["spine1", "spine2"]` (the diagnosed case) — extend to
   `["neck1", "neck2"]` or others only if the same pattern is confirmed
   there. General mechanism, narrow initial configuration.
2. **Stiffness tuning.** Both noise-std knobs need empirical tuning against
   real sequences — no principled starting value yet. Start conservative
   (weak pull) and strengthen only as far as needed to prevent early
   saturation, to minimize risk of fighting genuinely correct asymmetric
   poses.
3. **Does equal-split need to be limit-aware?** I.e. should the target
   itself avoid pushing a joint toward a value it can't reach without
   hitting its own limit, or is it fine to let `enforce_joint_limits()`
   remain the sole backstop and let the pseudo-measurement just change what
   the *unconstrained* update converges to? Current thinking: the latter —
   keep the two mechanisms orthogonal — but worth confirming once
   implemented.
4. **Validation beyond the one diagnosed case.** Need example timestamps
   from the *other* sequences where this pattern has been visually observed
   as unnatural-but-non-divergent poses, to confirm the fix generalizes
   rather than being overfit to Case 3's specific geometry.
5. **Does this actually improve Case 2's ill-conditioning**, as speculated
   in *Relationship to adaptive process noise*? Untested — would need a
   direct before/after comparison on that run once this exists.

---

## Phasing

**Phase 1 — Alternative 1 (pseudo-measurement), scoped to `spine1`/`spine2`
only.** Add the two residual types to `update()`'s measurement assembly,
config-gated per redundant chain. *Validation*: re-run Case 3's segment,
confirm `spine1` no longer saturates alone (or saturates later/less
severely) for the same real motion, confirm the outlier cascade at t≈59.18s
either doesn't happen or is materially smaller; re-verify Cases 1 and 2 and
a calm segment don't regress; spot-check at least one of the other
sequences where unnatural-but-non-divergent poses were observed, to confirm
the pose looks more natural post-fix.

**Phase 2 (only if Phase 1 measured insufficient) — Alternative 2, or a
hybrid.** Gated on Phase 1 not adequately preventing early saturation in
practice; not worth building ahead of that evidence, same principle as every
other phase gate in the adaptive-process-noise note.
