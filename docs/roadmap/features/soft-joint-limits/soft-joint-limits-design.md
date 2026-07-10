# Soft joint-limit repulsion — design sketch

> **Status (2026-07-08)**: Sketch only, not implemented. Written up after
> tracing the "arms completely lost" crisis (t≈59.1-66s,
> `ukemi-tommi-20260509.db`, sequence `a5da88ea-f7ba-4e0e-bbd4-43c68205dcf6`)
> back to `upper_arm.L`/`upper_arm.R` hitting their configured ball-joint
> limits, after extensive adaptive-process-noise tuning (Mechanisms A/B, see
> `docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md`)
> failed to touch this specific failure at all. Reuses the pseudo-measurement
> pattern from
> `docs/roadmap/features/pose-regularization/pose-regularization-design.md`
> (`apply_pose_regularization()` in `ukf.cpp`); read that doc first, this one
> assumes familiarity with the mechanism it establishes.

## Goals

Discourage a joint from approaching its own hard rotation limit *before* it
gets there, rather than only reacting once `enforce_joint_limits()` clamps it
after the fact. The clamp is a necessary backstop, but by the time it fires
the Kalman update has already overshot — this mechanism aims to make that
overshoot rare instead of routine.

## Non-goals

* **Not a replacement for `enforce_joint_limits()`.** The hard clamp remains
  the actual backstop that guarantees the stored state never exceeds
  `[lo, hi]`. This mechanism only changes what solution the *unconstrained*
  Kalman update converges to, same relationship pose-regularization has to
  the clamp.
* **Not anatomical accuracy.** Same caveat as pose-regularization: a linear
  repulsion starting at a fixed margin from each configured limit is a crude
  approximation of real joint behavior (which doesn't have a hard wall at
  all — see *Open questions* on parent-joint redistribution), not a
  biomechanical model.
* **Not changing the limit values themselves.** Whether `upper_arm`'s ±45°
  y-axis limit is anatomically correct is a separate question (initial
  read: probably close — see *Motivation*) from whether the *filter's*
  approach to that limit is well-behaved. This note is only about the
  latter.
* **Not redistributing rotation to parent joints.** `enforce_joint_limits()`
  currently clamps each joint's own angle independently — no mechanism
  pushes "leftover" rotation up to `shoulder.L/R` (the clavicle) when
  `upper_arm.L/R` maxes out, so real scapulohumeral-rhythm-style
  compensation isn't modeled at all. Worth a future note of its own; out of
  scope here (see *Open questions*).

---

## Motivation: the traced case

Full-trial run `824154de-1e8c-499a-9a5c-e544c9e148b3` (config
`e7bb4c0b-6473-4f6c-a0f9-acc699177ba0`, proximal/distal adaptive-process-noise
split), t≈59.09-66s. Frame numbers below are `tracker_step` at 120Hz,
`t = 38.1 + step/120`.

- **Steps 2505-2517 (t=58.98-59.08s): nominal.** Covariance condition number
  steady around 5×10⁵, 420+ inliers/step, worst per-marker mahalanobis
  distance in the 4-5.5 range (fingers, knee — normal noise).
- **Step 2518 (t=59.083s): covariance conditioning degrades first**, one
  frame before anything hits a limit. Condition number jumps 5.3× (508K →
  2.71M), min eigenvalue roughly halves (2.50e-6 → 9.99e-7).
  `MRK-elbow.R`'s mahalanobis distance rises to 3.87 — new, wasn't among the
  worst markers before. `MRK-elbow.R` is the marker attached near
  `upper_arm.R`/`forearm.R`, i.e. the joint that clamps next.
- **Step 2519 (t=59.092s): `upper_arm.R` first hits its limit**, on x and z
  simultaneously. `MRK-elbow.R` is now the single worst marker (mahal 6.92).
  Inliers start dropping (424 → 397).
- **Step 2520 (t=59.100s): `upper_arm.R` locks onto its y-axis limit
  (+45°, `0.7853981852531433` rad — exact bit-for-bit match to the
  configured bound)**, and `upper_arm.L` clamps on x the same step. Inliers
  crash 397 → 254; both arms show catastrophic mahalanobis (`elbow.L` 22.2,
  `wrist.L` 19.3, `elbow.R` 13.2, `wrist.R` 10.9).
- **Steps 2520-onward: recurring, not sustained.** `upper_arm.R`'s y-axis
  limit gets hit and released repeatedly through t≈66s (end of the tracked
  range) rather than locking once and staying stuck — bursts of 1-22 frames
  at 2520-2534, 2569, 2571-2572, 2620-2623, 2626-2638, 2655-2669, 2805-2827
  (longest single run, 0.18s), 2989-2997, 3341-3354.

Both arms hitting their proximal-joint limits within the same one-to-two-
frame window is consistent with a genuinely fast, bilateral motion (matching
the original "bilateral hand-raise" case this whole investigation started
from) pushing both shoulders toward a real anatomical limit at close to the
same moment — the person measured their own shoulder abduction limit at
close to 45°, so the *limit value* isn't obviously wrong. What's wrong is
the *filter's* behavior once it's approached: raw camera detections stay
smooth and continuous through this whole window (checked via
`tracking_obs_results.obs_blob`'s `actual_x/actual_y`, frames 2448-2568 on
`MRK-wrist.R`), so this isn't an observation-quality or occlusion problem —
coverage during t=58-61s averages 4.0-4.4 of 5 cameras, never below 2.
The divergence is internal to the estimator.

**Likely mechanism**: the bilateral raise increases both `upper_arm.L/R`'s
velocity; Mechanism A's adaptive process noise inflates their variance in
response (by design — that's its job); the wider sigma-point spread
interacts badly with FK's nonlinearity right at the limit boundary; the raw
(pre-clamp) Kalman update overshoots past `[lo, hi]`; `enforce_joint_limits()`
clamps the *position* but `damp_velocity_covariance_at_limits()` only shrinks
the *velocity* covariance for the clamped DOF, leaving the state's own
uncertainty about its position inconsistent with the deterministic override
that just happened. That inconsistency plausibly explains why recovery
oscillates (repeated clamp/release bursts) rather than settling immediately
once the true motion re-enters the representable range.

This means Mechanism A's adaptive process noise — built to help exactly this
kind of fast-motion case — was, for this specific failure, making the
overshoot larger rather than smaller: more slack for a joint that's
accelerating toward a hard wall widens the sigma cloud right when a narrow
one would behave better. Explains why no variant of the arms-scope /
proximal-distal-split tuning this session touched this crisis at all (see
`adaptive-process-noise-design.md`'s Case 3 and this doc's sibling
investigation) — it isn't a noise-budget problem.

## Relationship to adaptive process noise and pose regularization

Three additive mechanisms now operating on different parts of the same
causal chain:

- **Adaptive process noise (Mechanism A/B)**: reacts to *velocity* and
  *residual growth* respectively, at the whole-scope level. Doesn't know
  anything about joint limits — can (as observed here) make an
  already-marginal joint's overshoot worse by widening its uncertainty right
  as it approaches a wall.
- **Pose regularization**: discourages one joint in a *redundant chain*
  (multiple joints that can trade off to produce the same marker positions)
  from silently absorbing all the rotation and saturating alone. Different
  failure shape — this doc's case isn't about redundancy, `upper_arm.R`
  has no sibling joint to share rotation with (see *Non-goals* on parent
  redistribution).
- **This note**: acts directly on the *approach to a hard limit*, regardless
  of why the joint got there (fast motion, redundancy absorption, or
  anything else). Should compose with both of the above without conflict —
  same "additive pseudo-measurement, same sigma-point/Kalman-gain machinery"
  shape as pose-regularization, just a different residual definition.

---

## Proposed mechanism

Same pattern as pose-regularization's Alternative 1
(`apply_pose_regularization()`): a second, small Kalman update pass, run
alongside (not instead of) the real camera-observation correction, using the
same sigma-point / cross-covariance / Kalman-gain machinery. New residual
type instead of a new architecture.

**Per configured joint axis**, with hard limits `[lo, hi]` (already stored
per-joint in `JointDesc::limits`) and a configured margin `m`:

```
interior_lo = lo + m
interior_hi = hi - m
residual(θ) = max(0, θ - interior_hi) + min(0, θ - interior_lo)
```

Zero anywhere in `[interior_lo, interior_hi]`; grows linearly (unbounded, no
saturation) as θ leaves that interior band in either direction — including
past the hard limit itself, so a sigma point that's already overshot `hi`
still produces a proportionally larger residual, not a flat one. Pseudo-
measurement target is 0, same convention as pose-regularization's rest-pose
residual: when the sigma-point-averaged residual is nonzero, the Kalman
update pulls θ back toward the interior, with pull strength set by a
configurable noise-std (smaller std = stiffer pull), evaluated exactly like
`pose_reg_rest_pose_noise_std`.

Computed directly from each sigma point's `joint_angles()` — no FK, no
camera projection, same as pose-regularization's residuals. Assembled into
its own small residual vector, own innovation covariance, own Kalman gain;
kept as an independent pass from `apply_pose_regularization()` (own function,
own config knobs, own on/off switch) rather than merged into the same
residual vector, so the two can be tuned and validated independently — same
reasoning that kept Mechanism A's scopes as separate, composable pieces
rather than one shared knob.

**Where in the pipeline**: called from `update()` alongside
`apply_pose_regularization()`, after the real camera-observation correction,
before `enforce_joint_limits()`'s hard clamp — so the hard clamp still runs
every step as the final backstop, but should engage far less often once this
pass is doing the earlier work.

**Config surface** (mirroring `pose_reg_*`):

- `UnscentedKalmanFilter::set_soft_joint_limits(joint_names, margin, noise_std)`
- Private state: `soft_limit_joint_indices_` (resolved per-axis from
  `joint_names` against `layout_->joints()`), `soft_limit_margin_`,
  `soft_limit_noise_var_`.
- `TrackerConfig`: `soft_limit_joint_names` (empty = disabled, same idiom as
  every other opt-in mechanism), `soft_limit_margin_rad`,
  `soft_limit_noise_std`.
- Start scoped to the joints that actually saturated in the traced case —
  `upper_arm.L`, `upper_arm.R` — extend to `shoulder.L/R`, `forearm.L/R`, or
  leg joints only if the same pattern is confirmed there (same "general
  mechanism, narrow initial configuration" principle pose-regularization
  used for `spine1`/`spine2`).

---

## Open questions

1. **Margin width.** No principled starting value. User's own estimate:
   start around 5-10°, tune empirically against the traced crisis window
   (steps 2505-3400 of the run above) the same way pose-regularization's
   stiffness was tuned — confirm `upper_arm.R` either doesn't clamp for the
   same real motion, or clamps later/for fewer consecutive frames, without
   fighting genuinely correct near-limit poses elsewhere in the trial.
2. **One shared margin/noise-std, or per-joint?** Start with one shared pair
   of knobs across all configured joints (simplest, matches pose-
   regularization's Phase 1 scoping). Mechanism A needed splitting into
   multiple scopes once one gain stopped being enough (see
   `adaptive-process-noise-design.md`) — this may follow the same arc if a
   single margin/stiffness can't serve joints with very different limit
   ranges well.
3. **Does the margin need to be axis-specific rather than joint-specific?**
   `upper_arm.R`'s three axes have quite different ranges (x: -30° to 160°,
   y: -45° to 45°, z: -20° to 150°) — a fixed absolute margin (e.g. 7.5°)
   applies asymmetrically relative to each axis's own range. Worth checking
   once implemented whether that matters in practice, or whether an
   absolute-radians margin is fine since it's the *approach to the wall*
   that matters, not the wall's position.
4. **Parent-joint redistribution** (noted in *Non-goals*): even with this
   mechanism damping the *overshoot*, the underlying real motion may still
   need more range than `upper_arm.R` alone can represent — real shoulders
   compensate via the clavicle. Whether to add that as a follow-on (a
   pseudo-measurement or explicit redistribution rule coupling
   `shoulder.R`'s available range to `upper_arm.R` nearing its own limit) is
   a separate, larger design question, not attempted here.
5. **Validation beyond the one traced case.** Same caveat as pose-
   regularization: need to confirm this doesn't regress calm segments or
   fight genuinely correct near-limit poses elsewhere in this trial or
   others, not just fix the one diagnosed window.
6. **Interaction with `damp_velocity_covariance_at_limits()`.** If this
   mechanism reduces how often the hard clamp fires, it should also reduce
   how often that function's velocity-covariance-only damping runs — worth
   checking whether the position-covariance inconsistency noted in
   *Motivation* becomes rare enough in practice that fixing it directly
   (a separate, smaller change) isn't also needed, or whether both are worth
   doing.

---

## Phasing

**Phase 1 — implement as described above**, scoped to `upper_arm.L` and
`upper_arm.R` only, margin ≈5-10° (Open question 1), starting with one
shared noise-std. *Validation*: re-run the traced full-trial config
(`e7bb4c0b-6473-4f6c-a0f9-acc699177ba0` lineage) over t=58-66s; confirm
`upper_arm.L/R`'s y-axis clamp frequency/duration drops for the same real
motion; confirm the mahalanobis/outlier crisis at steps 2520+ is smaller or
shorter; re-check the step-1631 and other previously-diagnosed windows for
regressions; spot-check a calm segment doesn't pick up spurious pull.

**Phase 2 (only if Phase 1 proves insufficient)** — investigate parent-joint
redistribution (Open question 4), or extend scope to more joints if the same
saturation pattern is confirmed elsewhere. Not worth building ahead of that
evidence, same gating principle as every other phase split so far.
