```toml
name = "Soft Joint-Limit Repulsion"
status = "proposal"
description = """
A pseudo-measurement that discourages a joint from approaching its own hard rotation limit \
before it gets there, rather than only reacting once the hard clamp fires after the Kalman \
update has already overshot — targeting a traced crisis where adaptive process noise made a \
near-limit overshoot worse, not better, for a fast bilateral arm-raise.
"""
categories = ["tracker-core", "ukf-tuning"]
target_release = "TBD"
last_updated = 2026-08-06
```

# Soft Joint-Limit Repulsion — Implementation Status

See [soft-joint-limits-design.md](soft-joint-limits-design.md) for the full traced motivating
case, mechanism, and phasing. Reuses the pseudo-measurement pattern established by
[pose-regularization](../pose-regularization/status.md) — read that design first.

## Current state

Sketch only, not implemented. Proposed as a second, independent Kalman-gain pass (own residual
vector, own noise-std, own on/off switch) using the same sigma-point/cross-covariance machinery
as pose-regularization: zero residual inside a configured margin from each joint's hard limit,
growing linearly (unbounded) outside it. Scoped to start at exactly the two joints
(`upper_arm.L`/`upper_arm.R`) implicated in the traced crisis.

Motivating trace found that Mechanism A of
[adaptive-process-noise](../adaptive-process-noise/status.md) — built to help exactly this kind
of fast-motion case — plausibly made this specific failure *worse*: more slack for a joint
accelerating toward a hard wall widens the sigma cloud right when a narrower one would behave
better, explaining why no variant of adaptive-process-noise tuning touched this crisis at all.

## Known issues / open questions

- Margin width has no principled starting value — user's own estimate is 5-10°, to be tuned
  empirically against the traced crisis window.
- Whether one shared margin/noise-std across all configured joints is sufficient, or whether it
  needs to follow adaptive-process-noise's arc of splitting into per-joint/per-scope knobs once
  one gain stops being enough for joints with very different limit ranges.
- Whether the margin needs to be axis-specific (a joint's three axes can have quite different
  ranges) rather than one shared absolute-radians value.
- Parent-joint redistribution (shifting "leftover" rotation to the clavicle when the shoulder
  maxes out) is explicitly out of scope here — flagged as a separate, larger follow-on design
  question, not attempted in this proposal.
- Whether reducing hard-clamp frequency also makes `damp_velocity_covariance_at_limits()`'s
  known position/velocity-covariance inconsistency rare enough not to need its own direct fix.
