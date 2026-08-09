```toml
name = "Pose Regularization for Kinematically Redundant Chains"
status = "proposal"
description = """
A soft pseudo-measurement that discourages one joint in a redundant chain (e.g. spine1 vs. \
spine2 vs. root) from silently absorbing all rotation until it hits its own limit, in favor of \
distributing rotation sensibly across the chain — reducing both visually unnatural poses and a \
specific traced divergence where a saturated spine joint triggered a tracking crisis.
"""
categories = ["tracker-core", "ukf-tuning"]
target_release = "TBD"
last_updated = 2026-08-06
```

# Pose Regularization — Implementation Status

See [pose-regularization-design.md](pose-regularization-design.md) for the full motivation
(traced against a real diverging run), proposed mechanisms, and phasing.

## Current state

Sketch only, not implemented. Two alternative mechanisms are proposed:

- **Alternative 1 (preferred)**: a pseudo-measurement fused into the same weighted `update()`
  that fuses real camera observations — equal-split and rest-pose-pull residuals per configured
  redundant chain, computed directly from each sigma point's `joint_angles()` (no FK/camera
  projection needed).
- **Alternative 2 (fallback)**: a mean-reverting bias applied in `predict()`, same pattern as
  the existing velocity-damping half-life. Explicitly weaker (applies unconditionally regardless
  of how strongly real data resolves the ambiguity) — kept as a fallback if Alternative 1 proves
  insufficient.

Complementary to, not a replacement for,
[adaptive-process-noise](../adaptive-process-noise/status.md) — this targets the *trigger*
(why a joint saturated alone), that targets the *consequence* (keeping the outlier gate from
starving the filter once residual is already growing).

## Known issues / open questions

- Which chains beyond `spine1`/`spine2` need this — start narrow, extend only if the same
  pattern is confirmed elsewhere (e.g. `neck1`/`neck2`).
- Stiffness (noise-std) tuning has no principled starting value yet.
- Whether the equal-split target needs to be limit-aware, or whether letting
  `enforce_joint_limits()` remain the sole hard backstop is sufficient.
- Whether this also improves the ill-conditioning symptom from adaptive-process-noise's Case 2
  (a fast leg-swing) — speculated, not verified, since neither mechanism exists yet to test.
- Validation beyond the one diagnosed divergence case — needs example timestamps from other
  sequences where the "unnatural but non-divergent" pattern has been visually observed.
