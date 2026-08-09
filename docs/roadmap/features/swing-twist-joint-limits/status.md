+++
name = "Swing-Twist Joint Orientation & Limits"
status = "proposal"
description = """
Replaces spherical joints' raw rotation-vector storage (which can spuriously approach the \
representation's own topological boundary near 180° even when every individual axis stays \
within its configured limit) with a swing-twist decomposition, so combined per-axis limits \
can't inadvertently produce an unreasonable total rotation.
"""
categories = ["tracker-core", "ukf-tuning"]
target_release = "TBD"
last_updated = 2026-08-06
+++

# Swing-Twist Joint Limits — Implementation Status

See [swing-twist-joint-limits-design.md](swing-twist-joint-limits-design.md) for the full
diagnosis (unifying two previously-separate-looking findings: the "box corner" effect and the
frame-227/228 discontinuity) and proposed representation.

## Current state

Design sketch, not implemented — flagged as likely to need revision once a still-open
Karcher-mean-convergence instrumentation task (tracked in
`docs/roadmap/features/tracking-crisis-debugging-log.md`) reports results. This is a materially
bigger lift than the other tracking-crisis mechanisms
([pose-regularization](../pose-regularization/status.md),
[soft-joint-limits](../soft-joint-limits/status.md)) — it touches forward kinematics, sigma-point
generation/mean/error computation, the skeleton YAML file format, and inverse kinematics, not
just a new pseudo-measurement or noise-scaling term.

## Known issues / open questions

- Bone-axis convention per joint needs to be made explicit and validated (presumably derivable
  from each joint's existing rest-pose data, not yet confirmed).
- Circular vs. elliptical swing cap — a real shoulder's range isn't a circular cone; a circular
  cap is the simpler first cut.
- Migrating existing per-axis box limits to swing/twist has no clean mapping — either an
  approximate/conservative conversion from the box geometry, or a full recalibration from
  scratch (more correct, bigger undertaking, and current shoulder limits may already be
  miscalibrated independent of this representation question).
- Whether this is a new `JointType` or a per-joint flag on the existing `SPHERICAL` type —
  affects how much existing switch/if logic needs touching either way.
- CSV export (`joint_angles.csv`) semantics change for opted-in joints (`swing_x, swing_y,
  twist` instead of raw `angle_x/angle_y/angle_z`) — needs a documented divergence or a
  column-semantics flag, since existing analysis scripts assume raw-axis-angle semantics.
- Whether this fully obsoletes near-limit damping (very likely, for opted-in joints) and how it
  relates to parent-joint redistribution (orthogonal, still separately useful) — not confirmed.
