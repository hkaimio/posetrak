```toml
name = "Adaptive Process Noise"
status = "in_progress"
progress_pct = 40
description = """
Velocity-driven per-DOF process noise (Mechanism A) and a regional NIS-feedback fading safety \
net (Mechanism B), so the UKF's process noise can widen for fast motion (a throw, a deep bend) \
instead of using one static value that's always a compromise between calm-motion precision and \
fast-motion tracking.
"""
categories = ["tracker-core", "ukf-tuning"]
target_release = "TBD"
last_updated = 2026-08-06
```

# Adaptive Process Noise — Implementation Status

See [adaptive-process-noise-design.md](adaptive-process-noise-design.md) for the full
motivation, literature background, and design.

## Current state

**Mechanism A (velocity-driven per-DOF process noise)** is implemented
(`rebuild_process_noise()` in `src/filters/ukf.cpp`) and scoped by joint-name list
(`process_noise_vel_joint_names`). Validated in isolation on the original motivating case
(Case 1, a forward bend) — inlier count and covariance conditioning improved monotonically as
gain increased, with zero regressions against the pre-change test suite baseline.

**Mechanism B (regional NIS-feedback fading)** is design-only — the whole-skeleton version
originally sketched was revised to a per-scope (joint-name-list) version after Case 3 showed
the whole-skeleton aggregate would have missed a real, localized wrist rejection. Not yet
implemented.

**Status under review, not "done."** A later full-trial A/B comparison (both mechanisms on vs.
off, three people, otherwise identical config) found **on is worse than off for every person,
on every conditioning metric** — average NIS/DOF, % overconfident steps, average and worst-case
covariance condition number all regressed, most severely on the least-manually-cleaned data.
It is not yet isolated whether Mechanism A, Mechanism B (not even built at whole-trial scale),
or an interaction between the two mechanisms and the rest of the tuning stack is responsible.
Full writeup: `docs/roadmap/features/tracking-crisis-debugging-log.md`, sections "Adaptive
process noise (Mechanisms A+B) on/off comparison" and "Mechanism inventory — keep/kill initial
lean."

No decision has been made yet on whether to revert, retune, or keep either mechanism.

## Known issues / open questions

- Whether Mechanism A's per-DOF velocity gain is itself responsible for the regression, or
  whether it's downstream interaction with soft-joint-limits / pose-regularization (both built
  later, in response to failures Mechanism A didn't fix) — not isolated.
- Mechanism B's design was revised once (whole-skeleton → per-scope) but has never been coded,
  so the on/off comparison above only exercises Mechanism A in isolation despite testing "both."
- See the design doc's *Open questions* section (7 items) for tuning/scoping questions that
  were live even before the regression finding — root vs. joint gain, linear vs. saturating
  velocity mapping, interaction with `vel_half_life_s_`, and others.
