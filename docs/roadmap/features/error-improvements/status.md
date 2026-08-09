+++
name = "Measurement Error Model Improvements"
status = "in_progress"
progress_pct = 85
description = """
Replaces the tracker's single flat pixel-noise parameter with a model that accounts for crop \
scale and calibration error separately, adds relative (parent-child and spatially-close) \
keypoint measurements to cancel shared calibration bias, and — the highest-impact piece — \
cross-person relative observations for contact/interaction (grabs, throws, handshakes).
"""
categories = ["tracker-core", "measurement-model"]
target_release = "TBD"
last_updated = 2026-08-06
+++

# Measurement Error Model Improvements — Implementation Status

See:
- [error-improvement-brief.md](error-improvement-brief.md) — original problem statement (Ideas 1-2)
- [design-crop-scale-noise.md](design-crop-scale-noise.md), [design-calibration-error.md](design-calibration-error.md) — detailed design for the split noise model and calibration-error approaches
- [implementation-plan.md](implementation-plan.md) — phase-by-phase implementation plan (Phases 1-5)
- [phase5-cross-person-plan.md](phase5-cross-person-plan.md) — Phase 5's own implementation status (Stages 1-4)
- [phase5-cross-person-diagnostics-plan.md](phase5-cross-person-diagnostics-plan.md) — Phase 5 follow-on, diagnostics/observability

## Phase summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Split `measurement_noise_std` into `pose_noise_std` (`ep`) + `calib_noise_std` (`ec`), crop-scale-aware | ✅ Done |
| 2 | Robust (Huber) measurement likelihood | ⬜ Not implemented (no `huber_k` in codebase) |
| 3 | Relative keypoint measurements — parent-child pairs (`PAIR_DIFF`) | ✅ Done |
| 4 | Relative keypoint measurements — spatially-close pairs | ✅ Done |
| 5 | Cross-person relative observations (`MultiPersonTracker`, contact gating) | ✅ Done — all 4 stages, see below |
| 5 follow-on | Cross-person observation diagnostics (which keypoint anchored to what, surfaced in output) | ⬜ Design sketch only |

## Phase 5 detail (the priority item — see `first-release-backlog.md`)

Built in C++, in-process (a `MultiPersonTracker` orchestrator), not the Python-subprocess
orchestrator the original sketch assumed — corrected early after confirming there's no
Python↔C++ binding in this codebase. All four stages done:

1. Orchestrator harness (no coupling yet) — bitwise-identical to single-person runs.
2. Contact gating + anchor injection — three-level gate, rotating processing order, anchor-
   freshness extrapolation.
3. Per-marker anchor uncertainty via Jacobian (`Tracker::marker_projection_std()`), verified
   against a sigma-point-reprojection oracle, a finite-difference check, and a hand-computed case.
4. Config DB wiring (`cross_person_max_world_mm`/`cross_person_min_confidence`/
   `cross_person_max_n`), RTS smoothing (turned out already wired via a shared helper), and
   Python/UI/MCP surfacing — `run_tracker.py` gained a Trial → Person(s) picker driving
   `run_multi_person_tracker()`.

Contact-window summary UI (a data surface showing which frames/markers were cross-person
anchored) remains explicitly deferred — see the diagnostics follow-on plan.

## Known issues

- **Phase 2 (Huber robust likelihood) was never implemented** — the implementation plan scoped
  it as a quick, independent win, but no `huber_k` config field or code exists in the tree.
  Not blocking; not currently prioritized.
- **Cross-person observation diagnostics** (Phase 5 follow-on): nothing downstream records
  which marker was anchored to which other person's marker, in which camera/frame — the
  `Observation::anchor_position` value is transient and discarded after the UKF update. Design
  sketch only; motivated by wanting to distinguish "the cross-person coupling algorithm caused
  this jitter" from "there's just more occlusion when people are close."
- **A related pre-existing bug found while investigating the diagnostics gap**: direct
  detections and `PAIR_DIFF` observations for the same `(marker, camera, frame)` already collide
  in today's output (not specific to cross-person anchors — affects the existing within-person
  `PAIR_DIFF` observations from Phases 3/4 too). See `phase5-cross-person-diagnostics-plan.md`
  for the trace.
