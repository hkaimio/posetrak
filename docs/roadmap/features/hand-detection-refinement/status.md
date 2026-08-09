+++
name = "Hand Detection Refinement"
status = "in_progress"
progress_pct = 85
description = """
Three related mechanisms to fix hand/finger tracking, a recurring weak point: skipping the \
outlier gate for trusted human keypoint edits (capped, not unconditional), a dedicated \
hand-region detection pass layered on top of the whole-body detector, and automated hand \
redetection triggered when a user edits a wrist/elbow during interactive editing.
"""
categories = ["detection-pipeline", "tracker-core"]
target_release = "TBD"
last_updated = 2026-08-06
+++

# Hand Detection Refinement — Implementation Status

See [hand-detection-refinement-design.md](hand-detection-refinement-design.md) for the full
history (9 dated status updates) and design. Also see
`docs/roadmap/features/tracking-crisis-debugging-log.md` for the "Phase 0"/"Phase 0b" writeups
this doc's Idea 1 references, and `docs/roadmap/first-release-backlog.md` item 3 for the
remaining validation-at-scale gap.

## Idea summary

| Idea | Description | Status |
|------|-------------|--------|
| 1 | Skip/scale the outlier gate for trusted human keypoint edits | ✅ Implemented as "scale noise to land just inside gate threshold" (exact bisection solve, not the original hard bypass — that variant tested net-negative and was superseded). Off by default. |
| 2 | Hand-specific detection pass (`rtmlib.Hand`) after the whole-body pass, multi-row `pose_observations` schema (`source` column) | ✅ Implemented, validated end-to-end against real session data |
| 3 | Automated hand redetection triggered by editing wrist/elbow, `.refined` source-precedence convention, "Auto-detect" vs "keep existing state" UI toggle | ✅ Implemented and validated end-to-end 2026-07-15, after fixing two real bugs found via live testing (a frame-position desync on inter-frame-coded video, and a worker thread that silently died on its first request due to a missing `row_factory`) |

## Current state

All three ideas are implemented. Idea 1's hard-bypass variant (Phase 0) tested net-negative on
real data — a legitimate edit forced a 48σ correction through in one step, badly
ill-conditioning the covariance — and was superseded by a noise-scaling variant that caps how
far any single edited observation can pull the state, tested successful on two of three people
(Roosa: complete success; Tommi: a second apparent failure traced to a genuine data error, not
the mechanism, and confirmed resolved on rerun). Idea 2's crop/candidate-selection/validation-
gate formulas are empirically tuned across four rounds of offline stills. Idea 3's integration
(trigger debouncing, provenance/precedence, interpolation interaction, UI status color) was
finalized 2026-07-14 and validated live 2026-07-15.

## Known issues

- **Idea 1 still off by default** — not yet tested on the third person (Timo) or against the
  full trial-wide adaptive-process-noise on/off comparison this whole investigation arc started
  from.
- **Two-handed coordinated grip edge case** (Idea 2 and Idea 3 both inherit this): the
  proximity validation gate can reject the *correct* hand when both hands are close together
  (a sword grip, clasped hands), because it checks distance to a single wrist and the other
  hand's grip point can be closer. Not designed further; a rejection degrades gracefully
  (writes nothing) rather than writing a wrong value.
- **Idea 3's 700ms debounce window is untuned** — a guess, not validated against extended real
  use (`first-release-backlog.md` item 3).
- **Tracking-quality impact not measured at scale** — Idea 3 is confirmed working (no crashes,
  sensible writes) but its effect on tracking quality across a full trial, and a hand-editing
  completion-time comparison, are still open per the design doc's own validation criteria.
- **Idea 1(b), surfacing rejected edits**: proposed (MCP diagnostic tool sibling to
  `get_edit_coverage`, and/or a UI color distinguishing "edit was rejected by the gate") but
  not built.
