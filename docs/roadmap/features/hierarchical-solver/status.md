```toml
name = "Hierarchical Body/Hand Solver"
status = "in_progress"
progress_pct = 90
description = """
Splits tracking into a two-pass batch solve — a body-only parent filter, then a per-hand child \
filter conditioned on the parent's smoothed trajectory — because finger markers destabilize \
whole-skeleton tracking (cross-covariance leakage makes arm tracking visibly jerkier, not just \
the fingers) and dominate the cross-person contact-pair budget.
"""
categories = ["tracker-core"]
target_release = "TBD"
last_updated = 2026-08-06
```

# Hierarchical Body/Hand Solver — Implementation Status

See [hierarchical-solver-design.md](hierarchical-solver-design.md) for the full design (builds
on two earlier docs, `docs/hierarchical-ukf-design.md` and `docs/hierarchical-tracker-redesign.md`,
which are outside this roadmap folder).

## Phase summary

| Phase | Description | Status |
|-------|-------------|--------|
| 3a-3e | Low-level plumbing: `SkeletonLayout` refactor, subtree Pinocchio model, fixed-root UKF sigma-point path | ✅ Done (predates this doc) |
| 3f | `ForwardKinematics::world_transform(joint_name)` | ✅ Done |
| PR 1 | `BatchTrajectoryStream`/`TrajectoryStream` — parent's smoothed output as a pull-based stream | ✅ Done |
| PRs 2-7 | Child-stage solver, DB persistence (`tracking_run_stages`, RMW patching), CLI/config plumbing, integration test | ✅ Done, integration-tested against real production data (2026-07-22) |
| PR 8 | Python/UI/MCP surfacing | ✅ Done, live-verified by Harri (2026-07-23) — every UI setting exercised individually |

## Current state

All 8 PRs done. A source-reading pass after PR 8's live verification found one real,
previously-undocumented bug (`cov_diag` merge using the wrong length), fixed the same day
(2026-07-23) alongside a new centralized `SkeletonLayout::build_error_index_map_from()`.

Architecture is a **two-pass batch** solve (parent runs full forward+smooth over a
`main`-only skeleton layout; each hand's child filter then runs its own full forward+smooth
using the parent's *smoothed* trajectory as a fixed per-frame root, via `PAIR_DIFF` relative
observations against the wrist marker) — not the originally-sketched per-frame interleaved
model, which needed sync machinery this design avoids entirely once fingers are absent from
the parent's state.

## Known issues

- **Not exercised**: an actual end-to-end hierarchical run launched from `run_tracker.py`'s own
  UI (as opposed to config-file/CLI-driven, which is covered by PR 7's integration test) —
  deliberately skipped as disproportionately slow to validate as UI-driven.
- `run_tracker.py`'s tracker-configuration dialog was flagged as too complex independent of
  this feature's own fields — tracked as a separate UI redesign
  (see [configuration-improvements](../configuration-improvements/status.md)), not a gap here.
- Wrist *joint angle* (as opposed to finger pixel tracking) inherits the parent's forearm
  orientation bias, since the relative measurement model doesn't carry a term for parent-root
  error — a known, accepted caveat, not a bug (see the design doc's "What the relative model
  does and doesn't absorb").
- No cross-person coupling at finger level (v1 trade-off, recorded as a decision) — gross
  hand-scale contact coupling still works via `MRK-wrist`/`MRK-index`/`MRK-pinky`, which stay
  in the parent's `main` layout.
