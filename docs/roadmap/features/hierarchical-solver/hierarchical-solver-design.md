# Hierarchical body/hand solver — design proposal

## Status

Design proposal, not yet implemented. Builds directly on two existing design
docs already in the repo, which this doc does not duplicate:

- `docs/hierarchical-ukf-design.md` — root-cause analysis (why fingers
  destabilize the body chain) and a survey of solution options. Written
  earlier, independently of this round's investigation.
- `docs/hierarchical-tracker-redesign.md` — concrete engineering design for
  a hierarchical parent/child filter architecture, with phases **3a through
  3f** already scoped, and **3a–3e already implemented and tested**:

  | Phase | Status | What it did |
  |-------|--------|-------------|
  | 3a | done | Removed `Skeleton` arg from `UKF`/`ProcessModel`/`SigmaPointGenerator`; read from `SkeletonLayout` |
  | 3b | done | `SkeletonLayout` factories take `shared_ptr<const Skeleton>` |
  | 3c | done | Removed `set_active_groups()` from `Skeleton`; groups passed to layout factory |
  | 3d | done | `PinocchioModelBuilder::build_subtree_model()` + unified `ForwardKinematics` API (single `shared_ptr<const SkeletonLayout>` constructor, layout-aware `state_to_config()`), 9 passing `[subtree_model]` tests |
  | 3e | done | `UnscentedKalmanFilter::set_root_transform()` + fixed-root sigma-point path (root excluded from error state, held constant through predict/update), 3 passing `[child_filter]` tests |
  | 3f | **done — corrected 2026-07-20** | `ForwardKinematics::world_transform(joint_name)` was already implemented (`joint_id_map_` populated from `model_.names`, reads `data_.oMi[]`) and tested (`[world_transform]`, 18 assertions), contrary to this doc's earlier "next, not yet done" — verified directly against the code before starting implementation. |

  This means the low-level plumbing a hand child-filter needs — a
  subtree Pinocchio model rooted at an externally-supplied joint, a UKF that
  holds that root fixed through predict/update, sigma points that never
  perturb it — **already exists and is tested**, independent of anything
  decided in this doc.

  **Implementation status (2026-07-20)**: PR 1 done —
  `BatchTrajectoryStream`/`TrajectoryStream` (`include/posetrak/tracking/trajectory_stream.hpp`)
  wraps a completed `Tracker`'s smoothed output + its own `get_fk()` to yield
  one named joint's world transform per frame, pull-based (`next()` →
  `std::optional<FreeflyerPose>`), ready for a future fixed-lag producer to
  replace the batch implementation without changing consumers. 16 passing
  `[trajectory_stream]` assertions. Full test suite otherwise green; one
  pre-existing, unrelated failure (`test_statistics_tracker.cpp`'s CSV/JSON
  cleanup hits a Windows file-handle-release race) confirmed via `git status`
  to predate this work and left alone as out of scope.

This doc's job is narrower: (1) independently re-validate the root-cause
diagnosis against fresh production data, since it happened to get re-derived
from scratch this session before either old doc was consulted; (2) pin down
**exact** `main`/`HandL`/`HandR` group definitions against the actual
production skeleton (the old docs used an illustrative fixture skeleton,
not real data); and (3) propose a specific architecture for the phases after
3f — a **two-pass batch model**, not the per-frame interleaved model
`hierarchical-tracker-redesign.md` originally sketched — motivated by a
wrist-position-error argument that came out of this session's discussion
and isn't in either old doc.

**Revised 2026-07-20** after a schema-grounded design review (checked
against `db/session_schema.sql`, `db/registry_schema.sql`,
`src/db/result_writer.cpp`, and the implemented cross-person
`MultiPersonTracker`): the persistence design, checkpoint/restore
interaction, and multi-person composition sections below were reworked,
and phases 3h–3l updated. Inline "(review, 2026-07-20)" markers show
what changed.

## Independent re-validation of the root cause

`hierarchical-ukf-design.md` already diagnosed the mechanism precisely: a
chain-end marker's predicted measurement covariance `S = H·P·Hᵀ + R`
inherits uncertainty from every ancestor joint, amplified by lever-arm
length; a global, uniform Mahalanobis threshold is therefore structurally
more permissive for finger markers than for body markers regardless of
detector quality; and once a bad finger observation is accepted, the Kalman
gain's cross-covariance couples the correction back into the arm chain, not
just the fingers. That doc even reports the same validating experiment run
this session: *"Disabled most palm/finger markers, kept only minimal set for
wrist orientation. Result: Divergence significantly reduced."*

This session reran that ablation independently, on different production
data (three people, `optbuild` release build, real multi-camera capture),
without having read either old doc first, and got the same qualitative and
quantitative result:

- Wall time: fingers-on ~2100s → fingers-off 905.5s (state-dimension
  reduction from converting orphaned finger joints to `type: fixed`, not
  just fewer observations).
- Outlier rate, NIS mean, and covariance condition number all dropped
  substantially with fingers removed (see this session's earlier
  comparison — NIS mean roughly halved, condition number improved by up to
  ~11x for the worst-conditioned person).
- Visual inspection of exported BVH confirmed it qualitatively: fingers-on
  tracking was visibly jerkier in the arms, not just the fingers, matching
  `hierarchical-ukf-design.md`'s cross-covariance-leakage explanation
  exactly.

Treat this as a second, independent confirmation of an already-correct
diagnosis, not a new finding — the fix path was already scoped in
`hierarchical-tracker-redesign.md` before this session started; this
session's contribution is (a) confirming it still holds today, on the
production data and skeleton rig currently in use, and (b) the two design
changes below.

One additional, session-specific data point worth carrying forward: this
session also found that finger markers destabilize the **cross-person**
coupling feature in the same way — with ~60 markers per person including
many fingers clustered close together, finger-pair candidates dominate
`cross_person_max_n`'s per-camera cap, crowding out potentially more
meaningful non-finger contact pairs. Both problems point at the same root
cause and the same fix (get fingers out of the shared/monolithic DOF and
observation set).

## Exact group definitions

Verified directly against the actual production skeleton content (Timo /
Tommi / Roosa, the `reallusion-no-waist` template — pulled from the session
DB this session, not the illustrative fixture used in the older design
docs). All three performers share identical topology; only bone-length
`offset:` values differ.

**Important correction to the skeleton's current `groups:` section**: the
`main`, `HandL`, and `HandR` groups already exist in the production
skeletons and are *almost* exactly what's needed below — but they reference
`palm.01.L`/`palm.02.L`/`palm.03.L`/`palm.04.L` (and `.R`) joints that do
not exist in the current joint tree. This skeleton's fingers attach
directly to `hand.L`/`hand.R` (confirmed by reading the actual joint
`parent:` fields, and by the fact that `MRK-index.L` and `MRK-pinky.L`
attach to `f_index.01.L`/`f_pinky.01.L`, whose own parent is `hand.L`).
Today this is harmless — `skeleton_loader.cpp`'s group parsing just
builds a name→group lookup map and silently no-ops on unmatched joint
names — but once `SkeletonLayout::from_groups()` is used to build a real
child-filter subset (Phase 3d's machinery), a stale reference would either
be silently dropped (if `from_groups()` also just filters against joints
that exist) or become a real bug. **This should be corrected before
Phase 3h wiring** — replace the `palm.*` entries below with nothing (they
have no replacement).

Two cheap guards make this class of error visible instead of silent
(review, 2026-07-20): (1) `skeleton_loader.cpp` should warn — or error —
on group entries referencing nonexistent joints/markers, replacing
today's silent no-op (which is exactly how the phantom `palm.*` entries
survived); (2) a unit test asserting the production skeletons' `groups:`
sections are structurally identical, since group definitions become
load-bearing with this feature but stay duplicated per performer file
until the skeleton class/scaling split lands.

**Revision from Harri's review**: `hand.L`/`hand.R` themselves are *not*
the freeflyer boundary — see "wrist ownership" below. The freeflyer is one
level further up, `forearm.L`/`forearm.R`, matching Phase 3d/3e's
"freeflyer is the skeleton parent of the shallowest group joint" convention
against the *corrected* group membership (`hand.L` is now inside `HandL`,
not excluded from it). `forearm.L`'s joint type is also non-`FIXED`,
matching §11.2 of the redesign doc's note that real freeflyer boundary
joints always are.

```yaml
groups:
- name: "main"
  joints:
  - hips
  - waist
  - spine1
  - spine2
  - neck1
  - neck2
  - head
  - shoulder.L
  - upper_arm.L
  - forearm.L
  - hand.L
  - shoulder.R
  - upper_arm.R
  - forearm.R
  - hand.R
  - thigh.L
  - shin.L
  - foot.L
  - toe.L
  - thigh.R
  - shin.R
  - foot.R
  - toe.R
  markers:
  - MRK-nose
  - MRK-ear.R
  - MRK-ear.L
  - MRK-wrist.L
  - MRK-elbow.L
  - MRK-shoulder.L
  - MRK-wrist.R
  - MRK-elbow.R
  - MRK-shoulder.R
  - MRK-hip.L
  - MRK-hip.R
  - MRK-knee.L
  - MRK-knee.R
  - MRK-Ankle.L
  - MRK-Ankle.R
  - MRK-heel.L
  - MRK-heel.R
  - MRK-bigToe.L
  - MRK-smallToe.L
  - MRK-bigToe.R
  - MRK-smallToe.R
  - MRK-index.L
  - MRK-pinky.L
  - MRK-index.R
  - MRK-pinky.R
  optional: false

- name: "HandL"
  depends_on: "main"
  joints:
  - hand.L           # NOW included — see "wrist ownership" below (Harri's correction)
  - thumb.01.L
  - thumb.02.L
  - thumb.03.L
  - f_index.01.L
  - f_index.02.L
  - f_index.03.L
  - f_middle.01.L
  - f_middle.02.L
  - f_middle.03.L
  - f_ring.01.L
  - f_ring.02.L
  - f_ring.03.L
  - f_pinky.01.L
  - f_pinky.02.L
  - f_pinky.03.L
  markers:
  - MRK-wrist.L      # shared with "main"; the child's own PAIR_DIFF reference marker
  - MRK-index.L      # shared with "main"
  - MRK-index2.L
  - MRK-index3.L
  - MRK-index4.L
  - MRK-thumb.L
  - MRK-thumb1.L
  - MRK-thumb3.L
  - MRK-thumb4.L
  - MRK-middle1.L
  - MRK-middle2.L
  - MRK-middle3.L
  - MRK-middle4.L
  - MRK-ring1.L
  - MRK-ring2.L
  - MRK-ring3.L
  - MRK-ring4.L
  - MRK-pinky.L      # shared with "main"
  - MRK-pinky2.L
  - MRK-pinky3.L
  - MRK-pinky4.L

- name: "HandR"
  depends_on: "main"
  joints:
  - hand.R
  - thumb.01.R
  - thumb.02.R
  - thumb.03.R
  - f_index.01.R
  - f_index.02.R
  - f_index.03.R
  - f_middle.01.R
  - f_middle.02.R
  - f_middle.03.R
  - f_ring.01.R
  - f_ring.02.R
  - f_ring.03.R
  - f_pinky.01.R
  - f_pinky.02.R
  - f_pinky.03.R
  markers:
  - MRK-wrist.R
  - MRK-index.R
  - MRK-index2.R
  - MRK-index3.R
  - MRK-index4.R
  - MRK-thumb.R
  - MRK-thumb1.R
  - MRK-thumb3.R
  - MRK-thumb4.R
  - MRK-middle1.R
  - MRK-middle2.R
  - MRK-middle3.R
  - MRK-middle4.R
  - MRK-ring1.R
  - MRK-ring2.R
  - MRK-ring3.R
  - MRK-ring4.R
  - MRK-pinky.R
  - MRK-pinky2.R
  - MRK-pinky3.R
  - MRK-pinky4.R
```

`hand.L`/`hand.R` is now the first *estimated* joint in `HandL`/`HandR`
(a normal spherical joint in the child's own layout, not the freeflyer) —
the freeflyer moved one level up, to `forearm.L`/`forearm.R`, which stays
`main`-only, fixed from the smoothed parent per frame. See "wrist
ownership" below for why.

Notes:

- `MRK-index.{L,R}` and `MRK-pinky.{L,R}` deliberately appear in **both**
  `main` and `Hand{L,R}` — this is already how the existing (if stale)
  group scheme works, and `hierarchical-tracker-redesign.md` §2.2 already
  covers it: "A palm marker appearing in both parent and child observation
  groups is fine. Each FK instance is independent... contributes to two
  separate UKF updates." In the parent pass they're two of `main`'s ~25
  markers constraining the wrist. In the child pass they're additionally
  the **anchor reference markers** for the relative-observation scheme
  below.
- No `thumb2.{L,R}` marker exists in this skeleton (thumb has 3 markers —
  base, `1`, `3`, `4` — not 4 like the other fingers). Preserved exactly as
  found; not a bug to fix.
- Joint counts: `HandL`/`HandR` now each carry 16 joints (`hand.{L,R}`
  plus the 15 finger phalanges), not a clean match to the earlier
  no-fingers ablation's 26-fixed/2-free split — that experiment kept
  `f_index.01`/`f_pinky.01` free in the *parent* and fixed everything else;
  here, `hand.{L,R}` and all 15 descendants move to the child instead. The
  ablation's stability result (removing fingers from the parent stabilizes
  it) is still the motivating evidence; the exact DOF partition is
  different because the parent no longer needs `f_index.01`/`f_pinky.01`
  at all — `MRK-index`/`MRK-pinky` still constrain the parent's wrist
  *orientation* indirectly through `hand.{L,R}`'s parent-side estimate,
  same as the ablation, but the parent doesn't need to own the phalanx
  joints themselves to benefit from those markers.

## Architecture: two-pass batch, not per-frame interleaved

`hierarchical-tracker-redesign.md` §4.1 designs a per-frame **interleaved**
sequence: every frame, the parent predicts+updates, then each child reads
the parent's *live, current-frame* posterior root transform
(`parent_fk_->world_transform(...)`) and does its own predict+update. This
needs `enable_sync`/`sync_covariance` machinery to feed corrected child
DOFs back into the parent for temporal consistency (§4.1 steps 4–5, and
open questions Q3/Q4 about sync direction and failure handling).

This doc proposes a **two-pass batch** sequence instead:

```
Stage A (parent): run today's existing Tracker, unmodified, against the
                   "main"-only SkeletonLayout (zero finger DOFs, zero
                   finger observations) — full forward pass + RTS
                   smoothing, exactly like this session's no-fingers
                   ablation runs. No new code.

Stage B (child, per hand, per person): a new solver that runs its own
                   full forward pass + RTS smoothing over the SAME frame
                   range, using the parent's *smoothed* (not live) output
                   as a fixed, precomputed per-frame anchor.
```

### Why batch instead of interleaved

The motivating reason, from this conversation: **wrist position error from
the body solve is significant relative to hand/finger scale**, and a live
per-frame parent posterior still carries that error at the moment the child
would consume it. Waiting for the parent's *smoothed* trajectory doesn't
eliminate that error, but it minimizes it (smoothing is strictly at least
as accurate as filtering) before the hand solver depends on it, and —
combined with the relative-observation scheme below — the child filter
never needs to trust the wrist's *absolute* world position at all, only its
*projected pixel location*, which is what actually matters for finger
tracking accuracy.

This also removes the interleaved design's motivation for the sync
machinery in the first place. That machinery existed because the
interleaved design still had genuine bidirectional information value — the
child sees dense observations that could inform the parent's shared DOFs
better than the parent's own sparse observations of them. But once fingers
are **entirely absent from the parent's state and observation set** (as
they are here — `main` has zero finger DOFs, not just externally-fixed
finger DOFs synced from a shared state), the parent no longer has anything
to gain from the child's opinion: `main`'s own markers (including
`MRK-index`/`MRK-pinky`) are already sufficient to constrain the wrist well,
which this session's ablation runs already demonstrate (the no-fingers
parent-only runs were the *most* stable of everything tried, not just
"acceptable"). Concretely, this means:

- No `enable_sync`, `sync_covariance`, or live merge step. `HierarchicalTracker`'s
  open questions Q3 ("child predict without parent update") and Q4
  ("covariance sync direction") become moot — there's no interleaving for
  them to apply to.
- No `HierarchicalTracker` as a generalized, potentially-recursive N-level
  coordinator (redesign.md §7, §8 Q6) — see "Scope decision" below.
- Sequential full RTS per filter is sufficient, resolving the earlier
  sliding-window-smoothing question from this conversation: parent runs
  and fully smooths once, then each hand runs and fully smooths once. No
  fixed-lag/sliding-window smoother is required for this feature. (Fixed-lag
  smoothing remains valuable on its own for other reasons — the
  cross-person phase's memory scaling with person count, and interactive
  re-solving after edits — but isn't a prerequisite here.)

### Parent-trajectory input is a stream, not a materialized array

One API decision made now because it is cheap now and a solver rewrite
later (review, 2026-07-20): the Stage B solver consumes the parent's
smoothed freeflyer transforms as an **incremental stream**
(iterator/callback supplying frame `t`'s transform), not as a fully
materialized whole-sequence array. Under today's full-batch RTS the
driver simply streams the completed trajectory, so nothing about the
two-pass model changes. But if/when the tracker migrates to a fixed-lag
smoother (planned independently, for cross-person memory scaling and
edit-driven re-solving), the child becomes a pipeline stage running
`lag` frames behind the parent with zero solver changes — only the
driver changes. Baking "takes the whole trajectory up front" into
3f2/3h would turn that migration into a rewrite.

### Scope decision, revised: narrow *and* generic — not narrow *or* general

The first draft framed this as a binary: a purpose-built, hand-specific
pipeline (small, fast to ship) versus `hierarchical-tracker-redesign.md`'s
fully general, recursive `HierarchicalTracker` coordinator (§7, §8 Q6).
Harri flagged the real problem with the "narrow" side of that binary as
drafted: the phase plan named a `HandSolver` C++ class, implied
hand-specific database output structure, and implied Python export/merge
code that specifically understands "main + hand" — i.e. it hardcoded one
particular skeleton topology's structure into three separate layers of the
codebase. That's exactly the failure mode the skeleton-classes discussion
earlier this session was about: limb/topology structure baked into
application code instead of driven by the skeleton's own definition. This
design would have reproduced that problem in a new place while that broader
architectural fix is deliberately postponed.

The fix isn't "build the general recursive coordinator after all" — that
was correctly avoided above for its own reasons (no second use case yet,
and its sync/merge machinery solves a live-interleaving problem this batch
design doesn't have). The fix is to keep the same small, non-recursive,
two-stage *shape*, but stop naming and parameterizing it after "hand"
specifically:

- **C++**: no `HandSolver` class. The Phase 3d/3e subtree-filter machinery
  (subtree Pinocchio model + fixed-root UKF) is already generic — it takes
  a skeleton, a freeflyer joint name, and a group name list as data, not as
  template/class parameters. The Stage B solver is parameterized by
  `(freeflyer_joint_name, joint_groups, observation_groups,
  ref_marker_name, stage_config)` and gets instantiated twice, with
  `{"HandL"}`/`{"HandR"}` and `MRK-wrist.{L,R}`, for this feature —
  nothing in it knows what a hand is. Whether it is a new
  `ChildFilterSolver` class or `Tracker` taught a fixed-root mode is
  decided in 3h (review, 2026-07-20: prefer the latter — see the phase
  table for the duplication-drift argument).
- **Database — revised a third time (review, 2026-07-20)**: still no new
  run identity — but the previous revision's "genuinely zero schema
  change" claim does not survive contact with the actual write path
  (`src/db/result_writer.cpp`); one small bookkeeping table is warranted
  and three write-side conventions must be pinned down. The core shape
  is unchanged: `tracking_results` stores one row per
  `(run_id, person_id, tracker_step, is_smoothed)` with `state`/`cov_diag`
  as opaque full-width blobs, and `tracking_obs_results` stores
  `obs_blob` indexed by camera and marker across the *whole* skeleton's
  marker list (`tracking_runs.marker_names`) — already "one wide row per
  step," so a multi-stage pipeline needs no new run rows. Keep exactly
  one `tracking_runs`/`tracking_run_persons` row per person, referencing
  the performer's *full*, unmodified skeleton — the same one used today.
  Each solver stage internally uses its own smaller `SkeletonLayout`
  (`from_groups({"main"})`, `from_groups({"HandL"})`, ...), and at the
  DB-write boundary expands its compact state into the full skeleton's
  index range via `SkeletonLayout::build_index_map_from()` — the same
  merge-map mechanism `hierarchical-tracker-redesign.md` §5 designed for
  an in-memory merge. The parent writes first; each child then
  read-modifies-writes the same row, patching only the index range it
  owns.

  What the review changed or pinned down:

  1. **One new table: `tracking_run_stages`** —
     `(run_id, person_id, group_name, status, started_at, completed_at)`.
     Plain read-modify-write rows leave no record of which stages have
     run: a crash mid-child leaves rows silently half-patched,
     indistinguishable from complete output, and nothing can answer "has
     Stage B run?" or "is it stale?". The table gives an atomic
     stage-completion boundary (mark `complete` only after the stage's
     whole RMW pass commits), the staleness flag the checkpoint/edit
     workflow needs (see the checkpoint section below), and a progress
     surface for the UI. This deliberately trades away "zero schema
     change": one small table buys crash-safety, provenance, and
     invalidation tracking.

  2. **Placeholder and `is_smoothed` semantics, made explicit.** The
     parent writes rest-pose values with init-level `cov_diag` into
     child-owned DOF ranges; readers distinguish "not yet solved" via
     `tracking_run_stages`, never by sniffing blob values. The child
     patches **both** row families — its filtered states into
     `is_smoothed=0` rows, its smoothed states into `is_smoothed=1`
     rows — accepting that hand DOFs in the "filtered" rows are
     conditioned on the parent's *smoothed* trajectory (they are the
     best per-frame-causal hand estimate available; a pure-filtered
     hand estimate does not exist in this design). Merged `cov_diag` is
     similarly mixed: child-DOF variances are conditional on a fixed
     parent root and exclude parent uncertainty, so they are not
     comparable to body-DOF variances — anything displaying confidence
     must know this (surfacing phase 3l).

  3. **`tracking_obs_results` has a real collision to resolve.**
     `obs_blob` holds exactly one slot per (camera, marker)
     (`result_writer.cpp::write_obs_results`), and `MRK-wrist`,
     `MRK-index`, `MRK-pinky` are deliberately in *both* `main` and
     `Hand{L,R}` — parent and child would overwrite each other's slot.
     Additionally, the child's observations are all `PAIR_DIFF`, whose
     natural `actual`/`predicted` values are pixel *differences*, while
     every existing consumer (MCP `get_observation_gaps`, UI overlays)
     reads fields 0–3 as absolute pixels — writing raw differences
     would reproduce the known `observations.csv`/
     `marker_projections.csv` mode-collision bug inside the DB.
     Resolution: the child reconstructs absolute pixels before writing
     (add the wrist detection back to `actual`, the wrist reprojection
     back to `predicted`; `mahal_dist` stays the PAIR_DIFF value, and
     the currently-unused pad field, index 7, becomes a per-slot mode
     flag); for the shared-marker slots the **parent's entry wins** —
     those observations constrain the parent update, and the child's
     ~18 hand-only markers dominate its own statistics anyway.

  4. **Per-stage diagnostics stay a query, with one caveat.** The four
     scalar columns (`n_inlier_observations`, `cov_condition_number`,
     `nis_value`, `nis_dof`) are inherently per-filter-instance; they
     stay parent-only (today's behavior, unchanged), and per-stage
     NIS/outlier-rate is derived by bucketing `obs_blob` by
     marker→group — group membership is recoverable from the skeleton's
     own `groups:` metadata. The caveat the earlier draft missed:
     shared-marker slots carry only the parent's entry (point 3), so
     the child-stage bucket excludes them — acceptable given the
     child's marker count, but the bucketing query must not double-count
     them into both stages.

  **Per-stage tuning — revised: config layer, not skeleton metadata.**
  The previous draft attached child-solver tuning to the skeleton's own
  `HandL`/`HandR` group definitions. The review rejected that as a
  provenance mistake: skeletons are shared, referenced-by-id registry
  entities describing *topology*, while tuning is iterated per run —
  retuning hand noise would mean either mutating a shared skeleton row
  (breaking provenance for every past run referencing it, violating the
  same immutability principle the project enforces for detection runs)
  or versioning the whole skeleton per tweak. It also contradicts the
  codebase's own pattern: `tracker_configs` has grown a column per
  tuning feature through migrations v22–v28, and the v27 comment
  (`registry_schema.sql`) explicitly chose config-side scoping over
  skeleton groups. The dividing rule, agreed with Harri: **groups are
  structure and live in the skeleton** (joint list, marker list,
  `depends_on`, the reference/anchor marker; the freeflyer boundary is
  derived from topology and needs no storage at all); **tuning is
  numbers and lives in tracker config, keyed by group name**.
  Concretely: a registry child table
  `tracker_config_stages(tracker_config_id, group_name,
  <nullable tuning columns>)` where NULL = inherit from the parent
  config row — mirroring the `parent_id` inheritance `tracker_configs`
  already has. The run's single `tracker_config_id` stays the complete
  provenance record; the config never defines what a group *contains*,
  it only attaches numbers to a name the skeleton defines. Tuning
  columns, from `ChildFilterConfig` (`config.hpp:39`) minus its
  structural fields: `process_noise_std` (+ vel/half-life variants —
  finger dynamics are nothing like torso dynamics), `pose_noise_std`
  (hand keypoints come from a different detector head/crop resolution),
  `outlier_threshold` (the root-cause analysis *is* that one global
  threshold can't serve both chain depths), `min_inliers_ratio`,
  `max_innovation_norm`, min keypoint confidence,
  `init_joint_std`/`init_velocity_std` (child initialization, 3h), and
  optionally UKF `alpha` and the adaptive process-noise gains. A side
  benefit: whether to run hierarchically, and *which* of the skeleton's
  groups become stages, becomes a config-level choice — so monolithic
  vs. hierarchical can be A/B'd on the identical skeleton, which 3k
  needs anyway.
- **Python export/merge**: given the DB correction above, there may be
  nothing to merge at read time at all — if every stage already writes into
  the same `tracking_results` row, `state`/`cov_diag` are already complete
  once all three stages finish. What's still needed is generic (not
  hand-specific) code for the *write* side — decoding a stage's compact
  state and re-encoding it into the full blob's index range via the merge
  map — parameterized by group name, not hardcoded.

### Checkpoint/restore: the real interaction is invalidation, not snapshot cost

Harri flagged that this may need revisiting once mid-sequence UKF-state
snapshots (checkpoint/restore, already a separate planned work item per
the cross-person phase's plan doc) exist. The first draft's answer —
snapshot complexity is a property of running three filter instances,
regardless of whether their output lands in one row or three, so the
persistence layout is unaffected — is true but incomplete. The review
(2026-07-20) identified the interactions that actually matter, in both
directions:

- **Parent re-runs invalidate all child output.** RTS smoothing is a
  backward pass from the sequence end, so a parent re-solve from a
  mid-run checkpoint (the motivating use case: re-running after
  time-local edits) changes the smoothed trajectory well beyond the
  edited window — in principle everywhere. Children consume the
  *smoothed* parent trajectory, so any parent re-run makes every child
  stage stale — and plain result rows record nothing about that. This
  is half the argument for the `tracking_run_stages` table above: a
  parent re-run marks child stages stale, and the app knows to re-run
  them before trusting hand DOFs.
- **Hand-only edits get a fast path.** An edit touching only hand
  keypoints needs only that hand's Stage B re-run — a small filter over
  a sub-range with an externally supplied root, no parent involvement,
  no cross-person coupling. In the interactive keypoint-editing
  workflow this is arguably the strongest practical argument *for* the
  two-pass architecture, and a child-only checkpoint is far cheaper
  than a full-tracker one (small state, root supplied externally).
- **Fixed-lag smoothing bounds the cascade.** With full-batch RTS the
  invalidation above is formally sequence-wide; a fixed-lag smoother
  limits how far an edit's effect propagates. One more reason the child
  solver must consume the parent trajectory as a stream (see
  "Parent-trajectory input is a stream" above), so that migration stays
  a driver-level change.

This keeps the actual engineering scope identical to the narrow pipeline
(no recursion, no live sync, same phase count) — it only changes naming and
where the group list lives (config/skeleton data, not class/table/function
names) so that a future different split (different skeleton, different
groups, a third level) is a config change, not new code. It is explicitly
**not** the full skeleton-classes redesign from earlier this session — that
remains postponed — but it avoids adding new hardcoded structure that the
eventual classes work would just have to undo.

## Measurement model: relative observations against the child's own wrist marker

**Revised per Harri's review** — resolves both open questions this section
originally posed. The anchor is `MRK-wrist`, and `hand.{L,R}` is solved by
*both* filters, with the child's estimate winning in the merged output.

### Wrist ownership: solved twice, child wins

`hand.{L,R}` stays a `main`-solved DOF (unchanged — the parent estimates it
from arm markers + `MRK-wrist`/`MRK-index`/`MRK-pinky`, same as today), and
is now *also* a genuinely estimated DOF inside `HandL`/`HandR` (added to
the group's joint list above), using the full dense finger marker set.
Rationale, straight from `hierarchical-ukf-design.md`'s own original
"Option B: Child Overwrites Parent ⭐ RECOMMENDED": `main`'s ~2-3 wrist-area
markers are a weak constraint competing with ~40 other body markers, while
the same markers plus the full finger set are a strong constraint in the
child's much smaller filter. The child's wrist estimate is simply better,
so it should win.

This does **not** reopen the interleaved/sync design this doc otherwise
avoids. "Child wins" happens once, at the **output merge step** (3i below)
— the parent's own forward pass and smoothing run to completion first,
unaffected, exactly as in the batch model already described. There is no
live feedback into the parent's own running filter; the parent simply
produces a wrist estimate that the merge step discards in favor of the
child's, for that one DOF, after both are already finished.

### Freeflyer moves to `forearm.{L,R}`

Because `hand.{L,R}` is now an estimated DOF *inside* the child, it can no
longer be the child's external, fixed-root boundary — the boundary moves
one joint further up the chain, to `forearm.{L,R}`, which stays
`main`-only and is never touched by the child. Mechanically this is the
same Phase 3d/3e machinery as before (subtree Pinocchio model rooted at an
externally-supplied joint, `set_root_transform()` holding it fixed through
predict/update) — just anchored one joint higher, and fed from the
*smoothed* parent trajectory's `forearm.{L,R}` world transform per frame,
not a live per-frame value.

### Measurement construction: `PAIR_DIFF` with an in-state reference marker, not an external anchor

Because `hand.{L,R}` (and therefore `MRK-wrist`, which attaches directly to
it) is now inside the child's own skeleton, this **no longer needs
`Observation::anchor_position`** (the external-constant-reference mechanism
built for Phase 5 cross-person anchors) **at all**. It's simpler than that:
`MRK-wrist` is a normal marker in the child's own subtree model, so every
other hand marker's observation is exactly the existing, older within-person
relative-observation mechanism — `MeasurementMode::PAIR_DIFF` with
`ref_marker_id` pointing at `MRK-wrist`'s index in the child's *own*
skeleton, reprojected fresh per sigma point like any other in-state
reference marker (`ukf.cpp`'s existing `PAIR_DIFF` branch, unmodified):

- **Measured**: `z = detected_hand_kp − detected_wrist_kp` — both raw
  per-camera detections from the same frame. Same-camera calibration error
  cancels, exactly as in every other existing `PAIR_DIFF` use in this
  codebase.
- **Predicted**: `h(x) = project(hand_kp, x) − project(MRK-wrist, x)` —
  both terms reprojected per sigma point from the child's own state, since
  both markers are now genuinely in-state. No external constant, no anchor
  table, no `anchor_position` field involved.
- **Noise**: `noise_std_override = pose_noise_std * sqrt(2) * crop_scale`
  — unchanged from the earlier draft, matches the existing within-person
  `PAIR_DIFF` formula in `session_reader.cpp` exactly.

This is a genuine simplification versus the earlier draft, not just a
different choice — Phase 5's anchor machinery exists specifically to handle
a reference that lives *outside* the filter doing the estimating (another
person, in that case). Once `hand.{L,R}` moved inside the child's own
state, that problem doesn't apply here anymore.

What Phase 3f/3f2 still need to deliver, revised accordingly: not a
per-marker, per-camera *pixel* anchor table (the earlier draft's plan) —
just `forearm.{L,R}`'s **world transform** (position + orientation, no
camera involved) per frame, read from the parent's smoothed trajectory, to
seed `set_root_transform()` before each of the child's own predict/update
steps.

### What the relative model does and doesn't absorb (review, 2026-07-20)

Two honest caveats, neither blocking:

- **Parent-root error is unmodeled in the child's noise.** The noise
  formula above carries no term for the parent's `forearm.{L,R}` error.
  Root *position* error mostly cancels in the pixel difference (a small
  projection-scale/parallax residual remains); root *orientation* error
  is absorbed by the now-in-child `hand.{L,R}` DOF — but that
  absorption means the exported wrist *joint angle* inherits the
  parent's forearm orientation bias even when finger pixels track well.
  Remember this when judging BVH wrist angles in 3k. If tuning ever
  shows the child's outlier gate misbehaving during fast arm motion,
  the already-built `Tracker::marker_projection_std()` machinery
  (cross-person Stage 3) can inflate child noise from the parent's
  smoothed covariance — the tool exists; no new math needed.
- **Crop-scale source.** Hand keypoints come from dedicated hand-region
  detections (`detection_keypoints.region_type`), so the child's noise
  formula must use the hand crop's `noise_scale`, not the body crop's —
  the per-detection value the observation loading path already carries;
  stated here so 3h doesn't hardcode the body value.

## Composition with the multi-person (cross-person) tracker

Both features restructure the same orchestration layer — the
cross-person plan's `MultiPersonTracker` is implemented — so the
stacking order must be explicit (review, 2026-07-20):

- **Stage A is `MultiPersonTracker`'s coupled forward pass run over
  `main`-only layouts**, with per-person RTS smoothing after the coupled
  pass exactly as that feature already does; each person's hand children
  then run as Stage B off that person's smoothed trajectory. Children of
  different persons are fully independent and can run in parallel.
- **An automatic win**: the cross-person plan's own observation that
  finger-pair candidates crowd out `cross_person_max_n`'s per-camera cap
  disappears — `main`-only layouts have no finger markers, so the cap
  budget goes to meaningful body-contact pairs.
- **An explicit v1 loss, recorded as a decision**: finger markers now
  live only inside per-person child filters that run independently in
  batch, so there is **no cross-person coupling at finger level** — and
  for the motivating contact scenarios (grips, handshakes, assisted
  throws) contact is precisely hand-on-body. `MRK-wrist`, `MRK-index`,
  `MRK-pinky` stay in `main`, so gross hand-scale contact coupling
  survives; that is judged an acceptable v1 trade. Future option if it
  proves insufficient: cross-person anchors *into* a child solve via the
  `Observation::anchor_position` machinery Phase 5 built — the anchor
  reference there is external to the estimating filter, which is exactly
  that mechanism's job (unlike within-child observations, which
  correctly don't need it).

## Revised phase plan (supersedes `hierarchical-tracker-redesign.md` §12 from 3f onward)

| Phase | Work |
|-------|------|
| 3f | (unchanged from the existing plan) `ForwardKinematics::world_transform(joint_name)` — still needed, but now invoked while replaying the parent's *smoothed* state per frame (Stage B setup), not a live per-frame FK cache. |
| 3f2 (new, revised) | Given the parent's smoothed trajectory, compute the per-frame **joint world transform** (position + orientation) for whichever joint a child's config names as its freeflyer (`forearm.{L,R}` for this feature) — no camera/pixel projection involved (see measurement-model revision above). Generic over freeflyer joint name, not hand-specific, and **delivered to the child as an incremental stream** (see "Parent-trajectory input is a stream"). |
| 3g (revised) | No `Tracker` → `HierarchicalTracker` rename/generalization, and no recursive coordinator (see scope decision above). Stage A needs no new code at all — it's today's `Tracker` run against a `main`-only `SkeletonLayout`. |
| 3h (revised, 2026-07-20) | The child-stage solver: constructed from `(freeflyer_joint_name, joint_groups, observation_groups, ref_marker_name, stage_config)`, all data, none hardcoded; owns a subtree UKF + subtree FK (Phase 3d/3e machinery), runs a full forward pass building `PAIR_DIFF` observations against `ref_marker_name`, own RTS smoothing pass after; instantiated twice for this feature with `{"HandL"}`/`{"HandR"}` and `MRK-wrist.{L,R}` as config, not as class identity. **First decision inside this phase — reuse `Tracker`, don't sibling it**: prefer teaching `Tracker` a fixed-root mode (layout + streamed external root trajectory; Phase 3e already put fixed-root support in the UKF) over a new "structurally similar" `ChildFilterSolver` class — a second forward+smooth loop would duplicate outlier gating, NIS feedback, statistics, and result writing, and drift from `Tracker`. If a separate class still wins on inspection, name explicitly which shared pieces get factored out. **Also in this phase — child initialization, previously unspecified**: fixed-root IK over triangulated finger markers when enough are visible; rest pose with `init_joint_std`-wide covariance when occluded (common at sequence start); plus a re-init policy after the child loses tracking. |
| 3i (revised, 2026-07-20) | Output per the DB discussion above: one run row per person (full performer skeleton, as today); every stage — parent included — read-modifies-writes the *same* `tracking_results` rows (both `is_smoothed` families, with the defined placeholder semantics) via `SkeletonLayout::build_index_map_from()`. New `tracking_run_stages` status table + session-DB migration. `obs_blob`: absolute-pixel reconstruction for child `PAIR_DIFF` entries, pad-field mode flag, parent-wins rule for shared-marker slots. The four scalar diagnostics stay parent-only; per-stage stats are the marker→group bucketing query. Per-stage tuning via the `tracker_config_stages` registry table + migration — **not** skeleton group metadata. |
| 3j | CLI/config plumbing: extend `--person` (or a new mode) to select the multi-stage path, driven by however many/whichever named groups the skeleton defines beyond `main`, with per-stage tuning resolved from `tracker_config_stages` (NULL = inherit from the run's base config) — not hardcoded to exactly two hands. Stage selection is config-level, so monolithic vs. hierarchical A/B runs use the identical skeleton. |
| 3k | Integration test: compare against (a) today's monolithic fingers-on tracking and (b) this session's no-fingers-only baseline. Acceptance criteria: parent/body quality matches or exceeds the no-fingers baseline (should be near-identical, since Stage A *is* that baseline), and hand/finger quality is usably close to monolithic fingers-on tracking's finger output, without importing its arm-jerk regression. Re-run this session's visual BVH comparison as part of this, remembering the wrist-angle caveat from the measurement-model section. |
| 3l (new, 2026-07-20) | Python/UI/MCP surfacing, same shape as every other phase's final stage: MCP `get_filter_stats`/`get_run_info` label the scalar diagnostics as parent/body-only and expose the per-stage `obs_blob`-bucketed stats; `content_panels.py` shows stage structure and status from `tracking_run_stages`; `run_tracker.py` gains the hierarchical toggle + per-stage config editing; document the mixed `cov_diag` semantics wherever confidence is displayed. |

## Implementation plan

Sequenced as PR-sized units, each with an acceptance gate, checked
2026-07-20 against the actual code before being locked in:

- `src/db/result_writer.cpp`'s `tracking_results` write path is a plain
  `INSERT INTO tracking_results ...` via a batched `pending_` list — no
  `UPSERT`, no read-back. So PR 4 below needs a genuinely new
  `ResultWriter` capability (`SELECT` + patch + `UPDATE`), not just "the
  RMW pattern is already possible."
- `detection_keypoints.region_type` is real, with exactly
  `'full_body'/'face'/'hand_l'/'hand_r'` values (`session_schema.sql:308,316`)
  — the crop-scale caveat in the measurement-model section has stronger
  grounding than stated: hand detections are already a first-class,
  separately-tracked region type in the detection pipeline.
- The registry `v27` comment quoted in the persistence section is exact —
  independently found earlier in this same review — so the
  config-side-scoping precedent is solid.
- Two gaps found that the phase table doesn't make explicit enough to
  implement directly:
  1. `decode_obs_blob` and its MCP consumers (`get_filter_stats`,
     `get_camera_coverage`, `get_observation_gaps`) don't know about the
     new pad-field mode flag or the parent-wins shared-marker rule —
     writing that data is inert unless those readers change too. Folded
     into PR 5, not left for 3l.
  2. The hierarchical-mode toggle was never named. Resolved: existence-based
     — a `tracker_config_id` with rows in `tracker_config_stages` runs
     hierarchically; one without runs monolithic, unchanged. No separate
     boolean flag. Folded into PR 6.

| PR | Scope | Acceptance gate |
|----|-------|------------------|
| 1 — **done** | `ForwardKinematics::world_transform(joint_name)` (redesign.md §11 — found already implemented, not new work) + `TrajectoryStream`/`BatchTrajectoryStream`, a small streaming interface over a completed `Tracker`'s smoothed trajectory yielding one named joint's world transform per frame via its own `get_fk()`. | `[world_transform]` (pre-existing, 18 assertions) + `[trajectory_stream]` (new, 16 assertions), all passing. Commit `36ae6c5`. |
| 2 — **done** | Decided in favor of teaching `Tracker` a fixed-root mode rather than a sibling class: `TrackerConfig::fixed_root_joint_name` (empty = today's behavior exactly) lets `initialize_ukf()`'s existing subtree-model path anchor at an arbitrary joint instead of the skeleton's own root — no `SkeletonLayout` changes needed, since `from_groups()` already reports `has_floating_root()==false` correctly whenever the requested groups exclude the skeleton's true root (confirmed by reading `skeleton_layout.cpp` before assuming otherwise). New `Tracker::set_external_root_transform()` forwards to `UnscentedKalmanFilter::set_root_transform()` (Phase 3e). Removed the unused `ChildFilter`/`run_child_step()`/`children_` scaffolding (dead code — `children_` was never populated — left over from the superseded interleaved-coordinator design). | New `[fixed_root]` test reusing `tests/data/simple_humanoid.yaml`'s existing `right_arm`/`spine_upper` group shape (already exactly analogous to `HandL`/`forearm.L`) — root held exactly at the injected transform across 9 predict+update cycles, child joints converge to ground truth; 33 assertions, all passing. Full suite otherwise unaffected (473/482 "as expected", same one pre-existing unrelated flake). Commit `57ef7d5`. **Found and fixed a real bug along the way**: `UnscentedKalmanFilter::compute_state_mean()` (and the constructor's initial `state_`) sized `joint_angles`/`joint_velocities` from `layout->skeleton()->total_dof_count()` (the full skeleton) instead of `layout->total_storage_dof_count()` (the layout's own count) — silently padding the state with unused trailing slots for any subset layout. Harmless for the already-shipped `active_joint_groups`-only case (still a floating root; nothing downstream read past the active range), but it broke the compact-state contract PR4's read-modify-write DB merge depends on. Fixed in the same commit. |
| 3 — **done (synthetic-data verification only; the BVH spot check below is still outstanding)** | `Tracker::triangulate_markers()` (refactored out of `initialize()`) + `Tracker::initialize_with_fixed_root()` (triangulated-IK init with a caller-supplied, never-trusted-from-IK root; falls back to rest pose under 3 markers) + `build_ref_marker_pair_observations()` (`PAIR_DIFF`/`ref_marker_id` construction against a fixed reference marker such as `MRK-wrist`, mirroring `session_reader.cpp`'s within-person relative-pair logic), reusing the existing `PAIR_DIFF` branch in `ukf.cpp` unmodified. Re-init policy after tracking loss not addressed — deferred to PR 6/7 alongside the rest of the per-stage orchestration. | New `[relative_observations]` (6 cases, 22 assertions) + `[tracker][fixed_root][relative_observations]` combined init+tracking-loop test (2 sections, 48 assertions), all passing; full suite 267/267 test cases as expected (one pre-existing unrelated flake). Commit `72066eb`. One hand's forward pass against a short *real* sequence with a BVH spot check is still outstanding — noted as an acceptance gap, not silently dropped. |
| 4 — **done** | `tracking_run_stages` + `tracker_config_stages` migration (v37, single combined session DB — not the separate session/registry split originally described; corrected after reading `python/posetrak/db/db.py`'s actual migration mechanism). `ResultWriter` read-modify-write capability: a new attach-mode constructor (`db_path, run_id, person_id` — no `tracking_runs` insert) plus `patch_frame(step, is_smoothed, state_indices, state_values, cov_diag_indices, cov_diag_values)`, which SELECTs the existing row, decodes state/cov_diag as float64 vectors, overwrites the given indices, and UPDATEs the row back. Index semantics are entirely the caller's (e.g. built from `SkeletonLayout::build_index_map_from()`) — `ResultWriter` itself has no skeleton/layout/group-name knowledge. | Schema: `test_create_session_includes_hierarchical_solver_tables` + `test_migrate_session_v36_to_v37_adds_hierarchical_solver_tables`, both passing (`db/migrations/026_hierarchical_solver_stages.sql`, commit `5c6f745`). RMW: new `[result_writer][patch_frame]` suite (7 cases, 48 assertions) against a minimal fixture DB, covering state-only patches, state+cov_diag together, the smoothed-family row patched independently of the filtered one, and the four error paths (missing row, mismatched lengths, out-of-range index, cov_diag patch against a NULL cov_diag row). Commit `92dfd95`. |
| 5 — **done** | `obs_blob` patching: `reconstruct_pair_diff_absolute()` (relative_observations.hpp/.cpp) shifts a child's `PAIR_DIFF`-derived `ObservationResult` entries back to absolute pixels by adding the reference marker's own actual/predicted for the same camera (innovation/mahalanobis_distance are shift-invariant, left untouched). `ResultWriter::patch_obs_results()` writes those into an existing `obs_blob`, skipping any marker in `parent_owned_markers` (parent-wins) and setting the pad field (index 7) to a per-entry mode flag (1.0 reconstructed, 0.0 native). Companion update: `OBS_MODE_ABSOLUTE`/`OBS_MODE_PAIR_DIFF_RECONSTRUCTED` constants in `app/mcp/db.py`; `get_camera_coverage` renders reconstructed entries in lowercase (`i`/`x` vs `I`/`X`); `get_observation_gaps` appends an `r` status suffix — the new data isn't write-only. | New `[reconstruct_pair_diff_absolute]` (4 cases, 22 assertions) + `[result_writer][patch_obs_results]` (5 cases, 37 assertions) + a Python suite covering the pad-field roundtrip and both MCP tools' rendering (4 cases), all passing. Commit `11eecc8`. |
| 6, prerequisite — **done** | Discovered while starting PR 6: neither `freeflyer_joint_name` nor `ref_marker_id` had anywhere to live — the skeleton's `groups:` section had no such fields, and `tracker_config_stages` deliberately excludes them per its own migration comment ("live in the skeleton, not here"). Added two new optional `groups:` YAML fields, `freeflyer_joint`/`ref_marker` (`SkeletonGroup` struct + `Skeleton::add_group()`/`get_group()`, parsed by `skeleton_loader.cpp`, documented in `docs/skeleton-format.md`). Getting their values right for `HandL`/`HandR` surfaced a second gap: the design doc's "wrist ownership: solved twice" requires `hand.{L,R}` to belong to **both** `main` and `HandL`/`HandR` at once, but `Joint::group` is a single string and every group filter (`SkeletonLayout::build()`, `PinocchioModelBuilder`'s subtree builders) matched on it exactly — fixed with `Skeleton::is_joint_in_groups()`, which unions a group's own declared `joints:`/`markers:` list (falling back to `Joint::group` when no `SkeletonGroup` is registered, so every existing group-filtering test/caller is unaffected). Also added the "warn on group entries referencing nonexistent joints/markers" guard the design doc's "Exact group definitions" section called for as a prerequisite. | New `[skeleton_loader][groups]` + `[skeleton_layout][groups]` suites, all passing. Commit `a51ad74`. |
| 6, prerequisite — **done** | `python/tools/upgrade_skeleton_hand_groups.py`: applies the design doc's reviewed `HandL`/`HandR` corrections (phantom `palm.01-04.{L,R}`/`MRK-thumb2.{L,R}` references removed, `freeflyer_joint`/`ref_marker` added) to a skeleton YAML file or every matching row in a registry/session DB (content-addressed `skeletons` rows are never mutated in place — inserts a new row with `parent_id` set to the original). **Shipped a real bug in its first version**: it assumed every skeleton's `palm.*` references were phantom, and corrupted `tests/data/Harri_skeleton-regress-test.yaml`/`Harri_skeleton-shouldery-rot.yaml`, whose `palm.0N.{L,R}` are real, load-bearing joints (a different, unreviewed hand topology) — caught via `git diff` before committing, reverted, and fixed by checking joint existence directly instead of hardcoding the pattern, skipping the hand-group corrections entirely (with a printed note) whenever real `palm.*` joints are present. Applied to all 6 tracked skeleton YAML files. | New pytest suite (9 cases), including a regression test for the exact bug above. Commit `ddbeba7` (script) + `1558cd9` (applied to tracked skeletons). Full C++ suite (288/2980) and `python/tests/db` (253 passed) clean afterward. |
| 6 — **done, not yet integration-tested** | CLI/config plumbing: `src/tracking/hierarchical_solver.cpp`'s `build_stage_tracker_config()` (per-stage tuning resolution, NULL = inherit) + `run_hierarchical_child_stages()` (the existence-based toggle from gap 2, and the full per-stage driver — builds the fixed-root child `Tracker`, streams the parent's smoothed `freeflyer_joint` trajectory, builds each frame's `PAIR_DIFF`+own-position observations, runs forward pass + smoothing, merges into the parent's `tracking_results`/`tracking_obs_results` rows, tracks `tracking_run_stages` status). Wired into both of `cli/track.cpp`'s DB paths after `finalize_person_context()`. `cov_diag` is deliberately not merged yet — needs a parallel `error_index` mapping between layouts that doesn't exist. Also fixed a real, unrelated bug discovered while working out the exact frame alignment this needed: every RTS-smoothed `tracking_results`/`smoothed_*.csv` row was mislabeled by one `tracker_step`, for every `--smooth` run ever, because a warm-up `track_frame()` call before the main loop fed the smoother cache without a filtered-row counterpart. | New `[hierarchical_solver]` suite (3 cases) covering `build_stage_tracker_config()`'s inherit/override/partial-override behavior — the one piece usable without a full DB-backed `Tracker` fixture. The orchestration driver itself compiles and integrates cleanly (full suite green) but has **no automated test of its own yet** — PR 7's integration test, against real production data, is the first end-to-end exercise. Commits `58946db` (dead `HierarchicalConfig` scaffolding removed), `271650f` (smoothed-frame off-by-one fix), `99cf8b1` (DB-layer additions), `5b1d4d6` (this PR). |
| 7 | Integration test: merged output vs. (a) this session's no-fingers monolithic baseline for parent-owned DOFs, (b) monolithic fingers-on for hand-owned DOFs. Re-run this session's visual BVH comparison, checking the wrist-angle caveat (exported wrist angle inherits parent forearm orientation bias even when finger pixels track well). v1 runs children sequentially per person — no parallelism yet, even though the design supports it later. | (a) near-identical to the no-fingers baseline; (b) hand tracking usably close to monolithic fingers-on, without the arm-jerk regression. |
| 8 | Python/UI/MCP surfacing: `get_filter_stats`/`get_run_info` label scalars as parent/body-only + expose per-stage bucketed stats; `content_panels.py` stage structure/status from `tracking_run_stages`; `run_tracker.py` hierarchical toggle + per-stage config editing; document mixed `cov_diag` semantics wherever confidence is shown. **Requires live UI verification per CLAUDE.md** ("start the dev server and use the feature in a browser before reporting complete") — not something to mark done from source inspection alone. | Manual walkthrough in the running app; screenshots/description of what was exercised. |

PRs 1–7 are pure C++/DB/CLI and can proceed without live UI access. PR 8
cannot be verified from source reading alone and needs an interactive
session — flagged explicitly rather than silently skipped or guessed at.

**Status (2026-07-21): PRs 1–5 done.** Not a UI-verification gap (none
of PRs 1–5 need one). PR 2 (teaching `Tracker` a fixed-root mode) turned
out to be smaller than its risk profile suggested —
`SkeletonLayout::from_groups()` already did the hard part correctly, and
the only real surprise was a latent `compute_state_mean()` sizing bug,
now fixed with a regression test. PR 3 (child init + `PAIR_DIFF` wiring)
is verified against synthetic data only; the real-sequence BVH spot check
from its acceptance gate is still outstanding and not something to claim
without running it. PR 4's `ResultWriter` RMW capability turned up its
own real bug, in the new test's fixture rather than production code: a
`cov_diag` blob bound `SQLITE_STATIC` but declared inside the enclosing
`if`-block was freed before `sqlite3_step()` read it, writing freed-memory
garbage into the row — fixed by switching to `SQLITE_TRANSIENT`. PR 5
(`obs_blob` patching) extended the attach-mode constructor from PR 4 to
also load `camera_labels_`/`marker_names_` from the parent's
`tracking_runs` row, so a child stage's `ResultWriter` doesn't need the
caller to re-derive that metadata just to patch `obs_blob`.

**PR 6's two prerequisites and PR 6 itself are done** (`freeflyer_joint`/
`ref_marker` skeleton metadata + the dual-group-membership fix; the
skeleton `groups:` converter script and its application to every
tracked skeleton YAML; the child-stage orchestration driver and its CLI
wiring). All three surfaced real, previously-undiagnosed gaps rather
than being pure plumbing: dual membership wasn't representable at all;
the converter's first version corrupted a differently-topologized test
fixture; and working out PR 6's exact frame alignment surfaced a
pre-existing off-by-one bug affecting every `--smooth` run's smoothed
output, fixed alongside it. PR 6's own driver code compiles, integrates
into the CLI, and the full suite stays green, but it has **no
integration test of its own yet** — it has literally never been run.
PR 7 is that first run: a real production tracking run with hierarchical
hand stages enabled, + BVH export, per Harri's 2026-07-21 request. Given
PR 6 is completely unexercised, PR 7 should start cautiously (a short
sequence first, not the full production run) rather than assuming the
wiring is correct just because it compiles.

## Explicitly out of scope for this proposal

- The general/recursive `HierarchicalTracker` coordinator architecture —
  not needed; the batch model has no live interleaving for it to
  coordinate (see scope decision above).
- The full skeleton-classes redesign from earlier this session — this
  proposal takes the narrower step of keeping the *implementation* generic
  (config/group-driven, not hand-specific code/schema) without building
  class-based skeleton typing. Still postponed, per Harri's earlier
  request to iterate in small steps; this feature is written so it doesn't
  add structure the classes work would later have to undo. The
  structure-vs-tuning rule above (group membership in the skeleton, stage
  tuning in `tracker_config_stages`) was chosen specifically so that when
  the skeleton eventually splits into class (structure, joint limits) +
  per-performer scaling (dimensions), group definitions migrate verbatim
  into the class and nothing from this feature lands on the wrong side.
  Until then, the two guards in the group-definitions section (loader
  warning on stale group entries, cross-performer group-consistency test)
  keep the per-file duplication safe.
- Fixed-lag/sliding-window smoothing — not needed for this feature; remains
  a separate, later project for other reasons (cross-person memory scaling,
  edit-driven re-solving). The stream-based parent-trajectory input is
  what keeps that migration a driver-level change rather than a
  child-solver rewrite.
- The skeleton structure/dimension separation (template vs. per-performer
  scale) discussed earlier this session — explicitly postponed by request;
  this doc's group definitions are written against the current flat
  per-performer skeleton files and will need to move if/when that
  separation happens.
- Fixing the `observations.csv`/`marker_projections.csv` mode-collision bug
  found this session — independently useful but not required to implement
  this feature (this feature doesn't route hand-child observations through
  those CSVs at all, per the export step in 3i).
- A DB migration for per-stage diagnostic scalars, if per-stage
  `n_inlier_observations`/`cov_condition_number`/`nis_value`/`nis_dof`
  ever turn out to be needed as first-class columns rather than derived
  from `obs_blob` — not needed for v1 per the DB discussion above, revisit
  only if the query-based approach proves insufficient in practice.
