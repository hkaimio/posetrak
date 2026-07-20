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
  | 3f | **next in that doc, not yet done** | `ForwardKinematics::world_transform(joint_name)` — reads a named joint's world pose from the Pinocchio cache after `compute()` |

  This means the low-level plumbing a hand child-filter needs — a
  subtree Pinocchio model rooted at an externally-supplied joint, a UKF that
  holds that root fixed through predict/update, sigma points that never
  perturb it — **already exists and is tested**, independent of anything
  decided in this doc.

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
  template/class parameters. The Stage B solver should be a generic
  `ChildFilterSolver` (name TBD, no domain-specific naming), constructed
  from config: `(freeflyer_joint_name, joint_groups, observation_groups,
  ref_marker_name, tracker_config)`. It happens to get instantiated twice,
  with `{"HandL"}`/`{"HandR"}` and `MRK-wrist.{L,R}`, for this feature —
  nothing in the class itself knows what a hand is.
- **Database — revised again, per Harri's review**: no new run identity at
  all, not even a generic one. Checked the actual schema
  (`db/session_schema.sql`): `tracking_results` stores one row per
  `(run_id, person_id, tracker_step, is_smoothed)`, with `state`/`cov_diag`
  as opaque blobs — "full UKF state vector" — and `tracking_obs_results`
  stores `obs_blob` similarly, indexed by camera and marker across the
  *whole* skeleton's marker list (`tracking_runs.marker_names`). Both are
  already shaped as "one wide row per step," not "one row per DOF" — so a
  multi-stage pipeline doesn't need multiple run rows at all. Keep exactly
  one `tracking_runs`/`tracking_run_persons` row per person, referencing
  the performer's *full*, unmodified skeleton (not a `main`-only or
  `HandL`-only variant) — the same one used today. Each solver stage
  internally uses its own smaller `SkeletonLayout` (`from_groups({"main"})`,
  `from_groups({"HandL"})`, ...), but when writing results, expands its
  compact state into the full skeleton's index range using
  `SkeletonLayout::build_index_map_from()` — the exact same merge-map
  mechanism `hierarchical-tracker-redesign.md` §5 already designed for an
  in-memory merge, just applied at the DB-write boundary instead. The
  parent writes first; each child then read-modifies-writes the same row,
  patching only the index range it owns. Genuinely zero schema change for
  `tracking_results`/`tracking_obs_results`.

  One place this doesn't fully resolve cleanly: `tracking_results`'s four
  scalar diagnostic columns (`n_inlier_observations`, `cov_condition_number`,
  `nis_value`, `nis_dof`) are inherently per-filter-instance — each of the
  three solvers has its own independent covariance matrix and NIS, and
  there's no lossless single value once merged. Proposed resolution: leave
  those four columns reflecting the parent/body solver only (today's
  behavior, unchanged), and get hand-specific versions of the same
  diagnostics by bucketing `tracking_obs_results.obs_blob` post-hoc — group
  membership is a static property of marker name, recoverable from the
  skeleton's own `groups:` metadata, so per-stage NIS/outlier-rate doesn't
  need its own column either; it's a query, not new storage.

  Similarly, `tracking_runs.tracker_config_id` / `tracking_run_persons.skeleton_id`
  are single-valued per run — but each solver stage plausibly wants its own
  tuning (matching `ChildFilterConfig`'s original per-child noise/threshold
  fields). Rather than a second FK'd config row, keep child-solver tuning as
  a handful of lightweight fields attached directly to the skeleton's own
  `HandL`/`HandR` group definitions (which the run's existing `skeleton_id`
  already points at) — consistent with the "generic, group-driven" principle
  above, and again no new run-identity or schema concept.
- **Python export/merge**: given the DB correction above, there may be
  nothing to merge at read time at all — if every stage already writes into
  the same `tracking_results` row, `state`/`cov_diag` are already complete
  once all three stages finish. What's still needed is generic (not
  hand-specific) code for the *write* side — decoding a stage's compact
  state and re-encoding it into the full blob's index range via the merge
  map — parameterized by group name, not hardcoded.

### Checkpoint/restore doesn't change this calculus

Harri flagged, correctly as a caution, that this may need revisiting once
mid-sequence UKF-state snapshots (checkpoint/restore, already a separate
planned work item per the cross-person phase's plan doc) exist. On
reflection I don't think it actually favors one persistence option over the
other here: checkpoint/restore complexity comes from needing to snapshot
**three independent filter instances'** internal state (covariance,
sigma-point cache, NIS-feedback windows, `prev_observations_`, ...), which
is a property of *running three separate `UnscentedKalmanFilter` objects*
— true regardless of whether their eventual output lands in one
`tracking_results` row or three. The single-row output design doesn't add
to that cost, and three separate run rows wouldn't have reduced it either.

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

## Revised phase plan (supersedes `hierarchical-tracker-redesign.md` §12 from 3f onward)

| Phase | Work |
|-------|------|
| 3f | (unchanged from the existing plan) `ForwardKinematics::world_transform(joint_name)` — still needed, but now invoked while replaying the parent's *smoothed* state per frame (Stage B setup), not a live per-frame FK cache. |
| 3f2 (new, revised) | Given the parent's full smoothed trajectory, compute and store the per-frame **joint world transform** (position + orientation) for whichever joint a child's config names as its freeflyer (`forearm.{L,R}` for this feature) — no camera/pixel projection involved (see measurement-model revision above). Generic over freeflyer joint name, not hand-specific. |
| 3g (revised) | No `Tracker` → `HierarchicalTracker` rename/generalization, and no recursive coordinator (see scope decision above). Stage A needs no new code at all — it's today's `Tracker` run against a `main`-only `SkeletonLayout`. |
| 3h (revised) | New generic `ChildFilterSolver` (name TBD — not hand-specific): constructed from `(freeflyer_joint_name, joint_groups, observation_groups, ref_marker_name, tracker_config)`, all data, none hardcoded. Owns a subtree UKF + subtree FK (Phase 3d/3e machinery), runs its own full forward pass building `PAIR_DIFF` observations against `ref_marker_name` (per the measurement-model revision above), own RTS smoothing pass after. Structurally similar to two already-tested code paths — `Tracker`'s own forward+smooth loop, and the within-person `PAIR_DIFF` construction already in `session_reader.cpp` — rather than new UKF math. Instantiated twice for this feature, with `{"HandL"}`/`{"HandR"}` and `MRK-wrist.{L,R}` as config, not as class identity. |
| 3i (revised again) | Output: **no schema change** (see DB discussion above). One `tracking_runs`/`tracking_run_persons` row per person, referencing the full performer skeleton as today. Each solver stage — parent included — writes into the *same* `tracking_results`/`tracking_obs_results` row per step, expanding its compact state into the full skeleton's index range via `SkeletonLayout::build_index_map_from()` before a read-modify-write. `n_inlier_observations`/`cov_condition_number`/`nis_value`/`nis_dof` stay parent-only; per-stage versions are a query over `obs_blob` bucketed by marker→group, not new storage. Per-stage tuning lives as fields on the skeleton's own `HandL`/`HandR` group metadata, not a second `tracker_config_id`. |
| 3j | CLI/config plumbing: extend `--person` (or a new mode) to select the multi-stage path, driven by however many/whichever named groups the skeleton defines beyond `main`, with per-stage tracker config (process/measurement noise, outlier threshold — same shape as `ChildFilterConfig`, already fully specified in `config.hpp`) — not hardcoded to exactly two hands. |
| 3k | Integration test: compare against (a) today's monolithic fingers-on tracking and (b) this session's no-fingers-only baseline. Acceptance criteria: parent/body quality matches or exceeds the no-fingers baseline (should be near-identical, since Stage A *is* that baseline), and hand/finger quality is usably close to monolithic fingers-on tracking's finger output, without importing its arm-jerk regression. Re-run this session's visual BVH comparison as part of this. |

## Explicitly out of scope for this proposal

- The general/recursive `HierarchicalTracker` coordinator architecture —
  not needed; the batch model has no live interleaving for it to
  coordinate (see scope decision above).
- The full skeleton-classes redesign from earlier this session — this
  proposal takes the narrower step of keeping the *implementation* generic
  (config/group-driven, not hand-specific code/schema) without building
  class-based skeleton typing. Still postponed, per Harri's earlier
  request to iterate in small steps; this feature is written so it doesn't
  add structure the classes work would later have to undo.
- Fixed-lag/sliding-window smoothing — not needed for this feature; remains
  a separate, later project for other reasons (cross-person memory scaling,
  edit-driven re-solving).
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
