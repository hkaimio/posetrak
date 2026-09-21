<!--
SPDX-FileCopyrightText: 2026 Harri Kaimio

SPDX-License-Identifier: Apache-2.0
-->

# Dot assignment robustness: plan

Work plan for making the tracking-time assignment of anonymous reflective-dot
candidates to marker slots resistant to *confidently wrong* candidates: a
candidate that is sharp, passes the Mahalanobis gate, and is the wrong physical
point. It refines WS2 of
[productization-architecture-and-plan.md](productization-architecture-and-plan.md)
(§3.6.2, §3.6.3 and §5) against the code as it stands after the CLI workflows
were delivered. Where this page and that one differ, this page is the newer.

## 1. What exists now

The assignment lives in `cpp/src/tracking/dot_assignment.cpp`. For each camera
and tracker step it builds one cost matrix, candidates × the slots of every
dot-bearing subject, and solves it once (`solve_assignment()`, gated by
`dot_assignment_gate_mahalanobis`). The cost of a pair is the squared
Mahalanobis distance between the candidate and the slot's *predicted* pixel
position under the prediction covariance. Two things then change that cost:

| Mechanism | Where | Effect |
|---|---|---|
| Tracklet continuity | `resolve_dot_assignment()`, opt-in via `dot_tracklet_gate_multiplier > 1` | cost ÷ multiplier when the candidate's tracklet id equals the one resolved into this (subject, camera, marker) slot on the previous step |
| Candidate ownership | same loop, `subject_mask` | cost set to `kNotOwnedCost` when the candidate is not in the subject's own sequence |

Two other things the design doc lists as cost mechanisms are *not* in the
matrix. Backface culling happens earlier, in the prediction
(`predict_rigid_marker()` returns no prediction, so the slot has no column
for that camera). The streak-velocity observation is emitted after the solve
and does not affect who wins. Only the two above are refactored in WS2.1.

Per-step state the assignment can read: each subject's previous resolved
pixel position and tracklet id per (camera, marker)
(`Tracker::prev_observations()`, `prev_dot_tracklet_ids()`). Both are stale
until overwritten, so a slot that stopped resolving keeps its last entry, and
neither records *when* it was written. There is no per-slot gap counter yet.

There is no cross-camera information in the assignment. Each camera is solved
independently; the cameras only meet in the filter update afterwards.

### Evidence of the failure

- **Ball, `gopro13_02`.** With that camera's raw dot feed included, at least
  two near-static clutter candidates alternated with the real moving ball.
  Each jump stayed under the outlier threshold, three good cameras plus one
  wrong one still looked like four agreeing observations, and the fused
  position was pulled off course for tens of frames. Reprojection error over
  the throw peaked at 418 px until that camera was removed.
- **Leg module, ankle markers.** A marker reacquired at a wildly discontinuous
  position after a two-frame gap, with nothing flagging it.

The two failures differ in shape, which decides which mechanism addresses
which (§2.2, §2.3):

- The clutter case has a candidate **resolved on every step**, so no slot ever
  has a gap. A reacquisition gate does not see it.
- The ankle case has a **gap followed by a jump**, which is what a
  reacquisition gate is for.
- Tracklet continuity can *reinforce* the clutter case: once the slot has been
  resolved to a clutter candidate, that candidate's static tracklet gets the
  relaxed gate on the next step. This is a hypothesis to check in WS2.0, not an
  observed fact; the option is off by default.

## 2. Design

### 2.1 A seam for evidence: context and modifiers

Replace the growing positional parameter list of `resolve_dot_assignment()`
with a context and an ordered list of modifiers. A modifier is a pure function
of (slot, candidate, context) that returns a **support factor** and may exclude
the pair:

```cpp
struct DotAssignmentContext {
    TrackerConfig const& config;
    int frame_idx; double timestamp;
    PrevDotPositions const& prev_positions;
    PrevDotTrackletIds const& prev_tracklet_ids;
    SlotGapCounters const& gap_frames;      // new state, §2.2
    SlotEvidence const& evidence;           // per-step pre-pass, §2.3
    // ...
};
```

- The cost is **divided** by the support factor: 1 is a no-op, above 1 makes
  the pair cheaper, below 1 dearer. Division, not multiplication by a
  reciprocal, because the tracklet mechanism divides today and `x / m` and
  `x * (1 / m)` differ in the last bit. This is what makes the refactor
  bit-identical.
- The two existing mechanisms become the first two modifiers, applied in
  today's order (tracklet, then ownership, the latter overriding).
- The streak-velocity step is not a modifier and keeps its own configuration
  bundle; it is only moved onto the context.
- The Mahalanobis inverse in the inner loop is recomputed per pair. Hoisting it
  per column is safe (same arithmetic) and worth doing only if a profile asks
  for it; not part of this plan.

The gate-config agreement check (`find_dot_config_disagreement()`) already
exists, so that half of the original WS2 item 1 is done.

### 2.2 Reacquisition gate

After `dot_reacquire_gap_frames` consecutive steps in which a slot was not
resolved *in a camera*, that (slot, camera) may only be resolved to a candidate
with independent evidence: tracklet continuity with the slot's last resolved
tracklet, **or** a candidate within `dot_reacquire_max_px` of the slot's
prediction in at least two cameras. Otherwise the slot stays unresolved and the
filter coasts. Cross-camera corroboration (§2.3) joins the "or" list when it
exists.

- The gap is counted per (subject, camera, marker), matching the state that
  `prev_observations()` already has. A slot seen by other cameras is anchored
  by them; the restriction applies to the camera that lost it.
- New state: the frame at which each (subject, camera, marker) was last
  resolved, kept in the `Tracker` beside `prev_dot_tracklet_ids()`.
- Default `dot_reacquire_gap_frames = 0` means off, and the run is
  bit-identical to one without the feature.
- Cost of coasting: a legitimate reacquisition with no second camera is
  delayed until the evidence exists. Whether that trades well is what the
  ankle window measures.

### 2.3 Cross-camera corroboration

The assignment needs to know, for a candidate in camera A, whether another
camera holds a candidate that agrees with it for the same slot. Both this and
the "two cameras near the prediction" test of §2.2 need data from other cameras
inside a per-camera loop, so build it once per step in a pre-pass:

- `SlotEvidence`: for each slot and camera, the candidates within a radius of
  the slot's prediction. Candidates outside every gate are dropped first, so the
  pre-pass costs on the order of the gate graph, not of rows × columns.
- Corroboration test: candidate c in camera A for slot s is corroborated when
  some candidate c′ in another camera B, also in `SlotEvidence` for s, has an
  epipolar distance to c below `dot_cross_view_corroboration_px`. The
  fundamental matrix of each camera pair is computed once from the calibrated
  cameras. A corroborated candidate gets support > 1, an uncorroborated one 1.
  Never below 1: a lone camera with everyone else occluded must not be
  penalised.

**Feasibility comes first.** A candidate inside the gate is close to the true
one, so its epipolar line is close to the true candidate's line. The test
separates them only when the tolerance is *tighter than the offset of the
confidently wrong candidate*, and the tolerance cannot be tighter than the
calibration accuracy. Reprojection medians on the leg case are 16–32 px, which
mixes calibration and model error and says nothing directly about epipolar
error. WS2.3 therefore starts with a measurement, before any C++: the epipolar
residual of known-good correspondences (the hand-tracked ball in three
cameras) per camera pair, against the offset of the clutter candidates. If the
95th percentile of the good residuals is not clearly below the clutter offset,
this modifier cannot separate them and the plan moves to the alternatives
(triangulate the other cameras' resolved observations and test this camera's
candidate against the projection; that is the filter's prediction re-derived,
so expect it to add little).

`prototype_ball_tracking.py` made a multi-view RANSAC consistency check; read
it first and port the check it makes rather than a new one.

### 2.4 Per-camera trust

`dot_camera_noise_scale`: a per-camera factor on the measurement noise of the
resolved observations of that camera's dot candidates.

- Applied to the **measurement noise** of the resolved observation (where it
  changes the Kalman gain), not as a cost modifier. The original design lists
  it in both places; noise is where "trust" acts, and a cost factor only
  changes who wins, which a poor camera is not usually wrong about.
- Default: empty map, no change. The values come from data: the per-camera
  median residual of a baseline run, checked against NIS/dof and the
  reprojection medians of the leg case before and after.

### 2.5 Segmentation ROI and hue class: gated by offline tests

Neither is built until an offline test on data that already exists shows the
signal separates the confidently wrong candidates from the real ones. Ground
truth for "the real candidate" is the projection into `gopro13_02` of the
hand-tracked 3D ball; candidates in the slot's gate that are not near it are
the clutter.

| Test | Data | Go criterion (proposed, to confirm with the first numbers) |
|---|---|---|
| Segmentation ROI | the ball's stored Cutie masks and the dot candidates of `gopro13_02` | at least 90 % of real candidates inside the mask, and at least half of the clutter candidates outside it |
| Local hue class | video frames at the candidates' positions in `gopro13_02`; a hue sample around each candidate | the real and clutter hue distributions differ by more than their spread, so that a class threshold separates most of both |

A pass adds a soft modifier (support above 1 for agreement, below 1 for
disagreement, exactly 1 when the evidence is missing, never an exclusion, as a
thin fast prop can leave its mask). The hue test also decides whether the
detector should store a hue feature per candidate, which is an additive change
to the detection blob and is not started before the test passes. A failed test
is recorded in this page, and the item is dropped.

### 2.6 Configuration

All new settings default to off, so a run that does not set them is
bit-identical to today.

| Setting | Meaning |
|---|---|
| `dot_reacquire_gap_frames` | 0 = off; consecutive unresolved steps before a (slot, camera) needs evidence |
| `dot_reacquire_max_px` | radius for the "two cameras near the prediction" test |
| `dot_cross_view_corroboration_px` | epipolar tolerance; 0 = off |
| `dot_camera_noise_scale` | camera → factor; empty = off |

Each is added along the path `dot_tracklet_gate_multiplier` takes:
`TrackerConfig` and its TOML twin in `cpp/include/posetrak/core/config.hpp` and
`cpp/src/core/config.cpp`, the loader in `cpp/src/db/session_reader.cpp`, the
`tracker_configs` table (`python/posetrak/db/sql/registry_schema.sql`, with a
migration) and `python/posetrak/db/db.py`. The setting is also given to the
tracker run options where the tracklet multiplier is.

## 3. Work packages

Each is its own pull request, merged only after its validation passes. Sizes
follow the plan's legend (S about 1–2 days, M 3–5).

| # | Package | Size | Validation |
|---|---|---|---|
| 0 | **Baselines before any change.** A reference copy of the optimized tracker binary from the merged code. Windowed validation cases: the leg module around the ankle event, and the ball with the raw `gopro13_02` feed added back to a database copy (`sequence add-dots --camera`). A discontinuity metric in the validation driver (§4). Record the values on the *current* code | S | the new cases run and their numbers are recorded in `validation-baseline.md`; the ankle jump and the clutter divergence reproduce, or the case is rebuilt until they do |
| 1 | **Context and modifiers.** §2.1, no behaviour change | M | bit-identical output against the reference binary on every case (§4); unit tests of each modifier alone |
| 2 | **Reacquisition gate.** §2.2, including the per-slot gap state | S–M | ankle window: no discontinuity above the threshold, tracked percentage and NIS/dof within tolerance of the baseline; every other case bit-identical with the setting at 0 |
| 3 | **Corroboration.** Measurement of §2.3 first; then the pre-pass and modifier if it can separate | M | the measurement is recorded whichever way it goes; on the ball with `gopro13_02`, peak reprojection error close to the run without that camera (the 3-camera run is the reference), other cases unchanged |
| 4 | **Per-camera trust.** §2.4 | S | leg case: NIS/dof and per-camera medians no worse, and the changed cameras named |
| 5 | **Offline tests for ROI and hue.** Prototype scripts (exempt from the comment rules), results recorded here | S each | a go/no-go per the table in §2.5; a soft modifier PR only on a go |

WS2.0 and WS2.1 come first, because everything else is measured against them
and lands on the seam. WS2.5's tests do not depend on the C++ work and can run
in parallel with it. A modifier that fails its validation is dropped, not tuned
until it passes.

## 4. Validation

- **Reference binary.** Copy `optbuild/cpp/cli/posetrak-tracker.exe`, built from
  the merged code, before the first change. "Bit-identical" means the tracking
  results of the new binary and the reference binary are byte-equal for the same
  input: the same rows in `tracking_results` and the same
  `tracking_obs_results` blobs. The tracker is deterministic, so any difference
  is a real difference.
- **Cases.** The existing driver cases (pen, pad, single ball, leg module,
  person + ball) plus the two windows of WS2.0. The sword needs its capture
  drive mounted; it is run before the merge of WS2.1 and again at the end.
- **Discontinuity metric.** Per case, the number of steps at which a slot's
  resolved pixel position moves by more than a stated number of pixels from its
  previous resolved position in the same camera, counting only steps within a
  small number of frames of the previous one. It is the direct measure of the
  failure, which NIS/dof and reprojection medians average away.
- **Runtime.** The full leg case takes about 27 minutes; iteration uses the
  window. A modifier that adds more than a few percent to the per-step time of
  the leg case is measured with `frame_step_profile` and reported.
- **Unit tests.** `cpp/tests/test_dot_assignment.cpp` gets one test per
  modifier on fabricated predictions and candidates, and a test that a
  candidate is never claimed by two subjects with every modifier enabled.
- **Not covered.** All numbers come from the recorded captures. A change that
  improves tracking fails a check until the reference is deliberately updated
  and the reason recorded, as for the existing baselines.

## 5. Decisions

1. **Per-camera trust as measurement noise, not cost.** Recommended (§2.4).
   The original design puts it in both places.
2. **Sword before merging WS2.1.** Its capture is on a drive that must be
   mounted. Recommended: mount it for the refactor, since the refactor touches
   the code every dot-bearing run goes through; the other cases are then only a
   first line of defence.
3. **Defaults stay off in the tracker.** Whether a validated modifier is then
   turned on by default in the CLI or GUI is a separate decision, made per
   modifier once its validation is recorded.

## 6. Out of scope

- The multi-view seed provider and dots-only multi-dot initialisation
  (initialisation, not assignment).
- Reading per-source noise from the stored candidates.
- The update-cost profile at larger marker counts and the solver's scaling
  with the candidate count; both are measured before the torso and arm
  capture is processed, and neither is changed here.
- Automatic labelling of tracklets at calibration time.
