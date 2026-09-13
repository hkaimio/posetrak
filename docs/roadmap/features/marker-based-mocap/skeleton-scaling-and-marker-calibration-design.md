# Skeleton scaling & marker calibration — design (draft, 2026-09-13)

## Status and scope

Design draft only, no implementation attached. Companion to
[marker-catalog-and-assignment-redesign.md](marker-catalog-and-assignment-redesign.md)
(catalog spec + B1-B5 assignment/calibration pipeline, which this doc
assumes) and to `status.md`'s 2026-09-12 entries, which built and
validated the single-camera reprojection refit and, in the course of
fixing `ankle_lat_L`'s calibration, surfaced the circularity risk this
doc's core recommendation is built around.

Written from a real discussion with Harri after reviewing the first
end-to-end dot-augmented tracking run on the `2026-09-06-kare-tests`
capture, then revised after Harri's own review of the first draft
caught a real error in §2 (a rigid-cluster fit does **not** give bone
length on its own — see §2.2) and raised two substantial extensions
(§2.5, §5) this revision folds in.

Harri's own diagnosis, recorded close to verbatim since it is the
actual scope of this doc:

1. Bone lengths (specifically legs) are somewhat too long. Confirmed
   quantitatively, not just visually: every well-fit ankle marker's
   `along` fraction (fitted position along the shin bone, 0 = knee end,
   1 = ankle end) clusters at **0.85-0.87** across three independently
   fit markers (`ankle_lat_R`, `ankle_med_L`, `ankle_med_R`), not
   anywhere near 1.0 — consistent with the shin bone being modelled
   roughly `1/0.86 ≈ 1.16x`, i.e. **~16% too long**.
2. Torso (spine, clavicles) and neck/head are hard to calibrate from a
   handful of keypoints — plausibly why tracked results often show a
   torso bent slightly forward.
3. Hip keypoints are noisier than most, and pivotal to good results.
4. Manual per-bone calibration effort is reasonable for large bones but
   not feasible for fingers — and hand accuracy (pinches, grabs) is a
   real target use case.
5. The current method assumes keypoints sit at joints. Harri is unsure
   how true that is, hip especially.

## 1. Why a single joint bundle adjustment is the wrong first move

The obvious "correct" formulation is one big nonlinear optimization over
bone lengths, per-frame joint angles, and marker offsets together. Two
reasons to not build that first:

- **Circularity.** If it is seeded from (or scored against) the
  *current* tracked joint poses, it inherits whatever bone-length or
  joint-position error the current skeleton already has — exactly the
  failure mode `ankle_lat_L` just demonstrated at a different layer: the
  dot-assignment gate used that marker's own (wrong) offset to decide
  which observations to trust, and the resulting refit converged right
  back to the wrong answer from every tested starting point (verified
  directly, `status.md` 2026-09-12). A bone-length optimization that
  trusts today's tracked poses risks the identical trap.
- **Identifiability.** A longer bone plus a compensating shift in a
  marker's own offset can explain nearly the same marker trajectory —
  nothing in a single joint fit forces the *physically* correct
  decomposition over merely *a* locally-consistent one. Per-frame joint
  angles as free variables also make the problem large (thousands of
  frames) for comparatively little benefit if the real information is
  in the markers' own rigid geometry, not in re-deriving pose frame by
  frame.

## 2. Core idea: solve what markers already over-determine, before trusting anything the skeleton assumes

### 2.1 Rigid marker clusters per segment

Any skeleton segment with **3 or more** dot markers rigidly attached to
it can have its own 6-DOF pose solved directly from triangulated marker
positions via Kabsch/Umeyama rigid registration — **no bone length, no
joint hierarchy, no skeleton IK involved at all.** This is not new
math for this project: `calibrate_rigid_marker_body.py` already solves
exactly this problem for the sword prop (rigid marker-body geometry
from ordinary capture footage), and `predict_rigid_marker()` /
`initialize_rigid_body()` already implement the closed-form pose math
the tracker itself uses for a whole rigid object. Applying the same
approach to one *segment* of an articulated skeleton, instead of one
whole object, is the extension this doc proposes.

On the `leg` module today: `shin` has 5 markers (`knee_lat`, `knee_med`,
`knee_front`, `ankle_lat`, `ankle_med`) — comfortably enough. `foot`
has 2 (`heel`, `toe`) — borderline, see §6. `thigh` has 1 (`hip`) —
not enough on its own.

**What this actually buys you** (corrected from the first draft, which
overclaimed — see §2.2): a per-frame, skeleton-independent 6-DOF pose
for the segment, and — the more valuable, more clearly justified part —
a **jointly, more robustly fit local offset for every marker in the
cluster**, since markers sharing one rigid body mutually constrain each
other. A weakly-observed marker doesn't need its own assignment to
succeed in isolation: its offset can be solved from exactly the frames
where the *other* markers in the cluster already pin the segment's pose
down confidently. This reframes tonight's `ankle_lat_L` fix in a more
satisfying way — a joint rigid-cluster fit of all 5 `shin` markers
together would plausibly have solved `ankle_lat_L`'s offset correctly
from the start, carried by `knee_lat`/`knee_med`/`knee_front`/
`ankle_med`'s already-good coverage, with no need for the mirrored-guess
workaround this session actually used. Reuses the same
co-occurrence-and-average approach `calibrate_rigid_marker_body.py`
already validated: bootstrap from the skeleton's current (possibly
wrong) offsets, solve each frame's rigid pose via Kabsch against that
geometry, re-average the local geometry from all frames' aligned marker
clouds, iterate for robustness.

### 2.2 What a rigid-cluster fit does *not* give you: bone length

**Harri's correction to the first draft, and it's right.** A
rigid-cluster fit gives a self-consistent local coordinate frame for a
segment's *own* markers — it says nothing about where that frame's
origin sits relative to where the segment actually attaches to its
neighbours. "The markers span roughly this much space" is not the same
claim as "the joint centers are this far apart," and conflating the two
would just reintroduce finding (5) in a different guise (assuming
markers *are* the joint, only now averaged over several markers instead
of one). Recovering a true bone length needs one of:

- the two-sided functional joint center estimation in §2.3 — requires
  markers on **both** neighbouring segments; or
- the chain-calibration approach in §2.5 — relaxes that requirement
  using an assumed joint *type* instead of markers on every link, at
  the cost of needing enough distinct joint-angle configurations across
  the capture to be identifiable.

Concretely, on this dataset: `shin`'s own internal marker geometry is
directly useful for marker-offset calibration (§2.1) today, but `shin`'s
true knee-to-ankle *length* is only as good as whichever end's joint
center is independently known. The `ankle` end is reachable via §2.3
once `foot` is properly instrumented; the `knee` end is not, since
`thigh` has only one marker — so, contrary to the first draft's claim,
`shin` length is **not** fully solvable from today's capture without
either more `thigh` markers or the chain-calibration approach below.

### 2.3 Functional joint center estimation — needs two adjacent, instrumented segments

Where two adjacent segments *each* have their own independently-solved
rigid pose trajectory (`shin` + `foot`, once `foot`'s cluster is
trustworthy), the joint between them (`ankle`) has a well-established
biomechanics answer: fit the point that stays most consistent relative
to *both* segments' own frames across the whole range of motion (a
sphere-fit / SCoRE-style functional calibration), rather than assuming
any single marker sits at the joint.

This is the direct answer to finding (5) — for a joint with two
well-instrumented neighbours, "where is the joint" becomes a measured
quantity per capture, not an assumption baked into the catalog. It is
**not** yet available for `knee` or `hip` on this dataset, since `thigh`
has only one marker and so has no independent rigid trajectory of its
own to fit against `shin`'s (§2.5 is what recovers something here
anyway, without needing `thigh` itself instrumented).

### 2.4 Anchored regional refinement, as a fallback

For the residual case neither §2.3 nor §2.5 (below) can reach — no
segment near a joint has enough markers on *either* side, or the
motion captured doesn't condition a chain-calibration fit well — a
small, anchored bundle adjustment is still the right fallback tool, not
the global one rejected in §1:

- Fix whatever §2.1-2.3/2.5 already solved as boundary conditions, not
  free variables.
- Only let the genuinely unconstrained pieces float.
- Regularize with a soft left/right symmetry prior by default — not a
  new mechanism: `TrackerConfig` already carries this shape of soft
  regularization (`pose_reg_joint_names`, `pose_reg_equal_split_noise_std`)
  for a different purpose (spine posture); a bone-length symmetry prior
  reuses the same idea.
- Objective: marker reprojection error (identical math to tonight's
  single-camera refit) against the still-floating parameters, plus the
  symmetry term, plus a soft prior toward the nominal/previous bone
  length so the optimizer cannot wander arbitrarily far without
  evidence.

### 2.5 Calibrating a completely unmarked middle link (Harri's question)

Harri's sharper framing of §2.4: assume rigid solutions exist for links
1 and 3 in a chain (say `pelvis` and `shin`), plus a rough current
estimate of the two joints connecting them (`hip`, `knee`) — but the
link between them (`thigh`) carries **no markers at all**. Can the
thigh's length and the two joint locations still be improved from just
the two solved end trajectories?

**Yes, in principle** — this is structurally the same problem as
**kinematic calibration of a robot or exoskeleton arm from end-effector
pose measurements**, a well-studied problem outside mocap, not
something this project needs to invent from scratch. At every frame,
the chain `pelvis(t) -> hip_joint -> thigh -> knee_joint -> shin(t)`
must close consistently between the two *independently measured* end
poses. The unknowns are only: the small set of fixed geometric
parameters (hip-in-pelvis offset, knee-in-shin offset, thigh length),
plus per-frame joint angles for the chain's own (small) DOF between the
two solved ends — e.g. hip's 3 rotational DOF + knee's 1 flexion DOF,
**not** the whole body's DOF. That is a much smaller, better-conditioned
bundle adjustment than §1's rejected whole-skeleton version, and it
needs **no markers on the middle link at all** — only that the joint
*types* (spherical hip, hinge knee) are trusted, which is a far weaker
assumption than trusting the joint *locations* or bone *lengths* this
whole exercise is trying to establish. The skeleton schema already
carries joint type (REVOLUTE/SPHERICAL/etc. per
`docs/cpp-architecture-overview.md`), so this doesn't need new schema,
only new fitting code.

Same caveat as §2.3: needs enough independent joint-angle configurations
across the capture to be identifiable — a chain moved through only one
DOF's worth of motion under-constrains the rest (§6, open question 2).

This generalizes recursively: **any** uninstrumented link bounded on
both sides by a solved rigid segment — marked directly, or itself
already resolved by a previous chain-calibration step — can be
calibrated this way, working outward from wherever real markers exist
toward the body's extremities. It directly answers a question Harri
raised about the staging order too: once neighbouring segments have
*any* marker-derived trajectory (their own, or a previously chain-solved
one), a segment with zero markers of its own is not stuck at today's
keypoint-only accuracy — it inherits calibration from its neighbours.
Full marker coverage of every segment was never the actual requirement;
markerless remains the right default for segments that will never carry
markers, and this section is exactly what lets those segments still
benefit from wherever real markers do exist nearby.

## 3. Staging recommendation

Ordered by what is *actually instrumented today*, not by anatomical
region alone — a refinement of Harri's own "legs+hips, then torso+head,
then arms+hands" staging, made concrete about the boundary:

1. **`shin` rigid-cluster fit** (§2.1) — buildable now, on existing
   `2026-09-06-kare-tests` data, no new capture needed. Gives better,
   jointly-consistent marker offsets for all 5 `shin` markers; does
   *not* by itself give `shin`'s bone length (§2.2).
2. **`ankle` functional joint center** (§2.3), once `foot`'s 2-marker
   cluster is validated as trustworthy (§6) or a 3rd foot marker is
   added in a future capture.
3. **`thigh` length + `hip`/`knee` joint centers via chain calibration**
   (§2.5), using `pelvis` and `shin`'s already-solved trajectories as
   the two anchors — needs no new markers on `thigh` at all, only a
   trusted joint-type assumption and enough range-of-motion diversity.
4. **Anchored regional refinement** (§2.4) only for whatever §1-3 still
   can't reach.
5. **Torso/head, arms/hands**: the same mechanism (§2.1-2.5) applies
   mechanically once those regions have their own catalog and enough
   nearby anchors — this is a capture-design and module-registry
   question (see the productization plan), not a new algorithm. Until
   such a layout exists and is captured, these regions keep using
   today's keypoint-based joint assumption, unimproved.

## 4. Prerequisites / concrete next steps

- **Catalog schema addition** (call it D7, extending
  marker-catalog-and-assignment-redesign.md §1.10's open-decisions
  table): a way to declare that a *set* of a module's markers forms a
  rigid cluster on one segment, distinct from today's per-marker-only
  `parent_joint` attachment. The information already implicitly exists
  (markers sharing one non-hip `parent_joint` are already, today, a
  rigid-cluster candidate) — this is about making it an explicit,
  queryable property rather than leaving it emergent.
- Reuse `calibrate_rigid_marker_body.py`'s solved-geometry-from-capture
  math, generalized from "one designed rigid object" to "one skeleton
  segment with an a-priori-unknown local frame."
- The chain-calibration approach (§2.5) needs a small new fitting tool
  (not built anywhere in this project yet): given two segments' solved
  pose trajectories and the connecting chain's joint types, solve the
  small per-link geometric-parameter set. Scope this as its own
  follow-on once §2.1-2.3 are validated on real `shin`/`foot` data.
- Validate any resulting bone-length or joint-center change the same
  way every other change this session was checked: re-track, compare
  tracked-fraction and NIS/dof against the current baseline, and check
  the specific markers' own reprojection error before trusting it over
  today's keypoint-derived value.

## 5. A further payoff: correcting markerless keypoint bias (Harri's second idea)

Once a joint or segment has an independently marker-solved location
(from §2.1/2.3/2.5), that same ground truth can be reprojected into
each camera and compared against that *same frame's* raw markerless
keypoint detection (vitpose/openpose) for the corresponding joint —
giving a real, per-joint, per-camera-viewing-angle error sample.
Aggregated across a calibration trial (or several), this can expose
whether the markerless detector has a systematic, **viewpoint-dependent
bias**, not just noise — Harri specifically suspects this for hip.

If a real bias-vs-viewing-angle pattern shows up, a corrective term
(RBF or similar, keyed on the relative camera-to-joint viewing angle)
could be fit and applied to de-bias the markerless detector's own
keypoint output directly. This would be a materially different, and
arguably higher-leverage, payoff than the marker-augmented tracking work
itself — it improves the *baseline* every future markerless-only run
starts from, not just marker-augmented captures.

This needs a real "marker-derived joint ground truth vs. raw keypoint
detection" comparison harness — buildable from data this project
already has (the same predicted/observed export pattern used
throughout this session's own validation work, pointed at raw keypoint
detections instead of the tracker's own smoothed state). Treat this as
a distinct, parallel-track idea rather than folding it into the main
calibration effort: it depends on the same rigid-cluster/chain-
calibration machinery as its ground-truth source, but is otherwise a
separate analysis with a separate payoff, worth its own scoping once
§2's mechanism is built and produces joint-location ground truth
trustworthy enough to compare against.

## 6. Open questions

1. **Is 2 markers (`foot`) enough for a trustworthy rigid cluster?**
   Untested. Worth a synthetic-noise sensitivity check (how much does
   the fitted pose/geometry wobble under realistic detection noise with
   only 2 points, versus 3+) before deciding whether `foot`'s cluster
   can be trusted for the `ankle` functional-joint-center step, or
   whether a 3rd foot marker is needed in a future capture.
2. **Range-of-motion coverage for functional joint center and chain-
   calibration fits.** A squat mostly exercises sagittal-plane knee
   flexion — likely enough to condition the `ankle` sphere fit, but
   unlikely to be enough for a ball joint like the hip (needs real
   rotational diversity, not just flexion in one plane) — this applies
   to §2.5's chain calibration too, since it also relies on the chain
   being exercised through a representative range of its own joint
   angles. The productization plan's calibration-trial protocol should
   specify per-joint motion requirements, not assume one generic trial
   (e.g. squats) suffices for every joint.
3. **Interaction with the productization plan.** This design's staged
   approach (§3) should become one stage of the calibration-trial
   processing pipeline described there, not a separate workflow.
4. **Synthetic data for validation (Harri).** Rendering a known-ground-
   truth capture from animation software is a real option for testing
   and tuning both the calibration mechanism above and the bias-
   correction idea in §5 — the cost is that a synthetic skeleton/marker
   rig may not fully match real body/marker physics (soft tissue
   movement, marker slippage, real detector noise characteristics).
   Worth pursuing as a complement to real-capture validation, not a
   replacement — real capture is still needed to confirm the synthetic-
   vs-real gap isn't hiding something that matters.
