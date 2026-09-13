# Person-worn marker mocap — productization plan (draft, 2026-09-13)

## Status and scope

Companion to [marker-mocap-productization-plan.md](marker-mocap-productization-plan.md),
which covers productizing the **rigid-prop** use case (UC1 per
`marker-mocap-design.md`'s phasing). That plan's own §6 flags "a person
wearing markers to augment person tracking" as "genuinely new ground
this plan hasn't scoped at all" — this document is that scoping.

Written the day after the first real end-to-end validation of this
pipeline (`status.md`, 2026-09-12): a real capture
(`2026-09-06-kare-tests`, one person, a 16-marker `leg` module) went all
the way from raw detection through a calibrated, dot-augmented skeleton
to a validated UKF tracking run with real dot-slot predictions, and
then through a second iteration that found and fixed a real calibration
bug (`ankle_lat_L`) and a real performance bug (dot-slot prediction cost).
Every stage of that pipeline today is a **standalone script**, hardcoded
to this one capture's IDs, one skeleton topology
(`reallusion-no-waist`), one marker module (`leg`), and one person. This
plan is about generalizing a pipeline that has now been proven to work
once, not designing a new one from scratch.

Per Harri: the near-term product needs include both **multi-person**
captures and unification with the existing **prop** case, and some kind
of purpose-built **calibration trial** for onboarding a new marker
setup — this doc scopes all three.

## 1. Inventory: what today's pipeline actually is

| Stage | Script | General already? | Hardcoded to this capture |
|---|---|---|---|
| Detection | `MarkerDetectionPipeline` (production) | Yes — scene-wide, per-camera, no person concept at all | Nothing real |
| B2 cross-camera tracklet grouping | `build_tracklet_groups.py` | Algorithm yes; **never exercised with >1 person's dots present** | Nothing structural, but untested for multi-person (§3.2) |
| B3 manual review UI | `label_tracklet_groups_gui.py` | Algorithmically yes; **not production-ready** (see note below) | A standalone PySide6 window outside the main GUI app; not on the DB schema at all |
| Audit / conflict-review / reconciliation | `audit_cross_slot_consistency.py`, `prepare_conflict_review.py`, `reconcile_conflict_review.py` | Yes — general JSON I/O and geometry, only depends on the module's own slot vocabulary | Nothing beyond the module's own vocabulary (expected) |
| B4 calibration fit | `fit_calibrated_attachment_set.py` | No | `_SLOT_PARENT_BASE` is a hardcoded Python dict for the `leg` module; a second module needs its own copy of this script |
| Single-camera reprojection refit | `refit_attachment_set_single_camera.py` (2026-09-12) | No | Same `_SLOT_PARENT_BASE` duplication; also hardcodes `person_id=0` |
| Skeleton build | `build_dot_augmented_skeleton.py` | **Yes** — merges any calibrated-attachment-set-shaped YAML into any base skeleton | Nothing — a bright spot, reusable across modules unmodified |
| Finalisation (raw dots → `pose_observations`) | `copy_dot_candidates_to_sequence.py` (2026-09-12) | No — explicitly a validation-scoped stopgap (own docstring) | Routes around `finalise_object_to_db()`'s rigid-object-only scoping; needs a real production path (§2.1) |
| Tracking | `posetrak-tracker track` | **Yes** — `--skeleton`/`--tracker-config`/`--sequence`/`--person-id` are independent params, no marker-module concept baked in; `MultiPersonTracker`/`PersonSpec` already exist structurally | Never run with 2+ marker-augmented people simultaneously |

The pattern: **detection and tracking are already general**; review
tooling is algorithmically general but not product-ready; the real gap
is calibration fitting and finalisation, which grew as one-off scripts
during this session's prototyping and were never generalized because
there was only ever one module and one person to generalize *from*.

**Correction (Harri's review)**: `label_tracklet_groups_gui.py` doesn't
belong in the "already general" bucket as stated. It's a real, useful
tool for experimentation, but calling it done conflates "the algorithm
generalizes" with "this is shippable" — three separate gaps remain: it
needs integrating into the main GUI app (`python/app/ui/`) rather than
staying a standalone script outside it, its performance likely needs
work at real scale (never profiled against a large multi-module,
multi-person groups file), and its data model (JSON files passed
between scripts by filename convention) needs a real home in the DB
schema instead — today nothing about a review session (which groups,
whose decisions, when) is queryable from the database at all. Folded
into §2.1's finalisation work below, since a DB-backed review model and
a DB-backed finalisation path are really the same underlying gap.

## 2. Gaps to close

### 2.1 A real finalisation path for person-worn dots

`finalise_object_to_db()` requires a `capture_object_id` and refuses
otherwise — correct for the rigid-prop case, wrong fit for a person.
`copy_dot_candidates_to_sequence.py` is tonight's stopgap: it copies a
run's `detection_keypoints` (`region_type='dots'`) straight into an
*existing* person sequence's `pose_observations`, reusing the exact
copy shape `finalise_object_to_db()`'s own `_copy_region()` already
validated for the prop case.

Needed: promote this to a real `finalise_dots_to_person_sequence()` in
`finalise.py`, callable from the GUI/CLI the same way
`finalise_object_to_db()` already is, with the same immutability guards
(refuse to silently destroy a sequence with existing tracking results or
edits) `finalise.py`'s other functions already enforce. This is also the
prerequisite for the prop-side plan's own §3.2 (dot review/edit in the
GUI) to extend naturally to the person case instead of needing a
parallel implementation.

This is also where `label_tracklet_groups_gui.py`'s three real gaps
(§1's correction) get closed, not as a separate effort: a DB-backed
finalisation path implies a DB-backed home for groups/assignments too
(replacing today's by-convention JSON files), and once review state is
queryable, integrating the review UI itself into the main GUI app
(`python/app/ui/`) becomes a normal widget-wiring task rather than a
data-model migration. Performance at real scale should be profiled once
there is real multi-module or multi-person data to profile against —
premature before that.

### 2.2 A real marker-module registry

Today `_SLOT_PARENT_BASE` is a hand-written Python dict, duplicated
wherever a script needs it, for exactly one module (`leg`). But every
piece of that mapping already exists, declared, inside a module's own
nominal/calibrated catalog file (`parent_joint` per marker) — per
`marker-catalog-and-assignment-redesign.md`'s own proposed file shape
(§1.9). There is no reason `fit_calibrated_attachment_set.py` or
`refit_attachment_set_single_camera.py` should hardcode this mapping in
Python at all; both should derive it from the module's own catalog file
that they are already given as an input.

Once that's true, adding a **torso/arm module** (§4's actual next
target) — or a **hand module**, if markerless wrist-seeded hand pose
turns out not to be accurate enough for the pinch/grab use case (§4) —
becomes "author a new nominal catalog YAML," not "write a new Python
script." This is the single highest-leverage generalization in this
plan: everything downstream (calibration fit, refit, skeleton build)
already reads a module-shaped file; only the one hardcoded mapping
stands between "leg-only" and "any module."

### 2.3 Per-module scaling & calibration

[skeleton-scaling-and-marker-calibration-design.md](skeleton-scaling-and-marker-calibration-design.md)'s
rigid-cluster / functional-joint-center approach generalizes the same
way §2.2 does: *which* segments qualify for a rigid-cluster treatment is
a property of how many markers a module's catalog attaches to each
segment, not hardcoded per-body-region logic. A hand module with 3+
markers per finger segment (if physically feasible — see §5) would reuse
the identical mechanism without new code.

### 2.4 Multi-person scoping

Two separate problems, worth keeping apart:

- **Mechanical DB scoping** (cheap): `refit_attachment_set_single_camera.py`
  and `copy_dot_candidates_to_sequence.py` both currently assume
  `person_id=0`. Threading a real `person_id` through both is a small,
  mechanical change, matching the pattern the C++ side
  (`Tracker`/`PersonSpec`) already established.
- **Person disambiguation in scene-wide dot detection** (the real gap):
  dot detection is deliberately anonymous and scene-wide — it has no
  notion of "whose" a candidate is. With two marker-wearing people in
  frame, B2's grouping has never been exercised to check whether it
  cleanly separates their dots.

  **Harri: the solution path here is actually fairly clear** — Cutie
  segmentation already gives a reasonably reliable per-pixel person
  classification, and combining that mask with tracklets (each
  tracklet's candidates checked against which person's mask they fall
  inside, frame by frame) should give a strong disambiguation heuristic
  without needing a new algorithm. Mask errors will happen (a limb
  dropped from the mask on some frames is a known Cutie failure mode),
  but a tracklet spans many frames, so a handful of mask-miss frames
  shouldn't flip a whole tracklet's person assignment if the rest of it
  agrees.

  This is the same mechanism the *prop*-side plan already proposes for
  a different purpose (`marker-mocap-productization-plan.md` §5 point 3
  — scoring raw dot candidates against a Cutie mask as an object-init
  cueing signal) — worth building once and reusing for both: mask-vs-
  candidate scoring as a general cost-relaxation/disambiguation term
  wherever assignment needs to know whose candidate something is,
  whether that's "this object" or "this person." One real difference
  in this direction's favour: a person's whole-body silhouette is a
  much easier segmentation target than the prop plan's own flagged
  concern about a thin, fast-moving prop (a sword blade) — so mask
  quality is less likely to be the limiting factor here than it is
  there. Downgrades this item from "deepest, most novel, riskiest" to
  "a real integration task with a known-shaped solution" — still
  sequenced after single-person work is solid (§4), but with
  meaningfully lower risk than the first draft assumed.

### 2.5 Convergence with the prop case

The prop path (rigid, existing productization plan) and the person path
(articulated, this session) already correctly diverge in the C++
tracker at the *per-frame prediction* layer
(`Tracker::predict_dot_slot_predictions()` branches on
`skeleton_->is_rigid_body()` — `predict_rigid_marker()`'s closed form vs.
`UnscentedKalmanFilter::predict_marker_slots()`'s sigma-point FK), and
that split is correct: the underlying math genuinely differs for a
whole rigid object vs. an articulated chain.

But the scaling design's rigid-cluster idea (§2.1 there) means a
well-instrumented *segment* of an otherwise-articulated person is,
locally, exactly the same rigid-body geometry problem the prop
calibration path (`calibrate_rigid_marker_body.py`) already solves.
Worth a design note now, before it is discovered as a surprise later
(the existing plan's own stated concern about backward-incompatible
changes found after merge): the two paths could plausibly share more
machinery at the **calibration** layer even while staying split at the
**tracking** layer. Not building this now — flagging it so the two
plans do not silently diverge into duplicate implementations of the
same rigid-geometry-from-capture math.

The **detection** layer (`dot_blob_detector.py`, `DotTrackletLinker`) is
already fully shared and should stay that way — it has no notion of
"prop" vs. "person" today, which is a strength, not a gap. The
disambiguation work in §2.1/§2.4 belongs at grouping/assignment, never
pushed down into detection.

## 3. The calibration trial

Today, calibration happens as a side effect of reviewing an ordinary
performance capture — `2026-09-06-kare-tests` was not purpose-built for
calibration, it just happened to contain a squat-like section that
worked well enough. Manual review (B3) is the real bottleneck, echoing
the prop-side plan's own finding that dot review has no GUI workflow at
all yet (§3.2 there).

**Proposal**: a short, purpose-built calibration trial per (module,
subject) pair — subject being a person *or* a prop, reusing the same
concept for both once §2.5's convergence is real:

1. **What to attach**: derived directly from the module's own nominal
   catalog (marker names, approximate placement — the catalog schema
   already anticipates a `placement_note` field per marker).
2. **What motion to prescribe**: this session's most useful finding for
   trial design is that the **bar just got materially lower**. B4's
   original triangulation-based fit needed two cameras simultaneously
   seeing the same marker — expensive to guarantee by design and the
   reason `ankle_lat_L` was starved to 28 usable samples out of a whole
   capture. Tonight's single-camera reprojection refit needs only
   abundant *single*-camera coverage across a varied range of joint
   configurations — a far easier bar to satisfy by trial design. Where
   a joint is also getting the functional-joint-center treatment
   (scaling design §2.3), the trial additionally needs real rotational
   diversity at that joint, not just single-plane flexion — a squat
   alone is probably insufficient for a ball joint like the hip
   (flagged as an open question in that doc too).
3. **Processing pipeline** once captured: B1 detection (already fast +
   parallel per the prop-side plan's own performance work) → B2
   auto-grouping → a shorter, more targeted B3 review pass (a
   purpose-built trial should produce materially cleaner groups than an
   arbitrary performance capture) → single-camera reprojection refit as
   the *primary* calibration mechanism (B4's triangulation approach
   becomes a fallback/cross-check, not the primary path) → rigid-cluster
   / joint-center refinement wherever the module's catalog has enough
   markers per segment → register the resulting calibrated skeleton,
   tagged with `{module, subject, calibration_trial_id}` for
   provenance.
4. Once the pipeline above is validated on a second real module (§4),
   this should become a first-class CLI workflow — sketch only, not
   designed here — rather than a hand-run chain of scripts passing
   intermediate JSON files between each other by convention.

## 4. Sequencing

1. **Finalisation path** (§2.1) — promotes tonight's stopgap to
   production; unblocks a real GUI review workflow for person-worn
   dots, and should be built once, shared with the prop-side plan's own
   §3.2 rather than duplicated.
2. **Module registry** (§2.2) — needed before any second module can be
   added without duplicating scripts; cheap relative to its leverage.
3. **Calibration trial protocol + pipeline** (§3), validated against a
   **second real module — torso + arms** (Harri: this is the actual
   next module, not hands). This is the real product validation: does
   the pipeline work on a materially different body region using only a
   new catalog and a new capture, with no new script-writing?

   **Hands, separately: markerless may be enough.** Harri wants to
   explore whether a good, accurate wrist position (from the torso/arm
   module above) is sufficient to seed a markerless hand-pose estimate
   good enough for the pinch/grab use case, rather than assuming finding
   (4)'s hand-marker-infeasibility problem needs solving with markers at
   all. If true, this sidesteps the hardest physical-instrumentation
   question in this whole plan (§5, marker size vs. finger segment
   length) rather than solving it. Worth a real markerless-hand-pose
   accuracy check against ground truth once torso/arm calibration is
   solid, *before* committing to a marker-based hand module.
4. **Multi-person disambiguation** (§2.4) — deepest, most novel item;
   deliberately last, since a shaky single-person pipeline would only
   compound into a harder multi-person debugging problem.
5. **Prop/person calibration-layer convergence** (§2.5) — an
   architecture note for now; revisit for real code sharing once both
   sides have matured further.

## 5. Open questions

1. Does B2's grouping algorithm, unmodified, cleanly separate two
   simultaneously-captured people's dots, or does it need person-track-
   based pre-clustering first? Untested (§2.4).
2. Is "abundant single-camera coverage across varied configurations"
   sufficient for every future module, or do some segments (e.g. a
   twisting forearm, closely-spaced finger markers) still need genuine
   multi-camera triangulation at some frames? Untested outside `leg`.
3. Hand-module physical feasibility — marker size vs. finger segment
   length, and occlusion during a grasp — is a capture-hardware
   question for Harri, not something this doc resolves.
4. Where §2.5's shared calibration-layer code would actually live
   (a new module inside `posetrak`, vs. a shared `python/tools/`
   library both plans' scripts import) — not decided, deliberately
   deferred until there is a second real caller to design the interface
   against.
5. **Performance scaling with marker count — profiled and substantially
   fixed for the dot-specific path (`status.md`, 2026-09-13); one real
   risk remains, deliberately not started.** Real instrumentation (not
   estimates) found `predict_marker_slots()`/`predict_dot_slot_
   predictions()` sequential and called once per marker *and* once per
   camera — both fixed same day (OpenMP parallelization matching
   `update()`'s own existing pattern, then batching every (marker,
   camera) pair into one `predict_measurements()` call per sigma point).
   Combined effect on the 16-marker `leg` module: dot-slot prediction
   61.5 -> 4.1 ms/frame (**14.9x**), real throughput ~4.1 -> ~5.35 fps
   (+30%), verified with a bit-for-bit regression test, not just
   "looks reasonable."

   **The full per-frame budget is now accounted for (Harri caught a
   real arithmetic gap in an earlier version of this entry — see
   `status.md`, 2026-09-13 "closed the frame-time accounting").** On the
   same 2399-frame window, real observed time is 183.6 ms/frame
   (5.45 fps); every piece of that is now individually measured to
   within ~2%:

   | piece | ms/frame | % |
   |---|---|---|
   | `update()` (incl. Kalman-gain, ~61ms of it) | 108.1 | 58.9% |
   | `observations.get_all_in_range()` | 33.9 | 18.5% |
   | `predict()` | 11.1 | 6.0% |
   | `resolve_shared_dot_assignment()` (assignment only) | 10.9 | 6.0% |
   | `bucket_candidates_by_camera()` | 5.2 | 2.8% |
   | `predict_marker_slots()` (today's whole fix target) | 4.0 | 2.2% |
   | everything else (export/writer calls, posterior FK) | 6.8 | 3.7% |

   Two real findings from closing this out:
   1. **`update()`'s own Kalman-gain computation (~61 ms/frame) is
      confirmed the single largest piece (58.9% of the whole frame)** —
      and it barely moved when 16 dot markers were added (measured
      unchanged across all profiling passes). That's not evidence dot
      count doesn't matter here — 16 is small next to the ~366 pose
      observations (61 markers x 6 cameras) already feeding that same
      update(). At a Vicon-scale marker set (~53+24, more than doubling
      the observation count) and with Kalman-gain computation typically
      scaling worse than linearly in observation count, this could
      become a real, separate bottleneck today's 16-marker data point
      cannot predict. Materially bigger scope than today's dot-specific
      fixes (inside the shared body+hand `update()`) — needs its own
      profiling once a materially larger module (torso+arms, §4)
      exists, not extrapolation from here.
   2. **`observations.get_all_in_range()` (18.5%, 33.9 ms/frame) was a
      genuine surprise** — more than 8x today's whole dot-slot-
      prediction fix target, just to fetch this frame's real pose-
      keypoint observations before doing anything with them. Not
      dot-marker-specific at all (every tracking run calls this), so a
      fix would help every run in this project, not only marker-
      augmented ones. Smells like a linear scan or unindexed lookup
      against a large, session-wide observation collection, re-run
      every frame — flagged as a real, likely easier and higher-value
      target than anything left on the dot-prediction side, not
      investigated further this pass.
