# Hand-detection refinement & trusted-edit gate bypass — design sketch

> **Status (2026-07-14, update 7)**: The `.refined` suffix name is chosen
> (Harri picked it over `.corrected`/`.interactive`/`.auto`). The four
> remaining design-level open questions from update 6 are now resolved by
> checking the actual code rather than guessing: **editor write access to
> `pose_observations`** is safe (the DB already runs in WAL mode and
> background `QThread`s already write concurrently to `frame_cache_entries`
> today); **"select whole hand"** already exists (`"Left hand"`/`"Right
> hand"` groups in `kp_models.py`, wired into the right-click group-select
> menu); **provenance tracking through the merge** has a concrete algorithm
> now (strip the `.refined` suffix, decide per base-name which row wins,
> then apply existing per-source index placement); and the **new status
> color's scope is timeline-only** (the crop-grid canvas already has its
> own separate, unrelated color convention — `STATUS_BLUE` isn't used there
> either). Idea 3 has no remaining open design questions — what's left is
> the actual implementation.

> **Status (2026-07-14, update 6)**: Phase 2 (multi-source `pose_observations`,
> see *Idea 2*'s schema section) validated end-to-end against real session
> data — schema, batch hand-refinement pipeline, `finalise_to_db`, and C++
> tracking all confirmed working together (hand markers' median mahalanobis
> distance roughly halved vs. a body-only baseline). Along the way: fixed a
> real bug (`HandRefinementPipeline` hardcoded `device="cpu"` instead of
> using the same `_auto_device()` autodetection the whole-body pass already
> had), fixed a real bug in `finalise_to_db` (re-finalising a run whose
> sequences already had tracking/edits crashed on an unhandled FK violation
> instead of refusing cleanly — now fixed, see `python/app/pose/finalise.py`),
> and found a real gap in the real-data test workflow itself (there's no
> supported way to reuse an existing segmentation as a second detection run's
> bbox source — written up separately as
> `docs/roadmap/features/segmentation-reuse/segmentation-reuse-design.md`,
> postponed). **Idea 3's integration-level design is now finalized** (the
> crop/gate mechanism itself was already validated via Idea 2) — six
> previously-open questions (schema/provenance, reject/revert, interpolation
> integration, trigger convergence across edit operations, frame access, UI
> status color) are resolved in the *Idea 3* section below. The
> schema/provenance answer was generalized per Harri's comment into a
> **generic `<base>.refined` source-precedence convention** rather than a
> hand-specific mechanism, so future auto-detection-after-edit features
> (e.g. face landmarks) can reuse it directly. Not yet implemented.

> **Status (2026-07-12, update 5)**: Phase 1 (Idea 2, interim no-schema-
> change version) is implemented — `posetrak.detection.hand_refinement`
> (`HandRefinementPipeline`, `detect_hand_in_crop`), wired into both the
> GUI (`app/pose/main.py`, a "Refine hands" checkbox, on by default) and
> the CLI (`app/pose/cli.py run --refine-hands/--no-refine-hands`) as a
> step right after the full-body pass, before track assignment. Uses the
> exact crop/candidate-selection/gate formulas from the "Idea 2" section
> below, patches the refined 21-point hand into the existing 133-point
> `detection_keypoints` blob in place, and no-ops for 17-keypoint pose
> models. The sequential-frame decoder was pulled out of `DetectionPipeline`
> into a shared `posetrak.detection.frame_source` module so this pass can
> re-read the same frames a run's keypoints were built from. Unit-tested
> (`python/tests/app/test_hand_refinement.py`, 12 cases covering the crop
> math, nearest-candidate selection, gate accept/reject, and the DB patch
> round-trip) against a fake hand model — not yet run against a real
> trial (the before/after garbage-detection comparison and two-handed-grip
> frequency check from "Phasing" below are still open). The hand model's
> own per-keypoint confidence is scaled by a placeholder `×5.0` factor
> (`_HAND_CONF_SCALE`) to bring its 0-1 range into the same ballpark as
> the whole-body RTMPose model's raw SimCC logits before both land in the
> same blob's confidence column — reusing the project's existing
> ViTPose-conf-scale convention, not an empirically tuned value.

> **Status (2026-07-12, update 4)**: Idea 2's core mechanism — crop sizing,
> model choice, and a validation gate — is now empirically tuned against
> four rounds of offline stills (Tommi, Roosa, and Harri in a completely
> different trial with a two-handed sword grip; 60+ crops reviewed, several
> by eye). See the rewritten "Idea 2" section below for the validated
> formulas and `rtmlib.Hand` as the concrete model choice. Headline
> results: occlusion and motion blur correctly come back low-confidence
> (the core premise this whole idea was betting on, confirmed); a
> proximity gate on the hand model's *own* detected root keypoint —
> `reject if far from the tracked wrist relative to forearm length` —
> reliably catches wrong-hand/wrong-person detections without also
> rejecting good ones; and a crop offset away from the elbow (not
> centered on the wrist) fixed finger-clipping. One new, real edge case
> surfaced and not yet handled: a two-handed coordinated grip (sword,
> clasped hands) can fool the gate into rejecting the *correct* hand
> because it's checking distance to a single wrist, and the other hand's
> grip point can be closer. Phasing revised below to reflect this —
> pipeline integration (Phase 1) is now well-specified rather than a
> sketch.

> **Status (2026-07-11, update 3)**: Idea 1's revised design (scale-to-
> threshold, update 2 below) is now tested against real data (Roosa +
> Tommi) — see "(a) Scale noise to land just inside the gate threshold" in
> Idea 1's section below, and the crisis log's "Phase 0b" section for the
> full writeup. **Roosa: complete success**, on par with adaptive-off.
> **Tommi: initially a severe-looking new failure (condition number up to
> 1.7×10²⁸), traced to a genuine data error** (swapped shoulder/wrist
> keypoints in one camera for ~20 frames) rather than a mechanism
> limitation — fixed and confirmed resolved on rerun. Both real-data
> failures this mechanism has hit so far turned out to be bad data, not
> architecture, once actually traced.

> **Status (2026-07-11, update 2)**: Idea 1's revised design (scale an
> edited observation's noise, via bisection, so its mahalanobis distance
> lands exactly at `outlier_threshold` instead of an unconditional bypass)
> is now implemented too — see "(a) Scale noise to land just inside the
> gate threshold" in Idea 1's section below for the mechanism. `force_inlier`
> observations are still always kept, but no longer forced through with
> arbitrarily tight noise regardless of consequence. Superseded by update 3
> above (real-data result).

> **Status (2026-07-11, update 1)**: Idea 1's Phase 0 (hard `force_inlier`
> bypass, flat `edited_kp_noise_std`) was implemented and tested first — see
> "Phase 0 (trusted keypoint edits, Idea 1)" in
> `docs/roadmap/features/tracking-crisis-debugging-log.md`. **Result: net
> negative as configured** (Roosa's avg NIS/DOF went 1.52→28.84). Root cause
> traced to a real, legitimate edit forcing a 48σ correction through in a
> single step with tight (25px) noise, badly ill-conditioning the
> covariance — not a bad edit, but the *hard bypass + tight noise*
> combination proved too blunt for correcting a state that had already
> drifted significantly. Superseded by update 2 above. Ideas 2/3 and the
> multi-row schema discussion below are unaffected by this finding.

> **Status (2026-07-11)**: Sketch only, not implemented. Written up after the
> adaptive-process-noise on/off comparison
> (`docs/roadmap/features/tracking-crisis-debugging-log.md`, "Adaptive
> process noise (Mechanisms A+B) on/off comparison") surfaced two literal
> `(0,0)`-confidence-1.0 edited keypoints destabilizing Roosa's data, and a
> follow-up discussion of why hand/finger tracking is a recurring weak point
> (fast bilateral hand-raises still lose lock at the outlier gate; hand
> keypoints are frequently wrong from self-occlusion or identity mixup
> during grabs, and are slow/tedious to hand-edit given how many keypoints
> a hand has). Three related ideas, discussed together because two of them
> share the same underlying mechanism and the third needs a schema decision
> informed by both.

## Hygiene scan results (motivating data point, not a design item)

Read-only scan of every row in `pose_observation_edits` across the whole
trial (10,640 rows, all sequences, all three people) for edited keypoints
that are enabled (`is_outlier=False`) but sit at/near pixel `(0,0)`, are
`NaN`, or are wildly out of frame (`>8000px` or negative coordinates):

| finding | count |
|---|---|
| Near-origin (`≤3px` of `(0,0)`), enabled | **3**, all Roosa |
| `NaN`, enabled | 0 |
| Extreme-magnitude / negative, enabled | 0 |

All three: `gopro-11_mini_01` frame 5475 (`left_shoulder`) and frame 5476
(`left_shoulder`, same coordinates, same `created_at`), `gopro-11_mini_02`
frame 15225 (`right_ear`) — sequence `a5da88ea-f7ba-4e0e-bbd4-43c68205dcf6`
(Roosa). `created_at` 2026-07-03/04 — predates the marker/chain-placement/
interpolate-missing features, so not caused by that work; some earlier
edit path wrote a real coordinate for other keypoints in the same
multi-keypoint edit and a stray `(0,0)` for one that was swept in without
ever getting a real placement. Root tool/path not identified — out of
scope for this note, but worth a quick look given it's clearly reproducible
(the frame-5475/5476 pair suggests a single edit action touching both).

**Conclusion: narrow, bounded, and already fully enumerated** — three rows,
easy to fix directly, not a systemic problem across the trial. Nothing
further needed here beyond actually clearing those three rows (additive:
either re-edit them to the correct position, or set `is_outlier=True` to
disable them, both already-supported operations — no schema change).

---

## Idea 1 — skip the outlier gate for edited keypoints (Phase 0 tested, net negative — revised design below)

**Revised framing after this discussion**: this must apply to *human-placed*
edits specifically, not to anything written automatically — see the schema
note in Idea 3 below. A garbage edit like the hygiene-scan findings above
would be actively harmful under an unconditional bypass (forced into the fit
with zero mahalanobis safety net), so this also depends on the hygiene scan
staying clean (spot-check periodically, or before ever enabling this per
sequence).

**Mechanism** (traced, not yet built): `UnscentedKalmanFilter::update()`
(`ukf.cpp:2043-2074`) has two independent rejection paths — the standard
Mahalanobis threshold (`mahalanobis_distance > outlier_threshold`) and a
separate cross-camera median-consistency check (`ukf.cpp:2002-2041`,
rejects if a marker's per-camera mahalanobis is >3x the cross-camera
median). Both would need to respect a new flag. `session_reader.cpp`
(`:753-864`) already loads each edit's `kp_mask` bitmask per (camera,
frame) *before* building each `Observation` — the per-keypoint "was this
edited" bit is available at exactly the point `Observation` objects get
constructed (`:847-860`), it's just not threaded into the struct yet.

**Proposed schema/plumbing (revised — see per-keypoint correction below)**:
add a `bool force_inlier` field to `Observation`
(`include/posetrak/core/observation.hpp`), set from the edit's mask bit at
load time (per the corrected per-keypoint provenance model below). In
`update()`, when `force_inlier` is true, set `is_outlier=false`
unconditionally on both rejection paths but still compute and record
`mahalanobis_distance` for diagnostics.

**Correction, superseded by the multi-row design below**: an earlier
revision of this doc proposed a second `auto_mask` bitmask on
`pose_observation_edits` to separate human from automated provenance
within one edit row. That's no longer needed — once Idea 3's automated
writes target `pose_observations` (as their own `source='hand.L'/'hand.R'`
rows, see *Measurement noise for edited and automated observations* below)
instead of `pose_observation_edits`, **`pose_observation_edits` stays
unambiguously human-only**, exactly as it is today. `force_inlier` is
simply: bit set in `kp_mask`, full stop — no second bitmask, no schema
change to the edits table at all.

**Implemented and tested as Phase 0** (`edited_kp_noise_std` config field,
schema v34, `force_inlier` field on `Observation`) — see "Phase 0 (trusted
keypoint edits, Idea 1)" in the crisis log for the full writeup. **Result:
net negative as configured.** Traced concretely: a real, legitimate,
internally-consistent human edit (not garbage) got force-included with
tight 25px noise despite the filter's predicted state having already
drifted ~1275px / 48σ away by that point — the mechanism doesn't
distinguish "gently nudge the filter back on track" from "yank the state
by 48σ in one step," and the latter badly ill-conditioned the covariance
(the same "large sudden correction breaks local linearity" pathology
already documented for the frame-227/228 event and Proposal 1). This
wasn't one unlucky edit either — Roosa's degradation persists (though
smaller) even excluding the worst 50 steps, and with 6,672 edit rows in
her sequence there's no reason to expect this was the only case.

**The mechanism (unconditional force-include) may still be sound in
principle** — the traced case really was a correct edit fixing a real
problem, just applied too abruptly. Two follow-up ideas, both raised in
response to this finding, not yet built:

**(a) Scale noise to land just inside the gate threshold — implemented,
exactly, not the approximation originally sketched here (2026-07-11).**
Instead of a fixed `noise_std_override` (or a hard `force_inlier` bypass),
`reject_outliers()` now computes a *per-observation* noise value such that
the resulting mahalanobis distance comes out at exactly `outlier_threshold`
whenever the raw (unscaled) distance would exceed it — capping how much any
single edited observation can pull the state in one step, regardless of
how far the raw pixel gap is, while still guaranteeing it's never rejected.
Below threshold: included unchanged, no scaling at all.

**Turned out to be exact, not approximate, and cheap either way.** The
original plan here assumed solving for the `R` that lands mahalanobis
exactly on threshold wasn't closed-form since `R` sits inside
`S = H·P·H^T + R`. True, but `reject_outliers()` already has `cov_2x2`
(observation *i*'s own 2×2 diagonal block of the already-built `S`) and the
noise variance that went into it (recoverable via
`Observation::measurement_noise_std()`, same `pose_noise_std`/
`calib_noise_std` `update()` used to build `S` in the first place — now
threaded into `reject_outliers()` as two extra parameters for exactly this).
Subtracting that variance back out recovers `H·P·H^T`'s local block (the
*R-free*, purely-geometric part), PSD-clamped via
`Eigen::SelfAdjointEigenSolver` against floating-point noise. From there,
mahalanobis distance as a function of a candidate noise multiplier `k` is
one cheap 2×2 matrix inverse (`compute_mahalanobis_distance`, already
existing) — monotonically decreasing in `k`, so a plain bisection (exponential
bracket-expansion, then ~40 bisection iterations, all on 2×2 matrices)
converges to the exact `k` in a handful of microseconds. The resulting
`noise_std_override` — carefully un-doing the `/max(confidence, 0.1)`
`measurement_noise_std()` will re-apply downstream, since `noise_std_override`
is defined as the *pre*-division raw value — is written onto a **copy** of
the observation before it's pushed into `inliers`, which is exactly what
`update()`'s existing "recompute predictions/covariance for inliers only"
step (right after `reject_outliers()` returns) reads back from — no changes
needed anywhere else in `update()`'s structure at all, confirming the
original "doesn't require touching the two-pass structure" intuition, just
via a different (and now exact) mechanism than first planned.

**Base noise revised upward.** Phase 0's tested value (`edited_kp_noise_std
= 25.0`, exactly `calib_noise_std`) was flagged as likely too tight —
calibration error isn't the only source of uncertainty in a human-placed
point, and using it alone probably underweights the additional imprecision
of a human click. Revised interim value for the next test:
`sqrt(pose_noise_std² + calib_noise_std²)` ≈ 28.18px (13.0 and 25.0 in the
configs used) — reusing the codebase's existing quadrature-combination
convention for independent error sources (matches how RELATIVE-mode noise
is already computed as `pose_noise_std·√2·crop_scale` elsewhere) rather
than inventing a new dedicated "click precision" config field without any
empirical basis for its magnitude. Still explicitly a placeholder, not a
considered answer — same caveat as before. Note this base value now mostly
only matters for the *not-capped* case (an edit that's already within
threshold) — for a capped correction, the converged result is `threshold`
regardless of where the bisection started, so the base value barely
affects the failure mode Phase 0 actually hit.

New tests: `UKF force_inlier is kept but its noise is scaled to land at
threshold` (includes a regression guard — `root_position().norm() < 2.0`
after a would-be-wild correction, guarding against ever silently
regressing back to unconditional-bypass-shaped behaviour) and `UKF
force_inlier within threshold is included with its noise unchanged`
(`tests/test_ukf_update.cpp`). Full `[ukf]` suite passes.

**Tested against real data (Roosa + Tommi), full writeup in the crisis
log's "Phase 0b" section.** Roosa: complete success, every metric on or
fractionally better than the adaptive-off baseline — exactly fixed the
single-point 48σ case Phase 0 failed on. Tommi: a second, worse-looking
failure at first (covariance condition number up to 1.7×10²⁸), traced not
to a mechanism limitation but to a genuine data error — swapped
shoulder/wrist keypoints in one camera for ~20 frames, found by the user
reviewing the footage directly. Fixed and confirmed resolved on rerun
(back in line with the adaptive-off baseline). Both failures this
mechanism has hit so far turned out to be data problems, not architectural
ones, once actually traced — same pattern as Crisis B. Still off by
default; not yet tested on Timo or against the full trial-wide
adaptive-on/off comparison this whole investigation arc started from.

**(b) Surface which edits the gate rejects.** Right now there's no way to
tell, without manually cross-referencing `pose_observation_edits` against
`tracking_obs_results.obs_blob`'s `is_outlier` flag, that an edit a human
carefully placed got rejected by the tracker and had zero effect on the
result — exactly the blind spot that made this Phase 0 investigation
necessary in the first place (the corrupted-edit hygiene scan and the
48σ-edit trace above both required ad hoc one-off scripts). Two places
this could surface, not mutually exclusive:
- **MCP diagnostic tool**: a sibling to the existing `get_edit_coverage`
  (which already flags unedited key landmarks) — report, per edited
  keypoint/camera/frame, whether the corresponding `tracking_obs_results`
  row for the most recent run marked it `is_outlier=true`. Cheap: both
  tables already exist, this is a join, not new instrumentation.
  Read-only, fits the existing MCP server's scope directly.
- **Interactive editor UI**: visually distinguish an edited keypoint's dot
  (in `_ImageCanvas`/`PersonCropGridWidget`) when the *current* tracking
  run's `obs_blob` shows it as rejected — e.g. a distinct outline color,
  reusing the existing grey-for-rejected convention already used for
  unedited outliers (`content_panels.py`'s paint code, `QColor(120, 120,
  120)` for tracker-rejected non-edit-mode markers) but visually distinct
  from that, so "the tracker didn't listen to your edit" is obvious at a
  glance rather than requiring a diagnostic tool. Needs the crop-grid view
  to have the relevant tracking run's `obs_blob` loaded, which it doesn't
  today (it currently only shows raw/edited observations, not per-run
  gate outcomes) — a real, if bounded, addition.

Neither built yet — flagging both as valuable next steps once (a) has
something worth surfacing.

---

## Idea 2 — hand-specific detection pass in the original pipeline

Run the existing full-body detector (VitPose/RTMPose, COCO-133) first as
today, then for each frame/camera with a confident wrist detection, crop
around the estimated hand location and run a dedicated hand model on that
crop. If the hand is actually occluded by the body, a hand-specific
detector — trained on hand crops, not whole-body context — should be much
better calibrated to report low/no confidence than a whole-body model
guessing from pose priors (the calibration problem discussed for the
"tighter confidence gate" idea, which this sidesteps rather than solves
directly).

**Confirmed empirically, not just a hypothesis anymore (2026-07-12).** Four
rounds of offline stills against `rtmlib.Hand` (below) across Tommi, Roosa,
and Harri (a different trial, two-handed sword grip): occlusion and motion
blur consistently come back with low enough confidence that nothing gets
drawn/accepted, while clearly-visible hands detect cleanly. This was the
central bet the whole idea rested on and it held up across genuinely
different footage, not just the original motivating case.

### Model, crop, and validation — now a concrete, tuned mechanism

**Model: `rtmlib.Hand`**, already available via the `rtmlib` dependency the
whole-body pass already uses — no new package. It's a two-stage pipeline
internally (an `RTMDet` hand-region detector, then an `RTMPose` 21-keypoint
hand model, MMPose hand21 keypoint order — index 0 is the hand root/wrist,
matching this project's own `left_hand_root`/`right_hand_root` naming in
`kp_models.py`). Both checkpoints download once via `rtmlib`'s own cache
(`~/.cache/rtmlib/hub/checkpoints`, ~53MB total), same mechanism already in
use for the whole-body model.

**Crop, tuned across three rounds**:
- Half-width: `max(0.9 × forearm_len_px, 60px)`, `forearm_len_px = |wrist -
  elbow|` in the current tracked/edited position for that camera/frame.
  *Round 1's first attempt used a 150px floor that dominated every single
  case regardless of actual arm scale — an oversized, non-adaptive crop is
  exactly what let a neighbour's hand into frame in a grappling scene.*
  Lowering the floor and the multiplier made the crop genuinely scale-
  adaptive (round 2: 120-191px depending on actual limb size in that
  camera view, vs. a flat 300px before).
- Centre offset: shift `0.35 × forearm_len_px` from the wrist *away from
  the elbow* (along the elbow→wrist direction, continued past the wrist),
  rather than centering exactly on the wrist joint. Centering on the wrist
  wastes roughly half the box on forearm/sleeve where there's never a
  finger — round 2 clipped fingers in a few cases for exactly this reason;
  round 3's offset fixed it (spot-checked: full hand in frame, well
  centered, no clipping in the case reviewed).
- Fallback when the elbow isn't confidently known: no offset (crop
  centres on the wrist alone), `forearm_len_px` treated as 0, so the half-
  width floor (60px) applies directly, giving a small 120×120px box. This
  is the current answer to *Open questions* #3 below — a simple fallback,
  not a tuned one; hasn't come up often enough in testing to know if it
  needs more care.

**Validation gate, new — not in earlier revisions of this doc**: `Hand()`
can return multiple candidate hands per crop (observed 1-3 in testing).
Pick the candidate whose own detected root keypoint (hand21 index 0,
mapped back into full-frame coordinates) is *closest* to the tracked
wrist — not confidence, not detection order. Then reject that best
candidate anyway if the distance still exceeds `max(0.5 × forearm_len_px,
40px)`. This is the direct, tuned answer to the "surface which edits/
detections to trust" problem: in testing, 16 of 18 (round 3) and 22 of 28
(round 4, different trial) candidates passed, and every rejection spot-
checked had either no clearly visible hand in the crop or a legitimate
wrong-hand pick — no case found where a good detection got wrongly
rejected. Passing cases often landed within a few pixels of the tracked
wrist (one case: 0.4px), well inside the threshold, not borderline passes.

**New, real edge case, not yet handled**: a two-handed coordinated grip
(sword/bokken, held hands) can fool this gate — both hands are close
together by design, so the detector can find the *other* hand's grip
point, which may still be closer to *a* threshold than 40px but attributed
to the wrong wrist, or in the observed case, simply farther than the gate
allows even though the detection itself might be a legitimate hand (round
4: two rejects on a sword grip, both at a grip point plausibly belonging
to the other hand). This is a different failure shape from the identity-
mixup case the gate was built for — same person, ambiguous between their
*own* two hands — and isn't solved by tightening the crop (the grip is
often *inside* both crops if the hands are close). Not designed further
here; flagged as a known gap.

**Noise-model win falls out for free.** `Observation::measurement_noise_std()`
(`observation.hpp:49-53`) is `(pose_noise_std * crop_scale + calib_noise_std)
/ max(confidence, 0.1)`, where `crop_scale = bbox_width / pose_input_width`
comes straight from the detection pipeline per-observation. A tight
hand-only crop has a much smaller `bbox_width` than a whole-body crop for
the same `pose_input_width`, so `crop_scale` — and therefore the effective
measurement noise — drops automatically, with **no new field or tracker
change needed**. The only requirement is that the pipeline sets `crop_scale`
correctly per-observation (from the hand crop's own dimensions, not
inherited from the whole-body crop) when it writes these rows, which is
already the existing per-observation contract.

**Schema check (traced, answers Harri's question above): today's schema does
not support two models' results side by side, and merging them loses
precision.** `pose_observations`' primary key is `(sequence_id,
camera_instance_id, video_frame, person_id)` — **one row, one `kp_blob`,
one `noise_scale` per frame**, tied to a single `pose_model` /
`detection_run_id` on the parent sequence
(`pose_observation_sequences.detection_run_id` is a single FK). There is no
existing mechanism for "this sequence has both a COCO-133 pass and a
hand-model pass" as two coexisting result sets.

**Revised direction (per discussion): multi-row observations, not a
`noise_scale_blob`.** A per-keypoint noise blob would fix the immediate
symptom but not the underlying limitation — the schema still has no way to
represent "this frame has results from two different models" as a first-
class thing, which is exactly what's needed anyway once **marker
detections** (a distinct future capability, not designed further here) get
added: a fiducial/motion-capture-marker detector produces an entirely
different keypoint vocabulary from a human-pose model, has its own model
name, its own count, its own noise characteristics — that cannot be merged
into one `kp_blob` at all, blob-of-noise or not. **Better primitive: let
`pose_observations` hold multiple rows per (sequence, camera, frame,
person)**, one per detection source, and move the merge from
*write time* (today's plan: the pipeline overwrites indices in one shared
blob) to ***load* time** (the tracker's loader merges whatever rows exist
into the skeleton's full marker-index space before edits get applied on
top — same place `pose_observation_edits` already gets merged in today,
just one more merge step ahead of it).

Sketch: add a `source` column (`'body'` default / `'hand.L'` / `'hand.R'` /
future `'markers'`, ...) to the primary key —
`(sequence_id, camera_instance_id, video_frame, person_id, source)` — and
move `detection_run_id` from `pose_observation_sequences` (one per
sequence today) down onto each `pose_observations` row, since the marker
case genuinely needs a *different* detection run's output to coexist with
the body pass's, not just a different `source` label under the same run.
The hand-refinement case (this doc's actual topic) uses one
`detection_run_id` with two `source` values (`'body'`, `'hand.L'`/`'hand.R'`),
written together by one coordinated pipeline execution — markers would use
a second, independent `detection_run_id` with its own `source`, run
separately (possibly much later, on already-tracked footage) — the same
mechanism covers both without a special case.

**Forward reference**: Idea 3 below generalizes this further — an
auto-detection-after-edit feature (hand redetection here, conceivably face-
landmark refinement or similar later) doesn't need its own bespoke
`source` value and merge special-case. See *Source tiering* in Idea 3 for
the generic `<base>.refined` precedence convention this `source` column ends
up supporting.

**Backward compatibility**: existing rows all become `source='body'` by
migration default — reads stay correct unchanged. The real cost is the
primary key itself: SQLite can't alter a `PRIMARY KEY` in place, so this is
a rebuild-the-table migration (create new, copy, swap, drop old), not a
simple additive `ALTER TABLE ADD COLUMN` like every other migration in this
project's history so far (schema v26/v33 etc.) — flagging that cost
honestly since it's a genuinely bigger lift than anything else in this doc.

**Tracker-loader implications**: `session_reader.cpp`'s observation query
(`:786-791`) currently assumes exactly one row per (camera, frame, person)
and one `coco_to_marker_idx` mapping built once from the sequence's single
`pose_model`. Under this design it would need to: iterate *all* rows for a
(camera, frame, person) regardless of `source`; decode each row against
*its own* keypoint vocabulary/mapping (a `'hand.L'` row is 21 RTMPose
hand-model indices, not a slice of the 133-point body layout — needs its
own registered index→marker mapping, not built from `pose_model` alone
anymore); and merge all of them into the unified per-marker array before
edits apply. Real work, not a one-line change — sizing this properly is
future work, not resolved here.

**"No good hand detection"** still holds regardless of which schema
direction: write nothing for that `source`/frame — the merge step just has
one fewer row to fold in, same sparse-observation tolerance as everywhere
else.

---

## Measurement noise for edited and automated observations

Two distinct questions, both real, neither resolved before now:

**(a) Manual edits.** Today an edited point silently inherits the noise
formula and `crop_scale` of whatever `pose_observations` row it overrides
— i.e. a human's click is currently modeled as if it had the same error
characteristics as the original neural-net detection it replaced, which
isn't right. But **the correct value isn't obvious either** — a human
placing a keypoint is not zero-error, and there's no principled number to
reach for without empirical data. `Observation` already has the field this
would use, **`noise_std_override`** (`observation.hpp:43,49-51` — "when >
0, replaces computed noise for this observation," currently unused for
edits) — the open question is what to put in it, not how to plumb it.

**Interim default (needs real tuning later, not a considered answer)**:
reuse the run's existing calibration-error baseline (`calib_noise_std` —
the same value every raw detection's noise already includes, e.g. 25px in
the configs used for the on/off comparison) **unscaled** — i.e. skip the
`pose_noise_std * crop_scale` term entirely (that term models the pose
model's crop-relative regression error, which doesn't apply to a human
click) and use `calib_noise_std` alone as `noise_std_override`. This is a
placeholder, explicitly not a resolved design: it doesn't make edited
points especially "trusted" relative to a normal good detection, just puts
them on equal footing with one, and doesn't touch the actual gate-rejection
problem this idea originally targeted (correct edits arriving after the
state has drifted, discussed in the crisis log) — Idea 1's `force_inlier`
question stays live and separate from this.

**Tested**: 25px (`calib_noise_std`'s value in the configs used) as
`noise_std_override`, paired with `force_inlier`. Net negative — see Idea
1's section above and the crisis log's "Phase 0" writeup. The specific
25px number isn't obviously implicated on its own (the failure mode was a
single-step 48σ correction being forced through regardless of noise value,
which a hard bypass would do at *any* fixed noise level) — Idea 1's
revised "scale to just inside threshold" design is the more promising
next attempt, not a different constant for this same fixed-override
approach.

**The real fix, not built now**: let a human express their own confidence
per edit (a displayed precision estimate, or a coarse
confident/uncertain toggle) and feed *that* into `noise_std_override`
instead of one global constant — previously discussed but not written down
anywhere until now. Worth its own follow-up once there's a concrete UI
shape for it; the plumbing (`noise_std_override` already existing, this
doc's per-keypoint load-time hook) is the same either way.

**(b) Automated finger detections.** No new formula needed — this is
exactly what the existing `crop_scale`-based formula already computes
correctly, *once it can vary per source row* (the multi-row design above).
The hand-model row's own `noise_scale` = its own crop's `bbox_width /
pose_input_width`, same formula, same code path as any other detection —
this question is really the same one as the schema question above wearing
a different hat, and gets resolved together with it.

**Where this lives**: `python/app/pose` (the detection pipeline, per
CLAUDE.md's app boundary — this is squarely `PoseExtractionWindow` territory,
not the interactive editor). A new stage after the existing full-body
detection pass, before track assignment/finalization.

---

## Idea 3 — automated hand redetection after a manual edit

**Integration-level design finalized 2026-07-14** (crop/gate math below was
already validated via Idea 2's offline-stills testing; this round resolved
how the mechanism actually plugs into the editor — schema/provenance,
reject/revert, interpolation, trigger convergence, frame access, UI
surfacing). Not yet implemented.

**Trigger mechanism (revised — the original "trigger on editing wrist" was
underspecified, per Harri's questions; anchor set simplified below now that
testing has shown wrist+elbow alone are sufficient)**:

- **Debounced, not synchronous.** Fire on a short idle timer (e.g. 500ms-1s
  of no further edits to `wrist`/`elbow` on that specific camera/frame)
  rather than on every single edit event. Covers both the chain-placement
  case (wrist is placed a few keypoints into an ordered chain — firing
  immediately on it would run before a settled position is known) and a
  standalone nudge (arrow-key taps each write to the DB individually today
  — firing per-keystroke would be both wasteful and visually flickery).
  Anchor set used at fire time: current wrist + elbow position (edited if
  available, tracked otherwise), same as Idea 2's crop — no separate
  index_1/pinky_1 anchors needed (superseded, see *Shared code with Idea 2*
  below).
- **Never overwrite a human-placed finger keypoint — resolved for free by
  where the write lands, not by a mask bit.** Automated results write to
  `pose_observations` (a new `source='hand_l.refined'`/`'hand_r.refined'` row —
  see *Source tiering* below, supersedes the original `'hand.L'`/`'hand.R'`
  naming sketch), never to `pose_observation_edits`. Since the loader always
  applies `pose_observation_edits` *last*, on top of every merged
  `pose_observations` row regardless of source (today's existing merge
  order, unchanged), a human edit on any finger index automatically wins
  over whatever the automated row says for that same index — no provenance
  bit, no overwrite-detection logic needed at all. Simpler than the earlier
  `auto_mask` proposal, and gets the safety property for free from the
  existing precedence rule.
- **Interpolated values inherit whatever the wrist/anchor edit's own
  provenance is** — since they're written via the normal
  `pose_observation_edits` path (`update_single_keypoint_edit`, same as any
  edit), they carry the same "always wins over automated" property as a
  real click, with no special-casing needed. Whether an interpolated value
  *should* have the same trust level as an actual placement is a fair
  question but orthogonal to Idea 3 specifically now — it's a property of
  the edits layer in general, not something this feature needs to solve.

Once fired: recompute the crop from the current anchors, run the shared
detect+validate function (see below).

**Confirmed design decision (per discussion), refined**: results are
written as their own `pose_observations` row(s) and go through the *same*
outlier gate as any other detector output via the normal merge-then-gate
path — not through Idea 1's trusted-edit bypass, which only ever sees
genuine `pose_observation_edits` rows. This is the right call: an automated
redetection, however good the geometric-consistency check, is still a
machine guess, not a human verification — treating it identically to a
manual edit would undermine the whole point of keeping the gate meaningful
for edited data. It also means Idea 3's writes get correctly-scaled
measurement noise automatically (their own `crop_scale`, per the multi-row
mechanism), rather than inheriting the frame's original whole-body noise
level.

**Source tiering, resolved 2026-07-14 — no manufactured `detection_run_id`,
no schema change, and generalized beyond hand detection (revised per
Harri's comment).** The original sketch above ("own `detection_run_id` — an
'interactive redetection' run") turned out not to scale: a real editing
session produces thousands of these writes (this trial's sequences alone
carry 3,946 and 6,673 manual edit rows), so minting/managing a dedicated
`detection_runs` row for this would need its own lifecycle question (create
once per sequence? per app session? reused across days?) for no real
benefit.

Resolved with a **generic convention, not a hand-specific one**: any source
name of the form **`<base>.refined`** takes precedence over its corresponding
`<base>` source, per marker slot, for the same
`(sequence_id, camera_instance_id, video_frame, person_id)`. Hand
redetection is the first feature to use it (`hand_l` → `hand_l.refined`,
`hand_r` → `hand_r.refined`), but the rule itself doesn't know anything about
hands — a later "detect additional face points from a few manually placed
anchors" feature (the kind of case that prompted this generalization) would
just introduce its own `<face-source>.refined` pair and get the same
precedence behavior for free, no new merge special-case anywhere. Merge at
load time becomes: apply each base source, then apply its `.refined`
counterpart on top (per marker slot, wherever the `.refined` row has a
present/nonzero-confidence value) if one exists; `pose_observation_edits`
still applies last, on top of everything, unchanged:

```
<base>  →  <base>.refined (if present, per-slot override)  →  pose_observation_edits (human, always wins)
```

concretely, for hand detection today:

```
body  →  hand_l / hand_r (Idea 2, batch)  →  hand_l.refined / hand_r.refined (Idea 3, interactive)  →  pose_observation_edits (human, always wins)
```

No schema change needed either way: `source` is already a free-text column
inside `pose_observations`' primary key
(`sequence_id, camera_instance_id, video_frame, person_id, source`), so a
`.refined` row for a given frame is a different PK row from its base row — it
can never overwrite it, exactly mirroring how `pose_observation_edits` never
overwrites `pose_observations`. `detection_run_id` on `.refined` rows: left
`NULL` (the column is nullable: `db/session_schema.sql:197`) — sidesteps
the lifecycle question entirely, generically, for any future feature that
uses this convention.

**Naming: resolved 2026-07-14 — `.refined`.** Considered against
`.corrected`/`.interactive`/`.auto`; Harri picked `.refined`.

Real, bounded cost this generic rule does add: `session_reader.cpp`'s merge
(currently two tiers — body, then hand_l/hand_r) needs a generic
"apply `.refined` override if present" pass added (not hand-specific code —
implement it once against the naming convention, not per feature), and the
Python read-path (`posetrak.db.observation_merge.merge_observation_sources`,
used by `read_timeline_status` among others) needs the same generic pass for
UI display to agree with what the tracker actually sees.

**Reject / revert / further-edit, resolved 2026-07-14 — mostly
already-existing machinery, one new op.**
- **"Revert to the original detection"**: `DELETE FROM pose_observations
  WHERE sequence_id=? AND camera_instance_id=? AND video_frame=? AND
  person_id=? AND source IN ('hand_l.refined','hand_r.refined')` for the relevant
  side — the merge then naturally falls back to whatever `hand_l`/`hand_r`
  (batch) row exists for that slot, or nothing. Needs one small new DB
  helper (symmetric to `update_single_keypoint_edit`) and one new UI action
  (e.g. a "Revert hand redetection" context-menu item).
- **"Disable hand keypoints entirely"**: no new mechanism at all, confirmed
  2026-07-14 — `_set_outlier_selected` (`content_panels.py:4149`, the
  existing "Disable selected" context-menu action) already marks selected
  keypoints as outlier via `pose_observation_edits`, which already wins over
  every source tier including `hand_l.refined`. The "select whole hand"
  convenience already exists too: the right-click group-selection menu
  (`content_panels.py:3610-3654`) is populated from `PoseModel.group_names`,
  and `kp_models.py` already defines `"Left hand"`/`"Right hand"` groups
  covering the full 21-keypoint range each (`_LEFT_HAND_IDX`/`_RIGHT_HAND_IDX`,
  `kp_models.py:188-189, 207-208`). "Reject = disable hand entirely" is
  wiring, not new UI.
- **"Further edit a redetected finger"**: already works, unchanged — edits
  overlay on top of whatever's currently displayed regardless of which
  source tier produced it.

**Interpolation integration, resolved 2026-07-14 — maps directly onto
`_interpolate_range`/`_interpolate_missing_range` as they exist today, not a
new mode.** Both functions (`content_panels.py:4003` and `:4192`) already
loop `for kp_idx in self._sel_kp_indices` — i.e. **"interpolate only the
keypoints the user selected" is exactly their existing behavior.** If the
user selects only wrist (and maybe index1/pinky1) and presses "I", only
those indices get geometrically interpolated; every other finger index is
simply left untouched by these functions today, with no code change needed
to get that part of the behavior. The new piece: after either function
finishes writing edits for a range, if wrist (or wrist+elbow) was among the
touched indices for a hand side, queue one hand-redetect request per frame
in that range using the just-interpolated wrist/elbow position as anchor —
filling in whichever finger indices the user *didn't* select via detection
rather than geometric interpolation. This resolves the open question of
"interpolate hand keypoints or redetect them" in favor of redetection for
anything not explicitly selected, matching the intuition that finger
articulation doesn't move linearly the way a geometric interpolation would
assume.

**One converged trigger point across every editing operation, resolved
2026-07-14.** Every single-keypoint write in the editor funnels through one
function, `update_single_keypoint_edit` — confirmed 7 call sites in
`content_panels.py` (nudge, drag, chain-placement, toggle-outlier,
interpolate, interpolate-missing, and one more). Rather than duplicating
hand-redetect-trigger logic at each site, add **one** convergence point:
after any editing operation finishes, scan whichever `(frame, kp_idx)` pairs
it just touched for wrist/elbow indices, and enqueue one hand-redetect
request per `(camera, frame)` that changed. Keying the debounce by
`(camera, frame)` rather than "the operation that fired" makes single-edit
debouncing and interpolation-batch firing *the same mechanism* for free: a
single nudge produces one debounced key (coalescing rapid arrow-key taps as
originally intended); a big interpolation produces many independent keys
that don't interfere with each other's timers. No separate batch-mode logic
needed.

**Frame access mechanism, resolved 2026-07-14 — union of two patterns that
already exist in this codebase, not a new one.** A single post-edit trigger
needs one frame — the same shape as `CropBackfillWorker.prioritise()`'s
single-seek path. An interpolation-fill needs potentially many frames across
one contiguous range — per-frame seeking in compressed video is expensive,
so this needs sequential decode over the span instead, the same shape as
`WideCropExtractWorker`'s epoch-based sequential walk
(`wide_crop_cache.py`). Recommendation: a new worker that reuses both access
shapes (a prioritized single-frame path plus a sequential-range walk) rather
than literally merging into `WideCropExtractWorker` itself — that worker's
job is proactively caching crops for an entire run in the background; this
one fires reactively only on edited/interpolated spans. Same architectural
pattern and the same shared `detect_hand_in_crop` core function, not the
same worker instance.

**UI: new timeline status color for "value came from automated hand
redetection (`.refined`), not yet human-verified", resolved 2026-07-14.**
The keypoint-editing
timeline already has exactly this kind of per-keypoint status signal
(`python/app/pose/timeline_status.py`): `STATUS_GREEN` (original detection),
`STATUS_YELLOW` (original detection, outside segmentation boundary),
`STATUS_BLUE` (edited/kept as keyframe), `STATUS_GREY` (disabled/no data) —
rendered via a single `_STATUS_COLORS` dict
(`keypoint_timeline_widget.py:90-95`), already exactly the "easily
adjustable, one place" pattern wanted for this. **Note: `STATUS_YELLOW` is
already taken** for the segmentation-boundary-quality signal — reusing
yellow for "auto-redetected" would collide with an existing, different
meaning, so this needs its own code and color. Proposed: a new
`STATUS_ORANGE` code, value `2` (renumbering today's `STATUS_BLUE: 2 → 3`
and `STATUS_GREY: 3 → 4` — safe, since these codes are computed fresh from
the DB on every render and never persisted), sitting between YELLOW and
BLUE in the "ascending precedence, max code wins" aggregation scheme used
when collapsing several keypoints/cameras into one displayed cell
(`keypoint_timeline_widget.py:493-521`, an actual `max()` over these
integers — an auto-redetected value is worth surfacing but still ranks
below an actual human edit). Placeholder color: `QColor(220, 140, 50)`
(orange) — an engineering color, trivially changed in the one dict it's
defined in. Determining this status per-keypoint requires `read_timeline_status`
(and the underlying `merge_observation_sources`) to track *which source*
contributed each merged marker's final value, not just its coordinates —
today that provenance is discarded once merged.

**Scope, resolved 2026-07-14 — timeline only, not the crop-grid canvas.**
Checked `_ImageCanvas`'s dot-painting code (`content_panels.py:2341-2369`):
it already uses its own separate, hardcoded palette (grey/green/yellow/red,
keyed off *tracker* outlier status) — `STATUS_BLUE` (the timeline's edit
color) is never used there at all, only for logic (`content_panels.py:4064,
4514`, checking whether a keypoint is a keyframe for interpolation
anchoring). So there's no existing precedent of the timeline's edit-status
colors appearing in the crop view either — adding the new color there would
mean designing a second, independent convention, not extending this one.
Out of scope for now; timeline-only.

**Provenance mechanism, made concrete 2026-07-14.** `observation_merge.py`
today conflates two separate concerns: (a) placing each source's own
vocabulary into the right final-marker index range (necessarily
per-source — e.g. `hand_l` occupies indices 91-111 — a future face-
refinement source would need its own such mapping regardless of `.refined`),
and (b) precedence between a base source and its `.refined` variant. Untangle
them: strip a trailing `.refined` suffix to get the base name, decide per
base-name which row wins (prefer `.refined` if present), *then* apply the
existing per-source index-placement logic keyed by the base name — this
makes "prefer `.refined`" fully generic while keeping the necessarily-
per-source vocabulary mapping separate. `merge_observation_sources`'s
existing signature and its other caller (`db_cache.py`'s
`read_observations_with_edits`, which only wants merged coordinates) stay
unchanged; add one new, small companion function used only by
`read_timeline_status` that additionally returns, per merged marker index,
which source name won — `read_timeline_status` then colors a slot
`STATUS_ORANGE` when that winning source name ends in `.refined`.

**Validation before writing — now the validated proximity gate from Idea 2,
not the original vague "index_1/pinky_1 tolerance" sketch.** Pick the
candidate hand (of possibly several `Hand()` returns) whose own root
keypoint lands closest to the wrist anchor; reject it if that distance
still exceeds `max(0.5 × forearm_len_px, 40px)`. Below that: write nothing
for that camera/frame (matches the existing tolerance for sparse
per-camera observations — no special-casing needed elsewhere). This is a
stricter, cheaper check than the original plan — it needs only wrist +
elbow (already the trigger's anchor set above), not a separate
index_1/pinky_1 tolerance comparison, since the hand model's own root
keypoint already gives a direct identity signal the original proposal was
reaching for indirectly. Confidence (the hand model's own per-keypoint
score) is still worth surfacing alongside the write, but isn't part of the
gate itself — the proximity check alone caught every bad case found in
testing (wrong person, wrong hand) without also rejecting good ones.
**Known gap carried over from Idea 2**: a two-handed coordinated grip can
make the gate reject a correct hand because the *other* hand's grip point
is closer to the wrist anchor than the true hand is — Idea 3 inherits this
limitation unchanged; a rejection there means "wrote nothing," same
graceful degradation as any other reject, just possibly a missed
opportunity rather than a wrong write.

**Where this lives / how it's triggered**: the codebase already has a
precedent for exactly this shape of work — a background worker started
from the editor that runs heavier processing and writes results back
(`CropBackfillWorker`, `FrameCropCacheManager` in `content_panels.py`,
`_start_backfill`/`_start_wide_crop_cache`). A new worker following that
same pattern (armed by the debounce/convergence point above, running the
hand crop+detect step, writing the result as a `pose_observations` row with
`source='hand_l.refined'`/`'hand_r.refined'`, then triggering `_load_frame` to
refresh) is the natural fit rather than inventing a new async mechanism —
see *Frame access mechanism* above for how it should actually decode frames.
Writing to `pose_observations` from the interactive editor (today
exclusively a pipeline-write table) is a small boundary shift worth naming
explicitly — the editor process would need write access to that table,
which it doesn't exercise today (it only ever writes
`pose_observation_edits`). **Checked and resolved 2026-07-14: safe, given
existing precedent.** The DB already runs `PRAGMA journal_mode = WAL`
(`posetrak/db/db.py:137`), and `CropBackfillWorker`/`WideCropExtractWorker`
already open their own SQLite connections from background `QThread`s within
the editor process and write concurrently to `frame_cache_entries` — the
"multiple writers from one process" pattern is already proven here, not a
new risk category. The only actual change is which table gets a new writer,
not a new kind of technical risk.

**Shared code with Idea 2, signature now grounded in the validated
mechanism** (revised from the original `index1_hint`/`pinky1_hint`-based
sketch, which testing showed unnecessary — wrist + elbow alone drive both
the crop and the gate):

```python
def detect_hand_in_crop(
    image: np.ndarray, wrist: tuple[float, float], elbow: tuple[float, float] | None
) -> HandDetectionResult | None:
    """Crop, run rtmlib.Hand, pick nearest candidate, gate by proximity.

    Returns None if no candidate passes the gate. Otherwise a result
    carrying the 21 hand21 keypoints (crop-local, caller maps back to
    full-frame), per-keypoint confidence, and the winning candidate's
    root-to-wrist distance (worth logging/surfacing even on a pass).
    """
```

Used by both the batch pipeline (Idea 2, after every frame's full-body
pass) and the interactive post-edit worker (Idea 3, on demand) — same
crop formula, same candidate selection, same gate, one implementation.
Worth writing it once, shared, when this gets built; the offline test
scripts used for the four rounds of stills validation are a direct,
already-working reference implementation to start from.

---

## Phasing

Sequenced by dependency and risk — cheapest, most self-contained, most
directly tied to an already-diagnosed problem first; the expensive,
foundational schema work deferred until a cheaper version has validated
the underlying premise is worth building on.

**Phase 0 — Idea 1, edit noise + gate treatment.** First cut (hard
`force_inlier` bypass, flat `edited_kp_noise_std`) tested net negative
(Roosa avg NIS/DOF 1.52→28.84) — see "Phase 0" in the crisis log. **Phase
0b (revised: scale-to-threshold via bisection) done and now working for
both Roosa and Tommi** — see "Phase 0b" in the crisis log for the full
writeup, including a second real-data failure on Tommi that turned out to
be a data error (swapped shoulder/wrist keypoints), not a mechanism
limitation, fixed and confirmed resolved. Still off by default
(`edited_kp_noise_std=0`) — promoting to on-by-default is a separate
decision, not made here. The "surface gate-rejected edits" diagnostic idea
(section above) is still unbuilt and would help future tuning/validation
of this mechanism.

**Phase 0.5 — offline stills validation (done, cheaper than a pipeline
build, 2026-07-12).** Before writing any pipeline code, the crop/gate
mechanism itself was validated against curated real frames (60+ crops
across Tommi, Roosa, and Harri in a different trial/activity) using a
standalone script against `rtmlib.Hand`, iterated through four rounds
based on direct visual review. This is the premise-validation step Phase 1
below originally described doing *inside* the pipeline — doing it offline
first turned out to be strictly cheaper (no pipeline plumbing, no DB
writes, fast iterate-on-a-formula loop) and already answered the open
question ("does a dedicated hand-crop detector meaningfully reduce
occlusion/identity-mixup garbage") with a concrete yes, plus tuned,
specific formulas (crop sizing/offset, candidate selection, gate
threshold — see *Idea 2* above) that Phase 1 can now implement directly
instead of guessing at. One real limitation surfaced (two-handed
coordinated grips) that Phase 1 inherits knowingly rather than discovering
mid-build.

**Phase 1 — Idea 2, interim version (no schema change) — implemented,
2026-07-12.** Added the hand-specific detection stage as
`posetrak.detection.hand_refinement` (`HandRefinementPipeline`,
`detect_hand_in_crop`), run in `python/app/pose` right after the existing
full-body pass, before track assignment/finalization (wired into both
`DetectionJob` in `main.py`, behind a "Refine hands" checkbox, and the CLI
`run` command's `--refine-hands/--no-refine-hands` — see *Open questions*
#8 for the remaining integration-point questions this didn't need to
resolve, since it hooks in via the existing job/command rather than a new
pipeline phase). Uses the tuned crop/gate formulas from *Idea 2* above.
Merged into the existing single `kp_blob`/row for now (accept the noise
imprecision — hand keypoints inherit the frame's whole-body `noise_scale`
until Phase 2's multi-row schema exists to represent the difference
properly; this was always the accepted interim tradeoff). Unit-tested
against a fake hand model (`python/tests/app/test_hand_refinement.py`);
the `detect_hand_in_crop(image, wrist, elbow)` function ended up not
needing to be literally shared with Idea 3's sketch signature word-for-word
since Idea 3 isn't built yet, but it's the same function, ready to be
called from a future post-edit worker. **Not yet done**: *Validation*
against a real trial — the hygiene-scan-style before/after comparison of
near-origin/garbage finger detections (same scan already run this
session), and a targeted look at how often the two-handed-grip failure
mode actually occurs at trial scale (so far only observed twice, in
curated stills).

**Phase 2 — multi-row `pose_observations` migration — implemented and
validated against real data, 2026-07-14.** Schema v34→v35 (PK rebuild
adding `source`, `detection_run_id` moved per-row), C++ `session_reader.cpp`
loader merge, Python read/write-path merges — all committed and confirmed
working end-to-end on a real trial (new detection run copied from an
existing one, hand-refined, finalised, tracked): hand markers' median
mahalanobis distance roughly halved vs. a body-only baseline, per-source
`noise_scale` confirmed correctly tighter for hand crops. Upgraded Phase 1's
hand pass to correct per-source noise as intended, and is the prerequisite
for Idea 3 (below) and (out of scope here) future marker detections.

**Phase 3 — Idea 3, automated post-edit redetection. Integration-level
design finalized 2026-07-14** (see the *Idea 3* section above for all six
decisions: source tiering via a **generic `<base>.refined` precedence
convention** — hand detection is its first user (`'hand_l'`→`'hand_l.refined'`,
`'hand_r'`→`'hand_r.refined'`), not a hand-specific mechanism, so a future
auto-detection-after-edit feature (e.g. face landmarks) can reuse it without
its own merge special-case — instead of a manufactured `detection_run_id`;
reject/revert/further-edit; interpolation integration; one converged
trigger point across all 7 editing-operation call sites; frame access
mechanism; new timeline status color). All design-level open questions
resolved as of 2026-07-14 (editor write access, "select whole hand",
provenance tracking, status-color scope — see Open Questions below). Not
yet implemented — remaining work is the actual build: the generic
`.refined`-override pass in `session_reader.cpp` / `observation_merge.py`,
the new worker, and the DB/UI plumbing described above. *Validation once
built*: the debounce/trigger behavior against real editing sessions, and a
hand-completion-time comparison (does this actually reduce how long manual
hand-editing takes).

---

## Open questions

1. **Idea 1**: hard `force_inlier` + fixed `noise_std_override` (Phase 0)
   tested net negative; scale-to-threshold (Phase 0b) tested and now
   working for Roosa and Tommi, both real-data failures traced to bad data
   rather than the mechanism — see crisis log. Not yet: tested on Timo,
   tested against the full trial-wide adaptive-on/off comparison, or
   promoted to on-by-default. The "surface gate-rejected edits" diagnostic
   (MCP tool and/or editor UI, Idea 1's section above) is still unbuilt and
   would help validate further tuning without needing an ad hoc script
   each time (as this round's investigation needed twice). Per-edit
   user-specified confidence (*Measurement noise*, "The real fix, not
   built now") remains the longer-term direction.
2. **Idea 3 debounce window**: 500ms-1s proposed above is a guess, not
   tuned.
3. **Idea 2/3 crop and gate formulas** (half-width `max(0.9×forearm,
   60px)`, offset `0.35×forearm` away from elbow, gate
   `max(0.5×forearm, 40px)`): validated against 60+ curated stills across
   three people, two trials — see *Idea 2* above. Still open: tuning
   against a full trial run rather than curated cases (curated frames
   likely skew toward the interesting/hard cases the user picked, not a
   representative sample), and the elbow-unconfident fallback (small
   wrist-centered crop, no offset) has only come up rarely in testing so
   its adequacy is unconfirmed.
4. **Two-handed coordinated grip failure mode** (new, round 4, real
   footage): the proximity gate can reject a correct hand because the
   *other* hand's grip point is closer to the wrist anchor, seen twice on
   a sword grip. Not designed further in this doc. Possible directions,
   none evaluated: gate against both wrists simultaneously and disambiguate
   by which anchor the candidate is closer to (rather than a single
   pass/fail threshold); widen the gate specifically when both wrists are
   themselves close together (a detectable precondition); or accept this
   as an inherent limit and let it degrade to "no write" the same as any
   other reject, which is not silently wrong, just less complete for these
   scenes.
5. ~~Automated writes re-triggering~~ — resolved 2026-07-14: `hand_l.refined`/
   `hand_r.refined` reuse the same `source` value across repeated redetections
   of the same (camera, frame), so the existing PK
   `(sequence_id, camera_instance_id, video_frame, person_id, source)`
   naturally upserts (`INSERT OR REPLACE`) — plain overwrite, no
   history/accumulation, consistent with how edits and the batch hand pass
   already behave.
6. ~~Multi-row `pose_observations` migration~~ — done, see Phase 2 above.
7. ~~Editor write access to `pose_observations`~~ — resolved 2026-07-14:
   safe, given existing WAL-mode + multiple-background-writer precedent
   (`CropBackfillWorker`/`WideCropExtractWorker` already do this today for
   `frame_cache_entries`).
8. ~~Idea 2's pipeline integration point~~ — resolved, Phase 1 shipped
   (hooks into the existing `DetectionJob`/CLI `run` command).
9. ~~"Select whole hand" UI convenience~~ — resolved 2026-07-14: already
   built. The right-click group-selection menu already has `"Left hand"`/
   `"Right hand"` groups covering all 21 keypoints each
   (`kp_models.py:188-189, 207-208`), combined with the existing "Disable
   selected" action.
10. ~~Provenance tracking through the merge~~ — resolved 2026-07-14: concrete
    algorithm now specified in Idea 3's *UI status color* section (strip the
    `.refined` suffix, decide per base-name which row wins, then apply
    existing per-source index placement; one new small companion function,
    `merge_observation_sources`'s own signature/other caller unaffected).
11. ~~Scope of the new status color~~ — resolved 2026-07-14: timeline only.
    Checked `_ImageCanvas`'s dot-painting code — it already uses its own
    separate, tracker-outlier-based palette; `STATUS_BLUE` isn't used there
    either, so there's no existing-convention argument for extending this
    one to the crop view.
12. ~~The suffix name itself~~ — resolved 2026-07-14: `.refined`.
