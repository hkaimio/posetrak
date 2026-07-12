# Hand-detection refinement & trusted-edit gate bypass — design sketch

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

Run the existing full-body detector (VitPose, COCO-133) first as today,
then for each frame/camera with a confident wrist detection, crop around
the estimated hand location and run an RTMPose hand-specific model on that
crop (RTMPose has a dedicated hand model; VitPose in this pipeline does
not). If the hand is actually occluded by the body, a hand-specific
detector — trained on hand crops, not whole-body context — should be much
better calibrated to report low/no confidence than a whole-body model
guessing from pose priors (the calibration problem discussed for the
"tighter confidence gate" idea, which this sidesteps rather than solves
directly).

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

**Trigger mechanism (revised — the original "trigger on editing wrist" was
underspecified, per Harri's questions)**:

- **Debounced, not synchronous.** Fire on a short idle timer (e.g. 500ms-1s
  of no further edits to `wrist`/`index_1`/`pinky_1` on that specific
  camera/frame) rather than on every single edit event. Covers both the
  chain-placement case (wrist is the 3rd of 5 keypoints placed in order —
  firing immediately on it would run with a worse fallback crop seconds
  before better anchors arrive) and a standalone nudge (arrow-key taps each
  write to the DB individually today — firing per-keystroke would be both
  wasteful and visually flickery). Anchor set used at fire time: whichever
  of wrist/index_1/pinky_1 are currently known, same fallback rule as
  before (forearm-length crop if index_1/pinky_1 aren't set).
- **Never overwrite a human-placed finger keypoint — resolved for free by
  where the write lands, not by a mask bit.** Automated results write to
  `pose_observations` (a new `source='hand.L'`/`'hand.R'` row, see below),
  never to `pose_observation_edits`. Since the loader always applies
  `pose_observation_edits` *last*, on top of every merged `pose_observations`
  row regardless of source (today's existing merge order, unchanged), a
  human edit on any finger index automatically wins over whatever the
  automated row says for that same index — no provenance bit, no
  overwrite-detection logic needed at all. Simpler than the earlier
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
written as their own `pose_observations` row(s) (`source='hand.L'`/
`'hand.R'`, own `detection_run_id` — an "interactive redetection" run,
distinct from the sequence's original batch detection run) and go through
the *same* outlier gate as any other detector output via the normal
merge-then-gate path — not through Idea 1's trusted-edit bypass, which
only ever sees genuine `pose_observation_edits` rows. This is the right
call: an automated redetection, however good the geometric-consistency
check, is still a machine guess, not a human verification — treating it
identically to a manual edit would undermine the whole point of keeping
the gate meaningful for edited data. It also means Idea 3's writes get
correctly-scaled measurement noise automatically (their own `crop_scale`,
per the multi-row mechanism), rather than inheriting the frame's original
whole-body noise level.

**Validation before writing**: same geometric-consistency check as
originally proposed — redetected `index_1`/`pinky_1` should land within
some pixel tolerance of the anchors that generated the crop; combined with
the hand model's own confidence. Below either threshold: write nothing for
that camera/frame (matches the existing tolerance for sparse per-camera
observations — no special-casing needed elsewhere).

**Where this lives / how it's triggered**: the codebase already has a
precedent for exactly this shape of work — a background worker started
from the editor that runs heavier processing and writes results back
(`CropBackfillWorker`, `FrameCropCacheManager` in `content_panels.py`,
`_start_backfill`/`_start_wide_crop_cache`). A new worker following that
same pattern (armed by the debounce timer above, running the hand
crop+detect step, writing the result as a `pose_observations` row with
`source='hand.L'`/`'hand.R'`, then triggering `_load_frame` to refresh) is
the natural fit rather than inventing a new async mechanism. Writing to
`pose_observations` from the interactive editor (today exclusively a
pipeline-write table) is itself a small boundary shift worth naming
explicitly — the editor process would need write access to that table,
which it doesn't exercise today (it only ever writes `pose_observation_edits`).

**Shared code with Idea 2**: both ideas are "crop around an estimated hand
location, run the RTMPose hand model, validate the result" — the actual
detect+validate logic should be one function
(`detect_hand_in_crop(image, wrist, index1_hint, pinky1_hint) -> keypoints | None`
or similar) used by both the batch pipeline (Idea 2, after every frame's
full-body pass) and the interactive post-edit worker (Idea 3, on demand).
Worth writing it once, shared, when this gets built.

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

**Phase 1 — Idea 2, interim version (no schema change).** Add the
hand-specific detection stage to the pipeline, merged into the existing
single `kp_blob`/row (accept the noise imprecision for now — hand
keypoints inherit the frame's whole-body `noise_scale` until Phase 2's
multi-row schema exists to represent the difference properly). Cheap to
build, validates the actual
premise (does a dedicated hand-crop detector meaningfully reduce
occlusion/identity-mixup garbage before investing in schema work) against
real trial data. *Validation*: hygiene-scan-style before/after comparison
of near-origin/garbage finger detections (same scan already run this
session), plus visual QC on a known-bad segment.

**Phase 2 — multi-row `pose_observations` migration.** Only once Phase 1
has shown hand-specific detection is worth doing precisely. The PK
rebuild + `detection_run_id`-per-row + `session_reader.cpp` loader
rewrite from *Idea 2 — hand-specific detection pass*, above. Upgrades
Phase 1's already-shipped hand pass to correct per-source noise as a side
effect, and is the prerequisite for Idea 3 and (out of scope here) future
marker detections.

**Phase 3 — Idea 3, automated post-edit redetection.** Built on Phase 2's
multi-row schema (proper per-source noise, clean `source='hand.L'/'hand.R'`
separation, no provenance-bitmask complexity) and Phase 1's shared
detect+validate function. Includes resolving the editor's new write access
to `pose_observations` (open question below). *Validation*: the debounce/
trigger behavior against real editing sessions, and a hand-completion-time
comparison (does this actually reduce how long manual hand-editing takes).

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
3. **Idea 3 fallback crop sizing** (forearm-length-based, when index_1/
   pinky_1 aren't yet known) is a guess, not validated against real data —
   needs tuning once built.
4. **Automated writes re-triggering**: a later automated redetection for
   the same (camera, frame, source) — the multi-row PK
   `(sequence_id, camera_instance_id, video_frame, person_id, source)`
   naturally supports upsert-by-source, so overwrite is the easy default;
   not yet confirmed there's no reason to want history/accumulation instead.
5. **Multi-row `pose_observations` migration** is the biggest open item in
   this whole doc: PK rebuild (not a simple `ADD COLUMN`), moving
   `detection_run_id` down from the sequence to the row, and a real
   `session_reader.cpp` loader change (iterate + merge N rows with
   per-source keypoint vocabularies, not one). Sizing this is future work.
6. **Editor write access to `pose_observations`**: Idea 3 needs the
   interactive editor to write a table it has never written before
   (today: read-only there, all editor writes go through
   `pose_observation_edits`) — worth a deliberate look before building,
   not assumed safe by default.
7. Idea 2's pipeline integration point (before/after track assignment,
   whether it shares the sequence's existing `detection_run_id` or gets
   its own under the multi-row model) not yet traced against
   `python/app/pose`'s current pipeline structure.
