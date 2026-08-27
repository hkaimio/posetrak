# Mask-based pose treatment for segmentation-driven extraction — design sketch

> **Status (2026-08-26)**: Proposal only, nothing implemented. Written up
> after an exploratory study (`python/tools/segmentation_mask_steering_experiment.py`,
> not yet part of any pipeline) validated the core idea against real grab
> footage from two captures. This doc is the resulting design; see
> "The study" below for what was actually measured.
>
> **Decided (2026-08-26): two-phase plan.** Phase 1 (below, "Where this
> actually plugs in") is the near-term target — apply the treatment inside
> the existing offline `PoseWorker` path, no new architecture needed.
> "Fuse pose estimation into the interactive tracking loop itself" (a
> materially different architecture, see its own section below) is a
> deliberately deferred Phase 2 candidate: real GPU-memory and timing
> numbers were measured for it, but building it now isn't justified until
> Phase 1 ships and real usage shows segmentation-quality feedback is
> still a pain point worth the added interactive latency.
>
> **Phase 1: implemented** (commit `1b47033`, this branch). `PoseWorker`
> applies `suppress_others()` (`posetrak/detection/mask_treatment.py`) per
> person before estimation; `PoseExtractionJob.apply_mask_treatment`
> (default on) keeps it toggleable. Hand refinement is **not yet** wired
> to treated crops — `HandRefinementPipeline` is a shared,
> bbox-source-agnostic component with no mask awareness today; that's a
> separate follow-up, not folded into this change. Not yet validated
> against a real video/session (same caveat `PoseWorker` itself already
> carried before this change, per the segmentation-reuse status doc).
>
> **Real-tracking-run finding (2026-08-27), gates Phase 1 as shipped
> here.** Harri retracked trial `d70d5b60` end to end with the treatment
> on (run `d1fd9d62-...`) against the original baseline (`75e00861-...`)
> and visually spotted extra jerkiness in the arms during throws. Measured
> directly from `tracking_results` (RMS jerk of raw joint angle, both
> smoothed runs, `posetrak.db.load_session.load_tracking_run_data`):
> arm-joint jerk is elevated 1.4-2.5× for the treated run, and **not
> confined to contact moments** — it's present across nearly the entire
> trial, worse during throws but rarely near parity even in calm stretches.
> One severe, localized spike found directly (`hand.L` around t=44.2s,
> ratio 19.6× in that one-second bin): a genuine transient divergence the
> UKF's own outlier gate didn't fully catch, not just diffuse roughness.
>
> **Working hypothesis**: the treatment is computed independently per
> frame from that frame's own mask. Normal mask-boundary jitter is
> harmless to the untreated path (the model always sees the same true
> pixels regardless of mask stability), but feeding that jitter through
> `suppress_others()` gives the model a slightly different "context edit"
> every frame — an added source of input variability the baseline path
> structurally cannot have. Not yet tested against this hypothesis
> directly (e.g. by checking whether mask-boundary movement at t≈44.2s
> in the source footage actually correlates with the spike).
>
> The original study measured per-frame pose *correctness* (in-mask
> keypoint fraction) but never measured downstream *temporal* smoothness
> through the actual UKF tracker — a real gap, only caught because this
> was run for real rather than judged from the study script's own debug
> videos. **Whether Phase 1 should ship as-is, gain a temporal-smoothing
> mitigation, or be reconsidered isn't decided** — Harri is also now
> weighing whether image-treatment tricks are the right approach at all
> versus fine-tuning Cutie and/or the pose estimators for this use case;
> see "Open questions" below.

## Motivation

Close-contact frames — specifically a grab, where two people's bboxes
nearly coincide — are exactly where a top-down pose model's crop shows
both people at once, and it's the model's own judgement, not anything the
pipeline controls, which one it actually estimates joints for. Wrong-body
keypoint attribution during grabs is a known, recurring failure mode.

## The study

A standalone script (`python/tools/segmentation_mask_steering_experiment.py`,
built and iterated over several rounds on 2026-08-25/26, not wired into
any real pipeline) tested the direct fix: before running pose estimation
on a person's crop, alter every pixel *not* belonging to that person
(background and any other tracked person alike) so the model has less to
latch onto or hallucinate from. Four treatments were compared against an
unmodified baseline across ~40,000 metric rows spanning two captures
(a 2-person and a 3-person aikido/ukemi scene), several grab moments per
capture, both RTMPose and ViTPose, and the existing hand-specific
refinement pass (`posetrak.detection.hand_refinement.detect_hand_in_crop`)
riding along for a second, independent read on the same question:

- **hard** — fill everything outside the mask with flat gray.
- **blur** — Gaussian-blur everything outside the mask.
- **feather** — blur, but alpha-blended in over a 15px band outward from
  the mask boundary rather than a hard cutoff (the idea being that a hard
  boundary is an edge shape the pose model never saw in training).
- **feather2** — the same idea with a narrower 10px band *and* a
  contrast reduction blended in alongside the blur, tried after **feather**
  underperformed expectations.

**Headline result**: the ranking `hard > blur > feather2 > feather > none`
held consistently across every capture, camera, and model tested, on both
the wholebody in-mask-keypoint-fraction metric and the dedicated
hand-refinement model's own version of it (which showed an even larger
relative gain — masking matters more for a tight wrist-anchored crop than
a wider wholebody one). `feather2` closed most of the gap toward
`hard`/`blur` while keeping a noticeably smaller confidence penalty than
either.

**The metric has a known blind spot, caught only by manual video review**:
a hallucinated keypoint that lands anywhere inside the *correct* person's
own silhouette still scores as "in mask," whether or not a real
anatomical feature is actually there. `hard`/`blur` fully erase the palm
when a grabbing hand's mask boundary cuts through it, and were observed
(Harri, watching the actual debug videos) to sometimes hallucinate
confident-looking fingers on the bare forearm left behind — a failure the
aggregate numbers cannot see, since they only ever check silhouette
membership. `feather2` was Harri's verdict as best-so-far specifically
because it never fully erases the palm the way `hard`/`blur` do, while
still recovering most of their disambiguation benefit.

**The governing factor either way is mask trustworthiness, not treatment
choice**: all treatments help when the mask is right; none can when it
isn't (masking can't manufacture correctness the mask itself doesn't
have); and `hard`'s all-or-nothing erasure means it fails *worse* than
the gentler treatments the moment the mask is wrong, exactly where a grab
is hardest to segment in the first place.

Debug videos and the full metric CSVs from this study are local only
(`D:\mocap\segmentation-study\trial1` through `trial4` on Harri's
machine, ~24k + ~16k rows) — not checked into the repo.

## Why this can't be a live, automatic step

The natural next question — fuse mask generation (Cutie) directly into
the normal per-frame detection pipeline (`DetectionPipeline._process_camera`,
`posetrak/detection/pipeline.py`), so every detection run gets this
treatment automatically — **does not fit how segmentation actually
works today**. Cutie alone is not reliable enough on exactly the hard
grab frames this treatment targets; segmentation is, and needs to remain,
an **interactive, human-supervised operation**: a person reviews a Cutie
pass, finds the frame range where it drifted, and re-seeds it (often
running *backward* from a clean frame in the middle of the bad range) via
`app/pose/cutie_init_panel.py`'s existing click-correct-and-track
workflow. There is no way to make that judgement call automatically, and
trying to would defeat the entire point — the treatment only ever helps
when the mask it's built from is one a human has actually verified.

So mask generation stays exactly where it is: a separate, supervised step
that happens *before* pose estimation, on its own timeline, producing
curated rows in `seg_masks` (see `018_seg_masks.sql`). This feature is
about what happens *after* that step, not about replacing it.

## Where this actually plugs in — already 90% built

The segmentation-reuse feature
(`docs/roadmap/features/segmentation-reuse/`, ~85% done as of 2026-08-16)
already built exactly the seam this needs: `PoseWorker`/`PoseExtractionJob`
(`python/app/pose/pose_worker.py`) is a queued job — reachable today from
`CutieInitPanel`'s "Queue Pose" button, or `RunDetectionDialog`'s "Bbox
source" combo — that reads a curated `seg_masks` row per frame, derives a
per-person bbox from it (`_bboxes_from_mask`, currently a bare tight box,
no smart-padding), and runs the pose estimator over the *raw, untreated*
frame using those bboxes (`_run_pose`, around `pose_worker.py:198-202`).
It already runs strictly against human-reviewed masks, on a bounded,
already-supervised frame range — precisely the precondition the study's
"only helps if the mask is trustworthy" finding calls for. No new
architecture is needed to get a trustworthy mask to this code path; it's
already there.

The proposed change is narrowly scoped to this one function: before
`estimator.estimate(frame_bgr, detections)` is called, build a treated
version of `frame_bgr` per person (feather2, or whichever treatment wins
further validation) using that same frame's already-loaded mask, and
estimate against the treated crop instead of the raw one. The same
treated-crop input should feed `_run_hand_refinement`'s
`detect_hand_in_crop` call too, given the hand-refinement model showed
the *larger* relative benefit in the study.

## The real implementation wrinkle — resolved, it isn't one

`RTMPoseEstimator.estimate()` (`posetrak/detection/backends_rtmpose.py`)
takes one frame and a list of bboxes; per-person treatment means each
person needs their *own* treated frame (their own crop with *everyone
else* suppressed), so the call has to become one call per person instead
of one call per frame. Checked directly against rtmlib's own source
(`rtmlib/tools/pose_estimation/rtmpose.py` and `vitpose.py`,
`__call__`): there is no batching to lose — both already `for bbox in
bboxes: preprocess → inference → postprocess`, one ONNX session run per
bbox, regardless of how many bboxes arrive in one call. Per-person calls
are exactly what already happens today; this is not a throughput
regression at all.

Plain CPU/OpenCV treatment code, close to what the study script already
does, is adequate here regardless — `PoseWorker` is a queued, offline,
already non-realtime job over an already-bounded, human-corrected range,
not the live per-frame pipeline the GPU-fusion discussion earlier in
this investigation was worried about.

## Phase 2 candidate (deferred): fuse pose estimation into interactive tracking

A different idea surfaced while discussing this doc: instead of a
separate "Queue Pose" step after segmentation is finalized, have each
interactive tracking job (`CutieWorker`'s forward/backward jobs,
`cutie_init_panel.py`) *also* run pose estimation per frame as it goes,
using the mask it just produced, and write/overwrite
`detection_keypoints` directly. The appeal: immediate visual feedback on
whether a segmentation attempt is actually "good enough" to produce
correct pose, rather than only eyeballing the mask and finding out later
whether it worked.

**What first looked like blockers turned out not to be, once thought
through against the actual code**:

- *Segmentation's fragmented run identity* — each interactive session
  creates its own new `seg_quality_run` row, with older sessions kept as
  fallback coverage (`_load_stored_mask`'s multi-run lookup,
  `cutie_init_panel.py`). This only matters for a *later* consumer doing
  its own lookup by `seg_quality_run_id`, like `PoseWorker._load_mask`
  does today. Fused, pose estimation runs inline against the mask that
  was just computed in memory — there's nothing to look up.
- *Append-only detection runs* — resolved by assigning one
  `detection_run_id` up front for the whole segmentation effort and
  finalizing once, at the end. This isn't even a new exception: this
  codebase already treats a detection run as freely mutable until
  finalized (`finalise_to_db` refuses a second call once `tracking_runs`/
  `pose_observation_edits` reference it — see CLAUDE.md's data-model
  invariants), so repeated overwrites during interactive refinement are
  already the sanctioned pattern, just applied earlier in the pipeline
  than usual.
- *Track/person assignment* — a non-issue here specifically:
  segmentation-driven pose already has identity for free from
  `persons_ordered`, unlike the YOLO+tracker path where a generic
  track ID genuinely needs a separate person-assignment step.

**What actually matters is performance, and it was measured directly**
(real footage, real models, RTX 4080 Super, 2026-08-26):

| | GPU memory (resident) | added time/frame |
|---|---|---|
| Cutie alone (today) | ~500 MB model, ~1.24 GB during a step | 69.3 ms |
| + RTMPose | +483 MB | +55.2 ms → **124.5 ms total** |
| + ViTPose + hand refinement too | +2.16 GB more | +134.2 ms → **203.5 ms total** |

Memory is comfortable under any combination (~4 GB beyond Cutie's own
footprint, against ~9.7 GB free on this machine). Time is the real
trade-off: **Harri's call (2026-08-26): production needs hand refinement
too, and ~200ms/frame is too much for something meant to feel
interactive.** Fusing only the wholebody model (RTMPose, ~125ms/frame,
~1.8× slower than Cutie alone) is the more plausible version of this
idea; hand refinement would stay a separate, later pass — matching how
`_run_hand_refinement` already runs as a distinct step after body pose in
today's `PoseWorker`, just not pulled into the per-frame interactive
loop at all.

Decoding the frame only once (already resident for Cutie) instead of
once for segmentation and again for a later separate pose pass is a real
saving, but small relative to the added inference cost — not the
deciding factor.

**The actual deciding factor isn't a number this session could measure**:
today, pose is paid once, over whatever range is finally accepted. Fused,
it's paid on every forward/backward attempt, including discarded ones.
If a hard range typically takes 1-2 correction attempts, fusion is close
to a wash on total work (plus the feedback benefit); if it typically
takes many more, the repeated cost is real. That depends on Harri's own
correction workflow in practice, not on anything in this codebase.

## Open questions

0. **The jerkiness finding above, unresolved.** Options not yet weighed
   against each other: (a) temporally smooth the mask itself before
   deriving the treatment each frame, so boundary jitter doesn't reach
   the pose model frame-independently; (b) accept the trade-off for the
   disambiguation benefit in contact frames specifically, if a way is
   found to apply the treatment *only* during genuine grabs rather than
   always-on; (c) treat this as evidence that post-hoc image treatment is
   the wrong layer entirely, and prioritize the fine-tuning direction
   below instead.
1. **Confidence-weighted treatment as a `feather2` alternative.** Cutie's
   `InferenceCore.step()` (`cutie/inference/inference_core.py:139-170`,
   confirmed by reading it directly) returns a genuine per-pixel
   probability tensor per object *before* `output_prob_to_mask()`
   argmaxes it down to the hard label `seg_masks` actually stores —
   `1 - output_prob[target_label]` as a continuous alpha map would let
   the treatment's aggressiveness track Cutie's own boundary confidence
   directly, rather than committing to a fixed pixel radius. Not yet
   tried; would require capturing the soft probability at
   mask-*creation* time (`cutie_init_panel.py`'s tracking jobs), since
   `seg_masks` only ever persists the post-argmax label today. A real
   format/schema question, not just a `pose_worker.py` change.
2. **Should the treatment be user-toggleable per pose-extraction job**,
   similar to the install-time GPU segmentation checkbox elsewhere in
   this codebase, or always-on once validated?
3. **Which treatment, finally** — **decided for Phase 1: ship `feather2`
   as-is.** Its parameters (10px band, 0.4 contrast factor) came from one
   round of tuning after `feather` underperformed, not an exhaustive
   sweep, and confidence-weighting from (1) is a plausible improvement —
   but the mask feeding Phase 1 is already human-curated, which is
   exactly the case where confidence-weighting matters least (it's most
   valuable when the mask itself might be unreliable, i.e. Phase 2's
   automatic-during-tracking case). Revisit only if `feather2` proves
   inadequate in real use.
4. Should `python/tools/run_cutie_pose.py` (the older, related
   Cutie+RTMPose tool with its own tight/padded bbox derivation) gain
   the same treatment for consistency, or stay a separate experimental
   path?
5. **When to revisit Phase 2** — once Phase 1 has real usage, does
   segmentation-quality feedback (only discoverable today by running a
   separate later pose pass) remain a genuine pain point worth ~1.8×
   slower interactive tracking? Needs Harri's own sense of typical
   correction-attempt counts per hard range, not something measurable
   from the codebase alone.

## Future work (after Phase 1)

Harri (2026-08-26): once Phase 1 ships, revisit the original hand-painted
(brush) mask-correction idea from the study that led to this doc — the
existing interactive workflow only supports SAM2-point-click correction
today (`cutie_click_controller.py`), and a true freehand paint tool was
the first idea raised, before the study pivoted to the masking-treatment
question this doc is actually about. Bundle in whatever other
segmentation-UI improvements fall out of using Phase 1 in practice — not
scoped further than that yet.

**Harri (2026-08-27), raised alongside the jerkiness finding above**:
whether post-hoc image treatment (this whole doc's approach) is even the
right layer, versus fine-tuning Cutie and/or the pose estimators (RTMPose/
ViTPose) directly on this project's own close-contact footage. A
fine-tuned model wouldn't need any per-frame image manipulation at all —
it would just be inherently better at grabs — which would also sidestep
the jerkiness mechanism entirely (no mask-jitter-driven input variability
to introduce, since nothing about the input image changes frame to
frame). Not scoped or investigated at all yet: data volume needed, which
model(s) to target first, training infrastructure, whether curated
`seg_masks`/hand-corrected keypoints from this project's own sessions are
enough labeled data to fine-tune from directly.

## References

- `python/tools/segmentation_mask_steering_experiment.py` — the study's
  tooling (mask-source-agnostic: curated `seg_masks` or a fresh Cutie
  pass, both wired up).
- `docs/roadmap/features/segmentation-reuse/segmentation-reuse-design.md`
  and its `status.md` — the existing foundation this builds on.
- `docs/roadmap/features/hand-detection-refinement/hand-detection-refinement-design.md`
  — the hand-specific refinement pass this feature also treats.
