# Segmentation-mask-based dot-assignment filtering (design note, for review)

Scope note (2026-09-28): follow-up to the sword-swing tracking work in
[dot-assignment-architecture-design.md](dot-assignment-architecture-design.md).
Two independent bugs found there (normal-inheritance, the Hungarian solver's
padded-column asymmetry) are fixed and committed. This note addresses a third,
separate failure mode found on the same trial: around t=306.8s, a fast sword
swing causes several candidates on `gopro-11_mini_01`/`gopro13_02` to lock onto
the wrong tracklet (one that happens to sit near `dot2`/`dot3`'s prediction),
and that wrong lock persists — via `TrackletContinuityModifier`'s own
gate-relaxation — for 25+ steps after the ArUco tags are visible and correctly
detected again, because the wrong dot lock keeps outscoring the correct
detection on cost alone. Implemented and validated against the real capture
(§9); the mask-based mechanism (§1b) is in, ArUco `force_inlier` (§1a) is
still open, separate work.

Feature branch: `feature/segmentation-mask-dot-assignment` (branched from
`detect-segmentation-source-and-scrubber-fix`, which carries today's other,
unrelated fixes).

## 1. Two remedies, both wanted, kept separate

You proposed two remedies for this failure and confirmed both are likely
needed, but asked that the mask-based one be designed here as the "fundamentally
correct" solution:

- **(a) ArUco/labeled markers should always be assumed correct.** Small,
  precedented change: `Observation::force_inlier` already exists and already
  bypasses both the Mahalanobis outlier check and cross-camera consistency
  grouping in `UnscentedKalmanFilter::reject_outliers()`
  (`cpp/src/filters/ukf.cpp`); today it's set only for manually-edited
  keypoints, and explicitly left `false` for cross-person anchors ("always
  subject to the outlier gate (identity-switch guard)",
  `multi_person_tracker.cpp:1155`). Setting it for coded-marker (`prop_markers`
  track) observations would let a correctly-detected ArUco corner override the
  filter's current, wrongly-confident state immediately rather than waiting for
  Mahalanobis distance to shrink enough to pass the gate — which is exactly the
  25-step stall observed. **Not covered further in this note** — it's a
  one-line-per-call-site change in the observation-construction path, not a
  design question, and can be done independently of the work below whenever
  you want it.
- **(b) Detect and remove wrong dot-slot lock-ons using segmentation masks.**
  The rest of this note.

## 2. Why this isn't just the existing `_make_dot_mask_filter`

`python/app/pose/finalise.py`'s `_make_dot_mask_filter` already drops
out-of-mask dot candidates against a segmentation run, but it's a **one-time,
destructive bake** at `finalise_object_to_db` time: it permanently deletes rows
from `pose_observations`, has no per-tracklet memory (a single frame's mask
membership, no notion of "this tracklet has spent the last 40 frames near the
mask boundary"), and goes stale the moment the segmentation is re-edited after
finalising (already logged as a known gap in
`productization-architecture-and-plan.md`'s "Segmentation-gated prop marker-body
calibration UI" entry). It also runs in Python at candidate-generation time, not
inside the assignment cost function, so it can only ever exclude a candidate
outright — it can't do what's needed here: let a marginal, near-mask candidate
compete more expensively rather than either being silently kept or hard-deleted.

This proposal is a **live, per-run, non-destructive** filter: it reads
`seg_masks` at tracking time, keyed by whatever `seg_run_id` the run is
configured with, computes membership per candidate per step, and folds that
into `resolve_dot_assignment()`'s existing cost machinery — the same place
`TrackletContinuityModifier` and friends already run. Nothing in
`pose_observations` is touched; a different segmentation run or a re-edited one
just changes the next tracking run's own trace, same as any other tunable.

## 3. Two-level scoring, per your design

You asked for: (a) is *this frame's* candidate inside the mask, and (b) what
fraction of *the whole tracklet's* history has been inside the mask. Both
apply per (camera, candidate), independent of which slot is being costed —
a tracklet either has a good in-mask history or it doesn't, regardless of
which marker slot it's competing for this step.

- **(a) Current-frame membership** is a stateless lookup: decode this step's
  mask for that camera, test the candidate's (dilated) pixel against it. No
  history needed — but *not* the same test `_make_dot_mask_filter` does: that
  filter tests `mask > 0` ("near any tracked subject"), while this needs
  `mask == this_pair's_subject_label` — see §5.1, this only became clear once
  multi-subject scenes were worked through.
- **(b) Per-tracklet in-mask fraction** needs a running accumulator, exactly
  the shape `StreakKAccumulator` already established for the streak-velocity
  extension: plain data, no Tracker/skeleton/mask access of its own, updated in
  place by the caller each step, queried by the cost modifier. Call it
  `TrackletMaskAccumulator` — for a given (camera_id, tracklet_id), a running
  count of (frames-in-mask, frames-seen), evicting old samples past some window
  the same way `StreakKAccumulator::add()` does, so a long-lived tracklet's
  score reflects its *recent* trustworthiness, not "ever, once, years ago."

Both signals feed a new `CostModifier`:

```cpp
/// @brief Penalizes a (candidate, slot) pair whose candidate's current-frame
/// position or recent tracklet history falls outside *the slot's own
/// subject's* label in the segmentation mask.
///
/// Two independent checks, both against
/// DotAssignmentContext::seg_mask_config[slot.subject_id]:
///  - this frame's candidate position, dilated, tested against this step's
///    own decoded mask for the candidate's camera, restricted to the pixels
///    labeled for slot.subject_id specifically -- not "labeled for any
///    subject" (a "hard now" signal, and the thing that lets it arbitrate
///    between two subjects competing for the same stray candidate, see §5.1);
///  - the candidate's tracklet's running in-mask fraction *for this subject*
///    from DotAssignmentContext::tracklet_mask_state[slot.subject_id] (a
///    "soft, over time" signal -- built for exactly the case you named:
///    segmentation is wrong or a reflective dot is still visible outside a
///    strict silhouette during a fast swing, so the current-frame test alone
///    would wrongly reject a real detection).
///
/// A pair is excluded outright when the current-frame test fails by more
/// than dilate_px AND the tracklet's own fraction (for this subject) is
/// below min_tracklet_fraction (both signs agree: not just off-mask now, but
/// with a history of being off-mask too -- exactly the "confidently
/// outside" case this exists to catch, not a single ambiguous frame).
/// Otherwise the pair's cost is scaled by a factor derived from the
/// tracklet fraction alone (see class body) -- a soft push away from a
/// since-questionable tracklet without ever making a good candidate
/// mathematically unreachable on one bad frame.
///
/// A no-op for a subject absent from seg_mask_config (no segmentation
/// configured for it) -- every existing single-subject caller is
/// unaffected.
class SegmentationMaskModifier final : public CostModifier { ... };
```

This is deliberately a soft/hard combination, not a hard gate alone — matching
your "there likely needs to be some hysteresis" concern: a single wrong-side-of-
the-mask frame during fast motion (blur, mask lag, dilation edge) only nudges
the cost; only a *tracklet* with a sustained bad history gets excluded outright.
That's what stops the sword-blade failure mode: the wrong lock's tracklet would,
over its 25-step run, accumulate a low in-mask fraction (the mask belongs to the
sword, and the wrong-lock candidates are drifting off it during the fastest part
of the swing) — this is the mechanism that would have broken the lock well
before ArUco recovery instead of 25 steps after it.

## 4. Where the masks come from and why the boundary sits in C++

You were explicit that this belongs in the tracker, not a Python preprocessing
pass, even though it requires a new dependency there — the reasoning being that
the modifier needs the same per-step, per-candidate granularity the rest of
`resolve_dot_assignment()` already has, and a Python-computed side table would
either need to be joined back in per candidate (extra DB round-trips inside the
hot per-frame loop) or precomputed for the whole trial ahead of time (stale the
moment a mask changes, the exact problem you said to avoid).

C++ currently has **no image-codec dependency at all** — no OpenCV, no PNG
decoder of any kind (`cpp/meson.build`/`cpp/*/meson.build` wrap list is
`catch2, cli11, eigen, fmt, nlohmann_json, rerun_cpp_sdk, sqlite3,
tomlplusplus, yaml-cpp`). `seg_masks.mask_blob` is a plain single-channel
`cv2.imencode(".png", labeled)` blob — greyscale, no palette, no interlacing,
no animation. Decoding that doesn't need OpenCV's ~60MB of unrelated
functionality; **`stb_image.h`** (single header, public-domain, already the
de-facto standard minimal PNG decoder for exactly this kind of codebase) reads
it in a few lines. Proposed as a new Meson wrap alongside the existing ones —
smallest dependency that solves the actual problem, not a new build-system
subsystem.

## 5. Where masks are read, and the load-time relationship you asked for

Per your note: "masks may be edited after dot detection so this cannot be
stored in database like tracklet ids... keep these separate & build the
relationship at load time, should be quite cheap." Concretely:

- `seg_masks` is keyed by `(seg_quality_run_id, shot_video_id, frame_idx)` —
  `frame_idx` is the camera's own native video frame number, same convention
  `_load_mask()` (`pose_worker.py`) already uses, **not** a tracker step. The
  tracker already resolves tracker step → per-camera native frame elsewhere
  (wherever candidate blobs are pulled per camera per step today) — the mask
  lookup reuses that same resolved frame number, no new mapping needed.
- Nothing about mask membership is written back to any table. A tracking run
  configured with a `seg_run_id` decodes and tests masks live, every run, from
  whatever `seg_masks` rows exist *at that moment* — editing a segmentation
  and re-running tracking picks up the edit for free, by construction, because
  there's no cached derivative to go stale.
  - Note this differs from `tracking_assignment_trace` (this session's earlier
    instrumentation): that table intentionally *does* persist per-run decision
    data, but only because it's a diagnostic record of what a specific run in
    the past decided, not an input another run reads back.

### 5.1 Multi-subject scenes (dots are anonymous, candidates are shared)

`resolve_dot_assignment()` already runs **one shared Hungarian solve per
camera across every dot-bearing subject at once** — a stray candidate can
legitimately be in contention for two different subjects' slots in the same
call (that's the whole reason the architecture doc calls for one joint solve
instead of one independent solve per subject). So the mask test can't be a
single "is this candidate inside a mask" boolean; it has to be
subject-specific, and it turns out the data already supports that:

- `seg_masks.mask_blob` is not a binary silhouette — it's a **labeled** mask
  (`0=bg, 1..N=subject`), and `seg_quality_runs.persons_json[i]` names the
  subject at label `i+1` (confirmed this applies uniformly to persons *and*
  objects — `cutie_init_panel.py`'s own comment: "an object's name flows
  through identically to a person's"). `_make_dot_mask_filter` (the
  finalise-time filter, §2) throws this away today — it tests `mask > 0`,
  "near *any* tracked subject" — which is fine for a coarse prefilter but
  useless for arbitration.
- The live modifier must instead test `mask == label_for(slot.subject_id)`.
  `CostModifier::apply()` already receives `DotSlotRef::subject_id` per pair,
  so this needs no new plumbing to the modifier itself — the same shared
  candidate pool, tested against each competing subject's own label in turn,
  is exactly what lets a stray candidate score well against subject A's slots
  and poorly against subject B's in the same call. This is a strictly better
  version of the coarse "off-mask" idea in §3: it's the same mechanism that
  also settles ambiguity *between* subjects, not only between "the scene" and
  "off-scene."
- Consequence for config shape (see §6): `seg_run_id`/label etc. cannot be a
  single scalar shared across every subject in a call the way
  `dot_tracklet_gate_multiplier` is (that one is explicitly required to be
  *identical* across subjects sharing a solve —
  `find_dot_config_disagreement()` errors on a mismatch). Which seg_run
  applies, or whether one exists at all, is legitimately per-subject (a prop
  might have no segmentation; two people in frame have two different
  silhouettes in the *same* mask blob, distinguished only by label). Needs a
  `subject_id -> SegMaskConfig` map instead — see §6 for where the
  subject-to-label resolution actually happens (Python, not new schema).
- Consequence for the accumulator: a tracklet's in-mask fraction is only
  meaningful relative to *one* subject's label. A tracklet correctly on
  subject A but briefly considered (and rejected) for subject B's slot must
  not have its A-relative trust history contaminated by a B-relative score —
  so `TrackletMaskAccumulator` state needs the same
  `subject_id -> camera_id -> tracklet_id` nesting `PrevDotPositions`/
  `PrevDotTrackletIds` already use, not a flat `camera_id -> tracklet_id` map.

## 6. Config and plumbing

Mirrors the existing `dot_streak_*`/`dot_tracklet_gate_multiplier` precedent —
new `TrackerConfig` fields, a new `tracker_configs` migration
(schema is at v56 after `tracking_assignment_trace`; this would be v57),
`DotAssignmentContext` fields analogous to `streak_config`/`streak_k_state`,
now subject-scoped per §5.1:

```cpp
struct SegMaskConfig {
    bool enabled = false;
    std::string seg_run_id;   // seg_quality_runs.id; empty disables
    int mask_label;           // already-resolved 1..N label for this subject
                               // in that seg_run -- see resolution note below
    double dilate_px = 20.0;  // native-pixel dilation before testing
    int tracklet_window = 60;        // frames, same accumulator-window shape
                                      // as StreakKAccumulator's own window
    double min_tracklet_fraction = 0.3;  // below this + off-mask now -> excluded
    double cost_scale_at_zero_fraction = 4.0;  // soft-penalty ceiling
};
```

`DotAssignmentContext` gains
`std::unordered_map<int, SegMaskConfig> seg_mask_config;` (keyed by
subject_id; a subject absent from the map is a no-op, so a scene with no
segmentation at all, or where only some subjects have one, works unchanged)
and
`std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, TrackletMaskAccumulator>>>* tracklet_mask_state = nullptr;`
(subject_id -> camera_id -> tracklet_id).

**Name/label resolution stays in Python, not C++ — this was overcomplicated
in an earlier draft of this note; corrected after checking the actual
join.** `seg_quality_runs.persons_json[i]` names the subject at mask label
`i+1`; a tracking subject's own name is `sequence_persons.person_name` for a
person, or `capture_objects.name` for an object (both already exist,
resolved by `manage_person.py`'s `persons_ordered_for_seg_run()` for the
`persons_json` half, and by `finalise.py`/`sequence_persons` for the subject
half — no new schema needed to go from "subject" to "its own mask label").
But `cpp/src/db/session_reader.cpp` never reads either of those name-bearing
tables today, and shouldn't have to just for this: **the Python launcher
(`python/posetrak/tracker/runner.py`, which already builds a `PersonRunSpec`
per subject and shells out to the tracker binary) resolves `seg_run_id` +
subject name -> integer `mask_label` once, before invoking the binary, and
passes the plain resolved integer down** (TOML config or CLI flag). C++ only
ever compares an already-resolved `int` against a decoded mask byte — it
gains a PNG decoder, not a name-matching join.

Which `seg_run_id` applies to a given subject in the first place is **not
persisted anywhere new**, mirroring `dot_mask_seg_run_id`'s own existing
precedent (a `finalise_object_to_db` parameter, never written back to any
column): the operator supplies it per subject at track-invocation time
(TOML/CLI), the same way `RunDetectionDialog` already offers an explicit
segmentation picker for a detection run (`run_detection_dialog.py`,
`segmentation-reuse-design.md`'s "gap 2") rather than the schema
auto-discovering it. If a future UI wants a "run tracking with this
segmentation" picker, it can follow that same explicit-choice precedent
without any schema change either.

Masks themselves are decoded and cached per (camera, frame) for the duration
of one step's resolve call — same "not cached across frames, one full mask
retained at a time" memory posture `_make_dot_mask_filter`'s own
`dilated_mask()` already documents and justifies for the same data shape; one
decoded mask is reused for every subject's label test that step, only the
label byte compared against differs.

## 7. Testing

Same split as the rest of `dot_assignment.hpp`: `TrackletMaskAccumulator` is
pure data, unit-testable like `StreakKAccumulator` (window eviction, fraction
computation, no-samples-yet state). `SegmentationMaskModifier::apply()` is
testable against fabricated masks/context, no DB or Tracker involved. The PNG
decode step (`stb_image.h` wrapper) gets its own small test against a real
`seg_masks`-shaped blob. Integration is verified the same way the Hungarian-
solver fix was: re-run the sword trial's t≈306-308s window with
`--trace-assignment-start/-end` spanning it, confirm via
`get_assignment_trace` that the wrong tracklet's cost rises (or the pair gets
excluded) once its in-mask fraction degrades, rather than only trusting the
aggregate tracked-percentage number.

## 8. Open questions for your review

1. Is `min_tracklet_fraction=0.3`/`dilate_px=20`/`tracklet_window=60` a
   reasonable starting guess, or do you want these derived from the real sword
   trial's own mask/candidate data before picking defaults?
2. Should `SegmentationMaskModifier` run before or after
   `TrackletContinuityModifier` in `default_cost_modifiers()`'s order? As
   written above it's independent (operates on tracklet mask history, not
   tracklet *continuity*), but the order matters if both end up modifying the
   same pair's cost multiplicatively vs. one excluding first.
3. **Settled**: `_make_dot_mask_filter` in `finalise.py` stays as-is. It's a
   different job — a coarse, one-time, storage/solve-cost prefilter over the
   raw candidate pool, run once at detection-finalise time — not a cruder
   version of the live per-subject trust mechanism, which never shrinks the
   stored pool and would leave every noisy background detection riding along
   in every future run's candidate merge/solve if the finalise-time filter
   were dropped.
4. **Settled** (you flagged this was worth checking before this note added
   more schema — right call, an earlier draft here proposed a new
   `tracking_run_persons`/`tracking_run_objects` column it didn't need):
   subject-to-mask-label resolution reuses data that already exists —
   `sequence_persons.person_name` (person) or `capture_objects.name`
   (object) matched against `seg_quality_runs.persons_json` — done once in
   the Python launcher (`runner.py`) before invoking the tracker binary, not
   in C++ and not via any new column. Which `seg_run_id` applies to a
   subject is supplied at invocation time (TOML/CLI), never persisted,
   mirroring `dot_mask_seg_run_id`'s existing precedent. See §6.

## 9. Real-data validation

Sword trial (sequence `84d5b140…`, skeleton `b19afc91…`, base tracker config
`00a21d49…`, t=303.5–323.259s) against the session's one `seg_quality_runs`
row (`db73c8b8…`). Its `persons_json` lists only `["Harri"]`, yet the mask
blobs carry **three** labels: 1 = the person, 3 = the sword (label 2 unused).
The sword's label is therefore not resolvable by name
(`resolve_seg_mask_label()` returns None for it) and has to be given as an
explicit `--seg-mask-label 3`; a segmentation whose extra labels are not in
`persons_json` cannot be used by name.

**A first pass used label 1 (the person) as the sword's trust region and was
wrong**: the blade lies outside the person's label, so the correct blade-dot
(`dot0`/`dot1`) tracklets were excluded on the second step and tracking was
much worse than without the mask. That pass's conclusions (including "a
150 px dilation does not fix the bug") were artefacts of the wrong label and
are withdrawn. It also exposed a hysteresis flaw: the tracklet fraction was
trusted from a single sample, so one out-of-mask frame excluded an
established tracklet. The fraction now needs at least 5 samples
(`kMinTrackletSamples`, `dot_assignment.cpp`) before it can exclude or
penalise.

Correct runs, label 3:

| run | steps ≤120 with dot0 / dot1 resolved | 306.4–308.0 s: dot0 / dot1 / dot2 / dot3 resolved | mean / max NIS·DOF⁻¹ |
|---|---|---|---|
| baseline (no mask) | 36 / 86 | 16 / 63 / 114 / 116 | 0.154 / 3.79 |
| mask, `dilate_px=20` | 35 / 86 | 61 / 99 / 38 / 76 | 0.184 / 3.94 |
| mask, `dilate_px=150` | 36 / 86 | 63 / 75 / 33 / 64 | 0.181 / 4.41 |

The blade dots now behave like baseline at the start, and in the failure
window `dot2`/`dot3` resolutions drop (114→38, 116→76) while `dot0`/`dot1`
rise. Whether the remaining `dot2`/`dot3` resolutions are the correct ones,
and whether the ArUco recovery delay is gone, has not been checked against
video; the aggregate numbers alone do not show it. `dilate_px` 20 vs 150
shows no clear difference here.

## 10. Distance exclusion

The tracklet-history rule (section 3) cannot judge a candidate whose tracklet is new.
A static background light that a fast-moving prop passes in front of is detected as a
new tracklet every few frames, so the prop's slot can lock onto it and then hold it,
because the filter has no other measurement to pull it away.

`dot_seg_mask_hard_distance_px` (tracker config, 0 = off) adds a rule that needs no
history: a candidate farther than this from its subject's mask label in the current frame
cannot be assigned to that subject's slots. The search is `mask_label_present_near` with
the larger radius. It is skipped for a camera whose mask has no pixel of the label at all
(`DecodedSegMask::label_present`), because a frame where segmentation lost the subject
says nothing about where the subject is.

Evidence on a 2.6 s window of a fast sword swing, 4K cameras, tracker at 120 Hz, mask
label of the sword (1 pixel in the mask = 2 video pixels):

- Before the lock-on, every accepted dot was within 60 video pixels of the sword mask
  (281 inside, 71 within 20 px, 22 within 20-60 px, none beyond).
- With the mask rules but no distance rule, 632 of 689 accepted assignments after the
  lock-on were more than 150 px from the mask (up to 920 px).
- With a distance of 70 px, after the same moment 937 were inside, 199 within 20 px,
  68 within 60 px, 3 within 150 px and none farther. Mean inlier observations per
  step rose from 2.8 to 19.8 (steps 100-200) and from 5.4 to 14 (steps 200-310); 7
  steps with no usable measurement were reported lost.

The distance should exceed the largest distance at which a correct dot is seen from the
mask (mask boundary accuracy, a thin part the segmentation misses), and be small
compared with how far a stray light can be from the subject.
