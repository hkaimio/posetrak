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
detection on cost alone. Not yet implemented; this is the note to review before
any of it is built, per your request ("write a short design note for my review
before impl").

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
  history needed — same computation `_make_dot_mask_filter` already does per
  frame, just done once per (camera, frame) at tracking time instead of once
  at finalise time.
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
/// @brief Penalizes a candidate whose current-frame position or recent
/// tracklet history falls outside the subject's segmentation mask.
///
/// Two independent checks, both against DotAssignmentContext::seg_mask_config:
///  - this frame's candidate position, dilated, tested against this step's
///    own decoded mask for the candidate's camera (a "hard now" signal);
///  - the candidate's tracklet's running in-mask fraction from
///    DotAssignmentContext::tracklet_mask_state (a "soft, over time" signal
///    -- built for exactly the case you named: segmentation is wrong or a
///    reflective dot is still visible outside a strict silhouette during a
///    fast swing, so the current-frame test alone would wrongly reject a
///    real detection).
///
/// A pair is excluded outright when the current-frame test fails by more
/// than dilate_px AND the tracklet's own fraction is below
/// min_tracklet_fraction (both signs agree: not just off-mask now, but with
/// a history of being off-mask too -- exactly the "confidently outside"
/// case this exists to catch, not a single ambiguous frame). Otherwise the
/// pair's cost is scaled by a factor derived from the tracklet fraction
/// alone (see class body) -- a soft push away from a since-questionable
/// tracklet without ever making a good candidate mathematically
/// unreachable on one bad frame.
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

## 6. Config and plumbing

Mirrors the existing `dot_streak_*`/`dot_tracklet_gate_multiplier` precedent —
new `TrackerConfig` fields, a new `tracker_configs` migration
(schema is at v56 after `tracking_assignment_trace`; this would be v57),
`DotAssignmentContext` fields analogous to `streak_config`/`streak_k_state`:

```cpp
struct SegMaskConfig {
    bool enabled = false;
    std::string seg_run_id;          // seg_quality_runs.id; empty disables
    double dilate_px = 20.0;         // native-pixel dilation before testing
    int tracklet_window = 60;        // frames, same accumulator-window shape
                                      // as StreakKAccumulator's own window
    double min_tracklet_fraction = 0.3;  // below this + off-mask now -> excluded
    double cost_scale_at_zero_fraction = 4.0;  // soft-penalty ceiling
};
```

`DotAssignmentContext` gains `SegMaskConfig seg_mask_config;` and
`std::unordered_map<int, TrackletMaskAccumulator>* tracklet_mask_state =
nullptr;` (keyed by tracklet_id, one map per camera — same per-camera
partitioning `streak_k_state` uses, since a tracklet_id is only unique within
its own camera's detector). Masks themselves are decoded and cached per
(camera, frame) for the duration of one step's resolve call — same "not cached
across frames, one full mask retained at a time" memory posture
`_make_dot_mask_filter`'s own `dilated_mask()` already documents and justifies
for the same data shape.

CLI: `--seg-run-id` flag on `posetrak-tracker track`, threaded down next to the
existing `--trace-assignment-start/-end` flags; absent/empty disables the whole
mechanism (matches every other optional mechanism in this file — a
default-constructed `SegMaskConfig` is a no-op).

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
3. Does `_make_dot_mask_filter` in `finalise.py` stay as-is (a separate, coarser,
   opt-in pre-filter someone might still use without ever running a masked
   tracking pass), or should it eventually be superseded by this mechanism?
   Leaning toward "leave it, it's a different tool for a different moment
   (detection-time hygiene vs. tracking-time trust)" — flagging for your
   call rather than assuming.
