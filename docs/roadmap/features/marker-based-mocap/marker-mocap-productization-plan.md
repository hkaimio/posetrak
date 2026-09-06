# Marker-based mocap — productization plan

Scope: what has to be true before `hkaimio/marker-based-mocap` merges to
`main`, and how to get there. Written after the 2026-09-06 GUI/CLI/MCP gap
analysis (see status.md's entry of the same date) confirmed that every
GUI-facing capability on Harri's list is either fully missing or missing a
layer below the GUI too — this plan sequences the work, it does not
re-argue that finding.

## 1. Merge bar

Per Harri's own framing: merge to `main` once the **architecture and
plumbing** are robust enough that finishing the feature (real UI, phase 3)
will not require a backward-incompatible change — not once the feature is
UI-complete or production-polished. Concretely, three conditions:

1. **Schema validated for the multi-subject case** (multiple objects,
   multiple persons, both together in one trial) — not necessarily *used*
   yet, but exercised once end-to-end so a phase-3 schema change isn't
   discovered after merge.
2. **CLI-complete, GUI-minimal**: the marker pipeline (detect → finalise →
   review/edit → track) is fully usable from the command line, and the GUI
   supports *at minimum* viewing a marker-based run, editing marker
   detections the same way pose keypoints are edited today, and launching
   a tracker run for a marker object. Full authoring UI (marker-body
   definition, object init) stays post-merge.
3. **Detection performance in the few-minutes range** for the reference
   capture (currently ~49 minutes for 6 cameras / 47.7k frames) — a
   49-minute loop is incompatible with iterating on GUI-exposed detection
   parameters, which is itself a merge-bar item (§2 below), so this is a
   dependency of the bar, not a separate nice-to-have.

Phase 3 (marker-aided human tracking, i.e. person + object together) is
**designed as part of this plan** (§4) but is explicitly *not* a merge-bar
item to implement — only its DB/schema implications need to be accounted
for now, per condition 1.

## 2. Detection performance

Must be solved before, or alongside, exposing detection parameters in the
GUI (§3.1) — a parameter change nobody can afford to re-run isn't
meaningfully "GUI-adjustable."

**First step is measurement, not parallelization.** `MarkerDetectionPipeline._process_camera()`
runs a full sequential decode pass per camera and, when `dot_bg_subtract`
is on, an *extra* full decode pass first to build the background model
(`_compute_dot_background()`) — two decodes of the same footage before any
detection work happens on the second one. Before assuming per-camera
parallelism is sufficient, profile a representative camera to attribute
the ~8 min/camera (49 min / 6) between: video decode, the background-model
pass specifically, contour/shape classification, and the chroma/tracklet
filters. If decode dominates, hardware-accelerated decode (or reading a
lower-resolution decode target when the dot search doesn't need full
resolution) may matter more than parallelism; if per-frame CV work
dominates, parallelism alone gets most of the win.

**Parallelization, two independent axes, and they compose.**

- **Across cameras**: fully independent (own `VideoCapture`, own
  background model, own `DotTrackletLinker`) — a `ProcessPoolExecutor`
  over `_process_camera()` calls, one process per camera, is the natural
  fit (not threads: OpenCV's own per-call GIL release is inconsistent
  enough not to rely on for this). Ceiling: `max(per-camera time)` rather
  than `sum(per-camera time)` — 6x on this capture. Below that ceiling,
  though, worker count is capped at the camera count regardless of how
  many cores the machine actually has.
- **Within a camera, across frames** (raised again — noted here rather
  than dropped, since the reasoning changed since it was last raised):
  per-frame blob *detection* itself has no cross-frame state and is
  genuinely embarrassingly parallel — but `DotTrackletLinker.link_frame()`,
  which currently runs immediately after each frame's detection, is
  inherently sequential (each frame's linking decision depends on the
  previous frame's still-open tracklets). So this can't be "run
  `_process_camera()`'s whole per-frame loop across a pool" as-is; it
  needs splitting into (a) a parallel map over frames that does decode +
  `detect_blobs()` only, producing each frame's candidate list
  independently, gathered back in frame order, then (b) a single cheap
  sequential pass running `DotTrackletLinker` over the already-computed,
  ordered candidate lists — linking itself is lightweight
  nearest-neighbor matching, not the expensive CV work, so serializing
  just that step costs little.
  One more real wrinkle: video decode is not freely random-access for a
  worker assigned "frame 500, 1500, 2500, ..." — most consumer codecs
  decode forward from the nearest preceding keyframe, and this project
  already has a documented aversion to arbitrary reseeking (the "1080
  reseeks" trap noted elsewhere in this codebase). So the right unit of
  parallel work is a **contiguous chunk of frames per worker** (each
  worker decodes forward through its own chunk, no seeking within it),
  not literally one task per frame. That still delivers the benefit
  Harri's asking for — worker count is chunk count, not camera count, so
  it can be sized to the machine's actual core count independent of how
  many cameras a given trial has (6 cameras × 1 chunk each under-uses a
  32-core machine; 6 cameras × 4 chunks each doesn't).
  Combining both axes: total parallel units = cameras × chunks-per-camera,
  scheduled onto one process pool sized to the core count, with each
  camera's chunk results gathered in order before that camera's own
  single-threaded tracklet-linking pass runs.

**GPU**: worth splitting by algorithm, since the two detectors are not
equally GPU-friendly. Dot detection's core operation (threshold a
residual, find local blobs) is simple and local — plausibly portable to a
GPU kernel (OpenCV's CUDA module accelerates thresholding but not
`findContours` directly, so a real port likely means a custom connected-
components or local-maxima kernel, not a drop-in `cv2.cuda` swap). ArUco
detection (adaptive thresholding → contour finding → quad fitting →
perspective correction → dictionary matching, all fairly branchy) is a
worse GPU fit and has no standard turnkey GPU implementation to reach
for — expect it to stay CPU-bound regardless. The more universally
useful GPU win is likely **hardware-accelerated video decode** (NVDEC via
ffmpeg) ahead of either detector — it helps both algorithms equally and
doesn't require rewriting either one, and this development machine
already has a CUDA toolkit installed, so it's a real option here, not a
hypothetical. Worth a line item once profiling (above) shows how much of
the ~8 min/camera is decode vs. CV work — if decode dominates, GPU decode
alone could beat CPU parallelism on its own.

## 3. CLI-complete, GUI-minimal

### 3.1 Detection: expose in `RunDetectionDialog`

Add `dot_bg_subtract`/`dot_threshold`/`dot_max_saturation`/
`dot_bg_sample_count` and the per-camera dot-detection toggle to the
existing object-detection path in `RunDetectionDialog` — the same fields
`run_object_marker_detection.py` already exposes on the CLI side (added
2026-09-06). No new backend plumbing; `load_pipeline_for_capture_object()`
already accepts every one of these.

### 3.2 Dot review/edit: extend `ObjectCropGridWidget`

`ObjectCropGridWidget._load_observations()` currently hardcodes
`primary_source="markers"` (ArUco corners only) — dots are invisible in
the GUI today; every dot review this whole project has done went through
the standalone `render_tracking_debug_frames.py` script.

The real design question here isn't wiring — `read_observations_with_edits()`
is already source-agnostic — it's a genuine mismatch in *editing model*.
ArUco corners (and pose keypoints) are **fixed-width, named slots** per
frame: "corner 2 of aruco_3" is a stable index a drag-to-correct edit can
target via `pose_observation_edits`. Raw dot *candidates* are
**variable-N per frame with no persistent identity** until the tracker's
own assignment resolves one against a named marker slot (`dot3`, etc.).
Two different things could be meant by "edit marker detections the same
way as pose keypoints," and they have different costs:

- **Edit the resolved, per-marker-slot observation** (post-assignment: "the
  candidate the tracker picked for `dot3` this frame is wrong/missing") —
  this is the direct analogue of today's corner editing, same fixed-width
  shape, same `pose_observation_edits` mechanism, and the recommended
  scope for the merge bar. Requires an edit-time source of "what did
  assignment pick for this slot" (`tracking_obs_results`, or re-running
  `resolve_dot_assignment()` at review time) rather than raw
  `pose_observations` alone, since raw dot rows carry candidates, not
  resolved markers. This bucket also covers **manually placing a dot the
  detector missed entirely** (Harri — e.g. too much motion blur for
  `detect_blobs()` to find anything at all): writing a fresh value into an
  empty slot and correcting a wrong value in an occupied one are the same
  `pose_observation_edits` write, just starting from "nothing there"
  instead of "something wrong there" — no separate mechanism needed, only
  a UI affordance to click-and-place on empty space instead of dragging an
  existing point.
- **Relabel/add/remove raw candidates before assignment** ("this blob the
  detector found actually is `dot3`, don't discard it as noise") — a
  materially harder, separate feature: no fixed slot to drag, needs new UI
  affordances (click a candidate, assign a label) and a new edit
  representation entirely. Valuable, but should be scoped as a follow-on
  after the merge bar, not blocking it.

Recommend the first (resolved-slot editing, including manual placement)
for the merge bar; the second stays explicit future work so it isn't
silently dropped. Harri confirms both are useful and this ordering is
right.

**Tracklet-assisted propagation** (Harri): when the user fixes one frame's
assignment ("this is `dot3`"), and that frame's raw candidate carries a
`tracklet_id`, the same label can be propagated automatically to every
other candidate sharing that `tracklet_id` in the same camera — since a
tracklet already represents one physically-continuous blob across a
contiguous run of frames (per `DotTrackletLinker`), one correction could
fix an entire mislabeled segment instead of one frame. Real value, but
needs one safety check before it's trustworthy: tracklet linking can
itself mis-link two different real dots into one tracklet (e.g. two
streaks passing close together) — blindly propagating a label across the
whole tracklet would then silently mislabel frames that were never
actually the corrected dot. Recommend surfacing propagation as a
previewed suggestion (highlight which frames/timerange would change,
user confirms) rather than a silent blanket apply, at least until real
usage shows mislinked tracklets are rare enough to trust by default.

### 3.3 Tracker config fields in the GUI

`TrackerConfigWidget` is a hand-authored, per-field form (`run_tracker.py`)
— every one of this project's `dot_streak_*`/`dot_assignment_gate_*`/
`dot_tracklet_gate_multiplier` fields (and `outlier_threshold` was the only
one already wired) requires an explicit new form control to become
GUI-reachable, on top of the already-established five-touchpoint pattern
for adding the field itself (CLAUDE.md). That's a real, recurring tax on
every future dot-tuning parameter. Recommend a generic **"Advanced" tab**
driven by `PRAGMA table_info(tracker_configs)` (or an equivalent schema
introspection) that surfaces any column with no dedicated widget as a
plain labelled numeric/text field — not a replacement for hand-authored
widgets where a field deserves real UI (units, validation, a slider), but
a safety net so a new tunable is never *invisible* to the GUI, only
possibly under-designed. Scope: at minimum for the merge bar, the current
dot-tuning fields must be reachable somehow, even if only via this
generic fallback.

### 3.4 Object tracking launch

Already works (`ObjectPanel` → `ObjectRunTrackerDialog`, built and
functioning) — no gap here for the single-object case, which is what the
merge bar requires. Multi-subject launch (person + object together) is
explicitly Phase 3 scope (§4), not merge-bar scope.

## 4. Phase 3 design: marker-aided human tracking

Per `marker-mocap-design.md` §7's phasing table, phase 3 is "markerless
person + one prop (ArUco or dot) together" — track binding for multiple
subjects including objects, grip anchor points, contact gating tuning for
the prop case. Investigating the current code for this plan surfaced two
findings that change how risky phase 3 actually is:

**Encouraging**: the C++ side already looks subject-kind-agnostic at the
data-structure level. `PersonSpec` (`multi_person_tracker.hpp`) is just
`{sequence_id, skeleton_id, config_id, person_id}` — nothing requires a
person's skeleton specifically, and the `--person` CLI flag on
`posetrak-tracker track` could already load an object sequence into a
multi-subject run mechanically. The `cross_person_*` config fields
(`cross_person_max_world_mm`, `cross_person_min_confidence`,
`cross_person_max_n`) are generic distance/confidence/count parameters,
not anatomy-named (no hardcoded "hand"/"wrist" joint lookup found in
`multi_person_tracker.hpp`/`config.hpp`) — the grip-anchor mechanism looks
built to be marker-pair-generic already, which is exactly what phase 3's
own design note assumes ("reusing the existing Stage-2 mechanism").

**Not yet validated**: nobody has actually run a person + object trial
through this path. `tracking_run_persons` already has the right shape for
it (`PRIMARY KEY (run_id, person_id)`, optional `capture_object_id`
per-row) — so this is very plausibly a **validation task, not a schema or
tracker-code change** — but "plausibly" is exactly the risk this plan's
merge bar is meant to close out before merge. Concretely: what a
person-then-prop trial's `cross_person` anchors mean when one side is a
rigid body with no joints of its own hasn't been exercised at all; the
`dot`-specific assignment work (Phase B, this session) explicitly notes
prop-dot labeling stays single-body-scoped "even with a person present"
at phase 3 — meaning phase 3 doesn't yet ask the dot-assignment machinery
to do anything new, only the *tracker-level* subject coupling is new
ground.

**Phase 3 work breakdown** (design, to build after merge):

1. **Validate the mechanical path first**: construct a real (or synthetic)
   trial with one tracked person sequence and one tracked object sequence,
   run both through `MultiPersonTracker` via `--person` flags, confirm
   `build_cross_person_anchors()` produces sane anchors for a rigid-body
   subject and doesn't assume every subject has a comparable joint set.
   This is the schema/plumbing validation item from §1 condition 1 — do it
   as part of merge-bar work even though phase 3's UI/UX doesn't ship yet.
2. **Grip anchor points** — the design doc calls for these as a new
   concept (a marker or point on the object model representing "where a
   hand grips it") distinct from ordinary markers. Harri's own pushback is
   right: this needs real thinking and experimentation, not just a schema
   field, because a grip isn't static — it can slide/change during
   tracking (a two-handed regrip mid-swing), and the object itself can
   leave one subject's coupling entirely (thrown, passed to a second
   person, set down). Breaking that down:
   - *Where a grip point lives in the data model* is still genuinely
     open (a new marker role on the object, vs. metadata on an existing
     marker) — no answer yet.
   - *Whether the coupling itself needs to be dynamic* looks less novel
     than it first appears, on a closer read of the existing code:
     `update_contact_gate()` already re-evaluates the active contact set
     every step from current FK marker positions, not from a fixed
     binding decided once at init — so "gripped" vs. "not gripped" is
     already a per-step gate, not a static relationship, which is the
     right shape for a grip that changes or breaks. What that mechanism
     has never been asked to handle is the two genuinely new cases a
     prop introduces: **free flight** (thrown or set down — coupled to
     *nobody* — does the object's own process model produce sane motion
     with the grip constraint absent, or does it implicitly assume a
     coupling always exists?) and **handoff ambiguity** (a brief window
     where two subjects' grip gates could plausibly both be active, or
     neither is, while the object changes hands). Both need real-data
     experimentation against an actual throw/pass/regrip capture before
     trusting the existing mechanism as "just reuse it."
3. **Contact gating tuning for the prop case**: the existing contact-gate
   mechanism (`update_contact_gate()`) was built and tuned for
   person-person contact; whether its thresholds/timing generalize to a
   person-prop grip (very different contact geometry and duration
   profile — a held sword vs. a mutual grab) needs real-data tuning, not
   just enabling the existing mechanism. This is the same tuning work
   point 2's free-flight/handoff cases will also need, not a separate
   pass.
4. **GUI**: `RunTrackerWidget`'s roster is hardcoded to `capture_persons`
   rows (`run_tracker.py`) — needs to accept an object row alongside
   person rows, or `ObjectRunTrackerDialog` needs to grow into (or be
   superseded by) a shared multi-subject launcher. `ObjectRunTrackerDialog`'s
   own docstring already anticipates this ("a future trial-level launcher
   ... ends up calling into this same machinery from a different entry
   point") — the execution path is meant to be shared already, only the
   dialog/roster UI needs building.

## 5. Object initialization

This is a real, structural asymmetry the productization plan needs to
solve, not a corner case. A person has two independent, already-working
init mechanisms: **offline track-stitching** (`PoseExtractionWindow`
assigns per-frame YOLO detections into person tracks across the whole
trial before finalising — an identity problem solved from data already in
hand) and **live segmentation** (`CutieSegmentor`: YOLOX detects a person
box in one frame, SAM2 turns the box into a mask, Cutie propagates that
mask across the trial; the mask then scores keypoint detections
inside/boundary/outside/unavailable via `_score_keypoints()`). Both work
*because* "a person" is a supported detector class with a coherent,
trackable visual blob. An arbitrary marker-carrying prop has neither
property in general.

**Two init signals, not mutually exclusive, ranked by how much new work
each needs:**

1. **ArUco-anchored init (already built, should stay the default)**: any
   object carrying at least one ArUco tag gets a full 6-DOF rigid pose
   directly from that tag's own corners — no segmentation, no detector,
   no new work at all. This is the mechanism this entire project has used
   for the sword capture and should remain the first-choice path whenever
   the object has a coded marker at all. It doesn't help a dot-only prop,
   which is exactly the case §5.2 below targets.
2. **Cutie-segmentation-assisted init, for dot-only or partially-visible
   objects**:
   - **Near-term (infrastructure already exists, low risk)**: `CutieInitPanel`'s
     manual click-seed workflow (`init_mask=labeled_mask_from_ui`, which
     already bypasses YOLO+SAM entirely) does not know or care what class
     of object it's segmenting — Cutie propagates whatever mask it's
     seeded with. A person clicking the prop once in one frame, then
     letting Cutie propagate across the trial, should work today with
     **no new segmentation code**, only a GUI path that lets an object
     (not just a person) be the subject of a `CutieInitPanel` session and
     stores the resulting per-frame mask against the object's sequence
     instead of a person's. This is the recommended first step — it's
     nearly free relative to the alternative below, and validates whether
     mask-based dot cueing (next point) is even worth pursuing before
     investing further.
   - **Longer-term / stretch, only if manual seeding proves to be a real
     per-trial bottleneck**: automatic init via a text-prompted
     open-vocabulary detector (e.g. Grounding DINO or similar), driven by
     a free-text description field added to `marker_body_definitions`
     ("a wooden sword", "a black remote control"), producing a box that
     feeds SAM2 exactly the way YOLOX's person box does today — only the
     box source changes, everything downstream (SAM2 → Cutie → mask) is
     reused unmodified. Do not start here; the manual-click path already
     removes the "no object detector" blocker at near-zero cost, so this
     is only worth building once real usage shows manual seeding is
     actually the bottleneck, not a hypothetical one.
3. **Using the resulting mask as a dot-assignment cue** (Harri's second
   idea) — directly reuse the existing `_score_keypoints()`
   inside/boundary/outside/unavailable pattern, but score raw dot
   *candidates* against the object's per-frame Cutie mask instead of
   scoring named person keypoints against a person mask. Feed that score
   into `resolve_dot_assignment()`'s cost matrix as another
   cost-relaxation term, composing with (not replacing) the
   `dot_tracklet_gate_multiplier` mechanism just built for Phase B — a
   candidate outside the mask gets penalized, one inside gets relaxed,
   the same way a tracklet match relaxes cost today. This also gives a
   genuinely new, independent init-time signal for the very first frame,
   before any tracklet history exists — directly addressing the
   documented pain of manually scanning for a valid init window
   (`status.md`'s repeated "brute-force `--start-time` scan" entries).
   **Real, non-obvious risk to validate before trusting this as a hard
   filter**: a thin prop (a sword blade) is a much harder video-object-
   segmentation target than a person's whole-body silhouette — Cutie's
   mask accuracy on a thin/fast-moving object needs checking against real
   footage before this is used for anything stronger than a soft
   cost-relaxation term (never a hard reject).

These two are a starting point, not the full answer — Harri expects other
init signals will be needed too (not yet identified; add them here as
they come up rather than treating this list as closed).

**Re-initialization after full occlusion** (object leaves every camera's
frame and returns) is a distinct, deeper problem this plan does not
resolve: `Tracker::initialize()`/`initialize_with_fixed_root()` are
documented "call at most once per Tracker instance... re-initialize by
constructing a new Tracker instead" — there is no concept today of one
logical tracking result spanning multiple disjoint tracked segments. Both
init signals above (ArUco and mask-based) help find a *first* valid
window, but reappearance-after-loss needs its own design: likely a
tracking-run-level orchestration layer that detects a sustained loss,
constructs a fresh `Tracker`, re-initializes from whichever signal is
available at the reappearance point, and stitches the resulting segments
into one reported result. Flagging as a known gap for a future design
pass, not solving it here — it's out of scope for the merge bar and not
required for phase 3 either (phase 3's own scope assumes a person and
prop that stay in frame together).

## 6. Sequencing

1. **Profile + parallelize detection** (§2) — unblocks iterating on
   everything else without a 49-minute tax per change.
2. **Multi-subject schema validation** (§4 point 1 only — the mechanical
   path, not the rest of phase 3) — cheapest way to retire the "will this
   need a backward-incompatible change" risk the merge bar cares about
   most.
3. **CLI-complete, GUI-minimal** (§3) — detection params in
   `RunDetectionDialog`, resolved-slot dot editing in `ObjectCropGridWidget`,
   an Advanced-tab fallback for tracker config fields.
4. **Merge to `main`** once 1-3 are done and 2's validation found no
   schema change was needed (or the needed change has already been made,
   not just identified).
5. **Post-merge**: object-initialization work (§5, starting with the
   manual-click Cutie path since it is nearly free), phase 3's remaining
   items (grip anchors, contact-gating tuning, multi-subject GUI launcher),
   then the full authoring UI (marker-body definition/calibration,
   raw-candidate relabeling) before publishing the feature.

**Parallel track, not sequenced against 1-4 above**: Harri already has
real capture material for two future phases — (a) multiple objects with a
person, (b) a person wearing markers to augment person tracking — and
intends to start prototyping against both soon rather than waiting for
the merge bar to close first. This doesn't conflict with the sequencing
above: prototyping is investigative (following this project's own
established "prototype script, never touching production code until
validated" pattern), and for (a) specifically it's a natural way to
actually perform the phase-3 mechanical-path validation §4 point 1 and
condition 1 of the merge bar call for — real material, not synthetic,
is a better validation than one built from scratch for the purpose. (b)
is genuinely new ground this plan hasn't scoped at all (UC2, per
`marker-mocap-design.md`'s own phasing — markers *on* a person rather
than a standalone prop) and will need its own design pass once
prototyping surfaces what it actually needs.

## 7. Open questions

1. Where do "grip anchor points" live in the data model — a new marker
   role on the object's skeleton, or metadata attached to an existing
   marker? Not designed yet (§4 point 2) — open to Harri too, not just
   unresearched.
2. Does the background-model computation in dot detection actually need
   its own full decode pass, or can it share the main pass (e.g. sample
   background frames lazily from the same decode loop rather than a
   separate up-front pass)? Harri: the sampled frames likely need to
   span a wide enough time range to be a representative background
   (a handful of frames from early in the pass wouldn't be), which cuts
   against simple lazy sharing — computing the background needs to have
   *seen* the back of the range before any frame near the front can be
   detected against it, so a true single-pass design has a real
   chicken-and-egg problem, not just an implementation inconvenience.
   Likely conclusion: keep two passes, but make the background pass
   itself cheap (lower-resolution decode, or a coarser frame stride)
   rather than try to eliminate it — still worth confirming against
   §2's profiling result before committing to that.
3. Should resolved-slot dot editing (§3.2) read from `tracking_obs_results`
   (requires a tracking run to already exist) or re-run
   `resolve_dot_assignment()` live at review time (works without a prior
   tracking run, but duplicates assignment logic into the review path)?
   Related to the tracklet-propagation idea in §3.2 (both touch how much
   assignment logic the review UI needs to know about) — needs real
   experimentation against both approaches before deciding, not a desk
   decision.
