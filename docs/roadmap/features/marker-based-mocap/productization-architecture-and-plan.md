# Marker-based mocap — productization architecture and plan

**Status**: proposal, 2026-09-18; review round 1 folded in 2026-09-19
(Harri's inline comments are kept with a `> **Resolution**` block under
each; §8 has the summary table). Written against
`hkaimio/marker-based-mocap` at `d2e411d` plus the uncommitted work
listed in §1.3. Supersedes the *sequencing* sections of
[marker-mocap-productization-plan.md](marker-mocap-productization-plan.md)
(§6) and
[person-marker-mocap-productization-plan.md](person-marker-mocap-productization-plan.md)
(§4); their technical content (dot review editing model, Advanced config
tab, calibration trial, module registry) is referenced, not repeated.
Starting point for the facts: [productization-context-summary.md](productization-context-summary.md).

**Revision, 2026-09-19 (WS1 re-scope).** Reading the code before starting WS1
changed part of the design. The original text is kept unchanged in place and
marked with a `> **Revision**` block where it no longer describes what is
built; [§10](#10-revision-log-ws1-re-scope) records each change with the
original design, the reason, and when to reconsider.

The document answers three questions:

1. What is the **target architecture** that the four real cases already
   tracked on this branch (ArUco prop, ArUco+dot prop, person with a
   16-dot leg module, single-dot ball from an external 2D track) fit
   into without bespoke scripts — §3.
2. What **decisions** have to be made now so that the rest can land as
   additive increments on `main` — §2.
3. In what **order** the work lands, and what each step is validated
   against — §5, §6.

---

## 1. Where the branch stands

### 1.1 Production code that is already general and tested

| Layer | What exists | Tests |
|---|---|---|
| Schema (session v46→v53, registry v8→v12) | `capture_objects`, `detection_runs.detector_type/config_json/capture_object_id`, `pose_sequence_keypoints` manifest, `tracking_run_persons.capture_object_id`, dot/streak/gate/confidence-override columns on `tracker_configs`. Every migration additive; a session with no markers behaves as before. | `python/tests/db/*` |
| Detection (Python) | `dot_blob_detector.py` (round + streak, subtract/blacklist background, chroma filter), `dot_tracklet.py` (`MotionGatedLinker`), `marker_pipeline.py` (ArUco + dots per camera, `run_parallel`) | `python/tests/detection/*`, `test_marker_pipeline.py` |
| Definitions | `marker_body_to_skeleton.py`, `posetrak marker-body` CLI, `manage_capture_object.py` | `tests/skeleton`, `tests/cli`, `tests/db` |
| Finalisation | `finalise_object_to_db()` (object-bound detection run → sequence + manifest) | `test_finalise_object.py` |
| Tracker (C++) | `input_tracks`/`track`/`landmark`/`normal` on markers; `Skeleton::is_rigid_body()`; rigid Kabsch init incl. single-marker; init-window search; `Tracker::predict_step()/update_step()` split; `MarkerPrediction` seam with rigid closed form and articulated sigma-point implementation (batched, OpenMP); `resolve_dot_assignment()` (Hungarian, shared across subjects, tracklet gate relaxation, streak noise/velocity); three-pass wiring in both `run_track_from_db()` and `MultiPersonTracker::run()`; `load_unlabeled_candidates()`; `--seed-position`, `--subject-seed`; `get_in_range()` binary search | `test_dot_assignment.cpp` (761 lines), `test_marker_prediction`, `test_tracker_integration`, `test_tracker_predict_update_split`, `test_session_reader`, … |
| GUI | `CaptureObjectsSection`, `ObjectPanel`/`ObjectCropGridWidget`, `ObjectRunTrackerDialog`, `StandaloneRunPanel`, marker mode in `RunDetectionDialog`, objects as Cutie targets | `python/tests/app/*` |

### 1.2 What is validated only through scripts

Everything that turns raw data into a *tracked person with markers*, and
everything that got the ball tracked: cross-camera tracklet grouping,
manual tracklet-group labeling, attachment-set fitting and refitting,
skeleton augmentation, person-dot finalisation, rigid-body calibration
from footage, external 2D track import, and every diagnostic renderer.
See §4 for the full triage of the ~60 scripts in `python/tools/`.

### 1.3 Uncommitted work on the branch as of this writing

Modified: `dot_tracklet.py`, `marker_pipeline.py` and their tests,
`label_dot_ground_truth.py`, several prototypes. Untracked: 35 tools
(tracklet grouping, GT labeling GUIs, Blender helpers, ball prototypes,
moving-box extrinsics calibration) and three design docs. These need to
be committed or archived before anything else (§5, WS0).

### 1.4 Re-assessment of the existing merge bar

The prop plan's bar was: (1) schema validated for the multi-subject case,
(2) CLI-complete and GUI-minimal, (3) detection in the few-minutes range.

- (1) is **met**: person + pen + pad ran through `MultiPersonTracker` on
  2026-09-14 with no schema change; the `MarkerPrediction`, `input_tracks`,
  manifest and shared-assignment seams held unchanged across all four real
  cases, including the single-marker ball, which was the most different.
- (2) is **partly met**: prop detect → finalise → track works from the CLI
  and the GUI; person-worn dots and external tracks are script-only; dot
  review has no GUI.
- (3) is **partly met**: 2.8× from camera-level parallelism; decode
  threading unresolved.

Harri's own framing of the bar was "plumbing robust enough that finishing
the feature won't need a backward-incompatible change". Everything still
missing from (2) and (3) is additive on top of that plumbing. §2 D1
therefore recommends merging the foundation now and delivering (2) and
(3) as increments on `main`.

---

## 2. Decisions

Each decision is the load-bearing call behind one part of §3. They are
listed together so Harri can accept or overturn them in one pass.

**D1 — Merge the foundation now; finish on `main`.** `main` is one commit
ahead of the branch point, so conflict cost is minimal today and grows
with every week. The branch carries a fix every tracking run benefits
from (`get_in_range()`, 770×) and every change is gated behind "no
`input_tracks:` → identical behaviour". Merge after WS0 hygiene (§5), not
after the GUI is complete.

We need some branch from which to create bug fixes to the released prototype. Currently that is main. We can branch `rel-proto` from current main and merge this to main, or keep main as release branch for now and continue development here. Before merging to main I'd like to do some cleanup, e.g. there is some test data and possibly references to my local setup in this branch.

> **Resolution**: branch `rel-proto` from current `main`, then merge this
> branch into `main` and keep `main` as the development trunk. Keeping
> `main` as the release branch instead would mean every workstream below
> lives on a long-running feature branch for months, which is the exact
> divergence cost D1 exists to avoid; a release branch that only ever
> receives cherry-picked fixes is the cheaper side of the trade.
> `rel-proto` is cut from the *current* `main`, so the released prototype
> is unaffected by the merge.
>
> **Cleanup scope, measured rather than assumed** (scan run 2026-09-19):
> local-path references in tracked files are (a) the `catalog/*.calibrated.*`
> YAML provenance blocks, which leave the repo anyway under §4, and
> (b) `--usage` examples in ~20 tool docstrings, cosmetic. The only three
> hits outside `python/tools/` and `catalog/` are a build-command comment
> in `packaging/windows/posetrak.iss`, a help-text example in
> `python/posetrak/cli/trial.py`, and a fixture string in
> `python/tests/cli/test_extrinsics_rig.py` — none is real coupling to
> your machine, and all three predate this branch. Tracked test data added
> by this branch: none above 200 KB. So the cleanup is the §4 triage plus
> a docstring pass, not a hunt. WS0 item 2a.

**D2 — One `pose_observation_sequence` per subject per trial stays the
atomic tracking input for *labeled* observations; anonymous candidate
pools are scene-level and referenced, not copied.** Labeled sources
(`body`, `hand_*`, `markers`, and a labeled external track) resolve to a
named slot at load time, belong to exactly one subject, and live as rows
inside that subject's sequence — as all four real cases already do. An
*anonymous* pool (`dots`, and an anonymous external track) has no subject
until the tracker resolves it, so it stays one shared pool per detection
run and each subject's sequence merely references it. See the D3 comment
below for why this distinction is load-bearing rather than cosmetic.

The base design's alternative of binding several sequences to one
skeleton via `--track <input_track>:<sequence_id>`
(marker-mocap-design.md §5.1) is still dropped: it adds a second binding
mechanism for the labeled case, which no case needs. `Marker::track`
keeps its meaning as the *type* of source a marker reads (`coco133` /
`labeled_points` / `unlabeled_points`).

**D3 — Sequence composition becomes a first-class, provenance-recording
operation.** Today three bespoke scripts (`copy_dot_candidates_to_sequence`,
`finalize_ball_blender_detection`, `setup_pen_pad_capture_objects`) each
hand-assemble a sequence from detection runs. Replace them with one
operation — "compose subject sequence from these sources" — and a
`pose_sequence_sources` table recording which detection run contributed
which `source`, whether it was copied or referenced, and a per-source
trust/noise value. §3.5.

Harri: my understanding is that unlabeled marker (reflective dot) assignment utilizes the UKF tracker predict() step. How does that fall into the proposal? I.e. if a unlabeled markers from same detection run are used to track multiple objects.

> **Resolution: you found a real flaw in the first draft of D3, and it
> changes the design.** Checked against the code rather than reasoned
> about.
>
> How it works today: assignment runs *between* predict and update.
> `MultiPersonTracker::run()` (multi_person_tracker.cpp ~1330-1380) calls
> `step_person_context_predict()` on every dot-bearing subject, then
> concatenates each subject's own candidate list into one pool per camera
> (`combined_candidates`), runs one Hungarian solve per camera across
> every subject's predicted slots, and hands each subject its resolved
> share to update with. The mutual exclusion — one physical dot can be
> claimed by at most one subject — is the whole point of solving jointly.
>
> The flaw: candidates are loaded **per sequence**
> (`load_unlabeled_candidates(sequence_id)`, and
> `PersonContext::unlabeled_candidates` is per subject). So if the first
> draft's "compose by copy" put one scene-wide dots run's rows into two
> subjects' sequences, the concatenation would produce **two distinct
> rows for one physical dot**, and the solver would cheerfully give one
> copy to each subject. Mutual exclusion defeated, silently. This is
> exactly the prerequisite dot-assignment-architecture-design.md §5.4
> flags ("candidates must be a single shared pool, not N redundant
> per-subject lists"), and the code carries a comment saying the naive
> concatenation is a known, deferred limitation. Copying would turn that
> latent limitation into a guaranteed bug the moment a second dot-bearing
> subject exists — which W2-with-two-people and person+dotted-prop both
> are.
>
> **The fix, and it is a simplification**: anonymous pools are referenced,
> not copied (D2 above). `pose_sequence_sources` gets a `mode` of
> `'copy'` or `'reference'`; an anonymous source is always `'reference'`
> and writes no `pose_observations` rows. The orchestrator loads one pool
> per distinct detection run and shares it across every subject
> referencing it, so one physical dot is one row in the cost matrix by
> construction. Per subject, what remains is which pool(s) it
> participates in plus an optional camera filter.
>
> This also *delivers* §5.4's real fix at the data-model level rather than
> deferring it: the remaining duplication case is two different detection
> runs over the same footage, which composition now makes avoidable by
> construction (reference the one scene-wide run from both subjects). No
> de-dup bridge needed.
>
> The predict step itself is unchanged and stays per subject: every
> dot-bearing subject predicts its own slots, then one solve arbitrates
> the shared pool. Composition changes only where candidates come *from*,
> never when they are predicted or resolved. §3.5.1 has the mechanics.

> **Revision (2026-09-19): the schema half of D3 is deferred, and the
> mechanism that prevents double claims is different.** The target above is
> kept as the fuller design; what is built is smaller. *Not built now:*
> `pose_sequence_sources`, `mode='reference'`, the detection-run-keyed loader,
> orchestrator-owned pools, and `sequence compose`. *Built instead:* when the
> orchestrator merges subjects' candidate lists it merges byte-identical
> candidates into one and records which subjects' sequences hold it (candidate
> ownership, §10.1). Why: `pose_observations` rows already carry
> `detection_run_id` and `noise_scale`, so provenance and per-source trust are
> already stored per row; the defect D3 addressed sits in one place, the merge
> of per-subject lists; and no current case needs reference-mode storage
> savings. Accepted during WS1 planning. Reconsider when one of the triggers in
> §10.2 occurs.

**D4 — External 2D tracks are a detector type.** An imported Blender (or
any) track becomes a `detection_runs` row with `detector_type='external_2d'`
and `config_json` naming the file and its conventions, feeding
`detection_keypoints` in the `dots` blob layout. Nothing downstream
distinguishes it from automatic detection except its per-source trust
(D3). Export in the other direction is a plain per-camera CSV writer. §3.4.

Harri: Yes

> **Extended per your §3.4 comment**: the importer supports two layouts,
> chosen at import and recorded in `config_json`. *Anonymous* (the ball
> case) writes the `dots` blob layout — a reference-mode source per D2/D3.
> *Labeled* writes the fixed-slot layout plus `pose_sequence_keypoints`
> manifest rows, and is a copy-mode source like any other labeled input.
> Labeled mode carries a `label_map` in `config_json`: external track name
> → PoseTrak landmark name (`"Track.003": "hilt:c2"`, `"L_knee":
> "knee_lat_L"`), authored once per external project and reusable. An
> unmapped track is imported as anonymous rather than dropped, so a
> partially-mapped project still works.
>
> Worth building the *format* for this now even though the first real
> user is later: choosing the blob layout and carrying `label_map` costs
> nothing at import time, whereas retrofitting labels onto an
> anonymous-only importer would mean re-importing every external project.
> Labeled-mode import is a WS1 item; a GUI for authoring the map is not
> (edit the JSON until something needs better).

**D5 — Attachment sets are stored documents and are materialized into
generated skeletons.** Keep `build_dot_augmented_skeleton.py`'s model:
the C++ tracker always receives one `Skeleton`; the composed skeleton is
a derived, content-addressed `skeletons` row with provenance
(`parent_id` = base skeleton, `generated_from_attachment_set` = the set's
id). Attachment sets themselves get a registry table
`marker_attachment_sets` (content-addressed YAML, like skeletons) so
"which calibration produced this run" is queryable. Nominal catalog
modules stay YAML files in the repo (`catalog/modules/`), per the
redesign doc's own recommendation. §3.2.

**D6 — Two different assignment problems, kept apart.**
*Calibration-time labeling* (which tracklet is which named slot, before
the offsets are known) is a Python, offline, tracklet-group-level
problem; the manual-first path (redesign doc §4, B1–B5) is what worked
and is what gets productized. *Tracking-time assignment* (which candidate
feeds which slot this frame, offsets known) is the C++
`resolve_dot_assignment()` phase, already production. Automatic
calibration-time labeling (redesign doc §3) remains research behind the
GT metric harness. §3.6.

**D7 — Tracking-time assignment gets an evidence pipeline and a
reacquisition gate.** The twice-observed "confidently wrong candidate"
failure (ball; leg ankle markers on 2026-09-14) is addressed
architecturally: `resolve_dot_assignment()`'s growing parameter list is
replaced by a context struct plus a list of cost modifiers, and a slot
that has been unresolved for more than N frames may only reacquire on a
candidate corroborated by a second camera (or by tracklet continuity).
§3.6.3.

**D8 — Tracking runs are made of segments.** A run may contain several
disjoint tracked segments, each one `Tracker` lifetime. Loss is detected
automatically; re-initialization is triggered from user-editable
re-init points and whichever init signal is available. Additive schema:
`tracking_run_segments` plus a nullable `segment_index` on
`tracking_results`. Export representation of gaps is decided by a spike
against real downstream tools. §3.7.

Harri: Just to confirm: tracking runs (and thus segments) are per-object, right?

> **No — and the distinction matters for this schema.** A `tracking_runs`
> row is per *trial*, and holds a roster of one or more subjects in
> `tracking_run_persons` (`person_id` is a subject index within the run;
> `capture_object_id` is set when that subject is an object). Both shapes
> exist in real use today: `ObjectRunTrackerDialog` creates a one-subject
> run per object, while the 2026-09-14 A/B test was a single run holding
> Nelli + pen + pad. Results are keyed `(run_id, person_id)` throughout.
>
> So **segments are per (run, subject)**, not per run and not per object —
> which is what the proposed key
> `tracking_run_segments (run_id, person_id, segment_index, …)` says. That
> is the right granularity on the evidence: in a person-plus-thrown-prop
> trial the prop needs re-initialization at each catch while the person
> tracks continuously, so a run-level segment boundary would force
> spurious re-inits on every subject that never lost tracking.


**D9 — Trust is per source and per camera, not only per marker.**
`pose_sequence_sources.noise_std` (D3) feeds `Observation::noise_std_override`
for every row of that source; a `dot_camera_noise_scale` JSON map on
`tracker_configs` scales a camera's dot candidates. Both default to
"no change". §3.9.

> **Revision (2026-09-19): the per-source half of D9 is deferred.** Per-row
> `noise_scale` is already stored, but the C++ dot loader ignores it
> (`load_unlabeled_candidates()`), and no current case needs a trusted source
> to weigh differently. Reading it is a few lines to add when a case does. The
> per-camera half (`dot_camera_noise_scale`, WS2 item 4) is unchanged.

**D10 — Person↔prop coupling defaults off for object subjects.** The one
A/B run measured no benefit and a localized jitter regression. Grip
anchors and free-flight/handoff stay a post-merge research item.

**D11 — Tools triage rule.** Anything a user needs to run one of the three
proven workflows (§3.0) becomes a `posetrak` CLI command in the package
with tests. Genuinely superseded scripts move to
`python/tools/prototypes/` with an index; a script that is still the only
working path to a real workflow **stays where it is until its replacement
exists**, and is deleted in the same change that lands the replacement.
Capture-specific artefacts (`catalog/*.calibrated.*`, dot-augmented
skeleton YAMLs) leave the repo; they are already imported into the
session DB. §4.

**D12 — Decode threading is resolved by process isolation, not by trusting
a fix.** Each camera chunk decodes in its own worker process with
threaded decode enabled; the worker returns its results *before* closing
the container; the parent joins with a timeout and terminates a worker
whose close hangs. A hung close then costs one process, never the run.
Validate with a sustained multi-hour stress run before making it the
default. §5 WS4.

**D13 — GUI editing model for dots is resolved-slot editing, reading
`tracking_obs_results`.** Requires a tracking run to exist; raw-candidate
relabeling stays explicit future work. Both already agreed in the prop
plan §3.2; recorded here as final.

**D14 — Naming follows the data model.** `PersonSpec`/`PersonContext`/
`MultiPersonTracker` are subject-generic in behaviour; rename to
`Subject*` only when a file is being rewritten anyway. No rename-only
commits.

**D15 — The skeleton-document split is a separate feature; this plan adds
only the topology identity that bounds its risk.** Splitting structure /
person metrics / marker attachment into separate documents is the right
end state and is not scoped here. What *is* scoped here is a topology
name plus a structural hash that catalog modules and attachment sets
declare and the composer validates, so a module can never be silently
applied to a skeleton it was not authored against. Full reasoning,
scope sketch and the triggers that should schedule the redesign: §9.

---

## 3. Target architecture

### 3.0 The three workflows the architecture must serve

Each is already proven once on real data; the architecture is judged by
whether each becomes a CLI/GUI workflow with no per-capture script.

**W1 — Rigid prop.** Characterize once (`marker-body calibrate` from
footage, or hand-authored YAML) → add object to capture → detect
(ArUco and/or dots) → compose sequence → track → review resolved slots →
export.

**W2 — Person with worn markers.** Attach dots per a catalog module →
detect dots scene-wide → group tracklets across cameras → label groups
to slots (GUI) → fit attachment set → augment person skeleton → compose
person sequence (pose + dots) → track → review.

**W3 — Subject from an external 2D track.** Track in Blender (or any
tool) → import per-camera CSV as an `external_2d` detection run →
compose sequence (optionally with automatic sources, each with its own
trust) → seed/init → track per segment → export, with gaps represented.

### 3.1 Layers

```
┌─────────────────────────── Definitions (registry) ───────────────────────────┐
│ marker_body_definitions   catalog/modules/*.yaml   marker_attachment_sets    │
│ (rigid geometry)          (nominal person markers)  (calibrated, per subject) │
│            └── generate ──▶ skeletons (prop skeleton | dot-augmented skeleton)│
└──────────────────────────────────────────────────────────────────────────────┘
┌──────────────────────── Capture participants (session) ──────────────────────┐
│ capture_persons                     capture_objects                          │
└──────────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────── Detection runs (session) ────────────────────────┐
│ detector_type: pose | hand | aruco | dots | external_2d | segmentation        │
│ append-only; scene-wide for dots; object-bound for aruco; file-bound for ext │
└──────────────────────────────────────────────────────────────────────────────┘
┌────────────────────────── Sequence composition (session) ────────────────────┐
│ one pose_observation_sequence per subject per trial                          │
│   rows by source ('body','hand_l','markers','dots',…) + keypoint manifest    │
│   pose_sequence_sources: (source ← detection_run, trust/noise, notes)        │
│   pose_observation_edits: human corrections on labeled slots                 │
└──────────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Tracking (C++) ────────────────────────────────┐
│ per segment: init (rigid | IK | seed | external) → per frame:                │
│   predict all subjects → shared dot assignment (evidence pipeline)           │
│   → update all subjects (contact anchors optional) → loss detection          │
│ tracking_run_segments; tracking_results(+segment_index); tracking_obs_results│
└──────────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────── Review, export (GUI/CLI) ────────────────────────┐
│ resolved-slot review/edit  tracklet-group labeling  coverage/confidence view │
│ BVH/glTF/USD (+gap policy)  external 2D CSV  Blender scene helpers           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Definitions

- **Marker body definitions** (rigid props): unchanged. `marker-body
  calibrate` (from `calibrate_rigid_marker_body.py`) becomes the CLI path
  for solving one from footage; the orbit variant
  (`calibrate_harness_from_orbit.py`) is a mode of the same command only if
  it generalizes without sword-specific code, else it stays a prototype.
- **Catalog modules** (`catalog/modules/<module>.marker-module.yaml`):
  the redesign doc §1.9 file shape (`parent_joint`, `along/lateral/
  anterior`, `normal`, `mirror`). A module loader in `posetrak/markers/
  catalog.py` replaces every `_SLOT_PARENT_BASE`-style dict. The first
  file is `leg`, written from the values the current scripts hardcode.
- **Attachment sets**: same file shape plus a `calibration:` block
  (source capture, detection run, tracking run, method, per-slot
  residuals). Stored in `marker_attachment_sets` (registry; id = content
  hash; `person_id`, `module`, `created_at`). Session DBs copy the row on
  use, like skeletons.
- **Generated skeletons**: `marker_body_to_skeleton` (props) and
  `augment_skeleton_with_attachment_set` (persons, from
  `build_dot_augmented_skeleton.py`). Both write provenance into the YAML
  header and register a `skeletons` row. Generated skeletons are never
  hand-edited.

### 3.3 Subjects

`capture_persons` and `capture_objects` are the two participant kinds; a
tracking run's roster is `tracking_run_persons` rows with a nullable
`capture_object_id`. No change. The GUI roster (`RunTrackerWidget`) must
list both kinds (WS3).

### 3.4 Detection sources

`detector_type` taxonomy, all existing or additive:

| type | scope | produces | notes |
|---|---|---|---|
| `pose`, plus hand refinement | person track | `full_body`, `hand_*` regions | existing |
| `aruco` | one `capture_object` | fixed-slot corner blobs | existing |
| `dots` | scene-wide, per camera | variable-N candidate blobs with `tracklet_id` | existing, in both object-bound and standalone (unbound) modes via scripts and `StandaloneRunPanel`; the `posetrak detect run` CLI covers pose detection only today |
| `external_2d` | file-bound | `dots`-layout blobs, one candidate per frame per track | **new** (D4); importer = productized `finalize_ball_blender_detection.py` minus the sequence-writing half |
| `segmentation` (Cutie) | person or object | masks | existing; used as evidence (§3.6.2), never as a position source |

Harri: For external_2d, I think the format should support also labeled markers (likely with some kind of mapping tom external tool labels -> Posetrak labels) At least as a future development topic.

> **Agreed — folded into D4 above** as a two-layout importer with a
> `label_map` (external track name → PoseTrak landmark name) in
> `config_json`. Anonymous stays the default; labeled mode is a WS1 item
> so the wire format is settled before external projects accumulate.

Detection runs stay append-only. The dots detector gets the parameter
surface (`dot_bg_subtract`, thresholds per camera, `max_saturation` per
camera, camera subset) on both `posetrak detect run` and
`RunDetectionDialog`.

> **Revision (2026-09-19).** `posetrak detect run` now covers marker runs:
> `--type pose|aruco|dots` (§10.2 explains the name), object-bound with
> `--object` or unbound with `--marker-ids`, and the whole dot parameter
> surface including per-camera overrides and `--dots-camera` as the camera
> subset. `RunDetectionDialog` still lacks the dot parameters. The scripts
> `run_object_marker_detection.py` and `run_standalone_marker_detection.py`
> are deleted.

### 3.5 Sequence composition

New module `posetrak/db/compose_sequence.py`, CLI `posetrak sequence
compose`, GUI dialog on the person/object panel.

Input: subject (person or object), trial, and a list of
`(detection_run_id, source_filter, trust)` entries. Output: one
`pose_observation_sequence` (new row, or refuse if the target has
tracking runs/edits, same guard `finalise_to_db` already enforces),
`pose_sequence_keypoints` rows for labeled sources, `pose_observations`
rows for **copy-mode** sources only, and one `pose_sequence_sources` row
per contribution:

```sql
CREATE TABLE pose_sequence_sources (
    sequence_id       TEXT NOT NULL REFERENCES pose_observation_sequences(id),
    source            TEXT NOT NULL,          -- 'body','hand_l','markers','dots',…
    detection_run_id  TEXT NOT NULL REFERENCES detection_runs(id),
    mode              TEXT NOT NULL,          -- 'copy' (labeled) | 'reference' (anonymous pool)
    camera_filter     TEXT,                   -- JSON list of camera_instance_ids, NULL = all
    noise_std         REAL,                   -- per-source noise_std_override, NULL = default
    notes             TEXT,
    PRIMARY KEY (sequence_id, source, detection_run_id)
);
```

`finalise_to_db()` (person stitching) and `finalise_object_to_db()`
become thin callers of the same composer: the person path copies
`body`+`hand_*` from one pose run; the object path copies `markers` and
references `dots` from one object-bound run. The person-with-dots path
copies `body`+`hand_*` from a pose run and references a scene-wide dots
run. The external path references an `external_2d` run, optionally also
referencing an automatic run restricted to named cameras
(`camera_filter`) at lower trust — exactly the mix the ball needed,
minus the script.

`SessionReader` reads `pose_sequence_sources.noise_std` and applies it
as `noise_std_override` to every row of that source when set. That is
the whole C++ change for D9's per-source half.

> **Revision (2026-09-19): `sequence compose` is not built now.** Its job is
> covered by narrower commands, one per real workflow, added as separate pull
> requests: `capture object add/list/rename/rm`, `sequence finalise-object`
> (wrapping `finalise_object_to_db()`), and `sequence add-dots` (the
> `copy_dot_candidates_to_sequence.py` script moved into the package, with the
> source run stamped correctly). The `pose_sequence_sources` table above is
> deferred with D3.

#### 3.5.1 Anonymous pools are shared, not copied (D2/D3)

The mechanics behind the D3 resolution:

- **Loading.** A reference-mode source names a detection run, so
  candidates load once per `(trial, detection_run)` rather than once per
  subject. `load_unlabeled_candidates()` gains a detection-run-keyed
  sibling; the existing sequence-keyed overload stays for legacy
  sequences that have dot rows physically copied in (detected by "this
  sequence has no reference-mode source rows"), so nothing already in a
  session DB has to be migrated.
- **Ownership.** `PersonContext::unlabeled_candidates` /
  `unlabeled_candidates_by_camera` move out of the per-subject context
  into an orchestrator-owned pool map keyed by detection run. What stays
  per subject is which pools it participates in, plus its camera filter.
  `bucket_candidates_by_camera()`'s binary search over a per-camera
  sorted vector works unchanged on the shared pool.
- **Mutual exclusion.** `MultiPersonTracker::run()` stops concatenating
  per-subject lists and instead passes the shared pool straight to
  `resolve_shared_dot_assignment()`. One physical dot is one column in
  the cost matrix, so the joint solve arbitrates it exactly as designed.
  The "naive concatenation" comment and the §5.4 limitation it records
  are both deleted, not deferred again.
- **Per-subject camera filters** become an eligibility view over the
  shared pool, not a separate pool: a subject's slot predictions are
  simply not produced for a camera its filter excludes, so that camera's
  candidates can never be costed against it. The ball's "external track
  on three cameras, automatic run on one" is expressible without the
  pool fragmenting.
- **Gate config reconciliation.** Today the shared solve takes its gate
  config from whichever subject happens to be first in the processing
  order (an explicit shortcut in the code, harmless while every subject
  shares a value). With a shared pool this becomes visible as a real
  question; resolve it in WS2 item 1 by taking the gate from the pool's
  own config and refusing a run whose subjects disagree, rather than
  silently picking one.
- **Storage.** A 6-camera, 47k-frame dots run stops being duplicated once
  per participating subject.

> **Revision (2026-09-19): what is built for the shared pool.** The bullets
> above describe the reference-mode design. What exists (WS1, first pull
> request):
>
> - Loading is unchanged: each subject's candidates load from its own
>   sequence, as before.
> - **Merge with ownership.** `MultiPersonTracker::run()` merges the
>   subjects' per-camera lists with `append_unique_candidates()`. Candidates
>   that agree on frame, timestamp, distorted position and tracklet id are one
>   physical detection and are merged; every candidate carries a
>   `subject_mask` of the subjects whose sequences hold it, and the assignment
>   only lets a subject claim candidates it owns. One physical dot is one
>   candidate, arbitrated jointly; a candidate only one subject holds can only
>   be claimed by that subject. This replaces both the pool-ownership refactor
>   and the camera-level "eligibility view".
> - **Gate config reconciliation** is done as written: the constructor refuses
>   a run whose dot-bearing subjects disagree on any setting the shared solve
>   reads (`find_dot_config_disagreement()`), naming the setting.
> - **Storage** is still duplicated per subject; accepted (§10.2).
> - Single-subject runs are unchanged: candidates keep the default mask (all
>   subjects).

### 3.6 Tracking

#### 3.6.1 Loop shape (unchanged; candidate source changes)

`MultiPersonTracker::run()` already does, per step: predict every
dot-bearing subject → one shared `resolve_dot_assignment()` per camera
across all subjects → update every subject. Subjects without a dot
track use the plain `step_person_context()` path. This three-pass shape
stays exactly as it is.

The one change is *where the candidates come from*: the shared pool of
§3.5.1 instead of a concatenation of per-subject lists. Predict and
update are untouched, and a single-subject run behaves identically
either way.

#### 3.6.2 Evidence pipeline for tracking-time assignment (D7)

`resolve_dot_assignment()` currently takes eleven positional parameters
and grows one per mechanism. Replace with:

```cpp
struct DotAssignmentContext {
    TrackerConfig const& config;
    int frame_idx; double timestamp;
    PrevDotPositions const& prev_positions;
    PrevDotTrackletIds const& prev_tracklet_ids;
    SlotGapCounters const& gap_frames;          // frames since slot last resolved
    std::unordered_map<int, StreakKAccumulator>* streak_k_state;
    std::vector<CostModifier const*> modifiers; // ordered
};
struct CostModifier {
    // Returns a multiplier on the squared-Mahalanobis cost (1 = no-op),
    // or +inf to exclude the pair. Pure; unit-testable in isolation.
    virtual double apply(SubjectSlotRef slot, UnlabeledCandidate const& c,
                         DotAssignmentContext const& ctx) const = 0;
};
```

Modifiers, in the order they are expected to be built:

1. **Tracklet continuity** (exists: `dot_tracklet_gate_multiplier`).
2. **Backface culling** (exists for rigid via `Marker::normal`; the
   articulated path should honour the same field once attachment sets
   carry fitted normals).
3. **Cross-camera corroboration** (new, from `prototype_ball_tracking.py`'s
   RANSAC check): a candidate whose implied 3D position is confirmed by
   another camera's candidate near the same slot's prediction gets a
   multiplier < 1; an uncorroborated one keeps 1. Cheap: only candidates
   already inside a slot's gate are tested.
4. **Per-camera trust** (D9): `dot_camera_noise_scale[camera]` applied as
   a cost multiplier.
5. **Segmentation ROI** (new, soft): candidate inside the subject's mask
   → < 1, outside → > 1, mask unavailable → 1. Never +inf, per both
   prior plans' warning about thin/fast props.
6. **Local hue class** (new, needs the cheap validation first): candidate
   whose stored surround-hue class matches the slot's learned class → < 1.
   Requires the detector to store a hue feature per candidate (additive
   blob column, versioned via `config_json`).

#### 3.6.3 Reacquisition gate (D7)

After `dot_reacquire_gap_frames` consecutive unresolved frames for a
slot, that slot's row in the cost matrix is restricted to candidates
with independent evidence: tracklet continuity with the slot's last
resolved tracklet, **or** cross-camera corroboration (modifier 3), **or**
a candidate within `dot_reacquire_max_px` of the slot's prediction in at
least two cameras. Otherwise the slot stays unresolved and the filter
coasts. This is the "never reseed from a bad guess after a gap" rule
from the 2026-09-14 finding, applied uniformly to props and persons.
Two config fields, both defaulting to "off" (gap threshold = 0).

#### 3.6.4 Initialization signals

One interface, several providers, selected per subject per segment:

| signal | subject kinds | status |
|---|---|---|
| triangulation + IK | articulated persons | existing |
| rigid Kabsch from labeled markers (ArUco) | props with ≥1 coded marker | existing |
| single-marker placement | single-dot props | existing |
| explicit seed (`--seed-position`, `--subject-seed`, `--object NAME=SKELETON@X,Y,Z`, or a re-init point's pose) | any; sufficient alone only for a single-dot body (position, identity orientation) | existing; per subject since the person + prop pull request |
| multi-dot body without a coded marker | dots-only props with several dots | **not supported**: refused with an error. Needs position *and* orientation, i.e. pairwise-distance template registration against the marker body (algorithms doc §4.1; out of scope in the dot-assignment design §9). A coded marker on the body avoids it |
| multi-view dot triangulation seed | dots-only subjects | **new**: productize the ad-hoc "triangulate a seed from whichever cameras have a sample" work from the ball case |
| Cutie manual-click mask → coarse position | dots-only props | later (prop plan §5) |

#### 3.6.5 Cost of `update()`

The Kalman-gain computation is 59% of a frame at 16 dots and unprofiled
beyond that. Not an architecture item; a gate in WS6 (profile with the
torso+arm module before designing anything). Candidate remedies if it
binds: sequential per-camera updates, or exploiting the block structure
of the innovation covariance — both are internal to `ukf.cpp`.

The dot assignment is a second cost that grows with the candidate count, not
the marker count. `solve_assignment` is O(n³) on a square matrix padded to
max(rows, cols), and it runs once per camera per tracker step. Measured on the
optimized build: 340 candidates × 240 subject slots take 17.7 ms per camera per
step, 700 × 240 take 175 ms, while the candidate-pool merge before it takes
0.07 ms and 0.3 ms. Three subjects with 80 dots each plus props and false
detections sit between those two. Remedy when it binds: gate before solving
(drop candidates that no slot's gate reaches, or split the matrix into
connected components of the gate graph); both are internal to
`dot_assignment.cpp`.

### 3.7 Segments and re-initialization (D8)

- `tracking_run_segments (run_id, person_id, segment_index, start_step,
  end_step, init_method, init_seed_json, ended_by)` and a nullable
  `tracking_results.segment_index`. Existing single-segment runs have
  NULL, meaning segment 0.
- **Loss detection** in the orchestrator: `dot_loss_frames` consecutive
  `tracking_lost` steps, or a covariance trace above a threshold, ends
  the segment and records `ended_by`.
- **Re-init points** are run inputs (`tracking_run_reinit_points (run_id,
  person_id, time_s, seed_json)`), written by the user from the GUI
  timeline or the CLI; the orchestrator also proposes them automatically
  at the first frame after a loss where an init signal succeeds, but a
  proposal is stored as a proposal until accepted (the "semi-automatic"
  requirement).
- **Orchestration** is a new layer above `PersonContext`:
  `SubjectSegmentRunner` owns the sequence of `Tracker` instances for one
  subject; `MultiPersonTracker` drives runners instead of contexts. The
  ball's "one run per throw" workaround becomes "one run, four segments".
- **Export**: BVH has no gap concept. Spike three representations (hold
  last pose, snap to rest, per-segment files + manifest) against Blender
  and one other consumer before choosing; the chosen policy is an export
  option, and the segment table is the ground truth either way.

### 3.8 Review and edit

- **Resolved-slot review** (D13): `ObjectCropGridWidget` and
  `PersonCropGridWidget` overlay `tracking_obs_results` dot slots
  (`mode==0` only, per observation-results-semantics.md) and write
  corrections through `pose_observation_edits`, including click-to-place
  on an empty slot. Tracklet-assisted propagation is offered as a
  previewed suggestion (prop plan §3.2).
- **Tracklet-group labeling** (W2): `label_tracklet_groups_gui.py`
  becomes a panel in `python/app/ui/` reading/writing DB tables instead of
  JSON files: `dot_tracklet_groups (group_id, detection_run_id, members
  JSON, stats JSON, flags)` and `dot_tracklet_group_labels (group_id,
  slot_name | 'rejected', edited_at)`. Grouping is recomputed by
  `posetrak marker-set group-tracklets`; labels survive regrouping by
  member overlap.
- **Coverage view**: per segment and per slot, cameras contributing and
  median reprojection error, from `tracking_obs_results` — the "a
  trajectory should not look uniformly trustworthy" point from the ball
  case. Table first; the crop-grid overlay later.

### 3.9 Config surface

- `tracker_configs` additions (all NULL = off): `dot_reacquire_gap_frames`,
  `dot_reacquire_max_px`, `dot_camera_noise_scale` (JSON),
  `dot_cross_view_corroboration_px`, `dot_loss_frames`.
- **Advanced tab** (prop plan §3.3) from `PRAGMA table_info(tracker_configs)`
  so none of the above needs a hand-built widget to be reachable.
- `confidence_threshold_marker_names/override` are already committed;
  they stay as the per-marker mechanism, distinct from D9's per-source and
  per-camera ones.

### 3.10 Export and import

- Import: `posetrak detect import-2d --camera <label> <csv>…` →
  `external_2d` run. Blender-side exporter stays a Blender script under
  `python/tools/blender/`.
- Export: `posetrak track export-2d <run> --source predicted|observed`
  writes per-camera CSV in the same convention, so a run can be handed to
  an external tool for correction and re-imported (the "reverse
  direction" Harri asked for).
- Rigid-body BVH/glTF export already works for a free-flyer root; the
  Blender scene helpers (`blender_add_*`, `make_blender_proxy_videos`) are
  kept as tools, not productized.

### 3.11 Invariants preserved

Detection runs append-only; one sequence per subject per trial for
labeled observations; one shared candidate pool per anonymous detection
run, never duplicated per subject (§3.5.1); `SkeletonLayout` is the only
DOF index authority; the C++ tracker sees one `Skeleton` per subject and
never reads definitions; sessions without markers are unaffected (N3);
`mode` is checked before treating `actual_x/y` as a position.

---

## 4. Code organization: tools triage (D11)

| Script(s) | Disposition |
|---|---|
| `calibrate_rigid_marker_body.py` | → `posetrak/calibration/rigid_marker_body.py`, CLI `marker-body calibrate` (test exists) |
| `calibrate_harness_from_orbit.py`, `prototype_calibrate_pen_and_pad.py` | prototypes |
| `run_object_marker_detection.py`, `run_standalone_marker_detection.py` | → `posetrak detect run --detector aruco\|dots` (unbound dots run allowed) |
| `copy_dot_candidates_to_sequence.py`, `setup_pen_pad_capture_objects.py`, `finalize_ball_blender_detection.py`, `finalize_ball_cutie_detection.py` | → `posetrak sequence compose` + `posetrak detect import-2d`. Stay in place until that lands (D11), then deleted in the same change |
| `blender_export_2d_tracks.py`, `blender_add_*`, `blender_render_scene_range.py`, `make_blender_proxy_videos.py`, `export_pen_pad_to_blender.py`, `blender_add_pen_pad_animation.py` | → `python/tools/blender/` with a README; the pen/pad pair generalized to "export object run to Blender" if cheap, else prototypes |
| `build_tracklet_groups.py`, `fit_calibrated_attachment_set.py`, `refit_attachment_set_single_camera.py`, `build_dot_augmented_skeleton.py`, `audit_cross_slot_consistency.py`, `prepare_conflict_review.py`, `reconcile_conflict_review.py` | → `posetrak/markers/` package, CLI group `marker-set` (`group-tracklets`, `fit`, `refit`, `augment-skeleton`, `audit`) |
| `label_tracklet_groups_gui.py` | → `python/app/ui/tracklet_group_panel.py` (DB-backed) |
| `label_marker_slots_gui.py`, `label_dot_ground_truth.py`, `refine_dot_ground_truth.py`, `build_gt_frame_manifest.py`, `eval_dot_detection.py`, `eval_fk_prediction.py`, `validate_dot_detector.py`, `sweep_dot_detection.py`, `sweep_linker_and_b2.py` | → `python/tools/gt/` (validation harness, redesign doc §2), kept as tools with a README |
| `render_tracking_debug_frames.py`, `render_dot_detection_overview_video.py` | → `posetrak track render-debug`, `posetrak detect render` (tests exist) |
| `annotate_dots_manually.py`, `estimate_streak_exposure_ratio.py` | keep as tools (tested) |
| `calibrate_extrinsics_from_moving_box.py`, `reframe_box_rig.py`, `spot_check_box_calibration.py` | → `posetrak extrinsics-rig from-moving-box` (WS7; belongs to the extrinsics feature) |
| `prototype_*` (dot_blob_detector, streak_detector, motion_gated_linker, fk_marker_prediction, multi_camera_fusion, shin_rigid_cluster_fit, pen_band_tracking, track_pen_trajectory, ball_tracking) | → `python/tools/prototypes/` with an index noting what each validated and which production module absorbed it |
| `prototype_person_marker_assignment.py`, `prototype_hybrid_*`, `prototype_fused_*`, `prototype_marker_normal_assignment.py`, `prototype_marker_person_filter.py`, `prototype_tracklet_smoothed_assignment.py`, `render_person_marker_assignment_video.py`, `render_hybrid_assignment_video.py`, `render_marker_normal_debug_frame.py`, `prototype_ball_cutie_segmentation.py`, `cutie_segment_ball_all_cameras.py` | superseded (redesign doc §0; ball status entries) → `python/tools/prototypes/superseded/` for the duration of the plan, deleted when the workstream that replaces each one is done (per Harri: keep during migration, remove before completing the plan). The `prototypes/` index records which workstream retires which file, so "before completing the plan" is a checklist, not a memory. |
| `catalog/*.calibrated.*.yaml`, `catalog/reallusion-no-waist.dot-augmented.*.yaml`, `catalog/ball.*`, `pen.*`, `pad.*` | move to the data drive beside the session DB; keep `catalog/modules/` only |

New package layout: `posetrak/markers/` (catalog, attachment sets,
tracklet grouping, fitting, augmentation), `posetrak/calibration/`
(rigid marker body), `posetrak/db/compose_sequence.py`,
`posetrak/detection/external_import.py`.

---

## 5. Plan

Effort legend: S ≈ 1–2 days, M ≈ 3–5 days, L ≈ 1–2 weeks, XL = needs
its own design pass first.

### WS0 — Branch hygiene and merge (L)

0. Cut `rel-proto` from current `main` (D1) so prototype bug fixes have a
   home before `main` becomes the trunk again.
1. Commit the uncommitted production changes (`dot_tracklet.py`,
   `marker_pipeline.py`, tests) as their own commits; commit the three
   design docs.
2. Apply the §4 triage: move superseded scripts to
   `prototypes/superseded/`, leave still-needed ones in place (D11), add
   the `prototypes/`, `gt/`, `blender/` READMEs, and move
   capture-specific catalog files out.
   2a. Docstring pass over the moved tools replacing machine-specific
   `--usage` paths with placeholders (D1's measured cleanup scope).
3. **Establish a green baseline before merging.** Run `./run_tests.sh`
   and `pytest python/tests/` on `main` *and* on this branch separately,
   so pre-existing failures are told apart from ones the triage
   introduced; fix both, or record any left failing with an explicit
   reason. Fix what the move breaks (imports in `python/tests/tools/`).
4. Regenerate `docs/roadmap/README.md`; give `status.md` the frontmatter
   the generator expects so the feature shows up.
5. Update CLAUDE.md's schema notes (`pose_sequence_keypoints`,
   `capture_objects`, `detector_type`) and `docs/data-model-and-storage.md`
   §3 (the manifest is no longer "future").
6. Merge to `main`. Everything after this lands on `main` in normal
   commits.

Validation: tests green on both sides of the merge; the pen/pad and leg
runs re-executed from the committed CLI produce the recorded numbers (§6).

Harri: I think there have been some tests failing in main; likely worth handling  also existing test regressions at this point.

> **Agreed, and made item 3 above.** Running the suite on both branches
> separately is the cheap part that makes it actionable: without that
> split, a failure found after the merge costs a bisect to attribute.
> Worth doing even if some pre-existing failure turns out to need real
> work — knowing which is which is the deliverable here, and a
> deliberately-recorded known failure is fine, an unexplained one is not.

### WS1 — CLI-complete workflows (L)

1. `posetrak sequence compose` + `pose_sequence_sources` (D3), refactor
   the two `finalise*` functions onto it, `SessionReader` per-source
   noise (D9 half). Delete the three bespoke scripts.
   1a. Shared anonymous candidate pools (§3.5.1): detection-run-keyed
   `load_unlabeled_candidates()`, pool ownership moved to the
   orchestrator, per-subject camera filters, legacy copied-rows path
   kept. This is the C++ half of the D3 correction and is a prerequisite
   for any second dot-bearing subject, so it lands with the composer
   rather than after it.
2. Marker detection in the `posetrak detect` CLI group (`--detector
   aruco|dots`, object-bound or standalone, full dot parameter surface),
   absorbing `run_object_marker_detection.py` and
   `run_standalone_marker_detection.py`; `posetrak detect import-2d`
   with both anonymous and labeled layouts and `label_map` (D4);
   `posetrak track export-2d`.
3. `posetrak marker-body calibrate`; `posetrak marker-set` group with
   `group-tracklets`, `fit`, `refit`, `augment-skeleton`, reading
   `parent_joint` from the catalog module (kills `_SLOT_PARENT_BASE`);
   `catalog/modules/leg.marker-module.yaml`; `marker_attachment_sets`
   table (D5), in the registry with session copy-on-use.
   **Prerequisite for processing the torso+arm capture without new
   scripts** — see the sequencing note below.
   3a. Topology identity and compatibility validation (D15/§9): a
   `topology: {name, version}` block plus derived structural hash in
   skeleton YAML, `requires_topology`/`requires_joints` on modules and
   attachment sets, and a refuse-with-a-clear-error check in the
   composer and in `augment-skeleton`.
4. Multi-view dot triangulation seed provider (§3.6.4); the per-subject
   seed exists (§10.1).
5. Object subjects allowed in `posetrak track run-persons` rosters (they
   already are mechanically; make the CLI/GUI resolvers list them).
   5a. **First step of this item, before the composer work is called
   done**: survey existing captures for a trial containing a dotted
   person and a dotted prop at the same time, to serve as the
   two-dot-bearing-subject test case. Expected to exist (§8 follow-up 3);
   if it does not, say so early — a purpose-made 30-second capture is
   cheap, but only if it is identified while the torso+arm session is
   still being planned rather than after.

Validation: W1, W2, W3 each executed end to end from the CLI on the
existing captures with no script outside the package; results within
the §6 baselines. Additionally, a two-dot-bearing-subject run confirms
one physical candidate is never claimed twice — the case §3.5.1 exists
for, and one no current capture has ever exercised.

> **Revision (2026-09-19): WS1 is delivered as separate pull requests,
> smaller than the list above.** Each ends by running the validation driver.
> Items marked *open* are proposed and confirmed per pull request.
>
> | PR | Content | Replaces plan items |
> |---|---|---|
> | 1 | Person + prop tracking: candidate ownership and config agreement (§3.5.1 revision), a per-subject seed, `track run-persons --object`, a person + ball validation case | 1a (reduced), 4 (reduced), 5, 5a |
> | 2 | `detect run --type aruco\|dots`, object-bound or standalone, with the dot parameter surface (§10.2) | 2 (first part) |
> | 3 | `capture object` commands, `sequence finalise-object`, `sequence add-dots` | 1 (as narrower commands) |
> | 4 | `detect import-2d`, anonymous layout; the labeled layout's `config_json` format is documented, not implemented | 2 (second part, reduced) |
> | 5 | Catalog module YAML and loader replacing the hard-coded slot tables in the existing scripts; topology name and hash check | 3 (reduced), 3a |
> | 6 | `marker-body calibrate` as a library function plus CLI, with the old script re-exporting its shared helpers | 3 (first part) |
>
> *Deferred, with triggers in §10.2:* `pose_sequence_sources` and reference mode,
> `sequence compose`, per-subject camera filters as a separate mechanism,
> per-source noise, `track export-2d`, the multi-view seed provider, and the
> `marker-set` CLI with the `marker_attachment_sets` table. *Open:* whether the
> torso+arm capture may be processed with the existing scripts once PR 5 lands
> (relaxing the gate in WS6).
>
> **Item 5a is answered.** One shot holds both a person with worn dots
> (full-trial sequence, scene-wide dot run) and a dots-only ball (hand-tracked
> points, 56.5–70.5 s inside the person's span). It exercises the joint
> assignment. Its stored sequences share no candidates, so the double-claim
> case is built by adding copies of the person's candidates to the ball's
> sequence in a database copy (§10.1).

### WS2 — Assignment robustness (M, then S per modifier)

1. Refactor `resolve_dot_assignment()` onto `DotAssignmentContext` +
   `CostModifier` with the two existing mechanisms as modifiers;
   bit-for-bit regression against the current outputs on the sword and
   leg runs (the `frame_step_profile` harness plus the existing
   regression test pattern). Also resolve the gate-config shortcut
   (§3.5.1): take the gate from the shared pool's config, refuse a run
   whose dot-bearing subjects disagree.
2. Reacquisition gate (§3.6.3). Validate on the 2026-09-14 ankle case
   (frames 1018–1019) and the ball's `gopro13_02` clutter.
3. Cross-camera corroboration modifier, ported from
   `prototype_ball_tracking.py`.
4. Per-camera trust.
5. Segmentation ROI modifier — only after the cheap offline test on the
   ball's existing mask + dot data says it separates anything.
6. Hue class — only after the offline test on the `gopro13_02` clutter.

### WS3 — GUI-minimal (L)

1. Dot parameters in `RunDetectionDialog` (S).
2. Advanced config tab (S).
3. Roster with objects in `RunTrackerWidget`; retire
   `ObjectRunTrackerDialog` into it (M).
4. Compose-sequence and import-2d dialogs on the person/object panels (M).
5. Resolved-slot dot review/edit with click-to-place (M–L).
6. Coverage/confidence table per run (S).

### WS4 — Detection performance (M)

1. Process-isolated decode workers with timeout-terminate (D12), threaded
   decode on inside the worker; chunked within-camera parallelism sized
   to core count (prop plan §2). Stress-test for hours on the reference
   capture before flipping the default.
2. Hardware decode as a follow-on only if decode still dominates after 1.

### WS5 — Segments and re-initialization (XL: design pass, then L)

1. Export-gap spike: three representations (hold last pose, snap to
   rest, per-segment files + manifest) against **Blender and DaVinci
   Resolve** (per Harri). Resolve is the more informative of the two
   here: it is the stricter importer, so a representation that survives
   it will almost certainly survive Blender, and it represents the
   editorial/finishing consumer rather than the animation one.
2. Design doc for `SubjectSegmentRunner`, loss detection, re-init points,
   schema (§3.7); then implement. Re-track the ball as one four-segment
   run as the acceptance test.

### WS6 — Second person-marker module and calibration trial (L, gated)

**Timing changed by Harri's answer**: the capture happens within 1–2
weeks, i.e. during WS0/WS1 rather than after WS5. Treat the *capture*
as parallel and unblocked (it needs no software), and the *processing*
as gated on WS1 item 3 — the module registry and `marker-set` CLI are
what make a second module "author a YAML" instead of "write another
script". If the capture lands before WS1 item 3 is done, the data simply
waits; do not process it with one-off scripts, since that would recreate
exactly the per-capture-script problem this plan exists to end.

1. Author `torso_arm` module; capture the calibration trial per the person
   plan §3; run W2 with no new code.
2. Profile `update()` at the larger marker count (§3.6.5) before any UKF
   change.
3. Tracklet-group labeling panel in the main GUI (§3.8) once two modules
   have exercised the DB-backed model.
4. Skeleton scaling from rigid clusters (skeleton-scaling design §3),
   staged as that doc describes.

### WS7 — Extrinsics from a moving box (M, independent)

Promote `calibrate_extrinsics_from_moving_box.py` into
`posetrak extrinsics-rig from-moving-box` (stationary-window finder +
per-window free marker groups + final-position anchor). Belongs to the
extrinsics feature; listed here because Harri flagged the ~2 px vs
>10 px result as worth keeping and the script is on this branch.

### Deferred / research (not scheduled)

Automatic calibration-time labeling (redesign §3); body-model spike
(redesign §6.1); person↔prop coupling tuning, grip anchors, free flight
and handoff; hand markers vs wrist-seeded markerless hands; UC3 moving
camera (not started); raw-candidate relabeling UI; the Cutie manual-click
init path.

### Sequencing and dependencies

```
WS0 ─▶ WS1 ─▶ WS2(1–2) ─▶ WS3 ─▶ WS5
        │        │
        │        └── WS2(3–6) ──▶      (each modifier lands when validated)
        └── WS1.3 ─▶ WS6 processing
WS4 ─▶ pulled before WS6 processing (see below)
WS7 independent, any time after WS0
torso+arm capture: parallel, no software dependency
```

WS2's first two items come before the GUI because a review UI over
assignments that can still lock onto clutter reviews the wrong thing.

**WS4 moves ahead of WS6's processing** (revised from the first draft,
on the 1–2 week capture timing). A new module means a fresh full
detection pass over new footage, plus the per-camera threshold sweeps a
new marker placement always needs. At the current 2.8×, that is roughly
a 17-minute loop per parameter iteration on a 6-camera capture — which
is the difference between tuning a new module in an afternoon and
tuning it over days. The cost of moving WS4 earlier is low: it touches
only `frame_source.py` and the pipeline's worker layer, and nothing in
WS1–WS3 depends on it.

WS5 stays last of the core items: it is the only one needing a new
design pass, and every other workstream is useful without it.

---

## 6. Validation

**Regression baselines** (recorded on this branch; each WS re-runs the
affected ones from the package, not from scripts):

| case | metric | value |
|---|---|---|
| sword (ArUco + dots) | tracked steps | 91.4 % |
| pen (ArUco) | tracked, rigid-init RMS | 97.2 %, 1.8 mm |
| pad (ArUco) | tracked, rigid-init RMS | 94.9 %, 3.6 mm |
| person, leg module | tracked, NIS/dof vs markerless | 100 %, equal |
| ball, throw 1 (3 cameras) | reprojection median per camera | 12–57 px |

The `scripts/validate_marker_mocap.py` driver re-runs each case from a
cases file kept beside the data, skips when the data is absent, and
compares against expected values with tolerances. It runs on demand, not
in CI. [validation-baseline.md](validation-baseline.md) records the values
a correct build produces; they reproduce the table above except for the
sword (whose stored detections had to be redone) and the ball's
reprojection figure, both explained there.

**Ground-truth harness** (redesign doc §2) for WS2 and WS6: the CAL and
HARD labeled frame sets and the per-stage metric scripts under
`python/tools/gt/`. Any assignment change ships with the before/after
table.

**Visual check** remains mandatory for any change touching assignment or
init: the rendered reprojection video, cropped to the union of tracked
and observed extents (the round-2 lesson from the ball).

---

## 7. Risks

- **Merge before GUI (D1)** exposes `main` users to CLI-only features.
  Mitigated by N3 (no markers → no change) and by the roadmap README
  marking the feature "in progress".
- **`pose_sequence_sources` on existing sequences**: every existing
  sequence has no rows. Readers treat "no rows" as "single implicit
  source, default trust, dot rows copied in"; a backfill is optional,
  and the legacy copied-dots load path (§3.5.1) stays for them.
- **Moving candidate ownership out of `PersonContext`** (§3.5.1) touches
  the tracking loop that every marker run depends on. Mitigated by the
  bit-for-bit regression in WS2 item 1 being run *before* it as the
  baseline, and by single-subject runs being behaviourally identical
  either way. *(Revision 2026-09-19: this move is not done; the merge with
  ownership changes only the multi-subject candidate merge, and the validation
  driver's single-subject cases are the regression check.)*
- **Topology hash churn**: if the structural subset is chosen badly, a
  cosmetic YAML change invalidates every module's compatibility check.
  Hash the parsed structure, not the text, and exclude anything a
  generator or scaler rewrites (§9.3).
- **Reacquisition gate too strict** could lower tracked % on fast
  motion. It defaults off; the sword run is the sensitivity test.
- **Decode process isolation** costs memory per worker and pickling of
  results; measure on the 6-camera reference before adopting.
- **Segments** touch the result writer, the exporters, the MCP server
  and the GUI timeline. That is why it has its own design pass.
- **Kalman-gain scaling** is unquantified beyond 16 dots; WS6's profile
  gate exists so this is measured before the torso+arm module is relied
  on.

---

## 8. Review round 1 — resolutions

Answered by Harri 2026-09-19; each is now folded into the decision or
workstream it belongs to, and the original comment is kept inline above
for the trail.

| # | Question | Resolution | Lands in |
|---|---|---|---|
| 1 | D1: merge after WS0, or hold? | Merge after WS0; cut `rel-proto` from current `main` first; cleanup scope measured and small | D1, WS0.0/2a |
| 2 | D2: drop multi-sequence binding? | Dropped for labeled data; **but the anonymous-pool half was wrong and is now reference-not-copy** | D2, D3, §3.5.1 |
| 3 | D5: attachment sets in registry? | Registry, with session copy-on-use, like skeletons | D5, WS1.3 |
| 4 | §4: delete superseded prototypes? | Keep under `prototypes/superseded/` during migration; delete per-file as its replacement lands | §4 |
| 5 | WS5: which tools for the gap spike? | Blender + DaVinci Resolve | WS5.1 |
| 6 | WS6: capture scheduled? WS4 first? | Capture in 1–2 weeks, parallel; **WS4 moves ahead of WS6 processing**; WS1.3 is the real gate | WS6, sequencing |
| 7 | §3.4: labeled external tracks? | Two-layout importer with `label_map`; format settled now, labeled import in WS1 | D4, §3.4 |
| 8 | D8: are runs per-object? | No — runs are per trial with a subject roster; segments are per (run, subject) | D8 |
| 9 | Skeleton document split? | Separate feature; topology identity only, pulled into WS1 | D15, §9 |

### Round-1 follow-ups, now closed

1. **D6 and D13 confirmed** (Harri: both ok). Calibration-time labeling
   stays offline/Python at tracklet-group granularity while tracking-time
   assignment stays the C++ shared phase; dot review edits resolved slots
   read from `tracking_obs_results`, with raw-candidate relabeling
   explicitly future work. Both now constrain WS3's review UI as settled
   input rather than open questions.
2. **Tools that are current but destined for replacement stay in
   `python/tools/`** until their replacement actually exists (Harri: ok).
   Only genuinely superseded scripts move to `prototypes/superseded/`.
   The tree therefore never loses a working path to a real workflow
   mid-migration — which matters because the torso+arm capture may land
   while WS1 is still in flight. §4 reflects this.
3. **The two-subject validation is expected to come from existing data,
   confirmed during WS1** (Harri: assume so, check in WS1). Made an
   explicit first step of WS1 item 5 rather than left as an assumption:
   if no existing trial has a dotted person and a dotted prop
   simultaneously, that is worth knowing before the composer's shared-pool
   work is declared validated, since a purpose-made short capture is
   cheap but only if it is identified early.
   *(Closed 2026-09-19: the case exists, see the WS1 revision above.)*

---

## 9. The skeleton document split (answer to D15)

### 9.1 What is actually mixed today

One content-addressed `skeletons` row carries four things with four
different lifetimes:

| | Content | Lifetime | Written by |
|---|---|---|---|
| a | Structure: joint names, hierarchy, types, limits, groups, `bone_tip_offset` directions | Changes with modeling work; shared by everyone | Hand-authored, rarely |
| b | Marker definitions and binding: `markers:`, `input_tracks:`, `track`/`landmark`/`normal` | Per capture at worst; markers are re-taped every session | Generators and calibration |
| c | Person metrics: bone lengths | Durable per person, across sessions | `scale_skeleton.py` |
| d | Marker placement: per-marker offsets | Per session | Attachment-set fitting |

The base design already called for splitting (b)+(d) out
(marker-mocap-design.md §5.2, "marker attachment set"), and the redesign
doc deferred the fuller split explicitly (§1.11) with your own reasoning
that the requirements were not yet understood.

### 9.2 Why it bites now, specifically

Catalog modules and attachment sets reference **joint names as strings**
with no way to declare, or check, which structure they were authored
against. That is the duck typing you identified. The concrete failure is
quiet rather than loud: a `leg` module authored against
`reallusion-no-waist` applied to a skeleton with a waist joint, or with
differently-named leg joints, either errors deep inside the fitting code
with a confusing message, or — worse — silently resolves a subset of its
markers and produces a plausible-looking attachment set fitted against
the wrong parent bones. Nothing in the current model can catch that.

It is worth being precise about what is *not* broken: composition
currently works because there is exactly one topology in use. The risk
arrives with the second one, which is also roughly when the torso+arm
module arrives.

### 9.3 Recommendation: separate feature, one piece pulled forward

**The full split should be its own feature** (`skeleton-model-redesign`,
sibling of this one), not folded into this plan. Three reasons:

1. **Scope.** It touches skeleton loading on both language sides,
   content-addressing and hashing, scaling calibration, every tracking
   run's provenance, the GUI skeleton panels, export, and the migration
   of every existing session DB whose skeletons are already composed.
   That is comparable in size to this entire plan, and it would delay
   work that is one step from being usable.
2. **The requirements still are not known.** Your own §1.11 conclusion
   has not been overtaken by evidence yet. Three of this plan's
   workstreams will generate that evidence directly: WS6's second module
   (does a twist joint become necessary for a forearm?), the
   rigid-cluster scaling work (does an explicit link/bone object become
   necessary?), and WS5 (does a segment want its own per-segment
   metrics?). Designing before those land means designing from one
   module and one rig.
3. **This plan already separates the pieces at the authoring level**, in
   a way that makes the later split a mechanical change rather than a
   redesign. D5 says the composed skeleton is a *derived* artefact,
   generated from a base skeleton plus an attachment set, never
   hand-edited, with provenance recorded. So (b)+(d) already have their
   own source of truth; what the future feature changes is *when*
   composition happens — stop materializing a row, start referencing the
   parts — not what the parts are. Doing D5 first makes the split
   cheaper; doing them together makes both slower.

**What to pull forward now** (WS1 item 3a, small): give topology an
explicit identity so the duck typing is bounded.

- Skeleton YAML gains `topology: {name, version}` and a derived
  `topology_hash` computed over the structural subset only — joint names,
  parent links, joint types, and `bone_tip_offset` *directions* —
  deliberately excluding bone lengths and markers, so that a scaled
  skeleton and a dot-augmented skeleton share their base's topology hash.
- Catalog modules and attachment sets gain `requires_topology: <name>`
  (this is the redesign doc's own open decision D2, `skeleton_topology`)
  and keep `requires_joints`.
- The composer, `augment-skeleton`, and the fitting commands validate
  both and refuse with a message naming the mismatch.

That is a YAML field, a hash over a filtered subtree, and one validation
call. It costs little, it makes a silent failure loud, and it is
strictly forward-compatible: when structure becomes its own stored
document, `topology_hash` becomes its primary key and every consumer of
the name or hash keeps working unchanged.

### 9.4 Triggers that should schedule the full redesign

So the decision is made on evidence rather than by calendar. Any **two**
of these should open the feature:

1. A marker-session-derived bone length needs to reach a markerless
   session — the reuse problem base design §5.2 describes. WS6 item 4
   (rigid-cluster scaling) will very likely hit this first.
2. A second skeleton topology comes into real use (a different base rig,
   or an adopted body model).
3. A module needs twist joints, which the current parent-joint attachment
   cannot express (redesign §1.3 deferred this; a forearm module may
   force it).
4. Rigid-cluster calibration wants an explicit link/bone object rather
   than working through the parent joint.

### 9.5 Scope sketch for that feature, so it is not a blank cheque

Four documents — topology, person metrics, marker catalog/attachment,
and the composed run-time skeleton — plus a composition step and a
migration path for existing content-addressed skeletons (decompose where
possible, grandfather where not). One ordering constraint worth
recording now: the body-model spike (redesign §6.1, SOMA-X and ANNY
licence audit plus a reference topology choice) belongs **before** that
redesign, not after, since picking the joint-naming standard is exactly
what it was meant to feed.

---

## 10. Revision log: WS1 re-scope

Recorded 2026-09-19, when WS1 was about to start. Each row keeps the original
design (still in the sections above), what replaces it, why, and what would
bring the original back.

### 10.1 The shared candidate pool: what was found, and what was built

**Original (D3, §3.5.1):** anonymous candidates are referenced, not copied: a
`pose_sequence_sources` row with `mode='reference'` names the detection run,
the orchestrator loads each run once and owns the pool, and per-subject camera
filters make a camera eligible or not.

**What the code showed.**

- Candidates load per sequence (`load_unlabeled_candidates(sequence_id)`), and
  `pose_observations` rows already carry `detection_run_id` and `noise_scale`.
- The double claim (§2, D3 resolution) arises in one loop: the concatenation of
  per-subject lists in `MultiPersonTracker::run()`.
- Every subject is predicted on every camera that has candidates in the merged
  pool and can claim any of them.

**A first cut was not enough.** Skipping byte-identical candidates in the merge
and limiting each subject to the cameras its own sequence holds candidates for
prevented double claims, but a person + ball run then failed. The ball is
tracked from hand-placed points on three cameras, and the person's automatic
detections on those same cameras were in the merged pool. Right after seeding,
the ball's gate is very wide (a 57 px error read as a Mahalanobis distance of
0.15), so with about 35 person candidates per camera on average it took wrong ones at
the first update (1.35 m off after one step). Measured against the ball tracked
alone: mean NIS/dof 24.4 against 0.150, per-camera reprojection medians of
179–1037 px against 19–71 px.

**What is built.** Candidate ownership: the merge records which subjects' sequences
hold each candidate, and a subject can only claim candidates it owns (§3.5.1
revision). Same physical dot in two sequences: one candidate, both may claim it,
the joint assignment arbitrates. A candidate in one sequence only: that subject
alone. This is the effect of D3's reference model (a subject participates only
in the pools it references) obtained from the data that is already stored.

**Measured after the change** (throw 1, 0.75 s, person and ball tracked
together against each alone):

| Subject | Tracked steps | Mean NIS/dof | Reprojection medians |
|---|---|---|---|
| Person with leg dots | 89 / 89 | 1.577 / 1.577 | identical to 0.1 px on 6 cameras |
| Ball | 89 / 89 | 0.150 / 0.150 | identical on 3 cameras |

No pixel is used by two subjects at the same time and camera, on the plain run
and on the variant where exact copies of the person's candidates are added to
the ball's sequence.

**Other findings recorded here.**

- The multi-subject loop steps every subject by index from its own start time.
  Subjects with different sequence ranges are therefore not time-aligned unless
  they share a start and end; `track run-persons` now defaults to the
  intersection of the subjects' ranges and takes `--start-time`/`--end-time`.
- `--seed-position` was only wired for a single subject. The first multi-subject
  version applied it to "the one subject that cannot initialise", which leaves
  it ambiguous when two objects need a seed and lets the wrong object take it
  when only one does. The seed is now a property of the subject:
  `PersonSpec::seed_position`, `--subject-seed INDEX X Y Z` on the tracker and
  `--object NAME=SKELETON@X,Y,Z` on `track run-persons`. A dots-only subject
  without a seed, and a multi-dot dots-only body with or without one, are
  errors that name the subject.
- No code writes `tracking_run_persons.capture_object_id`; objects in a roster
  therefore need their skeleton given explicitly (`--object NAME=SKELETON`,
  repeated for several objects).

### 10.2 Changes and deferrals

| Item | Original | Now | Why | Reconsider when |
|---|---|---|---|---|
| D3 sources table and reference mode | `pose_sequence_sources`, `mode='reference'`, orchestrator-owned pools, detection-run-keyed loader | Candidate ownership in the merge (§10.1); no schema change | Provenance and trust are already stored per row; the defect sits in one merge loop; ownership gives reference mode's effect | Provenance must be queryable per sequence; detection output grows so per-subject copies cost real storage; a subject must share a pool without holding a copy of it |
| Camera filter as eligibility view | Per-subject `camera_filter` on the sources row | Ownership of individual candidates, which is finer than a camera filter | A camera filter cannot separate two sources on the same camera (§10.1) | A subject must exclude a camera that its own sequence contains |
| D9 per-source noise | `pose_sequence_sources.noise_std` read by `SessionReader` | Deferred; per-row `noise_scale` already stored | No case weighs sources differently; the loader ignores the stored value today | A case combines hand-placed and automatic sources and needs them weighed differently |
| `sequence compose` (§3.5) | One composer, refactoring both `finalise*` functions | Narrower commands (`capture object`, `sequence finalise-object`, `sequence add-dots`) as separate pull requests | Each replaces one real script without the sources table it was designed around | The sources table is built |
| Legacy copied-rows path (§3.5.1) | Kept alongside the new loader | Not needed; there is no second loader | Nothing changed in loading | With the sources table |
| Multi-view seed provider (§3.6.4) | Productized triangulation seed | Deferred; a seed is given per subject | Choosing the ball among candidates is the hard part; an imported single track removes the ambiguity | The external-track import (PR 4) exists and can supply a seed by convention |
| Multi-dot dots-only initialization | Not in this plan; base design phase 2 and dot-assignment design §9 place it out of scope | Refused with an error; a single-dot body starts from a seed position | A seed carries no orientation; a body with several dots needs template registration, which is a separate algorithm | A dots-only prop with several dots and no coded marker is captured (registration by pairwise-distance RANSAC, algorithms doc §4.1) |
| Dot-assignment solver scaling (§3.6.5) | Not identified | Unchanged (O(n³) per camera per step); measured, with the remedy named | Real captures so far stay near 340 candidates | A case with several hundred candidates per camera and step runs too slowly to validate |
| `track export-2d` (§3.10) | Per-camera CSV writer | Deferred | No consumer | Something reads the export |
| Labeled `detect import-2d` layout (D4) | Implemented in WS1 | The `config_json` format and `label_map` are documented; only the anonymous layout is implemented | The first labeled user is later; the format costs nothing to settle now | A labeled external project exists |
| `marker-set` CLI, `marker_attachment_sets` table (D5, §3.2) | WS1 item 3, gating the torso+arm processing | The catalog module file and loader come first (PR 5); the CLI and table follow the first GUI consumer | The scripts work and are protected by D11; what blocked a second module was the hard-coded slot tables | The tracklet-group labeling panel or another app feature needs the fitting functions; `python/tools` is not shipped, so app code must not import from it |
| `detect run` selector (§3.4, WS1 item 2) | `--detector aruco\|dots` | `--type pose\|aruco\|dots`; `--detector` stays the person detector model. `dots` runs the dot detector without a coded-marker pass (`MarkerDetectionPipeline(detect_coded=False)`) | `--detector` already names the YOLOX model on `detect run`, so reusing it for the run kind would make one option mean two things. Marker runs, with or without dots, are stored as `detector_type='aruco'`, as before, so there is no data or schema change and no `dots` value in the column | A consumer needs to tell a dots-only run from an ArUco run without reading `config_json.marker_ids` |
| `marker-body calibrate` (§3.2) | CLI command | Kept, and scheduled as PR 6 with the old module re-exporting its helpers | The application needs calibration; the re-export keeps about 27 importing tools working | (not deferred) |

### 10.3 Status of decisions

- **Accepted:** the D3 deferral in the form of §10.1, and starting WS1 with the
  person + prop pull request.
- **Open:** relaxing the WS6 gate so the torso+arm capture may be processed with
  the existing scripts once PR 5 lands.
- **Proposed, confirmed per pull request:** the order and content of PRs 2–6 in
  the WS1 revision above.
