# Marker-based motion capture — design

**Status**: Design proposal (2026-08-19). Based on
[marker-mocap-brief.md](marker-mocap-brief.md) and analysis of the current
implementation.

**Relationship to earlier documents** — this design consolidates and, where
they conflict, supersedes:

- `docs/aruco-prop-tracking-design.md` — feasibility analysis for ArUco prop
  tracking. Its key decisions (all markers of one prop = one track; blob
  storage; raw corners as observations, no PnP; skeleton-declared input
  tracks) are adopted here largely unchanged.
- `docs/roadmap/features/pose-detect-improvements/marker-detection-analysis.md`
  — marker-type trade-offs and the anonymous-marker labeling architecture.
  Adopted; the labeling stack is specified in detail in
  [marker-mocap-algorithms.md](marker-mocap-algorithms.md).
- `docs/roadmap/features/extrinsics-improvements/extrinsics-improvements-design.md`
  §3/§9/§10 — the fiducial detection framework, marker body YAML format, and
  `marker_body_definitions`/`scene_marker_bodies` storage. This design is the
  "future moving-marker feature" those sections repeatedly deferred to; it
  consumes that infrastructure as-is rather than reworking it, **except**
  one runtime assumption in §10's "Reflective dots" note, which this design
  supersedes: that doc reasons dots can only be tracked "once *some* marker
  on the same body ... has already solved a pose for it" (i.e. a coded
  anchor is always present). That held for the calibration box that
  motivated it, but not for a *dot-only* prop (phase 2, §7) — the
  definition-format-level requirement to fix the body-local frame at
  *characterization* time (§6.1 item 1: at least one coded marker or a
  manual axis definition) is unaffected and still applies once, offline,
  but per-frame *runtime* correspondence for an all-dot body has no coded
  anchor to lean on and needs the unlabeled rigid-template registration in
  [marker-mocap-algorithms.md](marker-mocap-algorithms.md#41-establishing-correspondence)
  §4.1 instead.

Algorithm-level detail (detection, association, initialization, calibration,
drift detection) lives in [marker-mocap-algorithms.md](marker-mocap-algorithms.md).

---

## 1. Use cases and requirements

### UC1 — Rigid prop tracking (priority 1)

Track a rigid object (bokken, jo, calibration box, …) carrying physical
markers, producing a 6-DOF pose trajectory. Motivating example: aikido bokken
practice, where blade trajectory and edge angle matter, hand-pose errors are
amplified by the lever arm, and the grip shifts during cuts — so the sword
must be tracked as its own body, not inferred from the hands.

Requirements:

- **R1.1** A prop is characterized once (geometry + marker layout in a
  body-local frame) and reused across sessions.
- **R1.2** A prop is added to a capture the same way a person is, then
  detected and tracked through trials.
- **R1.3** Output is a rigid-body pose trajectory (position + orientation per
  tracker step), stored in `tracking_results` like person states, exportable
  like person tracking output.
- **R1.4** Physically unobservable DOFs (roll of a rotationally symmetric
  staff) must not be chased by the filter.
- **R1.5** (Phase 3+) A prop held by a tracked person can be coupled to the
  person's solution so the prop's precise markers also improve hand/arm pose
  ("use its relative keypoint locations to also optimize hand pose").

### UC2 — Person tracking augmentation (priority 2)

Attach markers to a performer to add observations at body locations the pose
estimators cover poorly (spine, hips, shoulders) or measure noisily.

Requirements:

- **R2.1** Marker observations fuse into the same per-person UKF as pose
  keypoints — one solver, one state. Markers are *additional* observations,
  never required: everything degrades gracefully when markers are occluded,
  fall off, or are absent (markerless-first invariant).
- **R2.2** Marker→segment attachment (which joint frame, what local offset)
  is *calibrated, not authored*: the user seeds markers once, then an
  automatic calibration pass over a well-tracked movement window estimates
  each marker's constant offset in its segment's frame.
- **R2.3** Anonymous markers (reflective/colored dots carry no identity) must
  be labeled robustly; a mislabeled marker is worse than a missing one
  ("drop, don't guess" — matches existing outlier-gating philosophy).
- **R2.4** Markers on loose clothing must be usable with honestly inflated
  noise rather than silently corrupting the solution.

### UC3 — Moving-camera detection and recovery (priority 3)

Fiducials fixed in the environment (already used for extrinsics anchoring)
are monitored during trials to detect camera movement — accidental or
intentional — and to recompute extrinsics for the moved camera.

Requirements:

- **R3.1** Per-camera drift is detected and surfaced to the user with a time
  estimate of when movement happened; no silent corruption of tracking runs.
- **R3.2** A moved camera's extrinsics can be re-solved from the scene
  markers it still sees (single-camera re-anchor — Phase 9 of the extrinsics
  design already validated this mechanism) and applied to the affected time
  range.
- **R3.3** (Later) Deliberate camera repositioning mid-session is a supported
  workflow, not an error.

### Marker-type flexibility (cross-cutting)

The design must not hard-code one marker technology. Supported detection
families, per the marker-type analysis:

| Type | ID | Role |
|---|---|---|
| ArUco / AprilTag corners | hard | props, static scene refs, re-anchoring (poor under motion blur) |
| Reflective dots (+ per-camera visible-light ring) | none | fast-motion body/prop points |
| Colored dots / bands / gloves | soft (color) or hard (region=ID) | body points, identity cues, thin-weapon bands |
| ChArUco boards | hard | calibration only (already supported) |

The tracker-facing contract is uniform: a marker detection is a 2-D point
(or 4 corner points for a coded quad) with a confidence and, possibly, an
identity. Detector families are pluggable behind the existing
`FiducialDetector` protocol (`python/app/setup/fiducial_markers.py`), which
was deliberately built per-frame and stateless so this feature could consume
it without rework.

### Non-functional requirements

- **N1** Marker detection runs offline/batch like pose detection (it is not
  in the interactive path); full-resolution CPU detection cost must be
  acceptable at that stage, with per-camera parallelism.
- **N2** All existing invariants hold: detection runs are append-only;
  pixels stored undistorted where the schema says so; `SkeletonLayout` is the
  single source of truth for DOF indexing; UKF alpha/weight constraints.
- **N3** Sessions without any markers behave exactly as today (zero-cost
  when unused).

---

## 2. Concepts and terminology

- **Marker body** — a named rigid body carrying markers at fixed body-local
  positions; defined by the §10 YAML format, stored in
  `marker_body_definitions`. Already exists for calibration rigs. A *prop* is
  a marker body that moves and gets tracked; a *scene marker body* is one
  fixed in the environment (`scene_marker_bodies` rows).
- **Tracked object** — a prop instance participating in a capture, the
  object analog of `capture_persons`.
- **Marker track** — one detection run's stream of marker detections for one
  object (or one person's attached markers), analogous to a person's pose
  track. Landmarks within a track are named slots ("marker `hilt`, corner 2",
  "dot `spine_mid`"), structurally identical to "COCO keypoint 5".
- **Coded vs. anonymous markers** — coded markers (ArUco etc.) carry
  identity in every detection; anonymous markers (dots) must be labeled by
  the association pipeline before the UKF can consume them.

---

## 3. Architecture overview

The guiding principle: **once a detection has an identity mapped to a
skeleton marker index, the entire existing tracker pipeline works
unchanged.** The UKF does not care whether an `Observation` came from
RTMPose or an ArUco corner — it already supports per-observation
`noise_std_override`, per-observation gating, PAIR_DIFF references, and
cross-tracker anchors. Almost all new machinery therefore sits *before* the
filter:

```
                       ┌──────────────────────────────────────────────┐
                       │ Python: detection phase (offline, per camera)│
 videos ──────────────▶│  pose detection   (existing)                 │
                       │  marker detection (NEW: aruco/blob/… runs)   │
                       └──────────────┬───────────────────────────────┘
                                      │ detection_keypoints blobs
                                      ▼
                       ┌──────────────────────────────────────────────┐
                       │ Python: finalise (per trial)                 │
                       │  person sequences (existing)                 │
                       │  marker sequences + keypoint manifest (NEW)  │
                       └──────────────┬───────────────────────────────┘
                                      │ pose_observation_sequences
                                      ▼
                       ┌──────────────────────────────────────────────┐
                       │ C++: SessionReader (multi-source load, NEW)  │
                       │  map (track, landmark) → skeleton marker idx │
                       │  fuse into one ObservationSet per subject    │
                       └──────────────┬───────────────────────────────┘
                                      ▼
        ┌──────────────────────── per frame ─────────────────────────┐
        │ [anonymous markers only] MarkerAssociator (NEW, phase 2,   │
        │   single-body scope; generalised to full cross-subject     │
        │   scope in phase 4): tracklets + multi-view + prediction-  │
        │   gated assignment                                         │
        │                                                            │
        │ UKF predict → update            (existing, unchanged)      │
        │ MultiPersonTracker contact anchors: person ↔ prop (reused) │
        └────────────────────────────────────────────────────────────┘
```

The genuinely new algorithmic pieces are: rigid-body initialization
(replacing IK for prop skeletons), the anonymous-marker association stage,
the marker-offset calibration post-process, and the camera-drift monitor.
Everything else is data-model plumbing and UI.

---

## 4. Data model changes

### 4.1 Detection layer

Following `aruco-prop-tracking-design.md`:

- `detection_runs` gains **`detector_type`** (`'pose'` default, `'aruco'`,
  `'blob'`, …) and **`config_json`** (detector parameters plus the blob
  decode key). Existing columns (`detector_model`, thresholds, status,
  timestamps) are shared.
- Marker detections are stored in **`detection_keypoints`** with the
  existing `(detection_run_id, capture_video_id, video_frame, track_id,
  region_type)` key and the existing `float32[n, 3]` (x, y, conf) blob
  layout, so `blob_codec.hpp::decode_keypoints()` and the editing overlay
  machinery are reused as-is. Layouts per detector type:
  - **Coded markers** (one prop = one track, `track_id = 0`): fixed-slot
    layout `n = 4 × n_markers`, ordered by the run's `config_json`
    `marker_ids` list (list-position-major, corners 0–3 within each marker).
    Undetected corners are NaN with conf 0. Confidence is 1.0 (ArUco corners
    are detected-or-not); sub-pixel refinement quality can lower it later
    without a format change.
  - **Anonymous dots**: one row per (frame, camera) with a *variable-length*
    blob of candidate centroids `(x, y, conf)`, `track_id = 0`,
    `region_type = 'markers'`. Count = blob length / 12. Identity does not
    exist at this layer by definition; per-camera tracklet ids assigned by
    the association stage are stored alongside (see §4.3 below and the
    algorithms doc §3).

This deliberately avoids a new observation table until a concrete need
appears (same reasoning as the ArUco analysis: typed blobs in one table,
decode key on the run).

### 4.2 Objects as first-class capture participants

New table, mirroring `capture_persons`:

```sql
CREATE TABLE capture_objects (
    id                         TEXT PRIMARY KEY,
    capture_id                 TEXT NOT NULL REFERENCES captures(id),
    name                       TEXT NOT NULL,     -- e.g. "bokken-A"
    marker_body_definition_id  TEXT NOT NULL REFERENCES marker_body_definitions(id),
    notes                      TEXT
);
```

The prop's *tracking skeleton* is generated from its marker body definition
(§5.3) and imported into `skeletons` (content-addressed, like every
skeleton), so downstream tables need nothing new: a tracked object
participates in a tracking run as an additional `tracking_run_persons` row
whose `skeleton_id` is the generated prop skeleton. To make the object link
explicit rather than convention:

```sql
ALTER TABLE tracking_run_persons ADD COLUMN capture_object_id TEXT
    REFERENCES capture_objects(id);   -- NULL for actual persons
```

*Alternative considered*: a parallel `tracking_run_objects` +
`tracking_object_results` table family. Rejected — it duplicates every
results/diagnostics/export path for no semantic gain; the existing
`person_id` integer is already just a subject index within the run
(`MultiPersonTracker` treats subjects uniformly, and the marker-detection
analysis explicitly recommends keeping the orchestrator free of
person-specific assumptions).

#### Definition vs. capture object vs. skeleton — roles and lifecycle

(Raised in review, Harri 2026-08-19: the geometry seemingly lives in two
places.) The three entities have disjoint roles per pipeline stage; the
skeleton's copy of the geometry is a *derived, compiled* form, not a second
authority:

| Entity | Scope | Written by | Read by |
|---|---|---|---|
| `marker_body_definitions` | registry (global, one per physical design) | characterization (§6.1), once | **detection** (which dictionaries/ids to scan for, which detections belong to this body); the skeleton generator; drift/anchor tooling |
| `capture_objects` | one capture | user, when setting up the capture (§6.2 step 1) | UI (session tree, run dialogs): "this physical body was present here, call it *bokken-A*". Carries no geometry — it links a capture to a definition, exactly as `capture_persons` links a capture to a person |
| generated prop skeleton (`skeletons` row) | referenced per tracking run | the generator (§5.3), never by hand | **tracking only** — the C++ side keeps its single geometry interface (`Skeleton`/`SkeletonLayout`/FK) and never learns about marker bodies |

The duplication is therefore a deterministic derivation (same definition →
same generated YAML → same content hash), analogous to how skeleton YAML is
already compiled to URDF/Pinocchio models internally — it can never diverge
by editing, only by regeneration, and regeneration is idempotent.
Provenance is kept in the generated YAML itself
(`generated_from_marker_body: <definition id>` in the header, so the
skeleton hash also pins its source); the run's
`tracking_run_persons.skeleton_id` records which compiled form was used,
same as for persons. Two rules keep the boundary clean: generated skeletons
are never hand-edited (fix the definition, regenerate), and detection-time
code never reads skeletons while tracking-time code never reads
definitions.

### 4.3 Finalized observation layer

Marker detections finalize into ordinary `pose_observation_sequences` +
`pose_observations` rows — one sequence per tracked object per trial (and,
for UC2, one *marker* sequence per person alongside their pose sequence).
Blob layout is declared by the keypoint manifest table already sketched as
the extensibility seam in `docs/data-model-and-storage.md` §3:

```sql
CREATE TABLE pose_sequence_keypoints (
    sequence_id   TEXT NOT NULL REFERENCES pose_observation_sequences(id),
    keypoint_idx  INTEGER NOT NULL,
    name          TEXT NOT NULL,   -- "hilt:c0".."hilt:c3", "dot:spine_mid", …
    source        TEXT NOT NULL,   -- "aruco", "reflective_dot", "manual", …
    PRIMARY KEY (sequence_id, keypoint_idx)
);
```

A sequence *without* manifest rows keeps today's implied-by-`pose_model`
layout — fully backward compatible. Landmark names for a prop derive from
the marker body definition (`<marker name>:c<corner>` for quads, the dot's
own `name` for dots), so the same physical body always yields the same
names regardless of which ArUco ids it carries.

Consequences worth calling out:

- `pose_observation_edits` (manual keypoint editing) works on marker
  sequences with zero changes — the editing UI becomes the manual-labeling
  fallback for markers for free.
- The MCP diagnostic server sees marker runs through the same tables.
- Anonymous-dot sequences are only written *after* labeling (association
  assigns landmark slots); unlabeled candidates stay in the detection layer.
  A slot with no confident label at frame t is NaN — exactly an occluded
  keypoint.

### 4.4 What is *not* changed

- `marker_body_definitions` / `scene_marker_bodies` are consumed unchanged.
- `pose_observations` schema, blob encoding, sync/extrinsics chains:
  unchanged.
- Detection runs remain append-only; marker detection runs are new runs,
  never additions to old ones.

---

## 5. Skeleton format and C++ changes

### 5.1 Input tracks (skeleton YAML extension)

Adopted from the ArUco analysis. The skeleton declares named observation
sources; markers reference a track + landmark; tracks are bound to concrete
sequence ids at invocation:

```yaml
input_tracks:
  - id: person_pose
    type: coco133          # layout implied by pose model (today's behavior)
  - id: body_markers
    type: labeled_points   # layout from the bound sequence's keypoint manifest

markers:
  - name: left_wrist
    parent: elbow.L
    offset: [0.0, 0.28, 0.0]
    openpose_keypoint: 7          # legacy sugar ≡ {track: person_pose, landmark: 7}

  - name: spine_mid_dot
    parent: spine2
    offset: [0.0, 0.11, -0.06]    # written by offset calibration, not authored
    track: body_markers
    landmark: "dot:spine_mid"
    noise_std: 1.0                # optional per-marker measurement noise (px)
```

- A skeleton with no `input_tracks` behaves exactly as today
  (`person_pose`/coco133 implied).
- Binding at invocation: `posetrak-tracker track-db … --track
  body_markers:<sequence_id>` (and the equivalent columns in
  `tracking_runs`' provenance). `PersonSpec` grows a
  `map<track_id, sequence_id>`; `SessionReader::load_observations()` gains a
  multi-source variant that loads each bound sequence, resolves
  `(track, landmark)` → skeleton marker index (via the manifest), and merges
  into one `ObservationSet`. This is precisely the multi-source
  `SessionReader`/`ObservationSet` work already flagged in
  `phase5-cross-person-plan.md`.
- Per-source noise: marker observations get `noise_std_override` from the
  marker's `noise_std` (or a per-track default in `tracker_configs`), so a
  1 px dot rightly dominates a 5 px interpolated spine keypoint. `crop_scale`
  is 1.0 for markers (detected on full frames, no crop pipeline).

### 5.2 Skeleton information lifecycle: structure vs. person scale vs. session markers

(Raised in review, Harri 2026-08-19.) A person skeleton mixes information
with three distinct lifetimes, and today all of it is baked into one
content-addressed `skeletons` row, with `parent_id` chains recording
derivation:

1. **Structure** — bone hierarchy, joint types/limits, groups, pose-keypoint
   marker slots. Common across persons; changes only with modeling work.
2. **Person scale** — bone lengths from `scale` calibration. A durable
   property of one person, reusable across sessions.
3. **Marker attachment** — which physical markers, on which segment, at
   what offset, with what noise. Valid for one session at best: markers are
   re-taped every time, possibly in different places, and many sessions are
   markerless.

Baking (3) into the same object as (1)+(2) creates a real reuse problem: a
scale improvement obtained during a marker session would live in a leaf
skeleton that also carries dead session-specific marker data, and the
durable part would not flow to the markerless skeleleton other sessions use
without manual re-derivation.

**Recommendation: keep (1)+(2) in the skeleton; move (3) into a separate
per-session *marker attachment set* document.**

- New document type (small YAML, content-addressed, stored like skeletons —
  e.g. table `marker_attachment_sets`): a list of
  `{name, parent joint, offset, noise_std, track/landmark}` entries that
  reference the target skeleton's *joint names*, not a specific skeleton
  hash.
- At invocation, the loader **composes** skeleton + attachment set into the
  effective tracker-facing skeleton (a pure merge into `markers:`). The C++
  tracker still receives exactly one `Skeleton` — the single-interface
  property of §4.2 is preserved — and the tracking run records both source
  ids for provenance.
- The offset-calibration pass (§6.3) writes an attachment set, **not** a
  skeleton variant. If a marker session's data also improves bone lengths,
  that goes through the existing scale calibration and produces a new
  person-scale skeleton that is marker-free by construction — so the
  improvement propagates to markerless sessions automatically, and the
  attachment set applies on top of the new scale unchanged (joint-name
  anchored; offsets survive a mild rescale, and a large rescale is itself a
  reason to re-run the cheap offset calibration).
- Props do **not** need this split: a prop's markers *are* its durable
  geometry, so the single generated skeleton (§5.3) is already the right
  granularity.

*Alternative (status quo)*: keep everything in one skeleton object and rely
on `parent_id` discipline. Workable for phases 1–4 (props only, which never
hit the problem), and acceptable as long as no person marker data exists —
so the attachment-set mechanism should land with phase 5, before real
person-marker sessions are recorded.

### 5.3 Prop skeletons are generated, not authored

A prop is a degenerate skeleton: one free-flyer root, `FIXED` structure,
markers only. A generator (CLI `posetrak marker-body to-skeleton` + called
implicitly by the GUI when adding an object to a capture) converts a marker
body definition into skeleton YAML:

- each coded marker → 4 markers `<name>:c0..c3` at
  `marker_local_corners`-resolved body-local positions;
- each reflective dot → 1 marker with its `center`;
- `input_tracks: [{id: prop_markers, type: labeled_points}]` with every
  marker referencing it.

The generated YAML is imported into `skeletons` (content-addressed →
idempotent, stable across sessions for the same body). Keeping the
generator outside the C++ loader follows the §10 principle: the loader
stays dumb and literal; generation is an inspectable offline step.

**Symmetric props (R1.4)**: a cylinder's roll is unobservable. Options:

1. **(Recommended)** Generator emits the root as free-flyer plus a
   `locked_dofs` annotation; C++ handles it as a tiny per-step
   regularization pulling the locked axis's angular velocity to zero
   (pseudo-observation), keeping `State`/root representation untouched.
2. Support non-freeflyer roots (5-DOF root) in `SkeletonLayout` — cleaner
   conceptually but touches the most invariant-laden code in the tracker
   (root quaternion handling, error-state indexing) for a niche case.

Option 1 first; revisit if the regularization proves fiddly.

### 5.4 Tracker changes (C++)

- **Rigid-body initialization**: for a root-only skeleton, triangulation +
  full IK is overkill and fragile. Add an analytic path: triangulate ≥3
  non-collinear markers, then closed-form rigid fit (Kabsch/Umeyama, scale
  fixed) of body-local → world; fall back to IK path otherwise. See
  algorithms doc §4.
- **Association hook (phase 2+, anonymous markers only)**: anonymous-dot
  labeling needs the predicted state each frame. New component
  `MarkerAssociator` sits in the per-frame loop between observation fetch
  and `track_frame()`; it consumes the tracker's predicted marker
  projections and the frame's anonymous candidates, and emits labeled
  `Observation`s (or drops them). It reuses `Tracker::marker_projection_std()`
  for gating covariance. The UKF itself is untouched. Phase 2 needs only its
  single-body scope (one marked rigid body, no cross-subject term); phase 4
  generalises it to the full joint-Hungarian-across-all-subjects form. See
  algorithms doc §3 for the three-layer design and why assignment stays
  *outside* the UKF update (the brief's "per-marker prediction vs. combine
  into update" question).
- **Person↔prop coupling (R1.5)**: a prop is another subject in
  `MultiPersonTracker`. The existing Stage-2 machinery — contact gating +
  cross-subject PAIR_DIFF anchors with `anchor_position`, rotation of
  processing order, `marker_projection_std`-based anchor noise — applies
  verbatim to "person hand marker ↔ prop grip marker". Grip markers (e.g. a
  point on the tsuka) are declared in the prop's marker body definition as
  dots even if nothing physical is there; they exist to give the gate an
  anchor point. Later (beyond the phasing in §7 of this feature), a spliced
  single skeleton (`prop_root` with `parent: right_hand`, per the ArUco
  analysis phase 3) becomes possible for rigidly-attached props, but the
  anchor mechanism is the right default for hand-held props whose grip
  changes.

---

## 6. User experience

### 6.1 Characterizing a prop (R1.1)

Home: the setup app's existing marker-body management ("Manage rigs…"),
extended from calibration-rig scope to general props.

1. **Author or capture geometry.** Three entry paths, in increasing
   automation:
   - write the §10 YAML by hand (measured geometry — works today);
   - **characterization capture**: wave the prop through the calibrated
     capture volume (or orbit a camera around it); the tool detects coded
     markers across frames, bundle-adjusts the rigid constellation, and
     emits fully-resolved YAML (`corners:` form). The
     `characterize_rig_from_video.py` prototype is the seed; productize as
     a setup-app page with per-frame detection preview;
   - for dot-only props (thin weapons with bands/dots): seed dots by
     clicking them in a few frames from ≥2 cameras; triangulation + the
     same bundle adjust recovers body-local positions. At least one coded
     marker or manual axis definition is needed to fix the body frame.
2. **Import** into `marker_body_definitions` (idempotent).
3. Optional: mark a rotational symmetry axis. Concrete example: a jo is a
   cylinder, and if its markers are bands (points on the long axis),
   rotating the staff about that axis moves no marker at all — that roll
   angle is both invisible to the cameras and physically meaningless for
   the prop. Recording the axis in the definition lets the skeleton
   generator lock the corresponding root DOF so the filter doesn't wander
   in a direction the data can never correct (R1.4; mechanism in §5.3's
   "Symmetric props"). Props with asymmetric marker layouts (a bokken with
   tsuba-mounted markers) skip this — every DOF is observable.

### 6.2 Tracking a prop in a session (R1.2, R1.3)

1. In the main window, add an **object** to the capture (picker over
   `marker_body_definitions`) — appears in the session tree alongside
   persons.
2. Run a **marker detection run** for the trial (new detector type in the
   run-detection dialog; parameters: dictionaries in the body, perimeter
   rate, frame step). Runs per camera, full resolution, batch.
3. **Review** in a new `ObjectPanel` (sibling of `PersonPanel` in
   `python/app/ui/content_panels.py` — this is Main-window territory per
   CLAUDE.md's two-GUI note): crop grid with detected corners/dots
   overlaid, scrubber, and the existing keypoint-edit mode for fixing bad
   frames. No stitching step — a *coded* prop's track assignment is
   trivial (one object, ids are hard).

**Uncoded markers, and multiple dotted subjects in one scene** (review
question, Harri 2026-08-19): a dot detection carries no identity, so the
steps above describe the coded phase-1 case, and dots gain identity at
three levels of the same mechanism, in increasing order of what they need
(details in algorithms doc §3):

- *Mixed bodies* (coded markers + dots on the same rigid body — the
  calibration box pattern, and Harri's actual existing props): the coded
  markers pose the body each frame, the pose predicts every dot's position,
  and dots are labeled by that rigid geometry alone. No association stack
  needed at all — this works throughout phases 1–4 regardless of how many
  other subjects are in the scene, since each such body's dots are labeled
  purely from its own already-posed geometry, never in competition with
  anyone else's. This is the recommended way to build props whenever a
  coded anchor is acceptable.
- *Dot-only bodies, standalone* (phase 2): with no coded anchor to pose the
  body first, bootstrap is a genuine unlabeled point-set registration
  problem — match the unordered triangulated 3-D points against the known
  rigid template by pairwise-distance consistency (algorithms doc §4).
  Once posed, steady-state relabeling is nearest-predicted-position per
  marker, same as the mixed-body case above — still no cross-subject
  mutual exclusion needed with only one marked body in frame.
- *Multiple dot-only bodies, or dotted persons, competing in one scene*
  (phase 4 for props; phase 6 once persons carry dots too): labeling runs
  through the full association stage, and its assignment is solved
  **globally across every tracked subject in the run** — all persons' and
  props' predicted markers compete in one mutual-exclusion assignment per
  frame. A dot that two subjects could claim with comparable cost (a
  hand-on-prop moment, two performers close together) is dropped for that
  frame, not guessed; the subject's filter coasts. Identity re-attaches
  through tracklet continuity, color class, or any coded marker on the same
  body once the ambiguity passes.
- Two dot-only bodies with near-identical, symmetric layouts are
  inherently indistinguishable at bootstrap. The characterization tool
  should warn when two bodies registered in the same capture have
  near-identical inter-marker distance patterns; distinct constellations or
  different color classes break the tie physically.
4. **Finalise** to an object sequence (+ manifest), then run tracking —
   the object shows up in the run-tracker dialog as another subject; person
   + prop in one run uses `MultiPersonTracker`.
5. Results browse/export identical to persons (BVH export of a free-flyer
   root is just a root trajectory; TRC gets the marker trajectory).

### 6.3 Person with markers (R2.x)

Following the "let the markerless tracker calibrate them" principle:

1. Attach markers anywhere they stick; no anatomical placement needed.
2. Run pose detection as usual **and** a marker detection run.
3. **Seeding** (coded markers: skip — identity is in the detection): in the
   person's crop grid, on one frame, click each dot and give it a name
   (reuses the keypoint-editing interaction). One frame in one camera
   suffices when the association can extend it; more clicks help.
4. **Calibration pass**: the user selects a well-tracked window (or the
   tool proposes one from tracking stats); the system runs markerless
   tracking, triangulates labeled markers, attaches each to the nearest
   segment, averages the local offset, then refines by least squares over
   the window (algorithms doc §5). Output: a **marker attachment set**
   (§5.2) bound to the person's scaled skeleton by joint names, containing
   the marker entries plus a proposed `noise_std` per marker from residual
   spread — cloth-mounted markers get honestly large noise automatically
   (R2.4). The user can override segment attachment in a review dialog
   (answering the brief's "does the user need to link markers to joints" —
   default automated, override available).
   Scale improvements discovered in a marker session do **not** live in
   this output: bone-length refinement goes through the existing `scale`
   calibration and updates the person-scale skeleton itself, which is
   marker-free by construction — so it propagates to markerless sessions
   with no extra step (see §5.2 for the layering rationale).
5. Subsequent tracking runs bind the marker sequence and just work; if the
   markers are absent in some trial, nothing breaks (R2.1).

Soft/flexible mounting (chest markers influenced by several joints — the
brief's "weight map" question): **v1 models this as a single-segment
attachment with inflated noise**, not a skinned measurement model. A
weight-map measurement model is sketched in algorithms doc §5.3 with its
costs; it is deferred until single-segment + noise demonstrably limits
accuracy.

### 6.4 Moving camera (R3.x)

Phased shallow-to-deep:

1. **Monitor + alert** (cheap, high value): during any marker/pose
   detection run — or as its own lightweight scan — detect the session's
   `scene_marker_bodies` fiducials in each camera at a coarse frame stride,
   reproject their known world positions through the camera's calibration,
   and track the residual over time. A step change flags "camera X moved at
   ≈ t"; surfaced on the capture in the UI and in tracking-run preflight.
   (Algorithms doc §6: robust statistics, thresholds, why a step-change
   detector and not per-frame gating.)
2. **Re-solve**: one-click action solving the moved camera's new pose from
   scene markers it sees after the step (single-camera PnP re-anchor — the
   validated Phase 9 mechanism), producing a new `extrinsic_calibrations`
   row.
3. **Time-windowed application**: new table
   `extrinsic_calibration_windows (capture_id, valid_from_s,
   extrinsic_calibration_id)`. Sequence/tracking loaders select the window
   covering their time range; a tracking run **may not span a boundary** in
   v1 (preflight splits the trial at the detected movement time). This
   avoids time-varying `Camera` objects inside the C++ tracker entirely.
   Continuous per-frame camera-pose estimation (camera pose in the filter
   state) is explicitly out of scope — it is a different product
   (SLAM-adjacent) and nothing in the priority list needs it.

---

## 7. Phasing

UC1 (props) and UC2 (person markers) are staged as two successive
first-iteration targets rather than interleaved phase-by-phase: **UC1 is the
first iteration in full** — real captures already combine ArUco anchors with
reflective dots on the same prop, and a dots-only prop is an equally valid
configuration, so both marker types belong in UC1 rather than treating dots
as a UC2-only concern. UC2 (markers *on people*) is explicitly the next
project after UC1 is working end to end, not a phase inside it.

The phase split within UC1 is driven by labeling difficulty, not by marker
type: a rigid prop's anonymous dots keep constant pairwise distances, so
identifying and tracking them is a *single-body* problem (cold-start
template registration + prediction-gated relabeling scoped to that one
body's own markers, per [marker-mocap-algorithms.md](marker-mocap-algorithms.md#3-anonymous-marker-association-labeling)
§3) — no cross-subject mutual exclusion is needed until *two or more*
marked bodies (props and/or people) can plausibly be confused for each
other in the same volume. That only happens once multiple props are tracked
together, which is deliberately the last UC1 phase rather than the first.

| Phase | Scope | Delivers | Main new pieces |
|---|---|---|---|
| **1** | Rigid ArUco prop, standalone | UC1 core (R1.1–R1.4), coded markers | `detector_type`/`config_json`; ArUco detection runs writing `detection_keypoints`; `capture_objects`; skeleton generator; manifest table; multi-source `SessionReader` (single track case); rigid init; ObjectPanel review; finalise for objects. Zero labeling ambiguity — validates the object/track plumbing before any unlabeled-detection problem is layered on it. |
| **2** | Rigid dot-only (or ArUco+dot) prop, standalone | UC1 core, anonymous markers | Dot detector (algorithms §1.2); multi-view correspondence (algorithms §3.2); single-body cold-start rigid-template registration (new — algorithms §4); scoped single-body prediction-gated relabeling per frame (algorithms §3.3, restricted to one body, no cross-subject term). Proves the anonymous-marker pipeline end to end in the easiest labeling context: exactly one marked rigid body in frame. |
| **3** | Markerless person + one prop (ArUco or dot) together | R1.5 | Track binding for multiple subjects incl. objects in `MultiPersonTracker`; grip anchor points; contact gating tuning for the prop case. The person contributes no marker detections, so prop-dot labeling stays single-body-scoped even with a person present — the only new cross-subject question is the contact/grip anchor coupling, reusing the existing Stage-2 mechanism. |
| **4** | Multiple props (mixed ArUco/dot-only) + person interacting — **UC1 complete** | Full R1.1–R1.5 | Generalise the single-body assignment from phase 2 to the full joint-Hungarian-across-all-subjects form (algorithms §3.3) — this is the first point where two dotted props' point clouds can actually cross-contaminate, so it is the first phase that needs mutual exclusion at all. |
| **5** | Identified body markers | UC2 for coded/colored markers (no labeling problem) | Marker sequence per person; skeleton `track`/`landmark` markers; marker attachment sets (§5.2) + offset-calibration pass; colored-band/glove detector |
| **6** | Anonymous dots on people | UC2 fully (R2.3) | Full cross-subject `MarkerAssociator` scope extended to articulated bodies: per-person *and* cross-person mutual exclusion, occlusion/reacquisition on a moving skeleton, tentative-confirmation window under real pose ambiguity (wrist grabs, limb crossings) — the genuinely hard labeling case, now isolated from the much simpler rigid-prop case solved in phases 2 and 4 |
| **7** | Moving camera | UC3 | Drift monitor + UI alert; single-camera re-solve action; `extrinsic_calibration_windows` |

Phase ordering rationale: props first is unanimous across the brief and both
prior analyses — self-contained, no labeling problem, exercises the
multi-source loading and the extra-subject orchestrator path that every
later phase needs. Phase 1 before phase 2 so the object/track/finalise
plumbing is validated against the zero-ambiguity coded case before an
unlabeled-detection problem is layered on it; phase 2's mechanisms
(detector, triangulated correspondence, prediction-gated assignment) are
then reused, not rebuilt, when phase 6 extends them to people. Phase 5
before 6 because coded/colored markers deliver UC2 value without the
association stack, and the offset-calibration machinery it builds is
required by phase 6 anyway. Phase 7 is independent of 3–6 and can be
reordered if a real moved-camera incident makes it urgent; its monitor half
(6.4 step 1) is cheap enough to pull forward.

### 7.1 Phase 1 breakdown

Phase 1 alone touches four separate subsystems (DB schema, Python
detection, Python finalisation, C++ tracker, two different GUIs) that are
each independently buildable and checkable — there is no reason to land
it as one slab. Six sub-phases, each with its own pass/fail check, in
dependency order (1a and 1b have no dependency on each other and can run
in parallel; everything else is sequential):

| Sub-phase | Delivers | Validation |
|---|---|---|
| **1a** — Detection layer | `detector_type`/`config_json` on `detection_runs` (§4.1); ArUco detector run (reusing `FiducialDetector`/`ArucoDetector`) writing the coded-marker `detection_keypoints` blob layout. No UI, no `capture_objects`, no skeleton — a standalone batch job invocable from a script/CLI against an existing capture's video. | Run detection against a real recorded clip with a known ArUco prop (or a synthetic rendered sequence with known corner positions). Query `detection_keypoints` directly: correct corner count per detected marker, ids match `config_json.marker_ids`, confidence 1.0 when seen, NaN+conf 0 for frames where the marker is genuinely occluded. No tracker or finalisation involved yet. |
| **1b** — Skeleton generator | `posetrak marker-body to-skeleton` CLI (§5.3): converts a `marker_body_definitions` row into skeleton YAML (`input_tracks`, `markers`, locked-DOF annotation for symmetric props — resolves open question 3). Pure offline transform, no capture/detection/tracker-run involvement. | Generate YAML for an existing definition (the calibration box is the natural first target — it already has both ArUco and dot markers per §10) and diff against a hand-verified expected structure; load the generated YAML through the existing `SkeletonLoader`/`SkeletonLayout` unit-test harness and confirm it parses as a valid root-only, `FIXED`-structure skeleton with the right marker count and (for a symmetric test prop) the locked-DOF annotation present. |
| **1c** — Capture-object plumbing | `capture_objects` table + `tracking_run_persons.capture_object_id` (§4.2); "add object to capture" UI in the session tree (picker over `marker_body_definitions`); the 1a detector wired in as a selectable type in the existing run-detection dialog. | In the GUI, add a real object to a real capture, launch a marker-detection run against a trial, confirm the object appears in the session tree and the run's `detection_keypoints` rows are correctly scoped to it. A DB-level test covers the new table/column and the run-launch code path (same style as existing `test_posetrak_db.py`/session-tree tests). |
| **1d** — ObjectPanel review | New `ObjectPanel` (sibling of `PersonPanel`, §6.2 step 3): crop grid with detected corners overlaid, scrubber, keypoint-edit mode reused for fixing bad frames. | Open a real 1c detection run in the panel; corners visibly overlay correctly across cameras and frames; deliberately corrupt one frame's corner via the edit UI, confirm the correction persists across a panel reload before finalisation touches it. |
| **1e** — Finalisation + manifest | `pose_sequence_keypoints` manifest table (§4.3); finalise pipeline extended to emit one object `pose_observation_sequence` (+ manifest rows) per trial from the (possibly edited) 1d detection run. | Finalise a reviewed object run; inspect the resulting sequence's manifest names (`<marker>:c0`..`c3` per §4.3) against the marker body definition, and confirm blob shapes/landmark count match; run the existing finalisation consistency checks (whatever already validates person sequences) against the new object sequence. |
| **1f** — Tracker: multi-source load + rigid init | C++ `input_tracks` binding, `(track, landmark)` → marker-index resolution via the manifest, `PersonSpec`'s track map, multi-source `SessionReader::load_observations()` (single-track case); rigid Kabsch/Umeyama initialization path in `Tracker::initialize()` (algorithms §4.2). | Run `posetrak-tracker track` against the 1e sequence with the 1b-generated skeleton on a real prop-only trial (or a synthetic rigid-trajectory sequence per the algorithms-doc §7 test pattern); assert successful init (residual RMS within the existing retry threshold) and a plausible 6-DOF trajectory written to `tracking_results`/`root_pose.csv`. This is phase 1's actual finish line — first end-to-end tracked prop. |

---

## 8. Open questions

1. **Reflective vs. colored dots** — hardware shoot-out (ring lights,
   locked WB, marker sizes vs. FOV mode) remains empirical; phase 2 should
   start with whichever wins a bench test. Tracked in
   marker-detection-analysis.md's open questions; nothing in this design
   depends on the outcome (both are anonymous point detections).
2. **Confidence semantics for marker detections** — corners are
   detected-or-not; is a uniform 1.0 (with noise carried entirely in
   `noise_std_override`) right, or should blur/oblique-angle metrics
   modulate it? Start uniform; the format allows refinement.
3. **Locked-DOF mechanism** (§5.3) — pseudo-observation regularization vs.
   reduced-DOF root. Decide during phase 1 implementation against a real
   symmetric prop.
4. **2-D-only marker observations** — is a marker seen in one camera
   (labeled via tracklet continuity) worth using in v1 of phase 2/4, or does
   requiring multi-view labeling keep assignment materially simpler?
   Default: require multi-view for *labeling*, allow single-view for
   *continued* observation of an already-labeled tracklet.
5. **BVH/TRC export conventions for objects** — how prop trajectories are
   best delivered to animation software (separate BVH per object vs. one
   combined hierarchy). Defer to phase 1 export work; ask the consuming
   workflow.
6. **Trial splitting UX for moved cameras** — automatic split proposal vs.
   manual; interacts with the trial data model. Decide in phase 7.
7. **Marker attachment set details** (§5.2) — storage scope (session-only
   vs. registry with copy-to-session like other documents), whether the
   composition step materializes the composed skeleton as a `skeletons` row
   (best provenance, more rows) or composes in the loader at run time
   (fewer artifacts, provenance = pair of ids on the run), and whether an
   attachment set should validate against a structure id or stay purely
   joint-name anchored. Decide at phase 5 start; the recommendation in §5.2
   only fixes the *split*, not these mechanics.
