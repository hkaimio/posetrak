# Marker catalog & assignment — redesign (design phase, 2026-09-10)

## Status and scope

This is a **design document**, no implementation attached. It supersedes
the grouping/direction/assignment approach in
[person-marker-assignment-current-state.md](person-marker-assignment-current-state.md)
(which now describes the "before" state) following Harri's 2026-09-10
review. It does **not** replace
[person-marker-assignment-design.md](person-marker-assignment-design.md)
wholesale — the marker layout catalog (§1), attachment set (§2), and
phased plan there are still the frame; this doc fills in the two pieces
that review found under-specified or wrong (the catalog's geometry, and
evaluation) and re-frames assignment around tracklets.

Priority order for this doc, per Harri: **(1) catalog spec, (2) metrics**,
then the rest sketched.

Two strategic decisions this doc records but does not finally settle
(they need their own short spikes — see §6):
- whether to align the skeleton/marker representation with an emerging
  permissively-licensed body model (SOMA-X, ANNY);
- manual-tracklet-assignment-first vs. auto-assignment-first for getting
  marker-augmented tracking working.

---

## 0. Why the previous approach was wrong

Recorded so the redesign doesn't drift back into it:

1. **Grouping by one shared joint anchor.** Every marker in an "ambiguous
   group" (`knee_L_medial/lateral/front`) was mapped to a single vitpose
   keypoint index and matched by proximity to it. Real markers are not
   all at a joint — they sit mid-bone, or near the *child* joint, to
   observe segment twist or to get a clean line of sight. And the
   dominant real failure in the latest validation video was not
   same-joint ambiguity at all: it was **two markers that are close in
   3D projecting close together in one camera**, which anchor-proximity
   cannot represent or resolve.

2. **Pelvis-proxy anatomical direction.** Stage C derived medial/lateral
   from the hip-to-hip vector and anterior from `cross(up, pelvis)`. The
   hip is a ball joint; the thigh can rotate about its own long axis
   (internal/external rotation) without the pelvis moving, which breaks
   the proxy — and it is far worse for arms/hands. The direction a marker
   faces is a property of the **link it is glued to**, not of the pelvis.

3. **Per-frame slot assignment as the unit.** Each frame was assigned
   independently and then majority-voted along tracklets after the fact.
   Tracklets can very likely be built with *higher* confidence than a
   single frame's slot assignment (a within-camera temporal association
   is a much easier problem than a cross-subject identity decision), so
   the tracklet should be the thing we assign, with per-frame assignment
   as a fallback for un-trackleted detections.

4. **Ad-hoc metrics.** "Purity" and "coverage" were debugging
   conveniences measured against a downstream majority vote, not ground
   truth. Not a basis for deciding anything.

---

## 1. Marker catalog specification

### 1.1 What the catalog is for

The catalog answers, for a marker set, **before any per-person
calibration**:

- **Where** would marker *S* project in camera *C* this frame, given a
  markerless skeleton pose? (→ 2D search location / clustering seed)
- **Is** marker *S* facing camera *C* this frame, or is it on the far
  side of the limb? (→ visibility prior, backface culling)
- Which physical **link** does *S* move rigidly with? (→ which FK
  transform to apply, which tracklet-consistency to expect)

and it is the thing a per-person calibration **refines** (nominal offset
→ measured offset for this subject/session).

### 1.2 Relationship to the existing skeleton `markers:` block

`docs/skeleton-format.md` already defines a `markers:` list —
`name`, `parent` (joint), `offset [x,y,z]` in the parent joint's local
frame, optional `openpose_keypoint`. The catalog is **not** a parallel
format; a resolved catalog entry must be expressible as an (extended)
entry of that block. The catalog adds: a surface **normal**, a
scale-robust way to place the offset, side-mirroring, modularity, and the
nominal-vs-calibrated distinction. Per
[person-marker-assignment-design.md §5.2](person-marker-assignment-design.md),
session-scoped attachments live in a separate *marker attachment set*
document, not baked into the shared skeleton — the catalog is the
reusable template those attachment sets are instantiated from.

> **§1.3–1.4 revised 2026-09-10 per Harri's comments** — terminology
> aligned to the existing schema (a marker has a **parent joint**, no
> "link" concept is introduced here); the local frame is defined
> canonically for the **right side** with a mirror flag; the long axis
> reference uses the schema's existing `bone_tip_offset` rather than an
> undefined "primary child joint". Skeleton-schema redesign is explicitly
> **out of scope for this phase** (Harri: "let's not do it yet... I don't
> think we understand all skeleton schema requirements yet well enough")
> — recorded as future work in §1.11.

### 1.3 Attachment model: parent joint (existing schema)

A marker has a **parent joint**, exactly as `docs/skeleton-format.md`'s
`markers:` block already defines — the joint whose child bone the marker
moves rigidly with. A marker on the mid-thigh or just above the knee has
parent joint **hip** (it rides the femur, which is the hip's child bone);
a marker just below the knee has parent joint **knee**. FK gives the
parent joint a world pose every frame; the marker's world position and
normal follow rigidly.

The catalog does **not** invent a "link"/"bone" object — it uses the
parent-joint attachment the schema already has, and adds geometry fields
(§1.5–1.6). ("Bone" appears below only informally, meaning "the parent
joint's child bone segment".)

**Twist markers** (e.g. two markers above the wrist for forearm
pronation/supination) are out of scope for this phase. The current
skeleton merges forearm twist into the elbow joint, so a rigid
parent-joint attachment can't represent a marker that observes twist
below the elbow. Adding explicit twist joints is plausible future work
but changes tracker configuration — deferred (§1.11). The leg marker set
has no twist markers.

### 1.4 The parent-bone-local coordinate frame (canonical = right side)

Each parent joint's child bone gets a right-handed frame, built from the
skeleton **rest pose** (not the YAML rest `orientation`, so the catalog
stays portable):

- **`e_long`**: the parent joint's `bone_tip_offset` (already in the
  schema, "typically points toward the child joint"), normalized, in the
  rest pose. "Down the bone." If a parent joint has no `bone_tip_offset`,
  the catalog entry must give `long_axis_ref` explicitly (a child joint
  name, or a rest-pose vector) — an authoring requirement, not a schema
  change.
- **`e_lat`**: `normalize(e_long × ref_up)`, `ref_up` = rest-pose global
  up. (Degenerate for a near-vertical bone → fall back to `ref_forward`;
  per-entry flag.)
- **`e_ant`**: `e_lat × e_long`.

**This frame, and every catalog entry, is defined for the RIGHT side
only.** For the right thigh: `e_long` points down, `e_lat` points to the
subject's right (away from midline = lateral), `e_ant = e_lat × e_long`
points forward — a proper right-handed frame, and the axis names match
anatomy. A catalog entry that also applies to the left side carries
`mirror: true`; the left instance is generated by reflecting
`offset` and `normal` across the sagittal plane and swapping the
side token in the parent joint name. This sidesteps the
handedness/anatomy inconsistency that arises if you try to define one
frame that's "correct" for both sides at once.

A marker's `normal` and `offset` are expressed in `(e_long, e_lat,
e_ant)`. At runtime the marker's world normal is FK's parent-bone
rotation applied to `normal` — no per-frame anatomical heuristic.

> **Open decision D1.** `e_lat`/`e_ant` still bake in the rest pose's
> rotation about `e_long`. Fine for a well-defined neutral stance;
> revisit if a rest-twist-ambiguous bone appears in a later marker set.

### 1.11 Deferred: skeleton schema redesign

Recorded, not for this phase (Harri, 2026-09-10). The skeleton YAML
schema has accumulating shortcomings worth a dedicated redesign later:

- Decouple **skeleton topology** (joints, parents, default limits) from
  **marker/keypoint definitions** and from **performer-specific metrics**
  (bone lengths) and **capture-specific marker placement** — currently
  all can end up in one file.
- Introduce an explicit **link/bone** object (this doc deliberately
  avoids it for now and works through the parent joint).
- Explicit **twist joints** (§1.3).

Not started because the full set of schema requirements isn't understood
yet; the marker work here must fit the current schema and will surface
requirements for that redesign.

### 1.5 Offset parametrization (scale-robust)

Do **not** store the offset as raw local `[x,y,z]` cm — that does not
survive per-person bone-length scaling. Store:

```yaml
along:   0.55      # fraction of proximal→child bone length, from proximal joint
lateral: 0.041     # metres, perpendicular offset along e_lat  (signed)
anterior: -0.010   # metres, perpendicular offset along e_ant  (signed)
```

Rationale: the *along-bone* position of a marker scales with the person
(a mid-shank marker stays mid-shank); the *perpendicular* standoff is a
near-constant physical quantity (limb radius + a couple of cm of marker
base) that scales weakly, so metres is the honest unit and per-person
calibration adjusts it. Resolving to a local `[x,y,z]` for FK is
`e_long · (along · bone_len) + e_lat · lateral + e_ant · anterior`.

### 1.6 Normal

```yaml
normal: [0.0, 1.0, 0.2]   # in (e_long, e_lat, e_ant); normalized on load
```

The outward surface normal at the marker, in link-local axes. For a
marker on a roughly cylindrical limb this is close to the perpendicular
offset direction (`normalize([0, lateral, anterior])`) and the catalog
loader may **default** it to that when omitted — but it is a separate
field because it isn't always true (a marker on a flat-ish surface like
the shin, or angled deliberately toward a camera bank).

Used for the visibility prior:
`facing = dot(normal_world, unit(camera_center − marker_world))`;
`facing > cos(θ_max)` → plausibly visible in that camera (θ_max ~100–110°
to allow grazing angles, tuned against GT — see §2.4).

### 1.7 Sides, mirroring, modularity

- Author one side; the catalog declares a **mirror plane** (the
  skeleton's sagittal plane) and generates the opposite side by
  reflecting `offset_local` and `normal_local` and swapping the joint
  name's side token (`hip_L` ↔ `hip_R`). `lateral` stays "away from
  midline" on both sides by construction (§1.4).
- The catalog is a set of **modules** (`leg`, `pelvis`, `torso`,
  `arm`, …), each a list of marker definitions plus the joints they
  reference. A capture's *marker set* is a composition of modules plus
  ad-hoc extra markers. Composition just concatenates; name collisions
  are an error.

### 1.8 Nominal vs. calibrated

Catalog values are **nominal** — hand-authored from a marker-placement
guide / anatomy, good enough to (a) predict projection within the §3
clustering radius and (b) predict visibility sign correctly most of the
time. A **calibrated marker attachment set** for a subject/session
carries the same fields with values fitted from that session's data
(from manual tracklet→slot assignment, then a bundle fit — see §4). The
file format is identical; a `calibration:` block records provenance
(source capture, method, residuals, date).

### 1.9 Proposed file shape

```yaml
# catalog/leg.marker-module.yaml
module: leg
skeleton_topology: <name>          # see §6 — which joint-naming standard
requires_joints: [hip, knee, ankle, foot]   # right-side names; L generated on mirror

# Every entry is defined for the RIGHT side. mirror: true generates the
# left instance (reflect offset/normal across sagittal, swap side token).
markers:
  - name: knee_lat            # -> knee_lat_R, and knee_lat_L via mirror
    parent_joint: knee_R      # rides the tibia
    mirror: true
    along: 0.06               # fraction of knee_R -> ankle_R bone length
    lateral: 0.045            # metres along e_lat (right = away from midline)
    anterior: 0.005           # metres along e_ant
    normal: [0.0, 1.0, 0.1]   # in (e_long, e_lat, e_ant); normalized on load
    placement_note: "lateral tibial plateau, ~1 finger below joint line"
  - name: knee_ant
    parent_joint: knee_R
    mirror: true
    along: 0.05
    lateral: 0.0
    anterior: 0.052
    normal: [0.0, 0.0, 1.0]
  - name: shin_mid            # a genuinely mid-bone marker
    parent_joint: knee_R
    mirror: true
    along: 0.45
    lateral: 0.0
    anterior: 0.04
    normal: [0.0, 0.0, 1.0]
  # ... hip_*, ankle_*, heel, toe ...
```

No "group" field, no vitpose-index field. Grouping is emergent (§3); the
pose keypoint is only a bootstrap anchor and lives in the *bootstrap*
config, not the catalog.

### 1.10 What still has to be decided

| ID | Decision |
|----|----------|
| D1 | Rest-twist reference for bones where `e_lat` is ill-conditioned (§1.4) |
| D2 | `skeleton_topology` / joint-naming standard — feeds from the §6.1 spike |
| D3 | Whether `along` should also be allowed as absolute metres for markers whose position genuinely doesn't scale (e.g. a fixed distance below a joint line) |
| D4 | Twist markers / intermediate joints — deferred with the schema redesign (§1.11) |
| D5 | Stored file: explicit named fields (`along`/`lateral`/`anterior`/`normal`) as in §1.9, vs. the schema's `offset [x,y,z]` + `[z,y,x]` Euler. This doc proposes named fields for authoring, resolved to schema `offset` at load. |
| D6 | Does `bone_tip_offset` reliably point child-ward on the real skeleton in use, or do several joints need `long_axis_ref` (§1.4)? Check against the actual skeleton YAML. |

---

## 2. Metrics and ground truth

Every stage gets its own metric, each against ground truth, each reported
before/after a change. No single "score".

### 2.1 Ground-truth datasets

Two captures minimum, both with multi-camera hand-labeled dot GT:

- **CAL** — a calibration-grade capture: slow, deliberate moves, markers
  well-lit, minimal occlusion. This is the "can the pipeline do it at
  all" bar and the source for per-person calibration.
- **HARD** — fast motion, limbs crossing, one subject passing another,
  self-occlusion. This is the "does it hold up" bar.

GT contents, per capture, on a sampled set of synchronized frames
(target: ≥200 frames per capture, spread across the clip, not one
contiguous block):

1. **Per (camera, frame)**: pixel location of every visible reflective
   dot, each tagged with either a **slot name** (`knee_lat_L`, …) or
   `unlabeled` (prop dot, other subject's marker, glare the labeler is
   sure is not a slot). Pixel locations refined to the real centroid
   (existing `refine_dot_ground_truth.py`).
2. **Per dot, a GT track id** stable across frames within one camera —
   built by linking (1) by hand or semi-automatically and checked.
3. **Per (frame, slot)**: GT 3D position, from triangulating that slot's
   labeled pixels across the cameras that labeled it (≥2).

Tooling gap: `label_dot_ground_truth.py` produces (1)'s pixels but not
slot labels or track ids. Extending it is part of this phase.

### 2.2 Stage A — detection, **per camera**

Match a detection to a GT dot if within ε px (ε = 4 after refinement).

- **Recall** = matched GT slot-dots / visible GT slot-dots.
- **Precision (slots)** = detections matched to a GT slot-dot / all detections.
- **Precision (any reflective)** = detections matched to *any* GT dot
  (slot or `unlabeled`) / all detections — separates "false positive" from
  "correctly found a real dot that just isn't a body marker".
- Reported as a table, one row per camera. The 2026-09-10 finding
  (gopro13_01 / oneplus / insta under-detecting) means the aggregate is
  meaningless; per-camera is mandatory.
- Swept over `dot_threshold_by_camera` and a (new) per-camera
  `max_saturation` to pick per-camera values.

### 2.3 Stage A — tracklet linking

Against GT track ids (§2.1.2), standard multi-object-tracking framing:

- **Fragmentation** = mean tracklets per GT track (1.0 ideal).
- **Tracklet purity** = for each tracklet, fraction of its detections
  belonging to its single most common GT track (1.0 = never spans two
  physical dots). *This is the principled version of the ad-hoc "purity"
  — against GT, not a downstream vote.*
- **Track coverage** = fraction of a GT track's frames covered by its
  dominant tracklet.
- **ID switches** per 100 frames.
- Optionally the composite **IDF1** for a single comparable number.

### 2.4 Stage B — fusion

Match a `FusedPoint` to a GT 3D slot within radius r (r = 3 cm).

- **3D position error** = distance to matched GT slot — distribution
  (median, p90), not just mean.
- **3D recall** = GT slots visible in ≥2 cameras that have a matched
  FusedPoint / all such GT slots.
- **False-merge rate** = FusedPoints whose `views` candidates map to ≥2
  different GT slots / all FusedPoints.
- **View completeness** = for a correctly matched FusedPoint, cameras in
  `views` / cameras whose GT says the slot is visible. (Directly measures
  the "only the winning pair" problem the 2026-09-10 rewrite targeted.)
- Also used to **tune θ_max** for the §1.6 visibility prior: over GT,
  the ROC of `facing` score vs. "GT says this slot is visible in this
  camera".

### 2.5 Stages C–E — assignment

Against GT slot labels, per slot, then macro-averaged:

- **Per-slot precision** = frames slot *S* assigned to a detection GT
  says is *S* / frames slot *S* assigned anything.
- **Per-slot recall** = frames slot *S* assigned correctly / frames GT
  says *S* is visible (in the camera / in ≥2 cameras — reported both).
- **Split** by confidence (`3d` vs `2d`) and by whether the slot has a
  near neighbour that projects within N px in that camera (the case the
  old approach failed) vs. isolated.
- **Temporal consistency** = tracklets that are a single GT slot
  throughout but receive ≥2 assigned names / all single-GT-slot
  tracklets. (Independent of whether the name is *right* — measures
  stability.)
- **Headline** = macro-averaged per-slot F1, plus the temporal-
  consistency number. Two numbers, always reported together.

### 2.6 Reporting

A change is accepted only with a before/after table across **all** stages
it could plausibly touch (a fusion change can move assignment numbers; a
detection change moves everything downstream). Same CAL and HARD frame
sets every time.

---

## 3. Assignment, re-framed around projected clusters + tracklets (sketch)

Full design deferred; the shape, per Harri's proposal:

1. **Predict** every marker slot's 3D position and world normal this
   frame via FK from the markerless result (CAL/calibration) or the
   tracker predict step (live), using the calibrated attachment set
   (§1.8). Project into every camera; compute the §1.6 visibility prior.

2. **Per camera, cluster** {dot detections} ∪ {predicted slot
   projections} by 2D distance. Each cluster is "in this view, these
   detections plausibly explain some of these slots" — an explicitly
   *ambiguous set*, not a forced 1:1.

3. **Narrow each cluster** by the visibility prior (drop slots the normal
   says face away) and by detection shape/size if informative.

4. **Resolve across cameras in 3D**: a set of detections (one per camera,
   from these clusters) that triangulates consistently *and* lands near a
   predicted slot's 3D position is a strong slot match. Competing
   resolutions scored by 3D reprojection error + distance to prediction +
   visibility prior.

5. **Tracklets as the unit** (Harri's hypothesis, to be tested with the
   §2.3 metrics first): rather than resolving per frame, resolve per
   tracklet. Declare two tracklets (different cameras) the same physical
   dot if the per-frame 3D triangulation error over their shared frames
   has median/p90/max under threshold *and* is clearly better than the
   next-best pairing. Assign a whole linked tracklet-group to one slot if
   its per-frame distance-to-prediction is under threshold for
   (nearly) all frames. Per-frame assignment (steps 1–4) becomes the
   fallback for detections no tracklet covers.

6. **Manual override** at the tracklet level (§4).

This subsumes the old Stages C (pelvis normals → replaced by FK normals,
step 1), D (per-camera hybrid → replaced by cluster+3D resolve, steps
2–4) and E (post-hoc majority vote → replaced by tracklet-native
assignment, step 5).

---

## 4. P-B: tracklet-centric calibration (detailed design, 2026-09-11)

Goal: turn real capture data into a **calibrated marker attachment set**
(§1.8) — real, data-fitted `along`/`lateral`/`anterior`/`normal` per slot
— by having a human assign whole **tracklet groups** to slots (a handful
of clicks per clip), not individual frames. This is the manual-first
deliverable (§6.2): it unblocks marker-augmented tracking without
waiting for auto-assignment, and it replaces P-A's guessed `±X`/`±Z`
probe offsets with real ones (P-A's own diagnostic already showed the
guessed probes don't bracket the real ankle markers well).

Five pieces, B1–B5, each a concrete script:

### B1 — per-camera tracklets (no new code)

Already have this: `decode_dot_candidates()`'s 9th column is
`DotTrackletLinker`'s `tracklet_id`. A "per-camera tracklet" is just the
set of `(video_frame, px, py)` for one `(camera_instance_id,
tracklet_id)` pair. Filter to tracklets with a minimum lifetime (propose
**15 frames**) before B2 touches them — a 1-2 frame blip is noise and
not usable for calibration anyway.

### B2 — cross-camera tracklet grouping

`build_tracklet_groups.py` (new). For a detection run + time window:

1. **Pre-filter** each camera's tracklets to those with ≥15 frames *and*
   at least one frame within a generous radius (propose 80px) of *any*
   slot's FK-predicted projection (using the current trial/calibrated
   attachment set) — cuts prop dots, glare, and the other person out
   before the expensive step, without deciding *which* slot yet.
2. **Pairwise test**: for every pair of cameras with overlapping
   calibration, and every pair of their (pre-filtered) tracklets with
   overlapping time ranges (via the sync table): find their **shared
   frames** (both have a detection at the same global time); require
   ≥5 shared frames; triangulate each shared frame's pair
   (`cv2.triangulatePoints`, undistorted) and compute reprojection
   error into both cameras. A genuine match has **low reprojection
   error on (nearly) every shared frame** — this is the real difference
   from per-frame fusion (§3 step 4): a temporally-sustained agreement,
   not a single lucky frame. Propose: median ≤ 3px *and* p90 ≤ 6px
   across all shared frames.
3. **Disambiguate**: for a tracklet with multiple candidate partners
   passing step 2 (common near an ambiguous joint), keep only the
   partner whose median reprojection error is *clearly* better than the
   runner-up (propose: runner-up's median ≥ 1.5× the best's) — otherwise
   leave both candidate edges out and flag the tracklet for manual
   review rather than guessing.
4. **Group**: treat accepted pairs as edges in a graph over all
   tracklets; connected components are tracklet groups (one physical dot
   possibly seen by 3+ cameras). If a component contains a pair that
   *fails* step 2 directly (A–B and B–C accepted but A–C rejected) --
   flag the whole component for manual review instead of forcing it.
5. **Annotate each group** (for B3's UI, not authoritative) with its
   fused 3D trajectory and its mean distance to every slot's current
   FK-predicted trajectory, sorted ascending — a ranked slot suggestion,
   never an auto-assignment.

Output: `tracklet_groups.json` — one entry per group: member
`(camera_instance_id, tracklet_id)` list, frame range, per-pair
reprojection stats, fused 3D trajectory, ranked slot suggestions.

**Validating B2 itself, cheaply**: the existing slot-labelled GT
(`pc_labels_refined.json`) already carries real detections' pixel
positions, which can be matched back to their own `tracklet_id` in the
detection data (the same lookup `prototype_tracklet_smoothed_
assignment.py` already does) — giving **real cross-camera tracklet-group
ground truth for free**, no extra labeling. Before trusting B2's
thresholds, measure grouping precision/recall against this: does the
algorithm link the GT's own same-slot tracklets across cameras, and does
it avoid linking different-slot ones? Tune the median/p90/margin
thresholds above against that, the same way Stage A's thresholds were
tuned against GT rather than guessed.

### B3 — manual tracklet-group → slot assignment UI — **built 2026-09-11**

`label_tracklet_groups_gui.py` (PySide6). Built as designed below, with
two adjustments found via a headless smoke test: suggestions are
translated through `_SLOT_TO_PROBE` (the empirical mapping
`eval_fk_prediction.py` derived) so they're offered under real slot
names, not `_DEFAULT_TRIAL`'s own unlabelled probe names; and the
anchor/member comparison in the trajectory re-derivation had to key on
the full `(camera_id, tracklet_id)` pair, not just the camera, since a
flagged/ambiguous group can contain multiple tracklets from the same
camera. Smoke-tested against the validated 42-48s/5-camera run (24
groups) -- not yet exercised by an actual human review pass. See
status.md for details.

Interaction (as built):

- A list of groups from B2 (sorted by lifetime, longest first), each
  showing its top slot suggestion.
- Selecting a group shows: a few sample frames (one per member camera)
  with the tracklet's own dot highlighted, and its 3D trajectory's
  distance-over-time to its top few suggested slots (a small plot or
  table — "this trajectory sits 8mm from knee_lat_R's prediction and
  41mm from knee_med_R's, throughout").
- Operator actions per group: **assign** a slot (dropdown, defaults to
  the top suggestion); **reject** (not a body marker — prop, other
  person, glare that slipped through B2's pre-filter); **split** (the
  linker silently carried one tracklet_id across two different physical
  dots — a known, accepted `DotTrackletLinker` failure mode, same one
  Stage E's docstring already flags); **merge** (two groups B2 failed to
  link, e.g. only ever visible from non-overlapping camera pairs).
- Output: `tracklet_group_assignments.json` — `{group_id: slot_name |
  "rejected"}`, plus any manual split/merge edits to the group
  membership itself.

### B4 — calibration fit

`fit_calibrated_attachment_set.py` (new). For each assigned group:

- Per frame *t* in the group's lifetime, the parent joint's FK transform
  `T(t)` (from the markerless tracking result, same one P-A validated)
  and the group's fused 3D position `p(t)` give one equation:
  `offset_local ≈ T(t)[:3,:3]^T @ (p(t) - T(t)[:3,3])`. Since
  `offset_local` is constant (rigid attachment), solve the linear
  least-squares fit over every frame at once (equivalently, a robust
  average — use the median or a trimmed mean to absorb the occasional
  bad fused point).
- Convert the fitted local offset to the catalog's scale-robust
  parametrization (§1.5): `along` = the local-Y component ÷ the parent
  bone's length (from `bone_tip_offset`); `lateral`/`anterior` = the
  local-X/Z components directly (already in metres).
- `normal`: default to the offset's own perpendicular direction
  (`normalize([0, lateral, anterior])`, i.e. radially outward from the
  bone axis — §1.6's stated default for a marker on a roughly
  cylindrical limb). A proper fit against observed visibility (which
  cameras saw the marker each frame, tested against "facing" the
  fitted normal) is a real improvement but out of scope for the first
  pass — flag as a follow-up, not blocking.
- **Report residuals**: per-frame `‖predicted_world(t) − p(t)‖` using the
  *fitted* offset, median/p90 — this is the real calibration-quality
  number (bounded below by the markerless tracker's own accuracy and
  the group's own fusion noise, so it's informative even though it's
  not zero).

### B5 — output

A calibrated marker attachment set, same file shape as the nominal
catalog (§1.9), plus a `calibration:` block: source capture/session,
`tracking_run_id`, `detection_run_id`, method, per-slot residual
median/p90, n_frames, date.

### Validation

Re-run `eval_fk_prediction.py` (extended to accept `--attachment-set
<path>` instead of the hard-coded trial probes) using the calibrated
set: purity should rise sharply (real offsets separate real slots,
unlike the generic probe ring) and median error should drop toward
tracking-noise level. Directly answers whether the earlier
`ankle_lat_R`/`ankle_med_R` "both closer to the plain joint anchor"
finding was a probe-placement problem (should resolve here) or something
deeper (would still show up here).

### Open scope questions for this pass

1. **Window to calibrate on**: propose the same capture, full detection
   range (33.62–130.185s, `detection_run 6abcba67`) rather than just the
   40–60s GT window — more shared frames per tracklet pair, better
   grouping confidence, and the GT window still serves as the
   cheap-validation slice within it.
2. **Thresholds in B2** (§ above: 15-frame minimum, 80px pre-filter
   radius, 3px/6px median/p90, 1.5× disambiguation margin) are proposed
   starting points, not tuned — tune against the GT-derived tracklet
   groups before trusting them on the full run.
3. **Reuse vs. new UI** for B3: proposed as a new tool sharing
   `label_marker_slots_gui.py`'s image-display helpers rather than
   extending that tool in place — the interaction model (group list +
   trajectory view, not a frame-by-frame canvas) is different enough
   that bolting it on would likely make both harder to use.
4. Normal fitting (B4) deferred to the radial-direction default, per
   above — revisit only if the visibility prior (§1.6) turns out to need
   better accuracy than that gives.

---

## 5. FK-based prediction (sketch)

- **Calibration context**: the markerless tracking result already exists
  for the capture (`tracking_runs`). Per frame it gives joint angles →
  Pinocchio FK → each link's world pose → predicted slot world position
  and normal (§1.4–1.6). This is not circular for evaluating *assignment*
  because the markerless result never saw the dots.
- **Live context**: the tracker's predict step gives the same joint
  poses one step ahead; the assignment gate uses those. This is exactly
  the design doc's original "Stage B: FK-projected, tight" idea, now with
  the catalog geometry to make it real.
- Shared code: a `predict_marker_slots(skeleton_state, attachment_set,
  cameras) -> {slot: (xyz, normal_world, {camera: (u,v, facing)})}`
  used by both.

---

## 6. Two open strategic questions

### 6.1 Align with an emerging body model?

Context (Harri): as the skeleton model accrues information, and given
that occlusion/collision/physics and result visualization will eventually
need a **body surface**, not just a skeleton, it is worth checking
whether to align with a permissively-licensed parametric human model now,
even if adoption waits.

- **SMPL / SMPL-X**: rejected. Licence restricts commercial use and
  redistribution; joints are regressed from the surface (weak anatomical
  placement, no real joint limits).
- **SOMA / SOMA-X** (NVlabs, `github.com/NVlabs/SOMA-X`): directly
  relevant as **prior art for the assignment problem itself** — SOMA is
  an automatic optical-marker *labeling* method (learned per-frame
  point-cloud → labels, then temporal consistency). Worth studying as an
  *alternative architecture* to §3, not just a dependency. Caveat to
  verify in the spike: the original SOMA's training/runtime leaned on
  SMPL-X assets; confirm SOMA-X's Apache licence covers everything it
  actually needs at runtime, model files included.
- **ANNY** (Naver, `github.com/naver/anny`): a more recently published,
  Apache-licensed parametric body model positioned as a permissive SMPL
  alternative. *Details to verify in the spike* — rig quality, whether it
  ships a usable skeleton topology and surface, dependency weight.

**Recommendation**: do **not** adopt now (agree with Harri that this
isn't the moment). Do a **1–2 day spike** that produces: (a) a licence
audit of SOMA-X and ANNY including model/data assets, (b) a chosen
**reference skeleton topology** to make `skeleton_topology` in §1.9 a
real value — pick joint names and segment definitions that map cleanly to
whichever of these we're most likely to adopt later, so the catalog and
attachment-set files are forward-compatible at near-zero cost, (c) a
short assessment of SOMA's labeling approach as an alternative to §3.
Feed all three back into this doc before building §3–§5.

Cheap alignment we can commit to regardless: keep joint naming and the
link-local frame convention (§1.4) standard and documented, never
bespoke; keep the attachment set a separate document from the skeleton
(already the plan); keep offsets scale-parametrized (§1.5) so they port
across skeleton scales and definitions.

### 6.2 Manual-first vs. auto-first

- **Manual-first** (recommended): §4's tracklet→slot manual assignment
  for calibration captures → calibrated attachment sets → marker-
  augmented tracking working on real data now. Auto-assignment (§3)
  built and tuned afterward against the labels and calibration this
  produces.
- **Auto-first**: build §3 (FK prediction + cluster + 3D resolve +
  tracklet assignment) well enough to calibrate automatically. Higher
  risk, and it has nothing solid to be evaluated against until the GT of
  §2 exists anyway.

Manual-first unblocks the headline goal sooner and de-risks; auto-first
has no evaluation substrate yet. Both need §1 and §2 first, which is why
they lead this doc.

---

## 7. Suggested phase order

1. **§1 catalog spec** finalized (resolve D1–D6; §6.1 spike feeds D2).
2. **§2 ground truth**: extend labeling tooling for slot labels + track
   ids; label CAL and HARD.
3. **§2 metric harness**: one script per stage, run against GT, emit the
   before/after tables.
4. **Stage A per-camera calibration** using the §2.2 harness (the
   cheapest real win — `dot_threshold_by_camera` + new per-camera
   `max_saturation`).
5. **§4 manual tracklet→slot assignment** + calibrated attachment set
   fit → first marker-augmented tracking on calibrated data.
6. **§6.1 spike** (can run parallel to 2–4).
7. **§3 auto-assignment**, built against everything above.

---

## 8. How prototyping continues (concrete, within the current skeleton schema)

The redesign above is mostly still hypothesis. Before committing to it,
build the cheapest experiments that would confirm or kill each load-
bearing assumption, reusing what the `2026-09-06-kare-tests` capture
already has: 3 markerless tracking runs (`tracking_results`, ~69.5k state
vectors + skeleton YAML in `skeletons.yaml_content`), the blacklist-mode
dot detection run `01c2e3c1…`, 6-camera calibration + sync, and the
existing prototype scripts.

### P-A — FK marker-slot prediction (the keystone) — **DONE 2026-09-10**

`python/tools/prototype_fk_marker_prediction.py`, video
`scratch/dot_ground_truth/fk_prediction.mp4` (6-camera grid, 40–50 s).

**Findings:**
- **The keystone premise holds.** On the three cameras that both
  calibrate well and detect the leg dots (gopro13_02, gopro-11_mini_01,
  pixel9), FK-predicted slots from a *nominal* trial attachment set +
  the existing markerless result land a **median ~10–40 px (≈2–4 cm)**
  from a real detection, on the correct body part — comfortably inside a
  usable clustering radius. gopro13_02: 90–98 % of frames matched within
  60 px, median 10–26 px.
- **Extrinsics are fine on all six cameras.** Reprojecting a
  tracker-trusted FK 3D knee vs. the vitpose knee keypoint gives
  ~15–30 px everywhere. An earlier "3 cameras are 200–1500 px off"
  reading was a metric artefact — "distance to nearest *detected* dot"
  is meaningless on a camera that detects almost none.
- **Stage A per-camera detection is the binding constraint, not
  prediction.** gopro13_01 / insta_ace2_pro / oneplus9pro-01 match only
  0–15 % of predictions simply because they detect so few leg dots.
  **P-D is a hard prerequisite for using the other three cameras**, not
  optional polish.
- **The FK-carried normal / visibility prior works.** Paired probe
  markers (`+X` vs `−X` local offset) show consistent *opposite* facing
  signs across cameras; the video's filled-vs-hollow rendering correctly
  separates front- from back-facing predictions. The `Z`-local axis was
  weakly discriminative in this clip (near the viewing direction for most
  cameras) — the per-skeleton anatomical frame (D1) still needs pinning,
  but the FK → world-normal → facing path is sound.
- **D6 resolved**: this skeleton's leg joints have
  `bone_tip_offset = [0, +L, 0]` (local +Y = toward child), usable as
  `e_long` directly.
- Occlusion of the tracked person (by the second person / the man) drops
  prediction usefulness in some views — an argument for the multi-camera
  resolve to lean on whichever cameras have a clear line of sight.

**Follow-ups before P-B leans on this:** author a real trial leg
attachment set with medial/lateral/anterior offsets in the correct local
directions (this pass used unlabelled `±X`/`±Z` probes to discover the
mapping); optionally run P-D first so P-B has more than three usable
cameras.

---

*(original P-A plan, for reference:)*

Purpose: confirm the single assumption everything in §3/§5 rests on —
that a **nominal** attachment set + the markerless pose predicts marker
positions well enough to drive clustering, and predicts visibility sign
correctly.

- Read `tracking_results.state` + the skeleton YAML for the kare capture.
  Do FK in Python (Pinocchio is a system dep; or minimal FK from the
  YAML). Output per frame: each joint's world pose.
- Hand-author a **trial right-leg attachment set** (~8 markers: `hip`,
  `knee_med/lat/ant`, `ankle_med/lat`, `heel`, `toe`) as §1.9 entries
  against the real skeleton's joint names, `mirror: true`.
- Per frame: resolve each slot to world xyz + world normal (§1.4–1.6),
  project into all 6 cameras, compute `facing` (§1.6).
- Render a diagnostic grid video: predicted slot projections (colour per
  slot, dimmed when `facing` says back-facing) over the actual detected
  dots — same renderer family as `render_dot_detection_overview_video.py`.
- **Read-outs**: predicted-vs-nearest-detection pixel error per slot per
  camera (is it < the intended clustering radius?); does `facing` agree
  with "is there actually a detection there"; how much does error vary
  over the clip (→ how badly is per-person calibration needed).
- Also resolves **D6** (is `bone_tip_offset` usable as `e_long`) against
  the real skeleton.

If P-A's nominal prediction is wildly off even after eyeballing the
offsets, that's a strong signal the manual-first path (calibrate first,
predict second) is the only viable order — which is already the
recommendation, but P-A tells us how far nominal gets us for free.

### P-B — cross-camera tracklet grouping + manual assignment — **detailed design in §4**

Purpose: deliver the manual-first path and test Harri's "tracklets are
higher-confidence than per-frame slots" hypothesis. Full B1–B5 design
(algorithm, thresholds, UI interaction, calibration fit, validation
plan, open scope questions) now in §4 above — build in that order.

### P-C — minimal ground truth + metric harness — **annotation tool done 2026-09-10**

- **Done**: `label_marker_slots_gui.py` — a PySide6 GUI (the cv2 tool
  proved too clunky: no menus, every left-click adds a point). A frame
  opens with the detector's dot candidates pre-placed (grey = untagged,
  no text clutter); **right-click a dot → menu → pick slot** (or
  `unlabeled` / Delete); **Ctrl+left-click empty → add a missed dot**;
  left-click selects, number keys `1`–`9`/`0` set the selected dot's
  slot, Delete removes it. Previous same-camera frame's tagged dots show
  as dashed ghosts; "Carry from prev" snaps each to the nearest untagged
  current dot. **Track id is derived, not entered** — one physical
  marker per slot, so `track = slot_palette.index(slot)` (downstream
  metrics key on (camera, track), so a value shared across cameras is
  fine); `unlabeled` dots get no track. Output JSON is the same shape
  (`points: [[x,y],...]` + parallel `point_meta: [{slot, track},...]`),
  so `refine_dot_ground_truth.py` / `validate_dot_detector.py` consume
  it unchanged.
- `label_dot_ground_truth.py` also gained a `--slots` mode + a
  `--seed-detection-run` flag along the way, but the GUI supersedes it
  for slot labeling; keep the cv2 tool for plain (no-slot) point marking.
- **Done**: `build_gt_frame_manifest.py` — resolves each requested global
  time to each camera's own `video_frame` via the sync table and emits
  the manifest `label_dot_ground_truth.py` consumes.
- **Next (the actual labeling)**: build a ~25-time × 6-camera manifest
  across the kare 40–60 s window and hand-label it in slot mode. Then
  the metric scripts (one per stage per §2), starting with Stage A
  detection per camera and a "prediction error vs GT slot" number for
  P-A.

### P-D — per-camera detection calibration

- Using P-C's small GT, sweep `dot_threshold_by_camera` and a **new
  per-camera `max_saturation`** override for gopro13_01 / oneplus /
  insta_ace2_pro (the 2026-09-10 under-detection finding).
- Re-run detection on a short window with tuned params; re-measure.

### P-E — body-model spike (§6.1, independent)

Licence audit (incl. model/data assets) of SOMA-X and ANNY; pick the
reference `skeleton_topology`; assess SOMA's labeling approach as an
alternative to §3.

### Order

P-A and P-C start now, in parallel. P-B after P-A (needs prediction).
P-D after P-C. P-E any time. Reconvene after P-A + P-B to decide whether
§3 auto-assignment is worth building or whether manual-first + calibrated
tracking is enough for the near term.
