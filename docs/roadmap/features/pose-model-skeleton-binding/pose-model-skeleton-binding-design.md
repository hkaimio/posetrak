# Pose model and skeleton binding

How a pose model's keypoints reach skeleton markers, how the SOMA-77 marker offsets are stored
and shipped, what happens to existing skeletons and tracking runs, and the anatomical joint-limit
table of the default skeletons.

## 1. Today

- A keypoint blob slot reaches a skeleton marker in one of two ways (`SessionReader::load_observations`,
  `session_reader.cpp`):
  - **Implied layout.** A sequence without `pose_sequence_keypoints` rows has the COCO-133 layout
    implied by `pose_model`. A marker with `openpose_keypoint: i` (loaded as `Marker::coco_id`) binds
    slot `i`. Hand refinement rows (`source = 'hand_l' | 'hand_r'`) are merged into the same index
    space at fixed offsets 91 and 112 (`kHandBaseIdxLeft/Right`, mirrored from `hand_refinement.py`).
  - **Manifest.** A sequence with manifest rows binds slot `keypoint_idx` to the marker whose
    `landmark` equals the row's `name`.
- The reader tries the manifest first and the implied COCO map second for every slot. Its comment
  says the two "never overlap in practice", which holds while a skeleton carries either COCO markers
  or landmark markers. It stops holding when one skeleton carries both: a SOMA slot that no landmark
  marker claims (for example a finger slot whose marker was left out) falls through to the COCO map
  and binds to whatever COCO marker has that index (slot 15, `LeftHandThumb1`, would bind to the
  COCO left ankle).
- The manifest has one layout per sequence (`PRIMARY KEY (sequence_id, keypoint_idx)`), so the
  hand rows of a SOMA sequence cannot be given their own layout.
- The SOMA-77 skeleton is made by a prototype (`build_soma77_skeleton.py`) from a base skeleton in a
  session: COCO markers dropped, one marker per SOMA joint added with `landmark` set and zero offsets;
  offsets are fitted separately (`calibrate_soma77_offsets.py`) against a reference run of one
  capture. Nothing bundled carries SOMA markers.
- Default skeletons are bundled YAML (`python/posetrak/data/skeletons/default_male.yaml`,
  `default_female.yaml`) seeded into databases by content hash (`seed_default_skeletons`). Skeletons
  are immutable and content-addressed; tracking runs record the exact skeleton id.

## 2. Design

### 2.1 A pose model is a keypoint vocabulary

A pose model is described in code by a `PoseModelSpec`: name (the `pose_model` value), keypoint
names in blob order, flip pairs, and the `source` tag written to the manifest (`soma77`, `coco133`,
`hand21`). One registry (`posetrak/detection/pose_models.py`) holds the specs; the estimator, the
viewer (`app/pose/kp_models.py`) and finalisation read it, replacing the copies of the SOMA names in
the prototype scripts.

### 2.2 Every new sequence carries a manifest per observation source

- `pose_sequence_keypoints` gains `obs_source`, the `pose_observations.source` whose blob the row
  indexes (`body`, `hand_l`, `hand_r`); the primary key becomes `(sequence_id, obs_source,
  keypoint_idx)`. Existing rows get `obs_source = 'body'`.
- Finalisation writes manifest rows for every new sequence, whatever the model: a SOMA body, a
  COCO-133 body and hand-refinement rows (`hand21` names) all name their slots.
- The reader binds by manifest only, per source, when the sequence has manifest rows for that
  source; the implied COCO map is used only for a source without manifest rows (every existing
  sequence). The fixed hand offsets 91/112 stay for those old sequences only.

This removes the fall-through hazard of section 1 and lets a skeleton carry markers for several
models: whichever model's sequence is tracked binds its own markers, and the others get no
observations.

### 2.3 Landmark names are model-qualified

A marker's `landmark` is `"<source>:<keypoint name>"`, e.g. `soma77:LeftForeArm`,
`coco133:left_elbow`, `hand21:index_mcp`. Manifest names are written the same way. The prototype's
bare SOMA names stay readable: a manifest row without a prefix matches a landmark without one, so
spike sequences and skeletons keep working.

### 2.4 Default skeletons carry both marker sets

Each bundled default skeleton carries its existing COCO markers and a SOMA-77 marker set:

- 29 SOMA body markers (trunk, head, limbs, hands, feet) on the `main` group, plus the first joints
  of the index and little fingers (`LeftHandIndex1`, `LeftHandPinky1` and their right-side
  counterparts) attached to the hand segment of the `main` group. Wrist and hand markers alone do not
  fix forearm twist and wrist rotation; the two finger joints span the hand's width and do, without
  a hand pass. The other 44 finger keypoints of the body crop are not tracked: they localise at about
  0.20 of the hand size, against 0.05-0.07 for a hand crop, and an accurate finger pose needs the
  hand refinement.
- Hand markers for refined hands bound by `hand21:` landmarks in the hand groups. The existing COCO
  hand markers (`openpose_keypoint` 91-132) are given the matching `hand21:` landmarks, so one hand
  marker set serves both pose models.
- Offsets: the per-bone offsets calibrated in the spike (reference capture, upright poses) are the
  nominal values. They are kept as catalog data, `catalog/pose-models/soma77.marker-offsets.yaml`:
  per SOMA joint, the parent joint of the default skeleton and the offset in that joint's frame, in
  metres, with a header naming the calibration method. A generator (`posetrak skeleton add-pose-model
  soma77 BASE`) applies the file to any skeleton, which is how the default YAML files are produced
  and how a user's own skeleton gets the SOMA markers.

The spike measured that the offsets are pose-dependent (ankle displaced 21% of the segment upright,
less in rolls), so a fixed offset is a calibration, not a constant. Per-subject offset calibration from
the subject's own triangulated joint centres (including non-upright poses) is a later feature of
skeleton scaling, not part of this one.

### 2.5 Existing skeletons and tracking runs

- Nothing existing changes. Tracking runs keep their skeleton ids and results; the defaults get new
  ids because their content changes, and the old default rows stay in every database that has them.
- `capture_persons.default_skeleton_id` pointing at an old default, or at a scaled descendant of it,
  keeps tracking COCO sequences as before. To track a SOMA sequence the person needs a skeleton with
  SOMA markers: the UI offers "add SOMA-77 markers", which runs the generator and stores the result as
  a new version (`parent_id` = the old skeleton). Scale-group calibration results carry over because
  joints and scale groups are unchanged.
- Starting a tracking run whose skeleton binds no marker to the sequence's manifest fails with a
  message naming the missing pose model, instead of the "all keypoints filtered out" error.

### 2.6 Joint limits of the default skeletons

The default skeletons model the knee, the elbow and the wrist as ball joints whose off-axis limits are
anatomically wrong, verified in `default_male.yaml`:

| Joint | Axis limits today (rad) | Problem |
|---|---|---|
| `shin.*` (knee) | x [0, 2.71], y +-1.57, z +-1.57 | side bend and twist of a hinge of +-90 degrees |
| `forearm.*` (elbow) | x [0, 2.44], y +-1.57, z +-1.57 | sideways bend of +-90 degrees |
| `hand.*` (wrist) | x +-1.57, y +-0.52, z +-1.57 | twist +-30 degrees and deviation +-90 degrees |

One table, `python/posetrak/data/skeletons/joint_limits.yaml`, holds the anatomical limits per joint
name and axis for the default skeletons. The default YAML files are generated from the base geometry
plus this table, and a test asserts that hinge joints (knee, elbow, finger interphalangeal joints)
have zero, or a stated small tolerance, on their off-axis limits.

After the spike locked these limits, the tracked wrist deviation still sat at its new limit for much
of a trial. The cause (marker placement on the hand, the hand frame's axes, or forearm twist range) is
examined with the limit change; the limits are not loosened to hide it.

### 2.7 Layered skeleton definition

A skeleton today is one immutable YAML document: joint hierarchy, joint limits, segment dimensions and
every marker in one content-addressed row. Any change to one part, a performer's segment lengths or a
second pose model's markers, produces a whole new skeleton, and nothing records which parts two
skeletons share. The skeleton is split into four layers with different owners and lifetimes:

| Layer | Holds | Varies per | Default |
|---|---|---|---|
| Joint hierarchy | joints, types, axes, parent links, groups, scale groups | tracking need (body, body and hands, prop) | bundled per default |
| Joint limits | lower and upper limit per joint and axis | hierarchy; overridden per performer | `joint_limits.yaml` (section 2.6) |
| Marker set | markers with parent joint, offset, landmark or keypoint binding | detection model, performer, capture | one catalog file per pose model (`coco133`, `soma77`, `hand21`) and per dot layout |
| Dimensions | segment lengths or scale factors | performer | nominal, from the base geometry |

The tracker keeps consuming one resolved skeleton. A resolved skeleton is composed from one layer of
each kind (limits and markers with their overrides applied) and stays immutable and content-addressed,
so every tracking run still records exactly what it tracked and nothing in the tracker changes. What
changes is how skeletons are made and edited: the generator of section 2.4 becomes
`posetrak skeleton compose`, which takes a hierarchy, a limit table, one or more marker sets and a
dimensions set, and records the layers a resolved skeleton came from next to it.

In this feature the layers are files and the default skeletons are composed from them: the
hierarchy and nominal dimensions taken from the current default YAML, the joint-limit table, and the
COCO-133, SOMA-77 and `hand21` marker sets. Storing layers in the registry, per-performer limit and
marker overrides, per-capture marker sets (the calibrated dot attachment sets that live beside a
session database today) and scaling that writes a dimensions layer instead of a new skeleton are a
separate feature, skeleton layers, which designs their tables; the file layers and the composition
step defined here are its starting point and do not need to change for it.

## 3. Acceptance

- A SOMA sequence and a COCO sequence of the same person track with the same default skeleton, each
  binding only its own markers (checked by a reader unit test with a dual-set skeleton and a SOMA blob
  that has unclaimed finger slots).
- A SOMA sequence with refined hands binds the hand rows through their own manifest.
- The SOMA markers of the bundled default reproduce the spike's calibrated-offset results on the
  reference capture's held-out window (leg / arm / head errors within 0.3 cm of the spike figures).
- The joint-limit test fails on the current knee, elbow and wrist limits and passes on the new table.
- Composing the current default's hierarchy, dimensions and COCO-133 marker set with the old limits
  reproduces today's default skeleton apart from key order.
