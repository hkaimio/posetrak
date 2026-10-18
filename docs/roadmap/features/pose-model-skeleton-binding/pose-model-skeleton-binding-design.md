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

A pose model is described in code by a `PoseModelSpec` (`posetrak/detection/pose_models.py`): the
`source` tag (`soma77`, `coco133`, `hand21`) that prefixes its landmarks and is written to the
manifest, the keypoint names in blob order, the left/right flip pairs and, for a tree-shaped model,
the parent of each keypoint. One registry (`SPECS`, `spec_for_pose_model`) holds the specs; the
estimator, the viewer (`app/pose/kp_models.py`), finalisation, the skeleton composition and the
binding check read it. `COCO133_HAND_BASE` (91 and 112) gives the COCO-133 slots of the two hands.

### 2.2 Every new sequence carries a manifest per observation source

- `pose_sequence_keypoints` gains `obs_source`, the `pose_observations.source` whose blob the row
  indexes (`body`, `hand_l`, `hand_r`; a `.refined` source uses its plain source's rows); the primary
  key becomes `(sequence_id, obs_source, keypoint_idx)`. Existing rows get `obs_source = 'body'`,
  which also stands for the primary row of a sequence that has no body, such as the `markers` row of
  an object sequence.
- Finalisation writes manifest rows for every new person sequence: the body blob named by its pose
  model (SOMA-77 or COCO-133) and each hand source by the hand21 names of its side. A model the
  registry does not know (the 17-point models) gets none and keeps the implied layout.
- The reader binds each *layer* of a sequence, the primary row and the two hand rows, by its own
  manifest when it has rows, and by the implied COCO map (slot `k` is COCO index `k`, hand slot `k` is
  91 + `k` or 112 + `k`) only when it has none. A slot that no marker claims stays unbound.
- The merged array the editor and the reader work on is the body row with the hand rows laid over its
  COCO hand slots when the body row is COCO-133 wide. A narrower body row (SOMA-77, 77 slots) has no
  hand slots: the left hand is appended after it and the right hand after the left, and only for hand
  rows that have a manifest. `posetrak/db/observation_merge.py` and the C++ reader apply the same rule.

This removes the fall-through hazard of section 1 and lets a skeleton carry markers for several
models: whichever model's sequence is tracked binds its own markers, and the others get no
observations.

### 2.3 Landmark names are model-qualified

A marker's `landmark` is `"<source>:<keypoint name>"`, e.g. `soma77:LeftForeArm`,
`coco133:left_elbow`, `hand21:left_index_1`. Manifest names are written the same way. A hand21 blob
has no side of its own (it is stored as `hand_l` or `hand_r`), so its landmarks carry the side as a
prefix of the keypoint name, the way the COCO-133 hand slots are named (`left_index_1`,
`right_hand_root`).

Two rules keep older data and skeletons working:

- A manifest name without a model prefix (the SOMA sequences of the prototype) matches a landmark
  with the same bare name and, failing that, the landmark `<source>:<name>`, where `source` is the
  manifest row's own source column.
- A `coco133` slot of the body layer and a `hand21` slot of a hand layer are also bound to the marker
  whose `openpose_keypoint` is the slot's COCO-133 index, when no landmark matched. A skeleton that has
  only `openpose_keypoint` markers therefore binds the manifest of a new COCO sequence as before.

A run whose skeleton binds no slot of a sequence's manifest fails before it starts
(`posetrak.skeleton.binding.missing_pose_models` in `track run`, and the C++ reader as a second line)
with a message naming the pose model and the command that adds its markers.

### 2.4 Default skeletons carry both marker sets

Each bundled default skeleton carries its COCO markers and a SOMA-77 marker set:

- 33 SOMA markers on the `main` group: the 29 body markers (trunk, head, limbs, hands, feet) and the
  first joints of the index and little fingers (`LeftHandIndex1`, `LeftHandPinky1` and their right-side
  counterparts) attached to the hand segment. Wrist and hand markers alone do not fix forearm twist and
  wrist rotation; the two finger joints add a lateral extent without a hand pass. The other 44 finger
  keypoints of the body crop are not tracked: they localise at about 0.20 of the hand size, against
  0.05-0.07 for a hand crop, and an accurate finger pose needs the hand refinement.
- The COCO-133 markers (21: nose, ears, arms, legs and feet) carry `coco133:` landmarks and the hand
  markers (40: the COCO-133 keypoints 92-132; the hand root has no marker of its own, the wrist marker
  serves) carry `hand21:` landmarks, so one hand marker set serves both pose models. All keep their
  `openpose_keypoint`.
- Offsets: the per-bone offsets calibrated against a reference run are the nominal values. They are kept
  as catalog data, `catalog/pose-models/soma77.marker-offsets.yaml`: per SOMA joint, the parent joint of
  the default skeleton and the offset in that joint's frame, in metres, with a header naming the
  calibration method. `posetrak skeleton compose --base SKELETON --marker-set soma77` applies the file
  to any skeleton, which is how a user's own skeleton gets the SOMA markers (markers whose parent joint
  the skeleton lacks are left out and reported).

The offsets are pose-dependent (the ankle is displaced by 21% of the segment upright, less in rolls),
so a fixed offset is a calibration, not a constant. Per-subject offset calibration from the subject's
own triangulated joint centres (including non-upright poses) is a later feature of skeleton scaling,
not part of this one.

The SOMA index and little-finger base joints lie only 2.3 cm apart across the hand (the index and
little-finger knuckles of the COCO hand markers are 6 cm apart) and 4-5 cm from the wrist, so they are
a weak lever for the hand's twist; see section 2.6.

### 2.5 Existing skeletons and tracking runs

- Nothing existing changes. Tracking runs keep their skeleton ids and results; the defaults get new
  ids because their content changes, and the old default rows stay in every database that has them.
- `capture_persons.default_skeleton_id` pointing at an old default, or at a scaled descendant of it,
  keeps tracking COCO sequences as before (the implied layout, or the `coco133` manifest of a new
  sequence, binds its `openpose_keypoint` markers). To track a SOMA sequence the person needs a skeleton
  with SOMA markers: `posetrak skeleton compose --base ... --marker-set soma77` makes one, to be stored as
  a new version (`parent_id` = the old skeleton) with `skeleton import`. A UI action "add SOMA-77
  markers" that does this in one step is not part of this feature. Scale-group calibration results
  carry over because joints and scale groups are unchanged.
- Starting a tracking run whose skeleton binds no marker to the sequence's manifest fails with a
  message naming the missing pose model, instead of the "all keypoints filtered out" error.

### 2.6 Joint limits of the default skeletons

Limits are per joint and axis, the components of the joint's rotation vector in its rest frame. The
default skeletons modelled the knee, the elbow and the wrist as ball joints whose off-axis limits were
anatomically wrong:

| Joint | Axis limits before (rad) | Problem |
|---|---|---|
| `shin.*` (knee) | x [0, 2.71], y +-1.57, z +-1.57 | side bend and twist of a hinge of +-90 degrees |
| `forearm.*` (elbow) | x [0, 2.44], y +-1.57, z +-1.57 | sideways bend of +-90 degrees, and a twist that competes with the wrist's |
| `hand.*` (wrist) | x +-1.57, y +-0.52, z +-1.57 | twist +-30 degrees, deviation and flexion +-90 degrees |

In the hand frame y runs along the hand (twist), x is the palm normal (radial and ulnar deviation) and
z the axis across the palm (flexion and extension).

`python/posetrak/data/skeletons/joint_limits.yaml` holds the anatomical limits per joint name and axis.
The knee and the elbow are hinges (x only; y and z locked), the forearm's pronation and supination moves
to the wrist (y +-85 degrees), the wrist's flexion and extension is +-70 degrees and its deviation +-25
degrees. A test (`tests/skeleton/test_joint_limits.py`) asserts that hinge joints (knee, elbow, finger
interphalangeal joints) have zero off-axis limits and that the check fails on the earlier table.

Why the tracked wrist sat at its limit. Tracked states of a SOMA run with the earlier table show the
axis at its limit to be the wrist twist (`hand.*` y, +-30 degrees): in a multi-person trial of falls and throws it
sits within 3% of that limit in 58% (left) and 81% (right) of the states, while the elbow's twist
(`forearm.*` y, +-90 degrees) saturates at the same time (71% on the right) with the opposite sign, and
the elbow's rotation-vector y is not a twist when the elbow is also flexed by 90 degrees. The twist the
poses need is larger than the wrist allowed (the net twist of elbow and wrist spans -66 to +110
degrees, 5th to 95th percentile, on the right), the elbow absorbed the rest in a parametrisation that mixes flexion and twist, and
the markers observe the twist poorly: the SOMA index and little-finger base joints are 2.3 cm apart.
Deviation (x) and flexion (z) stay well inside +-90 degrees in the same states. The limits are not
loosened to hide this: the twist range moved to the joint that owns it, the elbow lost its twist, and
what remains is a measurement problem (a hand pass, or markers with a wider base) and, for large twists
combined with large swings, the per-axis rotation-vector limits that the swing-twist design replaces.

### 2.7 Layered skeleton definition

A skeleton today is one immutable YAML document: joint hierarchy, joint limits, segment dimensions and
every marker in one content-addressed row. Any change to one part, a performer's segment lengths or a
second pose model's markers, produces a whole new skeleton, and nothing records which parts two
skeletons share. The skeleton is split into four layers with different owners and lifetimes:

| Layer | Holds | Varies per | File |
|---|---|---|---|
| Joint hierarchy | joints, types, rest orientations, parent links, groups | tracking need (body, body and hands, prop) | `data/skeletons/layers/hierarchy-body-hands.yaml` |
| Joint limits | lower and upper limit per joint and axis | hierarchy; overridden per performer | `data/skeletons/joint_limits.yaml` (section 2.6) |
| Marker set | markers with parent joint, offset, landmark or keypoint binding, group membership | detection model, performer, capture | `data/skeletons/layers/marker-sets/{coco133,hand21}.yaml`, `catalog/pose-models/soma77.marker-offsets.yaml` |
| Dimensions | joint offsets and bone tips (segment lengths) | performer | `data/skeletons/layers/dimensions-{male,female}.yaml` |

The tracker keeps consuming one resolved skeleton. A resolved skeleton is composed from one layer of
each kind and stays immutable and content-addressed, so every tracking run still records exactly what it
tracked and nothing in the tracker changes. `posetrak skeleton compose` (module
`posetrak.skeleton.compose`) takes a hierarchy, a limit table, marker sets and a dimensions set (the
bundled ones by default) and writes the YAML; the layers it came from are recorded under a top-level
`layers:` key, which the tracker ignores. With `--base SKELETON` it adds marker sets to an existing
skeleton instead. `python -m posetrak.skeleton.compose` regenerates the bundled default skeletons after
a layer changes, and a test fails when the files and their composition differ.

A marker set file names the markers of one pose model (`pose_model:`, `markers:`, and either a `group:`
every marker joins or per-group `groups:` lists whose order is kept). Marker lists are concatenated set
by set, so the markers of the composed default are listed by pose model (COCO body, hand, SOMA) and not
interleaved as in the earlier file; the group lists are identical. Composing the earlier hierarchy,
dimensions, limits and COCO-133 and hand sets without landmarks reproduces the earlier default skeleton
apart from key order and marker order (a test pins this).

Storing layers in the registry, per-performer limit and marker overrides, per-capture marker sets (the
calibrated dot attachment sets that live beside a session database today) and scaling that writes a
dimensions layer instead of a new skeleton are a separate feature, skeleton layers, which designs their
tables; the file layers and the composition step defined here are its starting point and do not need to
change for it.

## 3. Acceptance

- A SOMA sequence and a COCO sequence of the same person track with the same default skeleton, each
  binding only its own markers (checked by a reader unit test with a dual-set skeleton and a SOMA blob
  that has unclaimed finger slots).
- A SOMA sequence with refined hands binds the hand rows through their own manifest.
- The SOMA markers of the bundled default reproduce the calibrated-offset results of the prototype on
  the reference capture's held-out window (leg / arm / head errors within 0.3 cm of 2.4 / 4.0 / 2.6 cm).
- The joint-limit test fails on the current knee, elbow and wrist limits and passes on the new table.
- Composing the earlier default's hierarchy, dimensions and COCO-133 marker sets with the earlier limits
  reproduces the earlier default skeleton apart from key order and marker order.
