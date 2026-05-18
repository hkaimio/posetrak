# ArUco Prop Tracking — Design Analysis

**Status**: Feasibility analysis, not yet scheduled for implementation.

---

## Use cases

1. **Rigid prop** (immediate need): A prop with ArUco markers attached. The "skeleton" is just a root joint with the ArUco corners as markers. Tracking produces 6-DOF pose of the prop per frame.
2. **Articulated prop** (future): Markers distributed across a jointed rigid body (e.g. a two-segment tool). Skeleton is a small kinematic chain.
3. **Person + prop** (future): Prop held by or attached to a tracked person. Ideally tracked in one solver run so physical constraints between person and prop are enforced.
4. **Person with attached markers** (future): Additional optical markers on a person's body segments, supplementing keypoint observations.

---

## Key design decisions

### All ArUco markers in one sequence = one track

A detection run for a prop detects potentially multiple ArUco markers per frame simultaneously. These should be treated as a **single track** — analogous to a person track. "Top-left corner of marker ID 1" is then structurally identical to "COCO133 keypoint ShoulderR": a named landmark index on a track.

This means the number of ArUco corners per frame is `4 × n_markers`, where `n_markers` is fixed for a given detection run (set of markers on one prop). This is known at skeleton definition time.

### Blob storage, not row-per-landmark

Person keypoints already use a blob per track per frame (not one DB row per keypoint) for performance. ArUco should follow the same pattern. The blob is indexed by **position in the `marker_ids` list**, not by raw marker ID:

```
float32[n_markers × 4 × 2]   — (cx, cy) for each corner, list-position-major order
```

For example, a run configured with `marker_ids = [3, 6, 7]` produces a `3×4×2 = 24 float` blob where index 0 = marker 3, index 1 = marker 6, index 2 = marker 7. Undetected markers are NaN at their list position. No wasted slots for IDs between used values.

The `marker_ids` list is the decode key and must be persisted on the detection run (not only in the skeleton YAML, since the skeleton should be reusable with different physical marker sets across sessions). Stored as a JSON column on `detection_runs` (e.g. `config_json`) alongside other run parameters.

### Raw corners as observations, not derived pose

ArUco detection yields 4 pixel-space corners per marker per camera. These are fed directly into posetrak as observations. The solver derives the rigid body pose via the UKF/FK pipeline — no intermediate PnP step in the detector. This is consistent with how keypoint triangulation + IK works for persons.

### Skeleton declares named input tracks

Rather than embedding observation source on individual markers, the skeleton definition declares **named input tracks** at the top level. Each track has a type that determines the blob format and landmark naming convention. Individual markers then reference a track + landmark index. At invocation time the tracks are bound to actual detection run IDs — skeleton YAML remains reusable across sessions.

---

## DB schema

### Detection runs

`detection_runs` gets a `detector_type` column (e.g. `'pose'`, `'aruco'`) to distinguish run types. Other columns (`detector_model`, `detector_conf`, etc.) remain shared.

### Observation storage

Reuse the existing `detection_keypoints` table pattern. For an ArUco run, `track_id` is always `0` (single track per run), and the `keypoints` blob encodes the ArUco corners:

```
keypoints blob layout (ArUco, marker_ids = [3, 6, 7]):
  [m3_c0x, m3_c0y, m3_c1x, m3_c1y, m3_c2x, m3_c2y, m3_c3x, m3_c3y,
   m6_c0x, m6_c0y, ..., m6_c3y,
   m7_c0x, m7_c0y, ..., m7_c3y]   — float32, NaN where marker not detected
```

`marker_ids` and `n_markers` are stored in a `config_json` column on the detection run. The C++ reader fetches this once and uses it to decode all blobs for that run.

Alternatively (if the two blob formats become confusing in one table), a `keypoints_format` column (`'coco133'` / `'aruco_corners'`) provides self-description.

### No new observation table needed

Using `detection_keypoints` with typed blobs keeps the schema minimal and avoids a migration until there is a concrete reason to generalise further.

---

## Skeleton YAML

```yaml
# Named input tracks — bound to actual run IDs at invocation time
input_tracks:
  - id: person_pose
    type: coco133
  - id: prop_markers
    type: aruco_corners
    marker_ids: [1, 2, 3, 4]   # which physical ArUco IDs this prop carries

joints:
  - name: root
    type: free
  # ... person kinematic chain ...
  - name: prop_root
    type: free
    parent: right_hand          # attaches prop to person skeleton (use case 3)
                                # or to world root for standalone prop

markers:
  # Person keypoints reference COCO133 landmark indices
  - name: left_wrist
    track: person_pose
    landmark: 94               # COCO133 index

  # ArUco corners: track + marker ID within the set + corner index
  - name: m1_corner_0
    track: prop_markers
    aruco_marker_id: 1
    corner: 0
  - name: m1_corner_1
    track: prop_markers
    aruco_marker_id: 1
    corner: 1
  # ... remaining corners for markers 1-4 ...
```

### Invocation

Tracks are bound at run time via CLI flags (or TOML config):

```
posetrak track skeleton.yaml \
  --track person_pose:run_abc123 \
  --track prop_markers:run_def456
```

For standalone prop tracking (no person), omit `--track person_pose` and use a skeleton with only the prop structure.

---

## Phases

| Phase | Scope | Key changes |
|-------|-------|-------------|
| 1 | ArUco rigid prop (standalone) | `detector_type` column on `detection_runs`. Python ArUco detector writes corners blob to `detection_keypoints` (track_id=0). C++ solver extended to read ArUco blob format. Skeleton `input_tracks` + `aruco_marker_id`/`corner` fields added. CLI `--track` binding. |
| 2 | Multi-track solver | Solver accepts multiple `--track` bindings simultaneously. `input_tracks` list in skeleton can reference more than one track. Enables person+prop in one run. |
| 3 | Combined skeletons | Skeleton can declare joints from different tracks in one kinematic tree. Constraint propagation between person and prop. |

---

## Open questions

1. **Marker ID scoping per detection run**: The same physical ArUco marker ID (e.g. ID 42) could be attached to different props in different trials within a session. Marker IDs are therefore scoped to a detection run, not session-global. The skeleton `marker_ids` list identifies which IDs within a given run are relevant to this prop.

2. **Confidence / partial detection**: No per-corner confidence value — ArUco corners are either detected or not. Missing corners stored as NaN in the blob. Solver must handle frames where some or all markers are absent (same as occluded keypoints).

3. **`config_json` on detection runs**: Both `marker_ids` and `n_markers` stored in a JSON metadata column on `detection_runs`. The skeleton YAML may also carry `marker_ids` for documentation, but the run record is authoritative at decode time (skeleton is reusable across sessions with different physical markers).

4. **Solver initialization for rigid prop**: Person tracking initializes via triangulation + IK. For a rigid prop, initialization from 4×n corners across ≥2 cameras could use DLT triangulation of individual corners followed by a rigid-body fit — a new code path but structurally similar to the existing triangulator.

5. **Keypoints format column**: Whether to add a `keypoints_format` column to `detection_keypoints` for self-description, or rely on the `detector_type` of the parent run. The latter avoids a column addition but requires a join to decode.
