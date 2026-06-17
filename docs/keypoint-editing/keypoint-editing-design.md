# Keypoint editing — technical design

## Goals

Allow a user to manually correct keypoint observations for a named-person
sequence **after stitching** (i.e. after `finalise_to_db` has produced
`pose_observations`).  The two fundamental operations are:

1. **Mark as outlier / inlier** — override the automatic outlier flag on one
   or more frames of a keypoint.
2. **Move to new position** — drag or nudge a keypoint to a corrected pixel
   location in one or more frames.

Non-goals for the initial implementation:
* Editing the tracker's smoothed output (only raw observations).
* Real-time propagation of edits to the UKF (edits are written to the DB;
  the tracker must be re-run to pick them up).  A future improvement would be
  incremental re-tracking over the affected frame window.
* Multi-skeleton editing or re-assignment of keypoints between persons.

---

## Pipeline context

Editing fits into the following sequence of steps:

```
1. Detection run
   YOLO + RTMPose per camera → detection_keypoints (anonymous track_ids)

2. Stitch (finalise_to_db)
   User assigns anonymous tracks to named persons
   → pose_observation_sequences + pose_observations

3. Keypoint editing  ← NEW STEP
   User corrects observations for a named person
   → pose_observation_edits  (overlay on pose_observations)

4. Tracking
   C++ UKF reads pose_observations + pose_observation_edits
   → tracking_results
```

Edit mode is entered when the user selects a named person under a detection
run in the session tree.  No tracker run is required.  Editing is useful
before any tracker run as well as between runs.

Re-running step 2 (Finalize) creates a new `pose_observation_sequences.id`,
which will not inherit existing edits.  Users should complete stitching before
editing keypoints.

---

## UI context

The editing view is a new `PersonCropGridWidget` that becomes the central
editing surface.  **It does not seek raw video files.**  Instead it reads JPEG
crop blobs from `frame_cache_entries`, which makes frame scrubbing
instantaneous.

The widget shows one camera crop cell per camera for the selected person +
frame, with the keypoint overlay drawn on top.

```
┌──────────────────────────────────────────────────────────────┐
│ Cam A │ Cam B │ Cam C │ Cam D │  ← crops at current frame   │
│  [img]│  [img]│  [img]│  [img]│                             │
│   ○ ○ │   ○ ○ │   ○ ○ │   ○ ○ │  ← skeleton overlay        │
└──────────────────────────────────────────────────────────────┘
```

### Crop loading

Crops are in `frame_cache_entries`, keyed by the original detection run, not
by the sequence.  The lookup path:

1. `pose_observation_sequences.detection_run_id` → `detection_run_id`
2. `detection_track_assignments` → `(shot_video_id, track_id)` for each
   camera + person.
3. `capture_videos.camera_instance_id` → `shot_video_id` (to go from the
   sequence's `camera_instance_id` to the cache key).
4. Query `frame_cache_entries` by `(detection_run_id, shot_video_id,
   track_id, frame_idx, cache_type='full_body', region_type='full_body')`.

The `src_x, src_y, src_w, src_h` columns from `frame_cache_entries` record
the crop region in the original full-resolution frame.

### Coordinate conversion

Keypoints in `pose_observations` are in full-frame pixel coordinates (native
resolution of the video, post-undistortion).  To draw them on a displayed crop:

```python
display_x = (frame_x - src_x) / src_w * display_w
display_y = (frame_y - src_y) / src_h * display_h
```

Inverse (mouse click → full-frame coordinate for storage in `pose_observation_edits`):

```python
frame_x = src_x + click_x / display_w * src_w
frame_y = src_y + click_y / display_h * src_h
```

### Missing crops (no detection)

When no detection exists for a frame, `frame_cache_entries` has no row for
that camera + frame + track.  See *Bounding box backfill* below.

---

## Data model

### New table: `pose_observation_edits`

The table is an overlay on `pose_observations`.  It is keyed by
`(sequence_id, camera_instance_id, video_frame)`, exactly matching the
`pose_observations` primary key (minus `person_id`, since each sequence has
exactly one person at `person_id = 0`).

```sql
CREATE TABLE pose_observation_edits (
    id                 TEXT PRIMARY KEY,
    sequence_id        TEXT NOT NULL REFERENCES pose_observation_sequences(id),
    camera_instance_id TEXT NOT NULL,
    video_frame        INTEGER NOT NULL,
    -- float32[N, 3]: x, y, is_outlier for each keypoint slot.
    -- Slots not set in kp_mask retain their pose_observations value.
    kp_blob            BLOB NOT NULL,
    -- uint8[ceil(N/8)]: bitmask of which slots this row overrides.
    kp_mask            BLOB NOT NULL,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE UNIQUE INDEX pose_observation_edits_unique
    ON pose_observation_edits (sequence_id, camera_instance_id, video_frame);
```

`N` matches the keypoint count of the pose model that produced the sequence
(`pose_observation_sequences.pose_model`).

### Merge logic

Applied whenever observations are read for display or tracking:

1. Read `pose_observations.kp_blob` for the frame: `float32[N, 3]` =
   (x, y, confidence).  If no row exists (ghost frame), start from zeros.
2. If a `pose_observation_edits` row exists for the same
   `(sequence_id, camera_instance_id, video_frame)`, unpack `kp_blob` as
   `float32[N, 3]` = (x, y, is_outlier) and `kp_mask` as a bitmask.
3. For each keypoint index `i` where `kp_mask[i] == 1`:
   - Replace `x, y` with the edit values (a move edit).
   - Set confidence to `0.0` if `is_outlier == 1`, or restore the original
     confidence if `is_outlier == 0` (forced inlier).
4. Return the merged `float32[N, 3]` and an `is_edited` bool array (True for
   overridden slots) for the overlay to mark edited keypoints.

### Why blob-per-frame with a mask

- Matches `pose_observations.kp_blob` format exactly (same deserialization).
- One UPSERT covers all edits in a frame, including multi-keypoint adjustments.
- The `kp_mask` records only overridden slots without rewriting unchanged ones.

### Ghost frames (no observation)

When the user places a keypoint on a frame with no `pose_observations` row,
a `pose_observation_edits` row is written with `kp_mask` marking only the
placed keypoint slot.  The C++ reader (see below) checks for edit rows even
when no corresponding `pose_observations` row exists.

---

## Tracker integration (C++)

### Reading edits in `load_observations` (`session_reader.cpp`)

The merge query is a simple lookup — no complex join is needed because
`pose_observation_edits` shares the same key space as `pose_observations`:

```sql
SELECT kp_blob, kp_mask
FROM pose_observation_edits
WHERE sequence_id         = :sequence_id
  AND camera_instance_id  = :camera_instance_id
  AND video_frame         = :video_frame
```

For each `pose_observations` row, this lookup is performed once (by frame).
If a row is found, `apply_keypoint_edits()` is called to produce the merged
blob before creating the `Observation` objects.

For ghost-frame edits (no `pose_observations` row), a complementary query
collects orphan edit rows per camera:

```sql
SELECT video_frame, kp_blob, kp_mask
FROM pose_observation_edits poe
WHERE poe.sequence_id        = :sequence_id
  AND poe.camera_instance_id = :camera_instance_id
  AND NOT EXISTS (
      SELECT 1 FROM pose_observations po
      WHERE po.sequence_id        = poe.sequence_id
        AND po.camera_instance_id = poe.camera_instance_id
        AND po.video_frame        = poe.video_frame
        AND po.person_id          = 0
  )
```

These frames are merged from zeros (no underlying detection).

### `apply_keypoint_edits` helper (`src/db/blob_codec.{hpp,cpp}`)

Shared merge function used by `load_observations`:

```cpp
// Applies kp_mask bits from edit_blob over base_blob (in-place).
// base_blob: float32[N, 3] (x, y, confidence)
// edit_blob: float32[N, 3] (x, y, is_outlier)
// mask_blob: uint8[ceil(N/8)]
void apply_keypoint_edits(
    std::vector<Keypoint>& kps,
    void const* edit_blob, int edit_bytes,
    void const* mask_blob, int mask_bytes);
```

### Pre-migration compatibility

`pose_observation_edits` does not exist in DBs created before the migration.
`load_observations` catches the SQLite `SQLITE_ERROR` on the prepare step and
skips edit application silently, making it backward-compatible.

---

## Keypoint trail

When a keypoint is selected (by clicking a dot in any camera cell), a trail
is drawn in that camera's crop view:

- **Past N frames**: red dots connected by a polyline.
- **Future N frames**: blue dots connected by a polyline.
- **Ghost positions** (frames with no `pose_observations` entry): semi-
  transparent grey dots, linearly interpolated between the nearest known
  positions.  Ghost positions are UI-only and are not stored in the DB unless
  the user moves one or marks it as inlier.
- **Edited keypoints**: small yellow marker overlaid on any slot overridden
  by a `pose_observation_edits` row.

Trail radius N is configurable (default: 10 frames each direction).  The
trail always extends to the next/previous real observation, even if beyond N.

Keypoints are identified by their **index in the detection blob** (0-based,
same ordering as the pose model output — up to 133 for full-body COCO-133).
The UI labels each index with the COCO keypoint name derived from
`pose_observation_sequences.pose_model`.

---

## Interaction model

### Mouse (per camera cell in `PersonCropGridWidget`)

| Event | Action |
|---|---|
| Click on keypoint dot | Select that keypoint index; trail updates all cells |
| Click on empty area | Deselect |
| Drag from keypoint dot | Move keypoint; write `pose_observation_edits` row on release |
| Click on ghost dot | Select; drag or Space creates an edit row |

Mouse events arrive in display-crop coordinates; the inverse transform converts
them to full-frame pixel coordinates for storage.

### Keyboard

Key events are captured by `PersonCropGridWidget` (focusable widget).

| Key | Action |
|---|---|
| `a` | Previous frame (loads crop from DB — no video file seek) |
| `d` | Next frame |
| `Shift+A` | Extend frame-range selection to the left |
| `Shift+D` | Extend frame-range selection to the right |
| `←` `→` `↑` `↓` | Nudge selected keypoint ±1 px (full-frame coords) |
| `Shift+←/→/↑/↓` | Nudge ±10 px |
| `Space` | Toggle outlier/inlier for the selected keypoint at the current frame |
| `Esc` | Deselect keypoint / exit edit mode |

### Frame range operations

When a range `[first, last]` is active (extended via `Shift+A/D`):
- `Space` toggles the outlier flag on every frame in the range.
- A drag applies the same pixel delta to every frame in the range (relative,
  not absolute).

---

## Bounding box backfill for undetected frames

Keypoint editing is most useful on frames where detection failed.  Those frames
currently have no JPEG crop in `frame_cache_entries`.

### Two-pass approach

1. **Pass 1 (existing)**: detection pipeline writes crops at the exact
   detected bbox per frame.
2. **Pass 2 (new, post-run)**: for every frame in the run's time range with
   no crop, compute the extended bbox from the union of real detections within
   ±N frames (default N=10), padded by 10%, and write a synthetic crop to
   `frame_cache_entries`.

Pass 2 runs automatically at the end of the pipeline and can also be triggered
manually for existing runs via a menu action.  This avoids a ring buffer in
the hot path (Pass 1) at the cost of bounded re-reading of video frames after
the run completes.

---

## Schema migration

New session DB migration (next available version):

```sql
CREATE TABLE pose_observation_edits (
    id                 TEXT PRIMARY KEY,
    sequence_id        TEXT NOT NULL,
    camera_instance_id TEXT NOT NULL,
    video_frame        INTEGER NOT NULL,
    kp_blob            BLOB NOT NULL,
    kp_mask            BLOB NOT NULL,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE UNIQUE INDEX pose_observation_edits_unique
    ON pose_observation_edits (sequence_id, camera_instance_id, video_frame);
```

---

## Implementation phases

### Phase 1 — data layer (Python + C++)

**Schema:**
- Add `pose_observation_edits` table via a DB migration.

**Python — `db_cache.py`:**
- `read_observations_with_edits(session, sequence_id, camera_instance_id)`
  → reads `pose_observations` + `pose_observation_edits` and returns merged
  blobs per frame (including ghost-frame edit rows).
- `write_observation_edit(session, sequence_id, camera_instance_id,
  video_frame, edits: dict[int, tuple[float, float, int]])` → packs into
  blob/mask format and upserts one row.

**C++ — `src/db/blob_codec.{hpp,cpp}`:**
- `apply_keypoint_edits(kps, edit_blob, edit_bytes, mask_blob, mask_bytes)`
  helper: unpacks both blobs, applies the mask, modifies `kps` in place.

**C++ — `src/db/session_reader.cpp` (`load_observations`):**
- After resolving the camera list, precompute a prepared statement for the
  edit lookup `(sequence_id, camera_instance_id, video_frame)`.
- For each `pose_observations` frame, fetch the edit row and call
  `apply_keypoint_edits` if found.
- After the main frame loop, execute the ghost-frame orphan query and inject
  additional `Observation` objects.
- Catch `SQLITE_ERROR` on statement prepare and skip edits gracefully for
  pre-migration DBs.

**Tests:**
- Python: merge with no edits returns original; edited slots override; ghost
  frame returns merged-from-zeros blob.
- C++: `apply_keypoint_edits` with a known blob+mask produces the expected
  merged array; missing table does not crash `load_observations`.

### Phase 2 — crop grid widget
- Implement `PersonCropGridWidget` loading per-camera JPEG blobs from
  `frame_cache_entries` (using detection_run_id + track_id from
  `detection_track_assignments`).
- Draw merged keypoints from `read_observations_with_edits` as an overlay
  using the crop coordinate transform.
- Support `a`/`d` frame navigation from cached blobs.
- Wire into `PoseExtractionWindow` when a named person is selected.

### Phase 3 — trail overlay
- Implement `KeypointTrailData` and linear interpolation for ghost positions.
- Draw past/future trail polylines and ghost dots on the overlay.
- Update trail when keypoint selection changes.

### Phase 4 — mouse interaction
- Click-to-select: hit-test against displayed dot positions in crop space.
- Drag-to-move: track drag delta in crop space, convert to full-frame coords
  on release, call `write_observation_edit`.
- Ghost-dot interaction: click or drag creates a new `pose_observation_edits`
  row.

### Phase 5 — keyboard shortcuts
- Capture key events in `PersonCropGridWidget`.
- Implement `a`/`d` frame nav, cursor nudge, `Space` toggle.
- Implement `Shift+A/D` frame range selection and bulk operations.

### Phase 6 — bounding box backfill (two-pass)
- `backfill_crops(session, detection_run_id, shot_video_id, n_context=10)`
  in `db_cache.py`: write synthetic crops for undetected frames.
- Call automatically at the end of the detection pipeline.
- Expose as a menu action for existing runs.

---

## Open questions

1. **Freeze / version management** — the brief envisions marking the current
   edit state as a named frozen version.  A `pose_observation_edit_snapshots`
   table linking edit rows to a named snapshot is the natural extension.

2. **COCO keypoint name mapping** — derive from `pose_model` field.  A static
   lookup table (`rtmpose-l-133kp` → 133-entry name list, etc.) should live
   in a new `pose_models.py` module.

3. **Incremental re-tracking** — re-solving only the affected time window by
   warm-starting the UKF from a state checkpoint just before the first edit.
   This requires caching UKF state, which is a significant C++ change.
