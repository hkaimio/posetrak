# Keypoint editing — technical design

## Goals

Allow a user to manually correct keypoint detections inside the detection UI
without running a new detection pass.  The two fundamental operations are:

1. **Mark as outlier / inlier** — override the automatic outlier flag on one
   or more frames of a keypoint.
2. **Move to new position** — drag or nudge a keypoint to a corrected pixel
   location in one or more frames.

Non-goals for the initial implementation:
* Editing the tracker's smoothed output (only raw detections).
* Real-time propagation of edits to the UKF (edits are written to the DB;
  the tracker must be re-run to pick them up).  A future improvement would be
  incremental re-tracking over the affected frame window to show the edit's
  impact without a full rerun.
* Multi-skeleton editing or re-assignment of keypoints between tracks.

---

## UI context

The editing view is a new `PersonCropGridWidget` that becomes the central
editing surface.  **It does not seek raw video files.**  Instead it reads JPEG
crop blobs from `frame_cache_entries`, which makes frame scrubbing
instantaneous.

The widget is added to the existing `PoseExtractionWindow` (or to a new
dedicated editing tab/panel in that window).  Edit mode is entered as soon as
a detection run and a track_id are selected — no tracker run is required.
Editing should be possible and useful before the tracker has ever been run,
specifically to fix obvious keypoint errors that would otherwise corrupt the
tracked result.

---

## `PersonCropGridWidget`

The widget displays one row of camera views for the selected person + frame.
Each cell shows the cached JPEG crop for that camera (loaded from
`frame_cache_entries`), with the keypoint overlay drawn on top.

```
┌──────────────────────────────────────────────────────────────┐
│ Cam A │ Cam B │ Cam C │ Cam D │  ← crops at current frame   │
│  [img]│  [img]│  [img]│  [img]│                             │
│   ○ ○ │   ○ ○ │   ○ ○ │   ○ ○ │  ← skeleton overlay        │
└──────────────────────────────────────────────────────────────┘
```

### Crop loading

For each camera at the current frame:

```sql
SELECT image_data, width_px, height_px, src_x, src_y, src_w, src_h
FROM frame_cache_entries
WHERE detection_run_id = ?
  AND shot_video_id    = ?
  AND frame_idx        = ?
  AND track_id         = ?
  AND cache_type       = 'full_body'
  AND region_type      = 'full_body'
```

`src_x, src_y, src_w, src_h` record the crop region in the original full-
resolution frame, which is needed to:
- draw keypoints in the correct position within the crop
- convert mouse clicks in crop display space back to full-frame pixel
  coordinates for storing edits

### Coordinate conversion

Keypoints are stored in full-frame pixel coordinates (native video
resolution).  To draw them on a displayed crop:

```python
display_x = (frame_x - src_x) / src_w * display_w
display_y = (frame_y - src_y) / src_h * display_h
```

Inverse (mouse click → full-frame coordinate):

```python
frame_x = src_x + click_x / display_w * src_w
frame_y = src_y + click_y / display_h * src_h
```

### Missing crops (no detection)

When no detection exists for a frame, `frame_cache_entries` has no row for
that camera + frame + track.  The widget shows the crop from the nearest
detected frame (±N, same camera) extended to contain the missing frame's
region of interest.  See *Bounding box backfill* below.

---

## Data model

### Design choice: edit overlay table

Edits form a versioned overlay on top of the immutable detection run; original
`detection_keypoints` rows are never mutated.

#### New table: `keypoint_edits`

The table stores one row per **frame** (not per keypoint index), matching the
blob-per-frame structure of `detection_keypoints`.  The `kp_blob` column is a
`float32[N, 4]` array (x, y, is_outlier, is_edited flag) in the same keypoint
order as the source detection model.  A `kp_mask` bitmask records which
keypoint slots are actually overridden; slots with `kp_mask[i] = 0` inherit
the original detection value.

```sql
CREATE TABLE keypoint_edits (
    id               TEXT PRIMARY KEY,
    detection_run_id TEXT NOT NULL REFERENCES detection_runs(id),
    shot_video_id    TEXT NOT NULL,
    track_id         INTEGER NOT NULL,
    video_frame      INTEGER NOT NULL,
    -- float32[N, 3]: x, y, is_outlier for each keypoint slot
    -- Slots not in kp_mask are ignored (original detection value kept)
    kp_blob          BLOB NOT NULL,
    -- uint8[ceil(N/8)]: bitmask of which slots this row overrides
    kp_mask          BLOB NOT NULL,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE UNIQUE INDEX keypoint_edits_unique
    ON keypoint_edits (detection_run_id, shot_video_id, track_id, video_frame);
```

`N` matches the number of keypoints in the source detection blob (17 for
COCO-17, up to 133 for full-body COCO-133, depending on the pose model used
for the run).

**Merge logic** (Python, called from `read_keypoints_with_edits()`):

1. Read the `detection_keypoints.keypoints` blob for the frame:
   `float32[N, 3]` = (x, y, confidence).
2. If a `keypoint_edits` row exists for this frame, unpack `kp_blob` as
   `float32[N, 3]` = (x, y, is_outlier) and `kp_mask` as a bitmask.
3. For each keypoint index `i` where `kp_mask[i] == 1`:
   - Replace `x, y` with the edit values if they differ from 0 (a move edit).
   - Replace the confidence slot with `0.0` if `is_outlier == 1` (outlier
     suppression), or with the original confidence if `is_outlier == 0`
     (forced inlier).
4. Return the merged `float32[N, 3]` alongside an `is_edited` bool array
   (True for each overridden slot) for the overlay to mark edited keypoints.

Frames with a `keypoint_edits` row but no corresponding `detection_keypoints`
row (ghost edits — user placed a keypoint on a frame with no detection) are
handled by a UNION query that returns the edit blob directly with all slots
set to `is_edited=True`.

#### Why blob-per-frame instead of row-per-keypoint

- Matches `detection_keypoints` storage format (same deserialization path).
- A single UPSERT covers an entire frame's edits, including cases where the
  user adjusts several keypoints in the same frame.
- The `kp_mask` records exactly which slots are overridden without
  materialising unmodified keypoints.

### Data pipeline and tracker integration

Edits stored in `keypoint_edits` must reach the C++ UKF tracker.  The current
pipeline is:

```
detection_keypoints  ──(finalise_to_db)──►  pose_observations  ──►  C++ SessionReader
keypoint_edits ──┘ (not yet applied)                                 load_observations()
```

Edits must be applied at **two points** in this pipeline so that the tracker
always uses the correct merged observations.

#### Integration point 1 — `finalise_to_db` (Python, `finalise.py`)

`finalise_to_db` already reads `detection_keypoints` row-by-row and writes the
blob to `pose_observations`.  It must be extended to:
1. Fetch the `keypoint_edits` row for the same
   `(detection_run_id, shot_video_id, track_id, video_frame)`, if one exists.
2. Apply the mask-based merge to the detection blob.
3. Write the merged blob to `pose_observations`.

This ensures that after the user edits and re-finalizes, the tracker always
receives the correct merged observations.  Re-finalization is a single button
click already in the workflow.

#### Integration point 2 — `SessionReader::load_observations` (C++, `session_reader.cpp`)

The C++ tracker loads observations from `pose_observations` keyed by
`sequence_id`.  It must also apply any `keypoint_edits` that post-date the
last finalization — otherwise the user would always have to re-finalize before
tracking, even for quick iteration.

The join chain to resolve `keypoint_edits` from a `pose_observations` row:

```sql
-- (1) From sequence_id → detection_run_id, shot_id, person_name
SELECT pos.detection_run_id, pos.shot_id, sp.person_name
FROM pose_observation_sequences pos
JOIN sequence_persons sp ON sp.sequence_id = pos.id
WHERE pos.id = :sequence_id AND sp.person_id = :person_id

-- (2) From camera_instance_id → shot_video_id (within the shot)
SELECT id AS shot_video_id
FROM capture_videos
WHERE camera_instance_id = :camera_instance_id AND shot_id = :shot_id

-- (3) From (detection_run_id, shot_video_id, person_name) → track_id
SELECT track_id FROM detection_track_assignments
WHERE detection_run_id = :detection_run_id
  AND shot_video_id    = :shot_video_id
  AND person_name      = :person_name

-- (4) Fetch edit for this frame
SELECT kp_blob, kp_mask FROM keypoint_edits
WHERE detection_run_id = :detection_run_id
  AND shot_video_id    = :shot_video_id
  AND track_id         = :track_id
  AND video_frame      = :video_frame
```

In practice, steps (1)–(3) are resolved once per camera (precomputed into a
lookup map before the frame loop) and step (4) is a single indexed lookup per
frame.  If no `keypoint_edits` row exists the observation is used unmodified.

The merge logic mirrors the Python version: unpack `kp_blob` and `kp_mask`,
iterate over set bits, update `x`/`y` and clamp `confidence` to 0 for forced
outliers.  A helper `db::apply_keypoint_edits(kp_data, kp_bytes, edit_blob,
mask_blob)` should be added to `src/db/blob_codec.{hpp,cpp}` and used by
both the Python binding (if any) and the C++ reader.

`SessionReader` is opened `SQLITE_OPEN_READONLY`; no writes occur.  The C++
schema version check is not enforced by the reader (it reads whatever tables
are present), so the migration adding `keypoint_edits` does not require a
corresponding C++ version bump — but it does require the C++ code to handle
the absence of the table gracefully (i.e., treat a missing `keypoint_edits`
table as "no edits").

---

## Keypoint trail

When a keypoint is selected (by clicking a dot in any camera cell), a trail
is drawn in that camera's crop view:

- **Past N frames**: red dots connected by a polyline.
- **Future N frames**: blue dots connected by a polyline.
- **Ghost positions** (frames with no detection): semi-transparent grey dots,
  linearly interpolated between the nearest known positions on each side.
  Ghost positions are UI-only and are not stored in the DB unless the user
  moves one or marks it as inlier.
- **Edited keypoints**: small yellow marker overlaid on any slot overridden by
  a `keypoint_edits` row.

Trail radius N is configurable per session (default: 10 frames each
direction).  The trail always extends all the way to the next/previous real
detection, even if that is further than N frames, so the user can see the
nearest anchor for interpolation.

A keypoint is identified by its **index in the detection blob** (0-based,
same ordering as the pose model output).  The UI labels each index with the
COCO keypoint name (nose, left_eye, …, right_ankle for COCO-17; full set for
COCO-133).  A `kp_index → name` lookup table is needed; it can be derived
from the `pose_model` field of the `detection_runs` row.

---

## Interaction model

### Mouse (per camera cell in `PersonCropGridWidget`)

| Event | Action |
|---|---|
| Click on keypoint dot | Select that keypoint index (trail updates all cells) |
| Click on empty area | Deselect |
| Drag from keypoint dot | Move keypoint; write `keypoint_edits` row on mouse-release |
| Click on ghost dot | Select; drag or Space write a synthetic edit row |

Mouse events arrive in display-crop coordinates.  The inverse transform above
converts them to full-frame pixel coordinates for storage.

### Keyboard

Key events are captured by `PersonCropGridWidget` (focusable widget).

| Key | Action |
|---|---|
| `a` | Previous frame (loads crop from DB) |
| `d` | Next frame |
| `Shift+A` | Extend frame-range selection to the left |
| `Shift+D` | Extend frame-range selection to the right |
| `←` `→` `↑` `↓` | Nudge selected keypoint ±1 px (full-frame coords) |
| `Shift+←/→/↑/↓` | Nudge ±10 px |
| `Space` | Toggle outlier/inlier for the selected keypoint at the current frame |
| `Esc` | Deselect keypoint / exit edit mode |

Frame navigation loads crops from `frame_cache_entries` — no video file seek.

### Frame range operations

When a range `[first, last]` is active (extended via `Shift+A/D`):
- `Space` toggles the outlier flag on every frame in the range.
- A drag applies the same pixel delta to every frame in the range (relative
  move, not absolute repositioning).

---

## Bounding box backfill for undetected frames

Keypoint editing is most useful on frames where detection failed; those frames
currently have no crop in `frame_cache_entries`.

### During a detection run

The detection pipeline processes frames sequentially, so at the time a frame
is written the future detections are not yet known.  The simplest approach is
a **two-pass crop**:

1. **Pass 1 (existing)**: run detector + pose estimator; write detection rows
   and crops at the exact detected bbox per frame.
2. **Pass 2 (new, post-run)**: for every frame in the run's time range that
   has no crop, compute the extended bbox from the union of real detections
   within ±N frames (configurable, default N=10), padded by 10%, and write a
   synthetic crop to `frame_cache_entries`.  Also re-crop detected frames
   using this wider bbox so that the displayed region is stable across frames.

Pass 2 runs automatically at the end of the pipeline.  It requires re-reading
N frames from the video around each gap, which is acceptable since it happens
once at pipeline time, not interactively.  Existing runs can be backfilled via
a menu action that calls the same function.

The two-pass approach avoids the N-frame ring buffer during Pass 1 (which
would complicate the hot path) at the cost of re-reading a bounded number of
video frames in Pass 2.

---

## Schema migration

New session DB migration (next available version after current):

```sql
CREATE TABLE keypoint_edits (
    id               TEXT PRIMARY KEY,
    detection_run_id TEXT NOT NULL,
    shot_video_id    TEXT NOT NULL,
    track_id         INTEGER NOT NULL,
    video_frame      INTEGER NOT NULL,
    kp_blob          BLOB NOT NULL,
    kp_mask          BLOB NOT NULL,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE UNIQUE INDEX keypoint_edits_unique
    ON keypoint_edits (detection_run_id, shot_video_id, track_id, video_frame);
```

---

## Implementation phases

### Phase 1 — data layer (Python + C++)

**Schema:**
- Add `keypoint_edits` table via a DB migration (schema version bump).

**Python — `db_cache.py`:**
- Implement `read_keypoints_with_edits(session, run_id, svid, track_id)`:
  reads the detection blob and applies the mask-based merge.
- Implement `write_keypoint_edit(session, run_id, svid, track_id, frame,
  edits: dict[int, tuple[float, float, int]])`: packs edits into the
  blob/mask format and upserts the row.

**Python — `finalise.py`:**
- Extend `finalise_to_db` to fetch `keypoint_edits` rows per frame and
  apply the mask merge before writing to `pose_observations`.

**C++ — `src/db/blob_codec.{hpp,cpp}`:**
- Add `apply_keypoint_edits(kp_data, kp_bytes, edit_blob, mask_blob)`
  helper: unpacks both blobs, applies the mask, returns the merged
  `float32[N, 3]`.

**C++ — `src/db/session_reader.cpp` (`load_observations`):**
- After resolving camera metadata, precompute the
  `camera_instance_id → shot_video_id` and `shot_video_id → track_id`
  mappings using `capture_videos` and `detection_track_assignments`
  (requires `detection_run_id` and `person_name` from the sequence row).
- For each observation frame, query `keypoint_edits` (single indexed lookup)
  and call `apply_keypoint_edits` if a row is found.
- Handle the case where `keypoint_edits` does not exist (pre-migration DBs):
  catch the SQLite error and skip edit application silently.

**Tests:**
- Python unit tests: merge with no edits returns original; edited slots
  override; ghost frame returns edit blob directly.
- C++ unit tests: `apply_keypoint_edits` with a known blob+mask produces
  the expected merged array; missing table does not crash `load_observations`.

### Phase 2 — crop grid widget
- Implement `PersonCropGridWidget` in `python/app/pose/`:
  - Loads per-camera JPEG blobs from `frame_cache_entries`.
  - Draws merged keypoints (from `read_keypoints_with_edits`) as an overlay
    using the crop coordinate transform.
  - Supports `a`/`d` frame navigation from cached blobs.
- Wire into `PoseExtractionWindow` so it activates when a track segment is
  selected in the stitcher.

### Phase 3 — trail overlay
- Implement `KeypointTrailData` and linear interpolation for ghost positions.
- Extend the overlay to draw past/future trails and ghost dots.
- Update trail when keypoint selection changes.

### Phase 4 — mouse interaction
- Click-to-select: hit-test against displayed dot positions in crop space.
- Drag-to-move: track drag delta in crop space, convert to full-frame coords
  on mouse-release, call `write_keypoint_edit`.
- Ghost-dot interaction: click or drag on an interpolated position creates a
  new `keypoint_edits` row.

### Phase 5 — keyboard shortcuts
- Capture key events in `PersonCropGridWidget`.
- Implement `a`/`d` frame nav, cursor nudge, `Space` toggle.
- Implement `Shift+A/D` frame range selection.
- Apply range operations (bulk toggle, bulk delta move) to all frames in
  range.

### Phase 6 — bounding box backfill (two-pass)
- Implement `backfill_crops(session, run_id, svid, n_context=10)` in
  `db_cache.py`: for each frame without a crop, compute the extended bbox and
  write a synthetic `frame_cache_entries` row.
- Call automatically at the end of the detection pipeline.
- Expose as a menu action in `PoseExtractionWindow` for existing runs.

---

## Open questions

1. **Freeze / version management** — the brief envisions marking the current
   edit state as a named frozen version.  A `keypoint_edit_snapshots` table
   linking edit rows to a named snapshot is the natural extension.  Deferred
   to after Phase 1.

2. **COCO keypoint name mapping** — the pose model name (stored in
   `detection_runs.pose_model`) determines the keypoint count and ordering.
   A static lookup table (`rtmpose-l-133kp` → 133-entry name list, etc.)
   should be added to `db_cache.py` or a new `pose_models.py` module.

3. **Incremental re-tracking** — re-running the full tracker to see the
   effect of edits is slow.  A future optimisation is to re-solve only the
   affected time window by warm-starting the UKF from a checkpoint just
   before the first edit.  This requires caching UKF state, which is a
   significant change to the C++ tracker.
