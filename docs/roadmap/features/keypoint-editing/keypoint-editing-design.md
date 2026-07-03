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

Keypoints in `pose_observations` are in **distorted pixel space** (K_original
— raw video frame coordinates).  `finalise_to_db` writes
`pixels_are_undistorted = 0`; the C++ tracker undistorts them internally.
`pose_observation_edits` must store coordinates in the same distorted space so
the tracker applies undistortion consistently.

To draw keypoints on a displayed crop:

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

**Validation:**
- Select a person with a known detection run; verify each camera cell shows
  the correct JPEG crop at the correct frame and that `a`/`d` advances without
  seeking video files (confirm via profiling or log output).
- Verify keypoints drawn on a crop correspond to the correct pixel position:
  place a known detection (e.g. right shoulder) and confirm the dot lands on
  the shoulder in the displayed crop image.
- Verify switching between frames with and without detections (ghost frames)
  shows a placeholder / fallback crop rather than crashing.

### Phase 3 — trail overlay
- Implement `KeypointTrailData` and linear interpolation for ghost positions.
- Draw past/future trail polylines and ghost dots on the overlay.
- Update trail when keypoint selection changes.

**Validation:**
- Select a keypoint at a frame in the middle of a detected run; verify that
  red (past) and blue (future) trail dots appear at the correct positions and
  connect with polylines.
- Introduce a gap (frames with no detection) and verify ghost dots appear
  at linearly interpolated positions between the flanking real detections.
- Verify ghost dots are not written to the DB (check `pose_observation_edits`
  remains empty until the user interacts with them).

### Phase 4 — mouse interaction
- Click-to-select: hit-test against displayed dot positions in crop space.
- Drag-to-move: track drag delta in crop space, convert to full-frame coords
  on release, call `write_observation_edit`.
- Ghost-dot interaction: click or drag creates a new `pose_observation_edits`
  row.

**Validation:**
- Click a known keypoint dot; verify it is highlighted and its index / name
  is shown in the active keypoint panel.
- Drag a keypoint by a known pixel delta (e.g. 20 px right in crop space);
  after release, read the `pose_observation_edits` row and verify the stored
  `x` coordinate has changed by the expected full-frame delta (accounting for
  the `src_w / display_w` scale factor).
- Drag to a ghost frame position; verify a new `pose_observation_edits` row
  is created with the correct `video_frame`.
- Verify clicking empty space deselects the keypoint.

### Phase 5 — keyboard shortcuts
- Capture key events in `PersonCropGridWidget`.
- Implement `a`/`d` frame nav, cursor nudge, `Space` toggle.
- Implement `Shift+A/D` frame range selection and bulk operations.

**Validation:**
- Press `a`/`d` and verify the displayed frame index changes by ±1 and the
  crop updates immediately.
- Select a keypoint, press `→` once; verify the stored x coordinate in
  `pose_observation_edits` increases by 1 (full-frame pixel).
- Press `Shift+→` and verify an increase of 10 px.
- Press `Space`; verify `is_outlier = 1` is written.  Press `Space` again;
  verify `is_outlier = 0`.
- Extend a range with `Shift+D` over 5 frames, then press `Space`; verify
  `pose_observation_edits` rows are written for all 5 frames with
  `is_outlier = 1`.

### Phase 6 — union-bbox crop for undetected frames

On-demand backfill (started in the `CropBackfillWorker` thread when edit mode
is entered) currently uses the single nearest available bbox as the crop
region for frames without a detection.  This can be too tight when the
person has moved significantly.

**Change**: replace nearest-single-bbox with the **union** of all detected
bboxes within ±N frames (N = 10), padded by 10% on each side.  The result is
`(min_x1, min_y1, max_x2, max_y2)` across the window, converted to a
`(cx, cy, w, h)` bbox and passed to `_encode_crop`.

- Short detection gaps: the union stays tight (nearby bboxes are similar).
- Long gaps or fast motion: the union widens to cover the person's last-known
  range on both sides.
- No nearby bboxes at all (camera with no detections): fall back to full-frame
  downscale as before.

Memory-backed crops for undetected frames (those generated by the union-bbox
path) are **not** written to `frame_cache_entries` — they live in
`CropBackfillWorker._mem_results` for the lifetime of the edit session.
This avoids polluting the persistent cache with synthetic entries.

The status bar at the bottom of the main window is used to show backfill
progress ("Generating crops… N remaining") and other transient operation
feedback (copy count, interpolation result, etc.).

**Validation:**
- Scrub to a frame in the middle of a detection gap; verify the displayed
  crop is wider than a single-nearest-bbox crop and encompasses the
  person's position on both flanking detected frames.
- Verify long gaps (>10 frames) display a crop wide enough to show the
  person after they reappear.
- Verify detected frames are unaffected (their DB-backed crops are used,
  not the union-bbox path).

### Phase 7 — ghost-frame keypoint placement

Frames with no `pose_observations` row can now be displayed (via the
memory-backed crop from Phase 6).  This phase allows the user to place
keypoints on those frames.

**Data layer**: `read_observations_with_edits` is extended to also return
ghost-frame rows — i.e. frames that have a `pose_observation_edits` row but
no corresponding `pose_observations` row.  These are merged from zeros
(all-zero kp array) and returned alongside real observation frames.

**UI**: when the user is in edit mode and has a keypoint selected
(`_sel_kp_idx is not None`), clicking **anywhere** on a cell that has no
existing observation (obs_kp is None) places the selected keypoint at the
click location.  This writes a `pose_observation_edits` row for that
`(sequence_id, camera_instance_id, video_frame)` with only the one keypoint
slot marked in `kp_mask`.  The existing drag-to-move path then works
identically for subsequent edits on the same frame.

Clicking on an empty area when obs_kp is None (and no kp selected) deselects
as normal — there is no "place-all" mode.

**Validation:**
- Scrub to a ghost frame; verify a crop is displayed (Phase 6 prerequisite).
- Select a keypoint (e.g. nose) on a flanking detected frame, then scrub to
  the ghost frame and click; verify a `pose_observation_edits` row is written
  with only the nose slot set.
- Verify the placed dot appears in the cell immediately after placement.
- Verify that scrubbing away and back to the ghost frame shows the placed dot
  (merged from the edit row).

### Phase 8 — multi-keypoint selection

Selection model changes from a single selected index to a **set**:

```python
_sel_kp_indices: set[int]       # all currently selected
_primary_kp_idx: int | None     # last explicitly clicked — drives trail display
```

Existing operations (nudge, Space toggle) apply to all indices in
`_sel_kp_indices`.  Trail is shown for `_primary_kp_idx` only to avoid
visual clutter.

**Selection interactions:**

| Interaction | Result |
|---|---|
| Click on dot | Sole selection (replaces set); sets primary |
| Ctrl+click on dot | Add/remove from set; sets primary to clicked index |
| Drag on empty area | Rubber-band rect; all dots inside the rect replace the set |
| Right-click on cell | Context menu with named groups (see below) |
| Esc | Clear selection |

**Named group context menu** (populated from `PoseModel.group_names`):

```
Select…
  ├ Face
  ├ Left arm
  ├ Right arm
  ├ Left hand
  ├ Right hand
  ├ Left leg
  ├ Right leg
  ├ Left foot
  ├ Right foot
  ├ Body
  ├ Upper body
  ├ Lower body
  └ All
────────────
  Deselect all
```

Groups not present in the current model (e.g. "Left hand" in a COCO-17
sequence) are hidden from the menu.  Group index sets come directly from
`self._pose_model.group_indices(group_name)`.

**Validation:**
- Ctrl+click three keypoints; verify all three show selection rings.
- Rubber-band drag across two dots; verify exactly those two are selected.
- Right-click → "Left hand"; verify all 21 left-hand indices are selected
  (COCO-133 sequence), or item is absent (COCO-17 sequence).
- Nudge with multi-selection; verify all selected keypoints move by the
  same pixel delta.

### Phase 9 — copy / paste keypoints

**Clipboard** (in-memory, per session):

```python
_clipboard: dict[int, tuple[float, float, float]] | None
# kp_idx → (x, y, is_outlier)  — full-frame coords
```

| Key | Action |
|---|---|
| Ctrl+C | Snapshot all `_sel_kp_indices` from current merged obs into `_clipboard` |
| Ctrl+V | Write `_clipboard` entries to current frame via `pose_observation_edits` |

Paste only touches keypoints present in the clipboard; other slots are
unchanged.  Pasting onto a ghost frame is valid (creates a new edit row).

The status bar shows "Copied N keypoints" after Ctrl+C and
"Pasted N keypoints" after Ctrl+V.

**Validation:**
- Copy 5 keypoints from frame 100; scrub to frame 150 (ghost frame); paste;
  verify 5 `pose_observation_edits` slots written at the same full-frame
  coordinates.
- Verify paste on a frame with existing edits merges correctly (only copied
  slots are overwritten).

### Phase 10 — frame range selection + linear interpolation

**Frame range** (extends the `Shift+A/D` keyboard shortcuts from Phase 5):

```python
_range_start: int | None   # frame index
_range_end:   int | None   # frame index (inclusive)
```

`Shift+A` anchors at the current frame and extends the range leftward;
`Shift+D` extends rightward.  The active range is highlighted on the
time scrubber bar.  Pressing `A` or `D` without Shift clears the range.

**Linear interpolation** (`I` key, active only when a range and at least one
keypoint are selected):

For each `kp_idx` in `_sel_kp_indices`:
1. Scan `_obs_kp[cam_id]` (merged observations + edits) backwards from
   `_range_start - 1` to find the nearest frame with `conf > 0` →
   *left anchor* `(frame_l, x_l, y_l)`.
2. Scan forwards from `_range_end + 1` → *right anchor*
   `(frame_r, x_r, y_r)`.
3. If either anchor is missing for this keypoint, skip it.
4. For each frame `f` in `[_range_start, _range_end]`, compute:
   ```
   t = (f - frame_l) / (frame_r - frame_l)
   x = x_l + t * (x_r - x_l)
   y = y_l + t * (y_r - y_l)
   ```
   Write as a `pose_observation_edits` row (is_outlier = 0).
5. Reload `_obs_kp` and refresh display.

This supports both gap-filling (place anchors at detection edges, interpolate
the gap) and outlier smoothing (select the noisy range, interpolate between
the last good frame on each side).

Anchor lookup uses merged observations so previously placed (Phase 7) or
pasted (Phase 9) keypoints can serve as anchors.

**Validation:**
- Keypoint at frame 1 (1.0, 1.0) and frame 10 (10.0, 10.0); select range
  4–7; press I; verify frames 4–7 receive edits at (4,4), (5,5), (6,6),
  (7,7) and frames 2, 3, 8, 9 are untouched.
- Select a range where one keypoint has no left anchor; verify that keypoint
  is skipped and others are interpolated normally.
- Verify the range highlight disappears after interpolation completes.

---

## Keypoint model definitions

`python/app/pose/kp_models.py` (implemented) provides ordered keypoint name
tuples and named selection groups for each supported pose model.  The factory
`get_pose_model(model_name)` maps DB strings like `"rtmpose-l-133kp"` to the
appropriate `PoseModel` instance, falling back to COCO-17 for unknown names.

`PersonCropGridWidget` queries `pose_model` from
`pose_observation_sequences` on load and stores `self._pose_model`, which
drives keypoint name display (already wired) and named-group selection
(Phase 8).

Hierarchy (parent→child joint tree) is omitted for now: COCO-133 has no
single hip or neck node, so a proper tree requires virtual root nodes.  The
reference implementation in the `pose-editor` project uses `anytree` with
virtual "Hip" and "Neck" nodes; the same approach can be added to
`kp_models.py` when hierarchy-based selection is needed.

---

# Improvements (second iteration)

The sections below extend this design per
`keypoint-editing-improvements-brief.md`, plus a fourth section (*Background
wide-crop frame cache*) addressing a follow-up problem raised during review.
They continue the phase numbering from the phases above and follow the same
validation-per-phase convention.

## Timeline view

### Motivation

The crop grid is efficient for spatially correcting a keypoint once you know
which frame it's wrong on, but there is no way to see *where in the trial*
problems are without scrubbing through it manually.  A dope-sheet-style
timeline — one row per keypoint, colored by status, collapsible into body-part
groups — surfaces the temporal pattern at a glance, similar to Blender's dope
sheet or Cascadeur's timeline.

### Status signal (no new tables for classification)

Status has **two independent axes**, not one merged color scale. An earlier
draft of this section collapsed "tracker-classified outlier" and
"user-disabled" into a single grey; that's a problem because (a) the
enable/disable and multi-keyframe-interpolation logic only care about the
user's edit intent, not the tracker's verdict, and (b) an edited/enabled
keypoint can still come back as an outlier the next time the tracker runs —
merging the two would make a cell's color contradict itself across runs.

**Axis 1 — edit state** (stable; independent of any tracking run), the cell's
fill color:
- **grey** — user-disabled: an edit row exists with `is_outlier == 1`.
- **blue** — edited/moved: an edit row exists with `is_outlier == 0` and a
  position different from the original.
- **yellow** — original detection, outside person segmentation.
- **green** — original detection, inside person segmentation (also the
  default when no segmentation-quality run exists for the sequence —
  segmentation is a refinement signal, not a requirement).
- Precedence when computing an aggregate cell (see *Row hierarchy* below):
  grey > blue > yellow > green.
- Segmentation lookup: `keypoint_obs_quality.quality_blob`, keyed by
  `(seg_run_id, shot_video_id, video_frame, track_id)` and already populated
  by `python/tools/add_seg_quality.py`. The `seg_run_id` / `shot_video_id` /
  `track_id` for a given camera + person resolve via the same
  `detection_track_assignments` lookup path already used for crop loading
  (see *Crop loading* above).

**Axis 2 — last-run tracker verdict** (only present once a tracking run is
selected; changes on every re-run), drawn as an **overlay** on top of the
axis-1 fill rather than folded into it — a thin red outline/hatch, sourced
from the selected run's `tracking_obs_results.obs_blob` (`is_outlier` field,
per the schema notes in `CLAUDE.md`). This lets both facts coexist: a
keypoint the user edited (blue fill) that the tracker's outlier rejection
still didn't trust in the last run (red outline) is a legible, useful signal
— "your edit didn't fix it" — without the edit-state color changing every
time the tracker re-runs.

### Row hierarchy and camera scope

Editing is inherently per-camera — moving or disabling a keypoint only
affects the camera cell it was edited in.
`PersonCropGridWidget._sel_cam_idx` (`content_panels.py:2218`) already tracks
exactly this: "the camera that last emitted `keypoint_selected`", and is what
nudge/`Space` already act on today (`content_panels.py:3048`, `:3077`). The
timeline reuses the same scope instead of inventing a new definition of
"selected camera":

- **Default view**: single camera = `_sel_cam_idx`, i.e. whichever crop cell
  the user last clicked a keypoint dot in. This keeps the timeline consistent
  with what `Space`/nudge would actually edit if pressed right now.
- A camera tab/dropdown at the top of the timeline switches which camera it
  displays — and, symmetrically, setting `_sel_cam_idx` (e.g. by clicking a
  keypoint in a crop cell) switches the timeline's tab to match, so the two
  controls stay in sync in either direction.
- An explicit **"all cameras" toggle** switches to an aggregated overview:
  one row per keypoint, striped by the axis-1 precedence across cameras (the
  "split coloring" technique the brief describes for body-part rows, applied
  one level down), with an expand arrow revealing one sub-row per camera.
  This is a scan-for-problems view, not the primary editing view — editing
  interactions (rubber-band, `Space`, drag) stay disabled while it's active,
  since there's no single camera for them to target.
- **Inlier-count hint**: every single-camera-mode cell also gets a thin bar
  beneath it, filled to `n_cameras_with_inlier / n_cameras` for that
  (keypoint, frame) — computed once from the same merged-observation data
  already loaded for all cameras (no extra query). This directly answers "do
  I need to bother fixing this camera": a mostly-filled bar means several
  other cameras already have a good observation for this keypoint at this
  frame, so a gap or outlier in the *currently shown* camera is usually not
  worth manually correcting. The bar's value is the same regardless of which
  camera tab is active, since it counts across all cameras.

Group rows (Face, Left arm, …) reuse `PoseModel.groups` (already implemented
in `kp_models.py`) directly — no need to build the anytree parent/child
hierarchy flagged as future work in *Keypoint model definitions* above; a
flat named-group tree is sufficient for a dope sheet (it doesn't need
joint-chain semantics, only grouping). Group rows aggregate the same way as
the "all cameras" keypoint rows: striped swatch across child keypoints,
collapsible.

### Widget

New `KeypointTimelineWidget(QWidget)`, custom-painted (`paintEvent`) to match
the existing idiom — the codebase has no `QGraphicsView` usage anywhere
(`_ImageCanvas`, `FilmstripBarItem` are both hand-painted `QWidget`s) and a
dope sheet with hundreds of small flat-colored cells doesn't benefit from a
scene graph. Docked below the crop grid inside `PersonPanel`.

- X axis: frame index. Independent horizontal zoom/scroll (mouse wheel +
  modifier, or `+`/`-`), since one-pixel-per-frame won't fit a multi-minute
  trial. The existing global `QSlider` (`content_panels.py:2429`) remains the
  coarse scrub control; a vertical playhead line on the timeline stays in
  sync with it via the existing `time_changed` signal.
- Y axis: fixed-height tree rows (~16 px), vertically scrollable.
- Zoomed-out cells aggregate multiple frames using the same precedence
  ordering (worst-status-wins striping) plus a mean fill fraction for the
  inlier-count bar, so problem regions stay visible even when zoomed out to
  see the whole trial.
- A camera tab strip and the "all cameras" toggle (see *Row hierarchy* above)
  sit above the tree.

### Interaction

- Rubber-band drag over rows × frame span → sets both `_sel_kp_indices`
  (rows touched) and `_range_start`/`_range_end` (columns touched) in one
  gesture — a natural generalization combining Phase 8 (multi-select) and
  Phase 10 (frame range). Disabled while the "all cameras" overview is active
  (see *Row hierarchy* above).
- `Ctrl`+click a single cell → toggle a **keyframe** at that (keypoint,
  frame): writes (or removes) an edit row at the keypoint's *current* merged
  position with `is_outlier == 0`, i.e. it freezes the frame's existing value
  as an explicit anchor without moving it. This is the same underlying action
  as nudging or re-enabling that frame — it becomes an ordinary edit row — it
  just doesn't require the user to actually change the position first. See
  *Multi-keyframe interpolation* below for why this distinction from "just an
  inlier frame" matters.
- Selection and range state are owned by `PersonPanel` (or a small shared
  controller), not duplicated per-widget, so `Space`, arrow-key nudge, and `I`
  behave identically regardless of whether the crop grid or the timeline has
  focus.

### Multi-keyframe interpolation

Generalizes Phase 10's two-anchor interpolation to N anchors, matching the
brief's workflow: disable a broad range, then re-enable/place a few frames
inside it (including the two ends) as keyframes, then interpolate. This
supersedes the anchor-scan described in Phase 10 — there is a single `I`
interpolation algorithm, described here, used by both the crop grid and the
timeline.

**The anchor predicate is the key decision, and it can't be "any inlier
frame."** A common editing pattern is: a handful of frames in an otherwise
fine range have bad detections; the user selects a slightly wider range
covering them and the surrounding good frames, and hits interpolate expecting
the *whole* range to be overwritten by a straight line between its two ends
(the original Phase 10 behavior) — none of the interior frames were
deliberately preserved, they just happen to still hold their original,
untouched (and in this case wrong) detections. If the anchor-scan treated
"any inlier frame inside the range" as an anchor, it would pick up those bad
detections as extra anchors and only interpolate around them — the opposite
of what the user wants, and indistinguishable from the multi-keyframe case
where the interior frames the user chose to keep genuinely are correct.

The two cases are only distinguishable by **whether the user explicitly
touched that frame**, not by whether it happens to be an inlier. That's
exactly what `is_edited` (the `pose_observation_edits.kp_mask` bit, already
returned by the merge) tracks. So the anchor predicate is:

> An interior frame is an anchor **iff** it has an edit row with
> `is_outlier == 0` (i.e. `is_edited and not is_outlier`) — the user
> explicitly moved it, re-enabled it, or `Ctrl`-clicked it as a keyframe (see
> *Interaction* above). An untouched original detection, even if it's
> currently an inlier, is never an interior anchor.

Algorithm:
1. Interior anchors = frames in `[range_start, range_end]` satisfying the
   predicate above.
2. Boundary anchors = nearest inlier frame outside the range on each side
   (unchanged from Phase 10) — guarantees at least two anchors even with zero
   interior ones.
3. Sort all anchors by frame number; interpolate piecewise-linearly between
   each consecutive pair; overwrite every non-anchor frame in the range.

This resolves the brief's open question of how to signal "plain overwrite"
vs. "keyframed" without a mode switch, and does so correctly for the "fix a
few bad frames in a wider selection" case above: if the user never touched
the interior of the range, there are zero interior anchors and step 3
degrades exactly to Phase 10's two-anchor overwrite. If the user disabled a
broad range and then re-enabled/placed/`Ctrl`-clicked specific interior
frames, those become anchors and everything else in between is filled by
piecewise interpolation.

### Phasing

**Phase 11 — status data plumbing.** Extend the Python read path with a
`read_timeline_status(session, sequence_id, camera_instance_id)` helper that
joins `pose_observations` + `pose_observation_edits` + `keypoint_obs_quality`
per `(keypoint, frame)` for one camera at a time (matching the single-camera
default view), plus a separate cross-camera inlier-count helper for the
bar/aggregate view. *Validation*: unit test with a synthetic sequence
covering all four axis-1 states and confirm precedence ordering (e.g. a
disabled-after-being-edited keypoint reports grey, not blue); unit test the
inlier-count helper with a keypoint visible/inlier in 3 of 4 cameras and
confirm it returns 0.75.

**Phase 12 — `KeypointTimelineWidget` skeleton.** Tree rows from
`PoseModel.groups`, flat-colored cells reading `_sel_cam_idx`, camera tab
strip, playhead synced to the existing slider. *Validation*: open a person
with a known detection run; verify row colors match `read_timeline_status`
output at a few spot-checked frames for the currently selected camera, and
that clicking a different camera tab both changes the displayed colors and
updates `_sel_cam_idx` (verify a subsequent `Space` press affects that
camera's edit rows).

**Phase 13 — selection & keyboard parity.** Rubber-band + ctrl-click wired to
shared selection/range state. *Validation*: rubber-band select 3 keypoints ×
10 frames on the timeline, then press `Space` — verify the crop grid shows
those keypoints as disabled at those frames, in the camera the timeline was
scoped to (state applied from the timeline, displayed in the grid).

**Phase 14 — multi-keyframe interpolation.** Anchor-scan extended to also
collect interior anchors satisfying `is_edited and not is_outlier`, ranked by
frame number alongside the existing boundary anchors. *Validation* (three
cases):
- *Plain overwrite*: keypoints at frames 1 and 20 are inliers with different
  positions; frames 5–15 are inliers too (untouched, "wrong" detections);
  select range 1–20, press `I`; verify frames 2–19 are all overwritten by a
  single straight line from frame 1 to frame 20 (the interior inliers at
  5–15 must **not** act as anchors).
- *Multi-keyframe*: keypoint disabled across frames 1–20 (`Space` over the
  range), then re-enabled/moved at frames 1, 10, 20 with different positions;
  press `I`; verify frames 1–10 and 10–20 interpolate as two independent
  linear segments (not one straight line from 1 to 20), and frames 1, 10, 20
  themselves are unchanged.
- *`Ctrl`-click keyframe*: same as above but frame 10 is kept at its
  original (untouched) position via `Ctrl`-click instead of being moved;
  verify it still acts as an anchor (an edit row was written at its existing
  position) even though its value didn't change.

## Partial tracking

### Motivation

Full re-tracking after every edit is slow and breaks the edit/verify loop.
`Tracker::initialize_from_state()` (`include/posetrak/tracking/tracker.hpp:148`)
already exists and is already used by `cli/track.cpp` to seed the UKF mean
from an externally supplied `State` — this is the entry point partial
tracking needs; it just isn't wired to a persisted mid-trial checkpoint yet.

### Checkpoints

New table `tracking_checkpoints`: `(run_id, step, timestamp, state_blob,
cov_blob)`. `state_blob` reuses the existing `State` vector encoding used by
`tracking_results.state`. `cov_blob` stores the **full** covariance matrix
(float64, row-major), not just the diagonal that `tracking_results.cov_diag`
stores today — resuming a filter from a diagonal-only covariance discards the
cross-correlations the UKF has built up, so the resumed run would be
overconfident and could diverge differently than a true continuation.

Written by a new `ResultWriter::write_checkpoint(step, timestamp, state,
full_covariance)`, called from the tracking loop (`cli/track.cpp`) whenever
`timestamp - last_checkpoint_time >= checkpoint_interval_s` (new
`TrackerConfig` field, default 1.0 s).

**Size, corrected against real runs**: the ~58-DOF estimate above was wrong —
`SkeletonLayout::error_state_dim()` on the actual regress-test skeleton (27
`SPHERICAL` + 32 `REVOLUTE` joints + a floating root) comes out around
120-130, matching Harri's number from real runs, not the toy estimate. At
error-state dim 125, `cov_blob` is 125×125×8 bytes ≈ 122 KB dense (not ~27
KB). At the default 1 Hz checkpoint interval that's ~7.3 MB/minute of
*tracked video time* — roughly 37 MB for a 5-minute trial, 73 MB for 10
minutes. That's per tracking run, and a session accumulates one run per
tracking attempt as the user iterates on config/edits.

On the write-cost side, 1 Hz is checkpointing against *video timestamp*, not
wall-clock: the tracker currently processes at roughly 10 video-frames/second
of wall-clock compute, so a 120 fps capture takes ~10-15 s of wall-clock work
per second of footage. One checkpoint write per second of *video* time is
therefore one write per ~10-15 s of *wall-clock* compute — cheap relative to
the surrounding work, so the interval shouldn't go much below 1 s (a much
tighter interval would multiply both the write overhead and the storage size
below for little benefit), but doesn't need to go higher either.

The storage number is the real problem, not the write cost: unbounded
accumulation across many tracking attempts needs pruning, not more careful
frequency tuning. See *Checkpoint retention* below — folded into the same
ephemeral-run garbage collection introduced under *Temporary ("ephemeral")
tracking runs*, since both are "don't keep every run's full state forever"
problems with the same shape.

### Checkpoint retention

Checkpoints are pruned using the *same* recency policy as ephemeral tracking
runs (see below), rather than a separate knob — one retention setting to
reason about instead of two that can drift out of sync:

- When an ephemeral run is garbage-collected (dropped beyond the most recent
  *M* for its `observation_sequence_id`, or older than *N* days), its
  `tracking_checkpoints` rows are deleted in the same transaction — a
  discarded test run has no future use for its checkpoints either.
- For **non-ephemeral (committed) runs**, checkpoints are useful for as long
  as the user might want to "test from here" against that run again. Once a
  run is no longer among the *M* most recent tracking runs for its
  `observation_sequence_id`, its checkpoints (but not its
  `tracking_results` — the committed output stays) are pruned the same way.
  A user who wants to test further edits against an old run is expected to
  re-run tracking first, which produces fresh checkpoints.
- An explicit **"Discard checkpoints"** action (independent of deleting the
  run itself) lets a user free the space for a run early once they're
  confident in it and no longer need to test further partial edits against
  it, without losing the committed `tracking_results`.

This keeps steady-state storage bounded by *M* runs' worth of checkpoints
(a small, fixed number) rather than growing with total tracking attempts
over a session's lifetime.

### RTS smoothing interaction

The brief asks whether checkpoints should store the smoothed state. They
should not — checkpoint the **forward-pass (filtered) state only**:

- A smoothed state at step *k* (`FrameSmootherData`, `rts_smoother.hpp`)
  depends on the *entire* forward pass, including frames after *k* that the
  user is about to edit. Checkpointing it would silently bake in the
  pre-edit future, defeating the point of testing the edit's impact.
- The filtered state at step *k* depends only on frames ≤ *k*, which is
  exactly "resume tracking from here" semantics.

This part is not in question: the checkpoint *state to resume from* must be
unsmoothed, for the reason above. What's separately open is whether the
**test run's own forward pass** should be RTS-smoothed after it completes,
rather than left as raw filtered output.

There's a real argument for smoothing it. RTS smoothing's main visible
benefit is reducing the lag the filter shows when reacting to fast motion —
and fast, hard-to-track motion is disproportionately the reason a segment
needed manual keypoint edits in the first place. A test run that's
deliberately shown unsmoothed could look worse than the edit actually is,
because the comparison is unsmoothed-test-run against a smoothed main trial,
not an apples-to-apples check of whether the edit fixed the problem.

Given that, treat this as a follow-on refinement rather than something Phase
18 needs to get right immediately: ship test runs unsmoothed first (simpler,
and still answers "does the filter recover at all"), and add an option to
also run RTS over just the test window once there's a feel for whether the
unsmoothed comparison is actually misleading users in practice. If added,
smoothing a test run only needs the test window's own forward-pass cache
(`FrameSmootherData`) — it doesn't need the main trial's smoother state,
since the whole point of a test run is that it doesn't touch the committed
trial until promoted.

Smoothing of the *committed* trial is unaffected either way: after a
promising test run is folded back into the main trial (or the full trial is
re-tracked with edits applied throughout), the existing full-trial RTS
smoothing pass still runs once at the end as it does today.

### Temporary ("ephemeral") tracking runs

- `tracking_runs.is_ephemeral INTEGER NOT NULL DEFAULT 0` (migration).
- Ephemeral runs write `tracking_results` / `tracking_obs_results` exactly
  like normal runs — no viewer or analysis code path needs to special-case
  them for reading.
- Excluded from `list_tracking_runs` (UI tree and MCP tool) by default,
  nested under their parent run when a "show test runs" toggle is on.
- Garbage-collected: on session open, drop ephemeral runs beyond the most
  recent *M* (default configurable, e.g. 5) or older than *N* days, grouped
  by `tracking_runs.observation_sequence_id` — the `pose_observation_sequences`
  row a run tracked, which is the actual grouping key `tracking_runs` carries
  (not "trial": a trial can have more than one observation sequence, e.g.
  after re-stitching, and each keeps its own independent set of tracking
  runs and checkpoints). Also exposed as an explicit "Discard test run" UI
  action.
- Inherit `tracker_configs` from the parent run by default (no separate
  config UI needed for the common "just test my edit" case).

### C++ entry point

No changes needed to existing public methods — `Tracker::get_ukf()`
(`tracker.hpp:205`) and `UnscentedKalmanFilter::set_covariance()`
(`ukf.hpp:126`) are both already public:

```cpp
Tracker tracker(skeleton, cameras, config);
tracker.initialize_from_state(checkpoint.state, checkpoint.timestamp);
tracker.get_ukf()->set_covariance(checkpoint.covariance);
for (auto const& [obs, t] : observations_from(checkpoint.timestamp)) {
    tracker.track_frame(obs, t);
}
```

New surface area is limited to `ResultWriter::write_checkpoint`, a
`load_checkpoint(run_id, before_timestamp)` reader in `session_reader.cpp`,
and a CLI flag, e.g.:

```bash
posetrak-tracker track config.toml --resume-from-run <run_id> --resume-time 12.5
```

### UI: visualization switch-over

- The tracking-run selector in `PersonPanel` shows ephemeral test runs nested
  under their parent (not as tree siblings).
- Trajectory/overlay views use a thin `CompositeRunView`: read from the
  parent run for frames before the checkpoint's timestamp, and from the test
  run from that timestamp onward. This avoids copying the pre-edit segment
  into the test run's own rows.
- Status bar shows "Test run — resumed from t=12.5s" while an ephemeral run
  is selected, so it isn't mistaken for the trial's authoritative result.
- "Test from here" is triggered from the crop grid / timeline: it resolves
  the checkpoint nearest-but-before the earliest edit in the current
  selection or active range, and launches a resumed run. On launch, the
  tracking-run selector automatically switches to the new ephemeral run —
  the whole point is a fast look at the result, so making the user manually
  find and select it afterward would defeat that.

**How long does a test run track for?** Tied to whatever selection state the
user already has, rather than a fixed duration or a separate "how far"
prompt that adds a step to the common case:

- If a frame **range** is active (the same range used for `Space`/`I`), the
  test run covers checkpoint-before-range-start through range-end **plus a
  fixed trailing buffer** (default a few seconds). The buffer matters
  because the interesting question isn't just "does the filter recover by
  the end of my edit" — it's "does it *stay* recovered for a bit after,"
  which the exact edit boundary can't show.
- If no range is active (a single-frame edit, e.g. one Ctrl-click keyframe),
  the test run covers checkpoint-before-current-frame through
  current-frame-plus-the-same-default-buffer.
- An explicit **"Extend test run"** action continues the *same* ephemeral run
  further (no new checkpoint resolution, just more `track_frame()` calls
  appended) for when the default window wasn't enough to tell — this avoids
  needing to guess the right window length upfront, at the cost of a second
  click when the default guess undershoots.

### Phasing

**Phase 15 — checkpoint writer/reader.** `tracking_checkpoints` migration,
`ResultWriter::write_checkpoint`, `session_reader` loader,
`checkpoint_interval_s` config field. *Validation*: track `tests/regress.toml`
with `checkpoint_interval_s=1.0`; verify one checkpoint row per second with
correct step/timestamp, and that the covariance blob has nonzero off-diagonal
entries (proving it's the full matrix, not the diagonal).

**Phase 16 — resume entry point.** CLI flag using `initialize_from_state` +
`set_covariance`. *Validation*: resume from a mid-trial checkpoint with
unmodified observations and verify the resumed run's state trajectory matches
the original run's from that point forward (within numerical tolerance);
resume with a deliberately edited observation and verify the trajectory
diverges only after the checkpoint.

**Phase 17 — ephemeral run bookkeeping + checkpoint retention.**
`is_ephemeral` column, retention GC, "show test runs" UI toggle, and the
shared checkpoint-pruning policy from *Checkpoint retention* above (applies
to ephemeral *and* non-ephemeral runs' checkpoints — `tracking_results`
itself is never pruned by this GC). *Validation*: create several test runs,
verify only the most recent *M* survive after session reopen, and that
non-ephemeral runs' `tracking_results` are never garbage-collected even
though their `tracking_checkpoints` are once they age out of the *M* most
recent; verify the explicit "Discard checkpoints" action removes a run's
checkpoints without touching its `tracking_results`.

**Phase 18 — UI wiring.** "Test from here" action (auto-selects the new
ephemeral run in the tracking-run selector on launch, and resolves the test
window from the active frame range + trailing buffer, or the current frame +
buffer if no range is active — see *UI: visualization switch-over* above),
"Extend test run", `CompositeRunView`, status bar indicator, "Promote"
(rename, clear `is_ephemeral`) / "Discard" actions. *Validation*: edit a
keypoint range, trigger "Test from here", verify the tracking-run selector
switches to the new run automatically and the crop grid / trajectory plot
shows original data before the checkpoint and test-run data after; verify
the test run's last tracked frame matches range-end-plus-buffer; click
"Extend test run" and verify it continues past that point without
re-resolving the checkpoint; discard the test run and verify its
`tracking_results` rows are removed.

## Keypoint / camera / frame-specific measurement error

### Motivation

`pose_noise_std` / `calib_noise_std` (`include/posetrak/core/config.hpp:86-87`)
are single scalars per tracker config today — overridable per hierarchical
child filter and per velocity-mode camera, but not per keypoint. Detection
noise is known to vary a lot by keypoint (finger tips are precise; hips are
often noisy). `Observation::noise_std_override`
(`include/posetrak/core/observation.hpp:51`) already exists as a per-
observation override — currently populated only for `PAIR_DIFF` mode — and is
exactly the hook this feature needs. No change to the UKF's measurement
update (`UnscentedKalmanFilter::update()`) is required.

### Static per-keypoint defaults

New TOML config field, for configs launched directly via the CLI:

```toml
[tracking.keypoint_noise_multiplier]
left_hip = 2.0
right_hip = 2.0
left_pinky_tip = 0.5
```

**DB mapping**: TOML is only how the CLI accepts a config; configs launched
from the UI are rows in the registry's `tracker_configs` table
(`db/registry_schema.sql:79`), which the TOML loader (`config.cpp`) doesn't
touch at all. The existing precedent for a variable-length, per-config
structure on that table is `velocity_mode_camera_ids` — a JSON array in a
`TEXT` column — rather than a child table, because it's a single
config-scoped blob with no need to join or query into it. The same shape
fits here: add `keypoint_noise_multipliers TEXT` (JSON object, keypoint name
→ multiplier, e.g. `{"left_hip": 2.0, "right_hip": 2.0}`) to
`tracker_configs`, keyed by name rather than index so it stays meaningful
across pose models with different keypoint orderings.

Both the TOML loader and whatever reads a `tracker_configs` row into a
`TrackerConfig` (`session_reader.cpp` or a dedicated config loader — same
place `pose_noise_std` etc. already get read from the DB) need to populate
`TrackerConfig::keypoint_noise_multiplier`; a config created via one path and
edited via the other must agree on the same field. Shipping a UI for editing
this column is separate, later work — `cross_pair_max_px` and several other
advanced `tracker_configs` fields already ship DB/TOML-only today without a
settings-page editor, so there's precedent for that split too.

Applied as `pose_noise_std * multiplier` (default multiplier 1.0) when each
`Observation` is constructed from a detection. Ship a default multiplier
table per pose model (co-located with `python/app/pose/kp_models.py`, e.g. as
a small JSON/TOML asset) that users can override in their tracking config —
this satisfies the brief's "add a default for each keypoint" while keeping it
a *multiplier*, so it composes with existing `pose_noise_std`/`calib_noise_std`
pixel calibration instead of duplicating it.

### Editor-driven per-camera/frame overrides

Static defaults don't cover "this specific hip detection on this specific
camera in this specific frame is untrustworthy," which is what the editor
needs.

**Separate table vs. a fourth channel on `pose_observation_edits.kp_blob`** —
weighing both instead of asserting the separate table by default:

*Extend `pose_observation_edits` (float32[N,4]: x, y, is_outlier, multiplier):*
- **For**: one table, one lookup per frame instead of two, one write path.
  Natural for the "nudge this hip's position *and* mark it less trustworthy"
  workflow in a single write.
- **Against**: `pose_observation_edits.kp_mask` currently means "this row
  overrides x/y/is_outlier for keypoint *i* as one unit." Noise overrides
  need *independent* presence-tracking from position/outlier overrides — you
  want to widen a keypoint's uncertainty without necessarily moving it or
  touching its outlier flag, and vice versa. That forces a second mask
  alongside the first, which is most of the cost of the separate-table
  option, paid *in addition to* a breaking blob-format change: every
  existing `pose_observation_edits` row (already shipped, already written by
  users across Phases 1-14) would need a migration from float32[N,3] to
  float32[N,4], and the C++ `apply_keypoint_edits` codec, `read_observations_with_edits`,
  `write_observation_edit`, and `update_single_keypoint_edit` would all need
  updating in lockstep. High blast radius for a feature most frames will
  never use — noise overrides are expected to be far rarer than position/
  outlier edits.

*Separate `pose_observation_noise_overrides` table (this design):*
- **For**: purely additive — no change to the already-shipped
  `pose_observation_edits` format or codec, no migration risk to existing
  data. Matches the actual usage pattern: sparse and optional, most frames
  never get a row. Keeps two conceptually different facts ("what value is
  this keypoint" vs. "how much do we trust it") in independently-reasoned-
  about tables instead of one row format serving two purposes.
- **Against**: two lookups per frame in `session_reader.cpp` instead of one
  (both cheap, prepared-statement lookups — same pattern already used for
  `pose_observation_edits`, so not a new kind of cost). Two Python
  read/write helper sets instead of one. A combined "move + adjust
  trust" edit is two DB writes instead of one, though both still happen
  within the same user-visible edit action.

The separate table wins mainly because it's non-breaking: the fourth-channel
option pays for a second mask *and* a format migration, while the separate
table only pays for the second mask's cost (a second lookup), without ever
touching code that Phases 1-14 already shipped and tested.

```sql
CREATE TABLE pose_observation_noise_overrides (
    id                 TEXT PRIMARY KEY,
    sequence_id        TEXT NOT NULL REFERENCES pose_observation_sequences(id),
    camera_instance_id TEXT NOT NULL,
    video_frame        INTEGER NOT NULL,
    -- float32[N]: per-keypoint multiplier applied to pose_noise_std.
    mult_blob          BLOB NOT NULL,
    -- uint8[ceil(N/8)]: bitmask of which slots this row overrides.
    kp_mask            BLOB NOT NULL,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE UNIQUE INDEX pose_observation_noise_overrides_unique
    ON pose_observation_noise_overrides (sequence_id, camera_instance_id, video_frame);
```

`session_reader.cpp` applies it the same way it applies
`pose_observation_edits` today: one more optional prepared-statement lookup
per frame, with the same pre-migration `SQLITE_ERROR`-catch compatibility
pattern, setting `Observation::noise_std_override = pose_noise_std *
multiplier * crop_scale + calib_noise_std` (reusing the existing split-noise
formula rather than replacing it) before `measurement_noise_std()` is called.

### UI: stddev circle/ellipse

- When a keypoint is selected, draw a circle of radius `noise_std_override`
  around its dot, in display-crop pixels, reusing the existing
  `display_x = (frame_x - src_x) / src_w * display_w` transform. Rendered
  only for the selected keypoint(s), matching the trail-only-for-primary
  convention from Phase 8 to avoid clutter.
- New shortcuts `[` / `]` (unused in the current binding table) shrink/grow
  the selected keypoint(s)' multiplier by a fixed step (e.g. ×0.9 / ×1.1);
  `Shift+[` / `Shift+]` for a bigger step. Applies to the current frame, or
  the whole active range if one is set — matching the existing range
  semantics for `Space`/drag.
- Interpolation (`I`, Phase 10 / multi-keyframe from the timeline section
  above) also interpolates the multiplier channel when an override exists at
  both anchors, alongside x/y.

### Phasing

**Phase 19 — static per-keypoint config.** TOML section, `tracker_configs.
keypoint_noise_multipliers` DB column (JSON, keyed by keypoint name — see
*DB mapping* above), `Observation` construction wiring reading from
whichever source produced the running config, default multiplier tables for
COCO-17 and COCO-133. *Validation*: track a config with one keypoint's
multiplier set to 10× and confirm (via `tracking_obs_results` / the MCP
`get_filter_stats` tool) that its innovation covariance scales accordingly
and it's outlier-rejected less aggressively than an unmodified keypoint with
equally noisy input; verify a config loaded from a `tracker_configs` DB row
and one loaded from the equivalent TOML produce the same per-keypoint
multipliers.

**Phase 20 — override storage + merge.** `pose_observation_noise_overrides`
migration, Python read/write helpers mirroring `db_cache.py`'s edit helpers,
C++ reader wiring with pre-migration compatibility. *Validation*: write an
override for one keypoint on one frame; verify `load_observations` produces
an `Observation` with the expected `noise_std_override` and that a DB
predating the migration still loads (edits skipped, no crash).

**Phase 21 — editor UI.** Stddev circle rendering, `[`/`]` shortcuts (single
keypoint and active range), multiplier interpolation. *Validation*: select a
keypoint, press `]` five times, verify the drawn circle grows and the stored
multiplier matches the expected compounded step; select a range, repeat, and
verify all frames in the range receive the same multiplier.

## Background wide-crop frame cache

*(Originally scoped as "full-resolution frame extraction" — superseded by the
cropped-cluster design below after benchmarking; see Motivation.)*

### Motivation

The cached crops in `frame_cache_entries` trade fidelity for scrub speed:
they're downscaled to 240 px height at JPEG quality 75
(`_CROP_TARGET_HEIGHT`/`_CROP_JPEG_QUALITY`, `python/app/pose/db_cache.py:13-14`)
and cropped to the detection bbox (even after the Phase 6 union-bbox
widening, still bounded by nearby detections). That's the right tradeoff for
instantaneous scrubbing, but it's too coarse to place a keypoint precisely
once you're zoomed in on it, and too tight when the person has moved further
than nearby frames suggest. The alternative — seeking the source video on
demand — has the opposite problem: `FrameCache.get_frame()`
(`python/app/pose/frame_cache.py:50`) does a `cap.set(CAP_PROP_POS_FRAMES,
…)` random seek per frame, which is too slow for interactive scrubbing (this
is exactly why `frame_cache_entries` exists — see *UI context* above: "It
does not seek raw video files").

The original idea here was to extract every full (or 1920px-capped) frame in
the background. Benchmarked on real 4K/120fps footage
(`cam1.mp4`, mpeg4, `D:\mocap\desktop\2026-03-10-posetrak-test\...`), that
doesn't hold up once multiplied across a multi-camera rig:

| | full 4K | 1920px-capped | generous crop (~1.2-1.6 Mpx) |
|---|---|---|---|
| decode+encode | ~19-25 fps | ~40 fps | ~37-49 fps |
| size/camera/10s | ~767 MB | ~313 MB | ~104-268 MB |

Even the downscaled full-frame variant is both slower *and* bigger than
encoding only a generous crop around the person — and the crop also gives
back the resolution that downscaling the whole frame throws away, since the
person typically occupies a small fraction of a 4K frame. So the design
below replaces "extract every frame" with "extend the existing crop cache to
be wider, higher quality, and shared across nearby people," which is a much
smaller change against the current Phase 1-6 pipeline.

That third column also isn't the whole story: an ukemi (close-contact
throwing/rolling) capture has multiple people whose generous crop windows
frequently overlap. Simulating both encoding strategies against real per-frame
bboxes from detection run `b05e51a9-5fd8-49de-98aa-958c8a84ce3a` (5 cameras ×
3 tracked persons, ~28s each) — one crop per person vs. one shared crop per
overlapping cluster of persons — showed:

| | separate per-person crops | merged clusters | savings |
|---|---|---|---|
| storage (5 cams, ~28s) | 2.42 GB | 1.46 GB | ~40% |
| encode time (5 cams) | 67.6 s | 53.2 s | ~21% |
| cached images | baseline | — | ~55% fewer |

98% of the simulated windows had at least one beneficial merge — for this
kind of capture, treating "which persons to cache together" as a first-class
decision is worth the extra bookkeeping, not just a nice-to-have.

### Design: background sequential extraction with per-cluster crops

When edit mode is entered for a person, launch a background worker — same
`QThread` + priority-queue architecture as the existing `CropBackfillWorker`
(`content_panels.py:932`) — that walks each active camera's video
**sequentially** (no seeking) over the trial's frame range, decoding every
frame once. Sequential decode avoids the per-seek GOP-reparse cost that makes
random access slow, even though it visits every frame rather than only the
ones the user has scrubbed to — for a multi-minute trial at video frame rate
this is a one-time linear pass, not repeated per scrub.

Unlike the original design, the worker does not encode one JPEG per frame —
it encodes one JPEG per **(camera, frame, person-cluster)**, where a cluster
is a group of one or more tracked persons whose generous crop windows
overlap closely enough to be worth sharing a single cached image. See
*Algorithm: deciding which crop areas to cache* below.

- **Storage**: reuse the `tempfile.mkdtemp(prefix=...)` pattern from
  `FrameCache` (`frame_cache.py:40`), not the SQLite `frame_cache_entries`
  table. Even with cropping, a multi-minute trial across several cameras is
  still large (extrapolating the simulation above, a 5-minute, 5-camera trial
  is on the order of 15 GB) — too large and too session-specific to belong in
  the durable, backed-up session DB. An in-memory index (see below) maps
  `(shot_video_id, frame_idx)` → the cluster file(s) covering that frame, so
  no SQL schema change is needed at all.
- **Resolution/quality**: no fixed low cap — encode the padded cluster crop
  at its natural pixel size (optionally capped at something generous, e.g.
  1200 px on the long edge, well above the current 240 px), JPEG quality
  ~90 (vs. the scrub cache's 75).
- **Priority**: same `prioritise(svid, frame_idx)` mechanism as
  `CropBackfillWorker`, so frames near the user's current scrub position
  extract first, with the same "fills in behind you as you work" status-bar
  convention ("Generating wide crops… N remaining").

### Algorithm: deciding which crop areas to cache

Runs per camera, over fixed-length, non-overlapping **epochs** of the
trial's frame range (default: ~0.4s of wall-clock time, converted to frames
via that camera's actual fps — e.g. 48 frames at 120fps). The epoch length is
a direct tradeoff: longer epochs recompute the crop window less often
(cheaper, and the resulting crop is generous enough to tolerate more motion
before the next recompute) but also grow each crop's average size. The
simulation above swept 24/48/96-frame epochs and found savings *improve*
with longer epochs (storage savings 37%→42%, encode-time 17%→26% from 24 to
96 frames) — so there's no sharp cost to erring toward a longer epoch;
0.4s is a reasonable starting default, not a value to tune finely up front.

**The frames that most need editing are exactly the ones detection struggled
with**: wrong-person assignment, or a person not detected at all for a
stretch that can outlast one epoch during a fast, high-motion sequence (a
throw or roll can occlude someone for well over 0.4s). The crop must stay
generous precisely there, not shrink to nothing just because an epoch
happens to contain zero real detections. So step 1 below deliberately looks
past the epoch's own boundary rather than only at detections strictly inside
it:

For each epoch, per camera:

1. Compute each track's **raw rect** for the epoch:
   a. First, union the track's real `person_detections` bboxes over
      `[epoch_start − 10, epoch_end + 10)` frames — the epoch widened by
      Phase 6's existing `±10`-frame margin, not just the frames strictly
      inside it. This absorbs short gaps straddling an epoch edge and
      epochs with only a couple of real detections near their boundary,
      the same way Phase 6 already does for a single missing frame.
   b. If that widened window still has *zero* real detections — a gap
      longer than the epoch itself — search further outward from the
      epoch's edges for the nearest real detection before it and the
      nearest one after it, up to a bounded search radius (a few seconds;
      configurable). Union whichever anchor(s) are found within that
      radius. This is intentionally generous: it produces a box spanning
      from wherever the person was last seen to wherever they reappear,
      because a large "known bounds" box is far more useful for editing a
      long undetected stretch than a tight, stale, or missing one. If the
      radius is exceeded on one side (track truly gone — e.g. leaves the
      camera's view for the rest of the trial), use only the anchor found
      on the other side; if neither side has one within the radius, this
      track has no crop for this epoch and falls through to the existing
      layered fallback (see *Serving frames* below).
2. Pad the raw rect by a configurable fraction on each side (default 35%,
   noticeably more generous than Phase 6's 10% — this is what gives room for
   motion before the next recompute, on top of whatever gap-handling already
   widened the rect in step 1). This is the track's candidate crop rect for
   the epoch.
3. Cluster tracks whose padded rects overlap: a simple union-find /
   connected-components pass over the (typically single-digit) tracks active
   in the camera — merge any two clusters whose rects intersect, taking the
   bounding union as the merged cluster's rect, and repeat until no cluster
   changes. This is cheap (O(n²) on a handful of tracks) and needs no
   external library.
4. **Merge guard**: only accept a merge if the union rect's area doesn't
   exceed roughly `1.3×` the sum of the merged rects' individual areas. Two
   rects that only graze at a corner produce a union much larger than either
   one alone — in that case keep them as separate clusters rather than
   caching one mostly-empty shared crop. (The simulation above merged
   unconditionally and still landed net-positive; this guard is a
   robustness refinement for the real implementation, not a correction to
   those numbers.)
5. The resulting clusters (each with a member track-id set and a union crop
   rect) are what gets cached for every frame in the epoch: one JPEG per
   `(shot_video_id, frame_idx, cluster)`, decoded from that frame (already in
   hand from the sequential walk) and cropped to the cluster's rect —
   *the rect is fixed for the whole epoch; only the pixels change frame to
   frame.*
6. Update the in-memory index: `(shot_video_id, frame_idx) → [ (track_ids,
   file_or_blob, src_x, src_y, src_w, src_h, {track_id: own_padded_rect}),
   ... ]` for every frame written. `own_padded_rect` is each member's
   individual rect from step 2, before the union in step 3 — kept alongside
   the shared cluster image so display can sub-crop to just one person
   instead of showing the whole cluster (see the selection algorithm below).

A track can belong to different clusters in different epochs (people
separating and rejoining) — that's expected and handled automatically since
clustering is recomputed independently per epoch. Because step 1's gap
handling already reaches outside the epoch boundary when needed, the fixed
epoch length is mostly a cost/cadence knob from here on, not a correctness
requirement — even a track invisible for several epochs in a row still gets
a generous, well-formed crop from its nearest real detections on either
side.

### Alternative considered: fixed-grid tiles

Before settling on cluster-merge, we evaluated a simpler-looking
alternative: partition each frame into a fixed tile grid and cache any tile
intersecting a person's padded window, stitching the needed tiles together
at display time. This is appealing on paper — it removes the
clustering/merge-guard/epoch bookkeeping entirely, since sharing between
overlapping people is implicit (two people whose windows touch the same
grid cell just get that cell cached once) and the caching decision per
frame is only "which tiles does this window touch."

Simulated against the same real bboxes, using the *same* padded-window
definition as the cluster approach so only the sharing mechanism differs,
across tile sizes from 400px to 2400px:

| Strategy | Storage | Encode time | Images |
|---|---|---|---|
| separate crop per person | 2.42 GB | 67.8 s | 50,288 |
| cluster-merge | **1.47 GB** | **53.2 s** | 22,892 |
| fixed tiles, best size (1400px) | 5.28 GB | 271.0 s | 32,182 |
| fixed tiles, 400-800px | 5.3-6.1 GB | 112-204 s | 74k-163k |
| fixed tiles, 1800-2400px | 7.8-10.4 GB | 425-597 s | 24k-30k |

Tiling loses at *every* size tested — 3.5-8× worse than cluster-merge on
both storage and encode time, and often worse than not sharing at all. The
reason is structural, not a tuning problem: a person's padded motion window
is an elongated, arbitrarily-positioned rectangle (someone rolling or being
thrown covers a lot of ground in a fraction of a second), and a fixed square
grid can't conform to that shape — it always rounds outward to whole tiles
at the boundary, and an elongated region wastes disproportionately more
tile area than a compact one, whichever grid size is chosen. Serving isn't
meaningfully simpler either in practice: stitching 2-4 tiles back into a
display crop is about as much code as cluster-merge's single sub-crop.
Rejected in favor of cluster-merge.

### Algorithm: selecting the cached image for a frame & scaling for display

Given the frame being displayed (`shot_video_id`, `frame_idx`) and the
track being edited in that camera cell:

This went through two wrong revisions before landing here, both caught by
actually using the feature rather than by review, so both are worth
recording:

- **Attempt 1** re-derived a tight, current-frame window from the person's
  live bbox before display, on the theory that it would keep scrubbing
  visually smooth between epoch recomputes. That re-crop used a much
  smaller margin than the cache's own (roughly Phase 6's 10-20%, not the
  35% the cluster crop was cached with), so it silently cancelled out the
  wider framing on every frame that had a real detection — which is most
  frames — making the feature look unwired even though the cache was
  populating correctly.
- **Attempt 2**, reacting to that, went to the other extreme: display the
  cached cluster crop directly, no re-crop at all. That fixed the
  cancelled-margin problem but exposed a different one — a cluster can
  legitimately span several spread-out people (that's the whole point of
  merging), so showing the *entire* cluster image for someone editing just
  one of them can mean most of the frame is empty room with the target
  person as a small figure in it.

The actual algorithm needs a rect that's tighter than the whole cluster but
not so tight it fights the cache's own margin, **and** it needs to account
for something neither attempt considered: the cache is built entirely from
raw `person_detections`, which is exactly the data this feature exists to
*correct*. An edited keypoint can legitimately sit far from where the
original (wrong) detection placed the person, and the cache has no way to
know that ahead of time.

1. Look up `(shot_video_id, frame_idx)` in the in-memory cluster index.
2. Find the entry whose `track_ids` contains the target track. (Cheap linear
   scan — there are only ever a few clusters per camera per frame.) Each
   entry stores not just the shared cluster image but also, per member
   track, that track's own padded window from *before* clustering (step 2 of
   the caching algorithm above) — narrower than the merged cluster rect,
   still with the cache's full 35% margin.
3. If found: sub-crop the decoded cluster JPEG to the target track's own
   window — not the whole cluster, and not a re-derived tight window.
4. Widen that sub-crop to cover whatever keypoints are actually about to be
   drawn for this frame:
   - the merged `pose_observations` + edits result (same confidence cutoff
     the overlay uses to decide what's visible), and
   - the tracked skeleton's projected joints/markers, if a tracking run is
     selected and available. Keypoints can in principle be edited before a
     tracking run exists, or a selected run's own projection can drift from
     the raw detection the cache was built from — either way this is a
     second, independent overlay that needs the same guarantee.
5. Grow (never shrink) the widened window to match the display cell's own
   aspect ratio, so the cell fills with image content instead of showing a
   letterboxed black border around whatever aspect ratio the padded/widened
   window happened to end up with.
6. Clamp the result to what the cluster image actually has decoded — a
   sub-crop can't show pixels outside the cached region. (The aspect-ratio
   growth in step 5 happens *before* this clamp, so it only fills in extra
   margin where the cache actually has it — it never causes a fallback on
   its own.)
7. If even the clamped, keypoint-widened window (before the aspect-ratio
   growth) still doesn't cover the displayed keypoints/joints — the edit or
   tracking result moved something outside this cluster's own cached extent
   entirely — don't use this cache entry for this frame. Fall through to the
   existing chain instead (step 9) rather than show a crop that's silently
   missing part of what should be visible.
8. If two people sharing the same cluster both have edit panels open at
   once, both read the same cached cluster JPEG and each does its own local
   sub-crop from steps 3-6 — no extra decode or storage cost. This is where
   the sharing actually pays off, and is also why the cache is scoped to the
   detection run rather than to a single panel (see *Cache scope and
   lifecycle* below) — a second person's panel should reuse, not rebuild,
   crops the first person's panel already generated.
9. If no cluster entry exists yet for this `(shot_video_id, frame_idx,
   track)` (background worker hasn't reached it, or hasn't started for this
   camera), or step 7 rejected the one that does exist: fall back to the
   existing layered chain unchanged — DB blob → Phase 6 in-memory synthetic
   crop → placeholder.

### Serving frames: layered fallback, not a replacement

`PersonCropGridWidget`'s frame source already layers DB blob → Phase 6
in-memory synthetic crop → placeholder. Insert the wide-crop cache as a
**preferred** layer above the DB blob, using the selection algorithm above:

1. Wide-crop cluster cache, if an entry exists for this frame + track — sharp,
   generously framed, and (per the merge guard) shared across nearby people
   where that helps.
2. Else the existing `frame_cache_entries` DB blob (instant, but 240p and
   tightly cropped).
3. Else the existing Phase 6 in-memory synthetic crop / on-demand path.

Because extraction runs in the background, editing stays usable immediately
on layer 2/3 and silently upgrades to layer 1 as extraction catches up — no
explicit "wait for extraction" step or mode switch.

### Cache scope and lifecycle

Because clusters are shared across persons by construction, scoping the
worker and its temp directory to a single `PersonPanel` instance (as
originally designed) would throw away the sharing benefit the moment a
second person in the same trial is opened for editing — exactly the "if
there are multiple persons in a trial it's likely all of them are edited at
one go" case this design is meant to help. Instead:

- Scope the worker + temp dir + in-memory index to the **detection run**
  (`detection_run_id`), not the panel. A small reference-counted
  `FrameCropCacheManager` keyed by `detection_run_id`: opening a
  `PersonPanel` acquires (starting the worker on first acquire), closing one
  releases (tearing down — `shutil.rmtree`, mirroring `FrameCache.close()` —
  only once the last referencing panel has closed).
- Frame range limited to the sequence's trial span (`t_start`/`t_end`,
  already used by the time slider), not the whole source video.
- No persistence across sessions: reopening the trial later re-extracts.
  Simpler than an on-disk persistent cache, and the DB-backed
  `frame_cache_entries` remains the durable, instant-load fast path.

### Tiling / digital zoom — future work, not needed for the first cut

Harri's brief also raises tiled JPEGs and digital zoom as options. The
cluster-crop cache already gets most of the way there for free: because the
cached crop is deliberately wider than what's displayed, zooming in on a
region of it is just a tighter sub-crop in step 3 of the selection algorithm
above, no new cache entries required, up to the resolution the crop was
encoded at. True fixed-size tiling (indexed by `(video, frame_idx, tile_row,
tile_col)`) remains a possible later extension if profiling shows
whole-cluster decode is a bottleneck even after cropping — worth deferring
until measured, same as before.

### Phasing

**Phase 22 — cluster-crop worker.** New `WideCropExtractWorker(QThread)`
mirroring `CropBackfillWorker`'s structure (per-camera `cv2.VideoCapture`,
sequential read loop, priority queue, `frame_ready` signal), implementing
the epoch/union/cluster/merge-guard algorithm above and writing one JPEG per
`(camera, frame, cluster)` plus the in-memory index. *Validation*: open a
person in a trial with ≥2 close-together tracked persons; verify clusters
form where bboxes overlap and stay separate where they don't (log or debug-
dump cluster membership per epoch); verify the background thread doesn't
stall UI scrubbing; find or construct a gap longer than one epoch (a track
undetected across several consecutive epochs) and verify the crop for those
epochs still spans a generous, well-formed region derived from the nearest
real detections before/after the gap, not an empty or missing crop.

**Phase 23 — layered frame serving.** Extend the crop-source fallback chain
to check the cluster cache first via the selection algorithm above:
sub-crop to the target track's own padded window, widen to cover the
frame's actual displayed keypoints and tracked-skeleton projection
(including edits, and independent of the tracking run's own drift), grow to
the display cell's aspect ratio, clamp to the cluster's decoded extent, and
fall through to the existing chain if that still doesn't cover everything
(see that section for the two wrong revisions this went through first).
*Validation*: scrub to a frame before extraction reaches it — verify the
existing 240p crop displays; scrub back after extraction has passed that
frame — verify the display seamlessly upgrades to the sharper, more
generously framed image (visibly wider margin than the old 240p crop, not
just higher resolution, but zoomed to the person being edited rather than
the whole cluster) with no visible reload glitch and no letterboxed black
border around the crop cell; open a second person's panel in the same trial
and verify it reuses the first panel's already-built cache instead of
restarting extraction from scratch; edit a keypoint to a position well
outside the original detection's bbox and verify the crop either widens to
include it or falls back cleanly, never silently clipping it off-frame;
select a tracking run whose projected skeleton reaches outside the person's
own detected bbox and verify the same widen-or-fall-back behavior applies
to it too.

**Phase 24 — priority + status bar + lifecycle.** `prioritise()` called on
scrub, status bar message while active, reference-counted worker/cache
teardown on last panel close for a detection run. *Validation*: scrub to an
unextracted frame — verify it's prioritized and appears within a bounded
time well under a raw video seek's latency; close all panels for a trial
mid-extraction — verify the temp directory is removed and the worker thread
stops (no orphaned thread or leaked temp files); close one of two panels
sharing a cache — verify the cache survives until the second also closes.

---

## Zoom and pan in the camera crop views

### Motivation

The wide-crop cache (see above) already caches more pixel detail than the
display currently uses — crops are encoded up to `MAX_LONG_EDGE` (1200px)
long edge, well beyond the ~240-400px a typical crop cell renders at. That
headroom was explicitly left for future zoom (see *Tiling / digital zoom*
above). Precise keypoint placement — small joints (fingers, ankles), or fast
motion where the auto-detected position is close but not exact — needs the
ability to zoom in past "fit to cell" and pan around, independently per
camera.

### Interaction model: designed around the touchpad, not added to it after

This needs to work identically with a mouse and a laptop touchpad. Qt
reports two-finger touchpad scroll (Windows Precision Touchpad) through the
same `QWheelEvent` a mouse wheel produces — vertical `angleDelta().y()`, and,
for a two-finger swipe in any direction, `angleDelta().x()` too — so there is
no separate touchpad code path to write. The design just needs to not lean
on gestures a touchpad can't produce: a real middle mouse button, or
pinch-to-zoom (`QNativeGestureEvent`), which Qt's Windows touchpad support is
inconsistent about across driver versions and not something to depend on for
a first cut.

That's what rules out the initial "plain scroll = zoom" idea: reserving
scroll for zoom would leave touchpad users with no way to pan at all (no
middle button, and holding a modifier while swiping is asking a touchpad to
do something a mouse doesn't need to). Flipping the convention around fixes
both devices with the same gesture set, and reuses a modifier this app's
users are already trained on:

| Input | Action |
|---|---|
| Plain wheel scroll / two-finger scroll | Pan (vertical delta → vertical pan; a two-finger diagonal swipe pans both axes) |
| `Ctrl` + wheel scroll / `Ctrl` + two-finger scroll | Zoom, centered on the cursor position |
| Middle-mouse drag | Pan (mouse-only convenience; touchpad users already have scroll-to-pan) |
| Double-click, or a small per-cell "Fit" affordance | Reset to fit (today's behavior, unchanged default) |

`Ctrl`+wheel matches the existing timeline zoom convention (Round 1 in
status.md) — same modifier, no new convention for users to learn, and it's
also the general-purpose-app convention (browsers, image viewers) for
exactly the reason above: plain scroll is claimed by panning/scrolling
everywhere else, so zoom needs a modifier regardless of device.

### View state: a full-frame rectangle, not a display-pixel offset

Zoom/pan state must survive scrubbing to a nearby frame, but the *source
image itself can change* between frames — the wide-crop cache's cluster crop
is only stable within one ~0.4s epoch (see *Background wide-crop frame
cache* above); crossing an epoch boundary can shift the underlying crop's
position and size. Storing pan as a fixed offset in *display* pixels would
silently show a different part of the scene the moment the source crop
shifts under it.

Instead, `_ImageCanvas` stores the user's desired view as a rectangle in
**full-frame pixel coordinates** — `self._zoom_rect: (x0, y0, x1, y1) | None`,
`None` meaning "fit whatever crop is given" (today's behavior, unchanged
default):

- **Zoom**: if `_zoom_rect` is `None`, initialize it from the pixmap's
  current full-frame extent (`self._x1, self._y1` plus its size divided by
  `self._src_scale` — already tracked for the existing crop-to-full-frame
  overlay math). Then shrink/grow it around the full-frame point under the
  cursor (inverse of the existing `to_pt` transform) by the wheel's zoom
  factor.
- **Pan**: translate `_zoom_rect` by the wheel/drag delta converted from
  display pixels to full-frame pixels (divide by the current combined
  scale).
- **At paint time**: clamp `_zoom_rect` to the *current* pixmap's own
  full-frame extent (same "can't show pixels that were never decoded" clamp
  already used for the wide-crop cache's sub-crop, in `_load_frame` /
  `_display_crop_result`) before computing the display transform. If the
  clamped rectangle has shrunk to nothing — the new frame's crop doesn't
  overlap the desired view at all, e.g. after a large epoch-boundary shift —
  fall back to fit mode for that frame rather than show a blank cell, but
  don't clear the stored `_zoom_rect` itself, so the zoomed view resumes as
  soon as an overlapping frame is reached again.

This is the same clamp-and-fall-back shape already built for the wide-crop
cache's own sub-cropping — the two features compose rather than fighting
each other.

### Consolidating the coordinate transform

`_ImageCanvas.paintEvent`, its mouse handlers (click hit-testing, drag,
rubber-band, context menu), and `_image_rect()` each independently derive a
`combined = self._src_scale * disp_scale` today (four call sites). Zoom/pan
adds a second scale factor and an offset every one of them needs too, so
this needs consolidating into one method — e.g. `_view_transform() ->
(off_x, off_y, combined_scale)` — used by every consumer, rather than adding
a fifth ad hoc copy of the same math. This is mechanical but not optional:
any interaction that doesn't route through it will silently misregister the
moment a cell is zoomed.

### Phasing

**Phase 25 — canonical view transform + zoom.** Consolidate the four
existing `combined = src_scale * disp_scale` call sites into one
`_view_transform()`; add `_zoom_rect`, `Ctrl`+wheel zoom centered on the
cursor, clamped to the pixmap's own extent. *Validation*: zoom in on a
keypoint, verify click/drag/rubber-band hit-testing still lines up exactly
with what's drawn at the new scale; verify plain (non-`Ctrl`) scroll and all
existing shortcuts are unaffected on an unzoomed cell.

**Phase 26 — pan.** Plain wheel/two-finger-scroll and middle-drag translate
`_zoom_rect`; clamp-and-fall-back-to-fit at frame load when the new frame's
crop doesn't cover the stored view. *Validation*: zoom in, scrub across an
epoch boundary, verify the view either follows smoothly or falls back to fit
without an error or blank cell — never shows stale/wrong pixels; verify a
two-finger diagonal touchpad swipe pans both axes at once.

**Phase 27 — reset/fit + persistence check.** Double-click (or a small
per-cell button) resets `_zoom_rect` to `None`. *Validation*: zoom one
camera cell, scrub several frames within the same epoch, verify the view
stays anchored to the same real-world region (not just visually similar) by
parking on a specific static object in the background; reset and verify it
returns to today's fit-to-cell behavior exactly.

---

## Keypoint-placement toolbar

### Motivation

The existing move-a-keypoint interaction (drag from an existing dot) has a
hard prerequisite: a dot to grab. Two of the cases keypoint editing exists
for don't have one:

- The person wasn't detected at all in this frame (no observation, no dot)
  — Phase 7's ghost-frame placement already covers this, but only through
  the *currently selected* keypoint and only by clicking empty space, so
  switching which keypoint you're placing means going back to a *different*,
  already-drawn dot elsewhere to reselect first.
- Detection exists but is bad enough that the wrong keypoint's dot is what
  you'd have to click on, or dots are stacked closely enough that hitting
  the right one is fiddly.

Both are solved by picking the target keypoint from a list instead of from
the canvas.

### Design

A new persistent panel — the same `pose_model.tree_groups` hierarchy the
timeline already uses for its row tree (`build_rows` in
`keypoint_timeline_widget.py`), reused here as a plain clickable tree rather
than the timeline's per-frame status columns.

1. Clicking a **leaf** row (single keypoint) sets
   `self._pending_place_kp_idx` and changes every camera canvas's cursor to
   a crosshair (`Qt.CursorShape.CrossCursor`). Clicking a **group** row can
   still drive the existing multi-select context-menu behavior (unchanged)
   but does not enter placement mode — placement is one keypoint at a time,
   since a single click only has one location to give it.
2. The **next click on any camera canvas**, in any state — empty space, on
   top of an existing dot, ghost frame or not — places
   `self._pending_place_kp_idx` at the clicked location via the existing
   `_on_kp_moved(cam_idx, kp_idx, new_x, new_y)` write path
   (`update_single_keypoint_edit` → re-read merged observations → refresh
   timeline → reload frame). This is a strict superset of what
   `_on_empty_area_clicked` already does for ghost frames — same write path,
   just no longer gated on "no observation for this frame at all" or
   restricted to whichever keypoint happens to be the current primary
   selection.
3. Placement mode **stays active** after a placement, so the same keypoint
   can be placed across several consecutive frames/cameras in one sweep
   (matches the existing range-editing spirit of `Shift+A/D`) — it doesn't
   revert to a passive/select mode after one click, since scanning through a
   bad-detection stretch and re-placing the same joint frame by frame is the
   expected use, not a one-off correction.
4. `Esc` clears `self._pending_place_kp_idx` and reverts the cursor.
   Existing `Esc` semantics (deselect keypoint / exit edit mode, see
   *Interaction model* above) still apply once nothing is pending —
   placement-cancel takes priority over them, not instead of them.
5. Picking a *different* keypoint from the list while one is already
   pending simply retargets it — no need to `Esc` first.

### Phasing

**Phase 28 — toolbar panel.** New dock/panel widget listing
`pose_model.tree_groups` (group headers + expandable leaf rows — worth
sharing `build_rows` from `keypoint_timeline_widget.py` rather than
reimplementing the same tree derivation). Clicking a leaf sets
`_pending_place_kp_idx` and the crosshair cursor on every canvas.
*Validation*: open the panel for a skeleton with several groups, verify
every keypoint is reachable and clicking one visibly changes the cursor over
the camera cells.

**Phase 29 — canvas placement override.** Canvas mouse-press, when
`_pending_place_kp_idx` is set, short-circuits the existing
hit-test/select/drag/ghost-frame branches and calls `_on_kp_moved` directly
at the click location; `Esc` clears the pending state ahead of its existing
deselect/exit-edit-mode handling. *Validation*: with a keypoint picked from
the list, click a frame that has other correctly-detected keypoints and
verify only the picked one moves; click across several consecutive frames
and verify placement mode persists between clicks; press `Esc` mid-placement
and verify the cursor reverts and a normal click no longer places anything.

---

## Open questions

1. **Freeze / version management** — the brief envisions marking the current
   edit state as a named frozen version.  A `pose_observation_edit_snapshots`
   table linking edit rows to a named snapshot is the natural extension, and
   would naturally also cover "promoting" an ephemeral test run (see
   *Partial tracking* above) to a named, permanent one.

2. ~~**Incremental re-tracking**~~ — addressed by *Partial tracking* above
   (Phases 15–18): checkpoint the forward-pass UKF state + full covariance
   periodically, and resume via the already-existing
   `Tracker::initialize_from_state()`.
