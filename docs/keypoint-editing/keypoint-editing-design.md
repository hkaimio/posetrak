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

## Open questions

1. **Freeze / version management** — the brief envisions marking the current
   edit state as a named frozen version.  A `pose_observation_edit_snapshots`
   table linking edit rows to a named snapshot is the natural extension.

2. **Incremental re-tracking** — re-solving only the affected time window by
   warm-starting the UKF from a state checkpoint just before the first edit.
   This requires caching UKF state, which is a significant C++ change.
