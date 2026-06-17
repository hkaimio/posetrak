# Keypoint editing — technical design

## Goals

Allow a user to manually correct keypoint detections inside the detection UI
(`PoseExtractionWindow`) without running a new detection pass.  The two
fundamental operations are:

1. **Mark as outlier / inlier** — override the automatic outlier flag on one
   or more frames of a keypoint.
2. **Move to new position** — drag or nudge a keypoint to a corrected pixel
   location in one or more frames.

Non-goals for the initial implementation:
* Editing the tracker's smoothed output (only raw detections).
* Real-time propagation of edits to the UKF (edits are written to the DB;
  the tracker must be re-run to pick them up).
* Multi-skeleton editing or re-assignment of keypoints between tracks.

---

## Data model

### Design choice: edit overlay table

The preferred approach from the brief is option **c** — edits form a versioned
overlay on top of the immutable detection run, rather than overwriting it.

#### New table: `keypoint_edits`

```sql
CREATE TABLE keypoint_edits (
    id              TEXT PRIMARY KEY,
    detection_run_id TEXT NOT NULL
                        REFERENCES detection_runs(id),
    shot_video_id   TEXT NOT NULL,
    track_id        INTEGER NOT NULL,
    video_frame     INTEGER NOT NULL,
    kp_index        INTEGER NOT NULL,   -- COCO keypoint index 0-16
    -- Nullable fields: NULL means "keep original value from detection"
    x               REAL,              -- pixel x (native video resolution)
    y               REAL,              -- pixel y (native video resolution)
    is_outlier      INTEGER,           -- 0 = forced inlier, 1 = forced outlier
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE UNIQUE INDEX keypoint_edits_unique
    ON keypoint_edits (detection_run_id, shot_video_id, track_id, video_frame, kp_index);
```

**Rationale:**
- Original `detection_keypoints` rows are never mutated, so any detection run
  remains fully reproducible.
- `UPSERT` (INSERT OR REPLACE) keeps edits instant and auto-versioned by
  `created_at`.
- Reading merged keypoints is a simple LEFT JOIN: detection row overridden by
  the edit row where one exists.
- A future "freeze" mechanism can snapshot the edit table to a named version
  without changing the schema.

#### Reading merged keypoints

A helper function `read_keypoints_with_edits(session, detection_run_id,
shot_video_id, track_id)` replaces `read_keypoints_for_run` for the editing
view:

```sql
SELECT
    dk.video_frame,
    COALESCE(ke.x,          dk_x)   AS x,
    COALESCE(ke.y,          dk_y)   AS y,
    COALESCE(ke.is_outlier, 0)      AS is_outlier,
    ke.id IS NOT NULL               AS is_edited
FROM detection_keypoints dk
    LEFT JOIN keypoint_edits ke
        ON  ke.detection_run_id = dk.detection_run_id
        AND ke.shot_video_id    = dk.shot_video_id
        AND ke.track_id         = dk.track_id
        AND ke.video_frame      = dk.video_frame
        AND ke.kp_index         = ?
WHERE dk.detection_run_id = ?
  AND dk.shot_video_id    = ?
  AND dk.track_id         = ?
ORDER BY dk.video_frame
```

(The keypoint blob must be unpacked per-index in Python as today — only the
relevant column changes.)

#### Handling frames with no detection

When the user wants to place a keypoint in a frame where the detector found
nothing (no row in `detection_keypoints`), the edit is written as a synthetic
row:

```sql
INSERT OR REPLACE INTO keypoint_edits
    (id, detection_run_id, shot_video_id, track_id, video_frame, kp_index, x, y, is_outlier)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0);
```

`x` and `y` are populated from the interpolated ghost position the user
clicked on (see *Trail interpolation* below).  These rows have no
corresponding `detection_keypoints` parent row, so the query above uses a
UNION variant in the reader:

```sql
-- ghost edits: edits with no matching detection row
SELECT ke.video_frame, ke.x, ke.y, ke.is_outlier, 1 AS is_edited, 1 AS is_synthetic
FROM keypoint_edits ke
WHERE ke.detection_run_id = ?
  AND ke.shot_video_id    = ?
  AND ke.track_id         = ?
  AND ke.kp_index         = ?
  AND NOT EXISTS (
      SELECT 1 FROM detection_keypoints dk
      WHERE dk.detection_run_id = ke.detection_run_id
        AND dk.shot_video_id    = ke.shot_video_id
        AND dk.track_id         = ke.track_id
        AND dk.video_frame      = ke.video_frame
  )
```

---

## UI components

### Mode toggle

A new **Edit keypoints** toggle button (or toolbar action) in
`PoseExtractionWindow` switches the frame view between *browse mode* (existing
behaviour) and *edit mode*.  Edit mode is only available when a detection run
and a specific `track_id` are selected.

### `KeypointEditOverlay` (new class, `frame_view.py`)

Replaces / extends `SkeletonDetectionOverlay` when edit mode is active.
Responsibilities:

| Concern | Details |
|---|---|
| Full skeleton | Draw all keypoints at the current frame, same colours as today. |
| Selected keypoint | Larger dot, white ring, no confidence colour. |
| Trail (past) | Red dots + polyline for the N preceding frames. |
| Trail (future) | Blue dots + polyline for the N following frames. |
| Ghost positions | Semi-transparent grey dots at interpolated positions on frames with no detection. |
| Edited marks | Small yellow square overlaid on any keypoint whose position/outlier state has been overridden. |
| Drag handle | Mouse hover over a keypoint shows a grab cursor; press-drag moves it. |

The overlay receives a `KeypointTrailData` structure (see below) instead of
the raw per-frame dict.

#### Trail data structure

```python
@dataclass
class KeypointTrailEntry:
    frame: int
    x: float
    y: float
    is_outlier: bool
    is_edited: bool
    is_synthetic: bool   # interpolated ghost; not in DB

@dataclass
class KeypointTrailData:
    kp_index: int
    trail: list[KeypointTrailEntry]   # sorted by frame
    current_frame: int
    trail_radius: int = 10            # frames each direction
```

The overlay draws `trail_radius` entries on each side of `current_frame`,
extending further if the nearest real detection is more than `trail_radius`
frames away.

#### Trail interpolation

Gaps between real detections are filled by linear interpolation in pixel
space.  Ghost entries are marked `is_synthetic=True` and are *not* written to
the DB.  When the user moves or marks a ghost entry as inlier, the resolved
pixel coordinates are written as a new `keypoint_edits` row.

### Keypoint selection

A keypoint is identified by its COCO index (0–16).  Clicking a keypoint dot
in any camera view selects that keypoint index globally — all camera views
update their trails simultaneously.

Selection state lives in `PoseExtractionWindow` and is propagated to each
`FrameViewWidget`.  An "active keypoint" panel (could be a label row) shows
the keypoint name (e.g. "Right wrist"), current pixel position, and outlier
state.

### Multi-camera synchronisation

Edit mode requires all camera views to be visible simultaneously.  The current
single `FrameViewWidget` layout is not ideal here — a follow-up task should
introduce a tiled multi-camera view.  For the initial implementation, the
existing single-camera view with the camera dropdown can be used; the trail is
always computed from all cameras but only the active camera's trail is
interactive.  Edits made in one camera view are immediately reflected when
switching to another camera.

---

## Frame view interaction

### Mouse

| Event | Action |
|---|---|
| Click on keypoint dot | Select that keypoint index |
| Click on empty area | Deselect |
| Drag from keypoint dot | Move keypoint; write edit on mouse-release |
| Click on ghost dot | Select; if user drags, write a synthetic edit |

Coordinate conversion uses `VideoCanvas.canvas_to_image()` already present in
the codebase.  The resulting image-space coordinates are stored in native
video resolution (same convention as `detection_keypoints.keypoints` blob).

### Keyboard

All keyboard events are captured by the active `FrameViewWidget` (or by the
main window and dispatched).

| Key | Action |
|---|---|
| `a` | Previous frame |
| `d` | Next frame |
| `Shift+A` | Extend frame selection left (multi-frame edit range) |
| `Shift+D` | Extend frame selection right |
| `←` `→` `↑` `↓` | Nudge selected keypoint ±1 px |
| `Shift+←/→/↑/↓` | Nudge ±10 px |
| `Space` | Toggle outlier/inlier for the selected keypoint at the current frame |
| `Esc` | Deselect keypoint / exit edit mode |

For **frame range operations** (`Shift+A/D`): when a range `[first, last]` is
active, `Space` and drag operations apply to all frames in the range.  A drag
moves every keypoint in the range by the same delta (not to the same absolute
position).

---

## Bounding box caching for undetected frames

Currently `frame_cache_entries` only stores crops for frames where a person
was detected.  The brief correctly notes that this prevents inspecting frames
where detection failed.

### Solution

Extend the detection pipeline post-step to write synthetic bbox crops for
frames within the run's time range that have no detection.  The bounding box
is chosen as the union of all real detections within ±N frames (configurable,
default N=10), padded by 10%.  If no detection exists within ±N frames, use
the last known bbox.

This crop is written to `frame_cache_entries` with `cache_type='full_body'`
and `track_id` of the nearest detected track.  Because these rows already
exist in the schema they require no migration.

A one-shot backfill function `backfill_undetected_crops(session,
detection_run_id, shot_video_id, n_context=10)` runs after the pipeline
completes, so existing runs can also be patched.

---

## Schema migration

Requires a new session DB migration (schema version bump to 21 or next
available):

```sql
CREATE TABLE keypoint_edits (
    id               TEXT PRIMARY KEY,
    detection_run_id TEXT NOT NULL,
    shot_video_id    TEXT NOT NULL,
    track_id         INTEGER NOT NULL,
    video_frame      INTEGER NOT NULL,
    kp_index         INTEGER NOT NULL,
    x                REAL,
    y                REAL,
    is_outlier       INTEGER,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE UNIQUE INDEX keypoint_edits_unique
    ON keypoint_edits (detection_run_id, shot_video_id, track_id, video_frame, kp_index);

PRAGMA user_version = 21;
```

---

## Implementation phases

### Phase 1 — data layer
- Add `keypoint_edits` table via a DB migration.
- Implement `read_keypoints_with_edits()` in `db_cache.py`.
- Implement `write_keypoint_edit()` in `db_cache.py` (single-frame upsert).
- Unit tests for the read path (merged output matches expectations for edited,
  unedited, and ghost frames).

### Phase 2 — overlay and trail
- Implement `KeypointTrailData` dataclass and trail-building logic (with
  linear interpolation for gaps).
- Implement `KeypointEditOverlay` with trail rendering in
  `frame_view.py`.
- Wire overlay into `FrameViewWidget` behind the edit-mode flag.

### Phase 3 — mouse interaction
- Implement keypoint click-to-select (hit-test against rendered dot
  positions).
- Implement drag-to-move with coordinate conversion and DB write on release.
- Add ghost-dot interaction (click/drag creates a synthetic edit row).

### Phase 4 — keyboard shortcuts
- Capture key events in `FrameViewWidget` (override `keyPressEvent`).
- Implement `a`/`d` frame navigation, cursor nudge, `Space` toggle.
- Implement `Shift+A/D` frame range extension.

### Phase 5 — multi-frame operations
- Apply Space toggle and drag delta to entire selected frame range.
- UI indicator showing active frame range.

### Phase 6 — bounding box backfill
- Implement `backfill_undetected_crops()` in `db_cache.py`.
- Call automatically at the end of the detection pipeline.
- Expose as a menu action for existing runs.

---

## Open questions

1. **Freeze / version management** — the brief envisions being able to mark
   the current edit state as a named frozen version.  A `keypoint_edit_snapshots`
   table (linking `keypoint_edits` rows to a named snapshot) could support
   this.  Deferred to after Phase 1.

2. **Multi-camera tiled view** — edit mode is most useful when all cameras are
   visible at once.  A grid layout (2×2 or N×1) should be designed as a
   separate feature but kept in mind when wiring trail synchronisation.

3. **Propagation to the tracker** — after editing, the user must re-run the
   tracker to see the effect.  A future improvement would be to let the
   tracker read `keypoint_edits` directly so that only the affected time
   window is re-solved.

4. **COCO keypoint indexing vs. skeleton marker names** — the current skeleton
   YAML uses named markers (e.g. `right_wrist`) while COCO uses integer
   indices.  A mapping table (already partially implied by `skeleton_layout.py`)
   will be needed to label keypoints in the editing UI.
