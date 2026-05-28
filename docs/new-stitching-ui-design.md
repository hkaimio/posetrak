# New Stitching Widget — Design Proposal

*Based on `docs/new-stitching-ui-concept.md`. Formalises the concept into
testable components, clarifies ambiguities, and identifies required DB and
framework changes.*

---

## 1. Goals and scope

The current `StitcherWidget` + `FrameViewWidget` + `PersonPreviewWidget` combo
is cumbersome for complex scenes where the tracker loses a person frequently:
the user must repeatedly click single-frame positions to split tracks, with no
visual feedback about what is inside each bar.  The new design replaces this
with a filmstrip-based timeline where each detection bar is visually filled with
sampled JPEG crops, making track content immediately recognisable.

**In-scope for the first prototype:**
- Filmstrip bar rendering (JPEG crops from `frame_cache_entries`)
- Hover tooltip showing a larger crop under the cursor
- Person colour strip + name at bar bottom
- Configurable bar height and label visibility
- Click to select a full bar; drag to select a sub-range within a bar
- Assignment, person creation, and detach — operating on the selection
- Implicit auto-split: if only part of a bar is selected, assignment/detach
  automatically splits the bar at the selection boundaries
- Implicit auto-merge: adjacent segments with the same assignment are merged
  into one after every write
- Overlap conflict highlighting: red overlay when one person is assigned to
  overlapping time ranges in the same camera
- Two view modes: **By Detection** (current layout) and **By Person**
  (tracks grouped per person, compacted into minimum rows)
- Status bar showing global timestamp and per-camera frame number at the
  current cursor/playhead position

**Explicitly out-of-scope for first prototype** (noted in concept as future):
- Full video frame display read from DB cache
- Manual bounding-box drawing and background re-detection
- Pose keyframe editing

---

## 2. Changes to `StitcherPanel` layout

### 2.1 Removed widgets

| Widget | Reason |
|--------|--------|
| `FrameViewWidget` (left half of splitter) | Replaced by filmstrip bars + hover tooltip |
| `PersonPreviewWidget` (right of stitcher row) | Made redundant by filmstrip content |

### 2.2 New layout

```
┌────────────────────────────────────────────────────────────────┐
│  Run selector / header                          [By Detection ▼]│
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FilmstripStitcherWidget  (QGraphicsView, fills remaining space) │
│                                                                  │
├────────────────────────────────────────────────────────────────┤
│  t = 00:39.120   gopro_01: frame 2847   gopro_02: frame 2851   │  ← status bar
├────────────────────────────────────────────────────────────────┤
│  Person: [Combo ▼]  [+Add]    [Assign]       [Apply]           │  ← unchanged
└────────────────────────────────────────────────────────────────┘
```

The view-mode combo ("By Detection" / "By Person") sits in the header row.
Switching modes replaces the scene content but preserves all assignment state
held in `StitcherPanel`.

---

## 3. Data model — unchanged

The assignment data model in `StitcherPanel` stays the same:

```
_assignments: dict[(svid, tid, seg_first), person_name]
_segments:    dict[(svid, tid), list[(seg_first, seg_last)]]
```

All changes are in the *presentation* layer (`FilmstripStitcherWidget`) and
the *interaction* layer (drag selection, auto-split/merge).

---

## 4. `FilmstripStitcherWidget`

### 4.1 Visual anatomy of a bar

```
┌─────────────────────────────────────────────────────────────┐  ← height = ROW_H (default 48 px)
│  [crop t0] [crop t1] [crop t2] [crop t3] [crop t4  ···      │  ← filmstrip
│ ─────────────────────────────────────────────────────────── │
│ ████████████████  Roosa                                      │  ← person colour strip + name (3 px)
└─────────────────────────────────────────────────────────────┘

 ↑ when selected: thick black border around entire bar (or around selected sub-range)
 ↑ when overlap conflict: semi-transparent red overlay on conflicting frames
```

The leftmost thumbnail is for the bar's first frame; thumbnails tile leftward
with a step equal to `bar_width_px / n_thumbnails` where `n_thumbnails` is
chosen so thumbnails are roughly square given `ROW_H`.  The rightmost thumbnail
is cropped at the bar's right edge.  Only thumbnails for frames currently
visible in the viewport are loaded.

### 4.2 Constants and configuration

```python
ROW_H_DEFAULT = 48          # px — bar height
ROW_H_MIN, ROW_H_MAX = 24, 120
LABEL_HEIGHT = 3            # px — person colour strip at bar bottom
LABEL_WIDTH  = 80           # px — camera label column (unchanged)
ROW_GAP      = 4            # px — gap between rows (unchanged)
```

Both `ROW_H` and `show_label` (the bottom strip) are user-configurable at
runtime via a settings row above the widget (or a right-click menu option).

### 4.3 Filmstrip loading strategy

Thumbnails are loaded **synchronously on demand**, restricted to frames visible
in the current viewport.  Each bar maintains a `_loaded_range: tuple[int, int]`
of already-loaded frame indices.  On scroll or resize, bars call
`_refill_thumbnails()` to fetch missing frames from `frame_cache_entries`.

For the first prototype, load at most `MAX_THUMBS_PER_BAR = 200` thumbnails per
bar to cap memory use.  If the bar spans more frames than this, thumbnails are
sampled uniformly (every *k*-th frame).

Query pattern (single bar, one call per refill):
```sql
SELECT frame_idx, image_data, height_px
FROM frame_cache_entries
WHERE shot_video_id = ?
  AND cache_type    = 'person_crop'
  AND track_id      = ?
  AND detection_run_id = ?
  AND frame_idx BETWEEN ? AND ?
ORDER BY frame_idx
```

Loaded `QPixmap` objects are stored in a `dict[int, QPixmap]` on the bar item.
They are discarded when the bar scrolls out of the viewport.

### 4.4 Hover tooltip

`QGraphicsView.mouseMoveEvent` maps the cursor x to a global timestamp →
per-camera frame index.  For the bar under the cursor, query
`frame_cache_entries` for the nearest cached frame (±3 frames, same query as
`_load_frame` in `PersonCropGridWidget`).  Show the JPEG as a `QToolTip`-style
floating widget (a `QLabel` inside a frameless `QWidget` parented to the
viewport) positioned just above or below the bar.  Update it as the cursor
moves along the bar.  The tooltip widget is `hide()`d when the cursor leaves
the bar.

### 4.5 Signals

```python
class FilmstripStitcherWidget(QGraphicsView):
    # User selected a segment or a sub-range within a segment.
    # frame_sel_first / frame_sel_last are the selected sub-range
    # (equal to seg_first/seg_last when the full bar is selected).
    segment_selected = Signal(str, int, int, int, int, int)
    #                         svid tid seg_f seg_l sel_f sel_l

    # User requests an assignment change (assignment logic lives in StitcherPanel)
    assignment_changed = Signal(str, int, int, object)  # svid tid seg_first person|None

    # Time cursor moved (hover or click)
    time_hovered   = Signal(float)   # global_s (hover, for status bar)
    time_clicked   = Signal(float)   # global_s (click, to update playhead)
```

`split_requested` is **removed** — splits are triggered implicitly when
`StitcherPanel` processes a partial-range assignment (see §5.2).

### 4.6 Mouse interaction

| Event | Action |
|-------|--------|
| Left press on bar (no drag) | Select full bar → emit `segment_selected` with full range |
| Left press + drag within bar | Rubber-band selection within one bar; release emits `segment_selected` with the sub-range |
| Drag starting outside a bar | No selection change |
| Right-click on bar or selection | Context menu (see §5.3) |
| Mouse move over bar | Update hover tooltip; emit `time_hovered` |

Drag is detected once the cursor moves more than 4 px after press.  While
dragging, a semi-transparent blue overlay rect is drawn over the bar indicating
the selected region.

---

## 5. `StitcherPanel` changes

### 5.1 Selection state

Replace the current `_current_svid / _current_track_id / _current_seg_first`
triple with:

```python
@dataclass
class _Selection:
    svid: str
    tid: int
    seg_first: int   # segment identifier (unchanged)
    sel_first: int   # selected frame range within the segment
    sel_last: int
```

`_Selection.sel_first == seg_first and _Selection.sel_last == seg_last`
means the full bar is selected (no split required on assign).

### 5.2 Auto-split and auto-merge

`_do_assign(svid, tid, seg_first, sel_first, sel_last, name)` replaces the
current `_do_assign`:

```
1. If sel_first > seg_first:
       split segment at sel_first  →  left half [seg_first, sel_first-1]
2. If sel_last < seg_last:
       split segment at sel_last+1  →  right half [sel_last+1, seg_last]
3. Assign the middle segment [sel_first, sel_last] to name
4. Run _auto_merge(svid, tid):
       scan adjacent segments; if two neighbours have the same person_name
       (or both None), merge them into one by removing the split boundary
```

`_auto_merge` is also called after every detach operation.

**Merge rule**: two segments `(sf1, sl1)` and `(sf2, sl2)` where
`sf2 == sl1 + 1` are merged if
`assignments.get((svid, tid, sf1)) == assignments.get((svid, tid, sf2))`.
The merged segment takes `seg_first = sf1`.

### 5.3 Context menu (right-click)

Same as current, **except**:
- "Split here" action is removed (splitting is now implicit on partial assignment)
- When a sub-range is selected, menu header shows "Selection: frames F–L"
  instead of track info

### 5.4 Overlap conflict detection

After every assignment write, `StitcherPanel._check_overlaps(svid)` is called
for the affected camera.  It scans all segments in the camera and builds a
`dict[person_name, list[(t0, t1)]]` of time intervals.  Intervals of the same
person that overlap produce a set of conflicting `seg_keys`.  These are passed
to `FilmstripStitcherWidget.set_conflict_segments(keys)` which draws a
semi-transparent red overlay on the offending bars.

The `set_conflict_segments` call on the widget replaces the current
`find_assignment_conflicts` + dialog flow: **conflicts are no longer blocking**.
The user sees them highlighted and must resolve before `apply()` succeeds
(Apply button shows a count badge of unresolved conflicts; clicking it when
conflicts exist shows a summary dialog).

---

## 6. By-Person view

`ByPersonScene` is an alternative `QGraphicsScene` populated by
`FilmstripStitcherWidget` when the view mode is "By Person".

Layout algorithm:
1. For each known person, iterate cameras in alphabetical order.
2. For each camera, collect all segments assigned to this person (there may be
   several, from splits and multi-track assignments).
3. Pack segments onto rows using a greedy interval-scheduling algorithm
   (each new segment goes on the first row where the last segment's end time
   < new segment's start time). This minimises the number of rows.
4. If the camera contributes zero assigned segments, it is omitted.
5. Overlap conflicts (same person, overlapping times) always go on separate rows
   (conflict highlighting still applies).

Row labelling: left column shows `person / camera` instead of just the camera
label.

The camera-label column width may need to grow to accommodate `person / camera`
text — a two-line label (person name bold, camera name normal) at `LABEL_WIDTH =
120 px` is recommended.

---

## 7. Status bar

A `QLabel` below the scene (or inside the `FilmstripStitcherWidget` footer)
shows:

```
t = 00:39.120  |  gopro_01: frame 2847  |  gopro_02: frame 2851  |  pixel9: frame 3002
```

Updated on `time_hovered`.  Frame numbers are computed via `SyncTable.lookup(t, svid)`.

---

## 8. Database schema changes

### 8.1 No changes required for Phase 1

All data needed by the filmstrip is already present:

| Data | Table | Key columns |
|------|-------|-------------|
| JPEG crops for filmstrip | `frame_cache_entries` | `cache_type='person_crop'`, `track_id`, `frame_idx`, `image_data` |
| Crop-to-full-frame transform | `frame_cache_entries` | `src_x`, `src_y`, `src_w`, `src_h` |
| Track spans | `person_tracks` | `first_frame`, `last_frame` |
| Person assignments with ranges | `detection_track_assignments` | `person_name`, `first_frame`, `last_frame` |

The existing `person_crop` JPEGs are stored at up to 240 px height
(`_CROP_TARGET_HEIGHT`) which is sufficient to render at 24–120 px bar height.

### 8.2 Optional optimisation (not Phase 1): `timeline_thumb` cache type

For long recordings with many cameras the filmstrip may load hundreds of
240 px JPEGs per render.  A dedicated `timeline_thumb` cache type at
`_THUMB_TARGET_HEIGHT = 48` (or equal to `ROW_H`) would reduce blob sizes by
~5×.  This requires:

1. A new `cache_type` value `'timeline_thumb'` — **no schema change**, just a
   new value in the existing `frame_cache_entries` table.
2. Writing thumbnails in `DetectionBatchWriter.add_frame()` alongside the
   existing `person_crop` entries.
3. Updating `FilmstripStitcherWidget` to prefer `timeline_thumb` when available,
   falling back to `person_crop`.

This is a pure data-layer addition with no migration required.

### 8.3 Future schema additions (manual detections)

If manual bounding-box detections are added later, a new table is needed:

```sql
CREATE TABLE manual_detections (
    id                  TEXT PRIMARY KEY,
    detection_run_id    TEXT NOT NULL REFERENCES detection_runs(id),
    shot_video_id       TEXT NOT NULL,
    frame_first         INTEGER NOT NULL,
    frame_last          INTEGER NOT NULL,
    -- Keyframe bbox interpolation data (JSON array of {frame, cx, cy, w, h})
    keyframes_json      TEXT NOT NULL,
    created_at          TEXT NOT NULL
);
```

Not part of this design.

---

## 9. Framework / infrastructure changes

### 9.1 `FrameViewWidget` and `PersonPreviewWidget`

These classes are **not deleted** — they remain in use in `PoseExtractionWindow`
(the standalone pose extraction app in `python/app/pose/main.py`).  They are
simply **no longer instantiated** inside `StitcherPanel._build_ui()`.

### 9.2 `StitcherWidget` (existing)

Kept unchanged as a fallback; `StitcherPanel` gets a toggle to switch between
`StitcherWidget` (current, solid-colour bars) and `FilmstripStitcherWidget`
(new).  Once the new widget is validated the old one can be removed.

### 9.3 New module layout

```
python/app/pose/
    stitcher.py                  # existing — solid-colour StitcherWidget (kept)
    filmstrip_stitcher.py        # NEW — FilmstripStitcherWidget
    filmstrip_bar.py             # NEW — FilmstripBarItem (QGraphicsItem subclass)
    by_person_scene.py           # NEW — ByPersonScene layout engine
    stitcher_panel.py            # modified — switch old/new widget; _Selection dataclass;
                                 #            auto-split/merge; overlap detection
```

### 9.4 `SyncTable.lookup()` dependency

`FilmstripStitcherWidget` calls `SyncTable.lookup(global_t, svid)` on every
mouse-move to compute per-camera frame numbers for the status bar and hover
tooltip.  This is O(log n) per call (binary search) and safe to call from the
main thread during paint/mouse events.  No async plumbing needed.

---

## 10. Open questions / decisions needed

| # | Question | Recommendation |
|---|----------|----------------|
| 1 | Should drag selection work across multiple bars (multi-bar multi-camera)? | **No for Phase 1** — single bar only. Cross-bar multi-select adds complexity without clear use case. |
| 2 | What happens if the user drags across a split point within a bar? | Select from drag-start to drag-end frame regardless; when assigned, the two resulting sub-segments get the same name and are immediately merged by auto-merge. |
| 3 | Should "By Person" show unassigned tracks? | **No** — only tracks with at least one assigned segment are shown. Unassigned tracks remain visible only in "By Detection" mode. |
| 4 | Bar height slider: global or per-camera? | **Global** for Phase 1. Per-camera zoom is a useful later addition. |
| 5 | Should the hover tooltip disappear immediately on mouse-leave or fade? | Immediate hide for simplicity. |
| 6 | Apply button conflict badge vs blocking dialog? | Non-blocking badge (described in §5.4) is preferred. Keeps the user unblocked while resolving overlaps. |
| 7 | Do we keep `split_requested` signal for backward compatibility with `PoseExtractionWindow`? | `StitcherWidget` keeps `split_requested` unchanged. `FilmstripStitcherWidget` never emits it. |
