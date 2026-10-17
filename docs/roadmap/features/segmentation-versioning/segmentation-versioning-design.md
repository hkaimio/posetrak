# Segmentation mask versioning

## 1. Problem

Every other input of a tracking run (detections, observation sequences, calibrations, skeletons,
tracker configurations) is append-only, so a run can be reproduced and compared with another.
Segmentation masks are not: a `seg_quality_runs` row keeps its id while its `seg_masks` rows are
overwritten in place. Nothing records which masks a derived result saw:

- boxes of a mask-sourced detection run and the heatmap gate of a SOMA run,
- the mask gate of a tracking run (`SegmentationMaskModifier`, given a segmentation per subject at
  invocation time and not stored),
- candidate mask labels and the mask overlay of the person view.

A result computed before an edit cannot be reproduced, and whether a detection run is stale after an
edit cannot be decided.

The write path is narrow: the segmentation panel's `_flush_masks` (`app/pose/cutie_init_panel.py`)
writes masks with `INSERT OR REPLACE`; the session tree deletes a whole segmentation run. About
twenty queries in eight files read `seg_masks` directly, including the C++ `SessionReader`.

## 2. Design: append-only revisions

### 2.1 Storage

- `seg_mask_revisions` (new): one row per revision of a segmentation run
  `(seg_quality_run_id, revision, created_at, closed_at, note)`. Revision 0 is the state of every
  segmentation when versioning starts. `closed_at` is NULL while the revision is open for editing.
- `seg_masks` gains `revision` and its primary key becomes
  `(seg_quality_run_id, shot_video_id, frame_idx, revision)`. A row holds the mask a frame has from
  that revision on. `mask_blob` NULL records that the frame has no mask from that revision on (a
  deletion).
- The mask of a frame at revision `r` is the row with the largest `revision <= r` for that frame;
  no row, or a NULL blob, means no mask.
- A view `seg_masks_latest (seg_quality_run_id, shot_video_id, frame_idx, revision, mask_blob)`
  returns the newest row per frame among closed revisions. Readers that want "the masks as they are now" (the overlay, the
  segmentation panel) read the view, which is a change of table name only; readers that reproduce a
  result read at a recorded revision through one helper per language.

### 2.2 Writing

- The segmentation panel opens a revision when the user starts editing a segmentation run (first
  write after opening it, or after the last save), writes changed frames under that revision, and
  closes it on save, on closing the panel, or when a long propagation job finishes. One revision is
  one user-visible save, not one brush stroke.
- Only changed frames are written; the panel already knows which frames a propagation or an edit
  touched. A propagation job that rewrites a range writes that range only.
- `INSERT OR REPLACE` is used only within the open revision (re-editing the same frame before the
  save replaces the row of the open revision, not an older one).

### 2.3 Concurrent use

A segmentation panel and command-line jobs (detection from masks, tracking with a mask gate,
`detect stale`) run at the same time on one session database. The rules that keep their results
consistent:

- **Readers pin a revision.** A job resolves the newest closed revision of its segmentation run
  once, when it starts, records it (section 2.4), and reads every frame at that revision. A save made
  while the job runs neither changes what the job reads nor mixes two revisions within one result.
- **Open revisions are invisible.** Rows of an open revision are read only by the panel that owns
  it. The latest view and the pinned read skip them, so a job never sees half an edit, and an edit
  abandoned by a crashed panel stays invisible until it is saved or discarded.
- **One editor per segmentation run.** A partial unique index allows one open revision per run. A
  second panel opening the same run gets it read-only, with the name of the open revision. An open
  revision left by a crash is offered for saving or discarding when the run is next opened.
- **Saving is one transaction.** Closing a revision sets `closed_at` in a single write, so a job
  starting at the same moment sees the revision either entirely or not at all. Writes use the busy
  timeout of the session-db concurrency design.
- **Pruning respects running jobs.** `seg prune` keeps revisions referenced by runs whose status is
  `running` as well as by finished runs.

### 2.4 Recording what a result saw

| Result | Records |
|---|---|
| Detection run from masks, or gated by masks | `detection_runs.seg_quality_run_id`, `detection_runs.seg_revision` |
| Tracking run subject with a mask gate | `tracking_run_persons.seg_quality_run_id`, `seg_mask_label`, `seg_revision` |
| Candidate mask labels, overlay | computed at read time from the recorded revision, not stored |

`seg_revision` is NULL on rows made before versioning, meaning "unknown".

### 2.5 Staleness of detections

After an edit, `posetrak detect stale RUN` compares the boxes the run used with the boxes of the
current revision per (camera, frame, label) and lists frames whose crop changed by more than a
tolerance (the spike used one pixel of the model input). Refreshing them creates a new detection run
(append-only): unchanged frames are copied with `INSERT ... SELECT`, changed frames are re-run, and the
new run records the current revision. This is the spike's incremental re-extraction, productized.

### 2.6 Space

Revisions store only changed frames. Masks are indexed PNGs at reduced resolution, so a revision
costs roughly the size of the frames it touched. `posetrak seg prune` deletes rows of revisions no
detection or tracking run references and that are older than the newest revision; it is explicit,
never automatic.

## 3. Migration

SQLite cannot change a primary key in place, so `seg_masks` is rebuilt: new table, copy with
`revision = 0`, drop, rename, recreate the index. On a large session this copies every mask blob once;
the migration reports progress. One `seg_mask_revisions` row with revision 0 is written per existing
segmentation run.

## 4. Acceptance

- A tracking run with a mask gate, re-run after the masks were edited, reproduces its result when
  given the recorded revision.
- The overlay shows the current masks; a tracking run's overlay can show the revision the run used.
- Editing one frame adds one row; no existing row changes.
- `detect stale` lists exactly the frames whose box changed beyond the tolerance (unit test on a
  fabricated segmentation).
- A tracking run started while a revision is open, and running while it is saved, reads only the
  revision that was newest and closed at its start (two-process test).
