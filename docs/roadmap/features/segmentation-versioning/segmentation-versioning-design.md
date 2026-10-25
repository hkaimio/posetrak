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
  segmentation when versioning starts, and a trigger gives every segmentation created later a closed,
  empty revision 0 too, so masks written by tools that do not version (the `revision` column defaults to
  0) are visible. `closed_at` is NULL while the revision is open for editing; `note` holds the host and
  process id of the editor that opened it (§2.3). A revision opens only after the previous one is closed,
  so the closed revisions of a run are always `0..n`.
- `seg_masks` gains `revision` and its primary key becomes
  `(seg_quality_run_id, shot_video_id, frame_idx, revision)`. A row holds the mask a frame has from
  that revision on. `mask_blob` NULL records that the frame has no mask from that revision on (a
  deletion).
- The mask of a frame at revision `r` is the row with the largest `revision <= r` for that frame;
  no row, or a NULL blob, means no mask.
- A view `seg_masks_latest (seg_quality_run_id, shot_video_id, frame_idx, revision, mask_blob)`
  returns the newest row per frame among closed revisions, leaving out frames whose newest row is a
  deletion (NULL blob), so a reader of the view sees exactly the masked frames. Readers that want "the masks as they are now" (the overlay, the
  segmentation panel) read the view, which is a change of table name only; readers that reproduce a
  result read at a recorded revision through one helper per language.

### 2.2 Writing

- The segmentation panel opens a revision on the first mask write after opening a segmentation run (or
  after the last save), writes changed frames under that revision, and closes it when a propagation job
  finishes or fails, when the job queue ends or is stopped, and when the panel closes. One revision is
  one user-visible save, not one brush stroke. The panel has no separate save button: clicks and the
  paint and erase tools only change the mask on screen, and the database is written only by the
  propagation jobs that turn them into masks, so "save" and "job finished" are the same moment.
- Only changed frames are written: a job reports every frame of the range it propagated, and the panel
  skips a frame whose new mask equals the one the revision already sees. A job that rewrites a range
  writes that range only.
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
- **One editor per segmentation run.** A partial unique index allows one open revision per run, so two
  processes racing to open one cannot both succeed. A second panel opening the same run gets it
  read-only, with the name of the open revision (its tracking and pose buttons are disabled). A revision
  records `host=<name> pid=<n>` of its editor; one whose process is gone on the same host was left by a
  crash and is offered for saving or discarding when the run is next opened. A revision of another host,
  or without such a note, counts as live, since nothing can tell otherwise.
- **Saving is one transaction.** Closing a revision sets `closed_at` in a single write, so a job
  starting at the same moment sees the revision either entirely or not at all. Writes use the busy
  timeout of the session-db concurrency design.
- **Pruning respects running jobs.** `seg prune` keeps revisions referenced by runs whose status is
  `running` as well as by finished runs.

### 2.4 Recording what a result saw

| Result | Records |
|---|---|
| Detection run from masks, or gated by masks | `detection_runs.seg_quality_run_id`, `detection_runs.seg_revision` (the box source reports them; a run built from several jobs keeps the first job's revision) |
| Tracking run subject with a mask gate | `tracking_run_persons.seg_quality_run_id`, `seg_mask_label`, `seg_revision`, written by the tracker (`ResultWriter::record_segmentation`) |
| Candidate mask labels, overlay | computed at read time from the recorded revision, not stored |

`seg_revision` is NULL on rows made before versioning, meaning "unknown". Deleting a segmentation
clears the references to it in these rows (the foreign keys would refuse otherwise); the results keep
their data. `posetrak-tracker track` takes the revision as `--seg-revision`, defaulting to the newest
closed one at the start of the run; `track run` resolves it itself and passes it on, so it is known
before the tracker starts.

### 2.5 Staleness of detections

After an edit, `posetrak detect stale RUN` compares the boxes the run used with the boxes of the
current revision per (camera, frame, label) and lists frames whose crop changed by more than a
tolerance. The tolerance (`--tolerance`, default 1) is in pixels of the pose model's input: a box edge
that moved by `d` frame pixels moves the crop by `d * input_width / box_width`, so a large box tolerates
a larger shift. A person appearing in or vanishing from a frame is always a change. When the run's
`seg_revision` is known only frames that revisions after it touched are compared; with an unknown
revision every frame between a camera's first and last detection is. `--refresh` creates a new
detection run (append-only): unchanged frames are copied with `INSERT ... SELECT` (detections,
keypoints, preview crops, tracks), changed frames are re-run, and the new run records the current
revision and names the old run in `source_detection_run_id`. This is the spike's incremental
re-extraction, productized.

### 2.6 Space

Revisions store only changed frames. Masks are indexed PNGs at reduced resolution, so a revision
costs roughly the size of the frames it touched. `posetrak seg prune [SEG_RUN] [--dry-run]` removes
the revisions no detection run or tracking run subject references (whatever the run's status) other
than the newest closed one, and never an open one; it is explicit, never automatic. Revision 0 is
removable like any other. A row of a removed revision that a kept revision still reads (no later
row of its frame up to that revision) moves to the next kept revision, so every kept revision reads
exactly as before; rows a later revision covers are deleted, and that is where the space comes back.

## 3. Migration

SQLite cannot change a primary key in place, so `seg_masks` is rebuilt: new table, copy with
`revision = 0`, drop, rename. On a large session this copies every mask blob once; the migration copies
run by run and reports progress through the `posetrak.db.db` logger, all in one transaction. One
`seg_mask_revisions` row with revision 0 (closed at the run's creation time) is written per existing
segmentation run. The rebuilt table is the same size, but the dropped one leaves free pages in the file:
the database file grows by about the size of its masks until a `VACUUM` (measured in §4).

## 4. Acceptance

- A tracking run with a mask gate, re-run after the masks were edited, reproduces its result when
  given the recorded revision.
- The overlay shows the current masks; a tracking run's overlay can show the revision the run used.
- Editing one frame adds one row; no existing row changes.
- `detect stale` lists exactly the frames whose box changed beyond the tolerance (unit test on a
  fabricated segmentation).
- A tracking run started while a revision is open, and running while it is saved, reads only the
  revision that was newest and closed at its start (two-process test).

### 4.1 Results

Measured on copies made with the SQLite backup API; no real session or registry database was written.

- **Reproduction.** On a copy of the marker-props session (ukemi trial, person with a mask gate,
  `track run --seg-run ... --dot-proposals`, 14.2 s, about 4 minutes per run), run A read revision 0 and
  recorded `(segmentation, label 1, revision 0)`. After the masks of the trial's frames (8,506 masks, the
  person's mask shifted sideways) were saved as revision 1, run B (default) recorded revision 1 and
  differed from A; run C (`--seg-revision 0`) recorded revision 0 and its `tracking_results` (state,
  covariance) and `tracking_obs_results` rows hashed identically to A's (3,404 and 1,702 rows).
- **Two processes.** A second process held revision 1 open (written, unsaved) while run P started, and
  saved it about ten seconds into P's tracking. P recorded revision 0 and its results were identical to A's.
- **Overlay.** Unit test: with a tracking run's recorded revision selected the overlay decodes that
  revision's mask, otherwise the current one.
- **One row per edit.** Unit test: saving one changed frame adds one row; rows of earlier revisions are
  unchanged.
- **`detect stale`.** Unit test on a fabricated segmentation: exactly the frames whose box edge moved by more
  than the tolerance, or whose person vanished or appeared, are listed; a refresh copies the others.
- **Migration.** Marker-props session, schema 62 to 68: 13.59 GB, one segmentation of 120,684 masks
  (1.03 GB of blobs); 7.4 s; file 13.588 GB before, 14.715 GB after (the freed pages of the old table stay
  in the file until `VACUUM`); the mask hash of the run unchanged. A second real session at schema 66, 3.97 GB,
  two segmentations (40,192 and 44,436 masks, 617 MB): 3.1 s; 3,974.8 MB to 4,656.9 MB; row count and
  SHA-256 of every run's masks unchanged; 47 other tables unchanged in row count and content hash, and
  `tracker_configs` changed only by the column that the 66-to-67 step adds.
- **GUIs.** The migrated copy of the second session opens in the main viewer (tree with current mask counts,
  person view overlay decoding masks for 4 of 4 cameras, segmentation window editable) and in the pose
  detection window (offscreen Qt smoke test).
