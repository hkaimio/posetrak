# Session database concurrency and run status

## 1. Problem

Several processes write one session database: the GUI, `posetrak` CLI commands (detection,
finalisation, segmentation) and the C++ tracker. The tracker's result writer opens the database with
`sqlite3_open_v2` and no busy timeout (`cpp/src/db/result_writer.cpp`), so a write that meets another
writer fails at once with "database is locked". One run died 8 minutes in this way and left a run row
without results; all progress was lost. A detection job, a finalisation, the UI opening the run or a
segmentation edit is enough to trigger it.

Failed jobs leave rows that look like results: `tracking_runs` has no status at all, so the UI
cannot tell a half-written run from a complete one. `detection_runs.status` exists ('running',
'complete', 'failed') but a killed process leaves 'running' forever.

## 2. Design

1. **Busy timeout everywhere.** The tracker's connections call `sqlite3_busy_timeout` (60 s). Python
   connections from `open_session` (`_connect` in `db.py`) already use WAL mode but keep Python's
   default 5 s timeout; they get the same 60 s. Connections opened with plain `sqlite3.connect`
   elsewhere (for example `PoseWorker`) go through `_connect`.
2. **Short write transactions.** The result writer commits every batch of steps (it already uses
   `BEGIN`/`COMMIT` per write; the batch size is kept so a transaction holds the lock for well under a
   second). Long Python writers (detection) commit per batch, as `DetectionBatchWriter` does.
3. **Run status.** `tracking_runs` gains `status` ('running', 'complete', 'failed'), `completed_at`
   and `failure` (the error text). The tracker inserts the run as 'running', sets 'complete' after
   the last stage, and 'failed' with the message in its exception handler. A hierarchical run's
   per-stage status stays in `tracking_run_stages`.
4. **Stale 'running' rows.** A run row stays 'running' only while its process lives. The row
   records nothing about the process; the newest activity is dated by `run_progress`
   (`run_id`, `updated_at`), which the tracker's result writer updates inside every batch commit
   (and at most every 30 s while patching a hierarchical child stage's rows) and the detection
   writers update with every batch. A run that has not reported yet is dated by its start time
   (`tracking_runs.ran_at`, `detection_runs.created_at`). A 'running' row with no progress for 10
   minutes is shown as interrupted (`posetrak.db.run_status`), and `posetrak runs cleanup` marks
   such rows 'failed' (and `posetrak runs list` shows every run that is not complete). The same
   applies to `detection_runs`. The progress row is deleted when the run ends or is cleaned up.
   Results have no wall-clock column, so a separate small table is the smallest way to date them;
   it needs no change to `detection_runs`, whose columns belong to a later stage.
5. **The UI respects status.** The session tree shows running, failed and interrupted runs distinctly
   (greyed, with the status in the label). A run that is not complete opens in the tracking run
   panel with a warning and without the actions that build on a result: exports, skeleton scaling
   and "Set as trial default" are disabled.

Existing `tracking_runs` rows are migrated as 'complete': there is no way to tell the broken ones
apart, and a run without results already shows as empty.

## 3. Acceptance

- A tracking run proceeds when another connection holds the write lock for several seconds during
  the run (C++ test: a second connection holds `BEGIN IMMEDIATE` while the result writer
  commits; Python test: `_connect` waits for a lock held by another process).
- A tracker killed mid-run leaves a row the UI shows as interrupted, and `runs cleanup` marks it
  failed (Python test on a 'running' row without recent progress).
- A tracker that throws leaves a 'failed' row with the error text (`finish_tracking_run`, called
  from the `track` command's exception handlers).
- A result batch commit holds the write lock well under a second (C++ test times a 500-step batch).
