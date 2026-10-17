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
4. **Stale 'running' rows.** A run row stays 'running' only while its process lives. Each run records
   nothing about its process; instead, the UI treats a 'running' row whose newest result is older
   than a timeout (10 minutes without a new step) as interrupted and shows it so, and `posetrak runs
   cleanup` marks such rows 'failed'. The same applies to `detection_runs`.
5. **The UI respects status.** The session tree shows running, failed and interrupted runs distinctly
   and does not open a run that is not complete as if it were (it may open it read-only with a
   warning, for diagnosis).

Existing `tracking_runs` rows are migrated as 'complete': there is no way to tell the broken ones
apart, and a run without results already shows as empty.

## 3. Acceptance

- A tracking run proceeds when a detection job writes to the same database during the run (scripted
  test with two processes and a 1-second writer lock held repeatedly).
- A tracker killed mid-run leaves a row the UI shows as interrupted, and `runs cleanup` marks it
  failed.
- A tracker that throws leaves a 'failed' row with the error text.
