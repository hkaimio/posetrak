```toml
name = "Session DB Concurrency and Run Status"
status = "in_progress"
progress_pct = 90
description = """
Busy timeouts and short transactions for every session database writer, a status for tracking \
runs, detection of interrupted runs, and a UI that does not open unfinished runs as results.
"""
categories = ["data-model", "tracker-core", "ui"]
target_release = "TBD"
last_updated = 2026-10-17
```

# Session DB Concurrency and Run Status — Implementation Status

See [session-db-concurrency-design.md](session-db-concurrency-design.md). Planned as part of
[the SOMA/GEM-X productization plan](../pose-model-spike/soma-gemx-productization-plan.md). Closes
the open issue "Tracker run aborts with 'database is locked' when another process writes the session
DB" in `docs/open-issues.md`.

| Item | Status |
|---|---|
| Busy timeout in the tracker (result writer, session reader, run finishing) and in `_connect`; direct `sqlite3.connect` sites routed through `_connect` | Implemented |
| Short write transactions: batch commit of the result writer measured under 0.5 s per 500-step batch | Implemented |
| `tracking_runs.status`, `completed_at`, `failure` (session schema v64; provisional number) | Implemented |
| `run_progress` table dating the newest activity of tracking and detection runs | Implemented |
| Interrupted-run detection (`posetrak.db.run_status`), `posetrak runs list` / `runs cleanup` | Implemented |
| Detection writers update progress per batch; pose worker marks a run failed on an exception | Implemented |
| Session tree and tracking run panel show and respect run status | Implemented |
| Migration run on a copy of a real session database | See the stage 1 report |

Not covered: a tracker that fails while it is still building its subjects (before the result writers
of all subjects exist) leaves 'running' rows that show as interrupted after 10 minutes and are
cleaned up by `runs cleanup`. Detection pipelines other than the pose worker's batch writer and the
dot candidate writer (for example the marker pipeline's keypoint writer) do not update
`run_progress` yet, so a long run of those is dated by its start time.
