```toml
name = "Session DB Concurrency and Run Status"
status = "planned"
progress_pct = 0
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
| Busy timeout in the tracker and in `open_session` | Planned |
| `tracking_runs.status`, `completed_at`, `failure` | Planned |
| Interrupted-run detection and `runs cleanup` | Planned |
| Session tree shows and respects run status | Planned |
