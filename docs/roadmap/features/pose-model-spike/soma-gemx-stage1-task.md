# Task: stage 1 of the SOMA/GEM-X productization — session database robustness

Implement stage 1 of [soma-gemx-productization-plan.md](soma-gemx-productization-plan.md) (section 4)
as designed in [session-db-concurrency-design.md](../session-db-concurrency/session-db-concurrency-design.md).
Work in a separate git worktree on a new branch made from `main`. Do not use the spike branch
`feature/soma77-pose-spike`: it is a reference and is not merged (plan section 5, branches and
schema versions). Measurement runs are going on in the main work area, so do not touch its build
directories or its databases.

## Read first

- `CLAUDE.md`, all of it: build and test commands, schema documentation rules (every column of a
  touched table documented, the `DOCUMENTED_TABLES` ratchet), comment rules and git conventions.
- The session-db concurrency design and status, and the open issue "Tracker run aborts with
  'database is locked' when another process writes the session DB" in `docs/open-issues.md`.
- [soma-gemx-architecture-and-schema.md](soma-gemx-architecture-and-schema.md) §3.1 and §4. Stage 1
  owns session schema version 64, and §3.1 lists every column of `tracking_runs` after the change.
  Version 64 also adds the segmentation columns of `tracking_run_persons` (§3.2). They belong to this
  migration, with NULL values written until segmentation versioning exists. The number 64 is
  provisional: use the next free session version on `main` when the branch merges, and run the
  migration only on copies of databases until then.

## Scope

1. A busy timeout of 60 s on every tracker connection (`cpp/src/db/`) and in Python `_connect`.
   Route connections that bypass `_connect` through it.
2. Short write transactions in the result writer. Keep the batch commit, and check that the lock is
   held well under a second.
3. `tracking_runs.status`, `completed_at` and `failure`, written by the tracker: 'running' at
   insert, 'complete' after the last stage, 'failed' with the message on an exception. Existing rows
   migrate as 'complete'.
4. Interrupted runs: a 'running' row with no new step for 10 minutes is shown as interrupted, and a
   `posetrak runs cleanup` command marks such rows 'failed'. The same applies to
   `detection_runs.status`.
5. The session tree shows running, failed and interrupted runs distinctly. It opens an incomplete
   run only read-only, with a warning.

Out of scope: anything of later stages. Do not change the observation loading, the smoother or the
segmentation tables beyond the version-64 columns.

## Acceptance

- The design's §3 tests pass: the two-process lock test (a writer holding the database for several
  seconds does not kill a tracker run) and the killed-tracker test (the run shows as interrupted and
  `runs cleanup` marks it failed).
- The migration runs on a copy of a real session database. Every existing run is 'complete'
  afterwards, and nothing else changes; compare row counts and checksums of the other tables.
- `session_schema.sql` documents every column of `tracking_runs` and `tracking_run_persons`, both
  tables are in `DOCUMENTED_TABLES`, and the migration test passes.
- C++ tests (`./run_tests.sh`, or `meson test` in the worktree's own build directory) and
  `pytest python/tests/` pass.

## Rules

- Never write to a real session database. Test against copies, made outside the repository.
- Keep runs of the C++ build and tests short where possible, because measurement runs share the
  machine.
- Commits: cohesive, titled `comp: short description` (`db:`, `cpp:`, `ui:`, `cli:`, `docs:`), no
  Co-authored-by line, pre-commit installed in the worktree, tests passing before each commit. Stage
  paths explicitly; never `git commit -a`.
- Update the feature's `status.md`, and close the open issue in `docs/open-issues.md` (move it to
  the resolved section) when the tracker no longer fails on a locked database.
- Where the code disagrees with the design, change the design document to match what was built and
  say so in your report. Do not silently deviate.

## Report back

What was built, the test results (with output for any failure), the migration check on the database
copy, any deviation from the design, and the branch name, ready to merge.
