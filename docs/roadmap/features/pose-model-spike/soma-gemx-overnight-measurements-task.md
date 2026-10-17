# Task: overnight measurement queue after the restricted-parent tracker rerun

You are working in the posetrak repository on branch `feature/soma77-pose-spike`. Stage 0 of
[soma-gemx-productization-plan.md](soma-gemx-productization-plan.md) is done except M2b: the
tracker baseline with the parent stage restricted to `main` (variants `main120` and `main60` of
`scripts/tracker_baseline.py`), which is running now. Prepare three more measurements and
one script that runs them in sequence after M2b. Then smoke-test each job on a short slice, and hand
the script to the operator.

Do not start any long run yourself. Agent background tasks are killed after about 20 minutes (open
issue), and anything heavy running next to M2b distorts its timings, which are compared with the
earlier M2 runs.

## Read first

- `CLAUDE.md`.
- Plan §4, stage 0 (its outcome) and stages 3 and 6.
- [tracker performance §2-§4](../tracker-performance/tracker-performance-design.md), including the
  procedure and results of §3 and the correction about the restricted parent.
- [detection pipeline §2-§3](../detection-pipeline/detection-pipeline-design.md), especially the
  instrumentation, the results, and the note on threaded decode hanging on close.
- `scripts/tracker_baseline.py`, `python/tools/gt/README.md`, `python/tools/gt/eval_kp_candidates.py`
  and `kp_eval_set.toml`.

## Jobs, in queue order

**J1 Tracker with the first release's parent marker set.** Make a copy of the SOMA-77 skeleton used
by the baseline, without the 44 body-crop finger markers. Keep `LeftHandIndex1`, `LeftHandPinky1` and
their right-side counterparts (see pose-model skeleton binding §2.4). Store it as a new skeleton
version in the work copy of the database, with `parent_id` set to the original. Add variants to
`tracker_baseline.py`: `main120m33` for part a and part b (parent restricted to `main`, the reduced
skeleton, 120 Hz), and `single120m33` for part a. Record the same columns as the other variants,
plus observations per step and the Kalman-gain time. The question it answers: how much of the gap to
the 8 GB and real-time-factor targets the stage 2 marker set closes on its own.

**J2 Accuracy of the baseline runs.** Score every tracking run of M2, M2b and J1 with the evaluation
harness against the label-fed run, on the frozen evaluation set: mean body-joint distance per person
and range, and the failure events at 40 cm. Where a run lacks a labelled range, say so instead of
scoring it. Add a table to tracker performance §3 that puts accuracy next to cost per variant. It
decides the default configuration (stage 6 acceptance). Read-only.

**J3 Parallel-decode soak test.** A prototype under `python/tools/prototypes/`. It opens, decodes and
closes the five videos of the M1 trial in parallel, in two modes:
- one thread per camera with PyAV (`thread_type="NONE"` and `"AUTO"`);
- one process per camera.

Use random 10 s windows, for a fixed number of cycles or a time limit (default 3 h). A watchdog
counts each hang on open, decode or close (no progress for 60 s), with its mode and window. Also
record the decode throughput per mode against the serial 124 s of M1. Record the result in detection
pipeline §3. It decides between decode threads and decode processes for stage 3. It is CPU-heavy, so
it runs last.

## The queue script

- One script, `scripts/overnight_queue.py` or a small `.ps1`, runs J1 part a, J1 part b, J2 and J3 in
  order. It first waits until no `posetrak-tracker` process is running, so it can be started before
  M2b finishes.
- It logs each job's start, end and exit status to one file in the work directory. A failed job does
  not stop the next one.
- Every job uses the work copy of the session database. The source databases are never written.
  Until session-db concurrency is built, a second writer on the database a tracker run uses kills
  that run, so J2 reads its own snapshot copy.
- It can be resumed: a job with a result file is skipped, as `tracker_baseline.py` does it.

## Rules

- Smoke-test every job yourself on a short slice (`--end-time-s`, or a few soak cycles), so the
  overnight run does not fail on a typo. Keep these tests short, below about 5 minutes each.
- Put results in the design documents as tables, with the capture, configuration, build and command.
  Update the touched `status.md` files. Design documents contain no dates, people's names, planning
  step identifiers or references to chats.
- Commits: cohesive, titled `comp: short description` (`scripts:`, `tools:`, `docs:`), no
  Co-authored-by line, pre-commit installed, unit tests passing. Stage paths explicitly. Leave
  unrelated changes alone: the modified prototype scripts, `cohn_db.json`, and `python/tools/iclone/`
  (locally excluded).
- Commit the scripts before the operator runs them. Results are committed when the operator reports
  that the queue has finished and you have written them up.

## Report back

The command for the operator to start the queue, where its log and results go, the expected total
run time per job, and the results of the smoke tests.
