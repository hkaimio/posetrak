# Trial concept — refactoring design

## Goal

Align the DB schema, CLI, and UI with the Phase 5 design model where `trial` is the
first-class, self-contained unit of a motion capture workflow:

```
session → capture → trial → detection run → tracking run
```

A trial is a named time window within a capture. It is the minimal portable atom:
exporting a trial carries its detection runs, tracking runs, cameras, calibrations,
skeleton, and config.

---

## Current state

The DB mostly has this structure already — migration 012 added the `trials` table and
`trial_id` on `detection_runs`. Three concrete gaps remain.

---

## Gap 1 — DB schema: `tracking_runs.trial_id` is missing

`tracking_runs` has no `trial_id` column. Reaching a trial from a tracking run currently
requires a 3-hop join:

```sql
tracking_runs.observation_sequence_id
  → pose_observation_sequences.detection_run_id
  → detection_runs.trial_id
```

### Options

**A. Add `trial_id` FK to `tracking_runs` (migration)**

A new migration adds `trial_id TEXT REFERENCES trials(id)` (nullable to handle rows
that predate trial assignment). A data migration backfills existing rows:

```sql
UPDATE tracking_runs
SET trial_id = (
    SELECT dr.trial_id
    FROM detection_runs dr
    JOIN pose_observation_sequences s ON s.id = tracking_runs.observation_sequence_id
    WHERE dr.id = s.detection_run_id
)
```

Downstream impact:
- `trial_export.py` anchor resolution simplifies (direct FK lookup).
- `session_tree.py` `_add_tracking_runs()` query simplifies.
- `trial list` tracking run count subquery simplifies.

**B. Keep the 3-hop join (no schema change)**

The join always reflects the true relationship. The only cost is query verbosity.

### Recommendation: Option A

The redundant column is worth it for clarity and query simplicity. The relationship is
deterministic so the backfill is safe. The migration is straightforward.

---

## Gap 2 — CLI: `export` and `import` are top-level, not under `trial`

The design doc specifies `posetrak trial export` / `posetrak trial import`, but the
current implementation registers them as top-level commands (`posetrak export`,
`posetrak import`). The current implementation has a more capable multi-anchor interface
(`--trial`, `--capture`, `--detection`, `--tracking-run`, `--scope`) that covers export
scenarios beyond trials (e.g. exporting a single capture or detection run). Moving these
under `trial` would imply they only operate on trials.

**Decision: keep `export` and `import` as top-level commands.** The commands already
support `--trial ID` as one of several anchor types. Nesting them under `trial` would be
misleading given the broader scope.

The design doc interface was written before the multi-anchor implementation existed; the
implementation supersedes it.

### Missing CLI commands

`trial create` and `trial show` are needed to make trial a first-class scripted concept:

```
posetrak trial create --capture ID --name STR [--start S] [--end S]
posetrak trial show ID
posetrak trial list
```

`trial list` already exists. `trial create` and `trial show` are small additions to
`trial.py`.

---

## Gap 3 — UI: context menu items are disabled

The tree hierarchy is already correct (Capture → Trial → Detection run → Person track →
Tracking run). Most context menu items are stubbed out with `setEnabled(False)`.

| Item | Status | Notes |
|---|---|---|
| New trial… | Disabled | Dialog: name + start/end time → insert into `trials` |
| Run detection… | Disabled | Depends on detection pipeline UI; leave for later |
| Finalise → person tracks… | Disabled | Same; leave for later |
| Run tracker… | Redundant | Already accessible via PersonPanel "Run Tracker" button |
| View results… | Redundant | Tree selection already navigates to TrackingRunPanel |
| Export BVH… | Disabled | Wire existing `track export bvh` CLI to a file-chooser dialog |

**`TrialPanel`** currently shows trial metadata only. Once `tracking_runs.trial_id`
exists (Gap 1), a tracking runs list can be added so the user can navigate from trial
directly to any tracking run without going through the detection run → person track path.

---

## Recommended implementation order

| Priority | Area | Change | Complexity |
|---|---|---|---|
| 1 | CLI | Add `trial create` and `trial show` commands | Small |
| 2 | DB | Add `trial_id` to `tracking_runs` with backfill migration | Medium |
| 3 | UI | Enable "New trial…" context menu item | Small |
| 4 | UI | Enable "Export BVH…" context menu item | Small |
| 5 | UI | `TrialPanel` shows associated tracking runs | Medium (easier after priority 2) |
| 6 | CLI/UI | Simplify trial navigation using direct `trial_id` FK | Small (after priority 2) |
