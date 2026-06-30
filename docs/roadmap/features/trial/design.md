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

## Segmentation scope

Segmentation (person body masks, used to improve keypoint confidence) belongs at the
**trial** level, not the detection run level.

Rationale:
- Segmentation is computed over a time range, independent of which detection model
  or parameters are used. A new detection run on the same trial reuses the existing
  segmentation.
- Future addition of specialised segmentations (e.g. "face", "hands") would still
  be scoped to the trial — they differ by body-part category, not by detection run.
- This means the trial page owns the "create / edit segmentation" action, and each
  detection run page has no segmentation controls.

For now a single segmentation per trial is sufficient. A future `segmentation_category`
column (defaulting to `"body"`) would allow multiple named segmentations per trial
without schema-breaking changes.

---

## UI panel design per level

### Capture panel

Current state is mostly correct. The one wrong label:

- **"Detect pose…" → "New trial…"** — the action at capture level is creating a
  trial (time window), not running detection. Detection is initiated from inside the
  trial. The dialog that currently opens (detection settings) should be replaced with
  a simpler "New trial" dialog (name, optional start/end time). Detection is then
  triggered from the trial page after the trial is created and opened.

### Trial panel — needs full rework

Current: embeds the full detection assignment editor (`StitcherPanel`), which duplicates
the Detection page. Correct design:

**Trial panel should contain:**
1. Basic info: trial name, time range, capture label.
2. Segmentation section: "Create segmentation" button if none exists, "Edit
   segmentation" button if one does. Segmentation belongs to the trial, not a
   detection run.
3. Detection runs list: one row per detection run (model, status, date, frame count).
   "Run detection…" button below the list to start a new one.
4. Tracking runs list: one row per tracking run across all detection runs in this
   trial (skeleton, status, date). "Run tracker…" button below the list.
5. Breadcrumb at the top showing `Session / Capture / Trial name` so the user can
   navigate up.

Clicking a detection run row navigates to the Detection panel. Clicking a tracking run
row navigates to the Tracking run panel.

### Detection panel — minor changes

Current is mostly correct. Changes needed:

- **Title says "trial" — change to detection run label/ID.** Add breadcrumb:
  `Session / Capture / Trial / Detection run`.
- **Remove "Run" dropdown** — this page is specific to one detection run. Run
  selection belongs in the trial panel's detection runs list.
- **Remove "Create/Edit segmentation" button** — segmentation is managed from the
  trial panel.
- **Rename "By detection" sort option → "By camera"** in the camera sort dropdown.
- **Rename "Apply" button → "Save assignments"** for clarity.

### Tracking run panel — lower priority, design TBD

The tracking run panel largely duplicates the person/tracking run page reached via
the tree. The right long-term design is to consolidate, but this is deferred — the
current state is usable.

---

## Gap 1 — DB schema: `tracking_runs.trial_id` is missing

`tracking_runs` has no `trial_id` column. Reaching a trial from a tracking run currently
requires a 3-hop join:

```sql
tracking_runs.observation_sequence_id
  → pose_observation_sequences.detection_run_id
  → detection_runs.trial_id
```

### Recommendation: add `trial_id` FK to `tracking_runs`

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
- Trial panel can list tracking runs without the join.

---

## Gap 2 — CLI: `export` and `import` are top-level, not under `trial`

**Decision: keep `export` and `import` as top-level commands.** They accept multiple
anchor types (`--trial`, `--capture`, `--detection`, `--tracking-run`, `--scope`) and
are not trial-specific. The Phase 5 design doc interface (`trial export ID`) was written
before the multi-anchor implementation existed; the implementation supersedes it.

### Missing CLI commands

`trial create` and `trial show` are needed to make trial a first-class scripted concept:

```
posetrak trial create --capture ID --name STR [--start S] [--end S]
posetrak trial show ID
posetrak trial list   # already exists
```

---

## Gap 3 — UI: context menu and panel content

The tree hierarchy is already correct (Capture → Trial → Detection run → Person track →
Tracking run). Panel content and context menu actions need alignment with the level
design above.

| Level | Action | Current state | Target state |
|---|---|---|---|
| Capture | New trial | "Detect pose…" button | Rename to "New trial…"; dialog creates trial only |
| Trial | Content | Shows StitcherPanel (detection editor) | Rework per trial panel design above |
| Trial | Segmentation | Not surfaced | "Create / Edit segmentation" button in trial panel |
| Trial | Run detection | Disabled context menu item | Button in trial panel's detection runs section |
| Detection | Title | Says "trial" | Show detection run ID; add breadcrumb |
| Detection | Run dropdown | Present | Remove; run selected from trial panel |
| Detection | Segmentation button | Present | Remove; belongs in trial panel |
| Detection | Sort option label | "By detection" | Rename to "By camera" |
| Detection | Apply button | "Apply" | Rename to "Save assignments" |
| Any | Navigation up | No breadcrumb | Add breadcrumb at top of each panel |

---

## Recommended implementation order

| Priority | Area | Change | Complexity |
|---|---|---|---|
| 1 | DB | Add `trial_id` to `tracking_runs` with backfill migration | Medium |
| 2 | UI | Rework TrialPanel (info + segmentation + detection/tracking run lists) | Large |
| 3 | UI | Rename "Detect pose…" → "New trial…" on capture panel; simplify dialog | Small |
| 4 | UI | Detection panel: remove run dropdown + segmentation btn; rename labels | Small |
| 5 | UI | Add breadcrumb to all panels | Medium |
| 6 | CLI | Add `trial create` and `trial show` commands | Small |
| 7 | UI | Enable "Export BVH…" context menu item | Small |
