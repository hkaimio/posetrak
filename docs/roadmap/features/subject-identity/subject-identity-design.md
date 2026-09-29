# Capture-scoped subject identity (persons and objects) — design proposal

Status: phases 1, 2 and 3 implemented (registry, triggers, migration, exception
table, label-mapping UI; segmentation editor allocates by registry label,
viewer overlays share the editor's palette and show subject names);
phase 4 open.

## 1. What is inconsistent today

Identity for "something that is tracked in a capture" exists in five
different forms:

| Where | Identity used | Notes |
|---|---|---|
| `capture_persons` / `capture_objects` | separate UUID tables | Persons and props are unrelated tables with different link columns. |
| Mask labels (`seg_masks.mask_blob`) | integer label 1..N | Assigned by the Cutie panel as **position in a sorted union of names** (persons ∪ track-assignment names ∪ object names). Adding a subject shifts every later label. |
| `seg_quality_runs.persons_json` | ordered name list, index+1 = label | A snapshot at creation, names only. Does not have to cover every label present: the sword is label 3 in the sword session's mask but `persons_json` is `["Harri"]`, so it cannot be resolved by name. |
| `sequence_persons`, `pose_observations.person_id` | sequence-local integer + name (+ `capture_person_id` for persons only) | Objects have no `sequence_persons` row. |
| `tracking_run_persons` | sequence-local integer + `capture_object_id` (objects only) | Not written at all by the single-subject CLI path. Persons have no id here. |

Consequences seen so far: the tracker was given the wrong mask label; the UI
could not colour or name the sword's mask; resolving "which label is this
subject" needs a name join that is per-run and can silently fail.

## 2. Proposal

**One capture-scoped registry, `capture_subjects`, with a permanent integer
label per subject that is used directly as the mask label.**

```sql
CREATE TABLE capture_subjects (
    id         TEXT PRIMARY KEY,   -- equals capture_persons.id / capture_objects.id
    capture_id TEXT NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN ('person','object')),
    label      INTEGER NOT NULL CHECK (label BETWEEN 1 AND 255),
    name       TEXT NOT NULL,
    deleted    INTEGER NOT NULL DEFAULT 0,
    UNIQUE (capture_id, label)
);
CREATE UNIQUE INDEX idx_capture_subjects_live_name
    ON capture_subjects (capture_id, name) WHERE deleted = 0;
```

- `id` reuses the existing person/object UUID, so no existing foreign key
  changes and there is no id collision (UUIDs).
- `label` is allocated once (`MAX(label)+1` over all rows, deleted ones
  included) and **never reused**, so old masks stay unambiguous.
- The person/object tables keep their own payload (default skeleton, marker
  body); only identity is shared.
- The cross-table uniqueness a plain column cannot give (a person and an
  object cannot share a name) is enforced by the registry's constraints.

`seg_quality_run_subjects` holds **exceptions only**; a subject with no row in
a run means "this run uses the subject's registry label":

```sql
CREATE TABLE seg_quality_run_subjects (
    seg_quality_run_id TEXT NOT NULL REFERENCES seg_quality_runs(id),
    label      INTEGER NOT NULL,
    subject_id TEXT REFERENCES capture_subjects(id),   -- NULL = explicitly unassigned
    PRIMARY KEY (seg_quality_run_id, label)
);
```

It exists for segmentations whose labels are not the registry labels (all
segmentations made before the registry, and any later run that covers only
some subjects). A segmentation made with registry labels needs no rows. A
subject moved to another label in a run is not also claimed by its registry
label (`manage_subject.subject_for_label_in_run`).

The migration backfills existing subjects with labels in **sorted-name order
per capture** (the order the Cutie panel has numbered labels so far, so most
existing masks match with no exception row), then writes exception rows
from each run's `persons_json` where its ordinal disagrees with the registry
label. Registry rows are kept in step with `capture_persons` and
`capture_objects` by SQL triggers (insert allocates `MAX(label)+1`, rename
updates the name, delete marks the row deleted and retires its label), so
raw-SQL inserts are covered too. Labels are limited to 255 (uint8 PNG).
`capture_subjects.capture_id` has no foreign key: rows come only from the
triggers, which already reference `captures` through the person/object
tables.

## 3. What each consumer does afterwards

- **Cutie panel**: SAM object id = `subject.label`, not sorted position. A
  subject added later cannot shift anyone. Old runs keep their own mapping
  via `seg_quality_run_subjects`.
- **Tracker CLI**: `--seg-mask-label` is resolved from
  `(seg_run, subject)` through `seg_quality_run_subjects`, or is simply
  `subject.label` for new runs; C++ stays a plain integer compare. A
  `--subject <name>` convenience in the Python launcher does the lookup.
- **UI overlays**: colour by label (stable per subject) and show the
  subject's name in a small legend, replacing the anonymous all-labels
  colours added for object views.
- **Sequences and runs**: add nullable `subject_id` to `sequence_persons`
  (persons and objects both get a row), `tracking_run_persons`, and
  `detection_track_assignments` (replacing name matching). The sequence-local
  `person_id` integer stays as the array index in blobs; only its meaning
  ("which subject") becomes an explicit foreign key. `detection_runs.
  capture_object_id` stays as is.
- **Detection track ids** (`person_tracks.track_id`) stay run-local. In a
  YOLOX run they are the detector tracker's ids (unreliable, restarted in
  difficult videos); stitching maps them to a person in
  `detection_track_assignments`, and that association is what gets a
  `subject_id`. In a segmentation-driven run `track_id` is the mask label
  itself, so with registry labels it is already the subject's label.

## 4. Phasing (each step shippable)

1. Registry table, allocator helper, migration, `seg_quality_run_subjects`
   backfill. Add a UI action to map an unmapped label to a subject (fixes the
   sword session immediately).
2. Cutie panel allocates by subject label; UI overlays use registry names
   and colours.
3. `subject_id` columns on sequences/runs/track assignments; tracker
   launcher resolves labels through the registry; CLI writes
   `tracking_run_persons` rows.
4. Later, optional: drop `persons_json` and name-based lookups.

## 5. Alternatives considered

- **Columns only** (`label` on each of the two tables): smallest change but
  cross-table uniqueness is only convention.
- **Merge into one `capture_subjects` table carrying skeleton/marker body**:
  cleanest model, but rewrites every `capture_person_id`/`capture_object_id`
  reference; the registry above gets the identity benefits without that.
- **Keep names as keys**: what we have; fails as soon as a subject is renamed
  or a run predates a subject.

## 5b. Phase 2 as built

- The Cutie panel builds its name list from the registry: index i holds the
  name of label i+1, `""` for a label no subject has (so a deleted subject
  leaves a gap instead of renumbering everyone). Names known only from an
  earlier detection run's track assignments become persons first, so they get
  a registry label. `persons_json` written for a new segmentation is that
  same list, so old readers still work; consumers skip `""` entries (Cutie
  worker objects, auto-assign, the detection dialog's summary).
- Registry order is creation order, no longer sorted-name order, for
  subjects created after the registry existed.
- The tracking-run viewer colours labels with the editor's
  `label_to_color` palette and draws a swatch and the subject's name per
  painted label (from `subject_for_label_in_run`, so a Mask labels
  correction shows up there).

## 5c. Phase 3 as built (schema v59)

- Nullable `subject_id` on `sequence_persons`, `tracking_run_persons` and
  `detection_track_assignments`. The migration backfills from
  `capture_person_id` (a person's registry id is its `capture_persons` id),
  else by name within the capture; a run's subject is its tracked object or,
  for a person, its sequence's person. Tracking runs that predate any
  `tracking_run_persons` row get one. `finalise_to_db` writes `subject_id`
  on new sequences and assignments.
- **Deviation from section 3:** object sequences still get **no**
  `sequence_persons` row (the session tree treats their absence as "this is
  an object sequence"). `manage_subject.subject_id_for_sequence` answers for
  both kinds instead: a person's `sequence_persons.subject_id`, else the
  detection run's `capture_object_id`.
- The C++ tracker now writes the `tracking_run_persons` row when it creates a
  run (`capture_object_id` and `subject_id` resolved by plain SQL from the
  sequence, no name matching). It is best effort: a database without those
  tables is left alone so provenance can never block a run.
- `posetrak track run --seg-run <id>` looks the tracked subject's label up
  through the registry (`seg_mask_label_for_sequence`, honouring the run's
  exception rows) and passes it to the tracker; `--seg-label` overrides it.
  The tracker config's `dot_seg_mask_enabled` still has to be set.

## 6. Decisions and open points

- 255 subjects per capture is enough (confirmed).
- Renaming a subject is allowed; the only constraint is a unique live name
  per capture (a person and an object cannot share a name).
- Old segmentations: the migration writes exception rows from
  `persons_json`; a label that had no name (the sword's label 3 in the sword
  session) is left to identity or assigned once in the segmentation panel's
  "Mask labels" table.
