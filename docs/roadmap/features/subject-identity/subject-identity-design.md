# Capture-scoped subject identity (persons and objects) — design proposal

Status: proposal for review, nothing implemented.

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
    capture_id TEXT NOT NULL REFERENCES captures(id),
    kind       TEXT NOT NULL CHECK (kind IN ('person','object')),
    label      INTEGER NOT NULL CHECK (label BETWEEN 1 AND 255),
    name       TEXT NOT NULL,
    UNIQUE (capture_id, label),
    UNIQUE (capture_id, name)
);
```

- `id` reuses the existing person/object UUID, so no existing foreign key
  changes and there is no id collision (UUIDs).
- `label` is allocated once (lowest unused, or max+1) and **never reused**,
  even after a subject is deleted, so old masks stay unambiguous. 255 fits a
  uint8 PNG.
- The person/object tables keep their own payload (default skeleton, marker
  body); only identity is shared. Creating a person or object goes through
  one helper that also inserts the registry row.
- The cross-table uniqueness a plain column cannot give is enforced by the
  registry constraints.

For masks of segmentations that already exist:

```sql
CREATE TABLE seg_quality_run_subjects (
    seg_quality_run_id TEXT NOT NULL REFERENCES seg_quality_runs(id),
    label      INTEGER NOT NULL,
    subject_id TEXT REFERENCES capture_subjects(id),   -- NULL = unmapped
    PRIMARY KEY (seg_quality_run_id, label)
);
```

Migration fills it from `persons_json` (ordinal i+1 → subject by name) and
also records every label actually present in the run's masks, leaving
labels with no name (label 3 here) `NULL` until a person assigns them once
in the UI. New segmentations write `label = subject.label` and one row per
subject; `persons_json` stays for old readers and is then derived.

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

## 6. Open questions

1. Are 255 subjects per capture enough (yes for any realistic capture)?
2. Should renaming a subject be allowed? With permanent labels it is safe;
   `name` uniqueness within a capture is the only constraint.
3. For old segmentations, is a one-time "assign label to subject" UI step
   acceptable, or should the migration guess (e.g. by label frequency)?
4. Should a person's pose track ids (`person_tracks.track_id`, per detection
   run) also map to a subject at detection time, or stay run-local as now?
