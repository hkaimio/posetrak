-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- 028_capture_subjects.sql
-- Capture-scoped identity for persons and objects (see
-- docs/roadmap/features/subject-identity/subject-identity-design.md).
-- capture_subjects gives every person/object a permanent integer label,
-- used directly as its segmentation-mask label. Triggers keep it in step
-- with capture_persons/capture_objects, so a raw INSERT gets a label too.
-- seg_quality_run_subjects holds only *exceptions*: label meanings of
-- segmentations whose labels are not the subjects' registry labels.

BEGIN;

CREATE TABLE IF NOT EXISTS capture_subjects (
    id         TEXT PRIMARY KEY,
    capture_id TEXT NOT NULL,   -- no FK: rows come only from capture_persons/objects triggers
    kind       TEXT NOT NULL CHECK (kind IN ('person', 'object')),
    label      INTEGER NOT NULL CHECK (label BETWEEN 1 AND 255),
    name       TEXT NOT NULL,
    deleted    INTEGER NOT NULL DEFAULT 0,
    UNIQUE (capture_id, label)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_capture_subjects_live_name
    ON capture_subjects (capture_id, name) WHERE deleted = 0;

CREATE TABLE IF NOT EXISTS seg_quality_run_subjects (
    seg_quality_run_id TEXT    NOT NULL REFERENCES seg_quality_runs(id),
    label              INTEGER NOT NULL,
    subject_id         TEXT    REFERENCES capture_subjects(id),
    PRIMARY KEY (seg_quality_run_id, label)
);

-- A deleted subject's row stays (its label is never reused: the next label is
-- MAX over all rows, deleted included).
CREATE TRIGGER IF NOT EXISTS trg_capture_persons_subject_ins
AFTER INSERT ON capture_persons
BEGIN
    INSERT INTO capture_subjects (id, capture_id, kind, label, name)
    VALUES (NEW.id, NEW.capture_id, 'person',
            COALESCE((SELECT MAX(label) FROM capture_subjects
                      WHERE capture_id = NEW.capture_id), 0) + 1,
            NEW.name);
END;
CREATE TRIGGER IF NOT EXISTS trg_capture_objects_subject_ins
AFTER INSERT ON capture_objects
BEGIN
    INSERT INTO capture_subjects (id, capture_id, kind, label, name)
    VALUES (NEW.id, NEW.capture_id, 'object',
            COALESCE((SELECT MAX(label) FROM capture_subjects
                      WHERE capture_id = NEW.capture_id), 0) + 1,
            NEW.name);
END;
CREATE TRIGGER IF NOT EXISTS trg_capture_persons_subject_rename
AFTER UPDATE OF name ON capture_persons
BEGIN
    UPDATE capture_subjects SET name = NEW.name WHERE id = NEW.id;
END;
CREATE TRIGGER IF NOT EXISTS trg_capture_objects_subject_rename
AFTER UPDATE OF name ON capture_objects
BEGIN
    UPDATE capture_subjects SET name = NEW.name WHERE id = NEW.id;
END;
CREATE TRIGGER IF NOT EXISTS trg_capture_persons_subject_del
AFTER DELETE ON capture_persons
BEGIN
    UPDATE capture_subjects SET deleted = 1 WHERE id = OLD.id;
END;
CREATE TRIGGER IF NOT EXISTS trg_capture_objects_subject_del
AFTER DELETE ON capture_objects
BEGIN
    UPDATE capture_subjects SET deleted = 1 WHERE id = OLD.id;
END;

-- Backfill: existing subjects get labels in sorted-name order per capture --
-- the same order the segmentation tool has used to number labels so far, so
-- most existing masks already match without any exception row.
INSERT OR IGNORE INTO capture_subjects (id, capture_id, kind, label, name)
SELECT id, capture_id, kind,
       ROW_NUMBER() OVER (PARTITION BY capture_id ORDER BY name), name
FROM (SELECT id, capture_id, 'person' AS kind, name FROM capture_persons
      UNION ALL
      SELECT id, capture_id, 'object' AS kind, name FROM capture_objects);

-- Exception rows for segmentations whose recorded label->name list disagrees
-- with the registry labels.
INSERT OR IGNORE INTO seg_quality_run_subjects (seg_quality_run_id, label, subject_id)
SELECT sq.id, CAST(je.key AS INTEGER) + 1, cs.id
FROM seg_quality_runs sq, json_each(sq.persons_json) je
JOIN capture_subjects cs ON cs.capture_id = sq.shot_id AND cs.name = je.value
WHERE sq.persons_json IS NOT NULL AND cs.deleted = 0
  AND CAST(je.key AS INTEGER) + 1 <> cs.label;

PRAGMA user_version = 58;
COMMIT;
