-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- Session schema v67 -> v68: append-only segmentation mask revisions.
-- This file holds the schema objects that do not copy data; _migrate_session_v67_to_v68 in
-- db.py rebuilds seg_masks run by run (so it can report progress), backfills revision 0 and
-- bumps the version. Column documentation lives in session_schema.sql.
CREATE TABLE seg_mask_revisions (
    seg_quality_run_id TEXT    NOT NULL REFERENCES seg_quality_runs(id),
    revision           INTEGER NOT NULL,
    created_at         TEXT    NOT NULL,
    closed_at          TEXT,
    note               TEXT,
    PRIMARY KEY (seg_quality_run_id, revision)
);
CREATE UNIQUE INDEX idx_seg_mask_revisions_open
    ON seg_mask_revisions (seg_quality_run_id) WHERE closed_at IS NULL;

CREATE TABLE seg_masks_new (
    seg_quality_run_id TEXT    NOT NULL REFERENCES seg_quality_runs(id),
    shot_video_id      TEXT    NOT NULL REFERENCES capture_videos(id),
    frame_idx          INTEGER NOT NULL,
    revision           INTEGER NOT NULL DEFAULT 0,
    mask_blob          BLOB,
    PRIMARY KEY (seg_quality_run_id, shot_video_id, frame_idx, revision)
);

-- @@ after copy
-- (run once seg_masks_new holds every row and has replaced seg_masks)
CREATE VIEW seg_masks_latest AS
SELECT m.seg_quality_run_id, m.shot_video_id, m.frame_idx, m.revision, m.mask_blob
FROM seg_masks m
WHERE m.mask_blob IS NOT NULL
  AND m.revision = (
      SELECT MAX(m2.revision)
      FROM seg_masks m2
      JOIN seg_mask_revisions r2
        ON r2.seg_quality_run_id = m2.seg_quality_run_id AND r2.revision = m2.revision
      WHERE m2.seg_quality_run_id = m.seg_quality_run_id
        AND m2.shot_video_id = m.shot_video_id
        AND m2.frame_idx = m.frame_idx
        AND r2.closed_at IS NOT NULL);

CREATE TRIGGER trg_seg_quality_runs_revision0
AFTER INSERT ON seg_quality_runs
BEGIN
    INSERT INTO seg_mask_revisions (seg_quality_run_id, revision, created_at, closed_at)
    VALUES (NEW.id, 0, NEW.created_at, NEW.created_at);
END;
