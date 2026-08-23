-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- Migration v16 → v17: add detection_run_id to frame_cache_entries primary key.
--
-- Allows PERSON_CROP entries from multiple detection runs on the same shot to
-- coexist.  Non-crop entries (THUMB, FULL_FRAME) use detection_run_id = ''.
--
-- frame_cache_entries is a performance cache; losing existing rows is safe.
-- person_crop entries from the old schema lack a known detection_run_id so they
-- are discarded; all other cache types are migrated with detection_run_id = ''.
--
-- Also removes the now-redundant detection_crops table added in v16.

BEGIN;

CREATE TABLE frame_cache_entries_new (
    shot_video_id       TEXT    NOT NULL REFERENCES capture_videos(id),
    frame_idx           INTEGER NOT NULL,
    cache_type          TEXT    NOT NULL,
    track_id            INTEGER NOT NULL DEFAULT -1,
    region_type         TEXT    NOT NULL DEFAULT '',
    width_px            INTEGER NOT NULL DEFAULT 0,
    height_px           INTEGER NOT NULL DEFAULT 0,
    image_data          BLOB    NOT NULL,
    detection_run_id    TEXT    NOT NULL DEFAULT '',
    PRIMARY KEY (shot_video_id, frame_idx, cache_type, track_id, region_type, width_px, detection_run_id)
);

INSERT OR IGNORE INTO frame_cache_entries_new
    (shot_video_id, frame_idx, cache_type, track_id, region_type,
     width_px, height_px, image_data, detection_run_id)
SELECT shot_video_id, frame_idx, cache_type, track_id, region_type,
       width_px, height_px, image_data, ''
FROM frame_cache_entries
WHERE cache_type != 'person_crop';

DROP TABLE frame_cache_entries;
ALTER TABLE frame_cache_entries_new RENAME TO frame_cache_entries;

DROP TABLE IF EXISTS detection_crops;
DROP INDEX IF EXISTS idx_detection_crops_lookup;

PRAGMA user_version = 17;
COMMIT;
