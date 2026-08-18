-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- Migration: session schema v7 → v8
-- Adds person_detections, person_tracks, and frame_cache_entries tables
-- for the Phase 2 setup application (capture pipeline).

BEGIN;

-- Person detections: one row per (video, frame, track, region type).
-- Model-agnostic replacement for the former yolo_detections concept.
-- region_type: 'full_body' | 'face' | 'hand_l' | 'hand_r'
-- track_id: detection tracker ID, assigned before person identity is known.
CREATE TABLE IF NOT EXISTS person_detections (
    shot_video_id  TEXT    NOT NULL REFERENCES shot_videos(id),
    video_frame    INTEGER NOT NULL,
    track_id       INTEGER NOT NULL,
    region_type    TEXT    NOT NULL DEFAULT 'full_body',
    model_name     TEXT,
    bbox_x         REAL,
    bbox_y         REAL,
    bbox_w         REAL,
    bbox_h         REAL,
    confidence     REAL,
    PRIMARY KEY (shot_video_id, video_frame, track_id, region_type)
);

-- Person tracks: one row per continuous track span within a shot video.
-- Created by the detection pipeline; referenced by frame_cache_entries.
CREATE TABLE IF NOT EXISTS person_tracks (
    id             TEXT    PRIMARY KEY,
    shot_video_id  TEXT    NOT NULL REFERENCES shot_videos(id),
    track_id       INTEGER NOT NULL,
    first_frame    INTEGER NOT NULL,
    last_frame     INTEGER NOT NULL,
    UNIQUE (shot_video_id, track_id)
);

-- Frame cache: stores decoded/cropped image data for fast UI access.
-- cache_type: 'THUMB' | 'PERSON_CROP'
-- track_id:   -1 for cache types that do not require a person track (e.g. THUMB)
-- region_type: '' (empty) for THUMB; 'full_body'/'face'/'hand_l'/'hand_r' for PERSON_CROP
-- width_px / height_px: output image dimensions.
CREATE TABLE IF NOT EXISTS frame_cache_entries (
    shot_video_id  TEXT    NOT NULL REFERENCES shot_videos(id),
    frame_idx      INTEGER NOT NULL,
    cache_type     TEXT    NOT NULL,
    track_id       INTEGER NOT NULL DEFAULT -1,
    region_type    TEXT    NOT NULL DEFAULT '',
    width_px       INTEGER NOT NULL,
    height_px      INTEGER NOT NULL,
    image_data     BLOB    NOT NULL,
    PRIMARY KEY (shot_video_id, frame_idx, cache_type, track_id, region_type, width_px)
);

PRAGMA user_version = 8;
COMMIT;
