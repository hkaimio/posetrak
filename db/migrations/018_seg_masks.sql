-- Migration 018: add seg_masks table for interactive Cutie init widget.
--
-- Stores a labeled segmentation mask (indexed PNG blob) per frame, keyed by
-- (seg_quality_run_id, shot_video_id, frame_idx).  Populated incrementally
-- during interactive tracking; rows with frame_idx > correction_frame are
-- deleted when the user corrects a tracked frame and re-tracks.

CREATE TABLE IF NOT EXISTS seg_masks (
    seg_quality_run_id TEXT    NOT NULL REFERENCES seg_quality_runs(id),
    shot_video_id      TEXT    NOT NULL REFERENCES capture_videos(id),
    frame_idx          INTEGER NOT NULL,
    -- Indexed PNG: uint8 per pixel.  Label 0 = background, 1..N = person.
    -- Person label -> name mapping: seg_quality_runs.persons_ordered JSON array.
    -- Typical compressed size at 1080p: 5–15 KB per frame.
    mask_blob          BLOB    NOT NULL,
    PRIMARY KEY (seg_quality_run_id, shot_video_id, frame_idx)
);

PRAGMA user_version = 19;
