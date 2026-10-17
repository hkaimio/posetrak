-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- 028_capture_events.sql
-- Capture-control event log: one row per camera per capture, recording when
-- the start and stop commands were sent, executed and confirmed. See
-- docs/roadmap/features/camera-control/camera-control-design.md §4.
--
-- All timestamps are ISO-8601 UTC on the controller clock.
--   start_cmd_ts / stop_cmd_ts   command sent by the controller
--   start_ack_ts                 camera (or its bridge) executed the command
--   start_confirm_ts / stop_confirm_ts
--                                positive confirmation that the camera is
--                                recording / has stopped; NULL when the camera
--                                type cannot confirm
-- confidence: 'confirmed' | 'commanded' | 'manual'
-- status:     'ok' | 'failed' | 'operator_override'
-- camera_instance_id references registry: camera_instances(id).

CREATE TABLE IF NOT EXISTS capture_events (
    capture_id         TEXT NOT NULL REFERENCES captures(id),
    camera_instance_id TEXT NOT NULL,
    start_cmd_ts       TEXT,
    start_ack_ts       TEXT,
    start_confirm_ts   TEXT,
    stop_cmd_ts        TEXT,
    stop_ack_ts        TEXT,
    stop_confirm_ts    TEXT,
    confidence         TEXT NOT NULL CHECK (confidence IN ('confirmed', 'commanded', 'manual')),
    status             TEXT NOT NULL CHECK (status IN ('ok', 'failed', 'operator_override')),
    notes              TEXT,
    PRIMARY KEY (capture_id, camera_instance_id)
);
