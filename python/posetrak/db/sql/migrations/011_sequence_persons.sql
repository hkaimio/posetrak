-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- Migration: session schema v11 → v12
-- Adds sequence_persons to persist the person_name → person_id mapping that was
-- previously only computed in-memory during finalise_to_db.  Without this table
-- the assignment colours cannot be reconstructed when a detection run is reopened.

CREATE TABLE IF NOT EXISTS sequence_persons (
    sequence_id TEXT    NOT NULL REFERENCES pose_observation_sequences(id),
    person_id   INTEGER NOT NULL,
    person_name TEXT    NOT NULL,
    PRIMARY KEY (sequence_id, person_id)
);
