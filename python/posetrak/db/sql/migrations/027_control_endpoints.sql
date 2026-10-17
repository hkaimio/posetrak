-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- 027_control_endpoints.sql
-- How a physical camera is controlled remotely during capture. See
-- docs/roadmap/features/camera-control/camera-control-design.md §4.
--
-- One row per camera_instance that capture control can drive; cameras
-- without a row are operated manually. COHN credentials for 'gopro_cohn'
-- cameras are secrets and are deliberately not stored here: the capture
-- controller keeps them in a local file under the user config directory.

CREATE TABLE IF NOT EXISTS control_endpoints (
    camera_instance_id TEXT PRIMARY KEY REFERENCES camera_instances(id),
    method             TEXT NOT NULL CHECK (method IN (
                           'gopro_cohn', 'gopro_ble', 'insta360_ble',
                           'adb', 'ble_hid', 'manual')),
    gopro_serial       TEXT,  -- gopro_cohn: stable camera identity
    ble_mac            TEXT,  -- gopro_ble: camera BLE address
    wake_payload       BLOB,  -- insta360_ble: per-unit wake advertisement payload
    adb_serial         TEXT,  -- adb: persistent device serial
    notes              TEXT
);
