-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- registry_private_schema.sql
-- Tables that exist only in the registry database and are never part of a session database.
--
-- A session database embeds registry_schema.sql so that it stays usable on its own, and
-- sessions are archived and shared with the captures. Nothing in this file may be copied
-- into a session or exported with one.

-- COHN credentials of the cameras: how to reach a Hero13 on a WLAN. The password gives full
-- control of the camera, so this table is private to the registry. Written by
-- `posetrak-capture provision`; read by the capture daemon.
CREATE TABLE IF NOT EXISTS control_credentials (
    camera_instance_id TEXT NOT NULL REFERENCES camera_instances(id), -- the camera the credentials open
    ssid               TEXT NOT NULL,  -- WLAN the credentials were created on; a camera has one entry per WLAN
    ip_address         TEXT NOT NULL,  -- address the camera got on that WLAN when it was provisioned
    username           TEXT NOT NULL,  -- HTTP basic authentication user name
    password           TEXT NOT NULL,  -- HTTP basic authentication password; grants full control of the camera
    certificate        TEXT NOT NULL,  -- PEM certificate the camera presents; connections are pinned to it
    PRIMARY KEY (camera_instance_id, ssid)
);
