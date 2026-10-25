# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""capture_control.py — DB access for remote capture control.

Two tables back the capture controller (see
docs/roadmap/features/camera-control/camera-control-design.md §4):

``control_endpoints``
    How a physical camera is controlled (registry table, also present in
    session databases). Cameras without a row are operated manually.

``capture_events``
    Session table with one row per camera per capture: when the start and
    stop commands were sent, executed and confirmed, and how trustworthy the
    confirmation is.

All timestamps are ISO-8601 UTC strings on the controller clock.
"""

from __future__ import annotations

import datetime
import sqlite3
from dataclasses import dataclass
from typing import Final

CONTROL_METHODS: Final[tuple[str, ...]] = (
    "gopro_cohn",
    "gopro_ble",
    "insta360_ble",
    "adb",
    "ble_hid",
    "manual",
)
CONFIDENCE_LEVELS: Final[tuple[str, ...]] = ("confirmed", "commanded", "manual")
EVENT_STATUSES: Final[tuple[str, ...]] = ("ok", "failed", "operator_override")

_EVENT_TS_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "start_cmd_ts",
        "start_ack_ts",
        "start_confirm_ts",
        "stop_cmd_ts",
        "stop_ack_ts",
        "stop_confirm_ts",
    }
)


def utc_now_iso() -> str:
    """Return the current time as an ISO-8601 UTC string with millisecond precision."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class ControlEndpoint:
    """How one camera instance is controlled."""

    camera_instance_id: str
    method: str
    gopro_serial: str | None = None
    ble_mac: str | None = None
    wake_payload: bytes | None = None
    adb_serial: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class ControlCredential:
    """How to reach one camera over COHN on one WLAN (registry only; see registry_private_schema.sql)."""

    camera_instance_id: str
    ssid: str
    ip_address: str
    username: str
    password: str
    certificate: str


@dataclass(frozen=True)
class CaptureEvent:
    """Control-event log for one camera in one capture."""

    capture_id: str
    camera_instance_id: str
    confidence: str
    status: str
    start_cmd_ts: str | None = None
    start_ack_ts: str | None = None
    start_confirm_ts: str | None = None
    stop_cmd_ts: str | None = None
    stop_ack_ts: str | None = None
    stop_confirm_ts: str | None = None
    notes: str | None = None
    #: registry camera_modes.id the camera was armed with; None if no registry mode was used
    camera_mode_id: str | None = None
    #: JSON object of the settings the camera ended up with (or was asked to have)
    settings_json: str | None = None


def set_control_endpoint(conn: sqlite3.Connection, endpoint: ControlEndpoint) -> None:
    """Insert or replace the control endpoint of a camera instance.

    Raises
    ------
    ValueError
        If ``endpoint.method`` is not one of :data:`CONTROL_METHODS`.
    """
    if endpoint.method not in CONTROL_METHODS:
        raise ValueError(f"unknown control method {endpoint.method!r}")
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO control_endpoints "
            "(camera_instance_id, method, gopro_serial, ble_mac, wake_payload, adb_serial, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                endpoint.camera_instance_id,
                endpoint.method,
                endpoint.gopro_serial,
                endpoint.ble_mac,
                endpoint.wake_payload,
                endpoint.adb_serial,
                endpoint.notes,
            ),
        )


def get_control_endpoint(conn: sqlite3.Connection, camera_instance_id: str) -> ControlEndpoint | None:
    """Return the control endpoint of a camera instance, or None if it has none."""
    row = conn.execute(
        "SELECT camera_instance_id, method, gopro_serial, ble_mac, wake_payload, adb_serial, notes "
        "FROM control_endpoints WHERE camera_instance_id = ?",
        (camera_instance_id,),
    ).fetchone()
    return None if row is None else ControlEndpoint(*tuple(row))


def list_control_endpoints(conn: sqlite3.Connection) -> list[ControlEndpoint]:
    """Return all control endpoints ordered by camera instance id."""
    rows = conn.execute(
        "SELECT camera_instance_id, method, gopro_serial, ble_mac, wake_payload, adb_serial, notes "
        "FROM control_endpoints ORDER BY camera_instance_id"
    ).fetchall()
    return [ControlEndpoint(*tuple(r)) for r in rows]


def set_control_credential(conn: sqlite3.Connection, credential: ControlCredential) -> None:
    """Insert or replace the credentials of a camera on one WLAN."""
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO control_credentials "
            "(camera_instance_id, ssid, ip_address, username, password, certificate) VALUES (?, ?, ?, ?, ?, ?)",
            (
                credential.camera_instance_id,
                credential.ssid,
                credential.ip_address,
                credential.username,
                credential.password,
                credential.certificate,
            ),
        )


def list_control_credentials(conn: sqlite3.Connection, camera_instance_id: str) -> list[ControlCredential]:
    """The credentials of a camera, one per WLAN, ordered by SSID."""
    rows = conn.execute(
        "SELECT camera_instance_id, ssid, ip_address, username, password, certificate "
        "FROM control_credentials WHERE camera_instance_id = ? ORDER BY ssid",
        (camera_instance_id,),
    ).fetchall()
    return [ControlCredential(*tuple(r)) for r in rows]


def delete_control_credential(conn: sqlite3.Connection, camera_instance_id: str, ssid: str) -> bool:
    """Remove the credentials of a camera on one WLAN; return whether there were any."""
    with conn:
        cur = conn.execute(
            "DELETE FROM control_credentials WHERE camera_instance_id = ? AND ssid = ?", (camera_instance_id, ssid)
        )
    return cur.rowcount > 0


def upsert_capture_event(conn: sqlite3.Connection, event: CaptureEvent) -> None:
    """Insert a capture event, or update the existing row for the same camera and capture.

    Timestamp fields that are None in *event* leave an existing value untouched,
    so the start and stop phases of a capture can be written separately.

    Raises
    ------
    ValueError
        If ``confidence`` or ``status`` is not an allowed value.
    """
    if event.confidence not in CONFIDENCE_LEVELS:
        raise ValueError(f"unknown confidence {event.confidence!r}")
    if event.status not in EVENT_STATUSES:
        raise ValueError(f"unknown status {event.status!r}")
    with conn:
        conn.execute(
            "INSERT INTO capture_events "
            "(capture_id, camera_instance_id, start_cmd_ts, start_ack_ts, start_confirm_ts, "
            " stop_cmd_ts, stop_ack_ts, stop_confirm_ts, confidence, status, notes, "
            " camera_mode_id, settings_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (capture_id, camera_instance_id) DO UPDATE SET "
            " start_cmd_ts = COALESCE(excluded.start_cmd_ts, start_cmd_ts), "
            " start_ack_ts = COALESCE(excluded.start_ack_ts, start_ack_ts), "
            " start_confirm_ts = COALESCE(excluded.start_confirm_ts, start_confirm_ts), "
            " stop_cmd_ts = COALESCE(excluded.stop_cmd_ts, stop_cmd_ts), "
            " stop_ack_ts = COALESCE(excluded.stop_ack_ts, stop_ack_ts), "
            " stop_confirm_ts = COALESCE(excluded.stop_confirm_ts, stop_confirm_ts), "
            " confidence = excluded.confidence, "
            " status = excluded.status, "
            " notes = COALESCE(excluded.notes, notes), "
            " camera_mode_id = COALESCE(excluded.camera_mode_id, camera_mode_id), "
            " settings_json = COALESCE(excluded.settings_json, settings_json)",
            (
                event.capture_id,
                event.camera_instance_id,
                event.start_cmd_ts,
                event.start_ack_ts,
                event.start_confirm_ts,
                event.stop_cmd_ts,
                event.stop_ack_ts,
                event.stop_confirm_ts,
                event.confidence,
                event.status,
                event.notes,
                event.camera_mode_id,
                event.settings_json,
            ),
        )


def set_event_timestamp(
    conn: sqlite3.Connection,
    capture_id: str,
    camera_instance_id: str,
    field: str,
    ts: str | None = None,
) -> None:
    """Set one timestamp field of an existing capture event (default: now).

    Raises
    ------
    ValueError
        If *field* is not a timestamp column of ``capture_events``.
    KeyError
        If no event exists for the capture and camera.
    """
    if field not in _EVENT_TS_FIELDS:
        raise ValueError(f"not a capture_events timestamp field: {field!r}")
    with conn:
        cur = conn.execute(
            f"UPDATE capture_events SET {field} = ? WHERE capture_id = ? AND camera_instance_id = ?",
            (ts or utc_now_iso(), capture_id, camera_instance_id),
        )
    if cur.rowcount == 0:
        raise KeyError((capture_id, camera_instance_id))


def get_capture_events(conn: sqlite3.Connection, capture_id: str) -> list[CaptureEvent]:
    """Return the control events of a capture, ordered by camera instance id."""
    rows = conn.execute(
        "SELECT capture_id, camera_instance_id, confidence, status, start_cmd_ts, start_ack_ts, "
        "start_confirm_ts, stop_cmd_ts, stop_ack_ts, stop_confirm_ts, notes, camera_mode_id, settings_json "
        "FROM capture_events WHERE capture_id = ? ORDER BY camera_instance_id",
        (capture_id,),
    ).fetchall()
    return [CaptureEvent(*tuple(r)) for r in rows]
