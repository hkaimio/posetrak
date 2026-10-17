# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for posetrak.db.capture_control and its schema migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from posetrak.db.capture_control import (
    CaptureEvent,
    ControlEndpoint,
    get_capture_events,
    get_control_endpoint,
    list_control_endpoints,
    set_control_endpoint,
    set_event_timestamp,
    upsert_capture_event,
    utc_now_iso,
)
from posetrak.db.db import (
    REGISTRY_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION,
    create_camera_instance,
    create_camera_model,
    create_capture,
    create_mocap_session,
    create_registry,
    create_session,
    get_schema_version,
    open_registry,
    open_session,
)


@pytest.fixture()
def camera_id(registry_db: sqlite3.Connection) -> str:
    model = create_camera_model(registry_db, manufacturer="GoPro", model_name="Hero13 Black")
    return create_camera_instance(registry_db, model, label="gopro-13_01", serial_number="C3501")


@pytest.fixture()
def capture_id(session_db: sqlite3.Connection) -> str:
    session_id = create_mocap_session(session_db, location="test")
    return create_capture(session_db, session_id)


def test_control_endpoint_roundtrip(registry_db: sqlite3.Connection, camera_id: str) -> None:
    assert get_control_endpoint(registry_db, camera_id) is None
    ep = ControlEndpoint(camera_id, "gopro_cohn", gopro_serial="1234")
    set_control_endpoint(registry_db, ep)
    assert get_control_endpoint(registry_db, camera_id) == ep
    assert list_control_endpoints(registry_db) == [ep]


def test_control_endpoint_replace(registry_db: sqlite3.Connection, camera_id: str) -> None:
    set_control_endpoint(registry_db, ControlEndpoint(camera_id, "manual"))
    set_control_endpoint(registry_db, ControlEndpoint(camera_id, "adb", adb_serial="abc"))
    got = get_control_endpoint(registry_db, camera_id)
    assert got is not None and got.method == "adb" and got.adb_serial == "abc"
    assert len(list_control_endpoints(registry_db)) == 1


def test_control_endpoint_rejects_unknown_method(registry_db: sqlite3.Connection, camera_id: str) -> None:
    with pytest.raises(ValueError):
        set_control_endpoint(registry_db, ControlEndpoint(camera_id, "carrier_pigeon"))


def test_control_endpoint_requires_camera_instance(registry_db: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        set_control_endpoint(registry_db, ControlEndpoint("no-such-camera", "manual"))


def test_capture_event_two_phase_upsert(session_db: sqlite3.Connection, capture_id: str) -> None:
    """Start and stop phases are written separately and do not clobber each other."""
    upsert_capture_event(
        session_db,
        CaptureEvent(
            capture_id,
            "cam-a",
            "confirmed",
            "ok",
            start_cmd_ts="2026-10-17T10:00:00.000+00:00",
            start_ack_ts="2026-10-17T10:00:00.120+00:00",
            start_confirm_ts="2026-10-17T10:00:00.480+00:00",
        ),
    )
    upsert_capture_event(
        session_db,
        CaptureEvent(
            capture_id,
            "cam-a",
            "confirmed",
            "ok",
            stop_cmd_ts="2026-10-17T10:05:00.000+00:00",
            stop_confirm_ts="2026-10-17T10:05:00.300+00:00",
        ),
    )
    (ev,) = get_capture_events(session_db, capture_id)
    assert ev.start_cmd_ts == "2026-10-17T10:00:00.000+00:00"
    assert ev.start_confirm_ts == "2026-10-17T10:00:00.480+00:00"
    assert ev.stop_confirm_ts == "2026-10-17T10:05:00.300+00:00"
    assert ev.stop_ack_ts is None


def test_capture_event_status_can_change(session_db: sqlite3.Connection, capture_id: str) -> None:
    upsert_capture_event(session_db, CaptureEvent(capture_id, "cam-a", "confirmed", "failed", notes="timeout"))
    upsert_capture_event(session_db, CaptureEvent(capture_id, "cam-a", "manual", "operator_override"))
    (ev,) = get_capture_events(session_db, capture_id)
    assert (ev.confidence, ev.status, ev.notes) == ("manual", "operator_override", "timeout")


def test_capture_events_are_per_camera_and_ordered(session_db: sqlite3.Connection, capture_id: str) -> None:
    for cam in ("cam-b", "cam-a"):
        upsert_capture_event(session_db, CaptureEvent(capture_id, cam, "commanded", "ok"))
    assert [e.camera_instance_id for e in get_capture_events(session_db, capture_id)] == ["cam-a", "cam-b"]


def test_capture_event_validation(session_db: sqlite3.Connection, capture_id: str) -> None:
    with pytest.raises(ValueError):
        upsert_capture_event(session_db, CaptureEvent(capture_id, "cam-a", "sure", "ok"))
    with pytest.raises(ValueError):
        upsert_capture_event(session_db, CaptureEvent(capture_id, "cam-a", "confirmed", "great"))


def test_capture_event_requires_capture(session_db: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        upsert_capture_event(session_db, CaptureEvent("no-such-capture", "cam-a", "confirmed", "ok"))


def test_set_event_timestamp(session_db: sqlite3.Connection, capture_id: str) -> None:
    upsert_capture_event(session_db, CaptureEvent(capture_id, "cam-a", "confirmed", "ok"))
    set_event_timestamp(session_db, capture_id, "cam-a", "start_confirm_ts", "2026-10-17T10:00:01.000+00:00")
    set_event_timestamp(session_db, capture_id, "cam-a", "stop_cmd_ts")
    (ev,) = get_capture_events(session_db, capture_id)
    assert ev.start_confirm_ts == "2026-10-17T10:00:01.000+00:00"
    assert ev.stop_cmd_ts is not None and ev.stop_cmd_ts.endswith("+00:00")


def test_set_event_timestamp_errors(session_db: sqlite3.Connection, capture_id: str) -> None:
    with pytest.raises(ValueError):
        set_event_timestamp(session_db, capture_id, "cam-a", "notes")
    with pytest.raises(KeyError):
        set_event_timestamp(session_db, capture_id, "cam-a", "start_cmd_ts")


def test_utc_now_iso_is_utc_with_milliseconds() -> None:
    ts = utc_now_iso()
    assert ts.endswith("+00:00") and len(ts.split(".")[1]) == len("123+00:00")


def test_registry_migration_v14_to_v15_adds_control_endpoints(tmp_path: Path) -> None:
    path = tmp_path / "registry.db"
    conn = create_registry(path)
    conn.execute("DROP TABLE control_endpoints")
    conn.execute("PRAGMA user_version = 14")
    conn.commit()
    conn.close()

    conn = open_registry(path)
    assert get_schema_version(conn) == REGISTRY_SCHEMA_VERSION
    assert conn.execute("SELECT COUNT(*) FROM control_endpoints").fetchone()[0] == 0
    conn.close()


def test_session_migration_v55_to_v56_adds_capture_tables(tmp_path: Path) -> None:
    path = tmp_path / "session.db"
    conn = create_session(path)
    conn.execute("DROP TABLE capture_events")
    conn.execute("DROP TABLE control_endpoints")
    conn.execute("PRAGMA user_version = 55")
    conn.commit()
    conn.close()

    conn = open_session(path)
    assert get_schema_version(conn) == SESSION_SCHEMA_VERSION
    assert conn.execute("SELECT COUNT(*) FROM capture_events").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM control_endpoints").fetchone()[0] == 0
    conn.close()
