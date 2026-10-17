# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Assembling the controller for the configured cameras."""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Callable
from pathlib import Path

from pipeline.capture.config import CameraConfig, CaptureConfig, ConfigError
from pipeline.capture.devices import CameraDriver, ManualDriver
from pipeline.capture.service import CameraSlot, CaptureController
from pipeline.capture.simulated import SimulatedDriver
from posetrak.db.db import (
    DEFAULT_REGISTRY_PATH,
    create_mocap_session,
    create_session,
    open_registry,
    open_session,
)


def build_driver(cam: CameraConfig) -> CameraDriver:
    if cam.method == "manual":
        return ManualDriver(cam.label)
    raise ConfigError(f"camera {cam.label!r}: the {cam.method} driver is not available yet")


def resolve_camera_instances(registry: sqlite3.Connection, labels: list[str]) -> dict[str, str]:
    """Map camera labels to ``camera_instances.id``; every label must be registered."""
    ids: dict[str, str] = {}
    for label in labels:
        row = registry.execute("SELECT id FROM camera_instances WHERE label = ?", (label,)).fetchone()
        if row is None:
            raise ConfigError(f"camera {label!r} is not in the camera registry")
        ids[label] = row[0]
    return ids


def latest_or_new_session(session: sqlite3.Connection) -> str:
    row = session.execute("SELECT id FROM mocap_sessions ORDER BY recorded_at DESC LIMIT 1").fetchone()
    return row[0] if row else create_mocap_session(session, location="capture daemon")


def controller_factory(cfg: CaptureConfig, *, demo_cameras: int = 0) -> Callable[[], CaptureController]:
    """Return a function that builds the controller on the server's event loop thread."""

    def factory() -> CaptureController:
        if demo_cameras:
            session = create_session(Path(tempfile.mkdtemp(prefix="posetrak-capture-demo-")) / "demo.db")
            sid = create_mocap_session(session, location="demo")
            slots = [
                CameraSlot(f"demo-{i + 1:02d}", f"demo-instance-{i + 1:02d}", SimulatedDriver(f"demo-{i + 1:02d}"))
                for i in range(demo_cameras)
            ]
        else:
            if cfg.daemon.session_db is None:
                raise ConfigError("daemon.session_db is not set in the configuration")
            if not cfg.cameras:
                raise ConfigError("no [[camera]] entries in the configuration")
            registry = open_registry(cfg.daemon.registry_db or DEFAULT_REGISTRY_PATH)
            try:
                ids = resolve_camera_instances(registry, [c.label for c in cfg.cameras])
            finally:
                registry.close()
            session = open_session(cfg.daemon.session_db)
            sid = latest_or_new_session(session)
            slots = [CameraSlot(c.label, ids[c.label], build_driver(c)) for c in cfg.cameras]
        return CaptureController(slots, session, sid, settings=cfg.profile.settings)

    return factory
