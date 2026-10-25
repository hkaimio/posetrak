# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Assembling the controller for the configured cameras."""

from __future__ import annotations

import dataclasses
import sqlite3
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from pipeline.capture.config import CameraConfig, CaptureConfig, ConfigError
from pipeline.capture.credentials import CredentialStore, RegistryCredentialStore
from pipeline.capture.devices import CameraDriver, ManualDriver
from pipeline.capture.modes import ModePlanner, PlannedCamera, load_modes, model_keys
from pipeline.capture.registry_link import LinkError, find_camera
from pipeline.capture.service import CameraSlot, CaptureController
from pipeline.capture.simulated import SimulatedDriver
from posetrak.db.db import (
    DEFAULT_REGISTRY_PATH,
    create_mocap_session,
    create_session,
    open_registry,
    open_session,
)


def build_driver(
    cam: CameraConfig,
    cfg: CaptureConfig | None = None,
    store: CredentialStore | RegistryCredentialStore | None = None,
) -> CameraDriver:
    """Create the driver a camera's configuration asks for.

    A COHN camera gets one endpoint per WLAN it was provisioned on. They are
    ordered like the ``[[network]]`` blocks (networks not in the configuration
    last), and a camera with a reserved address on a network is reached there
    rather than at the address recorded when it was provisioned.
    """
    if cam.method == "manual":
        return ManualDriver(cam.label)
    if cam.method == "gopro_cohn":
        from pipeline.capture.gopro_cohn import CohnEndpoint, GoProCohnDriver

        cfg = cfg or CaptureConfig()
        store = store or CredentialStore(cfg.daemon.credentials_dir)
        known = [n.ssid for n in cfg.networks]
        creds = sorted(
            store.load(cam.gopro_serial or ""),
            key=lambda c: known.index(c.ssid) if c.ssid in known else len(known),
        )
        if not creds:
            raise ConfigError(
                f"camera {cam.label!r} (serial {cam.gopro_serial}) has no COHN credentials; "
                "run `posetrak-capture provision` for it"
            )
        endpoints = []
        for c in creds:
            net = cfg.network_for_ssid(c.ssid)
            endpoints.append(
                CohnEndpoint.from_credentials(c, net.name if net else c.ssid, net.addresses.get(cam.label) if net else None)
            )
        return GoProCohnDriver(cam.label, endpoints)
    raise ConfigError(f"camera {cam.label!r}: the {cam.method} driver is not available")


def resolve_registry_cameras(registry: sqlite3.Connection, cameras: Sequence[CameraConfig]) -> list[tuple[CameraConfig, str]]:
    """Complete the configured cameras from the registry; returns each with its ``camera_instances.id``.

    The control method and the COHN device serial come from the camera's
    ``control_endpoints`` row unless the configuration states them (an explicit
    value that contradicts the registry is an error). A camera without an
    endpoint is operated manually.
    """
    resolved = []
    for cam in cameras:
        try:
            reg = find_camera(registry, cam.label)
        except LinkError as exc:
            raise ConfigError(str(exc)) from exc
        method = cam.method or (reg.endpoint.method if reg.endpoint else "manual")
        serial = cam.gopro_serial
        if method == "gopro_cohn":
            if serial and reg.gopro_serial and serial != reg.gopro_serial:
                raise ConfigError(
                    f"camera {cam.label!r}: the configuration says camera {serial}, the registry says {reg.gopro_serial}"
                )
            serial = serial or reg.gopro_serial
            if not serial:
                raise ConfigError(f"camera {cam.label!r} is not linked to a Hero13; run `posetrak-capture provision {cam.label}`")
        resolved.append((dataclasses.replace(cam, method=method, gopro_serial=serial), reg.instance_id))
    return resolved


def build_planner(
    registry: sqlite3.Connection, slots: Sequence[CameraSlot], cfg: CaptureConfig
) -> ModePlanner:
    """The mode planner for these cameras: their models' registry modes and the configured setups."""
    cameras = []
    for slot in slots:
        row = registry.execute(
            "SELECT m.id, m.manufacturer, m.model_name FROM camera_instances i "
            "JOIN camera_models m ON m.id = i.camera_model_id WHERE i.id = ?",
            (slot.camera_instance_id,),
        ).fetchone()
        if row is None:
            raise ConfigError(f"camera {slot.label!r} has no camera model in the registry")
        model = f"{row[1] or ''} {row[2] or ''}".strip() or "unknown model"
        cameras.append(
            PlannedCamera(slot.label, model_keys(row[1], row[2]), model, load_modes(registry, row[0]), slot.driver)
        )
    _check_setups(cfg, cameras)
    return ModePlanner(cameras, cfg.setups, cfg.daemon.default_setup, cfg.profile.settings)


def _check_setups(cfg: CaptureConfig, cameras: list[PlannedCamera]) -> None:
    """Refuse setups that mention cameras that are not configured."""
    labels = {c.label for c in cameras}
    for setup in cfg.setups:
        for label in setup.cameras:
            if label not in labels:
                raise ConfigError(f"setup {setup.name!r} names camera {label!r}, which has no [[camera]] block")


def demo_registrar(session, registry, session_id, capture_id, camera_instance_id, path, *, expected_mode_id=None, **_):  # type: ignore[no-untyped-def]
    """Stands in for registering a copied video when the cameras are simulated."""
    from pipeline.capture.register import RegisteredVideo, VideoStatus

    names = {"demo-4k": "4K 120 linear", "demo-2k": "2.7K 240 linear", "demo-hd": "1080 60"}
    return RegisteredVideo("demo", path, expected_mode_id, names.get(expected_mode_id), VideoStatus.OK, "")


def demo_planner(slots: Sequence[CameraSlot]) -> ModePlanner:
    """Modes and setups for simulated cameras; the last of three or more is set by hand."""
    from pipeline.capture.config import SetupConfig
    from pipeline.capture.devices import ModeSpec

    modes = (
        ModeSpec("demo-4k", "4K 120 linear", 3840, 2160, 120, {"resolution": "4K", "fps": 120, "lens": "linear"}),
        ModeSpec("demo-2k", "2.7K 240 linear", 2704, 1520, 240, {"resolution": "2.7K", "fps": 240, "lens": "linear"}),
        ModeSpec("demo-hd", "1080 60", 1920, 1080, 60, {"resolution": "1080", "fps": 60}),
    )
    if len(slots) >= 3:
        slots[-1].driver.settings_control = "manual"
    cameras = [PlannedCamera(s.label, model_keys("", "Demo camera"), "Demo camera", modes, s.driver) for s in slots]
    setups = (
        SetupConfig("default", {"Demo camera": "4K 120 linear"}),
        SetupConfig("slow motion", {"Demo camera": "2.7K 240 linear"}),
    )
    return ModePlanner(cameras, setups, "default")


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
        planner: ModePlanner | None = None
        if demo_cameras:
            session = create_session(Path(tempfile.mkdtemp(prefix="posetrak-capture-demo-")) / "demo.db")
            sid = create_mocap_session(session, location="demo")
            slots = [
                CameraSlot(f"demo-{i + 1:02d}", f"demo-instance-{i + 1:02d}", SimulatedDriver(f"demo-{i + 1:02d}"))
                for i in range(demo_cameras)
            ]
            planner = demo_planner(slots)
        else:
            if cfg.daemon.session_db is None:
                raise ConfigError("daemon.session_db is not set in the configuration")
            if not cfg.cameras:
                raise ConfigError("no [[camera]] entries in the configuration")
            registry = open_registry(cfg.daemon.registry_db or DEFAULT_REGISTRY_PATH)
            try:
                resolved = resolve_registry_cameras(registry, cfg.cameras)
                store = RegistryCredentialStore(registry, CredentialStore(cfg.daemon.credentials_dir))
                slots = [CameraSlot(c.label, instance_id, build_driver(c, cfg, store)) for c, instance_id in resolved]
                planner = build_planner(registry, slots, cfg)
            finally:
                registry.close()
            session = open_session(cfg.daemon.session_db)
            sid = latest_or_new_session(session)
        offload_dir = None
        registrar = None
        if demo_cameras:
            offload_dir = Path(tempfile.mkdtemp(prefix="posetrak-capture-demo-videos-"))
            registrar = demo_registrar
        else:
            offload_dir = cfg.daemon.offload_dir or (cfg.daemon.session_db.parent / "videos" if cfg.daemon.session_db else None)
        return CaptureController(
            slots,
            session,
            sid,
            settings=cfg.profile.settings,
            planner=planner,
            offload_dir=offload_dir,
            registry_path=cfg.daemon.registry_db,
            auto_offload=cfg.daemon.auto_offload,
            offload_registrar=registrar,
        )

    return factory
