# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests of the daemon assembly helpers: addresses, QR code, camera resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("qrcode")

from fastapi.testclient import TestClient  # noqa: E402

from pipeline.capture.api import create_app  # noqa: E402
from pipeline.capture.config import CameraConfig, ConfigError, parse_config  # noqa: E402
from pipeline.capture.credentials import CohnCredentials, CredentialStore  # noqa: E402
from pipeline.capture.daemon import (  # noqa: E402
    build_driver,
    controller_factory,
    latest_or_new_session,
    resolve_camera_instances,
)
from pipeline.capture.devices import ManualDriver  # noqa: E402
from pipeline.capture.network import access_url, lan_addresses, qr_text  # noqa: E402
from posetrak.db.db import (  # noqa: E402
    create_camera_instance,
    create_camera_model,
    create_mocap_session,
    create_registry,
    create_session,
)


def test_lan_addresses_exclude_loopback_and_link_local() -> None:
    for addr in lan_addresses():
        assert not addr.startswith(("127.", "169.254.", "0."))
    assert len(set(lan_addresses())) == len(lan_addresses())


def test_access_url_keeps_token_in_the_fragment() -> None:
    url = access_url("192.168.8.2", 8765, "tok")
    assert url == "http://192.168.8.2:8765/#token=tok"


def test_qr_text_variants() -> None:
    url = access_url("192.168.8.2", 8765, "abc123")
    blocks, plain = qr_text(url), qr_text(url, ascii_only=True)
    assert len(blocks.splitlines()) * 2 >= len(plain.splitlines()) - 1
    assert set(plain) <= {" ", "#", "\n"}
    plain.encode("ascii")


def test_build_driver() -> None:
    assert isinstance(build_driver(CameraConfig("a", "manual")), ManualDriver)
    with pytest.raises(ConfigError, match="no COHN credentials"):
        build_driver(CameraConfig("a", "gopro_cohn", "1234"), parse_config({}), CredentialStore(Path("nowhere")))


def test_build_cohn_driver_orders_networks_and_applies_reserved_addresses(tmp_path: Path) -> None:
    from pipeline.capture.gopro_cohn import GoProCohnDriver

    store = CredentialStore(tmp_path)
    store.save("1234", CohnCredentials("HomeNet", "192.168.1.50", "gopro", "p1", "C1"))
    store.save("1234", CohnCredentials("site-ap", "192.168.8.77", "gopro", "p2", "C2"))
    store.save("1234", CohnCredentials("other", "10.0.0.5", "gopro", "p3", "C3"))
    cfg = parse_config(
        {
            "network": [
                {"name": "location", "ssid": "site-ap", "addresses": {"cam-1": "192.168.8.101"}},
                {"name": "home", "ssid": "HomeNet"},
            ]
        }
    )
    driver = build_driver(CameraConfig("cam-1", "gopro_cohn", "1234"), cfg, store)
    assert isinstance(driver, GoProCohnDriver)
    endpoints = driver._endpoints
    assert [(e.network, e.base_url) for e in endpoints] == [
        ("location", "https://192.168.8.101"),
        ("home", "https://192.168.1.50"),
        ("other", "https://10.0.0.5"),
    ]


def test_resolve_camera_instances(tmp_path: Path) -> None:
    reg = create_registry(tmp_path / "reg.db")
    model = create_camera_model(reg, manufacturer="GoPro", model_name="Hero13")
    inst = create_camera_instance(reg, model, label="gopro-13_01")
    assert resolve_camera_instances(reg, ["gopro-13_01"]) == {"gopro-13_01": inst}
    with pytest.raises(ConfigError, match="not in the camera registry"):
        resolve_camera_instances(reg, ["missing"])
    reg.close()


def test_latest_or_new_session(tmp_path: Path) -> None:
    session = create_session(tmp_path / "s.db")
    created = latest_or_new_session(session)
    assert latest_or_new_session(session) == created
    other = create_mocap_session(session, location="later", recorded_at="2999-01-01T00:00:00")
    assert latest_or_new_session(session) == other
    session.close()


def test_configured_daemon_needs_database_and_cameras() -> None:
    with pytest.raises(ConfigError, match="session_db"):
        controller_factory(parse_config({}))()
    with pytest.raises(ConfigError, match=r"no \[\[camera\]\]"):
        controller_factory(parse_config({"daemon": {"session_db": "x.db"}}))()


def test_demo_mode_serves_simulated_cameras() -> None:
    cfg = parse_config({"daemon": {"token": "t"}})
    app = create_app(controller_factory(cfg, demo_cameras=2), "t", web_dir=None)
    headers = {"Authorization": "Bearer t", "X-Client-Id": "c"}
    with TestClient(app) as client:
        state = client.get("/api/state", headers=headers).json()
        assert sorted(state["cameras"]) == ["demo-01", "demo-02"]
        client.post("/api/lease", json={}, headers=headers)
        assert client.post("/api/capture/arm", headers=headers).json()["ready"]
