# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests of linking registry cameras to physical cameras, and the provision command."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from pipeline.capture import gopro_ble
from pipeline.capture.cli import main
from pipeline.capture.config import CameraConfig, ConfigError, parse_config
from pipeline.capture.credentials import CohnCredentials, CredentialStore, RegistryCredentialStore
from pipeline.capture.daemon import resolve_registry_cameras
from pipeline.capture.registry_link import (
    LinkError,
    SerialCheck,
    cameras_by_gopro_serial,
    check_link,
    find_camera,
    link_gopro,
    reconcile_serial,
)
from posetrak.db.capture_control import ControlEndpoint, get_control_endpoint, set_control_endpoint
from posetrak.db.db import (
    create_camera_instance,
    create_camera_model,
    create_registry,
    create_session,
    list_camera_models,
    open_registry,
)

FULL_SERIAL = "C3501324514609"


@pytest.fixture
def registry(tmp_path: Path):
    reg = create_registry(tmp_path / "registry.db")
    create_camera_model(reg, manufacturer="GoPro", model_name="Hero13")
    yield reg
    reg.close()


def add_camera(reg, label: str, serial: str | None = None) -> str:
    return create_camera_instance(reg, list_camera_models(reg)[0]["id"], label=label, serial_number=serial)


# -- registry link ----------------------------------------------------------------------------------


def test_find_camera_reports_unknown_labels_and_lists_the_known_ones(registry) -> None:
    add_camera(registry, "gopro13_01")
    with pytest.raises(LinkError, match=r"'nope'.*registered: gopro13_01"):
        find_camera(registry, "nope")
    assert find_camera(registry, "gopro13_01").endpoint is None


def test_link_creates_the_endpoint_and_keeps_its_notes(registry) -> None:
    instance = add_camera(registry, "gopro13_01")
    set_control_endpoint(registry, ControlEndpoint(instance, "gopro_cohn", notes="roof rig"))
    link_gopro(registry, find_camera(registry, "gopro13_01"), "4609")
    endpoint = get_control_endpoint(registry, instance)
    assert (endpoint.method, endpoint.gopro_serial, endpoint.notes) == ("gopro_cohn", "4609", "roof rig")
    assert cameras_by_gopro_serial(registry) == {"4609": "gopro13_01"}
    # linking again to the same camera is harmless
    link_gopro(registry, find_camera(registry, "gopro13_01"), "4609")


def test_link_refuses_contradictions(registry) -> None:
    add_camera(registry, "gopro13_01")
    add_camera(registry, "gopro13_02")
    add_camera(registry, "gopro13_03", serial=FULL_SERIAL.replace("4609", "9999"))
    add_camera(registry, "phone", serial=None)
    link_gopro(registry, find_camera(registry, "gopro13_01"), "4609")

    with pytest.raises(LinkError, match="already linked to camera 4609, not 4243"):
        check_link(registry, find_camera(registry, "gopro13_01"), "4243")
    with pytest.raises(LinkError, match="already linked to gopro13_01"):
        check_link(registry, find_camera(registry, "gopro13_02"), "4609")
    with pytest.raises(LinkError, match="does not end in 4243"):
        check_link(registry, find_camera(registry, "gopro13_03"), "4243")
    set_control_endpoint(registry, ControlEndpoint(find_camera(registry, "phone").instance_id, "adb", adb_serial="x"))
    with pytest.raises(LinkError, match="controlled by method 'adb'"):
        check_link(registry, find_camera(registry, "phone"), "4243")


def test_reconcile_serial_fills_an_empty_field_and_never_overwrites(registry) -> None:
    add_camera(registry, "gopro13_01")
    add_camera(registry, "gopro13_02", serial="C0000000000000")

    camera = find_camera(registry, "gopro13_01")
    assert reconcile_serial(registry, camera, FULL_SERIAL, fill=False) is SerialCheck.FILLED
    assert find_camera(registry, "gopro13_01").serial_number is None
    assert reconcile_serial(registry, camera, FULL_SERIAL) is SerialCheck.FILLED
    assert find_camera(registry, "gopro13_01").serial_number == FULL_SERIAL
    assert reconcile_serial(registry, find_camera(registry, "gopro13_01"), FULL_SERIAL) is SerialCheck.SAME

    other = find_camera(registry, "gopro13_02")
    assert reconcile_serial(registry, other, FULL_SERIAL) is SerialCheck.DIFFERS
    assert find_camera(registry, "gopro13_02").serial_number == "C0000000000000"


# -- provision command ------------------------------------------------------------------------------


class StoredCredentials:
    """What the registry holds, read through a fresh connection after a command has run."""

    def __init__(self, registry_path: Path) -> None:
        self._path = registry_path

    def load(self, serial: str):
        reg = open_registry(self._path)
        try:
            return RegistryCredentialStore(reg).load(serial)
        finally:
            reg.close()


@pytest.fixture
def cli_env(tmp_path: Path, registry, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text(
        f'[daemon]\ntoken = "t"\nregistry_db = "{(tmp_path / "registry.db").as_posix()}"\n'
        f'credentials_dir = "{(tmp_path / "creds").as_posix()}"\n'
        '[[network]]\nname = "location"\nssid = "site-ap"\n',
        encoding="utf-8",
    )
    seen: dict = {}
    reported = {"serial": FULL_SERIAL}

    async def fake_provision(serial, ssid, password, *, force=False, state_dir=None):
        seen.update(serial=serial, ssid=ssid, password=password, force=force)
        return gopro_ble.ProvisionResult(CohnCredentials(ssid, "192.168.8.60", "gopro", "pw", "CERT"), reported["serial"])

    monkeypatch.setattr(gopro_ble, "provision", fake_provision)

    def invoke(*args: str, input: str | None = None):
        return CliRunner().invoke(main, ["--config", str(config), *args], input=input)

    return invoke, seen, reported, StoredCredentials(tmp_path / "registry.db")


def test_provision_links_the_camera_stores_credentials_and_fills_the_serial(registry, cli_env) -> None:
    invoke, seen, _, store = cli_env
    add_camera(registry, "gopro13_01")
    result = invoke("provision", "gopro13_01", "--serial", "4609", "--network", "location", "--password", "wlan-pw", input="y\n")
    assert result.exit_code == 0, result.output
    assert seen == {"serial": "4609", "ssid": "site-ap", "password": "wlan-pw", "force": False}
    assert [c.ssid for c in store.load("4609")] == ["site-ap"]
    camera = find_camera(registry, "gopro13_01")
    assert camera.gopro_serial == "4609" and camera.serial_number == FULL_SERIAL
    assert "192.168.8.60" in result.output


def test_provision_leaves_the_registry_serial_empty_when_declined(registry, cli_env) -> None:
    invoke, _, _, store = cli_env
    add_camera(registry, "gopro13_01")
    result = invoke("provision", "gopro13_01", "--serial", "4609", "--ssid", "site-ap", "--password", "x", input="n\n")
    assert result.exit_code == 0, result.output
    camera = find_camera(registry, "gopro13_01")
    assert camera.gopro_serial == "4609" and camera.serial_number is None
    assert store.load("4609")


def test_provision_of_a_linked_camera_needs_no_serial(registry, cli_env) -> None:
    invoke, seen, _, _ = cli_env
    add_camera(registry, "gopro13_01", serial=FULL_SERIAL)
    link_gopro(registry, find_camera(registry, "gopro13_01"), "4609")
    result = invoke("provision", "gopro13_01", "--ssid", "site-ap", "--password", "x")
    assert result.exit_code == 0, result.output
    assert seen["serial"] == "4609"
    contradicting = invoke("provision", "gopro13_01", "--serial", "4243", "--ssid", "site-ap", "--password", "x")
    assert contradicting.exit_code != 0 and "linked to camera 4609" in contradicting.output


def test_provision_picks_an_unlinked_camera_from_the_scan(registry, cli_env, monkeypatch) -> None:
    invoke, seen, _, _ = cli_env
    add_camera(registry, "gopro13_01")
    add_camera(registry, "gopro13_02")
    link_gopro(registry, find_camera(registry, "gopro13_02"), "4243")

    async def fake_scan(timeout_s=6.0):
        return [gopro_ble.NearbyCamera("4243", "AA", -50), gopro_ble.NearbyCamera("4609", "BB", -60)]

    monkeypatch.setattr(gopro_ble, "scan_cameras", fake_scan)
    result = invoke("provision", "gopro13_01", "--ssid", "site-ap", "--password", "x", input="2\ny\n")
    assert result.exit_code == 0, result.output
    assert "(already gopro13_02)" in result.output
    assert seen["serial"] == "4609"
    assert find_camera(registry, "gopro13_01").gopro_serial == "4609"


def test_provision_stores_nothing_when_the_camera_is_a_different_one(registry, cli_env) -> None:
    invoke, _, _, store = cli_env
    add_camera(registry, "gopro13_01", serial="C0000000000000")
    result = invoke("provision", "gopro13_01", "--serial", "4609", "--ssid", "site-ap", "--password", "x")
    assert result.exit_code != 0 and "does not end in 4609" in result.output
    assert store.load("4609") == []

    add_camera(registry, "gopro13_02", serial="C3509999994609")
    result = invoke("provision", "gopro13_02", "--serial", "4609", "--ssid", "site-ap", "--password", "x")
    assert result.exit_code != 0 and "differs" in result.output
    assert store.load("4609") == [] and find_camera(registry, "gopro13_02").gopro_serial is None


def test_provision_reports_an_unknown_label_and_a_missing_network(registry, cli_env) -> None:
    invoke, _, _, _ = cli_env
    add_camera(registry, "gopro13_01")
    unknown = invoke("provision", "nope", "--serial", "4609", "--ssid", "x", "--password", "x")
    assert unknown.exit_code != 0 and "not in the camera registry" in unknown.output
    no_network = invoke("provision", "gopro13_01", "--serial", "4609", "--password", "x")
    assert no_network.exit_code != 0 and "--network or --ssid" in no_network.output
    bad_network = invoke("provision", "gopro13_01", "--serial", "4609", "--network", "nope", "--password", "x")
    assert bad_network.exit_code != 0 and "unknown network" in bad_network.output


# -- the daemon reads the link ----------------------------------------------------------------------


def test_camera_blocks_need_only_a_label() -> None:
    cfg = parse_config({"camera": [{"label": "gopro13_01"}]})
    assert cfg.cameras == (CameraConfig("gopro13_01"),)
    with pytest.raises(ConfigError, match="unknown method"):
        parse_config({"camera": [{"label": "a", "method": "carrier-pigeon"}]})


def test_daemon_takes_method_and_device_from_the_registry(registry) -> None:
    add_camera(registry, "gopro13_01")
    link_gopro(registry, find_camera(registry, "gopro13_01"), "4609")
    add_camera(registry, "old-cam")  # no endpoint
    cams = [CameraConfig("gopro13_01"), CameraConfig("old-cam")]
    (a, a_id), (b, b_id) = resolve_registry_cameras(registry, cams)
    assert (a.method, a.gopro_serial) == ("gopro_cohn", "4609")
    assert (b.method, b.gopro_serial) == ("manual", None)
    assert a_id == find_camera(registry, "gopro13_01").instance_id and b_id != a_id


def test_daemon_refuses_cameras_that_are_not_linked_or_contradict_the_registry(registry) -> None:
    add_camera(registry, "gopro13_01")
    with pytest.raises(ConfigError, match="run `posetrak-capture provision gopro13_01`"):
        resolve_registry_cameras(registry, [CameraConfig("gopro13_01", "gopro_cohn")])
    link_gopro(registry, find_camera(registry, "gopro13_01"), "4609")
    with pytest.raises(ConfigError, match="configuration says camera 4243, the registry says 4609"):
        resolve_registry_cameras(registry, [CameraConfig("gopro13_01", "gopro_cohn", "4243")])
    with pytest.raises(ConfigError, match="not in the camera registry"):
        resolve_registry_cameras(registry, [CameraConfig("missing")])


# -- reading the serial over Bluetooth --------------------------------------------------------------


def test_serial_is_read_from_the_hardware_info_response() -> None:
    import asyncio
    from types import SimpleNamespace

    def camera(response):
        async def get_hardware_info():
            if isinstance(response, Exception):
                raise response
            return response

        return SimpleNamespace(ble_command=SimpleNamespace(get_hardware_info=get_hardware_info))

    info = SimpleNamespace(data=SimpleNamespace(serial_number=f" {FULL_SERIAL} "))
    assert asyncio.run(gopro_ble._read_serial(camera(info))) == FULL_SERIAL
    assert asyncio.run(gopro_ble._read_serial(camera(RuntimeError("no answer")))) is None


# -- credentials in the registry --------------------------------------------------------------------


CREDS = CohnCredentials("site-ap", "192.168.8.60", "gopro", "pw", "CERT")


def test_credentials_are_kept_per_camera_and_wlan(registry) -> None:
    add_camera(registry, "gopro13_01")
    link_gopro(registry, find_camera(registry, "gopro13_01"), "4609")
    store = RegistryCredentialStore(registry)
    assert store.load("4609") == []
    store.save("4609", CREDS)
    store.save("4609", CohnCredentials("home", "10.0.0.5", "gopro", "pw2", "CERT2"))
    store.save("4609", CohnCredentials("site-ap", "192.168.8.61", "gopro", "pw3", "CERT3"))  # replaces
    loaded = {c.ssid: c for c in store.load("4609")}
    assert set(loaded) == {"home", "site-ap"} and loaded["site-ap"].password == "pw3"
    assert store.forget("4609", "home") and not store.forget("4609", "home")
    assert [c.ssid for c in store.load("4609")] == ["site-ap"]
    with pytest.raises(LookupError, match="not linked"):
        store.save("9999", CREDS)
    assert store.load("9999") == []


def test_the_registry_store_falls_back_to_files_until_they_are_imported(registry, tmp_path: Path) -> None:
    add_camera(registry, "gopro13_01")
    add_camera(registry, "gopro13_02")
    link_gopro(registry, find_camera(registry, "gopro13_01"), "4609")
    legacy = CredentialStore(tmp_path / "creds")
    legacy.save("4609", CREDS)
    legacy.save("4243", CohnCredentials("site-ap", "192.168.8.62", "gopro", "x", "C"))  # not linked
    store = RegistryCredentialStore(registry, legacy)
    assert [c.ip_address for c in store.load("4609")] == ["192.168.8.60"]
    assert store.import_legacy() == {"4243": 0, "4609": 1}
    legacy._path("4609").unlink()
    assert [c.ssid for c in store.load("4609")] == ["site-ap"]
    assert store.instance_id("4243") is None


def test_import_credentials_command(registry, tmp_path: Path, cli_env) -> None:
    invoke, _, _, stored = cli_env
    add_camera(registry, "gopro13_01")
    link_gopro(registry, find_camera(registry, "gopro13_01"), "4609")
    CredentialStore(tmp_path / "creds").save("4609", CREDS)
    CredentialStore(tmp_path / "creds").save("4243", CREDS)
    result = invoke("import-credentials", "--delete-files")
    assert result.exit_code == 0, result.output
    assert "camera 4609: 1 WLAN entries imported" in result.output and "4243: not linked" in result.output
    assert [c.ssid for c in stored.load("4609")] == ["site-ap"]
    assert not (tmp_path / "creds" / "4609.json").exists() and (tmp_path / "creds" / "4243.json").exists()


def test_credentials_exist_only_in_the_registry(tmp_path: Path) -> None:
    session = create_session(tmp_path / "session.db")
    tables = {r[0] for r in session.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "control_credentials" not in tables and "control_endpoints" in tables
    session.close()
    registry = create_registry(tmp_path / "registry.db")
    assert registry.execute("SELECT COUNT(*) FROM control_credentials").fetchone()[0] == 0
    registry.close()


def test_an_older_registry_gains_the_credentials_table(tmp_path: Path) -> None:
    path = tmp_path / "registry.db"
    reg = create_registry(path)
    reg.execute("DROP TABLE control_credentials")
    reg.execute("PRAGMA user_version = 22")
    reg.commit()
    reg.close()
    reg = open_registry(path)
    assert reg.execute("SELECT COUNT(*) FROM control_credentials").fetchone()[0] == 0
    reg.close()


def test_the_daemon_builds_a_driver_from_registry_credentials(registry) -> None:
    pytest.importorskip("httpx")
    from pipeline.capture.daemon import build_driver

    add_camera(registry, "gopro13_01")
    link_gopro(registry, find_camera(registry, "gopro13_01"), "4609")
    store = RegistryCredentialStore(registry)
    with pytest.raises(ConfigError, match="no COHN credentials"):
        build_driver(CameraConfig("gopro13_01", "gopro_cohn", "4609"), parse_config({}), store)
    store.save("4609", CREDS)
    assert build_driver(CameraConfig("gopro13_01", "gopro_cohn", "4609"), parse_config({}), store).label == "gopro13_01"
