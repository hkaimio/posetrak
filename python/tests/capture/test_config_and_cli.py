# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the capture daemon configuration, CLI and the manual driver."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from click.testing import CliRunner

from pipeline.capture.cli import main
from pipeline.capture.config import (
    ConfigError,
    load_config,
    new_token,
    parse_config,
    render_default_config,
)
from pipeline.capture.devices import DeviceState, ManualDriver

FULL = {
    "daemon": {"port": 9000, "token": "t", "session_db": "D:/x/s.db"},
    "profile": {"name": "mocap", "camera_mode": "m1", "settings": {"fps": 120}},
    "network": [
        {"name": "home", "ssid": "HomeNet", "discovery": "mdns"},
        {"name": "location", "ssid": "ap", "addresses": {"gopro-13_01": "192.168.8.101"}},
    ],
    "camera": [
        {"label": "gopro-13_01", "method": "gopro_cohn", "gopro_serial": "1234"},
        {"label": "old-cam", "method": "manual"},
    ],
}


def test_parse_full_config() -> None:
    cfg = parse_config(FULL)
    assert cfg.daemon.port == 9000 and cfg.daemon.session_db == Path("D:/x/s.db")
    assert cfg.profile.settings == {"fps": 120}
    assert cfg.network("location").addresses == {"gopro-13_01": "192.168.8.101"}
    assert cfg.network_for_ssid("HomeNet").name == "home"
    assert cfg.network_for_ssid("elsewhere") is None and cfg.network_for_ssid(None) is None
    assert [c.label for c in cfg.cameras] == ["gopro-13_01", "old-cam"]


def test_empty_config_has_defaults() -> None:
    cfg = parse_config({})
    assert cfg.daemon.port == 8765 and cfg.cameras == () and cfg.networks == ()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["daemon"].update(port=0), "port"),
        (lambda d: d["network"][0].update(discovery="telepathy"), "discovery"),
        (lambda d: d["network"][1].update(name="home"), "duplicate network name"),
        (lambda d: d["network"][1].update(ssid="HomeNet"), "duplicate network ssid"),
        (lambda d: d["camera"][0].update(method="vhs"), "unknown method"),
        (lambda d: d["camera"][0].pop("gopro_serial"), "gopro_serial"),
        (lambda d: d["camera"][1].update(label="gopro-13_01"), "duplicate camera label"),
        (lambda d: d["camera"][1].pop("label"), "label"),
    ],
)
def test_invalid_config_rejected(mutate, message: str) -> None:
    import copy

    data = copy.deepcopy(FULL)
    mutate(data)
    with pytest.raises(ConfigError, match=message):
        parse_config(data)


def test_load_config_missing_and_malformed(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[daemon\nport=1", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_rendered_default_config_is_valid(tmp_path: Path) -> None:
    token = new_token()
    path = tmp_path / "config.toml"
    path.write_text(render_default_config(token, Path("D:/caps/s.db")), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.daemon.token == token and cfg.daemon.session_db == Path("D:/caps/s.db")
    assert len(token) >= 24 and new_token() != token


def test_cli_init_and_show_config(tmp_path: Path) -> None:
    path = tmp_path / "cfg" / "config.toml"
    runner = CliRunner()
    result = runner.invoke(main, ["--config", str(path), "init-config"])
    assert result.exit_code == 0, result.output
    assert path.exists()

    again = runner.invoke(main, ["--config", str(path), "init-config"])
    assert again.exit_code != 0 and "already exists" in again.output
    assert runner.invoke(main, ["--config", str(path), "init-config", "--force"]).exit_code == 0

    shown = runner.invoke(main, ["--config", str(path), "show-config"])
    assert shown.exit_code == 0 and "listen:" in shown.output


def test_cli_show_config_reports_errors(tmp_path: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_path / "missing.toml"), "show-config"])
    assert result.exit_code != 0 and "not found" in result.output


def test_manual_driver_lifecycle() -> None:
    async def scenario():
        seen = []
        driver = ManualDriver("cam-a")
        driver.on_status(lambda label, st: seen.append((label, st.state)))
        await driver.connect()
        arm = await driver.arm({})
        started = await driver.start()
        stopped = await driver.stop()
        await driver.close()
        return driver, seen, arm, started, stopped

    driver, seen, arm, started, stopped = asyncio.run(scenario())
    assert arm.ok and started.ok and stopped.ok
    assert started.confirm_ts is None and driver.confidence == "manual"
    assert [s for _, s in seen] == [
        DeviceState.READY,
        DeviceState.ARMED,
        DeviceState.RECORDING,
        DeviceState.STOPPED,
        DeviceState.OFFLINE,
    ]
