# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests of the Hero13 COHN driver against the fake camera."""

from __future__ import annotations

import asyncio
import contextlib
from base64 import b64encode
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("httpx")

from pipeline.capture.credentials import CohnCredentials, CredentialStore  # noqa: E402
from pipeline.capture.devices import LinkState  # noqa: E402
from pipeline.capture.gopro_cohn import CohnEndpoint, GoProCohnDriver, NotProvisionedError  # noqa: E402
from pipeline.capture.gopro_profile import ProfileError, translate  # noqa: E402
from pipeline.capture.service import CameraSlot, CaptureController  # noqa: E402
from posetrak.db.capture_control import get_capture_events  # noqa: E402
from posetrak.db.db import create_mocap_session, create_session  # noqa: E402
from tests.capture.fake_gopro import (  # noqa: E402
    FLAT_PHOTO,
    FLAT_TIME_LAPSE,
    FLAT_VIDEO,
    SET_FPS,
    SET_LENS,
    SET_RESOLUTION,
    FakeGoPro,
)

PROFILE = {"resolution": "4K", "fps": 120, "lens": "linear"}


def run(coro):
    return asyncio.run(coro)


@contextlib.asynccontextmanager
async def driver_for(fake: FakeGoPro, **kwargs) -> AsyncIterator[GoProCohnDriver]:
    kwargs.setdefault("poll_interval_s", 0.05)
    kwargs.setdefault("request_timeout_s", 1.0)
    kwargs.setdefault("confirm_timeout_s", 2.0)
    driver = GoProCohnDriver("cam", [CohnEndpoint("home", fake.base_url)], **kwargs)
    await driver.connect()
    try:
        yield driver
    finally:
        await driver.close()


@pytest.fixture()
def fake():
    with FakeGoPro("1234") as f:
        yield f


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


# -- profile translation ------------------------------------------------------


def test_translate_symbolic_and_raw_settings() -> None:
    assert translate({"resolution": "4k", "fps": "120", "lens": "Linear"}) == {2: 1, 3: 1, 121: 4}
    assert list(translate({"lens": "wide", "fps": 60, "resolution": "1080"})) == [2, 3, 121]  # application order
    assert translate({"108": 1, "fps": 30}) == {3: 8, 108: 1}


@pytest.mark.parametrize(
    "bad",
    [{"colour": "red"}, {"resolution": "9K"}, {"fps": 77}, {"lens": "fisheye"}],
)
def test_translate_rejects_unknown_names_and_values(bad: dict) -> None:
    with pytest.raises(ProfileError):
        translate(bad)


# -- credentials ------------------------------------------------------------------


def test_credential_store_keeps_one_entry_per_ssid(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path)
    assert store.load("1234") == []
    home = CohnCredentials("home", "192.168.1.50", "gopro", "pw1", "CERT1")
    site = CohnCredentials("site", "192.168.8.50", "gopro", "pw2", "CERT2")
    store.save("1234", home)
    store.save("1234", site)
    store.save("1234", CohnCredentials("home", "192.168.1.51", "gopro", "pw3", "CERT3"))
    loaded = {c.ssid: c for c in store.load("1234")}
    assert set(loaded) == {"home", "site"} and loaded["home"].password == "pw3" and loaded["site"] == site
    assert store.forget("1234", "home") and not store.forget("1234", "home")
    assert [c.ssid for c in store.load("1234")] == ["site"]
    assert store.forget("1234", "site") and store.load("1234") == []
    with pytest.raises(ValueError):
        store.load("../x")


def test_driver_requires_endpoints() -> None:
    with pytest.raises(NotProvisionedError):
        GoProCohnDriver("cam", [])


# -- link and status ---------------------------------------------------------------


def test_connect_reports_status(fake: FakeGoPro) -> None:
    fake.battery_percent, fake.remaining_video_s = 77, 5400

    async def go():
        async with driver_for(fake) as d:
            st = d.status
            assert (st.link, st.battery_pct, st.storage_remaining_s, st.recording) == (LinkState.UP, 77, 5400, False)
            assert not st.overheating

    run(go())


def test_connect_to_dead_camera_leaves_link_down_and_recovers(fake: FakeGoPro) -> None:
    fake.link_up = False

    async def go():
        async with driver_for(fake, request_timeout_s=0.3) as d:
            assert d.status.link == LinkState.DOWN
            fake.link_up = True
            for _ in range(60):
                if d.status.link == LinkState.UP:
                    return
                await asyncio.sleep(0.1)
            raise AssertionError("link did not come back")

    run(go())


def test_link_loss_is_reported_after_consecutive_failures(fake: FakeGoPro) -> None:
    async def go():
        async with driver_for(fake, request_timeout_s=0.3) as d:
            assert d.status.link == LinkState.UP
            fake.link_up = False
            for _ in range(80):
                if d.status.link == LinkState.DOWN:
                    return
                await asyncio.sleep(0.1)
            raise AssertionError("link loss not reported")

    run(go())


def test_status_follows_camera_changes(fake: FakeGoPro) -> None:
    async def go():
        async with driver_for(fake) as d:
            fake.system_hot, fake.battery_percent = True, 12
            await asyncio.sleep(0.3)
            assert d.status.overheating and d.status.battery_pct == 12

    run(go())


def test_credentials_are_sent_and_wrong_ones_fail(fake: FakeGoPro) -> None:
    fake.require_auth = "Basic " + b64encode(b"gopro:secret").decode()

    async def go():
        good = GoProCohnDriver("cam", [CohnEndpoint("n", fake.base_url, "gopro", "secret")], poll_interval_s=0.05)
        bad = GoProCohnDriver("cam", [CohnEndpoint("n", fake.base_url, "gopro", "wrong")], poll_interval_s=0.05)
        try:
            await good.connect()
            await bad.connect()
            assert good.status.link == LinkState.UP and bad.status.link == LinkState.DOWN
        finally:
            await good.close()
            await bad.close()

    run(go())


def test_falls_back_to_another_network(fake: FakeGoPro) -> None:
    async def go():
        endpoints = [CohnEndpoint("site", "http://127.0.0.1:9"), CohnEndpoint("home", fake.base_url)]
        d = GoProCohnDriver("cam", endpoints, poll_interval_s=0.05, request_timeout_s=0.5)
        try:
            await d.connect()
            for _ in range(40):  # the first poll tries only the preferred network
                if d.status.link == LinkState.UP:
                    break
                await asyncio.sleep(0.1)
            assert d.status.link == LinkState.UP and d.status.detail["network"] == "home"
        finally:
            await d.close()

    run(go())


# -- arming ------------------------------------------------------------------------------


def test_arm_pushes_settings_and_reads_them_back(fake: FakeGoPro) -> None:
    async def go():
        async with driver_for(fake) as d:
            res = await d.arm(PROFILE)
            assert res.ok, res
            assert fake.settings[SET_RESOLUTION] == 1 and fake.settings[SET_FPS] == 1 and fake.settings[SET_LENS] == 4

    run(go())


def test_arm_leaves_matching_settings_alone(fake: FakeGoPro) -> None:
    fake.settings.update({SET_RESOLUTION: 1, SET_FPS: 1, SET_LENS: 4})

    async def go():
        async with driver_for(fake) as d:
            assert (await d.arm(PROFILE)).ok
            assert fake.commands_to("/gopro/camera/setting") == []

    run(go())


@pytest.mark.parametrize("wrong_mode", [FLAT_TIME_LAPSE, FLAT_PHOTO])
def test_arm_switches_a_camera_out_of_the_wrong_mode(fake: FakeGoPro, wrong_mode: int) -> None:
    fake.flat_mode = wrong_mode

    async def go():
        async with driver_for(fake) as d:
            res = await d.arm(PROFILE)
            assert res.ok, res
            assert fake.flat_mode == FLAT_VIDEO
            assert fake.commands_to("/gopro/camera/presets/set_group")

    run(go())


def test_arm_reports_settings_the_camera_ignores(fake: FakeGoPro) -> None:
    fake.ignore_settings = True

    async def go():
        async with driver_for(fake) as d:
            res = await d.arm(PROFILE)
            assert not res.ok
            assert set(res.mismatches) == {"resolution", "fps", "lens"}
            assert res.mismatches["fps"][0] == "120"

    run(go())


def test_arm_reports_refused_settings(fake: FakeGoPro) -> None:
    fake.supported_options[SET_FPS] = {8}

    async def go():
        async with driver_for(fake) as d:
            res = await d.arm(PROFILE)
            assert not res.ok and any("refused fps=120" in p for p in res.problems)

    run(go())


def test_arm_rejects_a_profile_the_camera_type_cannot_do(fake: FakeGoPro) -> None:
    async def go():
        async with driver_for(fake) as d:
            res = await d.arm({"fps": 77})
            assert not res.ok and "77" in res.problems[0]

    run(go())


def test_arm_refuses_a_recording_or_unreachable_camera(fake: FakeGoPro) -> None:
    async def go():
        async with driver_for(fake, request_timeout_s=0.3) as d:
            assert (await d.arm(PROFILE)).ok
            assert (await d.start()).ok
            res = await d.arm(PROFILE)
            assert not res.ok and "recording" in res.problems[0]
            await d.stop()
            fake.link_up = False
            res = await d.arm(PROFILE)
            assert not res.ok and "not reachable" in res.problems[0]

    run(go())


# -- start and stop --------------------------------------------------------------------------


def test_start_and_stop_are_confirmed_by_the_encoding_flag(fake: FakeGoPro) -> None:
    fake.encoding_delay_s = 0.4

    async def go():
        async with driver_for(fake) as d:
            assert (await d.arm(PROFILE)).ok
            started = await d.start()
            assert started.ok and started.confirm_ts
            assert ts(started.cmd_ts) <= ts(started.ack_ts) <= ts(started.confirm_ts)
            assert (ts(started.confirm_ts) - ts(started.ack_ts)).total_seconds() >= 0.3  # acked before recording
            assert fake.is_encoding() and d.status.recording
            stopped = await d.stop()
            assert stopped.ok and stopped.confirm_ts and not fake.is_encoding()

    run(go())


def test_start_that_never_records_is_not_confirmed(fake: FakeGoPro) -> None:
    fake.never_encodes = True

    async def go():
        async with driver_for(fake, confirm_timeout_s=0.6) as d:
            assert (await d.arm(PROFILE)).ok
            res = await d.start()
            assert not res.ok and res.ack_ts and res.confirm_ts is None and "not confirmed" in res.error

    run(go())


def test_refused_command_is_reported(fake: FakeGoPro) -> None:
    fake.fail_commands = 1

    async def go():
        async with driver_for(fake) as d:
            assert (await d.arm(PROFILE)).ok
            res = await d.start()
            assert not res.ok and "HTTP 500" in res.error

    run(go())


def test_start_refuses_when_the_mode_changed_after_arming(fake: FakeGoPro) -> None:
    async def go():
        async with driver_for(fake) as d:
            assert (await d.arm(PROFILE)).ok
            fake.flat_mode = FLAT_TIME_LAPSE
            await asyncio.sleep(0.2)  # the next poll sees it
            res = await d.start()
            assert not res.ok and "video mode" in res.error
            assert not fake.commands_to("/gopro/camera/shutter/start")

    run(go())


def test_start_refuses_when_settings_changed_after_arming(fake: FakeGoPro) -> None:
    async def go():
        async with driver_for(fake) as d:
            assert (await d.arm(PROFILE)).ok
            fake.settings[SET_FPS] = 8
            await asyncio.sleep(0.2)
            res = await d.start()
            assert not res.ok and "fps" in res.error

    run(go())


def test_start_refuses_a_camera_already_recording(fake: FakeGoPro) -> None:
    async def go():
        async with driver_for(fake) as d:
            assert (await d.arm(PROFILE)).ok
            assert (await d.start()).ok
            res = await d.start()
            assert not res.ok and "already recording" in res.error

    run(go())


def test_camera_stopping_by_itself_shows_in_status(fake: FakeGoPro) -> None:
    fake.stop_after_s = 0.3

    async def go():
        async with driver_for(fake) as d:
            assert (await d.arm(PROFILE)).ok
            assert (await d.start()).ok
            await asyncio.sleep(0.6)
            assert d.status.recording is False

    run(go())


# -- with the controller ---------------------------------------------------------------------------


def test_controller_captures_with_two_cameras_and_records_events(tmp_path: Path) -> None:
    async def go():
        with FakeGoPro("1111") as a, FakeGoPro("2222") as b:
            a.flat_mode = FLAT_TIME_LAPSE  # one camera was left in the wrong mode
            session = create_session(tmp_path / "s.db")
            sid = create_mocap_session(session, location="test")
            drivers = [
                GoProCohnDriver(label, [CohnEndpoint("n", fake.base_url)], poll_interval_s=0.05, confirm_timeout_s=2.0)
                for label, fake in (("a", a), ("b", b))
            ]
            ctl = CaptureController(
                [CameraSlot(d.label, f"inst-{d.label}", d) for d in drivers], session, sid, settings=PROFILE
            )
            await ctl.connect()
            try:
                armed = await ctl.arm()
                assert armed.ready, armed
                started = await ctl.start_capture("take")
                assert started.ok and a.is_encoding() and b.is_encoding()
                stopped = await ctl.stop_capture()
                assert stopped.ok and not a.is_encoding() and not b.is_encoding()
                events = get_capture_events(session, ctl.capture_id)
                assert {e.camera_instance_id for e in events} == {"inst-a", "inst-b"}
                assert ctl.alerts == []  # a commanded stop is not "stopped by itself"
                assert all(e.confidence == "confirmed" and e.start_confirm_ts and e.stop_confirm_ts for e in events)
            finally:
                await ctl.close()

    run(go())


# -- provisioning command and route check --------------------------------------------------------


def test_provision_command_resolves_the_network_and_stores_credentials(tmp_path: Path, monkeypatch) -> None:
    from click.testing import CliRunner

    from pipeline.capture import gopro_ble
    from pipeline.capture.cli import main

    config = tmp_path / "config.toml"
    config.write_text(
        f'[daemon]\ntoken = "t"\ncredentials_dir = "{(tmp_path / "creds").as_posix()}"\n'
        '[[network]]\nname = "location"\nssid = "site-ap"\n',
        encoding="utf-8",
    )
    seen = {}

    async def fake_provision(serial, ssid, password, store, *, force=False, state_dir=None):
        seen.update(serial=serial, ssid=ssid, password=password, force=force)
        creds = CohnCredentials(ssid, "192.168.8.60", "gopro", "pw", "CERT")
        store.save(serial, creds)
        return creds

    monkeypatch.setattr(gopro_ble, "provision", fake_provision)
    result = CliRunner().invoke(main, ["--config", str(config), "provision", "1234", "--network", "location", "--password", "wlan-pw"])
    assert result.exit_code == 0, result.output
    assert seen == {"serial": "1234", "ssid": "site-ap", "password": "wlan-pw", "force": False}
    assert [c.ssid for c in CredentialStore(tmp_path / "creds").load("1234")] == ["site-ap"]
    assert "192.168.8.60" in result.output

    unknown = CliRunner().invoke(main, ["--config", str(config), "provision", "1234", "--network", "nope", "--password", "x"])
    assert unknown.exit_code != 0 and "unknown network" in unknown.output


def test_route_warning_is_silent_for_cameras_on_other_networks_and_without_routes(monkeypatch) -> None:
    from pipeline.capture import network

    monkeypatch.setattr(network, "lan_addresses", lambda: ["192.168.1.20"])
    assert network.route_warning("not-an-address") is None
    assert network.route_warning("10.77.0.5") is None  # not our network: unreachable, nothing to explain
