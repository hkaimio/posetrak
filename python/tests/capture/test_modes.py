# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests of capture modes and setups: the planner, the configuration, and arming with modes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from pipeline.capture.config import ConfigError, SetupCameraConfig, SetupConfig, parse_config
from pipeline.capture.daemon import build_planner
from pipeline.capture.devices import ModeSpec
from pipeline.capture.gopro_profile import ProfileError, derive
from pipeline.capture.modes import ModeError, ModePlanner, PlannedCamera, model_keys
from pipeline.capture.service import CameraSlot, CaptureController, CapturePhase
from posetrak.db.capture_control import get_capture_events
from posetrak.db.db import (
    create_camera_instance,
    create_camera_mode,
    create_camera_model,
    create_mocap_session,
    create_registry,
    create_session,
)
from tests.capture.scripted_driver import ScriptedDriver

M4K = ModeSpec("m-4k", "4K 120 linear", 3840, 2160, 120, {"resolution": "4K", "fps": 120, "lens": "linear"})
M27 = ModeSpec("m-27", "2.7K 120 linear", 2704, 1520, 120, {"resolution": "2.7K", "fps": 120, "lens": "linear"})
BARE = ModeSpec("m-bare", "bare", 3840, 2160, 60, {})


def planned(label: str, model: str = "GoPro Hero13", modes=(M4K, M27), driver=None) -> PlannedCamera:
    manufacturer, _, name = model.partition(" ")
    return PlannedCamera(label, model_keys(manufacturer, name), model, tuple(modes), driver or ScriptedDriver(label))


def run(coro):
    return asyncio.run(coro)


# -- the Hero13 mapping ------------------------------------------------------------------------


def test_hero13_settings_are_derived_from_size_and_rate() -> None:
    assert derive(3840, 2160, 120) == {"resolution": "4K", "fps": 120}
    assert derive(2704, 1520, 59.94 + 0.06) == {"resolution": "2.7K", "fps": 60}
    with pytest.raises(ProfileError, match="cannot derive"):
        derive(1000, 1000, 30)
    with pytest.raises(ProfileError, match="cannot derive"):
        derive(3840, 2160, 77)


def test_hero13_driver_prefers_the_modes_own_settings() -> None:
    pytest.importorskip("httpx")
    from pipeline.capture.gopro_cohn import GoProCohnDriver

    driver = object.__new__(GoProCohnDriver)
    off = {"hypersmooth": "off"}
    assert driver.mode_settings(M4K) == {**M4K.capture_settings, **off}
    assert driver.mode_settings(BARE) == {"resolution": "4K", "fps": 60, **off}
    stabilized = ModeSpec("x", "s", 3840, 2160, 120, {"hypersmooth": "high"})
    assert driver.mode_settings(stabilized)["hypersmooth"] == "high"
    lens_only = ModeSpec("x", "lens", 3840, 2160, 120, {"lens": "linear"})
    assert driver.mode_settings(lens_only) == {"resolution": "4K", "fps": 120, "lens": "linear", **off}
    overridden = ModeSpec("x", "o", 3840, 2160, 120, {"fps": 100})
    assert driver.mode_settings(overridden) == {"resolution": "4K", "fps": 100, **off}
    unknown_size = ModeSpec("x", "u", 0, 0, 0, {"resolution": "5K", "fps": 30, "lens": "wide"})
    assert driver.mode_settings(unknown_size) == {**unknown_size.capture_settings, **off}
    with pytest.raises(ProfileError):
        driver.mode_settings(ModeSpec("x", "u", 0, 0, 0, {"lens": "wide"}))
    assert driver.settings_control == "driver" and "lens" in driver.supported_settings()


# -- configuration -----------------------------------------------------------------------------


def test_setups_are_read_from_the_configuration() -> None:
    cfg = parse_config(
        {
            "daemon": {"default_setup": "indoor"},
            "setup": {
                "indoor": {
                    "mode": {"GoPro Hero13": "4K 120 linear"},
                    "camera": {"cam1": {"mode": "2.7K 120 linear", "settings": {"bit_rate": "high"}}},
                },
                "outdoor": {},
            },
        }
    )
    assert cfg.daemon.default_setup == "indoor"
    assert cfg.setup("indoor") == SetupConfig(
        "indoor", {"GoPro Hero13": "4K 120 linear"}, {"cam1": SetupCameraConfig("2.7K 120 linear", {"bit_rate": "high"})}
    )
    with pytest.raises(ConfigError, match="unknown setup"):
        cfg.setup("nope")
    with pytest.raises(ConfigError, match="default_setup"):
        parse_config({"daemon": {"default_setup": "nope"}, "setup": {"a": {}}})
    with pytest.raises(ConfigError, match="setup.a.mode"):
        parse_config({"setup": {"a": {"mode": "4K"}}})


# -- planner -----------------------------------------------------------------------------------


def planner_for(setups, *cameras, default=None, profile=None) -> ModePlanner:
    return ModePlanner(list(cameras), tuple(setups), default, profile)


def test_a_model_entry_covers_every_camera_of_the_model() -> None:
    planner = planner_for([SetupConfig("s", {"Hero13": "4K 120 linear"})], planned("a"), planned("b"))
    plan = planner.plan(["a", "b"])
    assert plan["a"].mode_id == plan["b"].mode_id == "m-4k"
    assert plan["a"].settings == M4K.capture_settings and plan["a"].problem is None


def test_a_camera_entry_beats_the_model_and_adds_settings() -> None:
    setup = SetupConfig(
        "s", {"GoPro Hero13": "4K 120 linear"}, {"b": SetupCameraConfig("2.7K 120 linear", {"bit_rate": "high"})}
    )
    plan = planner_for([setup], planned("a"), planned("b")).plan(["a", "b"])
    assert plan["a"].name == "4K 120 linear"
    assert plan["b"].name == "2.7K 120 linear"
    assert plan["b"].settings == {**M27.capture_settings, "bit_rate": "high"}


def test_an_operator_override_beats_the_setup() -> None:
    planner = planner_for([SetupConfig("s", {"Hero13": "4K 120 linear"})], planned("a"), planned("b"))
    plan = planner.plan(["a", "b"], overrides={"b": "2.7K 120 linear"})
    assert (plan["a"].name, plan["b"].name) == ("4K 120 linear", "2.7K 120 linear")
    with pytest.raises(ModeError, match="unknown camera"):
        planner.plan(["a"], overrides={"zzz": "x"})


def test_different_models_get_their_own_modes() -> None:
    hero11 = ModeSpec("m-11", "4K 60", 3840, 2160, 60, {"resolution": "4K"})
    setup = SetupConfig("s", {"GoPro Hero13": "4K 120 linear", "GoPro Hero11": "4K 60"})
    plan = planner_for([setup], planned("a"), planned("b", "GoPro Hero11", [hero11])).plan(["a", "b"])
    assert (plan["a"].mode_id, plan["b"].mode_id) == ("m-4k", "m-11")


def test_cameras_that_cannot_be_given_a_mode_get_a_problem() -> None:
    setup = SetupConfig("s", {"GoPro Hero13": "no such mode"})
    plan = planner_for([setup], planned("a"), planned("b", "Pixel 9", [])).plan(["a", "b"])
    assert "not a capture mode of GoPro Hero13" in plan["a"].problem and "4K 120 linear" in plan["a"].problem
    assert "no capture mode for Pixel 9 in setup 's'" in plan["b"].problem

    class Underivable(ScriptedDriver):
        def mode_settings(self, mode):
            raise ProfileError("cannot derive anything")

    odd = planner_for([SetupConfig("s", {"Hero13": "bare"})], planned("c", modes=[BARE], driver=Underivable("c")))
    assert "cannot derive anything" in odd.plan(["c"])["c"].problem


def test_without_setups_the_profile_settings_apply() -> None:
    planner = planner_for([], planned("a"), profile={"fps": 60})
    choice = planner.plan(["a"])["a"]
    assert (choice.mode_id, choice.settings, choice.problem) == (None, {"fps": 60}, None)
    with pytest.raises(ModeError, match="no setups are configured"):
        planner.plan(["a"], setup="x")


def test_choosing_among_several_setups() -> None:
    a, b = SetupConfig("a", {"Hero13": "4K 120 linear"}), SetupConfig("b", {"Hero13": "2.7K 120 linear"})
    cam = planned("c")
    with pytest.raises(ModeError, match="choose a setup: a, b"):
        planner_for([a, b], cam).plan(["c"])
    assert planner_for([a, b], cam, default="b").plan(["c"])["c"].name == "2.7K 120 linear"
    assert planner_for([a, b], cam).plan(["c"], setup="a")["c"].name == "4K 120 linear"
    assert planner_for([a], cam).default_setup == "a"
    with pytest.raises(ModeError, match="unknown setup 'z'"):
        planner_for([a, b], cam).plan(["c"], setup="z")


def test_the_planner_describes_what_the_operator_can_choose() -> None:
    manual = ScriptedDriver("p")
    manual.settings_control = "manual"
    planner = planner_for([SetupConfig("s")], planned("a"), planned("p", "Google Pixel 9", [BARE], manual))
    info = planner.describe()
    assert info["setups"] == ["s"] and info["default"] == "s"
    assert info["cameras"]["a"]["modes"] == ["4K 120 linear", "2.7K 120 linear"]
    assert info["cameras"]["p"]["settings_control"] == "manual"


# -- registry ------------------------------------------------------------------------------------


def test_the_planner_is_built_from_the_registry(tmp_path: Path) -> None:
    reg = create_registry(tmp_path / "registry.db")
    model = create_camera_model(reg, manufacturer="GoPro", model_name="Hero13")
    create_camera_mode(reg, model, width_px=3840, height_px=2160, nominal_fps=120, name="4K 120 linear",
                       capture_settings={"lens": "linear"})
    create_camera_mode(reg, model, width_px=1920, height_px=1080, nominal_fps=30)  # unnamed: not offered
    instance = create_camera_instance(reg, model, label="cam1")
    cfg = parse_config({"setup": {"s": {"mode": {"GoPro Hero13": "4K 120 linear"}}}})
    planner = build_planner(reg, [CameraSlot("cam1", instance, ScriptedDriver("cam1"))], cfg)
    assert planner.options("cam1") == ["4K 120 linear"]
    choice = planner.plan(["cam1"])["cam1"]
    assert choice.settings == {"lens": "linear"} and choice.mode_id

    stray = parse_config({"setup": {"s": {"camera": {"ghost": {"mode": "x"}}}}})
    with pytest.raises(ConfigError, match="ghost"):
        build_planner(reg, [CameraSlot("cam1", instance, ScriptedDriver("cam1"))], stray)
    reg.close()


# -- controller ----------------------------------------------------------------------------------


class Rig:
    def __init__(self, tmp_path: Path, planner: ModePlanner, drivers: list[ScriptedDriver]) -> None:
        self.session = create_session(tmp_path / "s.db")
        sid = create_mocap_session(self.session, location="test")
        self.drivers = drivers
        slots = [CameraSlot(d.label, f"inst-{d.label}", d) for d in drivers]
        self.ctl = CaptureController(slots, self.session, sid, planner=planner, settings={"fps": 30})


def test_each_camera_is_armed_with_its_modes_settings_and_the_event_records_them(tmp_path: Path) -> None:
    a, b = ScriptedDriver("a"), ScriptedDriver("b")
    setup = SetupConfig("s", {"Hero13": "4K 120 linear"}, {"b": SetupCameraConfig("2.7K 120 linear")})
    rig = Rig(tmp_path, planner_for([setup], planned("a", driver=a), planned("b", driver=b)), [a, b])

    async def scenario():
        await rig.ctl.connect()
        report = await rig.ctl.arm()
        assert report.ready
        assert rig.ctl.snapshot()["cameras"]["b"]["mode"] == "2.7K 120 linear"
        assert rig.ctl.snapshot()["modes"]["selected"] == "s"
        await rig.ctl.start_capture()
        await rig.ctl.stop_capture()

    run(scenario())
    assert a.armed_with == M4K.capture_settings and b.armed_with == M27.capture_settings
    events = {e.camera_instance_id: e for e in get_capture_events(rig.session, rig.ctl.capture_id)}
    assert events["inst-a"].camera_mode_id == "m-4k"
    assert json.loads(events["inst-b"].settings_json) == M27.capture_settings
    assert events["inst-b"].camera_mode_id == "m-27"


def test_arming_with_an_override_and_an_unknown_setup(tmp_path: Path) -> None:
    a = ScriptedDriver("a")
    rig = Rig(tmp_path, planner_for([SetupConfig("s", {"Hero13": "4K 120 linear"})], planned("a", driver=a)), [a])

    async def scenario():
        await rig.ctl.connect()
        with pytest.raises(ModeError):
            await rig.ctl.arm(setup="nope")
        assert rig.ctl.phase == CapturePhase.IDLE
        report = await rig.ctl.arm(modes={"a": "2.7K 120 linear"})
        assert report.ready

    run(scenario())
    assert a.armed_with == M27.capture_settings


def test_a_camera_without_a_usable_mode_is_not_ready_and_is_not_armed(tmp_path: Path) -> None:
    a = ScriptedDriver("a")
    rig = Rig(tmp_path, planner_for([SetupConfig("s", {"Pixel": "x"})], planned("a", driver=a)), [a])

    async def scenario():
        await rig.ctl.connect()
        return await rig.ctl.arm()

    report = run(scenario())
    assert not report.ready and "no capture mode for GoPro Hero13" in report.problems["a"][0]
    assert a.armed_with is None


def test_a_camera_set_by_hand_must_be_confirmed_by_the_operator(tmp_path: Path) -> None:
    a, phone = ScriptedDriver("a"), ScriptedDriver("phone", "manual")
    phone.settings_control = "manual"
    pixel = ModeSpec("m-px", "1080 60", 1920, 1080, 60, {"resolution": "1080p", "fps": 60})
    setup = SetupConfig("s", {"Hero13": "4K 120 linear", "Pixel 9": "1080 60"})
    planner = planner_for([setup], planned("a", driver=a), planned("phone", "Google Pixel 9", [pixel], phone))
    rig = Rig(tmp_path, planner, [a, phone])

    async def scenario():
        await rig.ctl.connect()
        first = await rig.ctl.arm()
        item = first.open_checklist
        assert item == ("mode:phone",)
        listed = [i for i in rig.ctl.snapshot()["checklist"] if i["id"] == "mode:phone"][0]
        assert "1080 60" in listed["label"] and "fps: 60" in listed["label"] and not listed["done"]
        rig.ctl.tick("mode:phone")
        second = await rig.ctl.arm()
        assert second.ready
        again = await rig.ctl.arm(modes={"phone": "1080 60"})
        assert again.ready  # the same instruction keeps its tick
        await rig.ctl.start_capture()

    run(scenario())
    events = {e.camera_instance_id: e for e in get_capture_events(rig.session, rig.ctl.capture_id)}
    assert events["inst-phone"].camera_mode_id == "m-px"


def test_the_settings_command_lists_the_hero13_vocabulary() -> None:
    from click.testing import CliRunner

    from pipeline.capture.cli import main

    result = CliRunner().invoke(main, ["settings"])
    assert result.exit_code == 0, result.output
    for expected in ("resolution", "5.3K", "fps", "240", "lens", "linear", "hypersmooth", "bit_rate", "performance", "raw Open GoPro"):
        assert expected in result.output


def test_hypersmooth_is_a_setting_applied_after_the_image_settings() -> None:
    from pipeline.capture.gopro_profile import translate

    assert list(translate({"hypersmooth": "off", "lens": "linear", "fps": 120})) == [3, 121, 135]
    assert translate({"hypersmooth": "boost"}) == {135: 3}
    with pytest.raises(ProfileError, match="hypersmooth"):
        translate({"hypersmooth": "sometimes"})
