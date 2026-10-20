# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests of CaptureController against scripted drivers."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from pipeline.capture.devices import ArmResult, DeviceState, LinkState, ManualDriver
from pipeline.capture.service import (
    CameraSlot,
    CaptureController,
    CapturePhase,
    CaptureStateError,
    ChecklistItem,
)
from posetrak.db.capture_control import get_capture_events
from posetrak.db.db import create_mocap_session, create_session
from tests.capture.scripted_driver import ScriptedDriver


class Rig:
    """A controller over three scripted cameras and a temporary session DB."""

    def __init__(self, tmp_path: Path, confidences=("confirmed", "confirmed", "confirmed"), **kw) -> None:
        self.session = create_session(tmp_path / "s.db")
        self.session_id = create_mocap_session(self.session, location="test")
        self.drivers = [ScriptedDriver(f"cam{i}", c) for i, c in enumerate(confidences)]
        slots = [CameraSlot(d.label, f"inst-{d.label}", d) for d in self.drivers]
        self.ctl = CaptureController(slots, self.session, self.session_id, **kw)

    def events(self):
        return {e.camera_instance_id: e for e in get_capture_events(self.session, self.ctl.capture_id)}

    def close(self) -> None:
        self.session.close()


@pytest.fixture()
def make_rig(tmp_path: Path):
    rigs = []

    def make(**kw) -> Rig:
        rig = Rig(_sub(tmp_path, len(rigs)), **kw)
        rigs.append(rig)
        return rig

    yield make
    for r in rigs:
        r.close()


def _sub(tmp_path: Path, i: int) -> Path:
    p = tmp_path / f"rig{i}"
    p.mkdir()
    return p


def run(coro):
    return asyncio.run(coro)


async def ready(rig: Rig):
    await rig.ctl.connect()
    report = await rig.ctl.arm()
    assert report.ready, report
    return report


def test_all_cameras_start_and_stop(make_rig) -> None:
    rig = make_rig(settings={"fps": 120})

    async def scenario():
        await ready(rig)
        assert all(d.armed_with == {"fps": 120} for d in rig.drivers)
        started = await rig.ctl.start_capture(label="take 1")
        assert rig.ctl.phase == CapturePhase.RECORDING
        stopped = await rig.ctl.stop_capture()
        return started, stopped

    started, stopped = run(scenario())
    assert started.ok and stopped.ok and rig.ctl.phase == CapturePhase.STOPPED
    events = rig.events()
    assert set(events) == {"inst-cam0", "inst-cam1", "inst-cam2"}
    for ev in events.values():
        assert ev.status == "ok" and ev.confidence == "confirmed"
        assert ev.start_cmd_ts and ev.start_ack_ts and ev.start_confirm_ts
        assert ev.stop_cmd_ts and ev.stop_confirm_ts
    assert [rig.ctl.camera_state(d.label) for d in rig.drivers] == [DeviceState.STOPPED] * 3


def test_checklist_blocks_ready_until_ticked(make_rig) -> None:
    rig = make_rig(checklist=[ChecklistItem("mode", "Insta360 mode set", "cam1")])

    async def scenario():
        await rig.ctl.connect()
        first = await rig.ctl.arm()
        with pytest.raises(CaptureStateError):
            await rig.ctl.start_capture()
        rig.ctl.tick("mode")
        second = await rig.ctl.arm()
        return first, second

    first, second = run(scenario())
    assert not first.ready and first.open_checklist == ("mode",)
    assert second.ready and rig.ctl.phase == CapturePhase.ARMED


def test_arm_reports_battery_and_setting_mismatch(make_rig) -> None:
    rig = make_rig()
    rig.drivers[0].arm_result = ArmResult(False, mismatches={"fps": (120, 60)})

    async def scenario():
        await rig.ctl.connect()
        rig.drivers[1].push_status(battery_pct=5)
        return await rig.ctl.arm()

    report = run(scenario())
    assert not report.ready
    assert "fps: wanted 120, camera has 60" in report.problems["cam0"][0]
    assert any("battery 5%" in p for p in report.problems["cam1"])
    assert "cam2" not in report.problems
    assert rig.ctl.camera_state("cam0") == DeviceState.FAULT
    assert rig.ctl.camera_state("cam2") == DeviceState.ARMED


def test_one_camera_rejects_start_others_record(make_rig) -> None:
    rig = make_rig()
    rig.drivers[1].start_error = "camera busy"

    async def scenario():
        await ready(rig)
        report = await rig.ctl.start_capture()
        return report

    report = run(scenario())
    assert not report.ok
    assert report.outcomes["cam0"].ok and report.outcomes["cam2"].ok
    assert report.outcomes["cam1"].error == "camera busy"
    assert rig.ctl.phase == CapturePhase.RECORDING
    assert rig.ctl.camera_state("cam1") == DeviceState.FAULT
    assert rig.events()["inst-cam1"].status == "failed"
    assert [a.kind for a in rig.ctl.alerts] == ["start_failed"]


def test_operator_override_after_failed_start(make_rig) -> None:
    rig = make_rig()
    rig.drivers[1].start_error = "camera busy"

    async def scenario():
        await ready(rig)
        await rig.ctl.start_capture()
        rig.ctl.operator_override("cam1", recording=True)
        assert rig.ctl.camera_state("cam1") == DeviceState.RECORDING
        assert rig.ctl.alerts == []
        await rig.ctl.stop_capture()

    run(scenario())
    ev = rig.events()["inst-cam1"]
    assert ev.status == "operator_override" and ev.confidence == "manual"
    assert "stop" not in rig.drivers[1].calls  # the operator stops that camera
    assert rig.ctl.camera_state("cam1") == DeviceState.STOPPED


def test_late_confirmation_within_timeout_is_ok_beyond_is_failure(make_rig) -> None:
    rig = make_rig(command_timeout_s=0.5)
    rig.drivers[0].start_delay_s = 0.2
    rig.drivers[1].start_delay_s = 5.0

    async def scenario():
        await ready(rig)
        t0 = time.monotonic()
        report = await rig.ctl.start_capture()
        return report, time.monotonic() - t0

    report, elapsed = run(scenario())
    assert report.outcomes["cam0"].ok and report.outcomes["cam2"].ok
    assert not report.outcomes["cam1"].ok and "not confirmed within" in report.outcomes["cam1"].error
    assert elapsed < 1.5  # a slow camera does not hold up the group
    assert rig.events()["inst-cam1"].start_cmd_ts is not None


def test_confirmed_driver_without_confirmation_is_a_failure(make_rig) -> None:
    rig = make_rig()
    rig.drivers[2].confirm = False

    report = run(_start(rig))
    assert not report.outcomes["cam2"].ok
    assert "did not confirm" in report.outcomes["cam2"].error


async def _start(rig: Rig):
    await ready(rig)
    return await rig.ctl.start_capture()


def test_commanded_cameras_succeed_without_confirmation(make_rig) -> None:
    rig = make_rig(confidences=("confirmed", "commanded", "manual"))
    report = run(_start(rig))
    assert report.ok
    ev = rig.events()
    assert ev["inst-cam1"].confidence == "commanded" and ev["inst-cam1"].start_confirm_ts is None
    assert ev["inst-cam1"].status == "ok"


def test_driver_exception_becomes_failed_outcome(make_rig) -> None:
    rig = make_rig()
    rig.drivers[0].start_raises = RuntimeError("boom")
    report = run(_start(rig))
    assert not report.outcomes["cam0"].ok and "boom" in report.outcomes["cam0"].error
    assert report.outcomes["cam1"].ok


def test_camera_dropping_mid_take_alerts_and_stop_still_works(make_rig) -> None:
    rig = make_rig()

    async def scenario():
        await _start(rig)
        rig.drivers[1].push_status(link=LinkState.DOWN)
        alerts_during = [a.kind for a in rig.ctl.alerts]
        rig.drivers[1].stop_error = "link lost"
        stopped = await rig.ctl.stop_capture()
        return alerts_during, stopped

    alerts, stopped = run(scenario())
    assert alerts == ["link_lost"]
    assert stopped.outcomes["cam0"].ok and stopped.outcomes["cam2"].ok
    assert not stopped.outcomes["cam1"].ok
    assert rig.events()["inst-cam1"].status == "failed"
    assert rig.events()["inst-cam0"].status == "ok"
    assert rig.ctl.phase == CapturePhase.STOPPED


def test_camera_stopping_by_itself_raises_alert(make_rig) -> None:
    rig = make_rig()

    async def scenario():
        await _start(rig)
        rig.drivers[2].push_status(recording=False)
        return [(a.kind, a.camera) for a in rig.ctl.alerts]

    assert run(scenario()) == [("stopped_by_itself", "cam2")]


def test_camera_stopped_by_command_is_not_stopped_by_itself(make_rig) -> None:
    rig = make_rig()

    async def scenario():
        await _start(rig)
        rig.drivers[0].stop_delay_s = 0.2

        async def camera_stops_mid_command():
            await asyncio.sleep(0.05)
            rig.drivers[0].push_status(recording=False)

        await asyncio.gather(rig.ctl.stop_capture(), camera_stops_mid_command())
        return [a.kind for a in rig.ctl.alerts]

    assert run(scenario()) == []


def test_alerts_clear_when_condition_ends(make_rig) -> None:
    rig = make_rig()

    async def scenario():
        await _start(rig)
        rig.drivers[0].push_status(link=LinkState.DOWN)
        rig.drivers[0].push_status(link=LinkState.UP)
        return rig.ctl.alerts

    assert run(scenario()) == []


def test_stop_with_camera_already_stopped(make_rig) -> None:
    rig = make_rig()

    async def scenario():
        await _start(rig)
        await rig.drivers[0].stop()  # stopped behind the controller's back
        return await rig.ctl.stop_capture()

    assert run(scenario()).ok


def test_overlapping_start_requests_rejected(make_rig) -> None:
    rig = make_rig()
    for d in rig.drivers:
        d.start_delay_s = 0.2

    async def scenario():
        await ready(rig)
        return await asyncio.gather(
            rig.ctl.start_capture(), rig.ctl.start_capture(), return_exceptions=True
        )

    results = run(scenario())
    assert sum(isinstance(r, CaptureStateError) for r in results) == 1
    assert sum(not isinstance(r, Exception) for r in results) == 1
    assert [d.calls.count("start") for d in rig.drivers] == [1, 1, 1]


def test_commands_rejected_in_wrong_phase(make_rig) -> None:
    rig = make_rig()

    async def scenario():
        await rig.ctl.connect()
        with pytest.raises(CaptureStateError):
            await rig.ctl.start_capture()
        with pytest.raises(CaptureStateError):
            await rig.ctl.stop_capture()
        await rig.ctl.arm()
        await rig.ctl.start_capture()
        with pytest.raises(CaptureStateError):
            await rig.ctl.arm()

    run(scenario())


def test_force_start_leaves_out_unarmed_cameras(make_rig) -> None:
    rig = make_rig()
    rig.drivers[0].arm_result = ArmResult(False, ("SD card full",))

    async def scenario():
        await rig.ctl.connect()
        report = await rig.ctl.arm()
        assert not report.ready
        with pytest.raises(CaptureStateError):
            await rig.ctl.start_capture()
        return await rig.ctl.start_capture(force=True)

    report = run(scenario())
    assert set(report.outcomes) == {"cam1", "cam2"}
    assert rig.drivers[0].calls.count("start") == 0
    ev = rig.events()["inst-cam0"]
    assert ev.status == "failed" and ev.notes == "not armed, left out of capture"


def test_rearm_after_stop_starts_a_new_capture(make_rig) -> None:
    rig = make_rig()

    async def scenario():
        await _start(rig)
        first = rig.ctl.capture_id
        await rig.ctl.stop_capture()
        await rig.ctl.arm()
        await rig.ctl.start_capture()
        return first, rig.ctl.capture_id

    first, second = run(scenario())
    assert first != second


def test_fallback_to_manual_driver(make_rig) -> None:
    rig = make_rig()

    async def scenario():
        await rig.ctl.connect()
        await rig.ctl.fallback_to_manual("cam1")
        report = await rig.ctl.arm()
        started = await rig.ctl.start_capture()
        return report, started

    report, started = run(scenario())
    assert report.ready and started.ok
    assert rig.drivers[1].closed
    assert rig.events()["inst-cam1"].confidence == "manual"


def test_disabled_camera_is_skipped(make_rig) -> None:
    rig = make_rig()

    async def scenario():
        await rig.ctl.connect()
        rig.ctl.set_enabled("cam2", False)
        report = await rig.ctl.arm()
        assert report.ready
        return await rig.ctl.start_capture()

    report = run(scenario())
    assert set(report.outcomes) == {"cam0", "cam1"}
    assert "inst-cam2" not in rig.events()


def test_events_and_snapshot_describe_the_same_state(make_rig) -> None:
    rig = make_rig()

    async def scenario():
        async with rig.ctl.bus.subscribe() as queue:
            await _start(rig)
            await rig.ctl.stop_capture()
            seen = []
            while not queue.empty():
                seen.append(queue.get_nowait())
        return seen

    seen = run(scenario())
    seqs = [e.seq for e in seen]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    phases = [e.data["phase"] for e in seen if e.kind == "phase"]
    assert phases == ["arming", "armed", "starting", "recording", "stopping", "stopped"]
    snap = rig.ctl.snapshot()
    assert snap["phase"] == "stopped" and snap["seq"] == seqs[-1]
    last_state = {}
    for e in seen:
        if e.kind == "camera":
            last_state[e.data["label"]] = e.data["state"]
    assert last_state == {k: v["state"] for k, v in snap["cameras"].items()}


def test_manual_driver_cameras_work_in_the_controller(tmp_path: Path) -> None:
    session = create_session(tmp_path / "m.db")
    sid = create_mocap_session(session, location="x")
    ctl = CaptureController([CameraSlot("old", "inst-old", ManualDriver("old"))], session, sid)

    async def scenario():
        await ctl.connect()
        assert (await ctl.arm()).ready
        await ctl.start_capture()
        return await ctl.stop_capture()

    assert run(scenario()).ok
    (ev,) = get_capture_events(session, ctl.capture_id)
    assert ev.confidence == "manual" and ev.status == "ok"
    session.close()
