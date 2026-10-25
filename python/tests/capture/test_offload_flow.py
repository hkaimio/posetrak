# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests of copying a capture's recordings off the cameras and registering them after it stops."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pipeline.capture import register
from pipeline.capture.config import SetupConfig
from pipeline.capture.mode_detect import VideoFacts
from pipeline.capture.modes import ModePlanner, PlannedCamera, load_modes, model_keys
from pipeline.capture.service import CameraSlot, CaptureController, CapturePhase, CaptureStateError
from posetrak.db.db import (
    create_camera_instance,
    create_camera_mode,
    create_camera_model,
    create_mocap_session,
    create_registry,
    create_session,
)
from tests.capture.scripted_driver import ScriptedDriver

SERIAL = "C3531325084609"


def run(coro):
    return asyncio.run(coro)


class Rig:
    """Two cameras with a card each, a registry with one mode and a session database."""

    def __init__(self, tmp_path: Path, monkeypatch, *, auto=True, lens="Linear", with_offload=True) -> None:
        self.tmp = tmp_path
        self.registry_path = tmp_path / "registry.db"
        registry = create_registry(self.registry_path)
        model = create_camera_model(registry, manufacturer="GoPro", model_name="Hero13")
        self.mode_id = create_camera_mode(
            registry, model, width_px=3840, height_px=2160, nominal_fps=120, name="4K 120 linear",
            capture_settings={"lens": "linear"},
        )
        ids = [create_camera_instance(registry, model, label=f"cam{i}") for i in range(2)]
        modes = load_modes(registry, model)
        registry.close()
        self.session = create_session(tmp_path / "session.db")
        session_id = create_mocap_session(self.session, location="test")
        self.drivers = [ScriptedDriver(f"cam{i}") for i in range(2)]
        for i, d in enumerate(self.drivers):
            d.supports_offload = True
            d.media = {"100GOPRO/OLD.MP4": b"from an earlier day"}
            d.recorded = {f"100GOPRO/GX01000{i}.MP4": f"clip {i}".encode() * 100}
        planner = ModePlanner(
            [PlannedCamera(d.label, model_keys("GoPro", "Hero13"), "GoPro Hero13", modes, d) for d in self.drivers],
            (SetupConfig("s", {"Hero13": "4K 120 linear"}),),
        )
        slots = [CameraSlot(d.label, ids[i], d) for i, d in enumerate(self.drivers)]
        self.videos = tmp_path / "videos"
        self.ctl = CaptureController(
            slots, self.session, session_id, planner=planner,
            offload_dir=self.videos if with_offload else None, registry_path=self.registry_path, auto_offload=auto,
        )
        monkeypatch.setattr(
            register, "read_video_facts",
            lambda path: VideoFacts(path, 3840, 2160, 119.88, "GoPro", "HERO13 Black", SERIAL, lens, duration_s=1.0),
        )

    async def take(self) -> None:
        await self.ctl.connect()
        assert (await self.ctl.arm()).ready
        await self.ctl.start_capture()
        await self.ctl.stop_capture()

    def videos_in_session(self):
        return self.session.execute(
            "SELECT * FROM capture_videos ORDER BY source_path"
        ).fetchall()


@pytest.fixture
def make(tmp_path: Path, monkeypatch):
    rigs = []

    def build(**kw) -> Rig:
        sub = tmp_path / f"rig{len(rigs)}"
        sub.mkdir()
        rig = Rig(sub, monkeypatch, **kw)
        rigs.append(rig)
        return rig

    yield build
    for r in rigs:
        r.session.close()


def test_the_new_recordings_are_copied_and_registered_after_the_capture_stops(make) -> None:
    rig = make()

    async def go():
        await rig.take()
        assert rig.ctl.snapshot()["offload"]["busy"] or rig.ctl.snapshot()["offload"]["cameras"]
        await rig.ctl._offloader.wait()

    run(go())
    copied = sorted(p.relative_to(rig.videos).as_posix() for p in rig.videos.rglob("*.MP4"))
    assert copied == ["capture-001/cam0/100GOPRO/GX010000.MP4", "capture-001/cam1/100GOPRO/GX010001.MP4"]  # not OLD.MP4
    rows = rig.videos_in_session()
    assert [r["source_path"] for r in rows] == ["100GOPRO/GX010000.MP4", "100GOPRO/GX010001.MP4"]
    assert all(r["camera_mode_id"] == rig.mode_id and r["verify_status"] == "ok" and r["sha256"] for r in rows)
    assert {r["shot_id"] for r in rows} == {rig.ctl.capture_id}
    view = rig.ctl.snapshot()["offload"]
    assert not view["busy"] and all(c["state"] == "done" and c["files_done"] == 1 for c in view["cameras"].values())
    assert rig.ctl.alerts == []


def test_nothing_is_copied_unless_asked_to(make) -> None:
    rig = make(auto=False)

    async def go():
        await rig.take()
        assert rig.ctl.snapshot()["offload"]["cameras"] == {} and not rig.videos.exists()
        rig.ctl.offload_now()
        await rig.ctl._offloader.wait()

    run(go())
    assert len(rig.videos_in_session()) == 2


def test_offloading_is_refused_where_it_does_not_apply(make) -> None:
    rig = make(with_offload=False)

    async def go():
        await rig.ctl.connect()
        with pytest.raises(CaptureStateError, match="offload"):
            rig.ctl.offload_now()  # nothing has stopped yet
        with pytest.raises(CaptureStateError, match="no directory"):
            rig.ctl.set_auto_offload(True)
        await rig.take()
        with pytest.raises(CaptureStateError, match="no directory"):
            rig.ctl.offload_now()

    run(go())
    assert rig.ctl.snapshot()["offload"]["available"] is False


def test_the_cameras_cannot_be_armed_while_videos_are_being_copied_and_a_copy_can_be_cancelled(make) -> None:
    rig = make()
    for d in rig.drivers:
        d.download_delay_s = 5.0

    async def go():
        await rig.take()
        await asyncio.sleep(0.05)
        assert rig.ctl.snapshot()["offload"]["busy"]
        with pytest.raises(CaptureStateError, match="being copied"):
            await rig.ctl.arm()
        rig.ctl.cancel_offload()
        await rig.ctl._offloader.wait()
        view = rig.ctl.snapshot()["offload"]
        assert not view["busy"] and {c["error"] for c in view["cameras"].values()} == {"cancelled"}
        for d in rig.drivers:
            d.download_delay_s = 0.0
        assert (await rig.ctl.arm()).ready  # armed again; the files are copied by the next run

    run(go())
    assert rig.videos_in_session() == []


def test_one_camera_failing_does_not_stop_the_other(make) -> None:
    rig = make()
    rig.drivers[0].fail_download = "the card was ejected"

    async def go():
        await rig.take()
        await rig.ctl._offloader.wait()

    run(go())
    cams = rig.ctl.snapshot()["offload"]["cameras"]
    assert cams["cam0"]["state"] == "failed" and "ejected" in cams["cam0"]["error"] and cams["cam1"]["state"] == "done"
    assert [r["source_path"] for r in rig.videos_in_session()] == ["100GOPRO/GX010001.MP4"]
    assert [(a.camera, a.kind) for a in rig.ctl.alerts] == [("cam0", "offload_failed")]


def test_a_camera_whose_earlier_recordings_were_not_listed_is_skipped(make) -> None:
    rig = make()

    async def go():
        await rig.ctl.connect()
        rig.drivers[0].supports_offload = False  # the list before the capture fails
        assert (await rig.ctl.arm()).ready
        rig.drivers[0].supports_offload = True
        await rig.ctl.start_capture()
        await rig.ctl.stop_capture()
        await rig.ctl._offloader.wait()

    run(go())
    cams = rig.ctl.snapshot()["offload"]["cameras"]
    assert cams["cam0"]["state"] == "skipped" and cams["cam1"]["state"] == "done"
    assert any(a.camera == "cam0" and a.kind == "offload_failed" for a in rig.ctl.alerts)


def test_a_video_in_another_mode_than_armed_raises_an_alert(make) -> None:
    rig = make(lens="Super View")

    async def go():
        await rig.take()
        await rig.ctl._offloader.wait()

    run(go())
    rows = rig.videos_in_session()
    assert all(r["verify_status"] == "mode_mismatch" and r["camera_mode_id"] is None for r in rows)
    kinds = {(a.camera, a.kind) for a in rig.ctl.alerts}
    assert kinds == {("cam0", "video_mode"), ("cam1", "video_mode")}
    assert all("armed with 4K 120 linear" in a.message for a in rig.ctl.alerts)


def test_progress_is_published_as_events(make) -> None:
    rig = make()
    seen = []
    rig.ctl.bus.publish = (  # type: ignore[method-assign]
        lambda kind, _orig=rig.ctl.bus.publish, **data: (seen.append((kind, data)), _orig(kind, **data))[1]
    )

    async def go():
        await rig.take()
        await rig.ctl._offloader.wait()

    run(go())
    offload = [d for k, d in seen if k == "offload"]
    assert offload[0]["busy"] and not offload[-1]["busy"]
    assert offload[-1]["cameras"]["cam0"]["state"] == "done" and offload[-1]["cameras"]["cam0"]["videos"][0]["status"] == "ok"
    assert rig.ctl.phase == CapturePhase.STOPPED


def test_a_missing_registry_is_reported_and_does_not_leave_the_copy_running(make) -> None:
    rig = make()
    rig.ctl._offloader._registry_path = rig.tmp / "no-such-registry.db"

    async def go():
        await rig.take()
        await rig.ctl._offloader.wait()

    run(go())
    view = rig.ctl.snapshot()["offload"]
    assert not view["busy"] and {c["state"] for c in view["cameras"].values()} == {"failed"}
    assert [(a.camera, a.kind) for a in rig.ctl.alerts] == [(None, "offload_failed")]
