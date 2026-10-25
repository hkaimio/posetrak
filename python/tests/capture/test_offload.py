# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests of copying recordings off a Hero13 (against the fake camera)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("httpx")

from click.testing import CliRunner  # noqa: E402

from pipeline.capture import daemon  # noqa: E402
from pipeline.capture.cli import main  # noqa: E402
from pipeline.capture.collect import already_copied, offload_camera, sidecar_path, target_path  # noqa: E402
from pipeline.capture.devices import ManualDriver, OffloadError  # noqa: E402
from pipeline.capture.gopro_cohn import CohnEndpoint, GoProCohnDriver  # noqa: E402
from pipeline.capture.registry_link import find_camera, link_gopro  # noqa: E402
from posetrak.db.db import create_camera_instance, create_camera_model, create_registry, list_camera_models  # noqa: E402
from tests.capture.fake_gopro import FakeGoPro  # noqa: E402
from tests.capture.test_gopro_cohn import driver_for, run  # noqa: E402

CLIP_A = bytes(range(256)) * 4000  # ~1 MB
CLIP_B = b"second clip" * 50_000
INFO_A = {"w": "3840", "h": "2160", "fov": "linear", "cre": "1700000000"}


@pytest.fixture()
def fake():
    with FakeGoPro("1234") as f:
        f.media = {
            "100GOPRO/GX010001.MP4": (CLIP_A, 1_700_000_000, INFO_A),
            "100GOPRO/GX010002.MP4": (CLIP_B, 1_700_000_100, {"w": "1920", "h": "1080"}),
            "100GOPRO/GOPR0003.JPG": (b"jpeg", 1_700_000_200, {}),
        }
        yield f


def test_the_media_list_is_read(fake: FakeGoPro) -> None:
    async def go():
        async with driver_for(fake) as d:
            files = await d.list_media()
            assert d.supports_offload
            by_path = {f.path: f for f in files}
            assert set(by_path) == set(fake.media)
            clip = by_path["100GOPRO/GX010001.MP4"]
            assert (clip.name, clip.size, clip.created) == ("GX010001.MP4", len(CLIP_A), 1_700_000_000)
            assert (await d.media_info(clip)) == INFO_A

    run(go())


def test_videos_are_copied_with_a_sidecar_and_not_copied_twice(fake: FakeGoPro, tmp_path: Path) -> None:
    async def go():
        async with driver_for(fake) as d:
            seen = []
            first = await offload_camera(d, tmp_path, progress=lambda f, done, total: seen.append((f.path, done, total)))
            second = await offload_camera(d, tmp_path)
            return first, second, seen

    first, second, seen = run(go())
    assert [r.file.path for r in first] == ["100GOPRO/GX010001.MP4", "100GOPRO/GX010002.MP4"]  # the photo is left
    target = target_path(tmp_path, "cam", first[0].file)
    assert target == tmp_path / "cam" / "100GOPRO" / "GX010001.MP4" and target.read_bytes() == CLIP_A
    side = json.loads(sidecar_path(target).read_text(encoding="utf-8"))
    assert side["sha256"] == hashlib.sha256(CLIP_A).hexdigest() == first[0].sha256
    assert (side["camera"], side["camera_path"], side["created"], side["camera_info"]) == (
        "cam", "100GOPRO/GX010001.MP4", 1_700_000_000, INFO_A,
    )
    assert not list(tmp_path.rglob("*.part"))
    assert seen[-1][1] == seen[-1][2] and {p for p, *_ in seen} == {"100GOPRO/GX010001.MP4", "100GOPRO/GX010002.MP4"}
    assert all(r.skipped for r in second) and len(second) == 2


def test_selecting_recordings(fake: FakeGoPro, tmp_path: Path) -> None:
    async def go():
        async with driver_for(fake) as d:
            newest = await offload_camera(d, tmp_path / "a", last=1)
            everything = await offload_camera(d, tmp_path / "b", include_all=True)
            nothing = await offload_camera(d, tmp_path / "c", last=0)
            return newest, everything, nothing

    newest, everything, nothing = run(go())
    assert [r.file.name for r in newest] == ["GX010002.MP4"]
    assert [r.file.name for r in everything] == ["GX010001.MP4", "GX010002.MP4", "GOPR0003.JPG"]
    assert nothing == []


def test_an_interrupted_copy_resumes(fake: FakeGoPro, tmp_path: Path) -> None:
    fake.cut_download_at = 300_000

    async def go():
        async with driver_for(fake) as d:
            with pytest.raises(OffloadError, match="GX010001.MP4 failed after"):
                await offload_camera(d, tmp_path)
            part = target_path(tmp_path, "cam", (await d.list_media())[0]).with_name("GX010001.MP4.part")
            assert 0 < part.stat().st_size < len(CLIP_A)
            return await offload_camera(d, tmp_path)

    results = run(go())
    first = results[0]
    assert first.path.read_bytes() == CLIP_A and first.sha256 == hashlib.sha256(CLIP_A).hexdigest()


def test_a_copy_of_the_wrong_size_is_rejected(fake: FakeGoPro, tmp_path: Path) -> None:
    async def go():
        async with driver_for(fake) as d:
            file = (await d.list_media())[0]
            from dataclasses import replace

            with pytest.raises(OffloadError, match="announced"):
                await d.download(replace(file, size=file.size + 1), tmp_path / "x.mp4")
            assert not (tmp_path / "x.mp4").exists()

    run(go())


def test_a_partial_file_is_not_a_copy(fake: FakeGoPro, tmp_path: Path) -> None:
    async def go():
        async with driver_for(fake) as d:
            file = (await d.list_media())[0]
            target = target_path(tmp_path, "cam", file)
            target.parent.mkdir(parents=True)
            target.write_bytes(CLIP_A[:100])
            assert not already_copied(target, file)  # no sidecar
            sidecar_path(target).write_text("{}", encoding="utf-8")
            assert not already_copied(target, file)  # wrong size
            target.write_bytes(CLIP_A)
            assert already_copied(target, file)

    run(go())


def test_cameras_that_cannot_be_offloaded_say_so() -> None:
    async def go():
        driver = ManualDriver("phone")
        assert not driver.supports_offload
        with pytest.raises(OffloadError, match="cannot be offloaded"):
            await driver.list_media()

    run(go())


# -- command --------------------------------------------------------------------------------------


@pytest.fixture()
def cli(tmp_path: Path, fake: FakeGoPro, monkeypatch):
    reg = create_registry(tmp_path / "registry.db")
    create_camera_model(reg, manufacturer="GoPro", model_name="Hero13")
    create_camera_instance(reg, list_camera_models(reg)[0]["id"], label="gopro13_01")
    link_gopro(reg, find_camera(reg, "gopro13_01"), "4609")
    reg.close()
    config = tmp_path / "config.toml"
    config.write_text(f'[daemon]\ntoken = "t"\nregistry_db = "{(tmp_path / "registry.db").as_posix()}"\n', encoding="utf-8")

    def fake_driver(cam, cfg=None, store=None):
        return GoProCohnDriver(cam.label, [CohnEndpoint("home", fake.base_url)], poll_interval_s=0.05, request_timeout_s=1.0)

    monkeypatch.setattr(daemon, "build_driver", fake_driver)

    def invoke(*args: str):
        return CliRunner().invoke(main, ["--config", str(config), *args])

    return invoke


def test_the_offload_command_lists_and_copies(cli, tmp_path: Path) -> None:
    listed = cli("offload", "gopro13_01", "--list")
    assert listed.exit_code == 0, listed.output
    assert "GX010001.MP4" in listed.output and "GOPR0003.JPG" not in listed.output

    dest = tmp_path / "out"
    copied = cli("offload", "gopro13_01", "--dest", str(dest))
    assert copied.exit_code == 0, copied.output
    assert (dest / "gopro13_01" / "100GOPRO" / "GX010001.MP4").read_bytes() == CLIP_A
    assert "GX010001.MP4" in copied.output and "MB" in copied.output

    again = cli("offload", "gopro13_01", "--dest", str(dest))
    assert again.output.count("already copied") == 2


def test_the_offload_command_reports_unknown_cameras(cli) -> None:
    result = cli("offload", "nope")
    assert result.exit_code != 0 and "not in the camera registry" in result.output
