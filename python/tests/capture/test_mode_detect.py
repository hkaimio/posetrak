# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests of telling the capture mode of a video file from what the file says about itself."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from pipeline.capture import mode_detect
from pipeline.capture.cli import main
from pipeline.capture.devices import ModeSpec
from pipeline.capture.mode_detect import (
    ProbeError,
    VideoFacts,
    detect_mode,
    lens_key,
    match_registry,
    read_video_facts,
    score_mode,
)
from pipeline.capture.registry_link import find_camera, link_gopro
from posetrak.db.db import (
    create_camera_instance,
    create_camera_mode,
    create_camera_model,
    create_registry,
)

UHD_120_LINEAR = ModeSpec("m1", "4K 120 linear", 3840, 2160, 120, {"lens": "linear"})
UHD_120_SUPER = ModeSpec("m2", "4K 120 superview", 3840, 2160, 120, {"lens": "superview"})
FHD_240 = ModeSpec("m3", "FHD 240 linear", 1920, 1080, 240, {"lens": "linear"})
NO_LENS = ModeSpec("m4", "4K 120", 3840, 2160, 120, {})
MODES = (UHD_120_LINEAR, UHD_120_SUPER, FHD_240, NO_LENS)

SERIAL = "C3531325084609"


def facts(width=3840, height=2160, fps=119.88, lens="Linear", serial=SERIAL, model="HERO13 Black") -> VideoFacts:
    return VideoFacts(Path("clip.mp4"), width, height, fps, "GoPro", model, serial, lens)


# -- matching -------------------------------------------------------------------------------------


def test_the_size_rate_and_lens_pick_the_mode() -> None:
    found = detect_mode(facts(), MODES)
    assert [m.mode.name for m in found.matches] == ["4K 120 linear", "4K 120"]
    assert found.unique.mode.name == "4K 120 linear"
    assert found.matches[0].confidence == 1.0 and found.matches[1].confidence == 0.7
    assert "lens Linear" in found.matches[0].evidence
    assert "lens (the mode does not state one)" in found.matches[1].unchecked


def test_the_lens_tells_modes_of_the_same_size_apart() -> None:
    assert detect_mode(facts(lens="Super View"), MODES).unique.mode.name == "4K 120 superview"
    assert lens_key("Super View") == lens_key("superview") == lens_key("Super_View")


def test_a_frame_rate_must_agree_within_a_percent() -> None:
    assert score_mode(facts(fps=119.88), UHD_120_LINEAR) is not None
    assert score_mode(facts(fps=100.0), UHD_120_LINEAR) is None
    assert score_mode(facts(fps=239.76, width=1920, height=1080), FHD_240) is not None
    assert score_mode(facts(width=1920, height=1080), UHD_120_LINEAR) is None


def test_a_file_that_does_not_state_its_lens_is_not_certain() -> None:
    found = detect_mode(facts(lens=None), (UHD_120_LINEAR, UHD_120_SUPER))
    assert len(found.matches) == 2 and found.unique is None
    assert all(m.confidence == 0.7 and m.unchecked == ("lens",) for m in found.matches)


def test_unknown_sizes_leave_no_match() -> None:
    assert detect_mode(facts(width=999, height=999), MODES).matches == ()


# -- reading the file ----------------------------------------------------------------------------

EXIFTOOL = {
    "GoPro:Model": "HERO13 Black",
    "GoPro:CameraSerialNumber": SERIAL,
    "GoPro:FieldOfView": "Super View",
    "GoPro:DigitalZoom": "No",
    "GoPro:ElectronicImageStabilization": "HS EIS",
    "QuickTime:Duration": "7.86 s",
    "Track1:ImageWidth": 5312,
    "Track1:ImageHeight": 2988,
    "Track1:VideoFrameRate": 59.94,
    "UserData:FirmwareVersion": "H24.01.02.10.00",
}


def fake_exiftool(monkeypatch, tags=None, available=True, fail=False) -> None:
    monkeypatch.setattr(mode_detect.shutil, "which", lambda name: "exiftool" if available else None)

    def run(cmd, **kwargs):
        if fail:
            raise subprocess.CalledProcessError(1, cmd)
        assert "-j" in cmd and cmd[-1] == "clip.mp4"
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps([tags or EXIFTOOL]), stderr="")

    monkeypatch.setattr(mode_detect.subprocess, "run", run)


def test_the_recording_details_are_read_from_exiftool_output(monkeypatch) -> None:
    fake_exiftool(monkeypatch)
    got = read_video_facts(Path("clip.mp4"))
    assert (got.width, got.height, got.fps, got.model, got.make) == (5312, 2988, 59.94, "HERO13 Black", "GoPro")
    assert (got.serial_number, got.lens, got.duration_s) == (SERIAL, "Super View", 7.86)
    assert got.extra == {"stabilization": "HS EIS", "digital_zoom": "No"}


def test_unreadable_files_are_reported(monkeypatch) -> None:
    fake_exiftool(monkeypatch, available=False)
    with pytest.raises(ProbeError, match="not installed"):
        read_video_facts(Path("clip.mp4"))
    fake_exiftool(monkeypatch, fail=True)
    with pytest.raises(ProbeError, match="could not read"):
        read_video_facts(Path("clip.mp4"))
    fake_exiftool(monkeypatch, tags={"GoPro:Model": "HERO13 Black"})
    with pytest.raises(ProbeError, match="no video track"):
        read_video_facts(Path("clip.mp4"))


# -- registry -------------------------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path: Path):
    reg = create_registry(tmp_path / "registry.db")
    model = create_camera_model(reg, manufacturer="GoPro", model_name="HERO 13 black")
    for mode in MODES:
        create_camera_mode(
            reg,
            model,
            width_px=mode.width_px,
            height_px=mode.height_px,
            nominal_fps=mode.fps,
            name=mode.name,
            capture_settings=mode.capture_settings,
        )
    yield reg, model
    reg.close()


def test_the_camera_is_found_by_the_serial_number(registry) -> None:
    reg, model = registry
    create_camera_instance(reg, model, label="gopro13_01", serial_number=SERIAL)
    found = match_registry(reg, facts())
    assert (found.label, found.model_name) == ("gopro13_01", "HERO 13 black")
    assert found.detection.unique.mode.name == "4K 120 linear" and found.notes == ()


def test_a_camera_linked_before_its_serial_was_recorded_is_found_too(registry) -> None:
    reg, model = registry
    create_camera_instance(reg, model, label="gopro13_01")
    link_gopro(reg, find_camera(reg, "gopro13_01"), "4609")
    assert match_registry(reg, facts()).label == "gopro13_01"


def test_an_unregistered_camera_still_gets_the_modes_of_its_model(registry) -> None:
    reg, _ = registry
    found = match_registry(reg, facts(serial="C0000000009999"))
    assert found.label is None and "not registered" in found.notes[0]
    assert found.detection.unique.mode.name == "4K 120 linear"
    nothing = match_registry(reg, facts(serial=None, model="Pixel 9"))
    assert nothing.detection.matches == () and any("no registry camera model" in n for n in nothing.notes)


def test_the_detect_mode_command(registry, tmp_path: Path, monkeypatch) -> None:
    reg, model = registry
    create_camera_instance(reg, model, label="gopro13_01", serial_number=SERIAL)
    config = tmp_path / "config.toml"
    config.write_text(f'[daemon]\ntoken = "t"\nregistry_db = "{(tmp_path / "registry.db").as_posix()}"\n', encoding="utf-8")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")
    monkeypatch.setattr(mode_detect, "read_video_facts", lambda path: facts(lens=None))
    result = CliRunner().invoke(main, ["--config", str(config), "detect-mode", str(clip)])
    assert result.exit_code == 0, result.output
    assert "gopro13_01" in result.output and "no lens in the file" in result.output
    assert "4K 120 linear - unsure (lens not checked)" in result.output
