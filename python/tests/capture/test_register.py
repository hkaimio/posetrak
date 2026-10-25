# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests of adding video files to a capture of a session."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.capture.mode_detect import VideoFacts
from pipeline.capture.register import RegisterError, VideoStatus, register_video
from posetrak.db.db import (
    create_camera_instance,
    create_camera_mode,
    create_camera_model,
    create_capture,
    create_mocap_session,
    create_registry,
    create_session,
)

SERIAL = "C3531325084609"


@pytest.fixture
def world(tmp_path: Path):
    registry = create_registry(tmp_path / "registry.db")
    model = create_camera_model(registry, manufacturer="GoPro", model_name="HERO 13 black")
    modes = {
        "linear": create_camera_mode(registry, model, width_px=3840, height_px=2160, nominal_fps=120,
                                     name="4K 120 linear", capture_settings={"lens": "linear"}),
        "super": create_camera_mode(registry, model, width_px=3840, height_px=2160, nominal_fps=120,
                                    name="4K 120 superview", capture_settings={"lens": "superview"}),
        "fhd": create_camera_mode(registry, model, width_px=1920, height_px=1080, nominal_fps=240,
                                  name="FHD 240", capture_settings={"lens": "linear"}),
    }
    camera = create_camera_instance(registry, model, label="gopro13_01", serial_number=SERIAL)
    calibration = "calib-4k-linear"
    with registry:
        registry.execute(
            "INSERT INTO intrinsics_calibrations (id, camera_mode_id, calibrated_at, fx, fy, cx, cy) "
            "VALUES (?, ?, '2026-10-01', 1800, 1800, 1920, 1080)",
            (calibration, modes["linear"]),
        )
        registry.execute(
            "UPDATE camera_modes SET default_intrinsics_calibration_id = ? WHERE id = ?", (calibration, modes["linear"])
        )
    session = create_session(tmp_path / "session.db")
    session_id = create_mocap_session(session, location="test")
    capture_id = create_capture(session, session_id, label="take 1")
    yield registry, session, session_id, capture_id, camera, modes, calibration, tmp_path
    registry.close()
    session.close()


def facts(path: Path, width=3840, height=2160, fps=119.88, lens="Linear") -> VideoFacts:
    return VideoFacts(path, width, height, fps, "GoPro", "HERO13 Black", SERIAL, lens, duration_s=2.0)


def add(world, name="clip.mp4", expected=None, **facts_kw):
    registry, session, session_id, capture_id, camera, modes, calibration, tmp = world
    path = tmp / name
    path.write_bytes(b"")
    return register_video(
        session, registry, session_id, capture_id, camera, path, expected_mode_id=expected, facts=facts(path, **facts_kw),
        source_path="100GOPRO/GX010001.MP4", sha256="abc", label="front",
    )


def row(world, video_id):
    return world[1].execute("SELECT * FROM capture_videos WHERE id = ?", (video_id,)).fetchone()


def test_a_file_that_agrees_with_one_mode_gets_that_mode_and_its_calibration(world) -> None:
    modes, calibration = world[5], world[6]
    video = add(world, expected=modes["linear"])
    assert (video.status, video.camera_mode_id, video.mode_name) == (VideoStatus.OK, modes["linear"], "4K 120 linear")
    stored = row(world, video.video_id)
    assert stored["camera_mode_id"] == modes["linear"] and stored["intrinsics_calibration_id"] == calibration
    assert (stored["first_video_frame"], stored["last_video_frame"]) == (0, 239)  # 2.0 s at 119.88 fps
    assert stored["actual_fps"] == pytest.approx(119.88)
    assert (stored["source_path"], stored["sha256"], stored["verify_status"]) == ("100GOPRO/GX010001.MP4", "abc", "ok")
    assert stored["collected_at"] and stored["verify_notes"] is None


def test_the_session_is_made_self_contained(world) -> None:
    registry, session, session_id, _, camera, modes, calibration, _ = world
    add(world)
    add(world, name="second.mp4")  # a second video of the same camera adds nothing new
    assert session.execute("SELECT label FROM camera_instances WHERE id = ?", (camera,)).fetchone()[0] == "gopro13_01"
    assert session.execute("SELECT name FROM camera_modes WHERE id = ?", (modes["linear"],)).fetchone()[0] == "4K 120 linear"
    assert session.execute("SELECT 1 FROM intrinsics_calibrations WHERE id = ?", (calibration,)).fetchone()
    assert session.execute("SELECT label FROM session_cameras WHERE session_id = ?", (session_id,)).fetchall()[0][0] == "front"
    assert session.execute("SELECT COUNT(*) FROM capture_videos").fetchone()[0] == 2


def test_a_file_that_contradicts_the_armed_mode_is_flagged_and_gets_the_mode_it_is(world) -> None:
    modes = world[5]
    video = add(world, expected=modes["super"])
    assert video.status is VideoStatus.MODE_MISMATCH and video.camera_mode_id == modes["linear"]
    assert "armed with 4K 120 superview, but the file is 4K 120 linear" in video.notes
    assert row(world, video.video_id)["verify_status"] == "mode_mismatch"


def test_a_file_that_fits_no_registered_mode_is_not_given_the_armed_one(world) -> None:
    modes = world[5]
    video = add(world, expected=modes["linear"], width=5312, height=2988, fps=59.94)
    assert video.status is VideoStatus.MODE_MISMATCH and video.camera_mode_id is None
    assert "fits no registered mode" in video.notes
    unknown = add(world, name="other.mp4", width=5312, height=2988, fps=59.94)
    assert unknown.status is VideoStatus.MODE_UNKNOWN and unknown.camera_mode_id is None


def test_a_file_that_cannot_decide_takes_the_armed_mode_when_it_fits(world) -> None:
    modes = world[5]
    video = add(world, expected=modes["linear"], lens=None)  # linear and superview look alike without a lens
    assert video.status is VideoStatus.MODE_ASSUMED and video.camera_mode_id == modes["linear"]
    assert "does not state its lens" in video.notes
    undecided = add(world, name="b.mp4", lens=None)
    assert undecided.status is VideoStatus.MODE_UNKNOWN and "several modes" in undecided.notes


def test_a_file_that_cannot_be_read_still_gets_registered(world, monkeypatch) -> None:
    from pipeline.capture import register

    registry, session, session_id, capture_id, camera, modes, _, tmp = world
    path = tmp / "broken.mp4"
    path.write_bytes(b"")
    monkeypatch.setattr(register, "_frame_info", lambda p: (500, 120.0))
    monkeypatch.setattr(register, "read_video_facts", lambda p: (_ for _ in ()).throw(register.ProbeError("exiftool is not installed")))
    video = register_video(session, registry, session_id, capture_id, camera, path, expected_mode_id=modes["fhd"])
    assert video.status is VideoStatus.UNREADABLE and video.camera_mode_id == modes["fhd"]
    assert "exiftool is not installed" in video.notes and "FHD 240" in video.notes
    stored = row(world, video.video_id)
    assert (stored["last_video_frame"], stored["actual_fps"]) == (499, 120.0)


def test_unusable_input_is_refused(world) -> None:
    registry, session, session_id, capture_id, _, _, _, tmp = world
    path = tmp / "x.mp4"
    path.write_bytes(b"")
    with pytest.raises(RegisterError, match="not in the registry"):
        register_video(session, registry, session_id, capture_id, "nobody", path, facts=facts(path))
    empty = VideoFacts(path, 3840, 2160, 119.88, "GoPro", None, SERIAL, "Linear")  # no duration, and not a video
    with pytest.raises(RegisterError, match="number of frames"):
        register_video(session, registry, session_id, capture_id, world[4], path, facts=empty)
