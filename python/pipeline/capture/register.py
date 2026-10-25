# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Adding video files to a capture of a session.

A registered file becomes a ``capture_videos`` row with its capture mode and that mode's
default intrinsic calibration. The mode is what the file says it is
(:mod:`pipeline.capture.mode_detect`); the mode the camera was armed with only fills in
when the file is not clear, and a file that contradicts it is flagged instead of trusted.
The registry rows of the camera and the mode are copied into the session, so the session
stays usable without the registry.
"""

from __future__ import annotations

import datetime
import enum
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from pipeline.capture.mode_detect import ProbeError, VideoFacts, detect_mode, read_video_facts
from pipeline.capture.modes import load_modes
from posetrak.db.db import add_capture_video, copy_camera_rows


class RegisterError(RuntimeError):
    """A file cannot be registered."""


class VideoStatus(enum.StrEnum):
    #: the file agrees with exactly one mode, which is the armed one if there was one
    OK = "ok"
    #: the file fits no mode, or only modes other than the armed one
    MODE_MISMATCH = "mode_mismatch"
    #: the file could not decide between modes; the armed mode, which fits, was taken
    MODE_ASSUMED = "mode_assumed"
    #: the file could not be matched to a mode and none was armed
    MODE_UNKNOWN = "mode_unknown"
    #: the recording details of the file could not be read
    UNREADABLE = "unreadable"


@dataclass(frozen=True)
class RegisteredVideo:
    video_id: str
    path: Path
    camera_mode_id: str | None
    mode_name: str | None
    status: VideoStatus
    notes: str


def _frame_info(path: Path) -> tuple[int, float]:
    """``(frame count, fps)`` of a video as OpenCV reports them; zeros if it cannot open the file."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()


def register_video(
    session: sqlite3.Connection,
    registry: sqlite3.Connection,
    session_id: str,
    capture_id: str,
    camera_instance_id: str,
    path: Path,
    *,
    expected_mode_id: str | None = None,
    facts: VideoFacts | None = None,
    source_path: str | None = None,
    sha256: str | None = None,
    collected_at: str | None = None,
    label: str = "",
) -> RegisteredVideo:
    """Add a video file to a capture.

    Parameters
    ----------
    expected_mode_id:
        Registry mode the camera was armed with; None if the file was not recorded by the
        capture daemon.
    facts:
        What the file states about itself; read from the file if not given.
    source_path, sha256, collected_at:
        Provenance of a file copied off a camera.

    Raises
    ------
    RegisterError
        If the camera is not in the registry or the number of frames cannot be told.
    """
    instance = registry.execute(
        "SELECT camera_model_id FROM camera_instances WHERE id = ?", (camera_instance_id,)
    ).fetchone()
    if instance is None:
        raise RegisterError(f"camera {camera_instance_id} is not in the registry")
    read_error = None
    if facts is None:
        try:
            facts = read_video_facts(path)
        except ProbeError as exc:
            read_error = str(exc)
    frames, fps = _frame_info(path)
    if frames <= 0 and facts is not None and facts.duration_s and facts.fps:
        frames = round(facts.duration_s * facts.fps)
    fps = (facts.fps if facts is not None else 0.0) or fps
    if frames <= 0 or fps <= 0:
        raise RegisterError(f"cannot tell the number of frames of {path}")

    modes = load_modes(registry, instance["camera_model_id"])
    names = {m.id: m.name for m in modes}
    chosen, status, notes = _choose_mode(facts, modes, expected_mode_id, names, read_error)

    copy_camera_rows(session, registry, session_id, camera_instance_id, chosen, label=label)
    video_id = add_capture_video(session, capture_id, camera_instance_id, path.as_posix(), 0, frames - 1, fps)
    calibration = None
    if chosen is not None:
        row = registry.execute(
            "SELECT default_intrinsics_calibration_id FROM camera_modes WHERE id = ?", (chosen,)
        ).fetchone()
        calibration = row[0] if row else None
    with session:
        session.execute(
            "UPDATE capture_videos SET camera_mode_id = ?, intrinsics_calibration_id = ?, collected_at = ?, "
            "source_path = ?, sha256 = ?, verify_status = ?, verify_notes = ? WHERE id = ?",
            (
                chosen,
                calibration,
                collected_at or (datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds") if source_path else None),
                source_path,
                sha256,
                str(status),
                notes or None,
                video_id,
            ),
        )
    return RegisteredVideo(video_id, path, chosen, names.get(chosen), status, notes)


def _choose_mode(facts, modes, expected, names, read_error):  # type: ignore[no-untyped-def]
    """The mode to record for a file, how sure that is, and why."""
    expected_name = names.get(expected) if expected else None
    if facts is None:
        notes = read_error or "the recording details of the file could not be read"
        if expected:
            return expected, VideoStatus.UNREADABLE, f"{notes}; recorded as the armed mode {expected_name}"
        return None, VideoStatus.UNREADABLE, notes
    detection = detect_mode(facts, modes)
    found = detection.unique
    if found is not None:
        if expected and expected != found.mode.id:
            return (
                found.mode.id,
                VideoStatus.MODE_MISMATCH,
                f"armed with {expected_name}, but the file is {found.mode.name} ({', '.join(found.evidence)})",
            )
        return found.mode.id, VideoStatus.OK, ""
    fits = [m for m in detection.matches]
    if expected and any(m.mode.id == expected for m in fits):
        unsure = next(m for m in fits if m.mode.id == expected)
        return expected, VideoStatus.MODE_ASSUMED, f"the file does not state its {', '.join(unsure.unchecked)}; taken as the armed mode {expected_name}"
    if expected:
        what = "fits no registered mode" if not fits else "fits " + ", ".join(m.mode.name or "?" for m in fits)
        return None, VideoStatus.MODE_MISMATCH, f"armed with {expected_name}, but the file {what} ({facts.width}x{facts.height} at {facts.fps:g} fps, lens {facts.lens or 'not stated'})"
    if not fits:
        return None, VideoStatus.MODE_UNKNOWN, f"the file fits no registered mode ({facts.width}x{facts.height} at {facts.fps:g} fps, lens {facts.lens or 'not stated'})"
    return None, VideoStatus.MODE_UNKNOWN, "the file fits several modes: " + ", ".join(m.mode.name or "?" for m in fits)
