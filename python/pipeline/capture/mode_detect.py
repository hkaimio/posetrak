# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Telling which registry capture mode a video file was recorded in.

The file alone is enough, no camera has to be connected. What a file says about
how it was recorded is read with ``exiftool`` (image size, frame rate, and for a
GoPro the model, serial number and field of view), and matched against the
registry modes of the camera's model. Every mode comes with the evidence for and
against it, so that an ambiguous file can be shown to the operator as a choice.

A match needs the image size and the frame rate to agree (the frame rate within
one percent, since cameras record 119.88 fps for a nominal 120). A mode that
states a lens agrees when the file's field of view is that lens; a file or a
mode that does not say leaves the match short of certain.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.capture.devices import ModeSpec
from pipeline.capture.modes import load_modes

#: relative difference up to which two frame rates are the same mode
FPS_TOLERANCE = 0.01


class ProbeError(RuntimeError):
    """The file could not be read."""


@dataclass(frozen=True)
class VideoFacts:
    """What a video file says about how it was recorded."""

    path: Path
    width: int
    height: int
    #: frames per second the camera recorded at
    fps: float
    make: str | None = None
    model: str | None = None
    #: full manufacturer serial number of the camera body
    serial_number: str | None = None
    #: field of view or lens the camera reports, as written in the file ("Linear", "Super View")
    lens: str | None = None
    duration_s: float | None = None
    #: other recording settings the file states, for display (stabilization, digital zoom)
    extra: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModeMatch:
    """How well one registry mode explains a file."""

    mode: ModeSpec
    #: 1.0 when everything the file and the mode say agrees; less when a part could not be checked
    confidence: float
    #: what agrees
    evidence: tuple[str, ...]
    #: what could not be checked
    unchecked: tuple[str, ...] = ()


@dataclass(frozen=True)
class Detection:
    """The modes that fit a file, best first."""

    facts: VideoFacts
    matches: tuple[ModeMatch, ...]

    @property
    def unique(self) -> ModeMatch | None:
        """The match to assign without asking: one mode agrees with everything, and no other is as good."""
        certain = [m for m in self.matches if m.confidence >= 0.99]
        return certain[0] if len(certain) == 1 else None


def _norm(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def lens_key(text: str | None) -> str:
    """A lens or field of view name reduced to what compares equal: 'Super View' and 'superview' are one."""
    return _norm(text)


def _number(value: Any) -> float | None:
    try:
        return float(re.match(r"[\d.]+", str(value)).group())  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        return None


def _first(tags: dict[str, Any], *names: str) -> Any:
    """The first present tag among *names*, wherever its group is (``GoPro:Model`` matches ``Model``)."""
    for name in names:
        for key, value in tags.items():
            if key.split(":")[-1] == name and value not in (None, ""):
                return value
    return None


def read_video_facts(path: Path) -> VideoFacts:
    """Read the recording details of a video file with exiftool.

    Raises
    ------
    ProbeError
        If exiftool is missing, cannot read the file, or the file has no video size.
    """
    exe = shutil.which("exiftool")
    if exe is None:
        raise ProbeError("exiftool is not installed (https://exiftool.org); it is needed to read recording details")
    try:
        done = subprocess.run(
            [exe, "-j", "-G1", "-api", "LargeFileSupport=1", str(path)],
            capture_output=True, text=True, check=True, timeout=120,
        )
        tags = json.loads(done.stdout)[0]
    except (subprocess.SubprocessError, OSError, ValueError, IndexError) as exc:
        raise ProbeError(f"exiftool could not read {path}: {exc}") from exc
    width, height = _number(_first(tags, "ImageWidth")), _number(_first(tags, "ImageHeight"))
    fps = _number(_first(tags, "VideoFrameRate"))
    if not width or not height:
        raise ProbeError(f"{path} has no video track")
    model = _first(tags, "Model")
    make = _first(tags, "Make") or ("GoPro" if str(model or "").upper().startswith("HERO") else None)
    extra = {
        k: str(v)
        for k, v in (
            ("stabilization", _first(tags, "ElectronicImageStabilization")),
            ("digital_zoom", _first(tags, "DigitalZoom")),
        )
        if v is not None
    }
    return VideoFacts(
        path=path,
        width=int(width),
        height=int(height),
        fps=fps or 0.0,
        make=str(make) if make else None,
        model=str(model) if model else None,
        serial_number=str(_first(tags, "CameraSerialNumber", "SerialNumber") or "") or None,
        lens=str(_first(tags, "FieldOfView") or "") or None,
        duration_s=_number(_first(tags, "Duration")),
        extra=extra,
    )


def score_mode(facts: VideoFacts, mode: ModeSpec) -> ModeMatch | None:
    """How well *mode* explains *facts*, or None if the file contradicts it."""
    evidence: list[str] = []
    unchecked: list[str] = []
    if mode.width_px and mode.height_px:
        if (mode.width_px, mode.height_px) != (facts.width, facts.height):
            return None
        evidence.append(f"size {facts.width}x{facts.height}")
    else:
        unchecked.append("size")
    if mode.fps and facts.fps:
        if abs(facts.fps - mode.fps) / mode.fps > FPS_TOLERANCE:
            return None
        evidence.append(f"frame rate {facts.fps:g} fps (mode {mode.fps:g})")
    else:
        unchecked.append("frame rate")
    mode_lens = mode.capture_settings.get("lens")
    if mode_lens and facts.lens:
        if lens_key(str(mode_lens)) != lens_key(facts.lens):
            return None
        evidence.append(f"lens {facts.lens}")
    else:
        unchecked.append("lens" if not facts.lens else "lens (the mode does not state one)")
    confidence = 1.0 if not unchecked else 0.7 if len(unchecked) == 1 else 0.4
    return ModeMatch(mode, confidence, tuple(evidence), tuple(unchecked))


def detect_mode(facts: VideoFacts, modes: list[ModeSpec] | tuple[ModeSpec, ...]) -> Detection:
    """The modes among *modes* that fit *facts*, certain ones first."""
    matches = [m for m in (score_mode(facts, mode) for mode in modes) if m is not None]
    matches.sort(key=lambda m: (-m.confidence, m.mode.name or ""))
    return Detection(facts, tuple(matches))


# -- registry --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistryMatch:
    """Which registry camera and modes a file belongs to."""

    detection: Detection
    #: registry camera the serial number points at; None if it is not registered
    instance_id: str | None
    label: str | None
    camera_model_id: str | None
    model_name: str | None
    notes: tuple[str, ...] = ()


def _find_instance(registry: sqlite3.Connection, serial: str | None) -> sqlite3.Row | None:
    if not serial:
        return None
    row = registry.execute(
        "SELECT i.id, i.label, i.camera_model_id FROM camera_instances i WHERE i.serial_number = ?", (serial,)
    ).fetchone()
    if row is not None:
        return row
    # A camera linked for control before its serial was recorded is known by the end of it.
    for r in registry.execute(
        "SELECT i.id, i.label, i.camera_model_id, e.gopro_serial FROM control_endpoints e "
        "JOIN camera_instances i ON i.id = e.camera_instance_id WHERE e.gopro_serial IS NOT NULL"
    ):
        if serial.endswith(r["gopro_serial"]):
            return r
    return None


def match_registry(registry: sqlite3.Connection, facts: VideoFacts) -> RegistryMatch:
    """Find the registry camera a file came from and the registry modes that fit it.

    The camera is found by the serial number in the file. Without one, or with a serial that
    is not registered, the camera model decides which modes are considered.
    """
    instance = _find_instance(registry, facts.serial_number)
    notes: list[str] = []
    model_id = instance["camera_model_id"] if instance else None
    if model_id is None:
        wanted = _norm(facts.model)
        for model in registry.execute("SELECT id, manufacturer, model_name FROM camera_models"):
            name = _norm(model["model_name"])
            if wanted and name and name in wanted:
                model_id = model["id"]
                break
        if instance is None:
            notes.append(
                f"serial number {facts.serial_number} is not registered" if facts.serial_number else "no serial number in the file"
            )
    modes = load_modes(registry, model_id) if model_id else ()
    if model_id is None:
        notes.append(f"no registry camera model fits {facts.model or 'this file'}")
    model_name = None
    if model_id:
        row = registry.execute("SELECT model_name FROM camera_models WHERE id = ?", (model_id,)).fetchone()
        model_name = row[0] if row else None
    return RegistryMatch(
        detect_mode(facts, modes),
        instance["id"] if instance else None,
        instance["label"] if instance else None,
        model_id,
        model_name,
        tuple(notes),
    )
