# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Camera driver interface and the types drivers exchange with the controller.

A driver wraps one physical camera and one way of controlling it. Drivers
report expected failures (timeout, camera refused, link lost) in their result
objects instead of raising, so the controller can fan a command out to many
cameras and always get a per-camera outcome back. Timestamps are ISO-8601 UTC
strings on the controller clock (:func:`posetrak.db.capture_control.utc_now_iso`).
"""

from __future__ import annotations

import enum
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from posetrak.db.capture_control import utc_now_iso

logger = logging.getLogger(__name__)


class DeviceState(enum.StrEnum):
    """Lifecycle of one camera during a capture."""

    OFFLINE = "offline"
    DISCOVERED = "discovered"
    READY = "ready"
    ARMED = "armed"
    RECORDING = "recording"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAULT = "fault"


class LinkState(enum.StrEnum):
    UP = "up"
    DOWN = "down"
    #: the camera is reachable, but on a different network than the controller
    OTHER_NETWORK = "other_network"


@dataclass(frozen=True)
class DeviceStatus:
    """Snapshot of what is known about one camera."""

    state: DeviceState = DeviceState.OFFLINE
    link: LinkState = LinkState.DOWN
    #: why the device is in ``DeviceState.FAULT``, if it is
    fault: str | None = None
    battery_pct: int | None = None
    storage_remaining_s: int | None = None
    overheating: bool = False
    #: True when the camera reports it is recording (None: cannot tell)
    recording: bool | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandResult:
    """Outcome of a start or stop command on one camera.

    ``ok`` means the camera did what was asked as far as the driver can tell.
    ``confirm_ts`` is set only when the camera positively reported the new
    state; a driver whose camera type cannot confirm leaves it None and has a
    ``confidence`` other than ``"confirmed"``.
    """

    ok: bool
    cmd_ts: str
    ack_ts: str | None = None
    confirm_ts: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ArmResult:
    """Outcome of pre-flight on one camera."""

    ok: bool
    #: human-readable reasons why the camera is not ready (empty when ok)
    problems: tuple[str, ...] = ()
    #: settings that were pushed but did not read back as requested
    mismatches: dict[str, tuple[Any, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ModeSpec:
    """A registry capture mode as a driver sees it."""

    id: str
    name: str | None
    width_px: int
    height_px: int
    fps: float
    #: the mode's ``capture_settings``: the driver's own settings vocabulary
    capture_settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SettingSpec:
    """One setting a driver understands, for building settings editors."""

    name: str
    #: allowed values; None when any value of the kind is accepted
    choices: tuple[Any, ...] | None = None
    description: str = ""


@dataclass(frozen=True)
class MediaFile:
    """A recording on a camera's storage."""

    #: location on the camera, ``"100GOPRO/GX010001.MP4"``
    path: str
    #: size in bytes; None if the camera does not say
    size: int | None = None
    #: creation time on the camera's own clock, seconds since the epoch; None if unknown
    created: int | None = None

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class Downloaded:
    """A file copied from a camera."""

    path: Path
    size: int
    sha256: str


class OffloadError(RuntimeError):
    """Media could not be listed or copied from a camera."""


#: ``(bytes done, bytes total or None)``
ProgressCallback = Callable[[int, "int | None"], None]

StatusCallback = Callable[[str, DeviceStatus], None]


class CameraDriver(ABC):
    """One camera and one way of controlling it.

    Parameters
    ----------
    label:
        The ``camera_instances.label`` of the camera; used to identify it in
        logs, events and the API.
    """

    #: ``"confirmed"``, ``"commanded"`` or ``"manual"``; see capture_events.confidence
    confidence: str = "commanded"
    #: ``"driver"``: :meth:`arm` applies the settings and reads them back. ``"manual"``: the
    #: camera cannot be configured remotely, the operator sets it by hand and confirms it.
    settings_control: str = "driver"

    def __init__(self, label: str) -> None:
        self.label = label
        self._status = DeviceStatus()
        self._callback: StatusCallback | None = None

    @property
    def status(self) -> DeviceStatus:
        return self._status

    def on_status(self, callback: StatusCallback | None) -> None:
        """Register the function called with ``(label, status)`` on every change."""
        self._callback = callback

    def _set_status(self, status: DeviceStatus) -> None:
        if status == self._status:
            return
        self._status = status
        if self._callback is not None:
            try:
                self._callback(self.label, status)
            except Exception:  # noqa: BLE001  (a bad listener must not break the driver)
                logger.exception("%s: status callback failed", self.label)

    #: True when the driver can list and copy the camera's recordings
    supports_offload: bool = False

    async def list_media(self) -> list[MediaFile]:
        """The recordings on the camera."""
        raise OffloadError(f"{self.label}: this camera type cannot be offloaded over the link")

    async def media_info(self, file: MediaFile) -> dict[str, Any]:
        """Metadata the camera keeps about a recording; empty if it has none."""
        return {}

    async def download(self, file: MediaFile, dest: Path, progress: ProgressCallback | None = None) -> Downloaded:
        """Copy a recording to *dest*, resuming a partial copy left by an earlier attempt.

        Raises
        ------
        OffloadError
            If the copy fails or is not the size the camera announced.
        """
        raise OffloadError(f"{self.label}: this camera type cannot be offloaded over the link")

    def supported_settings(self) -> dict[str, SettingSpec]:
        """The settings this driver understands, by name (empty if it takes none)."""
        return {}

    def mode_settings(self, mode: ModeSpec) -> dict[str, Any]:
        """The settings that make the camera record in *mode*.

        The default is the mode's own ``capture_settings``. A driver that can derive
        settings from the mode's size and frame rate does so and lets ``capture_settings``
        override what it derived.
        """
        return dict(mode.capture_settings)

    def describe_settings(self, settings: dict[str, Any]) -> list[str]:
        """Instructions for setting the camera by hand (for ``settings_control == "manual"``)."""
        return [f"{name}: {value}" for name, value in settings.items()]

    @abstractmethod
    async def connect(self) -> None:
        """Bring the link up and keep it up (reconnecting) until :meth:`close`."""

    @abstractmethod
    async def arm(self, settings: dict[str, Any]) -> ArmResult:
        """Run pre-flight checks and apply *settings*, reading them back."""

    @abstractmethod
    async def start(self) -> CommandResult:
        """Start recording."""

    @abstractmethod
    async def stop(self) -> CommandResult:
        """Stop recording."""

    @abstractmethod
    async def close(self) -> None:
        """Release the link and stop background work."""


class ManualDriver(CameraDriver):
    """A camera operated by hand.

    Used for cameras without remote control, and as the fallback when an
    automated driver is broken: the operator starts and stops the camera and
    confirms it in the UI. Commands succeed without a camera round trip and
    carry no confirmation.
    """

    confidence = "manual"
    settings_control = "manual"

    async def connect(self) -> None:
        self._set_status(DeviceStatus(state=DeviceState.READY, link=LinkState.UP))

    async def arm(self, settings: dict[str, Any]) -> ArmResult:
        self._set_status(DeviceStatus(state=DeviceState.ARMED, link=LinkState.UP))
        return ArmResult(ok=True)

    async def start(self) -> CommandResult:
        ts = utc_now_iso()
        self._set_status(DeviceStatus(state=DeviceState.RECORDING, link=LinkState.UP))
        return CommandResult(ok=True, cmd_ts=ts, ack_ts=ts)

    async def stop(self) -> CommandResult:
        ts = utc_now_iso()
        self._set_status(DeviceStatus(state=DeviceState.STOPPED, link=LinkState.UP))
        return CommandResult(ok=True, cmd_ts=ts, ack_ts=ts)

    async def close(self) -> None:
        self._set_status(DeviceStatus())
