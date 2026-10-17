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
