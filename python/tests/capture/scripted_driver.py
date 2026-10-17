# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""An in-memory camera driver with programmable behaviour for controller tests."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from pipeline.capture.devices import (
    ArmResult,
    CameraDriver,
    CommandResult,
    DeviceState,
    DeviceStatus,
    LinkState,
)
from posetrak.db.capture_control import utc_now_iso


class ScriptedDriver(CameraDriver):
    """Behaves like a camera, as scripted by the test.

    Attributes set by tests: ``arm_result``, ``start_delay_s`` / ``stop_delay_s``,
    ``start_error`` / ``stop_error`` (command reports failure), ``start_raises``
    (command raises), ``confirm`` (whether commands return a confirmation
    timestamp), and ``recording`` (what the "camera" is doing).
    """

    def __init__(self, label: str, confidence: str = "confirmed") -> None:
        super().__init__(label)
        self.confidence = confidence
        self.arm_result = ArmResult(ok=True)
        self.start_delay_s = 0.0
        self.stop_delay_s = 0.0
        self.start_error: str | None = None
        self.stop_error: str | None = None
        self.start_raises: Exception | None = None
        self.confirm = confidence == "confirmed"
        self.recording = False
        self.calls: list[str] = []
        self.armed_with: dict[str, Any] | None = None
        self.closed = False

    # -- test controls -----------------------------------------------------

    def push_status(self, **changes: Any) -> None:
        """Change the reported status as a real driver's poller would."""
        self._set_status(replace(self._status, **changes))

    # -- CameraDriver ------------------------------------------------------

    async def connect(self) -> None:
        self.calls.append("connect")
        self._set_status(
            DeviceStatus(
                state=DeviceState.READY,
                link=LinkState.UP,
                battery_pct=90,
                storage_remaining_s=7200,
                recording=False,
            )
        )

    async def arm(self, settings: dict[str, Any]) -> ArmResult:
        self.calls.append("arm")
        self.armed_with = dict(settings)
        return self.arm_result

    async def start(self) -> CommandResult:
        self.calls.append("start")
        cmd = utc_now_iso()
        if self.start_raises is not None:
            raise self.start_raises
        await asyncio.sleep(self.start_delay_s)
        if self.start_error:
            return CommandResult(False, cmd, error=self.start_error)
        self.recording = True
        self.push_status(recording=True)
        ts = utc_now_iso()
        return CommandResult(True, cmd, ack_ts=ts, confirm_ts=ts if self.confirm else None)

    async def stop(self) -> CommandResult:
        self.calls.append("stop")
        cmd = utc_now_iso()
        await asyncio.sleep(self.stop_delay_s)
        if self.stop_error:
            return CommandResult(False, cmd, error=self.stop_error)
        self.recording = False
        self.push_status(recording=False)
        ts = utc_now_iso()
        return CommandResult(True, cmd, ack_ts=ts, confirm_ts=ts if self.confirm else None)

    async def close(self) -> None:
        self.closed = True
