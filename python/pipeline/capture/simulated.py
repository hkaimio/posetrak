# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""A simulated camera for developing and demonstrating the capture UI.

Behaves like a confirmed-control camera with the response times measured on
Hero13s over COHN: START is acknowledged after about a second, STOP is
confirmed about a second and a half after the command.
"""

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


class SimulatedDriver(CameraDriver):
    confidence = "confirmed"

    def __init__(
        self,
        label: str,
        *,
        start_delay_s: float = 1.2,
        stop_delay_s: float = 1.6,
        battery_pct: int = 90,
    ) -> None:
        super().__init__(label)
        self.start_delay_s = start_delay_s
        self.stop_delay_s = stop_delay_s
        self._battery = battery_pct

    def _update(self, **changes: Any) -> None:
        self._set_status(replace(self._status, **changes))

    async def connect(self) -> None:
        self._set_status(
            DeviceStatus(
                state=DeviceState.READY,
                link=LinkState.UP,
                battery_pct=self._battery,
                storage_remaining_s=3 * 3600,
                recording=False,
            )
        )

    async def arm(self, settings: dict[str, Any]) -> ArmResult:
        await asyncio.sleep(0.3)
        return ArmResult(ok=True)

    async def start(self) -> CommandResult:
        cmd = utc_now_iso()
        await asyncio.sleep(self.start_delay_s)
        self._update(recording=True)
        ts = utc_now_iso()
        return CommandResult(True, cmd, ack_ts=ts, confirm_ts=ts)

    async def stop(self) -> CommandResult:
        cmd = utc_now_iso()
        await asyncio.sleep(0.25)
        ack = utc_now_iso()
        await asyncio.sleep(self.stop_delay_s)
        self._update(recording=False)
        return CommandResult(True, cmd, ack_ts=ack, confirm_ts=utc_now_iso())

    async def close(self) -> None:
        self._set_status(DeviceStatus())
