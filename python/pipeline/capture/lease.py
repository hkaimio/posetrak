# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""The control lease: at most one client at a time may send capture commands.

Any number of clients (phones, the desktop app) can watch a capture. Only the
lease holder's arm / start / stop / lighting commands are accepted, so two
phones cannot send contradictory commands. A lease lapses when its holder stops
refreshing it, so a phone that fell off the WLAN does not block the operator
who picks up another device; taking a lease from a live holder is an explicit
action and is logged.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class LeaseInfo:
    holder: str | None
    label: str | None
    expires_in_s: float | None


@dataclass(frozen=True)
class AcquireResult:
    granted: bool
    info: LeaseInfo
    #: set when a live holder was displaced by an explicit takeover
    taken_from: str | None = None


class ControlLease:
    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S, clock: Callable[[], float] = time.monotonic) -> None:
        self._timeout = timeout_s
        self._clock = clock
        self._holder: str | None = None
        self._label: str | None = None
        self._expires = 0.0

    def _expire_if_due(self) -> bool:
        """Drop an expired lease; return True if one was dropped."""
        if self._holder is not None and self._clock() >= self._expires:
            logger.info("control lease of %s (%s) expired", self._holder, self._label)
            self._holder = self._label = None
            return True
        return False

    def sweep(self) -> bool:
        """Expire a lapsed lease; True if the lease changed."""
        return self._expire_if_due()

    def info(self) -> LeaseInfo:
        self._expire_if_due()
        if self._holder is None:
            return LeaseInfo(None, None, None)
        return LeaseInfo(self._holder, self._label, max(0.0, self._expires - self._clock()))

    def acquire(self, client_id: str, label: str | None = None, *, takeover: bool = False) -> AcquireResult:
        """Take the lease, refresh it if already held, or fail if another client holds it."""
        self._expire_if_due()
        taken_from = None
        if self._holder not in (None, client_id):
            if not takeover:
                return AcquireResult(False, self.info())
            taken_from = self._holder
            logger.warning("control lease taken over by %s (%s) from %s", client_id, label, taken_from)
        self._holder, self._label = client_id, label or self._label
        self._expires = self._clock() + self._timeout
        return AcquireResult(True, self.info(), taken_from)

    def release(self, client_id: str) -> bool:
        self._expire_if_due()
        if self._holder != client_id:
            return False
        self._holder = self._label = None
        return True

    def holds(self, client_id: str | None) -> bool:
        """True if *client_id* currently holds the lease; a valid check refreshes it."""
        self._expire_if_due()
        if client_id is None or client_id != self._holder:
            return False
        self._expires = self._clock() + self._timeout
        return True
