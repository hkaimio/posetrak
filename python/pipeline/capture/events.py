# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""In-process event bus for the capture controller.

The controller publishes every state change here; the API layer subscribes to
stream them to clients. A subscriber that falls behind loses its oldest
events rather than blocking the controller, and is expected to recover by
asking for a fresh snapshot (events carry a sequence number so a gap is
detectable).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from posetrak.db.capture_control import utc_now_iso


@dataclass(frozen=True)
class Event:
    """One thing that happened: a camera state change, a phase change or an alert."""

    seq: int
    kind: str
    ts: str
    data: dict[str, Any] = field(default_factory=dict)


class EventBus:
    def __init__(self, queue_size: int = 256) -> None:
        self._queue_size = queue_size
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._seq = 0

    @property
    def seq(self) -> int:
        """Sequence number of the last published event."""
        return self._seq

    def publish(self, kind: str, /, **data: Any) -> Event:
        self._seq += 1
        event = Event(self._seq, kind, utc_now_iso(), data)
        for queue in self._subscribers:
            if queue.full():
                queue.get_nowait()  # drop the oldest; the subscriber sees a seq gap
            queue.put_nowait(event)
        return event

    @contextlib.asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[Event]]:
        queue: asyncio.Queue[Event] = asyncio.Queue(self._queue_size)
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)
