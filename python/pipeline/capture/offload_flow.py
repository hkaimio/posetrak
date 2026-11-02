# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Copying a capture's recordings off the cameras and adding them to the capture.

Which files belong to a capture is told by the cameras' own media lists, not by clocks: a
camera's list is read when the cameras are armed and again after they have stopped, and the
videos that appeared in between are the capture's. The copies go to
``<destination>/capture-<number>/<camera label>/<camera folder>/``, and each one is
registered in the capture with its capture mode (:mod:`pipeline.capture.register`).

Copying runs in the background after the capture stops. While it runs the cameras are busy,
so the controller does not arm them again until it has finished or has been cancelled; a
copy that was cancelled resumes the next time it is run.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.capture import register as registration
from pipeline.capture.collect import is_video, offload_camera
from pipeline.capture.devices import CameraDriver, MediaFile, OffloadError
from pipeline.capture.events import EventBus
from posetrak.db.db import DEFAULT_REGISTRY_PATH, open_registry

logger = logging.getLogger(__name__)

#: how often progress of a copy is published at most
PROGRESS_INTERVAL_S = 0.5


@dataclass(frozen=True)
class OffloadTarget:
    """A camera that took part in the capture being copied."""

    label: str
    driver: CameraDriver
    camera_instance_id: str
    #: registry mode the camera was armed with, if any
    mode_id: str | None


@dataclass
class _Progress:
    state: str = "waiting"  # waiting | copying | registering | done | failed | skipped
    files: int = 0
    files_done: int = 0
    bytes_done: int = 0
    bytes_total: int = 0
    error: str | None = None
    videos: list[dict[str, Any]] = field(default_factory=list)

    def view(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "files": self.files,
            "files_done": self.files_done,
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
            "error": self.error,
            "videos": list(self.videos),
        }


class CaptureOffloader:
    """Copies and registers the recordings of captures.

    Parameters
    ----------
    session:
        Session database connection; used from the event loop thread only.
    session_id:
        ``mocap_sessions.id`` of the captures.
    bus:
        Receives an ``offload`` event whenever the state changes.
    destination:
        Directory the copies go to; None means offloading is not available.
    registry_path:
        Camera registry used when registering files; None means the default registry.
    auto:
        Copy automatically when a capture has stopped.
    concurrency:
        Cameras copied at the same time.
    on_alert:
        ``(camera label or None, kind, message, active)`` raises or clears an operator alert.
    registrar:
        Replaces :func:`pipeline.capture.register.register_video`, for simulated cameras.
    """

    def __init__(
        self,
        session: sqlite3.Connection,
        session_id: str,
        bus: EventBus,
        *,
        destination: Path | None,
        registry_path: Path | None = None,
        auto: bool = False,
        concurrency: int = 2,
        on_alert: Callable[[str | None, str, str, bool], None] | None = None,
        registrar: Callable[..., Any] | None = None,
    ) -> None:
        self._session = session
        self._session_id = session_id
        self._bus = bus
        self.destination = destination
        self._registry_path = registry_path
        self.auto = auto
        self._concurrency = max(1, concurrency)
        self._on_alert = on_alert or (lambda *args: None)
        self._registrar = registrar
        self._baseline: dict[str, set[str] | None] = {}
        self._progress: dict[str, _Progress] = {}
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._capture_id: str | None = None
        self._last_publish = 0.0

    # -- state ---------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self.destination is not None

    @property
    def busy(self) -> bool:
        return self._running

    def view(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "auto": self.auto,
            "busy": self.busy,
            "capture_id": self._capture_id,
            "cameras": {label: p.view() for label, p in self._progress.items()},
        }

    def _publish(self) -> None:
        self._last_publish = time.monotonic()
        self._bus.publish("offload", **self.view())

    def set_auto(self, enabled: bool) -> None:
        if enabled and not self.available:
            raise OffloadError("no directory to copy to is configured (daemon.offload_dir)")
        self.auto = enabled
        self._publish()

    # -- before the capture --------------------------------------------------

    async def take_baseline(self, targets: list[OffloadTarget]) -> None:
        """Remember what is on the cameras now; what appears later belongs to the next capture."""
        if not self.available:
            return

        async def one(target: OffloadTarget) -> None:
            if not target.driver.supports_offload:
                self._baseline[target.label] = None
                return
            try:
                self._baseline[target.label] = {f.path for f in await target.driver.list_media()}
            except OffloadError as exc:
                logger.warning("%s: could not list the recordings before the capture: %s", target.label, exc)
                self._baseline[target.label] = None

        await asyncio.gather(*(one(t) for t in targets))

    # -- after the capture ---------------------------------------------------

    def start(self, capture_id: str, targets: list[OffloadTarget]) -> None:
        """Copy the recordings of *capture_id* in the background.

        Raises
        ------
        OffloadError
            If offloading is not available or a copy is already running.
        """
        if not self.available:
            raise OffloadError("no directory to copy to is configured (daemon.offload_dir)")
        if self.busy:
            raise OffloadError("a copy is already running")
        self._capture_id = capture_id
        self._progress = {t.label: _Progress() for t in targets}
        for t in targets:
            self._on_alert(t.label, "offload_failed", "", False)
            self._on_alert(t.label, "video_mode", "", False)
        self._running = True
        self._task = asyncio.create_task(self._run(capture_id, targets))
        self._publish()

    def cancel(self) -> None:
        """Stop copying; files already copied stay and the rest resumes next time."""
        if self.busy:
            assert self._task is not None
            self._task.cancel()

    async def wait(self) -> None:
        """Wait until the copy that is running, if any, has finished."""
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self, capture_id: str, targets: list[OffloadTarget]) -> None:
        number = self._session.execute("SELECT capture_number FROM captures WHERE id = ?", (capture_id,)).fetchone()
        root = self.destination / f"capture-{number[0]:03d}" if number else self.destination / "capture"  # type: ignore[operator]
        gate = asyncio.Semaphore(self._concurrency)
        registry: sqlite3.Connection | None = None
        try:
            if self._registrar is None:
                registry = open_registry(self._registry_path or DEFAULT_REGISTRY_PATH)
            opened = registry

            await asyncio.gather(*(self._one(capture_id, t, root, opened, gate) for t in targets))
        except asyncio.CancelledError:
            for p in self._progress.values():
                if p.state in ("waiting", "copying", "registering"):
                    p.state, p.error = "failed", "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001  (nothing awaits this task; report it)
            logger.exception("offload failed")
            for p in self._progress.values():
                if p.state in ("waiting", "copying", "registering"):
                    p.state, p.error = "failed", str(exc)
            self._on_alert(None, "offload_failed", f"copying the videos failed: {exc}", True)
        finally:
            if registry is not None:
                registry.close()
            self._running = False
            self._publish()

    async def _one(
        self, capture_id: str, target: OffloadTarget, root: Path, registry: sqlite3.Connection | None, gate: asyncio.Semaphore
    ) -> None:
        progress = self._progress[target.label]
        try:
            files = await self._new_recordings(target)
            if files is None:
                progress.state = "skipped"
                progress.error = "the recordings from before the capture were not listed, so the new ones cannot be told"
                self._on_alert(target.label, "offload_failed", f"{target.label}: {progress.error}", True)
                return
            progress.files = len(files)
            progress.bytes_total = sum(f.size or 0 for f in files)
            self._publish()
            offsets: dict[str, int] = {}
            running = 0
            for f in files:
                offsets[f.path] = running
                running += f.size or 0

            def on_progress(file: MediaFile, done: int, total: int | None) -> None:
                progress.bytes_done = offsets[file.path] + done
                if time.monotonic() - self._last_publish >= PROGRESS_INTERVAL_S:
                    self._publish()

            async with gate:
                progress.state = "copying"
                self._publish()
                results = await offload_camera(target.driver, root, files=files, progress=on_progress)
                progress.state = "registering"
                self._publish()
                for result in results:
                    await self._register(capture_id, target, result, registry, progress)
                    progress.files_done += 1
            progress.bytes_done = progress.bytes_total
            progress.state = "done"
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001  (one camera's failure must not stop the others)
            logger.exception("%s: offload failed", target.label)
            progress.state, progress.error = "failed", str(exc)
            self._on_alert(target.label, "offload_failed", f"{target.label}: copying the videos failed: {exc}", True)
        finally:
            self._publish()

    async def _new_recordings(self, target: OffloadTarget) -> list[MediaFile] | None:
        before = self._baseline.get(target.label)
        if before is None:
            return None
        now = await target.driver.list_media()
        return sorted((f for f in now if f.path not in before and is_video(f)), key=lambda f: (f.created or 0, f.path))

    async def _register(self, capture_id: str, target: OffloadTarget, result: Any, registry: sqlite3.Connection | None, progress: _Progress) -> None:
        facts = None
        if self._registrar is None:
            try:
                facts = await asyncio.to_thread(registration.read_video_facts, result.path)
            except registration.ProbeError:
                facts = None
        try:
            video = (self._registrar or registration.register_video)(
                self._session,
                registry,
                self._session_id,
                capture_id,
                target.camera_instance_id,
                result.path,
                expected_mode_id=target.mode_id,
                facts=facts,
                source_path=result.file.path,
                sha256=result.sha256,
                label=target.label,
            )
        except registration.RegisterError as exc:
            progress.videos.append({"file": result.file.name, "status": "not_registered", "notes": str(exc)})
            self._on_alert(target.label, "offload_failed", f"{target.label}: {result.file.name} was copied but not registered: {exc}", True)
            return
        progress.videos.append(
            {"file": result.file.name, "status": str(video.status), "mode": video.mode_name, "notes": video.notes}
        )
        if video.status not in (registration.VideoStatus.OK, registration.VideoStatus.MODE_ASSUMED):
            self._on_alert(target.label, "video_mode", f"{target.label}: {result.file.name}: {video.notes}", True)
