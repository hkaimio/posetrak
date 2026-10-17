# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""CaptureController: arm, start and stop a group of cameras and log what happened.

One controller owns a set of camera drivers and one session database. A take
goes through the phases ``IDLE -> ARMING -> ARMED -> STARTING -> RECORDING ->
STOPPING -> STOPPED``; every camera has its own :class:`DeviceState` inside
that, so a single failing camera never blocks the others. Commands fan out to
all cameras in parallel with a timeout, and each camera's outcome is written
to ``capture_events`` as soon as it is known.

All methods run on the event loop thread; the session connection is used from
that thread only.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pipeline.capture.devices import (
    ArmResult,
    CameraDriver,
    CommandResult,
    DeviceState,
    DeviceStatus,
    LinkState,
    ManualDriver,
)
from pipeline.capture.events import EventBus
from posetrak.db.capture_control import CaptureEvent, upsert_capture_event, utc_now_iso
from posetrak.db.db import create_capture

logger = logging.getLogger(__name__)

BATTERY_LOW_PCT = 15
STORAGE_LOW_S = 300


class CapturePhase(enum.StrEnum):
    IDLE = "idle"
    ARMING = "arming"
    ARMED = "armed"
    STARTING = "starting"
    RECORDING = "recording"
    STOPPING = "stopping"
    STOPPED = "stopped"


class CaptureStateError(RuntimeError):
    """A command was issued in a phase that does not allow it."""


@dataclass(frozen=True)
class CameraSlot:
    """A camera taking part in captures."""

    label: str
    camera_instance_id: str
    driver: CameraDriver


@dataclass(frozen=True)
class ChecklistItem:
    """Something the operator must confirm before arming completes."""

    id: str
    label: str
    camera: str | None = None


@dataclass(frozen=True)
class Alert:
    kind: str
    camera: str | None
    message: str
    ts: str


@dataclass(frozen=True)
class CameraOutcome:
    """What a start or stop command did on one camera."""

    label: str
    ok: bool
    confidence: str
    cmd_ts: str | None = None
    ack_ts: str | None = None
    confirm_ts: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ArmReport:
    ready: bool
    cameras: dict[str, ArmResult]
    problems: dict[str, tuple[str, ...]]
    open_checklist: tuple[str, ...]


@dataclass(frozen=True)
class CaptureReport:
    capture_id: str | None
    outcomes: dict[str, CameraOutcome]

    @property
    def ok(self) -> bool:
        return bool(self.outcomes) and all(o.ok for o in self.outcomes.values())


@dataclass
class _Camera:
    slot: CameraSlot
    state: DeviceState = DeviceState.OFFLINE
    fault: str | None = None
    status: DeviceStatus = field(default_factory=DeviceStatus)
    enabled: bool = True
    in_capture: bool = False
    overridden: bool = False
    start: CameraOutcome | None = None
    stop: CameraOutcome | None = None

    @property
    def label(self) -> str:
        return self.slot.label

    @property
    def driver(self) -> CameraDriver:
        return self.slot.driver


class CaptureController:
    """Orchestrates a group of cameras for one session.

    Parameters
    ----------
    slots:
        The cameras. ``camera_instance_id`` is the registry id recorded in
        ``capture_events``.
    session:
        Open session database connection.
    session_id:
        ``mocap_sessions.id`` new captures are created under.
    settings:
        Profile settings handed to every driver at arm time.
    checklist:
        Items the operator must tick before the controller reports ready.
    command_timeout_s:
        How long a start or stop may take on one camera before it counts as
        unconfirmed.
    """

    def __init__(
        self,
        slots: list[CameraSlot],
        session: sqlite3.Connection,
        session_id: str,
        *,
        settings: dict[str, Any] | None = None,
        checklist: list[ChecklistItem] | None = None,
        command_timeout_s: float = 5.0,
        arm_timeout_s: float = 30.0,
        bus: EventBus | None = None,
        clock: Callable[[], str] = utc_now_iso,
    ) -> None:
        labels = [s.label for s in slots]
        if len(set(labels)) != len(labels):
            raise ValueError("duplicate camera labels")
        self._cameras = {s.label: _Camera(s) for s in slots}
        self._session = session
        self._session_id = session_id
        self._settings = dict(settings or {})
        self._checklist = {c.id: c for c in checklist or []}
        self._ticked: set[str] = set()
        self._timeout = command_timeout_s
        self._arm_timeout = arm_timeout_s
        self.bus = bus or EventBus()
        self._clock = clock
        self.phase = CapturePhase.IDLE
        self.capture_id: str | None = None
        self._busy = False
        self._alerts: dict[tuple[str | None, str], Alert] = {}

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        """Bring all drivers up. A driver that fails to connect leaves its camera faulted."""
        for cam in self._cameras.values():
            cam.driver.on_status(self._on_driver_status)

        async def one(cam: _Camera) -> None:
            try:
                await asyncio.wait_for(cam.driver.connect(), self._arm_timeout)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: connect failed: %r", cam.label, exc)
                self._set_state(cam, DeviceState.FAULT, f"connect failed: {exc!r}")

        await asyncio.gather(*(one(c) for c in self._cameras.values()))

    async def close(self) -> None:
        await asyncio.gather(
            *(c.driver.close() for c in self._cameras.values()), return_exceptions=True
        )

    # -- operator inputs ---------------------------------------------------

    def tick(self, item_id: str, done: bool = True) -> None:
        if item_id not in self._checklist:
            raise KeyError(item_id)
        (self._ticked.add if done else self._ticked.discard)(item_id)
        self.bus.publish("checklist", item=item_id, done=done)

    def set_enabled(self, label: str, enabled: bool) -> None:
        """Include or exclude a camera from the next capture."""
        if self.phase in (CapturePhase.STARTING, CapturePhase.RECORDING, CapturePhase.STOPPING):
            raise CaptureStateError("cameras cannot be added or removed during a capture")
        self._cameras[label].enabled = enabled
        self.bus.publish("camera", **self._camera_view(self._cameras[label]))

    async def fallback_to_manual(self, label: str) -> None:
        """Replace a camera's driver with a :class:`ManualDriver` (broken-node escape hatch)."""
        cam = self._cameras[label]
        if self.phase in (CapturePhase.STARTING, CapturePhase.STOPPING):
            raise CaptureStateError("operation in progress")
        await asyncio.gather(cam.driver.close(), return_exceptions=True)
        driver = ManualDriver(label)
        driver.on_status(self._on_driver_status)
        cam.slot = CameraSlot(label, cam.slot.camera_instance_id, driver)
        cam.fault = None
        await driver.connect()
        self._set_state(cam, DeviceState.READY)

    def operator_override(self, label: str, recording: bool) -> None:
        """Record that the operator started (or stopped) a camera by hand.

        Used when a start did not confirm but the camera is seen recording, or
        for a camera with no remote control.
        """
        cam = self._cameras[label]
        if self.phase not in (CapturePhase.RECORDING, CapturePhase.STOPPING, CapturePhase.STOPPED):
            raise CaptureStateError("no capture to override")
        if self.capture_id is None or not cam.in_capture:
            raise CaptureStateError(f"{label} is not part of this capture")
        cam.overridden = True
        self._set_state(cam, DeviceState.RECORDING if recording else DeviceState.STOPPED)
        self._write_event(cam, status="operator_override", confidence="manual", notes="operator override")
        self._clear_alert(label, "start_failed")

    # -- commands ----------------------------------------------------------

    async def arm(self) -> ArmReport:
        """Pre-flight every enabled camera and report whether the group is ready."""
        self._require_phase("arm", CapturePhase.IDLE, CapturePhase.ARMED, CapturePhase.STOPPED)
        self._enter_busy()
        try:
            self._set_phase(CapturePhase.ARMING)
            self.capture_id = None
            for cam in self._cameras.values():
                cam.in_capture = cam.overridden = False
                cam.start = cam.stop = None
            cams = [c for c in self._cameras.values() if c.enabled]
            results = await asyncio.gather(*(self._arm_one(c) for c in cams))
            arm_results = {c.label: r for c, r in zip(cams, results)}
            problems: dict[str, tuple[str, ...]] = {}
            for label, res in arm_results.items():
                issues = list(res.problems) + [
                    f"{k}: wanted {want!r}, camera has {got!r}" for k, (want, got) in res.mismatches.items()
                ]
                issues += self._status_problems(self._cameras[label])
                if not res.ok and not issues:
                    issues.append("arm failed")
                if issues:
                    problems[label] = tuple(issues)
                    self._set_state(self._cameras[label], DeviceState.FAULT, "; ".join(issues))
                else:
                    self._set_state(self._cameras[label], DeviceState.ARMED)
            open_items = tuple(i for i in self._checklist if i not in self._ticked)
            ready = bool(cams) and not problems and not open_items and all(
                c.state == DeviceState.ARMED for c in cams
            )
            self._set_phase(CapturePhase.ARMED if ready else CapturePhase.IDLE)
            return ArmReport(ready, arm_results, problems, open_items)
        finally:
            self._busy = False

    async def start_capture(self, label: str | None = None, *, force: bool = False) -> CaptureReport:
        """Start recording on every armed camera.

        With ``force=False`` the group must be ready (:meth:`arm` returned
        ``ready``). With ``force=True`` cameras that are not armed are left
        out and logged as failed, and the rest start.
        """
        if self.phase != CapturePhase.ARMED:
            if not (force and self.phase == CapturePhase.IDLE):
                raise CaptureStateError(f"cannot start in phase {self.phase}")
        self._enter_busy()
        try:
            enabled = [c for c in self._cameras.values() if c.enabled]
            participants = [c for c in enabled if c.state == DeviceState.ARMED]
            if not participants:
                raise CaptureStateError("no armed camera to start")
            self._set_phase(CapturePhase.STARTING)
            self.capture_id = create_capture(self._session, self._session_id, label=label)
            now = self._clock()
            for cam in enabled:
                cam.in_capture = cam in participants
                # Pessimistic until the outcome is known: a crash leaves 'failed'.
                self._write_event(
                    cam,
                    status="failed",
                    notes="start pending" if cam.in_capture else "not armed, left out of capture",
                    start_cmd_ts=now if cam.in_capture else None,
                )
            outcomes = dict(
                await asyncio.gather(*(self._command(c, "start") for c in participants))
            )
            for cam in participants:
                out = outcomes[cam.label]
                cam.start = out
                self._write_event(
                    cam,
                    status="ok" if out.ok else "failed",
                    notes=out.error,
                    start_cmd_ts=out.cmd_ts,
                    start_ack_ts=out.ack_ts,
                    start_confirm_ts=out.confirm_ts,
                )
                if out.ok:
                    self._set_state(cam, DeviceState.RECORDING)
                else:
                    self._set_state(cam, DeviceState.FAULT, out.error)
                    self._raise_alert(cam.label, "start_failed", f"start failed: {out.error}")
            started = any(o.ok for o in outcomes.values())
            self._set_phase(CapturePhase.RECORDING if started else CapturePhase.IDLE)
            return CaptureReport(self.capture_id, outcomes)
        finally:
            self._busy = False

    async def stop_capture(self) -> CaptureReport:
        """Stop every camera that took part in the capture."""
        self._require_phase("stop", CapturePhase.RECORDING)
        self._enter_busy()
        try:
            self._set_phase(CapturePhase.STOPPING)
            participants = [c for c in self._cameras.values() if c.in_capture and not c.overridden]
            outcomes = dict(await asyncio.gather(*(self._command(c, "stop") for c in participants)))
            for cam in participants:
                out = outcomes[cam.label]
                cam.stop = out
                start_ok = cam.start is not None and cam.start.ok
                self._write_event(
                    cam,
                    status="ok" if (out.ok and start_ok) else "failed",
                    notes=out.error,
                    stop_cmd_ts=out.cmd_ts,
                    stop_ack_ts=out.ack_ts,
                    stop_confirm_ts=out.confirm_ts,
                )
                if out.ok:
                    self._set_state(cam, DeviceState.STOPPED)
                else:
                    self._set_state(cam, DeviceState.FAULT, out.error)
                    self._raise_alert(cam.label, "stop_failed", f"stop failed: {out.error}")
            for cam in self._cameras.values():
                if cam.overridden and cam.state == DeviceState.RECORDING:
                    self._set_state(cam, DeviceState.STOPPED)
            self._set_phase(CapturePhase.STOPPED)
            return CaptureReport(self.capture_id, outcomes)
        finally:
            self._busy = False

    # -- state -------------------------------------------------------------

    @property
    def alerts(self) -> list[Alert]:
        return list(self._alerts.values())

    def snapshot(self) -> dict[str, Any]:
        """Complete controller state, JSON-serializable."""
        return {
            "seq": self.bus.seq,
            "phase": str(self.phase),
            "capture_id": self.capture_id,
            "cameras": {c.label: self._camera_view(c) for c in self._cameras.values()},
            "checklist": [
                {"id": i.id, "label": i.label, "camera": i.camera, "done": i.id in self._ticked}
                for i in self._checklist.values()
            ],
            "alerts": [vars(a) for a in self._alerts.values()],
        }

    def camera_state(self, label: str) -> DeviceState:
        return self._cameras[label].state

    # -- internals ---------------------------------------------------------

    def _require_phase(self, what: str, *phases: CapturePhase) -> None:
        if self.phase not in phases:
            raise CaptureStateError(f"cannot {what} in phase {self.phase}")

    def _enter_busy(self) -> None:
        if self._busy:
            raise CaptureStateError("another operation is in progress")
        self._busy = True

    def _set_phase(self, phase: CapturePhase) -> None:
        self.phase = phase
        self.bus.publish("phase", phase=str(phase), capture_id=self.capture_id)

    def _set_state(self, cam: _Camera, state: DeviceState, fault: str | None = None) -> None:
        if cam.state == state and cam.fault == fault:
            return
        cam.state, cam.fault = state, fault
        self.bus.publish("camera", **self._camera_view(cam))

    def _camera_view(self, cam: _Camera) -> dict[str, Any]:
        st = cam.status
        return {
            "label": cam.label,
            "state": str(cam.state),
            "fault": cam.fault,
            "confidence": cam.driver.confidence,
            "enabled": cam.enabled,
            "in_capture": cam.in_capture,
            "overridden": cam.overridden,
            "link": str(st.link),
            "battery_pct": st.battery_pct,
            "storage_remaining_s": st.storage_remaining_s,
            "overheating": st.overheating,
            "recording": st.recording,
        }

    async def _arm_one(self, cam: _Camera) -> ArmResult:
        try:
            return await asyncio.wait_for(cam.driver.arm(self._settings), self._arm_timeout)
        except Exception as exc:  # noqa: BLE001
            return ArmResult(False, (f"arm failed: {exc!r}",))

    def _status_problems(self, cam: _Camera) -> list[str]:
        st = cam.status
        out = []
        if st.link != LinkState.UP:
            out.append(f"link {st.link}")
        if st.battery_pct is not None and st.battery_pct < BATTERY_LOW_PCT:
            out.append(f"battery {st.battery_pct}%")
        if st.storage_remaining_s is not None and st.storage_remaining_s < STORAGE_LOW_S:
            out.append(f"storage {st.storage_remaining_s}s remaining")
        if st.overheating:
            out.append("overheating")
        return out

    async def _command(self, cam: _Camera, op: str) -> tuple[str, CameraOutcome]:
        """Run start or stop on one camera, always returning an outcome."""
        fn = cam.driver.start if op == "start" else cam.driver.stop
        sent = self._clock()
        conf = cam.driver.confidence
        try:
            res: CommandResult = await asyncio.wait_for(fn(), self._timeout)
        except asyncio.TimeoutError:
            return cam.label, CameraOutcome(cam.label, False, conf, sent, error=f"{op} not confirmed within {self._timeout} s")
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s: %s failed", cam.label, op)
            return cam.label, CameraOutcome(cam.label, False, conf, sent, error=f"{op} raised {exc!r}")
        ok = res.ok and (conf != "confirmed" or res.confirm_ts is not None)
        error = res.error
        if res.ok and not ok:
            error = f"{op} sent but the camera did not confirm"
        return cam.label, CameraOutcome(
            cam.label, ok, conf, res.cmd_ts or sent, res.ack_ts, res.confirm_ts, error
        )

    def _write_event(
        self,
        cam: _Camera,
        *,
        status: str,
        confidence: str | None = None,
        notes: str | None = None,
        **timestamps: str | None,
    ) -> None:
        if self.capture_id is None:
            return
        upsert_capture_event(
            self._session,
            CaptureEvent(
                self.capture_id,
                cam.slot.camera_instance_id,
                confidence or cam.driver.confidence,
                status,
                notes=notes,
                **timestamps,
            ),
        )

    # driver status -> controller state and alerts

    def _on_driver_status(self, label: str, status: DeviceStatus) -> None:
        cam = self._cameras[label]
        cam.status = status
        if cam.state in (DeviceState.OFFLINE, DeviceState.DISCOVERED, DeviceState.READY):
            self._set_state(cam, DeviceState.READY if status.link == LinkState.UP else DeviceState.OFFLINE)
        elif cam.state == DeviceState.ARMED and status.link != LinkState.UP:
            self._set_state(cam, DeviceState.FAULT, f"link {status.link} while armed")
        else:
            self.bus.publish("camera", **self._camera_view(cam))
        self._check_alerts(cam)

    def _check_alerts(self, cam: _Camera) -> None:
        st = cam.status
        recording = cam.state == DeviceState.RECORDING
        self._toggle_alert(
            cam.label, "link_lost", recording and st.link != LinkState.UP, f"{cam.label}: link {st.link} during capture"
        )
        self._toggle_alert(
            cam.label,
            "stopped_by_itself",
            recording and cam.driver.confidence == "confirmed" and st.recording is False,
            f"{cam.label}: camera stopped recording by itself",
        )
        self._toggle_alert(
            cam.label,
            "battery_low",
            st.battery_pct is not None and st.battery_pct < BATTERY_LOW_PCT and cam.in_capture,
            f"{cam.label}: battery {st.battery_pct}%",
        )
        self._toggle_alert(
            cam.label,
            "storage_low",
            st.storage_remaining_s is not None and st.storage_remaining_s < STORAGE_LOW_S and cam.in_capture,
            f"{cam.label}: {st.storage_remaining_s} s of storage left",
        )
        self._toggle_alert(cam.label, "overheating", st.overheating and cam.in_capture, f"{cam.label}: overheating")

    def _toggle_alert(self, camera: str | None, kind: str, active: bool, message: str) -> None:
        if active:
            self._raise_alert(camera, kind, message)
        else:
            self._clear_alert(camera, kind)

    def _raise_alert(self, camera: str | None, kind: str, message: str) -> None:
        key = (camera, kind)
        if key in self._alerts:
            return
        alert = Alert(kind, camera, message, self._clock())
        self._alerts[key] = alert
        logger.warning("alert: %s", message)
        self.bus.publish("alert", active=True, **vars(alert))

    def _clear_alert(self, camera: str | None, kind: str) -> None:
        alert = self._alerts.pop((camera, kind), None)
        if alert is not None:
            self.bus.publish("alert", active=False, **vars(alert))
