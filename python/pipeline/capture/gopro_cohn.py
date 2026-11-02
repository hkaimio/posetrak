# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Hero13 driver: control over COHN (Camera On the Home Network).

A camera provisioned for COHN joins a WLAN and serves the Open GoPro HTTP API
over HTTPS on the address it got there, with Basic authentication and a
self-signed certificate. Bluetooth is only needed to provision a camera and to
wake it (see :mod:`pipeline.capture.gopro_ble`); everything while capturing
goes through this module.

Behaviour that follows from measurements on Hero13s:

* The camera closes connections made by some HTTP stacks after about two
  seconds, so the client is ``httpx`` with the certificate pinned (no hostname
  check: the certificate does not carry the camera's address).
* The shutter command is acknowledged long before recording is established,
  so START and STOP are only reported as confirmed when the ``encoding`` status
  has changed.
* The shutter command is accepted in any mode, including time lapse, which
  records the wrong thing without complaint. Arming therefore switches the
  camera to the video mode and reads the profile settings back; starting
  checks the mode again.
* A camera records for roughly twenty minutes at the highest-quality settings
  before overheating; the ``overheating`` status is passed on so the operator
  can be warned about five minutes ahead of the shutdown.

A camera provisioned on several WLANs has one endpoint per WLAN; whichever
answers is used, so the same configuration works at home and on location.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import ssl
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from pipeline.capture.credentials import CohnCredentials
from pipeline.capture.devices import (
    ArmResult,
    CameraDriver,
    CommandResult,
    DeviceState,
    DeviceStatus,
    Downloaded,
    LinkState,
    MediaFile,
    ModeSpec,
    OffloadError,
    ProgressCallback,
    SettingSpec,
)
from pipeline.capture.gopro_profile import (
    CONTROL_MODE_PRO,
    FLAT_MODE_VIDEO,
    PRESET_GROUP_VIDEO,
    SETTING_CONTROL_MODE,
    ST_BATTERY_PERCENT,
    ST_ENCODING,
    ST_FLAT_MODE,
    ST_OVERHEATING,
    ST_REMAINING_VIDEO_S,
    SUPPORTED,
    ProfileError,
    derive,
    label,
    translate,
)
from pipeline.capture.network import route_warning
from posetrak.db.capture_control import utc_now_iso

logger = logging.getLogger(__name__)

#: A state older than this is read again before a command relies on it.
STATE_MAX_AGE_S = 3.0

#: How long an offload request is retried when the connection to the camera fails, and the pause
#: between attempts (grows with every attempt up to the maximum). A camera on a marginal WLAN
#: link is unreachable for several seconds at a time; a copy only gives up when it gets no
#: further for the whole window.
TRANSIENT_WINDOW_S = 60.0
TRANSIENT_PAUSE_S = 1.0
TRANSIENT_PAUSE_MAX_S = 5.0


class NotProvisionedError(RuntimeError):
    """No COHN credentials are stored for the camera."""


class _CopyInterrupted(Exception):
    """A download stopped after *done* bytes because of *cause*."""

    def __init__(self, done: int, cause: Exception) -> None:
        super().__init__(f"{done} bytes copied; {cause!r}")
        self.done = done
        self.cause = cause


@dataclass(frozen=True)
class CohnEndpoint:
    """Where and how to reach one camera on one WLAN."""

    network: str
    base_url: str
    username: str = ""
    password: str = ""
    #: PEM certificate to pin; None for plain HTTP (the test camera)
    certificate: str | None = None

    @classmethod
    def from_credentials(cls, creds: CohnCredentials, network: str, ip_address: str | None = None) -> CohnEndpoint:
        """Endpoint for stored credentials, optionally at a reserved address instead of the stored one."""
        return cls(
            network=network,
            base_url=f"https://{ip_address or creds.ip_address}",
            username=creds.username,
            password=creds.password,
            certificate=creds.certificate,
        )

    @property
    def host(self) -> str:
        return self.base_url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]

    def make_client(self) -> httpx.AsyncClient:
        verify: ssl.SSLContext | bool = True
        if self.certificate:
            verify = ssl.create_default_context(cadata=self.certificate)
            # The certificate has no authority key identifier and its name does
            # not match the camera's address; trust is the pinned certificate.
            verify.check_hostname = False
            verify.verify_flags &= ~ssl.VERIFY_X509_STRICT
            verify.verify_flags &= ~ssl.VERIFY_X509_PARTIAL_CHAIN
        headers = {}
        if self.username:
            headers["Authorization"] = "Basic " + b64encode(f"{self.username}:{self.password}".encode()).decode()
        return httpx.AsyncClient(base_url=self.base_url, verify=verify, headers=headers)


def _describe(exc: Exception | None) -> str:
    """A short reason for a failed request: the HTTP status, or the kind of error."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__ if exc is not None else "unknown error"


class GoProCohnDriver(CameraDriver):
    """One Hero13 controlled over COHN.

    Parameters
    ----------
    label:
        Camera label, as in the registry.
    endpoints:
        One per WLAN the camera is provisioned on, in order of preference.
    poll_interval_s:
        How often status is read while idle. Commands poll much faster.
    request_timeout_s:
        Timeout of one request.
    confirm_timeout_s:
        How long start and stop wait for the camera to confirm.
    failures_before_down:
        Consecutive failed polls before the link is reported down.
    """

    confidence = "confirmed"
    settings_control = "driver"

    def supported_settings(self) -> dict[str, SettingSpec]:
        return {
            name: SettingSpec(name, choices, "or a numeric setting id for any other setting")
            for name, choices in SUPPORTED.items()
        }

    def mode_settings(self, mode: ModeSpec) -> dict[str, Any]:
        """Resolution and frame rate derived from the mode's size, with the mode's own settings on top.

        The derived values are the base, so a mode only has to add what its size cannot
        tell (the lens, the bit rate) and may state resolution or frame rate to override
        the derivation. A size that maps to no Hero13 resolution is an error unless the
        mode's settings state the resolution and frame rate themselves. Image stabilization
        is off unless the mode asks for it: it makes the image geometry change from frame to
        frame, which an intrinsic calibration of the mode cannot describe.
        """
        settings: dict[str, Any] = {"hypersmooth": "off"}
        try:
            settings.update(derive(mode.width_px, mode.height_px, mode.fps))
        except ProfileError:
            if not {"resolution", "fps"} <= mode.capture_settings.keys():
                raise
        settings.update(mode.capture_settings)
        return settings

    def __init__(
        self,
        label: str,
        endpoints: list[CohnEndpoint],
        *,
        poll_interval_s: float = 2.0,
        request_timeout_s: float = 3.0,
        confirm_timeout_s: float = 4.0,
        failures_before_down: int = 2,
    ) -> None:
        super().__init__(label)
        if not endpoints:
            raise NotProvisionedError(f"{label}: no COHN credentials; provision the camera first")
        self._endpoints = endpoints
        self._clients: list[httpx.AsyncClient] = []
        self._active = 0
        self._poll_interval = poll_interval_s
        self._request_timeout = request_timeout_s
        self._confirm_timeout = confirm_timeout_s
        self._failures_before_down = failures_before_down
        self._failures = 0
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._state: dict[str, Any] | None = None
        self._state_at = 0.0
        self._expected: dict[int, int] = {}
        self._route_warnings: list[str] = []

    # -- link ----------------------------------------------------------------

    async def connect(self) -> None:
        self._clients = [ep.make_client() for ep in self._endpoints]
        for ep in self._endpoints:
            warning = route_warning(ep.host)
            if warning:
                logger.warning("%s: %s", self.label, warning)
                self._route_warnings.append(warning)
        await self._poll_once()
        self._task = asyncio.create_task(self._poll_loop(), name=f"{self.label}-poll")

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await asyncio.gather(*(c.aclose() for c in self._clients), return_exceptions=True)
        self._clients = []
        self._set_status(DeviceStatus())

    async def _poll_loop(self) -> None:
        while True:
            delay = self._poll_interval if self._failures == 0 else min(1.0 + self._failures, 5.0)
            await asyncio.sleep(delay)
            try:
                await self._poll_once()
            except Exception:  # noqa: BLE001  (the loop must outlive any one bad poll)
                logger.exception("%s: status poll failed unexpectedly", self.label)

    async def _poll_once(self) -> None:
        """Read the state; after a failure, also try the camera's other networks."""
        order = [self._active] + [i for i in range(len(self._clients)) if i != self._active]
        last: Exception | None = None
        for index in order if self._failures else order[:1]:
            try:
                await self._refresh(index)
                return
            except Exception as exc:  # noqa: BLE001
                last = exc
        self._failures += 1
        logger.debug("%s: poll failed (%d): %r", self.label, self._failures, last)
        if self._failures >= self._failures_before_down and self._status.link != LinkState.DOWN:
            logger.warning("%s: link lost: %r", self.label, last)
            self._set_status(DeviceStatus(state=DeviceState.OFFLINE, link=LinkState.DOWN))

    async def _get(self, path: str, params: dict[str, Any] | None = None, *, index: int | None = None) -> httpx.Response:
        client = self._clients[self._active if index is None else index]
        async with self._lock:
            response = await client.get(path, params=params, timeout=self._request_timeout)
        response.raise_for_status()
        return response

    async def _get_retrying(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """Like :meth:`_get`, but tries again when the connection is dropped.

        A camera closes a connection it has not used for a while, and a request sent over
        such a connection fails without the camera having seen it. Offload requests are
        read-only, so repeating one is safe.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + TRANSIENT_WINDOW_S
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self._get(path, params)
            except httpx.TransportError as exc:
                if loop.time() >= deadline:
                    raise
                logger.info("%s: %s failed (%s); trying again", self.label, path, type(exc).__name__)
                await asyncio.sleep(min(TRANSIENT_PAUSE_S * attempt, TRANSIENT_PAUSE_MAX_S))

    async def _refresh(self, index: int | None = None) -> dict[str, Any]:
        """Read ``camera/state``, publish it as the status, and return it."""
        data = (await self._get("/gopro/camera/state", index=index)).json()
        if index is not None and index != self._active:
            logger.info("%s: now reached on network %s", self.label, self._endpoints[index].network)
            self._active = index
        self._state, self._state_at = data, asyncio.get_running_loop().time()
        if self._failures:
            logger.info("%s: link back after %d failed polls", self.label, self._failures)
        self._failures = 0
        st = data["status"]
        self._set_status(
            DeviceStatus(
                state=DeviceState.READY,
                link=LinkState.UP,
                battery_pct=st.get(ST_BATTERY_PERCENT),
                storage_remaining_s=st.get(ST_REMAINING_VIDEO_S),
                overheating=bool(st.get(ST_OVERHEATING)),
                recording=bool(st.get(ST_ENCODING)),
                detail={"network": self._endpoints[self._active].network, "flat_mode": st.get(ST_FLAT_MODE)},
            )
        )
        return data

    async def _recent_state(self) -> dict[str, Any]:
        age = asyncio.get_running_loop().time() - self._state_at
        if self._state is not None and age <= STATE_MAX_AGE_S and self._failures == 0:
            return self._state
        return await self._refresh()

    def _unreachable(self, exc: Exception) -> str:
        text = f"camera not reachable ({type(exc).__name__})"
        return f"{text}; {self._route_warnings[0]}" if self._route_warnings else text

    # -- arm -----------------------------------------------------------------

    async def arm(self, settings: dict[str, Any]) -> ArmResult:
        try:
            wanted = translate(settings)
        except ProfileError as exc:
            return ArmResult(False, (str(exc),))
        self._expected = wanted
        try:
            data = await self._refresh()
        except Exception as exc:  # noqa: BLE001
            return ArmResult(False, (self._unreachable(exc),))
        if data["status"].get(ST_ENCODING):
            return ArmResult(False, ("camera is recording",))

        problems: list[str] = []
        try:
            if data["settings"].get(str(SETTING_CONTROL_MODE), CONTROL_MODE_PRO) != CONTROL_MODE_PRO:
                if not await self._enter_pro_controls():
                    return ArmResult(
                        False,
                        ("camera is in Easy controls and did not switch to Pro controls (set it on the camera)",),
                    )
                data = await self._refresh()
            if data["status"].get(ST_FLAT_MODE) != FLAT_MODE_VIDEO:
                if not await self._enter_video_mode():
                    return ArmResult(False, ("could not switch the camera to video mode",))
                data = await self._refresh()
            for setting_id, option in wanted.items():
                if data["settings"].get(str(setting_id)) == option:
                    continue
                try:
                    await self._get("/gopro/camera/setting", {"setting": setting_id, "option": option})
                except httpx.HTTPStatusError as exc:
                    name, value = label(setting_id, option)
                    problems.append(f"camera refused {name}={value} (HTTP {exc.response.status_code})")
            mismatches = await self._read_back()
        except httpx.HTTPError as exc:
            return ArmResult(False, (self._unreachable(exc),))
        if mismatches:
            problems.append("settings did not take effect")
        return ArmResult(not problems and not mismatches, tuple(problems), mismatches)

    async def _enter_pro_controls(self) -> bool:
        """Switch the camera from Easy to Pro controls, which is what allows changing its settings."""
        try:
            await self._get("/gopro/camera/setting", {"setting": SETTING_CONTROL_MODE, "option": CONTROL_MODE_PRO})
        except httpx.HTTPStatusError:
            return False
        for _ in range(25):
            data = await self._refresh()
            if data["settings"].get(str(SETTING_CONTROL_MODE)) == CONTROL_MODE_PRO:
                return True
            await asyncio.sleep(0.2)
        return False

    async def _enter_video_mode(self) -> bool:
        await self._get("/gopro/camera/presets/set_group", {"id": PRESET_GROUP_VIDEO})
        for _ in range(25):
            if (await self._refresh())["status"].get(ST_FLAT_MODE) == FLAT_MODE_VIDEO:
                return True
            await asyncio.sleep(0.2)
        return False

    async def _read_back(self) -> dict[str, tuple[Any, Any]]:
        """Compare the camera's settings with the expected ones, allowing a moment to settle."""
        for attempt in range(4):
            data = await self._refresh()
            diffs = self._drift(data)
            if not diffs or attempt == 3:
                return diffs
            await asyncio.sleep(0.3)
        return {}

    def _drift(self, data: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
        """Settings of *data* that differ from the ones applied at arm time, as name -> (wanted, actual)."""
        diffs: dict[str, tuple[Any, Any]] = {}
        for setting_id, option in self._expected.items():
            actual = data["settings"].get(str(setting_id))
            if actual != option:
                name, wanted = label(setting_id, option)
                diffs[name] = (wanted, label(setting_id, actual)[1] if actual is not None else None)
        return diffs

    # -- offload ---------------------------------------------------------------

    supports_offload = True

    async def list_media(self) -> list[MediaFile]:
        try:
            listing = (await self._get_retrying("/gopro/media/list")).json()
        except httpx.HTTPError as exc:
            raise OffloadError(f"{self.label}: could not list the media ({self._unreachable(exc)})") from exc
        files = []
        for folder in listing.get("media", []):
            for item in folder.get("fs", []):
                size, created = item.get("s"), item.get("cre")
                files.append(
                    MediaFile(
                        f"{folder['d']}/{item['n']}",
                        int(size) if size is not None else None,
                        int(created) if created is not None else None,
                    )
                )
        return files

    async def media_info(self, file: MediaFile) -> dict[str, Any]:
        # The documented form has the folder's slash as it is; fall back to the encoded one.
        attempts: list[tuple[str, dict[str, Any] | None]] = [
            (f"/gopro/media/info?path={file.path}", None),
            ("/gopro/media/info", {"path": file.path}),
        ]
        error: Exception | None = None
        for url, params in attempts:
            try:
                return (await self._get_retrying(url, params)).json()
            except (httpx.HTTPError, ValueError) as exc:
                error = exc
        logger.warning("%s: no metadata for %s (%s)", self.label, file.path, _describe(error))
        return {}

    async def download(self, file: MediaFile, dest: Path, progress: ProgressCallback | None = None) -> Downloaded:
        # A dropped connection resumes from what has been copied. The window and the pauses start
        # again whenever an attempt got further than the one before it.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + TRANSIENT_WINDOW_S
        attempt, furthest = 0, -1
        while True:
            attempt += 1
            try:
                return await self._download_once(file, dest, progress)
            except _CopyInterrupted as exc:
                message = f"{self.label}: copying {file.path} failed after {exc.done} bytes ({type(exc.cause).__name__})"
                if exc.done > furthest:
                    furthest, attempt, deadline = exc.done, 1, loop.time() + TRANSIENT_WINDOW_S
                if loop.time() >= deadline or not isinstance(exc.cause, httpx.TransportError):
                    raise OffloadError(message) from exc.cause
                logger.info("%s; resuming", message)
                await asyncio.sleep(min(TRANSIENT_PAUSE_S * attempt, TRANSIENT_PAUSE_MAX_S))

    async def _download_once(self, file: MediaFile, dest: Path, progress: ProgressCallback | None) -> Downloaded:
        part = dest.with_name(dest.name + ".part")
        dest.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        done = 0
        if part.exists():
            with part.open("rb") as old:
                while chunk := old.read(1 << 20):
                    digest.update(chunk)
                    done += len(chunk)
        client = self._clients[self._active]
        headers = {"Range": f"bytes={done}-"} if done else {}
        timeout = httpx.Timeout(self._request_timeout, read=30.0)
        try:
            async with self._lock, client.stream("GET", f"/videos/DCIM/{file.path}", headers=headers, timeout=timeout) as response:
                response.raise_for_status()
                if done and response.status_code != 206:  # the camera ignored the range: start over
                    digest, done = hashlib.sha256(), 0
                with part.open("ab" if done else "wb") as out:
                    async for chunk in response.aiter_bytes(1 << 18):
                        out.write(chunk)
                        digest.update(chunk)
                        done += len(chunk)
                        if progress is not None:
                            progress(done, file.size)
        except httpx.HTTPError as exc:
            raise _CopyInterrupted(done, exc) from exc
        if file.size is not None and done != file.size:
            raise OffloadError(f"{self.label}: {file.path} is {done} bytes, the camera announced {file.size}")
        part.replace(dest)
        return Downloaded(dest, done, digest.hexdigest())

    # -- start and stop --------------------------------------------------------

    async def start(self) -> CommandResult:
        cmd_ts = utc_now_iso()
        try:
            data = await self._recent_state()
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, cmd_ts, error=self._unreachable(exc))
        st = data["status"]
        if st.get(ST_ENCODING):
            return CommandResult(False, cmd_ts, error="camera is already recording")
        if st.get(ST_FLAT_MODE) != FLAT_MODE_VIDEO:
            return CommandResult(False, cmd_ts, error=f"camera is not in video mode (mode {st.get(ST_FLAT_MODE)})")
        drift = self._drift(data)
        if drift:
            changed = ", ".join(f"{k}: wanted {w}, is {a}" for k, (w, a) in drift.items())
            return CommandResult(False, cmd_ts, error=f"settings changed since arming ({changed})")
        return await self._shutter(True)

    async def stop(self) -> CommandResult:
        return await self._shutter(False)

    async def _shutter(self, on: bool) -> CommandResult:
        op = "start" if on else "stop"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._confirm_timeout
        cmd_ts = utc_now_iso()
        ack_ts: str | None = None
        while True:
            try:
                await self._get(f"/gopro/camera/shutter/{op}")
                ack_ts = utc_now_iso()
                break
            except httpx.HTTPStatusError as exc:
                return CommandResult(False, cmd_ts, error=f"camera refused {op} (HTTP {exc.response.status_code})")
            except httpx.HTTPError as exc:
                # The command may have arrived although the answer did not.
                if await self._encoding_is(on):
                    return CommandResult(True, cmd_ts, confirm_ts=utc_now_iso())
                if loop.time() >= deadline:
                    return CommandResult(False, cmd_ts, error=f"{op} not delivered: {self._unreachable(exc)}")
                await asyncio.sleep(0.3)
        while loop.time() < deadline:
            if await self._encoding_is(on):
                return CommandResult(True, cmd_ts, ack_ts, utc_now_iso())
            await asyncio.sleep(0.1)
        return CommandResult(False, cmd_ts, ack_ts, error=f"{op} not confirmed within {self._confirm_timeout:g} s")

    async def _encoding_is(self, on: bool) -> bool:
        try:
            data = await self._refresh()
        except Exception:  # noqa: BLE001
            return False
        return bool(data["status"].get(ST_ENCODING)) == on
