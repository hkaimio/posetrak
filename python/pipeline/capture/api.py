# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""HTTP + WebSocket API of the capture daemon.

``GET /api/state``                       controller snapshot plus the lease
``WebSocket /api/events?token=...``      snapshot, then every controller event
``POST /api/lease``                      take or refresh the control lease
``DELETE /api/lease``                    release it
``POST /api/capture/{arm,start,stop}``   capture commands (lease holder only)
``POST /api/checklist/{item}``           tick or untick a checklist item
``POST /api/cameras/{label}/override``   record a by-hand start/stop
``POST /api/cameras/{label}/enabled``    include or exclude a camera
``POST /api/cameras/{label}/manual``     fall back to manual operation

Every ``/api`` request carries the access token, as ``Authorization: Bearer
<token>`` or a ``token`` query parameter (browsers cannot set headers on a
WebSocket). Commands also carry ``X-Client-Id``, a random id the client keeps;
only the lease holder's id is accepted. Static files (the web UI) are served
without a token: they contain nothing secret and must load before the page
knows the token.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import logging
import secrets
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline.capture.lease import DEFAULT_TIMEOUT_S, ControlLease
from pipeline.capture.modes import ModeError
from pipeline.capture.service import ArmReport, CaptureController, CaptureReport, CaptureStateError

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"


class _RevalidatedFiles(StaticFiles):
    """Static files a browser must revalidate on every load.

    Without a cache header a browser may reuse an old copy of the UI for hours, which
    leaves it out of step with the API of a newer daemon.
    """

    async def get_response(self, path, scope):  # type: ignore[no-untyped-def]
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


class LeaseRequest(BaseModel):
    label: str | None = None
    takeover: bool = False


class ArmRequest(BaseModel):
    #: setup that chooses the capture modes; None takes the default one
    setup: str | None = None
    #: camera label -> mode name, overriding the setup for this capture
    modes: dict[str, str] | None = None


class AutoOffloadRequest(BaseModel):
    enabled: bool


class StartRequest(BaseModel):
    label: str | None = None
    force: bool = False


class TickRequest(BaseModel):
    done: bool = True


class OverrideRequest(BaseModel):
    recording: bool


class EnabledRequest(BaseModel):
    enabled: bool


def _arm_report(report: ArmReport) -> dict[str, Any]:
    return {
        "ready": report.ready,
        "cameras": {k: dataclasses.asdict(v) for k, v in report.cameras.items()},
        "problems": {k: list(v) for k, v in report.problems.items()},
        "open_checklist": list(report.open_checklist),
    }


def _capture_report(report: CaptureReport) -> dict[str, Any]:
    return {
        "ok": report.ok,
        "capture_id": report.capture_id,
        "outcomes": {k: dataclasses.asdict(v) for k, v in report.outcomes.items()},
    }


def create_app(
    controller_factory: Callable[[], CaptureController],
    token: str,
    *,
    lease_timeout_s: float = DEFAULT_TIMEOUT_S,
    web_dir: Path | None = WEB_DIR,
) -> FastAPI:
    """Build the API around a controller.

    The controller is created by *controller_factory* on the server's event
    loop thread (its database connection must be used from that thread only)
    and connected during application startup.
    """
    if not token:
        raise ValueError("an access token is required")

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        controller = controller_factory()
        app.state.controller = controller
        app.state.lease = ControlLease(lease_timeout_s)
        await controller.connect()

        async def sweep() -> None:
            while True:
                await asyncio.sleep(1.0)
                if app.state.lease.sweep():
                    controller.bus.publish("lease", **_lease_dict(app.state.lease))

        sweeper = asyncio.create_task(sweep())
        try:
            yield
        finally:
            sweeper.cancel()
            await controller.close()

    app = FastAPI(title="posetrak capture", lifespan=lifespan)

    def check_token(authorization: str | None = Header(None), token_q: str | None = Query(None, alias="token")) -> None:
        supplied = token_q
        if authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:]
        if supplied is None or not secrets.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(401, "unauthorized")

    def lease_holder(request: Request, x_client_id: str | None = Header(None)) -> str:
        if not x_client_id:
            raise HTTPException(400, "X-Client-Id header required")
        if not request.app.state.lease.holds(x_client_id):
            raise HTTPException(409, {"error": "not lease holder", "lease": _lease_dict(request.app.state.lease)})
        return x_client_id

    def controller_of(request: Request) -> CaptureController:
        return request.app.state.controller

    def _lease_dict(lease: ControlLease) -> dict[str, Any]:
        info = lease.info()
        return {"holder": info.holder, "label": info.label, "expires_in_s": info.expires_in_s}

    api = APIRouter(prefix="/api", dependencies=[Depends(check_token)])

    @api.get("/state")
    async def state(request: Request) -> dict[str, Any]:
        return {**controller_of(request).snapshot(), "lease": _lease_dict(request.app.state.lease)}

    @api.post("/lease")
    async def take_lease(
        body: LeaseRequest, request: Request, x_client_id: str | None = Header(None)
    ) -> dict[str, Any]:
        if not x_client_id:
            raise HTTPException(400, "X-Client-Id header required")
        lease: ControlLease = request.app.state.lease
        result = lease.acquire(x_client_id, body.label, takeover=body.takeover)
        if not result.granted:
            raise HTTPException(409, {"error": "lease held by another client", "lease": _lease_dict(lease)})
        controller_of(request).bus.publish("lease", taken_from=result.taken_from, **_lease_dict(lease))
        return _lease_dict(lease)

    @api.delete("/lease")
    async def release_lease(request: Request, x_client_id: str | None = Header(None)) -> dict[str, Any]:
        lease: ControlLease = request.app.state.lease
        if x_client_id and lease.release(x_client_id):
            controller_of(request).bus.publish("lease", **_lease_dict(lease))
        return _lease_dict(lease)

    commands = APIRouter(dependencies=[Depends(lease_holder)])

    def _guard(fn):
        @functools.wraps(fn)
        async def run(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except CaptureStateError as exc:
                raise HTTPException(409, {"error": str(exc)}) from exc
            except ModeError as exc:
                raise HTTPException(400, {"error": str(exc)}) from exc
            except KeyError as exc:
                raise HTTPException(404, f"unknown {exc}") from exc

        return run

    @commands.post("/capture/arm")
    @_guard
    async def arm(request: Request, body: ArmRequest | None = None) -> dict[str, Any]:
        body = body or ArmRequest()
        return _arm_report(await controller_of(request).arm(body.setup, body.modes))

    @commands.post("/capture/start")
    @_guard
    async def start(body: StartRequest, request: Request) -> dict[str, Any]:
        return _capture_report(await controller_of(request).start_capture(body.label, force=body.force))

    @commands.post("/capture/offload")
    @_guard
    async def offload(request: Request) -> dict[str, Any]:
        controller_of(request).offload_now()
        return {"ok": True}

    @commands.post("/offload/cancel")
    @_guard
    async def cancel_offload(request: Request) -> dict[str, Any]:
        controller_of(request).cancel_offload()
        return {"ok": True}

    @commands.post("/offload/auto")
    @_guard
    async def auto_offload(body: AutoOffloadRequest, request: Request) -> dict[str, Any]:
        controller_of(request).set_auto_offload(body.enabled)
        return {"ok": True}

    @commands.post("/capture/stop")
    @_guard
    async def stop(request: Request) -> dict[str, Any]:
        return _capture_report(await controller_of(request).stop_capture())

    @commands.post("/checklist/{item}")
    @_guard
    async def tick(item: str, body: TickRequest, request: Request) -> dict[str, Any]:
        controller_of(request).tick(item, body.done)
        return {"item": item, "done": body.done}

    @commands.post("/cameras/{label}/override")
    @_guard
    async def override(label: str, body: OverrideRequest, request: Request) -> dict[str, Any]:
        controller_of(request).operator_override(label, body.recording)
        return {"label": label, "recording": body.recording}

    @commands.post("/cameras/{label}/enabled")
    @_guard
    async def enabled(label: str, body: EnabledRequest, request: Request) -> dict[str, Any]:
        controller_of(request).set_enabled(label, body.enabled)
        return {"label": label, "enabled": body.enabled}

    @commands.post("/cameras/{label}/manual")
    @_guard
    async def manual(label: str, request: Request) -> dict[str, Any]:
        await controller_of(request).fallback_to_manual(label)
        return {"label": label, "manual": True}

    api.include_router(commands)
    app.include_router(api)

    @app.websocket("/api/events")
    async def events(ws: WebSocket, token_q: str | None = Query(None, alias="token")) -> None:
        if token_q is None or not secrets.compare_digest(token_q.encode(), token.encode()):
            await ws.close(code=4401)
            return
        await ws.accept()
        controller: CaptureController = ws.app.state.controller
        # Subscribe before taking the snapshot so no event falls in between;
        # events with seq <= snapshot.seq are already reflected in it.
        async with controller.bus.subscribe() as queue:
            await ws.send_json(
                {"type": "snapshot", **controller.snapshot(), "lease": _lease_dict(ws.app.state.lease)}
            )
            try:
                while True:
                    event = await queue.get()
                    await ws.send_json(
                        {"type": "event", "seq": event.seq, "kind": event.kind, "ts": event.ts, "data": event.data}
                    )
            except (WebSocketDisconnect, RuntimeError):
                return

    if web_dir is not None and web_dir.is_dir():
        app.mount("/", _RevalidatedFiles(directory=web_dir, html=True), name="web")

    return app
