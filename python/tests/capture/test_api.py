# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests of the capture daemon's HTTP/WebSocket API and the control lease."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from pipeline.capture.api import create_app  # noqa: E402
from pipeline.capture.lease import ControlLease  # noqa: E402
from pipeline.capture.service import CameraSlot, CaptureController, ChecklistItem  # noqa: E402
from posetrak.db.db import create_mocap_session, create_session  # noqa: E402
from tests.capture.scripted_driver import ScriptedDriver  # noqa: E402

TOKEN = "s3cret-token"
A = {"Authorization": f"Bearer {TOKEN}", "X-Client-Id": "phone-a"}
B = {"Authorization": f"Bearer {TOKEN}", "X-Client-Id": "phone-b"}


def make_app(tmp_path: Path, drivers=None, checklist=None, lease_timeout_s=30.0, web_dir=None):
    drivers = drivers if drivers is not None else [ScriptedDriver("cam0"), ScriptedDriver("cam1")]

    def factory() -> CaptureController:
        session = create_session(tmp_path / "s.db")
        sid = create_mocap_session(session, location="test")
        slots = [CameraSlot(d.label, f"inst-{d.label}", d) for d in drivers]
        return CaptureController(slots, session, sid, checklist=checklist, command_timeout_s=2.0)

    return create_app(factory, TOKEN, lease_timeout_s=lease_timeout_s, web_dir=web_dir), drivers


@pytest.fixture()
def client(tmp_path: Path):
    app, _ = make_app(tmp_path)
    with TestClient(app) as c:
        yield c


def lease(client, headers=A, **body):
    return client.post("/api/lease", json=body, headers=headers)


# -- authentication -----------------------------------------------------------


def test_api_requires_token(client: TestClient) -> None:
    assert client.get("/api/state").status_code == 401
    assert client.get("/api/state", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/state", headers={"Authorization": TOKEN}).status_code == 401
    assert client.get("/api/state", headers=A).status_code == 200
    assert client.get(f"/api/state?token={TOKEN}").status_code == 200
    assert client.post("/api/capture/arm").status_code == 401


def test_websocket_requires_token(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/api/events?token=wrong"):
            pass
    assert exc.value.code == 4401
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/events"):
            pass


def test_empty_token_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        create_app(lambda: None, "")


def test_static_files_need_no_token(tmp_path: Path) -> None:
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<h1>capture</h1>", encoding="utf-8")
    app, _ = make_app(tmp_path, web_dir=web)
    with TestClient(app) as c:
        assert c.get("/").status_code == 200 and "capture" in c.get("/").text
        assert c.get("/api/state").status_code == 401


# -- state ----------------------------------------------------------------------


def test_state_snapshot(client: TestClient) -> None:
    state = client.get("/api/state", headers=A).json()
    assert state["phase"] == "idle"
    assert set(state["cameras"]) == {"cam0", "cam1"}
    assert state["cameras"]["cam0"]["state"] == "ready"
    assert state["lease"] == {"holder": None, "label": None, "expires_in_s": None}


# -- lease -------------------------------------------------------------------------


def test_commands_need_the_lease(client: TestClient) -> None:
    assert client.post("/api/capture/arm", headers=A).status_code == 409
    no_id = {"Authorization": f"Bearer {TOKEN}"}
    assert client.post("/api/capture/arm", headers=no_id).status_code == 400
    assert lease(client, A, label="Harri's phone").status_code == 200
    assert client.post("/api/capture/arm", headers=A).status_code == 200


def test_second_client_is_view_only_until_takeover(client: TestClient) -> None:
    assert lease(client, A, label="phone").status_code == 200
    denied = lease(client, B, label="tablet")
    assert denied.status_code == 409 and denied.json()["detail"]["lease"]["holder"] == "phone-a"
    assert client.post("/api/capture/arm", headers=B).status_code == 409
    assert client.get("/api/state", headers=B).status_code == 200  # viewing is always allowed

    assert lease(client, B, label="tablet", takeover=True).status_code == 200
    assert client.post("/api/capture/arm", headers=A).status_code == 409
    assert client.post("/api/capture/arm", headers=B).status_code == 200


def test_lease_release_and_refresh(client: TestClient) -> None:
    lease(client, A)
    assert client.delete("/api/lease", headers=B).json()["holder"] == "phone-a"  # not the holder: no effect
    assert client.delete("/api/lease", headers=A).json()["holder"] is None
    assert lease(client, B).status_code == 200


def test_lease_expires_without_refresh(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path, lease_timeout_s=0.3)
    with TestClient(app) as c:
        assert lease(c, A).status_code == 200
        assert lease(c, B).status_code == 409
        time.sleep(0.4)
        assert lease(c, B).status_code == 200  # the lapsed lease did not block the operator


def test_lease_unit_semantics() -> None:
    now = [0.0]
    ls = ControlLease(10.0, clock=lambda: now[0])
    assert ls.acquire("a", "A").granted
    assert not ls.acquire("b").granted
    now[0] = 9.0
    assert ls.holds("a")  # a successful check refreshes
    now[0] = 18.0
    assert ls.holds("a")
    now[0] = 40.0
    assert not ls.holds("a") and ls.info().holder is None
    first = ls.acquire("a")
    taken = ls.acquire("b", takeover=True)
    assert first.granted and taken.granted and taken.taken_from == "a"
    assert not ls.release("a") and ls.release("b")


# -- commands -----------------------------------------------------------------------


def test_full_capture_over_http(client: TestClient) -> None:
    lease(client, A)
    armed = client.post("/api/capture/arm", headers=A).json()
    assert armed["ready"] and set(armed["cameras"]) == {"cam0", "cam1"}

    started = client.post("/api/capture/start", json={"label": "take 1"}, headers=A).json()
    assert started["ok"] and started["capture_id"]
    assert started["outcomes"]["cam0"]["confirm_ts"]
    assert client.get("/api/state", headers=A).json()["phase"] == "recording"

    stopped = client.post("/api/capture/stop", headers=A).json()
    assert stopped["ok"] and client.get("/api/state", headers=A).json()["phase"] == "stopped"


def test_wrong_phase_is_a_conflict_not_an_error(client: TestClient) -> None:
    lease(client, A)
    r = client.post("/api/capture/stop", headers=A)
    assert r.status_code == 409 and "cannot stop" in r.json()["detail"]["error"]
    assert client.post("/api/capture/start", json={}, headers=A).status_code == 409


def test_checklist_and_unknown_items(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path, checklist=[ChecklistItem("mode", "Mode set")])
    with TestClient(app) as c:
        lease(c, A)
        report = c.post("/api/capture/arm", headers=A).json()
        assert not report["ready"] and report["open_checklist"] == ["mode"]
        assert c.post("/api/checklist/mode", json={"done": True}, headers=A).status_code == 200
        assert c.post("/api/checklist/nope", json={}, headers=A).status_code == 404
        assert c.post("/api/capture/arm", headers=A).json()["ready"]
        item = c.get("/api/state", headers=A).json()["checklist"][0]
        assert item["done"] is True


def test_partial_failure_and_override_over_http(tmp_path: Path) -> None:
    bad = ScriptedDriver("cam1")
    bad.start_error = "camera busy"
    app, _ = make_app(tmp_path, drivers=[ScriptedDriver("cam0"), bad])
    with TestClient(app) as c:
        lease(c, A)
        c.post("/api/capture/arm", headers=A)
        started = c.post("/api/capture/start", json={}, headers=A).json()
        assert not started["ok"] and started["outcomes"]["cam1"]["error"] == "camera busy"
        state = c.get("/api/state", headers=A).json()
        assert [a["kind"] for a in state["alerts"]] == ["start_failed"]
        assert c.post("/api/cameras/cam1/override", json={"recording": True}, headers=A).status_code == 200
        assert c.get("/api/state", headers=A).json()["alerts"] == []
        assert c.post("/api/cameras/nope/override", json={"recording": True}, headers=A).status_code == 404


def test_disable_camera_and_manual_fallback(client: TestClient) -> None:
    lease(client, A)
    assert client.post("/api/cameras/cam1/enabled", json={"enabled": False}, headers=A).status_code == 200
    assert client.get("/api/state", headers=A).json()["cameras"]["cam1"]["enabled"] is False
    assert client.post("/api/cameras/cam0/manual", headers=A).status_code == 200
    assert client.get("/api/state", headers=A).json()["cameras"]["cam0"]["confidence"] == "manual"


# -- event stream --------------------------------------------------------------------


def drain_until(ws, predicate, limit=50):
    seen = []
    for _ in range(limit):
        msg = ws.receive_json()
        seen.append(msg)
        if predicate(msg):
            return seen
    raise AssertionError(f"condition not reached; saw {[m.get('kind') for m in seen]}")


def test_event_stream_delivers_ordered_events(client: TestClient) -> None:
    lease(client, A)
    with client.websocket_connect(f"/api/events?token={TOKEN}") as ws:
        first = ws.receive_json()
        assert first["type"] == "snapshot" and first["phase"] == "idle"
        client.post("/api/capture/arm", headers=A)
        seen = drain_until(ws, lambda m: m.get("kind") == "phase" and m["data"]["phase"] == "armed")
        seqs = [m["seq"] for m in seen]
        assert seqs == sorted(seqs) and seqs[0] > first["seq"]
        assert [m["data"]["phase"] for m in seen if m["kind"] == "phase"] == ["arming", "armed"]


def test_reconnect_snapshot_equals_live_state(client: TestClient) -> None:
    lease(client, A)
    client.post("/api/capture/arm", headers=A)
    client.post("/api/capture/start", json={}, headers=A)
    live = client.get("/api/state", headers=A).json()
    with client.websocket_connect(f"/api/events?token={TOKEN}") as ws:
        snap = ws.receive_json()
    snap.pop("type")
    for state in (snap, live):
        state["lease"].pop("expires_in_s")  # a running countdown
    assert snap == live


def test_lease_changes_are_broadcast(client: TestClient) -> None:
    with client.websocket_connect(f"/api/events?token={TOKEN}") as ws:
        ws.receive_json()
        lease(client, A, label="phone")
        msg = drain_until(ws, lambda m: m.get("kind") == "lease")[-1]
        assert msg["data"]["holder"] == "phone-a"
