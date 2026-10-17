# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests of the fake camera itself: the capture tests rely on its behaviour."""

from __future__ import annotations

import http.client
import json
import time
from urllib.parse import urlparse

import pytest

from tests.capture.fake_gopro import (
    SET_FPS,
    ST_BATTERY_PERCENT,
    ST_ENCODING,
    FakeGoPro,
)


def get(fake: FakeGoPro, path: str, headers: dict | None = None, timeout: float = 2.0):
    u = urlparse(fake.base_url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    try:
        conn.request("GET", path, headers=headers or {})
        r = conn.getresponse()
        body = r.read()
        return r.status, json.loads(body) if body else None
    finally:
        conn.close()


@pytest.fixture()
def fake():
    with FakeGoPro("1234") as f:
        yield f


def test_state_and_shutter(fake: FakeGoPro) -> None:
    status, state = get(fake, "/gopro/camera/state")
    assert status == 200 and state["status"][ST_ENCODING] == 0
    assert state["status"][ST_BATTERY_PERCENT] == 100

    assert get(fake, "/gopro/camera/shutter/start")[0] == 200
    assert get(fake, "/gopro/camera/state")[1]["status"][ST_ENCODING] == 1
    assert get(fake, "/gopro/camera/shutter/stop")[0] == 200
    assert get(fake, "/gopro/camera/state")[1]["status"][ST_ENCODING] == 0


def test_encoding_flag_rises_after_delay(fake: FakeGoPro) -> None:
    fake.encoding_delay_s = 0.3
    get(fake, "/gopro/camera/shutter/start")
    assert get(fake, "/gopro/camera/state")[1]["status"][ST_ENCODING] == 0
    time.sleep(0.35)
    assert get(fake, "/gopro/camera/state")[1]["status"][ST_ENCODING] == 1


def test_never_encodes_and_stops_on_its_own(fake: FakeGoPro) -> None:
    fake.never_encodes = True
    assert get(fake, "/gopro/camera/shutter/start")[0] == 200
    assert get(fake, "/gopro/camera/state")[1]["status"][ST_ENCODING] == 0

    fake.never_encodes = False
    fake.stop_after_s = 0.2
    get(fake, "/gopro/camera/shutter/start")
    assert get(fake, "/gopro/camera/state")[1]["status"][ST_ENCODING] == 1
    time.sleep(0.25)
    assert get(fake, "/gopro/camera/state")[1]["status"][ST_ENCODING] == 0


def test_injected_command_failures(fake: FakeGoPro) -> None:
    fake.fail_commands = 1
    assert get(fake, "/gopro/camera/shutter/start")[0] == 500
    assert get(fake, "/gopro/camera/shutter/start")[0] == 200


def test_link_down_drops_connections(fake: FakeGoPro) -> None:
    fake.link_up = False
    with pytest.raises((http.client.HTTPException, OSError)):
        get(fake, "/gopro/camera/state")
    fake.link_up = True
    assert get(fake, "/gopro/camera/state")[0] == 200


def test_settings_validation(fake: FakeGoPro) -> None:
    assert get(fake, f"/gopro/camera/setting?setting={SET_FPS}&option=10")[0] == 200
    assert get(fake, "/gopro/camera/state")[1]["settings"][SET_FPS] == 10
    assert get(fake, f"/gopro/camera/setting?setting={SET_FPS}&option=99")[0] == 403
    assert get(fake, "/gopro/camera/setting?setting=555&option=1")[0] == 403


def test_auth_required(fake: FakeGoPro) -> None:
    fake.require_auth = "Basic abc"
    assert get(fake, "/gopro/camera/state")[0] == 401
    assert get(fake, "/gopro/camera/state", {"Authorization": "Basic abc"})[0] == 200


def test_response_delay_and_command_log(fake: FakeGoPro) -> None:
    fake.response_delay_s = 0.2
    t0 = time.monotonic()
    get(fake, "/gopro/camera/state")
    assert time.monotonic() - t0 >= 0.2
    assert len(fake.commands_to("/gopro/camera/state")) == 1


def test_unknown_path(fake: FakeGoPro) -> None:
    assert get(fake, "/nope")[0] == 404
