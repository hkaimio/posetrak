# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""A fake GoPro camera speaking the subset of the Open GoPro HTTP API that the
capture driver uses, with programmable misbehaviour.

Endpoints (plain HTTP on a loopback port; the real camera serves HTTPS):

``GET /gopro/camera/state``
    ``{"status": {...}, "settings": {...}}``. Status ids used: 6 system hot,
    10 encoding active, 54 remaining storage (kB), 70 battery percent.
``GET /gopro/camera/shutter/start`` and ``/stop``
``GET /gopro/camera/setting?setting=<id>&option=<value>``
``GET /gopro/camera/presets/get``
``GET /gopro/camera/presets/set_group?id=<group>``
    Switches the active mode: group 1000 is video, 1001 photo, 1002 time lapse.

Behaviour knobs, all settable while the server runs:

``encoding_delay_s``   time between a start command and the encoding flag rising
``response_delay_s``   added to every response
``link_up``            False: connections are accepted and dropped without a reply
``fail_commands``      number of upcoming shutter commands answered with HTTP 500
``never_encodes``      shutter/start is acknowledged but the camera never records
``stop_after_s``       the camera stops recording by itself this long after start
``supported_options``  setting id -> allowed option values; others get HTTP 403
``ignore_settings``    setting changes are acknowledged but do not take effect
                       (a camera in a locked-down "easy" control mode)
``flat_mode``          active mode, status 89: 12 video, 13 time lapse, 16 photo
``require_auth``       expected ``Authorization`` header value, or None
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ST_SYSTEM_HOT = "6"
ST_ENCODING = "10"
ST_SD_REMAINING_KB = "54"
ST_BATTERY_PERCENT = "70"
ST_REMAINING_VIDEO_S = "35"
ST_FLAT_MODE = "89"

FLAT_VIDEO, FLAT_TIME_LAPSE, FLAT_PHOTO = 12, 13, 16
GROUP_TO_FLAT_MODE = {"1000": FLAT_VIDEO, "1001": FLAT_PHOTO, "1002": FLAT_TIME_LAPSE}

# Open GoPro setting ids
SET_RESOLUTION = "2"
SET_FPS = "3"
SET_LENS = "121"


class FakeGoPro:
    """One fake camera. Use as a context manager or call :meth:`start` / :meth:`stop`."""

    def __init__(self, serial: str = "1234") -> None:
        self.serial = serial
        self.encoding_delay_s = 0.0
        self.response_delay_s = 0.0
        self.link_up = True
        self.fail_commands = 0
        self.never_encodes = False
        self.stop_after_s: float | None = None
        self.require_auth: str | None = None
        self.supported_options: dict[str, set[int]] = {
            SET_RESOLUTION: {1, 4, 9, 12},
            SET_FPS: {0, 1, 5, 8, 9, 10, 13},
            SET_LENS: {0, 4, 7},
        }
        self.settings: dict[str, int] = {SET_RESOLUTION: 9, SET_FPS: 8, SET_LENS: 0}
        self.ignore_settings = False
        self.flat_mode = FLAT_VIDEO
        self.remaining_video_s = 20_000
        self.battery_percent = 100
        self.system_hot = False
        self.storage_kb = 100_000_000
        self.commands: list[tuple[float, str]] = []  # (monotonic time, path) of every request served
        self._record_started_at: float | None = None
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    # -- lifecycle ---------------------------------------------------------

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> FakeGoPro:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> FakeGoPro:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- camera model ------------------------------------------------------

    def is_encoding(self) -> bool:
        with self._lock:
            started = self._record_started_at
            if started is None or self.never_encodes:
                return False
            now = time.monotonic()
            if self.stop_after_s is not None and now - started >= self.stop_after_s:
                self._record_started_at = None
                return False
            return now - started >= self.encoding_delay_s

    def state_json(self) -> dict:
        return {
            "status": {
                ST_ENCODING: int(self.is_encoding()),
                ST_BATTERY_PERCENT: self.battery_percent,
                ST_SYSTEM_HOT: int(self.system_hot),
                ST_SD_REMAINING_KB: self.storage_kb,
                ST_REMAINING_VIDEO_S: self.remaining_video_s,
                ST_FLAT_MODE: self.flat_mode,
            },
            "settings": dict(self.settings),
        }

    def commands_to(self, path_prefix: str) -> list[float]:
        """Monotonic times at which requests whose path starts with *path_prefix* arrived."""
        return [t for t, p in self.commands if p.startswith(path_prefix)]

    # -- HTTP --------------------------------------------------------------

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        if not self.link_up:
            handler.close_connection = True
            handler.connection.close()
            return
        if self.response_delay_s:
            time.sleep(self.response_delay_s)
        url = urlparse(handler.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        with self._lock:
            self.commands.append((time.monotonic(), url.path))
        if self.require_auth is not None and handler.headers.get("Authorization") != self.require_auth:
            return self._reply(handler, 401, {"error": "unauthorized"})

        if url.path == "/gopro/camera/state":
            return self._reply(handler, 200, self.state_json())
        if url.path in ("/gopro/camera/shutter/start", "/gopro/camera/shutter/stop"):
            with self._lock:
                if self.fail_commands > 0:
                    self.fail_commands -= 1
                    return self._reply(handler, 500, {"error": "injected failure"})
                if url.path.endswith("start"):
                    if self._record_started_at is None:
                        self._record_started_at = time.monotonic()
                else:
                    self._record_started_at = None
            return self._reply(handler, 200, {})
        if url.path == "/gopro/camera/setting":
            sid, option = query.get("setting"), int(query.get("option", "-1"))
            if sid not in self.supported_options or option not in self.supported_options[sid]:
                return self._reply(handler, 403, {"error": "unsupported setting"})
            if not self.ignore_settings:
                self.settings[sid] = option
            return self._reply(handler, 200, {})
        if url.path == "/gopro/camera/presets/set_group":
            mode = GROUP_TO_FLAT_MODE.get(query.get("id", ""))
            if mode is None:
                return self._reply(handler, 403, {"error": "unknown group"})
            with self._lock:
                if self._record_started_at is not None:
                    return self._reply(handler, 500, {"error": "busy"})
            self.flat_mode = mode
            return self._reply(handler, 200, {})
        if url.path == "/gopro/camera/presets/get":
            return self._reply(handler, 200, {"presetGroupArray": [], "serial": self.serial})
        self._reply(handler, 404, {"error": "not found"})

    @staticmethod
    def _reply(handler: BaseHTTPRequestHandler, code: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        handler.send_response(code)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def _make_handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                try:
                    fake._handle(self)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args) -> None:  # silence request logging
                pass

        return Handler
