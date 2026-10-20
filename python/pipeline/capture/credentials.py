# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Local store of camera COHN credentials.

A Hero13 provisioned for COHN on a WLAN gets a user name, a password, a
certificate and an address on that WLAN. The password grants full control of
the camera, so these live in the user's configuration directory, one file per
camera, and never in a session database (which is shared and archived with the
captures). A camera provisioned on several WLANs has one entry per SSID.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from pipeline.capture.config import DEFAULT_CONFIG_DIR

DEFAULT_CREDENTIALS_DIR: Final[Path] = DEFAULT_CONFIG_DIR / "cohn"


@dataclass(frozen=True)
class CohnCredentials:
    """How to reach one camera over COHN on one WLAN."""

    ssid: str
    ip_address: str
    username: str
    password: str
    #: PEM certificate the camera presents; the connection is pinned to it
    certificate: str


class CredentialStore:
    """Credentials of cameras, keyed by serial and SSID, as JSON files in *directory*."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or DEFAULT_CREDENTIALS_DIR

    def _path(self, serial: str) -> Path:
        if not serial.isalnum():
            raise ValueError(f"invalid camera serial {serial!r}")
        return self.directory / f"{serial}.json"

    def load(self, serial: str) -> list[CohnCredentials]:
        """Credentials stored for the camera, one per SSID; empty if it was never provisioned."""
        path = self._path(serial)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [CohnCredentials(ssid=ssid, **fields) for ssid, fields in data.items()]

    def save(self, serial: str, creds: CohnCredentials) -> None:
        """Add or replace the entry for ``creds.ssid``."""
        self.directory.mkdir(parents=True, exist_ok=True)
        entries = {c.ssid: c for c in self.load(serial)}
        entries[creds.ssid] = creds
        self._write(serial, entries)

    def forget(self, serial: str, ssid: str) -> bool:
        """Remove the entry for one SSID; return whether there was one."""
        entries = {c.ssid: c for c in self.load(serial)}
        if entries.pop(ssid, None) is None:
            return False
        if entries:
            self._write(serial, entries)
        else:
            self._path(serial).unlink()
        return True

    def _write(self, serial: str, entries: dict[str, CohnCredentials]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        data = {ssid: {k: v for k, v in asdict(c).items() if k != "ssid"} for ssid, c in entries.items()}
        path = self._path(serial)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        with contextlib.suppress(OSError):  # not meaningful on every platform
            os.chmod(tmp, 0o600)
        tmp.replace(path)
