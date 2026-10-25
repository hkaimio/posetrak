# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Stores of camera COHN credentials.

A Hero13 provisioned for COHN on a WLAN gets a user name, a password, a
certificate and an address on that WLAN. The password grants full control of
the camera, so the credentials never go in a session database (which is shared
and archived with the captures). They live in a registry table that only the
registry has (:class:`RegistryCredentialStore`). Cameras provisioned before
that are read from one file per camera in the user's configuration directory
(:class:`CredentialStore`) until they are imported. A camera provisioned on
several WLANs has one entry per SSID.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from pipeline.capture.config import DEFAULT_CONFIG_DIR
from posetrak.db.capture_control import (
    ControlCredential,
    delete_control_credential,
    list_control_credentials,
    set_control_credential,
)

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


class RegistryCredentialStore:
    """Credentials in the registry's ``control_credentials`` table, keyed by device serial.

    A camera is found through its ``control_endpoints`` row, so it has to be linked to a
    registry camera before credentials can be saved for it. Cameras without credentials in
    the registry fall back to the files of *legacy*, if given.
    """

    def __init__(self, registry: sqlite3.Connection, legacy: CredentialStore | None = None) -> None:
        self._registry = registry
        self.legacy = legacy

    def instance_id(self, serial: str) -> str | None:
        """The registry camera linked to the device serial, if any."""
        row = self._registry.execute(
            "SELECT camera_instance_id FROM control_endpoints WHERE method = 'gopro_cohn' AND gopro_serial = ?",
            (serial,),
        ).fetchone()
        return row[0] if row else None

    def load(self, serial: str) -> list[CohnCredentials]:
        """Credentials of the camera, one per SSID; empty if it was never provisioned."""
        instance = self.instance_id(serial)
        found = (
            [CohnCredentials(c.ssid, c.ip_address, c.username, c.password, c.certificate)
             for c in list_control_credentials(self._registry, instance)]
            if instance
            else []
        )
        if not found and self.legacy is not None:
            return self.legacy.load(serial)
        return found

    def save(self, serial: str, creds: CohnCredentials) -> None:
        """Add or replace the entry for ``creds.ssid``.

        Raises
        ------
        LookupError
            If no registry camera is linked to the device serial.
        """
        instance = self.instance_id(serial)
        if instance is None:
            raise LookupError(f"camera {serial} is not linked to a registry camera")
        set_control_credential(
            self._registry,
            ControlCredential(instance, creds.ssid, creds.ip_address, creds.username, creds.password, creds.certificate),
        )

    def forget(self, serial: str, ssid: str) -> bool:
        """Remove the entry for one SSID; return whether there was one."""
        instance = self.instance_id(serial)
        return bool(instance) and delete_control_credential(self._registry, instance, ssid)

    def import_legacy(self) -> dict[str, int]:
        """Copy the credentials of the legacy files into the registry.

        Returns
        -------
        dict[str, int]
            Entries imported per device serial. Files of cameras that are not linked to a
            registry camera are skipped (they are listed with a count of 0).
        """
        imported: dict[str, int] = {}
        if self.legacy is None or not self.legacy.directory.is_dir():
            return imported
        for path in sorted(self.legacy.directory.glob("*.json")):
            serial = path.stem
            if self.instance_id(serial) is None:
                imported[serial] = 0
                continue
            entries = self.legacy.load(serial)
            for creds in entries:
                self.save(serial, creds)
            imported[serial] = len(entries)
        return imported
