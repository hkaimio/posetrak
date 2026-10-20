# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Bluetooth side of a Hero13: provisioning COHN on a WLAN.

COHN credentials come from the camera, over Bluetooth, once per camera and
WLAN: the camera is told to join the WLAN, then to create COHN credentials
(user name, password, certificate) on it, which are read back and stored with
:class:`~pipeline.capture.credentials.CredentialStore`. A camera can be
provisioned on several WLANs; it then joins whichever it finds.

The Open GoPro SDK is imported lazily; it is only needed for this step.

Two details matter on Windows:

* The SDK builds a controller for the host's WiFi even for Bluetooth-only
  use, and closing the camera object runs ``netsh wlan disconnect``, which
  drops this machine from its own WLAN. :func:`null_wifi_adapter` replaces
  that controller with one that does nothing.
* A camera that has never been paired must be in pairing mode (hold the
  mode button until it shows the pairing screen) for the first connection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.capture.config import DEFAULT_CONFIG_DIR
from pipeline.capture.credentials import CohnCredentials, CredentialStore

logger = logging.getLogger(__name__)

#: Where the SDK keeps its own COHN bookkeeping (separate from our credential store).
SDK_STATE_DIR = DEFAULT_CONFIG_DIR / "sdk"


class ProvisioningError(RuntimeError):
    """The camera could not be provisioned."""


@dataclass(frozen=True)
class VisibleNetwork:
    ssid: str
    frequency_mhz: int
    signal_bars: int
    #: the camera already holds credentials for this network
    configured: bool
    associated: bool


def null_wifi_adapter() -> Any:
    """A WiFi controller class for the SDK that never touches the host's WiFi."""
    from open_gopro.network.wifi.controller import SsidState, WifiController

    class NullWifi(WifiController):
        def __init__(self, interface: Any = None, password: Any = None) -> None:
            super().__init__(interface, password)

        async def connect(self, ssid: str, password: str, timeout: float = 15) -> bool:
            return False

        async def disconnect(self) -> bool:
            return True

        def current(self) -> Any:
            return None, SsidState.DISCONNECTED

        def available_interfaces(self) -> list[str]:
            return []

        def power(self, power: bool) -> bool:
            return False

        @property
        def is_on(self) -> bool:
            return False

    return NullWifi


def open_camera(serial: str, state_dir: Path | None = None) -> Any:
    """An SDK camera object connected over Bluetooth only; use as an async context manager."""
    from open_gopro import WirelessGoPro

    state_dir = state_dir or SDK_STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)
    return WirelessGoPro(
        target=serial,
        interfaces={WirelessGoPro.Interface.BLE},
        cohn_db=state_dir / f"{serial}.json",
        wifi_adapter=null_wifi_adapter(),
    )


async def list_networks(serial: str, state_dir: Path | None = None) -> list[VisibleNetwork]:
    """The WLANs the camera sees, and which of them it has credentials for."""
    from open_gopro.models.proto import EnumScanEntryFlags as Flags

    async with open_camera(serial, state_dir) as gopro:
        await gopro.cohn.wait_until_ready()
        result = await gopro.access_point.scan_wifi_networks()
        return [
            VisibleNetwork(
                e.ssid,
                e.signal_frequency_mhz,
                e.signal_strength_bars,
                bool(e.scan_entry_flags & Flags.SCAN_FLAG_CONFIGURED),
                bool(e.scan_entry_flags & Flags.SCAN_FLAG_ASSOCIATED),
            )
            for e in result.unwrap().entries
        ]


async def provision(
    serial: str,
    ssid: str,
    password: str,
    store: CredentialStore,
    *,
    force: bool = False,
    state_dir: Path | None = None,
) -> CohnCredentials:
    """Join *ssid*, create COHN credentials on it and store them.

    Parameters
    ----------
    force:
        Create new credentials even if the camera already has some for this network.

    Raises
    ------
    ProvisioningError
        If the camera did not end up with a complete set of credentials.
    """
    async with open_camera(serial, state_dir) as gopro:
        await gopro.cohn.wait_until_ready()
        joined = await gopro.access_point.connect(ssid, password)
        logger.info("%s: joined %r: %s", serial, ssid, joined)
        await gopro.cohn.configure(force_reprovision=force)
        creds = gopro.cohn.credentials
        if creds is None or not creds.is_complete:
            raise ProvisioningError(f"camera {serial} did not return complete COHN credentials on {ssid!r}")
        result = CohnCredentials(ssid, creds.ip_address, creds.username, creds.password, creds.certificate)
    store.save(serial, result)
    return result
