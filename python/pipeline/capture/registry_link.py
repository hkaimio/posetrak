# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Linking physical cameras to the camera registry.

A registry ``camera_instances`` row is the camera as the rest of posetrak knows
it (its label, calibrations, videos). Capture control reaches the physical
camera through a ``control_endpoints`` row, which is the link between the two:
it names the control method and the device identity (for a Hero13 the last
digits of its serial number, which is also its Bluetooth name).

``camera_instances.serial_number`` is entered by hand and is often empty, so
the endpoint row is the link and the serial number is only a cross-check. When
the camera reports its full serial it is offered for an empty registry field,
and a differing one is reported and never overwritten.
"""

from __future__ import annotations

import enum
import sqlite3
from dataclasses import dataclass

from posetrak.db.capture_control import ControlEndpoint, get_control_endpoint, set_control_endpoint


class LinkError(RuntimeError):
    """A camera cannot be resolved or linked in the registry."""


@dataclass(frozen=True)
class RegistryCamera:
    """A registry camera instance and what capture control knows about it."""

    instance_id: str
    label: str
    #: manufacturer serial as entered in the registry; None or empty when unknown
    serial_number: str | None
    endpoint: ControlEndpoint | None

    @property
    def gopro_serial(self) -> str | None:
        """Device identity of a COHN-controlled Hero13, or None if not linked."""
        if self.endpoint is None or self.endpoint.method != "gopro_cohn":
            return None
        return self.endpoint.gopro_serial or None


class SerialCheck(enum.Enum):
    FILLED = "filled"  # the registry field was empty and now holds the camera's serial
    SAME = "same"  # the registry already held this serial
    DIFFERS = "differs"  # the registry holds another serial; left untouched


def find_camera(registry: sqlite3.Connection, label: str) -> RegistryCamera:
    """The registry camera with this label.

    Raises
    ------
    LinkError
        If no camera has the label; the message lists the labels that exist.
    """
    row = registry.execute(
        "SELECT id, label, serial_number FROM camera_instances WHERE label = ?", (label,)
    ).fetchone()
    if row is None:
        known = [r[0] for r in registry.execute("SELECT label FROM camera_instances ORDER BY label")]
        hint = f" (registered: {', '.join(known)})" if known else " (the registry has no cameras)"
        raise LinkError(f"camera {label!r} is not in the camera registry{hint}")
    return RegistryCamera(row[0], row[1], row[2] or None, get_control_endpoint(registry, row[0]))


def cameras_by_gopro_serial(registry: sqlite3.Connection) -> dict[str, str]:
    """Labels of the cameras linked as COHN cameras, keyed by device serial."""
    rows = registry.execute(
        "SELECT e.gopro_serial, c.label FROM control_endpoints e "
        "JOIN camera_instances c ON c.id = e.camera_instance_id "
        "WHERE e.method = 'gopro_cohn' AND e.gopro_serial IS NOT NULL"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def serial_matches(full_serial: str, device_serial: str) -> bool:
    """Whether a full manufacturer serial belongs to a device identity (its trailing digits)."""
    return bool(device_serial) and full_serial.endswith(device_serial)


def check_link(registry: sqlite3.Connection, camera: RegistryCamera, device_serial: str) -> None:
    """Raise :class:`LinkError` unless *camera* can be linked to the Hero13 with this device serial.

    The camera must not be linked to a different Hero13 or controlled by another
    method, no other registry camera may hold the serial, and a serial number
    already in the registry must be consistent with it.
    """
    if camera.gopro_serial and camera.gopro_serial != device_serial:
        raise LinkError(
            f"{camera.label} is already linked to camera {camera.gopro_serial}, not {device_serial}"
        )
    if camera.endpoint is not None and camera.endpoint.method != "gopro_cohn":
        raise LinkError(f"{camera.label} is controlled by method {camera.endpoint.method!r}, not gopro_cohn")
    other = cameras_by_gopro_serial(registry).get(device_serial)
    if other is not None and other != camera.label:
        raise LinkError(f"camera {device_serial} is already linked to {other}")
    if camera.serial_number and not serial_matches(camera.serial_number, device_serial):
        raise LinkError(
            f"{camera.label} has serial number {camera.serial_number} in the registry, "
            f"which does not end in {device_serial}"
        )


def link_gopro(registry: sqlite3.Connection, camera: RegistryCamera, device_serial: str) -> None:
    """Make *camera* the controllable Hero13 with this device serial (see :func:`check_link`)."""
    check_link(registry, camera, device_serial)
    notes = camera.endpoint.notes if camera.endpoint else None
    set_control_endpoint(
        registry, ControlEndpoint(camera.instance_id, "gopro_cohn", gopro_serial=device_serial, notes=notes)
    )


def reconcile_serial(registry: sqlite3.Connection, camera: RegistryCamera, full_serial: str, *, fill: bool = True) -> SerialCheck:
    """Compare the serial the camera reported with the registry's, filling an empty field.

    Parameters
    ----------
    fill:
        Write the camera's serial into an empty registry field. With False an
        empty field is reported as FILLED without being written.
    """
    if camera.serial_number:
        return SerialCheck.SAME if camera.serial_number == full_serial else SerialCheck.DIFFERS
    if fill:
        with registry:
            registry.execute("UPDATE camera_instances SET serial_number = ? WHERE id = ?", (full_serial, camera.instance_id))
    return SerialCheck.FILLED
