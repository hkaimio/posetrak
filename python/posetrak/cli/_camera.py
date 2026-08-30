# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Shared camera-instance resolution for CLI commands that take a camera by
label (e.g. ``session add-camera``, ``trial export-video``)."""

from __future__ import annotations

import sqlite3

import click

from posetrak.db.db import resolve_id_prefix


def resolve_camera_instance(conn: sqlite3.Connection, label_or_id: str) -> str:
    """Resolve *label_or_id* to a camera_instances.id.

    Tries an exact label match first (the natural way to refer to a camera
    day-to-day, e.g. ``pixel9``), then falls back to a UUID/prefix match via
    ``resolve_id_prefix`` for the rarer case of two instances sharing a label.
    """
    rows = conn.execute(
        "SELECT id FROM camera_instances WHERE label = ?", (label_or_id,)
    ).fetchall()
    if len(rows) == 1:
        return rows[0][0]
    if len(rows) > 1:
        ids = ", ".join(r[0] for r in rows)
        raise click.ClickException(
            f"Camera label {label_or_id!r} is ambiguous — {len(rows)} instances "
            f"share it ({ids}). Pass a camera_instances.id prefix instead."
        )
    try:
        return resolve_id_prefix(conn, "camera_instances", label_or_id)
    except ValueError as exc:
        raise click.ClickException(
            f"No camera instance found with label or id {label_or_id!r}"
        ) from exc
