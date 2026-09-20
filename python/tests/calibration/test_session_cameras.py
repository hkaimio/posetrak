# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for loading an intrinsics calibration by id."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from posetrak.calibration.session_cameras import IntrinsicsNotFoundError, load_intrinsics
from posetrak.db.db import create_session


def _database(tmp_path: Path, *ids: str) -> sqlite3.Connection:
    conn = create_session(tmp_path / "s.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF")
    for i, calibration_id in enumerate(ids):
        conn.execute(
            "INSERT INTO intrinsics_calibrations (id, camera_mode_id, calibrated_at, distortion_model, fx, fy, cx, cy, "
            "image_width, image_height) VALUES (?, 'mode', '2026-09-06', 'radtan', ?, 1895.0, 1920.0, 1080.0, 3840, 2160)",
            (calibration_id, 1900.0 + i),
        )
    conn.commit()
    return conn


def test_a_calibration_is_found_by_its_id_or_a_unique_prefix(tmp_path: Path) -> None:
    conn = _database(tmp_path, "cc817fba-8d98-42fe", "1b6fa3e5-0000-0000")

    by_prefix = load_intrinsics(conn, "cc81")
    by_id = load_intrinsics(conn, "1b6fa3e5-0000-0000")

    assert by_prefix.calibration_id == "cc817fba-8d98-42fe" and by_id.calibration_id == "1b6fa3e5-0000-0000"
    assert (by_prefix.image_width, by_prefix.image_height) == (3840, 2160)
    assert by_prefix.state.K[0, 0] == pytest.approx(1900.0) and by_id.state.K[0, 0] == pytest.approx(1901.0)
    assert np.array_equal(by_prefix.state.K[:2, 2], [1920.0, 1080.0]) and by_prefix.state.R is None
    assert not by_prefix.state.fisheye


def test_an_unknown_id_and_an_ambiguous_prefix_are_told_apart(tmp_path: Path) -> None:
    conn = _database(tmp_path, "cc817fba-1", "cc817fba-2")

    with pytest.raises(IntrinsicsNotFoundError, match="no intrinsics calibration 'zzzz'"):
        load_intrinsics(conn, "zzzz")
    with pytest.raises(ValueError, match="ambiguous") as ambiguous:
        load_intrinsics(conn, "cc81")
    assert not isinstance(ambiguous.value, IntrinsicsNotFoundError)
