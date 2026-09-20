# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""The old calibration script is a shim: it re-exports what other tools import from it."""
from __future__ import annotations

import pytest

import posetrak.calibration.rigid_marker_body as library
import posetrak.calibration.session_cameras as cameras
from tools import calibrate_rigid_marker_body as shim


@pytest.mark.parametrize(
    ("name", "home"),
    [
        ("load_camera_states", cameras),
        ("load_sync_table", cameras),
        ("_resolve_intrinsics", cameras),
        ("robust_mean", library),
        ("triangulate_point_multi_view", library),
        ("cluster_dot_samples", library),
        ("_point_in_dilated_quad", library),
    ],
)
def test_every_helper_other_tools_import_is_still_there_and_is_the_library_function(name: str, home) -> None:
    assert getattr(shim, name) is getattr(home, name)


def test_running_the_script_points_to_the_new_command() -> None:
    with pytest.raises(SystemExit) as exit_info:
        shim.main()

    assert "posetrak --session session.db marker-body calibrate" in str(exit_info.value)
