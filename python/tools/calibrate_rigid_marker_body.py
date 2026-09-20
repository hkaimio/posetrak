# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""calibrate_rigid_marker_body.py — moved to ``posetrak marker-body calibrate``.

The calibration itself is ``posetrak.calibration.rigid_marker_body`` and its
command is::

    posetrak --session session.db marker-body calibrate --capture <id> \\
        --time-start 34.4 --time-end 100.6 --marker-size 0.05 \\
        --marker-ids 2,3 --reference-id 2 --detect-dots --output sword_body.yaml

The helpers below are re-exported because many tools import them from here
(``from tools.calibrate_rigid_marker_body import load_camera_states``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posetrak.calibration.rigid_marker_body import (  # noqa: E402,F401
    _point_in_dilated_quad,
    cluster_dot_samples,
    robust_mean,
    triangulate_point_multi_view,
)
from posetrak.calibration.session_cameras import (  # noqa: E402,F401
    _resolve_intrinsics,
    load_camera_states,
    load_sync_table,
)


def main() -> None:
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
