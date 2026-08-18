# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""colors.py — Shared color utilities for the pose UI."""
from __future__ import annotations

import hashlib

from PySide6.QtGui import QColor

# matplotlib tab10 palette (perceptually designed for categorical data).
# Gray (#7f7f7f) is omitted — it is reserved for unassigned tracks.
_PALETTE: list[QColor] = [
    QColor(31, 119, 180),   # blue
    QColor(214, 39, 40),    # red
    QColor(44, 160, 44),    # green
    QColor(255, 127, 14),   # orange
    QColor(148, 103, 189),  # purple
    QColor(23, 190, 207),   # cyan
    QColor(227, 119, 194),  # pink
    QColor(140, 86, 75),    # brown
    QColor(188, 189, 34),   # olive
]

UNASSIGNED_COLOR = QColor(100, 100, 100)


def person_color(name: str) -> QColor:
    """Return a stable, palette-based color for *name*.

    Uses MD5 (non-security) so the mapping is deterministic across Python
    process restarts regardless of PYTHONHASHSEED.
    """
    digest = hashlib.md5(name.encode(), usedforsecurity=False).hexdigest()
    idx = int(digest[:8], 16) % len(_PALETTE)
    return _PALETTE[idx]
