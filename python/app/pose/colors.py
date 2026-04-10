"""colors.py — Shared color utilities for the pose UI."""
from __future__ import annotations

from PySide6.QtGui import QColor

# matplotlib tab10 palette (perceptually designed for categorical data).
# Gray (#7f7f7f) is omitted — it is reserved for unassigned tracks.
_PALETTE: list[QColor] = [
    QColor(31, 119, 180),   # blue
    QColor(255, 127, 14),   # orange
    QColor(44, 160, 44),    # green
    QColor(214, 39, 40),    # red
    QColor(148, 103, 189),  # purple
    QColor(140, 86, 75),    # brown
    QColor(227, 119, 194),  # pink
    QColor(188, 189, 34),   # olive
    QColor(23, 190, 207),   # cyan
]

UNASSIGNED_COLOR = QColor(100, 100, 100)


def person_color(name: str) -> QColor:
    """Return a stable, palette-based color for *name*."""
    return _PALETTE[hash(name) % len(_PALETTE)]
