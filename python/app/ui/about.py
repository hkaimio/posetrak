# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""about.py — the About box and the notices that come with the SOMA-77 pose model."""
from __future__ import annotations

from posetrak.detection.model_weights import DINOV3_NOTICE, NVIDIA_NOTICE


def about_html() -> str:
    """The text of the About box: the program, and the notices the licences of the optional pose model require."""
    return (
        "<h3>PoseTrak</h3>"
        "<p>Multi-camera video motion capture. Source code under the Apache License 2.0 and the MIT License.</p>"
        "<p><b>SOMA-77 pose model</b> (optional, downloaded separately on first use):<br>"
        f"{NVIDIA_NOTICE}.<br>{DINOV3_NOTICE}.</p>"
        "<p>The model weights are not part of PoseTrak; they are used under the NVIDIA Open Model License "
        "Agreement and the DINOv3 License Agreement, which you accept before they are downloaded.</p>"
    )


def show_about(parent) -> None:
    from PySide6.QtWidgets import QMessageBox

    QMessageBox.about(parent, "About PoseTrak", about_html())
