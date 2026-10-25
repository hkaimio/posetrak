# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""pose_model_gui.py — choosing the pose model in the GUI: licence dialog and weights download.

SOMA-77 is the recommended pose model. Its weights are not part of the program; before the first use the user is
shown the NVIDIA Open Model License and the DINOv3 License, and on acceptance the weights are downloaded
(``posetrak.detection.model_weights``). A user who declines, or whose download fails, gets RTMPose-133 and is told
why. Call :func:`prepare_pose_model` on the main thread before a detection job is started.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QMessageBox, QProgressDialog, QTabWidget, QTextBrowser,
    QVBoxLayout,
)

from posetrak.detection import model_weights
from posetrak.detection.pose_model_choice import FALLBACK_POSE_MODEL, is_soma

_log = logging.getLogger(__name__)


class LicenceDialog(QDialog):
    """Shows the text of each licence on its own tab; accepted only through the Accept button."""

    def __init__(self, texts, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Licences of the SOMA-77 pose model")
        self.resize(760, 600)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "The SOMA-77 pose model is NVIDIA's GEM-X 2D model. It is downloaded from Hugging Face and used under "
            "the two licences below, which you must accept to use it. Without it, RTMPose-133 is used."
        ))
        tabs = QTabWidget()
        for licence, text in texts:
            view = QTextBrowser()
            view.setPlainText(text)
            tabs.addTab(view, licence.name)
        layout.addWidget(tabs)
        layout.addWidget(QLabel(f"{model_weights.NVIDIA_NOTICE}.  {model_weights.DINOV3_NOTICE}."))
        buttons = QDialogButtonBox()
        accept = buttons.addButton("Accept both and download (about 3.4 GB)", QDialogButtonBox.ButtonRole.AcceptRole)
        accept.setDefault(False)
        buttons.addButton("Decline", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class _Download(QThread):
    progress = Signal(str)
    failed = Signal(str)

    def run(self) -> None:
        try:
            model_weights.install_gemx(self.progress.emit)
        except Exception as exc:  # noqa: BLE001 -- reported to the user, who then gets RTMPose
            _log.warning("SOMA-77 weights download failed", exc_info=True)
            self.failed.emit(str(exc))


def _download_with_progress(parent) -> str | None:
    """Run the download with a busy dialog; the error message, or None on success."""
    dialog = QProgressDialog("Downloading the SOMA-77 weights…", None, 0, 0, parent)
    dialog.setWindowTitle("SOMA-77 pose model")
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    worker = _Download(parent)
    error: list[str] = []
    worker.progress.connect(dialog.setLabelText)
    worker.failed.connect(error.append)
    worker.finished.connect(dialog.close)
    worker.start()
    dialog.exec()
    worker.wait()
    return error[0] if error else None


def prepare_pose_model(parent, requested: str) -> tuple[str, str | None]:
    """The pose model to run for *requested*, downloading SOMA-77 first when that needs the user's consent.

    Returns ``(model, reason)``; *reason* says why the result is not *requested* (None when it is) and has
    already been shown to the user.
    """
    if not is_soma(requested) or model_weights.soma77_model_path() is not None:
        return requested, None
    texts = model_weights.bundled_licences()
    if LicenceDialog(texts, parent).exec() != QDialog.DialogCode.Accepted:
        return _fallback(parent, requested, "the licences were not accepted")
    error = _download_with_progress(parent)
    if error is not None:
        return _fallback(parent, requested, f"the weights could not be downloaded ({error})")
    return requested, None


def _fallback(parent, requested: str, why: str) -> tuple[str, str]:
    reason = f"{requested}: {why}"
    QMessageBox.information(parent, "Pose model", f"Using {FALLBACK_POSE_MODEL} instead.\n\n{reason}")
    return FALLBACK_POSE_MODEL, reason
