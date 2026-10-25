# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""The licence dialog of the SOMA-77 pose model shows the licence texts bundled with posetrak."""
import socket

from PySide6.QtWidgets import QTabWidget, QTextBrowser

from app.pose.pose_model_gui import LicenceDialog
from posetrak.detection import model_weights


def test_dialog_shows_the_bundled_texts_without_network(qapp, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("the licence texts are bundled")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    dialog = LicenceDialog(model_weights.bundled_licences())
    tabs = dialog.findChild(QTabWidget)
    shown = [tabs.widget(i).toPlainText() for i in range(tabs.count())]
    assert tabs.count() == 2 and all(isinstance(tabs.widget(i), QTextBrowser) for i in range(2))
    assert shown == [text for _, text in model_weights.bundled_licences()]
    assert "Built with DINOv3" in shown[1]
