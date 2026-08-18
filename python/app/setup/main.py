# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""posetrak-setup — Session setup wizard entry point."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_PATH = _LOG_DIR / "posetrak-setup.log"

_formatter = logging.Formatter(
    "%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S",
)
_stream_handler = logging.StreamHandler(sys.stderr)
_stream_handler.setFormatter(_formatter)
_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_PATH, maxBytes=20_000_000, backupCount=5, encoding="utf-8",
)
_file_handler.setFormatter(_formatter)

logging.basicConfig(level=logging.WARNING, handlers=[_stream_handler, _file_handler])
# Verbose debug output for the setup wizard and LED sync algorithm.
for _name in ("app.setup", "app.setup.led_sync"):
    logging.getLogger(_name).setLevel(logging.DEBUG)

print(f"Logging to {_LOG_PATH}", file=sys.stderr)


def main() -> int:
    from PySide6.QtWidgets import QApplication, QWizard

    from app.setup.camera_registry import CameraRegistryWidget
    from app.setup.page_extrinsics import ExtrinsicsPage
    from app.setup.page_session import SessionPage
    from app.setup.page_shots import ShotsPage
    from app.setup.page_skeleton import SkeletonPage
    from app.setup.page_sync import SyncPage

    app = QApplication(sys.argv)

    wizard = QWizard()
    wizard.setWindowTitle("posetrak Setup Wizard")
    wizard.resize(1000, 700)

    # Placeholders for page 1 to write to; downstream pages read these.
    wizard.session_conn   = None
    wizard.session_id     = None
    wizard.db_context     = None
    wizard.registry_conn  = None  # set by SessionPage when a registry DB is opened
    wizard.new_shot_ids   = []    # set by ShotsPage so SyncPage can pre-select them

    # "Manage Cameras…" custom button — opens CameraRegistryWidget backed by
    # whichever DB connection is currently active (registry preferred, else session).
    wizard.setOption(QWizard.WizardOption.HaveCustomButton1, True)
    wizard.setButtonText(QWizard.WizardButton.CustomButton1, "Manage Cameras…")

    shots_page = ShotsPage()

    def _open_camera_registry() -> None:
        conn = wizard.registry_conn or wizard.session_conn
        if conn is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                wizard,
                "No database open",
                "Open or create a session database first.",
            )
            return
        session_conn = wizard.session_conn if conn is not wizard.session_conn else None
        dlg = CameraRegistryWidget(conn, session_conn=session_conn, parent=wizard)
        dlg.cameras_changed.connect(shots_page.refresh_camera_combos)
        dlg.exec()

    wizard.customButtonClicked.connect(lambda btn: _open_camera_registry())

    wizard.addPage(SessionPage())
    wizard.addPage(shots_page)
    wizard.addPage(SyncPage())
    wizard.addPage(ExtrinsicsPage())
    wizard.addPage(SkeletonPage())

    wizard.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
