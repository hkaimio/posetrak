"""posetrak-setup — Session setup wizard entry point."""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
# Verbose debug output for the setup wizard and LED sync algorithm.
for _name in ("app.setup", "app.setup.led_sync"):
    logging.getLogger(_name).setLevel(logging.DEBUG)


def main() -> int:
    from PySide6.QtWidgets import QApplication, QWizard

    from app.setup.page_session import SessionPage
    from app.setup.page_shots import ShotsPage
    from app.setup.page_sync import SyncPage

    app = QApplication(sys.argv)

    wizard = QWizard()
    wizard.setWindowTitle("posetrak Setup Wizard")
    wizard.resize(1000, 700)

    # Placeholders for page 1 to write to; downstream pages read these.
    wizard.session_conn = None
    wizard.session_id   = None
    wizard.db_context   = None

    wizard.addPage(SessionPage())
    wizard.addPage(ShotsPage())
    wizard.addPage(SyncPage())

    wizard.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
