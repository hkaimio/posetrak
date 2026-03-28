"""posetrak-setup — Session setup wizard entry point."""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication, QWizard

    from app.setup.page_session import SessionPage
    from app.setup.page_shots import ShotsPage

    app = QApplication(sys.argv)

    wizard = QWizard()
    wizard.setWindowTitle("posetrak Setup Wizard")
    wizard.resize(800, 600)

    # Placeholders for page 1 to write to; downstream pages read these.
    wizard.session_conn = None
    wizard.session_id   = None
    wizard.db_context   = None

    wizard.addPage(SessionPage())
    wizard.addPage(ShotsPage())

    wizard.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
