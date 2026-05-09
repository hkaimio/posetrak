"""posetrak-ui — Unified application entry point."""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
for _name in ("app.setup", "app.setup.led_sync", "app.pose", "app.ui"):
    logging.getLogger(_name).setLevel(logging.DEBUG)


def main() -> int:
    from pathlib import Path

    from PySide6.QtWidgets import QApplication

    from posetrak.db.db import DEFAULT_REGISTRY_PATH

    from app.ui.main_window import MainWindow, open_or_create_registry

    app = QApplication(sys.argv)
    app.setApplicationName("posetrak-ui")
    app.setOrganizationName("posetrak")

    registry_conn = open_or_create_registry(DEFAULT_REGISTRY_PATH)
    window = MainWindow(registry_conn)

    window.show()

    # CLI arg takes priority; otherwise reopen the last-used session.
    if len(sys.argv) > 1:
        window.open_session_file(Path(sys.argv[1]))
    else:
        window.auto_open_last_session()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
