"""posetrak-ui — Unified application entry point."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_PATH = _LOG_DIR / "posetrak-ui.log"

_formatter = logging.Formatter(
    "%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S",
)
_stream_handler = logging.StreamHandler(sys.stderr)
_stream_handler.setFormatter(_formatter)
# Rotate at 20MB so a long session doesn't grow the file unboundedly; a few
# backups is enough to grep back through recent runs without hunting for a
# per-run filename.
_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_PATH, maxBytes=20_000_000, backupCount=5, encoding="utf-8",
)
_file_handler.setFormatter(_formatter)

logging.basicConfig(level=logging.WARNING, handlers=[_stream_handler, _file_handler])
for _name in ("app.setup", "app.setup.led_sync", "app.pose", "app.ui", "app.setup.db_context", "app.setup.page_sync"):
    logging.getLogger(_name).setLevel(logging.DEBUG)

print(f"Logging to {_LOG_PATH}", file=sys.stderr)


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
