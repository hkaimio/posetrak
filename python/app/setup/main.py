"""posetrak-setup — Session setup wizard entry point."""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication, QLabel, QMainWindow

    app = QApplication(sys.argv)

    window = QMainWindow()
    window.setWindowTitle("posetrak setup")
    window.resize(800, 600)
    placeholder = QLabel("posetrak setup wizard — coming soon", window)
    placeholder.setAlignment(placeholder.alignment())
    window.setCentralWidget(placeholder)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
