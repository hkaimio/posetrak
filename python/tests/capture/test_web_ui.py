# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Runs the web UI's client-state tests with Node, when Node is installed."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

WEB_TESTS = Path(__file__).parent / "web"
WEB_DIR = Path(__file__).parents[2] / "pipeline" / "capture" / "web"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_client_store() -> None:
    result = subprocess.run(
        ["node", "--test", str(WEB_TESTS)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_ui_files_are_present() -> None:
    for name in ("index.html", "app.js", "store.js", "style.css", "vendor/htm-preact-standalone.js"):
        assert (WEB_DIR / name).is_file(), name
