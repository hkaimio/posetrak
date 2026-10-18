# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""stage_timing: with the environment variable unset nothing is wrapped; with it set, times are exclusive."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[2]


def _run(code: str, env_value: str | None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "POSETRAK_PROFILE_DETECTION"}
    if env_value is not None:
        env["POSETRAK_PROFILE_DETECTION"] = env_value
    env["PYTHONPATH"] = str(PYTHON_DIR)
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, capture_output=True, text=True, check=True)


def test_off_path_leaves_functions_and_iterators_untouched():
    out = _run(
        """
        from posetrak.detection import stage_timing as st, backends_soma77 as soma

        def f(x):
            return x + 1

        it = iter(range(3))
        assert not st.ENABLED
        assert st.timed("a")(f) is f
        assert st.timed_iter("a", it) is it
        assert st.stage("a") is st.stage("b") and st.section("c") is st.stage("a")
        assert not hasattr(soma.preprocess, "__wrapped__") and not hasattr(soma.decode, "__wrapped__")
        with st.stage("a"):
            pass
        assert set(st.snapshot()) == {"_wall_s"}
        print("ok")
        """,
        None,
    )
    assert out.stdout.strip() == "ok" and out.stderr == ""


def test_on_path_times_are_exclusive_and_sectioned(tmp_path):
    path = tmp_path / "t.json"
    _run(
        """
        import time
        from posetrak.detection import stage_timing as st

        with st.stage("outer"):
            time.sleep(0.05)
            with st.stage("inner"):
                time.sleep(0.05)
        with st.section("hand_pass"):
            for _ in st.timed_iter("decode", range(3)):
                pass
        """,
        str(path),
    )
    data = json.loads(path.read_text())
    assert data["inner"]["calls"] == 1 and data["outer"]["calls"] == 1
    assert 0.04 < data["outer"]["seconds"] < 0.09 and 0.04 < data["inner"]["seconds"] < 0.09
    assert data["hand_pass/decode"]["calls"] == 4  # three items and the exhausted step
