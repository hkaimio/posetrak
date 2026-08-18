# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure argument-building helper behind run_multi_person_tracker().

The subprocess/progress-callback plumbing itself follows the project's usual
manual-validation convention (see test_run_tracker.py's docstring); this
covers only the part that's meaningful to unit-test: the CLI argument shape.
"""
from __future__ import annotations

from pathlib import Path

from posetrak.tracker.runner import PersonRunSpec, _build_multi_person_args, _tracker_binary_name


def test_build_multi_person_args_repeats_person_flag_with_four_values_each():
    persons = [
        PersonRunSpec(sequence_id="seq1", skeleton_id="skelA", config_id="cfg1", person_id=0),
        PersonRunSpec(sequence_id="seq1", skeleton_id="skelB", config_id="cfg1", person_id=1),
    ]
    args = _build_multi_person_args(
        Path("posetrak-tracker"),
        Path("session.db"),
        persons,
        Path("out"),
        start_time=None,
        end_time=None,
        smooth=False,
    )

    assert args == [
        "posetrak-tracker", "track",
        "--session-db", "session.db",
        "--output-dir", "out",
        "--person", "seq1", "skelA", "cfg1", "0",
        "--person", "seq1", "skelB", "cfg1", "1",
    ]


def test_build_multi_person_args_appends_optional_flags():
    persons = [PersonRunSpec(sequence_id="s", skeleton_id="k", config_id="c", person_id=0)]
    args = _build_multi_person_args(
        Path("posetrak-tracker"),
        Path("session.db"),
        persons,
        Path("out"),
        start_time=1.5,
        end_time=9.0,
        smooth=True,
    )

    assert "--start-time" in args
    assert args[args.index("--start-time") + 1] == "1.5"
    assert "--end-time" in args
    assert args[args.index("--end-time") + 1] == "9.0"
    assert args[-1] == "--smooth"


def test_tracker_binary_name_adds_exe_suffix_on_windows():
    assert _tracker_binary_name("win32") == "posetrak-tracker.exe"


def test_tracker_binary_name_no_suffix_elsewhere():
    assert _tracker_binary_name("linux") == "posetrak-tracker"
    assert _tracker_binary_name("darwin") == "posetrak-tracker"
