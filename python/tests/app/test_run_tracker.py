# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for _describe_windows_exit_code (app.pose.run_tracker).

Regression coverage for a real crash: a tracker subprocess killed by
Windows before main() ran (missing DLL) reports its NTSTATUS as the exit
code, which previously overflowed a Qt `int` signal argument. Covers only
this pure helper -- the Qt signal/subprocess plumbing around it follows
the project's usual manual-validation convention.
"""
from __future__ import annotations

from app.pose.run_tracker import _describe_windows_exit_code


def test_describe_windows_exit_code_none_for_ordinary_exit_codes():
    assert _describe_windows_exit_code(0) is None
    assert _describe_windows_exit_code(1) is None
    assert _describe_windows_exit_code(255) is None


def test_describe_windows_exit_code_names_dll_not_found():
    msg = _describe_windows_exit_code(0xC0000135)
    assert msg is not None
    assert "DLL" in msg
    assert "CONTRIBUTING.md" in msg


def test_describe_windows_exit_code_generic_for_other_ntstatus_errors():
    msg = _describe_windows_exit_code(0xC0000005)  # STATUS_ACCESS_VIOLATION
    assert msg is not None
    assert "0xC0000005" in msg


def test_describe_windows_exit_code_boundary_just_below_ntstatus_range():
    assert _describe_windows_exit_code(0x7FFF_FFFF) is None


def test_describe_windows_exit_code_boundary_at_ntstatus_range():
    assert _describe_windows_exit_code(0x8000_0000) is not None
