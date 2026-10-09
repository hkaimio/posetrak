# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Column documentation in the SQL schema files.

Guideline (also in CLAUDE.md): every column of a table you create or modify
gets a comment saying what it holds -- unit or coordinate frame, what NULL
means, and which table/column an id refers to. Put it on the column's own
line (``-- ...``) or on the comment line directly above it.

Older tables are not fully documented yet, so this is a ratchet: add a table
to DOCUMENTED_TABLES in the same change that documents it, and never remove
one. A new or modified table that is not listed here is a review question.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SQL_DIR = Path(__file__).resolve().parents[2] / "posetrak" / "db" / "sql"

# (schema file, table). Tables that are fully column-documented.
DOCUMENTED_TABLES = [
    ("session_schema.sql", "capture_subjects"),
    ("session_schema.sql", "dot_assignment_edits"),
    ("session_schema.sql", "dot_assignment_proposals"),
    ("session_schema.sql", "seg_quality_run_subjects"),
    ("session_schema.sql", "seg_quality_runs"),
    ("session_schema.sql", "sequence_persons"),
    ("session_schema.sql", "tracking_run_persons"),
    ("session_schema.sql", "tracking_runs"),
    ("session_schema.sql", "detection_track_assignments"),
    ("session_schema.sql", "tracking_assignment_trace"),
    ("registry_schema.sql", "tracker_configs"),
]

_COLUMN = re.compile(r"^(\w+)\s+(TEXT|INTEGER|REAL|BLOB|NUMERIC)\b", re.IGNORECASE)
_NOT_COLUMNS = {"PRIMARY", "UNIQUE", "FOREIGN", "CHECK", "CONSTRAINT"}


def _table_columns(sql_file: str, table: str) -> list[tuple[str, bool]]:
    """(column name, documented) for each column of *table* in *sql_file*."""
    lines = (SQL_DIR / sql_file).read_text(encoding="utf-8").split("\n")
    header = re.compile(rf"^\s*CREATE TABLE IF NOT EXISTS {table}\s*\(")
    start = next((i for i, line in enumerate(lines) if header.match(line)), None)
    if start is None:
        return []
    columns: list[tuple[str, bool]] = []
    comment_above = False
    for line in lines[start + 1:]:
        stripped = line.strip()
        if re.match(r"^\)\s*;", stripped):
            break
        if stripped.startswith("--"):
            comment_above = True
            continue
        match = _COLUMN.match(stripped)
        if match and match.group(1).upper() not in _NOT_COLUMNS:
            columns.append((match.group(1), "--" in line or comment_above))
        comment_above = False
    return columns


@pytest.mark.parametrize("sql_file, table", DOCUMENTED_TABLES)
def test_every_column_of_a_documented_table_has_a_comment(sql_file: str, table: str) -> None:
    columns = _table_columns(sql_file, table)
    assert columns, f"{table} not found in {sql_file}"
    missing = [name for name, documented in columns if not documented]
    assert not missing, (
        f"{table} in {sql_file} has undocumented columns: {missing}. "
        "Add a '-- what it holds' comment to each (units, NULL meaning, referenced table)."
    )
