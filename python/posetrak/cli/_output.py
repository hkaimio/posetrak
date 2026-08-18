# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Output helpers for the posetrak CLI.

Two modes:
- Human-readable aligned tables (default).
- JSONL: one JSON object per line (--json mode).

UUID display in tables uses first 8 chars + ellipsis for readability.
JSONL uses full UUIDs.
"""

from __future__ import annotations

import json
import sys


def abbrev_id(uuid_str: str | None) -> str:
    """Return the first 8 characters of a UUID followed by an ellipsis."""
    if not uuid_str:
        return ""
    return uuid_str[:8] + "…"


def print_table(rows: list[dict], columns: list[str], *, json_mode: bool) -> None:
    """Print a list of dicts as an aligned table or as JSONL.

    Parameters
    ----------
    rows:
        List of row dicts.  Each dict may contain keys beyond *columns*.
    columns:
        Ordered list of column names to include in the output.
    json_mode:
        When True, emit one JSON object per line (full UUIDs).
        When False, print an aligned table with abbreviated UUIDs.
    """
    if not rows:
        return

    if json_mode:
        for row in rows:
            # Convert sqlite3.Row / dict-like objects to plain dicts.
            obj = {k: row[k] for k in columns if k in row or _has_key(row, k)}
            sys.stdout.write(json.dumps(obj) + "\n")
        return

    # Build display rows with abbreviated UUIDs.
    display_rows = []
    for row in rows:
        display_row = {}
        for col in columns:
            val = _get(row, col)
            if isinstance(val, str) and _looks_like_uuid(val):
                display_row[col] = abbrev_id(val)
            else:
                display_row[col] = "" if val is None else str(val)
        display_rows.append(display_row)

    # Compute column widths.
    widths = {col: len(col) for col in columns}
    for row in display_rows:
        for col in columns:
            widths[col] = max(widths[col], len(row[col]))

    # Header.
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    separator = "  ".join("-" * widths[col] for col in columns)
    sys.stdout.write(header + "\n")
    sys.stdout.write(separator + "\n")

    for row in display_rows:
        line = "  ".join(row[col].ljust(widths[col]) for col in columns)
        sys.stdout.write(line.rstrip() + "\n")


def print_record(record: dict, *, json_mode: bool = False) -> None:
    """Print a single record as a key-value table or a JSON object.

    Parameters
    ----------
    record:
        The record to display.
    json_mode:
        When True, emit a single-line JSON object.
        When False, print ``key: value`` pairs aligned on the colon.
    """
    if json_mode:
        obj = {k: v for k, v in record.items()}
        sys.stdout.write(json.dumps(obj) + "\n")
        return

    if not record:
        return

    key_width = max(len(k) for k in record)
    for key, val in record.items():
        sys.stdout.write(f"{key:<{key_width}} : {val}\n")


def print_jsonl(rows: list[dict]) -> None:
    """Emit each row as a JSON line to stdout."""
    for row in rows:
        sys.stdout.write(json.dumps(row) + "\n")


def fail(message: str, exit_code: int = 1) -> None:
    """Print *message* to stderr and exit."""
    sys.stderr.write(f"Error: {message}\n")
    raise SystemExit(exit_code)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_UUID_LEN = 36


def _looks_like_uuid(val: str) -> bool:
    """Return True if *val* looks like a UUID (36 chars with hyphens)."""
    return len(val) == _UUID_LEN and val.count("-") == 4


def _has_key(row, key: str) -> bool:
    try:
        row[key]  # noqa: B018
        return True
    except (KeyError, IndexError):
        return False


def _get(row, key: str):
    try:
        return row[key]
    except (KeyError, IndexError):
        return None
