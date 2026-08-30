# Posetrak Python coding guidelines

## Project structure

The Python codebase lives under `python/` and is installed as a single package
from the root `pyproject.toml`:

- `python/posetrak/db/` — SQLite DB layer (`posetrak-db` CLI + import helpers)
- `python/app/setup/` — Session setup wizard (PySide6, `posetrak-setup` CLI)
- `python/app/pose/` — Pose extraction app (PySide6 + YOLO/RTMPose, `posetrak-pose` CLI)
- `python/app/analysis/` — Marimo analysis scripts
- `python/pipeline/` — Standalone calibration and pose-extraction tools
- `python/tests/` — pytest suite

The actual motion capture tracker is **C++** (see `src/`, `include/`, `cli/`).
Python does not implement any tracking logic.

## Running commands

Always use `uv run` from the **repository root** (where `pyproject.toml` lives):

```bash
# Run tests
uv run pytest

# Run a specific test file
uv run pytest python/tests/db/test_db.py -v

# Run the setup wizard
uv run posetrak-setup

# Run the DB CLI
uv run posetrak-db --help
```

## Adding dependencies

```bash
uv add package-name          # runtime dependency
uv add --dev package-name    # dev dependency
uv sync                      # install all groups
```

## Coding standards

### Python version and style

- Python **3.13+**
- **PEP 8** compliance; maximum line length **100 characters**
- **Type annotations** on all functions and methods
- **Docstrings** on all public APIs (Google style)
- `from __future__ import annotations` at the top of every module

### Type annotations

```python
from __future__ import annotations

import sqlite3
from pathlib import Path


def open_session(path: Path) -> sqlite3.Connection:
    """Open an existing session database, applying any pending migrations.

    Args:
        path: Path to the SQLite session file.

    Returns:
        Open connection with row_factory set to sqlite3.Row.

    Raises:
        ValueError: If the file is not a valid session database.
    """
    ...
```

### Error handling

- Validate inputs at public API boundaries; trust internal code.
- Raise specific exception types with descriptive messages.

```python
def get_camera_mode(conn: sqlite3.Connection, mode_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM camera_modes WHERE id = ?", (mode_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Camera mode '{mode_id}' not found in database")
    return row
```

## SQLite / DB layer patterns

### Always use parameterised queries

```python
# Good
conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,))

# Bad — SQL injection risk
conn.execute(f"SELECT * FROM shots WHERE id = '{shot_id}'")
```

### Use `sqlite3.Row` for named column access

```python
conn.row_factory = sqlite3.Row
row = conn.execute("SELECT id, label FROM camera_instances").fetchone()
print(row["label"])   # not row[1]
```

### Migrations

Each migration is a standalone `.sql` file in `python/posetrak/db/sql/migrations/`
(package data, loaded via `importlib.resources` — see `_DB_DIR` in `db.py`).  The calling
Python function in `db.py` sets `PRAGMA user_version` after running the SQL —
never put `PRAGMA user_version` inside the SQL file so the same file can be
reused for multiple DB types.

```python
def _migrate_registry_v4_to_v5(conn: sqlite3.Connection) -> None:
    sql = (_DB_DIR / "migrations" / "009_camera_modes_default_calib.sql").read_text(
        encoding="utf-8"
    )
    conn.executescript(sql)
    conn.execute("PRAGMA user_version = 5")
    conn.commit()
```

## PySide6 / GUI patterns

### Widget initialisation order

Always follow: create widgets → connect signals → lay out → set initial state
(via a dedicated `_update_*` or `_load` method called at the end of `__init__`).

```python
class ModeDialog(QDialog):
    def __init__(
        self,
        conn: sqlite3.Connection,
        mode_id: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._conn = conn
        self._mode_id = mode_id

        # 1. Create widgets
        self._width = QLineEdit()
        self._height = QLineEdit()
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )

        # 2. Connect signals
        self._buttons.accepted.connect(self._save)
        self._buttons.rejected.connect(self.reject)

        # 3. Layout
        form = QFormLayout(self)
        form.addRow("Width:", self._width)
        form.addRow("Height:", self._height)
        form.addWidget(self._buttons)

        # 4. Populate if editing
        if mode_id:
            self._load()
```

### Override naming

Qt override methods use PascalCase (`validatePage`, `cleanupPage`,
`initializePage`).  Suppress ruff's `N802` on these:

```python
def validatePage(self) -> bool:  # noqa: N802
    ...
```

### Background work

Use `QThread` / signals for any work that would block the event loop (video
probing, DB migrations, LED sync).  Never call `time.sleep()` on the main
thread.  See `job_runner.py` for the project's established pattern.

### Testing GUI code

Use `pytest-qt` and `QApplication` fixtures.  Test widget state, signal
emissions, and DB side-effects — not pixel-level rendering.

```python
def test_session_page_rejects_missing_path(qtbot):
    page = SessionPage()
    qtbot.addWidget(page)
    assert not page.validatePage()
```

## Testing

### Running tests

```bash
uv run pytest                               # all tests
uv run pytest python/tests/db/ -v          # DB layer only
uv run pytest python/tests/app/ -v         # app tests
uv run pytest -k test_open_session         # by name
uv run pytest --tb=short -q                # compact output
```

### Writing tests

```python
import pytest
from pathlib import Path
from posetrak.db.db import SESSION_SCHEMA_VERSION, create_session, open_session


class TestOpenSession:
    def test_fresh_db_has_correct_version(self, tmp_path: Path) -> None:
        conn = create_session(tmp_path / "test.db")
        ver = conn.execute("PRAGMA user_version").fetchone()[0]
        assert ver == SESSION_SCHEMA_VERSION
        conn.close()

    def test_missing_file_raises(self) -> None:
        with pytest.raises(Exception):
            open_session(Path("/nonexistent/test.db"))
```

### Test naming

- Files: `test_*.py`
- Classes: `Test*`
- Methods: `test_*` with descriptive names (`test_duplicate_serial_rejected`,
  not `test2`)

## Development workflow

### Branch and commit

```bash
git checkout -b feature/my-feature
# ... implement, run tests ...
uv run pytest
git add python/app/setup/my_widget.py python/tests/app/test_my_widget.py
git commit
```

Commit message format follows the project convention in `CLAUDE.md`:

```
comp: short description

Longer explanation if needed. Reference GitHub issues as #123.
```

`comp` is the affected component, e.g. `python/app/setup`, `python/posetrak/db`,
`docs`, `tests`.  Do not use conventional-commit prefixes (`feat:`, `fix:`).
