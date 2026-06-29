# Posetrak CLI — Implementation Status

See [cli-design.md](cli-design.md) for the full requirements and design.

## Phase summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Consolidate existing CLI into Click | ✅ Done |
| 2 | `detect run/list/show` commands | ✅ Done |
| 3 | `track list/show` and `track export bvh/gltf/usd` | ✅ Done |
| 4 | Extract `run_tracker()`; add `track run`; MCP workflow tools | ✅ Done |
| 5 | `trial export/import` | ✅ Done |
| 6 | `detect finalise` (person assignment) | ⬜ Not started |
| 7 | Retire old entry points (`posetrak-pose`, `posetrak-setup`) | ⬜ Not started |

## Implemented commands

### Phase 1 — `python/posetrak/cli/registry.py`, `session.py`, `skeleton.py`, `config.py`

```
registry init/info
camera-model add/list
camera-mode add/list
camera add/list/show/import-session
calib import/import-h5/list
skeleton import/list/show/export/scale
config create/list/show/edit
session create/list/show/import-yaml
capture create/list/show/add-video
extrinsics import/list
sync import/list
```

C++ binary rename (`posetrak` → `posetrak-tracker` in `cli/meson.build`) is part of Phase 1 scope
but has not been done yet — it is a prerequisite for Phase 4.

### Phase 2 — `python/posetrak/cli/detect.py`

```
detect run    --capture ID --sync ID --start S --end S [--detector] [--pose-model] [--conf]
detect list   [--capture ID]
detect show   ID
```

Detection backends moved to `posetrak/detection/`; `app/pose/backends_*.py` are now thin shims.

### Phase 3 — `python/posetrak/cli/track.py`

```
track list              [--sequence ID]
track show              ID
track export bvh        ID OUTPUT [--smoothed] [--fps N] [--units m|cm] [--coord yup|zup]
track export gltf       ID OUTPUT [--smoothed] [--fps N] [--units m|cm] [--coord yup|zup]
track export usd        ID OUTPUT [--smoothed] [--fps N] [--units m|cm] [--coord yup|zup]
```

## Pending work

### Phase 4 — tracker subprocess extraction ✅

All items complete:

- `posetrak/tracker/runner.py` — `run_tracker()` / `TrackerResult`; `RunTrackerWidget`
  refactored to use `_TrackerThread(QThread)` calling the extracted function
- `track run` — inline config overrides on the command line; inserts a `cli-run` config row;
  prints `tracking_run_id` on success
- MCP workflow read tools: `list_captures`, `list_detection_runs`, `get_capture_info`
- MCP workflow write tools (`--mcp-allow-write` opt-in): `run_detection`, `run_tracking`

Also added outside the original Phase 4 scope (prompted by session DB portability need):

- `camera list`, `camera-model list`, `camera-mode list` are session-aware: pass `--session`
  to read from a session DB instead of the registry
- `camera import-session` — copies `camera_models`, `camera_modes`, `camera_instances`, and
  `intrinsics_calibrations` from a session DB into the registry; INSERT OR IGNORE so re-runs
  and partial states (model present, modes/calibrations missing) are handled correctly;
  `--dry-run` flag available

### Phase 5 — trial portability ✅

All items complete:

- `python/posetrak/db/trial_export.py` — library layer:
  - `ExportScope` enum: `capture-only | trial-only | detection-only | full`
  - `AnchorSpec` dataclass: `trial_ids`, `capture_ids`, `detection_ids`, `tracking_run_ids` lists
  - `export_trials(src, dst, anchor, *, scope, include_cache, skip_tables, dry_run)` — two-phase
    dependency resolution (walk UP for ancestors, walk DOWN for descendants) then ordered INSERT OR IGNORE copy
  - `import_trials(src, dst_session, dst_registry, anchor, *, skip_tables, dry_run)` — full-scope
    import; optionally mirrors registry tables to a separate registry DB
  - `open_source_readonly(path)` — URI mode open (`?mode=ro`) so corrupted or foreign DBs are
    safe to read without triggering migrations
  - Per-table error resilience: each table copy catches `sqlite3.DatabaseError` and returns a
    `TableResult` with an error string rather than aborting the whole run
- `python/posetrak/cli/trial.py` — CLI commands:
  - `trial list` — shows trials with capture label, detection count, tracking run count;
    falls back to capture list when no trials exist
  - `trial export OUTPUT.db [--trial|--capture|--detection|--tracking-run ID]... [--scope ...]
    [--include-cache] [--skip-tables T,...] [--dry-run]`
  - `trial import SRC.db [--trial|--capture|--detection ID]... [--sync-registry] [--skip-tables T,...]
    [--dry-run]`
- 26 new tests in `python/tests/cli/test_trial.py` covering library and CLI (total CLI tests: 101)

### Phase 6 — `detect finalise`

Design is deferred. Single-person case (`--assign "0:Subject"`) can ship first; multi-person
requires an interactive TUI or two-step list-then-assign flow.

### Phase 7 — retire old entry points

Remove from `pyproject.toml`:
```toml
posetrak-pose  = "app.pose.cli:main"
posetrak-setup = "app.setup.main:main"
```
Delete: `app/pose/cli.py`, `app/setup/main.py`, `app/setup/page_session.py`

## Other notes

- `pose list/import` commands exist in `python/posetrak/cli/pose.py` as a carry-over from
  the old CLI. These are not in the design doc and should be reviewed before Phase 7 retirement.
- Tests live in `python/tests/cli/`; all 101 tests pass as of Phase 5 completion.
