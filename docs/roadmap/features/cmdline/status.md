# Posetrak CLI — Implementation Status

See [cli-design.md](cli-design.md) for the full requirements and design.

## Phase summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Consolidate existing CLI into Click | ✅ Done |
| 2 | `detect run/list/show` commands | ✅ Done |
| 3 | `track list/show` and `track export bvh/gltf/usd` | ✅ Done |
| 4 | Extract `run_tracker()`; add `track run`; MCP detection tools | ⬜ Not started |
| 5 | `trial export/import` | ⬜ Not started |
| 6 | `detect finalise` (person assignment) | ⬜ Not started |
| 7 | Retire old entry points (`posetrak-pose`, `posetrak-setup`) | ⬜ Not started |

## Implemented commands

### Phase 1 — `python/posetrak/cli/registry.py`, `session.py`, `skeleton.py`, `config.py`

```
registry init/info
camera-model add/list
camera-mode add/list
camera add/list/show
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

### Phase 4 — tracker subprocess extraction

- Rename C++ binary to `posetrak-tracker` in `cli/meson.build`
- Extract subprocess logic from `app/pose/run_tracker.py` (`RunTrackerWidget`) into
  `posetrak/tracker/runner.py`:
  ```python
  def run_tracker(session_path, sequence_id, skeleton_id, config_id, output_dir, *,
                  binary_path=None, person_id=0, on_progress=None) -> int
  ```
- Refactor `RunTrackerWidget` to call `run_tracker()` with a Qt signal as `on_progress`
- Add `posetrak track run --sequence ID --skeleton ID --config ID [--output-dir PATH] [--person-id N] [--fps N]`
- Add MCP detection read tools (`list_captures`, `list_detection_runs`, `list_tracking_runs`,
  `get_capture_info`) and `run_detection` write tool

### Phase 5 — trial portability

- `posetrak trial export ID OUTPUT_PATH` — write a new session DB with only the specified trial
  and its dependencies (cameras, calibrations, skeleton, config)
- `posetrak trial import PATH [ID ...]` — merge trials into `--registry` and `--session`
- Add MCP tracker and export tools

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
- Tests live in `python/tests/cli/`; all 60 tests pass as of Phase 3 completion.
