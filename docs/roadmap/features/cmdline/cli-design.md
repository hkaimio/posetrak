# Posetrak CLI — Requirements & Design Proposal

## Goals

Expose all meaningful Posetrak functionality via a single `posetrak` command that mirrors the UI
workflow. The CLI is a first-class interface, not a thin wrapper — it must be fully scriptable for
batch pipelines and automated testing. The existing `posetrak-db` entry point covers a large part of
this already and will be absorbed rather than replaced.

---

## Command hierarchy

The unified entry point is `posetrak` with subcommand groups matching the domain model:

```
posetrak [--db PATH] [--session PATH] [--json] <group> <command> [args...]

Global options:
  --db PATH       Registry database (default: ~/.posetrak/registry.db)
  --session PATH  Session database; most commands require one of --db or --session

Output options:
  --json          Emit JSONL instead of human-readable tables (all list/show commands)
  -v / --verbose  Increase log verbosity
```

### Registry & camera setup

```
posetrak registry init [PATH]
posetrak registry info

posetrak camera-model add --manufacturer STR --model STR [--sensor-width MM]
posetrak camera-model list

posetrak camera-mode add --model ID --width PX --height PX --fps N [--codec STR]
posetrak camera-mode list [--model ID]

posetrak camera add --mode ID --label STR [--serial STR]
posetrak camera list
posetrak camera show ID

posetrak calib import PATH --camera-mode ID  # TOML (Pose2Sim format)
posetrak calib import-h5 PATH --camera-mode ID  # HDF5 (legacy)
posetrak calib list [--camera ID]
```

### Skeleton management

```
posetrak skeleton import PATH [--name STR]
posetrak skeleton list
posetrak skeleton show ID
posetrak skeleton export ID PATH
posetrak skeleton scale ID --height M [--arm-span M] ...
  # Runs scale_skeleton_yaml() and prints scaling_summary(); writes scaled YAML to PATH or stdout
```

### Tracker config

```
posetrak config create PATH [--name STR]  # from TOML file
posetrak config list
posetrak config show ID
posetrak config edit ID [--alpha F] [--measurement-noise-std F] ...
```

### Session & capture management

```
posetrak session create [--label STR] [--location STR]
posetrak session list
posetrak session show ID
posetrak session import-yaml PATH  # bulk import from YAML description file

posetrak capture create --session ID [--label STR]
posetrak capture list [--session ID]
posetrak capture show ID

posetrak capture add-video ID VIDEO_PATH --camera ID [--first-frame N] [--last-frame N]
posetrak capture add-camera ID --camera ID

posetrak extrinsics import PATH --capture ID [--method pose2sim]
posetrak extrinsics list [--capture ID]

posetrak sync import PATH --capture ID  # JSON sync description
posetrak sync list [--capture ID]
```

### Pose detection

```
posetrak detect run --capture ID --sync ID --start S --end S
                    [--detector yolo11x] [--pose-model rtmpose-l-133kp]
                    [--conf 0.3]
  # Wraps DetectionPipeline; prints progress to stderr, run-id to stdout on completion

posetrak detect list [--capture ID]
posetrak detect show ID

posetrak detect finalise ID [--assignment "YOLO_TRACK_ID:PERSON_NAME" ...]
  # Wraps finalise_to_db(); converts detection run to pose observation sequences
```

### Tracking

```
posetrak track run --sequence ID --skeleton ID --config ID [--output-dir PATH]
                   [--person-id N] [--fps N]
  # Invokes the C++ tracker binary as a subprocess; streams progress to stderr

posetrak track list [--sequence ID]
posetrak track show ID
```

### Export

```
posetrak export bvh ID OUTPUT_PATH [--smoothed]
posetrak export gltf ID OUTPUT_PATH [--smoothed]
posetrak export usd ID OUTPUT_PATH [--smoothed]
  # ID is a tracking-run ID; wraps export_bvh(), export_gltf(), export_usd()
```

### Data portability

```
posetrak db export-trial ID OUTPUT_PATH
  # Creates a self-contained session DB with the trial and all dependencies
  # (cameras, calibrations, skeleton, tracking run, pose sequences)

posetrak db import-trial PATH
  # Merges a trial DB into the current registry/session
```

---

## Reuse of existing code

Most of the required business logic exists today in well-tested, GUI-free modules. The CLI is
primarily an argument-parsing layer over these.

### Absorb `posetrak/db/cli.py` wholesale (2 400 lines)

The existing `posetrak-db` command already implements the registry, camera, skeleton, config,
session, capture, extrinsics, sync, pose, and tracking-run subcommands in full. The conversion plan:

- Keep all existing library call sites unchanged.
- Lift the argparse structure into Click command groups to match the new hierarchy.
- Merge the `--db` / `--session` global option handling.
- The current `resolve_id_prefix()` helper and interactive prompts can be reused verbatim.

The only migration cost is replacing `argparse` with Click (preferred for the new CLI) or keeping
argparse and restructuring the entry point. Click is recommended for consistency with `app/pose/cli.py`
and better subcommand discoverability.

### Absorb `app/pose/cli.py` (137 lines)

The `run` and `list-runs` commands call `DetectionPipeline`, `YOLOv11Detector`,
`RTMPoseEstimator`, and `list_detection_runs()` directly — no Qt. These map to
`posetrak detect run` and `posetrak detect list` with minimal changes.

### Directly callable library functions

| CLI command | Library function | Module |
|---|---|---|
| `skeleton import/export/scale` | `import_skeleton()`, `export_skeleton()`, `scale_skeleton_yaml()`, `scaling_summary()` | `db/manage_skeleton.py`, `db/scale_skeleton.py` |
| `config create/edit` | `create_config_from_toml()`, `edit_config()` | `db/manage_config.py` |
| `calib import` | `import_calib_toml()`, `import_calib_h5()` | `db/import_calib_toml.py`, `db/import_calib_h5.py` |
| `extrinsics import` | `import_extrinsics()` | `db/import_extrinsics.py` |
| `sync import` | `import_sync_json()` | `db/import_sync_json.py` |
| `session import-yaml` | `import_session_yaml()` | `db/import_session_yaml.py` |
| `detect finalise` | `finalise_to_db()` | `app/pose/finalise.py` |
| `detect run` | `DetectionPipeline.run()` | `app/pose/detection_pipeline.py` |
| `export bvh/gltf/usd` | `export_bvh()`, `export_gltf()`, `export_usd()` | `export/bvh.py`, `export/gltf.py`, `export/usd.py` |

---

## Required refactoring (minimal)

Only one meaningful piece of business logic is currently locked inside a Qt widget.

### Extract tracker subprocess invocation from `RunTrackerWidget`

`app/pose/run_tracker.py` uses `QProcess` to invoke the C++ tracker binary. The process
management logic needs to be extracted into a pure function:

```python
# Proposed: posetrak/tracker/runner.py
def run_tracker(
    session_path: Path,
    sequence_id: str,
    skeleton_id: str,
    config_id: str,
    output_dir: Path,
    *,
    binary_path: Path | None = None,
    person_id: int = 0,
    on_progress: Callable[[str], None] | None = None,
) -> int:  # returns exit code
```

`RunTrackerWidget` is then refactored to call this function (passing a Qt signal as `on_progress`)
rather than managing the subprocess itself. The CLI calls the same function with a stderr printer.
This also makes the tracker invocation unit-testable with a mock binary.

### Consider splitting `app/pose/detection_pipeline.py`

`DetectionPipeline` is already GUI-free but lives in `app/pose/`. As the CLI grows, it may be
cleaner to move it (and `backends.py`, `backends_yolo.py`, `backends_rtmpose.py`) to
`posetrak/detection/` to make the library boundary explicit. This is optional and can wait until the
CLI is otherwise working.

### `detect finalise` needs an assignment input format

`finalise_to_db()` takes a `dict[int, str]` mapping YOLO track IDs to person names. The CLI needs
a convention for this — proposed: `--assign "0:Alice" --assign "1:Bob"`, or a JSON file
`--assignments assignments.json`. The UI version assigns these interactively via `StitcherPanel`.

---

## Entry point consolidation

Replace the current scattered entry points with a single `posetrak` command:

```toml
[project.scripts]
posetrak    = "posetrak.cli.main:main"  # new unified entry point
posetrak-db = "posetrak.cli.main:main"  # alias for backwards compatibility
posetrak-mcp = "app.mcp.server:main"   # keep separate (server lifecycle is different)
```

Remove when ready (after `posetrak-ui` absorbs remaining functionality):
```toml
# posetrak-pose = "app.pose.cli:main"   # retire: absorbed by posetrak detect
# posetrak-setup = "app.setup.main:main" # retire: functionality in UI
```

### File layout

```
python/posetrak/cli/
    __init__.py
    main.py          # entry point, global options, Click group wiring
    registry.py      # registry, camera-model, camera-mode, camera commands
    session.py       # session, capture, extrinsics, sync commands
    skeleton.py      # skeleton commands
    config.py        # config commands
    detect.py        # detect run/list/finalise commands
    track.py         # track run/list/show commands
    export.py        # export bvh/gltf/usd commands
    db.py            # db export-trial/import-trial commands
    _output.py       # shared table/JSONL formatter helpers
```

Migrate the existing `posetrak/db/cli.py` content into these modules; do not attempt to do it in
one PR — migrate group by group.

---

## Output format

All `list` and `show` commands support two output modes:

**Human-readable (default):** aligned tables, coloured status indicators, UUID abbreviations (first 8 chars with `…` suffix).

**JSONL (`--json`):** one JSON object per line for `list`, a single JSON object for `show`. All UUIDs are emitted in full. This is what scripts and the future MCP server will consume.

Example:
```
$ posetrak --session session.db capture list
ID          LABEL        VIDEOS  CAMERAS  SYNC
a3f7b2c1…  outdoor-01   4       4        ✓
8d14ef90…  indoor-02    3       3        ✗

$ posetrak --session session.db capture list --json
{"id": "a3f7b2c1-...", "label": "outdoor-01", "n_videos": 4, "n_cameras": 4, "has_sync": true}
{"id": "8d14ef90-...", "label": "indoor-02", "n_videos": 3, "n_cameras": 3, "has_sync": false}
```

---

## Testing strategy

The library layer is already well tested (`tests/db/`, `tests/app/`). CLI tests should focus on
the argument parsing and output formatting layer, calling real library functions against in-memory
or `tmp_path` databases — no mocking of business logic.

### Fixtures (extend existing patterns)

```python
# conftest.py additions
@pytest.fixture
def cli_runner():
    return CliRunner()  # Click's test runner

@pytest.fixture
def populated_registry(tmp_path):
    # Reuse existing registry_db fixture; seed with camera model + mode
    ...

@pytest.fixture
def populated_session(tmp_path, populated_registry):
    # Seed session with capture, videos, sync, pose sequences
    ...
```

### Test scope per command group

| Group | Test focus |
|---|---|
| `registry`, `camera*`, `skeleton`, `config` | Round-trip: create → list → show output format; ID prefix resolution |
| `session`, `capture`, `extrinsics`, `sync` | Create then query; `--json` output is valid JSONL |
| `detect run` | Mock `YOLOv11Detector` and `RTMPoseEstimator`; verify DB writes |
| `track run` | Mock subprocess (replace binary path with echo script); verify output dir |
| `export *` | Run against fixture tracking run; verify output file is valid BVH/glTF |
| Global | `--json` flag propagation; `--db` / `--session` path resolution; error exit codes |

The tracker subprocess extraction (`run_tracker()`) is specifically designed to be testable with a
trivial mock binary, unlike the current `QProcess`-based widget.

---

## Implementation phases

**Phase 1 — Consolidate existing CLI (low risk)**
Migrate `posetrak/db/cli.py` into `posetrak/cli/` with the new Click structure.
Add `posetrak-db` as a backwards-compatible alias. No new functionality, no refactoring.
Adds `--json` output to all existing commands.

**Phase 2 — Detect commands**
Add `posetrak detect run/list/finalise` by porting `app/pose/cli.py`.
Define the `--assign` format for `finalise`.

**Phase 3 — Export commands**
Add `posetrak export bvh/gltf/usd`. These are simple wrappers; no refactoring needed.

**Phase 4 — Tracker subprocess extraction**
Extract `run_tracker()` from `RunTrackerWidget` into `posetrak/tracker/runner.py`.
Add `posetrak track run` command.
Add `RunTrackerWidget` refactor to call the extracted function.

**Phase 5 — Data portability**
Implement `posetrak db export-trial` / `import-trial`. This is new functionality with no existing
implementation; requires defining the subset-copy schema logic.

**Phase 6 — Retire old entry points**
Remove `posetrak-pose` and `posetrak-setup` from `pyproject.toml`.
Remove `app/pose/cli.py`, `app/setup/main.py`, `app/setup/page_session.py`.
