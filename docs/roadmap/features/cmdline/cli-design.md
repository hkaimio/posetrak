# Posetrak CLI — Requirements & Design Proposal

## Goals

Expose all meaningful Posetrak functionality via a single `posetrak` command that mirrors the UI
workflow. The CLI is a first-class interface, not a thin wrapper — it must be fully scriptable for
batch pipelines and automated testing. The existing `posetrak-db` entry point covers a large part of
this already and will be absorbed rather than replaced.

---

## Binary naming

The C++ tracker executable built by Meson is currently also named `posetrak`, which conflicts with
the Python CLI. The C++ binary should be renamed to `posetrak-tracker` (updated in
`cli/meson.build`). The Python CLI takes the `posetrak` name. The `run_tracker()` extration (see
Refactoring section) will need to reference the new binary name.

---

## Command hierarchy

The unified entry point is `posetrak` with subcommand groups matching the domain model:

```
posetrak [--registry PATH] [--session PATH] [--json] [-v] <group> <command> [args...]

Global options:
  --registry PATH   Registry database.
                    Default: $POSETRAK_REGISTRY or ~/.posetrak/registry.db
  --session PATH    Session database. Most write commands require --session;
                    read-only commands on registry data only require --registry.

Output options:
  --json            Emit JSONL instead of human-readable tables (all list/show commands)
  -v / --verbose    Increase log verbosity
```

`POSETRAK_REGISTRY` env var sets the default registry path so users working with a single
registry do not have to pass `--registry` every time.

---

## Data model clarification

The session DB contains two levels above capture:

```
registry.db  (camera hardware, skeletons, configs)
session.db
  └── mocap_session  (a recording event: date, location)
        └── capture  (one recording take: set of synchronised video files)
              ├── capture_videos  (one per camera)
              ├── sync_config / sync_points
              ├── detection_run → pose_observation_sequences
              └── tracking_run
```

The `--session` flag always refers to the **session DB file**. Within that file, a `mocap_session`
row (created via `posetrak session create`) is the parent of one or more captures. In practice many
projects have a single mocap_session per DB, but the model supports multiple.

---

## Command groups

### Registry & camera setup

Commands that operate on the registry use `--registry` only; no `--session` needed.

```
posetrak registry init [PATH]          # create new registry DB
posetrak registry info                 # print schema version, counts

posetrak camera-model add --manufacturer STR --model STR [--sensor-width MM]
posetrak camera-model list

posetrak camera-mode add --model ID --width PX --height PX --fps N [--codec STR]
posetrak camera-mode list [--model ID]

# Camera instances live in the registry by default; pass --session to add to a session DB instead
posetrak camera add --mode ID --label STR [--serial STR] [--session PATH]
posetrak camera list [--session PATH]  # lists registry cameras if --session omitted
posetrak camera show ID

posetrak calib import PATH --camera-mode ID   # TOML (Pose2Sim format)
posetrak calib import-h5 PATH --camera-mode ID  # HDF5 (legacy)
posetrak calib list [--camera ID]
```

### Skeleton management

Skeletons live in the registry. The scaling parameters are the actual bone-length measurements
exposed by `template_measurements()` in `scale_skeleton.py`:

```
posetrak skeleton import PATH [--name STR]
posetrak skeleton list
posetrak skeleton show ID
posetrak skeleton export ID PATH

posetrak skeleton scale ID \
    [--femur M] [--shin M] [--upper-arm M] [--lower-arm M] \
    [--torso-height M] [--shoulder-width M] \
    [--name STR | --output PATH]
  # All lengths in metres. Omitted measurements are left unscaled.
  # Prints scaling_summary() table to stderr.
  #
  # --name STR    Save scaled skeleton back to the registry DB under a new name (primary use case).
  # --output PATH Write scaled YAML to file instead (use - for stdout).
  # Exactly one of --name or --output must be provided.
```


### Tracker config

```
posetrak config create PATH [--name STR]   # import from TOML file
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

posetrak capture create --session-id ID [--label STR]
posetrak capture list [--session-id ID]
posetrak capture show ID

# Cameras are linked to a capture via their videos, not separately.
# --camera-mode is required; it records which capture mode was used for this video.
posetrak capture add-video CAPTURE_ID VIDEO_PATH \
    --camera ID --camera-mode ID \
    [--first-frame N] [--last-frame N]

posetrak extrinsics import PATH --capture ID [--method pose2sim]
posetrak extrinsics list [--capture ID]

posetrak sync import PATH --capture ID
posetrak sync list [--capture ID]
```

### Pose detection

```
posetrak detect run --capture ID --sync ID --start S --end S \
                    [--detector yolo11x] [--pose-model rtmpose-l-133kp] \
                    [--conf 0.3]
  # Wraps DetectionPipeline; streams progress to stderr, prints run-id to stdout.

posetrak detect list [--capture ID]
posetrak detect show ID
```

**Note on `detect finalise`:** Converting a detection run to labelled pose sequences requires
assigning YOLO track IDs to named persons. For single-person captures this is trivial
(`--assign "0:Subject"`), but for multi-person captures it is inherently interactive —
the user needs to inspect which track ID belongs to which person across cameras. The CLI
design for this command needs more thought and is deferred to a later phase (see Phases).

### Tracking

```
posetrak track run --sequence ID --skeleton ID --config ID \
                   [--output-dir PATH] [--person-id N] [--fps N]
  # Invokes posetrak-tracker binary as subprocess; streams output to stderr.

posetrak track list [--sequence ID]
posetrak track show ID

posetrak track export bvh  ID OUTPUT_PATH [--smoothed]
posetrak track export gltf ID OUTPUT_PATH [--smoothed]
posetrak track export usd  ID OUTPUT_PATH [--smoothed]
  # ID is a tracking-run ID.
```

### Trial portability

A "trial" is the minimal self-contained unit for sharing: a capture + its detection + tracking
run + all dependencies (cameras, calibrations, skeleton, config).

```
posetrak trial export ID OUTPUT_PATH
  # Writes a new session DB containing only the specified trial and its dependencies.

posetrak trial import PATH [ID ...]
  # Merges trials from the source DB into the current --registry and --session.
  # If one or more IDs are given, imports only those trials; default is all trials in PATH.
```

---

## Reuse of existing code

Most business logic exists today in GUI-free modules. The CLI is primarily an argument-parsing
layer over these.

### Absorb `posetrak/db/cli.py` wholesale (2 400 lines)

The existing `posetrak-db` argparse CLI covers registry, camera, skeleton, config, session,
capture, extrinsics, sync, pose, and tracking-run commands. Migration plan:

- Keep all library call sites unchanged.
- Replace argparse with Click groups matching the new command hierarchy.
- Merge `--registry` / `--session` global option handling.
- Reuse `resolve_id_prefix()` helper and existing interactive prompts verbatim.

### Absorb `app/pose/cli.py` (137 lines)

The `run` and `list-runs` commands call `DetectionPipeline`, `YOLOXDetector`,
`RTMPoseEstimator`, and `list_detection_runs()` without any Qt. These become
`posetrak detect run` and `posetrak detect list`.

### Directly callable library functions

| CLI command | Library function | Module |
|---|---|---|
| `skeleton import/export/scale` | `import_skeleton()`, `scale_skeleton_yaml()`, `scaling_summary()` | `db/manage_skeleton.py`, `db/scale_skeleton.py` |
| `config create/edit` | `create_config_from_toml()`, `edit_config()` | `db/manage_config.py` |
| `calib import` | `import_calib_toml()`, `import_calib_h5()` | `db/import_calib_toml.py`, `db/import_calib_h5.py` |
| `extrinsics import` | `import_extrinsics()` | `db/import_extrinsics.py` |
| `sync import` | `import_sync_json()` | `db/import_sync_json.py` |
| `session import-yaml` | `import_session_yaml()` | `db/import_session_yaml.py` |
| `detect run` | `DetectionPipeline.run()` | `app/pose/detection_pipeline.py` |
| `detect finalise` | `finalise_to_db()` | `app/pose/finalise.py` |
| `track export bvh/gltf/usd` | `export_bvh()`, `export_gltf()`, `export_usd()` | `export/bvh.py`, `export/gltf.py`, `export/usd.py` |

---

## Required refactoring (minimal)

### Extract tracker subprocess invocation from `RunTrackerWidget`

`app/pose/run_tracker.py` uses `QProcess` to invoke the C++ binary. Extract into a pure function:

```python
# Proposed: posetrak/tracker/runner.py
def run_tracker(
    session_path: Path,
    sequence_id: str,
    skeleton_id: str,
    config_id: str,
    output_dir: Path,
    *,
    binary_path: Path | None = None,  # defaults to ~/.posetrak/posetrak-tracker
    person_id: int = 0,
    on_progress: Callable[[str], None] | None = None,
) -> int:  # returns exit code
```

`RunTrackerWidget` is refactored to call this function with a Qt signal as `on_progress`.
The CLI calls it with a stderr printer. Also makes tracker invocation testable with a mock binary.

### Move detection backends to `posetrak/detection/`

**Done** (as part of the ultralytics removal, see `docs/license-analysis.md`):
`backends.py` and the detector backend (`backends_rtmdet.py`, replacing the old
`backends_yolo.py`) now live in `posetrak/detection/`, with only a thin
`app/pose/backends_rtmpose.py` re-export shim remaining for backwards
compatibility. Import paths are cleaner for both the CLI and future MCP tools.

### Rename C++ binary

In `cli/meson.build`, change the executable name from `posetrak` to `posetrak-tracker`.
Update `run_tracker()` default binary path accordingly.

---

## Entry point consolidation

```toml
[project.scripts]
posetrak     = "posetrak.cli.main:main"   # unified CLI
posetrak-db  = "posetrak.cli.main:main"   # backwards-compatible alias
posetrak-mcp = "app.mcp.server:main"      # keep separate (server lifecycle differs)
# retire when ready:
# posetrak-pose  = "app.pose.cli:main"
# posetrak-setup = "app.setup.main:main"
```

### File layout

```
python/posetrak/cli/
    __init__.py
    main.py        # entry point, global options, Click group wiring
    registry.py    # registry, camera-model, camera-mode, camera commands
    session.py     # session, capture, extrinsics, sync commands
    skeleton.py    # skeleton commands
    config.py      # config commands
    detect.py      # detect run/list commands
    track.py       # track run/list/show/export commands
    trial.py       # trial export/import commands
    _output.py     # shared table/JSONL formatter helpers
```

---

## Output format

All `list` and `show` commands support two modes:

**Human-readable (default):** aligned tables, coloured status, UUID abbreviations (first 8 chars + `…`).

**JSONL (`--json`):** one JSON object per line for `list`, single object for `show`. Full UUIDs.
This is what scripts, CI pipelines, and MCP tools consume.

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

The library layer is already well tested (`tests/db/`, `tests/app/`). CLI tests call real library
functions against `tmp_path` databases via Click's `CliRunner` — no mocking of business logic.

```python
# conftest.py additions
@pytest.fixture
def cli_runner():
    return CliRunner()

@pytest.fixture
def populated_registry(tmp_path): ...   # extends existing registry_db fixture

@pytest.fixture
def populated_session(tmp_path, populated_registry): ...
```

| Group | Test focus |
|---|---|
| `registry`, `camera*`, `skeleton`, `config` | Round-trip: create → list → show; ID prefix resolution |
| `session`, `capture`, `extrinsics`, `sync` | Create then query; `--json` output is valid JSONL |
| `detect run` | Mock `YOLOXDetector` / `RTMPoseEstimator`; verify DB writes |
| `track run` | Mock binary with echo script; verify output dir created |
| `track export` | Run against fixture tracking run; verify output is parseable BVH/glTF |
| Global | `--json` propagation; `--registry` / `--session` resolution; error exit codes |

---

## MCP server integration (future phase)

The existing MCP server (`app/mcp/server.py`) is currently read-only and diagnostic: it exposes
filter statistics, camera coverage, and observation data for investigating tracking runs. Once the
CLI library layer is in place, the MCP server can be extended to cover the full workflow —
letting Claude (or any MCP client) orchestrate captures, run detection, and trigger exports through
conversation.

### Architecture

Both the CLI and the MCP server are thin interface layers over the same `posetrak/` library code.
The MCP server should call library functions directly (not shell out to the CLI), sharing the
exact same code path:

```
posetrak/db/      ←── CLI commands (posetrak/cli/)
posetrak/export/  ←── MCP tools   (app/mcp/tools/)
posetrak/tracker/ ←── UI widgets  (app/pose/, app/ui/)
```

This means there is no MCP-specific business logic to maintain — any bug fixed in the library
benefits all three consumers.

### Proposed MCP tool additions

**Read tools (extend existing):**
```
list_captures(session_path)            → capture metadata list
list_detection_runs(session_path)      → detection run list
list_tracking_runs(session_path)       → tracking run list
get_capture_info(session_path, id)     → videos, cameras, sync status
```

**Write tools (new — require explicit user confirmation in the MCP client):**
```
run_detection(session_path, capture_id, sync_id, start_s, end_s, ...)
  → Calls DetectionPipeline.run(); returns detection_run_id

run_tracker(session_path, sequence_id, skeleton_id, config_id, output_dir)
  → Calls posetrak/tracker/runner.run_tracker(); returns exit code + output path

export_tracking_run(session_path, run_id, format, output_path)
  → Calls export_bvh() / export_gltf() / export_usd(); returns output_path
```

**Utility tools:**
```
get_skeleton_template_measurements(session_path, skeleton_id)
  → Returns the template bone lengths (useful before calling scale)

describe_config(session_path, config_id)
  → Human-readable summary of UKF parameters
```

### Phasing

MCP write tools should be added after Phase 4 (tracker extraction), since `run_tracker()` is the
shared primitive that both `posetrak track run` and the MCP `run_tracker` tool will call. The
detection tools can be added earlier (after Phase 2) since `DetectionPipeline` is already
GUI-free.

The MCP server's read-only posture should be made explicit in the server configuration so users
can opt in to write tools intentionally, given that they can modify session databases.

---

## Implementation phases

**Phase 1 — Consolidate existing CLI**
Migrate `posetrak/db/cli.py` into `posetrak/cli/` with Click structure and `--registry` global.
Add `--json` output to all commands. Keep `posetrak-db` alias. No new functionality.
Rename C++ binary to `posetrak-tracker`.

**Phase 2 — Detect commands**
Port `app/pose/cli.py` → `posetrak detect run/list`.
Move detection backends to `posetrak/detection/` (optional, can follow).

**Phase 3 — Track export commands**
Add `posetrak track export bvh/gltf/usd`. Simple wrappers; no refactoring needed.

**Phase 4 — Tracker subprocess extraction**
Extract `run_tracker()` from `RunTrackerWidget` into `posetrak/tracker/runner.py`.
Add `posetrak track run`. Refactor `RunTrackerWidget` to call the extracted function.
Add MCP detection tools (reads + `run_detection` write tool).

**Phase 5 — Trial portability**
Implement `posetrak trial export/import`. New functionality; requires subset-copy schema logic.
Add MCP tracker and export tools.

See [Phase 5 detailed design](#phase-5-detailed-design) below.

**Phase 6 — Detect finalise**
Design and implement `posetrak detect finalise` with a workable CLI convention for
person assignment. Single-person case (`--assign "0:Subject"`) can ship first; multi-person
needs more design (possibly an interactive TUI or a two-step list-then-assign flow).

**Phase 7 — Retire old entry points**
Remove `posetrak-pose`, `posetrak-setup` from `pyproject.toml`.
Remove `app/pose/cli.py`, `app/setup/main.py`, `app/setup/page_session.py`.

---

## Phase 5 detailed design

### Concept hierarchy

The intended domain hierarchy (trial is a first-class concept, not optional metadata):

```
session
  └── capture  (synchronised set of camera videos, shared extrinsics + sync)
        └── trial  (named time window within a capture: one technique, one attempt)
              ├── detection run  (pose estimation output over the trial window)
              │     └── person detections  (stitched per-person observation sequences)
              └── tracking run   (UKF run over person detections)
```

The current schema partially reflects this: `trials` exists and `detection_runs.trial_id`
links to it. However `tracking_runs` does not yet reference `trials` directly — it links via
`pose_observation_sequences → capture`. This gap will be bridged in a future schema migration;
for now the export logic resolves tracking runs for a trial by time-range overlap with the
trial's `[time_start_s, time_end_s]` window and the capture's observation sequences.

### Export scopes

Four levels, named to indicate "up to and including this level":

| Scope | What's included | Default when anchor is |
|-------|-----------------|------------------------|
| `capture-only` | capture + videos + calibrations + sync | `--capture` |
| `trial-only` | above + trial time-window rows | — |
| `detection-only` | above + detection runs + pose observations + edits + seg data | `--trial`, `--detection` |
| `full` | above + tracking runs + results | `--tracking-run` |

Detections without their keypoint blobs are not useful — if a detection run is included,
all of `detection_keypoints`, `person_detections`, and `pose_observations` travel with it.

### CLI interface

```bash
# Browse trials and their detection/tracking status
posetrak --session SRC.db trial list

# Export by anchor type; --scope overrides the default for that anchor
posetrak --session SRC.db trial export OUTPUT.db
    (--trial ID | --capture ID | --detection ID | --tracking-run ID)  # one or more
    [--scope capture-only|trial-only|detection-only|full]
    [--include-cache]              # include frame_cache_entries (excluded by default)
    [--skip-tables TABLE,...]      # exclude known-corrupted tables by name
    [--dry-run]                    # report row counts without writing

# Import from an exported (or any) session DB
posetrak --session DST.db trial import SRC.db
    [--trial ID | --capture ID ...]   # subset; default: everything in SRC.db
    [--dry-run]
```

Each anchor flag can be repeated to export multiple items in one output DB:

```bash
# Export two specific trials
posetrak --session src.db trial export out.db --trial ab12cd34 --trial ef56gh78

# Export a whole capture (infrastructure only, no detections)
posetrak --session src.db trial export out.db --capture a3f7b2c1 --scope capture-only
```

If a DB has no `trials` rows (older workflow), `trial list` falls back to listing tracking
runs and captures; `trial export` accepts `--tracking-run ID` or `--capture ID` directly.

Recovery workflow for a corrupted source DB:

```bash
# 1. See what trials / tracking runs are present
posetrak --session corrupted.db trial list

# 2. Dry-run to assess what is readable
posetrak --session corrupted.db trial export /dev/null \
    --trial TRIAL_ID --scope full --dry-run

# 3. Export, skipping the corrupted table
posetrak --session corrupted.db trial export clean.db \
    --trial TRIAL_ID --scope full --skip-tables pose_observation_edits

# 4. Import on the target machine
posetrak --session new.db trial import clean.db
```

### Dependency graph (full scope)

```
Registry tables (only rows referenced by this trial):
  camera_models
  camera_modes
  camera_instances
  intrinsics_calibrations
  skeletons                         (used by tracking_runs)
  tracker_configs                   (used by tracking_runs)

Session tables (dependency order):
  mocap_sessions
  extrinsic_calibrations
  extrinsic_entries
  captures
  capture_videos
  sync_configs
  sync_points
  sync_anchors
  sync_anchor_observations
  trials                            (the exported trial row)
  detection_runs                    (linked to trial, or time-range matched)
  detection_keypoints               [excluded below detection scope]
  person_detections                 [excluded below detection scope]
  person_tracks
  detection_track_assignments
  pose_observation_sequences
  sequence_persons
  pose_observations                 [excluded below detection scope]
  pose_observation_edits            [skippable via --skip-tables]
  seg_quality_runs                  [included in detection scope; large]
  keypoint_obs_quality              [included in detection scope; large]
  seg_masks                         [included in detection scope; large blobs]
  tracking_runs                     [full scope only]
  tracking_run_persons              [full scope only]
  tracking_results                  [full scope only]
  tracking_obs_results              [full scope only]
```

Note on segmentation tables (`seg_quality_runs`, `keypoint_obs_quality`, `seg_masks`):
these feed keypoint confidence weighting in tracking — excluding them produces a valid but
lower-fidelity export. They are included by default in detection and full scopes because
stripping them silently degrades tracking reproducibility. Excluded with `--scope capture-only`
or `--scope trial-only`.

`frame_cache_entries` — excluded by default (large JPEG blobs, can be regenerated by the UI
from the original video files), but included when `--include-cache` is given. Needed if
the exported DB will be used on a machine without access to the original video files.

### Robustness requirements

1. **Source opened read-only** via URI (`?mode=ro`) — migrations never run on a potentially
   corrupted source DB.
2. **Per-table error handling** — each table copy is wrapped independently; a failure emits a
   warning and continues rather than aborting the whole export.
3. **`--skip-tables`** — explicit opt-out for known-bad tables by name (e.g. `pose_observation_edits`).
4. **`--dry-run`** — traces the dependency walk and reports row counts per table without writing
   anything; safe to run against a corrupted DB to assess what is recoverable.
5. **FK enforcement OFF** on destination during import (same pattern as `camera import-session`).
6. **Progress output** — for large tables (pose_observations, tracking_results) print a one-line
   summary per table so the user can see progress on long exports.

### Library layer

New module `python/posetrak/db/trial_export.py` (CLI is a thin wrapper over this):

```python
from enum import Enum

class ExportScope(Enum):
    CAPTURE_ONLY   = "capture-only"    # capture + calibrations + sync
    TRIAL_ONLY     = "trial-only"      # above + trial time-window rows
    DETECTION_ONLY = "detection-only"  # above + detections + pose observations + seg data
    FULL           = "full"            # above + tracking runs + results

@dataclass
class AnchorSpec:
    """What to export: one or more items identified by type + ID."""
    trial_ids:        list[str] = field(default_factory=list)
    capture_ids:      list[str] = field(default_factory=list)
    detection_ids:    list[str] = field(default_factory=list)
    tracking_run_ids: list[str] = field(default_factory=list)
    # Empty lists mean "all" for the corresponding type.

@dataclass
class TableResult:
    table: str
    rows_copied: int
    error: str | None    # None = success

@dataclass
class ExportResult:
    anchor: AnchorSpec
    tables: list[TableResult]

def export_trials(
    src: sqlite3.Connection,   # read-only source
    dst: sqlite3.Connection,   # fresh session DB created by caller
    anchor: AnchorSpec,
    *,
    scope: ExportScope,
    include_cache: bool = False,
    skip_tables: set[str] = frozenset(),
    on_progress: Callable[[str], None] | None = None,
) -> ExportResult: ...

def import_trials(
    src: sqlite3.Connection,
    dst_session: sqlite3.Connection,
    dst_registry: sqlite3.Connection | None,  # None = skip registry sync
    anchor: AnchorSpec,                        # empty = all trials in src
    *,
    skip_tables: set[str] = frozenset(),
    dry_run: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> ExportResult: ...
```

### `trial list` output

Shows trial name and capture context, plus detection and tracking run counts:

```
TRIAL_ID    CAPTURE          TRIAL NAME          DETECTIONS  TRACKING RUNS
----------  ---------------  ------------------  ----------  -------------
ab12cd34…   ukemi-2026-05    shomenuchi-1                 2              1
ef56gh78…   ukemi-2026-05    shomenuchi-2                 1              0
```

### `trial import` and registry

`trial import` syncs camera/skeleton/config data to `--registry` as well as `--session`
so the imported trial is immediately usable on a fresh machine without a separate
`camera import-session` step.

### Open question

- **Schema migration**: `tracking_runs` should gain a `trial_id` FK column so the hierarchy
  is fully explicit in the DB. This would simplify the export dependency walk (no time-range
  heuristic needed). Deferred to a future migration; the export logic is written to work
  without it.
