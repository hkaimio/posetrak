# Camera Management — User Stories & UI Design

This document covers the gap identified during the first end-to-end test: there is no UI
for managing camera models, capture modes, or physical camera instances, and the shot
creation wizard does not link videos to registry camera records.  It extends the existing
`capture-pipeline-architecture.md` Phase 2 design.

---

## Current State

### Session DB is already self-contained

`create_session()` in `db.py` runs both `registry_schema.sql` and `session_schema.sql`
into the same file, so every session DB already contains its own copies of
`camera_models`, `camera_modes`, `camera_instances`, `intrinsics_calibrations`,
`skeletons`, and `tracker_configs`.  The registry is a shared catalog for avoiding
re-entry across sessions; the session file works standalone without it.  When cameras are
added from the registry, their records are copied into the session-local tables (preserving
the same UUIDs so re-linking is possible if the registry is later available).

### Gaps

The shot wizard (`page_shots.py`) ignores all of this.  It accepts a free-text
`camera_id` string per video and writes it directly to `shot_videos.camera_instance_id`,
bypassing the session-local camera tables entirely.  As a result:

- The session-local `camera_instances` table is never populated by the wizard.
- `session_cameras` is never populated, so there is no authoritative record of which
  cameras participated in a session.
- `intrinsics_calibration_id` is never set, so the pose pipeline cannot load
  undistortion maps.
- There is no UI to create `camera_models`, `camera_modes`, or `camera_instances` in
  either the registry or the session DB.

### Schema problem: camera mode is at the wrong level

`session_cameras` currently has `camera_mode_id` and `intrinsics_calibration_id` at
session level (primary key `(session_id, camera_instance_id)`).  This means a session
can only record one mode per physical camera — a camera that shoots 4K in one shot and
1080p in another cannot be represented.  Both fields must move to `shot_videos` level
where each video can declare its own mode and calibration.  See §Schema Changes below.

---

## User Stories

### US-CM-1 — View camera catalog

*As a practitioner setting up a session, I want to see all camera models, their capture
modes, and all registered physical camera units in one place, so I can quickly confirm
which cameras are available and which calibrations are attached.*

Acceptance criteria:
- A camera registry panel lists all `camera_models` from the registry DB.
- Each model expands to show its `camera_modes` (resolution, fps, codec) and the
  latest associated `intrinsics_calibration` for each mode (rms_error, date).
- A separate section lists all `camera_instances` (label, model, serial number).
- The panel is accessible without starting a session wizard (camera setup is a
  one-time prerequisite, not per-session).

### US-CM-2 — Register a new camera model

*I buy a new camera. Before I can record anything, I need to register the make/model so I
can attach instances and capture modes to it.*

Acceptance criteria:
- An "Add Camera Model" form collects manufacturer, model name, and optional sensor size.
- The new model appears immediately in the catalog and in any open camera pickers.
- Validation prevents saving a model with an empty model name.

### US-CM-3 — Add capture modes to a camera model

*The same GoPro records at 4K/120fps for slow motion and 1080p/60fps for other shots.
These modes have different intrinsics, so they must be registered separately.  I need to
add modes both when first registering the model and later when I configure a new mode.*

Acceptance criteria:
- From the camera model row an "Add Mode" action opens a form: width, height, nominal fps,
  codec, and freeform notes.
- Modes can be added to an existing model without re-creating the model.
- A mode row shows its resolution/fps alongside the default intrinsics calibration (if any).
- Editing a mode (notes, codec) is possible after creation.

### US-CM-4 — Import an intrinsics calibration from the UI

*Running `calibrate_intrinsics.py` produces an HDF5 file.  I want to import it from the
camera registry panel without switching to the command line.*

Acceptance criteria:
- An "Import calibration…" button on a camera mode row opens a file picker for an HDF5
  file and runs the same import logic as `posetrak-db calib import-h5`.
- The imported calibration is stored in both the registry (if open) and the current
  session's local `intrinsics_calibrations` table.
- On success the mode row updates to show the new calibration's rms_error and date.
- If no HDF5 is available, the user can enter calibration values manually (fx, fy, cx,
  cy, distortion coefficients) via a form; undistortion maps are computed on save using
  `cv2.initUndistortRectifyMap`.

### US-CM-5 — Set default intrinsics calibration for a mode

*After importing a calibration I want to mark it as the default for its mode so the shot
wizard offers it automatically.*

Acceptance criteria:
- The mode row shows all available `intrinsics_calibrations` for that mode.
- A "Set as default" control on each calibration marks it as the default.
  Only one calibration per mode can be the default.
- The default calibration is shown with its rms_error and calibration date.
- The field is nullable: a mode can exist without any calibration (produces a warning in
  the shot wizard but does not block progress).
- **Schema addition required**: a `default_intrinsics_calibration_id` nullable FK column
  on `camera_modes` pointing to `intrinsics_calibrations(id)`.

### US-CM-6 — Register a physical camera instance

*I have two identical GoPro Hero 12 bodies. I label them "cam1" and "cam2" by serial
number so I can track which unit captured which video across sessions.*

Acceptance criteria:
- An "Add Camera" form collects a user label, a camera model (from the catalog), and an
  optional serial number.
- The instance appears in the camera picker in the shot wizard immediately.
- If a serial number is entered, the app validates it is unique within the registry.

### US-CM-7 — Assign camera instance and mode when adding videos to a shot

*When adding a video file to a shot, I must link it to a specific physical camera and the
mode it was recording in, so the correct intrinsics are applied and
`session_cameras` is populated.*

Acceptance criteria:
- Each video row in the shot wizard shows a **camera instance picker** (combo box showing
  label + model for all instances in the registry, plus a "Create new camera…" option).
- After a camera instance is picked, a **mode picker** shows only modes belonging to that
  instance's model, filtered to those whose resolution and fps match the video's probe
  result (other modes accessible via "Show all").
- After a mode is picked, the **intrinsics calibration** field shows the mode's default
  calibration (or "(none — calibration missing)" with a warning icon if absent).
- `session_cameras` is upserted (insert if not present, otherwise no-op) when the page is
  committed, using the selected `camera_instance_id`, `camera_mode_id`, and
  `intrinsics_calibration_id`.
- `shot_videos.camera_instance_id` is set to the registry UUID of the selected instance
  (not a user-typed string).
- Validation on "Next": every video must have a camera instance and mode assigned.  The
  page warns (but does not block) if any mode lacks an intrinsics calibration.

### US-CM-8 — Create a camera inline during shot setup

*A new camera is used for the first time.  I should be able to register it without
leaving the shot wizard.*

Acceptance criteria:
- Selecting "Create new camera…" in the camera instance picker opens a compact dialog
  (not a new window) asking for: camera model (dropdown with existing models + "New
  model…" option), label, and optional serial number.
- If "New model…" is chosen, a nested dialog asks for manufacturer, model name, and
  sensor size.
- Saving the dialog creates the registry records and selects the new instance in the
  picker automatically.
- The mode picker is updated to show the new model's modes (initially empty; the user
  will need to add modes in the camera registry panel before modes are available).

### US-CM-9 — Auto-detect camera mode from video metadata

*When I add a video the wizard should auto-suggest the matching mode so I don't have to
look it up manually.*

Acceptance criteria:
- After the video probe completes (background), the mode picker pre-selects the mode whose
  `width_px × height_px × nominal_fps` matches the probe result (within ±1 fps tolerance).
- If no single mode matches, the picker shows "(no match — select manually)" and highlights
  the closest candidates.
- If a serial number is found in the video metadata (`video_probe.py` already extracts
  this from GoPro GPMF / exiftool), the camera instance picker pre-selects the instance
  with that serial number.

---

## Required Schema Changes

### 1. Move camera mode + intrinsics to `shot_videos` level

`session_cameras` currently stores `camera_mode_id` and `intrinsics_calibration_id` at
session level, allowing only one mode per camera per session.  Both columns must move to
`shot_videos` so each video can declare its own mode.  This is a breaking migration; it
requires updating the C++ tracker and `load_session.py` to read intrinsics from
`shot_videos` instead of `session_cameras`.

```sql
-- Migration 009 (session DB)
CREATE TABLE session_cameras_new (
    session_id         TEXT NOT NULL REFERENCES mocap_sessions(id),
    camera_instance_id TEXT NOT NULL,
    label              TEXT,
    PRIMARY KEY (session_id, camera_instance_id)
);
INSERT INTO session_cameras_new (session_id, camera_instance_id, label)
    SELECT session_id, camera_instance_id, label FROM session_cameras;
DROP TABLE session_cameras;
ALTER TABLE session_cameras_new RENAME TO session_cameras;

-- Per-shot mode + intrinsics on shot_videos
-- Both nullable: a video without mode/intrinsics is valid (produces a warning,
-- not an error; the pose pipeline can still run but cannot apply undistortion).
ALTER TABLE shot_videos ADD COLUMN camera_mode_id TEXT;
ALTER TABLE shot_videos ADD COLUMN intrinsics_calibration_id TEXT;
```

### 2. Default calibration on `camera_modes`

```sql
-- registry_schema.sql and session-local camera_modes (same change in both)
ALTER TABLE camera_modes
    ADD COLUMN default_intrinsics_calibration_id TEXT
        REFERENCES intrinsics_calibrations(id);
```

These are the only schema changes needed; all other tables already support the design.

---

## Software Design

### 5.1 Where camera management lives in the app

The current app is a flat `QWizard`.  Camera management is a registry-level concern (not
session-specific) so it does not belong inside the wizard flow.

**Approach**: introduce a `QMainWindow` shell that hosts both the wizard (as its central
widget when creating a new session) and the camera registry panel (always accessible from
the toolbar/menu).  This is the evolution toward the tab-based layout described in
`capture-pipeline-architecture.md § UI Flow` and can be done incrementally.

```
SetupMainWindow (QMainWindow)
├── Toolbar / Menu
│   ├── File → Open Registry / Open Session DB
│   └── View → Camera Registry
├── Central widget (stacked)
│   ├── SessionWizard  (current QWizard, unchanged)
│   └── CameraRegistryWidget  (new)
└── Status bar
```

For an MVP, `CameraRegistryWidget` can be opened as a `QDialog` from a toolbar button
("Manage Cameras") rather than requiring the full main window refactor.  The widget is
self-contained and works either as a dialog or as a tab.

### 5.2 CameraRegistryWidget

A single widget (usable as dialog or embedded panel) that owns a read/write connection to
the registry DB.

```
python/app/setup/
    camera_registry.py      ← CameraRegistryWidget + sub-dialogs
```

Layout:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Camera Registry                                       [+ Add Model]    │
├───────────────────────┬─────────────────────────────────────────────────┤
│  Models & Modes       │  Physical Cameras                               │
│                       │                              [+ Add Camera]     │
│  ▼ GoPro Hero 12      │  ┌──────────┬──────────────────┬─────────────┐ │
│    [Edit] [Delete]    │  │ Label    │ Model            │ Serial #    │ │
│    Sensor: 1/1.9"     │  ├──────────┼──────────────────┼─────────────┤ │
│                       │  │ cam1     │ GoPro Hero 12    │ C3491234567 │ │
│    Modes [+ Add Mode] │  │ cam2     │ GoPro Hero 12    │ C3498765432 │ │
│    ┌─────────────┬────┴──┤ cam3     │ GoPro Hero 11    │ GP123456    │ │
│    │ 4K Lin 120  │ calib │          │                  │             │ │
│    │ 1080p 60    │ none  │          │                  │             │ │
│    └─────────────┴──────┘└──────────┴──────────────────┴─────────────┘ │
│                                                                         │
│  ▼ GoPro Hero 11                                                        │
│    [Edit] [Delete]                                                      │
│    ...                                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

The left pane is a `QTreeWidget`: top-level items are models; child items are modes.
Each mode item shows a summary line and a calibration status icon (✓ green / ⚠ orange /
✕ red).

Clicking a mode item expands an inline detail area (or opens a small panel to the right)
showing:
- Resolution, fps, codec, notes
- List of all calibrations for this mode (date, rms_error, notes)
- A "Set as default" button next to each calibration; current default marked with ●

The right pane is a `QTableWidget` of camera instances.

#### Add/Edit Camera Model dialog

```
┌─── Add Camera Model ─────────────────────────────┐
│  Manufacturer:   [GoPro                        ]  │
│  Model name:     [Hero 12 Black                ]  │
│  Sensor size:    [1/1.9" (optional)            ]  │
│                                                   │
│                          [Cancel]  [Save]         │
└───────────────────────────────────────────────────┘
```

#### Add/Edit Camera Mode dialog

```
┌─── Add Camera Mode for GoPro Hero 12 ─────────────────────────────┐
│  Resolution:  Width [3840]  Height [2160]                           │
│  Nominal fps: [120.0]   Codec: [h265 (optional)               ]    │
│  Notes:       [4K Linear 120fps                                ]   │
│                                                                     │
│  Default intrinsics calibration:                                   │
│  (none available — import a calibration first with                  │
│   posetrak-db calib import-h5)                                      │
│                                                                     │
│                              [Cancel]  [Save Mode]                 │
└─────────────────────────────────────────────────────────────────────┘
```

When calibrations are available for this mode:

```
│  Default intrinsics calibration:                                   │
│  ○ (none)                                                           │
│  ● 2026-03-10 — rms 0.42 — "checkerboard 25mm"                    │
│  ○ 2025-11-20 — rms 0.61 — "checkerboard 25mm (older)"            │
```

#### Add Camera Instance dialog

```
┌─── Register Physical Camera ──────────────────────────────┐
│  Label:         [cam1                                   ]  │
│  Camera model:  [GoPro Hero 12 Black                  ▼]  │
│  Serial number: [C3491234567 (optional)                ]  │
│                                                            │
│                             [Cancel]  [Register]          │
└────────────────────────────────────────────────────────────┘
```

### 5.3 Shot wizard video row — enhanced

Current layout (single row):
```
[filename.mp4              ] Cam: [cam1     ] [probe info] [✕]
```

New layout (two sub-rows per video):
```
┌──────────────────────────────────────────────────────────────────────────┐
│ filename.mp4                          4K 2160p 120fps  00:45:23     [✕] │
│ Camera: [cam1 — GoPro Hero 12 ▼]  Mode: [4K Linear 120fps ▼]  [calib ✓]│
└──────────────────────────────────────────────────────────────────────────┘
```

- The camera instance picker shows `"label — model_name"` for each registry instance,
  with "Create new camera…" as the last item.
- The mode picker is enabled only after a camera instance is selected; it filters to
  modes belonging to that instance's model.  Modes are shown as `"WxH fps"` with an
  annotation if they match the video probe: `"3840×2160 120fps ✓"`.
- The calibration indicator is a small icon + tooltip:
  - `✓` (green) — mode has a default calibration
  - `⚠` (orange) — mode exists but has no default calibration; processing will skip
    undistortion
  - `✕` (red) — no mode selected
- The `camera_id` free-text field is removed.

The `VideoEntry` dataclass gains:
```python
@dataclass
class VideoEntry:
    path: str
    probe: VideoProbeResult | None = None
    error: str | None = None
    camera_instance_id: str = ""   # registry UUID (was: free-text label)
    camera_mode_id: str = ""       # registry UUID (new)
    intrinsics_calibration_id: str = ""  # registry UUID (new, from mode default)
```

### 5.4 Inline "Create new camera" dialog

Opened from the camera instance picker when "Create new camera…" is selected:

```
┌─── Register New Camera ─────────────────────────────────────────────────┐
│  This camera is not yet in the registry.                                │
│                                                                          │
│  Camera model:  [GoPro Hero 12 Black ▼]  [+ New model…]                │
│  Label:         [cam1                 ]                                  │
│  Serial #:      [C3491234567          ]  ← pre-filled from video probe  │
│                                                                          │
│                              [Cancel]   [Register & Select]             │
└──────────────────────────────────────────────────────────────────────────┘
```

If the registry has no models yet, `[+ New model…]` opens the Add Camera Model dialog and
returns to this dialog with the new model pre-selected.

On "Register & Select": creates the `camera_instances` row, selects it in the picker,
and closes the dialog.  The mode picker then offers modes for the new model (likely empty
initially, showing a hint to add modes via the Camera Registry panel).

### 5.5 DBContext changes

`create_shot_video` signature change — `cam_instance_id` must now be a valid registry
UUID.  The free-text shortcut is removed.

New method `upsert_session_camera`:

```python
def upsert_session_camera(
    self,
    camera_instance_id: str,  # registry UUID
    camera_mode_id: str,       # registry UUID
    intrinsics_calibration_id: str | None,
) -> None:
    """Insert a session_cameras row if not already present.

    If the row already exists (same session_id + camera_instance_id), no-op.
    Raises if camera_instance_id is not in the registry.
    """
    self._conn.execute(
        "INSERT OR IGNORE INTO session_cameras "
        "(session_id, camera_instance_id, camera_mode_id, "
        "intrinsics_calibration_id) VALUES (?, ?, ?, ?)",
        (self._session_id, camera_instance_id, camera_mode_id,
         intrinsics_calibration_id),
    )
```

Called from `ShotsPage.validatePage()` for every video entry before committing.

`DBContext` also needs the registry DB connection to populate camera pickers:

```python
class DBContext:
    def __init__(
        self,
        conn: sqlite3.Connection,      # session DB
        session_id: str,
        registry_conn: sqlite3.Connection,  # registry DB (read-only for pickers)
    ) -> None: ...
```

New read methods on `DBContext`:

```python
def list_camera_instances(self) -> list[CameraInstanceInfo]:
    """All camera_instances from the registry, ordered by label."""
    ...

def list_camera_modes(self, camera_model_id: str) -> list[CameraModeInfo]:
    """All modes for a model, with default calibration if set."""
    ...
```

### 5.6 Registry DB access in the wizard

The wizard currently opens only the session DB.  The session page (`page_session.py`)
also opens (or creates) the registry DB.  Both connections are passed to `DBContext`.

The registry DB path comes from the same settings as before
(`settings.project_root` or a `--registry` CLI flag).  The wizard's "Manage Cameras"
toolbar button opens `CameraRegistryWidget` with the registry connection.

### 5.7 Module layout additions

```
python/app/setup/
    camera_registry.py   ← CameraRegistryWidget, ModelDialog, ModeDialog,
                            InstanceDialog, InlineCreateCameraDialog
```

No new DB or backend files are needed; `CameraRegistryWidget` uses the registry
`sqlite3.Connection` directly (it is a simple CRUD panel).

---

## Open Questions

**Q1 — Manual calibration entry**
If no HDF5 exists (e.g. calibration was computed by another tool and only the K matrix is
available), should the UI offer a manual entry form?  Decision: yes, as a fallback in
US-CM-4.  The form accepts fx, fy, cx, cy, dist_coeffs (comma-separated) and computes
undistortion maps via `cv2.initUndistortRectifyMap`.

**Q2 — Multi-mode per video file**
Deferred.  One mode per `shot_video` row is assumed.  If a GoPro switches modes mid-clip
the video should be split into separate files first.

**Q3 — Legacy sessions with free-text camera IDs**
Sessions created before this change have `shot_videos.camera_instance_id` set to a
user-typed string rather than a registry UUID.  These need a one-time migration:
`posetrak-db session fix-camera-refs <session.db>` prompts the user to map each dangling
string to a real `camera_instances` row (or create one), then updates both `shot_videos`
and `session_cameras`.

---

## Implementation Plan

---

### Phase 1 — Camera Registry panel

**Outcome**: A practitioner can open a "Manage Cameras" dialog from the setup wizard
and perform the full camera lifecycle — create models, add capture modes, import
intrinsics calibrations, designate a default calibration per mode, and register
physical camera units — entirely from the UI.  The wizard itself is unchanged; this
phase is a prerequisite that must be complete before Phase 2.

#### Deliverables

| # | File / location | Change |
|---|---|---|
| 1a | `db/migrations/009_camera_modes_default_calib.sql` | `ALTER TABLE camera_modes ADD COLUMN default_intrinsics_calibration_id TEXT REFERENCES intrinsics_calibrations(id)` — applied to both registry and session DBs |
| 1b | `python/posetrak/db/db.py` | Bump `REGISTRY_SCHEMA_VERSION` and `SESSION_SCHEMA_VERSION`; add `_migrate_*_v*` functions that run migration 009 |
| 1c | `python/app/setup/camera_registry.py` | New file: `CameraRegistryWidget`, `ModelDialog`, `ModeDialog`, `CalibrationImportDialog`, `InstanceDialog` |
| 1d | `python/app/setup/main.py` | Add "Manage Cameras" toolbar button that opens `CameraRegistryWidget` as a `QDialog` backed by the registry connection |
| 1e | `python/app/setup/page_session.py` | Open (or create) the registry DB alongside the session DB; store the registry connection on the wizard for downstream use |

#### CameraRegistryWidget sub-components

- **Model tree** (`QTreeWidget`): top-level = models, children = modes.  Each mode item
  shows `WxH fps — [calib ✓/⚠/✕]`.  Right-click or inline buttons: Add Mode, Edit,
  Delete (with a confirmation if calibrations are attached).
- **ModelDialog**: manufacturer, model name, sensor size fields.
- **ModeDialog**: width, height, nominal fps, codec, notes.  Below the fields: a
  read-only list of calibrations for this mode with "Set as default" radio buttons and
  an "Import calibration…" button.
- **CalibrationImportDialog** (opened from ModeDialog or from a mode's context menu):
  - Primary path: HDF5 file picker → calls `import_calib_h5(registry, mode_id, h5_path)`.
  - Fallback path: "Enter manually" tab with fields for fx, fy, cx, cy,
    dist_coeffs (space-separated floats), image width/height.  On save, computes
    undistortion maps via `cv2.initUndistortRectifyMap` and stores both maps and scalars.
  - On success: refreshes the mode row; if no default is set yet, auto-sets this
    calibration as the default.
- **Instance table** (`QTableWidget`): columns = Label, Model, Serial #.  "Add Camera"
  button opens **InstanceDialog**: label (required), model dropdown (existing models),
  serial number (optional, validated unique within registry).

#### Test criteria

1. **Schema migration**: opening a registry DB at v3 applies migration 009 and sets
   `user_version = 4` without errors; `camera_modes.default_intrinsics_calibration_id`
   is `NULL` for all existing rows.  Idempotent: running migration twice is a no-op.
2. **Add model**: fill in manufacturer + model name → Save → model appears as a
   top-level tree item with zero mode children.  Saving with an empty model name is
   rejected with an inline error.
3. **Add mode**: select a model → Add Mode → fill in 3840×2160 / 120 fps → Save →
   mode appears as a child item showing "3840×2160 120fps — ✕".
4. **Import calibration (HDF5)**: click "Import calibration…" on a mode → pick a real
   HDF5 file → dialog closes → mode item now shows "3840×2160 120fps — ✓" → the
   calibration row appears in the mode detail with a non-null rms_error and
   `default_intrinsics_calibration_id` set on the mode row.
5. **Import calibration (manual)**: switch to "Enter manually" tab → supply valid fx, fy,
   cx, cy, dist_coeffs → Save → `intrinsics_calibrations` row created with non-null
   `undistort_mapx`/`undistort_mapy` blobs (verified by reading the DB directly).
6. **Set default**: when a mode has two calibrations, clicking "Set as default" on the
   older one updates `camera_modes.default_intrinsics_calibration_id` to the older
   calibration's ID.
7. **Add instance**: fill in label "cam1", select model, optionally serial number →
   Register → row appears in the instance table.  Entering a duplicate serial number is
   rejected.
8. **Delete model with attached modes**: confirmation dialog warns that N modes will also
   be deleted; cancelling leaves the model intact.
9. **Registry-less session**: opening the wizard without a registry DB disables the
   "Manage Cameras" button and shows a tooltip explaining that a registry must be
   configured first.

---

### Phase 2 — Schema migration + shot wizard integration

**Outcome**: The shot wizard links every video to a real camera instance and capture mode
from the session-local camera tables (copied from the registry at assignment time).
`session_cameras` records which cameras are in the session.  `shot_videos` records the
mode and intrinsics for each video.  After the wizard the session DB is self-contained:
intrinsics (including undistortion maps) are present without the registry.  The C++
tracker and `load_session.py` are updated to read intrinsics from `shot_videos`.

#### Deliverables

| # | File / location | Change |
|---|---|---|
| 2a | `db/migrations/010_shot_videos_mode_intrinsics.sql` | Rebuild `session_cameras` without mode/intrinsics columns; add `camera_mode_id` + `intrinsics_calibration_id` (both nullable) to `shot_videos` |
| 2b | `python/posetrak/db/db.py` | Bump `SESSION_SCHEMA_VERSION`; add `_migrate_session_v*` function applying migration 010 |
| 2c | `python/posetrak/db/load_session.py` | Change intrinsics lookup from `session_cameras` → `shot_videos`; join `shot_videos → intrinsics_calibrations` for K matrix, dist_coeffs, and undistortion maps |
| 2d | `src/` or `cli/` (C++ tracker) | Update camera loading to read intrinsics FK from `shot_videos` rather than `session_cameras` |
| 2e | `python/app/setup/db_context.py` | Accept `registry_conn` parameter; add `upsert_session_camera()`, `upsert_camera_records()`, `list_camera_instances()`, `list_camera_modes()` |
| 2f | `python/app/setup/page_shots.py` | Replace free-text `camera_id` field in `_VideoRow` with three-picker layout; update `VideoEntry`; update `validatePage()` to call `upsert_session_camera()` and copy camera records |
| 2g | `python/app/setup/page_session.py` | Pass registry connection to `DBContext` constructor |

#### Key design points

**`upsert_camera_records()`** in `DBContext`: when a camera instance is selected in the
picker, this method copies the `camera_models`, `camera_modes`, and
`camera_instances` rows (and the selected `intrinsics_calibrations` row including maps)
from the registry into the session-local tables using `INSERT OR IGNORE`.  The same UUIDs
are preserved so the session can be re-linked to the registry later.  If no registry
connection is present (session-only mode), this is a no-op.

**`upsert_session_camera()`**: inserts `(session_id, camera_instance_id)` into
`session_cameras` with `INSERT OR IGNORE`.  No mode or intrinsics columns — those are
now on `shot_videos`.

**`create_shot_video()`** signature change: the `cam_instance_id` parameter now accepts
only a UUID that exists in the session-local `camera_instances` table; the free-text
fallback is removed.  `camera_mode_id` and `intrinsics_calibration_id` are new
parameters (nullable).

**Picker behaviour**:
- Camera instance picker is populated from the session-local `camera_instances` table
  (which may already contain rows from the registry via `upsert_camera_records`, or from
  Phase 3 inline creation).  If a registry connection exists, it is also queried to show
  instances not yet copied into the session.
- Mode picker is enabled after an instance is chosen; it shows modes for that instance's
  model.  Modes whose `width_px × height_px` and `nominal_fps` match the probe result
  (within ±1 fps) are shown at the top with a "✓ matches video" annotation.
- After a mode is chosen, the calibration indicator reads `default_intrinsics_calibration_id`
  from the mode row: green if set, orange if NULL.

#### Test criteria

1. **Schema migration**: a session DB at v8 (old schema) is opened; migration 010 runs;
   `session_cameras` no longer has `camera_mode_id` or `intrinsics_calibration_id`
   columns; `shot_videos` has both new columns (nullable).  Existing `shot_videos` rows
   have NULL in the new columns.
2. **Tracker/load_session**: `load_session.py` with a migrated session that has
   `intrinsics_calibration_id` on `shot_videos` returns the correct fx, fy, cx, cy,
   dist_coeffs for each video; `load_session.py` with a session that has NULL
   intrinsics on `shot_videos` returns no intrinsics for that video (not an exception).
3. **Wizard — picker pre-fill from probe**: add a GoPro video whose GPMF metadata
   contains a serial number matching a registry instance → camera picker pre-selects
   that instance; probe returns 3840×2160 / 120fps → mode picker pre-selects the
   matching mode.
4. **Wizard — no match**: add a video whose resolution does not match any mode for the
   selected instance → mode picker shows "(no match — select manually)"; "Next" is still
   reachable after the user manually picks a mode.
5. **Wizard — missing calibration warning**: select a mode that has no default
   calibration → orange warning icon appears; "Next" proceeds (not blocked);
   `shot_videos.intrinsics_calibration_id` is NULL in the session DB after commit.
6. **Wizard — validation**: attempt "Next" with one video having no camera instance
   selected → error message shown; wizard does not advance.
7. **Session DB self-containment**: after the wizard commits, detach the registry DB
   (move the file); open the session DB alone; `load_session.py` returns complete
   intrinsics (fx, fy, dist_coeffs, undistortion maps) without touching the registry.
   This confirms the `upsert_camera_records()` copy step worked.
8. **Python tests** (`python/tests/`):
   - `test_db_context_upsert_session_camera`: upsert twice → exactly one row in
     `session_cameras`.
   - `test_db_context_copy_camera_records`: populate a registry with a model + mode +
     instance + calibration; call `upsert_camera_records()` on an empty session;
     verify all four rows appear in the session-local tables with the same IDs.
   - `test_shot_videos_intrinsics_nullable`: create session, create shot_video with NULL
     intrinsics → `load_session` does not raise; returns None for intrinsics.

---

### Phase 3 — Inline camera creation + legacy session repair

**Outcome**: A first-time user with an empty registry can complete the full wizard
without ever opening the Camera Registry panel.  Old sessions with free-text camera IDs
can be repaired without manually editing the DB.

#### Deliverables

| # | File / location | Change |
|---|---|---|
| 3a | `python/app/setup/camera_registry.py` | Add `InlineCreateCameraDialog` (instance label, model dropdown with "New model…", serial) and `InlineCreateModelDialog` (manufacturer, model name) |
| 3b | `python/app/setup/page_shots.py` | "Create new camera…" item in the camera instance picker opens `InlineCreateCameraDialog`; on accept the new instance is selected and camera records are written to session-local tables (and registry, if connected) |
| 3c | `python/posetrak/db/cli.py` | `posetrak-db session fix-camera-refs <session.db> [--registry <registry.db>]` command |

#### `InlineCreateCameraDialog` behaviour

1. Model dropdown shows models from the session-local `camera_models` table (already
   populated from registry if connected) plus a "New model…" sentinel.
2. Choosing "New model…" opens `InlineCreateModelDialog` (manufacturer, model name,
   sensor size); on save the new model is inserted into both the session-local
   `camera_models` and the registry (if connected), and is pre-selected in the dropdown.
3. "Register & Select" creates the `camera_instances` row in session-local tables
   (and registry if connected) and dismisses the dialog.  The new instance is immediately
   selected in the `_VideoRow` picker.
4. If no model exists at all (neither in session nor registry), the model dropdown
   launches `InlineCreateModelDialog` automatically on open.

#### `posetrak-db session fix-camera-refs` CLI command

Repairs sessions created before Phase 2 where `shot_videos.camera_instance_id` is a
free-text string (e.g. "cam1") rather than a UUID.

Algorithm:
1. Collect all distinct `camera_instance_id` values in `shot_videos` that do not exist
   in `camera_instances`.
2. For each dangling string, prompt interactively:
   - "Match to existing instance" (shows list from registry)
   - "Create new instance" (asks for model + label + serial)
3. Update `shot_videos.camera_instance_id` to the matched/created UUID.
4. Upsert `session_cameras` rows for the newly linked instances.
5. Optionally prompt for `camera_mode_id` and `intrinsics_calibration_id` per
   `shot_video` if still NULL after the fix.

#### Test criteria

1. **Inline instance — happy path**: start wizard with an empty registry; add a video;
   camera picker shows "(no cameras registered)"; select "Create new camera…"; fill in
   model name + label; "Register & Select" → instance is selected; mode picker shows
   "(no modes for this model — add modes via Manage Cameras)".  Wizard can proceed to
   "Next" with a warning about missing mode.
2. **Inline instance — new model**: click "Create new camera…" when registry has models;
   choose "New model…" → `InlineCreateModelDialog` opens; save → returns to instance
   dialog with new model pre-selected; save instance → new model and instance both appear
   in the camera registry panel when opened afterward.
3. **Inline creation goes to both session and registry**: after inline creation with an
   active registry connection, open the Camera Registry panel → new model and instance
   appear there; close wizard; open a fresh session wizard → new instance appears in the
   picker.
4. **fix-camera-refs — identifies dangling IDs**: run command against a legacy session;
   output lists the dangling string IDs; interactive prompt appears for each; selecting
   "Create new instance" creates the registry row and updates `shot_videos`.
5. **fix-camera-refs — idempotent**: running the command on an already-fixed session
   reports "No dangling camera references found" without modifying the DB.
6. **fix-camera-refs — registry not supplied**: run without `--registry`; command operates
   on session-local tables only; dangling IDs are mapped to session-local instances.
