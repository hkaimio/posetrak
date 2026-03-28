# Phase 2 Implementation Plan: Setup Application

Implementation steps for the setup wizard described in `capture-pipeline-architecture.md`.
Steps within the same group have no inter-group dependencies and can be worked in parallel.

---

## Group A — Schema and project skeleton

No UI, no inter-dependencies. Start here.

### A1. DB schema migration

- Add `person_detections`, `person_tracks`, `frame_cache_entries` tables to the SQL schema files in `db/`.
- Update `create_session()` in `posetrak/db/db.py` to include the new tables.
- Write a migration script (or bump `PRAGMA user_version`) for existing session DBs.
- Add `person_detections` to the pytest fixture in `test_posetrak_db.py`.

### A2. Project skeleton

- Create `python/app/setup/` directory with empty `__init__.py` files.
- Add `[project.scripts] posetrak-setup = "posetrak.app.setup.main:main"` to `pyproject.toml`.
- Write stub `main.py` (creates `QApplication`, shows a placeholder window, exits cleanly).
- Verify `uv run posetrak-setup` launches without import errors.

---

## Group B — Foundation components

Depend on A. Can be implemented in parallel with each other.

### B1. `DBContext`

- Define typed `dataclass`/`NamedTuple` return types: `ShotVideoInfo`, `SyncPoint`, `ExtrinsicEntry`.
- Implement `DBContext` with methods: `create_shot`, `create_shot_video`, `write_sync_config`, `write_extrinsics`, `get_shot_videos`, `get_active_sync`.
- Unit-test each method against an in-memory SQLite DB.

### B2. `FrameCache`

- Define `CacheType` enum and `CacheKey` dataclass (with `track_id`, `region_type`, `width_px`/`height_px`).
- Implement `VideoCapture` pool with sequential-read optimisation (`_last_frame` tracking avoids unnecessary seeks).
- Implement in-memory LRU first — get `get()` working without DB persistence.
- Add DB persistence: read from `frame_cache_entries` on miss; write asynchronously via a `queue.Queue` drained by a daemon thread.
- Unit tests: sequential seek does not call `CAP_PROP_POS_FRAMES`; random seek does; LRU eviction works correctly.

### B3. `Overlay` protocol + concrete overlay stubs

- Define the `Overlay` `Protocol` in `overlay.py` with typed `paint()`, `mouse_press()`, `mouse_move()`, `mouse_release()` signatures.
- Implement `ROIDrawOverlay` (rubber-band rectangle): paint + full mouse event handling.
- Implement `SyncAnchorOverlay` (vertical tick mark at a given frame): paint only.
- Leave `AnnotationPointOverlay` and `ReprojectionOverlay` as stubs (needed in D4) — `pass` in all methods.

### B4. `BackgroundJob` base class

- `QThread` subclass with signals: `progress = Signal(int, str)`, `finished = Signal(object)`, `error = Signal(str)`.
- Standard `try/except` wrapper around `run()` that emits `error` on any unhandled exception.

---

## Group C — `MultiVideoScrubber`

Depends on B2 and B3. Implement C1 before C2.

### C1. `CameraCell` widget

- `QLabel` subclass that displays a `np.ndarray` frame (convert via `QImage`).
- Accepts a `list[Overlay]`; calls `paint()` on each after frame render.
- Forwards `mousePressEvent` / `mouseMoveEvent` / `mouseReleaseEvent` to overlays in reverse-z order, mapping display coordinates to video-frame coordinates.
- Draws a focus border when the cell is focused.

### C2. `MultiVideoScrubber` widget

- Grid layout of `CameraCell`s.
- **Synced mode**: `seek_synced(timestamp_s)` updates all cells via `SyncTable.lookup()`.
- **Independent mode**: `seek_camera(cell_idx, frame_idx)` updates one cell; focused cell receives keyboard input. Mode is active when no sync table is loaded, or when the user clicks/tabs to a specific cell.
- Keyboard shortcuts: `←`/`→` (±1 frame), `Shift+←`/`Shift+→` (±10 frames), `Space` (play/pause), `Home`/`End`.
- `reload_sync(sync_table)` slot: updates sync source and immediately re-renders all cells.
- Integration test: open two video files, verify synced seek moves both cells and independent mode moves only the focused one.

---

## Group D — Wizard pages

All pages depend on B1 and B4. Pages using the scrubber additionally depend on C2. Pages can largely be implemented in parallel once their dependencies are met.

### D1. Page 0 — Open / create session

*Depends on: B1*

- `QWizardPage` with a file-open dialog.
- **Open existing**: validates DB schema, displays summary (shot count, camera count, sync state).
- **Create new**: name + directory picker → calls `create_session()`.
- `validatePage()` stores the DB path on the wizard so subsequent pages can access it.

### D2. Page 1 — Add videos

*Depends on: B1, B2*

- Lists `session_cameras` rows; one file-path row per camera with a "Browse" button.
- On file selection: probe with `cv2.VideoCapture` to read `actual_fps`, `frame_count`, `width`, `height`.
- **Shot boundary widget**: thumbnail strip for the reference camera (populated by a background job using `FrameCache.get(CacheKey(..., THUMB))`). Start/end drag handles define the shot time range. Multiple shots can be defined in one pass.
- `validatePage()`: writes `Shot` and `ShotVideo` rows via `DBContext`.

### D3. Page 2 — Sync (rough + LED, single page)

*Depends on: B1, B3, B4, C2*

Split into sub-tasks; implement in order:

**D3a.** Embed `MultiVideoScrubber` in independent mode (no sync table yet). Wire focus-click and keyboard navigation. Verify that cameras scroll independently.

**D3b.** Rough sync panel:
- "Set anchor" button records current frame for the focused camera.
- `SyncAnchorOverlay` marks the anchor on each cell's timeline.
- "Apply rough sync": compute frame offsets, write `SyncConfig(method="manual-rough")` via `DBContext`, call `scrubber.reload_sync()`. Scrubber switches to synced mode.

**D3c.** LED sync panel:
- "Draw LED ROI" toggle activates `ROIDrawOverlay` on all cells; user draws one rectangle per camera.
- "Run LED sync" launches `LedSyncJob` (B4 subclass):
  - Reads each camera's video **sequentially** (no random seek).
  - Crops the ROI patch each frame and computes `patch.mean()` — avoids full-frame processing cost.
  - Finds brightness peaks; cross-correlates against reference camera.
  - Emits `progress(percent, "Analyzing camera N/M · frame F of T")`.
  - Returns `dict[cam_label, list[SyncPoint]]`.
- Single progress bar in the panel; no per-camera dialogs.
- On completion: per-camera quality metrics (peak SNR, correlation score) shown inline. Anchor dots appear on the scrubber timeline strip.
- "Accept": writes `SyncConfig(method="led-auto")` with all sync points; reloads scrubber.

**D3d.** Failure handling: if correlation score is below threshold, show an inline warning with suggested actions (re-draw ROI, fall back to manual-rough). No dialog boxes.

### D4. Page 3 — Extrinsics annotation

*Depends on: B1, B3, C2*

Split into sub-tasks; implement in order:

**D4a.** `AnnotationPointOverlay` — zoom-to-refine interaction:
- In overview mode: click places a labelled dot; cell switches to 1:1 zoom with the clicked pixel pinned under the cursor via `QTransform`.
- With mouse button held: moving adjusts the dot at full pixel resolution.
- On mouse release: commit coordinate in original video-pixel space; return to overview.
- Click existing dot: re-enter zoom-refine for that point.
- Right-click dot: delete it.

**D4b.** `ReprojectionOverlay`: draw circles at reprojected positions, residual lines from annotation to reprojection, per-camera RMS error label.

**D4c.** Control point set editor: `QTableWidget` with rows `{name, x_m, y_m, z_m}` or a JSON file loader.

**D4d.** "Compute extrinsics": run `cv2.solvePnPRansac` per camera; activate `ReprojectionOverlay` with results.

**D4e.** "Bundle adjustment" (optional): SciPy `least_squares` over all cameras jointly; update overlay with refined results. Guard the import so the button is disabled if SciPy is not installed.

**D4f.** "Accept": write `ExtrinsicCalibration` + `ExtrinsicEntry` rows via `DBContext`.

---

## Group E — Integration and polish

Depends on all of D.

### E1. Wire the wizard

- `wizard.py`: instantiate `DBContext`, add all pages, make `DBContext` accessible to pages via `self.wizard().db_context`.
- "Back" button: implement `cleanupPage()` on pages that write to DB to roll back uncommitted changes.
- Unhandled exceptions from `DBContext` methods surface as `QMessageBox` errors rather than crashes.

### E2. End-to-end test with a real session

- Run the full wizard on an existing session from `/mnt/d/mocap/`.
- Verify: expected DB rows exist after each page; sync table round-trips correctly; computed extrinsics match the existing calibration.

### E3. Entry point

- `main.py`: parse optional positional `session.db` argument; if provided, skip Page 0 and open that DB directly.
- Verify `uv run posetrak-setup` and `uv run posetrak-setup /path/to/session.db` both work from repo root.

---

## Dependency graph

```
A1 ──► B1 ──► D1 ──────────────────────────────────────────► E1
       B1 ──► D2 ──────────────────────────────────────────► E1
A2 ────────────────────────────────────────────────────────► E3

A1 ──► B2 ──► D2
       B2 ──► C1 ──► C2 ──► D3a ──► D3b ──► D3c ──► D3d ──► E1
                     C2 ──► D4a ──► D4b ──► D4c
                                            D4c ──► D4d ──► D4e ──► D4f ──► E1
A1 ──► B3 ──► C1
       B3 ──► D4a

A1 ──► B4 ──► D3c

                                                              E1 ──► E2 ──► E3
```

## Algorithmic pieces that can be prototyped standalone

Two sub-tasks are algorithmic-heavy and worth prototyping as standalone scripts before wiring into the wizard:

- **`LedSyncJob` (D3c)**: write a command-line script that takes a video path and an ROI rectangle, prints the brightness curve and detected peaks. Validates the algorithm in isolation before any Qt integration.
- **Extrinsics PnP + bundle adjustment (D4d/D4e)**: write a script that takes a JSON of `{cam: [(point_name, x_px, y_px)]}` annotations and a control point JSON, prints R/t per camera and reprojection errors. Can be tested against the existing VIA annotations from prior sessions.
