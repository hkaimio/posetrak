# Posetrak UI — Status and Roadmap

**Last updated:** 2026-05-17 (session 2)

---

## 1. Current application structure

Three separate entry points today:

| Entry point | Description |
|---|---|
| `posetrak-setup` | Qt wizard: session DB, captures, camera sync |
| `posetrak-pose` | Qt window: pose detection, stitching, finalise |
| `posetrak-ui` | Unified shell (Phase 1 in progress): session tree + content panels |
| `posetrak-db` | CLI: extrinsics, skeleton, config, run tracker, export |

`posetrak-ui` now covers: run tracker (via dialog), BVH export, skeleton manager — all
formerly CLI-only.  Extrinsics import, tracker config editing, and the merge of
`posetrak-setup`/`posetrak-pose` into the shell remain as future work.

---

## 2. Status: what is implemented

### posetrak-setup wizard (3 pages)

| Feature | Status |
|---|---|
| Session DB open / create | Done |
| Camera registry (models, modes, instances) | Done |
| Intrinsics import — HDF5 (Pose2Sim output) | Done |
| Intrinsics entry — manual (pinhole + Brown-Conrady only) | Done; fisheye model not selectable |
| Shots page — add shot, attach video files | Done |
| Camera instance and mode assignment per video | Done |
| Sync page — LED auto-sync with ROI selection | Done (pairwise graph solver; accuracy on real data unverified — see open issues) |
| Sync page — manual frame-stepping fallback | Done |
| Sync page — Export current frames of all cameras as PNG | Done |

### posetrak-pose window

| Feature | Status |
|---|---|
| Shot / sync / detection-run selectors | Done |
| Time-range marking (Mark Start / Mark End) | Done |
| Background detection pipeline (YOLO + RTMPose → DB) | Done |
| Frame range via SyncTable (piecewise-linear, not single-anchor fps extrapolation) | Done |
| Stitcher timeline with person assignment | Done |
| Assignment conflict detection + resolution dialog | Done |
| Frame view with skeleton overlay | Done |
| Person preview panel (live bbox crop) | Done |
| Finalise → one `pose_observation_sequences` row per person | Done |
| Assignment state persisted via `detection_track_assignments` (survives restart) | Done |
| Re-finalize replaces sequences (cascade delete + recreate) | Done |
| `data_changed` signal → main window tree auto-reloads after finalize | Done |
| Detection pipeline stores JPEG person crops in `frame_cache_entries` | Done |
| Confidence sparkline | Postponed (Phase C) |
| Manual bbox correction + partial re-run | Postponed (Phase C) |

### posetrak-ui shell (Phase 1, in progress)

| Feature | Status |
|---|---|
| Shell window: File menu, registry auto-open, recent files | Done |
| Session tree: Capture → Trial → Detection run → Person → Tracking run | Done |
| Context menus: rename trial, rename person | Done |
| Skeleton manager dialog (import YAML, rename, delete, scale dimensions) | Done |
| CapturePanel: capture metadata, video list, open pose extraction | Done |
| TrialPanel: trial metadata | Done |
| DetectionRunPanel: run metadata, open pose extraction | Done |
| PersonPanel: person info, tracking runs list with stats | Done |
| PersonPanel: Run tracker dialog (pre-selects sequence, person 0) | Done |
| PersonPanel: Export BVH via `export_bvh.py` subprocess | Done |
| PersonPanel: Delete tracking run (cascade) | Done |
| PersonPanel: multi-camera crop grid with time scrubber | Done |
| PersonPanel: pose_observations skeleton overlay on crop grid | Done |
| PersonPanel: 3D view placeholder cell | Done |
| TrackingRunPanel: run stats summary | Done |

### Known issues in posetrak-pose

#### High-frame-rate video decode (Pixel 9 / 120 fps cameras) — FIXED 2026-05-17

Two bugs were fixed in `_iter_frames_av` (`detection_pipeline.py`):

1. **Frame range** (fixed 2026-05-16): the range calculation now uses `SyncTable.lookup()`
   (piecewise-linear interpolation through all sync points) instead of single-anchor fps
   extrapolation.  Frame numbers come from the sync configuration; `actual_fps` is no
   longer used for range math.

2. **Decode cadence / wrong frames** (fixed 2026-05-17): PyAV `pts`-derived `frame_idx`
   jumped by 4 for the Pixel 9 (120 fps content in a 30 fps container) — only 25 % of
   frames were yielded.  Fix: `_iter_frames_av` now reads `container_fps` from
   `stream.average_rate` (the container's own metadata, independent of `actual_fps`)
   and uses sequential counting from `first_frame` after the initial pts-based seek.
   `actual_fps` is no longer referenced inside `_iter_frames_av` at all.

#### pose-extraction window UI structure

The current window conflates four concerns in one screen:

1. **Navigation** — shot, sync config, detection run selectors at the top.
2. **Trial definition** — "Mark Start / Mark End" sets the time range for a new run.
3. **Run parameters** — detector model, confidence threshold, "Run detection" button.
4. **Results review** — stitcher timeline + frame view for the selected detection run.

This makes it unclear which detection run the stitcher is showing, and mixes the
"what to detect" (trial concept) with the "how to detect" (run parameters).  See
section 4.1 for the target architecture.  The long-term fix is part of Phase 3
(merge into single app with session tree), but the following interim improvements
are worth making before then:

- The stitcher should always label which detection run it is showing.
- "Mark Start / Mark End" should be labelled as defining a *trial*, not a run.
- The shot/sync selectors should be read-only once a detection run is selected
  (they are navigation, not parameters).

### Backend readiness for distorted-pixel pipeline

`finalise.py` writes `pixels_are_undistorted = 0`; the C++ tracker reads this flag and
calls `Camera::undistort()` on every observation.  The tracker implements both
Brown-Conrady and fisheye (kb4) iterative inversion.

For cameras calibrated via HDF5 (Pose2Sim), `K_original` (distorted) and `K_new`
(optimal undistorted) are both stored and the tracker's undistortion is fully correct.

**Gap:** The manual calibration UI entry hardcodes `radtan` and does not compute or store
`matrix_original`.  For cameras with mild distortion this is acceptable (the tracker
remains self-consistent, just using a sub-optimal `K_new`).  For true fisheye lenses
the distortion model must be set correctly; intrinsics for such cameras must currently
come from the HDF5 import path, not the manual entry form.

---

## 3. What is still needed for a full end-to-end GUI run

The steps below are today CLI-only.  Each is a self-contained feature that can be added
independently.

### 3.1 Extrinsics import

**What:** A UI action to import a Pose2Sim `cameras.toml` (or HDF5 equivalent) and
create an `extrinsic_calibrations` row linked to the current session.

**Backend:** `import_extrinsics.py` / `posetrak-db extrinsics import` — fully
implemented.  The UI is a single file-picker dialog plus a camera-matching step (same
pattern as the existing intrinsics import).

**Note:** This is a placeholder.  A new extrinsics calibration method (wand-based or
similar) is planned; when that lands the import dialog can remain as a fallback.

### 3.2 Skeleton setup

**What:** A wizard page (or sidebar panel) for choosing the skeleton a tracking run
will use.

**Two entry paths:**

1. **Pick from registry** — select an existing skeleton stored in the DB (previously
   imported or scaled).
2. **Import YAML** — browse for a skeleton YAML file; wraps
   `posetrak-db skeleton import`.

**Rough scaling before the first run:**
The UKF works best when bone lengths are at least roughly right from the start.
Before the first tracking run the user should be able to set a simple scale factor
(or per-segment overrides: shin, femur, upper arm, lower arm, torso height, shoulder
width, head) applied to the template skeleton.  The UI pre-fills these from rough
physical measurements or body-height estimate.  The scaled skeleton is saved to the
DB and used for the run.

After the short calibration tracking run, if the template was "roughly right",
outlier rejection will have produced a clean set of inlier observations.  Those
inliers can be triangulated to measure actual joint-to-joint distances, and the
skeleton can be re-scaled to match — this is what `body_measurements.py` (Marimo) and
`posetrak-db skeleton scale` do today.  Integrating this into the Qt UI removes the
need to open the notebook.

**Final scaling workflow (integrated):**
1. Pick a tracking run from the current session.
2. The UI triangulates inlier observations, computes per-segment medians, and shows
   them against the template values (the table from `body_measurements.py`).
3. User reviews and optionally overrides individual measurements.
4. "Save scaled skeleton" writes a new skeleton row to the DB and optionally sets it
   as the default for future runs in this session.

**Backend:** `scale_skeleton.py`, `manage_skeleton.py` — fully implemented.

### 3.3 Tracker config (UKF parameters)

**What:** A form for creating or editing a `tracker_configs` row: UKF noise parameters
(process noise, measurement noise, initial covariance), outlier rejection threshold,
joint limit enforcement, and which skeleton + pose sequence to use.

**Backend:** `manage_config.py` / `posetrak-db config create|edit` — fully
implemented.

### 3.4 Run tracker

**What:** A "Run Tracker" button that:
1. Resolves the `posetrak` binary path (configurable, defaults to
   `optbuild/cli/posetrak`).
2. Invokes `posetrak track --session-db … --config-id …` in a `QThread`.
3. Streams stdout progress to a progress bar / log panel.
4. On completion writes the `tracking_runs` row ID and offers to proceed to results.

**Backend:** The C++ CLI, `tracking_runs` schema, and `run_project.py` (batch script)
all exist.  This is purely a Qt wrapper.

### 3.5 Skeleton manager dialog

**What:** A standalone `SkeletonManagerDialog` accessible from the main window menu
(Tools → Manage skeletons…).  Replaces the skeleton-management UI that currently lives
only inside the New Capture wizard.

**Operations:**
- List skeletons in the session DB (name, parent, joint count, created date).
- **Import YAML** — browse for a `.yaml` file; validates the `joints` key; imports via
  `manage_skeleton.import_skeleton_str`.
- **Rename** selected skeleton — updates the `name` column in place.
- **Delete** selected skeleton — guarded by a confirmation dialog.
- **Scale dimensions…** — opens `SkeletonScaleDialog` (extracted from the wizard's
  `SkeletonSetupWidget` scaling panel) for the selected skeleton.

**Backend:** `manage_skeleton.py` (import, rename, delete) and `scale_skeleton.py`
(scaling) — all implemented.

**Note:** Skeletons are stored in the session DB (the `skeletons` table), not the
registry.  The dialog works with the session connection only; registry copy is deferred.

### 3.6 Person panel

**What:** The main panel shown when a *Person* node is selected in the session tree.
Replaces the thin `PersonTrackPanel` and consolidates everything needed to run and
iterate on the tracker for one person.

**Layout (top to bottom):**

1. **Header** — person name, time range, observation count.
2. **Skeleton & parameters** — always visible:
   - Skeleton picker (dropdown from session DB `skeletons`).
   - "Adjust dimensions…" button → opens `SkeletonScaleDialog`.
   - Collapsible "Tracker parameters" section (UKF noise values, fps, outlier threshold).
3. **Run control** — "Run tracker" button; progress bar + log output while running;
   summary stats (inlier %, avg observations) after completion.
4. **Tracking runs list** — previous runs for this person (skeleton, timestamp, stats);
   selecting one makes it the active result shown in the views below.
5. **Frame display strip** — time scrubber on the global timeline; per-camera images
   showing detections at the scrubbed time (reuses `FrameViewWidget`).
6. **3D skeleton view** *(deferred, Phase 5)* — stick-figure skeleton drawn from
   `tracking_results` at the scrubbed time; placeholder until Phase 5.

**One person at a time:** the panel runs a single tracker invocation with a specific
person_id.  Multi-person orchestration is deferred.

**Sequence pre-selection:** the panel opens with its `sequence_id` already resolved from
the tree selection, so the user never picks it from a combo.

### 3.7 Results viewer and BVH export

**What:** A read-only panel showing the selected tracking run:
- Summary stats (frames tracked, inlier %, per-camera breakdown).
- A playback view overlaying the tracked skeleton onto the original video (reuses
  `FrameViewWidget` from the pose window with a different data source).
- "Export BVH" button wrapping `export_bvh.py`.

**Backend:** `export_bvh.py` (standalone script), tracking results in DB — both exist.
The overlay rendering for tracked results is currently only in `visualize_tracking.py`
(Rerun-based); a Qt equivalent needs to be written or the Rerun viewer launched as a
subprocess.

**Note:** Summary stats and BVH export are included in the Person panel (section 3.6)
since they belong to the same workflow.  This section covers only the video overlay
(Phase 5 work).

---

## 4. Future enhancements (after e2e is complete)

### 4.1 Merge setup wizard and pose window into one application

#### Terminology and domain model

The current `shots` table name conflates two distinct concepts.  New terminology:

| Old term | New term | DB table | Meaning |
|---|---|---|---|
| Shot | **Capture** | `captures` (rename from `shots`) | One continuous camera recording: cameras on → off.  Owns the video files and sync config. |
| — | **Trial** | `trials` (new) | A named, bounded time window within a capture: one technique, one attempt.  The user-facing unit of analysis ("shomenuchi shihonage take 1"). |
| Detection run | **Detection run** | `detection_runs` | Technical execution of pose detection over a trial's time window.  Multiple detection runs per trial are allowed (e.g. different model or confidence threshold). |
| Observation sequence | **Person track** | `pose_observation_sequences` | One performer's finalised pose data from a detection run. |
| Tracking run | **Tracking run** | `tracking_runs` | One 3D tracking execution on a person track. |

The `trials` table is a thin concept layer:

```sql
CREATE TABLE trials (
    id           TEXT PRIMARY KEY,
    capture_id   TEXT NOT NULL REFERENCES captures(id),
    name         TEXT,
    time_start_s REAL,
    time_end_s   REAL,
    notes        TEXT
);
```

`detection_runs` gains a `trial_id TEXT REFERENCES trials(id)` column (nullable:
existing runs without a trial remain valid).

#### Target architecture

A single `posetrak-ui` shell replaces both `posetrak-setup` and `posetrak-pose`.

```
┌──────────────────────────────────────────────────────────────────────┐
│ File   Session   Cameras   Help                       [status bar]   │
├──────────────────┬───────────────────────────────────────────────────┤
│ Session tree     │  Main panel (stacked/content area)                │
│                  │                                                    │
│ ▼ Capture "morning"         • Nothing selected: welcome / empty      │
│   ▼ Trial "shomen take 1"   • Capture selected: capture detail,      │
│     ▼ Detection [yolo11x]     video files, sync status               │
│       Harri                 • Trial selected: trial metadata,        │
│         Tracking run 1        list of detection runs                 │
│       Sensei                • Detection run selected:                │
│     Detection [yolo8n]        PoseExtractionWindow panel             │
│   ▼ Trial "shomen take 2"   • Person track selected:                 │
│     ▼ Detection [yolo11x]     finalised track view (same panel)      │
│       Harri                 • Tracking run selected:                 │
│         Tracking run 1        tracking summary (Phase 5: visualizer) │
└──────────────────┴───────────────────────────────────────────────────┘
```

Tree hierarchy:
- **Capture** (label, sync status indicator)
  - **Trial** (user name; time range shown as subtitle)
    - **Detection run** (model name + timestamp)
      - **Person track** (performer name; one per `pose_observation_sequences` row)
        - **Tracking run** (skeleton name + `ran_at` + notes tooltip)

#### Session DB and registry

- On startup the app auto-opens `~/.posetrak/registry.db` (creating it if absent).
  No prompt unless the user wants a custom location ("File → Change registry…").
- Session DB is opened via **File → New session database…** / **File → Open session
  database…** (or equivalent toolbar buttons).  No wizard page for this step.
- Recent databases listed in File menu for quick re-open.

#### Capture creation: wizard stays, session-open page removed

The wizard's remaining pages (videos → sync → extrinsics → skeleton) are
capture-specific and stay intact.  The old "Session" page (page 1) moves out: the
session DB is already open before the wizard runs.

Launching the wizard: **Session → New Capture…** (menu or toolbar button).
The wizard creates the capture, attaches video files, sets up sync and extrinsics,
and picks a skeleton.  When it finishes the new capture appears in the tree.

Trials are created inside the pose extraction panel (by marking a time range and
naming it), or via "New Trial…" from the capture's context menu.

#### Context-menu actions per tree level

| Selected item | Context-menu actions |
|---|---|
| Capture | New trial… / Set up sync… / Import extrinsics… / Edit metadata… / Delete capture |
| Trial | Run detection… / Edit name & notes / Delete trial |
| Detection run | Assign persons / Finalise → person tracks / Delete run |
| Person track | Rename / Run tracker… / Delete track |
| Tracking run | View results (Phase 5) / Export BVH… / Edit notes / Delete run |

#### Schema changes required (migrations)

1. Rename `shots` → `captures`; update all FK references (`shot_id` columns in
   `detection_runs`, `shot_videos`, `sync_configs`, `extrinsic_calibrations`,
   `pose_observation_sequences`, `tracking_runs`).
2. New `trials` table (see above).
3. Add `trial_id TEXT REFERENCES trials(id)` to `detection_runs`.
4. Add `name TEXT` to `pose_observation_sequences` (person track display name).
5. Add `notes TEXT` to `tracking_runs`.

All changes are additive or renames — no existing data is lost.

#### Relation to existing code

The two existing windows share no global state (each opens its own DB connection).
The merge is mainly navigation and layout: `SyncPage` and `PoseExtractionWindow`
widget trees embed as-is into the content area, driven by the tree selection.
The terminology change (shots → captures, new trials level) affects Python query
strings and UI labels but not algorithmic logic.

### 4.2 Intrinsics calibration UI

**What:** An in-app calibration wizard:
1. Select camera mode.
2. Record or load a set of calibration frames (checkerboard or ChArUco).
3. Run OpenCV calibration; display reprojection error and per-frame residuals.
4. Accept → writes `intrinsics_calibrations` row and generates undistortion maps.

This removes the dependency on Pose2Sim / external HDF5 files for intrinsics.

The UI should also fix the current gap in the manual entry form:
- Add a distortion model selector (Brown-Conrady / Fisheye).
- Compute and store `matrix_original` and the optimal `K_new` when distortion
  coefficients are provided, so the C++ tracker's undistortion is fully correct.

### 4.3 Extrinsics calibration UI (placeholder)

A wand-based or structured-light extrinsics calibration method is planned; the
specific algorithm is TBD.  When implemented it replaces the import-from-TOML step
(3.1) with an in-app capture + solve workflow.  For now, import from Pose2Sim TOML
(section 3.1) is the production path.

---

## 5. Implementation phasing

Agreed order: Phase 1 → Phase 3 → Phase 5 → Phase 2 → Phase 4.
Phase 2 (skeleton scaling) fits naturally inside the merged shell; Phase 4 (calibration
UIs) is deferred as it is self-contained and not on the critical path.

### Phase 1 — Complete the e2e pipeline (tracker workflow unblocked)

Priority: a practitioner can go from finalised person tracks to a tracker result without
touching the CLI.  The tree restructure (formerly Phase 3 work) is pulled forward because
the new Person panel needs to live in the right place from the start.

#### Step 1 — Skeleton manager dialog (3.5)

`SkeletonManagerDialog`: standalone dialog (Tools → Manage skeletons…) wrapping
`SkeletonSetupWidget` and adding rename/delete.  Also extracts the scaling panel into a
reusable `SkeletonScaleDialog`.

#### Step 2 — Session tree restructure (partial pull-forward from Phase 3 T3.3)

Surface the existing `trials` and `pose_observation_sequences` data in the session tree
with the correct hierarchy:

```
Capture → Trial → Detection Run → Person → Tracking Run
```

- Add Trial nodes (between Capture and Detection Run).  Detection runs with
  `trial_id = NULL` appear under a grey "(no trial)" grouping.
- Rename "Person track" → "Person" at tree and panel level.
- Add Tracking Run nodes under each Person node.
- Context menus: Trial (rename), Person (rename), Tracking Run (view stats, export BVH,
  delete).

No schema changes required — all tables exist (v15 and later).

#### Step 3 — Person panel: tracker control (3.6, parts 1–4)

New `PersonPanel` replacing `PersonTrackPanel`:
- Skeleton picker pre-wired to the session's skeleton list; "Adjust dimensions…" opens
  `SkeletonScaleDialog`.
- Collapsible tracker parameters form (reuses `RunTrackerWidget` internals).
- "Run tracker" button with progress bar + log.  Sequence is pre-selected from the
  tree — no combo needed.
- Tracking runs list (previous runs; click to select as active).

#### Step 4 — Person panel: frame display strip (3.6, parts 5–6) — Done

- Time scrubber spanning only the sequence time range (not the full capture).
- All cameras simultaneously in a grid (one cell per camera + one 3D placeholder cell).
- Camera name label at top of each cell; empty cell if no detection for this person at that frame.
- Crops served from `frame_cache_entries` (JPEG bbox crops stored during detection, keyed by
  detection_run_id so multiple runs on the same capture coexist).
- `pose_observations` skeleton overlaid at display time (not baked into stored JPEG).
- 3D skeleton view: placeholder cell reserved.

Note: `frame_cache_entries` PK was extended to include `detection_run_id` (schema v17) to
support this.  Previously the PK was `(shot_video_id, frame_idx, cache_type, track_id,
region_type, width_px)`; `detection_run_id TEXT NOT NULL DEFAULT ''` was added so PERSON_CROP
entries from different detection runs coexist, while non-crop entries use `''`.

| # | Task | Effort | Status |
|---|---|---|---|
| T1.0 | Skeleton manager dialog + SkeletonScaleDialog extraction | S | Done |
| T1.1a | Tree: Trial nodes, Person rename, Tracking Run nodes | S | Done |
| T1.1b | PersonPanel: tracker control, tracking runs list, BVH export | M | Done |
| T1.1c | PersonPanel: multi-camera crop grid + time scrubber | M | Done |
| T1.2 | Extrinsics import dialog (3.1) | S | — |
| T1.3 | Tracker config form improvements (3.3) | S | — |
| T1.4 | Fix manual calibration: distortion model selector + `matrix_original` (4.2 partial) | S | — |

`S` = 1–2 days, `M` = 2–4 days.

**Deliverable:** A practitioner can go from raw videos to BVH using only the Qt apps;
the CLI remains available but is no longer required for the tracker workflow.

**Current schema version:** v17 (`SESSION_SCHEMA_VERSION = 17`).  Migration chain:
v14 (sync anchors) → v15 (detection_track_assignments) → v16 (detection_crops, superseded) →
v17 (frame_cache_entries gains detection_run_id in PK; detection_crops removed).

### Phase 2 — Skeleton scaling from tracking run (integrated)

Integrates the `body_measurements.py` notebook workflow into the Qt UI.  Depends on
Phase 1 (needs at least one tracking run in the DB).

| # | Task | Effort |
|---|---|---|
| T2.1 | Post-run body measurements panel (triangulate inliers, show per-segment medians) | M |
| T2.2 | Override fields + "Save scaled skeleton" action | S |
| T2.3 | Wire scaled skeleton as default for next run in session | S |

**Deliverable:** Full skeleton sizing workflow without opening a browser or CLI.

### Phase 3 — Merge into single application

Combines the two Qt entry points into one unified shell.  Design spec in section 4.1.

| # | Task | Effort |
|---|---|---|
| T3.1 | Schema migrations: rename `shots`→`captures`; new `trials` table; `trial_id` on `detection_runs`; `name` on `pose_observation_sequences`; `notes` on `tracking_runs` | M |
| T3.2 | Shell window: File menu (open/new session DB), registry auto-open, recent files | S |
| T3.3 | Session tree widget (captures → trials → detection runs → person tracks → tracking runs) with context menus | M |
| T3.4 | Stacked content area: wire tree selection to embed existing widgets (PoseExtractionWindow, SyncPage, ExtrinsicsImportDialog) | M |
| T3.5 | Strip session-open page from wizard; launch wizard from "New Capture…" action | S |
| T3.6 | Session persistence — remember last-opened DB, restore tree selection | S |

**Deliverable:** `posetrak-setup` and `posetrak-pose` replaced by a single `posetrak-ui`
entry point; old entry points kept as thin aliases for backwards compatibility.

### Phase 3.5 — Multi-clip sync redesign

Motivation: captures where cameras have non-overlapping time ranges, or where the sync LED is not visible in all cameras, cannot be synced with the single-reference-moment model. This phase replaces it with a graph-based model: pairwise anchor observations feed a BFS solver that produces the existing `sync_points` output consumed by the tracker.

**New DB tables** (input layer, schema v14):
- `sync_anchors (id, shot_id, notes)` — one row per shared real-world event
- `sync_anchor_observations (id, sync_anchor_id, shot_video_id, video_frame, subframe)` — per-video frame number; `subframe` carries LED peak sub-frame precision (0.0 for manual)

**Output layer unchanged:** `sync_configs` + `sync_points` with piecewise-linear `timestamp_s`

| # | Task | Effort | Status |
|---|---|---|---|
| T_S1 | DB migration 013: `sync_anchors` + `sync_anchor_observations` tables | S | Done |
| T_S2 | `db.py`: bump `SESSION_SCHEMA_VERSION` → 14; add `_migrate_session_v13_to_v14` | S | Done |
| T_S3 | `db_context.py`: anchor CRUD helpers + `SyncAnchorObservation` dataclass | M | — |
| T_S4 | New `sync_solver.py`: graph BFS solver → `list[SyncPoint]` | M | — |
| T_S5 | New `pair_scrubber.py`: two-camera side-by-side widget (ref + target) | L | — |
| T_S6 | Redesigned `SyncPage` / new `SyncWidget` + `SyncDialog` | L | — |
| T_S7 | `_LedSyncDialog`: per-camera LED checkboxes; subset ≥ 2; route accept through solver | M | — |
| T_S8 | Extract `FrameReader` from `page_sync.py` → `video_reader.py` | S | Done |
| T_S9 | Wire "Set up sync…" button in `CapturePanel` | S | — |
| T_S10 | Tests: `DBContext` anchor CRUD | S | — |
| T_S11 | Tests: `sync_solver.py` | M | — |
| T_S12 | Tests: `PairScrubber` | S | — |
| T_S13 | Tests: revised `SyncPage` / `SyncWidget` | M | — |

**Dependency order:** T_S1 → T_S2 → T_S3 → {T_S4, T_S5, T_S10} → T_S6 → {T_S7, T_S9, T_S12, T_S13}
(T_S8 is independent; T_S5 also needs T_S8.)

**Key risks:**
- `_on_accept` in `_LedSyncDialog` must route through solver (T_S7); needs `event_frames` added to `CameraSyncResult` in `led_sync.py`
- `test_page_sync.py` tests many removed internals — substantial rewrite in T_S13
- `SyncPage` must remain a valid `QWizardPage` (wrap `SyncWidget`)
- Any test hard-coding `SESSION_SCHEMA_VERSION = 13` will break after T_S2

### Phase 4 — Calibration UIs

| # | Task | Effort |
|---|---|---|
| T4.1 | Intrinsics calibration wizard (checkerboard / ChArUco, OpenCV) | L |
| T4.2 | Extrinsics calibration UI (TBD algorithm; placeholder) | L |

`L` = 1–2 weeks.

**Deliverable:** Self-contained calibration without Pose2Sim or external files.

### Phase 5 — Results viewer with video overlay

| # | Task | Effort |
|---|---|---|
| T5.1 | Qt-native skeleton overlay on original video (frame-by-frame, reusing FrameViewWidget) | L |
| T5.2 | Replace / supplement Rerun-based `visualize_tracking.py` | M |
| T5.3 | Joint angle plots alongside video | M |

**Deliverable:** In-app visual validation of tracking results.
