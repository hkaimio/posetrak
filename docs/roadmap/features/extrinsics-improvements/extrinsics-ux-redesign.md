# Extrinsics calibration UX — current design

**Status: implemented and landed.** This document describes the
`ExtrinsicsAutoCalibDialog` UI as it exists today, after nine rounds of
live-testing-driven UX cleanup (2026-08-13 through 2026-08-15, "UX Phases
1–7" plus several unnumbered follow-up rounds). It is the authoritative
current-state spec, not a historical plan — see
[status.md](status.md) for the dated, blow-by-blow change history and the
reasoning behind individual decisions. A first real live-testing pass of
the whole assembled UI (load rig, detect markers, solve, save, load into
another capture) passed on 2026-08-15; broader testing is still ongoing.

## Why this document exists

Phases 1–9 of `extrinsics-improvements-design.md` (the feature's own
numbering, distinct from this document's "UX Phase" numbering below) built
out real capability — ArUco/ChArUco detection, a portable calibration rig,
scattered scene tags, named marker groups. Each landed as an *addition* to
one dialog, driven by whatever the previous round of live testing
surfaced. After nine rounds of "this one thing is confusing, fix it," the
accumulated result no longer clearly communicated what it was doing, so
work paused for a UX design pass (this document, originally a proposal)
before further feature additions. That proposal has since been
implemented and iterated on through several more live-testing rounds well
beyond its original scope — this rewrite replaces the "proposed
restructuring" framing with a description of what actually got built.

## Entry points

`CapturePanel`'s **`Extrinsics…`** button (`content_panels.py`) opens
`ExtrinsicsStatusDialog` — a per-camera solved/not-solved summary table
plus `Calibrate…` / `Import TOML…` / `Close`. The button's own text/
tooltip reflects solved state (e.g. `Extrinsics ✓ (6/6)` vs.
`Extrinsics (not set)`), refreshed on panel build and whenever the status
dialog closes (`CapturePanel._refresh_extrinsics()`, mirroring the
pre-existing `_refresh_sync()` pattern).

- **`Calibrate…`** opens `ExtrinsicsAutoCalibDialog` — the real GUI-native
  calibration workflow, described below.
- **`Import TOML…`** opens `ExtrinsicsImportWidget` — a purely TOML-import
  screen (file browse, camera-assignment table, Import), unrelated to the
  rig/marker workflow.

The legacy `Auto-calibrate (image folder)…` path (still-frame PNG folder
input, predating direct video scrubbing) has been removed entirely, along
with `_on_auto_calibrate`/`_load_states_from_images`.

## `ExtrinsicsAutoCalibDialog` layout

Top to bottom: a camera grid (one thumbnail per camera, with per-camera
"Detect ArUco"/"Detect ChArUco"/"Detect Rig" buttons and click/drag
control-point placement) alongside a fixed-width sidebar, then a
vertically-resizable tab container (Cameras / Data) below the grid.

### Sidebar: four always-visible sections

No tabs, no collapse-by-default — an explicit steer against progressive
disclosure, since calibration is an iterative workflow in practice (add a
CP after seeing solve results, redetect a marker mid-review, jump back and
forth), not a linear wizard. `_build_cp_panel()` assembles exactly these
four sections, each its own method:

1. **Control Points** (`_build_cp_group`) — Add/Delete/Load…/Save… for
   manually-placed control points. Select a CP's row in the Data table
   (below) to arm it for click-to-place; double-click to rename.
2. **Calibration rig setup** (`_build_rig_setup_group`) — `Calib rig…`
   (opens `_CalibRigDialog`, see below), a status label, "Min cameras to
   anchor" (guards against auto-anchoring from a stray single-camera
   glimpse of a rig left over from an earlier capture — see status.md's
   2026-08-12 "moved rig" entry), `Anchor Rig`/`Clear`, and `Manage
   rigs…` (opens `_RigRegistryManagerDialog`).
3. **Markers** (`_build_markers_group`) — `Detect markers…` (opens
   `_DetectMarkersDialog`, ArUco-only bulk detect), `Load markers…`
   (re-anchor from a named `scene_marker_bodies` configuration saved in an
   earlier capture, no physical rig needed), `Save markers…` (opens
   `_SaveMarkersDialog`, disabled until something is anchored/solved), and
   `Manage markers…` (opens `_SceneMarkerManagerDialog`).
4. **Solve** (`_build_solve_group`) — SIFT-matching checkbox, RANSAC
   threshold, `Solve`/`Cancel`, `Load from DB…`, and a shared status label
   that every bulk action/detect/anchor/solve writes its own outcome to.

There is deliberately no separate ChArUco section: per Harri, a ChArUco
board is close enough to a calibration rig that `Calib rig…`'s own
ChArUco Board tab is the only anchor entry point needed. There is also no
separate Cameras/Intrinsics section — intrinsics selection and the
Refine/Lock/Excl per-camera settings live as extra columns on the Cameras
tab's results table instead (see below), since that data is inherently
one-row-per-camera and didn't fit the Data/Actions framing of anything
else in the sidebar.

### Headless per-action settings

ArUco (`_init_aruco_detect_settings`), ChArUco
(`_init_charuco_detect_settings`), and rig (`_init_rig_detect_settings`)
detection settings (dictionary, sizes, min-marker-%, etc.) are real
`QWidget` instances constructed and held as instance attributes, but never
added to any visible sidebar layout — they no longer have their own
standalone sidebar sections (removed once each one's settings were fully
covered by a bulk dialog). Both the per-camera "Detect ArUco"/"Detect
ChArUco"/"Detect Rig" buttons under each thumbnail and the sidebar's bulk
dialogs (`_DetectMarkersDialog`, `_CalibRigDialog`) read and write this
same shared state: a bulk dialog pre-fills itself from whatever's
currently held here when opened, and writes back on OK, so the per-camera
buttons always use whatever was last confirmed in a bulk dialog (or the
constructor defaults, if none has ever been opened).

### Bulk dialogs

**`_DetectMarkersDialog`** — dictionary / default size / min-marker-%,
then detects ArUco markers across every camera with an image loaded at
once.

**`_CalibRigDialog`** — two tabs, one result kind each:

- **Physical Rig**: a table of rigs already imported into this session's
  DB (`marker_body_definitions`, via `list_marker_bodies`) plus a "From
  file…" button and a min-marker-% spinbox (pre-filled from the sidebar's
  current rig min-marker-% setting, not a fixed default, so a value tuned
  in an earlier dialog session survives being reopened). Picking a row and
  clicking OK, or double-clicking a row, loads that rig; "From file…"
  loads a YAML file directly. Either way the rig is immediately detected
  across every camera's current frame and anchored if found in enough of
  them.
- **ChArUco Board**: the board settings (dictionary/squares/lengths/
  face-up/legacy-pattern/min-marker-%), pre-filled from the sidebar's
  current headless ChArUco state. OK detects the board across every camera
  with an image loaded and anchors from it if found anywhere.

The caller (`_on_calib_rig_bulk`) dispatches on `dlg.result_kind()` and
does the actual loading/detecting/anchoring — the dialog only collects a
choice.

## The Cameras/Data tab container

Below the camera grid, one `QTabWidget` (Cameras / Data) sits in its own
pane of a vertical `QSplitter`, so its height is adjustable by dragging
the splitter handle instead of being capped at a fixed size.

**Cameras tab** — one row per camera: position (X/Y/Z), CP reprojection
error, intrinsics-calibration selection (a combo leading with the
calibration's own notes, date/RMS as extra columns), and Refine/Lock/Excl
checkboxes. Populated and visible from dialog construction, not just after
a solve — position/CP-error columns show "—" until solved, but the
intrinsics/refine/lock/excl settings are needed before solving too.

**Data tab** — one unified table, one row per data point currently
contributing to (or available to) the solve: manually-placed control
points, detected ArUco marker groups, ChArUco board corners, physical-rig
corners, loaded-marker-config corners, and camera-position observations
(one camera's manually-marked sighting of another camera's position).
Before this table existed these were three inconsistent representations
(a list widget, a table widget, and "nothing, look at the image overlay
only" for rig corners and camera-position observations); this replaces
all of them. Columns: **Type**, **Label**, **Cameras** (each observing
camera's 1-based order number, e.g. "2, 3" — not a bare count, so a user
can tell *which* cameras without opening a tooltip), **World position**,
**Source**, **Size (m)** (editable per-marker size override, the one
column beyond the original four-column proposal — dropping the
size-override editing `_marker_table` used to provide would have been a
real regression).

**Type** values: `CP`, `Marker`, `Board corner`, `Rig corner`, `Loaded
marker corner`, `Cam pos obs`. The last two both come from the same
`_rig_control_points()` detector/anchor mechanism (see "Two-source rig
anchoring" below) — only the label differs, by `_rig_source`. Selecting a
CP row arms it for click-to-place, matching the pre-Data-table `_cp_list`
selection behavior; double-clicking renames it. Marker/Board-corner/
Rig-corner/Loaded-marker-corner/Cam-pos-obs rows are read-only in the
table itself (populated by their own sidebar-section triggers), but
selecting one:

- **Highlights the corresponding point(s) in every camera view** — a
  lighter fill and thicker white ring versus the normal marker style. A
  `Marker` row highlights its whole group; `Board corner`/`Rig corner`/
  `Loaded marker corner` highlight just that one corner; `Cam pos obs`
  highlights that one camera-position marker (a separate mechanism,
  `_ClickableImageWidget.set_selected_cam_pos_subject`, since those
  markers aren't stored the same way as the others).
- **Populates a detail pane** to the right of the table (`QStackedWidget`,
  one page per Type): CP → World position controls (fix X/Y/Z for BA,
  moved here from the sidebar's old standalone group). Marker → "Clear"
  (removes just that one marker). Board corner/Rig corner → "Clear"
  (removes the whole detected board/rig — individual corners aren't
  prunable for a genuine physical instrument, since piece-by-piece pruning
  of a real rig's own geometry isn't a normal workflow). **Loaded marker
  corner** → "Remove Corner" / "Remove Marker" (in addition to "Clear")
  — pruning one stale/misdetected corner or marker from a *loaded*
  configuration, without discarding the whole thing, is a normal workflow
  the way pruning a physical rig's geometry isn't. Cam pos obs → "Remove"
  (deletes just that one observation — previously the only way to change
  one was to overwrite it by dragging again).

The table does a full rebuild on every refresh (`_refresh_data_table`),
restoring the previously-selected row afterward (via `blockSignals`) so an
unrelated refresh — e.g. right after applying a CP's world position —
doesn't silently drop the user's current selection and close the detail
pane they're using.

## Two-source rig anchoring: `_rig_source`

A loaded rig configuration comes from one of two sources, tracked in
`self._rig_source: str | None`:

- **`"file"`** — a genuine physical rig, loaded from a YAML file or the
  session's rig registry (`marker_body_definitions`). Shows as **Rig
  corner** in the Data table.
- **`"scene_markers"`** — a reconstructed configuration built from
  previously-saved `scene_marker_bodies` rows (`Load markers…`), with no
  physical rig behind it — e.g. a set of sized ArUco markers saved from an
  earlier capture. Shows as **Loaded marker corner**.

Both funnel through the same `_rig_detector`/`_rig_detections_by_camera`/
`_rig_control_points()`/`_apply_loaded_rig_config` machinery; several call
sites branch on `_rig_source` because the two need different treatment:

- Data table row **Type**/**Source** labels (above).
- Detail pane's Remove Corner/Remove Marker buttons only shown for
  `"scene_markers"`.
- `_save_markers_items()` only offers the rig's own anchor pose as a
  checklist item when `_rig_source == "file"` — a `"scene_markers"`
  config isn't a rig to re-save as one (though its own markers can still
  be saved again if re-detected independently, e.g. via `Detect
  markers…`).
- `Min cameras to anchor`'s auto-anchor guard applies only to `"file"` —
  re-anchoring from just one already-known tag in a `"scene_markers"`
  config is the expected, common case there.
- Loading a `"file"` rig excludes its own `(dictionary, marker_id)`s from
  `Detect markers…` output going forward (and retroactively purges any
  already picked up before the rig was loaded) — a `"scene_markers"`
  config's marker ids are ordinary scattered tags, meant to stay
  redetectable.

Per-corner/per-marker removals from a `"scene_markers"` config
(`_rig_excluded_corners`/`_rig_excluded_markers`) are filtered inside
`_rig_control_points()` itself, so both the Data table display and the
actual solve respect them. They reset when a genuinely different
configuration is loaded, but persist across `Clear`+redetect of the same
one — an exclusion is a judgment about that configuration's own markers,
not about one detection pass.

## Save/Load Markers

**`Save markers…`** (`_SaveMarkersDialog`) — a checklist of everything the
session currently has eligible (a file-sourced rig's own anchor pose, any
sized ArUco/ChArUco marker pose from the last solve), default all-checked,
a required name field (`Save` disabled until non-empty). Writes via
`upsert_scene_marker_body` per checked item. There is no implicit
save-on-Accept path — this is the only way scene markers get persisted.
Manually-anchored control points are out of scope (see "Deferred" below).

**`Load markers…`** — always opens a picker of named configurations
(`_SceneMarkerGroupPickerDialog`); there is no anonymous/ungrouped
fallback. Selecting one runs the same detect-and-anchor-immediately flow a
physical rig load does.

**Confirm-before-clobber** (`_confirm_replace_existing_anchor`) — loading
a new rig or marker configuration over something already anchored from a
potentially different source asks first (`QMessageBox.question`) rather
than silently replacing it. This follows CLAUDE.md's "automation vs. prior
human edits" design principle: scope the check to the moment of the write,
ask rather than silently deciding, rather than building a general
precedence system.

## Manage dialogs

Two separate dialogs, matching the two separate concepts a user actually
works with ("we have calibration rigs and named saved scene marker sets"
— Harri), replacing an earlier single "Manage Scene Markers…" dialog that
conflated both and was reported as confusing ("I don't really understand
what it does"):

- **`_RigRegistryManagerDialog`** ("Manage rigs…", reachable from
  Calibration rig setup) — table of `marker_body_definitions` (Name/
  Source/Created), "From file…" (import without detecting/anchoring),
  "Delete Selected" (`delete_marker_body`), Close.
- **`_SceneMarkerManagerDialog`** ("Manage markers…", reachable from
  Markers) — table of every `scene_marker_bodies` row for the session
  (Label/Group/Source/Dictionary/Marker ID/Size/Primary anchor/Updated/
  Rig match?), flagging rows whose (dictionary, marker_id) coincidentally
  matches a known rig's own marker (a residue of an earlier bug — see
  status.md's 2026-08-12 "eighth live-testing round" entry), "Delete
  Selected", Close.

Neither dialog enforces referential integrity against the other table or
against past `extrinsic_calibrations` rows — deleting a rig or a scene
marker that a past run referenced leaves that run's own stored numbers
intact, it just won't resolve back to a name/config anymore. Consistent
with `delete_scene_marker_body`'s existing "let the user prune, don't
overprotect" precedent.

## Deferred / open

- **D2 — manual control points in saved configurations.** Everything
  above only covers marker-based anchors (rig instances, scattered ArUco
  tags). A saved configuration carrying a manually-placed, non-detectable
  control point (fixing world coordinates by hand, with no rig or marker
  involved) needs a reference-image mechanism to be locatable again in a
  later capture — real new scope (a new table, a "match against a
  picture" load-side interaction), not a small addition to the marker
  save/load flow. Not started.
- **Per-camera calibration quality persistence.** `extrinsic_entries`
  stores only R/t per camera, no reprojection error or observation count
  — the per-camera stats the dialog computes at solve time live only in
  its status label, never written to the DB. Would let a status screen
  show calibration quality, not just solved/not-solved, and could feed a
  better-grounded per-camera measurement-noise estimate for the C++
  tracker's UKF. Not decided whether it's worth the schema change; not
  blocking anything today.
- **Re-saving an edited `"scene_markers"` configuration.** Removing a
  corner or marker from a loaded configuration (via the detail pane) edits
  what that session solves with, but there is no direct way to persist the
  *edited* subset under a new name — only markers that get independently
  re-detected (e.g. via `Detect markers…`) appear in `Save markers…`'s
  checklist. Minor, not yet raised as a concrete request.

See [status.md](status.md) for the full dated history, including the
"purple marker mixup" cross-dictionary bug (fixed 2026-08-15) and every
individual UX round's own reasoning.
