# Extrinsics calibration UX redesign — draft proposal

**Status: draft, not yet approved.** No implementation should start from this
document until the open questions at the end are resolved with Harri.

## Why this exists

Phases 1–9 of `extrinsics-improvements-design.md` built out real capability —
ArUco/ChArUco detection, a portable calibration rig, scattered scene tags,
named marker groups — and each landed with real tests against real capture
data. But each one also landed as an *addition* to one already-crowded
dialog, driven by whatever the previous round of live testing surfaced. After
nine rounds of "this one thing is confusing, fix it," the accumulated result
is a screen that does a lot but doesn't clearly communicate what it's doing.
Harri's assessment after the ninth round: the capability is there, the UX is
not, and further feature work should pause for a design pass rather than
adding a tenth patch on top.

This document is that pass: an inventory of the current screens (with file/
line references, so it's checkable), the specific problems Harri named, and
a proposed restructuring. It intentionally stops short of an implementation
plan — the open questions at the end need answers first.

## Current state (grounding)

### Entry point: `CapturePanel` ("the main page")

`python/app/ui/content_panels.py:704` — the capture detail panel (shown when
a capture is selected in the main window's tree) has a flat toolbar of
buttons: `Mark Start`, `Mark End`, `Set up sync…`, **`Extrinsics…`**,
`New trial…`. `Extrinsics…` unconditionally opens `ExtrinsicsImportDialog`
(`page_extrinsics.py:4010`) — there is no visual difference between "this
capture has no extrinsics yet" and "this capture is fully calibrated."

Contrast: `_refresh_sync()` (`content_panels.py:750`) already does the thing
that's missing here — it queries `sync_configs` for this capture and uses
the result to enable/disable `New trial…`. There's no reason extrinsics
status couldn't follow the same pattern; it just never got the pass.

### What "Extrinsics…" actually opens

`ExtrinsicsImportDialog` wraps `ExtrinsicsImportWidget`
(`page_extrinsics.py:3589`), which is, top to bottom:

1. A file row: `Browse TOML…`, `Auto-calibrate…` (opens the real GUI
   calibration dialog), `Auto-calibrate (image folder)…` (legacy).
2. An "existing calibrations" label — one line of terse text (`date [method]
   id…`, `_refresh_existing_label`, `page_extrinsics.py:3845`), not a status
   view. Nothing here says which cameras are actually solved or where they
   are.
3. A "Camera assignment" table — mapping TOML `camX` entries to session
   camera instances. **This table, and everything below it, only makes
   sense once a TOML file is already loaded.** For anyone who doesn't have a
   Pose2Sim TOML — i.e. anyone using the GUI-native rig/marker workflow this
   project has spent nine rounds building — this entire screen is dead
   weight between the button and the thing they actually want.
4. An `Import` button, disabled until a TOML is loaded and matched.

So the *only* generic, format-agnostic entry point into extrinsics is a
dialog whose entire visible surface is about editing a TOML import. Harri's
point 2 exactly: this needs to be a status view first, with TOML import as
one explicit action reachable from it — not the default screen.

### The legacy image-folder path

`_on_auto_calibrate()` (`page_extrinsics.py:3742`) — runs the same
`ExtrinsicsAutoCalibDialog` as the primary `Auto-calibrate…` button, but
sourced from a directory of previously-exported PNG frames instead of
scrubbing video directly. Predates `VideoScrubBar`'s direct-video-scrubbing
support (see the design doc's "Frame source & scrubbing" section — the
video-scrubbing path was built specifically to remove the need for a
still-frame export step). Grounding for removal:

- No test references it (`_on_auto_calibrate`, `_load_states_from_images`,
  or the button) anywhere in `python/tests/`.
- No other code path depends on `_load_states_from_images` (one comment in
  `camera_registry.py` references it descriptively, nothing calls it).
- It's the third button crammed into a file row that only has room for two,
  which is a large part of why that row is cramped in the first place.

Proposed: remove it. Flagged as an open question below in case Harri knows
of a use case (e.g. a capture with no usable video, only exported stills)
that the video-scrubbing path can't cover.

### The `ExtrinsicsAutoCalibDialog` sidebar

`_build_cp_panel()` (`page_extrinsics.py:1554`) stacks, top to bottom, in a
**300px fixed-width** scrollable sidebar (`page_extrinsics.py:1645`):

1. **Control Points** — list + add/remove/rename, per-camera click-to-place.
2. **World Position** — XYZ fields to fix a selected CP's world coordinates.
3. **ArUco Markers** *(collapsible)* — dictionary, default size, per-marker
   size table, detect button semantics.
4. **ChArUco Board** *(collapsible)* — board dimensions, square/marker
   length, face-up toggle.
5. **Marker Rig / Scene Markers** *(collapsible)* — the panel this session
   built out repeatedly: three load buttons (file/registry/scene-markers),
   a manage-scene-markers button, a group-name field, two spinboxes
   (min-marker-size%, min-cameras-to-anchor), status label, anchor/clear
   buttons.
6. **Intrinsics** *(collapsible)* — per-camera calibration selection,
   3 rows each.

Above the splitter: a solve row (`Match & Solve`, `Cancel`, `Load from DB…`,
a SIFT checkbox, a RANSAC threshold spinbox, a status label). Below: a
camera-positions results table.

Six independently-collapsible concern areas plus a persistent action row is
a lot of surface for one 300px column, and — this is the sharper problem —
**they're presented as six peers with no indication of which ones are
alternatives to each other and which are complementary.** ArUco, ChArUco,
and Marker Rig are three different ways to *establish the world frame*;
Control Points can also do that (via World Position) or just add free
correspondences; Intrinsics is a completely different concern (per-camera
calibration selection, not scene geometry) that happens to live in the same
list. Nothing in the current layout communicates that grouping.

### The data model has no per-camera calibration quality

`extrinsic_calibrations` has one `rms_error REAL` column — session-scoped,
not per-camera (`db/session_schema.sql:48`). `extrinsic_entries` stores only
`R`/`t` per camera (`db/session_schema.sql:59`) — no reprojection error, no
observation count, nothing that would let a status screen show "camera 3 is
solved but noisy." The per-camera reprojection stats the current dialog
*does* compute at solve time (`compute_reprojection_errors`,
`compute_cp_errors`) live only in the dialog's status label — never
written to the DB. A "show calib data for each camera" status screen can
show position (derived from stored R/t) and "solved / not solved" for free,
but can't show quality without either persisting these stats or
re-deriving them from stored observations (which aren't persisted either).
Flagged as an open question below — it changes how much the status screen
can promise.

### The scene-marker group feature just shipped doesn't match what Harri wants

The `group_name` feature (session schema v40→v41, landed just before this
message) defaults to an ungrouped `''` when no name is given, and "From
Scene Markers…" silently loads that ungrouped bucket when no named groups
exist yet — a deliberate zero-friction default at the time. Harri's
message supersedes that: **no default bucket, only named configurations.**
Every save should require a name; every load should be picking from a list
of named configurations, never silently falling back to an anonymous one.
This needs an explicit revisit — see the proposal and open question below.

## Problems, restated against the above

1. No way to tell from the main page whether a capture has extrinsics.
2. The generic entry point is a TOML-editing screen, not a status view.
3. `Auto-calibrate (image folder)…` is legacy clutter.
4. Scene marker groups have an implicit "default"/ungrouped state that
   shouldn't exist — every save should be named, every load should pick
   from named configurations.
5. Saving scene markers currently happens silently (an optional text field
   read at Accept time) rather than as an explicit, reviewable action with
   a picker (which markers) and a required name.
6. Loading a named marker configuration doesn't check for or warn about
   clobbering whatever's already anchored in the current session (a
   previously-loaded rig, or manually-placed world-position control
   points).
7. The right-hand sidebar presents six concern areas as undifferentiated
   peers, with no visual grouping of "these are alternative ways to anchor
   the world frame" vs. "this is a per-camera setting" vs. "these are
   detection settings that feed the anchor."

## Proposed restructuring

### A. Status-first entry point

Replace the unconditional `ExtrinsicsImportDialog` launch with a new,
lightweight **Extrinsics Status** dialog as the default target of
`Extrinsics…`:

```
┌─ Extrinsics — Capture 2026-08-12 (room7) ──────────────────┐
│                                                              │
│  5 / 6 cameras solved · rig-anchor · 2026-08-12 14:03        │
│                                                              │
│  Camera          Position (m)              Source            │
│  ─────────────────────────────────────────────────────────  │
│  gopro-01        1.20, -0.85, 1.60          rig-anchor         │
│  gopro-02        2.05,  0.40, 1.55          rig-anchor         │
│  gopro-03        -0.30, 1.10, 1.58          rig-anchor         │
│  gopro-04        0.90, -2.00, 1.61          rig-anchor         │
│  gopro-05        —                          not solved         │
│  gopro-06        1.75,  0.05, 1.59          rig-anchor         │
│                                                              │
│  [ Calibrate… ]   [ Import TOML… ]   [ History… ]   [Close]  │
└──────────────────────────────────────────────────────────────┘
```

- "Calibrate…" opens the existing `ExtrinsicsAutoCalibDialog` (the real
  workflow, restructured per part D below) — unchanged entry semantics,
  just moved one level deeper.
- "Import TOML…" opens today's TOML-import screen, unchanged, but now
  reached explicitly rather than being the default.
- "History…" — optional, could defer: list past `extrinsic_calibrations`
  rows for the session, let the user pick one to preview/restore. Not
  required for the first cut.
- Quality columns (reprojection error, etc.) depend on the persistence gap
  noted above — see open question.

`CapturePanel`'s `Extrinsics…` button gets a companion status refresh,
mirroring `_refresh_sync()`: on panel build and whenever the status dialog
closes, requery whether this capture's session has a solved calibration and
update the button's text/tooltip (e.g. `Extrinsics ✓ (6/6)` vs.
`Extrinsics (not set)`). Cheap, and answers problem 1 directly without a
new visual language.

### B. Split TOML import out cleanly

`ExtrinsicsImportWidget` keeps its file-browse + camera-assignment-table +
import flow exactly as-is, but drops the two `Auto-calibrate…` buttons
(both move to be reached from the status screen — video-scrubbing via
"Calibrate…", image-folder path removed per part C). It becomes a
single-purpose "import this TOML" screen instead of a screen that's also,
confusingly, an entry point to the entire GUI-native workflow.

### C. Remove the legacy image-folder path

Delete `_on_auto_calibrate()`, the `Auto-calibrate (image folder)…` button,
and `_load_states_from_images` (if nothing else uses it once the button is
gone). Confirm with Harri first (open question below) since this is a
one-way deletion of a working, if unused, code path.

### D. Explicit save/load for scene markers, always named

Replace the current implicit flow (a `group_name` text field read silently
at Accept time) with two explicit actions in the rig/marker panel:

**"Save Markers…"** — enabled only when the current session has at least
one anchored thing (rig anchored, or manually-placed control points with a
fixed world position, or sized ArUco markers with a solved pose — anything
that would currently be eligible for `scene_marker_bodies`). Opens a picker:

```
┌─ Save Markers ──────────────────────────────────┐
│  ☑ rig:aikido-calib-box     (primary anchor)      │
│  ☑ tag:3                                          │
│  ☑ tag:7                                          │
│  ☐ tag:12          (only 1 camera — check pose)   │
│                                                    │
│  Configuration name:  [___________________]  *required │
│                                                    │
│                          [ Cancel ]   [ Save ]    │
└────────────────────────────────────────────────────┘
```

Defaults to all-checked (matching Harri's "default all"). `Save` is
disabled until the name field is non-empty — no ungrouped fallback. This
replaces the current `_scene_marker_group_edit` field entirely; Accept no
longer silently persists anything on its own.

**"Load Markers…"** (renamed from "From Scene Markers…") — always opens a
picker of named configurations (the existing `_SceneMarkerGroupPickerDialog`
minus its `(ungrouped)` row, since ungrouped won't exist as a save target
going forward). If none exist yet: "No saved marker configurations for this
session." Selecting one runs today's detect-and-anchor-immediately flow
unchanged.

**Before applying a loaded configuration**, check whether the dialog
already has something anchored from a *different* source (a previously
loaded rig, or manual world-position control points) and, if so, confirm
before replacing it:

```
"This will replace the current world-frame anchor (rig
'aikido-calib-box'). Continue?"                    [Cancel] [Replace]
```

This is the same principle CLAUDE.md's automation-vs-prior-state section
already establishes elsewhere in this codebase (hand-detection-refinement's
auto-redetect toggle is the worked example) — scope the check to the
moment of the write, ask rather than silently deciding, don't build a
global precedence system for it.

### E. Sidebar reorganization

This is the part with real design forks — three concrete options, roughly
increasing in how much they change:

**Option 1 — Regroup into fewer top-level sections.** Keep the flat
collapsible-groupbox structure, but reduce six peers to three:
- **"Anchor World Frame"**: Control Points + World Position + Marker
  Rig/Scene Markers folded into one section (they're all ways to fix scale/
  origin), probably as inner tabs (Manual / Rig / Scene Markers) rather
  than three inner subsections.
- **"Marker Detection"**: ArUco + ChArUco (detection settings that feed
  free correspondences and/or the anchor).
- **"Cameras"**: Intrinsics selection, unchanged.

Lowest-risk, smallest diff from today's code — still a scrollable sidebar,
just fewer, more meaningful toggles. Doesn't fully solve "which of these
are alternatives" but goes a long way.

**Option 2 — Tabs instead of a scrolling accordion.** Same three groupings
as Option 1, but as actual `QTabWidget` tabs instead of collapsible
sections — only one tab's content occupies the sidebar's height at a time,
no scrolling. Slightly more code churn (tab widget + state preserved across
tab switches) but a cleaner result — matches how the six sections were
never meant to all be open simultaneously anyway (that's exactly what
"collapsible" already assumes).

**Option 3 — Reframe as a sequential flow.** 1) Anchor world frame → 2)
Detect markers / configure SIFT → 3) Solve → 4) Review & Accept, with
Back/Next and only the current step's controls visible. Most aligned with
"progressive disclosure," but the current dialog's workflow isn't strictly
sequential in practice (you go back to add a CP after seeing solve results,
redetect a marker mid-review, etc.) — a step model would need an "any step,
any time" escape hatch to not become more annoying than the current
free-form layout. Biggest rework of the three.

**Recommendation: Option 2.** It solves the actual complaint (too much
visible at once, unclear grouping) with a well-understood, low-risk Qt
pattern, and doesn't fight the dialog's inherently non-linear workflow the
way Option 3 would.

## Open questions for Harri

1. **Sidebar reorg**: Option 1, 2, or 3 above (or something else)?
2. **Legacy image-folder path**: confirm removal, or is there a real case
   (no usable video, only exported stills) it still covers?
3. **Per-camera calibration quality**: is "position + solved/not-solved"
   enough for the status screen's first cut, or is per-camera reprojection
   error/observation count important enough to justify persisting it (a
   schema addition to `extrinsic_entries`, and threading solve-time stats
   through `write_extrinsics_to_db`)?
4. **Always-named groups**: should the CLI's `anchor-rig`/`reanchor --name`
   stay optional (ungrouped default) for power users, or should the CLI
   also require `--name` for consistency with the GUI's new mandatory
   naming? And what happens to scene markers already saved ungrouped in
   existing session DBs — leave them reachable only via "Manage Scene
   Markers…" for cleanup/rename, or something else?
5. **"History…" button** on the status screen — worth building now, or
   defer until something concrete needs it?

Once these are answered this becomes a phased implementation plan, same
shape as the rest of this feature's roadmap — each phase landing with its
own tests against real data.
