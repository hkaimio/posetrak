# Extrinsics calibration UX redesign — draft proposal

**Status: UX Phases 1–7 landed (2026-08-14).** Still awaiting the
live-testing pass Phase 7 itself called for (real capture footage, click-
through of every anchoring path) before considering it fully proven, not
just unit-tested. Harri has flagged a possible follow-up: moving the Data
table back into the sidebar (its construction was factored into its own
`_build_data_table()` method in anticipation of this). UX Phase 8/D2
(manual CPs in saved configurations) remains deferred behind that. One
open question remains, non-blocking (per-camera calibration quality
persistence — see "Open questions").

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

**Decided:** remove it.

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
   3 rows each. *(2026-08-13, landed independently of this redesign: the
   picker now leads with the calibration's own notes rather than
   date/RMS/model — notes are what a user actually recognises a
   calibration by; the technical summary moved to a detail label below,
   shown for whichever item is selected.)*

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
gone). **Decided** — see below.

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

### D2. Manual control points belong in saved configurations too

Everything above only covers marker-based anchors (rig instances, scattered
ArUco tags). Harri's addition: a saved configuration should also be able to
carry **manually-placed control points that fix world coordinates** — the
case where a scene has no calibration rig but the user has typed in 3D
coordinates for a few known points to anchor the frame directly. Two cases,
worth distinguishing precisely because they get different treatment:

- **(a) Anchoring CPs** — a control point with `world_xyz` set, used to fix
  scale/origin. Valuable to save: this is exactly the same *kind* of thing
  a rig or a scattered tag is (something with a known world position that a
  later capture might want to reuse), just placed by hand instead of
  detected.
- **(b) Free CPs** — a control point with no `world_xyz`, added only to
  help this particular solve (e.g. not enough markers visible in every
  camera). **Not worth saving** — without a world position they're only
  meaningful as pixel correspondences within the one solve they were placed
  for; importing them into an unrelated later capture brings nothing.

So the eligibility rule for "Save Markers…" (or whatever it ends up
called — see below) extends cleanly: *any* control point with `world_xyz`
set is eligible, regardless of whether it came from a rig, a detected tag,
or a manual click. Free CPs never appear in the picker at all.

The real complication is on the **load side**. A saved marker/tag has an
ID an `ArucoDetector` can search for automatically — that's the whole
mechanism "Load Markers…" relies on. A manually-placed point has no such
signature; the only way a human can find "this same physical point" again
in a new capture's camera views is to look at where it was the first time.
That means a saved manual CP needs **a reference image** — a crop/thumbnail
of the camera view it was placed in, showing roughly where it sits — and
the load-side UI needs to show that reference next to the live camera
panes so the user can click the matching spot (reusing the existing
select-CP-then-click-on-camera interaction, just pre-seeded with the saved
world position and a picture to match against instead of starting blind).

This is real new scope, not a small addition to D:

- A new place to store reference imagery per saved manual CP — a crop
  BLOB (or a path, if this project's convention is to keep large binaries
  out of the DB — worth checking that convention before deciding) tied to
  which camera/frame it was captured from. Likely a new table (a manual CP
  doesn't fit `scene_marker_bodies`' shape — no dictionary/marker_id, and
  it needs the image column that table was never going to need), something
  like `scene_control_points` (session_id, group_name, label, world_xyz,
  reference_camera_label, reference_frame_idx, reference_image, created_at).
- A "Load Markers…" experience that, for a manual CP, doesn't auto-detect
  but instead surfaces the reference image and waits for the user to click
  the corresponding point in a current camera view.

Proposed: land this as its own follow-on phase, after the marker-only
save/load flow in D lands and proves the picker/naming pattern works —
same incremental, test-each-step approach the rest of this feature has
used throughout. The eligibility rule (only `world_xyz`-set CPs) and the
picker/name UI from D should carry over directly; only the reference-image
storage and the load-side "match against a picture" interaction are new.

### E. Sidebar reorganization

My original pass at this (three groupings — Anchor World Frame / Marker
Detection / Cameras, in tabs) is superseded by a better model Harri
proposed: organize by **role**, not by *which detector produced the data*.
Three roles:

- **Data** — everything currently contributing to the solve: detected
  ArUco/ChArUco markers, manually-placed control points, a loaded rig's
  corners, and camera-position observations (one camera's manually-marked
  sighting of another camera, the `CamPosObs`/`_on_cam_pos_set` mechanism
  at `page_extrinsics.py:2311` — real, already-shipped, but currently has
  no list representation at all, only the gold/cyan markers drawn directly
  on camera images). All of these are "a data point that helps solve
  camera poses"; today they're scattered across three separate,
  inconsistent representations (`_cp_list`, `_marker_table`, and nothing
  at all for rig corners or cam-pos observations beyond image overlays).
- **Actions** — how you *add* data: detect ArUco, detect ChArUco, load a
  rig, place a manual control point. Each action's own settings (ArUco
  dictionary/size, ChArUco board dimensions, rig min-marker-%) live with
  the action that uses them, not as separate peer sections.
- **Anchoring** — how you *fix the world frame*: anchor from a loaded rig,
  or give one or more control points explicit world coordinates. This is
  the "commit" step — everything in Data can exist without anchoring
  anything (free correspondences still help the SIFT solve); Anchoring is
  specifically about which data points get a fixed `world_xyz`.

This is a real improvement over my original grouping: it separates "what
do I have" from "how do I get more" from "how do I fix the frame," which
is a cleaner match for how calibration actually gets used than grouping by
detector type. It also surfaces a genuine current gap — a unified Data
list would replace three inconsistent representations (a list widget, a
table widget, and "nothing, look at the image overlay") with one, and
would be the natural place for the "Save Markers…"/"Load Markers…" picker
in D to draw its rows from (anything in Data with `world_xyz` set is
exactly the D picker's eligibility set).

**Where does Cameras/Intrinsics go? Decided** — it doesn't merge into
Data/Actions/Anchoring at all (it isn't scene data, doesn't add scene
data, and isn't part of anchoring — those three roles were never going
to have a natural home for it). Instead: fold it into the **existing
per-camera results table** (`_cam_pos_table`, currently Camera / X / Y /
Z / CP error, one row per camera) as additional columns — intrinsics
calibration selection (now notes-first, per the 2026-08-13 fix above) and
the Refine/Lock/Excl settings that today live as three checkboxes per
camera block in the sidebar's Intrinsics section. One row per camera,
everything about that camera in one place, always visible, full width.
This eliminates the "fourth section" question outright rather than
answering it — there's no separate Cameras area left in the sidebar at
all once this table carries it.

**Layout mechanism.** Harri's steer: list over tabs, and no
progressive-disclosure/wizard model — extrinsics calibration is iterative
in practice (add a CP after seeing solve results, redetect a marker
mid-review, jump back and forth), so hiding things behind steps or tab
switches fights how the work actually happens. Tabs (my original
recommendation) are out for the same reason a wizard is out; my Option 3
(sequential flow) is out for the same reason Harri already gave.

The open constraint is fit: a **Data** list with useful columns (type,
label, cameras observing it, world position if anchored, source) is
wider than the sidebar's 300px allows to stay readable. Proposal to
resolve that directly: don't put Data in the sidebar at all. This dialog
already has a full-width table below the camera grid for solve results
(`_cam_pos_table`, `page_extrinsics.py:1523`) — put the unified Data list
there too, full width, always visible, no tab/accordion hiding. So the
full-width area below the camera grid ends up hosting **two** tables,
different row granularities (one row per camera vs. one row per data
point) rather than forced into one: the enriched per-camera **Cameras**
table (position, CP error, intrinsics, refine/lock/excl) and the unified
**Data** table (markers, CPs, rig corners, cam-pos observations). That
leaves the sidebar for just **Actions** and **Anchoring** — two compact,
always-visible (not tabbed, not collapsed-by-default) sections, each
considerably smaller than today's six, since their per-action settings
(dictionary, board size, etc.) only need to be visible when that action's
controls are, which a plain stacked layout already gives for free without
needing tabs to achieve it.

**Alternatives considered, now superseded:** my original three options
(regroup-into-three / tabs / sequential wizard) are kept below only as a
record of what was considered and why the Data/Actions/Anchoring model
plus a full-width Data table is the better fit — not as live options.

<details>
<summary>Original options (superseded)</summary>

**Option 1 — Regroup into fewer top-level sections**, keeping the flat
collapsible-groupbox structure but reducing six peers to three grouped by
detector type (Anchor World Frame / Marker Detection / Cameras).

**Option 2 — Tabs instead of a scrolling accordion**, same three
groupings as Option 1 but as `QTabWidget` tabs.

**Option 3 — Reframe as a sequential flow** (Anchor → Detect → Solve →
Review), with Back/Next and only the current step visible.

All three grouped by *which detector produced the data* rather than by
*role*, and Options 2/3 both hide sections behind switches/steps — the
wrong direction for an iterative workflow per Harri's steer above.

</details>

## Decided

- **Legacy image-folder path**: remove it (`Auto-calibrate (image
  folder)…`, `_on_auto_calibrate`, `_load_states_from_images`).
- **Sidebar reorg**: Data/Actions/Anchoring model confirmed.
- **Cameras/Intrinsics placement**: no separate section at all — folds
  into the existing per-camera results table as extra columns (position,
  CP error, intrinsics selection, refine/lock/excl), full width, always
  visible. See the revised §E above.
- **CLI naming**: `anchor-rig`/`reanchor --name` becomes **required**,
  matching the GUI's mandatory naming — "copy the GUI model to CLI."
  Residual detail, not blocking: what to do with scene markers already
  saved ungrouped (`group_name = ''`) in existing session DBs before this
  change. Proposal, unconfirmed: leave them as-is, reachable only through
  "Manage Scene Markers…" for cleanup — nothing forces a retroactive name
  onto data that predates the requirement.
- **"History…" button**: deferred, not in the first cut.
- **D2 (manual CPs in saved configs)**: confirmed as its own follow-on
  phase after D, not bundled into it. Its own sub-questions (reference-
  image storage mechanism, "Save Markers…" vs. "Save Anchors…" naming)
  stay open but non-blocking — they only need answers when D2 itself gets
  scoped, not now.

## Open questions for Harri

Only one substantive question left; everything else above is resolved
enough to plan implementation around.

1. **Per-camera calibration quality**: worth persisting (schema addition
   to `extrinsic_entries`, threading solve-time stats through
   `write_extrinsics_to_db`), or defer? Harri's lean: not sure, but it
   could feed a better-grounded per-camera measurement-noise estimate for
   the C++ tracker's UKF (`measurement_noise_std` in the `[tracking.ukf]`
   config today is presumably one flat value or hand-tuned per camera —
   real per-camera reprojection stats from calibration would let that be
   derived instead of guessed). That's a real downstream use, but a
   separate piece of work from this UI redesign — doesn't need to be
   decided before starting A–E; only changes how rich the (deferred)
   History view and the status screen's quality column could eventually
   be. Fine to leave open and revisit whenever that tracker-side work is
   actually on deck.

## Phased implementation plan

Same shape as the rest of this feature's roadmap: each phase lands as its
own self-contained, tested change. Numbered independently from the main
design doc's Phases 1–9 (this is UI restructuring, not new capability) —
labelled "UX Phase" throughout, including wherever these land in
`status.md`, to keep the two numbering sequences from colliding.

Ordered low-risk/high-visibility first, riskiest structural change
(UX Phase 7) last, so each earlier phase's tests and live-testing pass
give more confidence before the biggest rework starts. UX Phase 8 (D2) is
listed for completeness but stays deferred per the Decided note above —
not scoped in detail, not part of this round.

**2026-08-14: UX Phases 1–4 landed** (an overnight batch, deliberately
scoped to the phases with no open design judgment calls left, additive/
mechanical changes, and no breaking behavior — see status.md's dated
entry for the reasoning). UX Phases 5–7 still need Harri's review before
landing: 5 ships a CLI breaking change, 6 depends on 5, and 7 is the
widest structural change in the whole plan and explicitly needs a live-
testing pass. See status.md for details of what shipped.

### UX Phase 1 — Remove the legacy image-folder path ✅ Done

- Delete `_on_auto_calibrate()`, the `Auto-calibrate (image folder)…`
  button and its wiring in `ExtrinsicsImportWidget.__init__`, and
  `_load_states_from_images` from `page_extrinsics.py`.
- Drop now-unused imports; update or remove the descriptive (non-calling)
  comment in `camera_registry.py` that references
  `_load_states_from_images` if it goes stale.

**Validation:** full regression sweep (no test currently references this
path — confirmed during the design pass, so none should need updating);
manual check that the file row now holds two buttons, not three, and
reads less cramped.

### UX Phase 2 — Status-first entry point ✅ Done

- New status dialog in `page_extrinsics.py` (e.g. `ExtrinsicsStatusDialog`):
  summary line (N/M cameras solved, method, date) + a per-camera table
  (Camera / Position / Source), and `Calibrate…` / `Import TOML…` / `Close`
  buttons (`History…` excluded — deferred).
- `CapturePanel._open_extrinsics()` (`content_panels.py:783`) launches
  this instead of `ExtrinsicsImportDialog` directly.
- New `CapturePanel._refresh_extrinsics()`, mirroring the existing
  `_refresh_sync()` (`content_panels.py:750`): queries whether the
  session has a solved `extrinsic_calibrations` row, updates the
  `Extrinsics…` button's text/tooltip. Called on panel build and whenever
  the status dialog closes.
- `Calibrate…` opens `ExtrinsicsAutoCalibDialog` unchanged at this phase;
  `Import TOML…` opens the (not yet slimmed — see UX Phase 3)
  `ExtrinsicsImportWidget`.

**Validation:** new tests for the status dialog (correct camera list/
positions/solved-state for a seeded session, with and without
`extrinsic_entries` rows) and for `_refresh_extrinsics()`'s button-text
behavior; manual check against one real session with extrinsics and one
without.

### UX Phase 3 — Split TOML import out cleanly ✅ Done

- Remove the (now sole survivor, since UX Phase 1) `Auto-calibrate…`
  video button from `ExtrinsicsImportWidget`'s file row — routing to the
  GUI-native workflow now happens via the UX Phase 2 status dialog's
  `Calibrate…` button instead.
- `ExtrinsicsImportWidget` becomes purely: file-browse, existing-
  calibrations label, camera-assignment table, `Import` button.
- Check the pose-window entry point (`app/pose/main.py:452`, also opens
  `ExtrinsicsImportDialog`) still makes sense pointed at this now-TOML-
  only widget — it may want the same status-first treatment as
  `CapturePanel`, or may be fine importing directly; note whichever way
  this goes, don't silently change its behavior without checking.

**Validation:** existing `ExtrinsicsImportWidget`/`ExtrinsicsImportDialog`
tests updated for the removed button; manual re-verification of the TOML
import flow end to end from both entry points.

### UX Phase 4 — Fold Intrinsics into the per-camera results table ✅ Done

- Extend `_cam_pos_table` (`page_extrinsics.py:1523`) with new columns:
  Intrinsics (combo, notes-first per the 2026-08-13 fix — reuses
  `_populate_intrinsics_combo`, detail text as a cell tooltip instead of
  a separate label since table cells are tighter than the old sidebar
  block), Refine, Lock, Excl (checkboxes, same `_refine_intrinsics`/
  `_locked_cameras`/`_excluded_cameras` set-wiring as today).
- Remove `_build_intrinsics_group()` and the "Camera Intrinsics"
  collapsible sidebar section entirely.

**Validation:** update `test_extrinsics_panel_layout.py`'s intrinsics
tests to look for the combo/checkboxes in the results table; verify
`_cam_pos_table` still populates correctly after a solve; live check that
changing a camera's intrinsics from its new table cell still re-solves
correctly.

### UX Phase 5 — Explicit, always-named Save/Load Markers ✅ Done

- New "Save Markers…" dialog: checklist of every `world_xyz`-eligible
  item this session currently has (rig anchor, sized ArUco/ChArUco
  markers with a solved pose — manually-anchored CPs are explicitly
  **out of scope for this phase**, since saving those needs UX Phase 8/D2's
  reference-image mechanism to be useful on reload; the picker only
  offers what can actually be auto-redetected), default all-checked,
  required name field (`Save` disabled until non-empty). Writes via
  `upsert_scene_marker_body(..., group_name=name)` per checked item.
- Removes `_scene_marker_group_edit` and Accept's implicit persistence —
  saving becomes this one explicit action.
- Rename "From Scene Markers…" → "Load Markers…"; `_SceneMarkerGroupPickerDialog`
  drops its `(ungrouped)` row (see the residual, unconfirmed proposal in
  Decided about pre-existing ungrouped data).
- Confirm-before-clobber: before applying a loaded configuration, check
  for an existing rig anchor or manually-anchored CPs from a different
  source; if found, `QMessageBox.question` before replacing.
- CLI: `anchor-rig`/`reanchor --name` becomes `required=True`.

**Validation:** new tests — Save Markers picker eligibility/defaults/
required-name enforcement; Load Markers always shows the named list, never
an ungrouped fallback; confirm-before-clobber prompts correctly on both
accept and cancel paths. CLI tests updated for `--name` now required.
Full regression sweep.

### UX Phase 6 — Sidebar reorg: Actions / Anchoring ✅ Done

- Restructure `_build_cp_panel()` into two always-visible sidebar groups
  (no tabs, no collapse-by-default, per Harri's iterative-workflow steer):
  - **Actions**: ArUco detect (dictionary/size settings + button), ChArUco
    detect (board settings + button), rig loading (Load Config…/From
    Registry…/Load Markers…), manual-CP placement.
  - **Anchoring**: rig Anchor button + min-cameras-to-anchor spinbox,
    ChArUco "Set origin & axes," manual World Position XYZ fields +
    Apply, Save Markers…/Manage Scene Markers….
- Every per-action setting moves with its action, not as a separate peer
  groupbox.

**Validation:** update `test_extrinsics_panel_layout.py`'s section tests
for the new two-groupbox structure; manual check the sidebar fits
comfortably at 300px with the smaller sections; live re-test of the full
rig-anchor and ArUco/ChArUco workflows end to end to confirm the move
didn't change behavior, only location.

### UX Phase 7 — Unified Data table ✅ Done

- New full-width table below the camera grid, alongside UX Phase 4's
  Cameras table: one row per data point — Type (marker/CP/rig-corner/
  cam-pos-obs), Label, Cameras observing it, World position (if
  anchored), Source.
- Replaces `_cp_list` and `_marker_table` as the CP-selection-for-
  placement mechanism: selecting a "CP" row arms click-to-place the same
  way `_cp_list`'s selection does today. Rig-corner and cam-pos-obs rows
  are read-only (populated by their own Action-section triggers, not
  directly editable here) — both get list representation for the first
  time; today they only exist as image overlays.

**Validation:** the most invasive phase — plan for the widest live-
testing pass of the whole plan. New tests for the table's population
from every source (CPs, markers, rig detections, cam-pos-obs) and for
CP-placement-via-table-selection behavioral parity with today's
`_cp_list`-driven flow. Live re-test of every anchoring path (manual CP,
rig, ChArUco, scattered tags) end to end against real capture footage
before calling this phase done.

### UX Phase 8 (deferred) — D2: manual CPs in saved configurations

Not scoped in detail — per the Decided note above, scoping happens when
this is actually picked up, not as part of this round. Placeholder shape
only: a `scene_control_points`-like table (world position + reference
image), the UX Phase 5 Save Markers picker extended to include anchoring
CPs once this lands, and a Load Markers experience for manual points that
shows the reference image and waits for a matching click instead of
auto-detecting.

**Validation:** TBD when scoped.

### Phase summary

| Phase | Description | Depends on |
|-------|-------------|------------|
| UX 1 | Remove legacy image-folder path | — |
| UX 2 | Status-first entry point + `CapturePanel` button refresh | — |
| UX 3 | Split TOML import out cleanly | UX 2 |
| UX 4 | Fold Intrinsics into the per-camera results table | — |
| UX 5 | Explicit, always-named Save/Load Markers + CLI `--name` required | — |
| UX 6 | Sidebar reorg: Actions / Anchoring | UX 5 (moves its buttons) |
| UX 7 | Unified Data table | UX 6 (lands in the reorganized sidebar's place) |
| UX 8 | *(deferred)* D2: manual CPs in saved configurations | UX 5, UX 7 |

UX 1/2/3 (entry point) and UX 4 (Cameras table) have no real dependencies
on each other or on UX 5/6/7 — could land in either order, or in
parallel, without conflict. UX 5 should land before UX 6 so "Actions"/
"Anchoring" are organizing already-final button behavior, not something
about to change again. UX 7 is sequenced last deliberately, per the
ordering note above.

With that, this document has enough resolved to become a phased
implementation plan — happy to draft one (same shape as the rest of this
feature's roadmap, each phase landing with its own tests against real
data) whenever you want to move forward.
