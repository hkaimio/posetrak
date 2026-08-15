# Extrinsics calibration — user guide

Extrinsic calibration finds where every camera in a capture actually is —
its 3-D position and orientation — in a shared, real-world coordinate
frame. Posetrak needs this to triangulate a person's body keypoints into
3-D and to solve joint angles from multiple camera views: without it, each
camera's 2-D detections have no way to be combined into one 3-D scene.

There are existing external tools for this (Pose2Sim among them); if
you've already calibrated a capture that way, you don't need this guide —
just import the resulting TOML file ("Import TOML…", below). This guide
covers Posetrak's own native calibration workflow: detecting markers or a
calibration rig directly in the capture's own video, with no separate
calibration session or extra software needed.

## Concepts

A few terms this guide uses throughout:

- **Control point (CP)** — a point clicked/dragged by hand in a camera
  view. A CP can be *free* (just a correspondence that helps the solver,
  no known real-world position) or have a **world position** set, in
  which case it also fixes scale and origin.
- **Fiducial marker** — an ArUco or ChArUco marker/board detected
  automatically. A detected ArUco marker's 4 corners act as one control-
  point group; give the marker a real-world size and it also contributes
  a rigid pose once seen by 2+ cameras.
- **Calibration rig** — several fiducial markers mounted at known,
  fixed positions relative to each other (ideally non-planar — see
  Prerequisites), captured in a small config file. Detecting a rig
  anywhere in frame anchors the whole world coordinate frame in one step,
  no per-marker sizing needed.
- **Anchoring** — the act of fixing the world coordinate frame's origin,
  scale, and orientation. This happens automatically when a rig or
  ChArUco board is detected, or manually by giving one or more control
  points a world position.
- **Scene markers** — fiducial markers that aren't part of a rig, saved
  under a name so a *later* capture (different session, cameras possibly
  moved) can re-anchor from them without a physical rig present. See
  "Save markers…"/"Load markers…" below.
- **Camera-position observation** — one camera manually sighting roughly
  where another camera is, when both happen to see each other. A weak
  extra position constraint, not required for normal use.

## Prerequisites

- **Camera intrinsics already calibrated** for every camera used in the
  capture (Posetrak menu **Cameras → Manage cameras…**). Extrinsics
  calibration assumes intrinsics (focal length, distortion) are already
  known and correct.
- **The scene needs something with known geometry visible to enough
  cameras**: either your own recognizable control points (surveyed or
  hand-measured 3-D positions) or fiducial markers — markers are
  recommended, since they're detected and matched automatically instead
  of requiring you to click the same point precisely in every camera.
- **At least 4 points with known world coordinates**, to fix scale and
  origin. Prefer **non-planar** geometry (not all 4 in one flat plane) —
  a purely planar anchor (e.g. one flat ChArUco board) has a genuine pose
  ambiguity a solver can flip into two mirror-image solutions with
  similar reprojection error, seen in practice as cameras solving with
  the wrong sign of Z. A non-planar calibration rig sidesteps this
  ambiguity entirely, which is why it's the recommended anchor over a
  flat board where practical.

> 📷 **Screenshot 1:** a physical calibration rig (or ChArUco board) as
> it looks in a real capture — gives readers a concrete idea of what
> "non-planar rig" means before the abstract explanation above.

## Setting up the capture

Before opening extrinsics calibration:

1. Make sure the session's cameras are registered with intrinsics
   calibration (**Cameras → Manage cameras…**).
2. Load the capture's videos and, if using multiple cameras, set up sync
   (the capture's **Set up sync…** button) so frames line up across
   cameras.

## Calibration workflow

### 1. Open the Extrinsics dialog

Select the capture, then click its **Extrinsics…** button. This opens a
status dialog — per-camera solved/not-solved summary — with
**Calibrate…** (the native workflow this guide covers) and
**Import TOML…** (import an externally-produced calibration instead).

> 📷 **Screenshot 2:** the Extrinsics status dialog, showing a mix of
> solved and not-solved cameras.

Click **Calibrate…** to open the main calibration dialog: a camera grid
at the top, a sidebar on the right (Control Points / Calibration rig
setup / Markers / Solve), and two tabs below the camera grid (Cameras /
Data) for reviewing results and data points.

> 📷 **Screenshot 3:** the full calibration dialog on first open (camera
> grid, sidebar, empty Data table) — the "orientation" shot for the rest
> of the guide.

### 2. Establish the anchor

If you have a **calibration rig or a ChArUco board**, click **Calib
rig…** in the "Calibration rig setup" sidebar section. It has two tabs:

- **Physical Rig** — pick a rig already in this session's registry, or
  load one from a file ("From file…"). Either way, on OK it's
  immediately detected across every camera's current frame and the world
  frame is anchored if the rig is seen by enough cameras (see "Min
  cameras to anchor," below).
- **ChArUco Board** — dictionary, square/marker dimensions, etc. On OK,
  the board is detected and anchored the same way.

> 📷 **Screenshot 4:** the "Calib rig…" dialog, Physical Rig tab, with a
> rig selected from the registry table.

If nothing was detected in enough cameras, the anchor doesn't apply
automatically — check **"Min cameras to anchor"** (default 2; a rig
glimpsed by only one camera is often left-over clutter from an earlier
capture, so this is a deliberate guard, not a bug). Lower it if a single
camera genuinely is your only view of the rig, then click **Anchor Rig**
to retry.

If you don't have a physical rig or board, you can anchor manually
instead: add control points (**Control Points → Add**, then click/drag
in a camera view to place each one), select a placed point's row in the
Data table, and set its world **X/Y/Z** in the detail pane on the right,
then **Apply**. At least 4 such points (non-planar, see Prerequisites)
are needed to anchor without a rig or board.

If you previously saved a marker configuration from an earlier capture
(see "Save markers for reuse," below), you can instead click **Load
markers…** in the Markers section and pick it by name — this re-anchors
from those saved positions with no physical rig or board present at all.

> 📷 **Screenshot 5:** the "Load markers…" picker showing a couple of
> named configurations.

Loading a rig or marker configuration over something already anchored
from a different source asks for confirmation first, so you don't
silently lose an existing anchor by clicking the wrong button.

### 3. Add more data

Beyond the anchor itself, more data generally improves the solve:

- **Detect markers…** (Markers section) — bulk-detects ArUco markers
  across every camera at once. Give a detected marker a size (its row's
  **Size (m)** column in the Data table) to also recover its own rigid
  world pose once 2+ cameras have seen it — useful even without a rig,
  and this is what "Save markers…" later persists.
- **Manual control points** — add more free (no world position) or
  anchoring (world position set) control points the same way as above.
  Free points still help the solver even without a known position, as
  long as the same physical point is placed consistently across cameras.
- **Camera-position observations** — if two cameras can see each other,
  drag that camera's marker in the observing camera's view to its actual
  pixel position. A rough extra position constraint; not needed for a
  normal setup.

Per-camera **Detect ArUco / Detect ChArUco / Detect Rig** buttons on each
camera thumbnail redo detection for just that one camera — useful after
scrubbing it to a different frame, without repeating the bulk dialog.

### 4. Review your data

The **Data** tab lists every control point, marker, rig/board corner, and
camera-position observation currently contributing to (or available for)
the solve — one row each, with type, label, which cameras observe it,
world position (if anchored), source, and size.

> 📷 **Screenshot 6:** the Data table with a mix of row types (CP,
> Marker, Rig corner, Cam pos obs), one row selected.

Selecting a row highlights that point in every camera view (a lighter
fill, thicker ring), and opens a matching detail pane on the right:

- **CP** — world-position controls (the same X/Y/Z/Apply from step 2).
- **Marker** — Clear (removes just this marker).
- **Rig corner / Board corner** — Clear (removes the whole detected
  rig/board — not prunable corner-by-corner, since editing a genuine
  physical instrument's own geometry isn't a normal workflow).
- **Loaded marker corner** (from "Load markers…", no physical rig) —
  Remove Corner or Remove Marker, for pruning just one bad corner/marker
  from a loaded configuration without discarding the rest, plus Clear for
  everything.
- **Cam pos obs** — Remove (deletes just that one observation).

This is the place to go looking for "how do I get rid of just this one
bad point" — most problems found after a first solve attempt are fixed
here rather than by starting over.

### 5. Solve

Click **Match & Solve** in the Solve section. **SIFT matching** (checked
by default) helps initialize camera poses from shared image features
alongside your control points/markers; uncheck it to rely on control
points alone (needs at least 4 world-position CPs per camera).
**RANSAC** sets the pixel-error threshold used while initializing poses —
raise it if a camera with imperfect intrinsics or near-planar points
fails to initialize at all.

After solving, check the **Cameras** tab: do the reported positions look
physically plausible, and is the CP reprojection error small (a few
pixels, typically)?

> 📷 **Screenshot 7:** the Cameras tab after a solve, showing position
> and CP-error columns for several cameras.

### 6. If the result isn't good

- Reselect the offending points in the Data table and check they're
  correctly (and consistently) placed in every camera that observes
  them — a single mis-clicked point in one camera can pull a whole solve
  off.
- Add more control points, especially ones seen by whichever camera(s)
  are solving worst.
- If most cameras look right but one or two don't, use the **Lock**
  checkbox (Cameras tab) on the good ones so re-solving doesn't disturb
  them while you fix the rest, or **Excl** to drop a camera from the
  solve entirely while diagnosing it.
- If a camera's own intrinsics might be slightly off for this particular
  footage (e.g. a zoom lens that refocused between calibration and
  capture), try **Refine** for that camera — lets the solver adjust its
  focal length within the bundle adjustment instead of trusting the
  stored intrinsics exactly.
- The solver is incremental — it optimizes from whatever's already
  solved rather than starting fresh each time. It often works best to
  start with a small number of control points, solve once, then add more
  and re-solve, rather than adding everything before the first solve.

### 7. Save markers for reuse in later captures

Once something is anchored and/or solved, **Save markers…** (Markers
section) opens a checklist of everything currently eligible — a
file-sourced rig's own anchor, any sized ArUco/ChArUco marker with a
solved pose — default all-checked, plus a required name.

> 📷 **Screenshot 8:** the "Save markers…" dialog with its checklist and
> name field.

Saved markers can be reloaded in a **different** capture via **Load
markers…** (step 2), letting that capture re-anchor from the same
physical points with no rig present and even if the cameras have moved —
useful for a room with permanently-mounted tags, or reusing a portable
rig's last-known scattered markers after the rig itself has been packed
away.

### 8. Housekeeping

**Manage rigs…** and **Manage markers…** (sidebar) each list what this
session has stored — imported rig configs, and saved scene markers
respectively — with a Delete button for pruning stale entries (a rig
that's been physically moved, a tag that's no longer where it was saved).

## Calibration rig definition format

A calibration rig is described by a YAML file: a flat list of markers,
each with its dictionary, marker ID, and real-world corner positions.
This is the only format currently supported — there's no shortcut for
generating one from just a shape (e.g. box dimensions), only from actual
measured or detected corner positions. Two practical ways to produce one:

- **Hand-measure** each marker's corners relative to a chosen reference
  point/orientation and write them into the YAML directly.
- **`python/tools/characterize_rig_from_video.py`** — solves marker
  geometry from an orbit video of the rig (treating sampled frames as
  unknown-pose cameras), useful when hand-measuring isn't practical. Note
  from experience: sample count needs to roughly match the physical
  baseline the orbit actually covers — oversampling a short clip *hurts*
  accuracy (shrinks the baseline between adjacent samples) rather than
  helping, so more samples is not a safe default to reach for.

Import a rig YAML into a session's registry via **Manage rigs… → From
file…**, or `posetrak marker-body import` (below).

## Command-line interface

Everything above except the interactive placement/review steps is also
available from the command line, useful for scripting or batch
processing:

- `posetrak marker-body import` / `list` / `show` / `export` — manage rig
  definitions in the session's registry.
- `posetrak extrinsics anchor-rig` — detect a named rig across given
  cameras/frames, anchor, solve, and write the result.
- `posetrak extrinsics reanchor` — re-anchor from previously-saved scene
  markers, no physical rig needed.
- `posetrak extrinsics import` / `list` — import a TOML calibration, or
  list what's stored for a session.
- `posetrak extrinsics scene-marker list` / `groups` / `delete` — inspect
  or prune saved scene markers.

Run any of these with `--help` for the full option list.

---

## Screenshots to capture

A consolidated list of every screenshot referenced above, in the order
they appear:

1. A physical calibration rig or ChArUco board as seen in a real capture
   (Prerequisites — illustrates "non-planar").
2. The Extrinsics status dialog with a mix of solved/not-solved cameras.
3. The full calibration dialog on first open (camera grid + sidebar +
   empty Data table) — the main orientation shot.
4. The "Calib rig…" dialog, Physical Rig tab, with a rig selected.
5. The "Load markers…" picker with a couple of named configurations.
6. The Data table with a mix of row types and one row selected
   (highlighted in a camera view + detail pane both visible).
7. The Cameras tab after a solve (position + CP-error columns
   populated).
8. The "Save markers…" dialog (checklist + name field).

Two more worth considering, not tied to a specific numbered step above:

9. A "before/after" pair of the Cameras tab — a bad solve (e.g. a mixed-
   sign-Z or high-error case) next to a fixed one — for the
   troubleshooting section.
10. The per-camera thumbnail's Detect ArUco/ChArUco/Rig buttons close up,
    since the guide references them without showing where they live.
