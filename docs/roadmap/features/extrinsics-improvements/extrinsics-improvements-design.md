# Extrinsics calibration improvements — design

## Introduction

Extrinsic calibration today (`python/app/setup/page_extrinsics.py` +
`python/app/setup/extrinsics_solver.py`) requires the user to first export a
single still frame per camera to PNG files on disk (via some external step),
then load that directory into `ExtrinsicsAutoCalibDialog`. From there, SIFT
feature matching bootstraps camera-pair poses, the user manually clicks
matching physical features across camera views to add
`ControlPoint`s (some with known `world_xyz` to fix scale/origin/axes), and
bundle adjustment (`run_calibration`) refines everything. The result is
written as a Pose2Sim TOML and imported into the session DB
(`extrinsic_calibrations` / `extrinsic_entries`, see
`posetrak/db/import_extrinsics.py`).

This has three practical problems:

1. **The still-image export step is a manual, out-of-band chore** — one more
   thing the user has to do correctly before calibration can even start, and
   it commits to exactly one frame per camera up front.
2. **One frame per camera is fragile.** If a person or object occludes a
   control point in the one frame that was exported, that point simply can't
   be placed for that camera — there's no way to pull a different, cleaner
   frame for just that one point.
3. **All correspondences are manual clicks, and the automatic fallback isn't
   usable in practice.** The SIFT pairwise bootstrap is meant to reduce
   manual work, but in real use it has not proven reliable enough to depend
   on — in practice control points are placed almost entirely by hand today.
   There is no automatic, *unambiguous* fiducial marker detection, and no way
   to give the solver metrically-known geometry short of manually entering
   `world_xyz` for hand-picked points.

This document proposes: (a) scrubbing calibration frames directly from the
already-known capture videos, with independent frame choice per camera and
per control point; (b) ChArUco board detection, optionally anchoring the
world coordinate system; (c) ArUco marker detection as correlated groups of
control points, usable both as an accuracy aid for calibration itself and as
**persistent physical scene fiducials** that make recalibration after a rig
change fast; and (d) a detector abstraction so other marker families
(AprilTag, …) can be added without touching the solver. Point (c) is
deliberately framed as *this feature's* use of marker detection, not the only
one ever intended: two related future directions — continuous, per-frame
extrinsics correction by re-checking marker positions instead of calibrating
once, and marker-based motion capture of objects that move during a
trial — are called out in §3 and §6 so this design doesn't back itself into
an assumption ("a marker's pose is fixed for all time") that those directions
would need to undo.

## Current state (grounding)

Relevant existing pieces this design builds on or must change:

- `capture_videos` (`db/session_schema.sql:124`) already has `file_path`,
  `first_video_frame`, `last_video_frame`, `actual_fps` per camera per
  capture — **the raw material for direct video scrubbing already exists in
  the DB**; no new capture-level metadata is needed to seek frames.
- `CamCalibState` (`extrinsics_solver.py:37`) holds exactly **one** `image`
  per camera, loaded once from a PNG matched by filename
  (`_load_states_from_images`, `page_extrinsics.py:122`, globbing
  `images_dir.glob("*.png")`).
- `ControlPoint` (`extrinsics_solver.py:52`) holds `obs: dict[video_id, (px,
  py)]` — one pixel observation per camera, with no frame association at
  all (implicitly "the one loaded image").
- `CamPosObs` (`extrinsics_solver.py:61`) is an existing, narrower precedent
  for "a point observed only in some camera views, contributing a BA
  residual" — camera-to-camera position sightings, not marker corners, but
  the same shape of residual this design needs for fiducial corners.
- Control points are **ephemeral working state**, explicitly by design (see
  `docs/extrinsics-calibration-design.md`, "Control points are ephemeral");
  they can be saved/loaded as a portable JSON file keyed by camera *label*
  (`save_control_points`/`load_control_points`, `extrinsics_solver.py:1419`
  onward, `version: 1`). **This JSON file is the "extrinsics configuration
  JSON file" referenced in this feature's requirements** — it is a working
  file for one calibration session, not the final camera-pose result (that
  result is the DB's `extrinsic_calibrations`/`extrinsic_entries`, written
  via `import_extrinsics.py` from a Pose2Sim TOML).
- Coordinate-system fixing today is entirely manual: scale from a known
  distance between two points, origin from a point or plane, axes from three
  floor points plus a forward point (`docs/extrinsics-calibration-design.md`,
  "Similarity transform").
- No fiducial-marker detection of any kind exists yet. `cv2.aruco` is
  available in the project's OpenCV build (checked: opencv-contrib 4.13,
  `cv2.aruco` present) — no new dependency needed for ArUco/ChArUco.

## Requirements

- **R1 — No manual still-image export.** Calibration frames are read
  directly from the capture's video files (`capture_videos.file_path`);
  the PNG-export-then-load step is removed as a requirement (may remain
  as an alternate/legacy path, not as the only path).
- **R2 — Per-camera frame scrubbing.** Each camera has its own independent
  scrub control; the frame shown for camera A is unrelated to the frame
  shown for camera B. This is safe specifically because *this* feature's
  markers are the static-scene-fixture case (§6) — nothing being calibrated
  against is expected to move between one camera's chosen frame and
  another's. A future feature that detects markers attached to moving
  objects would need cross-camera time alignment (i.e. the existing
  synchronization data, not independent per-camera scrubbing) for that
  separate use case — see §3 and §6.
- **R3 — Per-control-point frame choice.** Placing a control point in a
  given camera records *which frame* it was placed on for that camera;
  different control points may use different frames, including for the
  same camera.
- **R4 — Multiple frames per camera.** A user can pull several frames from
  the same camera over the course of one calibration session (e.g. because
  a person walked through the scene and occluded different points at
  different times).
- **R5 — ChArUco board detection.** User supplies board type/dictionary and
  dimensions; the app detects the board and its corners in a chosen frame
  per camera. Optionally, the user can anchor the world coordinate system
  (origin + axes, and implicitly scale) from a detected board pose.
- **R6 — ArUco marker detection.** Individual markers (not part of a
  ChArUco board) are detected and their four corners are treated as a
  *correlated group* of control points (one rigid planar quad), not four
  independent points. Marker physical size can optionally be supplied,
  either as one global default or per marker ID, so the solver can use it
  as a metric constraint.
- **R7 — Distinct handling for scene fiducials vs. manual control points in
  the exported/stored configuration.** Manually-clicked control points
  remain ephemeral, per-session working state. Detected (Ch)ArUco markers
  meant to stay fixed in the physical scene must additionally be
  representable as a **3D pose (position + orientation)**, not just a set
  of points, and must be persistable in a form that a *later* recalibration
  can reload and reuse without re-establishing world coordinates from
  scratch.
- **R8 — Extensible fiducial framework.** Adding a new marker family (e.g.
  AprilTag) must not require changes to the bundle-adjustment core or the
  storage format's shape, only a new detector implementation.

## Design

### 1. Frame source & scrubbing

**Recommendation: reuse the video-scrubbing machinery the sync page already
has, rather than building a new frame source or a new scrub widget.** This
project already scrubs individual camera videos by frame in more than one
place, and the pieces are already factored out of the page that uses them:

- `app/setup/video_reader.py`'s `FrameReader` — a small `QThread` that owns
  one `cv2.VideoCapture` per file, coalesces rapid seek requests (only the
  most recently requested frame is ever decoded), and reports back via a
  `frame_ready` signal. This is already the project's answer to "random-seek
  one video file from a Qt widget."
- `app/setup/pair_scrubber.py`'s `_VideoPane` — wraps a `FrameReader` +
  `CameraCell` + `QSlider` + frame-number label + "Go to…" button + arrow-key
  frame stepping into one reusable unit, currently used twice per
  `PairScrubber` (reference/target sides) for marking sync anchors.
- `_ROISelectDialog` (`page_sync.py`) additionally rolls its **own**,
  independent `QSlider` + `cv2.VideoCapture` wiring for the LED-ROI-picking
  step — i.e. there are already two separate scrub-control implementations
  in the codebase before this feature adds a third.

Extrinsics calibration needs an **N-camera grid** of independent scrub
panes, not `PairScrubber`'s fixed two-pane (reference/target) layout, so it
can't use `PairScrubber` directly — but the *pane* itself (`_VideoPane`) is
exactly the reusable unit needed per camera. Concretely:

- Promote `_VideoPane` out of `pair_scrubber.py` (where it is currently a
  private, module-local class) into a small shared module — the natural home
  is alongside `FrameReader` in `video_reader.py` — so it can be tiled
  N-across in a new extrinsics camera grid, reused unchanged by
  `PairScrubber`, and available to fold `_ROISelectDialog`'s bespoke slider
  into as a follow-up cleanup (not required for this feature, but a direct
  consequence of there being a shared component to consolidate onto).
- Each camera's pane is bound to `[first_video_frame, last_video_frame]`
  from that camera's `capture_videos` row, exactly as `_reload_scrubber_ref`/
  `_reload_scrubber_tgt` (`page_sync.py:1614`) already do today.
- `_ClickableImageWidget` (`page_extrinsics.py:300`) — the zoom-on-drag
  control-point placement widget — keeps its own interaction model (it does
  meaningfully more than `CameraCell`'s plain overlay painting: press-drag
  zoom for precise placement), but is driven by the shared pane's decoded
  frame instead of a single image loaded once from disk.
- `CamCalibState.image` (`extrinsics_solver.py:37`) changes meaning from "the
  one loaded image" to "the frame currently displayed in this camera's
  pane" — refreshed on every scrub, not loaded once.

**Seek performance**: random seeks into long-GOP consumer codecs (GoPro
H.264/H.265) are already known to be slow in this codebase — the sync
feature hits exactly this today. That feature's requirement is stricter than
this one, though: sync anchors need frame-accurate placement and are placed
often during a session, while extrinsics calibration needs only a handful of
one-off placements per camera. Given that, current performance is presumed
adequate for this feature without separate measurement — reusing the sync
page's scrub machinery rather than reinventing it means this feature
automatically inherits whatever perf work is done there later, instead of
needing its own.

### 2. Per-control-point, per-frame observations

`ControlPoint.obs` (`extrinsics_solver.py:52`) must record which frame each
per-camera observation came from, not just the pixel:

```python
@dataclass
class ObsPoint:
    frame_idx: int
    px: float
    py: float

@dataclass
class ControlPoint:
    name: str
    obs: dict[str, ObsPoint] = field(default_factory=dict)  # video_id -> ObsPoint
    world_xyz: np.ndarray | None = None
```

The bundle adjustment itself (`run_calibration`, `bundle_adjustment`) only
ever needs `(px, py)` per observation — `frame_idx` is provenance for the UI
and the save file, not a BA input. Call sites that currently destructure
`(px, py)` tuples need `.px, .py` instead; this is a mechanical, contained
change (the dataclass fields keep the tuple-like ordering so most sites need
only a rename, not restructuring).

UI-side, placing a control point captures the *currently scrubbed* frame
index for whichever camera the click landed in. Re-clicking the same
control point's name in the same camera at a different scrub position
overwrites that camera's `ObsPoint` (new frame, new pixel) — this is exactly
requirement R4's "use a different/later frame for the same point in the same
camera" case.

The pairwise SIFT bootstrap (`PairMatch`, essential-matrix step) still needs
*one* frame per camera to feature-match against — multi-frame support is a
per-control-point concept, not a per-bootstrap-step one. Each camera keeps a
designated "reference frame" (default: the first frame scrubbed to) used for
SIFT matching; control points may reference any frame independently of that
reference.

### 3. Fiducial marker detection framework

New module `python/app/setup/fiducial_markers.py`, kept separate from
`extrinsics_solver.py`'s BA code so marker-family-specific logic doesn't leak
into the solver:

```python
@dataclass
class MarkerCornerObs:
    marker_type: str      # "aruco", "charuco", "apriltag", ...
    marker_id: str        # dictionary-specific ID, or "<board>:<corner_id>" for ChArUco
    corner_index: int     # 0-3 for a quad marker; board-corner index for ChArUco
    video_id: str
    frame_idx: int
    px: float
    py: float

@dataclass
class FiducialDetection:
    marker_type: str
    marker_id: str
    corners: list[MarkerCornerObs]        # always 4 for a quad marker
    corner_local_xyz: list[np.ndarray] | None  # marker-local geometry, if size is known

class FiducialDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[FiducialDetection]: ...
```

**`FiducialDetector.detect()` is deliberately per-frame and stateless — it
makes no claim about whether a detected marker is fixed in the scene or
moving.** It answers only "what markers were seen where in *this* frame,"
the same way a pose detector reports keypoints for one frame without judging
whether the tracked person is standing still. Whether a marker's pose should
be treated as constant is a decision made by whatever *consumes* the
detections, not by the detector: this feature's calibration workflow (§6) is
one such consumer, and it chooses to collapse a static board/marker's
detections across a session into one fixed pose. Two future consumers of the
exact same detector output would decide differently — continuous per-frame
extrinsics correction (checking a static marker's apparent position every
frame to catch a camera that's drifted or been bumped) and marker-based
motion capture (a marker deliberately attached to something that moves, read
as a per-frame time series like any other tracked point) — and neither would
require changing this detection layer, only adding a new consumer of it.
This is called out explicitly so a future moving-marker feature is a new
consumer, not a rework of `FiducialDetector`/`FiducialDetection`.

- **`ArucoDetector`** wraps `cv2.aruco.ArucoDetector` with a configurable
  dictionary (e.g. `DICT_4X4_50`). Accepts an optional `size_by_id: dict[str,
  float]` (physical marker side length in metres) plus a single default
  size; markers without a known size still detect and contribute corner
  correspondences, just without a metric/rigidity constraint (R6's "size
  optional").
- **`CharucoDetector`** wraps `cv2.aruco.CharucoDetector`, configured with
  dictionary + `squares_x`/`squares_y`/`square_length`/`marker_length`.
  Because a ChArUco board's geometry is fully known, every detected corner
  already has an exact board-local `(x, y, 0)` — no size ambiguity at all.
  Also exposes `estimate_board_pose(detection, K, dist) -> (R, t)` (a
  `solvePnP` call) for the origin/axis anchoring in §4.
- **Future**: `AprilTagDetector`, same `FiducialDetector` protocol, different
  backend (e.g. `pupil-apriltags`). Adding it touches only this module; the
  BA and storage code operate on `FiducialDetection`/`MarkerCornerObs`,
  never on marker-family-specific types.

Integration with control points: detected corners become a new
`FiducialControlPoint` alongside today's `ControlPoint`
("`ManualControlPoint`" would be the more precise name for the existing
class, but renaming it is optional — the important part is a real type
distinction, not a name):

- **ChArUco corners, anchored** (§4 below performed): `world_xyz` fixed from
  board geometry through the anchor pose — behaves exactly like a manual
  `world_xyz` control point today, just generated automatically and far
  denser (dozens of corners vs. a handful of manual clicks).
- **ChArUco corners, not anchored**, or **ArUco corners without size**: free
  (no `world_xyz`), but a *perfectly labeled* correspondence across camera
  views — this can supplement or outright replace the SIFT bootstrap
  wherever a board/marker is visible, since there's no matching ambiguity
  and no RANSAC needed.
- **ArUco marker with known size**: contributes a **rigid group** residual,
  not four independent points (see §5) — this is the concrete mechanism
  behind requirement R6's "should act like groups of control points."

### 4. Coordinate-system anchoring from a ChArUco board

The existing three-step manual similarity transform (distance → scale,
point/plane → origin, three floor points + forward point → axes,
`docs/extrinsics-calibration-design.md` "Similarity transform") remains
available for boardless setups. When a ChArUco board is present, add a
single action that replaces all three steps at once:

1. User picks one camera + frame with a confident board detection.
2. `CharucoDetector.estimate_board_pose` solves the board's pose (R, t) in
   that camera's frame via `solvePnP` (intrinsics are already known).
3. Every corner's board-local `(x, y, 0)` — from *any* camera/frame where
   that same physical corner was detected — maps through this one pose to a
   world `xyz`, becoming a fixed `FiducialControlPoint`.

Because square size is known, this single action fixes scale, origin, and
axes together, correctly and consistently — the user chooses which board
corner is the origin and which board axis maps to which world axis (e.g.
"board lies flat on the floor, board-Z is world-up") as the only manual
input this step needs.

### 5. Rigid marker-group residual (new BA parameter block)

`bundle_adjustment` (`extrinsics_solver.py`) currently optimizes camera
`rvec`/`tvec` and free-point `xyz`. A size-known ArUco marker adds a
**per-(marker_id, frame-group) 6-DOF pose** parameter (its own `rvec`/`tvec`,
same shape as a camera pose), with each observed corner's residual computed
as: transform the marker's known local corner offset (`±size/2` square) by
this pose, project into the observing camera, compare to the detected pixel.
This is structurally the same kind of parameter block the BA already has
(camera poses), just attached to a marker instead of a camera — no new
residual *type*, a new *instance* of the existing camera-pose-shaped
residual pattern.

A marker observed by only one camera in one frame is under-constrained (6
DOF, only 8 pixel measurements, but they're coplanar so the pose has a
well-known ambiguity) — same caveat that already applies to `solvePnP` for a
single-view planar target. In practice this only matters when a marker is
visible in ≥2 cameras simultaneously, which is the expected common case for
scene fiducials placed to be visible from around the room.

Unknown-size markers skip this residual entirely and are treated as four
independent free points, as ChArUco-without-anchor corners already are.

This is flagged in *Open questions* below as worth a small synthetic-data
prototype before committing — it's new BA machinery, not just new input
data, and the project's own precedent
(`docs/extrinsics-calibration-design.md`'s "Checkpoint" gating) is to
validate solver changes against real or synthetic data before building UI
on top of them.

### 6. Storage: scene fiducial markers vs. manual control points

This is the question the feature request explicitly flagged, and the two
things really are different in kind:

- **Manual control points** (today's `ControlPoint`/`save_control_points`
  JSON) are ephemeral, per-session working state — ad hoc named points with
  no existence outside "the calibration run currently being worked on."
  They stay exactly as they are today: a portable JSON file, not part of the
  database.
- **Scene fiducial markers**, as this feature defines and uses them, are
  physically real objects fixed in the capture space for the life of a
  session. Their entire value (per the feature request: "recalibration is
  easy after modifying camera rig") is that a *future*, different
  calibration run can look them up and reuse their known pose — this is a
  **result to persist and query**, not a working file to remember to
  re-load by hand. This scoping is deliberate: `scene_fiducial_markers`
  below stores one pose per marker because that is exactly what this
  feature needs, not because a marker's pose is assumed constant in
  general. A future feature reading markers as a per-frame time series
  (continuous extrinsics correction, or marker-based motion capture of a
  moving object — see §3) is a different consumer with a different
  storage shape (more like `pose_observations`, keyed by frame, than a
  single fixed row) and is out of scope here; nothing in this table's design
  should be read as ruling that out later.

**Recommendation: a new session-DB table, not a sidecar file.** Every other
calibration artifact in this project already lives in the DB
(`extrinsic_calibrations`, `intrinsics_calibrations`) rather than as a
bespoke file the user manages — a marker-pose table is the same shape of
artifact and should follow the same convention. It also directly supports
the actual query the recalibration workflow needs ("give me marker X's last
known pose in this session"), which a file-based option would need the user
to remember to locate and load by hand.

```sql
CREATE TABLE IF NOT EXISTS scene_fiducial_markers (
    id                        TEXT PRIMARY KEY,
    session_id                TEXT NOT NULL REFERENCES mocap_sessions(id),
    marker_type               TEXT NOT NULL,   -- 'aruco' | 'charuco' | 'apriltag' | ... (open enum, like extrinsic_calibrations.method)
    marker_id                 TEXT NOT NULL,   -- dictionary ID, or "<board_name>" for a whole ChArUco board's own pose
    size                      REAL,            -- physical side length in metres; NULL if unknown
    R                         BLOB NOT NULL,   -- little-endian float64[9], row-major, world orientation
    t                         BLOB NOT NULL,   -- little-endian float64[3], world position
    source_extrinsic_calibration_id TEXT REFERENCES extrinsic_calibrations(id),
    updated_at                TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS scene_fiducial_markers_unique
    ON scene_fiducial_markers (session_id, marker_type, marker_id);
```

Scoped to the session, not the registry, for the first version — a rig's
physical markers plausibly outlive one session (multiple capture days in the
same room), which would argue for registry-level scoping analogous to
`camera_models` vs. `camera_instances`. That generalization is deferred
until there's a concrete second-session reuse case to design against,
consistent with this project's stated preference for scoping a new
mechanism to what's actually needed rather than a broader schema change
made speculatively (see `CLAUDE.md`'s "automation vs. prior human edits"
principle for the same shape of reasoning applied elsewhere in this
codebase). Session-scoping is strictly the smaller, reversible choice: a
later registry-level table can be populated from existing session rows,
but the reverse is not true.

`marker_type` is an open free-text enum rather than an integer/FK, matching
the existing convention for this kind of field
(`extrinsic_calibrations.method`, `tracking_runs.method`) — adding
`apriltag` later needs no migration.

**Superseded (2026-08-12) by §10's `scene_marker_bodies`.** The
per-*marker* schema above predates Phase 8/9's real implementation and the
marker-body-definition format work — §10 revises this to one row per
*body instance* (a rigid multi-marker body needs only one solved pose, not
one per marker) and generalizes `marker_type`/`marker_id` into a proper
`marker_body_definitions` reference for anything beyond a lone tag. The
reasoning above (ephemeral CPs vs. a persisted, queryable result; why a DB
table and not a file) still holds — only the table shape changed.

### 7. Recalibration workflow

1. Starting a new calibration in a session that already has
   `scene_fiducial_markers` rows offers **"Load scene markers as fixed
   control points"** — each stored marker becomes a fixed-`world_xyz`
   `FiducialControlPoint` group seeded from its persisted pose and size; the
   user only needs to get the same physical markers detected in the new
   footage, not re-establish world coordinates or re-click anything.
2. If a marker was moved or removed since it was last solved, its
   detected-vs-predicted reprojection error will be visibly high in the
   existing per-point residual reporting (`CalibResult.cp_reprojection_errors`,
   `extrinsics_solver.py:59`) — surface these as flagged/warned rather than
   silently trusting stale data, the same "don't silently trust possibly-
   stale automated state, surface it instead" shape of decision already
   made for hand-redetection
   (`docs/roadmap/features/hand-detection-refinement/hand-detection-refinement-design.md`,
   "Auto-detect vs keep existing state").
3. After a successful solve, upsert `scene_fiducial_markers` for every
   marker involved: new marker → insert; existing marker → overwrite pose
   unconditionally. The table represents "current believed pose," not a
   history, so there is no reason to keep stale rows around once a fresh
   solve has produced a new one.

### 8. Global timeline scrub (convenience layer over §1)

Added after Phase 1/2 UI testing (2026-08-09): §1 gives each camera its own
independent scrub bar, which is exactly right for placing control points at
different frames per camera (R3/R4). But by the time extrinsics calibration
runs, the capture has almost always already been through the session's sync
wizard page — the same `SyncTable` that page uses to line up all cameras on
one global clock already exists and is queryable
(`DBContext.get_active_sync(shot_id)`, `db_context.py:776`). Finding a good
calibration *moment* (e.g. a frame where a person or board is clearly
visible and stationary in every view) is currently a per-camera hunt-and-
peck across N independent sliders; a global timeline scrub that jumps every
camera to its locally-synced frame for the same instant would make that a
single drag, with each camera's own slider still available afterward for
per-point, per-camera fine adjustment exactly as before.

- **UI**: one additional scrub control above the per-camera grid, labeled
  something like "Global timeline (synced)". Dragging it computes, for each
  camera with sync data, that camera's own frame via
  `SyncTable.lookup(timestamp_s, shot_video_id)` (the same call
  `page_sync.py`'s `_frame_at_playhead` already makes,
  `page_sync.py:1634`) and calls that camera's existing `VideoScrubBar.seek()`
  — it does not replace the per-camera sliders, only drives them to a shared
  starting point.
- **Real implementation gotcha, worth flagging now rather than discovering
  mid-implementation**: `SyncTable` is keyed by `shot_video_id` — i.e.
  `capture_videos.id`, the video *row's* own primary key — while
  `CamCalibState.video_id` (§1, and everywhere else in this feature) is the
  camera *label* (`_load_states_from_capture` sets
  `video_id=r["cam_label"]`, matching the pre-existing
  `_load_states_from_images` convention). Wiring the global scrub needs an
  explicit `capture_videos.id → camera label` lookup (one extra query
  against `capture_videos`/`camera_instances`, the same join
  `_load_states_from_capture` already does) to translate between the two —
  not a blocker, just not a direct drop-in of `SyncTable.lookup()`'s result.
- **Time range**: derive the global slider's min/max from
  `SyncTable.frame_to_global_time()` (`db_context.py:156`) applied to each
  camera's own `first_frame`/`last_frame`, taking the union across cameras
  — a camera that starts recording later than another shouldn't shrink the
  range for cameras that were already rolling, it should just clamp (via
  `VideoScrubBar.seek()`'s existing clamping) when the global position falls
  outside that particular camera's own footage.
- **Graceful degradation**: `get_active_sync()` returns `None` when a
  capture has no sync config yet (e.g. calibration done before the sync
  step, or sync never run for this capture) — the global scrub bar should
  simply not appear in that case, falling back to today's per-camera-only
  scrubbing with no behavior change.
- **Out of scope for this addition**: the global scrub is a convenience for
  finding a shared moment, not a new control-point concept — it doesn't
  change `ObsPoint`, the file format, or anything solver-facing from §2.

### 9. Multi-instrument world-frame anchoring (added after Phase 4 live testing)

Live testing of Phase 4 (2026-08-09, see `status.md`'s fourth and fifth
live-testing rounds) surfaced two structural risks in relying on a single
flat ChArUco board as the *sole* mechanism for establishing the world
coordinate frame, not just implementation bugs within that mechanism:

1. **Spatial concentration bias.** A board that occupies a small region of
   the capture volume gives the bundle adjustment a point cluster with
   almost no depth variation and a narrow visual-angle spread. BA weights
   every pixel residual equally, so it fits that tight cluster very well —
   small angular errors in the recovered camera orientation are invisible
   at the board's own distance, but get lever-armed into large positional
   errors anywhere else in the working volume. This isn't a detection bug;
   it's inherent to anchoring from one small, planar object, however well
   it's detected.
2. **Single point of failure.** World-frame anchoring depended on every
   camera either detecting the board directly or chaining through one that
   did. One camera's detection failing for a physical reason (surface
   reflections, in the round that prompted this section) silently
   propagated a flipped or disconnected pose through the whole chain, with
   only a colored table cell as evidence — see §4's `init_poses_pnp` planar-
   pose-ambiguity handling and its `C_z > 0` heuristic, which cannot recover
   from this because it assumes a *correct* axis convention already exists,
   not because the ambiguity-handling itself is broken.

There is also a real-world portability constraint driving the shape of the
fix: sessions frequently happen at remote locations (this project's stated
use case is aikido, captured wherever a dojo/room is available), which rules
out a large purpose-built calibration frame as the primary instrument — it
has to travel well.

**Revised strategy: three complementary anchoring tiers, all expressed
through the `ControlPoint.world_xyz` fixed/free mechanism §2–§5 already
define — no new BA residual type, only new sources of fixed points and one
new way to seed `init_poses_pnp`.**

#### Tier A — Portable non-planar calibration rig (primary anchor)

A foldable, physically compact rig carrying ArUco markers across multiple
**non-coplanar** faces (e.g. a tent/pyramid fold, a hinged multi-panel
frame, an L-shape) — small enough to travel, opens to a rig geometry with
real depth variation when in use.

- Modeled with `cv2.aruco.Board` — the *generic* board class (distinct from
  `CharucoBoard`), which accepts arbitrary per-marker 3D corner coordinates
  rather than a flat grid. The rig's layout (each marker ID's four corners
  in a rig-local frame) is a small, versionable config: `MarkerRigConfig`
  (new, `fiducial_markers.py`) — `{rig_id, marker_corners: dict[marker_id,
  4×xyz]}` — loaded from JSON the same way `save_control_points`/
  `load_control_points` already version their file format. Measured once
  from the rig's physical dimensions/fold geometry (a hinge-angle + panel-
  size calculation) or itself established once via a precise one-off
  multi-camera capture and reused thereafter; either way this is a one-time
  cost per physical rig, not a per-session one.
- `MarkerRigDetector` (new, `fiducial_markers.py`) wraps `ArucoDetector` +
  `cv2.aruco.Board` + `solvePnP`, exposing `estimate_rig_pose(detections, K,
  dist) -> (R, t) | None` — same shape as `CharucoDetector`'s
  `estimate_board_pose`. Detecting only a subset of the rig's markers (folded
  partway, or some faces occluded from a given camera) is expected and
  handled the same way `CharucoDetector.detectBoard()` already tolerates
  missing corners — `cv2.aruco.Board.matchImagePoints()` works from whichever
  marker corners were actually found.
- Because the marker set spans genuine depth, not a plane, there is **no
  IPPE-style tilt/planar ambiguity to resolve at all** — this removes the
  entire class of failure found in Phase 4 live testing (§4's "no positive-Z
  solution" case), rather than adding another layer of disambiguation on top
  of it.
- Anchoring action generalizes §4's `anchor_from_charuco_board` into
  `anchor_from_marker_rig`: same shape (rig-local corner `xyz` → world `xyz`
  via the one solved rig pose → fixed `ControlPoint`s), different geometry
  source. Sets scale + origin + axes from a single rig detection, same as
  the board does today.
- ChArUco is **not removed** — it remains available as a supplementary,
  *non-anchoring* accuracy aid once the world frame is otherwise established
  (its corner grid is denser than a handful of rig-marker corners, useful
  for local refinement), and as a boardless-rig fallback for setups that
  don't have the rig yet. What changes is that it is no longer the
  recommended *sole* mechanism for fixing the world frame.

**Self-describing rig via an embedded QR code.** Rather than the user
having to locate and load the right `MarkerRigConfig` file for whatever
physical rig is in front of them (a manual pairing that's easy to get wrong
once more than one rig exists — a travel rig and a larger fixed-venue one,
say), print a QR code directly on the rig and read the geometry from the
rig itself. `cv2.QRCodeDetector` is already available in this project's
OpenCV build (checked: 4.13, core `objdetect` module — no new dependency,
same situation `cv2.aruco` was already in). This only needs to be read
*once* per rig, the same one-shot nature as picking "one camera + frame
with a confident detection" already is for the ChArUco anchor (§4) — not a
per-camera or per-frame requirement.

The practical way to keep the QR small and reliably scannable is to encode
geometry *parametrically* for common physical shapes rather than as raw
corner coordinates. A box (exactly tomorrow's planned cardboard-box
experiment) needs only its outer dimensions, one marker size, and a
marker-ID-per-face mapping — the four corners of each face's marker follow
automatically from the box geometry, no need to enumerate 3D points at all:

```json
{"v": 1, "shape": "box", "dict": "DICT_4X4_50",
 "dims_m": [0.20, 0.20, 0.20], "marker_size_m": 0.08,
 "faces": {"+x": "12", "-x": "13", "+y": "14", "-y": "15", "+z": "16", "-z": "17"}}
```

This is a compact *source* format that expands into the same
`MarkerRigConfig` (resolved per-marker corner coordinates) the file-based
loader already produces — detection/anchoring code only ever consumes the
resolved form, so it doesn't matter whether that form came from a hand-
edited JSON file or a decoded QR code, and adding a second parametric shape
later (e.g. a tetrahedron/tent fold) touches only the small expansion
function, not the solver-facing code. An `"shape": "explicit"` variant
(raw per-marker corner arrays, matching `MarkerRigConfig` directly) remains
available as a fallback for irregular, non-parametric geometries, at the
cost of a denser/larger QR code. Resolves the "rig config authoring UX"
open question below for the common case; hand-edited JSON remains available
for one-off or irregular rigs where a QR isn't worth designing for.

**Rig geometry from an orbit video (self-calibration).** Both the QR/`"box"`
path above and plain hand-measurement assume the rig's geometry is either
trivial to parametrize or easy to measure with a ruler — not true for an
irregular fold (a tent shape, an L-frame) or for a rig built without
precise fabrication. A third source needs no measurement at all: **a video
of one moving camera orbiting the rig once**, reusing existing solver
machinery rather than adding new math. Sampled frames from that video are
treated exactly like `run_calibration`/`solve_marker_groups` already treat
a multi-camera rig's simultaneous frame — each sampled frame is an
unknown-pose "camera" (same physical device, known intrinsics from its own
prior intrinsics calibration, just a different pose per frame), and the
existing marker-rigid-pose BA solves all per-frame poses and every marker's
rigid pose jointly. Gauge freedom (the whole solve is only defined up to a
rigid transform) is fixed by designating one marker — or the rig's own
geometric center — as the rig-local origin, the same kind of one-off manual
choice §4's board anchoring already asks for ("which corner is the
origin"). The result is a resolved `MarkerRigConfig`, produced directly
from footage instead of typed in by hand or measured with a ruler.

This is new *input*, not new BA machinery — `solve_marker_groups`'s rigid
marker-pose residual (§5, shipped in Phase 3) doesn't care whether the
poses it's jointly solving belong to distinct simultaneous cameras or to
one camera's sequential frames, only that each marker's corners are
observed from ≥2 sufficiently different viewpoints. The one new risk this
does introduce is sequential-SfM drift (frame-chain error accumulating
around a full orbit) that a genuinely simultaneous multi-camera solve
can't have — worth a real cross-check rather than trusting the video method
in isolation: where a synchronized multi-camera capture *also* sees the
rig (as it naturally will during ordinary Tier A anchoring), running
`solve_marker_groups` on that footage gives a second, independent
measurement of the same physical rig's marker geometry to compare the
video-derived config against.

#### Tier B — Scattered ArUco tags (redundancy / mid-session drift recovery)

Ordinary size-known ArUco markers placed around the capture room (not part
of the rig), meant to survive the whole session even if a camera gets
bumped.

- **Already mostly supported.** `solve_marker_groups()` (§5, shipped in
  Phase 3) already recovers a size-known marker's rigid world pose as a
  decoupled post-pass once cameras are solved from Tier A — no new solver
  machinery needed to capture these tags' positions the first time.
- **New, and small**: a "re-anchor one camera from known marker poses"
  action, for the case a camera is physically moved mid-session and
  shouldn't require redoing the whole rig-based calibration. Reuses
  `init_poses_pnp`'s PnP-plus-planar-disambiguation logic (§4), seeded with
  previously-solved tags' corners as fixed control points instead of the
  board/rig. A single scattered tag is still a planar target, so the
  existing `C_z`/IPPE handling in `init_poses_pnp` is directly relevant
  here and should be *reused*, not reimplemented — this is the motivation
  for factoring that disambiguation logic out of `init_poses_pnp` into a
  small shared helper (`_resolve_planar_pnp_pose_ambiguity` or similar) that
  both the existing per-session init path and this new single-camera
  re-anchor path call.
- **This directly closes a gap identified in live testing**: today,
  `MarkerGroup.as_control_points()` (§3) always yields *free* points — there
  is no path from "this marker was detected" to "treat it as a fixed point
  at a known/previously-solved world position." Tier B requires exactly that
  path: once a tag's pose is known (from the Tier-A-anchored solve), its
  corners need to be constructible as fixed `ControlPoint`s the same way
  `anchor_from_charuco_board` already does for board corners.
- Persistence: `scene_fiducial_markers` (§6) already models this generically
  enough (`marker_type` is a free-text enum) — no schema change needed to
  store a scattered tag's solved pose there, or even a whole rig's solved
  placement should that ever prove worth persisting (lower priority, since a
  portable rig is expected to be repositioned fresh each session, unlike
  wall-mounted tags in a venue used repeatedly).

#### Tier C — Manual control points (unchanged)

The existing "World position" panel (surveyed points, hand-clicked per
camera, §2) remains available for ad hoc known points and needs no changes
for this addendum.

### 10. Marker body definitions: format and storage

Added 2026-08-12, while planning the UI/persistence work needed to turn
Phases 8-9's already-validated core mechanism (see status.md) into a real
feature. Phase 8's rig, Phase 9's scattered tags, and (not yet in scope,
but deliberately not foreclosed — see §3's forward-compatibility note) a
future marker-based motion-capture object attached to a moving skeleton
joint are all the same underlying concept: **a named rigid body carrying
one or more fiducial markers at fixed positions in a body-local frame.** A
single scattered tag is the degenerate case (one marker, trivial local
frame). This section defines the file format and storage for that
concept, superseding §6's original `scene_fiducial_markers` sketch (see
that section for a forward pointer) with a design worked out against real
footage and real authoring concerns, not speculatively.

#### Format choice: YAML, mirroring `docs/skeleton-format.md`

This project already has an established, working precedent for "a named,
versioned, human-authored geometric definition, stored as YAML and
imported into the registry DB as a text blob": skeleton definitions
(`docs/skeleton-format.md`, the `skeletons` table, `import_skeleton()`).
Marker body definitions follow the identical pattern rather than
inventing a new one — same authoring workflow (edit a `.yaml` file, import
it), same storage shape (`yaml_content TEXT` in a DB row), and, per
Harri's own stated hope, a plausible future convergence path: a skeleton
joint could eventually reference (or splice in) a marker body's resolved
marker list, since both use the same "named point(s), offset from a local
origin" shape at their core (see "Forward compatibility" below).

#### Canonical schema

The file format is always **fully resolved** — a flat list of markers,
each with an explicit body-local position. There is deliberately no
shape-parsing (no `"box"`, `"tetrahedron"`, etc.) inside the format or its
loader. This revises this design's earlier sketch (§9 Tier A's QR-code
subsection originally proposed a compact `"box"` JSON shape the *loader*
itself would expand): that approach carries a real, hard-to-catch risk —
expanding a parametric shape requires *assuming* how each physical marker
is mounted (which printed corner is "up" on that face), and a wrong
assumption silently rotates that marker's corner correspondence with no
error raised anywhere (this exact risk is why `load_rig_config`'s
implementation deferred the `"box"` shape entirely — see its docstring).
Keeping the loader dumb and the file always literal (mirroring skeleton
YAML's own precedent — `skeleton_loader.cpp` never interprets "generate a
humanoid," it only ever walks a flat joint list) moves shape generation to
a separate, inspectable, testable **offline generator** step whose
*output* — not its intent — is what actually gets loaded and trusted. A
compact QR-embeddable descriptor (the original §9 idea) can still exist,
just as one *input* format for such a generator, not as something the
marker-body loader parses directly.

```yaml
name: "aikido-calib-box-v1"          # unique, stable identifier for this physical design
description: "Foldable cardboard calibration rig, ~50x33x31cm"  # optional, documentation only
units: "meters"

slots: ["top"]                        # OPTIONAL -- only present for a template (see "Templating" below)

markers:
  - name: "top"                       # stable, human-facing label, unique within this body.
                                       # Used for UI/debug/logging -- never used to match a real
                                       # detection (that's type+dictionary+id, see below).
    type: "aruco"                     # "aruco" | "apriltag" | "reflective_dot" | ... (open enum,
                                       # mirrors FiducialDetection.marker_type already in fiducial_markers.py)
    dictionary: "DICT_4X4_50"         # only meaningful for coded types (aruco/apriltag)
    id: "4"                           # the decoded marker id within `dictionary` -- the ONLY field
                                       # a real detection is matched against, together with
                                       # type+dictionary. String, matching MarkerCornerObs.marker_id.
    size: 0.145                       # side length, metres (aruco/apriltag only)
    center: [0.0, 0.0, 0.0]           # body-local marker-plane center
    normal: [0.0, 0.0, 1.0]           # body-local outward face normal (unit vector)
    up: [0.0, 1.0, 0.0]               # body-local "up" edge hint (~perpendicular to normal;
                                       # Gram-Schmidt-corrected at load time, same approach the
                                       # box rig config's gravity-up reframing already used --
                                       # doesn't need to be exact)
    # corners: [[x,y,z], ...]         # ALTERNATIVE to center/normal/up -- raw (4,3) corners in
                                       # the same TL/TR/BR/BL order marker_local_corners() uses.
                                       # Overrides center/normal/up if both given. This is the
                                       # provenance-preserving form for solved (not designed)
                                       # geometry -- e.g. what an orbit-video self-calibration
                                       # produces directly, with no lossy round-trip through an
                                       # idealized planar description.

  - name: "dot_top_left"
    type: "reflective_dot"
    center: [0.041, 0.038, 0.002]     # a point only -- no size/normal/up/id/dictionary; a
                                       # reflective dot is never decoded, so it has no identity of
                                       # its own beyond its name and its position in this body's
                                       # own known geometry (see "Reflective dots" below).
```

Required/optional fields are **type-discriminated** (aruco/apriltag need
orientation+size+id+dictionary; `reflective_dot` needs only a point),
mirroring skeleton format's own precedent of joint-`type`-specific fields
(`axis` for `revolute`, `limits.x/y/z` for `spherical`) rather than
inventing a new pattern.

#### `name` vs. `id` — why both exist

An earlier draft of this schema used one field for both jobs and was
caught as a real conflict (Harri, 2026-08-12): a body carrying markers
from two different dictionaries (exactly this project's actual box —
ArUco corners *and* reflective dots, see below) can have two markers that
legitimately share the same numeric `id` in different dictionaries, or a
human author may simply want a friendlier label than a bare dictionary id.
`name` is the body-author's own stable identifier (drives UI/debug/
`ControlPoint` naming); `id`, only meaningful together with
`type`+`dictionary`, is the literal value a real detection is matched
against. Any future loader/detector code keys its detection-matching
lookup on `(type, dictionary, id)`; `name` never participates in that
lookup.

#### Reflective dots

The project's actual calibration box already carries both ArUco markers
and reflective dots (added 2026-08-11 for a future marker-based
motion-capture direction, per §3's forward-compatibility note). A
reflective dot has no decodable identity — its correspondence across
cameras/frames can only come from the body's own known rigid geometry,
once *some* marker on the same body (an ArUco corner, typically) has
already solved a pose for it. This is exactly why dots belong in the
*same* body definition file as the coded markers rather than a separate
format: the coded markers anchor the body, and the dots ride along,
predictable from that one shared pose — the same mechanism conventional
reflective-marker motion-capture systems use. This isn't designed further
here (matching, tracking, and reprojecting dots per-frame is squarely the
future marker-based-mocap feature this design has repeatedly deferred,
most recently in §3/§9) — only the *definition format*'s ability to
describe a dot's position is in scope now.

#### Templating (deferred, syntax reserved only)

A real use case, not yet needed (Harri, 2026-08-12): multiple physical
copies of the same rig *shape* (e.g. several identical cubes),
distinguished only by which marker ids are printed on each copy. Building
the full mechanism now would be speculative — consistent with this
project's repeated preference (see §6's original scoping reasoning, and
CLAUDE.md's "automation vs. prior human edits" principle for the same
shape of call elsewhere) for deferring a generalization until a concrete
second case exists. What *is* worth doing now is reserving the syntax so
a future loader change is additive, not a breaking rewrite of files people
have already authored:

- `slots:` at the top of a definition lists symbolic names an *instance*
  must supply real ids for.
- A marker's `id:` becomes a structured `{slot: "name"}` reference instead
  of a literal string — unambiguous by YAML node shape (a mapping, not a
  scalar), not by string-sigil convention.
- A definition using only literal `id:` values (today's case,
  exclusively) is immediately usable and self-contained; one using
  `slot:` references is a **template**, not usable until a (not-yet-built)
  `marker_body_instances` table supplies a slot→id mapping.

Nothing about today's format or loader needs to anticipate the instance
tier beyond accepting this reserved shape; building `marker_body_instances`
is out of scope until a second physical copy of some rig actually exists.

#### Storage: registry-scoped `marker_body_definitions`

```sql
CREATE TABLE IF NOT EXISTS marker_body_definitions (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    yaml_content TEXT NOT NULL,
    source       TEXT,             -- e.g. "orbit_video_self_calibration" or an authoring file path
    created_at   TEXT NOT NULL,
    notes        TEXT
);
```

Directly mirrors `skeletons` (`id`, `yaml_content TEXT NOT NULL`, `notes`)
— same import shape (`import_marker_body()` reading a `.yaml` file,
analogous to `import_skeleton()`). Registry-scoped, not session-scoped
(unlike `skeletons`, whose `person_label`/`parent_id` columns suggest
per-person calibrated variants): a physical rig's *design* doesn't get a
per-session variant the way a skeleton's bone lengths do — it's copied
into a session DB unchanged, on demand, the same way
`camera_models`/`intrinsics_calibrations` already are.

#### Storage: session-scoped `scene_marker_bodies` (supersedes §6's `scene_fiducial_markers`)

Revised from §6's per-*marker* sketch to per-*body-instance*: a rigid
multi-marker body only ever needs **one** solved pose — every marker's
world position then follows from `definition + pose`, exactly matching
how `anchor_from_marker_rig` already treats a whole rig as a single
transform. A lone scattered tag needs no `marker_body_definitions` row at
all (its local geometry is always just `marker_local_corners(size)` —
nothing to author), so `marker_body_definition_id` is nullable, with
inline columns covering that common case directly (agreed with Harri,
2026-08-12 — a bespoke one-marker YAML definition per scattered tag id
would be pure authoring overhead for zero benefit):

```sql
CREATE TABLE IF NOT EXISTS scene_marker_bodies (
    id                         TEXT PRIMARY KEY,
    session_id                 TEXT NOT NULL REFERENCES mocap_sessions(id),
    label                      TEXT NOT NULL,   -- user-chosen, e.g. "calib-box", "wall-tag-north"
    marker_body_definition_id  TEXT REFERENCES marker_body_definitions(id),  -- NULL for a lone tag
    marker_type                TEXT,            -- only used when marker_body_definition_id IS NULL
    dictionary                 TEXT,            -- ditto
    marker_id                  TEXT,            -- ditto
    marker_size                REAL,            -- ditto
    R                          BLOB NOT NULL,   -- body-local -> world rotation, float64[9] row-major
    t                          BLOB NOT NULL,   -- body-local -> world translation, float64[3]
    is_primary_anchor          INTEGER NOT NULL DEFAULT 0,
    source_extrinsic_calibration_id TEXT REFERENCES extrinsic_calibrations(id),
    updated_at                 TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS scene_marker_bodies_unique
    ON scene_marker_bodies (session_id, label);
```

This single table covers every scenario this design needs:
- The calibration rig itself: one row, `R = I, t = 0, is_primary_anchor =
  1` (its own local frame *is* the world frame for that capture, per Tier
  A's anchoring mechanism).
- Each scattered tag: one row per tag, `marker_body_definition_id` NULL,
  pose from `solve_marker_pose()`.
- Cross-capture reuse (Tier B / Phase 9): a later session's solve reads
  these rows as fixed anchor input — exactly what
  `test_reanchor_capture2.py` already does today from a hand-carried file
  (see status.md's 2026-08-12 entries), minus the manual file-carrying.

#### Forward compatibility with skeleton joints

Not designed here — explicitly future work, same as §3/§6 already flag
for marker-based motion capture generally — but the per-marker fields
above (`name`, body-local `center`) were chosen close enough to skeleton
format's own marker fields (`name`, `offset`) that a future feature could
plausibly let a skeleton joint reference a `marker_body_definitions` row,
or splice its resolved markers into the skeleton's own `markers:` list
with `parent: <joint>`, without this format needing to change first.

## Phased implementation plan



### Phase 1 — Video frame source
- Promote `_VideoPane` from `pair_scrubber.py` into a shared module
  alongside `FrameReader` (`video_reader.py`); `PairScrubber` switches to the
  promoted class with no behavior change.
- Replace the directory-glob PNG loading path
  (`_load_states_from_images`, `page_extrinsics.py:122`) with a capture
  picker that resolves each camera's video file and tiles one `_VideoPane`
  per camera in a new grid, bound to `[first_video_frame,
  last_video_frame]` from that camera's `capture_videos` row.
- `_ClickableImageWidget` reads its displayed frame from the corresponding
  pane instead of a single image loaded once from disk.
- `CamCalibState.image` becomes "current scrub position's frame,"
  refreshed on scrub.

**Validation:** open a capture with several cameras, scrub each
independently; verify the displayed frame's content matches the expected
timestamp (spot-check against a visible event in the footage, e.g. a clap or
light change); verify existing manual-control-point click-to-place workflow
is functionally unaffected; verify `PairScrubber` (sync page) still behaves
identically after `_VideoPane` is promoted, since it's now a shared
dependency rather than a page-local class.

### Phase 2 — Per-control-point, per-frame observations
- `ControlPoint.obs` → `dict[video_id, ObsPoint]` (`frame_idx`, `px`, `py`);
  update BA glue and `save_control_points`/`load_control_points` (bump file
  version to 2; a version-1 file loads with `frame_idx` defaulting to the
  camera's current scrub position).
- UI: placing/re-placing a control point on a camera captures that camera's
  current scrub position; re-placing the same point on the same camera at a
  different frame overwrites its `ObsPoint`.

**Validation:** place one control point using frame 100 in camera 1 and
frame 250 in camera 2; verify BA output is unchanged from placing both at
the "same" logical point in different frames; verify a saved file round-trips
frame indices; verify loading a version-1 file still works.

### Phase 3 — ArUco marker detection
- `fiducial_markers.py`: `FiducialDetector` protocol + `ArucoDetector`.
- UI: "Detect ArUco markers" button per camera widget, acting on the
  currently scrubbed frame; configurable dictionary; a size table (global
  default + per-marker-ID override, override row shown only once more than
  one size has actually been entered).
- BA: rigid marker-pose parameter block (§5) for markers with known size,
  observed by ≥2 cameras in the same frame; free-corner fallback otherwise.

**Validation:** using footage with a printed marker of known physical size,
verify solved corner spacing matches the known size within tolerance;
verify a marker with no size entered still contributes usable
correspondences (BA converges, camera poses remain correct) without the
rigidity residual; verify a marker visible in only one camera degrades to
free-point behavior rather than an ill-conditioned single-view pose solve.

### Phase 4 — ChArUco board detection + coordinate-system anchoring
- `CharucoDetector` (dictionary + `squares_x`/`squares_y`/`square_length`/
  `marker_length`, all UI-configurable).
- "Set origin & axes from board" action (§4): pick one camera+frame,
  `solvePnP` the board pose there, propagate `world_xyz` to every detected
  corner of that physical board across all cameras/frames.
- Wire generated corners into the BA as `FiducialControlPoint`s.

**Validation:** calibrate a rig with a ChArUco board visible from ≥3
cameras; compare per-camera reprojection error against today's real-world
workflow on the same footage (manually-placed control points — per the SIFT
finding in *Open questions*, not a SIFT-assisted result that doesn't reflect
actual usage); verify the anchor action reproduces the board's own known
square size when measured in the solved world coordinates.

### Phase 5 — Scene marker persistence + recalibration reuse
- `scene_marker_bodies` table + migration (§10 — supersedes the earlier
  `scene_fiducial_markers` sketch).
- Write path: upsert one row per solved marker body (rig or lone tag)
  after any successful solve that included fiducial markers.
- Read path: "Load scene markers as fixed control points" on a new
  calibration run (via `anchor_from_marker_rig`, no new anchoring code —
  each persisted body's `definition + R/t` is just another
  `MarkerRigConfig` source), plus residual-based flagging of markers whose
  detected-vs-predicted reprojection is high.

**Validation:** run calibration A and verify markers are persisted; move one
camera (simulating a rig change) and run calibration B loading the
persisted markers; verify camera poses solve correctly from the fiducial
anchors alone (no manual control points needed); verify a deliberately
displaced marker is flagged rather than silently trusted.

### Phase 6 — AprilTag backend (extensibility proof)
- `AprilTagDetector` implementing the same `FiducialDetector` protocol.

**Validation:** swap the detector backend via configuration only; verify the
same downstream BA/persistence code path (§3, §5) runs unmodified against
AprilTag input on test footage.

### Phase 7 — Global timeline scrub (§8; UI-testing feedback, not originally scoped)

- Resolve `capture_videos.id → camera label` for the current capture (same
  join `_load_states_from_capture` already does) so `SyncTable` lookups
  (keyed by `shot_video_id`) can drive `VideoScrubBar`s (keyed by camera
  label).
- Add one global scrub control to `ExtrinsicsAutoCalibDialog`, shown only
  when `DBContext.get_active_sync(shot_id)` returns a `SyncTable` (hidden
  entirely otherwise — no sync config yet is a normal, expected state, not
  an error).
- Range: union of every camera's own frame range converted to global time
  via `SyncTable.frame_to_global_time()`.
- Dragging it calls `SyncTable.lookup(timestamp_s, shot_video_id)` per
  camera and seeks that camera's `VideoScrubBar` — each camera's own slider
  remains independently draggable afterward, unchanged from §1/§2.

**Validation:** open a capture that has been through the sync wizard page;
verify the global scrub bar appears and dragging it moves every camera to
the same real-world instant (spot-check against a visible synced event,
e.g. a clap); verify each camera's slider can still be moved independently
afterward without affecting the others (R2 unchanged); open a capture with
no sync config and verify the global scrub bar simply doesn't appear, with
the rest of the dialog behaving exactly as it does today.

### Phase 8 — Portable non-planar calibration rig (§9, Tier A)

**Core already implemented and validated against real capture-1 footage
(2026-08-12, standalone scripts — see status.md).** Remaining work is
switching the loader from the prototype's ad hoc JSON to §10's YAML
format, plus UI wiring:

- `load_marker_body()`: YAML loader for §10's format (`center`/`normal`/
  `up` and `corners` marker forms; `slot:` id references parsed but
  rejected with a clear "template not yet supported" error, per §10's
  deferred-templating note) — replaces `load_rig_config`'s JSON
  `"explicit"` shape. No parametric shape expansion in the loader itself
  (§10) — a `"box"`-shape *offline generator* producing this YAML is
  separate, optional tooling, not a loader responsibility.
- `MarkerRigDetector` and `anchor_from_marker_rig()`: already implemented
  (`fiducial_markers.py`), no changes needed beyond consuming
  `MarkerRigConfig` from the new loader instead of the old one.
- `marker_body_definitions` table + migration (§10) + registry↔session
  copy (mirroring `camera_models`/`intrinsics_calibrations`).
- Rig-from-video self-calibration: prototype already validated
  (`tools/characterize_rig_from_video.py`) against real orbit-video
  footage, cross-checked against a synchronized multi-camera solve of the
  same physical rig (see status.md, 2026-08-11/12) — **not yet promoted
  into a permanent tool or UI action; deliberately deferred, see Open
  Questions below** for what's missing before it should be (output format,
  reference-frame/origin selection, axis orientation).
- UI: "Load rig config" (file or registry picker) + "Detect rig" per
  camera + "Set origin & axes from rig" action, parallel to the existing
  ChArUco controls. QR-code scanning (reading a compact descriptor off
  the physical rig, expanding it into §10 YAML) deferred — not required
  for a first usable version, since a file/registry picker already
  removes the "which config goes with this rig" problem the QR idea was
  meant to solve, at lower implementation cost.

**Validation:** using the real physical rig already characterized this
session, verify the UI's "Detect rig" + "Set origin & axes from rig" flow
reproduces the same solved camera positions/reprojection errors already
recorded in status.md's 2026-08-12 capture-1 test (no regression from
re-implementing against §10's format); verify a single camera's rig
detection never exhibits the Phase-4 "no positive-Z solution" failure
regardless of viewing angle (already confirmed at the script level, per
status.md — this re-confirms it end-to-end through the UI); verify
partial visibility (some faces occluded from a given camera) still
produces a usable pose from whichever markers are visible.

### Phase 9 — Scattered-tag redundancy + single-camera re-anchor (§9, Tier B)

**Multi-camera re-anchor mechanism already validated against real
capture-2 footage (2026-08-12, standalone script — see status.md):**
`anchor_from_marker_rig()` needed zero new code — a set of previously-solved
scattered tags is already just another `MarkerRigConfig` source as far as
that function is concerned. Remaining work is persistence + UI, plus the
still-open single-camera case:

- Per-corner triangulation + reprojection-check for scattered tags (already
  implemented as part of the capture-1 test script,
  `tools/test_rig_anchor_capture1.py`'s `--save-scattered-tags`) becomes
  the real write path once `scene_marker_bodies` (§10, Phase 5) exists —
  same triangulation logic, writing DB rows instead of a JSON file.
- Read path: "Re-anchor from known scene markers" action — loads this
  session's (or a prior session's, once the registry-vs-session scoping
  question below is resolved) `scene_marker_bodies` rows and calls
  `anchor_from_marker_rig()` unchanged.
- **Still not attempted**: the *single-camera* re-anchor case specifically
  (one bumped/moved camera, re-anchored alone without re-running the whole
  session). Unlike the multi-camera case just validated, a *lone* scattered
  tag is still a planar target for that one camera — this is exactly where
  §4's `init_poses_pnp` planar-pose-ambiguity handling (`C_z`/IPPE logic)
  remains relevant and should be factored out into a shared helper, per
  the original phase plan below. Not yet re-evaluated in light of this
  session's findings; still believed necessary.

**Validation:** ~~after a Tier-A-anchored solve with several scattered tags
visible, verify each tag's recovered pose persists correctly~~ — done at
the script level (status.md, 2026-08-12); re-verify through the UI/DB path
once built. Simulate a bumped camera (perturb its pose, re-detect the same
tags from a fresh frame) and verify the single-camera re-anchor action
recovers a pose close to the camera's original one, without touching any
other camera's already-solved pose — this part is still untested at any
level.

## Open questions

- **Rig-from-video self-calibration needs real polish before it's part of
  posetrak proper, not just a validated prototype (Harri, 2026-08-12).**
  `tools/characterize_rig_from_video.py` proved the *mechanism* works
  against real footage (status.md, 2026-08-11/12), but three concrete gaps
  need closing before it should become a permanent tool or UI action:
  1. **Reference marker/origin selection is arbitrary.** `ref_id =
     min(complete_ids)` — whichever marker id happens to be lowest, with
     no regard for which physical face that is. This is exactly what
     produced the box rig config's original wrong-axis frame (status.md's
     "Z was backwards" entry) — a human had to notice it afterward and fix
     it with a separate, one-off script, not something the tool itself
     offered a way to control.
  2. **No axis-orientation control at all.** No way to specify "this
     direction is gravity-up," align to a known-vertical marker, or
     supply a manual override — the tool has no concept of a "correct"
     frame beyond "whatever the reference marker's own local axes happen
     to be."
  3. **Still emits the prototype's ad hoc JSON**, not §10 YAML.
  Not scheduled — deliberately left as future work, since the mechanism
  is validated and there's no immediate need pulling this forward (the
  box rig config in current use was fixed by hand, once, per status.md).
  Revisit if self-calibrating a *new* rig becomes a regular workflow
  rather than a one-off.
- **Registry- vs. session-level scoping** for `scene_fiducial_markers` —
  deferred to a follow-up once a concrete cross-session reuse case exists
  (§6). Confirmed as the right call for now.
- **Rigid marker-pose BA residual (§5) is new solver machinery**, not just
  new input data — worth a small synthetic-data prototype before Phase 3's
  UI work, per this project's existing practice of validating BA changes
  before building on top of them. Confirmed.
- **Video random-seek performance** is resolved, not open: reuse the sync
  page's existing scrub machinery (§1) rather than measuring or optimizing
  seek speed separately. Sync anchor placement already does random seeks
  more often and with a stricter (frame-accurate) requirement than
  extrinsics calibration ever will, so current performance is presumed
  adequate here without dedicated measurement — and sharing the code means
  this feature automatically inherits any future seek optimization made for
  syncing, rather than needing its own.
- **The SIFT pairwise bootstrap is not a baseline worth preserving as-is.**
  Real usage has found it unreliable enough that control points are placed
  almost entirely by hand today — it is not a "sometimes helps" fallback in
  practice. This raises the bar for §3's board/marker correspondences: they
  should be positioned as the primary automatic-correspondence mechanism
  going forward, with SIFT retained only as a legacy/manual-heavy path for
  boardless setups, not as the thing this feature merely "supplements."
  Phase 4's validation should compare against the current all-manual
  workflow (SIFT bootstrap not meaningfully contributing, matching real
  usage), not an idealized SIFT-assisted baseline.
- **Marker size input UX** — a single global default plus a per-ID override
  table (shown once more than one distinct size is entered) is proposed in
  Phase 3; revisit if real usage shows most rigs mix many different sizes.
  Confirmed as good enough for now.
- **Marker-based motion capture is a planned future direction, not a
  hypothetical.** Fiducial markers are expected to eventually be attached to
  moving objects and tracked per-frame, in addition to this feature's static
  scene-fixture use. §3 and §6 already scope the detector layer as
  per-frame/stateless and the persistence layer as this-feature-specific for
  exactly this reason, so that a future moving-marker feature is a new
  consumer of `FiducialDetector` output rather than a redesign of it. Flagged
  here so that assumption stays visible to whoever picks up that future
  work, not just buried in §3/§6's prose.
- **Physical rig geometry (§9, Tier A) is not yet designed.** This addendum
  specifies the *software* shape (`cv2.aruco.Board`, `MarkerRigConfig`,
  detection/anchoring flow) but not the fold pattern, panel count/size, or
  marker placement of the physical object itself — that's an industrial-
  design/fabrication question, not a software one, and is intentionally left
  open here. Whatever shape gets built, the config format only needs each
  marker's corner coordinates in a rig-local frame, so the software side
  doesn't need to change once a physical design is chosen. Note that
  *acquiring* an already-built rig's config is no longer a blocker on this
  question either way — the orbit-video self-calibration method above works
  for any marker layout, parametric or not, so an irregular fold doesn't
  need its own parametric shape primitive just to become usable.
- **Rig config authoring UX — superseded by §10.** The canonical format is
  now §10's YAML (`center`/`normal`/`up` or `corners`, always fully
  resolved); a `"box"`-shape descriptor (QR-embeddable or otherwise) is
  reframed as one *optional offline generator* that produces this YAML,
  not a loader-level shape — see §10's "Canonical schema" for why the
  earlier loader-level approach was revised. The underlying authoring
  paths are otherwise unchanged: hand-measurement, a future parametric
  generator, or orbit-video self-calibration (validated 2026-08-11/12, see
  status.md) all just need to produce §10-shaped YAML. Still open: which
  parametric generators beyond a `"box"` one are worth building (tent/
  pyramid, L-frame) versus always falling back to video self-calibration
  or hand-measurement for anything non-box-shaped.
- **`init_poses_pnp`'s planar-ambiguity helper extraction (§9, Tier B)**
  should land as part of Phase 9, not deferred — both the existing
  multi-camera init path and the new single-camera re-anchor path need
  identical handling, and duplicating it would let them drift (one fixed,
  one not) exactly the way this whole addendum is trying to avoid.
- **Rig pose persistence (§9, Tier A) — no longer a separate deferral,
  superseded by §10.** This was previously deliberately skipped since a
  portable rig is expected to be repositioned fresh each session rather
  than left in place. §10's unified `scene_marker_bodies` table ended up
  covering the rig's own row (`R=I, t=0, is_primary_anchor=1`) at no extra
  schema cost, since every solve needs a body-instance table regardless —
  so the rig's pose *is* persisted now, it's just that nothing yet reads
  it back across sessions (a rig is not currently expected to still be in
  the same physical place next session, so there's no read-path use for
  it yet). Revisit only if real usage shows rigs staying put across
  multiple sessions in the same venue.
- **Templating (§10) is deliberately syntax-only for now** — `slots:`/
  `{slot: "name"}` are reserved in the YAML format, but `marker_body_
  instances` (the table that would resolve a template into a concrete,
  usable definition) is not designed or built. Revisit once a second
  physical copy of some rig design actually exists (Harri, 2026-08-12).
- **Reflective-dot correspondence/tracking (§10) is out of scope beyond
  the definition format.** §10 lets a marker body describe a dot's
  position; nothing yet matches a detected dot in an image to the right
  `name` (no decoded id to key on, unlike ArUco/AprilTag), predicts its
  expected image position from an already-solved body pose, or tracks it
  per-frame for a moving object. This is squarely the future marker-based
  motion-capture feature §3/§6/§9 have repeatedly flagged as deliberately
  out of scope, not a gap in this addendum specifically.
