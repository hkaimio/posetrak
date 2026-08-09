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
3. **All correspondences are manual clicks.** SIFT gives free (but ambiguous
   and occasionally wrong) point correspondences for the pairwise bootstrap;
   there is no automatic, unambiguous fiducial marker detection, and no way
   to give the solver metrically-known geometry short of manually entering
   `world_xyz` for hand-picked points.

HarrI: In reality SIFT is worse than "occassionally wrong" - I have not found it to be usable in tis current form.

This document proposes: (a) scrubbing calibration frames directly from the
already-known capture videos, with independent frame choice per camera and
per control point; (b) ChArUco board detection, optionally anchoring the
world coordinate system; (c) ArUco marker detection as correlated groups of
control points, usable both as an accuracy aid for calibration itself and,
critically, as **persistent physical scene fiducials** that make
recalibration after a rig change fast; and (d) a detector abstraction so
other marker families (AprilTag, …) can be added without touching the solver.

HArri: Additional possible future use case for atuco markers would be support for moving cameras. I.e. instead of doing extrinsics calibration only once, the tracker app would check at every frame whether the markers have moved and adjust extrinsics. Also, aruco markers maight be used also for object tracking, so it cannot be assumed that all markers remain static between frames at different global time stamps.

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
  shown for camera B.
  Harri: See my commetn above that some aruco markers might intentionally be attached to moving objects
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

**Recommendation: direct per-camera random-seek reads via OpenCV
`VideoCapture`, not the frame-cache infrastructure built for keypoint
editing.** The keypoint-editing feature's `frame_cache_entries` /
`WideCropExtractWorker` machinery exists because that workflow scrubs
*constantly*, across an entire trial, for every person, every session.
Extrinsics calibration is a one-off, low-frequency activity — a handful of
frames per camera, once per rig setup. Building or reusing a persistent
cache is unwarranted complexity for this usage pattern.

Harri: we already have scrubbing in multiple places (e.g. syncing videos, so likely code & UI logic with these can be shared)

Concretely:

- A new small helper (`python/app/setup/video_frame_source.py`) wraps
  `cv2.VideoCapture` per `capture_videos.file_path`, exposing
  `get_frame(frame_idx) -> np.ndarray`, with a small per-camera LRU (a few
  dozen frames) since a user scrubbing near one spot will re-request nearby
  frames repeatedly.
- Each camera gets its own scrub widget (slider + spin box) bound to
  `[first_video_frame, last_video_frame]` from that camera's
  `capture_videos` row. Seeking is debounced (seek-on-release or after a
  short idle gap while dragging) to avoid hammering the decoder.
- **Known risk, flagged for early validation, not solved here**: random
  seeks into long-GOP consumer codecs (GoPro H.264/H.265) can be slow —
  potentially hundreds of ms per seek depending on keyframe interval. Phase
  1 should measure this against real capture footage before committing
  further; if unacceptable, fall back to decoding a bounded window around
  the last position rather than true random access.
- `CamCalibState.image` (`extrinsics_solver.py:37`) changes meaning from "the
  one loaded image" to "the frame currently displayed in this camera's
  widget" — refreshed on every scrub, not loaded once.

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
- **Scene fiducial markers** are physically real objects fixed in the
  capture space. Their entire value (per the feature request: "recalibration
  is easy after modifying camera rig") is that a *future*, different
  calibration run can look them up and reuse their known pose — this is a
  **result to persist and query**, not a working file to remember to
  re-load by hand.

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

## Phased implementation plan

### Phase 1 — Video frame source
- `video_frame_source.py`: per-camera random-seek reads keyed by
  `capture_videos.file_path` + frame index, small LRU cache.
- Replace the directory-glob PNG loading path
  (`_load_states_from_images`, `page_extrinsics.py:122`) with a capture
  picker that resolves each camera's video file and opens a per-camera
  scrub widget (slider + spin box bound to `[first_video_frame,
  last_video_frame]`) instead of a single static image.
- `CamCalibState.image` becomes "current scrub position's frame,"
  refreshed on scrub.

**Validation:** open a capture with several cameras, scrub each
independently; verify the displayed frame's content matches the expected
timestamp (spot-check against a visible event in the footage, e.g. a clap or
light change); verify existing manual-control-point click-to-place workflow
is functionally unaffected; measure random-seek latency against real
capture footage (flags the codec-seek-speed open question below).

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
cameras; compare per-camera reprojection error against the current
SIFT+manual-control-point baseline on the same footage; verify the anchor
action reproduces the board's own known square size when measured in the
solved world coordinates.

### Phase 5 — Scene marker persistence + recalibration reuse
- `scene_fiducial_markers` table + migration.
- Write path: upsert marker poses after any successful solve that included
  fiducial markers.
- Read path: "Load scene markers as fixed control points" on a new
  calibration run, plus residual-based flagging of markers whose
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

## Open questions

- **Registry- vs. session-level scoping** for `scene_fiducial_markers` —
  deferred to a follow-up once a concrete cross-session reuse case exists
  (§6).

  Harri: OK with that
- **Rigid marker-pose BA residual (§5) is new solver machinery**, not just
  new input data — worth a small synthetic-data prototype before Phase 3's
  UI work, per this project's existing practice of validating BA changes
  before building on top of them.
  Harri: OK
- **Video random-seek performance** on long-GOP consumer codecs (GoPros) is
  unmeasured; Phase 1 should check this early and fall back to a bounded
  prefetch window if true random access proves too slow.
  Harri: Other parts of the app (setting sync frames at least) already use random seek. It is slow but I think the need for random seeks in exterinsics calibration is smaller and less frequent that for sync frames 8which by definiton require frame accurate placement) -> I think current perf is OK for this but the scrubbing machinery should share code & UI logic with syncing videos for UI consistency & so that it inherits possible future optimizations.
- **Whether board/marker corners should outright replace the SIFT pairwise
  bootstrap** when present, rather than just supplementing it, is likely
  scene-dependent (textureless environments benefit most). Left as an
  internal solver heuristic (prefer labeled correspondences when available)
  rather than a user-facing toggle for the first version.
  Harri: As I said, I haven't found the SIFT bootstrap useful in practice. I almost never use it.
- **Marker size input UX** — a single global default plus a per-ID override
  table (shown once more than one distinct size is entered) is proposed in
  Phase 3; revisit if real usage shows most rigs mix many different sizes.
  Harri: OK for now.

Harri: As I hinted in many places, marker based mcoap is something that I am plannign as future addition. So fiducial markers will then be used both for calibration and attached to moving objects. Worth keeping in mind so that there are no big surprises/redesign needed.
