```toml
name = "Extrinsics Calibration Improvements"
status = "in_progress"
progress_pct = 55
description = """
Improvements to multi-camera extrinsic calibration: scrubbing calibration frames directly from \
capture video instead of a pre-extracted PNG folder, per-control-point per-frame observations, \
ArUco/ChArUco marker detection to anchor the coordinate system and provide a rigid marker-pose \
bundle-adjustment residual, and persisted fiducial markers for recalibration reuse.
"""
categories = ["calibration", "ui"]
target_release = "TBD"
last_updated = 2026-08-09
```

# Extrinsics Calibration Improvements — Implementation Status

See [extrinsics-improvements-design.md](extrinsics-improvements-design.md) for
the problem statement, requirements, and full technical design.

## Current state

Phases 1, 2, and 3 implemented (2026-08-09), grounded against the
pre-existing `python/app/setup/extrinsics_solver.py` / `page_extrinsics.py`
/ `posetrak/db/import_extrinsics.py` implementation and
`docs/extrinsics-calibration-design.md`. Phase 3 landed with one
significant, prominently-flagged scoping deviation from the design doc's
section 5 — see "Phase 3 notes" below before assuming it matches that
section literally. Phases 4-6 remain design-only.

## Phase summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Video frame source: per-camera random-seek reads, scrub UI replacing PNG-directory loading | ✅ Done |
| 2 | Per-control-point, per-frame observations (`ObsPoint`, file format v2) | ✅ Done |
| 3 | ArUco marker detection + rigid marker-pose BA residual | ✅ Done, with a scoping deviation — see "Phase 3 notes" (decoupled post-pass, not a joint BA parameter block) |
| 4 | ChArUco board detection + coordinate-system anchoring | ⬜ Not started |
| 5 | `scene_fiducial_markers` persistence + recalibration reuse | ⬜ Not started |
| 6 | AprilTag detector backend (extensibility proof) | ⬜ Not started |
| 7 | Global timeline scrub (§8) — jump every camera to the same synced instant | ⬜ Not started (design added 2026-08-09 from UI-testing feedback) |

## UI testing feedback (2026-08-09, Phases 1-2)

Harri ran the Phase 1/2 UI checklist. One bug found and fixed, one
improvement proposal captured as Phase 7 above (§8 in the design doc).

- **Bug, fixed** (`setup: fix "Go to..." frame-seek button using PyQt-style
  keyword names`): `VideoScrubBar._on_goto()`'s `QInputDialog.getInt()` call
  used `min=`/`max=` (the PyQt5/6 keyword spelling); PySide6 names the same
  parameters `minValue=`/`maxValue=` and raised `AttributeError: unsupported
  keyword 'min'` on every click. This was carried over verbatim from
  `pair_scrubber._VideoPane`'s original `_on_goto()` (confirmed via `git
  log` — predates the `VideoScrubBar` extraction), never caught before
  because no prior test exercised the real `QInputDialog.getInt()` call.
  Fixed; new tests call the real API (via a `QTimer.singleShot` to close
  the resulting modal dialog) to catch this class of bug going forward, not
  just a stub.
- **Improvement proposal, captured as design (Phase 7 / §8), not yet
  implemented**: since a capture has almost always already been through the
  sync wizard page by the time extrinsics calibration runs, a single global
  timeline scrub — driving every camera's `VideoScrubBar` to its own
  locally-synced frame for the same instant, via the existing `SyncTable` —
  would replace "hunt for a good calibration moment across N independent
  sliders" with one drag, while leaving each camera's own slider available
  afterward for per-point fine adjustment exactly as today.
- Everything else on the Phase 1/2 checklist: OK.

## Phase 1 notes

Implemented as two commits:

- `setup: extract VideoScrubBar shared component from pair_scrubber` —
  pulled the slider/label/"Go to…"/`FrameReader`-lifecycle logic out of
  `pair_scrubber._VideoPane` into a new, display-agnostic
  `VideoScrubBar` (`app/setup/video_scrub_bar.py`), per the design's
  "reuse the sync page's scrubbing machinery" recommendation.
  `PairScrubber`'s public API and behavior are unchanged.
- `setup: scrub extrinsics calibration frames directly from capture
  video` — `CamCalibState` gained `file_path`/`first_frame`/`last_frame`;
  new `_load_states_from_capture()` resolves cameras directly from
  `capture_videos` (no PNG filename matching); `ExtrinsicsAutoCalibDialog`
  gives each video-backed camera its own `VideoScrubBar`, updating both
  the `_ClickableImageWidget` and the underlying `CamCalibState.image` in
  place as the user scrubs. The PNG-directory workflow is kept as a
  secondary "Auto-calibrate (image folder)…" action, per the design
  doc's R1 ("may remain as an alternate/legacy path").

**Scope note (resolved in Phase 2, see below)**: every control point placed
in a session used to use whatever frame each camera happened to be scrubbed
to *at solve time* — there was no independent per-control-point frame
record. Placing point A while camera 1 was on frame 100, then scrubbing
camera 1 to frame 250 before placing point B, would silently move point A's
effective frame too, since `ControlPoint` had no frame field of its own.
Flagged at the time so it wasn't mistaken for Phase 1 already covering R3/R4
— Phase 2's `ObsPoint` is exactly the fix.

**Known pre-existing issue found while testing, not fixed (out of
scope for this change)**: three tests in `test_pair_scrubber.py`
(`test_key_left_steps_target_backward`, `test_key_right_steps_target_forward`,
`test_shift_right_steps_target_by_10`) fail on a clean checkout with no
Phase 1 changes applied — confirmed via `git stash` before starting this
work. They leave a `FrameReader` `QThread` running past the failing
assertion (no `ps.shutdown()` reached), which can later abort the whole
`pytest` process with "QThread: Destroyed while thread still running"
when enough such leaks accumulate across a full suite run. Reproduced
against the same file/test suite pre-Phase-1; unrelated to the
`VideoScrubBar` extraction. Still present and still unrelated after Phase
2 (re-confirmed the same three, and only those three, fail when running
Phase 2's test files alongside `test_pair_scrubber.py`). Worth a follow-up,
not addressed here.

## Phase 2 notes

Implemented as one commit, `setup: track per-control-point, per-camera
frame index`:

- `ObsPoint(frame_idx, px, py)` replaces the plain `(px, py)` tuple in
  `ControlPoint.obs`. `frame_idx` is provenance for the UI and the saved
  file only — every solver-facing consumer (`init_poses_pnp`,
  `_undistort_control_obs`, `compute_cp_errors`, the BA's observation list)
  reads only `.px`/`.py`, proven by a synthetic-camera PnP test that solves
  to a bit-identical pose regardless of what `frame_idx` values the
  observations carry.
- `ExtrinsicsAutoCalibDialog._on_cam_click` now records the clicked
  camera's *current* `VideoScrubBar` position into the new observation;
  re-clicking the same point on the same camera at a different scrub
  position overwrites its `ObsPoint` — R4's "different/later frame for the
  same point in the same camera" case.
- `save_control_points`/`load_control_points` bumped to file version 2
  (`obs` values become `[frame_idx, px, py]`); version 1 files still load,
  with `frame_idx` defaulting to a caller-supplied `default_frame_by_id`
  (wired to each camera's current scrub position when loading from the
  dialog) or 0 if none is given.

**Known limitation, not addressed here**: `_refresh_markers` still draws
every placed control point's marker on whatever frame a camera is
*currently* displaying, regardless of which frame that observation was
actually recorded on (`obs.frame_idx`). This was already true before Phase
2 in a lesser form (there was no `frame_idx` to compare against at all);
Phase 2 makes the mismatch meaningful without resolving it — a marker whose
own point moved between frame 100 and frame 250 will still be drawn at its
frame-100 pixel position even while the camera is scrubbed to frame 250,
which can look wrong for exactly the occlusion/motion case R4 was written
for. Not part of Phase 2's stated scope (data model + file format +
placement), but worth a follow-up: e.g. only drawing a point's marker when
the camera's displayed frame matches `obs.frame_idx`, or dimming/badging it
otherwise.

## Phase 3 notes

### Synthetic-data BA prototype (preparatory step, landed first)

Implemented as one commit, `setup: synthetic-data prototype for the rigid
marker-pose BA residual`, answering the "Open questions" item below before
starting Phase 3's actual ArUco-detection/UI work:

- `marker_local_corners(size)`, `project_marker_corners(...)`, and
  `solve_marker_pose(corner_obs, states_by_id, size)` added to
  `extrinsics_solver.py` as a self-contained prototype — a marker's four
  corners are treated as one rigid 6-DOF pose parameter (the same shape as
  a camera pose's own rvec/tvec), recovered via `least_squares` over every
  camera's corner residuals jointly, seeded by a single-camera `solvePnP`.
- Scope of the prototype: camera poses are fixed (as if already solved by
  the rest of `run_calibration`) — only the marker's own pose is recovered.
- Validated against a synthetic 3-camera rig
  (`test_marker_pose_prototype.py`, 12 cases, all passing): exact recovery
  (four marker orientations, atol 1e-5) from noise-free projections; 2
  cameras are sufficient; a single camera correctly raises; a camera with
  no solved pose is correctly excluded from the ≥2-camera count rather
  than crashing; convergence holds under 0.5px Gaussian pixel noise; and
  the result doesn't depend on a good initial guess.
- **Result: the residual math checks out.** This resolved the "worth a
  small synthetic-data prototype before committing" open question before
  any UI work began, per the design doc's own recommendation.

### Full implementation (two commits)

`setup: ArUco marker detection + solver integration (Phase 3, data/solver
layer)` and `setup: wire ArUco marker detection into the extrinsics
calibration UI (Phase 3, UI layer)`:

- New `app/setup/fiducial_markers.py`: `MarkerCornerObs`, `FiducialDetection`,
  `FiducialDetector` protocol, and `ArucoDetector` (wraps
  `cv2.aruco.ArucoDetector`; configurable dictionary, optional
  `default_size`/`size_by_id`). Corner order (top-left, top-right,
  bottom-right, bottom-left) was verified against real `cv2.aruco` output
  via `cv2.aruco.generateImageMarker`-rendered test images, not just
  assumed to match the prototype's `marker_local_corners()` convention —
  it does.
- `extrinsics_solver.MarkerGroup`: one physical marker's corner
  observations across cameras; `as_control_points()` represents its 4
  corners as independent free `ControlPoint`s.
- `ExtrinsicsAutoCalibDialog`: a "Detect ArUco" button per camera pane
  (video- or image-sourced), an "ArUco Markers" panel (dictionary combo,
  default-size spin box, per-marker size-override table, "Clear markers"),
  marker corners drawn as a gold overlay alongside manual control points,
  and `marker_groups` threaded through to `run_calibration` on solve.

### Scoping deviation from the design doc — read this before assuming §5's joint-BA design shipped as originally sketched

**Every detected marker's corners — known size or not — contribute to
camera-pose solving as four independent free control points**
(`MarkerGroup.as_control_points()`), reusing the existing,
already-tested free-CP/DLT-triangulation machinery. This alone delivers
real R6 value: automatic, correctly-labeled correspondences, no manual
clicking, usable to supplement or substitute for a weak SIFT bootstrap.

**What did *not* ship**: the design doc's section 5 sketched a marker's
corners as *replacing* four free points with a single rigid 6-DOF
parameter *inside* `run_bundle_adjustment`'s own joint optimization, so a
known-size marker could also help *correct* camera poses via its metric
constraint. That would mean injecting a new parameter type into
`run_bundle_adjustment`'s existing dense parameter-vector/Jacobian
machinery — real, risky surgery on tested, complex numerical code.
Instead, known-size markers get a **decoupled post-pass**
(`solve_marker_groups()`, called once after camera poses are already
solved): their corners still go through the free-CP path like every other
marker during the main solve, and *afterward*, using those now-fixed
camera poses, the validated `solve_marker_pose()` prototype recovers a
clean, correctly-scaled rigid pose — stored in the new
`CalibResult.marker_poses`, ready for Phase 5's persistence.

**Practical consequence**: a known-size marker's metric information does
not currently help *improve* camera pose accuracy beyond what its 4 free
corner points already contribute (same as an unknown-size marker); it only
produces a clean rigid pose *after* the fact. If real-world testing shows
camera poses need that extra constraint (e.g. too few SIFT features and
only a couple of known-size markers to anchor scale), true joint
optimization is the flagged follow-up — not attempted here, deliberately,
to keep this phase's risk bounded to already-validated pieces (the
free-CP path and the synthetic-data-tested `solve_marker_pose`).

### Test coverage

`test_fiducial_markers.py` (15 cases, real `cv2.aruco` round-trips, no
mocks), `test_marker_groups.py` (10 cases, including a full
`run_calibration` integration test using locked/pre-posed synthetic
cameras + `cp_only=True` so it needs no real footage or SIFT), and
`test_extrinsics_aruco_ui.py` (14 cases, dialog-level: detect button,
table population, scrub-frame capture, size override persistence across
re-detection, redetect-overwrite, multi-camera accumulation, marker/CP
overlay coexistence, and a spy-based confirmation that Match & Solve
actually forwards marker groups to `run_calibration`).

### Not addressed in Phase 3 (unchanged from the open questions below)

- The SIFT pairwise bootstrap is untouched — ArUco corners are *added*
  alongside whatever SIFT correspondences exist, not used to replace SIFT
  outright. Still a live open question (see below).
- Per-marker size override UX ships as an always-visible table column
  rather than the design doc's "shown only once more than one distinct
  size has been entered" progressive-disclosure idea — functionally
  equivalent, simpler to implement, not revisited.
- No manual UI validation yet against a real printed marker / real
  multi-camera footage (Phase 3's stated validation criterion in the
  design doc) — everything above is unit/integration-tested against
  synthetic data and rendered test images, not exercised live in the app.

## Known open questions (see design doc for detail)

- Registry- vs. session-level scoping for `scene_fiducial_markers`.
- ~~The rigid marker-pose BA residual (Phase 3) is new solver machinery and
  should be prototyped against synthetic data before UI work begins.~~
  Done — see "Phase 3 notes" above.
- Video random-seek performance on long-GOP consumer codecs (GoPros) is
  unmeasured — check early in Phase 1.
- Whether board/marker corners should replace the SIFT pairwise bootstrap
  outright, vs. only supplement it, is left as an internal heuristic for
  now — Phase 3 did not touch this either way (see "Phase 3 notes").
- Marker size input UX (global default + override table) shipped as an
  always-visible table in Phase 3, not the progressive-disclosure version
  originally sketched — may still need revisiting once real rigs are tried.
- **New from Phase 3**: whether the decoupled marker-pose post-pass's
  accuracy is sufficient in practice, or whether known-size markers need
  to become a true joint BA parameter block to meaningfully improve camera
  pose accuracy (not just produce their own clean pose after the fact).
  Only real-footage testing can answer this.
