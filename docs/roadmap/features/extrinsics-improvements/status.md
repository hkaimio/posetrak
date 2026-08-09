+++
name = "Extrinsics Calibration Improvements"
status = "in_progress"
progress_pct = 40
description = """
Improvements to multi-camera extrinsic calibration: scrubbing calibration frames directly from \
capture video instead of a pre-extracted PNG folder, per-control-point per-frame observations, \
ArUco/ChArUco marker detection to anchor the coordinate system and provide a rigid marker-pose \
bundle-adjustment residual, and persisted fiducial markers for recalibration reuse.
"""
categories = ["calibration", "ui"]
target_release = "TBD"
last_updated = 2026-08-09
+++

# Extrinsics Calibration Improvements — Implementation Status

See [extrinsics-improvements-design.md](extrinsics-improvements-design.md) for
the problem statement, requirements, and full technical design.

## Current state

Phases 1 and 2 implemented (2026-08-09), grounded against the pre-existing
`python/app/setup/extrinsics_solver.py` / `page_extrinsics.py` /
`posetrak/db/import_extrinsics.py` implementation and
`docs/extrinsics-calibration-design.md`. Phases 3-6 remain design-only.

## Phase summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Video frame source: per-camera random-seek reads, scrub UI replacing PNG-directory loading | ✅ Done |
| 2 | Per-control-point, per-frame observations (`ObsPoint`, file format v2) | ✅ Done |
| 3 | ArUco marker detection + rigid marker-pose BA residual | 🔶 In progress (synthetic-data residual prototype validated; detection/UI/`run_calibration` integration not started) |
| 4 | ChArUco board detection + coordinate-system anchoring | ⬜ Not started |
| 5 | `scene_fiducial_markers` persistence + recalibration reuse | ⬜ Not started |
| 6 | AprilTag detector backend (extensibility proof) | ⬜ Not started |

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

## Phase 3 progress — synthetic-data BA prototype (not yet Phase 3 proper)

Implemented as one commit, `setup: synthetic-data prototype for the rigid
marker-pose BA residual`, answering the "Open questions" item below before
starting Phase 3's actual ArUco-detection/UI work:

- `marker_local_corners(size)`, `project_marker_corners(...)`, and
  `solve_marker_pose(corner_obs, states_by_id, size)` added to
  `extrinsics_solver.py`, **deliberately not wired into `run_calibration`**
  — a marker's four corners are treated as one rigid 6-DOF pose parameter
  (the same shape as a camera pose's own rvec/tvec), recovered via
  `least_squares` over every camera's corner residuals jointly, seeded by a
  single-camera `solvePnP`.
- Scope of the prototype: camera poses are fixed (as if already solved by
  the rest of `run_calibration`) — only the marker's own pose is being
  recovered. Joint refinement of camera + marker poses together is Phase
  3's actual integration work, not this prototype's job.
- Validated against a synthetic 3-camera rig
  (`test_marker_pose_prototype.py`, 12 cases, all passing): exact recovery
  (four marker orientations, atol 1e-5) from noise-free projections; 2
  cameras are sufficient; a single camera correctly raises; a camera with
  no solved pose is correctly excluded from the ≥2-camera count rather
  than crashing; convergence holds under 0.5px Gaussian pixel noise; and
  the result doesn't depend on a good initial guess.
- **Result: the residual math checks out.** This resolves the "worth a
  small synthetic-data prototype before committing" open question — Phase
  3 proper can now wire this into `run_calibration`'s parameter vector with
  reasonable confidence in the underlying math, rather than discovering a
  sign error or a Jacobian issue after building UI on top of it.

## Known open questions (see design doc for detail)

- Registry- vs. session-level scoping for `scene_fiducial_markers`.
- ~~The rigid marker-pose BA residual (Phase 3) is new solver machinery and
  should be prototyped against synthetic data before UI work begins.~~
  Done — see "Phase 3 progress" above.
- Video random-seek performance on long-GOP consumer codecs (GoPros) is
  unmeasured — check early in Phase 1.
- Whether board/marker corners should replace the SIFT pairwise bootstrap
  outright, vs. only supplement it, is left as an internal heuristic for now.
- Marker size input UX (global default + override table) may need revisiting
  once real rigs are tried.
