# Extrinsics Calibration Improvements — Implementation Status

See [extrinsics-improvements-design.md](extrinsics-improvements-design.md) for
the problem statement, requirements, and full technical design.

## Current state

Phase 1 implemented (2026-08-09), grounded against the pre-existing
`python/app/setup/extrinsics_solver.py` / `page_extrinsics.py` /
`posetrak/db/import_extrinsics.py` implementation and
`docs/extrinsics-calibration-design.md`. Phases 2-6 remain design-only.

## Phase summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Video frame source: per-camera random-seek reads, scrub UI replacing PNG-directory loading | ✅ Done |
| 2 | Per-control-point, per-frame observations (`ObsPoint`, file format v2) | ⬜ Not started |
| 3 | ArUco marker detection + rigid marker-pose BA residual | ⬜ Not started |
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

**Scope note**: every control point placed in a session currently uses
whatever frame each camera happens to be scrubbed to *at solve time* —
there is no independent per-control-point frame record yet (that's
Phase 2's `ObsPoint`). Placing point A while camera 1 is on frame 100,
then scrubbing camera 1 to frame 250 before placing point B, will
silently move point A's effective frame too, since today's `ControlPoint`
has no frame field of its own. This is a real, known interim limitation,
not an oversight — flagging it here so it isn't mistaken for Phase 1
already covering R3/R4.

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
`VideoScrubBar` extraction. Worth a follow-up, not addressed here.

## Known open questions (see design doc for detail)

- Registry- vs. session-level scoping for `scene_fiducial_markers`.
- The rigid marker-pose BA residual (Phase 3) is new solver machinery and
  should be prototyped against synthetic data before UI work begins.
- Video random-seek performance on long-GOP consumer codecs (GoPros) is
  unmeasured — check early in Phase 1.
- Whether board/marker corners should replace the SIFT pairwise bootstrap
  outright, vs. only supplement it, is left as an internal heuristic for now.
- Marker size input UX (global default + override table) may need revisiting
  once real rigs are tried.
