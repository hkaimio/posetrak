# `tracking_obs_results` semantics are measurement-mode-dependent — and unmarked

**Status: fixed at the data-model level (2026-09-05).** Filed 2026-08-23
after a real bug: the skeleton scaling dialog
(`python/app/ui/skeleton_scaling_panel.py`) reported a measured femur
length in the thousands of centimetres against a real capture. Originally
worked around per-consumer only (skeleton scaling); this file's own "Open
question" section is now resolved -- see §2026-09-05 below.

## The problem

`tracking_obs_results.obs_blob` is `float32[n_cams, n_markers, 8]`, one slot
per (camera, marker, step):

```
[actual_x, actual_y, pred_x, pred_y, mahal_dist, used, is_outlier, pad]
```

`used`/`is_outlier` reliably answer "did the tracker's own Mahalanobis gate
accept this observation" — that part of the contract is sound and is exactly
what a downstream tool should read from this table.

`actual_x, actual_y` (and `pred_x, pred_y`) are a different story. They are
whatever the tracker's measurement model, `h(x)`, compared against —  and
`h(x)` is not always "project the 3D point to an absolute undistorted pixel
position." Per `cpp/include/posetrak/core/observation.hpp`, three modes
exist and all three get written into the same two float slots:

- `POSITION` (the default): `h(x) = project(x)`. `actual_x/y` is a genuine
  absolute undistorted pixel coordinate.
- `VELOCITY` (`tracker_configs.velocity_mode_camera_ids`, an escape hatch
  for a camera whose absolute calibration is less trusted than its
  frame-to-frame consistency): `h(x_t) = project(x_t) - project(x_t-1)`.
  `actual_x/y` is a frame-to-frame pixel *delta*, typically a few pixels.
- `PAIR_DIFF` (`tracker_configs.use_relative_observations`, applied per
  marker whenever both the marker and its kinematic parent marker clear a
  confidence threshold that frame): `h(x_t) = project(child, x_t) -
  project(parent, x_t)`. `actual_x/y` is a child-minus-parent pixel offset.

**Nothing in the stored blob says which mode produced a given slot.** The
8th field exists (`pad`) but is unused. Reading `actual_x/y` as an absolute
position is only safe for `POSITION`-mode slots, and there's no way to tell
which slots those are after the fact — not even by inspecting the values
(a small, plausible-looking delta or offset is indistinguishable from a
small, plausible-looking position near the image origin).

This is exactly what happened: `insta_ace2_pro` was in `VELOCITY` mode for
the affected run (confirmed unintentional — Harri meant to turn it off),
and the run's tracker config also had `use_relative_observations` on, which
converted every child marker (knee relative to hip, ankle relative to knee,
elbow relative to shoulder, ...) into `PAIR_DIFF` offsets whenever
confidence allowed. The skeleton scaling dialog triangulated those values as
if they were positions.

## The fix applied (skeleton scaling only)

`_MeasWorker` (`skeleton_scaling_panel.py`) now uses `tracking_obs_results`
*only* for its `is_outlier`/`used` verdict — which (camera, marker, step)
observations the tracker trusted. The actual 2D point triangulated for each
accepted observation is read from the original `pose_observations` row
instead (undistorted the same way the tracker itself would, via
`_undistort_point`, mirroring `Camera::undistort()`). Raw detections are
unambiguously always real pixel positions, regardless of which measurement
mode the tracker went on to use them in — so re-deriving from there
sidesteps the ambiguity entirely rather than needing to detect or reconstruct
which mode applied.

This also subsumed two smaller fixes that were tried first and are still
worth keeping for their own sake, independent of the mode-ambiguity issue:

- A `VELOCITY`-mode camera's data is now moot for this dialog specifically
  (raw detections don't care what mode the tracker used), but the general
  lesson holds elsewhere: don't assume every active camera's
  `tracking_obs_results` slot is a position.
- `_robust_triangulate`: DLT retried with the worst-reprojecting camera
  iteratively dropped, for the separate (and real, mode-independent)
  problem of a low-confidence detection that the tracker's own outlier gate
  didn't catch (inflated assumed noise from low confidence can make even a
  wildly-off detection look statistically unsurprising).

## 2026-09-05 — the open question resolved, and two deeper bugs it surfaced

Streak-derived dot velocity (`docs/roadmap/features/marker-based-mocap/
streak-velocity-design.md` §4) is the first thing in this codebase to give
a single (camera, marker) *two* Observations in the same step -- a
POSITION one and a VELOCITY one, deliberately alongside each other, not a
mode mutating in place the way the existing `velocity_mode_camera_ids`
escape hatch does. That "at most one Observation per (camera, marker) per
step" assumption turned out to be baked into three places, not one:

1. **This file's original open question**: `write_obs_results()`
   (`result_writer.cpp`) writes into one diagnostic slot per (camera,
   marker), last-observation-in-the-vector wins. With two Observations
   sharing a slot, the VELOCITY one's *displacement* (not a position)
   could silently overwrite the POSITION one's real pixel coordinate --
   caught visually as dot markers rendered nowhere near any actual
   detection in `render_tracking_debug_frames.py` output (Harri's own
   catch: predicted markers tracked the real sword correctly while
   "actual" markers floated over blank wall/door, nowhere near any
   possible detection -- the tell that this was a data bug, not a
   real-but-wrong correspondence, was that a genuinely wrong match would
   still land *on some real feature*, not in empty space).
2. **`UnscentedKalmanFilter::update()`'s innovation-recompute pass**:
   matched "this observation" between the full and inlier-only lists by
   `(marker_id, camera_id, frame_idx)` alone -- no longer unique once two
   Observations can share that triple, risking one's post-outlier-
   rejection predicted/innovation landing on the *other*'s
   `ObservationResult`.
3. **`reject_outliers()`'s cross-camera consistency check**: grouped
   observations by `marker_id` alone to compare Mahalanobis distances
   across cameras. A POSITION-mode distance (absolute-pixel residual) and
   a VELOCITY-mode one (pixel-delta residual) are not the same physical
   quantity -- pooling them into one median is statistically meaningless,
   and unlike the other two, **this one could affect which observations
   the real UKF update actually accepts**, not just a diagnostic table.

### The fix

- `ObservationResult` gained a `mode` field (mirroring the source
  `Observation::mode`), threaded through at all three construction sites
  in `ukf.cpp`.
- The innovation-recompute match key became `(marker_id, camera_id,
  frame_idx, mode)`.
- The cross-camera-consistency grouping key became `(marker_id, mode,
  ref_marker_id)` -- `ref_marker_id` included pre-emptively: a PAIR_DIFF
  child marker measured relative to two *different* parents (a real future
  case once cross-marker relative observations exist -- see below) would
  otherwise still get pooled incorrectly even with `mode` alone.
- `write_obs_results()` now writes the measurement mode into the
  previously-unused `pad` field (0=POSITION, 1=VELOCITY, 2=PAIR_DIFF,
  matching `MeasurementMode`'s declaration order), and a POSITION
  observation always wins a (camera, marker) slot over a VELOCITY/
  PAIR_DIFF sibling when both exist -- order-independent by construction
  (a flag per slot records once a POSITION has claimed it, checked before
  any later write, not "last write wins").

  **Caveat, checked and found narrow rather than fixed**: `pad` turned out
  to already carry an *established, different* meaning on hierarchical-
  solver runs -- `ResultWriter::patch_obs_results()` (called from
  `hierarchical_solver.cpp` once a child stage's PAIR_DIFF observations get
  reconstructed to absolute positions) writes a caller-supplied
  `pair_diff_reconstructed` boolean into the same field, unrelated to
  `obs.mode`. Left untouched (no live consumer reads `pad` today for
  either meaning, confirmed by search, so nothing breaks either way) --
  but this means a slot a hierarchical child stage later patches will have
  its `write_obs_results()`-written mode value overwritten by the
  reconstructed flag instead, so `pad` is genuinely ambiguous between the
  two schemes on such a run. Not urgent to reconcile today: streak
  velocity applies to rigid-prop dot tracks, hierarchical solving to
  articulated body/hand skeletons -- the two don't combine in any real
  capture yet. Worth a real decision (a second field, or a single
  consistent "is this actual_x/y trustworthy as a position" boolean both
  writers agree on) before they do.

### Forward note for cross-marker relative observations (person + prop)

The fix above makes today's shape safe: one marker, up to one POSITION
observation plus one non-POSITION one, sharing a slot with POSITION
winning. It does **not** yet solve the harder case raised while designing
this fix: person+prop integration wants PAIR_DIFF observations *between*
two markers that each also carry their own independent POSITION
observation and possibly their own separate diagnostic value worth keeping
(e.g. a sword-tip dot's PAIR_DIFF against a fingertip marker, alongside
both markers' own ordinary tracking). `tracking_obs_results`' one-slot-per-
(camera,marker) shape has no room to keep a cross-marker PAIR_DIFF
residual *and* that marker's own POSITION diagnostic simultaneously --
today's fix drops the PAIR_DIFF one from this table when both exist,
matching "POSITION wins," same as it now does for VELOCITY. If per-slot
history of every Observation (not just whichever one "wins") turns out to
matter for debugging person+prop contact once that's built, `obs_blob`
would need a real schema change (e.g. the same variable-length,
count-prefixed shape the 'dots' blob already uses, rather than a fixed
8-float slot) -- a decision to make deliberately when that feature is
actually designed, not preemptively here.
