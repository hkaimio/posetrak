# `tracking_obs_results` semantics are measurement-mode-dependent — and unmarked

**Status: documented, worked around once (skeleton scaling); not fixed at
the data-model level.** Filed 2026-08-23 after a real bug: the skeleton
scaling dialog (`python/app/ui/skeleton_scaling_panel.py`) reported a
measured femur length in the thousands of centimetres against a real
capture.

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

## Open question this doesn't resolve

Should `tracking_obs_results` record the measurement mode per slot (the
spare `pad` field is sitting right there), so any *other* consumer of this
table — the MCP diagnostic server's `get_observation_gaps`/
`get_filter_stats` tools, some future analysis script — doesn't have to
either know to route around it (as done here) or risk the same silent
misinterpretation? That's a real, contained C++ change
(`result_writer.cpp`, wherever `Observation.mode` is available at write
time) plus updating documented consumers, but it touches a format other
tooling already depends on and wasn't done as part of this fix. Worth
deciding deliberately rather than as a side effect of the next thing that
trips over it.
