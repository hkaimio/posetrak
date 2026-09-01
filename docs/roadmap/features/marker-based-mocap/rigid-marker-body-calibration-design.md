# Rigid marker-body calibration from multi-camera footage

Scope note (2026-08-31): triggered by a real capture — "Weapon test
2026-08-20", trial "Harri bokken" — a sword/bokken prop with ArUco markers
on both flat faces (4x4 dictionary, no other markers from that dictionary
in the scene) plus 7 reflective dots (4 on one face, 3 on the other). No
marker is ever visible from both faces at once, so the existing
single-camera turn-around approach used for props whose geometry is known
by construction (the calibration box) cannot characterize this rig at all.
This document scopes a tool to solve it from the capture's own 6-camera
footage instead. Not started — scoping only, per Harri's request.

## 1. Why the turn-around method is a dead end here

A single-camera turn-around is structure-from-motion: it needs a
correspondence linking the two halves of the rotation together, either a
marker seen across the transition or independently-known camera motion
through it. A double-sided flat object gives neither — no marker is ever
visible from both faces (true of *any* flat prop, not a symptom of sparse
placement), and a plain turn-around video has no external pose reference to
chain through the edge-on moment. Two disconnected marker clusters, no way
to relate their coordinate frames. Better footage doesn't fix this; it's a
different problem needing a different method.

## 2. Why a calibrated multi-camera rig sidesteps it

The two marker sets sit on opposite faces of a flat object, so their
surface normals are anti-parallel by construction. A rig that surrounds the
performer (not just faces them) will, at most orientations that aren't
exactly edge-on, have cameras on one side of the room see face A while
cameras on roughly the other side simultaneously see face B — in the same
synchronized frame. Since all cameras already share one calibrated world
frame (this capture's own solved extrinsics), that's enough to link the two
faces: no single camera, and no continuity through an edge-on transition,
is required.

The second thing working in this method's favor: **an ArUco marker is a
self-contained pose reference.** Its four corners are a known planar square
by construction, so triangulating one visible marker's corners from ≥2
cameras gives the prop's *full 6-DOF pose* at that instant directly, with
no dependency on which other markers happen to be visible. This means the
calibration doesn't need "face A and face B markers in the same frame" —
it only needs, across the whole capture and for every marker being placed:
some frame where that marker is visible together with *any* decodable
ArUco marker (either face, doesn't matter which). Each such co-occurrence
directly gives that marker's offset in the prop's body frame.

## 3. Algorithm

Given a capture with solved camera extrinsics/intrinsics and a time range
over which the prop is handled (no special turn-around recording needed —
ordinary performance footage is the input):

1. **Per-frame, per-camera detection.** ArUco: existing
   `ArucoDetector`/`MarkerRigDetector` (`app/setup/fiducial_markers.py`),
   unchanged. Reflective dots: **new** — see §4, gap 1.
2. **Per-frame anchor pose.** For each frame, for each decodable ArUco
   marker with ≥2 cameras seeing all 4 corners, solve its rigid world pose
   via **`extrinsics_solver.solve_marker_pose()`** — already implemented,
   already used by the extrinsics-calibration path for exactly this
   (single-camera PnP seed + multi-camera nonlinear refine against the
   marker's own known-size planar template, `least_squares(method="lm")`,
   returns RMS reprojection error too). No new code for this step.
3. **Per-marker offset extraction.** For every marker (dot or other
   ArUco) with a triangulated/PnP'd world position at a frame where some
   reference ArUco's pose is *also* available that frame:
   `offset = T_anchor_to_world(t)⁻¹ · world_position(marker, t)`. For an
   ArUco marker being calibrated (needs orientation, not just a point, per
   the `center`/`normal`/`up` fields `marker_body_definitions` YAML
   expects — see §10 marker-body infra), extract the full relative
   transform `T_marker_local_to_body = T_anchor(t)⁻¹ · T_marker(t)`, not
   just its center.
4. **Robust aggregation (seed).** Per marker, robust average (trimmed
   mean or geometric median) of its offset samples across all frames —
   same idea as the existing person-marker plan's §5.1 "Reference-window
   attach" seed step, minus that plan's dependency on markerless tracking
   (an ArUco's own known geometry supplies the per-frame reference pose
   here instead of a tracked skeleton). Orientation samples (for
   ArUco-type markers) need SE(3)-aware averaging (e.g. quaternion mean),
   not naive per-axis averaging.
5. **Joint least-squares refine (recommended).** With seed offsets as the
   initial guess, jointly refine over unknowns = {each marker's local
   offset/transform} ∪ {each sampled frame's prop pose
   `T_anchor_to_world(t)`}, minimizing total reprojection error across all
   markers/cameras/frames with a Huber robust loss — structurally the same
   problem as the person-marker plan's §5.2 whole-trial refinement, and the
   same residual-function shape `solve_marker_pose()` already uses per
   marker per frame, just with more unknowns in one solve. New code, but a
   direct extension of an established, working pattern in this file
   (`project_marker_corners()` already computes exactly this residual for
   one marker/frame/camera — the joint version stacks these across every
   marker and frame instead of solving them independently).
6. **Output.** Write a `marker_body_definitions` YAML: one `type: aruco`
   entry per coded marker (dictionary/id/size/center/normal/up from the
   refined local transform) and one `type: reflective_dot` entry per dot
   (center only) — the existing loader (`fiducial_markers.py`) already
   accepts both types; no schema change needed. Feeds directly into the
   existing `posetrak marker-body import` → `to-skeleton` pipeline
   unchanged.

## 4. What's genuinely new vs. reused

Reused as-is: `ArucoDetector`, `solve_marker_pose()`,
`project_marker_corners()`, `marker_local_corners()`, camera-state loading
from a capture's solved extrinsics (pattern already in `page_extrinsics.py`),
the `marker_body_definitions` YAML format and its import/`to-skeleton` CLI.
This is a larger fraction of the total problem than it first looked —
the hard per-marker pose-solving numerics already exist, tested, in
production use for the (different, but structurally identical) extrinsics-
calibration path.

Genuinely new:

1. **A reflective-dot blob detector.** Nothing in the codebase detects
   dots today (design doc's phase 2 scope, not yet built). For calibration
   specifically the bar is lower than live production tracking: a human
   can review/curate the calibration session's frames, there's no
   cross-frame identity/association problem to solve (only per-frame,
   per-camera 2D centroids feeding straight into triangulation — dot
   *labeling* across frames isn't needed since each frame's triangulated
   points are matched to body-local offsets independently, not tracked
   over time). A basic threshold + connected-components + centroid pass
   (or `cv2.SimpleBlobDetector`) is likely enough to start.
2. **Offset extraction + robust aggregation** (§3 steps 3–4). Small,
   maybe a hundred lines; no novel math, but doesn't exist yet.
3. **The joint least-squares refine** (§3 step 5). New, but a direct
   structural extension of `solve_marker_pose()`'s own residual pattern —
   low risk, not a new numerical method for this codebase.
4. **The orchestrating CLI/script** tying detection → per-frame anchor
   pose → aggregation → YAML output into one offline tool (likely a new
   `posetrak marker-body calibrate` subcommand, sibling to the existing
   `import`/`to-skeleton`).

## 5. Phasing

- **Phase A — ArUco-only.** Steps 1 (ArUco only) through 6, no dots. No
  new detector needed at all — pure reuse plus the small new aggregation
  glue (§4.2). This is the fastest path to something real, and it
  directly de-risks §2's core assumption (does the real capture actually
  have enough cross-face co-occurrence frames?) before investing in the
  dot detector.
- **Phase B — joint refine.** Add §3 step 5 once Phase A's seed method is
  validated end-to-end on real data. Likely worth folding into Phase A
  rather than deferring — the math isn't large, and validating the seed
  step alone doesn't tell you whether the refine is needed.
- **Phase C — reflective dots.** Build the blob detector (§4.1), extend
  offset extraction/aggregation to dot markers (mostly the same machinery,
  simpler per-marker unknowns: 3 DOF, not 6).

## 6. Validation before building anything

**Confirmed 2026-09-01, against the real capture.** Ran a plain
`ArucoDetector` (no rig config, no filtering — just whatever DICT_4X4_50
IDs are actually there) across all 6 cameras over the full trial
(34.4–100.6s, ~20 fps effective sampling; read-only, no DB writes). Two
IDs dominate and match Harri's description exactly (one per face): `2` and
`3`, both seen on nearly every camera. They co-occur constantly, not as an
edge case — 227 distinct 0.05s time-buckets (out of ~1,320 sampled) had
both visible simultaneously from different cameras, starting immediately
at t=35.5s and repeating on almost every sample for several seconds
straight (e.g. `t=36.05s`: camera `50819e8c` sees `2`, camera `a7f65681`
sees `3`, same synchronized instant). This is exactly §2's argument playing
out on real data: whenever the sword is being handled normally, opposite-
side cameras catch both faces together. The core assumption this whole
approach rests on is no longer speculative — Phase A has real data to work
with.

Two other IDs turned up rarely (`10` once on one camera, `17` a few times
on two cameras) — plausible false-positive decodes or something briefly in
frame, not a plausible third marker on the sword given how confined and
rare they are next to `2`/`3`'s near-ubiquity. Worth a quick visual check
on those specific frames before the calibration tool starts trusting
whatever IDs a detection pass reports, so it doesn't quietly try to place
a phantom third marker.

## 7. Open questions

- **Reference-marker convention for the seed step**: which ArUco anchors a
  given frame when multiple are visible? Doesn't matter for the final
  result once Phase B's joint refine runs (every frame's pose becomes a
  free variable resolved by all markers visible in it together), but
  matters for how simple/robust the seed step alone can be made.
- **Per-marker `noise_std` proposal**: the person-marker plan's §5.2 gets
  this for free from refinement residuals ("robust std of the final
  residuals, floored at detector accuracy"). The same idea should carry
  over directly once Phase B exists — worth generating for the new body
  definition's `noise_std` fields (design doc §5.1) rather than leaving
  them at a uniform default.
- **Multiple aruco markers per face**: answered by Harri — one per face
  (IDs `2` and `3`, confirmed above). No other markers from DICT_4X4_50
  are expected in the scene; the `10`/`17` stragglers found during
  validation are almost certainly noise, not a design consideration.
