# Marker-based motion capture — algorithm design

**Status**: Design proposal (2026-08-19). Companion to
[marker-mocap-design.md](marker-mocap-design.md); numbering below is
referenced from there. This document covers the algorithmic content: what is
computed, why these formulations, and where they plug into the existing
pipeline.

---

## 1. Detection

### 1.1 Coded markers (ArUco / AprilTag)

Reuse the `FiducialDetector` implementations in
`python/app/setup/fiducial_markers.py` (`ArucoDetector`,
`MarkerRigDetector`) — they were deliberately built per-frame and stateless
for exactly this consumer. Per-run configuration mirrors what the extrinsics
UI already learned the hard way:

- `min_marker_perimeter_rate` must be configurable and default low (~0.01):
  the cv2 default (0.03 of the larger image dimension) silently rejects
  room-distance markers in 4K frames.
- One detector per distinct dictionary the target body uses; membership
  filtering by the body's own `(type, dictionary, id)` set so unrelated tags
  sharing a dictionary don't leak in (the "purple marker mixup" lesson).

**Runtime strategy** (brief's "CPU and slow, needs full resolution"
concern): detection is an offline batch pass like pose detection, so
throughput, not latency, matters.

- Baseline: full-frame detection per camera, cameras processed in parallel
  (process pool — cv2.aruco releases the GIL unevenly). ArUco on a 4K frame
  is ~50–150 ms/core; at a detection-phase frame stride this is acceptable.
- Optimization tier 1 (cheap, generic): downscale-scan → full-res refine.
  Detect on a 2× downscaled frame (quad candidates survive), then re-run
  `cornerSubPix`-style refinement of the found quads' corners at full
  resolution. Corners keep full-res accuracy; cost drops ~4×.
- Optimization tier 2 (only if needed): temporal ROI — search near last
  frame's detections plus a periodic full-frame rescan (every N frames) to
  reacquire. Adds state to the detection pass; do not build until tier 1
  proves insufficient.

Motion blur: square tags die under fast motion; that is accepted (props in
fast phases are carried by dots/bands per the marker-type table, or by the
filter's process model coasting through short dropouts). CCTag-style
concentric fiducials remain the noted alternative if blur-robust *coded*
detection becomes necessary; keep them behind the same `FiducialDetector`
protocol.

### 1.2 Anonymous dots (reflective or colored)

New detector, deliberately simple:

1. Per-pixel score: for retro dots, luminance above an adaptive threshold
   (exposure biased down + ring light makes them near-saturated); for
   colored dots, distance in a chroma plane to per-run reference colors
   (white balance locked at capture — capture-settings prerequisite).
2. Connected components with area/circularity bounds derived from the
   expected marker size range at plausible depths.
3. Sub-pixel centroid via intensity-weighted mean over the component
   (~0.5–1 px accuracy); confidence from component contrast/compactness.
4. Static-highlight suppression: the opposing cameras' ring LEDs and other
   fixed speculars appear as stationary detections; a per-camera median
   background mask over the run removes anything that never moves.

Output per frame per camera: unordered `(x, y, conf[, color_class])` list —
the variable-length blob of design §4.1. Color class, when present, is a
*soft* prior for association (§3), never a hard ID (4–8 classes
realistically separable across cameras).

### 1.3 Colored bands / gloves (region markers)

Color segmentation (or a small learned segmenter later) → region centroid,
with two type-specific readings:

- A **glove/patch** yields a low-accuracy centroid with hard identity
  (region = ID): fed as a normal labeled observation with large
  `noise_std` (tens of px).
- A **band around a limb or staff** yields a point on the *axis* of the
  limb/prop, insensitive to roll: modeled as a marker whose body-local
  position lies on the segment axis. This is the primary thin-weapon
  mechanism (two bands + locked roll fully pose a staff).

---

## 2. Measurement model

Markers enter the UKF as ordinary `Observation`s — this is the load-bearing
design decision, and it needs *no* filter changes:

- **Coded quads: corners as 4 independent point observations**, not a
  derived pose. Rationale (unchanged from the ArUco analysis): the
  FK→project pipeline is exactly the corner predictor; a PnP pre-step would
  discard the multi-camera fusion and outlier gating the UKF already does,
  and would need its own covariance plumbing for an SE(3) measurement. A
  marker seen obliquely automatically contributes what it actually
  constrains. This also answers the brief's "markers that return orientation
  might need special handling" — they don't: orientation information *is*
  the corner spread, and the filter extracts it through the projection
  model.
- **Noise**: `noise_std_override` per marker (skeleton `noise_std` or track
  default), `crop_scale = 1.0`. Typical values: ArUco corners ~0.5–1 px,
  dot centroids ~0.5–1 px, band/region centroids 10–40 px, cloth-mounted
  dots whatever the offset calibration's residuals say (§5).
- **Gating**: existing Mahalanobis outlier rejection applies per corner —
  a half-occluded or mis-refined corner is dropped individually.
- **Partial detection**: NaN slots are simply absent observations; the
  existing empty/insufficient-observation handling covers frames where the
  whole prop is unseen (covariance grows, filter coasts; `tracking_lost`
  only when the existing sufficiency checks say so).

---

## 3. Anonymous-marker association ("labeling")

The brief asks: per-marker prediction before the pose UKF, or fold
assignment into the update step? **Recommendation: a separate assignment
stage immediately before the update, using the UKF's own prediction —
never inside it.** Folding assignment into the update (optimize
correspondences jointly with the state, JPDA/multi-hypothesis style) couples
a combinatorial problem into the sigma-point pipeline, breaks the
per-observation gating diagnostics the whole toolchain is built on, and
buys accuracy only in exactly the ambiguous situations where the project's
established policy is to *drop* the observation anyway. Prediction-gated
assignment is also how commercial optical mocap does online labeling.

Three cooperating layers (structure from marker-detection-analysis.md,
firmed up here):

### 3.1 Per-camera tracklets

Frame-to-frame nearest-neighbor linking of raw centroids within each
camera, with a velocity gate (constant-velocity extrapolation of the
tracklet, gate radius from expected pixel motion at the frame rate).
Detections are sparse (tens per frame), so this is Hungarian on a small
cost matrix per camera per frame. Tracklets:

- carry identity forward through frames where nothing else identifies them;
- **break honestly** at occlusions/merges (two candidates in one gate →
  terminate rather than guess);
- carry the soft color class as a merge veto.

Tracklet ids are per-camera, meaningless across cameras, and stored with
the detection layer for debuggability.

### 3.2 Multi-view correspondence (unlabeled 3-D candidates)

Per frame: match unlabeled 2-D detections across cameras by epipolar
consistency (point-to-epiline distance threshold from calibration error),
triangulate pairwise (existing DLT `Triangulator`), verify against a third
view where available (3+ cameras prune hard). RANSAC over camera pairs for
robustness to false blobs. Output: anonymous 3-D candidate points with an
inlier-camera list. This step is per-frame and stateless; tracklets are not
consumed here (they act in 3.1 and 3.3).

### 3.3 Model assignment (the labeling decision)

Inputs each frame: predicted 3-D marker positions (FK on the UKF's
*predicted* state) with per-marker projected uncertainty — the already
existing `Tracker::marker_projection_std()` gives the pixel-space σ; a 3-D
analog (J P Jᵀ in world space) is a small addition to the same machinery.
Assignment is gated global cost minimization:

- cost(candidate, marker) = squared Mahalanobis distance (3-D candidates
  against 3-D prediction covariance + triangulation covariance; or 2-D per
  camera for single-view tracklet continuation);
- evidence terms added to the cost: tracklet continuity with an already
  labeled tracklet (strong negative cost), color-class agreement (mild),
  coded-marker decode on the same physical body when readable (hard
  override);
- solve with Hungarian (mutual exclusion — the one genuinely new piece of
  machinery vs. today's per-observation gating). The candidate/marker pool
  the solver runs over is a scope parameter (§3.4): at its widest, the
  assignment is solved **once per frame across all subjects in the run** —
  every person's and prop's predicted markers form one joint problem, not
  one per subject. This is what makes mutual exclusion meaningful in the
  scenes that motivate this feature: a dot near two performers' wrists, or
  a hand closing on a dotted prop, is *cross-subject* ambiguity, invisible
  to any per-subject assignment. `MultiPersonTracker`'s frame loop already
  has all subjects' predictions in hand at the right moment (same data the
  contact gate uses);
- **ambiguity policy — drop, don't guess**: if the best and second-best
  assignments for a detection are within a margin (both gates pass with
  comparable cost — wrist-grab territory), discard the detection for this
  frame. Never `force_inlier` a recovered label.
- **confirmation**: a newly (re-)labeled tracklet is carried tentatively
  for K frames (K≈3) with inflated noise before full weight — cheap
  insurance against a confidently wrong re-association after a break.

Labeled detections become `Observation`s (marker index = the skeleton slot)
for this frame's update; the labels also write the finalized sequence
(design §4.3), making labeling reviewable and hand-editable afterwards.

**Bootstrapping**: no chicken-and-egg — the markerless pipeline initializes
and tracks on its own; markers join opportunistically once assignments are
confident. For a body carrying coded markers (design phase 1), this whole
section is bypassed at the detector: ids label detections directly, and
any dots sharing that body get their labels from the body's own rigid
geometry once the coded anchors have posed it — no assignment machinery at
all (§3.4 tier 1). For a *dot-only* body, steady-state relabeling still
needs no more than that same rigid-geometry lookup, but there is no coded
anchor to pose the body first — cold start is a genuine unlabeled
point-set registration problem, covered in §4.

### 3.4 Scope: single-body vs. cross-subject

`MarkerAssociator` is written once with a **scope parameter**, not rebuilt
when person markers arrive later. The cost function, gating, and Hungarian
solver above are identical at both scopes — only what competes in one
assignment problem changes:

- **Single-body scope** (design phase 2: a standalone dot-only prop): the
  candidate pool and the marker pool are both just that one body's own
  points. There is exactly one subject, so mutual exclusion is present in
  the machinery but never actually contested — no cross-subject evidence
  terms are needed, and the K-frame confirmation window mostly guards
  against a mis-registered restart after occlusion, not cross-subject
  mix-ups. This tier is also what a mixed coded+dot body uses at
  steady state (previous paragraph) even though it needs no cold start.
- **Cross-subject scope** (design phase 4 once multiple dot-only/mixed
  props share a scene; design phase 6 once persons carry dots too): the
  assignment is solved once per frame **across every tracked subject in
  the run** — every person's and prop's predicted markers form one joint
  problem, per the Hungarian bullet above. This is what makes mutual
  exclusion meaningful in the scenes that motivate this feature: a dot near
  two performers' wrists, or a hand closing on a dotted prop, is
  *cross-subject* ambiguity, invisible to any single-body assignment.

Implement single-body scope first (phase 2) as the general solver
restricted to a candidate/marker pool of size one body; widening the pool
to "all subjects `MultiPersonTracker` is stepping this frame" for phase 4/6
should be a call-site change, not a rewrite of the assignment logic itself.

**Placement**: `MarkerAssociator` is a C++ component owned by the per-frame
loop (`step_person_context()` level), holding tracklet state per camera. It
must run after `predict()` conceptually; practically it consumes the
previous posterior + process model extrapolation, identical to how
cross-person anchors use velocity-extrapolated positions for
not-yet-stepped subjects — reuse that convention rather than splitting the
UKF's predict/update entry point.

---

## 4. Rigid-body initialization (props)

Person initialization is triangulation + damped-least-squares IK. For a
root-only prop skeleton the analytic solution is better conditioned and
cannot fall into IK local minima. The fit itself (§4.2) needs, per marker,
one triangulated world point with a known correspondence to a body-local
template point; §4.1 covers how that correspondence is obtained, which
differs for coded and dot-only bodies.

### 4.1 Establishing correspondence

- **Coded or mixed bodies** (design phase 1, and steady state for any body
  with a coded anchor): decoded marker ids give per-marker-id triangulated
  points directly — proceed to §4.2.
- **Dot-only bodies, cold start** (design phase 2): the multi-view
  correspondence step (§3.2) yields only *unlabeled* 3-D candidates, so the
  body-local template point each one corresponds to is unknown. This is a
  rigid point-set registration problem without correspondence, solved by
  pairwise-distance consistency — the same technique commercial optical
  mocap uses to identify a rigid body from a fresh marker cloud:
  1. From the body's marker template, precompute all pairwise inter-marker
     distances.
  2. Pick a candidate triple from the frame's unlabeled 3-D points whose
     three pairwise distances match a template triple within tolerance
     (calibration + triangulation error); this fixes a candidate
     correspondence for those three points and, via Kabsch on the triple, a
     candidate rigid transform.
  3. Verify by transforming the remaining template points into world space
     under that transform and counting how many unlabeled candidates lie
     within tolerance of one (inlier count) — RANSAC over triple choices,
     keep the transform with the most inliers.
  4. A template with internal symmetry can yield multiple equally-good
     correspondences at this step; if so, treat it exactly like the
     collinear case below — the ambiguous relabeling maps to an
     unobservable DOF (or a discrete finite set of them) that must be
     locked or accepted as ambiguous, not resolved by guessing. Distinct,
     asymmetric marker constellations avoid this entirely and are the
     recommended way to build dot-only props (mirrors the "two symmetric
     bodies in one capture" warning in design §6.2).
  5. Once inliers are identified, they *are* the per-marker-id
     triangulated points §4.2 expects; run the fit below. Re-run this
     registration (not just re-triangulation) after any tracking loss,
     since a fresh cloud again has no correspondence.

  A short window of frames (rather than a single frame) improves inlier
  robustness against noise and partial occlusion at essentially no extra
  cost, since this only runs at init/re-acquisition, not every frame.

### 4.2 Closed-form fit

1. Triangulate all markers with ≥ `min_cameras_for_init` views (existing
   `Triangulator`), using the correspondence from §4.1.
2. Require ≥ 3 triangulated points, non-collinear (smallest singular value
   of the centered point matrix above a threshold; for deliberately
   collinear layouts — a two-band staff — fall to the reduced solution
   below).
3. Closed-form fit body-local → world: Kabsch/Umeyama with scale fixed at 1
   (geometry is metric by construction). Residual RMS is the initialization
   quality metric; reject and retry on a later frame above a threshold
   (same retry loop initialization already has).
4. Collinear case: position + axis direction are determined; the rotation
   about the axis is set arbitrarily and the corresponding DOF must be
   locked (design §5.3) — consistent, since that roll is exactly the
   unobservable DOF.
5. Initial velocities zero; initial covariance from
   `init_position_std`/`init_orientation_std` as today.

Implementation: `Tracker::initialize()` branches to this path when the
skeleton has no non-root active joints; no new public API needed.
Re-initialization after loss constructs a new Tracker (existing
constraint) and, for dot-only bodies, re-runs §4.1's registration first.

---

## 5. Body-marker offset calibration

### 5.1 Reference-window attach (seed)

Over a user-selected well-tracked window (or auto-proposed: lowest
NIS/condition-number stretch from `tracking_stats`):

1. Run markerless tracking (smoothed).
2. Triangulate each labeled marker per frame.
3. For each marker, for each candidate segment j: express the marker's
   world position in joint j's frame per frame,
   `p_j(t) = T_j(t)⁻¹ x(t)`; the attach segment is the one minimizing the
   spread `tr(Cov_t[p_j(t)])` — "most rigid explanation wins", which is
   better than nearest-segment-by-distance (a chest marker is *near* the
   upper arm in some poses but *rigid* only w.r.t. the torso).
4. Seed offset = robust mean of `p_j(t)` (geometric median or
   trimmed mean — outlier frames from tracking error shouldn't bias it).

### 5.2 Whole-trial least-squares refinement

Refine offsets (and optionally re-estimate per-marker noise) by minimizing
reprojection error over the whole trial with the tracked trajectory held
fixed:

    min_{p_local}  Σ_t Σ_cam ρ( π_cam( T_j(t) · p_local ) − z(t,cam) )

with a robust loss ρ (Huber). Given fixed `T_j(t)` this is a small
independent nonlinear problem per marker (3 unknowns) — structurally the
same job as the existing bone-length `scale` post-process and implemented
as a sibling CLI mode (`posetrak-tracker calibrate-markers config.toml`),
**not** by adding offsets to the UKF state (state-dimension growth for a
constant is not worth it; same call the marker-detection analysis already
made). One outer iteration of {refine offsets → re-track with markers →
refine again} is optionally exposed but expected unnecessary in practice.

Per-marker `noise_std` proposal: robust std of the final residuals,
floored at the detector accuracy. This is what makes cloth-mounted markers
self-limiting (R2.4): a marker on a flapping hakama gets 8 px noise and
correspondingly little influence, automatically.

### 5.3 Soft-mount weight maps (deferred option)

For markers demonstrably influenced by several joints (chest), the
measurement model could become a convex blend:
`x(t) = Σ_j w_j · T_j(t) · p_j_local`, with weights estimated in the same
least-squares pass (simplex-constrained). FK cost per sigma point rises
(k transforms per marker instead of 1) and `ForwardKinematics`' marker→
single-frame mapping assumption breaks, touching the hottest path in the
tracker. Deferred until single-segment + inflated noise measurably limits
accuracy on real captures; the calibration pass should, however, *report*
the rigidity spread per candidate segment (§5.1 step 3) so the evidence for
needing this accumulates for free.

---

## 6. Camera-drift monitoring and re-anchoring

### 6.1 Monitor

For each camera, at a coarse stride (e.g. 1 s) over a capture: detect the
session's `scene_marker_bodies` fiducials, reproject their known world
corners through the camera's current calibration, and record the robust
mean reprojection residual `r_c(t)` (median over corners — individual
mis-detections must not fire the alarm).

Movement shows as a **step change** in `r_c(t)`, so use a step detector
(CUSUM or a two-window mean comparison), not a per-sample threshold:
per-sample residuals also grow smoothly with, e.g., heat-induced focus
drift, and we specifically want "moved at time t₀" out. Alarm threshold in
pixels scaled to the calibration's own RMS (e.g. fire above
max(3 × calib RMS, 2 px) sustained for M consecutive samples). Occlusion of
the markers (people walking through) yields *missing* samples, not biased
ones — the detector must tolerate gaps.

Output per camera: OK / MOVED(t₀, before-RMS, after-RMS), stored with the
capture and surfaced in the UI + tracking preflight (design §6.4).

Deliberate camera *pans* (UC3's "intentional" case) appear as a residual
ramp rather than a step; v1 reports them as MOVED over an interval and
excludes the interval from windowed extrinsics validity.

### 6.2 Re-solve

For the post-movement range: accumulate scene-marker corner detections over
several sparse frames, solve the camera's new pose by PnP against the
corners' known world positions (RANSAC + refinement) — this is exactly the
single-camera re-anchor mechanism validated in extrinsics-improvements
Phase 9, applied at a new time range instead of a new session. Quality gate:
post-solve residual must return to the pre-movement band; otherwise the
tool asks for a proper recalibration instead of silently accepting a bad
pose. Result: new `extrinsic_calibrations` row + an
`extrinsic_calibration_windows` entry starting at t₀ (design §6.4).

Multi-camera consistency check: after re-solving camera X, triangulations
of scene markers using X + unchanged cameras must agree with the stored
world positions — catches the failure mode where the *scene markers*
moved, not the camera.

---

## 7. Validation strategy

- **Synthetic first, per phase**: the existing synthetic-sequence test
  pattern extends naturally — generate a rigid-body trajectory, project
  marker corners with noise/dropout, assert pose RMSE (phase 1); synthetic
  dot-only prop with an unlabeled marker cloud, asserting correct §4.1
  registration at cold start and correct relabeling through a simulated
  occlusion (phase 2); synthetic scene with **two** independent dot-only
  (or mixed) props whose point clouds overlap in the shared volume,
  asserting zero cross-prop mislabels — the case that motivates promoting
  §3.4's assignment to cross-subject scope (phase 4); synthetic anonymous
  dots on an articulated person with occlusion/swap scenarios asserting
  zero mislabels and bounded drop rate (phase 6); synthetic camera bump
  asserting detection latency and re-solve accuracy (phase 7).
- **Ground-truth props**: the calibration box itself is the first test prop
  — its geometry is exactly known, and `scene_marker_bodies` gives a
  static-pose ground truth (track the box while stationary → pose error is
  directly measurable; then hand-carry it).
- **Person-marker value test**: same trial tracked with and without the
  marker sequence bound; compare smoothness/NIS/reprojection on held-out
  cameras (leave-one-camera-out, which the pipeline already supports via
  `active_camera_ids`).
