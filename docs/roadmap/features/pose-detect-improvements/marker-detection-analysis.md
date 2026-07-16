# Marker detection to augment the markerless pipeline — analysis

Captured 2026-07-16 (discussion with Claude). Brainstorm for a future
marker-augmentation capability — **not scheduled for immediate
development**. Companion to
`pose-detect-improvements-analysis.md` in this directory.

This connects directly to a capability already flagged as planned in
`docs/roadmap/features/error-improvements/phase5-cross-person-plan.md`:
*"combining multiple detection sequences (markerless pose + physical
motion-capture markers, potentially spanning multiple people in one
marker stream)"* — there identified as a data-loading concern
(`SessionReader`/`ObservationSet` multi-source). This analysis is the
detection/labeling/model side of that same capability.

## Use cases

1. **Prop tracking** — weapons (bokken, jo, tanto) and other objects the
   performers interact with. Pose estimators produce nothing for these;
   markers are the only practical signal.
2. **Body augmentation** — markers on body parts poorly covered by pose
   estimator keypoints (the spine has no keypoints between neck and
   pelvis; hips are systematically biased from back views), and
   identity cues to distinguish performers wearing near-identical
   uniforms (and to disambiguate left vs. right hands in grabs).

The pipeline remains **markerless-first**: markers are additional
observations fused into the same UKF, never a requirement. Everything
must degrade gracefully when markers are occluded, fall off, or are
absent entirely.

## Question A — what marker types to support

### Passive dots (retroreflective or colored)

- **Retroreflective dots** shine when a light source sits near the
  camera axis; commercial mocap uses IR ring lights for this. A clip-on
  LED ring per action camera / phone is simple, cheap extra equipment —
  more hassle but perfectly field-viable when it earns its keep (Harri).
  Two practical notes: (a) action cameras have IR-cut filters, so the
  ring must be **visible-light** LED, not IR (unless cameras are
  modified); (b) retroreflection returns light toward its source, so
  each camera sees only its *own* ring's reflections — no cross-camera
  marker glare, though opposing cameras do see the ring LED itself as a
  static bright dot (trivially masked). With exposure biased slightly
  down, the reflections pop far above the scene, giving the most
  lighting-independent detection of any passive option — no white
  balance or color-constancy concerns at all.
- **Matte fluorescent / high-saturation colored dots** are the
  no-extra-equipment alternative: under normal lighting a saturated
  color blob is easier to detect than an unlit gray retro dot.
  Detection for either variant is cheap (threshold + connected
  components + sub-pixel centroid) and centroid accuracy is the best of
  any option here (~0.5–1 px).
- **Size math — with realistic action-cam FOV.** Wide modes are ~120°
  HFOV or more (SuperView more still), not 90°: at 4K/3840 px that is
  ~4–4.5 mm/px at 5 m mid-frame (fisheye projection is denser at frame
  center, ~2.7 mm/px, and worse toward the edges). A blob needs ~4+ px
  to detect reliably → **~20–25 mm dot at 5 m** in wide modes. Two
  levers if that is too big: narrower FOV capture modes (e.g. GoPro
  Linear ~87° recovers the ~2.6 mm/px figure at the cost of coverage),
  or accepting detection dropout beyond some range.
- **Color as soft ID**: a colored patch around (or instead of) the dot
  gives partial identity — but color *classification* needs materially
  more pixels than blob *detection* (edge pixels are contaminated;
  ~5–8 px of clean interior color is needed), so a color-coded patch is
  **~25–40 mm at 5 m** in wide FOV modes, noticeably larger than a
  detect-only dot. Realistically 4–8 colors are distinguishable across
  cameras; treat color as a *soft prior* in assignment (question B),
  never a hard ID. Requires locking white balance in capture settings —
  auto-WB shifts colors per camera and over time. (Retroreflective +
  ring light sidesteps the WB problem entirely, at the cost of carrying
  no color ID unless colored retro material proves distinguishable —
  empirical question.)
- Verdict: **primary body-marker candidate, in two flavors** —
  retroreflective + clip-on ring light (best robustness, extra
  equipment, no color ID) vs. colored dots (no equipment, WB-sensitive,
  soft ID for free). Which wins is an empirical shoot-out, not a design
  decision. Either way there is no hard ID — the labeling problem
  (question B) must be solved for these.

### Fiducial markers (ArUco / AprilTag / CCTag)

- Square binary-code tags give hard ID and even single-marker 6-DOF,
  but the size problem the user noted is real and quantifiable: an
  AprilTag needs roughly 50–80 px across to decode — at the realistic
  ~4–4.5 mm/px of wide-FOV 4K action cameras, that is **~25–35 cm at
  5 m**. On a body that is a billboard, not a marker; on a thin weapon
  it does not fit flat at all (cylindrical surfaces break the planar
  decode).
- Square tags also die first under exactly the conditions that matter
  here: motion blur and oblique viewing angles. **CCTag-style
  concentric-circle fiducials** are the exception worth knowing about —
  they were designed specifically for motion-blur robustness and decode
  from blurred frames where AprilTags fail. Fewer distinct IDs, but the
  ID space needed here is small.
- Verdict: **use fiducials where they excel — slow/static anchoring,
  not continuous tracking.** Concretely: prop identification, a
  reference-pose ID anchor at trial start, possibly a static
  world-frame reference on the floor/wall. When a fiducial *is*
  readable mid-trial it re-anchors identity for the blob tracklet
  carrying it (hybrid scheme, question B).

### Color-coded / patterned clothing

- Different-colored gloves per hand, colored belts, tape bands at known
  limb locations. Detection is region-level (color segmentation or a
  small learned detector), giving a deformable-region centroid — low
  positional accuracy (treat with large σ) but **extremely robust
  identity**.
- This is the cheapest and highest-leverage option for the two
  association problems markers can solve: *whose hand is this* during
  wrist grabs (a red glove on A's left hand answers it directly) and
  *which performer is which* for segmentation-ID verification. Note a
  colored glove also directly helps the existing hand-redetection
  pipeline pick the right hand.
- A tape band around a limb has a useful geometric reading: its
  centroid is a point on the limb *axis*, not on the surface — a
  different, arguably cleaner observation model than a surface dot
  (insensitive to roll of the band around the limb).
- Verdict: **do this first among body options.** No time-series problem
  exists at all — the region's color *is* the ID.

### Active markers (LEDs, blink-coded IDs)

Small battery LEDs with blink-coded identity are the classic active
solution. Rolling shutter and frame-rate sync on action cameras make
blink decoding fragile, and attaching powered hardware to performers
being thrown is impractical. **Out of scope**; recorded for
completeness.

### Summary

| Type | ID | 2D accuracy | Blur tolerance | Best role |
|---|---|---|---|---|
| Colored dot | none/soft (color) | high | good (with fast shutter) | body points (spine, hips) |
| Retro dot + ring light | none | high | good | body points; most lighting-robust, needs clip-on rings |
| AprilTag/ArUco | hard | high (corners) | poor | static refs, large props |
| CCTag (concentric) | hard, small ID space | high | **good** | prop ID, re-anchoring |
| Colored clothing/bands | hard (region = ID) | low | very good | identity, limb axis, gloves |

## Question B — building time series from anonymous detections

This is the classic mocap marker-labeling problem, but the pipeline has
a decisive advantage over standalone marker systems: **the UKF already
predicts every marker's 3D position each frame.** Labeling reduces to
prediction-gated assignment — exactly how commercial systems do online
labeling — rather than solving unconstrained multi-target tracking.

Proposed structure, three cooperating layers:

1. **Per-camera tracklets.** Frame-to-frame linking of raw blob
   detections in each camera independently (nearest-neighbor with a
   velocity gate — blobs are sparse, this is easy). Tracklets carry
   identity forward through frames where nothing else identifies them,
   and break honestly at occlusions rather than guessing.
2. **Multi-view correspondence.** Unlabeled 2D detections across
   cameras are matched by epipolar consistency and triangulated
   (RANSAC over camera pairs, verified against a third view where
   available). With 3+ cameras this prunes hard; output is a set of
   anonymous 3D candidate points per frame. This step is optional per
   marker — a marker seen in only one camera can still be used as a 2D
   observation once labeled.
3. **Model assignment.** Candidate points (3D) or tracklets (2D) are
   assigned to predicted marker positions by gated cost minimization
   (Hungarian / auction over Mahalanobis distances — the gating
   machinery already exists in the tracker; the *mutual-exclusion
   assignment* step is the new part). ID evidence enters as cost terms:
   tracklet continuity (strong), color agreement (soft), fiducial
   decode when readable (hard, overrides).

**Ambiguity policy — drop, don't guess.** When two predicted markers
gate the same detection with comparable cost (wrist grabs again),
discard the observation. This matches the existing outlier-gating
philosophy: a missing observation costs a little covariance growth; a
mislabeled one injects a confident lie. Never `force_inlier` a
recovered label.

**ID recovery after tracklet breaks**: re-association happens through
the same gated assignment against the (now more uncertain) prediction,
optionally confirmed over a few frames before the observation is
trusted at full weight. Color and fiducial evidence shortcut this.

**Bootstrapping**: no chicken-and-egg problem exists, because the
markerless pipeline initializes the tracker on its own. Markers join
opportunistically once their assignment is confident (see question C —
the markerless track is also what *calibrates* them).

## Question C — connecting markers to the skeleton

### Body markers: let the markerless tracker calibrate them

The `Skeleton` abstraction already models markers as points attached to
joints with fixed local offsets (`Marker` in
`include/posetrak/core/skeleton.hpp`) — a physical marker is just
another `Marker` entry whose offset is *estimated* instead of authored.

The key simplification: unlike classic mocap, **placement does not need
to be anatomical or repeatable.** A marker is an arbitrary point rigidly
attached to some segment; the markerless tracker measures where it
actually is:

1. During a reference window at trial start (a defined kamae or T-pose,
   or just any well-tracked low-speed segment), the markerless pipeline
   produces a confident skeleton state.
2. Each triangulated 3D marker point is attached to the nearest
   skeleton segment, and its constant offset in that joint's local
   frame is computed by averaging over the window's frames.
3. Offsets are refined afterwards by a least-squares pass over the
   whole trial — structurally the same job as the existing bone-length
   `scale` post-process (`posetrak-tracker scale`), and should follow
   that pattern rather than adding marker offsets to the UKF state
   (state-dimension growth for a constant is not worth it).

Consequences worth liking: markers can be slapped on wherever they
stick (hakama fabric permitting — cloth-mounted markers move relative
to bone, which is soft-tissue/cloth artifact; model it as extra
observation noise for markers on loose clothing, low noise only for
markers on skin/snug areas), and a marker that falls off mid-trial
simply stops producing gated observations.

### Props: rigid bodies as degenerate skeletons

A prop is a skeleton with a 6-DOF floating root and only `FIXED`
joints/markers — the existing skeleton machinery should handle it with
a small YAML file per prop type and no new joint math. Notes:

- **Marker layout must break symmetry.** A constellation of identical
  dots on a rigid body resolves pose only if the layout is asymmetric
  (no two inter-marker distance patterns alike). For rotationally
  symmetric props (a jo is a cylinder) roll is *physically*
  unobservable and also physically meaningless — lock that DOF in the
  prop skeleton rather than letting the filter chase it.
- **Layout calibration is a one-time job** because props are rigid and
  persistent (unlike bodies): wave the prop through the capture volume
  once, bundle-adjust the rigid point constellation from multi-view
  tracks, store it with the prop's skeleton YAML.
- **Thin weapons**: flat fiducials do not fit; practical options are
  colored bands (each band centroid = point on the axis; two bands +
  locked roll fully pose a staff) or small dots on the tsuba/hilt where
  there is some flat area.
- **Prop-in-hand coupling is Phase 5 for free.** The
  `MultiPersonTracker` orchestrator owns N `Tracker` instances; a prop
  tracker is just another instance with a rigid skeleton. Contact
  gating, cross-person anchor observations, rotating processing order —
  all of it applies verbatim to "person's hand ↔ prop grip point."
  This is a strong reason to keep the Phase 5 orchestrator free of
  person-specific assumptions (it already avoids two-person
  assumptions; avoid *human* assumptions too where cheap).

## Architecture fit

- **Data loading**: marker detections are one more observation sequence
  per camera, entering through the multi-source
  `SessionReader`/`ObservationSet` work already flagged in the Phase 5
  plan. The Phase 2 multi-source `pose_observations` schema precedent
  (`source` values like `.refined`) suggests the shape: a marker
  detection run is a source, with its own confidence semantics.
- **Fusion**: markers are ordinary `Observation`s with type-appropriate
  noise (`noise_std_override`): colored-dot centroids ~1 px, fiducial
  corners similar, clothing-region centroids large (tens of px). The
  UKF weighs them against pose-estimator keypoints automatically; a
  low-noise spine dot will rightly dominate the interpolated spine
  keypoints where present.
- **Detection runtime**: blob/color detection is orders of magnitude
  cheaper than pose estimation — it can run on full frames (not crops),
  which conveniently makes it independent of the segmentation/crop
  pipeline and its failure modes.
- **Capture settings**: everything here presupposes the fast-shutter
  recommendation from `pose-detect-improvements-analysis.md`, plus
  locked white balance for any color-based scheme.

## Suggested phasing (when this is picked up)

1. **Props first.** Self-contained (custom skeleton YAML, one-time
   layout calibration, no anonymous-labeling problem if bands/CCTags
   are used), delivers a capability the markerless pipeline cannot
   provide at all, and exercises both the multi-source data loading and
   the extra-tracker-instance path of the Phase 5 orchestrator.
2. **Colored clothing identity cues.** Detector-side only; feeds the
   existing association pain points (hand ownership, performer ID,
   segmentation verification) without any new tracker machinery.
3. **Anonymous body dots with the full labeling pipeline.** Highest
   accuracy payoff (spine, hips) but requires the tracklet +
   multi-view + gated-assignment stack and the offset-calibration
   post-process — the largest and last piece.

## Open questions

- **Retroreflective vs. colored dots shoot-out**: with a clip-on
  visible-light LED ring per camera, how far above the scene do retro
  reflections sit at realistic exposure, and how annoying is the ring
  light for performers? Also whether colored retro material keeps
  enough color separability to carry soft ID under ring lighting.
- Color palette size that survives real dojo lighting across cameras
  with locked WB — needs an empirical test (record swatches, measure
  separability), not a design decision.
- Practical marker/patch sizes vs. capture FOV mode: wide modes cost
  ~40–70 % marker size versus linear/narrow modes — is the coverage
  loss of narrower modes acceptable for marker-augmented trials, or do
  markers need sizing for wide FOV?
- Cloth-mounted marker noise: how bad is marker motion on a hakama or
  gi sleeve in practice? Determines whether spine/hip markers must go
  on snug base layers to be worth their weight.
- Whether 2D-only marker observations (marker seen in one camera,
  labeled via tracklet continuity) are worth supporting in v1, or
  whether requiring multi-view triangulation for labeling keeps the
  assignment problem materially simpler.
- CCTag detector maturity/licensing vs. rolling a simple
  concentric-ring detector in-house.
- Minimum marker count for a useful spine (one dot between the
  shoulder blades may already beat zero; three along the spine start
  constraining flexion shape).
