# Reflective-dot detection for the sword prop (Phase C)

Scope note (2026-09-01): follow-up to
[rigid-marker-body-calibration-design.md](rigid-marker-body-calibration-design.md)'s
Phase C, prompted by Harri's own tracking-quality read of the first real
sword baseline (status.md, 2026-09-01 entry): fast sword motion is
frequently invisible to the ArUco-only rig (2 markers), so the tracker
coasts on its constant-velocity model through the gap and visibly snaps
once a fresh, spatially distant observation arrives. Not started —
scoping only.

## 1. Why dots should actually help here

ArUco corner detection requires decoding a full bit pattern from a small
square; motion blur smears that pattern below the decode threshold well
before the marker becomes visually unrecognizable as *a* marker. A
reflective/colored dot only needs a locally bright or saturated blob and a
centroid — far more blur-tolerant, so it should keep producing
observations through exactly the fast-motion windows ArUco currently
drops. This is a specific, testable hypothesis, not just "more markers are
better": re-running the exact "Harri bokken" baseline trial with dots
added should show fewer/shorter observation gaps during fast segments
specifically, which is what should reduce the coast-then-snap artifact.
Worth checking this directly (gap duration histogram, ArUco-only vs.
dots-added) once dots exist, rather than only looking at aggregate
tracked-step-fraction.

Two secondary contributors to the snap that dots don't fix, worth keeping
separate so a partial improvement isn't misread as dots not working:
process-model mismatch (constant-velocity coasting is a poor model for a
prop that can change direction sharply mid-swing) and UKF tuning
(`process_noise_std`/`process_noise_vel_std` control how fast uncertainty
grows during a gap, hence how hard the eventual correction snaps).

## 2. Detection method — already resolved, not new scoping

`docs/roadmap/features/pose-detect-improvements/marker-detection-analysis.md`
(Question A) already settled this generically: threshold + connected
components + sub-pixel centroid, ~0.5–1px centroid accuracy, "orders of
magnitude cheaper than pose estimation." Open in that doc and *not*
resolved for this specific capture: retroreflective (needs a visible-light
clip-on ring light per camera — action cameras have IR-cut filters) vs.
matte fluorescent/colored (no extra equipment, WB-sensitive). **First
concrete step, before writing any detector code: look at the actual dots
in the Weapon test footage** to see which regime applies and roughly how
many pixels they cover at this room's camera distances (that doc's own
sizing math: ~4+ px needed, which becomes a real constraint at wide FOV
and several meters' distance -- this room looks closer than the doc's 5m
reference case from earlier frame grabs, but worth confirming rather than
assuming).

## 3. Two sub-problems, different difficulty

### 3.1 Calibration-time: determining the dots' body-local positions

This is the easier half, and follows Phase A/B directly: same
co-occurrence-with-a-reference-ArUco mechanism already validated and
built, just accumulating dot centroids (3 DOF: a point) instead of ArUco
corners (6 DOF: a rigid transform). No cold-start problem exists here --
restrict to frames where an ArUco is *also* solvable that instant (exactly
what `calibrate_rigid_marker_body.py` already does for the second ArUco
marker) and extend `samples_by_marker` to accept centroid-only candidates.
`marker-detection-analysis.md`'s own framing agrees: "Layout calibration is
a one-time job because props are rigid and persistent... bundle-adjust the
rigid point constellation from multi-view tracks, store it with the prop's
skeleton YAML" (Props section). Multi-view correspondence within one frame
(which blob in camera A is which blob in camera B) needs epipolar-
consistency matching when more than one dot is visible per camera -- that
doc's Question B, layer 2, already specifies this generically
(RANSAC over camera pairs, verified against a third view).

### 3.2 Live detection-time: labeling anonymous dots frame-by-frame

This is the real new work, and where a genuine architectural choice
exists. `marker-detection-analysis.md`'s Question B already designed the
general solution for anonymous markers on a person: **prediction-gated
assignment inside the tracker** -- "the UKF already predicts every
marker's 3D position each frame... labeling reduces to prediction-gated
assignment... exactly how commercial systems do online labeling." Verified
against the current C++ UKF: the Mahalanobis-distance primitive this needs
as a cost function already exists (used today for outlier gating of
already-labeled observations); the *assignment* solver over multiple
candidates (Hungarian/auction, or -- given only 7 dots -- a much simpler
greedy mutual-exclusion match would likely do) does not exist yet. That
doc says as much itself: "the *mutual-exclusion assignment* step is the
new part."

Two ways to actually build this, genuinely different in cost and where the
work lands:

**Option A -- tracker-side gated assignment (architecturally "correct").**
Pass anonymous per-frame dot candidates into the C++ tracker; at each
measurement update, project every named dot slot's *predicted* position
(from the UKF's current state, via FK -- already computed every step) into
each camera, gate candidates by Mahalanobis distance, resolve ties by
greedy mutual exclusion, treat matched candidates as ordinary Observations
for that step. Handles every case uniformly, including "no ArUco visible
this frame" (the UKF always has *a* prediction, even mid-gap, via
constant-velocity propagation) -- no separate cold-start logic needed
except at genuine track initialization. This is new tracker-side C++ work
(a real feature, reusable later for the general anonymous-person-marker
case in UC2, not thrown away), touching the measurement-update path.

**Option B -- detection-time (Python) slot resolution, no tracker
changes.** Resolve each frame's dot candidates to fixed named slots
*before* they ever reach the C++ side, reusing the exact manifest-based
`pose_sequence_keypoints`/`detection_keypoints` pipeline ArUco corners
already use (design doc §4.3) -- zero C++ changes. Needs a rough pose
estimate per frame to project expected dot positions against: trivial
when an ArUco is also decodable that instant (use its solved pose
directly), and needs the marker-mocap-algorithms.md §4.1 "unlabeled
rigid-template registration by pairwise-distance RANSAC" cold-start
mechanism (already designed generically, not yet implemented) for frames
with no ArUco anchor -- which is exactly the fast-motion case this whole
effort targets, so Option B can't actually skip building that piece
either. Cheaper than Option A only in the sense of not touching the C++
tracker; the hard part (labeling with no anchor) still needs real new
logic either way, just implemented once in Python against a rougher
(interpolated/RANSAC'd) pose guess instead of the tracker's own live
estimate.

**Recommendation, not yet decided**: Option A is more work up front but is
the design this project has already committed to conceptually, gets the
prediction quality of the tracker's real running state (not an
approximation), and isn't throwaway once UC2's person markers need the
identical capability. Option B is faster to a first result and touches
nothing on the C++ side, at the cost of building a narrower version of the
same hard sub-problem (cold-start labeling) that Option A gets for free
from the tracker's own state. Worth Harri's call given the effort
difference before either gets built.

## 4. Data model impact

Calibration-time (§3.1): none beyond what Phase A/B already write --
`type: reflective_dot` with just `center:` is already a supported
`marker_body_definitions` entry (`fiducial_markers.py`'s loader already
handles it).

Live detection-time: Option B needs no schema change at all (same fixed-
slot manifest ArUco already uses). Option A needs the C++ tracker to
accept an *unlabeled* observation stream alongside labeled ones for props
with a dot-bearing skeleton -- a real interface addition to
`SessionReader`/`ObservationSet`, not just internal UKF logic; scope that
properly before starting if Option A is chosen.

## 5. Phasing

- **C1 -- calibration-time dot geometry** (§3.1). Extends
  `calibrate_rigid_marker_body.py` directly; no cold-start needed, low
  risk, mostly reuses Phase A/B's own machinery.
- **C2 -- live per-frame labeling** (§3.2, Option A or B per Harri's
  call). The real new capability; needs the pairwise-distance RANSAC
  cold-start piece regardless of which option is chosen.
- **C3 -- re-run and compare.** Re-run the exact "Harri bokken" trial with
  dots included; compare against the 2026-09-01 baseline (0 lost /
  52.3% tracked / 8.4px reprojection error / jitter at fast-motion points)
  on tracked-step fraction, reprojection error, and specifically gap
  duration during fast segments (§1) -- the actual question this whole
  effort exists to answer.

## 6. Open questions

- **Retroreflective vs. colored, for this specific capture** (§2) --
  needs a look at the real footage before detector parameters can be
  chosen; not yet done.
- **Option A vs. B** (§3.2) -- real effort/architecture fork, not yet
  decided.
- **Symmetry/degenerate layouts**: `marker-detection-analysis.md`'s Props
  section flags that a rotationally-symmetric constellation leaves roll
  physically unobservable. Not a concern for this sword (asymmetric
  4-dots-one-side/3-dots-other-side layout, confirmed by Harri), but worth
  keeping in mind for future props.
- **Noise proposal for dots**: same per-marker `noise_std` open item
  Phase A/B already flagged -- dots and ArUco corners will have
  different, and possibly quite different, real detection precision;
  worth measuring once C1 has real residuals rather than assuming.
