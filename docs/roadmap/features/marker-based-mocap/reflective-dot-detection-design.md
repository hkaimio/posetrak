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

**Checked against real frames from the baseline run's biggest gap
(2026-09-01, §2.1)**: blur turned out not to be the whole story. Of the
individual misses inspected by eye, one showed the marker plate genuinely
edge-on to the camera (a viewing-angle failure, not blur — a flat ArUco
patch simply can't be read past a certain obliqueness, no matter how
sharp the frame is); another showed a clean, sharp, well-angled marker
that still produced no tracked observation (most likely a decode-margin
failure invisible at a glance — `min_marker_perimeter_rate` or a
confidence cutoff, not something visibly wrong with the frame). A small
round dot is more forgiving on *both* counts: it stays visible across a
much wider range of viewing angles than a flat decodable square (no
"read the pattern face-on" requirement at all), and doesn't carry ArUco's
fine bit-pattern margin to begin with. So the mechanism is broader than
originally framed — not just "blur-tolerant" but "robust to viewing angle
and decode-margin failures generally" — which if anything strengthens the
case for dots helping here, just for more reasons than the original
motion-blur framing alone.

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
magnitude cheaper than pose estimation."

### 2.1 Confirmed against real footage, with a working prototype (2026-09-01)

**Retroreflective, not colored** — settled by direct inspection: the dots
are bright, blown-out white points (the classic retroreflective-under-
on-axis-LED signature), and the capture visibly uses clip-on LED ring
lights on several tripods. Per Harri: `pixel7` has no ring light attached
and won't see any dots at all; the other Android phones' camera pipelines
may apply local tone-mapping that complicates a simple threshold approach.
**Start with the two GoPro cameras** (cleanest, most linear pipeline,
ring-lit) and extend to the other camera models only once those work.

`python/tools/prototype_dot_blob_detector.py` (throwaway spike, not
integrated) implements the method above with a compactness filter added
from the start (`4·π·area / perimeter²`, rejects elongated glare streaks
that pass a brightness+area test but aren't a round dot) and ran it
against one GoPro's footage across the baseline run's single biggest
observation gap (53.6–55.1s, §1) — exactly where the ArUco-only pipeline
produced zero usable observations for the whole 6-camera tracker:

- **Median 4 candidates/frame, only 14% of frames empty** (359 frames
  scanned) — real detections landing right where ArUco found nothing.
- **The area histogram cleanly separates three populations**: a tiny
  (~8px²) recurring false positive at a fixed image location (a static
  floor-mat glint — confirmed stationary across frames, i.e. not the
  moving sword), the real dots (30–70px², matching the tip + 3 dots near
  the marker plate visible by eye), and large (100–400px²) stationary
  blobs confirmed to be the LED ring lights on *other* tripods, not the
  sword. `min_area≈20, max_area≈90` (both already prototype parameters)
  cleanly isolates real dots from both false-positive sources — no need
  for anything beyond brightness+area+compactness at the detection stage.
- Manually checked the individual zero-candidate frames too: genuine,
  benign per-camera misses (the sword's dot-bearing face turned toward a
  different part of the room), exactly the case the multi-camera rig
  exists to cover, not a detector failure.

This is enough to design the Hungarian/Mahalanobis assignment architecture
(§3.2) against real numbers rather than assumptions -- next step there is
Harri's own call on Option A vs. B before more code gets written.

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
candidates does not exist yet. That doc says as much itself: "the
*mutual-exclusion assignment* step is the new part."

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

**Decided 2026-09-01: Option A.** Harri's call, explicitly for production
quality over a quick demo: gets the prediction quality of the tracker's
real running state (not an approximation), isn't throwaway once UC2's
person markers need the identical capability, and doesn't inherit Option
B's need to fake a pose estimate for the exact frames (no ArUco anchor)
this whole effort targets. Also explicitly **do not assume a fixed dot
count** (not "exactly 7") -- the assignment solver must handle a variable-
size candidate set per frame, which rules out a count-specific shortcut
and points at a real combinatorial assignment solver rather than anything
hand-tuned to "7". Concretely: the **Hungarian algorithm over a
Mahalanobis-distance cost matrix** (predicted dot-slot positions × observed
candidates, gated -- an entry above the outlier threshold is not a valid
assignment at all, handled by a dummy/infinite-cost row or by excluding
that pair from the assignment problem rather than forcing a match).

## 4. Data model impact

Calibration-time (§3.1): none beyond what Phase A/B already write --
`type: reflective_dot` with just `center:` is already a supported
`marker_body_definitions` entry (`fiducial_markers.py`'s loader already
handles it).

**Option A needs the C++ tracker to accept an *unlabeled*, variable-size
observation stream alongside labeled ones** for props with a dot-bearing
skeleton -- a real interface addition to `SessionReader`/`ObservationSet`
(today's manifest resolution assumes a fixed slot per keypoint index; an
anonymous candidate has no slot until assignment resolves it) and to the
skeleton/DB representation of "this prop has N named dot slots with
calibrated local positions, to be matched at tracking time" -- not just
internal UKF logic. This is the design round Harri flagged as needing to
happen before Hungarian-assignment code gets written; §7 below is a first
cut at what it needs to cover, not the round itself.

## 5. Phasing

- **C1 -- calibration-time dot geometry** (§3.1). Extends
  `calibrate_rigid_marker_body.py` directly; no cold-start needed, low
  risk, mostly reuses Phase A/B's own machinery.
  **Revised 2026-09-02 after real-data testing**: the automatic
  (`--detect-dots`) path this originally described -- restrict to
  instants where >=2 cameras each see exactly one candidate, avoiding
  general multi-view correspondence -- turned out not strict enough: two
  cameras each having exactly one candidate does not mean those
  candidates are the same physical point, and real footage produced a
  triangulated "dot" over 3 meters from the sword before a reprojection
  check was added, and still an implausible one after. Confirms
  `marker-detection-analysis.md`'s original recommendation (verify
  against a third view, not just two) was the right call to have
  followed the first time. Real reference+dot co-occurrence in the
  "Weapon test 2026-08-20" capture is also sparse enough (41 of 680
  buckets across the full ~66s range) that a properly-strict 3-view
  requirement would likely yield too few samples from this specific
  footage regardless. **Decided (Harri): manual annotation for the sword
  now** -- `tools/annotate_dots_manually.py`, human-confirmed
  correspondence via clicking a handful of known-good instants, reusing
  the same reprojection-checked triangulation. Automatic calibration
  (3-view verification, or single-camera multi-frame resection against a
  continuous tracked trajectory instead of requiring simultaneous
  multi-camera co-occurrence at all) remains a real goal for general use,
  not built this round -- see status.md's 2026-09-02 entry for the full
  account.
- **C2 -- live per-frame labeling, Option A** (§3.2). The real new
  capability: Hungarian/Mahalanobis assignment inside the C++ tracker,
  plus the schema/skeleton/`ObservationSet` design round this needs first
  (§7). Also needs the pairwise-distance RANSAC cold-start piece for
  genuine track initialization with no ArUco anchor at all.
- **C3 -- re-run and compare.** Re-run the exact "Harri bokken" trial with
  dots included; compare against the 2026-09-01 baseline (0 lost /
  52.3% tracked / 8.4px reprojection error / jitter at fast-motion points)
  on tracked-step fraction, reprojection error, and specifically gap
  duration during fast segments (§1) -- the actual question this whole
  effort exists to answer.

## 6. Open questions

- ~~Retroreflective vs. colored~~ -- resolved 2026-09-01 (§2.1):
  retroreflective, confirmed by direct inspection.
- ~~Option A vs. B~~ -- resolved 2026-09-01 (§3.2): Option A (tracker-side
  Hungarian/Mahalanobis assignment), for production quality and because a
  fixed dot count can't be assumed.
- **Symmetry/degenerate layouts**: `marker-detection-analysis.md`'s Props
  section flags that a rotationally-symmetric constellation leaves roll
  physically unobservable. Not a concern for this sword (asymmetric
  4-dots-one-side/3-dots-other-side layout, confirmed by Harri), but worth
  keeping in mind for future props.
- **Noise proposal for dots**: same per-marker `noise_std` open item
  Phase A/B already flagged -- dots and ArUco corners will have
  different, and possibly quite different, real detection precision;
  worth measuring once C1 has real residuals rather than assuming.

## 7. Design round scope for C2 (Option A)

**Done 2026-09-01** — full design round in
[dot-assignment-architecture-design.md](dot-assignment-architecture-design.md):
DB schema (new variable-row-count `detection_dot_candidates`/
`pose_observation_dot_candidates` tables, no manifest entries for dot
landmarks), skeleton representation (no format change -- a new
`unlabeled_points` `input_tracks:` type, already accepted by the loader
today since `type` is parsed as an opaque unvalidated string),
`SessionReader`/`ObservationSet` additions (a new `UnlabeledCandidate`
type alongside `Observation`), the exact `Tracker::run_parent_step()`
integration point (between `predict()` and `update()`, reusing
`prior_state`/`prior_cov` it already computes), a closed-form (not
sigma-point) Mahalanobis cost function specific to rigid-body skeletons,
and a hand-written Hungarian solver (no existing dependency covers this,
problem sizes are small). Below is the original first-cut agenda that
round worked from, kept for context:

- **Skeleton/DB representation of a variable-size dot set.** Today's
  `Marker`/manifest model assumes a fixed, known-at-load-time slot per
  keypoint index (an ArUco corner's `keypoint_idx` always means the same
  thing). A calibrated body's dots are still each a *named*, fixed local
  position (calibration doesn't change) -- what's unlabeled is only which
  *observed* candidate maps to which name, per frame. So the skeleton/DB
  side may not need to change much (dots are still named `Marker` entries
  with calibrated offsets); what changes is that `SessionReader` can no
  longer assume a manifest row's `keypoint_idx` deterministically picks
  out the right observation -- assignment has to happen somewhere between
  "raw candidate" and "which named marker did this update."
- **Where in the C++ pipeline unlabeled candidates enter.** Today
  `SessionReader::load_observations()` produces one `Observation` per
  resolved, already-labeled marker per camera per frame, consumed by
  `Tracker`/`UKF::update()`. Option A needs a new stage between "load
  observations" and "measurement update" (or folded into the update step
  itself) that: gets each dot-labeled marker's *predicted* position that
  frame (from the current UKF state via FK), projects into each camera,
  builds a cost matrix against that camera's unlabeled candidates
  (Mahalanobis distance), solves the assignment (Hungarian), and turns
  matched pairs into ordinary labeled `Observation`s before the existing
  update math ever runs -- ideally without that downstream math needing to
  know assignment happened at all.
- **Storage for raw unlabeled candidates.** `detection_keypoints`/
  `pose_observations` are fixed-width blobs today (one manifest-width row
  per frame/camera). A variable-count candidate list per frame/camera
  needs a different shape -- worth checking whether this is a new table,
  a variable-length blob, or reusing something closer to how
  `MarkerCornerObs`/`FiducialDetection` already represent detections
  in-memory on the Python side before they get resolved to fixed slots.
- **Gating/no-match policy.** `marker-detection-analysis.md`'s own
  "ambiguity policy — drop, don't guess" should carry over directly: an
  unresolved candidate (cost above the Mahalanobis gate for every slot) or
  an unresolved slot (no candidate within gate) is a dropped observation
  for that step, not a forced low-confidence match -- consistent with how
  the tracker already treats missing observations.
- **Testing strategy.** The assignment solver itself (given a set of
  predicted positions and a set of candidates with known ground-truth
  correspondence, does it recover the right assignment, including
  under partial occlusion and extra spurious candidates) is unit-testable
  independent of any real capture -- worth building that test harness
  alongside the solver rather than only validating against real footage.
