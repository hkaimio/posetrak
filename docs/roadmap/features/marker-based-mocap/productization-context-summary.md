# Marker-based mocap — branch summary for productization planning

**Purpose of this document**: a single, self-contained briefing on
`hkaimio/marker-based-mocap` (97 commits over `main`, 2026-08-19 through
2026-09-17) for an agent doing deep-dive productization-planning analysis.
It summarizes what has been built and validated, what was prototyped but
never productized, and what's still genuinely open — with pointers into
the much more detailed existing docs rather than duplicating them. Start
here, then follow the links for depth.

**Primary sources** (all in this same directory unless noted):
- [`marker-mocap-brief.md`](marker-mocap-brief.md) — the original ask (short).
- [`marker-mocap-design.md`](marker-mocap-design.md) +
  [`marker-mocap-algorithms.md`](marker-mocap-algorithms.md) — the base
  architecture/algorithm design, still largely current.
- [`marker-mocap-productization-plan.md`](marker-mocap-productization-plan.md) —
  the existing merge-to-`main` bar and plan for the **rigid-prop** case (UC1).
- [`person-marker-mocap-productization-plan.md`](person-marker-mocap-productization-plan.md) —
  the companion plan for **person-worn markers** (UC2).
- [`marker-catalog-and-assignment-redesign.md`](marker-catalog-and-assignment-redesign.md) +
  [`person-marker-assignment-current-state.md`](person-marker-assignment-current-state.md) —
  the person-marker assignment algorithm's current (flawed) state and its
  in-progress redesign. **Note**: as of this writing these two files are
  uncommitted, local work from a parallel session — read them for the
  current thinking, but confirm their status before treating them as settled.
- [`status.md`](status.md) — the full chronological lab notebook (4700+
  lines). Every claim below traces to a dated entry there; search it by
  date for the full account of anything summarized here.

---

## 1. What has been achieved

### 1.1 Rigid prop tracking (UC1) — the most mature path

A prop (sword, pen, pad, calibration box, a fully-reflective ball) can be
registered as a `capture_objects` row, detected, finalized, and tracked
through the **real production UKF tracker** (`posetrak-tracker`), end to
end, via CLI. Built in six phases (1a-1f, commits `fe6d496`..`79f169a`,
2026-08-19 to 2026-08-31):

- ArUco marker detection as its own detection-run type, independent of
  the person pipeline.
- Prop skeletons **generated** from a marker-body definition YAML
  (`marker_body_to_skeleton`), not hand-authored per object.
- `capture_objects` as first-class capture participants (own table, own
  detection runs, own sequences), reviewed and finalized through the same
  mechanisms as person detection where they apply.
- Rigid Kabsch/Umeyama initialization in the C++ tracker, with a real
  fix for a silent-bad-fallback bug (rigid init at a frame with
  insufficient coverage used to fall back to tracking from the world
  origin with a warning; now searches forward for a valid window and
  throws loudly if none exists — `Skeleton::is_rigid_body()`,
  `init_search_window_s`).
- Reflective-dot (anonymous marker) detection and assignment, entirely
  separate from ArUco: `dot_blob_detector.py` (round + motion-blur-streak
  classification, background-subtraction or brightness+blacklist modes,
  per-camera threshold calibration, chroma filtering), `DotTrackletLinker`
  (per-camera temporal identity), a shared cross-subject dot-assignment
  phase in the C++ tracker (Hungarian solver, one cost matrix per camera
  across every participating subject, closed-form rigid-body prediction),
  and Phase B tracklet-aware gate relaxation (a candidate matching the
  previous frame's resolved tracklet gets a relaxed Mahalanobis gate).

**Real, measured validation results** (not just "should work"):
- Sword prop: **91.4% tracked** (6057/6626 frames) after dot-detector
  production improvements, up from an 83.5% baseline.
- Pen: **97.2% tracked** (1747/1798 steps), rigid-init RMS 1.8mm.
- Pad: **94.9% tracked** (1708/1800 steps), rigid-init RMS 3.6mm.
- Both cross-validated against an independent Blender export of the same
  trajectory (identical physical story from two separate computation paths).

  Harri comments:
  - The dot tracking worked reasonably well, except in extremely fast moving objects (like the tip of the sword). The sword case was not successful due to insufficient marker target coverage (all markers were visible only from one side and none from front, so the sword rotation could not be tracked)
  - Pen & pad tracking worked reasonably well; the results were accurate enough to reconstruct the written text from tracking results

An existing plan
([`marker-mocap-productization-plan.md`](marker-mocap-productization-plan.md))
already defines the specific merge-to-`main` bar for this case (schema
validated for multi-subject, CLI-complete + GUI-minimal, detection
performance in the few-minutes range) and what's still missing to meet
it — see §3/§4 below rather than re-deriving it here.

### 1.2 Person-marker augmentation (UC2) — proven once, not yet general

A person's markerless tracking can be **augmented** with real reflective
dots at anatomical landmarks. First (and so far only) end-to-end run,
2026-09-12, on a real capture with a 16-marker `leg` module:

- **100% tracked** (11587/11587 steps), essentially identical NIS/dof to
  the no-dots baseline (no regression from adding dots).
- All 16 dot slots picked up real observations through the new
  articulated `UnscentedKalmanFilter::predict_marker_slot()` path (sigma
  points projected through FK, no measurement noise added, mirroring the
  rigid-body prediction's convention) — frame coverage 46.8%-87.3% per
  slot, median reprojection error 11.8-40.4px, near-zero outlier rate on
  most slots.
- A real calibration bug (`ankle_lat_L`, starved to 28 cross-camera
  samples) was found and is understood, not silently present.
- Real, measured performance fixes the same week: dot-slot prediction
  **14.9x** faster (61.5→4.1ms/frame, OpenMP + batching), and a
  `get_all_in_range()` linear-scan-to-binary-search fix, **770x**
  (33.9→0.044ms/frame) — the second one benefits every tracking run in
  the project, not just marker-augmented ones. Full per-frame budget is
  now individually accounted for down to ~2% (`update()`'s own Kalman-gain
  computation is 58.9% of the frame — see §4).

**But**: every stage of this pipeline except detection and tracking
itself is a **standalone, hardcoded-to-one-capture script** — calibration
fitting, the marker-module "catalog," and the review UI all need
generalizing before a second module (torso+arms is the stated next
target) or a second person is remotely easy. See
[`person-marker-mocap-productization-plan.md`](person-marker-mocap-productization-plan.md)
§1's own inventory table for exactly which stages are general already
and which aren't.

**The assignment *algorithm* itself (which raw dot goes to which named
marker) was found wrong on real data** and is mid-redesign, not
implemented: the original approach (group same-joint markers by a single
shared anchor, resolve ambiguity by a pelvis-proxy anatomical direction,
assign per-frame then majority-vote) failed on real footage — two markers
close in 3D but not at the same joint is the dominant real confusion, not
same-joint ambiguity; the pelvis-proxy direction is wrong for anything
that rotates independent of the pelvis (thigh long-axis rotation).
[`marker-catalog-and-assignment-redesign.md`](marker-catalog-and-assignment-redesign.md)
records the corrected direction (tracklets as the unit of assignment, a
proper per-parent-joint marker catalog, direction from FK not a proxy,
real ground-truth metrics) as a design, with implementation not yet
started.

### 1.3 The "3rd case" — a fully-reflective ball with no coded pattern

Not one of the brief's original use cases, but became the project's
hardest real stress test: an object with **exactly one** marker (a single
reflective dot, no ArUco, no orientation ever observable) and no visual
signature a general-purpose detector or segmenter reliably locks onto.

**What was tried and rejected, each for a documented, verified reason**:
- Reflective-dot detection + multi-view RANSAC correspondence alone:
  worked for some throws/cameras, not reliably across a whole throw with
  one initialization (per-camera reliability inconsistent — a global area
  threshold can't separate the ball from body markers across cameras with
  different focal lengths/distances).
- Cutie/SAM2 video-object segmentation, mask centroid as the 3D position
  source: looked numerically plausible (a believable-looking trajectory)
  but was confirmed **visually wrong** — the mask drifted onto a body part
  in every camera, in two separate attempts. Rejected as a ball detector.
  Adding non-person objects as selectable Cutie segmentation targets
  (`a8218f7`) was kept anyway — useful as a dot-assignment hint, not as a
  position source (see §2).

**What worked, and required real architecture fixes to get there**:
manually tracking the ball in Blender's Movie Clip Editor (human-verified
2D curves, cheap to do, 2-3 re-seeds per camera to survive occlusion) and
importing the result. This needed genuine new C++ tracker capability that
didn't exist before — a rigid body with a single anonymous marker had **no
path to ever initialize at all** (three separate fatal checks assumed
every tracked subject has either 3+ markers or some labeled observation).
Fixed and committed (`4d0f315`): single-marker Kabsch bypass (no
orientation fit needed/possible with one point), tolerance for a
dots-only skeleton's zero labeled observations, and a `--seed-position`
CLI escape hatch for cold-starting a skeleton with no observation to
search from. Import tooling committed (`6f41b5b`).

**Result, visually verified via reprojection video, not just trusted
numerically**: median reprojection error 12-57px per camera when 3
cameras have real coverage, 72-96px when only 2 do (an expected, explained
consequence of losing a redundant view, not a new bug).

**A genuinely important bug found and fixed along the way** (see §4 for
why this matters beyond the ball case): the first attempt folded in
PoseTrak's own automatic reflective-dot detector for a 4th camera
(reasoning: the tracker's own outlier gate should sort real candidates
from clutter, the same way it does for labeled person markers). It
didn't — a **confidently-wrong** static clutter candidate (sharp, low
apparent noise, just the wrong physical point) intermittently won
assignment over the real, moving ball, and no per-frame Mahalanobis gate
tuned for genuine detection noise could tell "confidently wrong" from
"correctly right" using one camera's geometry alone. Fixed by dropping
that camera's unfiltered feed entirely.

---

## 2. Prototyped features worth carrying into the productized version

Ranked roughly by how much of the hard part is already done vs. how much
is still genuinely new work.

1. **Segmentation (Cutie/SAM2) as a dot-assignment cost-relaxation cue**,
   not a position source. Already scoped in the prop plan (§5 point 3)
   and the person plan (§2.4) as the *same* mechanism serving two
   purposes: disambiguating whose candidate a dot belongs to (person A
   vs. person B, or object vs. background) by scoring it against a
   per-frame Cutie mask, composing with the tracklet-gate relaxation
   already built for Phase B. Non-person objects are already selectable
   Cutie targets (`a8218f7`); the scoring-into-cost-matrix wiring is not
   yet built. Real, non-obvious risk flagged in both plans: mask accuracy
   on a thin/fast prop is unvalidated and should stay a soft term, never
   a hard reject.
2. **Cross-camera geometric consistency as an assignment pre-filter**
   (new idea from the ball investigation, not yet in either existing
   plan). `prototype_ball_tracking.py`'s multi-view RANSAC consistency
   check (does this candidate's implied 3D position get independent
   corroboration from other cameras?) directly targets the
   "confidently-wrong single candidate" failure class found twice now
   (ball case; independently, in production leg-dot tracking's
   2026-09-14 occlusion-reacquisition finding, §4). Smaller and more
   isolated to validate than full tracklet-gating.
3. **External 2D-track import** (Blender-sourced, this session) as a
   real, general feature rather than a one-off script — useful for
   debugging any object/person, and as an escape hatch when automatic
   detection genuinely can't solve a case (the ball). The single-marker
   tracker support it needed is committed and general, not ball-specific.
   Export (the reverse direction) doesn't exist yet and was flagged by
   Harri as probably wanted too.
4. **ArUco-anchored + Cutie-manual-click object initialization**, ranked
   in the existing prop plan (§5) as the two init signals to build in
   that order — ArUco already works and should stay the default; the
   manual-click Cutie path is "nearly free" (`CutieInitPanel` already
   supports it structurally) and validates whether mask-based cueing is
   worth pursuing further before investing in point 1 above.
5. **A generic "Advanced" tracker-config tab** driven by schema
   introspection (prop plan §3.3) — every new dot-tuning field today
   needs a hand-authored widget on top of the five-touchpoint pattern for
   adding the field itself; a fallback generic form removes "invisible to
   the GUI" as a risk for every future tunable, without blocking a
   real widget where one's deserved.
6. **Soft per-dot classification via local hue/color** (Harri's
   recurring idea, not yet tested at all). This session's own
   clutter-candidate failure (§1.3) is a ready-made real test case for
   it — worth a cheap validation before deciding whether to build it in.
7. **Segmentation-mask-as-ROI-filter for raw dot candidates** (restrict
   which candidates are even considered, rather than trust a mask
   centroid as a position) — a weaker, more robust reframing of how
   segmentation and dot-detection combine than what was tried for the
   ball. Untested; the ball's own mask+dot data already exist to test it
   against cheaply.

Harri comments:
- Another idea tested as part of the capture sessions used for this that seems worth maintaining: We used video with moving box with aruco markers attached t it as extrinsics calibration target: we selected corresponding frames from all cameras where the box was stationary used these. The extrinsics reprojection error was significantly smaller than the previous mehtods used (~2 px vs >10 px in calibrations done from scene landmarks)

---

## 3. What's CLI-complete but has little or no GUI today

(Detail: prop plan §3, person plan §1's inventory table.)

- Detection parameter tuning (dot background-subtract/threshold/
  saturation) — CLI/script only, not in `RunDetectionDialog`.
- Dot review/edit — **no GUI path exists at all**; every dot review this
  project has done went through a standalone debug-frame-rendering
  script. A design for resolved-slot editing (the direct analogue of
  today's ArUco-corner/pose-keypoint editing) exists (prop plan §3.2) but
  isn't built. Raw-candidate relabeling before assignment is scoped as
  harder, explicit future work.
- Most dot-tuning tracker-config fields have no dedicated widget (only
  `outlier_threshold` does).
- Marker-body/module authoring — entirely YAML + one-off Python scripts;
  no UI at all for defining a new prop or person-marker module.
- Person-worn-dot finalization — a validation-scoped stopgap script
  (`copy_dot_candidates_to_sequence.py`), not a production entry point.
- Multi-subject (person+prop together) tracking launch — the underlying
  mechanism works (see §4), but `RunTrackerWidget`'s roster only knows
  about `capture_persons` rows today.

---

## 4. Major open items

Genuine unresolved problems, not just "not built yet" — ordered
roughly by how structural/cross-cutting each one is.

1. **Re-initialization after full occlusion has no design at all.**
   `Tracker::initialize()` is documented "call at most once... construct
   a new `Tracker` instead" — there's no concept of one logical tracking
   result spanning multiple disjoint segments, no gap-detection, and (per
   this session's own productization notes) no agreed way to represent a
   gap in exported BVH, which has no native timeline-gap concept. Flagged
   explicitly in the prop plan (§5) and hit directly by the ball case
   (which sidesteps it by tracking each throw as a separate run and
   leaving stitching to the user/animation software).

2. **A confidently-wrong candidate can beat the outlier gate — confirmed
   independently twice, still unfixed generally.** The ball
   investigation's root cause (§1.3) — a sharp, low-apparent-noise but
   physically-wrong candidate passing the same Mahalanobis gate tuned for
   genuine detection noise — is *not* a ball-specific problem: the
   2026-09-14 person+prop A/B test independently found and root-caused
   the identical failure shape in production leg-dot tracking (ankle
   markers reacquiring at a wildly discontinuous position after a 2-frame
   gap, with nothing flagging it as wrong). This is a real, twice-observed
   architectural gap in assignment/outlier-rejection design. The Phase B
   tracklet-gating mechanism already designed targets exactly this class
   of problem but hasn't been extended to cover it generally, and item 1
   above (cross-camera consistency pre-filter) is a second, independent
   angle on the same gap, also not yet built.

3. **Video-decode threading has a real, unresolved safety question.**
   A documented, timing-dependent hang (closing a multi-threaded FFmpeg
   decode context early) forced single-threaded decode as the safe
   default; re-enabling threading measures ~4.3x faster but a handful of
   clean tests is explicitly *not* meaningful evidence the original bug
   is actually fixed (it was diagnosed as more likely under sustained
   production load, not less). Blocks the detection-performance merge-bar
   item until resolved one of: a real timeout-guarded safe close, always
   decoding to true EOF, or a different decode backend.

4. **Person-marker assignment's core algorithm needs a real redesign,
   not a tune.** See §1.2 — the catalog/grouping/direction approach that
   was actually tried failed on real data for structural reasons, not
   parameter-tuning ones. The redesign doc exists; nothing from it is
   implemented yet.

5. **Multi-subject (person+prop) coupling doesn't yet help, and one
   specific side-effect isn't root-caused.** The one real A/B test run
   (2026-09-14) found no measurable hand/prop agreement improvement from
   coupling, plus a 3-4x localized jitter spike in one specific window
   tied to a contact-pair activating/deactivating — real, reproducible,
   not yet understood well enough to tune away.

6. **Kalman-gain cost is already the single largest per-frame cost
   (58.9%) and is unprofiled at anything beyond a 16-marker module.**
   With a Vicon-scale marker set (~53+24 markers, more than doubling
   today's ~366 pose observations) and Kalman-gain computation typically
   scaling worse than linearly in observation count, this is a real,
   currently-unquantified scaling risk for any module bigger than `leg` —
   flagged explicitly, not yet investigated.

7. **Object initialization has no answer for "reappeared after leaving
   every camera's frame."** Both existing init signals (ArUco, Cutie
   manual-click) find a *first* valid window; neither addresses
   reappearance, which is really the same gap as item 1.

8. **No schema or UI concept of "this object's tracker input is sourced
   from N detectors, combined by these rules."** Every real combination
   in use today (rigid prop: ArUco + dots + segmentation-as-hint; person:
   markerless + dot-augmented; the ball: single external track only, no
   orientation ever observable) is wired up by its own bespoke script.
   This is the natural next abstraction once 2+ more real combinations
   exist to design it against — deliberately not designed prematurely
   from one or two data points.

9. **Grip anchor points** (where a prop is "held," for phase-3
   person+prop coupling, including handoff/regrip/free-flight) have no
   answer for where they live in the data model. Open question in the
   prop plan (§7); not resolved by anything built since.

10. **Hand-marker physical feasibility is unresolved and may be a
    non-issue.** Marker size vs. finger-segment length and occlusion
    during a grasp is a real open capture-hardware question; Harri's own
    proposed sidestep (markerless hand pose seeded from a good,
    marker-derived wrist position) is untested but could make the
    marker-based version of this unnecessary — worth checking before
    investing in it.

11. **UC3 (moving-camera detection/recovery)** — the brief's third,
    lowest-priority use case (fiducial markers on the environment,
    tracked during a trial to detect and correct for camera movement) —
    has essentially no implementation work in this branch's history.
    Lowest priority per the original brief; flagging so it isn't assumed
    "later" without noting "later" currently means "not started."

12. **Reflective-dot per-camera reliability is inherently uneven** — a
    real, measured finding from the ball investigation (some cameras are
    reliable for a given object/lighting setup, some aren't, and this
    isn't predictable from resolution/framerate alone) with no general
    per-camera trust/confidence mechanism to act on it systematically
    yet (§4's item 8 territory: this is a specific case of "different
    sources need different trust weighting").

---

## 5. A note on process, for whoever plans next

Two working patterns recur throughout this branch's history and are
worth preserving in whatever comes next, not just the technical content:

- **Prototype scripts before touching production code**, validated
  against real capture data (not synthetic fixtures alone) before
  porting anything in. Nearly every real feature above went through a
  `python/tools/prototype_*.py` or standalone-script stage first.
- **Trust visual inspection over aggregate numeric plausibility.** The
  Cutie-segmentation ball attempt and the first Blender-import tracking
  attempt (§1.3) both looked numerically fine and were both actually
  wrong in ways only a rendered video caught — this has now happened
  more than once and should be treated as a standing rule, not a
  one-off lesson.
