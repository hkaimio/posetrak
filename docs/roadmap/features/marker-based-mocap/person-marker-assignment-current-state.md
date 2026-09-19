# Person marker assignment — current state (2026-09-10)

> **Under review (Harri feedback, 2026-09-10).** Stages C-E's core
> approach is being reconsidered, not just tuned. Specifically: (a) marker
> grouping by a single shared joint anchor is wrong — markers can sit
> mid-bone or near the child joint, and markers that are close in 3D
> often project close in one camera view (identified as the single most
> common mislabeling cause in the latest video); (b) the pelvis-proxy for
> anatomical direction (Stage C) is too weak — the intended design is a
> normal expressed in the *parent joint's* coordinate frame, transformed
> to world via forward kinematics from the markerless tracking result
> (calibration) or the tracker predict step (live); (c) tracklets should
> probably be the primary unit of assignment, not per-frame slots; (d)
> the metrics below ("purity", "coverage") are ad-hoc and need a proper,
> agreed definition with ground truth. A design phase precedes further
> implementation. This document still accurately describes what the code
> does *today* — treat it as the "before" reference for that redesign.

## Why this document exists

`person-marker-assignment-design.md` records the *design* and its phased
plan (P1-P7). Since then, a single fast-moving prototyping session built,
broke, diagnosed, and fixed a long chain of real issues across several
scripts, each building on the last. `status.md` records that history
chronologically, dated entry by entry — useful for "what happened and
why", hard to use for "how does the thing actually work right now".

This document is the second kind: a from-scratch description of the
**current** dot-detection and person-marker-assignment pipeline, as
actually implemented today, with no history in it. When it drifts out of
sync with the code (it will), trust the code and fix this doc, not the
other way round.

**Scope note**: dot *detection* (below, Stage A) is production code,
merged and tested. Everything from Stage B onward (fusion, hybrid
assignment, marker normals, tracklet smoothing) is **prototype-only** —
`python/tools/prototype_*.py` and `python/tools/render_*.py` scripts, not
wired into `MarkerDetectionPipeline` or the C++ tracker. It exists to
validate the assignment approach against real capture data before any of
it is ported to production (see the design doc's phased plan, P4).

---

## 1. Pipeline overview

One frame's full journey, top to bottom:

```
raw video frame (per camera)
    │
    ▼
[Stage A] dot_blob_detector.detect_blobs()          -- production
    → BlobCandidate list, one per camera, per frame
    │
    ▼
[Stage A] DotTrackletLinker                         -- production
    → each candidate gets a tracklet_id (temporal identity within one
      camera's own footage; does not know about other cameras)
    │  (stored in the DB as float32[N,9]: px,py,area,compactness,
    │   major_axis,minor_axis,dir_x,dir_y,tracklet_id)
    ▼
[Stage B] fuse_frame()                              -- prototype
    → exhaustive pairwise triangulation across all camera pairs, then
      reprojection-consensus grouping into FusedPoint objects, each
      possibly backed by 2+ cameras' candidates
    │
    ▼
[Stage C] compute_leg_frames()                      -- prototype
    → per-frame anatomical directions (medial/lateral/anterior) for
      each ambiguous joint (knee_L/R, ankle_L/R), derived from
      triangulated hip/knee/ankle pose keypoints
    │
    ▼
[Stage D] normal_aware_hybrid_assign_frame()        -- prototype
    → per-frame {camera: {marker_name: (px, py, confidence)}},
      combining a 3D pass (fused points, direction-aware for
      ambiguous groups) and a 2D per-camera fallback pass
    │
    ▼
[Stage E] run_tracklet_smoothed()                   -- prototype
    → majority-vote name correction per (camera, tracklet_id) across
      the whole clip, using Stage A's tracklet_id
    │
    ▼
rendered video / consumed by whatever comes next (tracker, review UI)
```

Each stage is its own section below, in this order.

---

## 2. Stage A — dot detection (production)

**File**: `python/posetrak/detection/dot_blob_detector.py`,
`python/posetrak/detection/marker_pipeline.py`

`detect_blobs(gray, threshold=235, min_area=4.0, max_area=400.0,
min_compactness=0.5, max_streak_length_px=60.0, background=None,
background_mode="subtract", blacklist_frac=0.7, blacklist_radius_px=6,
bgr=None, max_saturation=255.0)` finds anonymous reflective-dot
candidates in one grayscale frame:

1. Threshold the frame (raw brightness, or residual against a
   per-camera background image, depending on `background_mode`).
2. Contour each thresholded blob; compute its `minAreaRect` (major/minor
   axis) *before* any area-based rejection (an earlier ordering bug
   rejected legitimate motion-blur streaks and large round blobs by area
   alone — fixed, documented in the module's own docstring).
3. Classify: **round** (compact, `min_area <= area <= max_area`) or
   **streak** (`minor_axis` within a normal dot's width, `major_axis <=
   max_streak_length_px` — a fast-swing motion blur) or reject.
4. Optional chroma filter (`max_saturation`): rejects colored (skin,
   fabric) residuals a pure-brightness/residual check can't tell from a
   real near-white marker.

**`background_mode`** — two modes, same call signature:
- `"subtract"`: thresholds the *residual* after subtracting a per-camera
  median background image. Fails structurally when a marker is on the
  subject's own moving body: the marker's brightness is close to the
  (moving) skin/fabric behind it, so the residual is too small to clear
  threshold — the marker gets fused into the subject's own body blob and
  silently disappears.
- `"blacklist"` (current default for production runs): thresholds *raw
  brightness* directly, and uses the background image only to **veto**
  candidates whose raw brightness is not much above what the same spot
  was already showing *without* a subject there (`blacklist_frac`,
  `blacklist_radius_px`) — catches static false positives (glare, light
  fixtures) without the subtract-mode failure mode. Real measured
  improvement on ground truth: recall 65.1% → 80.7%, precision 50.0% →
  75.3%.

**Per-camera threshold calibration** (`dot_threshold_by_camera` on
`MarkerDetectionPipeline`): different cameras' sensors/tone-mapping cap a
real marker's brightness differently (GoPros saturate 251-254; some
phone cameras cap lower, ~194-232, due to local tone-mapping). A single
global threshold either misses real markers on the phones or merges
markers into skin on the GoPros — this dict overrides the global
`threshold` per camera.

**Tracklet linking**: after detection, `DotTrackletLinker` (existing
production code, originally built for the prop-dot case) assigns a
`tracklet_id` to every candidate — temporal continuity *within one
camera's own footage only*, no cross-camera knowledge. This is stored as
the 9th float in the DB blob (`encode_dot_candidates`/
`decode_dot_candidates`, `python/app/pose/db_cache.py`) and is the
temporal signal Stage E's smoothing uses.

---

## 3. Stage B — multi-camera fusion (prototype)

**File**: `python/tools/prototype_multi_camera_fusion.py`
**Function**: `fuse_frame(candidates_by_camera, states, max_reproj_px=8.0,
merge_radius_m=0.03) -> list[FusedPoint]`

Per (synchronized) frame, two stages:

1. **Seed 3D point hypotheses**: for every pair of cameras with
   candidates, triangulate every `(candidate_a, candidate_b)` combination
   (`cv2.triangulatePoints`), keep pairs with positive depth in both
   cameras and low reprojection error back into both. Greedily merge
   accepted pair-triangulations landing within `merge_radius_m` of each
   other in 3D into one seed hypothesis (position averaged across
   whichever pairs merged into it). This stage only produces candidate 3D
   *positions* — it does not decide final camera/candidate membership.

2. **Reprojection consensus** (rewritten 2026-09-10 — see below for why):
   reproject every seed hypothesis's 3D position into **every** camera
   with candidates this frame, not only the pair that produced it, and
   find each camera's closest candidate within `max_reproj_px`. Resolve
   all `(hypothesis, camera, candidate)` matches **globally** by greedy
   best-first claiming, sorted by reprojection error ascending, so one
   raw candidate ends up supporting at most one hypothesis — whichever it
   fits best — rather than silently supporting two different 3D
   interpretations of the same pixel. A hypothesis is kept only if it
   ends up with `>= 2` claimed views; its position is then refined by a
   proper N-view triangulation over its final claimed views
   (`triangulate_multiview`), not left at the pairwise-average estimate.

**Why the rewrite**: the original version only ever recorded, as a
`FusedPoint`'s `views`, the *specific pair* of cameras that happened to
seed it — a camera whose own candidate also reprojected close to the
resulting 3D point, but wasn't part of the winning pair, got no credit at
all. Two concrete real problems this caused, both found on real data:
- **Coverage loss**: a marker plainly visible in 3+ cameras only ever
  "counted" 2 of them, weakening every downstream confidence signal.
- **A real correctness bug**: two different seed hypotheses could each
  use the *same* raw pixel in one camera, paired with two *different*
  candidates in two other cameras, giving two different 3D positions from
  one 2D dot — each got assigned a different marker name downstream
  (`knee_L_medial` and `knee_L_lateral` both landing on the identical
  pixel at one real frame). The reprojection-consensus rewrite makes this
  structurally impossible: `matches.sort()` + greedy claiming guarantees
  no raw candidate is ever claimed by two different `FusedPoint`s.

**Trade-off, accepted deliberately** (Harri, 2026-09-10): this resolves
*more* ambiguity at the fusion stage than the original P7 design left
unresolved on purpose (competing 2-view hypotheses were meant to be left
for the assignment layer to weigh, not force-resolved by fusion). The
greedy claim can make a real second physical point lose its own
best-fitting camera views to a competing hypothesis that fit slightly
better globally. Measured net effect on the validation clip: fewer but
higher-quality fused points overall (ambiguous-group assignment count
dropped further, purity improved slightly) — quality over quantity, not
a strict coverage win.

`FusedPoint` fields: `xyz` (3D position), `views` (list of
`(camera_instance_id, candidate_idx)`), `max_reproj_err_px`.

---

## 4. The marker catalog and ambiguous groups

**File**: `python/tools/prototype_person_marker_assignment.py`

`_CATALOG: dict[name -> (vitpose_anchor_idx, group)]` — 16 leg markers,
each mapped to the vitpose-133 keypoint index used as its rough search
anchor, and a *group* name shared by every marker at the same physical
joint:

| Group     | Members                                  | vitpose idx |
|-----------|-------------------------------------------|-------------|
| hip_L/R   | 1 marker each (unambiguous)                | 11 / 12     |
| knee_L/R  | 3 markers: medial, lateral, front (ambiguous) | 13 / 14  |
| ankle_L/R | 2 markers: medial, lateral (ambiguous)     | 15 / 16     |
| heel_L/R  | 1 marker each (unambiguous)                | 19 / 22     |
| toe_L/R   | 1 marker each (unambiguous)                 | 17 / 20     |

`_UNAMBIGUOUS_GROUPS` = groups with exactly one marker — these can be
identity-resolved by proximity to their anchor alone. **Ambiguous
groups** (knee, ankle) cannot: every member shares the *same* vitpose
anchor index, so a plain nearest-candidate match has no way to tell
`knee_L_medial` from `knee_L_lateral` from `knee_L_front` — this is
exactly what Stage C/D's direction-aware matching exists to fix (see
below; before that fix, this was a real, confirmed source of arbitrary
mislabeling).

---

## 5. Stage C — per-frame anatomical directions (prototype)

**File**: `python/tools/prototypes/superseded/prototype_marker_normal_assignment.py`
**Function**: `compute_leg_frames(kp_by_cam, states) -> dict[joint_name -> {...}]`

For each ambiguous joint (`knee_L`, `knee_R`, `ankle_L`, `ankle_R`) that
has enough triangulated pose data this frame, computes:

- **`medial`**: unit vector from this leg's hip toward the *other* leg's
  hip (the pelvis-width vector), sign-flipped for the right side.
  "Medial" always means "toward the body's own midline" — this
  approximates it directly with no per-marker geometry needed.
  `lateral` is just `-medial`, not stored separately.
- **`anterior`**: `cross(world_up, pelvis_vector)` — **one shared
  direction for the whole pelvis** (both legs, both knee and ankle), not
  derived from either limb's own instantaneous thigh/shank direction.

**Why anterior is pelvis-anchored, not limb-anchored** (a real bug, found
and fixed by rendering the computed frame as arrows on a real image —
`render_marker_normal_debug_frame.py` — and checking anterior against
which way the subject was visibly facing): an earlier version derived
anterior from each limb's own thigh/shank direction, which swings with
hip/knee *flexion* (e.g. raising a leg) even though the knee's true
anterior-facing direction does not rotate with flexion — only hip axial
rotation would change it. The limb-direction version pointed backwards
on a raised, bent leg while looking correct on a straight standing leg;
the pelvis-anchored version matches on both. Known, accepted
simplification: true anterior would also track hip internal/external
rotation, which this doesn't model.

`world_up = (0, 0, 1)` — confirmed empirically (not assumed) by
triangulating a standing foot's ankle and checking it lands near Z=0.

---

## 6. Stage D — per-frame assignment (prototype)

**File**: `python/tools/prototypes/superseded/prototype_marker_normal_assignment.py`
**Function**: `normal_aware_hybrid_assign_frame(dots_raw_by_cam,
kp_by_cam, states, fusion_max_reproj_px=3.0, fusion_merge_radius_m=0.03,
match_radius_m=0.15, match_radius_px=80.0, min_direction_score=0.4,
disambiguate_2d_fallback=False) -> {camera: {marker_name: (px, py,
'3d'|'2d')}}`

This is the per-frame assignment entry point — everything above (Stages
A-C) feeds into it. Four passes, in order, each consuming the raw
candidates the previous passes didn't already claim:

1. **3D pass, unambiguous groups**: each unambiguous marker's own
   triangulated anchor position vs. every `FusedPoint` within
   `match_radius_m`, solved as one Hungarian (`scipy.linear_sum_assignment`)
   one-to-one match. Plain distance cost — no direction needed, there's
   only one marker per group.

2. **3D pass, ambiguous groups**: for each ambiguous joint with a
   computed anatomical frame (Stage C), find `FusedPoint`s within
   `match_radius_m` of the joint, then run a **separate, small Hungarian
   per group** (e.g. 3 names × N nearby fused points for a knee) scored
   by **directional agreement**: `cos(expected_direction,
   fused_point_offset_from_joint)`. A pairing scoring below
   `min_direction_score` is treated as infeasible (cost stays effectively
   infinite) rather than merely disfavored — "drop, don't guess" instead
   of accepting the least-bad option when nothing actually matches well.

3. **2D fallback pass, unambiguous groups, per camera**: for whatever
   unambiguous names are still unmatched, one-to-one match (plain 2D
   pixel distance, gated at `match_radius_px`) against that camera's own
   remaining candidates. This is P1's original mechanism, applied only to
   the fusion pass's leftovers.

4. **2D fallback pass, ambiguous groups, per camera** — **off by default**
   (`disambiguate_2d_fallback=False`). When enabled, this would project
   each ambiguous group's expected directions into the camera's own image
   plane and directionally match leftover 2D-only candidates the same way
   as pass 2, but in pixel space instead of 3D. **Why it's off**: a single
   camera's projected-offset geometry is far more sensitive to joint-
   position triangulation noise and to a real marker not sitting exactly
   on the idealized medial/lateral/front axis than the 3D pass is —
   confirmed on a real failing case where the only nearby candidate's
   real projected offset was ~32° off *every* modeled direction, so
   "medial" won only for being the least-wrong of three bad options
   (cosine 0.85) — not for being right. **A plain score threshold cannot
   catch this**: 0.85 looks like a confident match; it's simply the wrong
   one. Leaving this off means an ambiguous group with no cross-camera
   triangulation available that frame is simply left unassigned, rather
   than confidently mislabeled.

**Confidence tag**: every returned `(px, py, conf)` carries `'3d'` (pass
1/2, cross-camera confirmed) or `'2d'` (pass 3, single-camera fallback
only) so a consumer can tell them apart.

`hybrid_assign_frame` (`prototype_hybrid_person_marker_assignment.py`) is
the earlier, simpler version this extends — identical unambiguous-group
logic, but ambiguous groups are matched to an *undifferentiated shared
anchor* (every member of a group has the literal same triangulated
position, since they share one vitpose index) with no direction
awareness at all. It's kept as a baseline/comparison point, not because
it's still the recommended path for ambiguous groups.

---

## 7. Stage E — tracklet-majority-vote smoothing (prototype)

**File**: `python/tools/prototypes/superseded/prototype_tracklet_smoothed_assignment.py`
**Function**: `run_tracklet_smoothed(conn, shot_id, marker_detection_run,
pose_sequence, start_time, end_time, ref_camera_id=None,
assign_fn=hybrid_assign_frame) -> {video_frame: {camera: {name: (px, py,
conf)}}}`

Stage D is purely per-frame independent — no temporal continuity at all,
despite Stage A's tracklet linker already providing it. Two passes:

1. Run `assign_fn` (Stage D, or any function with the same signature)
   independently per frame. For each assigned `(camera, name)`, look up
   the assigned candidate's own `tracklet_id` by exact-position match
   against that frame's raw detection data (not threaded through Stage D
   itself — recovered here by position lookup, simpler than plumbing
   `tracklet_id` through every function's signature). Record every
   `(camera, tracklet_id, frame, name)` tuple seen.

2. **Rewrite pass**: for each `(camera, tracklet_id)`, take the
   majority-vote name across every frame that tracklet was assigned *any*
   name, and rebuild each frame's assignment dict from scratch mapping
   `tracklet_id -> majority_name`. **Must be a from-scratch rebuild, not
   an in-place `dict.pop`/`dict[key]=` mutation** — an earlier version did
   the latter and had a real mutual-swap-clobber bug: when two tracklets
   swap names within the *same* frame, correcting them one at a time on a
   shared dict lets the second correction overwrite the first, since both
   fight over the same two dict keys. Confirmed on a real case (`heel_L`
   and `ankle_L` briefly swapped names for 3 frames due to their 3D
   anchors coming within 2.6cm of each other) and fixed by rebuilding
   each frame's dict in one pass instead.

**Known, accepted limitation**: if the tracklet *linker itself* (Stage A,
production code, not touched by any of this) silently carries one
`tracklet_id` across what were actually two different physical dots (an
occlusion-recovery mistake, or two markers' 2D paths crossing), majority
voting forces the wrong name onto whichever portion is the minority. This
is exactly why the design's planned review UI includes a manual "split"
action — not something this prototype tries to detect on its own.

---

## 8. Known limitations / open items (as of 2026-09-10)

| # | Limitation | Status |
|---|---|---|
| 1 | 2D-only fallback disambiguation for ambiguous groups is disabled — real coverage loss when a marker is only ever seen by one camera at a time | Accepted trade-off, not fixed |
| 2 | A marker confirmed via 2 cameras' triangulation doesn't automatically get labeled in a 3rd camera that also sees it, unless that 3rd camera's candidate wins the reprojection-consensus claim in Stage B | Improved by Stage B's rewrite, not fully eliminated — a camera's own candidate can still lose the claim to a better-fitting one |
| 3 | `hybrid_assign_frame`'s own unambiguous-group 3D pass has the same latent "two FusedPoints could share a raw pixel" exposure Stage B's rewrite fixed structurally for anyone calling `fuse_frame` — but a caller doing its own ad-hoc fusion logic outside `fuse_frame` wouldn't get this protection | Not applicable now that Stage B is fixed centrally |
| 4 | Toe/knee markers often show no raw candidate nearby at all (toe: candidate present in only ~32% of visible-keypoint frames on one sampled camera) | Upstream of assignment — a detection/proximity limitation, not an assignment bug |
| 5 | Overall ambiguous-group coverage is low relative to how often a candidate is genuinely nearby (knee: candidate nearby 93% of the time, actually assigned ~45%) | Direct, understood consequence of items 1 and "drop, don't guess" gating — not a bug, a deliberate precision-over-recall choice |
| 6 | True anterior direction doesn't track hip axial (internal/external) rotation, only pelvis orientation | Accepted simplification (Stage C) |
| 7 | Not yet ported to production (C++ tracker, `MarkerDetectionPipeline`) | By design — this is still the P7/normals validation phase of the phased plan |

---

## 9. File map

| File | Role |
|---|---|
| `posetrak/detection/dot_blob_detector.py` | Stage A: blob detection |
| `posetrak/detection/marker_pipeline.py` | Stage A: pipeline wiring, per-camera config |
| `app/pose/db_cache.py` | dot candidate DB encode/decode, `DotCandidateWriter` |
| `tools/prototype_multi_camera_fusion.py` | Stage B: `fuse_frame`, `FusedPoint`, `triangulate_multiview` |
| `tools/prototype_person_marker_assignment.py` | marker catalog (`_CATALOG`), P1's original per-camera assignment |
| `tools/prototype_hybrid_person_marker_assignment.py` | Stage D baseline: `hybrid_assign_frame` (no direction awareness) |
| `tools/prototype_marker_normal_assignment.py` | Stage C + D: `compute_leg_frames`, `normal_aware_hybrid_assign_frame` |
| `tools/prototype_tracklet_smoothed_assignment.py` | Stage E: `run_tracklet_smoothed` |
| `tools/render_hybrid_assignment_video.py` | validation video renderer, single-camera or multi-camera grid, `--use-normals`/`--tracklet-smoothing`/`--min-direction-score` flags |
| `tools/render_marker_normal_debug_frame.py` | debug: draws Stage C's computed directions as arrows on a real frame |
| `tools/prototype_fused_person_marker_assignment.py` | earlier, superseded 3D-primary-only experiment (kept for reference; the hip-coverage-collapse finding that motivated the hybrid approach) |

---

## 10. Validation status (10s / 6-camera window, gopro-11_mini_01 as primary review camera)

Ambiguous-group per-tracklet name-purity (how consistently a physical dot
gets the same name across its own lifetime), measured on the same
validation clip throughout:

| Configuration | Purity | Ambiguous assignments |
|---|---|---|
| `hybrid_assign_frame` (no direction awareness) | 0.719 | 13,325 |
| + directional preference (2D fallback still buggy) | 0.797 | 10,188 |
| + 2D fallback disambiguation disabled | 0.814 | 7,436 |
| + Stage B reprojection-consensus fusion rewrite | 0.821 | 4,734 |

Coverage has monotonically dropped as each fix removed a source of wrong
labels rather than accepting the best-available guess — consistent
"drop, don't guess" throughout, not a free improvement. Whether this
trade-off point is the right one for the next phase (production port,
or further tuning) is an open question, not yet decided.
