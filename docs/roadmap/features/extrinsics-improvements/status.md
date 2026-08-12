```toml
name = "Extrinsics Calibration Improvements"
status = "in_progress"
progress_pct = 65
description = """
Improvements to multi-camera extrinsic calibration: scrubbing calibration frames directly from \
capture video instead of a pre-extracted PNG folder, per-control-point per-frame observations, \
ArUco/ChArUco marker detection to anchor the coordinate system and provide a rigid marker-pose \
bundle-adjustment residual, persisted fiducial markers for recalibration reuse, and (added after \
Phase 4 live testing) a portable non-planar calibration rig -- characterizable from a single \
orbit video, not just measured or hand-typed -- plus scattered-tag redundancy as a more robust \
alternative to anchoring the world frame from a single flat ChArUco board alone.
"""
categories = ["calibration", "ui"]
target_release = "TBD"
last_updated = 2026-08-12
```

# Extrinsics Calibration Improvements — Implementation Status

See [extrinsics-improvements-design.md](extrinsics-improvements-design.md) for
the problem statement, requirements, and full technical design.

## Current state

Phases 1-4 implemented (2026-08-09), grounded against the pre-existing
`python/app/setup/extrinsics_solver.py` / `page_extrinsics.py` /
`posetrak/db/import_extrinsics.py` implementation and
`docs/extrinsics-calibration-design.md`. Phase 3 landed with one
significant, prominently-flagged scoping deviation from the design doc's
section 5, and Phase 4 with a smaller one from section 4 — see each
phase's notes below before assuming either matches the design doc
literally. Manual, real-footage UI testing has confirmed Phases 1-4 work
in the live app (Phase 4 needed one settings fix along the way — see its
notes). Phases 5-6 remain design-only.

**Design addendum (2026-08-09, §9)**: live testing of Phase 4 surfaced two
structural risks in anchoring the world frame from a single flat ChArUco
board alone — spatial-concentration bias (a small board gives BA a point
cluster with little depth variation, so small angular errors there become
large positional errors elsewhere in the volume) and single-point-of-failure
(one camera's board detection failing, e.g. from reflections, silently
propagates a broken pose through anything chained to it — see the sixth
live-testing round below). Combined with a hard portability requirement
(sessions happen at remote locations, ruling out a large purpose-built
frame), the design doc's §9 now adds a three-tier anchoring strategy: a
portable non-planar calibration rig as the primary anchor (Phase 8), ArUco
tags scattered around the room for mid-session drift/bump recovery (Phase
9), and the existing manual control points, unchanged. ChArUco detection
(Phase 4) is not removed — it remains available as a supplementary accuracy
aid and boardless fallback, just no longer the recommended sole anchor.
Tier A's design also gained a same-day refinement: the rig's geometry can
be printed on the rig itself as a QR code (`cv2.QRCodeDetector`, already
available, no new dependency) and read automatically, using a compact
parametric shape descriptor (starting with `"box"` — dimensions + marker
size + one marker ID per face) rather than raw corner coordinates, so the
user never has to locate/pair a config file with a specific physical rig.

**Test captures acquired (2026-08-11) and continuation plan.** Harri shot
two real captures with the same 3 cameras: **capture 1** has a cardboard
box (one 4x4 ArUco marker per face — the Tier A rig prototype) plus 6
scattered 5x5 ArUco tags around the room (Tier B), and a separate video of
one camera orbiting the box; **capture 2** removes the box, keeps the same
6 scattered tags at the same physical positions, but the cameras have been
moved. Capture 1 also has reflective dots added to the box, and in capture
2 the target person picks the box up and carries it. These two captures
drive a concrete, ordered implementation plan (agreed 2026-08-11):

1. **Rig geometry from the orbit video** — a new design element (design
   doc, Tier A subsection "Rig geometry from an orbit video
   (self-calibration)"): treat sampled video frames as unknown-pose
   "cameras" and reuse the existing marker-rigid-pose BA
   (`solve_marker_groups`, shipped in Phase 3) to solve per-frame poses and
   the box's marker geometry jointly, gauge-fixed on one designated marker.
   No new solver machinery — new caller of already-tested code. Prototype
   as a standalone script against capture 1's orbit video before any UI
   work, and cross-check its output against `solve_marker_groups` run on
   capture 1's synchronized 3-camera footage of the same box (an
   independent, drift-free measurement of the same physical geometry) —
   this is now Phase 8's validation criterion for the video-derived path.
2. **Phase 8 core**, exercised against capture 1: `MarkerRigConfig` +
   `MarkerRigDetector` + `anchor_from_marker_rig`, anchoring the world frame
   from the box across all 3 cameras.
3. **Phase 5 persistence**, exercised against capture 1: once cameras are
   anchored via the box, `solve_marker_groups`'s existing post-pass already
   recovers the 6 scattered tags' world poses — persist them to
   `scene_fiducial_markers`.
4. **Phase 9 re-anchor**, validated against capture 2: load the tag poses
   persisted in step 3, solve capture 2's (moved) cameras from *only* those
   previously-known tags — no board, no rig, no manual points. This is the
   direct real-data test of §9's central motivation (robustness to a single
   anchoring instrument failing or being unavailable) and the milestone that
   closes out the addendum.

The reflective dots and the lifted-box portion of capture 2 are **not**
part of this plan — they're test data for a distinct, not-yet-scoped future
feature (per-frame tracking of a moving rigid body via markers, already
called out as deliberately out of scope in the design doc's §3/§6). Kept in
reserve rather than folded into Phases 8/9, which are calibration
(fixed-pose) problems, not motion-capture ones; will get its own design doc
when picked up.

Phases 8-9 are design-only; not started, per the plan above.

**Orbit-video self-calibration prototype (2026-08-11), step 1 of the plan
above.** `python/tools/characterize_rig_from_video.py` (see its own header
for the full approach) run against both of the box orbit videos from the
2026-08-10 test captures, against the real `registry.db` intrinsics:

- **OnePlus 9 Pro orbit video: clean end-to-end success.** 10 sampled
  frames, all 10 solved; 5 of the box's 6 markers recovered with edge
  lengths within ~4% of the configured 0.15m marker size; camera-to-rig
  distances (1.3-2.0m) plausible for the physical orbit. ~60s runtime for
  45 SIFT pairs — cheaper than the design doc's "expect this to be slow"
  caveat worried.
- **ACE2 Pro orbit video: surfaced a real triangulation-robustness gap, not
  just a bug.** First run: only 8/10 frames solved (weaker SIFT
  connectivity than the OnePlus footage, matching Harri's own note that
  this camera's lighting is worse), and 2 of the 4 recovered markers had
  catastrophically wrong edge lengths (up to 12m, for a 0.15m marker) —
  traced to marker corners triangulated from only 2 solved cameras with no
  sanity check at all before being trusted. Fixed by adding a per-corner
  reprojection-error check to `_triangulate_corner` (same idea
  `extrinsics_solver.triangulate_pair` already applies to SIFT points, just
  previously missing from both of this codebase's free-CP triangulation
  paths) — corners that don't reproject within 10px in every observing
  camera are now dropped rather than trusted.
- **The reprojection filter caught the worst case but didn't fully fix this
  particular run**: re-running after the fix left only 1 of 5 markers with
  all 4 corners surviving, and the *remaining* edges still showed a 126%
  relative spread. Diagnosis: reprojection error alone doesn't catch a
  different, more fundamental problem — a marker triangulated from only 2
  cameras at a narrow parallax angle (nearly-parallel viewing rays, e.g.
  two orbit samples taken close together in time) can reproject
  acceptably well at *many* different depths along the ray, so a
  low-reprojection-error triangulation can still have large real-world
  position error. This is a standard photogrammetry failure mode
  (triangulation uncertainty depends on baseline/parallax angle, not just
  reprojection residual at the found solution), not specific to this
  script, but nothing in either existing free-CP triangulation path in
  `extrinsics_solver.py` currently checks for it either.
- **Not yet resolved — needs a decision before Phase 8 core work starts**:
  whether to (a) increase `--num-samples` for weak-texture/weak-connectivity
  footage so more marker corners get 3+ camera observations at varied
  angles, (b) add an explicit parallax-angle check alongside the
  reprojection-error one, or (c) treat this as acceptable for now and rely
  on the OnePlus-derived rig config as the primary source, using ACE2 Pro
  only as a (currently unreliable) secondary cross-check. No implementation
  decision made yet.

**More samples made it worse, not better (2026-08-11), and marker size
was measured (2026-08-12).** Two follow-ups resolved the open question
above and changed direction:

- Re-ran both orbit videos at 24 samples (up from 10). Result: a clear
  regression on **both** cameras, not an improvement — OnePlus went from
  5/5 clean markers to 1/5 (13/20 corners rejected by the reprojection
  filter) with individual camera CP errors as high as 19.5±10.2px (vs.
  1.7-3.8px at 10 samples); ACE2 Pro went from a partial result to 0/20
  corners surviving at all. Root cause: sampling more frames from the
  *same* clip shrinks the baseline between adjacent samples, and
  `chain_poses_bfs` has no awareness of how well-conditioned a given
  SIFT-matched pair's baseline is — more samples added mostly
  short-baseline, noise-amplifying pose-chain edges, not new information.
  **Lesson for reuse**: for this method, sample count needs to match the
  physical baseline the orbit actually covers, not be maximized — "more
  samples" is not a safe default fix.
- Harri measured the actual printed marker size with calipers: **0.145m**,
  not the 0.15m nominal design size the orbit-video runs assumed. Comparing
  the (axis-corrected) 10-sample OnePlus result against a physical tape
  measurement of the box (49.5 x 31 x 33.5cm) showed a consistent ~5-7%
  *overestimate* on both real (non-guessed) axes — the same-direction bias
  across independent axes is the signature of a single multiplicative
  cause, and 0.15/0.145 = 1.034 explains roughly two-thirds of it on its
  own. The remainder is presumed to be ordinary triangulation noise from
  only 10 samples.
- **Decision (2026-08-12, per Harri):** for production use, a hand-measured
  physical rig with its geometry stored in a config file is the safer
  default — the orbit-video method is better framed as a fallback/
  self-calibration tool for rigs that can't be pre-measured, not the
  primary path. This doesn't waste the orbit-video work: the 10-sample
  OnePlus result, rescaled by 0.145/0.15, *is* the config now in use (see
  below) — its corner order came from real detections, so reusing it
  avoids a real, separate risk a hand-derived config would have carried
  (see `load_rig_config`'s docstring).

**Phase 8 core implemented and validated against real capture-1 footage
(2026-08-12).** `MarkerRigConfig` / `MarkerRigDetector` /
`anchor_from_marker_rig` added to `fiducial_markers.py` (15 new synthetic
tests, `test_marker_rig.py`), generalizing `anchor_from_charuco_board`'s
already-established mechanism (the instrument's own local frame becomes
the world frame directly) to a non-planar rig — no `solvePnP`-based
anchoring needed, same simplification Phase 4 already found. Only the
`"explicit"` rig-config shape is implemented; the `"box"` parametric shape
is deliberately deferred (see `load_rig_config`'s docstring — expanding it
requires *assuming* each physical marker's mounted orientation, a real,
hard-to-detect risk the `"explicit"` form sidesteps when the corner
geometry comes from real detections instead).

`python/tools/test_rig_anchor_capture1.py` (new, standalone, no
session-DB import needed — extrinsics calibration doesn't require sync)
ran the box rig config (`tools/rig_configs/box_2026-08-10.json`, the
rescaled orbit-video result above) against one real frame from each of
capture 1's 3 cameras (ace2pro frame 2069, gopro-11_mini_01 frame 1257,
oneplus9pro-01 frame 386, portrait mode):

- **All 3 cameras PnP-initialised directly from the rig anchor** (12 world
  CPs each — no SIFT chaining needed at all, though SIFT still ran
  alongside as usual). CP reprojection errors: 3.0-4.6px mean (max
  8.6-15.0px) — comparable to the earlier ChArUco-anchored runs.
- **Solved camera positions initially came out with mixed-sign world Z**
  (ace2pro/gopro at Z ≈ -1.6/-1.8, oneplus at Z ≈ +2.1) **with no
  ambiguity-resolution branch involved at all** — this is the concrete,
  real-data confirmation of §9's central motivation (see the sixth
  live-testing round above): a genuinely non-planar anchor has no
  IPPE-style tilt ambiguity to resolve in the first place. This claim is
  about *coplanarity* specifically, confirmed independently in the log
  (`init_poses_pnp`'s own "world CPs are coplanar" warning never fired for
  any of the 3 cameras) — it does not by itself say anything about which
  axis is "up" (see next point, caught live by Harri).
- **"Z" in this config was not gravity-up, and mixed-sign camera Z was
  briefly (and reasonably) mistaken for a red flag.** `characterize_
  rig_from_video.py`'s reference-marker choice (`min(complete_ids)`, i.e.
  "whichever marker id happens to be lowest") had no notion of which
  physical face that marker actually was — for this config it picked
  marker 0, one of the *side* faces, so the rig's Z axis was that face's
  own horizontal outward normal, not gravity-up. Reframed
  (`tools/rig_configs/box_2026-08-10.json`, rebuilt in place) onto marker
  4 — confirmed physically to be the box's top face — as the reference,
  matching this project's existing Z-up convention
  (`extrinsics_solver.similarity_from_floor_plane`'s docstring). The
  candidate frame's Z came out backwards on the first attempt (the other
  4 markers landed at positive Z, i.e. "above" the top face) and was
  corrected by flipping Y and Z together (an even number of axis flips,
  keeping the frame right-handed rather than silently mirroring it) —
  after which all 4 side markers landed consistently 13-14cm *below* the
  top marker, matching the box's real geometry. Re-running the capture-1
  test against the reframed config reproduced byte-identical reprojection
  errors (confirming this was a pure relabelling, not a re-solve) and now
  gives all 3 cameras positive Z (1.04-1.25m above the box top) —
  consistent with tripod-mounted cameras above a box on the floor.
- All 5 rig markers detected across the 3 cameras (each camera saw only
  3 of 5 — box faces aren't all visible from any one viewpoint, handled
  gracefully as designed); 5 scattered ArUco tags also recovered (2
  cameras each) as free-CP world centroids — a Phase 5/Tier B precursor,
  printed only, no `scene_fiducial_markers` persistence yet.

**Phase 9 re-anchor validated against real capture-2 footage (2026-08-12) —
mechanism confirmed, one camera's data quality flagged as suspect.**
`tools/test_rig_anchor_capture1.py --save-scattered-tags` now
DLT-triangulates each scattered tag's 4 corners individually (not just a
centroid), reprojection-checked the same way
`characterize_rig_from_video.py`'s `_triangulate_corner` already is, and
writes survivors out in the same rig-config JSON shape the physical box
rig uses (4/5 tags survived; tag 0 was triangulated from only 2 cameras
and none of its corners passed the check — correctly excluded rather than
trusted). New `python/tools/test_reanchor_capture2.py` loads that file and
calls `anchor_from_marker_rig` **completely unmodified** — the scattered
tags' already-solved geometry is just another rig config as far as that
function is concerned, so Tier B needed no new anchoring code at all, only
a new source of `MarkerRigConfig` input. This directly exercises the
design doc's Tier B goal (design doc §9): recover a shared world frame in
a *different* capture, with the physical rig absent and cameras moved,
using only ordinary tags anchored in a prior capture.

- **All 3 of capture 2's (moved) cameras solved**, all at positive world
  Z (0.87-1.58m) — the gravity-up convention established in capture 1
  propagated through correctly, and again no planar-pose-ambiguity branch
  was involved.
- **Accuracy was uneven across cameras, and the pattern points at one
  camera's data, not the mechanism.** ace2pro: 3.5±1.3px (clean, matches
  capture 1's quality). gopro: 9.2±6.6px (moderate — only 3 of 4 known
  tags visible, fewer/weaker-parallax anchor points). oneplus: 36.6±11.6px
  (max 64.2px) — and critically, the per-CP debug breakdown shows this
  elevated error on **every single tag/corner oneplus observes**, not
  concentrated on one bad tag. A uniform, all-correspondences-affected
  error for one camera is the signature of that camera's own calibration
  being wrong for this footage, not bad anchor geometry — and there's
  already a named suspect: Harri flagged at the very start of this test
  round that the OnePlus 9 Pro "has a habit to do sudden autofocusing
  which may change its intrinsics." The identical stored calibration gave
  a clean 3.0±2.8px for this same camera in capture 1, which is consistent
  with the lens having refocused (different scene, different session)
  between the two captures rather than the calibration or the anchor being
  wrong in general.
- **The autofocus/fx-fy-scale hypothesis is disproven, not confirmed.**
  Re-ran with `refine_intrinsics={"oneplus9pro-01"}`
  (`test_reanchor_capture2.py --refine-intrinsics`, new flag). Result:
  error got *worse*, not better — 239.2±62.6px (max 330.1px), up from
  36.6px — with fx/fy pushed -17.5%/-21.4% before `run_bundle_adjustment`'s
  bounds stopped it. A genuine pure focal-length shift (simple optical
  zoom from refocusing) should be *correctable* by floating fx/fy; making
  it worse means the real mismatch isn't a simple fx/fy scale error.
  Pulled and visually inspected the annotated frames for both captures
  (oneplus capture-1 frame 386 and capture-2 frame 219) — both show tight,
  correctly-localized marker corner boxes with no motion blur or
  misdetection, ruling out a gross per-frame problem. `init_poses_pnp`'s
  own initial RANSAC pass showed 12/16 inliers (75%) for oneplus in
  capture 2 — already somewhat elevated before BA ever ran, similar to
  gopro's 9/12 (75%), yet oneplus's final BA error is ~4x worse than
  gopro's — so whatever is wrong is present in the raw correspondences,
  not introduced by BA, and affects oneplus disproportionately more than
  a shared cause (e.g. generally weaker capture-2 anchor data) would
  predict on its own.
- **Root cause still open.** Remaining candidates, none yet tested:
  (a) a focus-distance-dependent shift in distortion coefficients or
  principal point (not just focal length) — a real, known effect in
  consumer phone lenses, and not something `refine_intrinsics` currently
  touches (it only floats fx/fy); (b) a portrait-mode rotation/calibration
  axis-convention mismatch whose visible impact depends on viewing
  geometry (would explain differing severity between capture 1 and 2 for
  the *same* camera/calibration without needing anything to differ between
  runs); (c) the underlying capture-1-solved tag positions themselves
  being subtly less accurate for whichever tags oneplus specifically
  relies on, with oneplus's capture-2 viewing angle happening to be more
  sensitive to that error than gopro's or ace2pro's. Not pursued further
  this round -- the actual Phase 9 mechanism validation (previous entry)
  doesn't depend on resolving this, and further diagnosis is better scoped
  as its own follow-up.

**Not yet done**: `scene_fiducial_markers` persistence (see below — now
superseded by `scene_marker_bodies`); any UI wiring; the oneplus
data-quality root cause above.

**Marker body definition format designed (2026-08-12), before continuing
to UI/persistence work.** Prompted by Harri's status check ("where are we
with implementing this feature") confirming three concrete UI gaps
(ArUco markers can't be locked to a known world position; calibration rigs
aren't supported in the UI; cross-capture marker reuse isn't available in
the UI) — all three reduce to the same missing piece: a proper format and
storage story for "a named rigid body carrying known markers," which the
prototype scripts had so far handled ad hoc (raw JSON, hand-carried
files). Worked out with Harri over several rounds; full write-up in
extrinsics-improvements-design.md §10 ("Marker body definitions: format
and storage"), which now supersedes §6's original `scene_fiducial_markers`
sketch. Highlights:

- **YAML, not JSON** — mirrors `docs/skeleton-format.md`'s existing
  precedent (a named, versioned, human-authored definition stored as a
  `yaml_content` text blob in a registry-DB row, imported from a `.yaml`
  file) rather than inventing a new pattern, and keeps a plausible future
  door open to a skeleton joint referencing the same marker-list shape
  (Harri's stated long-term hope).
- **The canonical format is always fully-resolved** — a flat per-marker
  list (`center`/`normal`/`up`, or raw `corners`), never a parametric
  shape the loader itself interprets. This is a direct revision of the
  earlier `"box"`-JSON idea (§9's QR-code subsection): expanding a shape
  requires *assuming* physical marker mounting orientation, which is
  exactly the risk `load_rig_config` already deferred building for (see
  its docstring). A `"box"` generator can still exist, just as an offline
  tool producing this YAML, never as loader-level parsing.
- **`name` vs `id` separated** — caught live as a real conflict, not a
  style preference: a body with markers from two different dictionaries
  (this project's actual box, which already has both ArUco corners and
  reflective dots) can have two markers that legitimately share a numeric
  `id`. `name` is the stable, human-facing label; `id` (with `type`+
  `dictionary`) is the only thing matched against a real detection.
- **`type` added, separate from `dictionary`** — mirrors the
  `FiducialDetection.marker_type` string already in `fiducial_markers.py`
  (`"aruco"`, `"charuco"`, `"apriltag"`, ...), extended with
  `"reflective_dot"`. Required fields are type-discriminated, same pattern
  skeleton format already uses for joint-type-specific fields.
- **Templating reserved, not built** — Harri flagged a real but
  not-yet-needed use case (several physical copies of the same rig shape,
  distinguished only by which marker ids are printed). `slots:` plus a
  structured `id: {slot: "name"}` reference are reserved in the format so
  a future `marker_body_instances` table is additive later, without
  building it now — same "defer until a concrete second case exists" call
  this design doc already made once for `scene_fiducial_markers`'s own
  registry/session scoping.
- **`scene_marker_bodies` replaces `scene_fiducial_markers`** — one row
  per solved *body instance* (rig or lone tag), not per marker, since a
  rigid multi-marker body only ever needs one pose. A lone scattered tag
  needs no YAML definition at all (`marker_body_definition_id` NULL,
  inline dictionary/id/size columns) — building a bespoke one-marker
  definition file per tag id would be pure overhead, agreed with Harri.

**Not yet done**: any of this is implemented — `import_marker_body()`,
the `marker_body_definitions`/`scene_marker_bodies` migrations, and
updating the existing prototype scripts (`characterize_rig_from_video.py`,
`test_rig_anchor_capture1.py`, `test_reanchor_capture2.py`) to read/write
§10's YAML instead of their current ad hoc JSON are all still ahead.

**`posetrak extrinsics anchor-rig`/`reanchor` implemented and validated
against real data (2026-08-12) — prototype scripts fully retired.**
Replaces `tools/test_rig_anchor_capture1.py`/`test_reanchor_capture2.py`
outright (both removed), not "updated in place." Prep work first:
extracted `page_extrinsics.py`'s private `_write_extrinsics_to_db` into
`extrinsics_solver.write_extrinsics_to_db` (Qt-free, shared by GUI and
CLI now — one write path, not two) and switched frame reading to the
already-shared, already-tested `posetrak.detection.frame_source.
iter_frames` instead of a fifth duplicated rotation-aware reader.

- **`anchor-rig`**: detects a named marker body (rig) across given
  cameras/frames, anchors the world frame, solves, writes straight to
  `extrinsic_calibrations`/`extrinsic_entries`, and upserts the rig's own
  anchor pose (+ any `--tag-size`'d scattered tags) into
  `scene_marker_bodies`.
- **`reanchor`**: re-anchors from previously-persisted `scene_marker_bodies`
  tags, no physical rig needed — `anchor_from_marker_rig` reused completely
  unmodified (Tier B).
- **Verified end-to-end against the real registry and 2026-08-10 capture
  footage**: converted `box_2026-08-10.json` to real section-10 YAML
  (`tools/rig_configs/box_2026-08-10.yaml`) for this. `anchor-rig`
  reproduced byte-identical camera positions to the original prototype's
  already-validated run and correctly persisted both DB writes.
  `reanchor` was also run against capture 2 (placeholder `--tag-size`, so
  its solved numbers aren't meaningful this time, but the full mechanism —
  DB read, rig-config construction, detection, anchoring, solve, write —
  executed correctly end to end).
- 10 new unit tests for the parsing/resolution helpers that don't need
  real video; the I/O-heavy command bodies validated against real data
  instead, consistent with this feature's practice throughout.

**Not yet done**: GUI wiring (Phase 8/9's panel in `ExtrinsicsAutoCalibDialog`)
is the one remaining piece from the original three gaps Harri's status
check identified — ArUco-locked-to-world-position, rigs, and cross-capture
reuse are all now real in the CLI; only the GUI front-end is still
outstanding.

**`posetrak marker-body` CLI group implemented (2026-08-12)** —
`import`/`list`/`show`/`export`, mirroring `posetrak skeleton`'s
structure directly. This is the first real, permanent entry point for
marker body definitions — replacing the throwaway `test_*.py` prototype
scripts' ad hoc JSON handling with actual production CLI, per Harri's
question about the path to GUI/CLI. Manually smoke-tested end-to-end
against a real registry via the installed `posetrak` console script; 11
new tests. Next: promote the rig-anchor/re-anchor logic the prototype
scripts already validated into real `posetrak extrinsics` commands using
this DB layer instead of JSON files — replacing the scripts outright, not
updating them in place.

**YAML loader implemented (2026-08-12)** — `load_marker_body_yaml()`/
`load_marker_body_yaml_file()` in `fiducial_markers.py` parse section 10's
canonical format into `MarkerRigConfig`. `center`/`normal`/`up` resolves
via the same Gram-Schmidt construction the box config's gravity-up
reframing already used; raw `corners` is used as-is. `reflective_dot`
entries land in a new `MarkerRigConfig.reflective_dots` field, parsed but
not matched against anything yet (correspondence/tracking for dots stays
future work). `MarkerRigConfig` also gained `marker_dictionaries` (empty
= legacy single-dictionary behaviour, unchanged for `load_rig_config`-
loaded configs); `MarkerRigDetector` now builds one `ArucoDetector` per
distinct dictionary a body actually uses. This works with a plain bare-id
lookup (no composite key needed) because the loader rejects a body where
two coded markers share an id in *any* dictionary — the one case a
bare-id lookup can't disambiguate — loudly, at load time. 38 new/updated
tests. Nothing calls this loader from the prototype scripts yet (they
still read `load_rig_config`'s JSON) or the DB CRUD layer above — those
are the next two pieces to connect.

**CRUD layer implemented (2026-08-12)** — `python/posetrak/db/
manage_marker_body.py`: `import_marker_body()`/`import_marker_body_str()`
(mirrors `manage_skeleton.import_skeleton()` exactly — content-addressed
id, idempotent), `copy_marker_body_to_session()` (reuses
`_copy_rows_if_missing()`), `upsert_scene_marker_body()` (upserts by
`(session_id, label)` — current believed pose, not history) and
`read_scene_marker_body_pose()`. 20 new tests. Nothing calls any of this
yet — no prototype-script or UI wiring.

**DB migration implemented (2026-08-12)** — `marker_body_definitions`
(registry v7→v8, embedded into every session DB the same way
camera_models/skeletons/tracker_configs already are) and
`scene_marker_bodies` (session v39→v40), per §10's schema. Both tables
exist now, empty and unused: `import_marker_body()` (the
`import_skeleton()`-equivalent CRUD helper) and the write/read paths that
would actually populate `scene_marker_bodies` from a solve are still
ahead, along with updating the prototype scripts to target the DB instead
of ad hoc JSON files. 4 new tests
(`python/tests/db/test_posetrak_db.py`) cover fresh-create, the
pre-existing-DB migration path, and `scene_marker_bodies`' two real usage
shapes (rig-anchor row with a real definition reference; lone-tag row
with the inline `marker_type`/`dictionary`/`marker_id`/`marker_size`
columns instead) plus its `(session_id, label)` uniqueness constraint.

## Phase summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Video frame source: per-camera random-seek reads, scrub UI replacing PNG-directory loading | ✅ Done, live-tested |
| 2 | Per-control-point, per-frame observations (`ObsPoint`, file format v2) | ✅ Done, live-tested |
| 3 | ArUco marker detection + rigid marker-pose BA residual | ✅ Done, live-tested (detection confirmed working) — see "Phase 3 notes" for a scoping deviation (decoupled post-pass, not a joint BA parameter block) |
| 4 | ChArUco board detection + coordinate-system anchoring | ✅ Done, live-tested (detection confirmed working after a settings fix, see "Phase 4 notes") — also see there for a scoping deviation (no solvePnP / reference camera needed) |
| 5 | `scene_fiducial_markers` persistence + recalibration reuse | ⬜ Not started |
| 6 | AprilTag detector backend (extensibility proof) | ⬜ Not started |
| 7 | Global timeline scrub (§8) — jump every camera to the same synced instant | ⬜ Not started (design added 2026-08-09 from UI-testing feedback) |
| 8 | Portable non-planar calibration rig — primary world-frame anchor (§9, Tier A) | 🟡 Core implemented + validated against real capture-1 footage (2026-08-12); no UI wiring yet, `"box"` shape deferred |
| 9 | Scattered-tag redundancy + single-camera re-anchor (§9, Tier B) | 🟡 Re-anchor mechanism validated against real capture-2 footage (2026-08-12), reusing Phase 8's anchor_from_marker_rig unmodified; no UI wiring, one camera's data quality still under investigation |

## UI testing feedback (2026-08-09, Phases 1-2)

Harri ran the Phase 1/2 UI checklist. One bug found and fixed, one
improvement proposal captured as Phase 7 above (§8 in the design doc).

- **Bug, fixed** (`setup: fix "Go to..." frame-seek button using PyQt-style
  keyword names`): `VideoScrubBar._on_goto()`'s `QInputDialog.getInt()` call
  used `min=`/`max=` (the PyQt5/6 keyword spelling); PySide6 names the same
  parameters `minValue=`/`maxValue=` and raised `AttributeError: unsupported
  keyword 'min'` on every click. This was carried over verbatim from
  `pair_scrubber._VideoPane`'s original `_on_goto()` (confirmed via `git
  log` — predates the `VideoScrubBar` extraction), never caught before
  because no prior test exercised the real `QInputDialog.getInt()` call.
  Fixed; new tests call the real API (via a `QTimer.singleShot` to close
  the resulting modal dialog) to catch this class of bug going forward, not
  just a stub.
- **Improvement proposal, captured as design (Phase 7 / §8), not yet
  implemented**: since a capture has almost always already been through the
  sync wizard page by the time extrinsics calibration runs, a single global
  timeline scrub — driving every camera's `VideoScrubBar` to its own
  locally-synced frame for the same instant, via the existing `SyncTable` —
  would replace "hunt for a good calibration moment across N independent
  sliders" with one drag, while leaving each camera's own slider available
  afterward for per-point fine adjustment exactly as today.
- Everything else on the Phase 1/2 checklist: OK.

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

**Scope note (resolved in Phase 2, see below)**: every control point placed
in a session used to use whatever frame each camera happened to be scrubbed
to *at solve time* — there was no independent per-control-point frame
record. Placing point A while camera 1 was on frame 100, then scrubbing
camera 1 to frame 250 before placing point B, would silently move point A's
effective frame too, since `ControlPoint` had no frame field of its own.
Flagged at the time so it wasn't mistaken for Phase 1 already covering R3/R4
— Phase 2's `ObsPoint` is exactly the fix.

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
`VideoScrubBar` extraction. Still present and still unrelated after Phase
2 (re-confirmed the same three, and only those three, fail when running
Phase 2's test files alongside `test_pair_scrubber.py`). Worth a follow-up,
not addressed here.

## Phase 2 notes

Implemented as one commit, `setup: track per-control-point, per-camera
frame index`:

- `ObsPoint(frame_idx, px, py)` replaces the plain `(px, py)` tuple in
  `ControlPoint.obs`. `frame_idx` is provenance for the UI and the saved
  file only — every solver-facing consumer (`init_poses_pnp`,
  `_undistort_control_obs`, `compute_cp_errors`, the BA's observation list)
  reads only `.px`/`.py`, proven by a synthetic-camera PnP test that solves
  to a bit-identical pose regardless of what `frame_idx` values the
  observations carry.
- `ExtrinsicsAutoCalibDialog._on_cam_click` now records the clicked
  camera's *current* `VideoScrubBar` position into the new observation;
  re-clicking the same point on the same camera at a different scrub
  position overwrites its `ObsPoint` — R4's "different/later frame for the
  same point in the same camera" case.
- `save_control_points`/`load_control_points` bumped to file version 2
  (`obs` values become `[frame_idx, px, py]`); version 1 files still load,
  with `frame_idx` defaulting to a caller-supplied `default_frame_by_id`
  (wired to each camera's current scrub position when loading from the
  dialog) or 0 if none is given.

**Known limitation, not addressed here**: `_refresh_markers` still draws
every placed control point's marker on whatever frame a camera is
*currently* displaying, regardless of which frame that observation was
actually recorded on (`obs.frame_idx`). This was already true before Phase
2 in a lesser form (there was no `frame_idx` to compare against at all);
Phase 2 makes the mismatch meaningful without resolving it — a marker whose
own point moved between frame 100 and frame 250 will still be drawn at its
frame-100 pixel position even while the camera is scrubbed to frame 250,
which can look wrong for exactly the occlusion/motion case R4 was written
for. Not part of Phase 2's stated scope (data model + file format +
placement), but worth a follow-up: e.g. only drawing a point's marker when
the camera's displayed frame matches `obs.frame_idx`, or dimming/badging it
otherwise.

## Phase 3 notes

### Synthetic-data BA prototype (preparatory step, landed first)

Implemented as one commit, `setup: synthetic-data prototype for the rigid
marker-pose BA residual`, answering the "Open questions" item below before
starting Phase 3's actual ArUco-detection/UI work:

- `marker_local_corners(size)`, `project_marker_corners(...)`, and
  `solve_marker_pose(corner_obs, states_by_id, size)` added to
  `extrinsics_solver.py` as a self-contained prototype — a marker's four
  corners are treated as one rigid 6-DOF pose parameter (the same shape as
  a camera pose's own rvec/tvec), recovered via `least_squares` over every
  camera's corner residuals jointly, seeded by a single-camera `solvePnP`.
- Scope of the prototype: camera poses are fixed (as if already solved by
  the rest of `run_calibration`) — only the marker's own pose is recovered.
- Validated against a synthetic 3-camera rig
  (`test_marker_pose_prototype.py`, 12 cases, all passing): exact recovery
  (four marker orientations, atol 1e-5) from noise-free projections; 2
  cameras are sufficient; a single camera correctly raises; a camera with
  no solved pose is correctly excluded from the ≥2-camera count rather
  than crashing; convergence holds under 0.5px Gaussian pixel noise; and
  the result doesn't depend on a good initial guess.
- **Result: the residual math checks out.** This resolved the "worth a
  small synthetic-data prototype before committing" open question before
  any UI work began, per the design doc's own recommendation.

### Full implementation (two commits)

`setup: ArUco marker detection + solver integration (Phase 3, data/solver
layer)` and `setup: wire ArUco marker detection into the extrinsics
calibration UI (Phase 3, UI layer)`:

- New `app/setup/fiducial_markers.py`: `MarkerCornerObs`, `FiducialDetection`,
  `FiducialDetector` protocol, and `ArucoDetector` (wraps
  `cv2.aruco.ArucoDetector`; configurable dictionary, optional
  `default_size`/`size_by_id`). Corner order (top-left, top-right,
  bottom-right, bottom-left) was verified against real `cv2.aruco` output
  via `cv2.aruco.generateImageMarker`-rendered test images, not just
  assumed to match the prototype's `marker_local_corners()` convention —
  it does.
- `extrinsics_solver.MarkerGroup`: one physical marker's corner
  observations across cameras; `as_control_points()` represents its 4
  corners as independent free `ControlPoint`s.
- `ExtrinsicsAutoCalibDialog`: a "Detect ArUco" button per camera pane
  (video- or image-sourced), an "ArUco Markers" panel (dictionary combo,
  default-size spin box, per-marker size-override table, "Clear markers"),
  marker corners drawn as a gold overlay alongside manual control points,
  and `marker_groups` threaded through to `run_calibration` on solve.

### Scoping deviation from the design doc — read this before assuming §5's joint-BA design shipped as originally sketched

**Every detected marker's corners — known size or not — contribute to
camera-pose solving as four independent free control points**
(`MarkerGroup.as_control_points()`), reusing the existing,
already-tested free-CP/DLT-triangulation machinery. This alone delivers
real R6 value: automatic, correctly-labeled correspondences, no manual
clicking, usable to supplement or substitute for a weak SIFT bootstrap.

**What did *not* ship**: the design doc's section 5 sketched a marker's
corners as *replacing* four free points with a single rigid 6-DOF
parameter *inside* `run_bundle_adjustment`'s own joint optimization, so a
known-size marker could also help *correct* camera poses via its metric
constraint. That would mean injecting a new parameter type into
`run_bundle_adjustment`'s existing dense parameter-vector/Jacobian
machinery — real, risky surgery on tested, complex numerical code.
Instead, known-size markers get a **decoupled post-pass**
(`solve_marker_groups()`, called once after camera poses are already
solved): their corners still go through the free-CP path like every other
marker during the main solve, and *afterward*, using those now-fixed
camera poses, the validated `solve_marker_pose()` prototype recovers a
clean, correctly-scaled rigid pose — stored in the new
`CalibResult.marker_poses`, ready for Phase 5's persistence.

**Practical consequence**: a known-size marker's metric information does
not currently help *improve* camera pose accuracy beyond what its 4 free
corner points already contribute (same as an unknown-size marker); it only
produces a clean rigid pose *after* the fact. If real-world testing shows
camera poses need that extra constraint (e.g. too few SIFT features and
only a couple of known-size markers to anchor scale), true joint
optimization is the flagged follow-up — not attempted here, deliberately,
to keep this phase's risk bounded to already-validated pieces (the
free-CP path and the synthetic-data-tested `solve_marker_pose`).

### Test coverage

`test_fiducial_markers.py` (15 cases, real `cv2.aruco` round-trips, no
mocks), `test_marker_groups.py` (10 cases, including a full
`run_calibration` integration test using locked/pre-posed synthetic
cameras + `cp_only=True` so it needs no real footage or SIFT), and
`test_extrinsics_aruco_ui.py` (14 cases, dialog-level: detect button,
table population, scrub-frame capture, size override persistence across
re-detection, redetect-overwrite, multi-camera accumulation, marker/CP
overlay coexistence, and a spy-based confirmation that Match & Solve
actually forwards marker groups to `run_calibration`).

### Not addressed in Phase 3 (unchanged from the open questions below)

- The SIFT pairwise bootstrap is untouched — ArUco corners are *added*
  alongside whatever SIFT correspondences exist, not used to replace SIFT
  outright. Still a live open question (see below).
- Per-marker size override UX ships as an always-visible table column
  rather than the design doc's "shown only once more than one distinct
  size has been entered" progressive-disclosure idea — functionally
  equivalent, simpler to implement, not revisited.
- ~~No manual UI validation yet against a real printed marker / real
  multi-camera footage~~ — confirmed working live (2026-08-09, see "UI
  testing feedback" below): ArUco detection tested against real footage
  in the running app. Full accuracy validation (solved corner spacing vs.
  known size, camera-pose comparison against a manual-CP baseline) not
  yet done.

## Phase 4 notes

Implemented as two commits (`setup: ChArUco board detection +
coordinate-system anchoring (Phase 4, detector layer)` and `setup: wire
ChArUco board detection + anchoring into the extrinsics UI (Phase 4, UI
layer)`):

- New `CharucoDetector` (`fiducial_markers.py`) wraps
  `cv2.aruco.CharucoBoard`/`CharucoDetector` (dictionary, squares X/Y,
  square/marker length all configurable). Every detected corner has an
  exact, pre-known board-local `(x, y, 0)` — no size ambiguity, unlike a
  plain ArUco marker.
- `ExtrinsicsAutoCalibDialog`: a "Detect ChArUco" button per camera pane
  (alongside Phase 3's "Detect ArUco"), a "ChArUco Board" panel (board
  geometry settings, a face-up/face-down checkbox, status line, "Set
  origin & axes from board", "Clear board detections"), and board corners
  drawn as a cyan overlay.
- Before anchoring, detected board corners behave exactly like unknown-size
  ArUco markers: free `ControlPoint`s contributing correspondences to
  camera-pose solving. Clicking "Set origin & axes from board" is the
  Phase-4-specific action: it fixes every detected corner's `world_xyz`
  to the board's own (metric, known) local coordinates, promoting them
  from free to fixed control points in place.

### Scoping deviation from the design doc — smaller than Phase 3's, but read this too

The design doc's section 4 describes anchoring as: pick one camera+frame,
`solvePnP` the board's pose *in that camera's frame*, then map every
corner through *that* pose to get world coordinates. That doesn't
actually work as literally stated — a camera's own world pose is exactly
what calibration is trying to solve, so a solvePnP result expressed in an
*unsolved* camera's own frame isn't "world" coordinates at all, it's
still camera-relative, and treating it as world coordinates would silently
bake that one camera's arbitrary-at-that-point frame in as ground truth.

The implemented mechanism (`anchor_from_charuco_board`) sidesteps this
rather than trying to fix it in place: **the board's own local coordinate
frame is used directly as the world frame** (it is already metric, via
`square_length`), with the only user-facing choice being whether the
board's own +Z is world +Z ("face up") or flipped ("face down", negating
Y and Z together to stay right-handed). This achieves the design's stated
goal exactly — scale, origin, and axes fixed together in one action — via
a mechanism that never needs a camera's intrinsics or an unsolved
camera's pose at all, and is simpler than the originally-sketched
mechanism, not just different from it. `CharucoDetector.estimate_board_pose`
(`solvePnP`, per-camera) is still implemented per the design's section-3
sketch, but only as a diagnostic building block — it is not on the
anchoring critical path. Unlike Phase 3's deviation, this one isn't a
capability trade-off (nothing is deferred or weaker) — it's a corrected
mechanism for the same result.

### Test coverage

`test_charuco_detector.py` (16 cases, real
`cv2.aruco.CharucoBoard.generateImage`-rendered board images, no mocks):
detection correctness (corner count, metric spacing, Z=0 plane), graceful
`None` on a blank image or mismatched board geometry,
`estimate_board_pose`'s rotation-matrix sanity, `anchor_from_charuco_board`'s
fixed-CP construction (single/multi-camera merging, partial corner
overlap, face-up/face-down axis flip), and a full `run_calibration` PnP
integration test that solves an entirely unposed synthetic camera from
scratch using only anchored board corners (R within 1e-4, t within
1e-3 of the known truth). `test_extrinsics_charuco_ui.py` (15 cases,
dialog-level) covers the same detect/anchor/clear flow through the actual
UI methods, plus overlay drawing and a spy-based confirmation that Match
& Solve forwards the anchored corners.

### Live-test finding, fixed (2026-08-09): board never detected

First live pass found the board undetected from every camera. Diagnosed
against a cropped photo of the actual printed board (not a guess) —
neither camera footage nor board size (small board, plausible initial
suspicion) was the cause. Two silent settings gotchas, both fixed in
`setup: fix ChArUco detection failing silently on real (calib.io) boards`:

- **`squares_x`/`squares_y` axis mismatch**: OpenCV's own axis convention
  didn't match how the board generator (calib.io) labels rows vs. columns
  on the page — this board is `(11, 8)` in OpenCV's terms, not the `8x11`
  the generator's own page implied.
- **Missing legacy-pattern support**: `CharucoDetector` had no way to
  request OpenCV's pre-4.7 ChArUco marker-placement convention, which
  calib.io's generator (and other older tools) still uses. Added a
  `legacy_pattern` parameter plus a dialog checkbox
  ("Legacy pattern (calib.io / older boards)").

Both are silent failures — ArUco marker detection succeeds regardless
(26/26 markers found in the diagnostic image every time), only chessboard-
corner interpolation depends on getting both settings right, so nothing
about the failure *looked* like a settings problem. Now documented
directly on `CharucoDetector`'s docstring and the "no board detected"
status message, and locked in with `python/tests/data/
charuco_board_sample.png` (a real cropped photo of the board) plus
regression tests proving both the correct settings work and either
gotcha alone reproduces the exact failure found live.

### Second live-test finding, fixed (2026-08-09): still nothing on the full camera frame

With the settings above confirmed correct (verified by cropping the board
directly out of the real capture video), detection *still* found nothing
against the full camera frame the real workflow actually hands it. Root
cause, again found by testing against real data rather than guessing: a
third, independent gotcha — `cv2.aruco`'s `minMarkerPerimeterRate`
defaults to 3% of the frame's larger dimension, which rejects a
calibration board's markers as "too small" once the camera frame is a
full 3840px-tall 4K capture and the board only occupies a modest region
of it (confirmed directly: plain marker detection went from 3/28 to 28/28
markers found on the exact same real frame, changing only this one
setting). Fixed in `setup: fix (Ch)ArUco detection failing on markers
small relative to a full 4K frame`:

- Both `ArucoDetector` and `CharucoDetector` gained a
  `min_marker_perimeter_rate` parameter (default `None` = OpenCV's own
  3%, so existing callers/tests are unaffected).
- The dialog gained a "Min marker size (%)" spin box in both the ArUco
  and ChArUco panels, **defaulting to 1%** rather than OpenCV's 3% — this
  project's actual use case (room-scale multi-camera rigs, markers seen
  from across a room) hits this default mismatch far more often than a
  typical close-up desk-calibration scenario cv2's own default assumes.

New fixture `python/tests/data/charuco_board_small_in_4k_frame.png` (a
real 3840px-tall crop of the actual capture frame, cropped in width only
since the bug depends on the frame's *height* staying large) locks in the
exact failure for both detectors, plus dialog-level tests through the
actual spin box widgets.

**Two real, independent gotchas found via two rounds of live testing on
real data, not one** — worth remembering if detection ever silently fails
again on different footage: check settings (dictionary/axes/legacy
pattern) *and* marker-size-vs-frame-resolution as separate possible
causes, since either alone produces the identical symptom (zero
detections, no error).

### Third live-testing round: still failing with both fixes applied — added diagnostic logging instead of guessing further

With both settings fixes above confirmed correct and applied, detection
*still* failed on real footage — but this time `ArucoDetector` found
*some* of the board's own markers while `CharucoDetector` still produced
zero corners, a symptom shape the first two fixes can't distinguish from
"still a wrong setting, just a different one." Debugging further by proxy
(screenshots, back-and-forth round trips) doesn't scale, so — requested
directly and implemented in `setup: add diagnostic logging to (Ch)ArUco
detection` — `CharucoDetector.detect()`/`ArucoDetector.detect()` now log,
on every call, the exact configuration used and how many of the board's
own expected marker ids were actually found (vs. how many exist); when
about to return `None`, a `WARNING` additionally lists any found ids that
do **not** belong to this board at all (a direct, mechanical check for
"a stray marker from elsewhere in the scene, sharing this dictionary, is
confusing detection" — plausible here, since the live-test room has
several other ArUco markers visible in frame using this same
`DICT_4X4_50`) plus the concrete next things to check. Both log at
INFO/WARNING unconditionally (not behind a verbose flag) since detection
runs a handful of times per session, not per video frame. Already
verified against the real problem frame — `app.setup` (and its children)
are already bumped to `DEBUG` in `main.py`, so this needed no logging
configuration changes to actually surface in the running app's console
and `logs/posetrak-setup.log`.

**Root cause of this third round, identified from the very next log
output**: the new logging paid off immediately. The real log showed
39/44 expected marker ids found — plenty — but with `min_marker_perimeter_rate`
turned down to the UI spin box's own floor (0.1%) chasing "more markers
found," and several ids appearing *more than once* in the found list (1,
3, 23, and 37 each decoded twice, 37 three times). Reproduced the
identical signature on our own fixture: 0.01-0.015 detects cleanly, while
0.008 and below all produce *more* raw markers than the working rate,
duplicate ids among them, and zero corners regardless. Fixed in
`setup: detect and warn about duplicate marker ids from too-low
min_marker_perimeter_rate`:

- **The real lesson**: `min_marker_perimeter_rate` has a **narrow working
  band**, not a "lower is always safer than OpenCV's too-high default"
  direction. Too high misses real markers (rounds 2's problem); too far
  below the board's own sweet spot starts accepting false-positive/
  misdecoded quads, and when the same id gets decoded from more than one
  candidate location, `detectBoard()`'s corner interpolation can't
  resolve the ambiguity — silently producing zero corners despite *more*
  raw markers being found than at the working rate.
- Both detectors now compute and log duplicate ids on every call; the
  ChArUco failure `WARNING` explicitly calls this out as a signal to
  **raise** the rate back up, replacing the previous (actively misleading
  in this exact case) "try lowering it further" suggestion. Both spin box
  tooltips and the "no board detected" status message carry the same
  warning now.
- Locked in as a regression test against our own fixture (not a
  hypothetical): 0.01 works, 0.005 finds more markers yet still produces
  zero corners with the duplicate-id warning present in the log.

**Not yet confirmed**: whether this was the *complete* explanation for
the live failure, or whether the user's next attempt (now with duplicate-id
guidance in hand) still needs a value search to find this specific board's
working band on this specific camera/lighting.

**Confirmed (2026-08-09, next round)**: the duplicate-id fix above was it —
raising `min_marker_perimeter_rate` back up from the UI's earlier 1.0%
default (which the user had already tried unsuccessfully before this fix
existed) got a clean detection. Note the earlier 1.0% *default* itself
wasn't the problem; what was fixed is that the guidance and floor now steer
users away from the false "lower is safer" attractor in the first place.

### Fourth live-testing round: board markers double-counted as plain ArUco, "unsolved cameras," sidebar too crowded

Three separate findings from the first live session where detection actually
worked end-to-end:

**1. ChArUco board's own markers also showing up in the plain ArUco marker
list.** Both detectors read the same dictionary, so a board's ~44 sub-markers
are indistinguishable from "real" scene markers to `ArucoDetector` unless
something excludes them. Considered recommending a different dictionary per
role (still the simplest, most robust fix if a project's ChArUco board and
scene markers can be planned together) but that doesn't help a project that
already committed to one dictionary for both, so added an automatic filter as
well: `CharucoDetector.expected_marker_ids()` returns the board's own ids, and
`_on_detect_aruco_clicked()` in `page_extrinsics.py` excludes them from the
plain-ArUco results whenever a board has actually been detected (not just
"same dictionary happens to be selected in both panels" — gated on
`self._charuco_detections` being non-empty, since two fresh panels both
default to `DICT_4X4_50` and would otherwise wrongly filter a project that
has never touched the ChArUco panel at all; this over-eager first version was
caught by the existing test suite failing 12 tests before ever reaching the
user). Status text now reports how many markers were excluded and why.
**Recommendation to the user**: use a different dictionary for the two roles
if convenient (simplest, avoids the filter's edge cases entirely); the
automatic filter exists as a safety net when that isn't practical.

**2. "All cameras remain unsolved" after Match & Solve, despite the board
being detected.** Not a bug — traced through `init_poses_pnp()`'s requirement
of ≥4 world-xyz control point observations **per camera**: a ChArUco
detection (or anchor) only supplies world-position points to a camera if
`Detect ChArUco` was actually clicked *under that camera*, exactly like a
manual control point. `Set origin & axes` fixes the world coordinate system
from whichever detections already exist, but doesn't retroactively give
un-detected cameras anything to solve from. This was already true of the
design but not obvious from the UI, which only ever showed one aggregate
"detected" count. Fixed by making the requirement visible instead of by
changing solver behavior: `_build_charuco_group()`'s hint text now says this
explicitly, and `_refresh_charuco_status()` / the anchor status message now
name which specific cameras are still missing a detection.

**3. Sidebar too crowded** once the ArUco and ChArUco panels joined the
pre-existing Control Points / World Position / Camera Intrinsics groups in
the fixed-width side panel — description text and tables clipped vertically,
the intrinsics calibration combo box too narrow to read. Fixed exactly per
the user's own suggestions: `ArUco Markers`, `ChArUco Board`, and
`Camera Intrinsics` are now collapsible (`QGroupBox.setCheckable`, via new
`_make_collapsible()`/`_set_layout_items_visible()` helpers — the latter
recurses into nested row layouts so unchecking a group hides everything in
it, not just its top-level children); `Control Points` and `World Position`
stay always-visible since they weren't reported as crowded and are the
primary controls. The whole right-hand panel is now wrapped in a
`QScrollArea` (`setWidgetResizable(True)`) as a fallback for whatever still
doesn't fit. `_build_intrinsics_group()` is restructured from one cramped
row per camera to three lines (camera label, intrinsics-selector combo alone
on its own line, then the three checkboxes on a third line), with a
`QFrame.Shape.HLine` separator between cameras. Covered by
`test_extrinsics_panel_layout.py` (collapsibility, content hiding, scroll
wrapper presence, un-squeezed combo, separator count).

### Fifth live-testing round: opaque crash on an invalid board size (square_length <= marker_length)

Trying `_on_detect_charuco_clicked` against a newly-configured board (larger
board, new dimensions typed in) crashed with `cv2.error` / `SystemError:
<class 'cv2.aruco.CharucoBoard'> returned a result with an exception set` —
`cv2.aruco.CharucoBoard`'s own constructor enforces `square_length >
marker_length` (and `squares_x/y > 1`), but via a C++ assertion that
surfaces to Python as an opaque `SystemError`, not a catchable `ValueError`,
and definitely not something to show a user. `CharucoDetector.__init__` in
`fiducial_markers.py` now validates these up front and raises a plain
`ValueError` with an actionable message ("square_length must be greater
than marker_length -- got these two swapped?"). Both `page_extrinsics.py`
call sites now handle it: `_on_detect_charuco_clicked` shows the message via
`QMessageBox.warning` instead of crashing; the ArUco/ChArUco overlap filter
in `_on_detect_aruco_clicked` (finding 1 above) logs a warning and skips the
exclusion rather than breaking plain ArUco detection over an unrelated
board misconfiguration. Regression-tested in `test_charuco_detector.py` and
`test_extrinsics_charuco_ui.py`.

### Sixth live-testing round: one camera's board reflections broke the whole session's world frame — led to the §9 design addendum

With the crash above fixed, the board detected cleanly across most cameras,
but one camera (surface reflections off the board in that view) could not
detect it. Two symptoms followed:

- **Solved camera positions never had positive world Z**, even though every
  camera is physically above the board. Root cause, diagnosed from
  `init_poses_pnp`'s own logic (`extrinsics_solver.py:562-609`): the planar-
  pose ambiguity `cv2.SOLVEPNP_IPPE` resolves is a *tilt* ambiguity (two
  locally-optimal orientations, both physically in front of the camera), not
  a full mirror-reflection through the plane — it does not guarantee one
  candidate solution per side of the board. Both real, physically-valid
  interpretations of a camera's view place it on the board's *actual*
  physical side; if the world +Z axis convention established when anchoring
  the board (the "Board face up" checkbox) doesn't match how the board was
  actually lying, then *every* valid solution reports negative Z, because
  the code's own `C_z > 0` preference assumes the axis convention is already
  correct — it can only pick between genuinely ambiguous solutions, it
  can't fix a wrong axis convention. The code comment describing this as
  "two valid solutions (above/below floor)" is an oversimplification worth
  fixing if this area gets touched again.
- **The camera without its own board detection reported a huge aggregate
  reprojection error while every individual control point looked fine.**
  These are two different numbers: the per-camera summary line comes from
  `compute_reprojection_errors()`, evaluated against SIFT-triangulated
  points; the `CP:` figure (and the DEBUG "Per-CP reprojection errors after
  BA" log block) comes from `compute_cp_errors()`, evaluated only against a
  camera's *own* control-point observations. A camera with no board
  detection of its own has no entries in the CP-error path at all — not a
  good number, no number — so the CP-based view shows nothing wrong while
  the SIFT-point view absorbs the full cost of that camera's pose having
  been chained in from a neighbor with no board-anchored constraint of its
  own to correct it.

Both symptoms trace back to the same root cause: one camera lacked a direct,
board-anchored pose and inherited an unconstrained one through chaining, and
nothing surfaced this as clearly as it should have (a colored table cell,
not an explicit warning). This — plus the pre-existing concern that a small
board concentrates the calibration points in a tiny region of the capture
volume, biasing the whole solve toward that region — is what prompted the
design doc's new §9 (portable non-planar calibration rig as primary anchor +
scattered ArUco tags for redundancy), added the same day. See the design
doc for the full three-tier strategy; Phases 8-9 there are design-only.

### Not yet done

- Camera-pose accuracy hasn't been validated against real multi-camera
  footage yet — detection now works, but the design doc's stated Phase 4
  validation criterion (compare reprojection error against a manual-CP
  baseline, verify the anchor reproduces the board's known square size)
  is still open.
- No test yet of the "detected but not anchored" board corners actually
  improving camera-pose solving via the free-CP path (mirrors Phase 3's
  analogous, already-covered case for unknown-size ArUco markers — the
  underlying mechanism is identical and already tested there, but not
  re-verified specifically with ChArUco corners as the input).

## Known open questions (see design doc for detail)

- Registry- vs. session-level scoping for `scene_fiducial_markers`.
- ~~The rigid marker-pose BA residual (Phase 3) is new solver machinery and
  should be prototyped against synthetic data before UI work begins.~~
  Done — see "Phase 3 notes" above.
- Video random-seek performance on long-GOP consumer codecs (GoPros) is
  unmeasured — check early in Phase 1.
- Whether board/marker corners should replace the SIFT pairwise bootstrap
  outright, vs. only supplement it, is left as an internal heuristic for
  now — Phase 3 did not touch this either way (see "Phase 3 notes").
- Marker size input UX (global default + override table) shipped as an
  always-visible table in Phase 3, not the progressive-disclosure version
  originally sketched — may still need revisiting once real rigs are tried.
- **New from Phase 3**: whether the decoupled marker-pose post-pass's
  accuracy is sufficient in practice, or whether known-size markers need
  to become a true joint BA parameter block to meaningfully improve camera
  pose accuracy (not just produce their own clean pose after the fact).
  Only real-footage testing can answer this.
- ~~New from Phase 4: no live UI test yet against a real printed/displayed
  ChArUco board.~~ Done — detection confirmed live (2026-08-09) after
  fixing the axis-order/legacy-pattern settings gotchas (see "Phase 4
  notes" above). Camera-pose accuracy against real multi-camera footage
  is still open.
