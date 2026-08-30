# Marker-based mocap — status

- **2026-08-30** — 1d (finalisation) and 1e (ObjectPanel review) built and
  merged. Building 1e surfaced a real bug in shared code, not specific to
  markers: `merge_observation_sources`/`infer_body_width`/
  `update_single_keypoint_edit` hardcoded `source='body'` throughout. For
  any sequence whose real source is never 'body' (an object's 'markers'
  source), the moment even one edit existed anywhere in the camera,
  `merge_observation_sources`'s "ghost frame → synthesize zero body"
  fallback fired regardless, silently discarding every real, untouched
  keypoint slot's data. Fixed by generalising all three functions to a
  `primary_source`/`source` parameter (default unchanged, so every
  existing person-panel call site is unaffected) rather than working
  around it in ObjectPanel — the whole point of finalising before
  reviewing (previous entry) was reusing this machinery genuinely, not
  papering over its person-only assumption.

- **2026-08-30** — Implementation progress: 1a (ArUco detection layer),
  1b (skeleton generator), and 1c (capture-object plumbing + GUI
  marker-detection run mode) built, tested against synthetic fixtures and
  real capture data (calibration box, `ukemi-tommi-20260509.db`), and
  merged. While starting 1d, found that review-before-finalisation (the
  original 1d→1e order) doesn't fit the codebase: pre-finalisation review
  for a person is track-to-person *stitching*, a real decision with no
  per-frame correction path anywhere in the project; per-frame correction
  only exists post-finalisation, via `pose_observation_edits`. Since an
  object's own phasing already established "no stitching step," swapped
  the order (§7.1 now runs 1d finalisation, then 1e review) so ObjectPanel
  reuses that existing mechanism directly instead of building a parallel
  one for raw `detection_keypoints`. Confirmed with Harri before proceeding.

- **2026-08-30** — Phase 1 broken into six independently-buildable
  sub-phases (design §7.1), each with its own validation check: detection
  layer (1a), skeleton generator (1b, parallel to 1a), capture-object
  plumbing (1c), ObjectPanel review (1d), finalisation + manifest (1e),
  tracker multi-source load + rigid init (1f, phase 1's actual finish
  line). Requested because phase 1 as originally scoped bundled DB schema,
  Python detection, Python finalisation, C++ tracker, and two GUIs into one
  slab with a single end-to-end validation criterion.

- **2026-08-30** — Second review round: UC1 phasing restructured so
  anonymous/reflective dots on props are pulled into the first iteration
  alongside ArUco, instead of waiting for UC2 (Harri: real props already
  combine ArUco + reflective dots, and a dots-only prop is also a valid
  configuration). Design §7 now runs seven phases — ArUco prop (1),
  dot-only prop (2), person + prop together (3), multiple mixed props +
  person — UC1 complete (4) — before UC2's identified (5) and anonymous
  (6) person markers, then moving camera (7). Split driven by labeling
  difficulty, not marker type: rigid-prop dot labeling is a single-body
  problem (algorithms §3.4 tier 1), so it doesn't need the cross-subject
  `MarkerAssociator` machinery UC2 requires until multiple marked bodies
  can compete in phase 4/6. Added the previously-missing cold-start
  procedure for a body with no coded anchor: unlabeled rigid-template
  registration by pairwise-distance RANSAC (algorithms §4.1). UC2 (person
  markers) is confirmed as the next project after UC1, not interleaved
  with it.

- **2026-08-19** — First review round: Harri's inline comments on the
  design addressed. Main outcome: new design §5.2 splits session-scoped
  marker attachments out of the skeleton into a composed *marker
  attachment set* document (so person-scale improvements from marker
  sessions propagate to markerless skeletons), plus clarifications on
  definition/capture-object/skeleton roles (§4.2), symmetry-axis marking
  (§6.1), and global cross-subject assignment for uncoded markers
  (§6.2, algorithms §3.3).

- **2026-08-19** — Design written from the brief + codebase analysis:
  [marker-mocap-design.md](marker-mocap-design.md) (requirements, data
  model, architecture, UX, phasing) and
  [marker-mocap-algorithms.md](marker-mocap-algorithms.md) (detection,
  measurement model, anonymous-marker association, rigid init, offset
  calibration, camera-drift monitoring). Consolidates and supersedes
  `docs/aruco-prop-tracking-design.md` where they conflict; builds on
  `pose-detect-improvements/marker-detection-analysis.md` and
  extrinsics-improvements §10 marker-body infrastructure. Not yet
  reviewed; no implementation started.
