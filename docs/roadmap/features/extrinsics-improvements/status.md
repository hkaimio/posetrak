# Extrinsics Calibration Improvements — Implementation Status

See [extrinsics-improvements-design.md](extrinsics-improvements-design.md) for
the problem statement, requirements, and full technical design.

## Current state

Design only — written 2026-08-09, grounded against the current
`python/app/setup/extrinsics_solver.py` / `page_extrinsics.py` /
`posetrak/db/import_extrinsics.py` implementation and
`docs/extrinsics-calibration-design.md`. No code has been written for this
feature yet.

## Phase summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Video frame source: per-camera random-seek reads, scrub UI replacing PNG-directory loading | ⬜ Not started |
| 2 | Per-control-point, per-frame observations (`ObsPoint`, file format v2) | ⬜ Not started |
| 3 | ArUco marker detection + rigid marker-pose BA residual | ⬜ Not started |
| 4 | ChArUco board detection + coordinate-system anchoring | ⬜ Not started |
| 5 | `scene_fiducial_markers` persistence + recalibration reuse | ⬜ Not started |
| 6 | AprilTag detector backend (extensibility proof) | ⬜ Not started |

## Known open questions (see design doc for detail)

- Registry- vs. session-level scoping for `scene_fiducial_markers`.
- The rigid marker-pose BA residual (Phase 3) is new solver machinery and
  should be prototyped against synthetic data before UI work begins.
- Video random-seek performance on long-GOP consumer codecs (GoPros) is
  unmeasured — check early in Phase 1.
- Whether board/marker corners should replace the SIFT pairwise bootstrap
  outright, vs. only supplement it, is left as an internal heuristic for now.
- Marker size input UX (global default + override table) may need revisiting
  once real rigs are tried.
