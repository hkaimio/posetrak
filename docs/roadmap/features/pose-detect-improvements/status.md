```toml
name = "Pose Detection Improvements (Aikido Capture)"
status = "proposal"
description = """
Brainstormed detection- and tracker-side improvements targeting aikido-specific failure modes: \
out-of-distribution poses (ukemi, inversions), limb-ownership ambiguity during grabs/throws, \
hakama-hidden knees, near-identical uniforms defeating re-identification, and back-view hip \
bias. Companion analysis covers marker-augmentation as a complementary, non-markerless option.
"""
categories = ["detection-pipeline"]
target_release = "TBD"
last_updated = 2026-08-06
```

# Pose Detection Improvements — Implementation Status

See:
- [pose-detect-improvements-analysis.md](pose-detect-improvements-analysis.md) — tiered
  brainstorm of detector- and tracker-side improvements (rotation-canonicalized crops,
  other-person masking, ViTPose fine-tuning, ZUPT floor-contact constraints, wrist-grab
  classifier, and others)
- [marker-detection-analysis.md](marker-detection-analysis.md) — companion analysis for a future
  marker-augmentation capability (passive dots, fiducials, colored clothing as identity cues)

## Current state

Both documents are analysis/brainstorm captures from a single discussion session
(2026-07-16), explicitly **not scheduled for immediate development**. Nothing here has been
implemented. Several items build on infrastructure that *is* already implemented elsewhere in
the codebase (Cutie segmentation, hand-detection-refinement's crop-conditioning trick,
`error-improvements` Phase 5's cross-person machinery) — the analysis docs are explicit about
what's already-done baseline vs. new proposal.

The analysis docs recommend three starting picks, in priority order: rotation-canonicalized
detector crops (using the tracker's own 3D orientation prediction to make an inverted/bent pose
look upright to the 2D detector), painting the other person out of a crop before running the
whole-body detector on it, and an edit-harvesting ViTPose fine-tuning loop (using
`pose_observation_edits` and multi-view-consistent pseudo-labels as free training data).

## Known issues / open questions

Not applicable in the usual sense — this is a proposal-stage document, not a partially-built
feature. See each analysis doc's own numbered items for the specific open questions listed
against each suggestion (e.g. marker-detection-analysis.md's retroreflective-vs-colored-dot
shoot-out, color palette size under real lighting, cloth-mounted marker noise).
