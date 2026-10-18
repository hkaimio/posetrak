```toml
name = "Track to Person Assignment"
status = "in progress"
progress_pct = 85
description = """
Automatic assignment of YOLOX track fragments to persons by cross-camera triangulation and 3D \
linking, writing the existing assignment rows; covers every frame with a box and reports where \
segmentation or a manual check is needed.
"""
categories = ["detection-pipeline"]
target_release = "TBD"
last_updated = 2026-10-18
```

# Track to Person Assignment — Implementation Status

See [track-person-assignment-design.md](track-person-assignment-design.md). Part of
[the SOMA/GEM-X productization plan](../pose-model-spike/soma-gemx-productization-plan.md), stage 4.

| Item | Status |
|---|---|
| Pieces: detector fragments cut where the detector swaps identities | Done |
| Per-instant cross-camera clustering (trunk keypoints from the pose model registry; RTMPose and SOMA) | Done |
| Tracks that end at contacts; labelling with fragments, distance and separation | Done |
| Boxes by projection of the persons, splits at identity switches | Done |
| Naming: seeds, `--persons`, `--subjects`, the capture's only person | Done |
| Existing rows: refused by default, `--fill-only`, `--replace` | Done |
| Report (cameras, contacts, hand-overs, splits, unassigned) as text and JSON | Done |
| `posetrak detect assign` with `--finalise` and `--dry-run` | Done |
| Stitcher "To check" selector from the report | Done |
| Scoring tool `python/tools/eval_track_assignment.py` | Done |
| Single-person footage without a manual step | Met (six-camera and three-camera runs of one person) |
| Agreement of persons apart on the labelled trial | Not met without seeds: 88.7% (98% required); 95.8% with ten seed rows of one camera; 98.1% with the right person for every track |
| Disagreements listed in the report | Not met without seeds: 69% (90% required); 99.9% with seeds |
| Sequences without a gap where a box exists in two cameras | Not met: one person has a 10 s gap at a contact of two persons 9 cm apart |
| Segmentation seeding from the report | Part of the segmentation UI feature |

Open: identity across a contact where persons are tangled. The cues of the design (the detector's
fragments, the distance between track ends) can point at the wrong person with full confidence; an
appearance cue or a seed after each contact is needed. See the limits in the design.
