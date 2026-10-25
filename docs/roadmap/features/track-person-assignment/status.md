```toml
name = "Track to Person Assignment"
status = "complete"
progress_pct = 100
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
| Several persons among bystanders (six cameras, five persons, 97 s, `--persons 5`) | Met: pure stretches per person, the subject 99.8% after the first 16 s; the label changes are listed in the report |
| Persons count with bystanders | Must be given with `--persons` (the automatic count found three of five) |
| Labelled contact trial (98% where apart) | Not met: 88.7% without seeds; kept as the record of the contact case |
| Segmentation seeding from the report | Part of the segmentation UI feature |

Not attempted: identity across a contact where persons are tangled. The cues of the design (the
detector's fragments, the distance between track ends) can point at the wrong person with full
confidence there; an appearance cue would be needed. Such stretches are listed for the stitcher or
segmentation. See the limits in the design.
