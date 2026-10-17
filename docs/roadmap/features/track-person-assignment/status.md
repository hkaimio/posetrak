```toml
name = "Track to Person Assignment"
status = "planned"
progress_pct = 0
description = """
Automatic assignment of YOLOX track fragments to persons by cross-camera triangulation and 3D \
linking, writing the existing assignment rows; covers every frame with a box and reports where \
segmentation or a manual check is needed.
"""
categories = ["detection-pipeline"]
target_release = "TBD"
last_updated = 2026-10-17
```

# Track to Person Assignment — Implementation Status

See [track-person-assignment-design.md](track-person-assignment-design.md). Planned as part of
[the SOMA/GEM-X productization plan](../pose-model-spike/soma-gemx-productization-plan.md).

| Item | Status |
|---|---|
| Per-instant cross-camera clustering | Planned |
| 3D linking and fragment labelling with splits | Planned |
| Subject naming from seeds | Planned |
| Contact and gap report | Planned |
