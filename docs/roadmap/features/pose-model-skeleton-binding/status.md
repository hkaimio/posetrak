```toml
name = "Pose Model and Skeleton Binding"
status = "planned"
progress_pct = 0
description = """
Pose models as keypoint vocabularies, a keypoint manifest per observation source for every new \
sequence, model-qualified landmark names, SOMA-77 markers and offsets as catalog data in the \
default skeletons, one anatomical joint-limit table with a hinge test, and default skeletons \ncomposed from hierarchy, limit, marker-set and dimension layers.
"""
categories = ["detection-pipeline", "tracker-core", "data-model"]
target_release = "TBD"
last_updated = 2026-10-17
```

# Pose Model and Skeleton Binding — Implementation Status

See [pose-model-skeleton-binding-design.md](pose-model-skeleton-binding-design.md). Planned as part
of [the SOMA/GEM-X productization plan](../pose-model-spike/soma-gemx-productization-plan.md).

| Item | Status |
|---|---|
| Pose model registry | Planned |
| Manifest per observation source (schema and reader) | Planned |
| Model-qualified landmarks | Planned |
| SOMA offsets catalog file and skeleton generator | Planned |
| Default skeletons with both marker sets | Planned |
| Joint-limit table and hinge test | Planned |
| Skeleton layer files and `skeleton compose` | Planned |
