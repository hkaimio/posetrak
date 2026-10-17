```toml
name = "Tracker Performance"
status = "planned"
progress_pct = 0
description = """
Run time and memory of long tracking runs for 77-keypoint skeletons: hierarchical configuration \
as the default, smoother memory without changing results, tracker rate, and per-person threads.
"""
categories = ["tracker-core"]
target_release = "TBD"
last_updated = 2026-10-17
```

# Tracker Performance — Implementation Status

See [tracker-performance-design.md](tracker-performance-design.md) for the cost model, options and
measurements. Planned as part of
[the SOMA/GEM-X productization plan](../pose-model-spike/soma-gemx-productization-plan.md).

| Item | Status |
|---|---|
| Baseline measurement: single-stage vs hierarchical, 120 vs 60 Hz | Part a and b done for all forms ([design §3](tracker-performance-design.md#procedure-and-results)). Part a, 10,214 steps: single-stage 1,103 s / 19.7 GB, hierarchical with parent `main` 584 s / 7.8 GB (below the 8 GB limit by 0.2 GB), unrestricted parent 1,320 s / 20.5 GB. Part b: 2,385 s / 25.7 GB single-stage, 793 s / 8.7 GB parent `main` |
| Default hierarchical tracker configuration for 77-keypoint skeletons | Planned; cost case made (3x faster, 3x less memory than single-stage on the three-person trial), but the restricted form moves the failure events by 4-164 cm against the unrestricted one, so the default needs the ground-truth comparison first; the groups move into `tracker_configs.active_joint_groups` |
| Observations loaded for the requested window only | Planned |
| Smoothed covariance kept as diagonal only | Planned |
| Forward cache released per stage | Planned |
| Forward cache spilled to disk | Only if the above are not enough |
| Per-person threads outside contact | Planned |
