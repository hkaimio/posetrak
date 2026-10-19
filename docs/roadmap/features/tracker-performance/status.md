```toml
name = "Tracker Performance"
status = "in-progress"
progress_pct = 90
description = """
Run time and memory of long tracking runs for 77-keypoint skeletons: hierarchical configuration \
as the default, smoother memory without changing results, tracker rate, and per-person threads.
"""
categories = ["tracker-core"]
target_release = "TBD"
last_updated = 2026-10-19
```

# Tracker Performance — Implementation Status

See [tracker-performance-design.md](tracker-performance-design.md) for the cost model, options and
measurements. Planned as part of
[the SOMA/GEM-X productization plan](../pose-model-spike/soma-gemx-productization-plan.md).

| Item | Status |
|---|---|
| Baseline measurement: single-stage vs hierarchical, 120 vs 60 Hz | Part a and b done for all forms ([design §3](tracker-performance-design.md#procedure-and-results)). Part a, 10,214 steps: single-stage 1,103 s / 19.7 GB, hierarchical with parent `main` 584 s / 7.8 GB (below the 8 GB limit by 0.2 GB), unrestricted parent 1,320 s / 20.5 GB. Part b: 2,385 s / 25.7 GB single-stage, 793 s / 8.7 GB parent `main` |
| Default hierarchical tracker configuration for 77-keypoint skeletons | Built: `tracker_configs.active_joint_groups` (schema 67), stage rows copied by `track run --config`, bundled hierarchical configuration with parent `["main"]`, start-of-run warning. Chosen on cost and the 8 GB limit; accuracy slightly below single-stage ([design §5](tracker-performance-design.md#5-results-of-the-changes)). Hand stages unmeasured: no baseline sequence has refined hands |
| Observations loaded for the requested window only | Built |
| Smoothed covariance kept as diagonal only | Built |
| Forward cache released per stage | Built |
| Compact observation record | Not built: observations do not dominate memory |
| Forward cache spilled to disk | Not built: the hierarchical runs peak at 4.3 GB (10,214 steps) and 6.0 GB (three persons) |
| Per-person threads outside contact | Built; gain within noise on the contact-heavy trial (23% of frames qualify) |
| Lower tracker rate | Dropped; `tracker_fps` stays |
