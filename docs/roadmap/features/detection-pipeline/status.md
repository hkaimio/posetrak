```toml
name = "Detection Pipeline"
status = "planned"
progress_pct = 0
description = """
One detect-run frame loop with a pluggable box source (YOLOX, segmentation masks, an existing \
run), SOMA-77 as a native batched pose model, a decode/infer/write pipeline that keeps the GPU \
busy, optional variable frame rate, and licence-compliant download of the GEM-X weights.
"""
categories = ["detection-pipeline"]
target_release = "TBD"
last_updated = 2026-10-17
```

# Detection Pipeline — Implementation Status

See [detection-pipeline-design.md](detection-pipeline-design.md). Planned as part of
[the SOMA/GEM-X productization plan](../pose-model-spike/soma-gemx-productization-plan.md).

| Item | Status |
|---|---|
| Profiling of the three routes | Done: timing hooks behind `POSETRAK_PROFILE_DETECTION`, results in [design §3](detection-pipeline-design.md#results); the loop is serial and the GPU about 40% busy |
| Box source protocol and the three sources | Planned |
| SOMA-77 as a registered, batched estimator | Planned |
| Decode / infer / write pipeline | Planned (scope set by the profiling) |
| Model download with licence acceptance | Planned; the DINOv3 question to NVIDIA is drafted in [design §4](detection-pipeline-design.md#open-item-licence-question-to-nvidia), not yet sent |
| Variable frame rate | Profiling shows decode dominates today but the model dominates once decode is overlapped; still off by default, decide after the pipeline exists |
