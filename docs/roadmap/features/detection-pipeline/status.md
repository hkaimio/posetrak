```toml
name = "Detection Pipeline"
status = "in-progress"
progress_pct = 75
description = """
One detect-run frame loop with a pluggable box source (YOLOX, segmentation masks, an existing \
run), SOMA-77 as a native batched pose model, a decode/infer/write pipeline that keeps the GPU \
busy, optional variable frame rate, and licence-compliant download of the GEM-X weights.
"""
categories = ["detection-pipeline"]
target_release = "TBD"
last_updated = 2026-10-18
```

# Detection Pipeline — Implementation Status

See [detection-pipeline-design.md](detection-pipeline-design.md). Planned as part of
[the SOMA/GEM-X productization plan](../pose-model-spike/soma-gemx-productization-plan.md).

| Item | Status |
|---|---|
| Profiling of the three routes | Done: timing hooks behind `POSETRAK_PROFILE_DETECTION`, results in [design §3](detection-pipeline-design.md#results); the loop is serial and the GPU about 40% busy |
| Box source protocol and the three sources | Done: `box_sources.py` (YOLOX, segmentation masks at mask resolution, existing run) |
| SOMA-77 as a registered, batched estimator | Done: `soma77-vitpose-h` in `backends_soma77.py`, `estimate_batch` for it and for RTMPose/ViTPose; identical to the prototype's keypoints on the same boxes (maximum difference 0.0 px over 10 keypoint sets, CPU) |
| Decode / infer / write pipeline | Done: `pipeline.py`, `frame_readers.py` (threads or child processes); `detect run`, `PoseWorker` and the dialogs use it. YOLOX is not batched (its ONNX graph has batch size 1) |
| Decode producers: threads or processes | Default threads; 30 s smoke test of both modes without a hang; the 30-minute soak run is the operator's (`decode_soak.py`, [design §3](detection-pipeline-design.md#decode-producers-threads-or-processes)) |
| Hand pass inside the loop | Done for COCO-133 models: batched hand pose model, no second decode (the hand detector has batch size 1) |
| Run inputs (schema 66) | Done: `detection_runs.seg_quality_run_id`, `seg_revision`, `source_detection_run_id`; migration checked on a copy of a real session database |
| Model download with licence acceptance | Done: dialog, `detect run` prompt / `--accept-licences`, About box, user guide. The licence texts are bundled (`posetrak/data/licences/`), accepted by their SHA-256 (a changed text in a release asks again), no network needed to show them. |
| Acceptance measurements | Done on the profiling clip ([design §3](detection-pipeline-design.md#results-of-the-unified-loop)): 2.9 times the prototype's throughput; GPU 79-82%, short of 85% (model-bound, crop preparation and serial YOLOX on the inference thread); RTMPose gap explained by GPU clock-down. Decode soak still to run |
| Variable frame rate | Profiling shows decode dominates today but the model dominates once decode is overlapped; still off by default, decide after the pipeline exists |
