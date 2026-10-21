```toml
name = "Keypoint Candidates"
status = "in-progress"
progress_pct = 85
description = """
Several heatmap maxima per keypoint stored per detection run, linked into tracklets at detection \
time, and assigned to skeleton markers by the tracker's shared dot-assignment phase with a \
keypoint-class restriction; strict mask adherence and multi-view agreement as tracking options.
"""
categories = ["detection-pipeline", "tracker-core", "data-model"]
target_release = "TBD"
last_updated = 2026-10-21
```

# Keypoint Candidates — Implementation Status

See [keypoint-candidates-design.md](keypoint-candidates-design.md). Planned as part of
[the SOMA/GEM-X productization plan](../pose-model-spike/soma-gemx-productization-plan.md).

| Item | Status |
|---|---|
| Measurement: candidates left after the strict mask filter | Done: [design §7](keypoint-candidates-design.md#7-measurements-before-and-with-the-build) item 1 |
| Ground-truth harness with the same-candidate metric | Done: `python/tools/gt/eval_kp_candidates.py`, frozen set `kp_eval_set.toml`; reads `kp_candidates` rows and scores them |
| Candidate extraction and online linking in detect run | Done: `posetrak/detection/kp_candidates.py`, `detect run --candidates`, checkbox of the pose extraction panel |
| Finalisation copies candidate rows | Done, refreshed runs keep the rows of unchanged frames |
| Loader, duplicate merge, `KeypointClassModifier` | Done (C++, unit-tested per modifier) |
| `tracker_configs.kp_candidates_enabled` | Done: session schema 69, registry 18; `track run --candidates`, `--strict-mask`, `--multi-view-agreement` |
| Reproduction of the prototype's tracker numbers | **Not met**: candidates on 18.7 / 7.6 cm against the prototype's cross-camera 10.2 / 3.4 (bar 10.7 / 3.9); candidates off identical to a run without candidates; options, cost and like-for-like in design §9; decision on the `suggest` rule and the `dot_*` defaults is the operator's |
| Tracklet editing, `suggest` rule from an automatic grouped choice | Not part of this stage (stage 10 / later) |

Outcome and figures: [design §9](keypoint-candidates-design.md#9-as-built-and-acceptance-results).
