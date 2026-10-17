```toml
name = "Keypoint Candidates"
status = "planned"
progress_pct = 0
description = """
Several heatmap maxima per keypoint stored per detection run, linked into tracklets at detection \
time, and assigned to skeleton markers by the tracker's shared dot-assignment phase with a \
keypoint-class restriction; strict mask adherence and multi-view agreement as tracking options.
"""
categories = ["detection-pipeline", "tracker-core", "data-model"]
target_release = "TBD"
last_updated = 2026-10-17
```

# Keypoint Candidates — Implementation Status

See [keypoint-candidates-design.md](keypoint-candidates-design.md). Planned as part of
[the SOMA/GEM-X productization plan](../pose-model-spike/soma-gemx-productization-plan.md).

| Item | Status |
|---|---|
| Measurement: candidates left after the strict mask filter | Done: [design §7](keypoint-candidates-design.md#7-measurements-before-and-with-the-build) item 1 |
| Ground-truth harness with the same-candidate metric | Done: `python/tools/gt/eval_kp_candidates.py`, frozen set `kp_eval_set.toml`; reads `kp_candidates` rows once they exist |
| Candidate extraction and online linking in detect run | Planned |
| Finalisation copies candidate rows | Planned |
| Loader, duplicate merge, `KeypointClassModifier` | Planned |
| `tracker_configs.kp_candidates_enabled` | Planned |
| Reproduction of the prototype's tracker numbers | Planned |
