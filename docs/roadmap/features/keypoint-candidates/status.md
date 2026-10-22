```toml
name = "Keypoint Candidates"
status = "in_progress"
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
| Likelihood cost and tracklet continuity of keypoint candidates | Done: `kp_candidate_peak_weight`, `kp_candidate_tracklet_multiplier` (same schema versions), `track run --kp-peak-weight`, `--kp-tracklet-multiplier`; unit-tested (dot-only assignment unchanged, stronger candidate wins at equal distance, continuity per keypoint); design §9.2 |
| Scoring of the tracker's picks without a label-fed run | Done: `eval_kp_candidates.py track-picks`, `kp_gt.tracker_picks` |
| Reproduction of the prototype's tracker numbers | **Not met**: with the likelihood cost, on boxes holding 82-83% of the prototype's crops, candidates on 16.8 / 5.6 cm (off 17.3 / 5.4) against the prototype's cross-camera 10.2 / 3.4 (bar 10.7 / 3.9); candidates off identical to a run without candidates; settings chosen on range 1 (flat grid), figures, cost and like-for-like in design §9; decision on the `suggest` rule and the `dot_*` defaults is the operator's |
| Tracklet editing, `suggest` rule from an automatic grouped choice | Not part of this stage (stage 10 / later) |

Outcome and figures: [design §9](keypoint-candidates-design.md#9-as-built-and-acceptance-results).
