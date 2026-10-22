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
last_updated = 2026-10-22
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
| Choice over a time window (`kp_candidate_choice = 'window'`) | Done: phase of `posetrak-tracker track` (`keypoint_window_choice`, `keypoint_choice_phase`), session schema 70 / registry 19 (provisional numbers), settings and `track run` options, picks stored in `tracking_kp_picks`, neighbour limits shipped for SOMA-77, `eval_kp_candidates.py window-picks`; design §10 |
| Pick equality with the prototype | The choice is the prototype's: it picks the same candidate for all but 1 of 176,113 (subject 1) and 3 of 184,523 (subject 2) keypoint-frames of the run's own candidates. Against the picks the prototype made on its frozen candidate file the share is 92.9% / 94.0% on frames with the same crop (bar 99%): the file's consensus flags and the frames extracted differently, design §10.7 |
| Reproduction with the windowed choice | **Not met**: 16.0 / 4.6 cm against the prototype's cross-camera 10.2 / 3.4 (bar 10.7 / 3.9); better than candidates off (17.8 / 5.2) and `'step'` (17.1 / 5.3). With the strict filter 15.9 / 4.5 and with the mask term 15.7 / 4.6. The default stays `'step'`; strict filter and mask term stay off |
| Tracklet editing, `suggest` rule from an automatic grouped choice | Not part of this stage (stage 10 / later) |

Outcome and figures: [design §9](keypoint-candidates-design.md#9-as-built-and-acceptance-results).
