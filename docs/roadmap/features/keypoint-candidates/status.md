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
last_updated = 2026-10-25
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
| Pick equality with the prototype | The choice is the prototype's: on the run's own candidates it picks the same candidate for all but 1 of 176,113 (subject 1) and 3 of 184,523 (subject 2) keypoint-frames. Against the picks the prototype made on `gt6m` it is 96.2% / 96.1% on frames with the same crop (bar 99%; 99.5% / 99.2% outside the hard windows); the rest is the consensus flags of the file, which cannot be reproduced, design §10.7 |
| Reproduction with the windowed choice | **Not met on the product path** (mask box source with the gate: `'window'` 13.1 / 3.8, with the body-only strict filter 10.73 / 3.48 against the bar 10.7 / 3.9; ungated candidates gave 9.4 / 3.7): on boxes rebuilt from the masks the prototype used (all its crops reproduced) 13.5 / 4.0 cm against the prototype's cross-camera 10.2 / 3.4 (bar 10.7 / 3.9); better than candidates off (14.3 / 4.7) and `'step'` (13.9 / 5.4). With the strict filter over all keypoint classes 9.4 / 6.0, with the mask term 12.9 / 3.9, and with the strict filter on the body keypoints only (`--kp-strict-classes body`, the default class setting) 9.4 / 3.7 on ungated candidates (chosen after seeing the range 3 figures of six variants; design §10.7). On the basis with the fitted ace2 intrinsics and calibration `b416cd21` the best setting is `'window'` with the ace2 outside the choice, 6.0 / 3.6 against the prototype's 5.5 / 2.5, and the filter makes subject 1 worse (6.8). The whole-trial check of the filter classes (against the operator-edited runs) is not run yet. The earlier box run was built from another version of the masks. The default stays `'step'`; strict filter and mask term stay off until the operator decides |
| Sticky choice and process noise against jitter | Done, off by default: `kp_choice_switch_penalty` / `kp_choice_jump_box` (session schema 71 / registry 20, provisional numbers), `track run --kp-switch-penalty`, `--kp-jump-box`; Viterbi chain in the last round, unit-tested. Grid of penalty 0 / 0.2 / 0.5 / 1.0 and process noise 0.3 / 0.2 / 0.15 (design §10.10): the best cell (1.0 / 0.15) has arms 14.7 and hands 22.6 mm (baseline 18.7 / 31.7, `'step'` 14.4 / 23.1) at the baseline's accuracy, but 15 episodes not in the edited runs against the target of 10 (`'step'` 8), so the penalty is not a default. Most of the gain is the process noise (0.3 to 0.15: arms plus hands 50.3 to 40.2 mm), a recommendation and not a changed default. A per-group process noise is not available for arms and hands (one parent stage). Explained for readers outside the feature, with terms, metrics and run ids, in [jitter-study-report.md](jitter-study-report.md). The recommended setting, exported and retargeted in iClone, looks clearly better than earlier runs on the same capture. The defaults (process noise, switch penalty) stay an open item until more captures have been tracked |
| Tracklet editing, `suggest` rule from an automatic grouped choice | Not part of this stage (stage 10 / later) |

Outcome and figures: [design §9](keypoint-candidates-design.md#9-as-built-and-acceptance-results).
