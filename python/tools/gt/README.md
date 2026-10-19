# Ground-truth harness

Tools for building hand-verified ground truth for reflective-dot detection
and marker assignment, and for scoring each pipeline stage against it. Any
change to detection, tracklet linking or assignment is meant to ship with a
before/after table from these tools
(`docs/roadmap/features/marker-based-mocap/marker-catalog-and-assignment-redesign.md` §2,
plan §6).

Scripts import shared helpers as `tools.<module>` (`python/` on `sys.path`),
so run them with `python python/tools/gt/<name>.py`. Usage examples use
`/path/to/...` placeholders; the labeled frame sets themselves are capture
data and live outside the repository.

## Workflow

1. **Choose frames.** `build_gt_frame_manifest.py` resolves a list of global
   times to each camera's own `video_frame` through the shot's sync table
   (cameras do not share frame numbering) and writes the manifest the
   labeling tools read.
2. **Label.**
   - `label_dot_ground_truth.py` — click every visible reflective dot. With
     `--slots` each point also gets a marker slot name and a per-camera
     track id.
   - `label_marker_slots_gui.py` — Qt tool for tagging *detected* dots with a
     slot name.
3. **Refine.** `refine_dot_ground_truth.py` snaps each click to the local
   intensity centroid. The default `blacklist` mode matches the production
   detector's centroid; use `subtract-ladder` for captures detected with
   background subtraction.
4. **Score a stage.**
   - `validate_dot_detector.py` — `detect_blobs()` against a labeled set, in
     seconds, without a multi-hour detection run.
   - `eval_dot_detection.py` — recall and precision per camera.
   - `eval_fk_prediction.py` — FK-predicted marker position against
     slot-labelled ground truth.
5. **Sweep parameters.**
   - `sweep_dot_detection.py` — per-camera `threshold` × `max_saturation`.
   - `sweep_linker_and_b2.py` — `MotionGatedLinker` parameters and the
     tracklet grouping thresholds together, without re-running detection.

These stay tools, not `posetrak` commands: they operate on labeled sets that
only exist while a capture is being tuned.

## Keypoint candidates and pose-model picks

A second harness scores the keypoints a pose pipeline picked per (camera, frame, keypoint), and tracking runs
fed with them, on a frozen evaluation set of operator labels. Its parts:

- `kp_gt.py` — library. `read_candidates()` is the one reader of keypoint candidates: the `kp_candidates` rows of
  a detection run (blob format of `docs/roadmap/features/keypoint-candidates/keypoint-candidates-design.md`
  §4.1) or a candidate file of the prototype labeller. `pick_metrics()` reports the within-3%-of-box-height
  share, the same-candidate share (the pick is the candidate the label holds), picks missing on visible labels
  and false picks on not-visible ones. `tracker_comparison()` and `event_distance()` compare tracking runs with
  the label-fed reference run and on the failure events.
- `kp_eval_set.toml` — the frozen evaluation set: labelled ranges, hard windows, failure-event list, reference
  runs and the scoring script, named in one place. The data (labels, candidate files, events, SHA-256 of
  each) is capture data and lives outside the repository: pass `--root` or set `POSETRAK_KP_EVAL_ROOT`; the
  tracking runs are in the session database of `--db` / `POSETRAK_KP_EVAL_DB`, opened read-only.
- `eval_kp_candidates.py` — `verify` (checksums), `picks`, `tracker`, `events`.

Picks come from the strongest maximum of the candidates, or from a pose observation sequence
(`picks --pick-sequence SUBJECT=SEQUENCE`), so a productized pipeline is scored without changing the harness.
