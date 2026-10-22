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
- `eval_kp_candidates.py` — `verify` (checksums), `picks`, `tracker`, `events`, `compare`, `track-picks`. `tracker --runs GROUP` scores
  a group of `[tracker.*]` runs of `kp_eval_set.toml` (the prototype's `cross_camera` and `cross_camera_mask` runs of range 3
  and its strict mask filter runs `mask_filter_proposal` and `mask_filter_cross_camera` among them); `tracker --run
  SUBJECT=RUN ...` scores runs of a work copy that the set does not name. `compare --detection-run RUN --subject N`
  reports how much of the prototype's candidate file of that subject a detection run with candidates reproduces within
  0.5 px, and the reverse, overall and for the frames whose crop is the same. The frozen file stores the heatmap cell
  index at the cell's left edge and the decoder now uses the cell centre: `--cell-shift` (default 0.5 cell) moves
  the file's positions to the same convention before comparing.

`window-picks --detection-run RUN --sequence SEQ --run TRACKING_RUN --subject N --prototype-picks FILE` compares the
  candidates a tracking run chose over a time window (`tracking_kp_picks`) with the prototype's picks for the same
  settings, on the frames whose crop is the same in the run and in the prototype's candidate file. It reports the share
  of keypoint-frames with the same candidate (picked candidates within 0.5 px, after the half-cell shift) and sorts
  the others by what the two candidate sets allow: the prototype's pick has no counterpart among the run's candidates,
  the run's has none among the prototype's, or both exist and the choices differ. Columns: all frames of the range,
  the interior of the range (a second from each end left out, where the two do not average over the same window),
  inside the hard windows, outside them. `--prototype-candidates FILE` names the candidate file the prototype's picks
  were made from when it is not the evaluation set's (a file made from the run's own candidates compares the choice
  alone). The prototype's picks file is made by running its cross-camera choice (`xcam.XAssign.run`) with the same
  neighbour limits; its layout is in the subcommand's help.

`track-picks --range RANGE --run SUBJECT=RUN ...` traces the candidate each used observation of a tracking run
  came from (`kp_gt.tracker_picks`: the keypoint of the marker, the sequence row nearest in time, the candidate whose
  undistorted position is nearest to the observation) and reports the share of labelled keypoint-frames, among those
  the run took a candidate for, within the tolerance of the label, next to the strongest maximum's share on the same
  keypoint-frames. It needs no label-fed tracker run, so it scores ranges that have only labels.

Picks come from the strongest maximum of the candidates, or from a pose observation sequence
(`picks --pick-sequence SUBJECT=SEQUENCE`), so a productized pipeline is scored without changing the harness.

### Neighbour limits of the windowed choice

The windowed choice of keypoint candidates needs the largest plausible distance between skeleton neighbours, in box
heights (`docs/roadmap/features/keypoint-candidates/keypoint-candidates-design.md`, section 10.4).

- `derive_neighbour_limits.py` computes the shipped table (`posetrak/detection/soma77_neighbour_limits.py`) from the
  labels of a range of the evaluation set: 1.15 times the 98th percentile of the labelled distance of each pair, for
  the pairs with more than 50 samples. The shipped table uses range 1 only, so that range 3 stays a test.
- `estimate_neighbour_limits.py` computes the same percentile in a tracked sequence from the candidates the
  multi-camera consensus flagged (the dump of `posetrak-tracker track --kp-choice-dump`), prints it against the
  shipped table and writes a choice model file that `track run --kp-choice-model` takes.
