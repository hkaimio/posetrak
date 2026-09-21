# Marker-based mocap: validation baseline

Marker-based tracking was developed by iterating on real captures, so many of
its results exist only as numbers recorded along the way. This page fixes them
as a reproducible baseline: eight real cases that can be re-run from the
committed code, with the values a correct build produces. Run them before and
after any change to marker detection, dot assignment, initialisation or the
UKF, and read any difference as either a regression or a change to be
explained.

The checks are numeric and do not replace visual inspection. A change to
assignment or initialisation still needs the rendered reprojection video,
cropped to the union of tracked and observed extents
(productization-architecture-and-plan.md §6).

## Running it

`scripts/validate_marker_mocap.py` re-runs each case and compares it with a
recorded tracking run and with expected values. Cases are described in a JSON
file kept beside the session databases, because it names databases and run IDs
that exist only where the captures are (format:
`scripts/validate_marker_mocap.example.json`).

```
python scripts/validate_marker_mocap.py --cases /path/to/cases.json
python scripts/validate_marker_mocap.py --cases cases.json --only pen pad
python scripts/validate_marker_mocap.py --cases cases.json --reuse-copies --reuse-detection
```

- **Databases are never modified.** Each session database is copied with
  SQLite's backup API (so a live WAL is included), migrated to the current
  schema, and every run is written to the copy. Copying takes about 10 seconds
  for a 7 GB database and 5 minutes for a 21 GB one.
- **The tracker is the repository's `optbuild` build.** Build it with
  `meson compile -C optbuild posetrak-tracker`. The driver deliberately does
  not use `default_binary_path()`, which prefers an installed copy under
  `~/.posetrak` that can be older than the source.
- **Comparing with another build.** `--reference-binary PATH` runs every case a
  second time with that binary and requires byte-identical results: the same
  stored tracking results and per-observation results. It is the check that a
  refactoring changes no behaviour. Keep a copy of the optimized tracker, and its
  DLLs, from before the change; a binary run from another directory needs the
  DLLs beside it.
- **Dot jumps.** Every run prints a count of resolved dot observations that leave
  their slot's own motion: for each camera and slot, an observation up to six
  steps after two consecutive ones is compared with the constant-velocity
  extrapolation of those two. It counts departures above 50 px after a gap
  and above 100 px at any gap. It is information for comparing two runs of one
  case, not a check: the raw candidates of a real capture jitter a great deal on
  their own (about 130 departures in 9900 observations on the leg window), and the
  count is only decisive where the failure is gross, as in the ball with clutter.
- The script skips, with exit code 0, when the cases file or a database used by
  a selected case is absent, so it is safe to leave in a workflow that runs
  elsewhere, and a capture on a drive that is not mounted does not block the
  cases on the other drives.

## Cases and baseline values

Measured with the driver on the optimized build. Reprojection is the median,
per camera, of the distance between the observed and the predicted pixel over
the position observations the filter used. NIS/dof is the mean over tracked
steps. The tracker is deterministic: re-running a case on unchanged code and
data reproduces every number below.

| Case | Tracked steps | Lost | Init RMS | Mean NIS/dof | Reprojection median per camera |
|---|---|---|---|---|---|
| Rigid pen (ArUco), 15 s | 1747 / 1798 (97.2 %) | 0 | 1.8 mm | 0.033 | 4.6 – 6.5 px (5 cameras) |
| Rigid pad (ArUco), 15 s | 1708 / 1800 (94.9 %) | 0 | 3.6 mm | 0.010 | 2.2 – 12.6 px (6 cameras) |
| Person with a 16-dot leg module, full trial | 11588 / 11588 (100 %) | 0 | – | 1.920 | 16.4 – 32.5 px (6 cameras) |
| Single reflective ball, one throw (0.75 s) | 89 / 90 (98.9 %) | 0 | – | 0.150 | 18.9, 71.3, 20.4 px (3 cameras) |
| Sword with ArUco tags and dots, 66 s | 6108 / 6626 (92.2 %) | 518 | 4.6 mm | 2.786 | 7.5 – 9.4 px (6 cameras) |

Two windowed cases of the leg module and the ball isolate the failure that
dot assignment robustness work targets, a resolved candidate that is the wrong
physical point. They are recorded on the current behaviour, which is the
reference the later changes are measured against, not a target:

| Case | Tracked steps | Mean NIS/dof | Reprojection medians | Dot jumps (after a gap > 50 px / any > 100 px) |
|---|---|---|---|---|
| Leg module, 4 s (40 – 44 s) around a short marker dropout | 480 / 480 | 1.996 | 18.5 – 29.3 px (6 cameras) | 132 / 200 |
| Ball, one throw, with the raw `gopro13_02` candidates added | 89 / 90 | 11.826 | 133.0, 175.6, 93.4 px (3 cameras) | 1 / 10 |

The ball case is the ball case above with one more camera's unfiltered dot
candidates added (the leg module's dots on that camera over the same span, which
include near-static clutter). Without them the medians are 18.9, 71.3 and 20.4 px,
so the added camera pulls the fused position far off course: its wrong candidates
pass the outlier gate although they are the wrong physical point. The leg window
sits around a two-step dropout of the ankle markers; the reacquisition jumps of
several hundred pixels in it are the same failure after a gap.

The seventh case tracks two subjects together, the person with the leg module and
the ball, over the ball's first throw (0.75 s), and compares each with the same
subject tracked alone over the same window:

| Subject | Tracked steps | Mean NIS/dof, joint / alone | Reprojection medians, joint = alone |
|---|---|---|---|
| Person with the leg module | 89 / 89 | 1.577 / 1.577 | 19.6 – 30.1 px (6 cameras) |
| Ball | 89 / 89 | 0.150 / 0.150 | 18.9, 71.3, 20.4 px (3 cameras) |

The case also requires that no observed pixel is used by both subjects at the same
time and camera, both on this run and on a variant where exact copies of the
person's dot candidates are added to the ball's sequence.

For the leg module the comparison that matters is against tracking the same
trial without dots: the recorded markerless run has mean NIS/dof 1.959, and the
case requires the dot-augmented run to stay within 10 % of it.

Approximate wall time on the development workstation: the pen, pad and ball
cases take a few seconds each, the leg module 28 minutes, the sword 16 minutes
of detection plus 15 seconds of tracking, and the two-subject case about a minute
(two solo runs and two joint runs of 17 seconds).

## What each case exercises

- **Leg module window and ball with clutter.** Robustness of the assignment to a
  confidently wrong candidate: after a gap, and when a camera's raw feed holds
  near-static clutter. Nothing in the current tracker addresses either, so these
  are the cases on which an assignment change has to show an effect.

- **Pen and pad.** Rigid-body initialisation by Kabsch fit over coded markers,
  free-flyer tracking and RTS smoothing, with factory-default tracker settings.
- **Leg module.** Articulated dot assignment: batched sigma-point marker
  prediction, the shared Hungarian assignment across cameras, and the check that
  adding dots does not degrade body tracking.
- **Ball.** A dots-only subject with one anonymous marker, started from a seed
  position, tracked from three hand-tracked cameras.
- **Sword.** The full pipeline: ArUco and dot detection on six cameras with
  background subtraction and the chroma filter, tracklet linking, finalisation
  into an object sequence, then tracking with streak velocity.
- **Person + ball.** Joint dot assignment across subjects: candidates that two
  subjects' sequences share are one candidate that either may claim, a candidate
  only one sequence holds can only be claimed by that subject, and a seed applies
  only to the dots-only subject. Joint and solo results are identical, which is
  what makes a prop tracked from hand-placed points safe alongside a person whose
  automatic detections cover the same cameras.

## Notes on how the values were established

- **The sword's stored detections could not be reused.** They were written in an
  earlier dot-candidate blob layout, which the current code refuses to read by
  design. The case therefore repeats the recorded detection run (same object,
  cameras, time range and dot settings) with the current pipeline, finalises it,
  and tracks the result. The earlier recorded run tracked 6057 of 6626 steps
  (91.4 %) with mean NIS/dof 3.26; the fresh detection tracks 6108 (92.2 %) with
  2.79, and the per-camera reprojection medians agree within about 1 px. Both
  detection (notably the motion-gated tracklet linker) and the tracker have
  changed since the recorded run, so a small difference is expected and the fresh
  run, not the historical one, is this case's reference. Detection makes up most of the case's run time; with
  `--reuse-detection` a repeat run tracks the sequence made earlier.
- **The ball is started from an approximate seed.** The seed used originally was
  not recorded, so the case starts from the root position of the first tracked
  step of the recorded run. The result matches the recorded run to the displayed
  precision, which shows the filter converges from a seed a few centimetres off.
  Producing that seed from the observations (multi-view triangulation) is
  future work.
- **Ball reprojection differs from the earlier summary.** An earlier summary
  quoted 12 – 57 px per camera for this throw; with the definition above the
  three cameras measure 18.9, 71.3 and 20.4 px, on the recorded run as well as
  the re-run. The earlier figure could not be reproduced with this definition,
  so the values here are the reference going forward.
- **The pen, pad and leg module reproduce their recorded runs exactly**, to the
  displayed precision, including every per-camera median.

## Limits

- The cases cover one capture each, so they detect regressions in the
  behaviour they exercise, not general accuracy.
- The two-subject case has no candidates that the two stored sequences share, so
  the shared-candidate path is exercised by a constructed variant (copies added
  in a database copy) and by unit tests, not by two sequences that were recorded
  that way. The window is one throw, since the subjects are stepped together and
  the ball's sequence covers only its throws.
- The reference values are current behaviour, not ground truth. A change that
  improves tracking will fail a check until the reference is deliberately updated
  and the reason recorded.
