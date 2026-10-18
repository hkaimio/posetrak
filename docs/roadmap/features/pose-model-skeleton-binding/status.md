```toml
name = "Pose Model and Skeleton Binding"
status = "in-progress"
progress_pct = 95
description = """
Pose models as keypoint vocabularies, a keypoint manifest per observation source for every new \
sequence, model-qualified landmark names, SOMA-77 markers and offsets as catalog data in the \
default skeletons, one anatomical joint-limit table with a hinge test, and default skeletons \
composed from hierarchy, limit, marker-set and dimension layers.
"""
categories = ["detection-pipeline", "tracker-core", "data-model"]
target_release = "TBD"
last_updated = 2026-10-18
```

# Pose Model and Skeleton Binding — Implementation Status

See [pose-model-skeleton-binding-design.md](pose-model-skeleton-binding-design.md). Stage 2 of
[the SOMA/GEM-X productization plan](../pose-model-spike/soma-gemx-productization-plan.md).

| Item | Status |
|---|---|
| Pose model registry (`posetrak/detection/pose_models.py`), viewer and finalisation read it | Done |
| SOMA-77 estimator (`backends_soma77.py`, cell-centre decoding and its test) | Done; registering it as a batched estimator and the frame loop belong to the detection pipeline stage |
| Manifest per observation source: schema (session version 65, provisional), migration, finalisation | Done |
| Manifest per observation source: Python merge of hand rows, `track run` check | Done |
| Manifest per observation source: C++ reader and its tests | Done; the C++ test suite passes in a debug build (including the five new binding tests) |
| Model-qualified landmarks | Done (reader accepts bare prototype names) |
| SOMA offsets catalog file (`catalog/pose-models/soma77.marker-offsets.yaml`) | Done |
| Default skeletons with both marker sets | Done (new ids) |
| Joint-limit table and hinge test | Done |
| Skeleton layer files, `skeleton compose` (also adds a pose model to a skeleton) | Done |
| Run fails naming the missing pose model | Done (Python before the tracker starts; C++ reader as a second line) |
| Migration check on a copy of a real session database | Done: see below |
| Held-out check on the reference capture | Done: 2.2 / 3.9 / 2.7 cm with the regenerated default skeleton and the new reader (see below) |
| Wrist saturation | Cause examined (design section 2.6); with the new table no wrist axis saturates in the calibration clip (see below); the trial in which the earlier table saturated was not re-run |
| UI action "add SOMA-77 markers" | Not done (out of scope; `skeleton compose --base` does the work) |

## Migration check

A copy of the reference capture's session database (13.6 GB, schema 62) migrated to 65: the 640 manifest
rows keep their content and get `obs_source = 'body'`; of the other 47 tables, the 45 that earlier versions
did not touch are unchanged (row counts and SHA-256 of every row), `tracking_runs` and
`tracking_run_persons` are unchanged apart from the columns versions 63 and 64 add, and `run_progress` is
new (version 64).

## Held-out check

Reference capture, calibration clip, tracker configuration of the prototype run, held-out window
123.5-135 s, joint position error against the reference tracking run (median over time of the mean over
the group, cm):

| Run | legs | arms | head |
|---|---|---|---|
| Prototype run (all 77 SOMA markers, earlier limits) | 2.4 | 4.0 | 2.6 |
| Regenerated default male skeleton (COCO, hand21 and SOMA markers) and the new reader | 2.2 | 3.9 | 2.7 |

The second row is within 0.3 cm of the first in every group (differences 0.2 / 0.1 / 0.1 cm), and in the
preceding window (112.5-123.5 s, where the offsets were fitted) it is 1.5 / 3.1 / 1.8 against 1.8 / 3.2 /
1.9. Tracking took 103 s with the optimised build of the worktree. The same figures come out of a run
with only the SOMA markers through the previous tracker binary. No wrist axis sits at a limit (at most 1%
of the states within 3% of one; twist 5th-95th percentile -35..55 degrees against the +-85 limit,
deviation -22..12 against +-25), and the elbows and knees move only about their flexion axis.

## C++ build

The C++ test suite passes in a debug build of the worktree and the optimised build tracks the reference
capture (meson with `-Ddefault_library=static` and the Pinocchio and Boost paths of `setup-windows.ps1`;
copy `boost_serialization.dll` and `yaml-cpp.dll` next to the tracker).
