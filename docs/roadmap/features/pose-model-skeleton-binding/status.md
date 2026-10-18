```toml
name = "Pose Model and Skeleton Binding"
status = "in-progress"
progress_pct = 85
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
| Manifest per observation source: C++ reader and its tests | Written and compiled for syntax only; **not built or run** (see below) |
| Model-qualified landmarks | Done (reader accepts bare prototype names) |
| SOMA offsets catalog file (`catalog/pose-models/soma77.marker-offsets.yaml`) | Done |
| Default skeletons with both marker sets | Done (new ids) |
| Joint-limit table and hinge test | Done |
| Skeleton layer files, `skeleton compose` (also adds a pose model to a skeleton) | Done |
| Run fails naming the missing pose model | Done (Python before the tracker starts; C++ reader as a second line, unbuilt) |
| Migration check on a copy of a real session database | Done: see below |
| Held-out check on the reference capture | Partly run: SOMA markers and limits through the previous tracker binary give 2.2 / 3.9 / 2.7 cm (see below); the regenerated default with the new reader **not run** |
| Wrist saturation | Cause examined (design section 2.6); with the new table no wrist axis saturates in the calibration clip (see below) |
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
| New SOMA marker set (33 markers from the catalog offsets) and new limits | 2.2 | 3.9 | 2.7 |

The second row is within 0.3 cm of the first in every group (differences 0.2 / 0.1 / 0.1 cm), and in the
preceding window (112.5-123.5 s, where the offsets were fitted) it is 1.5 / 3.1 / 1.8 against 1.8 / 3.2 /
1.9. It was tracked in 104 s with the previous tracker binary, on a skeleton composed from the new layers
with only the SOMA markers and bare landmarks (that reader needs exact landmark names and would bind the
COCO markers of the full default to the wrong SOMA slots). In that run no wrist axis sits at a limit
(at most 1% of the states within 3% of one; twist 5th-95th percentile -35..55 degrees against the
+-85 limit, deviation -22..12 against +-25), and the elbows and knees move only about their flexion axis.

What is **not** verified: the regenerated default skeleton (COCO, hand21 and SOMA markers together) read
by the new C++ reader.

## C++ build

On the machine used for this stage no new executable could be linked or started during the session (the
linker could not write its output, and a freshly linked test program failed to start with Windows error
1450, "insufficient system resources"; the `System` process held over 200 000 handles). The C++ reader
(`cpp/src/db/session_reader.cpp`) and its tests (`cpp/tests/test_session_reader.cpp`) were checked with
`cl /Zs` only. Before merging:

1. build and run the C++ tests in a fresh build directory of the worktree;
2. build `optbuild` and run `scripts/skeleton_heldout_check.py` on a copy of the reference capture's
   session database (migrated to the current schema): it tracks the SOMA-77 sequence of the calibration
   clip with the default male skeleton and prints leg / arm / head error against the reference tracking
   run next to the prototype run, plus how often the wrist, elbow and knee sit at a limit. The
   tracking takes under two minutes.
