# Task: stage 3 acceptance measurements (operator)

Stage 3 of [soma-gemx-productization-plan.md](soma-gemx-productization-plan.md) (the detection pipeline,
branch `feature/soma-gemx-stage3`, worktree `D:\mocap\wt-stage3`) is built and unit-tested. The measurements that
decide its acceptance need a quiet machine and run longer than an agent's background task survives, so they are
the operator's. Everything below runs in a shell (PowerShell or Git Bash), one job at a time, nothing else using the
GPU. Results go into the tables of [detection pipeline §3](../detection-pipeline/detection-pipeline-design.md)
(rows "pending") and the answers into its §5 status.

## Set-up (once)

1. Make a copy of the work database for the new schema. Opening a database with the stage 3 code migrates it to
   session version 66, which the spike branch cannot open, so never point the stage 3 code at the spike's work copy:

   ```
   python D:\mocap\stage3-work\backup.py D:\mocap\vanhaa-spike\session.db D:\mocap\stage3-work\profile-session.db
   ```
   Expected: 10-20 minutes (27 GB).

2. In every shell used below:

   ```
   $env:PYTHONPATH = "D:\mocap\wt-stage3\python"
   $env:PYTHONUTF8 = "1"
   $env:POSETRAK_SOMA77_MODEL = "D:\mocap\models\gem-x\onnx\vitpose.onnx"
   $S = "D:\mocap\stage3-work\profile-session.db"
   $PY = "D:\mocap\posetrak\.venv\Scripts\python.exe"
   function posetrak { & $PY -c "from posetrak.cli.main import main; main()" @args }
   ```
   The profiling clip is capture `b862ca88`, sync config `6680ef53`, 46-56 s, the edited segmentation run `e862c9af`,
   the mask-sourced RTMPose run `b05e51a9` as the run that supplies boxes.

3. GPU sampler, started before and stopped after each job (`Ctrl-C` or `Stop-Process -Name nvidia-smi`):

   ```
   nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,clocks.sm --format=csv,noheader -l 1 > gpu-<job>.csv
   ```
   Mean utilisation and peak memory of a job:
   ```
   python -c "import sys,csv; r=[l.split(',') for l in open(sys.argv[1])]; u=[float(x[1].split()[0]) for x in r]; m=[float(x[2].split()[0]) for x in r]; print('mean GPU %', sum(u)/len(u), ' peak MiB', max(m), ' idle MiB', min(m))" gpu-<job>.csv
   ```
   The peak above idle is the GPU memory figure of the job (ViT-H batches next to the decode threads and YOLOX).

## Jobs

Set `$env:POSETRAK_PROFILE_DETECTION = "D:\mocap\stage3-work\prof-<job>.json"` before each job; the stage table
is printed at its end and the JSON has the numbers. The expected run times come from a 13-21 s smoke test (SOMA
about 24 ms per person, decode on four threads) and are estimates; the 3-person clip has 6,000 camera-frames.

| Job | Command | Expected |
|---|---|---|
| J1 YOLOX + SOMA-77 | `posetrak --session $S detect run --capture b862ca88 --sync 6680ef53 --start 46 --end 56 --pose-model soma77-vitpose-h --accept-licences` | 8-12 min |
| J2 Masks + SOMA-77 | `... --pose-model soma77-vitpose-h --segmentation e862c9af` | 7-9 min |
| J3 Run boxes + SOMA-77 | `... --pose-model soma77-vitpose-h --from-run b05e51a9` | 7-9 min |
| J4 YOLOX + RTMPose-133 | `... --pose-model rtmpose-l-133kp` | 4-7 min |
| J5 Masks + RTMPose-133, no hand pass | `... --pose-model rtmpose-l-133kp --segmentation e862c9af --no-refine-hands` | 4-6 min |
| J6 Masks + RTMPose-133, hand pass | `... --pose-model rtmpose-l-133kp --segmentation e862c9af --refine-hands` | 8-12 min |

(`...` is `posetrak --session $S detect run --capture b862ca88 --sync 6680ef53 --start 46 --end 56`.) With
`--accept-licences` a first run downloads nothing here because the file is supplied by `POSETRAK_SOMA77_MODEL`;
drop the flag if you prefer to be asked.

J1-J3 are the acceptance routes. J4 and J5 together answer the RTMPose cost gap (design §3, "Unexplained, now
testable"): compare the `inference` milliseconds per crop of J4 and J5, and the `clocks.sm` column of the two GPU
logs. J6 is the hand pass in the loop (the earlier route needed 918 s).

## Acceptance checks

1. **GPU at least 85% busy**: the mean utilisation of J2 (and J3). If it is lower, the stage table shows which
   stage bounds it; the criterion also holds if that stage is at its decode floor (decode of the slowest camera
   alone is 37 s for the clip).
2. **Twice the prototype's throughput**: the prototype route for the three persons is the source pass (437 s for
   masks + RTMPose, design §3) plus `soma77_extract.py` once per person (339 s each): 1,454 s. J2's wall time
   must be at most half of that, 727 s. To re-measure the prototype on this machine instead, run from the spike
   checkout `D:\mocap\posetrak` against the spike's work copy (not `$S`): `profile_detection_routes.py soma
   SESSION SOURCE_RUN MODEL 46 56 --mask-run e862c9af --mask-label N` for N = 1, 2, 3; about 6 x 339 s.
3. **Parity with the prototype**: already checked on a copy of a real session on the same boxes (maximum
   difference 0.0 px over 10 keypoint sets, CPU); to repeat on the clip, run J3 and
   `soma77_extract.py $S b05e51a9 MODEL 46 46.2 --track-id N` for one person and compare the keypoints of the
   two new runs per (video, frame).

## Decode soak (long)

```
& $PY D:\mocap\wt-stage3\python\tools\prototypes\decode_soak.py --mode both --minutes 30 `
   --session $S --capture b862ca88 > soak.jsonl
```
Expected: 60 minutes plus start-up; the machine may be used lightly. One JSON line per mode: windows, frames
per second, hangs per step (open / decode / close), recovered stalls, leaked threads. Put both lines in the soak
table of design §3. The default is threads; if the thread line shows any hang or leaked thread, change
`DEFAULT_DECODE_MODE` in `posetrak/detection/pipeline.py` to `"process"` and say so in §3.

## Report

For each job the wall time, the stage table, mean GPU utilisation and peak memory above idle. The answers to the
three acceptance checks, the RTMPose gap comparison, and the soak lines. Update the "pending" rows of design §3
and the status document.
