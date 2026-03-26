# UKF Parameter Sweep

Guide for running the parameter sweep script to tune UKF noise parameters.

## Script

`python/tools/param_sweep.py`

## Usage

```bash
uv run python/tools/param_sweep.py \
    --session-db /mnt/d/mocap/<session>/session.db \
    --config     <base-tracker-config-id-or-prefix> \
    [--sequence  <pose-observation-sequence-id>] \
    [--skeleton  <skeleton-id>] \
    [--out-dir   /tmp/posetrak_sweep]
```

`--sequence` and `--skeleton` are optional — if omitted the script resolves them
from the most recent tracking run that used the given base config.

`--config` accepts a UUID prefix (first few characters are enough).

## What it does

1. Resolves the base `tracker_config` row and the sequence/skeleton to use.
2. Iterates over all combinations in `SWEEP_GRID` (Cartesian product).
3. For each combination: creates a child `tracker_config` row via `edit_config()`,
   then invokes `optbuild/cli/posetrak track --session-db ...` with a fixed time
   window (`TIME_RANGE`, default `0.0–10.0 s`).
4. Queries `tracking_results` in the session DB for NIS, condition number, inlier
   count, and tracking-lost rate.
5. Ranks by score = `|NIS/dof − 1| + 0.5*(lost%) + log10_cond_penalty`.
6. Prints the top-15 ranked configs and saves the full table to
   `<out-dir>/sweep_summary.csv`.

All child configs and tracking runs are stored in the session DB, so results can
be inspected with the tracker_debug Marimo app using their `run_id`.

## Editing the grid

Edit `SWEEP_GRID` and `FIXED_PARAMS` at the top of the script:

```python
SWEEP_GRID: dict[str, list] = {
    "process_noise_std":     [0.05, 0.1, 0.2],
    "process_noise_vel_std": [0.2, 0.5, 1.0],
    "velocity_half_life_s":  [0.25, 0.5, 1.0],
}

FIXED_PARAMS: dict[str, float | int | None] = {
    "measurement_noise_std": 60.0,
    "outlier_threshold":     4.0,
}

TIME_RANGE = (0.0, 10.0)
```

## Score interpretation

**Lower score is better.** A perfectly consistent filter (NIS/dof = 1.0, no lost
frames, low condition number) scores ≈ 0.

- `|NIS/dof − 1|`: filter consistency. NIS/dof > 1 → overconfident (noise
  underestimated). NIS/dof < 1 → underconfident.
- `0.5 * (tracking_lost_pct / 100)`: penalty for lost frames.
- `log10_cond_penalty`: penalty when covariance condition number exceeds 10⁶
  (`max(0, (log10(cond_p95) - 6) * 0.05)`).

## Results — 2026-03-26, teacup session

**Session:** `20260322-teacup-exc2`
**Fixed:** `measurement_noise_std=60`, `outlier_threshold=4`
**Grid:** 27 combinations (3×3×3), time window 0–10 s

| process_noise_std | process_noise_vel_std | velocity_half_life_s | NIS/dof | cond_p95 | avg_inliers | lost% | score |
|------------------:|----------------------:|---------------------:|--------:|---------:|------------:|------:|------:|
| 0.1 | 0.5 | 0.25 | 1.43 | 3.1e6 | 197 | 0 | **0.457** |
| 0.1 | 0.2 | 0.25 | 1.42 | 6.1e6 | 219 | 0 | 0.459 |
| 0.1 | 0.2 | 1.0  | 1.41 | 3.4e7 | 219 | 0 | 0.491 |
| 0.1 | 1.0 | 0.25 | 1.49 | 6.7e6 | 222 | 0 | 0.527 |
| 0.2 | 1.0 | 0.25 | 1.59 | 1.9e6 | 188 | 0 | 0.600 |
| 0.2 | 0.2 | 0.25 | 1.55 | 1.2e7 | 157 | 0 | 0.602 |
| 0.05 | 1.0 | 0.25 | 1.56 | 9.4e6 | 215 | 0 | 0.608 |

**Recommended config:** `process_noise_std=0.1`, `process_noise_vel_std=0.5`,
`velocity_half_life_s=0.25`, `measurement_noise_std=60`, `outlier_threshold=4`.

### Observations

- `velocity_half_life_s=0.25` consistently outperforms longer half-lives — shorter
  damping keeps condition numbers 5–10× lower (3–6×10⁶ vs 3×10⁷+).
- `process_noise_std=0.1` is the sweet spot; 0.05 and 0.2 both score worse.
- NIS/dof is 1.4–1.7 across **all** parameter combinations, suggesting a
  systematic filter overconfidence. Likely cause: `measurement_noise_std=60` is
  still slightly too small for this session's actual reprojection noise.
  A follow-up sweep over `measurement_noise_std` (80, 100, 120) is warranted.
- 0% tracking lost on the 10 s window for all 27 combinations.
