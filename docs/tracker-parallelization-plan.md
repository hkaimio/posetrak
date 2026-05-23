# Tracker Parallelization Plan

*Working doc — updated with measurements and results as work progresses.*

---

## Context and Motivation

The tracker currently runs at roughly 3–4 tracker steps per second on the full
whole-body+hands skeleton.  The goal is not strict real-time but to reduce per-step
wall time enough to make iterative workflow practical: shorter experiment turn-around,
the ability to re-track a 30-second clip in reasonable time, and headroom for future
skeleton or camera count growth.

**Benchmark dataset**: "Trial 1" in capture "bokken 20260518",
database `/home/harri/projects/mocap_videos/ukemi-tommi-20260509.db`,
sequence `2ec9c1a3-85ae-40fa-bd63-20c454902f46` (`time_start_s=250`, `time_end_s=252`),
skeleton `bcffc4b0cf41dce1e372817aa4bc567bac8ae1dec19f172415d1a0fe3dfbba69` ("Harri scaling 1 20260523"),
tracker config `438acd86-c413-492e-94a8-24555515044d` ("ui-run").
2-second clip, 5 cameras, full whole-body+hands skeleton.
This clip is short enough for rapid iteration but long enough to be representative.

**Benchmark command** (run from repo root with `optbuild`):

```bash
DB=/home/harri/projects/mocap_videos/ukemi-tommi-20260509.db
SEQ=2ec9c1a3-85ae-40fa-bd63-20c454902f46
SKEL=bcffc4b0cf41dce1e372817aa4bc567bac8ae1dec19f172415d1a0fe3dfbba69
CFG=438acd86-c413-492e-94a8-24555515044d
optbuild/cli/posetrak track \
  --session-db "$DB" \
  --sequence "$SEQ" \
  --skeleton "$SKEL" \
  --tracker-config "$CFG" \
  --start-time 250.0 \
  --end-time 252.0 \
  --output-dir /tmp/bench_timing
```

---

## Actual Problem Dimensions

The production skeleton (`Harri scaling 1 20260523`, SHA prefix `bcffc4b`) has:

| Quantity | Value |
|---|---|
| Ball joints | 51 |
| Active joint DOF | 153 (51 × 3) |
| Error-state dim | 318 — `2 × (6 root + 153 joint)` |
| **n\_sigma** | **637** — `2 × 318 + 1` |
| Cameras (current runs) | 5 |
| Tracked markers | 61 |
| **measurement\_dim** | **610** — `5 cams × 61 markers × 2` |

These are substantially larger than the body-only estimates in earlier planning.
637 sigma points each requiring a full 52-joint FK call (plus 5-camera projection)
explains why the tracker sits at 3–4 steps/s today.

---

## UKF Cost Structure

### Where time is spent — analytical estimates only

The following estimates are based on FLOP counts and code inspection.
**They are not measured.** Profiling is mandatory before starting optimisation
(see Implementation Plan Step 0).

Each `UnscentedKalmanFilter::update()` call does:

**1. Sigma generation** — Cholesky of the 318×318 covariance matrix.
Cost: O(n³/3) ≈ O(318³/3) ≈ 11M flops.  Runs once per step.

**2. FK + projection loop** — For each of the 637 sigma points:
- `pinocchio::forwardKinematics(model, data, q)` — 52-joint tree traversal
- `pinocchio::updateFramePlacements(model, data)` — frame transform propagation
- Project all 61 markers into all 5 cameras (with distortion)

  Each point is independent → primary parallelisation target.
  Raw FLOP count is modest (~10K per point, ~6M total), but FK is
  **latency-bound, not compute-bound**: the kinematic tree is a pointer-chasing
  traversal with sequential data dependencies between parent and child joints.
  Neither SIMD nor out-of-order execution helps much.  637 serial calls of
  non-vectorisable code is why this dominates despite low FLOP count.

**3. Weighted mean of measurements** — O(n\_sigma × measurement\_dim) ≈ 390K ops.  Fast.

**4. Innovation covariance S accumulation** — 637 iterations of:
```cpp
innovation_cov += weights_cov(i) * (innovation * innovation.transpose());
```
Each outer product is 610×610 ≈ 372K elements.
Total: 637 × 372K ≈ **237M element-wise operations**.
The 610×610 matrix is large enough that multi-threaded BLAS *could* help, but
the current implementation issues sequential rank-1 `+=` updates — Eigen never
sees a large enough single operation to dispatch to parallel BLAS.
Restructuring to a single batched `DSYRK` call (compute all innovations as a
matrix, then call `selfadjointView().rankUpdate(Z, w)`) would let BLAS
parallelize the whole accumulation in one shot.

**5. Cross-covariance Pxy** — same pattern, O(637 × 318 × 610) ≈ **123M ops**.
Same missed-BLAS issue as S.

**6. Kalman gain K = Pxy × S⁻¹** — currently uses `.inverse()` on the 610×610 S
matrix, which is O(610³) ≈ **226M flops** via full LU decomposition.
Eigen's `llt().solve()` (Cholesky, S is symmetric positive definite) would be
2× faster and numerically better.  The subsequent matrix multiply Pxy × S⁻¹ is
O(318 × 610²) ≈ 118M flops — this IS a standard DGEMM that Eigen parallelises
via BLAS.

**Summary of analytical estimates:**

| Step | Estimated cost | Parallelisable as-is? |
|------|---------------|----------------------|
| 1. Sigma generation (Cholesky 318×318) | ~11M flops | No (sequential) |
| 2. FK + projection loop (637 points) | ~6M flops, latency-bound | Yes — OpenMP Data pool |
| 3. Weighted measurement mean | ~390K ops | Trivial; not worth it |
| 4. S accumulation (637 rank-1 updates to 610×610) | ~237M ops | Needs DSYRK refactor |
| 5. Pxy accumulation (637 outer products 318×610) | ~123M ops | Needs DGEMM refactor |
| 6. Kalman gain (610×610 inverse + 318×610² multiply) | ~344M flops | Partial (DGEMM is parallel) |

Steps 4–6 together exceed step 2 in raw FLOP count by ~10×.  Whether they exceed it
in wall time depends on vectorisation and BLAS efficiency — which is why profiling
comes first.

### How camera count scales

Adding a camera increases `measurement_dim` by `n_markers × 2`:

| Cost component | Scaling with cameras (C) |
|---|---|
| FK + projection | O(C) — one more projection per sigma point |
| Weighted mean | O(C) |
| Innovation covariance S | O(C²) — S is (C×M)×(C×M) |
| S inversion | O(C³) |
| Cross-covariance Pxy | O(C) |

Going from 4 → 5 cameras is a 25% increase in FK/projection work but a 56% increase
in S computation and 95% increase in S inversion.  At 5 cameras and 61 markers the
610×610 S matrix is already the second most expensive operation after the FK loop.
If camera count grows further, S and the Kalman gain will become the bottleneck.

---

## Parallelization and Optimization Strategy

### Option A — OpenMP sigma point parallelization (primary plan)

Parallelize the FK projection loop in `update()` with `#pragma omp parallel for`.
`pinocchio::Data` is **not thread-safe** (it stores intermediate results `oMi`, `oMf`,
etc. in-place), but `pinocchio::Model` is read-only and safe to share.

**Implementation**: a `std::vector<pinocchio::Data>` pool, one per thread, each
constructed with `pinocchio::Data(model)`.  Threads index by `omp_get_thread_num()`.

Relevant loops in `ukf.cpp`:

| Loop | Line | FK? | Parallelize? |
|------|------|-----|--------------|
| Predict: process model propagation | 270 | No | Easy, small gain |
| Update: FK projection (first pass) | 819 | Yes | **Primary target** |
| Update: inlier re-projection | 932 | Yes | Same pool |
| Update: weighted mean | 833 | No | Marginal |
| Update: cross-covariance Pxy | 993 | No | Marginal |
| RTS smoother: FK projection | 1722 | Yes | Same pool |

With 637 sigma points and 8–16 physical cores the FK loop alone should yield
5–10× speedup (realistic: 4–8× after overhead and memory contention).

### Option B — Reduce S matrix cost (secondary plan)

Once the FK loop is parallelised, S and its inversion may become the new bottleneck.
Options:
- **Woodbury / reduced-rank update**: if most markers are outliers per frame, do the
  Kalman gain computation only in the inlier subspace (already partially done by the
  outlier rejection path at line 932, but the first S computation still uses full dim).
- **Structured S**: diagonal + low-rank approximation.  Requires a model assumption
  that per-marker noise is independent (already assumed in the current diagonal R).
- **Blocked Cholesky on S**: split S into camera-sized blocks and solve in parallel.
  Camera blocks are independent except for the cross-camera correlations introduced by
  shared marker visibility, which are small and can be ignored for a Cholesky.

### Option C — Spherical simplex sigma points (sigma count reduction)

The standard formulation uses `n_sigma = 2n+1 = 637`.  The spherical-simplex
formulation needs only `n+2 = 320`, halving FK work at the cost of higher-order
approximation error.  This is a correctness trade-off — evaluate only after the FK
loop is parallelised and profiled.

### Option D — Per-person parallelism

Each `Tracker` instance is independent.  For multi-person sessions, a thread pool
wrapping the CLI would give free parallelism.  Not useful for single-person sessions
(which is the current use case).

---

## MSVC OpenMP Compatibility

Native Windows builds use MSVC.  Understanding its OpenMP support up front avoids
writing code that has to be rewritten for Windows.

### What MSVC supports

| Feature | OpenMP version | MSVC `/openmp` | MSVC `/openmp:llvm` |
|---------|----------------|----------------|----------------------|
| `parallel for` | 2.0 | ✓ | ✓ |
| `schedule(static/dynamic/guided)` | 2.0 | ✓ | ✓ |
| `omp_get_thread_num()` | 2.0 | ✓ | ✓ |
| `omp_get_max_threads()` | 2.0 | ✓ | ✓ |
| `reduction` on built-in types | 2.0 | ✓ | ✓ |
| `critical`, `atomic` | 2.0 | ✓ | ✓ |
| Loop variable must be signed `int` | 2.0 restriction | required | required |
| `task`, `taskwait` | 3.0 | ✗ | ✓ |
| `collapse(N)` | 3.0 | ✗ | ✓ |
| `schedule(auto)` | 3.0 | ✗ | ✓ |
| `#pragma omp simd` | 4.0 | ✗ | ✓ |

`/openmp:llvm` requires VS 2019 16.9+ and links the LLVM OpenMP runtime instead of
Microsoft's.  It is production-quality but adds a runtime DLL dependency.

### Impact on our plan

The FK loop parallelisation (Step 3) uses only OpenMP 2.0 features and is fully
MSVC-compatible as written:

```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < n_sigma; ++i) {   // signed int: required by MSVC 2.0
    ForwardKinematics fk_local(...);
    predicted_measurements.col(i) = predict_measurements(..., fk_local);
}
```

The only code-level constraint to maintain for MSVC: keep loop variables as `int`
(not `size_t` or `auto`), and avoid `collapse`, `task`, and `schedule(auto)`.
None of the plans in this document require those features.

For the DSYRK/DGEMM refactors (steps 4–6), parallelism comes from BLAS, not OpenMP
pragmas — those are MSVC-compatible regardless of OpenMP version.

---

## Implementation Plan

### Step 0 — Establish baseline measurements

Instrument `Tracker::track_frame()` with `std::chrono::steady_clock` around:
- `ukf_.predict()` — predict step only
- `ukf_.update()` — update step only
- full `track_frame()` — including any overhead

Run on the benchmark clip (Trial 1, 2 s) with `optbuild/cli/posetrak track`.
Report: mean, p50, p95 ms/step.  Also report `OMP_NUM_THREADS` and CPU info.

Output: add `predict_ms`, `update_ms`, `total_ms` columns to `tracking_stats.csv`
or a companion `timing.csv`.

### Step 1 — Enable OpenMP in meson.build

```meson
openmp_dep = dependency('openmp', required: true)
```

Already commented out at `meson.build:106` — uncomment and wire into the tracker
library's `dependencies`.  Confirm `-fopenmp` appears in the compile/link commands.

### Step 2 — Data pool in UnscentedKalmanFilter

Add private member:

```cpp
mutable std::vector<pinocchio::Data> data_pool_;
```

Initialization helper (call before each parallel region):

```cpp
void ensure_data_pool(pinocchio::Model const& model) {
    int const n = omp_get_max_threads();
    if (static_cast<int>(data_pool_.size()) < n) {
        data_pool_.clear();
        for (int t = 0; t < n; ++t)
            data_pool_.emplace_back(model);
    }
}
```

### Step 3 — Parallelize the FK projection loop

Replace the serial loop at `ukf.cpp:819`:

```cpp
// Before:
for (int i = 0; i < n_sigma; ++i) {
    predicted_measurements.col(i) =
        predict_measurements(sigma_points[i], observations, cameras, fk);
}

// After:
ensure_data_pool(model_);
#pragma omp parallel for schedule(static)
for (int i = 0; i < n_sigma; ++i) {
    ForwardKinematics fk_local(model_, data_pool_[omp_get_thread_num()],
                               fk.marker_frame_map(), layout_);
    predicted_measurements.col(i) =
        predict_measurements(sigma_points[i], observations, cameras, fk_local);
}
```

Apply the same pattern to the inlier re-projection loop at line 932 and the RTS
smoother loop at line 1722.

Use `schedule(static)` first (evenly divides 637 points); switch to `dynamic` if
profiling shows significant load imbalance.

### Step 4 — Measure and decide next step

Same benchmark as Step 0.  With 637 sigma points, ideal speedup on 8 physical cores
is ~8×; realistic is 4–6× after memory bandwidth contention and thread overhead.
If profiling shows S/Pxy/Kalman gain together still dominate after FK parallelisation,
proceed to Option B (DSYRK/DGEMM refactor + Cholesky solve).  If the FK loop is
still dominant, investigate per-FK cache behaviour or increase thread count.

### Step 5 (optional) — Measure S matrix dominance

If S cost is the bottleneck after FK parallelisation, first instrument to confirm:
add a `steady_clock` timer around just the S accumulation loop.  Then choose between
the Woodbury / structured-S approaches in Option B.

---

## Environment: WSL2 vs Native Windows

**Short answer**: WSL2 is fine for this work.  Do not switch environments.

WSL2 (since Windows 11 / Win10 21H1) exposes all physical CPU cores and hardware
threads to the Linux kernel — `nproc` and `omp_get_max_threads()` see the full
hardware count, the same as a bare-metal Linux install.

The practical differences are:

| Factor | WSL2 | Native Windows |
|---|---|---|
| Thread count visible | All HW threads ✓ | All HW threads ✓ |
| Compute throughput | ~95–98% of native Linux | — |
| Memory bandwidth | Slight hypervisor overhead | — |
| L3 cache sharing | Shared with Windows processes | Shared with Windows processes |
| Build toolchain | GCC/Clang, full OpenMP support | MSVC (OpenMP limited), complex Pinocchio build |
| Profiling tools | `perf`, `callgrind` — both work | Intel VTune works natively |
| Scheduler preemption | Hyper-V can pause the VM briefly | No VM overhead |

The hypervisor overhead is real but small for CPU-bound workloads.  The bigger issue
is that porting the Meson + Pinocchio build to native Windows MSVC would be a
significant one-time investment for marginal throughput gain.

**Recommendation**: profile and optimise in WSL2.  If final measurements show a 5–10%
gap worth closing and the build can be done (e.g., with Clang-cl or LLVM on Windows),
re-benchmark natively at that point.  For now the measurement environment is WSL2,
and all log entries should note this.

---

## Measurements Log

*(Fill in as work progresses.  Each entry: date, commit, hardware, result.)*
*(Benchmark: Trial 1, `time_start_s=250`, `time_end_s=252`, `ukemi-tommi-20260509.db`.)*

### Baseline (serial, pre-OpenMP)

| Date | Commit | Hardware | mean ms/step | p95 ms/step | predict ms (mean/p95) | update ms (mean/p95) | Notes |
|------|--------|----------|-------------|-------------|----------------------|---------------------|-------|
| 2026-05-23 | 3e564ed | Ryzen 9 9900X (12c/24t), WSL2 | 298 | 315 | 55 / 58 | 243 / 259 | serial, no OpenMP; 238 frames, 5 cams, 61 markers, n_sigma=637 |

### After OpenMP FK parallelisation (Steps 1–3)

FK parallel loop only added no measurable speedup (update 246 ms vs 243 ms baseline).
FK is not the bottleneck; S and Pxy accumulation dominate.

| Date | Commit | Hardware | OMP_THREADS | mean ms/step | p95 ms/step | Speedup | Notes |
|------|--------|----------|-------------|-------------|-------------|---------|-------|
| 2026-05-23 | (unreleased) | Ryzen 9 9900X, WSL2 | 12 | 301 | 316 | 1.0× | FK parallel only; S/Pxy still sequential rank-1 loops |

### After S/Pxy DGEMM refactor (Option B)

S accumulation and Pxy replaced with batched DGEMM (Z matrix build + Eigen matrix multiply).
Eigen/OpenBLAS parallelizes the resulting (610×637)×(637×610) multiply across all 12 cores.

| Date | Commit | Hardware | OMP_THREADS | mean ms/step | p95 ms/step | predict (mean/p95) | update (mean/p95) | Speedup vs serial |
|------|--------|----------|-------------|-------------|-------------|-------------------|--------------------|------------------|
| 2026-05-23 | 720db4d | Ryzen 9 9900X (12c/24t), WSL2 | 12 | 152 | 164 | 57 / 61 ms | 95 / 104 ms | **2.0×** |

Note: the 95ms update figure was a coarse measurement. Fine-grained instrumentation
(added in commit 385ed5b) later revealed that `u_kalman_ms` alone was 133ms — three
`.inverse()` calls on the ~511×511 inlier innovation covariance dominated. The 95ms
estimate is believed to have been taken under lighter system load or with a different
CPU boost state; the instrumented baseline with the same code measures ~500ms update.

### After LDLT factorization + predict DGEMM refactor

Three O(n³) inversions of the ~511×511 inlier innovation covariance replaced with a
single LDLT factorization reused for Kalman gain, covariance update, and NIS.
`compute_state_covariance` and RTS cross-covariance rank-1 loops replaced with DGEMM
(same pattern as the earlier S/Pxy refactor).

Per-operation breakdown (238 frames, mean ms):

| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| predict total | 57.2 | 35.3 | 1.6× |
| &nbsp;&nbsp;p_mean_cov_ms | 26.0 | 15.6 | 1.7× |
| &nbsp;&nbsp;p_rts_ms | 24.5 | 12.6 | 1.9× |
| update total | 500 | 74.5 | 6.7× |
| &nbsp;&nbsp;u_inlier_ms | 33.9 | 16.4 | 2.1× |
| &nbsp;&nbsp;u_kalman_ms | 132.8 | 14.6 | 9.1× |
| &nbsp;&nbsp;u_cov_update_ms | 179.2 | 21.6 | 8.3× |

| Date | Commit | Hardware | OMP_THREADS | mean ms/step | p95 ms/step | predict (mean/p95) | update (mean/p95) | Speedup vs serial |
|------|--------|----------|-------------|-------------|-------------|-------------------|--------------------|------------------|
| 2026-05-23 | 92087e4 | Ryzen 9 9900X (12c/24t), WSL2 | 12 | 110 | 113 | 35 / 57 ms | 75 / 113 ms | **2.7×** |

Dominant remaining cost: `u_inlier_ms` (16ms) — Z_inlier build and S_inlier DGEMM.
Total throughput: ~9 steps/s (up from ~3.4 steps/s at serial baseline).

### Fine-grained timing baseline (post-LDLT)

Fine-grained per-operation timers (238 frames, skipping init frame):

| Operation | Mean ms | P50 ms | P95 ms | % of step |
|-----------|---------|--------|--------|-----------|
| predict total | 39.7 | 39.1 | 57.8 | 29% |
| &nbsp;&nbsp;p_sigma_gen_ms | 3.5 | 3.4 | 4.0 | 2.6% |
| &nbsp;&nbsp;p_propagate_ms | 2.6 | 2.6 | 3.0 | 1.9% |
| &nbsp;&nbsp;p_mean_cov_ms | 17.0 | 15.8 | 29.8 | 13% |
| &nbsp;&nbsp;p_rts_ms | 16.0 | 15.4 | 28.0 | 12% |
| update total | 95.7 | 91.4 | 149.3 | 71% |
| &nbsp;&nbsp;u_fk1_ms | 3.6 | 1.7 | 9.8 | 2.6% |
| &nbsp;&nbsp;u_s_ms | 18.6 | 17.3 | 36.2 | 14% |
| &nbsp;&nbsp;u_fk2_ms | 3.7 | 1.6 | 12.4 | 2.7% |
| &nbsp;&nbsp;u_inlier_ms | 24.6 | 23.6 | 47.1 | 18% |
| &nbsp;&nbsp;u_kalman_ms | 13.9 | 12.2 | 21.0 | 10% |
| &nbsp;&nbsp;u_cov_update_ms | 24.2 | 23.0 | 37.5 | 18% |
| **TOTAL step** | **135** | **131** | **198** | — |

CPU utilization: ~15% across 24 threads (Ryzen 9 9900X). Amdahl bottleneck: ~52ms is
parallel BLAS (u_s, u_inlier, p_propagate, FKs) and ~83ms is sequential. With BLAS
averaging ~8 threads during bursts: (52ms×8 + 83ms×1) / 135ms / 24 = ~16% — matches
observed utilization.

**Top three remaining bottlenecks:**

1. **u_cov_update_ms (24ms, 18%)** — `SelfAdjointEigenSolver` on the 318×318 posterior
   covariance runs every frame to clip negative eigenvalues (PSD fix). After convergence the
   covariance stays PD; the solver fires unconditionally. A Cholesky fast-path (LLT,
   O(n³/3) ≈ ~1ms vs ~24ms for eigensolver) can gate the expensive path so it only runs on
   failure. Expected savings: ~23ms on nearly all post-convergence frames.

2. **p_rts_ms (16ms, 12%) + p_mean_cov_ms (17ms, 13%)** — Both are sequential loops over
   637 sigma points computing `compute_state_error()`, followed by a DGEMM. The state-error
   function is a pure const operation (reads sigma points + reference state, writes to a
   distinct matrix column) — the same `#pragma omp parallel for` pattern used in the update's
   E-matrix build (line 923) applies directly. Parallelizing both loops saves ~29ms combined.

3. **Spherical simplex sigma points (deferred)** — Reducing n_sigma from 637 to 320 would
   halve u_fk1, u_s, and the predict loops. Left for separate analysis/experimentation as it
   changes filter numerics and requires validation.

### Numerical validation across optimization commits

To confirm the optimizations do not alter tracking results, three builds were run
against the same clip (pre-opt 385ed5b ukf.cpp, after S/Pxy DGEMM 720db4d, after
LDLT+predict DGEMM 92087e4):

| Metric | Pre-opt | Post-opt | Max diff |
|--------|---------|----------|----------|
| Mean inliers/frame | 255.4 | 255.4 | 0 |
| Mean outliers/frame | 49.3 | 49.3 | 0 |
| Mean reprojection error (px) | 17.597 | 17.597 | 0 |
| Body joint angles (max °) | — | — | < 0.02° |
| Distal finger joints (max °) | — | — | 3.4° |

Inlier/outlier counts and reprojection error are identical across all three versions.
Body joint differences (< 0.02°) are attributable to floating-point accumulation order
differences between DGEMM and sequential rank-1 loops — irrelevant for motion capture.
Distal finger joints (up to 3.4°, rms 0.28°) show similar order-dependent variation.

**Conclusion**: the optimizations are numerically equivalent for all practical purposes.

### After LLT-gated PSD fix (u_cov_update)

`SelfAdjointEigenSolver` replaced with `LLT` (Cholesky) as a fast PD check. The eigensolver
now only fires when LLT fails (covariance genuinely not PD). After convergence the covariance
stays PD so LLT succeeds on 236/239 frames; the eigensolver only fires on the first 3 frames
while the filter converges. `u_cov_update_ms` drops from ~24ms to ~7ms (70% reduction).

| Date | Commit | Hardware | mean ms/step (CSV) | p95 ms/step | u_cov_update (mean) | PSD fires | Speedup vs prev |
|------|--------|----------|--------------------|-------------|---------------------|-----------|-----------------|
| 2026-05-23 | — | Ryzen 9 9900X (12c/24t), WSL2 | 87 ms | — | 7.3 ms | 3/239 | **1.6×** |

### After parallel predict state-error loops (p_rts + p_mean_cov)

*(to be filled after implementation)*

| Date | Commit | Hardware | mean ms/step | p95 ms/step | p_rts (mean) | p_mean_cov (mean) | Speedup vs prev |
|------|--------|----------|-------------|-------------|--------------|-------------------|-----------------|
| — | — | — | — | — | — | — | — |

---

## Risks and Constraints

**Pinocchio Data pool correctness**: `pinocchio::Data(model)` allocates and
initialises all internal buffers independently.  Never copy a `Data` that has been
used by `forwardKinematics()` — the copy will contain stale intermediate state that
causes subtle numerical errors.

**Eigen + OpenMP interaction**: Eigen's internal BLAS parallelism can conflict with
outer OpenMP regions, causing thread over-subscription.  If throughput is worse than
expected with OpenMP enabled, add `Eigen::setNbThreads(1)` at the start of the
parallel region as a diagnostic step.

**ForwardKinematics construction inside the loop**: constructing `fk_local` per
iteration is cheap (it copies three pointers and builds a small `joint_id_map_`), but
if profiling shows this as significant, pre-allocate one `ForwardKinematics` per
thread outside the loop (as a `thread_local` or via the pool).

**Numerical equivalence**: column writes to `predicted_measurements` are non-overlapping
so results must be bit-identical to the serial version.  Add a regression test that
runs both paths on the same input and asserts zero difference.
