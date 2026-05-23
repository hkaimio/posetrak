# Tracker Parallelization Plan

*Working doc — updated with measurements and results as work progresses.*

---

## Context and Motivation

The tracker currently runs at roughly 3–4 tracker steps per second on the full
whole-body+hands skeleton.  The goal is not strict real-time but to reduce per-step
wall time enough to make iterative workflow practical: shorter experiment turn-around,
the ability to re-track a 30-second clip in reasonable time, and headroom for future
skeleton or camera count growth.

**Benchmark dataset**: "Trial 1" in `ukemi-tommi-20260509.db`, `time_start_s=250`,
`time_end_s=252` (2-second clip, 5 cameras, full whole-body+hands skeleton).
This clip is short enough for rapid iteration but long enough to be representative.

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

### Where time is spent

Each `UnscentedKalmanFilter::update()` call does:

**1. Sigma generation** — Cholesky of the 318×318 covariance matrix.
Cost: O(n³) ≈ O(318³) ≈ 32M flops.  Runs once per step.

**2. FK + projection loop** — For each of the 637 sigma points:
- `pinocchio::forwardKinematics(model, data, q)` — 52-joint tree traversal
- `pinocchio::updateFramePlacements(model, data)` — frame transform propagation
- Project all 61 markers into all 5 cameras (with distortion)

  Each point is independent → primary parallelisation target.
  Total: 637 calls, each touching ~52 joints × 5 cameras × 61 markers.

**3. Weighted mean of measurements** — O(n\_sigma × measurement\_dim) = O(637 × 610) ≈ 390K ops.
  Fast.

**4. Innovation covariance S** — outer-product accumulation:
O(n\_sigma × measurement\_dim²) = O(637 × 610²) ≈ 237M flops.
S is 610×610; this is non-trivial even with SIMD.

**5. Cross-covariance Pxy** — O(n\_sigma × state\_dim × measurement\_dim)
= O(637 × 318 × 610) ≈ 123M flops.  Pxy is 318×610.

**6. Kalman gain K = Pxy × S⁻¹** — S inversion is O(610³) ≈ 227M flops.
Then K × innovation is O(318 × 610) ≈ 194K ops.

Steps 4–6 are pure Eigen matrix algebra on matrices too large to dismiss but
too small for multi-threaded BLAS to be efficient (thread-launch overhead dominates
for matrices under ~500×500 in practice).  They will be addressed after step 2 is
profiled.

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

Same benchmark as Step 0.  If update time is now dominated by S computation (steps 4–6
in the cost breakdown), proceed to Option B.  If the FK loop is still dominant,
increase thread count or investigate per-FK cache behaviour.

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

| Date | Commit | Hardware | mean ms/step | p95 ms/step | predict ms | update ms | Notes |
|------|--------|----------|-------------|-------------|-----------|----------|-------|
| — | — | — | — | — | — | — | Not yet measured |

### After OpenMP FK parallelisation

| Date | Commit | Hardware | OMP_THREADS | mean ms/step | p95 ms/step | Speedup | Notes |
|------|--------|----------|-------------|-------------|-------------|---------|-------|
| — | — | — | — | — | — | — | — |

### After S matrix optimisation (if pursued)

| Date | Commit | Hardware | mean ms/step | p95 ms/step | Speedup vs serial | Notes |
|------|--------|----------|-------------|-------------|------------------|-------|
| — | — | — | — | — | — | — |

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
