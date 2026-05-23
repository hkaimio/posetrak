# Tracker Parallelization Plan

*Working doc — updated with measurements and results as work progresses.*

---

## Context and Motivation

The tracker processes real-time mocap at roughly one pose per camera frame.  For a
4-camera 60 fps session, the wall budget per frame is ~16 ms.  On a typical desktop
(Ryzen 9 / Core i9 class) the tracker currently runs well above that.  This doc records
the plan, measurements before and after each optimization attempt, and lessons learned.

---

## UKF Cost Structure

### Sigma points

For a full-body skeleton the error-state dimension is

```
error_state_dim = 2 × (root_error_dof + joint_active_dof)
                = 2 × (6 + joint_active_dof)
```

With a 23-active-DOF skeleton this is 58; with a fuller skeleton (e.g., hands + face)
it could reach 150+.  The number of sigma points is `n_sigma = 2 × error_state_dim + 1`.
For the standard body-only skeleton used in current tests, `n_sigma = 117`.

### Where time is spent

Each call to `UnscentedKalmanFilter::update()` does:

1. **Sigma generation** — Cholesky of the covariance matrix, then perturbation arithmetic.
   Cost: O(n³) for the Cholesky, dominated by matrix size.

2. **FK projection loop** (dominant) — For each of the `n_sigma` sigma points:
   - `pinocchio::forwardKinematics(model, data, q)` — full kinematic tree traversal
   - `pinocchio::updateFramePlacements(model, data)` — frame transform propagation
   - camera projection + distortion for every marker and camera

   These calls are currently serial.  Each FK call is independent of all others, making
   this loop the primary parallelization target.

3. **Weighted mean + covariance** — O(n_sigma × measurement_dim²) matrix algebra.
   Eigen already uses SIMD; the matrices are small enough that threading overhead
   would dominate.  Leave single-threaded.

4. **Kalman gain + state update** — O(n³) for matrix inversion.  Same reasoning as
   above; leave single-threaded.

The predict step also loops over sigma points (for the constant-velocity process model),
but the process model is pure arithmetic with no FK — it is fast and not a bottleneck.

---

## Parallelization Strategy

### Option A — OpenMP sigma point parallelization (primary plan)

Parallelize the FK projection loop in `update()` with `#pragma omp parallel for`.
Pinocchio's `pinocchio::Data` is **not thread-safe** (it stores intermediate results
`oMi`, `oMf`, joint Jacobians, etc. directly in the struct), but `pinocchio::Model`
is read-only and safe.

**Implementation**: maintain a `std::vector<pinocchio::Data>` pool, one per thread,
each constructed from the model (`pinocchio::Data(model)`).  Threads pick their own
copy via `omp_get_thread_num()`.

Relevant loops and their data dependencies:

| Loop | File / line | FK? | Can parallelize? |
|------|-------------|-----|-----------------|
| Predict: process model propagation (`sigma_points`) | `ukf.cpp:270` | No | Yes — trivial |
| Update: FK projection (`predicted_measurements`) | `ukf.cpp:819` | Yes | Yes — with Data pool |
| Update: inlier re-projection (after outlier rejection) | `ukf.cpp:932` | Yes | Yes — same pool |
| Update: weighted mean computation | `ukf.cpp:833` | No | Marginal gain |
| Update: cross-covariance Pxy | `ukf.cpp:993` | No | Marginal gain |
| RTS smoother: FK projection | `ukf.cpp:1722` | Yes | Yes — with Data pool |

The three FK loops (819, 932, 1722) are the valuable targets.

### Option B — Per-frame pipeline parallelism

If multiple persons are tracked simultaneously, each `Tracker` instance is independent.
This is already implicitly supported by having separate `Tracker` objects.  The
`posetrak track` CLI currently runs persons sequentially; a simple `std::async` or
thread-pool wrapper could run them concurrently.  Not useful for single-person sessions.

### Option C — Eigen threading (already available)

Eigen uses BLAS/LAPACK for large matrix ops.  With OpenBLAS linked, matrix multiply and
Cholesky automatically go multi-threaded for matrices above a threshold.  For the current
~58×58 matrices the single-threaded path is likely faster due to launch overhead.
**No action needed** — just ensure `OPENBLAS_NUM_THREADS` is not pinned to 1 in the
environment.

### Option D — Reduce sigma point count

Reduce `alpha` such that a smaller spread is acceptable, or use the minimal-sigma-point
formulation (spherical simplex, `n_sigma = n+2` instead of `2n+1`).  Spherical simplex
cuts sigma count from 117 to 60 for a 58-DOF system at the cost of higher-order error.
This is a correctness trade-off, not just an engineering one — evaluate only after
profiling shows the sigma count itself (not the per-point cost) is the bottleneck.

---

## Implementation Plan

### Step 0 — Establish baseline measurements

Before any changes, measure wall time per `track_frame()` on a representative sequence:

- Instrument `Tracker::track_frame()` with `std::chrono::steady_clock` timers around
  `ukf_.predict()`, `ukf_.update()`, and the full call.
- Run `optbuild/cli/posetrak track` on a 30-second clip; report mean, p50, p95, p99.
- Record hardware: CPU model, core count, frequency, OS.
- Output goes to `tracking_stats.csv` (add timing columns) or a separate timing file.

### Step 1 — Enable OpenMP in meson.build

```meson
openmp_dep = dependency('openmp', required: true)
# Add openmp_dep to tracker library link_with / dependencies
```

The line is already present but commented out (`meson.build:106`).  Verify it links
correctly with GCC/Clang on WSL2 (`-fopenmp`).

### Step 2 — Data pool in UnscentedKalmanFilter

Add to `UnscentedKalmanFilter` (private member):

```cpp
// One pinocchio::Data per OMP thread; initialized lazily from the model.
mutable std::vector<pinocchio::Data> data_pool_;
```

Initialization helper (called in the update body or on first use):

```cpp
void ensure_data_pool(pinocchio::Model const& model, int n_threads) {
    if (static_cast<int>(data_pool_.size()) < n_threads) {
        data_pool_.clear();
        for (int t = 0; t < n_threads; ++t)
            data_pool_.emplace_back(model);
    }
}
```

### Step 3 — Parallelize the FK projection loop

Replace the serial update loop at `ukf.cpp:819`:

```cpp
// Before:
for (int i = 0; i < n_sigma; ++i) {
    predicted_measurements.col(i) =
        predict_measurements(sigma_points[i], observations, cameras, fk);
}

// After:
#pragma omp parallel for schedule(dynamic, 4)
for (int i = 0; i < n_sigma; ++i) {
    int const tid = omp_get_thread_num();
    ForwardKinematics fk_local(model_, data_pool_[tid], fk.marker_frame_map(), layout_);
    predicted_measurements.col(i) =
        predict_measurements(sigma_points[i], observations, cameras, fk_local);
}
```

Same pattern for the inlier re-projection loop at line 932.

`schedule(dynamic, 4)` is suggested because FK cost per sigma point varies slightly
with joint configuration; dynamic scheduling avoids idle threads at the tail.

### Step 4 — Measure after OpenMP

Same benchmark as Step 0.  Expected speedup: roughly `min(n_threads, n_sigma / overhead)`.
For 8 physical cores and 117 sigma points, ideal is ~8×; realistic with scheduling and
memory bandwidth contention is 3–5×.

### Step 5 (optional) — Parallelize predict loop

The predict step's process-model loop has no FK; parallelizing it is simple but likely
yields only a small win since it's already fast.  Measure before implementing.

### Step 6 (optional) — Profile remaining bottlenecks

If FK parallelization does not reach the 16 ms budget, profile with `perf stat` or
`valgrind --tool=callgrind` to identify the next target.  Candidates:

- Cholesky in sigma generation (can use Eigen `LLT` with custom allocator)
- Covariance weighted sum (can use `Eigen::setFromTriplets` or blocked computation)
- Memory allocation inside the FK loop (pre-allocate `marker_positions` map)

---

## Measurements Log

*(Fill in as work progresses.  Each entry: date, code state/commit, hardware, result.)*

### Baseline

| Date | Commit | Hardware | mean ms/frame | p95 ms/frame | Notes |
|------|--------|----------|--------------|-------------|-------|
| — | — | — | — | — | Not yet measured |

### After OpenMP sigma parallelization

| Date | Commit | Hardware | mean ms/frame | p95 ms/frame | Speedup | Notes |
|------|--------|----------|--------------|-------------|---------|-------|
| — | — | — | — | — | — | — |

---

## Risks and Constraints

**Pinocchio Data pool correctness**: each `pinocchio::Data` must be fully independent.
The constructor `pinocchio::Data(model)` allocates and initializes all internal buffers;
do not share or copy a `Data` after it has been used by `forwardKinematics()`.

**WSL2 thread count**: `omp_get_max_threads()` on WSL2 sees all logical processors
including SMT siblings.  For the FK workload (compute-bound, moderate cache pressure)
binding to physical cores may outperform using all logical cores.  Test with
`OMP_NUM_THREADS=N` for N in {4, 8, 16} and pick empirically.

**Eigen + OpenMP interaction**: Eigen's internal parallelism (`EIGEN_DONT_PARALLELIZE`
not set) can conflict with outer OpenMP loops on some BLAS builds, causing thread
over-subscription.  Set `Eigen::setNbThreads(1)` inside the parallel region if
performance degrades unexpectedly.

**Numerical equivalence**: the parallel and serial implementations must produce
bit-identical results for the same input (column writes to `predicted_measurements` are
non-overlapping).  Add a regression test that runs both and compares outputs.
