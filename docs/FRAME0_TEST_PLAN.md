# Frame 0 Single-Update Test Plan

## Objective
Create a focused test harness that replicates Python's first UKF update (frame 0) in C++ and verifies exact numerical equivalence at each step.

## Available Python Debug Data
Located in: `python_results/debug/frame_0000/`

### Input Data (Python state before update):
- `prior_state.json` - Initial state (root pos, quat, joint angles, velocities)
- `prior_covariance.csv` - Initial covariance matrix
- `all_observations.csv` - All 341 observations with marker names, camera IDs, pixel coords

### Intermediate Computation Data (for verification):
- `sigma_points.csv` - All 145 sigma points generated from prior state
- `predicted_observations.csv` - Predicted pixel coords for each observation × sigma point
- `innovation_covariance.csv` - S matrix after outlier rejection
- `kalman_gain.csv` - K matrix

### Output Data (Python state after update):
- `posterior_state.json` - Updated state after UKF update
- `posterior_covariance.csv` - Updated covariance matrix
- Outlier info in `all_observations.csv` (is_outlier, outlier_reason, was_used_in_update)

## Test Implementation Strategy

### Option A: Integration Test (Recommended)
**File**: `tests/test_ukf_frame0_python_comparison.cpp`

**Advantages**:
- Part of test suite, runs on every build
- Uses Catch2 for assertions and clear failure messages
- Easy to add multiple test cases (e.g., frame 1, frame 2, etc.)
- Can use `REQUIRE_THAT` with custom matchers for floating point comparison

**Structure**:
```cpp
TEST_CASE("UKF Frame 0 matches Python", "[ukf][integration]") {
    SECTION("Load Python data") { ... }
    SECTION("Verify initial state") { ... }
    SECTION("Generate sigma points") { ... }
    SECTION("Predict measurements") { ... }
    SECTION("Compute innovation covariance") { ... }
    SECTION("Reject outliers") { ... }
    SECTION("Compute Kalman gain") { ... }
    SECTION("Update state") { ... }
    SECTION("Verify posterior state") { ... }
}
```

### Option B: Standalone CLI Tool
**File**: `cli/test_frame0.cpp`

**Advantages**:
- Can print detailed debug output
- Easier to run manually with custom arguments
- Can export comparison CSVs

**Structure**:
```cpp
int main() {
    // Load Python data
    // Run C++ UKF update
    // Compare at each step
    // Print detailed report
}
```

### Recommendation: **Start with Option A (Integration Test)**
Easier to maintain, better for CI/CD, clearer pass/fail criteria.

## Test Implementation Steps

### Step 1: Create Data Loading Utilities
**File**: `tests/test_helpers/python_data_loader.hpp`

```cpp
namespace test_helpers {

struct PythonFrame0Data {
    State prior_state;
    Eigen::MatrixXd prior_covariance;
    std::vector<Observation> observations;
    Eigen::MatrixXd sigma_points;  // For verification
    Eigen::MatrixXd predicted_obs;  // For verification
    Eigen::MatrixXd innovation_cov; // For verification
    Eigen::MatrixXd kalman_gain;    // For verification
    State posterior_state;
    Eigen::MatrixXd posterior_covariance;
    std::vector<bool> outlier_flags; // True if observation was outlier
};

PythonFrame0Data load_python_frame0_data(std::string const& debug_dir);

}
```

Functions needed:
- `load_state_from_json()` - Parse prior_state.json, posterior_state.json
- `load_covariance_from_csv()` - Parse CSV matrices
- `load_observations_from_csv()` - Parse all_observations.csv
- `load_sigma_points_from_csv()` - Parse sigma_points.csv

### Step 2: Create Comparison Utilities
**File**: `tests/test_helpers/matrix_comparison.hpp`

```cpp
namespace test_helpers {

// Compare matrices with tolerance
bool matrices_equal(Eigen::MatrixXd const& a, Eigen::MatrixXd const& b,
                   double tol = 1e-10);

// Find first differing element
void print_matrix_diff(Eigen::MatrixXd const& cpp, Eigen::MatrixXd const& python,
                      std::string const& name);

// Custom Catch2 matcher for matrices
class MatrixEquals : public Catch::MatcherBase<Eigen::MatrixXd> {
    // ...
};

}
```

### Step 3: Implement Main Test
**File**: `tests/test_ukf_frame0_python_comparison.cpp`

```cpp
TEST_CASE("UKF Frame 0 matches Python exactly", "[ukf][frame0]") {
    // Path to Python debug data
    std::string debug_dir = "tracking_tests/cpp-python-comparison/python_results/debug/frame_0000";

    SECTION("1. Load and verify Python data") {
        auto python_data = test_helpers::load_python_frame0_data(debug_dir);

        REQUIRE(python_data.observations.size() == 341);
        REQUIRE(python_data.sigma_points.cols() == 145); // 2*n+1 sigma points

        // Print loaded state for sanity check
        INFO("Python prior root position: "
             << python_data.prior_state.root_position().transpose());
    }

    SECTION("2. Initialize C++ UKF with Python's prior state") {
        auto python_data = test_helpers::load_python_frame0_data(debug_dir);

        // Load skeleton and cameras (from test fixtures)
        auto skeleton = load_test_skeleton();
        auto cameras = load_test_cameras();
        auto fk = create_test_fk(skeleton);

        // Create UKF with same parameters as Python
        double alpha = 0.5;  // Must match Python's alpha
        double beta = 2.0;
        double kappa = 0.0;
        double process_noise = 0.01;

        UnscentedKalmanFilter ukf(skeleton, process_noise, alpha, beta, kappa);
        ukf.set_state(python_data.prior_state);
        ukf.set_covariance(python_data.prior_covariance);

        // Verify state was set correctly
        REQUIRE_THAT(ukf.state().root_position(),
                    test_helpers::MatrixEquals(python_data.prior_state.root_position()));
    }

    SECTION("3. Compare sigma point generation") {
        auto python_data = test_helpers::load_python_frame0_data(debug_dir);
        auto ukf = create_ukf_from_python_prior(python_data);

        // Generate sigma points (no prediction yet, dt=0)
        auto cpp_sigma_points = ukf.generate_sigma_points_for_testing();

        // Compare with Python's sigma points
        REQUIRE(cpp_sigma_points.size() == 145);

        for (size_t i = 0; i < cpp_sigma_points.size(); ++i) {
            Eigen::VectorXd python_sp = python_data.sigma_points.col(i);
            Eigen::VectorXd cpp_sp = cpp_sigma_points[i].to_vector();

            if (!test_helpers::matrices_equal(cpp_sp, python_sp, 1e-10)) {
                FAIL("Sigma point " << i << " differs:\n"
                     << test_helpers::matrix_diff_string(cpp_sp, python_sp));
            }
        }
    }

    SECTION("4. Compare measurement prediction") {
        auto python_data = test_helpers::load_python_frame0_data(debug_dir);
        auto ukf = create_ukf_from_python_prior(python_data);
        auto skeleton = load_test_skeleton();
        auto cameras = load_test_cameras();
        auto fk = create_test_fk(skeleton);

        // Get sigma points
        auto sigma_points = ukf.generate_sigma_points_for_testing();

        // For each sigma point, predict observations
        for (size_t sp_idx = 0; sp_idx < sigma_points.size(); ++sp_idx) {
            auto predictions = ukf.predict_measurements_for_testing(
                sigma_points[sp_idx], python_data.observations, cameras, fk);

            // Compare with Python's predicted_observations
            for (size_t obs_idx = 0; obs_idx < python_data.observations.size(); ++obs_idx) {
                double python_u = python_data.predicted_obs(2*obs_idx, sp_idx);
                double python_v = python_data.predicted_obs(2*obs_idx+1, sp_idx);
                double cpp_u = predictions(2*obs_idx);
                double cpp_v = predictions(2*obs_idx+1);

                // Allow NaN == NaN comparison
                bool u_match = (std::isnan(python_u) && std::isnan(cpp_u)) ||
                              std::abs(python_u - cpp_u) < 1e-8;
                bool v_match = (std::isnan(python_v) && std::isnan(cpp_v)) ||
                              std::abs(python_v - cpp_v) < 1e-8;

                if (!u_match || !v_match) {
                    FAIL("Prediction mismatch at sigma_point=" << sp_idx
                         << ", obs=" << obs_idx
                         << ": Python=(" << python_u << "," << python_v << ")"
                         << ", C++=(" << cpp_u << "," << cpp_v << ")");
                }
            }
        }
    }

    SECTION("5. Compare outlier rejection") {
        auto python_data = test_helpers::load_python_frame0_data(debug_dir);
        auto ukf = create_ukf_from_python_prior(python_data);

        // Run update to get outlier results
        auto cameras = load_test_cameras();
        auto fk = create_test_fk(skeleton);

        auto update_result = ukf.update(python_data.observations, cameras, fk,
                                       5.0, 4.0); // measurement_noise, threshold

        // Compare outlier counts
        size_t python_inliers = std::count(python_data.outlier_flags.begin(),
                                          python_data.outlier_flags.end(), false);
        REQUIRE(update_result.num_inliers == python_inliers);

        // Compare per-observation outlier status
        for (size_t i = 0; i < python_data.observations.size(); ++i) {
            bool python_is_outlier = python_data.outlier_flags[i];
            bool cpp_is_outlier = update_result.observations[i].is_outlier;

            if (python_is_outlier != cpp_is_outlier) {
                INFO("Outlier mismatch at obs " << i
                     << " (marker=" << python_data.observations[i].marker_id
                     << ", camera=" << python_data.observations[i].camera_id << ")");
                REQUIRE(python_is_outlier == cpp_is_outlier);
            }
        }
    }

    SECTION("6. Compare innovation covariance") {
        // After outlier rejection, compare S matrix
        auto python_data = test_helpers::load_python_frame0_data(debug_dir);
        auto ukf = create_ukf_from_python_prior(python_data);

        auto cpp_innov_cov = ukf.get_innovation_covariance_for_testing();

        REQUIRE_THAT(cpp_innov_cov,
                    test_helpers::MatrixEquals(python_data.innovation_cov, 1e-8));
    }

    SECTION("7. Compare Kalman gain") {
        auto python_data = test_helpers::load_python_frame0_data(debug_dir);
        auto ukf = create_ukf_from_python_prior(python_data);

        auto cpp_kalman_gain = ukf.get_kalman_gain_for_testing();

        REQUIRE_THAT(cpp_kalman_gain,
                    test_helpers::MatrixEquals(python_data.kalman_gain, 1e-8));
    }

    SECTION("8. Compare posterior state") {
        auto python_data = test_helpers::load_python_frame0_data(debug_dir);
        auto ukf = create_ukf_from_python_prior(python_data);
        auto cameras = load_test_cameras();
        auto fk = create_test_fk(skeleton);

        // Run full update
        ukf.update(python_data.observations, cameras, fk, 5.0, 4.0);

        // Compare final state
        State cpp_posterior = ukf.state();

        REQUIRE_THAT(cpp_posterior.root_position(),
                    test_helpers::VectorEquals(python_data.posterior_state.root_position(), 1e-6));
        REQUIRE_THAT(cpp_posterior.root_orientation().coeffs(),
                    test_helpers::VectorEquals(python_data.posterior_state.root_orientation().coeffs(), 1e-6));
        // ... compare all state components
    }

    SECTION("9. Compare posterior covariance") {
        auto python_data = test_helpers::load_python_frame0_data(debug_dir);
        auto ukf = create_ukf_from_python_prior(python_data);
        auto cameras = load_test_cameras();
        auto fk = create_test_fk(skeleton);

        ukf.update(python_data.observations, cameras, fk, 5.0, 4.0);

        Eigen::MatrixXd cpp_posterior_cov = ukf.covariance();

        REQUIRE_THAT(cpp_posterior_cov,
                    test_helpers::MatrixEquals(python_data.posterior_covariance, 1e-6));
    }
}
```

### Step 4: Add Testing Accessors to UKF
**File**: `include/posetrak/filters/ukf.hpp`

Add these methods (only compiled in test builds):
```cpp
#ifdef POSETRAK_ENABLE_TESTING
    // Testing-only accessors
    std::vector<State> generate_sigma_points_for_testing() const;
    Eigen::VectorXd predict_measurements_for_testing(
        State const& state,
        std::vector<Observation> const& observations,
        std::unordered_map<int, Camera> const& cameras,
        ForwardKinematics const& fk) const;
    Eigen::MatrixXd get_innovation_covariance_for_testing() const;
    Eigen::MatrixXd get_kalman_gain_for_testing() const;
#endif
```

Or alternatively, use a testing-only subclass/friend class.

## Execution Plan

### Phase 1: Setup (Day 1)
1. Create `test_helpers/` directory structure
2. Implement Python data loading utilities
3. Implement matrix comparison utilities
4. Add basic test structure with Section 1 (data loading)

### Phase 2: Sigma Points (Day 1)
5. Add testing accessor for sigma point generation
6. Implement Section 2-3 (sigma point comparison)
7. **Debug any differences found**

### Phase 3: Measurement Prediction (Day 2)
8. Add testing accessor for measurement prediction
9. Implement Section 4 (measurement prediction comparison)
10. **Debug any differences found**

### Phase 4: Update Step (Day 2-3)
11. Implement Sections 5-9 (outlier rejection through posterior)
12. Add testing accessors for intermediate matrices
13. **Debug any differences found at each substep**

### Phase 5: Documentation (Day 3)
14. Document all findings in DEBUG_FINDINGS.md
15. Update this plan with final results
16. Create similar tests for frame 1, frame 2 if needed

## Expected Outcomes

### If Test Passes Completely:
- C++ and Python are numerically equivalent for frame 0
- Any remaining divergence in full tracking is due to:
  - Different initialization
  - Accumulation of numerical errors over time
  - Process noise randomness

### If Test Fails at Specific Step:
We'll know exactly where the implementations differ:
- Sigma point generation → check UKF parameters (alpha, beta, kappa)
- Measurement prediction → check FK computation or camera projection
- Innovation covariance → check NaN handling or matrix computation
- Kalman gain → check matrix inversion or cross-covariance
- State update → check state addition/composition

## Success Criteria

**All assertions must pass** with tolerance:
- Sigma points: `< 1e-10` (should be exact)
- Predicted measurements: `< 1e-8` (camera projection numerical precision)
- Innovation covariance: `< 1e-8`
- Kalman gain: `< 1e-8`
- Posterior state position: `< 1mm` (1e-3 m)
- Posterior state angles: `< 0.001 rad` (~0.057°)
- Posterior covariance: `< 1e-6`

## Notes
- All file paths in test should be relative to workspace root
- Test data files should be committed to git for reproducibility
- Consider adding `#ifdef POSETRAK_SKIP_SLOW_TESTS` for CI speed
- May need to update meson.build to include test data files
