# C++ Motion Capture Tracker - Implementation Plan

## Overview

This document provides a phased implementation plan for the C++ motion capture tracker. The first 4 phases are detailed with specific steps and exit criteria. Later phases are outlined at a high level and will be refined as we progress.

**Estimated Timeline**: 12-16 weeks total
**Development Approach**: Incremental validation against Python prototype

---

## Phase 0: Project Setup & Infrastructure (Week 1)

### Goals
- Establish build system
- Set up testing framework
- Configure development environment
- Create basic project structure

### Steps

#### 0.1: Create Project Structure
```
cpp-tracker-test/
├── meson.build
├── include/posetrak/
│   ├── core/
│   ├── kinematics/
│   ├── filters/
│   ├── tracking/
│   └── io/
├── src/
├── tests/
├── cli/
├── examples/
└── docs/
```

**Tasks**:
- [ ] Create directory structure
- [ ] Write root `meson.build` with project metadata
- [ ] Set up include directory structure
- [ ] Create placeholder `.cpp` files for core modules

**Exit Criteria**:
- ✅ `meson setup builddir` succeeds
- ✅ Project compiles (even if empty)
- ✅ Directory structure matches architecture

#### 0.2: Configure Dependencies
**Tasks**:
- [ ] Add Eigen dependency to meson.build
- [ ] Add fmt dependency
- [ ] Add nlohmann/json dependency
- [ ] Add yaml-cpp dependency
- [ ] Add toml11 dependency
- [ ] Add CLI11 dependency
- [ ] Test each dependency with minimal example

**Exit Criteria**:
- ✅ All dependencies resolve and link
- ✅ Can compile simple test using each library
- ✅ `pkg-config` or equivalent finds all libraries

#### 0.3: Set Up Testing Framework
**Tasks**:
- [ ] Choose between Catch2 and GTest (either acceptable)
- [ ] Add testing dependency to meson.build
- [ ] Create `tests/meson.build`
- [ ] Write first test (trivial, just to verify framework)
- [ ] Configure test runner

**Exit Criteria**:
- ✅ `meson test` runs successfully
- ✅ Can add new test files easily
- ✅ Test output is clear and informative

#### 0.4: Version Control & Documentation
**Tasks**:
- [ ] Update `.gitignore` (builddir, IDE files)
- [ ] Create `README.md` with build instructions
- [ ] Create `CONTRIBUTING.md` with coding standards
- [ ] Set up clang-format configuration

**Exit Criteria**:
- ✅ Clean git status (no build artifacts)
- ✅ Another developer can build from README
- ✅ Code formatting is consistent

### Phase 0 Exit Criteria Summary
- ✅ Project builds cleanly
- ✅ All required dependencies available
- ✅ Test framework operational
- ✅ Documentation covers basic setup

**Estimated Time**: 2-3 days

---

## Phase 1: Core Models (Week 1-2)

### Goals
- Implement State representation
- Implement Skeleton data structure
- Implement Camera model (without OpenCV initially)
- Implement Observation structures
- Full test coverage for core models

### Steps

#### 1.1: State Implementation
**Files**: `include/posetrak/core/state.hpp`, `src/core/state.cpp`

**Tasks**:
- [ ] Implement State class with:
  - Root position (Eigen::Vector3d)
  - Root orientation (Eigen::Quaterniond)
  - Joint angles (Eigen::VectorXd)
  - Velocities (root + joints)
- [ ] Implement `to_error_vector()` (error-state formulation)
- [ ] Implement `apply_error_update()` (manifold operations)
- [ ] Implement quaternion ↔ axis-angle conversions
- [ ] Implement JSON serialization

**Tests**:
- [ ] Test error state conversion round-trip
- [ ] Test quaternion composition
- [ ] Test axis-angle to quaternion conversion
- [ ] Test JSON serialization/deserialization
- [ ] Test with different DOF counts

**Exit Criteria**:
- ✅ All tests pass
- ✅ State can represent arbitrary skeleton (not hardcoded DOF)
- ✅ Error-state operations are correct
- ✅ No memory leaks (valgrind)

#### 1.2: Skeleton Implementation
**Files**: `include/posetrak/core/skeleton.hpp`, `src/core/skeleton.cpp`

**Tasks**:
- [ ] Implement Joint struct (name, parent, type, DOF, limits, group)
- [ ] Implement Marker struct (name, joint, local position, COCO ID)
- [ ] Implement Skeleton class:
  - Joint hierarchy (tree structure)
  - Marker definitions
  - Active joint filtering (by group or explicit list)
  - Validation (no cycles, root exists)
- [ ] Implement DOF counting (total, active)
- [ ] Implement JSON export

**Tests**:
- [ ] Test simple skeleton (pelvis + 2 joints)
- [ ] Test active group filtering
- [ ] Test cycle detection
- [ ] Test DOF calculation
- [ ] Test marker attachment

**Exit Criteria**:
- ✅ Can represent arbitrary skeleton hierarchies
- ✅ Active joint filtering works correctly
- ✅ Tree validation catches errors
- ✅ All tests pass

#### 1.3: Camera Implementation (Basic)
**Files**: `include/posetrak/core/camera.hpp`, `src/core/camera.cpp`

**Tasks**:
- [ ] Implement Intrinsics struct (fx, fy, cx, cy, distortion model enum, coeffs vector)
- [ ] Implement Extrinsics struct (position, rotation, get_transform)
- [ ] Implement SyncPoint struct
- [ ] Implement Camera class:
  - Constructor
  - `project_undistorted()` (basic pinhole)
  - `get_timestamp()` with sync point interpolation
  - `get_frame_at_time()` (inverse lookup)
- [ ] Implement Brown-Conrady distortion (manual, no OpenCV yet)
- [ ] Implement batch projection

**Tests**:
- [ ] Test pinhole projection (known ground truth)
- [ ] Test timestamp interpolation with sync points
- [ ] Test frame lookup
- [ ] Test Brown-Conrady distortion forward/inverse
- [ ] Test batch projection performance

**Exit Criteria**:
- ✅ Projection is correct (< 0.1 pixel error for test cases)
- ✅ Sync point interpolation works
- ✅ Distortion model matches OpenCV results (manual test)
- ✅ All tests pass

**Note**: OpenCV integration (fisheye, undistortion) deferred to Phase 4

#### 1.4: Observation Structures
**Files**: `include/posetrak/core/observation.hpp`, `src/core/observation.cpp`

**Tasks**:
- [ ] Implement Observation struct
- [ ] Implement ObservationSequence class:
  - `get_in_range(t_start, t_end)` (not exact equality)
  - `get_at_frame(frame_idx)`
- [ ] Implement ObservationSet class:
  - `get_all_in_range(t_start, t_end)`
  - `min_time()`, `max_time()`
  - `get_unique_timestamps()`

**Tests**:
- [ ] Test time range queries (inclusivity, edge cases)
- [ ] Test frame queries
- [ ] Test empty observation handling
- [ ] Test multi-camera aggregation

**Exit Criteria**:
- ✅ Time range queries work correctly (no floating point equality issues)
- ✅ All tests pass
- ✅ API is ergonomic

### Phase 1 Exit Criteria Summary
- ✅ All core models implemented and tested
- ✅ 80%+ test coverage for core models
- ✅ Can represent arbitrary skeletons and cameras
- ✅ Error-state formulation working
- ✅ No memory leaks or undefined behavior

**Estimated Time**: 5-7 days

---

## Phase 2: I/O Layer (Week 2-3)

### Goals
- Load skeleton from YAML
- Load cameras from TOML
- Load OpenPose observations from JSON
- Validate against Python prototype data

### Steps

#### 2.1: Skeleton Loader
**Files**: `include/posetrak/io/skeleton_loader.hpp`, `src/io/skeleton_loader.cpp`

**Tasks**:
- [ ] Parse YAML skeleton format (matching Python prototype)
- [ ] Build Skeleton object from YAML
- [ ] Implement joint limit parsing
- [ ] Implement marker definition parsing
- [ ] Handle groups
- [ ] Validate skeleton structure

**Tests**:
- [ ] Load `examples/simple_humanoid.yaml` (from Python repo)
- [ ] Verify joint count matches
- [ ] Verify marker count matches
- [ ] Test error handling (malformed YAML)
- [ ] Compare with Python loaded skeleton (JSON export)

**Exit Criteria**:
- ✅ Can load all example skeletons from Python repo
- ✅ Produces identical structure to Python (verified via JSON comparison)
- ✅ Error messages are clear
- ✅ All tests pass

#### 2.2: Camera Loader
**Files**: `include/posetrak/io/camera_loader.hpp`, `src/io/camera_loader.cpp`

**Tasks**:
- [ ] Parse TOML camera calibration format
- [ ] Build Camera objects from TOML
- [ ] Handle distortion model types (Brown-Conrady, fisheye placeholder)
- [ ] Parse extrinsics (multiple formats: position+rotation, matrix)
- [ ] Handle FPS and start_frame

**Tests**:
- [ ] Load camera calibration from Python repo
- [ ] Verify intrinsics match
- [ ] Verify extrinsics match
- [ ] Test projection against Python camera model
- [ ] Test error handling

**Exit Criteria**:
- ✅ Can load all calibration files from Python repo
- ✅ Projection matches Python (< 0.1 pixel difference)
- ✅ All tests pass

#### 2.3: Synchronization Metadata Loader
**Files**: `include/posetrak/io/sync_loader.hpp`, `src/io/sync_loader.cpp`

**Tasks**:
- [ ] Define JSON schema for sync metadata
- [ ] Implement parser
- [ ] Apply sync points to Camera objects
- [ ] Handle missing sync data (fallback to FPS)

**Tests**:
- [ ] Test sync point application
- [ ] Test timestamp calculation with sync points
- [ ] Test fallback behavior
- [ ] Test JSON parsing errors

**Exit Criteria**:
- ✅ Sync metadata loads correctly
- ✅ Timestamp calculation matches expected values
- ✅ All tests pass

#### 2.4: OpenPose Observation Loader
**Files**: `include/posetrak/io/observation_loader.hpp`, `src/io/observation_loader.cpp`

**Tasks**:
- [ ] Parse OpenPose JSON format (person array, keypoints_2d)
- [ ] Build Observation objects
- [ ] Handle multi-person selection (person_id)
- [ ] Apply confidence threshold
- [ ] Undistort coordinates (using Camera::undistort)
- [ ] Build ObservationSet from multi-camera directory structure
- [ ] Handle missing frames gracefully
- [ ] Apply frame range filtering (start_frame, max_frames)

**Tests**:
- [ ] Load observations from Python test data
- [ ] Verify observation count matches Python
- [ ] Verify confidence filtering
- [ ] Test person_id selection
- [ ] Test frame range filtering
- [ ] Compare observations with Python (JSON export)

**Exit Criteria**:
- ✅ Can load all test sequences from Python repo
- ✅ Observation count and content match Python
- ✅ Undistorted coordinates match (< 0.1 pixel)
- ✅ All tests pass

### Phase 2 Exit Criteria Summary
- ✅ All I/O operations functional
- ✅ Can load data from Python test cases
- ✅ Output matches Python prototype (verified via JSON comparison)
- ✅ Error handling is robust
- ✅ All tests pass

**Estimated Time**: 5-7 days

---

## Phase 3: Kinematics Layer (Week 3-4)

### Goals
- Integrate Pinocchio for forward kinematics
- Implement triangulation
- Implement basic IK for initialization
- Validate FK against Python prototype

### Steps

#### 3.1: Pinocchio Integration
**Files**: `include/posetrak/kinematics/forward_kinematics.hpp`, `src/kinematics/forward_kinematics.cpp`

**Tasks**:
- [ ] Add Pinocchio dependency to meson.build
- [ ] Convert Skeleton → Pinocchio Model
  - Map joint types
  - Build kinematic tree
  - Add marker frames
- [ ] Implement ForwardKinematics class:
  - Constructor (builds Pinocchio model)
  - `compute_marker_positions(State)` → vector<Vector3d>
  - `compute_marker_position(State, marker_id)` → Vector3d
  - `compute_joint_transforms(State)` → vector<Affine3d>
- [ ] Handle thread safety (one Data per thread)

**Tests**:
- [ ] Test FK with simple skeleton (known ground truth)
- [ ] Compare FK results with Python prototype
- [ ] Test multi-threaded FK (no race conditions)
- [ ] Benchmark FK performance

**Exit Criteria**:
- ✅ FK results match Python (< 1mm error)
- ✅ Thread-safe
- ✅ Performance is 10-20× faster than Python
- ✅ All tests pass

#### 3.2: Skeleton to URDF Export
**Files**: `include/posetrak/io/urdf_export.hpp`, `src/io/urdf_export.cpp`

**Tasks**:
- [ ] Implement `Skeleton::to_urdf()` method
- [ ] Generate valid URDF from Skeleton
- [ ] Test URDF can be loaded by Pinocchio
- [ ] Validate joint limits, types, offsets

**Tests**:
- [ ] Export simple skeleton to URDF
- [ ] Load URDF with Pinocchio
- [ ] Verify FK matches original Skeleton
- [ ] Test round-trip: YAML → Skeleton → URDF → Pinocchio

**Exit Criteria**:
- ✅ Generated URDF is valid
- ✅ Pinocchio can load it
- ✅ FK results are consistent
- ✅ All tests pass

#### 3.3: Triangulation
**Files**: `include/posetrak/kinematics/triangulation.hpp`, `src/kinematics/triangulation.cpp`

**Tasks**:
- [ ] Implement multi-camera triangulation:
  - DLT (Direct Linear Transform) method
  - Or mid-point of closest approach
  - Or least-squares optimization
- [ ] Handle 2+ cameras per marker
- [ ] Handle missing observations (std::optional)
- [ ] Weight by confidence

**Tests**:
- [ ] Test with synthetic data (known 3D points)
- [ ] Test with 2, 3, 4+ cameras
- [ ] Compare with Python triangulation
- [ ] Test edge cases (parallel rays, etc.)

**Exit Criteria**:
- ✅ Triangulation is accurate (< 1cm error for test cases)
- ✅ Handles degenerate cases gracefully
- ✅ Matches Python results
- ✅ All tests pass

#### 3.4: Inverse Kinematics (Basic)
**Files**: `include/posetrak/kinematics/inverse_kinematics.hpp`, `src/kinematics/inverse_kinematics.cpp`

**Tasks**:
- [ ] Research Pinocchio IK capabilities
- [ ] Implement basic IK solver:
  - Target: marker positions (subset)
  - Output: joint angles
  - Method: Jacobian-based or optimization
- [ ] Or: simple gradient descent on FK residual
- [ ] Good enough for initialization (doesn't need to be perfect)

**Tests**:
- [ ] Test IK with known configurations
- [ ] Test initialization from triangulated markers
- [ ] Compare with Python IK results (qualitatively)
- [ ] Test convergence rate

**Exit Criteria**:
- ✅ IK converges for reasonable poses
- ✅ Initialization produces valid starting state
- ✅ Comparable to Python IK quality
- ✅ All tests pass

### Phase 3 Exit Criteria Summary
- ✅ Pinocchio integration complete
- ✅ FK matches Python (< 1mm error)
- ✅ Triangulation works for initialization
- ✅ IK produces valid starting poses
- ✅ All tests pass
- ✅ Performance is significantly better than Python

**Estimated Time**: 7-10 days

---

## Phase 4: Basic UKF & Tracking Pipeline (Week 4-6)

### Goals
- Implement UKF prediction step
- Implement UKF update step
- Implement sigma point generation (error-state)
- Implement basic Tracker orchestration
- Track first sequence end-to-end
- Validate against Python UKF

### Steps

#### 4.1: Process Model
**Files**: `include/posetrak/filters/process_model.hpp`, `src/filters/process_model.cpp`

**Tasks**:
- [ ] Define ProcessModel interface (abstract)
- [ ] Implement ConstantVelocityModel:
  - Propagate position: p' = p + v*dt
  - Propagate orientation: q' = q ⊗ exp(ω*dt/2)
  - Propagate joints: θ' = θ + ω*dt
  - Velocities: constant (with process noise)
- [ ] Implement joint limit projection
- [ ] Implement process noise covariance generation

**Tests**:
- [ ] Test state propagation (known trajectories)
- [ ] Test joint limit enforcement
- [ ] Test quaternion integration
- [ ] Verify against Python process model

**Exit Criteria**:
- ✅ Process model matches Python behavior
- ✅ Joint limits are enforced
- ✅ All tests pass

#### 4.2: Sigma Point Generator
**Files**: `include/posetrak/filters/sigma_points.hpp`, `src/filters/sigma_points.cpp`

**Tasks**:
- [ ] Implement SigmaPointGenerator class:
  - Unscented transform parameters (alpha, beta, kappa)
  - Generate sigma points in error space
  - Compute weights (mean, covariance)
  - Cholesky decomposition
- [ ] Implement error-state to full-state conversion
- [ ] Implement mean state computation (special handling for quaternions)

**Tests**:
- [ ] Test sigma point generation (2n+1 points)
- [ ] Test weight computation (sum to 1)
- [ ] Test covariance reconstruction
- [ ] Test quaternion mean (multiple quaternions)
- [ ] Compare with Python sigma points

**Exit Criteria**:
- ✅ Sigma points match Python
- ✅ Covariance is preserved
- ✅ All tests pass

#### 4.3: UKF Prediction Step
**Files**: `include/posetrak/filters/ukf.hpp`, `src/filters/ukf.cpp`

**Tasks**:
- [ ] Implement UKF::predict():
  - Generate sigma points from current state + covariance
  - Propagate each sigma point through process model
  - Compute predicted mean state
  - Compute predicted covariance
  - Add process noise
- [ ] Implement parallel sigma point propagation (OpenMP)
- [ ] Handle error-state formulation correctly

**Tests**:
- [ ] Test prediction with simple state
- [ ] Compare covariance evolution with Python
- [ ] Test parallelization (results identical)
- [ ] Test with different process noise levels

**Exit Criteria**:
- ✅ Prediction matches Python (state and covariance)
- ✅ Parallelization works
- ✅ All tests pass

#### 4.4: UKF Update Step (No Outlier Rejection)
**Files**: `src/filters/ukf.cpp` (continued)

**Tasks**:
- [ ] Implement measurement prediction:
  - For each sigma point: FK → project to cameras
  - Compute mean predicted measurement
  - Compute innovation covariance
- [ ] Implement Kalman gain calculation
- [ ] Implement state update:
  - Innovation = observed - predicted
  - State correction
  - Covariance update (Joseph form for stability)
- [ ] Implement measurement noise matrix construction
- [ ] Handle missing observations (std::optional)

**Tests**:
- [ ] Test update with known observations
- [ ] Compare Kalman gain with Python
- [ ] Compare updated state with Python
- [ ] Compare updated covariance with Python
- [ ] Test with missing observations

**Exit Criteria**:
- ✅ Update matches Python (state and covariance)
- ✅ Innovation is correct
- ✅ Covariance remains positive definite
- ✅ All tests pass

#### 4.5: Basic Tracker
**Files**: `include/posetrak/tracking/tracker.hpp`, `src/tracking/tracker.cpp`

**Tasks**:
- [ ] Implement Tracker class:
  - Constructor (takes FilterBase, Skeleton, Cameras)
  - `initialize()` (from initial state + covariance)
  - `initialize_from_observations()` (triangulation + IK)
  - `track()` (main loop over observations)
  - `step()` (single frame)
  - Callback support (on_frame_start, on_frame_done, etc.)
- [ ] Implement observation timing:
  - Get observations for current timestamp
  - Handle per-camera frame offsets
  - Use time ranges (not exact equality)
- [ ] Implement TrackerResult structure

**Tests**:
- [ ] Test initialization from observations
- [ ] Test single frame step
- [ ] Test full sequence tracking (simple skeleton, few frames)
- [ ] Test callbacks are invoked
- [ ] Compare with Python tracker results

**Exit Criteria**:
- ✅ Can track simple sequence end-to-end
- ✅ Results roughly match Python (within 5-10°)
- ✅ Callbacks work
- ✅ All tests pass

#### 4.6: First Integration Test
**Files**: `tests/integration/test_simple_tracking.cpp`

**Tasks**:
- [ ] Create test data:
  - Simple skeleton (10-20 DOF)
  - 2-4 cameras
  - 50-100 frames
  - OpenPose detections
- [ ] Track sequence with C++ implementation
- [ ] Track same sequence with Python
- [ ] Compare results:
  - Joint angles (RMSE)
  - Reprojection errors
  - Covariance evolution
- [ ] Debug discrepancies

**Tests**:
- [ ] RMSE < 5° for all joints
- [ ] Reprojection errors comparable
- [ ] No divergence or NaN values

**Exit Criteria**:
- ✅ First successful end-to-end tracking
- ✅ Results are qualitatively similar to Python
- ✅ No crashes or numerical instabilities
- ✅ Tracking completes in reasonable time (< 10 seconds for 100 frames)

### Phase 4 Exit Criteria Summary
- ✅ UKF prediction and update implemented
- ✅ Basic tracker works end-to-end
- ✅ Results are comparable to Python (within 5-10°)
- ✅ No numerical instabilities
- ✅ All tests pass
- ✅ Performance is 10-50× faster than Python

**Estimated Time**: 10-14 days

---

## Phase 5: Outlier Rejection & Robustness (Week 6-7)

### Goals (High-Level)
- Implement Mahalanobis distance outlier rejection
- Implement whole-frame rejection
- Improve numerical stability
- Handle edge cases gracefully

### Key Deliverables
- Outlier rejection working (per-observation and per-frame)
- Tracking is robust to missing markers and occlusions
- Numerical stability improvements (covariance conditioning)
- Comprehensive edge case testing

### Exit Criteria
- ✅ Outlier rejection matches Python behavior
- ✅ Can handle 30-50% missing observations
- ✅ No numerical instabilities in long sequences (1000+ frames)
- ✅ RMSE < 2-3° compared to Python

**Estimated Time**: 5-7 days

---

## Phase 6: OpenCV Integration & Video Export (Week 7-8)

### Goals (High-Level)
- Integrate OpenCV for distortion handling
- Support fisheye camera model
- Implement video overlay export
- Validate distortion handling 100% compatible with Python

### Key Deliverables
- Camera distortion/undistortion using OpenCV
- Fisheye camera support
- Video overlay generation (tracking on original footage)
- All cameras work correctly (brown-conrady and fisheye)

### Exit Criteria
- ✅ Projection matches OpenCV exactly
- ✅ Fisheye cameras work correctly
- ✅ Video export produces usable output
- ✅ No performance regression from OpenCV

**Estimated Time**: 5-7 days

---

## Phase 7: Full I/O & Export (Week 8-9)

### Goals (High-Level)
- TRC export (optional)
- JSON export (states, metadata)
- ZIP diagnostics archive
- Statistics generation
- BVH export (via Python wrapper)

### Key Deliverables
- All export formats working
- ZIP diagnostics archive with per-frame data
- Statistics CSV
- Integration with Python for BVH

### Exit Criteria
- ✅ All export formats produce valid output
- ✅ Can load and visualize results in external tools
- ✅ ZIP archives are human-readable
- ✅ BVH playback works in Blender/MotionBuilder

**Estimated Time**: 5-7 days

---

## Phase 8: CLI & User Experience (Week 9-10)

### Goals (High-Level)
- Full CLI matching Python script
- Progress reporting and logging
- Error messages and validation
- Performance profiling support
- Documentation

### Key Deliverables
- Command-line tool with all features
- Help text and examples
- Console progress output
- User-friendly error messages
- README with usage examples

### Exit Criteria
- ✅ CLI feature parity with Python script
- ✅ User can track sequences without reading code
- ✅ Error messages are actionable
- ✅ Performance is visible to user

**Estimated Time**: 5-7 days

---

## Phase 9: Optimization & Performance (Week 10-11)

### Goals (High-Level)
- Profile and optimize hot paths
- Implement hierarchical UKF (optional)
- SIMD optimization (if needed)
- Memory optimization
- Multi-person tracking

### Key Deliverables
- Performance benchmarks
- Hierarchical UKF implementation (if beneficial)
- Optimized matrix operations
- Multi-person tracking support

### Exit Criteria
- ✅ 30-60 Hz tracking for 120 DOF skeleton
- ✅ < 1 GB memory usage
- ✅ Multi-person tracking works
- ✅ 20-50× faster than Python

**Estimated Time**: 7-10 days

---

## Phase 10: Validation & Polish (Week 11-12)

### Goals (High-Level)
- Regression test suite against Python
- Cross-platform testing
- Documentation completion
- Bug fixes and edge cases
- Release preparation

### Key Deliverables
- Comprehensive test suite (80%+ coverage)
- Validated on multiple platforms
- Complete API documentation
- User guide with examples
- Release package

### Exit Criteria
- ✅ All regression tests pass (< 1° RMSE vs Python)
- ✅ Works on Linux, Windows, macOS
- ✅ Documentation is complete
- ✅ Ready for production use

**Estimated Time**: 7-10 days

---

## Phase 11+: Advanced Features (Future)

### Potential Features
- Python bindings (pybind11)
- Factor graph optimization (offline refinement)
- Extended Kalman Filter (comparison)
- GUI (Qt or ImGUI)
- Real-time video processing
- Automatic synchronization (LED detection)
- Biomechanical constraints (collision, balance)
- GPU acceleration

### Timeline
To be determined based on project needs and priorities.

---

## Success Metrics

### Functional Parity
- [ ] Can track all Python test sequences
- [ ] RMSE < 1° compared to Python for same data
- [ ] All Python features implemented

### Performance
- [ ] 30-60 Hz for 120 DOF skeleton (vs 1-3 Hz Python)
- [ ] 20-50× overall speedup
- [ ] < 1 GB memory usage

### Code Quality
- [ ] 80%+ test coverage
- [ ] Zero memory leaks (valgrind)
- [ ] Zero warnings (GCC/Clang -Wall -Wextra)
- [ ] Clean static analysis (clang-tidy)

### Usability
- [ ] CLI matches Python interface
- [ ] Documentation covers all features
- [ ] Builds on 3 platforms
- [ ] Error messages are actionable

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Pinocchio integration harder than expected | Start early (Phase 3), test incrementally |
| Performance targets not met | Profile early, hierarchical UKF, OpenMP |
| Numerical instabilities | Validate against Python continuously, use Joseph form |
| OpenCV compatibility issues | Test against Python OpenCV, use same version |
| Scope creep | Defer advanced features to Phase 11+ |

---

## Development Workflow

### Per Phase
1. **Plan**: Review phase goals and steps
2. **Implement**: Write code incrementally
3. **Test**: Write tests alongside code
4. **Validate**: Compare with Python prototype
5. **Document**: Update documentation
6. **Review**: Check exit criteria
7. **Commit**: Clean git history

### Continuous Practices
- **Test-Driven**: Write tests before or alongside implementation
- **Incremental**: Small, working commits
- **Validated**: Always compare with Python
- **Documented**: Keep README and docs up to date

### Tools
- **Profiling**: perf, gprof, or built-in profiling
- **Memory**: valgrind, AddressSanitizer
- **Coverage**: gcov, lcov
- **Static Analysis**: clang-tidy

---

## Next Steps

1. **Review this plan**: Stakeholder approval
2. **Start Phase 0**: Set up project structure
3. **Weekly check-ins**: Review progress and adjust plan
4. **Iterate**: Refine later phases as we learn

**Ready to begin with Phase 0?**
