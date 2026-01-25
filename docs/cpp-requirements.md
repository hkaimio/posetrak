# C++ Motion Capture Tracker - Requirements Document

## 1. Executive Summary

This document defines requirements for a production C++ implementation of the UKF-based joint-space motion capture tracker. The system tracks skeletal motion from multi-camera 2D pose detections using forward kinematics and Unscented Kalman Filtering.

**Primary Goal**: Feature parity with `joint_space_tracker.py` with significant performance improvements.

**Target Performance**: 30-60 Hz real-time tracking for 120 DOF skeleton with 4-8 cameras.

---

## 2. Functional Requirements

### 2.1 Core Tracking (Priority: P0)

**FR-001**: Unscented Kalman Filter in joint space
- State: root position (3D), root orientation (quaternion), joint angles (configurable DOF), velocities
- Process model: constant angular velocity (extensible to more complex models)
- Measurement model: forward kinematics + camera projection
- Configurable for arbitrary skeleton structures (not hardcoded for any specific DOF count)
- Note: 120 DOF is typical target for full-body with hands, but system must support any skeleton

**FR-002**: Forward Kinematics
- Use Pinocchio library for FK computation
- Support arbitrary skeleton hierarchies from URDF/YAML
- Compute marker positions from joint angles
- Configurable marker set (not hardcoded for COCO keypoints)
- Note: 133 COCO keypoints is common, but system must support arbitrary marker definitions

**FR-003**: Multi-Camera Support
- Support arbitrary number of cameras (optimize for typical 4-8 camera setups)
- Perspective camera model with Brown-Conrady distortion
- Fisheye camera model with OpenCV fisheye distortion
- Batch projection for all cameras
- Per-camera observation confidence weighting

**FR-004**: Outlier Rejection
- Mahalanobis distance-based outlier detection
- Configurable threshold (e.g., 5.991 for 95%, 9.21 for 99%)
- Per-observation rejection (individual markers)
- Whole camera frame rejection (when most/all observations are outliers)
- Frame rejection criteria configurable (e.g., reject if >50% outliers)

**FR-005**: Initialization
- Triangulate initial 3D pose from multi-camera 2D detections
- Solve inverse kinematics for initial joint angles
- Initialize velocities from first 2-3 frames
- Handle missing markers during initialization

### 2.2 Camera Model (Priority: P0)

**FR-010**: Distortion Handling
- Support radial + tangential distortion (Brown-Conrady model)
- Support OpenCV fisheye distortion model (4-parameter model)
- Pre-undistort 2D detections to ideal projection plane before UKF
- Maintain mapping to project 3D results back to distorted coordinates
- Use OpenCV-compatible distortion parameters
- Consider using OpenCV for distortion handling to ensure compatibility

**FR-011**: Frame Rate & Synchronization
- Support per-camera frame rates (may differ between cameras)
- External synchronization metadata file
- Define synchronization points: (camera, frame_index) → timestamp
- Linear interpolation of timestamps between sync points
- Handle frame rate jitter

**FR-012**: Camera Calibration Format
- TOML input format (matching Python prototype)
- Store: intrinsics (fx, fy, cx, cy), distortion coefficients (Brown-Conrady or fisheye)
- Distortion model type identifier ("brown_conrady" or "fisheye")
- Extrinsics: position, rotation (or projection matrix)
- Per-camera FPS and start_frame offset (for backward compatibility)
- Synchronization points should be used when available (override FPS-based timing)

### 2.3 Input/Output (Priority: P0)

**FR-020**: Input Formats
- OpenPose JSON (multi-person, multi-camera)
- Camera calibration TOML
- Skeleton definition YAML (or URDF)
- Synchronization metadata JSON

**FR-021**: Output Formats
- **TRC** (OpenSim text format) - marker trajectories [Optional]
- **JSON** (structured: states, skeleton, metadata)
- **BVH** (via Python wrapper or FBX as alternative)
- **Archive format**: ZIP with JSON files (replaces HDF5 for diagnostics)
- **Video** (overlay tracking on original footage) via OpenCV

**FR-022**: Diagnostics Export
- Per-frame: state, covariance, innovation, outliers, residuals
- Per-observation: reprojection error, Mahalanobis distance
- Summary statistics: mean/std errors, outlier rates
- Format: ZIP archive with:
  - `metadata.json` (skeleton, cameras, parameters)
  - `states/*.json` (per-frame states)
  - `diagnostics/*.json` (per-frame diagnostics)
  - `summary.json` (aggregated statistics)

**FR-023**: Multi-Person Tracking
- Track multiple people from same input data in single run
- Each person has independent UKF state
- Shared camera observations distributed to appropriate trackers
- Parallel tracking of multiple people
- Output separate result files per person or combined archive

### 2.4 Skeleton Model (Priority: P0)

**FR-030**: Skeleton Representation
- Hierarchical joint structure (tree)
- Support 120+ DOF (core body + hands)
- Joint types: revolute, ball (3-DOF), 2-DOF
- Joint limits (min/max angles)
- Marker attachment points (133 COCO keypoints)

**FR-031**: Skeleton Configuration
- YAML input format (matching Python prototype)
- Support joint groups (e.g., core, left_arm, right_arm, legs)
- Configurable active joints (filter out unused joints)
- Export to URDF for Pinocchio

### 2.5 Progress Reporting (Priority: P1)

**FR-040**: Progress Callbacks
- Frame start/end events
- Prediction/update step events
- Outlier detection events
- Configurable callback functions (std::function)

**FR-041**: Console Output
- Real-time progress bar
- Per-frame timing and error statistics
- Outlier rejection summary
- Configurable verbosity levels

### 2.6 Command-Line Interface (Priority: P0)

**FR-050**: CLI Matching Python Script
```bash
posetrak track \
    --skeleton <path> \
    --calib <path> \
    --base-dir <path> \
    --output-dir <path> \
    --person-id <int> \
    --start-frame <int> [<int> ...] \
    --max-frames <int> \
    --fps <float> \
    --active-groups <group> [...] \
    --min-confidence <float> \
    --process-noise-std <float> \
    --measurement-noise-std <float> \
    --outlier-threshold <float> \
    --n-jobs <int> \
    --create-bvh \
    --create-statistics \
    --save-diagnostics \
    --profile
```

**FR-051**: Help & Documentation
- Comprehensive --help text
- Example usage in help
- Parameter descriptions with defaults
- Validation of input parameters

---

## 3. Non-Functional Requirements

### 3.1 Performance (Priority: P0)

**NFR-001**: Real-Time Tracking
- Target: 30-60 Hz for 120 DOF skeleton
- 10-30 Hz acceptable for full-body with hands
- Graceful degradation with increased DOF

**NFR-002**: Memory Efficiency
- < 1 GB RAM for typical tracking session (1000 frames)
- Streaming architecture (don't load all frames at once)
- Reuse allocation where possible

**NFR-003**: Parallelization
- Multi-threaded sigma point evaluation (OpenMP or similar)
- Configurable thread count (--n-jobs)
- Thread-safe camera projection

**NFR-004**: Startup Time
- < 1 second for skeleton and camera loading
- < 2 seconds for initialization (triangulation + IK)

### 3.2 Code Quality (Priority: P0)

**NFR-010**: Modern C++ Standards
- C++20 minimum (C++23 features where beneficial)
- Use: concepts, ranges, std::span, std::format, coroutines (if applicable)
- Avoid: raw pointers, manual memory management, C-style arrays

**NFR-011**: Clean Architecture
- Separation of concerns (core library vs CLI vs bindings)
- Dependency injection where appropriate
- Header-only where beneficial, compiled libraries where needed
- Clear interfaces between modules

**NFR-012**: Error Handling
- Exceptions for error propagation
- std::expected or std::optional for expected failures
- Meaningful error messages with context
- No silent failures

**NFR-013**: Testing
- Unit tests for core algorithms (GTest)
- Integration tests for full pipeline
- Regression tests against Python prototype results
- Performance benchmarks

### 3.3 Portability (Priority: P1)

**NFR-020**: Cross-Platform
- Linux (primary development target)
- Windows (WSL2 and native)
- macOS (best-effort)

**NFR-021**: Build System
- Meson (primary, as used in prototype)
- CMake support (optional, for wider ecosystem)
- Package manager integration (vcpkg or Conan)

**NFR-022**: Dependencies
- Minimize required system libraries
- Pin versions for reproducibility
- Clear installation documentation

### 3.4 Extensibility (Priority: P1)

**NFR-030**: Library Design
- Core library separate from CLI
- Clean C API for bindings
- Python bindings via pybind11
- Minimal coupling between modules

**NFR-031**: UI Integration Points
- Callback-based progress reporting
- Stateful tracker object (pause/resume)
- Frame-by-frame stepping mode
- Export intermediate results

**NFR-032**: Future Features
- Plugin architecture for new filters (EKF, Factor Graph)
- Custom camera models
- Custom skeleton formats
- Real-time video processing

**NFR-033**: Extensible Process Models (Priority: P1)
- Architecture supports pluggable process models
- Default: constant angular velocity
- Future: constant acceleration, physics-based, learned models
- Process model as abstract interface

**NFR-034**: Hierarchical Tracking (Priority: P1)
- Split skeleton into independent regions (torso, limbs)
- Track regions in parallel with UKF per region
- Condition limb tracking on torso state
- Reduces sigma point count and improves performance

### 3.5 Documentation (Priority: P1)

**NFR-040**: Code Documentation
- Doxygen-compatible comments
- API documentation generation
- Architecture diagrams

**NFR-041**: User Documentation
- Installation guide
- Usage examples
- Parameter tuning guide
- Troubleshooting section

**NFR-042**: Developer Documentation
- Build instructions
- Contributing guide
- Design decisions log
- Migration guide from Python

---

## 4. Technical Constraints

### 4.1 Dependencies

**Mandatory**:
- Eigen 3.4+ (linear algebra)
- Pinocchio 3.9+ (forward kinematics and inverse kinematics)
- fmt 10.0+ (formatting, until std::format widely available)
- yaml-cpp (configuration)
- toml11 (calibration)
- nlohmann/json (data interchange)
- CLI11 (command-line parsing)
- OpenCV 4.5+ (camera distortion handling, video I/O, visualization)
- libarchive or miniz (ZIP export)

**Optional**:
- OpenMP (parallelization)
- pybind11 (Python bindings)
- GTest or Catch2 (testing - either is acceptable)

**Python-Only** (via bindings):
- bvhsdk (BVH export - keep Python implementation)
- matplotlib (visualization)

### 4.2 Hardware Assumptions

- Multi-core CPU (4+ cores for good performance)
- 8+ GB RAM
- GPU optional (future work for FK/projection)

---

## 5. Success Criteria

### 5.1 Parity with Python Prototype

**SC-001**: Functional Parity
- [ ] Track same sequences as Python
- [ ] Produce equivalent joint angle trajectories (< 1° RMSE)
- [ ] Handle same input formats
- [ ] Generate same output formats (TRC, JSON)

**SC-002**: Performance Target
- [ ] 20-50× faster than Python for UKF steps
- [ ] 30-60 Hz for 120 DOF skeleton (vs 1-3 Hz Python)
- [ ] < 1 GB memory usage

**SC-003**: Robustness
- [ ] Pass all Python regression tests
- [ ] Handle edge cases (missing markers, occlusions)
- [ ] Graceful error messages

### 5.2 Production Ready

**SC-010**: Code Quality
- [ ] 80%+ test coverage (core library)
- [ ] Zero memory leaks (valgrind/asan)
- [ ] Zero warnings (GCC/Clang with -Wall -Wextra)

**SC-011**: Documentation
- [ ] Complete API documentation
- [ ] User guide with examples
- [ ] Installation tested on 3 platforms

**SC-012**: Maintainability
- [ ] Clear module boundaries
- [ ] Consistent code style (clang-format)
- [ ] CI/CD pipeline (build + test)

---

## 6. Out of Scope (Future Work)

**Phase 2+**:
- Factor Graph optimization (offline refinement)
- Extended Kalman Filter (comparison)
- Real-time video processing (no pre-computed detections)
- Automatic synchronization (LED detection)
- Multi-person tracking with contact constraints
- GPU acceleration (CUDA/OpenCL)
- GUI (Qt or ImGUI)
- Biomechanical constraints (collision, balance)
- Advanced visualization (3D viewport)

---

## 7. Migration Strategy

### 7.1 Development Phases

**Phase 1: Core Library** (4-6 weeks)
- State, Skeleton, Camera models
- Forward kinematics (Pinocchio integration)
- UKF implementation (monolithic)
- Basic I/O (YAML, TOML, OpenPose JSON)

**Phase 2: Full Pipeline** (2-3 weeks)
- Initialization (triangulation + IK)
- Outlier rejection
- Progress callbacks
- CLI tool

**Phase 3: Output & Diagnostics** (1-2 weeks)
- TRC export
- JSON export
- ZIP diagnostics archive
- Statistics generation

**Phase 4: Validation & Optimization** (2-3 weeks)
- Regression tests vs Python
- Performance profiling
- Hierarchical UKF optimization
- Documentation

**Phase 5: Bindings & Polish** (2-3 weeks)
- Python bindings (pybind11)
- BVH export wrapper
- Cross-platform builds
- Release packaging

### 7.2 Validation Approach

**Regression Testing**:
1. Track 5-10 sequences with Python
2. Export states, diagnostics, statistics
3. Track same sequences with C++
4. Compare:
   - Joint angle trajectories (< 1° RMSE)
   - Reprojection errors (< 1 pixel difference)
   - Outlier detection (same outliers)
   - Final skeleton poses (< 5mm marker error)

**Performance Benchmarking**:
1. Track reference sequence with both implementations
2. Measure:
   - Frames per second
   - Time per UKF step (predict/update)
   - Memory usage (peak and average)
   - Initialization time

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Pinocchio integration complexity | Medium | High | Start with simple skeleton, validate incrementally |
| Distortion handling introduces errors | Medium | Medium | Test with synthetic data, validate against OpenCV |
| Synchronization metadata format unclear | Low | Medium | Define format early, create test data |
| Performance targets not met | Low | High | Profile early, optimize hot paths, hierarchical UKF |
| Parity with Python not achieved | Low | High | Continuous regression testing, fix discrepancies early |
| ZIP format parsing issues | Low | Low | Use well-tested library (libarchive), simple JSON inside |
| BVH export complexity | Medium | Low | Keep Python wrapper, use proven bvhsdk |
| Cross-platform build issues | Medium | Medium | Use Meson, test on all platforms early |

---

## 9. Dependencies on External Work

**None**: All components can be developed independently with existing libraries and data formats.

**Future**:
- GUI framework selection (Qt vs ImGUI) - when Phase 5+ starts
- Factor Graph library (GTSAM) - when offline optimization added
- GPU framework (CUDA/OpenCL) - when acceleration needed

---

## 10. Acceptance Criteria Summary

The C++ implementation is considered **production ready** when:

1. ✅ Tracks reference sequences with < 1° RMSE vs Python
2. ✅ Runs at 30+ Hz for 120 DOF skeleton on 4-core CPU
3. ✅ Passes all regression tests
4. ✅ Has zero memory leaks and warnings
5. ✅ Complete documentation (API + user guide)
6. ✅ Builds on Linux, Windows, macOS
7. ✅ Python bindings with BVH export working
8. ✅ CLI matches Python feature set

**Target Delivery**: 12-14 weeks from start of development.
