# Rerun Visualization Design for Posetrak

**Created**: 2026-02-12
**Status**: Design Phase
**Target**: Comprehensive 3D/2D visualization of tracking results

---

## Overview

This document defines the design for integrating Rerun (https://rerun.io) into Posetrak for real-time and post-hoc visualization of motion capture tracking. The system will log multi-modal data (3D skeleton, 2D observations, metrics) in a hierarchical structure for interactive exploration.

---

## Architecture

### High-Level Design

```
┌─────────────────┐
│  Tracker        │
│  (orchestrator) │
└────────┬────────┘
         │
         ├───────► RerunLogger::log_frame_start()
         │
         ├───────► UKF::predict() ──► RerunLogger::log_prediction()
         │
         ├───────► UKF::update()  ──► RerunLogger::log_update()
         │
         └───────► RerunLogger::log_frame_end()
                           │
                           ▼
                   ┌──────────────┐
                   │ Rerun C++ SDK│
                   └──────┬───────┘
                          │
                          ▼
                   output.rrd (file)
                   or
                   tcp://localhost:9876 (live)
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| **Tracker** | Orchestrates tracking, calls RerunLogger at key points |
| **RerunLogger** | Encapsulates all Rerun API calls, maintains entity hierarchy |
| **UKF** | Provides diagnostics data to RerunLogger |
| **ForwardKinematics** | No changes (RerunLogger queries it) |
| **TrackerConfig** | Adds rerun-specific configuration options |

---

## Class Interface

### RerunLogger Class

**File**: `include/posetrak/visualization/rerun_logger.hpp`

```cpp
#pragma once

#include <rerun.hpp>
#include <optional>
#include <memory>
#include <filesystem>

#include "posetrak/core/camera.hpp"
#include "posetrak/core/observation.hpp"
#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/state.hpp"
#include "posetrak/filters/update_result.hpp"
#include "posetrak/kinematics/forward_kinematics.hpp"

namespace posetrak {

/**
 * @brief Configuration for Rerun logging
 */
struct RerunConfig {
    bool enabled = false;                              ///< Enable Rerun logging
    std::optional<std::filesystem::path> output_path;  ///< Save to .rrd file
    bool live_streaming = false;                       ///< Stream to Rerun viewer
    std::string live_address = "127.0.0.1:9876";       ///< Viewer TCP address

    // Visualization options
    bool log_camera_images = false;                    ///< Log camera images (if available)
    bool log_sigma_points = false;                     ///< Log UKF sigma points (expensive)
    bool log_covariance_ellipsoids = false;            ///< Log 3D uncertainty ellipsoids
    bool log_reprojection_vectors = true;              ///< Log 2D error vectors
    bool log_joint_frames = false;                     ///< Log coordinate frame at each joint

    // Performance
    int log_every_n_frames = 1;                        ///< Subsample logging (1=every frame)

    // Recording metadata
    std::string application_id = "posetrak";
    std::string recording_id;                          ///< Auto-generated if empty
};

/**
 * @brief Rerun visualization logger for motion capture tracking
 *
 * Logs tracking data to Rerun for interactive 3D/2D visualization and debugging.
 * Maintains a hierarchical entity structure under world/ namespace.
 *
 * Lifecycle:
 * 1. Construct with config and metadata (skeleton, cameras)
 * 2. Call log_initialization() once after tracker init
 * 3. For each frame:
 *    - log_frame_start()
 *    - log_observations() (raw OpenPose data)
 *    - log_prediction() (after UKF predict)
 *    - log_update() (after UKF update)
 *    - log_frame_end() (metrics, state)
 * 4. Destructor auto-flushes and closes recording
 */
class RerunLogger {
public:
    /**
     * @brief Construct logger with configuration
     *
     * @param config Rerun-specific configuration
     * @param skeleton Skeleton model (for marker/joint info)
     * @param cameras Map of camera_id -> Camera (for extrinsics)
     */
    RerunLogger(RerunConfig const& config,
                Skeleton const& skeleton,
                std::unordered_map<int, Camera> const& cameras);

    /**
     * @brief Destructor - flushes and closes recording
     */
    ~RerunLogger();

    // ========================================================================
    // Initialization (call once)
    // ========================================================================

    /**
     * @brief Log static metadata and scene setup
     *
     * Logs:
     * - Camera extrinsics and pinhole models
     * - Skeleton rest pose
     * - Marker definitions
     * - Time series blueprint
     */
    void log_initialization();

    /**
     * @brief Log initial state after tracker initialization
     *
     * @param initial_state State from triangulation + IK
     * @param marker_positions_3d Initial marker positions
     */
    void log_initial_state(State const& initial_state,
                           std::map<std::string, Eigen::Vector3d> const& marker_positions_3d);

    // ========================================================================
    // Per-Frame Logging
    // ========================================================================

    /**
     * @brief Mark start of new frame
     *
     * Sets Rerun timeline to current frame/timestamp.
     *
     * @param frame Frame number (0-indexed)
     * @param timestamp Timestamp in seconds
     */
    void log_frame_start(int frame, double timestamp);

    /**
     * @brief Log raw observations from all cameras
     *
     * Logs to world/cameras/cam_X/observations/{raw,labels}
     *
     * @param observations All observations for this frame
     */
    void log_observations(std::vector<Observation> const& observations);

    /**
     * @brief Log predicted state after UKF predict step
     *
     * Logs:
     * - Predicted skeleton pose (3D)
     * - Predicted marker positions (3D)
     *
     * @param predicted_state State after predict()
     * @param predicted_markers Marker positions from FK
     */
    void log_prediction(State const& predicted_state,
                       std::map<std::string, Eigen::Vector3d> const& predicted_markers);

    /**
     * @brief Log UKF update diagnostics
     *
     * Logs:
     * - Inlier/outlier observations (color-coded)
     * - Predicted 2D projections
     * - Reprojection error vectors
     * - Mahalanobis distances
     *
     * @param observations All observations
     * @param update_result Update diagnostics from UKF
     * @param fk Forward kinematics (for projections)
     */
    void log_update(std::vector<Observation> const& observations,
                   UpdateResult const& update_result,
                   ForwardKinematics& fk);

    /**
     * @brief Log final posterior state and metrics
     *
     * Logs:
     * - Posterior skeleton pose (3D)
     * - Joint angles/velocities (tensor)
     * - Tracking quality metrics (scalars)
     * - Covariance diagnostics
     *
     * @param posterior_state Final state after update
     * @param covariance State covariance matrix
     * @param update_result Update diagnostics
     * @param marker_positions_3d Final marker positions
     */
    void log_frame_end(State const& posterior_state,
                      Eigen::MatrixXd const& covariance,
                      UpdateResult const& update_result,
                      std::map<std::string, Eigen::Vector3d> const& marker_positions_3d);

    // ========================================================================
    // Optional: Advanced Diagnostics
    // ========================================================================

    /**
     * @brief Log UKF sigma points (3D cloud)
     *
     * Only if config.log_sigma_points = true (expensive)
     *
     * @param sigma_point_states List of sigma point states
     */
    void log_sigma_points(std::vector<State> const& sigma_point_states);

    /**
     * @brief Log camera images with overlays
     *
     * Only if config.log_camera_images = true
     *
     * @param camera_id Camera ID
     * @param image Image data (RGB or grayscale)
     */
    void log_camera_image(int camera_id,
                         cv::Mat const& image);  // Requires OpenCV

    /**
     * @brief Check if logging is enabled
     */
    bool enabled() const { return config_.enabled; }

    /**
     * @brief Manually flush recording (normally automatic)
     */
    void flush();

private:
    RerunConfig config_;
    Skeleton const& skeleton_;
    std::unordered_map<int, Camera> const& cameras_;

    std::unique_ptr<rerun::RecordingStream> rec_;

    int current_frame_ = -1;
    double current_timestamp_ = 0.0;
    int frames_logged_ = 0;

    // Cached data for trajectory visualization
    std::vector<Eigen::Vector3d> root_trajectory_;

    // Helper methods
    void setup_recording();
    void setup_timelines();

    void log_skeleton_3d(std::string const& entity_path,
                        State const& state,
                        std::map<std::string, Eigen::Vector3d> const& marker_positions);

    void log_observations_2d(int camera_id,
                            std::vector<Observation> const& camera_obs,
                            UpdateResult const* update_result = nullptr);

    void log_metrics(UpdateResult const& update_result,
                    Eigen::MatrixXd const& covariance);

    void log_reprojection_errors_2d(int camera_id,
                                   std::vector<Observation> const& observations,
                                   UpdateResult const& update_result,
                                   State const& state,
                                   ForwardKinematics& fk);

    Eigen::Vector3d get_joint_position_3d(State const& state,
                                         std::string const& joint_name);

    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
        get_bone_segments(State const& state);
};

}  // namespace posetrak
```

---

## Integration Points

### 1. TrackerConfig Extension

**File**: `include/posetrak/core/config.hpp`

```cpp
struct TrackerConfig {
    // ... existing fields ...

    // Visualization
    RerunConfig rerun;  ///< Rerun visualization configuration
};

struct TrackerAppConfig {
    // ... existing fields ...

    // Add [rerun] section parsing in load()
};
```

### 2. Tracker Class Integration

**File**: `include/posetrak/tracking/tracker.hpp`

```cpp
class Tracker {
public:
    // ... existing methods ...

    /**
     * @brief Set Rerun logger (optional)
     */
    void set_rerun_logger(std::shared_ptr<RerunLogger> logger);

private:
    std::shared_ptr<RerunLogger> rerun_logger_;
};
```

**File**: `src/tracking/tracker.cpp`

```cpp
bool Tracker::initialize(/* ... */) {
    // ... existing initialization ...

    // Log initialization
    if (rerun_logger_ && rerun_logger_->enabled()) {
        rerun_logger_->log_initialization();
        rerun_logger_->log_initial_state(state_, initial_marker_positions);
    }

    return true;
}

TrackingResult Tracker::track_frame(/* ... */) {
    // Frame start
    if (rerun_logger_ && rerun_logger_->enabled()) {
        rerun_logger_->log_frame_start(current_frame_, timestamp);
        rerun_logger_->log_observations(observations);
    }

    // Predict
    ukf_->predict(dt);

    if (rerun_logger_ && rerun_logger_->enabled()) {
        auto predicted_markers = fk_->compute(ukf_->state());
        rerun_logger_->log_prediction(ukf_->state(), predicted_markers);
    }

    // Update
    auto update_result = ukf_->update(observations, cameras_, *fk_);

    if (rerun_logger_ && rerun_logger_->enabled()) {
        rerun_logger_->log_update(observations, update_result, *fk_);
    }

    // Frame end
    if (rerun_logger_ && rerun_logger_->enabled()) {
        auto marker_positions = fk_->compute(ukf_->state());
        rerun_logger_->log_frame_end(ukf_->state(), ukf_->covariance(),
                                     update_result, marker_positions);
    }

    // ... rest of existing code ...
}
```

### 3. CLI Application Integration

**File**: `cli/track.cpp`

```cpp
int main(int argc, char* argv[]) {
    // ... parse config ...

    // Create Rerun logger if enabled
    std::shared_ptr<RerunLogger> rerun_logger;
    if (config.rerun.enabled) {
        rerun_logger = std::make_shared<RerunLogger>(
            config.rerun, skeleton, cameras
        );
        tracker.set_rerun_logger(rerun_logger);
    }

    // ... run tracking ...
}
```

---

## Configuration File Format

**File**: `example_config.toml`

```toml
[rerun]
enabled = true
output_path = "tracking_output/tracking.rrd"
live_streaming = false
live_address = "127.0.0.1:9876"

# Visualization options
log_camera_images = false          # Requires image loading
log_sigma_points = false           # Expensive (117 points per frame)
log_covariance_ellipsoids = false  # Requires eigendecomposition
log_reprojection_vectors = true    # Show prediction errors
log_joint_frames = false           # Show coordinate frames

# Performance
log_every_n_frames = 1             # Log every frame (1) or subsample (>1)

# Metadata
application_id = "posetrak"
recording_id = "kotegaeshi_run_001"
```

---

## Data Flow Example

### Tracking a Single Frame

```
Frame 42, t=1.400s
─────────────────────────────────────────────────────────────

1. tracker.track_frame(obs, 1.400)
   │
   ├─► rerun_logger.log_frame_start(42, 1.400)
   │     • Sets timeline: frame=42, timestamp=1.400
   │
   ├─► rerun_logger.log_observations(obs)
   │     • world/cameras/cam_0/observations/raw ← gray points
   │     • world/cameras/cam_1/observations/raw ← gray points
   │     • world/cameras/cam_2/observations/raw ← gray points
   │     • world/cameras/cam_3/observations/raw ← gray points
   │
   ├─► ukf.predict(dt)
   │
   ├─► rerun_logger.log_prediction(predicted_state, predicted_markers)
   │     • world/skeleton/predicted/joints ← yellow points
   │     • world/skeleton/predicted/bones ← orange lines
   │     • world/markers/fk_predicted ← purple points
   │
   ├─► ukf.update(obs, cameras, fk)
   │
   ├─► rerun_logger.log_update(obs, update_result, fk)
   │     • world/cameras/cam_0/observations/inliers ← green points
   │     • world/cameras/cam_0/observations/outliers ← red points
   │     • world/cameras/cam_0/observations/predicted ← blue points
   │     • world/cameras/cam_0/reprojection_vectors ← blue lines
   │     • (repeat for cam_1, cam_2, cam_3)
   │
   └─► rerun_logger.log_frame_end(state, cov, update_result, markers)
         • world/skeleton/current/joints ← yellow points
         • world/skeleton/current/bones ← orange lines
         • world/markers/fk_predicted ← purple points
         • world/tracking_quality/reprojection_error ← scalar(2.5)
         • world/tracking_quality/num_inliers ← scalar(285)
         • world/tracking_quality/num_outliers ← scalar(20)
         • world/tracking_quality/nis ← scalar(450.3)
         • world/state/joint_angles ← tensor([...])
```

---

## Entity Paths Detail

### Camera Observations (2D)

```
world/cameras/cam_0/
├── pinhole                         [Pinhole] Camera intrinsics (logged once)
├── transform                       [Transform3D] Camera pose (logged once)
├── observations/
│   ├── raw                         [Points2D] All detections (gray, radius=3px)
│   ├── inliers                     [Points2D] Accepted by UKF (green, radius=4px)
│   ├── outliers                    [Points2D] Rejected (red, radius=4px)
│   ├── predicted                   [Points2D] UKF prediction (blue, radius=3px)
│   └── labels                      [TextEntry] Marker names
└── reprojection_vectors            [LineStrips2D] actual→predicted (blue)
```

### Skeleton (3D)

```
world/skeleton/
├── rest_pose/                      (logged once at initialization)
│   ├── joints                      [Points3D] Joint positions (gray, radius=0.02m)
│   ├── bones                       [LineStrips3D] Bone connections (gray)
│   └── labels                      [TextEntry] Joint names
├── predicted/                      (after predict step)
│   ├── joints                      [Points3D] Predicted joints (yellow, radius=0.02m)
│   └── bones                       [LineStrips3D] Predicted bones (orange)
├── current/                        (after update step)
│   ├── joints                      [Points3D] Posterior joints (green, radius=0.02m)
│   ├── bones                       [LineStrips3D] Posterior bones (green)
│   ├── root_frame                  [Transform3D] Root coordinate frame
│   └── joint_frames/               [Transform3D] Per-joint frames (optional)
│       ├── pelvis
│       ├── spine
│       └── ...
└── history/
    └── root_trajectory             [LineStrips3D] Root path (blue trail)
```

### Markers (3D)

```
world/markers/
├── fk_predicted                    [Points3D] From FK (purple, radius=0.015m)
├── triangulated                    [Points3D] From multi-view (cyan, optional)
├── labels                          [TextEntry] Marker names
└── errors                          [LineStrips3D] predicted→triangulated (optional)
```

### Tracking Quality Metrics (Time Series)

```
world/tracking_quality/
├── reprojection_error              [Scalar] Mean error (pixels)
├── max_reprojection_error          [Scalar] Max error (pixels)
├── num_inliers                     [Scalar] Inlier count
├── num_outliers                    [Scalar] Outlier count
├── outlier_rate                    [Scalar] Percentage (0-100)
├── nis                             [Scalar] Normalized Innovation Squared
├── covariance_condition            [Scalar] Condition number (log scale)
└── covariance_min_eigenvalue       [Scalar] Min eigenvalue
```

### State (Time Series)

```
world/state/
├── joint_angles                    [Tensor] All angles (DOF-dimensional)
├── joint_velocities                [Tensor] All velocities
├── root_position                   [Vec3D] Root position
└── root_orientation                [Quaternion] Root orientation
```

---

## Color Scheme

| Entity | Color | Rationale |
|--------|-------|-----------|
| Raw observations | Gray (200,200,200) | Neutral, background |
| Inlier observations | Green (0,255,0) | Good/accepted |
| Outlier observations | Red (255,0,0) | Bad/rejected |
| Predicted projections | Blue (0,0,255) | Prediction |
| Reprojection vectors | Blue (0,0,255) | Error direction |
| Skeleton joints (predicted) | Yellow (255,255,0) | Prior state |
| Skeleton bones (predicted) | Orange (255,165,0) | Prior state |
| Skeleton joints (posterior) | Green (0,255,0) | Updated state |
| Skeleton bones (posterior) | Green (0,200,0) | Updated state |
| Markers (FK) | Purple (200,0,200) | Derived from state |
| Root trajectory | Blue (0,100,255) | Motion history |
| Camera frustums | Cyan (0,255,255) | Scene context |

---

## Performance Considerations

### Overhead Estimates

| Operation | Cost | Frequency | Impact |
|-----------|------|-----------|--------|
| log_frame_start() | ~0.1ms | Per frame | Negligible |
| log_observations() | ~1-2ms | Per frame | Low |
| log_prediction() | ~3-5ms | Per frame | Low |
| log_update() | ~5-10ms | Per frame | Medium |
| log_frame_end() | ~5-10ms | Per frame | Medium |
| log_sigma_points() | ~50-100ms | Per frame (opt) | High |
| log_camera_image() | ~10-50ms | Per frame (opt) | High |
| **Total (basic)** | ~15-30ms | Per frame | ~5-10% @ 30Hz |
| **Total (full)** | ~75-180ms | Per frame | ~30-50% @ 30Hz |

### Optimization Strategies

1. **Subsampling**: `log_every_n_frames = 10` → visualize every 10th frame
2. **Conditional features**: Disable expensive options by default
3. **Async logging**: Rerun SDK supports background flushing
4. **Data reduction**:
   - Log 3D skeleton but skip 2D overlays (much faster)
   - Log metrics but skip spatial data
5. **Recording sessions**: Log full detail for 100 frames, then sparse for rest

---

## Implementation Phases

### Phase 1: Core Infrastructure (2-3 days)
- [ ] Add Rerun C++ SDK dependency to meson.build
- [ ] Create `RerunLogger` class skeleton
- [ ] Implement `log_initialization()` (cameras, skeleton)
- [ ] Implement `log_frame_start()` and timeline management
- [ ] Basic 3D skeleton logging (`log_skeleton_3d`)
- [ ] Test with simple synthetic sequence

**Deliverable**: Can view 3D skeleton animation in Rerun viewer

### Phase 2: 2D Observations (2-3 days)
- [ ] Implement `log_observations()` (raw 2D points)
- [ ] Implement `log_update()` (inliers/outliers color-coded)
- [ ] Add reprojection error vectors
- [ ] Multi-camera view synchronization

**Deliverable**: Can see 2D observations per camera, color-coded by outlier status

### Phase 3: Metrics & Diagnostics (1-2 days)
- [ ] Implement `log_frame_end()` (metrics)
- [ ] Time series plots (reprojection error, NIS, etc.)
- [ ] Covariance diagnostics
- [ ] Joint angle tensor logging

**Deliverable**: Interactive time series plots alongside 3D view

### Phase 4: Integration & Testing (1-2 days)
- [ ] Integrate into Tracker class
- [ ] Add CLI flag `--rerun output.rrd`
- [ ] Configuration file parsing
- [ ] Test with real data (kotegaeshi sequence)
- [ ] Documentation and examples

**Deliverable**: Full working integration with CLI tool

### Phase 5: Advanced Features (Optional, 3-5 days)
- [ ] Camera image loading and overlay
- [ ] Sigma point visualization
- [ ] Covariance ellipsoids
- [ ] Comparison mode (Python vs C++ side-by-side)
- [ ] Export high-res images/videos

**Total estimated effort**: 8-15 days (depending on feature scope)

---

## Testing Strategy

### Unit Tests

```cpp
// tests/test_rerun_logger.cpp

TEST_CASE("RerunLogger initialization") {
    RerunConfig config;
    config.enabled = true;
    config.output_path = "/tmp/test.rrd";

    RerunLogger logger(config, skeleton, cameras);
    logger.log_initialization();

    // Verify recording file created
    REQUIRE(std::filesystem::exists("/tmp/test.rrd"));
}

TEST_CASE("RerunLogger frame logging") {
    // ... setup ...

    logger.log_frame_start(0, 0.0);
    logger.log_observations(observations);
    logger.log_frame_end(state, covariance, update_result, markers);

    logger.flush();

    // Verify data logged (check file size > 0, etc.)
}
```

### Integration Tests

1. **Small sequence test** (10 frames)
   - Verify all entities logged
   - Check timeline consistency
   - Validate data types

2. **Real sequence test** (kotegaeshi, 500 frames)
   - Performance check (overhead < 10%)
   - File size reasonable (< 100MB for 500 frames)
   - No crashes or memory leaks

3. **Comparison test**
   - Run Python and C++ on same data
   - Log both to separate recordings
   - Load both in Rerun, compare visually

---

## Usage Examples

### Minimal Usage (CLI)

```bash
# Track with Rerun logging to file
./posetrak track config.toml --rerun output.rrd

# View results
rerun output.rrd
```

### Live Streaming

```bash
# Terminal 1: Start Rerun viewer
rerun

# Terminal 2: Track with live streaming
./posetrak track config.toml --rerun-live
```

### Programmatic Usage (C++)

```cpp
// Create config
RerunConfig rerun_config;
rerun_config.enabled = true;
rerun_config.output_path = "results.rrd";
rerun_config.log_reprojection_vectors = true;

// Create logger
auto logger = std::make_shared<RerunLogger>(
    rerun_config, skeleton, cameras
);

// Attach to tracker
tracker.set_rerun_logger(logger);

// Run tracking (logger is automatically called)
tracker.initialize(initial_observations, 0.0);
for (auto const& [timestamp, obs] : observation_sequence) {
    tracker.track_frame(obs, timestamp);
}

// Logger flushes on destruction
```

### Python Analysis with Rerun

```python
import rerun as rr
import numpy as np

# Load C++ tracking results
rr.init("posetrak_analysis", recording_id="analysis_001")
rr.connect()  # Or rr.save("analysis.rrd")

# Load C++ recording
cpp_data = load_cpp_tracking_results("cpp_output.rrd")

# Add Python-side analysis
for frame in range(len(cpp_data)):
    rr.set_time_sequence("frame", frame)

    # Log additional analysis results
    rr.log("analysis/joint_angle_errors",
           rr.Scalar(compute_error(cpp_data[frame])))

    # Log comparison markers
    python_markers = compute_python_markers(frame)
    rr.log("comparison/python_markers",
           rr.Points3D(python_markers, colors=(255, 0, 255)))
```

---

## Dependencies

### Required

- **Rerun C++ SDK** (v0.20+)
  - Add to `meson.build` via wrap or system package
  - Headers: `<rerun.hpp>`
  - Link: `-lrerun_sdk`

### Optional

- **OpenCV** (for camera image logging)
  - Already planned for Phase 6
  - Used in `log_camera_image()`

---

## Open Questions

1. **Rerun SDK version**: Use latest (0.20+) or pin to stable version?
2. **Live streaming default**: Enable by default or opt-in?
3. **Recording size limits**: Implement auto-subsampling for long sequences (>1000 frames)?
4. **Multi-person support**: How to structure entities for multiple tracked people?
5. **Memory management**: Buffer frames before flushing or flush immediately?

---

## Future Enhancements

### Phase 6+: Advanced Visualization

- **Temporal analysis**: Heatmaps of error distribution over time
- **Marker-specific views**: Focus on individual markers with history
- **Joint angle plots**: 1D time series per joint DOF
- **Covariance animation**: Ellipsoid growing/shrinking over time
- **Camera coverage**: Visualize which cameras can see which markers

### Integration with Other Tools

- **OpenSim export**: Click marker in Rerun → export to OpenSim format
- **Blender export**: Export skeleton animation for rendering
- **Paper figures**: Screenshot tool with publication-quality settings

---

## Conclusion

This design provides a comprehensive visualization system that will greatly improve debugging, validation, and presentation of tracking results. The modular design allows incremental implementation, starting with core 3D skeleton visualization and gradually adding 2D overlays and metrics.

**Recommended priority**: Implement Phase 1-4 first (essential features), defer Phase 5 (nice-to-have) until after tracking divergence is resolved and basic validation is complete.
