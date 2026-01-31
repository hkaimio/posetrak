# Phase 7+8: Production Readiness - Detailed Plan

**Goal**: Make tracker usable for real-data testing with visualization in Jupyter/Marimo notebooks

**Estimated Time**: 7-10 days

---

## Overview

Instead of implementing video overlays in C++, we'll export comprehensive data to CSV/JSON that can be visualized in Python notebooks. This is faster to implement and more flexible for analysis.

---

## Part 1: Data Export Layer (3-4 days)

### 1.1: Tracking Results Export (Day 1)

**Goal**: Export all tracking results to structured files for analysis

#### Files to Create
- `include/posetrak/io/tracking_export.hpp`
- `src/io/tracking_export.cpp`

#### Export Format: CSV Per-Frame

**File 1: `tracking_results.csv`**
```csv
frame,timestamp,marker_id,marker_name,x_3d,y_3d,z_3d,is_visible
0,0.0000,0,pelvis_center,0.0,0.0,0.0,true
0,0.0000,1,spine_base,-0.007,0.150,0.001,true
...
```

**File 2: `joint_angles.csv`**
```csv
frame,timestamp,joint_name,angle_x,angle_y,angle_z,velocity_x,velocity_y,velocity_z
0,0.0000,pelvis,0.0,0.0,0.0,0.0,0.0,0.0
0,0.0000,spine,0.0,0.15,0.0,0.0,0.0,0.0
...
```

**File 3: `root_pose.csv`**
```csv
frame,timestamp,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z,vel_x,vel_y,vel_z,omega_x,omega_y,omega_z
0,0.0000,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
...
```

**File 4: `marker_projections.csv`**
```csv
frame,timestamp,marker_id,marker_name,camera_id,proj_x,proj_y,obs_x,obs_y,error_x,error_y,is_outlier
0,0.0000,0,pelvis_center,0,640.5,360.2,640.3,360.1,0.2,0.1,false
0,0.0000,0,pelvis_center,1,580.1,370.5,580.0,370.6,-0.1,0.1,false
...
```

**File 5: `observations.csv`**
```csv
frame,timestamp,marker_id,marker_name,camera_id,pixel_x,pixel_y,confidence,used_in_tracking
0,0.0000,0,pelvis_center,0,640.3,360.1,0.95,true
0,0.0000,1,spine_base,0,645.2,280.5,0.89,true
...
```

#### Implementation Tasks

```cpp
class TrackingExporter {
public:
    TrackingExporter(std::filesystem::path const& output_dir,
                     Skeleton const& skeleton,
                     std::unordered_map<int, Camera> const& cameras);

    // Open all CSV files for writing
    void open();

    // Write a single frame's results
    void write_frame(
        int frame_number,
        double timestamp,
        State const& state,
        std::map<std::string, Eigen::Vector3d> const& marker_positions_3d,
        std::vector<Observation> const& observations,
        UpdateResult const& update_result
    );

    // Close all files
    void close();

private:
    std::ofstream tracking_results_;
    std::ofstream joint_angles_;
    std::ofstream root_pose_;
    std::ofstream marker_projections_;
    std::ofstream observations_;
};
```

**Tasks**:
- [ ] Create `TrackingExporter` class
- [ ] Implement CSV writing for each file type
- [ ] Handle marker visibility (markers not in view)
- [ ] Compute projection errors (predicted vs observed)
- [ ] Write headers with column names
- [ ] Proper CSV escaping and formatting
- [ ] Tests: export sample tracking sequence, verify CSV format

**Exit Criteria**:
- ✅ Can export tracking results to 5 CSV files
- ✅ CSVs load correctly in pandas
- ✅ All data needed for visualization is present
- ✅ Tests pass

---

### 1.2: Summary Statistics Export (Day 1-2)

**Goal**: Export tracking quality metrics

**File: `tracking_stats.csv`**
```csv
frame,timestamp,num_observations,num_inliers,num_outliers,mean_reprojection_error,max_reprojection_error,covariance_min_eigenvalue,covariance_condition_number,nis_value,tracking_lost
0,0.0000,24,24,0,1.2,2.5,1.5e-5,1.2e6,18.5,false
1,0.0333,24,23,1,1.5,3.2,1.4e-5,1.3e6,22.1,false
...
```

**File: `overall_stats.json`**
```json
{
  "sequence_name": "person01_walk",
  "total_frames": 500,
  "frames_tracked": 498,
  "frames_lost": 2,
  "mean_reprojection_error": 2.1,
  "mean_num_inliers": 22.5,
  "outlier_rate": 0.08,
  "processing_time_ms": 1250.0,
  "fps": 400.0,
  "skeleton": {
    "name": "human_120dof",
    "num_joints": 45,
    "num_markers": 25
  },
  "cameras": [
    {"id": 0, "name": "cam1"},
    {"id": 1, "name": "cam2"}
  ]
}
```

#### Implementation

```cpp
class StatisticsTracker {
public:
    void add_frame_stats(int frame, double timestamp, UpdateResult const& result);

    void write_frame_stats(std::filesystem::path const& output_path);
    void write_summary_stats(std::filesystem::path const& output_path,
                             nlohmann::json const& metadata);

    // Get statistics for display
    double mean_reprojection_error() const;
    double mean_num_inliers() const;
    double outlier_rate() const;
};
```

**Tasks**:
- [ ] Create `StatisticsTracker` class
- [ ] Accumulate per-frame statistics
- [ ] Compute reprojection errors
- [ ] Track covariance condition numbers
- [ ] Write CSV and JSON outputs
- [ ] Tests: verify statistics calculation

**Exit Criteria**:
- ✅ Per-frame statistics CSV generated
- ✅ Summary JSON with overall metrics
- ✅ Tests pass

---

### 1.3: Debug Information Export (Day 2)

**Goal**: Export detailed information for debugging tracking issues

**File: `debug_ukf.csv`**
```csv
frame,timestamp,phase,min_eigenvalue,max_eigenvalue,condition_number,innovation_norm,kalman_gain_norm,state_correction_norm
0,0.0000,before_update,2.5e-5,1.8e-2,720.0,15.2,0.8,0.05
0,0.0000,after_update,2.3e-5,1.9e-2,826.1,0.0,0.0,0.0
...
```

**File: `debug_ik_initialization.csv`** (already exists at `/tmp/ik_iterations.csv`)
- Move to output directory
- Add frame number if re-initializing

**File: `debug_outliers.csv`**
```csv
frame,timestamp,marker_id,marker_name,camera_id,mahalanobis_distance,threshold,is_outlier,reason
0,0.0000,5,left_wrist,2,8.5,4.0,true,high_mahalanobis
0,0.0000,12,right_ankle,0,0.0,4.0,true,projection_failed
...
```

#### Implementation

Add to `TrackingExporter`:
```cpp
void write_debug_ukf(int frame, double timestamp,
                     Eigen::MatrixXd const& covariance_before,
                     Eigen::MatrixXd const& covariance_after,
                     Eigen::VectorXd const& innovation,
                     Eigen::MatrixXd const& kalman_gain);

void write_debug_outliers(int frame, double timestamp,
                          std::vector<ObservationResult> const& results,
                          double threshold);
```

**Tasks**:
- [ ] Export UKF internal state (covariance, innovation, gain)
- [ ] Export outlier detection details
- [ ] Copy IK debug files to output directory
- [ ] Tests: verify debug output format

**Exit Criteria**:
- ✅ Debug CSVs generated
- ✅ Sufficient detail for troubleshooting
- ✅ Tests pass

---

## Part 2: CLI Tool (2-3 days)

### 2.1: Configuration System (Day 3)

**Goal**: Load all parameters from configuration file

**File: `config.toml`**
```toml
[data]
skeleton = "skeletons/human_120dof.yaml"
cameras = "calibration/cameras.toml"
sync = "calibration/sync.json"
observations_dir = "openpose_output"
person_id = 0

[tracking]
process_noise_std = 0.5
measurement_noise_std = 2.0
outlier_threshold = 4.0

[tracking.initialization]
ik_max_iterations = 1000
ik_tolerance = 0.02
init_position_std = 0.1
init_orientation_std = 0.1
init_joint_std = 0.1
init_velocity_std = 0.1
min_cameras_for_init = 2

[tracking.ukf]
alpha = 0.5
beta = 2.0
kappa = 0.0

[output]
directory = "tracking_output"
export_tracking_results = true
export_statistics = true
export_debug = false

[processing]
start_frame = 0
max_frames = -1  # -1 means all frames
```

#### Files to Create
- `include/posetrak/core/config.hpp`
- `src/core/config.cpp`

#### Implementation

```cpp
struct TrackerConfig {
    // Data paths
    std::filesystem::path skeleton_path;
    std::filesystem::path cameras_path;
    std::filesystem::path sync_path;
    std::filesystem::path observations_dir;
    int person_id = 0;

    // Tracking parameters
    double process_noise_std = 0.5;
    double measurement_noise_std = 2.0;
    double outlier_threshold = 4.0;

    // Initialization
    int ik_max_iterations = 1000;
    double ik_tolerance = 0.02;
    double init_position_std = 0.1;
    double init_orientation_std = 0.1;
    double init_joint_std = 0.1;
    double init_velocity_std = 0.1;
    int min_cameras_for_init = 2;

    // UKF parameters
    double ukf_alpha = 0.5;
    double ukf_beta = 2.0;
    double ukf_kappa = 0.0;

    // Output
    std::filesystem::path output_dir = "tracking_output";
    bool export_tracking_results = true;
    bool export_statistics = true;
    bool export_debug = false;

    // Processing
    int start_frame = 0;
    int max_frames = -1;  // -1 = all frames

    // Load from TOML file
    static TrackerConfig load(std::filesystem::path const& config_path);

    // Validate configuration
    void validate() const;
};
```

**Tasks**:
- [ ] Create `TrackerConfig` struct with all parameters
- [ ] Implement TOML loading with toml11
- [ ] Implement validation (check files exist, parameters in valid ranges)
- [ ] Default values for all optional parameters
- [ ] Tests: load valid config, detect invalid configs

**Exit Criteria**:
- ✅ Can load configuration from TOML file
- ✅ Validation catches common errors
- ✅ All tracking parameters configurable
- ✅ Tests pass

---

### 2.2: CLI Application (Day 4)

**Goal**: Command-line tool for tracking

#### Files to Create
- `cli/track.cpp` (main CLI application)
- `cli/meson.build`

#### Implementation

```cpp
int main(int argc, char* argv[]) {
    CLI::App app{"Posetrak - Motion Capture Tracker"};

    std::string config_path;
    app.add_option("config", config_path, "Configuration file (TOML)")
        ->required()
        ->check(CLI::ExistingFile);

    bool verbose = false;
    app.add_flag("-v,--verbose", verbose, "Verbose output");

    bool quiet = false;
    app.add_flag("-q,--quiet", quiet, "Quiet mode (minimal output)");

    CLI11_PARSE(app, argc, argv);

    try {
        // Load configuration
        auto config = TrackerConfig::load(config_path);
        config.validate();

        // Load skeleton, cameras, observations
        // Initialize tracker
        // Run tracking with progress reporting
        // Export results

    } catch (std::exception const& e) {
        fmt::print(stderr, "Error: {}\n", e.what());
        return 1;
    }

    return 0;
}
```

**Features**:
- [ ] Load config file
- [ ] Progress reporting (frame N/M, percentage, ETA)
- [ ] Error handling with user-friendly messages
- [ ] Verbose mode (show per-frame statistics)
- [ ] Quiet mode (only show errors)
- [ ] Return exit codes (0=success, 1=error, 2=tracking lost)

**Progress Output**:
```
Loading skeleton: skeletons/human_120dof.yaml (120 DOF)
Loading cameras: calibration/cameras.toml (4 cameras)
Loading observations: openpose_output/ (500 frames detected)

Initializing from frame 0...
  Triangulated 24/25 markers
  IK converged: 0.025m RMS error in 234 iterations
  Initial pose set

Tracking: [====================>         ] 250/500 (50.0%) | 12.5 fps | ETA: 20s
  Frame 250: 23 inliers, 1 outlier, 1.8px reprojection error

Tracking complete!
  Tracked: 498/500 frames (99.6%)
  Lost: 2 frames
  Mean reprojection error: 2.1px
  Outlier rate: 8.2%
  Processing time: 40.2s (12.4 fps)

Results exported to: tracking_output/
  - tracking_results.csv
  - joint_angles.csv
  - root_pose.csv
  - marker_projections.csv
  - observations.csv
  - tracking_stats.csv
  - overall_stats.json
```

**Tasks**:
- [ ] Implement CLI with CLI11
- [ ] Progress bar using fmt
- [ ] Per-frame statistics display (verbose mode)
- [ ] Final summary with key metrics
- [ ] List exported files
- [ ] Error handling with actionable messages

**Exit Criteria**:
- ✅ CLI compiles and runs
- ✅ Can track sequence from config file
- ✅ Progress reporting works
- ✅ Results exported to specified directory

---

### 2.3: Example Configuration & Documentation (Day 4)

**Goal**: Make it easy to get started

#### Files to Create
- `examples/config_template.toml` (commented template)
- `examples/README.md` (how to use)
- `docs/cli-usage.md` (full documentation)

**Example Template**:
```toml
# Posetrak Tracking Configuration
# Copy this file and modify paths for your data

[data]
# Path to skeleton definition (YAML format)
skeleton = "path/to/skeleton.yaml"

# Path to camera calibration (Pose2Sim TOML format)
cameras = "path/to/cameras.toml"

# Path to sync metadata (JSON format, optional)
# sync = "path/to/sync.json"

# Directory containing OpenPose JSON files
# Structure: observations_dir/cam_name/frame_NNNNNN_keypoints.json
observations_dir = "path/to/openpose_output"

# Person ID to track (0 = first person in OpenPose output)
person_id = 0

[tracking]
# Process noise (higher = trust measurements more, lower = trust model more)
# Default: 0.5 (good for walking, general motion)
# Increase to 1.0+ for fast/unpredictable motion
process_noise_std = 0.5

# Measurement noise (pixels)
# Typical: 2.0 for OpenPose detections
measurement_noise_std = 2.0

# Outlier threshold (Mahalanobis distance)
# Higher = accept more observations, lower = reject more outliers
# Typical range: 3.0-5.0
outlier_threshold = 4.0

# ... rest of config with comments ...
```

**Documentation**:
- [ ] Configuration file reference
- [ ] Parameter tuning guide
- [ ] Examples for different use cases
- [ ] Troubleshooting common issues

**Exit Criteria**:
- ✅ Template config file with all options commented
- ✅ Usage documentation complete
- ✅ Examples for common scenarios

---

## Part 3: Real Data Testing (2-3 days)

### 3.1: Test Data Preparation (Day 5)

**Goal**: Set up real OpenPose data for testing

**Tasks**:
- [ ] Choose 2-3 test sequences:
  - Simple: standing person, minimal motion
  - Medium: walking, periodic motion
  - Complex: reaching, bending, varied motion
- [ ] Ensure camera calibration available
- [ ] Ensure sync metadata available (if needed)
- [ ] Verify OpenPose detections present
- [ ] Create configuration files for each sequence

**Test Sequences Directory Structure**:
```
test_data/
├── simple_standing/
│   ├── config.toml
│   ├── skeleton.yaml
│   ├── cameras.toml
│   ├── sync.json (optional)
│   └── openpose/
│       ├── cam1/
│       ├── cam2/
│       └── cam3/
├── medium_walking/
│   └── ... (same structure)
└── complex_reaching/
    └── ... (same structure)
```

**Exit Criteria**:
- ✅ 2-3 test sequences prepared
- ✅ Configuration files created
- ✅ All required files present

---

### 3.2: Initial Tracking Tests (Day 5-6)

**Goal**: Run tracker on real data and verify basic functionality

**Tasks**:
- [ ] Run CLI on each test sequence
- [ ] Verify tracking completes without crashes
- [ ] Check for common errors:
  - Missing observations
  - Camera calibration issues
  - Synchronization problems
  - IK initialization failures
- [ ] Export results to CSV
- [ ] Verify CSV files are well-formed

**Expected Issues** (to debug):
- IK may not converge on first frame
- Outlier rejection may be too aggressive/lenient
- Process noise may need tuning per sequence
- Synchronization may be incorrect

**Exit Criteria**:
- ✅ Tracker runs on all test sequences
- ✅ No crashes or exceptions
- ✅ CSV files generated
- ✅ Basic sanity checks pass (joint angles in reasonable ranges)

---

### 3.3: Visualization Notebook (Day 6-7)

**Goal**: Create Jupyter/Marimo notebook for result visualization

#### Files to Create
- `notebooks/visualize_tracking.py` (Marimo notebook)
- `notebooks/requirements.txt`

#### Notebook Sections

**1. Load Results**
```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Load CSVs
tracking_results = pd.read_csv("tracking_output/tracking_results.csv")
joint_angles = pd.read_csv("tracking_output/joint_angles.csv")
root_pose = pd.read_csv("tracking_output/root_pose.csv")
marker_projections = pd.read_csv("tracking_output/marker_projections.csv")
observations = pd.read_csv("tracking_output/observations.csv")
stats = pd.read_csv("tracking_output/tracking_stats.csv")
```

**2. Tracking Quality Metrics**
```python
# Plot reprojection errors over time
fig = px.line(stats, x='frame', y='mean_reprojection_error',
              title='Mean Reprojection Error Over Time')
fig.show()

# Plot number of inliers/outliers
fig = go.Figure()
fig.add_trace(go.Scatter(x=stats['frame'], y=stats['num_inliers'],
                         name='Inliers'))
fig.add_trace(go.Scatter(x=stats['frame'], y=stats['num_outliers'],
                         name='Outliers'))
fig.update_layout(title='Observations Per Frame')
fig.show()

# Covariance condition number (numerical stability indicator)
fig = px.line(stats, x='frame', y='covariance_condition_number',
              title='Covariance Condition Number (Numerical Stability)',
              log_y=True)
fig.show()
```

**3. 3D Skeleton Visualization**
```python
def plot_skeleton_3d(frame_num):
    """Plot 3D skeleton for given frame"""
    frame_data = tracking_results[tracking_results['frame'] == frame_num]

    # Plot markers
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=frame_data['x_3d'],
        y=frame_data['y_3d'],
        z=frame_data['z_3d'],
        mode='markers+text',
        text=frame_data['marker_name'],
        marker=dict(size=5)
    ))

    # Add skeleton connections (if bone structure available)
    # ...

    fig.update_layout(
        scene=dict(aspectmode='data'),
        title=f'3D Skeleton - Frame {frame_num}'
    )
    return fig

# Interactive frame selector
frame_slider = marimo.ui.slider(0, len(stats)-1, value=0, label="Frame")
plot_skeleton_3d(frame_slider.value)
```

**4. Camera View Overlays**
```python
def plot_camera_view(frame_num, camera_id):
    """Plot observations and projections for one camera"""
    # Get observations for this frame/camera
    obs = observations[
        (observations['frame'] == frame_num) &
        (observations['camera_id'] == camera_id)
    ]

    # Get projections for this frame/camera
    proj = marker_projections[
        (marker_projections['frame'] == frame_num) &
        (marker_projections['camera_id'] == camera_id)
    ]

    fig = go.Figure()

    # Plot observations (detected keypoints)
    fig.add_trace(go.Scatter(
        x=obs['pixel_x'], y=obs['pixel_y'],
        mode='markers',
        name='Observations',
        marker=dict(size=10, color='blue', symbol='circle')
    ))

    # Plot projections (predicted from 3D)
    fig.add_trace(go.Scatter(
        x=proj['proj_x'], y=proj['proj_y'],
        mode='markers',
        name='Projections',
        marker=dict(size=8, color='red', symbol='x')
    ))

    # Connect with lines to show errors
    for _, row in proj.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['proj_x'], row['obs_x']],
            y=[row['proj_y'], row['obs_y']],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False
        ))

    # Highlight outliers
    outliers = proj[proj['is_outlier']]
    fig.add_trace(go.Scatter(
        x=outliers['proj_x'], y=outliers['proj_y'],
        mode='markers',
        name='Outliers',
        marker=dict(size=15, color='orange', symbol='x-open')
    ))

    fig.update_layout(
        title=f'Camera {camera_id} - Frame {frame_num}',
        xaxis_title='X (pixels)',
        yaxis_title='Y (pixels)',
        yaxis=dict(autorange='reversed'),  # Image coordinates
        height=600
    )

    return fig

# Interactive selectors
frame_selector = marimo.ui.slider(0, len(stats)-1, value=0, label="Frame")
camera_selector = marimo.ui.dropdown(
    options=observations['camera_id'].unique().tolist(),
    value=0,
    label="Camera"
)

plot_camera_view(frame_selector.value, camera_selector.value)
```

**5. Joint Angle Time Series**
```python
# Plot joint angles over time
joint_selector = marimo.ui.dropdown(
    options=joint_angles['joint_name'].unique().tolist(),
    label="Joint"
)

joint_data = joint_angles[joint_angles['joint_name'] == joint_selector.value]

fig = go.Figure()
fig.add_trace(go.Scatter(x=joint_data['frame'], y=joint_data['angle_x'],
                         name='X rotation'))
fig.add_trace(go.Scatter(x=joint_data['frame'], y=joint_data['angle_y'],
                         name='Y rotation'))
fig.add_trace(go.Scatter(x=joint_data['frame'], y=joint_data['angle_z'],
                         name='Z rotation'))

fig.update_layout(
    title=f'Joint Angles: {joint_selector.value}',
    xaxis_title='Frame',
    yaxis_title='Angle (radians)'
)
fig.show()
```

**6. Error Analysis**
```python
# Per-marker error distribution
marker_errors = marker_projections.copy()
marker_errors['error'] = np.sqrt(
    marker_errors['error_x']**2 + marker_errors['error_y']**2
)

fig = px.box(marker_errors, x='marker_name', y='error',
             title='Reprojection Error Distribution by Marker',
             labels={'error': 'Reprojection Error (pixels)'})
fig.update_xaxis(tickangle=45)
fig.show()

# Per-camera error distribution
fig = px.box(marker_errors, x='camera_id', y='error',
             title='Reprojection Error Distribution by Camera',
             labels={'error': 'Reprojection Error (pixels)'})
fig.show()
```

**Tasks**:
- [ ] Create Marimo notebook with all sections above
- [ ] Add interactive widgets for frame/camera selection
- [ ] 3D skeleton visualization
- [ ] 2D camera view overlays
- [ ] Time series plots for joint angles
- [ ] Error analysis plots
- [ ] Summary statistics display
- [ ] Export as standalone HTML for sharing

**Exit Criteria**:
- ✅ Notebook loads and displays all data
- ✅ Interactive visualizations work
- ✅ Can inspect any frame/camera
- ✅ Easy to identify tracking issues
- ✅ Useful for subjective quality assessment

---

### 3.4: Validation Against Python (Day 7)

**Goal**: Compare C++ results with Python prototype

**Tasks**:
- [ ] Track same sequence with Python prototype
- [ ] Export Python results to same CSV format
- [ ] Load both in notebook
- [ ] Compare:
  - Joint angles (RMSE per joint)
  - 3D marker positions (RMSE per marker)
  - Reprojection errors
  - Tracking success rate
- [ ] Document differences and investigate if RMSE > threshold
- [ ] Tune parameters if needed

**Comparison Metrics**:
```python
def compare_tracking_results(cpp_results, python_results):
    """Compare C++ and Python tracking results"""

    # Merge on frame and marker
    merged = cpp_results.merge(
        python_results,
        on=['frame', 'marker_name'],
        suffixes=('_cpp', '_py')
    )

    # Compute RMSE for 3D positions
    merged['error_x'] = merged['x_3d_cpp'] - merged['x_3d_py']
    merged['error_y'] = merged['y_3d_cpp'] - merged['y_3d_py']
    merged['error_z'] = merged['z_3d_cpp'] - merged['z_3d_py']
    merged['rmse_3d'] = np.sqrt(
        merged['error_x']**2 +
        merged['error_y']**2 +
        merged['error_z']**2
    )

    # Overall statistics
    print(f"Mean 3D RMSE: {merged['rmse_3d'].mean():.4f} m")
    print(f"Max 3D RMSE: {merged['rmse_3d'].max():.4f} m")
    print(f"Median 3D RMSE: {merged['rmse_3d'].median():.4f} m")

    # Per-marker RMSE
    per_marker = merged.groupby('marker_name')['rmse_3d'].agg(['mean', 'max'])
    print("\nPer-marker RMSE:")
    print(per_marker.sort_values('mean', ascending=False))

    return merged
```

**Exit Criteria**:
- ✅ Both trackers run on same data
- ✅ Results exported in comparable format
- ✅ RMSE < 5cm for 95% of frames (or document why)
- ✅ Tracking success rate similar (±5%)
- ✅ Major discrepancies investigated and explained

---

## Part 4: Documentation & Polish (Day 8-9)

### 4.1: User Documentation

**Files to Create/Update**:
- `README.md` - Quick start guide
- `docs/installation.md` - Build instructions
- `docs/usage.md` - How to track sequences
- `docs/configuration.md` - Config file reference
- `docs/output-format.md` - CSV format specification
- `docs/visualization.md` - Using the notebook
- `docs/troubleshooting.md` - Common issues

### 4.2: Code Documentation

- [ ] Add docstrings to all public APIs
- [ ] Document CSV formats in headers
- [ ] Comment complex algorithms
- [ ] Add examples to function docs

### 4.3: Error Messages

Improve error messages to be actionable:

**Before**:
```
Error: File not found
```

**After**:
```
Error: Cannot load skeleton file
  Path: skeletons/human_120dof.yaml
  Reason: File does not exist

  Please check:
  - Is the path correct in your config file?
  - Did you download the skeleton file?
  - Are you running from the correct directory?
```

### 4.4: Examples

Create example scripts/configs for common scenarios:
- [ ] Simple tracking (minimal config)
- [ ] Multi-camera tracking
- [ ] Long sequence tracking (memory optimization)
- [ ] Tracking with poor detections (tuning)

---

## Testing Strategy

### Unit Tests
- [ ] `TrackingExporter` - CSV format validation
- [ ] `StatisticsTracker` - metric calculations
- [ ] `TrackerConfig` - TOML loading and validation

### Integration Tests
- [ ] Full tracking with export
- [ ] CLI argument parsing
- [ ] Config file loading

### Real Data Tests
- [ ] 3 test sequences track successfully
- [ ] Results validate against Python
- [ ] Visualization notebook works

### Documentation Tests
- [ ] Follow README to build and run
- [ ] All examples work
- [ ] Config template is valid

---

## Success Criteria

### Functional Requirements
- ✅ Can track real OpenPose data
- ✅ Exports comprehensive results to CSV
- ✅ CLI tool is easy to use
- ✅ Configuration via TOML file
- ✅ Progress reporting during tracking
- ✅ Visualization in Jupyter/Marimo notebook

### Quality Requirements
- ✅ RMSE < 5cm vs Python (95% of frames)
- ✅ No crashes on real data
- ✅ Error messages are actionable
- ✅ Documentation covers all features
- ✅ Examples demonstrate usage

### Performance
- ✅ Tracking completes in reasonable time (< 1min for 500 frames)
- ✅ CSV export doesn't significantly slow tracking
- ✅ Memory usage < 2GB for typical sequences

---

## Deliverables Checklist

### Code
- [ ] `src/io/tracking_export.cpp` - Export tracking results
- [ ] `src/core/config.cpp` - Configuration loading
- [ ] `cli/track.cpp` - CLI application

### Documentation
- [ ] `docs/cli-usage.md` - CLI documentation
- [ ] `docs/configuration.md` - Config reference
- [ ] `docs/output-format.md` - CSV format specs
- [ ] `examples/config_template.toml` - Annotated config
- [ ] `examples/README.md` - Examples guide

### Notebooks
- [ ] `notebooks/visualize_tracking.py` - Marimo notebook
- [ ] `notebooks/requirements.txt` - Python dependencies

### Tests
- [ ] Unit tests for new classes
- [ ] Integration test with real data
- [ ] Validation comparison with Python

---

## Timeline Summary

| Day | Task | Output |
|-----|------|--------|
| 1 | Tracking results export | 5 CSV files per sequence |
| 1-2 | Statistics export | Per-frame stats + summary JSON |
| 2 | Debug export | UKF, outliers, IK debug CSVs |
| 3 | Configuration system | TOML config loading |
| 4 | CLI tool | Working command-line tracker |
| 4 | Documentation & examples | Config template + usage docs |
| 5 | Test data preparation | 3 sequences ready |
| 5-6 | Initial tracking tests | First real data results |
| 6-7 | Visualization notebook | Interactive result viewer |
| 7 | Validation vs Python | Comparison and tuning |
| 8-9 | Documentation & polish | Complete user docs |

**Total: 7-10 days**

---

## Next Immediate Steps

1. **Start with exports** (Day 1)
   - Implement `TrackingExporter` class
   - Test CSV format with existing integration test
   - Verify pandas can load outputs

2. **Add configuration** (Day 3)
   - Implement `TrackerConfig` with TOML loading
   - Update `Tracker` to accept config struct
   - Add validation

3. **Build CLI** (Day 4)
   - Create `cli/track.cpp` with CLI11
   - Wire up config → data loading → tracking → export
   - Test with synthetic data first

4. **Prepare real data** (Day 5)
   - Identify test sequences
   - Create configs
   - Run CLI and debug issues

5. **Build visualization** (Day 6-7)
   - Create Marimo notebook
   - Test with exported CSVs
   - Iterate on visualizations

**Ready to start?** Begin with implementing `TrackingExporter` class!
