# C++ Motion Capture Tracker - Architecture Overview

## 1. System Context

```
┌─────────────────────────────────────────────────────────────────────┐
│                        External World                                │
│                                                                      │
│  Input:                                   Output:                   │
│  • OpenPose JSON (2D detections)         • TRC (marker trajectories) │
│  • Camera calibration (TOML)             • BVH (skeletal animation)  │
│  • Skeleton definition (YAML)            • JSON (states + metadata)  │
│  • Sync metadata (JSON)                  • ZIP (diagnostics)         │
│                                          • Statistics (CSV/JSON)     │
└────────────────────────┬──────────────────────────┬─────────────────┘
                         │                          │
                         ▼                          ▼
┌─────────────────────────────────────┐  ┌──────────────────────────┐
│     CLI Application                 │  │   Python Bindings        │
│  (posetrak executable)              │  │   (pyposetrak module)    │
│                                     │  │                          │
│  • Argument parsing (CLI11)         │  │  • pybind11 interface    │
│  • Orchestration                    │  │  • BVH export wrapper    │
│  • Progress display                 │  │  • Visualization helpers │
└──────────────┬──────────────────────┘  └───────────┬──────────────┘
               │                                     │
               └─────────────────┬───────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Core Library (libposetrak)                       │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Tracking Layer                               │ │
│  │                                                                 │ │
│  │  • Tracker (orchestration)                                      │ │
│  │  • ProgressCallbacks (observer pattern)                         │ │
│  │  • TrackingResult (states, diagnostics)                        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                 │                                    │
│                                 ▼                                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Filter Layer                                 │ │
│  │                                                                 │ │
│  │  • FilterBase (abstract interface)                              │ │
│  │  • UKF (Unscented Kalman Filter)                               │ │
│  │  • OutlierRejection (Mahalanobis distance)                     │ │
│  │  • SigmaPointGenerator (UT transform)                          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                 │                                    │
│                                 ▼                                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Kinematics Layer                             │ │
│  │                                                                 │ │
│  │  • ForwardKinematics (Pinocchio wrapper)                       │ │
│  │  • InverseKinematics (Pinocchio IK solver for initialization)  │ │
│  │  • Triangulation (multi-camera → 3D)                           │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                 │                                    │
│                                 ▼                                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Core Models                                  │ │
│  │                                                                 │ │
│  │  • State (joint angles, velocities, covariance)                │ │
│  │  • Skeleton (hierarchy, limits, markers)                       │ │
│  │  • Camera (projection, distortion)                             │ │
│  │  • Observation (2D detection with confidence)                  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                 │                                    │
│                                 ▼                                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    I/O Layer                                    │ │
│  │                                                                 │ │
│  │  • SkeletonLoader (YAML → Skeleton + URDF)                     │ │
│  │  • CameraLoader (TOML → Camera models)                         │ │
│  │  • ObservationLoader (OpenPose JSON → Observations)            │ │
│  │  • SyncMetadataLoader (JSON → sync points)                     │ │
│  │  • Exporters (TRC, JSON, ZIP archives)                         │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    External Libraries                                │
│                                                                      │
│  • Eigen (linear algebra)          • toml11 (calibration)           │
│  • Pinocchio (FK/IK)               • nlohmann/json (interchange)    │
│  • OpenCV (distortion, video I/O)  • CLI11 (CLI parsing)            │
│  • yaml-cpp (skeleton config)      • libarchive (ZIP export)        │
│  • fmt (formatting)                 • GTest/Catch2 (testing)        │
│  • OpenMP (parallelization)                                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Design Principles

### 2.1 Separation of Concerns
- **Core models**: Data structures with minimal logic (State, Skeleton, Camera)
- **Algorithms**: Stateless functions operating on models (FK, IK, UKF steps)
- **Orchestration**: Tracker coordinates algorithm execution
- **I/O**: Separate from core logic, pluggable formats

### 2.2 Dependency Injection
- Filters receive Camera and Skeleton as dependencies
- Tracker receives Filter implementation
- Enables testing with mock implementations
- No global state

### 2.3 Value Semantics Where Possible
- Small objects (Observation, Marker) passed by value
- Large objects (State, Skeleton, Cameras) passed by const&
- State updates return new State (functional style in UKF)
- Covariance stored separately (mutable, owned by UKF)

### 2.4 Zero-Cost Abstractions
- Templates for generic algorithms (sigma points, projection)
- Concepts for type constraints (C++20)
- Inline small functions
- Avoid virtual calls in hot paths

### 2.5 Error Handling
- Exceptions for unrecoverable errors (file not found, invalid input)
- std::optional for expected missing data (marker not detected)
- std::expected for operations that can fail gracefully
- Assert for logic errors (debug builds only)

### 2.6 Modern C++ Features
- **C++20**: Concepts, ranges, std::span, three-way comparison
- **C++23**: std::expected, std::print (via fmt fallback)
- **Eigen**: For all linear algebra (no raw arrays)
- **Smart pointers**: Ownership semantics (unique_ptr, shared_ptr)

---

## 3. Module Breakdown

### 3.1 Core Models (`posetrak/core/`)

**State** (`state.hpp`):
```cpp
class State {
public:
    // Root pose (position + orientation)
    Eigen::Vector3d root_position;
    Eigen::Quaterniond root_orientation;

    // Joint angles (dense, ordered by skeleton)
    Eigen::VectorXd joint_angles;  // DOF

    // Velocities
    Eigen::Vector3d root_velocity;
    Eigen::Vector3d root_angular_velocity;
    Eigen::VectorXd joint_velocities;  // DOF

    // Conversion to/from flat vectors
    Eigen::VectorXd to_vector() const;
    static State from_vector(const Eigen::VectorXd& v, const Skeleton& skel);

    // Pinocchio interface
    Eigen::VectorXd to_pinocchio_q() const;
    Eigen::VectorXd to_pinocchio_v() const;
};
```

**Skeleton** (`skeleton.hpp`):
```cpp
struct Joint {
    std::string name;
    std::string parent_name;
    JointType type;  // Revolute, Ball, Fixed, Prismatic
    Eigen::Vector3d local_offset;
    Eigen::VectorXd limits_min;  // Per DOF
    Eigen::VectorXd limits_max;
    int dof;
    std::string group;  // "core", "left_arm", etc.
};

struct Marker {
    std::string name;
    std::string joint_name;
    Eigen::Vector3d local_position;
    int coco_id;  // -1 if not a COCO keypoint
};

class Skeleton {
public:
    // Construction
    static Skeleton from_yaml(const std::string& path);
    static Skeleton from_urdf(const std::string& path);

    // Queries
    int get_total_dof() const;
    int get_active_dof() const;
    std::vector<Joint> get_active_joints() const;
    const Joint& get_joint(const std::string& name) const;
    const Marker& get_marker(const std::string& name) const;

    // Filtering
    void set_active_groups(const std::vector<std::string>& groups);
    void set_active_joints(const std::vector<std::string>& joints);

    // Export
    std::string to_urdf() const;

private:
    std::string root_name_;
    std::map<std::string, Joint> joints_;
    std::vector<Marker> markers_;
    std::set<std::string> active_groups_;
    std::set<std::string> active_joints_;
};
```

**Camera** (`camera.hpp`):
```cpp
class Camera {
public:
    // Intrinsics
    double fx, fy, cx, cy;

    // Distortion (Brown-Conrady)
    double k1, k2, k3, p1, p2;

    // Extrinsics
    Eigen::Vector3d position;
    Eigen::Matrix3d rotation;

    // Temporal
    double fps;
    int start_frame;
    std::vector<SyncPoint> sync_points;  // (frame_idx, timestamp)

    // Projection
    Eigen::Vector2d project_undistorted(const Eigen::Vector3d& p3d) const;
    Eigen::Vector2d project_distorted(const Eigen::Vector3d& p3d) const;
    Eigen::Vector2d undistort(const Eigen::Vector2d& p2d) const;
    Eigen::Vector2d distort(const Eigen::Vector2d& p2d) const;

    // Temporal queries
    double get_timestamp(int frame_idx) const;
    int get_frame_at_time(double timestamp) const;

    // Batched operations
    std::vector<Eigen::Vector2d> project_batch(
        const std::vector<Eigen::Vector3d>& points) const;
};
```

**Observation** (`observation.hpp`):
```cpp
struct Observation {
    int camera_id;
    int marker_id;
    Eigen::Vector2d position;  // Undistorted coordinates
    Eigen::Vector2d position_distorted;  // Original
    double confidence;
    double timestamp;
    int frame_idx;
};

struct ObservationSequence {
    int camera_id;
    std::string camera_name;
    std::vector<Observation> observations;
};

struct ObservationSet {
    std::map<std::string, ObservationSequence> sequences;
    int person_id;

    // Query observations for specific frame
    std::vector<Observation> get_observations_at_time(double t) const;
    std::vector<Observation> get_observations_for_frame(int frame_idx) const;
};
```

### 3.2 Kinematics Layer (`posetrak/kinematics/`)

**ForwardKinematics** (`forward_kinematics.hpp`):
```cpp
class ForwardKinematics {
public:
    explicit ForwardKinematics(const Skeleton& skeleton);

    // Compute marker positions in world frame
    std::vector<Eigen::Vector3d> compute_marker_positions(
        const State& state) const;

    // Compute single marker
    Eigen::Vector3d compute_marker_position(
        const State& state, int marker_id) const;

    // Compute joint transforms (for visualization)
    std::vector<Eigen::Affine3d> compute_joint_transforms(
        const State& state) const;

    // Jacobian (for EKF, future)
    Eigen::MatrixXd compute_marker_jacobian(
        const State& state, int marker_id) const;

private:
    // Pinocchio model and data
    pinocchio::Model model_;
    mutable pinocchio::Data data_;  // Mutable for internal cache

    // Mapping: marker_id → (frame_id, local_pos)
    std::vector<std::pair<int, Eigen::Vector3d>> marker_frames_;
};
```

**InverseKinematics** (`inverse_kinematics.hpp`):
```cpp
// Simple IK for initialization
State solve_ik(
    const Skeleton& skeleton,
    const std::map<int, Eigen::Vector3d>& marker_positions,
    const State& initial_guess = State(),
    int max_iterations = 100,
    double tolerance = 1e-4);

// Triangulation
std::map<int, Eigen::Vector3d> triangulate_markers(
    const std::vector<Observation>& observations,
    const std::vector<Camera>& cameras);
```

### 3.3 Filter Layer (`posetrak/filters/`)

**FilterBase** (`filter_base.hpp`):
```cpp
class FilterBase {
public:
    virtual ~FilterBase() = default;

    // Predict state at next time step
    virtual State predict(const State& state, double dt) = 0;

    // Update state with observations
    virtual State update(
        const State& predicted_state,
        const std::vector<Observation>& observations) = 0;

    // Covariance access (for diagnostics)
    virtual const Eigen::MatrixXd& get_covariance() const = 0;
    virtual void set_covariance(const Eigen::MatrixXd& cov) = 0;
};
```

**UKF** (`ukf.hpp`):
```cpp
struct UKFParams {
    double alpha = 1e-3;  // Spread of sigma points
    double beta = 2.0;    // Prior knowledge (Gaussian = 2)
    double kappa = 0.0;   // Secondary scaling
    double process_noise_std = 0.1;
    double measurement_noise_std = 5.0;
    double outlier_threshold = 0.0;  // 0 = disabled
    int n_jobs = -1;  // Parallelization
};

class UKF : public FilterBase {
public:
    UKF(const Skeleton& skeleton,
        const std::vector<Camera>& cameras,
        const UKFParams& params = UKFParams());

    State predict(const State& state, double dt) override;
    State update(const State& predicted_state,
                 const std::vector<Observation>& observations) override;

    const Eigen::MatrixXd& get_covariance() const override;
    void set_covariance(const Eigen::MatrixXd& cov) override;

    // Diagnostics
    const Eigen::VectorXd& get_innovation() const;
    const std::vector<OutlierInfo>& get_outliers() const;
    const Eigen::MatrixXd& get_kalman_gain() const;

private:
    // Sigma points
    std::vector<State> generate_sigma_points(
        const State& mean, const Eigen::MatrixXd& cov);

    State compute_mean_state(
        const std::vector<State>& sigma_points,
        const Eigen::VectorXd& weights);

    // Measurement prediction
    Eigen::VectorXd predict_measurement(const State& state);

    // Outlier rejection
    std::vector<Observation> filter_outliers(
        const std::vector<Observation>& observations,
        const Eigen::VectorXd& predicted_measurements);

    const Skeleton& skeleton_;
    const std::vector<Camera>& cameras_;
    UKFParams params_;
    ForwardKinematics fk_;

    // State
    Eigen::MatrixXd covariance_;
    Eigen::MatrixXd process_noise_;
    Eigen::MatrixXd measurement_noise_;

    // Diagnostics
    Eigen::VectorXd innovation_;
    std::vector<OutlierInfo> outliers_;
    Eigen::MatrixXd kalman_gain_;
};
```

**OutlierRejection** (`outlier_rejection.hpp`):
```cpp
struct OutlierInfo {
    int observation_idx;
    int camera_id;
    int marker_id;
    double mahalanobis_distance;
    Eigen::Vector2d residual;
};

std::vector<OutlierInfo> detect_outliers(
    const std::vector<Observation>& observations,
    const Eigen::VectorXd& predicted_measurements,
    const Eigen::MatrixXd& innovation_covariance,
    double threshold);
```

### 3.4 Tracking Layer (`posetrak/tracking/`)

**Tracker** (`tracker.hpp`):
```cpp
struct TrackerCallbacks {
    std::function<void(int)> on_frame_start;
    std::function<void(const State&)> on_predict_done;
    std::function<void(const State&, const std::vector<OutlierInfo>&)>
        on_update_done;
    std::function<void(int, const State&, double)> on_frame_done;
};

struct TrackerResult {
    std::vector<State> states;
    std::vector<double> timestamps;
    std::vector<int> frame_indices;
    TrackerDiagnostics diagnostics;
};

class Tracker {
public:
    Tracker(std::unique_ptr<FilterBase> filter,
            const Skeleton& skeleton,
            const std::vector<Camera>& cameras);

    // Initialization
    void initialize(const State& initial_state,
                   const Eigen::MatrixXd& initial_covariance);

    void initialize_from_observations(
        const std::vector<Observation>& first_frame_obs);

    // Tracking
    TrackerResult track(const ObservationSet& observations);

    // Frame-by-frame (for GUI)
    void step(const std::vector<Observation>& observations, double dt);
    const State& get_current_state() const;

    // Configuration
    void set_callbacks(const TrackerCallbacks& callbacks);
    void enable_diagnostics(bool enabled);

private:
    std::unique_ptr<FilterBase> filter_;
    const Skeleton& skeleton_;
    const std::vector<Camera>& cameras_;

    State current_state_;
    TrackerCallbacks callbacks_;
    bool diagnostics_enabled_;
    TrackerDiagnostics diagnostics_;
};
```

**TrackerDiagnostics** (`tracker_diagnostics.hpp`):
```cpp
struct FrameDiagnostics {
    int frame_idx;
    double timestamp;
    State state;
    Eigen::MatrixXd covariance;
    Eigen::VectorXd innovation;
    std::vector<OutlierInfo> outliers;
    std::map<int, double> marker_errors;  // marker_id → reprojection error
    double mean_error;
    double max_error;
    double predict_time_ms;
    double update_time_ms;
};

class TrackerDiagnostics {
public:
    void add_frame(const FrameDiagnostics& frame);

    const std::vector<FrameDiagnostics>& get_frames() const;

    // Summary statistics
    double get_mean_reprojection_error() const;
    double get_outlier_rate() const;
    double get_mean_frame_time() const;

    // Export
    void export_to_zip(const std::string& path) const;
    nlohmann::json to_json() const;
};
```

### 3.5 I/O Layer (`posetrak/io/`)

**Loaders**:
```cpp
// Skeleton
Skeleton load_skeleton_from_yaml(const std::string& path);

// Cameras
std::vector<Camera> load_cameras_from_toml(const std::string& path);

// Observations
ObservationSet load_openpose_observations(
    const std::string& base_dir,
    const std::vector<std::string>& camera_names,
    const Skeleton& skeleton,
    int person_id,
    int start_frame = 0,
    int max_frames = -1,
    double min_confidence = 0.0);

// Synchronization metadata
void load_sync_metadata(
    const std::string& path,
    std::vector<Camera>& cameras);
```

**Exporters**:
```cpp
// TRC (OpenSim marker trajectories)
void export_trc(
    const std::string& path,
    const std::vector<State>& states,
    const Skeleton& skeleton,
    double fps);

// JSON (structured format)
void export_json(
    const std::string& path,
    const TrackerResult& result,
    const Skeleton& skeleton,
    const std::vector<Camera>& cameras);

// ZIP diagnostics
void export_diagnostics_zip(
    const std::string& path,
    const TrackerDiagnostics& diagnostics,
    const Skeleton& skeleton,
    const std::vector<Camera>& cameras);

// Statistics CSV
void export_statistics_csv(
    const std::string& path,
    const TrackerDiagnostics& diagnostics);

// Video overlay (tracking visualization on original footage)
void export_video_overlay(
    const std::string& output_path,
    const std::vector<std::string>& video_paths,  // Per camera
    const std::vector<State>& states,
    const Skeleton& skeleton,
    const std::vector<Camera>& cameras,
    double fps);
```

---

## 4. Data Flow

### 4.1 Typical Tracking Session

```cpp
// 1. Load configuration
auto skeleton = load_skeleton_from_yaml("skeleton.yaml");
skeleton.set_active_groups({"core", "left_arm", "right_arm", "legs"});

auto cameras = load_cameras_from_toml("calibration.toml");
load_sync_metadata("sync.json", cameras);

// 2. Load observations
auto observations = load_openpose_observations(
    "data/aikido", camera_names, skeleton, person_id=0);

// 3. Initialize tracker
UKFParams params;
params.process_noise_std = 0.1;
params.measurement_noise_std = 5.0;
params.outlier_threshold = 5.991;  // 95% confidence

auto ukf = std::make_unique<UKF>(skeleton, cameras, params);
Tracker tracker(std::move(ukf), skeleton, cameras);

// 4. Set up callbacks
TrackerCallbacks callbacks;
callbacks.on_frame_done = [](int idx, const State& s, double t) {
    fmt::print("Frame {}: t={:.3f}s\n", idx, t);
};
tracker.set_callbacks(callbacks);
tracker.enable_diagnostics(true);

// 5. Initialize from first frame
auto first_frame = observations.get_observations_for_frame(0);
tracker.initialize_from_observations(first_frame);

// 6. Track
auto result = tracker.track(observations);

// 7. Export
export_trc("output.trc", result.states, skeleton, 30.0);
export_json("output.json", result, skeleton, cameras);
result.diagnostics.export_to_zip("diagnostics.zip");
```

### 4.2 Frame Processing Pipeline

```
For each frame t:
  ├─ Get observations at time t
  │  ├─ Query each camera for frame_idx at time t
  │  └─ Collect all 2D detections (undistorted)
  │
  ├─ Predict
  │  ├─ Generate sigma points from current state
  │  ├─ Propagate through process model (constant velocity)
  │  ├─ Compute predicted state (mean of propagated sigma points)
  │  └─ Update covariance
  │
  ├─ Measurement Prediction
  │  ├─ For each sigma point:
  │  │  ├─ Run forward kinematics → 3D markers
  │  │  └─ Project to each camera → 2D predictions
  │  └─ Compute mean predicted measurements
  │
  ├─ Outlier Rejection
  │  ├─ For each observation:
  │  │  ├─ Compute innovation (observed - predicted)
  │  │  ├─ Compute Mahalanobis distance
  │  │  └─ Reject if distance > threshold
  │  └─ Keep only inliers
  │
  ├─ Update
  │  ├─ Compute Kalman gain
  │  ├─ Update state with innovation
  │  └─ Update covariance
  │
  └─ Callback: on_frame_done(state, diagnostics)
```

---

## 5. Threading Model

### 5.1 Parallelization Points

**Sigma Point Evaluation** (embarrassingly parallel):
```cpp
#pragma omp parallel for
for (int i = 0; i < sigma_points.size(); ++i) {
    auto& sp = sigma_points[i];
    auto markers_3d = fk.compute_marker_positions(sp.state);
    sp.measurements = project_to_cameras(markers_3d, cameras);
}
```

**Camera Projection** (parallel across cameras):
```cpp
std::vector<Eigen::Vector2d> measurements(cameras.size() * markers.size());

#pragma omp parallel for
for (int cam_idx = 0; cam_idx < cameras.size(); ++cam_idx) {
    for (int m = 0; m < markers.size(); ++m) {
        measurements[cam_idx * markers.size() + m] =
            cameras[cam_idx].project_undistorted(markers[m]);
    }
}
```

**Hierarchical Tracking** (parallel limb updates):
```cpp
// Update torso first (serial)
torso_state = ukf_torso.predict_and_update(observations);

// Update limbs in parallel (conditioned on torso)
#pragma omp parallel sections
{
    #pragma omp section
    { left_arm_state = ukf_left_arm.update(obs, torso_state); }

    #pragma omp section
    { right_arm_state = ukf_right_arm.update(obs, torso_state); }

    #pragma omp section
    { legs_state = ukf_legs.update(obs, torso_state); }
}
```

### 5.2 Thread Safety

**Const-Correct Design**:
- `ForwardKinematics::compute_*` are const methods
- Pinocchio `Data` is `mutable` (internal cache)
- One `Data` object per thread (thread_local or vector of Data)

**No Shared Mutable State**:
- Each UKF instance owns its covariance
- States are immutable (copy-on-update)
- Cameras and Skeleton are const references

---

## 6. Memory Management

### 6.1 Ownership Model

- **Skeleton, Cameras**: Owned by Tracker, passed as const& to Filter
- **State**: Value semantics, copied on update (cheap with COW in Eigen)
- **Covariance**: Owned by UKF, mutated in place
- **Observations**: Loaded once, iterated by Tracker
- **ForwardKinematics**: Owned by UKF, holds Pinocchio model

### 6.2 Memory Layout

**State** (cache-friendly):
```
[root_pos (3) | root_quat (4) | joint_angles (DOF) |
 root_vel (3) | root_angvel (3) | joint_vels (DOF)]
```
Total: ~1.5 KB for 120 DOF

**Covariance** (error-state):
```
[3 + 3 + DOF + 3 + 3 + DOF] × [same]
```
Total: ~240 KB for 120 DOF (dense matrix)

**Per-Frame Memory**: ~250 KB (state + covariance)
**1000 Frames**: ~250 MB

---

## 7. Error Handling Strategy

### 7.1 Exception Hierarchy

```cpp
namespace posetrak {

class Error : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

class IOError : public Error { using Error::Error; };
class ParseError : public IOError { using IOError::IOError; };
class FileNotFoundError : public IOError { using IOError::IOError; };

class ConfigError : public Error { using Error::Error; };
class InvalidSkeletonError : public ConfigError { using ConfigError::ConfigError; };

class TrackingError : public Error { using Error::Error; };
class InitializationError : public TrackingError { using TrackingError::TrackingError; };
class ConvergenceError : public TrackingError { using TrackingError::TrackingError; };

}  // namespace posetrak
```

### 7.2 Expected Failures (std::optional)

```cpp
// Missing observations
std::optional<Observation> get_observation(int camera_id, int marker_id);

// Triangulation may fail
std::optional<Eigen::Vector3d> triangulate_marker(
    const std::vector<Observation>& obs);
```

### 7.3 Validation

```cpp
// At construction
Skeleton::Skeleton(...) {
    if (joints_.empty())
        throw InvalidSkeletonError("Skeleton has no joints");
    if (!joints_.count(root_name_))
        throw InvalidSkeletonError("Root joint not found");
    validate_tree_structure();  // Throws if cycle detected
}

// At runtime
void UKF::update(...) {
    if (observations.empty())
        throw TrackingError("No observations provided");
    if (!std::isfinite(covariance_.norm()))
        throw TrackingError("Covariance is not finite");
}
```

---

## 8. Testing Strategy

### 8.1 Unit Tests (GTest)

```cpp
// test_state.cpp
TEST(StateTest, VectorConversion) {
    Skeleton skel = create_simple_skeleton();
    State state = create_test_state();
    auto vec = state.to_vector();
    auto state2 = State::from_vector(vec, skel);
    EXPECT_EQ(state.joint_angles, state2.joint_angles);
}

// test_camera.cpp
TEST(CameraTest, ProjectionInvariance) {
    Camera cam = create_test_camera();
    Eigen::Vector3d p3d(1, 2, 5);
    auto p2d = cam.project_undistorted(p3d);
    auto p2d_distorted = cam.project_distorted(p3d);
    auto p2d_undistorted = cam.undistort(p2d_distorted);
    EXPECT_NEAR((p2d - p2d_undistorted).norm(), 0.0, 1e-3);
}

// test_forward_kinematics.cpp
TEST(FKTest, RootOnly) {
    Skeleton skel = create_root_only_skeleton();
    ForwardKinematics fk(skel);
    State state;
    state.root_position = Eigen::Vector3d(1, 2, 3);
    auto markers = fk.compute_marker_positions(state);
    EXPECT_EQ(markers.size(), 1);
    EXPECT_NEAR((markers[0] - state.root_position).norm(), 0.0, 1e-6);
}
```

### 8.2 Integration Tests

```cpp
// test_tracking_pipeline.cpp
TEST(TrackingTest, SyntheticSequence) {
    // Generate synthetic ground truth
    auto ground_truth = generate_synthetic_motion();

    // Simulate observations with noise
    auto observations = simulate_observations(ground_truth, cameras);

    // Track
    Tracker tracker(...);
    auto result = tracker.track(observations);

    // Compare
    double rmse = compute_rmse(result.states, ground_truth);
    EXPECT_LT(rmse, 1.0);  // < 1 degree RMSE
}
```

### 8.3 Regression Tests

```python
# Compare with Python reference
def test_regression():
    # Load Python results
    python_states = load_json("reference/python_tracked.json")

    # Run C++ tracker
    cpp_states = run_cpp_tracker("test_data/")

    # Compare joint angles
    rmse = compute_rmse(cpp_states, python_states)
    assert rmse < 1.0, f"RMSE too high: {rmse}"
```

---

## 9. Build System (Meson)

```meson
project('posetrak', 'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++20',
    'warning_level=3',
    'werror=true',
    'buildtype=release',
  ]
)

# Dependencies
eigen_dep = dependency('eigen3', version: '>=3.4')
pinocchio_dep = dependency('pinocchio', version: '>=3.9')
fmt_dep = dependency('fmt', version: '>=10.0')
yaml_cpp_dep = dependency('yaml-cpp')
toml11_dep = dependency('toml11')
json_dep = dependency('nlohmann_json')
cli11_dep = dependency('CLI11')
opencv_dep = dependency('opencv4', version: '>=4.5', modules: ['core', 'imgproc', 'videoio', 'calib3d'])
archive_dep = dependency('libarchive', required: false)
openmp_dep = dependency('openmp', required: false)

# Core library
posetrak_inc = include_directories('include')
posetrak_src = files(
  'src/core/state.cpp',
  'src/core/skeleton.cpp',
  'src/core/camera.cpp',
  'src/kinematics/forward_kinematics.cpp',
  'src/kinematics/inverse_kinematics.cpp',
  'src/filters/ukf.cpp',
  'src/filters/outlier_rejection.cpp',
  'src/tracking/tracker.cpp',
  'src/io/loaders.cpp',
  'src/io/exporters.cpp',
)

libposetrak = library('posetrak',
  posetrak_src,
  include_directories: posetrak_inc,
  dependencies: [eigen_dep, pinocchio_dep, fmt_dep,
                 yaml_cpp_dep, toml11_dep, json_dep,
                 archive_dep, openmp_dep],
  install: true
)

# CLI
executable('posetrak',
  'cli/track.cpp',
  include_directories: posetrak_inc,
  link_with: libposetrak,
  dependencies: [cli11_dep, fmt_dep],
  install: true
)

# Tests
if get_option('enable_tests')
  gtest_dep = dependency('gtest', main: true)
  test_exe = executable('test_posetrak',
    files('tests/test_state.cpp',
          'tests/test_camera.cpp',
          'tests/test_forward_kinematics.cpp'),
    include_directories: posetrak_inc,
    link_with: libposetrak,
    dependencies: [gtest_dep]
  )
  test('unit_tests', test_exe)
endif

# Python bindings
if get_option('enable_python')
  subdir('python')
endif
```

---

## 10. Next Steps

1. **Review & Approval**: Get stakeholder feedback on requirements and architecture
2. **Detailed Design**: Module-level design documents (next phase)
3. **Prototype**: Implement core models + FK + simple UKF
4. **Validate**: Test against Python on simple skeletons
5. **Iterate**: Add outlier rejection, diagnostics, hierarchical UKF
6. **Polish**: CLI, error handling, documentation
7. **Release**: Package, bindings, examples
