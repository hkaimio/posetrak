# C++ Motion Capture Tracker - Detailed Architecture

## 1. Camera Model with Distortion Handling

### 1.1 Problem Statement

**Challenge**: 2D pose detections (OpenPose) are extracted from original distorted camera footage. However, the UKF measurement model assumes ideal pinhole projection.

**Solution**: Maintain two coordinate systems:
1. **Distorted coordinates**: Original pixel coordinates from camera footage
2. **Undistorted (ideal) coordinates**: Normalized to ideal pinhole projection

### 1.2 Camera Class Design

```cpp
namespace posetrak {

struct Intrinsics {
    double fx, fy;  // Focal lengths
    double cx, cy;  // Principal point

    enum class DistortionModel {
        BrownConrady,  // k1, k2, k3, p1, p2 (5 coeffs)
        Fisheye        // k1, k2, k3, k4 (4 coeffs, OpenCV fisheye)
    };

    DistortionModel model;
    std::vector<double> distortion_coeffs;  // Size depends on model
};

struct Extrinsics {
    Eigen::Vector3d position;
    Eigen::Matrix3d rotation;  // World to camera

    // Convenience: world to camera transform
    Eigen::Affine3d get_transform() const {
        Eigen::Affine3d T = Eigen::Affine3d::Identity();
        T.linear() = rotation;
        T.translation() = position;
        return T;
    }
};

struct SyncPoint {
    int frame_idx;
    double timestamp_sec;
};

class Camera {
public:
    // Construction
    Camera(const std::string& name,
           const Intrinsics& intrinsics,
           const Extrinsics& extrinsics,
           double fps = 30.0,
           int start_frame = 0);

    // --- Projection API ---

    // Project 3D point to undistorted normalized coordinates
    // This is what UKF uses internally
    Eigen::Vector2d project_undistorted(const Eigen::Vector3d& p_world) const;

    // Project 3D point to distorted pixel coordinates
    // For final visualization/reprojection error
    Eigen::Vector2d project_distorted(const Eigen::Vector3d& p_world) const;

    // --- Distortion API ---

    // Undistort pixel coordinates (OpenPose → UKF input)
    Eigen::Vector2d undistort(const Eigen::Vector2d& p_distorted) const;

    // Distort normalized coordinates (UKF output → visualization)
    Eigen::Vector2d distort(const Eigen::Vector2d& p_undistorted) const;

    // --- Temporal API ---

    // Get timestamp for frame index (uses linear interpolation)
    double get_timestamp(int frame_idx) const;

    // Get frame index at given timestamp (inverse lookup)
    int get_frame_at_time(double timestamp) const;

    // Set synchronization points
    void set_sync_points(const std::vector<SyncPoint>& points);

    // --- Batch Operations ---

    // Project multiple points efficiently
    std::vector<Eigen::Vector2d> project_batch_undistorted(
        const std::vector<Eigen::Vector3d>& points) const;

    std::vector<Eigen::Vector2d> project_batch_distorted(
        const std::vector<Eigen::Vector3d>& points) const;

    // --- Accessors ---

    const std::string& name() const { return name_; }
    const Intrinsics& intrinsics() const { return intrinsics_; }
    const Extrinsics& extrinsics() const { return extrinsics_; }
    double fps() const { return fps_; }
    int start_frame() const { return start_frame_; }

private:
    std::string name_;
    Intrinsics intrinsics_;
    Extrinsics extrinsics_;
    double fps_;
    int start_frame_;
    std::vector<SyncPoint> sync_points_;

    // Cached ideal projection matrix (for undistorted)
    Eigen::Matrix<double, 3, 4> projection_matrix_;

    // Helpers
    void compute_projection_matrix();
    Eigen::Vector2d apply_distortion(const Eigen::Vector2d& p_norm) const;
    Eigen::Vector2d remove_distortion(const Eigen::Vector2d& p_distorted) const;
};

}  // namespace posetrak
```

### 1.3 Implementation Details

**Projection Pipeline**:
```cpp
Eigen::Vector2d Camera::project_distorted(const Eigen::Vector3d& p_world) const {
    // 1. Transform to camera frame
    Eigen::Vector3d p_cam = extrinsics_.get_transform() * p_world;

    // 2. Perspective division
    double x = p_cam.x() / p_cam.z();
    double y = p_cam.y() / p_cam.z();
    Eigen::Vector2d p_norm(x, y);

    // 3. Apply distortion
    Eigen::Vector2d p_distorted = apply_distortion(p_norm);

    // 4. Apply intrinsics
    double u = intrinsics_.fx * p_distorted.x() + intrinsics_.cx;
    double v = intrinsics_.fy * p_distorted.y() + intrinsics_.cy;

    return Eigen::Vector2d(u, v);
}

Eigen::Vector2d Camera::project_undistorted(const Eigen::Vector3d& p_world) const {
    // Skip step 3 (distortion)
    Eigen::Vector3d p_cam = extrinsics_.get_transform() * p_world;
    double x = p_cam.x() / p_cam.z();
    double y = p_cam.y() / p_cam.z();
    double u = intrinsics_.fx * x + intrinsics_.cx;
    double v = intrinsics_.fy * y + intrinsics_.cy;
    return Eigen::Vector2d(u, v);
}
```

**Distortion Model** (Brown-Conrady):
```cpp
Eigen::Vector2d Camera::apply_distortion(const Eigen::Vector2d& p_norm) const {
    double x = p_norm.x();
    double y = p_norm.y();
    double r2 = x * x + y * y;
    double r4 = r2 * r2;
    double r6 = r2 * r4;

    // Radial distortion
    double radial = 1.0 + intrinsics_.k1 * r2
                        + intrinsics_.k2 * r4
                        + intrinsics_.k3 * r6;

    // Tangential distortion
    double dx_tangential = 2.0 * intrinsics_.p1 * x * y
                         + intrinsics_.p2 * (r2 + 2.0 * x * x);
    double dy_tangential = intrinsics_.p1 * (r2 + 2.0 * y * y)
                         + 2.0 * intrinsics_.p2 * x * y;

    double x_distorted = x * radial + dx_tangential;
    double y_distorted = y * radial + dy_tangential;

    return Eigen::Vector2d(x_distorted, y_distorted);
}
```

**Undistortion** (iterative Newton-Raphson):
```cpp
Eigen::Vector2d Camera::remove_distortion(const Eigen::Vector2d& p_distorted) const {
    // Convert to normalized coordinates
    double x_d = (p_distorted.x() - intrinsics_.cx) / intrinsics_.fx;
    double y_d = (p_distorted.y() - intrinsics_.cy) / intrinsics_.fy;

    // Initial guess
    double x = x_d;
    double y = y_d;

    // Newton-Raphson iterations (typically 5 is enough)
    for (int i = 0; i < 5; ++i) {
        double r2 = x * x + y * y;
        double r4 = r2 * r2;
        double r6 = r2 * r4;

        double radial = 1.0 + intrinsics_.k1 * r2
                            + intrinsics_.k2 * r4
                            + intrinsics_.k3 * r6;

        double dx_tangential = 2.0 * intrinsics_.p1 * x * y
                             + intrinsics_.p2 * (r2 + 2.0 * x * x);
        double dy_tangential = intrinsics_.p1 * (r2 + 2.0 * y * y)
                             + 2.0 * intrinsics_.p2 * x * y;

        double x_new = (x_d - dx_tangential) / radial;
        double y_new = (y_d - dy_tangential) / radial;

        if (std::abs(x_new - x) < 1e-6 && std::abs(y_new - y) < 1e-6) {
            break;
        }

        x = x_new;
        y = y_new;
    }

    // Convert back to pixel coordinates
    double u = intrinsics_.fx * x + intrinsics_.cx;
    double v = intrinsics_.fy * y + intrinsics_.cy;

    return Eigen::Vector2d(u, v);
}
```

### 1.4 Temporal Synchronization

**Synchronization Points**:
```cpp
double Camera::get_timestamp(int frame_idx) const {
    if (sync_points_.empty()) {
        // Fallback: uniform frame rate
        return (frame_idx - start_frame_) / fps_;
    }

    // Find bracketing sync points
    auto it = std::lower_bound(sync_points_.begin(), sync_points_.end(),
                               frame_idx,
                               [](const SyncPoint& sp, int idx) {
                                   return sp.frame_idx < idx;
                               });

    if (it == sync_points_.begin()) {
        // Before first sync point: extrapolate backward
        double dt = (frame_idx - it->frame_idx) / fps_;
        return it->timestamp_sec + dt;
    }

    if (it == sync_points_.end()) {
        // After last sync point: extrapolate forward
        auto last = sync_points_.back();
        double dt = (frame_idx - last.frame_idx) / fps_;
        return last.timestamp_sec + dt;
    }

    // Linear interpolation between two sync points
    auto prev = std::prev(it);
    double t0 = prev->timestamp_sec;
    double t1 = it->timestamp_sec;
    int f0 = prev->frame_idx;
    int f1 = it->frame_idx;

    double alpha = static_cast<double>(frame_idx - f0) / (f1 - f0);
    return t0 + alpha * (t1 - t0);
}
```

**Synchronization Metadata Format** (`sync.json`):
```json
{
  "cameras": {
    "camera_1": {
      "sync_points": [
        {"frame": 0, "time": 0.0},
        {"frame": 1000, "time": 33.333},
        {"frame": 2000, "time": 66.700}
      ]
    },
    "camera_2": {
      "sync_points": [
        {"frame": 0, "time": 0.0},
        {"frame": 900, "time": 30.050},
        {"frame": 1800, "time": 60.120}
      ]
    }
  }
}
```

### 1.5 Usage in Tracking Pipeline

```cpp
// During initialization: undistort all observations
ObservationSet load_and_preprocess_observations(...) {
    auto raw_obs = load_openpose_json(...);

    for (auto& obs : raw_obs) {
        // Store both distorted and undistorted
        obs.position_distorted = obs.position;  // Original
        obs.position = cameras[obs.camera_id].undistort(obs.position);  // For UKF
        obs.timestamp = cameras[obs.camera_id].get_timestamp(obs.frame_idx);
    }

    return raw_obs;
}

// During UKF update: use undistorted
Eigen::VectorXd UKF::predict_measurement(const State& state) {
    auto markers_3d = fk_.compute_marker_positions(state);

    std::vector<double> measurements;
    for (const auto& camera : cameras_) {
        auto projections = camera.project_batch_undistorted(markers_3d);
        for (const auto& p : projections) {
            measurements.push_back(p.x());
            measurements.push_back(p.y());
        }
    }

    return Eigen::Map<Eigen::VectorXd>(measurements.data(), measurements.size());
}

// During visualization: use distorted
double compute_reprojection_error(const State& state,
                                  const Observation& obs,
                                  const Camera& camera) {
    auto marker_3d = fk_.compute_marker_position(state, obs.marker_id);
    auto predicted_distorted = camera.project_distorted(marker_3d);
    return (predicted_distorted - obs.position_distorted).norm();
}
```

---

## 2. State Representation

### 2.1 Full State vs Error State

**Full State** (for storage and visualization):
```cpp
struct State {
    // Root pose (6 DOF)
    Eigen::Vector3d root_position;
    Eigen::Quaterniond root_orientation;

    // Joint angles (DOF, varies by skeleton)
    Eigen::VectorXd joint_angles;

    // Velocities
    Eigen::Vector3d root_velocity;
    Eigen::Vector3d root_angular_velocity;
    Eigen::VectorXd joint_velocities;
};
```

**Error State** (for UKF covariance):
```cpp
// Covariance dimension: 2 * (3 + 3 + DOF)
// - Root position error: 3D vector
// - Root orientation error: 3D axis-angle (NOT 4D quaternion)
// - Joint angle errors: DOF-dimensional vector
// - Velocity errors: same as above
```

**Why Error-State for Orientation?**

Problem: Quaternions have 4 parameters but only 3 DOF (unit norm constraint). Filtering in 4D leads to:
- Covariance matrix not positive definite
- Need for normalization after each update
- Gimbal lock issues

Solution: Filter the **error** in tangent space (axis-angle), then apply to nominal quaternion:

```cpp
// UKF state: [pos_error (3), ori_error (3), joint_errors (DOF), vel_errors (6+DOF)]
// Nominal state: [pos, quat, joints, vels] (stored separately)

// After UKF update:
Eigen::Vector3d ori_error = ukf_state.segment<3>(3);
Eigen::Quaterniond error_quat = axis_angle_to_quaternion(ori_error);
state.root_orientation = state.root_orientation * error_quat;  // Compose
ori_error.setZero();  // Reset error to zero
```

### 2.2 State Class Implementation

```cpp
class State {
public:
    // Construction
    State() = default;
    explicit State(const Skeleton& skeleton);

    // --- Full State Access ---

    const Eigen::Vector3d& root_position() const { return root_position_; }
    const Eigen::Quaterniond& root_orientation() const { return root_orientation_; }
    const Eigen::VectorXd& joint_angles() const { return joint_angles_; }

    void set_root_position(const Eigen::Vector3d& pos) { root_position_ = pos; }
    void set_root_orientation(const Eigen::Quaterniond& ori) {
        root_orientation_ = ori.normalized();
    }
    void set_joint_angles(const Eigen::VectorXd& angles) { joint_angles_ = angles; }

    // --- Velocity Access ---

    const Eigen::Vector3d& root_velocity() const { return root_velocity_; }
    const Eigen::Vector3d& root_angular_velocity() const { return root_angular_velocity_; }
    const Eigen::VectorXd& joint_velocities() const { return joint_velocities_; }

    // --- Error State Conversion ---

    // Convert to error state vector (for UKF covariance)
    Eigen::VectorXd to_error_vector() const;

    // Apply error state update (from UKF)
    void apply_error_update(const Eigen::VectorXd& error_state);

    // --- Pinocchio Interface ---

    // Configuration vector: [root_pos, root_ori (quaternion xyzw), joint_angles]
    Eigen::VectorXd to_pinocchio_q() const;

    // Velocity vector: [root_vel, root_angvel, joint_vels]
    Eigen::VectorXd to_pinocchio_v() const;

    // Construct from Pinocchio vectors
    static State from_pinocchio(const Eigen::VectorXd& q,
                                const Eigen::VectorXd& v,
                                const Skeleton& skeleton);

    // --- Serialization ---

    nlohmann::json to_json() const;
    static State from_json(const nlohmann::json& j, const Skeleton& skeleton);

private:
    // Full state
    Eigen::Vector3d root_position_{0, 0, 0};
    Eigen::Quaterniond root_orientation_{1, 0, 0, 0};  // Identity
    Eigen::VectorXd joint_angles_;

    Eigen::Vector3d root_velocity_{0, 0, 0};
    Eigen::Vector3d root_angular_velocity_{0, 0, 0};
    Eigen::VectorXd joint_velocities_;

    // Cached for error state
    mutable Eigen::Vector3d root_orientation_error_{0, 0, 0};
};
```

**Error State Mapping**:
```cpp
Eigen::VectorXd State::to_error_vector() const {
    int dof = joint_angles_.size();
    Eigen::VectorXd error(2 * (3 + 3 + dof));

    error.segment<3>(0) = root_position_;
    error.segment<3>(3) = root_orientation_error_;  // Tangent space
    error.segment(6, dof) = joint_angles_;
    error.segment<3>(6 + dof) = root_velocity_;
    error.segment<3>(9 + dof) = root_angular_velocity_;
    error.segment(12 + dof, dof) = joint_velocities_;

    return error;
}

void State::apply_error_update(const Eigen::VectorXd& error_state) {
    int dof = joint_angles_.size();

    // Position: direct addition
    root_position_ += error_state.segment<3>(0);

    // Orientation: compose with error quaternion
    Eigen::Vector3d ori_error = error_state.segment<3>(3);
    if (ori_error.norm() > 1e-9) {
        Eigen::Quaterniond dq = axis_angle_to_quaternion(ori_error);
        root_orientation_ = (root_orientation_ * dq).normalized();
    }
    root_orientation_error_.setZero();  // Reset

    // Joints: direct addition
    joint_angles_ += error_state.segment(6, dof);

    // Velocities: direct addition
    root_velocity_ += error_state.segment<3>(6 + dof);
    root_angular_velocity_ += error_state.segment<3>(9 + dof);
    joint_velocities_ += error_state.segment(12 + dof, dof);

    // Apply joint limits
    apply_joint_limits();
}
```

---

## 3. Observation Model

### 3.1 Observation Structure

```cpp
struct Observation {
    int camera_id;
    int marker_id;

    // Coordinates
    Eigen::Vector2d position;           // Undistorted (for UKF)
    Eigen::Vector2d position_distorted; // Original (for diagnostics)

    // Metadata
    double confidence;  // [0, 1] from OpenPose
    double timestamp;   // Seconds
    int frame_idx;      // Camera-specific frame number

    // Noise model (per-observation)
    double measurement_noise_std() const {
        // Weight by confidence: low confidence → high noise
        return base_noise / std::max(confidence, 0.1);
    }

    static constexpr double base_noise = 5.0;  // pixels
};

struct ObservationSequence {
    int camera_id;
    std::string camera_name;
    std::vector<Observation> observations;

    // Query by time range (not exact equality for doubles)
    std::vector<Observation> get_in_range(
        double t_start, double t_end) const;
struct ObservationSet {
    std::map<std::string, ObservationSequence> sequences;  // camera_name → sequence
    int person_id;

    // Global queries
    std::vector<Observation> get_all_at_time(double t) const;
    std::vector<Observation> get_all_at_frame(
        const std::map<std::string, int>& frame_indices) const;

    // Time range
    double min_time() const;
    double max_time() const;
    std::vector<double> get_unique_timestamps() const;
};
```

### 3.2 Loading Observations

**OpenPose JSON Format**:
```json
{
  "version": 1.3,
  "people": [
    {
      "person_id": [0],
      "pose_keypoints_2d": [
        x0, y0, conf0,
        x1, y1, conf1,
        ...  // 25 keypoints × 3 = 75 values
      ],
      "hand_left_keypoints_2d": [...],   // 21 keypoints
      "hand_right_keypoints_2d": [...]   // 21 keypoints
    }
  ]
}
```

**Loader Implementation**:
```cpp
ObservationSet load_openpose_observations(
    const std::filesystem::path& base_dir,
    const std::vector<Camera>& cameras,
    const Skeleton& skeleton,
    int person_id,
    int start_frame = 0,
    int max_frames = -1,
    double min_confidence = 0.0)
{
    ObservationSet obs_set;
    obs_set.person_id = person_id;

    for (const auto& camera : cameras) {
        std::filesystem::path pose_dir = base_dir / "pose" / camera.name();

        ObservationSequence sequence;
        sequence.camera_id = camera_id;
        sequence.camera_name = camera.name();

        // Iterate over JSON files
        for (int frame_idx = start_frame; ; ++frame_idx) {
            if (max_frames > 0 && frame_idx >= start_frame + max_frames) {
                break;
            }

            std::string filename = fmt::format("{:012d}_keypoints.json", frame_idx);
            std::filesystem::path json_path = pose_dir / filename;

            if (!std::filesystem::exists(json_path)) {
                break;  // No more frames
            }

            // Parse JSON
            std::ifstream ifs(json_path);
            nlohmann::json j = nlohmann::json::parse(ifs);

            // Find person
            auto& people = j["people"];
            if (people.size() <= person_id) {
                continue;  // Person not detected
            }

            auto& person = people[person_id];
            auto& keypoints = person["pose_keypoints_2d"];

            // Extract observations
            for (const auto& marker : skeleton.markers()) {
                int coco_id = marker.coco_id;
                if (coco_id < 0 || coco_id >= 25) {
                    continue;  // Not a COCO body keypoint
                }

                double x = keypoints[coco_id * 3 + 0];
                double y = keypoints[coco_id * 3 + 1];
                double conf = keypoints[coco_id * 3 + 2];

                if (conf < min_confidence) {
                    continue;  // Too low confidence
                }

                Observation obs;
                obs.camera_id = camera_id;
                obs.marker_id = marker.id;
                obs.position_distorted = Eigen::Vector2d(x, y);
                obs.position = camera.undistort(obs.position_distorted);
                obs.confidence = conf;
                obs.frame_idx = frame_idx;
                obs.timestamp = camera.get_timestamp(frame_idx);

                sequence.observations.push_back(obs);
            }
        }

        obs_set.sequences[camera.name()] = std::move(sequence);
    }

    return obs_set;
}
```

---

## 4. UKF Implementation Details

### 4.1 Sigma Point Generation (Manifold-Aware)

```cpp
class SigmaPointGenerator {
public:
    SigmaPointGenerator(double alpha = 1e-3, double beta = 2.0, double kappa = 0.0)
        : alpha_(alpha), beta_(beta), kappa_(kappa) {}

    struct SigmaPoints {
        std::vector<Eigen::VectorXd> points;  // Error state vectors
        Eigen::VectorXd weights_mean;
        Eigen::VectorXd weights_cov;
    };

    SigmaPoints generate(const Eigen::VectorXd& mean,
                        const Eigen::MatrixXd& covariance) const {
        int n = mean.size();
        double lambda = alpha_ * alpha_ * (n + kappa_) - n;

        SigmaPoints result;
        result.points.reserve(2 * n + 1);
        result.weights_mean.resize(2 * n + 1);
        result.weights_cov.resize(2 * n + 1);

        // Weights
        result.weights_mean(0) = lambda / (n + lambda);
        result.weights_cov(0) = result.weights_mean(0) + (1 - alpha_ * alpha_ + beta_);

        for (int i = 1; i < 2 * n + 1; ++i) {
            result.weights_mean(i) = 0.5 / (n + lambda);
            result.weights_cov(i) = result.weights_mean(i);
        }

        // Cholesky decomposition
        Eigen::MatrixXd L = ((n + lambda) * covariance).llt().matrixL();

        // Central sigma point
        result.points.push_back(mean);

        // Positive perturbations
        for (int i = 0; i < n; ++i) {
            result.points.push_back(mean + L.col(i));
        }

        // Negative perturbations
        for (int i = 0; i < n; ++i) {
            result.points.push_back(mean - L.col(i));
        }

        return result;
    }

private:
    double alpha_, beta_, kappa_;
};
```

**Converting Error State to Full State**:
```cpp
std::vector<State> UKF::sigma_points_to_states(
    const State& nominal_state,
    const std::vector<Eigen::VectorXd>& error_points) const
{
    std::vector<State> states;
    states.reserve(error_points.size());

    for (const auto& error : error_points) {
        State state = nominal_state;  // Copy
        state.apply_error_update(error);
        states.push_back(state);
    }

    return states;
}
```

### 4.2 Prediction Step

```cpp
State UKF::predict(const State& state, double dt) {
    // 1. Get error state
    Eigen::VectorXd error_mean = state.to_error_vector();

    // 2. Generate sigma points in error space
    auto sigma_points_error = sigma_gen_.generate(error_mean, covariance_);

    // 3. Convert to full states
    auto sigma_states = sigma_points_to_states(state, sigma_points_error.points);

    // 4. Propagate each sigma point through process model
    #pragma omp parallel for if(params_.n_jobs != 1)
    for (size_t i = 0; i < sigma_states.size(); ++i) {
        sigma_states[i] = process_model(sigma_states[i], dt);
    }

    // 5. Compute predicted mean state
    State predicted_state = compute_mean_state(sigma_states, sigma_points_error.weights_mean);

    // 6. Compute predicted covariance
    Eigen::MatrixXd predicted_cov = compute_covariance(
        sigma_states, predicted_state, sigma_points_error.weights_cov);

    // 7. Add process noise
    covariance_ = predicted_cov + process_noise_;

    return predicted_state;
}
```

**Process Model** (constant velocity):
```cpp
State UKF::process_model(const State& state, double dt) const {
    State next_state = state;

    // Position: p' = p + v * dt
    next_state.root_position() += state.root_velocity() * dt;

    // Orientation: q' = q ⊗ exp(ω * dt / 2)
    Eigen::Vector3d dangle = state.root_angular_velocity() * dt;
    if (dangle.norm() > 1e-9) {
        Eigen::Quaterniond dq = axis_angle_to_quaternion(dangle);
        next_state.root_orientation() = (state.root_orientation() * dq).normalized();
    }

    // Joint angles: θ' = θ + ω * dt
    next_state.joint_angles() += state.joint_velocities() * dt;

    // Velocities: constant
    // (Process noise will add uncertainty)

    // Apply joint limits
    next_state.apply_joint_limits(skeleton_);

    return next_state;
}
```

### 4.3 Update Step

```cpp
State UKF::update(const State& predicted_state,
                  const std::vector<Observation>& observations)
{
    if (observations.empty()) {
        return predicted_state;
    }

    // 1. Generate sigma points from predicted state
    Eigen::VectorXd error_mean = predicted_state.to_error_vector();
    auto sigma_points_error = sigma_gen_.generate(error_mean, covariance_);
    auto sigma_states = sigma_points_to_states(predicted_state, sigma_points_error.points);

    // 2. Predict measurements for each sigma point
    std::vector<Eigen::VectorXd> sigma_measurements(sigma_states.size());

    #pragma omp parallel for if(params_.n_jobs != 1)
    for (size_t i = 0; i < sigma_states.size(); ++i) {
        sigma_measurements[i] = predict_measurements_for_state(sigma_states[i], observations);
    }

    // 3. Compute predicted measurement mean
    Eigen::VectorXd z_pred = compute_mean_measurement(
        sigma_measurements, sigma_points_error.weights_mean);

    // 4. Compute innovation covariance
    Eigen::MatrixXd Pzz = compute_measurement_covariance(
        sigma_measurements, z_pred, sigma_points_error.weights_cov);

    // 5. Add measurement noise
    Eigen::MatrixXd R = build_measurement_noise_matrix(observations);
    Pzz += R;

    // 6. Outlier rejection
    auto inliers = filter_outliers(observations, z_pred, Pzz);

    if (inliers.empty()) {
        return predicted_state;  // No valid observations
    }

    // 7. Recompute with only inliers
    z_pred = predict_measurements_for_observations(predicted_state, inliers);
    Pzz = recompute_innovation_covariance(sigma_measurements, inliers, z_pred,
                                         sigma_points_error.weights_cov, R);

    // 8. Compute cross-covariance
    Eigen::MatrixXd Pxz = compute_cross_covariance(
        sigma_states, predicted_state, sigma_measurements, z_pred,
        sigma_points_error.weights_cov);

    // 9. Compute Kalman gain
    kalman_gain_ = Pxz * Pzz.inverse();

    // 10. Compute innovation
    Eigen::VectorXd z_actual = extract_measurements(inliers);
    innovation_ = z_actual - z_pred;

    // 11. Update state
    Eigen::VectorXd error_update = kalman_gain_ * innovation_;
    State updated_state = predicted_state;
    updated_state.apply_error_update(error_update);

    // 12. Update covariance
    covariance_ = covariance_ - kalman_gain_ * Pzz * kalman_gain_.transpose();

    // Ensure positive definite (numerical stability)
    covariance_ = 0.5 * (covariance_ + covariance_.transpose());

    return updated_state;
}
```

**Measurement Prediction**:
```cpp
Eigen::VectorXd UKF::predict_measurements_for_state(
    const State& state,
    const std::vector<Observation>& observations) const
{
    // Forward kinematics: compute all marker positions
    auto markers_3d = fk_.compute_marker_positions(state);

    // Project to cameras
    std::vector<double> predictions;
    predictions.reserve(observations.size() * 2);

    for (const auto& obs : observations) {
        const auto& camera = cameras_[obs.camera_id];
        const auto& marker_3d = markers_3d[obs.marker_id];

        Eigen::Vector2d proj = camera.project_undistorted(marker_3d);
        predictions.push_back(proj.x());
        predictions.push_back(proj.y());
    }

    return Eigen::Map<Eigen::VectorXd>(predictions.data(), predictions.size());
}
```

### 4.4 Outlier Rejection

```cpp
std::vector<Observation> UKF::filter_outliers(
    const std::vector<Observation>& observations,
    const Eigen::VectorXd& predicted_measurements,
    const Eigen::MatrixXd& innovation_covariance) const
{
    if (params_.outlier_threshold <= 0.0) {
        return observations;  // Disabled
    }

    std::vector<Observation> inliers;
    outliers_.clear();

    for (size_t i = 0; i < observations.size(); ++i) {
        const auto& obs = observations[i];

        // Innovation for this observation
        Eigen::Vector2d z_actual(obs.position.x(), obs.position.y());
        Eigen::Vector2d z_pred(predicted_measurements(i * 2),
                              predicted_measurements(i * 2 + 1));
        Eigen::Vector2d innovation = z_actual - z_pred;

        // Innovation covariance for this observation
        Eigen::Matrix2d S = innovation_covariance.block<2, 2>(i * 2, i * 2);

        // Mahalanobis distance
        double mahal_dist = std::sqrt(innovation.transpose() * S.inverse() * innovation);

        if (mahal_dist > params_.outlier_threshold) {
            // Outlier
            outliers_.push_back(OutlierInfo{
                .observation_idx = static_cast<int>(i),
                .camera_id = obs.camera_id,
                .marker_id = obs.marker_id,
                .mahalanobis_distance = mahal_dist,
                .residual = innovation
            });
        } else {
            // Inlier
            inliers.push_back(obs);
        }
    }

    return inliers;
}
```

---

## 5. Output Format: ZIP Archive

### 5.1 Rationale

**Problem with HDF5**:
- Cross-language compatibility issues (Python ↔ C++)
- Schema versioning complexity
- Binary format (hard to inspect/debug)
- Heavy dependency

**ZIP Solution**:
- JSON inside (human-readable, universal)
- Easy compression
- Standard library support (C++20 or libarchive)
- Simple versioning (JSON schema)

### 5.2 Archive Structure

```
tracked_person_0.zip
├── metadata.json           # Skeleton, cameras, parameters
├── states/
│   ├── frame_0000000.json
│   ├── frame_0000001.json
│   └── ...
├── diagnostics/
│   ├── frame_0000000.json
│   ├── frame_0000001.json
│   └── ...
└── summary.json            # Aggregated statistics
```

### 5.3 JSON Schemas

**metadata.json**:
```json
{
  "version": "1.0.0",
  "skeleton": {
    "name": "simple_humanoid",
    "root": "pelvis",
    "joints": [...]
  },
  "cameras": [...],
  "parameters": {
    "process_noise_std": 0.1,
    "measurement_noise_std": 5.0,
    "outlier_threshold": 5.991,
    "fps": 30.0
  },
  "person_id": 0,
  "frame_count": 1000,
  "timestamp_range": [0.0, 33.333]
}
```

**states/frame_NNNNNNN.json**:
```json
{
  "frame_idx": 42,
  "timestamp": 1.4,
  "state": {
    "root_position": [0.5, 0.2, 1.0],
    "root_orientation": [1.0, 0.0, 0.0, 0.0],
    "joint_angles": {...},
    "root_velocity": [0.1, 0.0, 0.0],
    "root_angular_velocity": [0.0, 0.0, 0.0],
    "joint_velocities": {...}
  },
  "covariance_diag": [...]  // Diagonal of covariance (for compactness)
}
```

**diagnostics/frame_NNNNNNN.json**:
```json
{
  "frame_idx": 42,
  "innovation": [...],
  "outliers": [
    {
      "camera_id": 1,
      "marker_id": 15,
      "mahalanobis_distance": 7.3,
      "residual": [12.5, -8.2]
    }
  ],
  "marker_errors": {
    "nose": 2.3,
    "left_shoulder": 4.1,
    ...
  },
  "mean_error": 3.5,
  "max_error": 8.7,
  "predict_time_ms": 12.3,
  "update_time_ms": 45.6
}
```

**summary.json**:
```json
{
  "mean_reprojection_error": 3.8,
  "median_reprojection_error": 3.2,
  "outlier_rate": 0.05,
  "mean_frame_time_ms": 62.4,
  "total_tracking_time_sec": 62.4,
  "frames_per_second": 16.0
}
```

### 5.4 Export Implementation

```cpp
#include <archive.h>
#include <archive_entry.h>

void export_diagnostics_zip(const std::string& path,
                            const TrackerDiagnostics& diagnostics,
                            const Skeleton& skeleton,
                            const std::vector<Camera>& cameras)
{
    // Create ZIP archive
    struct archive* a = archive_write_new();
    archive_write_set_format_zip(a);
    archive_write_open_filename(a, path.c_str());

    // Write metadata
    {
        nlohmann::json metadata = build_metadata_json(skeleton, cameras, diagnostics);
        std::string content = metadata.dump(2);
        write_archive_entry(a, "metadata.json", content);
    }

    // Write states
    for (const auto& frame_diag : diagnostics.get_frames()) {
        std::string filename = fmt::format("states/frame_{:07d}.json", frame_diag.frame_idx);
        nlohmann::json state_json = frame_diag.state.to_json();
        state_json["frame_idx"] = frame_diag.frame_idx;
        state_json["timestamp"] = frame_diag.timestamp;
        // Optionally add covariance diagonal
        write_archive_entry(a, filename, state_json.dump(2));
    }

    // Write diagnostics
    for (const auto& frame_diag : diagnostics.get_frames()) {
        std::string filename = fmt::format("diagnostics/frame_{:07d}.json", frame_diag.frame_idx);
        nlohmann::json diag_json = frame_diagnostics_to_json(frame_diag);
        write_archive_entry(a, filename, diag_json.dump(2));
    }

    // Write summary
    {
        nlohmann::json summary = diagnostics.summary_to_json();
        write_archive_entry(a, "summary.json", summary.dump(2));
    }

    archive_write_close(a);
    archive_write_free(a);
}

void write_archive_entry(struct archive* a, const std::string& filename,
                        const std::string& content)
{
    struct archive_entry* entry = archive_entry_new();
    archive_entry_set_pathname(entry, filename.c_str());
    archive_entry_set_size(entry, content.size());
    archive_entry_set_filetype(entry, AE_IFREG);
    archive_entry_set_perm(entry, 0644);

    archive_write_header(a, entry);
    archive_write_data(a, content.data(), content.size());
    archive_entry_free(entry);
}
```

---

## 6. CLI Application

### 6.1 Argument Parser (CLI11)

```cpp
#include <CLI/CLI.hpp>

int main(int argc, char** argv) {
    CLI::App app{"Joint-space motion capture tracker"};

    // Input files
    std::string skeleton_path, calib_path, base_dir, output_dir;
    app.add_option("--skeleton", skeleton_path, "Skeleton YAML file")->required();
    app.add_option("--calib", calib_path, "Camera calibration TOML")->required();
    app.add_option("--base-dir", base_dir, "Base directory with pose/ subdirectory")->required();
    app.add_option("--output-dir", output_dir, "Output directory")->required();

    // Tracking parameters
    int person_id = 0;
    std::vector<int> start_frames = {0};
    int max_frames = -1;
    double fps = 30.0;

    app.add_option("--person-id", person_id, "Person ID to track");
    app.add_option("--start-frame", start_frames, "Starting frame index (per camera or single value)");
    app.add_option("--max-frames", max_frames, "Maximum frames to track");
    app.add_option("--fps", fps, "Frame rate in Hz");

    // Filter parameters
    std::vector<std::string> active_groups;
    double min_confidence = 0.3;
    double process_noise_std = 0.1;
    double measurement_noise_std = 5.0;
    double outlier_threshold = 0.0;
    int n_jobs = -1;

    app.add_option("--active-groups", active_groups, "Active skeleton groups");
    app.add_option("--min-confidence", min_confidence, "Minimum OpenPose confidence");
    app.add_option("--process-noise-std", process_noise_std, "Process noise std dev");
    app.add_option("--measurement-noise-std", measurement_noise_std, "Measurement noise std dev");
    app.add_option("--outlier-threshold", outlier_threshold, "Mahalanobis threshold (0=disabled)");
    app.add_option("--n-jobs", n_jobs, "Parallel jobs (-1=all cores)");

    // Output options
    bool create_bvh = false;
    bool create_statistics = false;
    bool save_diagnostics = false;

    app.add_flag("--create-bvh", create_bvh, "Export BVH file");
    app.add_flag("--create-statistics", create_statistics, "Generate statistics CSV");
    app.add_flag("--save-diagnostics", save_diagnostics, "Save diagnostics to ZIP");

    CLI11_PARSE(app, argc, argv);

    // ... rest of main
}
```

---

## 7. Key Interfaces Summary

### 7.1 Core Types
```cpp
State               // Full skeletal state
Skeleton            // Hierarchy + limits
Camera              // Projection + distortion + sync
Observation         // 2D detection with metadata
ObservationSet      // Multi-camera observations
```

### 7.2 Algorithms
```cpp
ForwardKinematics   // Joint angles → 3D markers (Pinocchio)
InverseKinematics   // 3D markers → joint angles (optimization)
UKF                 // State estimation with outlier rejection
Triangulation       // Multi-camera 2D → 3D
```

### 7.3 I/O
```cpp
load_skeleton_from_yaml
load_cameras_from_toml
load_openpose_observations
load_sync_metadata
export_trc
export_json
export_diagnostics_zip
```

### 7.4 Tracking
```cpp
Tracker             // Orchestrates UKF + observations
TrackerCallbacks    // Progress reporting
TrackerDiagnostics  // Performance metrics
```

---

## 8. Dependencies Summary

| Library | Version | Purpose | License |
|---------|---------|---------|---------|
| Eigen | 3.4+ | Linear algebra | MPL2 |
| Pinocchio | 3.9+ | Forward kinematics | BSD-2-Clause |
| fmt | 10.0+ | Formatting | MIT |
| yaml-cpp | Latest | Skeleton config | MIT |
| toml11 | Latest | Calibration | MIT |
| nlohmann/json | Latest | Data interchange | MIT |
| CLI11 | 2.3+ | CLI parsing | BSD-3-Clause |
| libarchive | Latest | ZIP export | BSD |
| OpenMP | Latest | Parallelization | OpenMP License |
| GTest | 1.12+ | Testing | BSD-3-Clause |

All dependencies are permissive licenses (no GPL).

---

## 9. Next Steps

1. **Prototype Core Models** (State, Skeleton, Camera)
2. **Integrate Pinocchio** (ForwardKinematics wrapper)
3. **Implement UKF** (Prediction + Update)
4. **Add I/O** (Loaders + Exporters)
5. **Build CLI** (Orchestration)
6. **Test & Validate** (Regression vs Python)
7. **Optimize** (Profiling + Parallelization)
8. **Document** (API docs + user guide)
