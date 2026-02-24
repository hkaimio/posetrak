/**
 * @file tracker.hpp
 * @brief Main tracking interface orchestrating UKF-based pose tracking
 *
 * The Tracker class provides a high-level interface for markerless motion capture:
 * 1. Initialize from first frame (triangulation + IK)
 * 2. Track through sequence (predict + update cycle)
 * 3. Report results via callbacks
 */

#pragma once

#include "posetrak/core/camera.hpp"
#include "posetrak/core/config.hpp"
#include "posetrak/core/observation.hpp"
#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/state.hpp"
#include "posetrak/filters/ukf.hpp"
#include "posetrak/filters/update_result.hpp"
#include "posetrak/kinematics/forward_kinematics.hpp"
#include "posetrak/kinematics/inverse_kinematics.hpp"
#include "posetrak/kinematics/triangulation.hpp"
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace posetrak {

// TrackerConfig is defined in config.hpp

/**
 * @brief Tracking result for a single frame
 */
struct TrackingResult {
    double timestamp;            ///< Frame timestamp
    State state;                 ///< Estimated state
    Eigen::MatrixXd covariance;  ///< State covariance
    UpdateResult update_info;    ///< Update diagnostics (outliers, NIS, etc.)
    int num_observations_used;   ///< Number of observations that passed filtering
    bool tracking_lost;          ///< True if tracking failed (no observations, etc.)
    std::string failure_reason;  ///< Reason for failure if tracking_lost=true
};

/**
 * @brief Callback function types for progress reporting
 */
using FrameCallback = std::function<void(TrackingResult const&)>;
using ProgressCallback = std::function<void(int frame_idx, int total_frames)>;

/**
 * @brief Main tracking class orchestrating UKF-based pose estimation
 *
 * Workflow:
 * 1. initialize(): Triangulate first frame, solve IK for initial pose
 * 2. track_frame(): Predict forward, update with observations
 * 3. Callbacks report progress and per-frame results
 *
 * Example:
 * @code
 * Tracker tracker(skeleton, cameras, config);
 * tracker.set_frame_callback([](auto const& result) {
 *     std::cout << "Frame " << result.timestamp << ": "
 *               << result.num_observations_used << " obs\n";
 * });
 *
 * if (tracker.initialize(initial_observations, 0.0)) {
 *     for (auto const& [timestamp, obs_set] : observation_sequence) {
 *         tracker.track_frame(obs_set, timestamp);
 *     }
 * }
 * @endcode
 */
class Tracker {
   public:
    /**
     * @brief Construct tracker for given skeleton and camera setup
     *
     * @param skeleton Skeleton model with joint hierarchy and markers
     * @param cameras Map of camera_id → Camera
     * @param config Tracking configuration parameters
     */
    Tracker(std::shared_ptr<const Skeleton> skeleton,
            std::unordered_map<int, Camera> const& cameras,
            TrackerConfig const& config = TrackerConfig{});

    /**
     * @brief Initialize tracker from first frame observations
     *
     * Steps:
     * 1. Triangulate visible markers from 2D observations
     * 2. Solve IK to find joint configuration matching marker positions
     * 3. Initialize UKF state and covariance
     *
     * @param observations Initial frame observations
     * @param timestamp Initial timestamp
     * @return True if initialization succeeded, false otherwise
     *
     * @note Requires sufficient markers visible in min_cameras_for_init cameras
     * @note Sets is_initialized() to true on success
     */
    bool initialize(std::vector<Observation> const& observations, double timestamp);

    /**
     * @brief Initialize tracker from skeleton rest pose (bypass IK)
     *
     * Initializes UKF with skeleton in rest configuration (zero joint angles).
     * Useful when IK initialization fails or for quick testing.
     *
     * @param timestamp Initial timestamp
     *
     * @note Sets is_initialized() to true
     */
    void initialize_from_rest_pose(double timestamp);

    /**
     * @brief Initialize tracker from a given state (e.g., from Python tracker)
     *
     * Initializes UKF with the provided state. Useful for validation by
     * initializing from a known-good external tracker.
     *
     * @param initial_state State to initialize with
     * @param timestamp Initial timestamp
     *
     * @note Sets is_initialized() to true
     */
    void initialize_from_state(State const& initial_state, double timestamp);

    /**
     * @brief Track a single frame
     *
     * Performs predict-update cycle:
     * 1. Predict state forward by dt = timestamp - last_timestamp
     * 2. Update with observations (with outlier rejection)
     * 3. Report result via callback
     *
     * @param observations Frame observations
     * @param timestamp Frame timestamp
     * @return Tracking result for this frame
     *
     * @note Requires is_initialized() == true
     * @throws std::runtime_error if not initialized
     */
    TrackingResult track_frame(std::vector<Observation> const& observations, double timestamp);

    /**
     * @brief Check if tracker is initialized and ready
     */
    bool is_initialized() const { return initialized_; }

    /**
     * @brief Get current state estimate
     * @note Only valid if is_initialized() == true
     */
    State const& state() const { return ukf_->state(); }

    /**
     * @brief Get current covariance estimate
     * @note Only valid if is_initialized() == true
     */
    Eigen::MatrixXd const& covariance() const { return ukf_->covariance(); }

    /**
     * @brief Set callback for per-frame results
     */
    void set_frame_callback(FrameCallback callback) { frame_callback_ = std::move(callback); }

    /**
     * @brief Set callback for progress updates
     */
    void set_progress_callback(ProgressCallback callback) {
        progress_callback_ = std::move(callback);
    }

    /**
     * @brief Reset tracker to uninitialized state
     */
    void reset();

    /**
     * @brief Get UKF for debug configuration
     * @return Pointer to UKF (or nullptr if not initialized)
     */
    UnscentedKalmanFilter* get_ukf() { return ukf_.get(); }

    /**
     * @brief Get the tracker's ForwardKinematics (initialized for the active layout).
     * @return Reference to FK (only valid after initialization)
     */
    ForwardKinematics& fk() { return *fk_; }

    /**
     * @brief Number of active child filters (0 in monolithic mode).
     * @note Only valid after initialization.
     */
    size_t num_children() const { return children_.size(); }

   private:
    /**
     * @brief Child filter — a self-contained subtree UKF.
     *
     * Runs after the parent predict+update each frame. The anchor joint's
     * world-transform (from the parent FK) is injected as a fixed floating
     * root before each predict, so the child only estimates its own joints.
     */
    struct ChildFilter {
        std::shared_ptr<const SkeletonLayout> layout;
        std::unique_ptr<UnscentedKalmanFilter> ukf;
        std::unique_ptr<ForwardKinematics> fk;
        std::unique_ptr<pinocchio::Model> model;
        std::unique_ptr<pinocchio::Data> data;
        std::map<std::string, pinocchio::FrameIndex> marker_frame_map;

        /// Name of the anchor joint in the parent skeleton whose world-transform
        /// is injected as this child's fixed floating root each frame.
        std::string anchor_joint_name;

        /// merge_map[i] = state_index in the full-skeleton layout for child DOF i.
        /// Used to write child joint angles back into the parent UKF state.
        std::vector<int> merge_map;

        /// Maps full-skeleton marker index → child-skeleton marker index.
        /// Only entries for markers that exist in this child filter are populated.
        /// Used in run_child_step to remap observation marker_ids before passing
        /// to the child UKF (which indexes its own subtree skeleton's markers()).
        std::unordered_map<int, int> marker_id_remap;

        double measurement_noise_std = 2.0;
        double outlier_threshold = 4.0;
    };

    /**
     * @brief Initialize UKF with given state and initial covariance
     */
    void initialize_ukf(State const& initial_state, double timestamp);

    /**
     * @brief Build all child filters from config_.child_filters.
     *
     * Called at the end of initialize_ukf() with the global initial state
     * (full-skeleton State from IK or rest pose). Must NOT use ukf_->state()
     * because the parent UKF only holds parent-group DOFs.
     */
    void build_children(State const& global_initial_state);

    /**
     * @brief Extract child-relevant joint angles from a full-skeleton State.
     *
     * @param global_state  Full-skeleton State (from IK or rest-pose init).
     * @param child_layout  Layout for the child filter.
     * @return A compact State with only the child's joints; root at identity.
     */
    State slice_state_for_child(State const& global_state,
                                SkeletonLayout const& child_layout) const;

    /**
     * @brief Run the parent (full-body) predict+update step.
     *
     * Calls ukf_->predict(), ukf_->update(), writes debug output, then
     * refreshes fk_ so children can query world_transform() immediately after.
     *
     * @param obs  Observations for this frame
     * @param dt   Time elapsed since last frame
     * @param timestamp  Frame timestamp (used only to populate the result)
     * @return TrackingResult for the parent filter
     */
    TrackingResult run_parent_step(std::vector<Observation> const& obs, double dt,
                                   double timestamp);

    /**
     * @brief Run one child filter's predict+update step (stub for Phase 3h).
     */
    void run_child_step(ChildFilter& child, std::vector<Observation> const& obs, double dt);

    /**
     * @brief Check if we have sufficient observations for tracking
     */
    bool has_sufficient_observations(std::vector<Observation> const& observations) const;

    std::shared_ptr<const Skeleton> skeleton_;
    std::unordered_map<int, Camera> const& cameras_;
    TrackerConfig config_;

    // Components
    std::unique_ptr<UnscentedKalmanFilter> ukf_;
    std::unique_ptr<Triangulator> triangulator_;
    std::unique_ptr<InverseKinematics> ik_solver_;
    std::unique_ptr<ForwardKinematics> fk_;

    // Pinocchio structures (owned by Tracker)
    std::unique_ptr<pinocchio::Model> model_;
    std::unique_ptr<pinocchio::Data> data_;
    std::map<std::string, pinocchio::FrameIndex> marker_frame_map_;

    // Child filters (populated in Phase 3h)
    std::vector<ChildFilter> children_{};

    // State
    bool initialized_ = false;
    double last_timestamp_ = 0.0;

    // Callbacks
    FrameCallback frame_callback_;
    ProgressCallback progress_callback_;
};

}  // namespace posetrak
