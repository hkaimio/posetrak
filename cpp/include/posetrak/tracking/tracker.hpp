// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

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
#include "posetrak/filters/rts_smoother.hpp"
#include "posetrak/filters/ukf.hpp"
#include "posetrak/filters/update_result.hpp"
#include "posetrak/kinematics/forward_kinematics.hpp"
#include "posetrak/kinematics/inverse_kinematics.hpp"
#include "posetrak/kinematics/triangulation.hpp"
#include "posetrak/tracking/marker_prediction.hpp"
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    double predict_ms = 0.0;     ///< Wall time for ukf_.predict() in milliseconds
    double update_ms = 0.0;      ///< Wall time for ukf_.update() in milliseconds

    // Predict sub-step timings (from PredictResult)
    double p_sigma_gen_ms = 0.0;
    double p_propagate_ms = 0.0;
    double p_mean_cov_ms = 0.0;
    double p_rts_ms = 0.0;

    // Update sub-step timings (from UpdateResult)
    double u_fk1_ms = 0.0;
    double u_s_ms = 0.0;
    double u_outlier_ms = 0.0;
    double u_fk2_ms = 0.0;
    double u_inlier_ms = 0.0;
    double u_kalman_ms = 0.0;
    double u_cov_update_ms = 0.0;
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
     * @brief Initialize a fixed-root ("child filter") Tracker: triangulate its
     * own markers and solve IK, with root pose supplied externally instead of
     * estimated from body landmarks.
     *
     * Unlike initialize() -- which analytically estimates root position/
     * orientation from hip/shoulder markers before running IK, irrelevant here
     * since a child filter's root comes from the parent's smoothed trajectory,
     * not its own markers -- this triangulates the child's own markers, runs
     * IK from a rest-pose guess anchored at the known root, and always
     * discards whatever root IK returned: root_position/root_orientation are
     * authoritative regardless of IK's result. Falls back to rest pose (zero
     * joint angles) at the known root if fewer than 3 markers triangulate
     * (e.g. occluded at sequence start).
     *
     * @param observations     This frame's raw per-camera detections for the
     *                         child's own marker group.
     * @param root_position    World-frame position of the freeflyer joint
     *                         (TrackerConfig::fixed_root_joint_name).
     * @param root_orientation World-frame orientation of the freeflyer joint.
     * @param timestamp        Initial timestamp.
     * @return True (matches initialize()'s signature; always succeeds, via
     *         the rest-pose fallback if triangulation is insufficient).
     *
     * @note Requires TrackerConfig::fixed_root_joint_name to be set.
     * @note Call at most once per Tracker instance. IK runs against the
     *       original full-skeleton model, before initialize_ukf() rebuilds
     *       model_/data_/fk_ to the fixed-root subtree -- ik_solver_ and
     *       triangulator_ are never rebuilt, so a second call after that
     *       rebuild would use stale (dangling) Pinocchio structures. This
     *       is a pre-existing constraint shared with initialize() (also
     *       expected to run once), not new to fixed-root mode -- just newly
     *       load-bearing here since a "re-initialize after tracking loss"
     *       policy would naturally want a second call. Re-initialize by
     *       constructing a new Tracker instead.
     * @note Sets is_initialized() to true.
     */
    bool initialize_with_fixed_root(std::vector<Observation> const& observations,
                                    Eigen::Vector3d const& root_position,
                                    Eigen::Quaterniond const& root_orientation, double timestamp);

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
     * @brief Predict-only half of the track_frame() cycle (marker-mocap design doc
     * dot-assignment-architecture-design.md §5.1): advances the UKF by dt with no
     * observations yet, and stashes everything update_step() needs to finish the
     * frame later.
     *
     * Exists so an external orchestrator can call predict_step() on every
     * dot-bearing subject in a scene *before* any of them commits an update --
     * the shared dot-assignment phase (design doc §5.2) needs every subject's
     * live prediction for the same instant, and
     * UnscentedKalmanFilter::predict() mutates state in place (not a peekable
     * dry run), so there is no other way to get that live prediction without
     * holding the result open across the resolution step. track_frame() itself
     * is just predict_step() immediately followed by update_step() -- a
     * subject with no dots to resolve sees no behavioural change.
     *
     * @param dt Time elapsed since last frame. Not validated here (unlike
     *        track_frame()'s own negative-dt guard) -- the caller (track_frame()
     *        itself, or an orchestrator's driving loop) owns that decision.
     * @note Requires is_initialized() == true.
     * @note Must be followed by exactly one update_step() call before this
     *       Tracker's state reflects "this frame" -- calling update_step()
     *       without a preceding predict_step() throws, and calling
     *       predict_step() again without an intervening update_step() discards
     *       the pending predict and starts over (not a supported usage, but not
     *       one worth guarding against either).
     */
    void predict_step(double dt);

    /**
     * @brief Every unlabeled_points-track marker's MarkerPrediction (design doc
     * §6), evaluated at the state predict_step() just computed, for one camera.
     *
     * This is the query surface a shared dot-assignment orchestrator (design
     * doc §5.2) actually calls -- it never touches Skeleton/State/covariance
     * directly, so which MarkerPrediction implementation runs (closed-form
     * rigid, §6.1, or the deferred general/articulated case) is entirely this
     * Tracker's own decision, invisible to the caller.
     *
     * @param camera_id Camera to project into.
     * @return skeleton().markers() index -> MarkerPrediction, for every
     *         unlabeled_points marker that projects in front of this camera.
     *         Empty if this skeleton declares no unlabeled_points input track.
     * @note Requires a predict_step() call this frame (i.e. call between
     *       predict_step() and update_step(), not before or after).
     * @throws std::runtime_error if skeleton_->is_rigid_body() is false --
     *         the general/articulated implementation is deferred (design doc
     *         §6) pending a real articulated dot-augmented capture to design
     *         it against -- or if camera_id is unknown.
     */
    std::unordered_map<int, MarkerPrediction> predict_dot_slot_predictions(int camera_id) const;

    /**
     * @brief Update-only half of the track_frame() cycle -- consumes the
     * prediction stashed by the most recent predict_step() call and finishes
     * the frame: observation annotation, sufficiency check, UKF update, NIS
     * feedback, RTS smoother bookkeeping, FK refresh on the posterior state,
     * and the same last_timestamp_/frame_count_/prev_observations_/
     * frame_callback_ bookkeeping track_frame() itself does after a
     * successful frame.
     *
     * @param observations Frame observations -- raw, not yet velocity-mode
     *        annotated (this method annotates them itself, exactly as
     *        track_frame() already does).
     * @param timestamp Frame timestamp (the same value used to compute the dt
     *        passed to the preceding predict_step() call).
     * @return TrackingResult for this frame -- identical to what
     *         track_frame() would have returned for the same
     *         (observations, timestamp) pair.
     * @throws std::runtime_error if called without a preceding predict_step()
     *         this frame.
     */
    TrackingResult update_step(std::vector<Observation> const& observations, double timestamp);

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
     * @brief Get ForwardKinematics for access by child filters (Phase 3h+)
     * @return Pointer to FK (or nullptr if not initialized)
     */
    ForwardKinematics* get_fk() { return fk_.get(); }

    /**
     * @brief Inject this frame's externally-known root transform (fixed-root
     * / child-filter mode -- see TrackerConfig::fixed_root_joint_name).
     *
     * Forwards to UnscentedKalmanFilter::set_root_transform(). The caller
     * (e.g. a hierarchical solver's Stage B driving this Tracker off a
     * TrajectoryStream) must call this once per frame before track_frame(),
     * with the value for the frame about to be tracked -- root position and
     * orientation are held fixed through that frame's predict+update, never
     * estimated. A safety no-op if this Tracker's layout has a floating
     * root (i.e. fixed_root_joint_name was never set) -- see
     * docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md.
     *
     * @note Requires is_initialized() == true.
     */
    void set_external_root_transform(Eigen::Vector3d const& position,
                                     Eigen::Quaterniond const& orientation);

    /**
     * @brief Per-marker projected-pixel-position standard deviation for one camera,
     * via linearized error propagation from the current posterior covariance
     * (error-improvements Phase 5, "Per-marker anchor uncertainty").
     *
     * pixel_covariance ≈ J P J^T, where J is the 2×n Jacobian of the
     * FK-then-project map at the current state (n = error_state_dim()) and P is
     * covariance(). J is assembled from: an analytic closed-form root position/
     * orientation block (matching the exact retraction State::apply_error_update()
     * and the sigma-point generator use -- see phase5-cross-person-plan.md for the
     * derivation) plus Pinocchio's per-joint frame Jacobian (computeJointJacobians +
     * getFrameJacobian, the same machinery InverseKinematics::compute_jacobian()
     * already uses) for every non-root joint DOF, selected per SkeletonLayout's
     * active_dof_mask. The returned std is the isotropic RMS of the 2x2 pixel
     * covariance's diagonal (sqrt((var_u + var_v) / 2)), matching how the rest of
     * the noise model treats noise as a single per-axis scalar.
     *
     * Lazy by design: only computes for the requested markers (each costs one
     * small GEMM chain, not a full sigma-point regeneration), so callers should
     * only ask for markers in currently-active contact pairs.
     *
     * @param camera_id Camera to project into; markers are omitted from the
     *        result if this id is unknown or a marker projects behind the camera.
     * @param marker_ids Marker indices (skeleton().markers() positions) to compute.
     * @return marker_id -> pixel std, for however many of marker_ids succeeded.
     */
    std::unordered_map<int, double> marker_projection_std(int camera_id,
                                                          std::vector<int> const& marker_ids) const;

    // ── RTS Smoothing ────────────────────────────────────────────────────────────

    /**
     * @brief Enable accumulation of RTS smoother data during tracking.
     *
     * Must be called BEFORE any track_frame() calls.  When enabled, the
     * tracker stores the per-frame forward-pass cache required by the RTS
     * backward sweep.  This slightly increases memory usage (O(N * edim^2))
     * but does not affect the filtered estimates produced by track_frame().
     *
     * @param enable  True to enable, false to disable (and clear the cache).
     */
    void enable_smoothing(bool enable);

    /**
     * @brief Run the RTS backward pass and return smoothed estimates.
     *
     * Requires enable_smoothing(true) to have been called before tracking,
     * and at least one tracked frame.
     *
     * @return Smoothed frames in chronological order (same order as track_frame calls).
     * @throws std::runtime_error if smoothing was not enabled or cache is empty.
     */
    std::vector<SmoothedFrame> smooth() const;

   private:
    /**
     * @brief Initialize UKF with given state and initial covariance
     */
    void initialize_ukf(State const& initial_state, double timestamp);

    /**
     * @brief Triangulate 3D positions for markers with enough per-camera
     * observations (>= config_.min_cameras_for_init). Shared by initialize()
     * and initialize_with_fixed_root(); does not set init_marker_positions_
     * or any other member -- callers do that with the return value.
     */
    std::map<std::string, Eigen::Vector3d>
    triangulate_markers(std::vector<Observation> const& observations) const;

    /**
     * @brief Analytic rigid-body initialization for a root-only skeleton (marker-mocap
     * algorithms doc §4.2) -- a prop skeleton generated from a marker body definition
     * (design §5.3) has one free-flyer root and no other active joints, so a closed-form
     * Kabsch/Umeyama fit of body-local marker positions (rest-pose FK) to their
     * triangulated world positions is better-conditioned than IK and cannot fall into a
     * local minimum the way IK can for this degenerate case.
     *
     * Called from initialize() when the skeleton has no non-root active joints, in place
     * of the human-skeleton analytic-estimate + limb-warm-start + IK path.
     *
     * @param marker_positions Triangulated world positions, keyed by marker name (from
     *        triangulate_markers()); must have >= 3 entries (checked by the caller).
     * @param timestamp Frame timestamp, passed through to initialize_ukf().
     * @return false if fewer than 3 triangulated markers have a body-local counterpart,
     *         the layout is collinear (not yet supported -- see algorithms doc §4 step 4
     *         and design doc open question 3), or the fit residual exceeds
     *         config_.rigid_init_max_residual_m (caller's existing retry-on-a-later-frame
     *         loop applies, same as a failed human-skeleton IK).
     */
    bool initialize_rigid_body(std::map<std::string, Eigen::Vector3d> const& marker_positions,
                               double timestamp);

    /**
     * @brief Check if we have sufficient observations for tracking
     */
    bool has_sufficient_observations(std::vector<Observation> const& observations) const;

    /**
     * @brief Print per-marker 3D FK vs triangulated error for a given state.
     *
     * Uses init_marker_positions_ as the 3D reference (triangulated at init time).
     * Prints root position, per-marker errors sorted descending, and RMS.
     */
    void print_init_debug(State const& state, std::string const& label) const;

    /**
     * @brief Annotate observations for velocity-mode cameras.
     *
     * For each observation whose camera_id is in config_.velocity_mode_camera_ids,
     * sets mode=VELOCITY and fills prev_position from prev_observations_ if available.
     * Observations with no stored previous pixel stay in POSITION mode (first frame behaviour).
     */
    std::vector<Observation>
    build_annotated_observations(std::vector<Observation> const& observations) const;

    /**
     * @brief Windowed NIS/DOF bookkeeping for one NIS-feedback scope (Mechanism B) --
     * see docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md.
     * Not intrinsic to the UKF's own state (see UnscentedKalmanFilter::
     * set_nis_feedback_scopes() doc comment), so it lives here instead.
     */
    struct NisFeedbackScopeWindow {
        std::string name;
        std::unordered_set<std::string> joint_names;
        std::deque<double> step_sum_mahal_sq;  ///< Sum of mahalanobis_distance^2, per step
        std::deque<int> step_dof_count;        ///< 2 per attributed observation (u,v), per step
        double running_sum_mahal_sq = 0.0;
        int running_dof_count = 0;
    };
    std::vector<NisFeedbackScopeWindow> nis_feedback_windows_;
    /// marker_name -> parent joint name, built once at construction from skeleton_.
    std::unordered_map<std::string, std::string> marker_to_joint_name_;

    /**
     * @brief Update each configured scope's windowed NIS/DOF from this step's
     * per-observation Mahalanobis distances, and push the resulting multiplier
     * into the UKF for the next predict() call. No-op if nis_feedback_windows_
     * is empty (Mechanism B not configured).
     */
    void update_nis_feedback_scopes(std::vector<ObservationResult> const& observations);

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

    // State
    bool initialized_ = false;
    double last_timestamp_ = 0.0;
    int frame_count_ = 0;

    // Triangulated 3-D positions from the initialization frame, used by debug output.
    std::map<std::string, Eigen::Vector3d> init_marker_positions_;

    // predict_step()/update_step() split (design doc §5.1): everything run_parent_step()
    // used to compute in its own predict half and consume immediately in its update
    // half now has to survive the gap where an orchestrator resolves dot assignment.
    // pending_predict_result_ in particular holds PredictResult::cross_cov_future --
    // launched async inside ukf_->predict(), deliberately resolved late (in
    // update_step()) so its work overlaps ukf_->update()'s -- so the whole struct, not
    // just the prior state/cov, must live here across the boundary.
    bool predict_pending_ = false;
    std::optional<PredictResult> pending_predict_result_;
    std::optional<State> pending_prior_state_;  // State has no default ctor
    Eigen::MatrixXd pending_prior_cov_;
    double pending_predict_ms_ = 0.0;

    // Previous-frame undistorted pixels per camera and marker, for velocity-mode cameras.
    // Populated at the end of each successful track_frame() call.
    std::unordered_map<int, std::unordered_map<int, Eigen::Vector2d>> prev_observations_;

    // RTS smoother
    bool smoothing_enabled_ = false;
    std::vector<FrameSmootherData> smoother_cache_;

    // Callbacks
    FrameCallback frame_callback_;
    ProgressCallback progress_callback_;
};

}  // namespace posetrak
