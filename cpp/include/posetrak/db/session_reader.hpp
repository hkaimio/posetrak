// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <posetrak/core/camera.hpp>
#include <posetrak/core/config.hpp>
#include <posetrak/core/observation.hpp>
#include <posetrak/core/skeleton.hpp>

#include <sqlite3.h>

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace posetrak {

/// @brief TrackerConfig loaded from DB plus the tracker sample rate
struct DbTrackerConfig {
    TrackerConfig tracker;
    double tracker_fps = 100.0;
};

/// @brief One tracker_config_stages row: per-stage tuning overrides for a
/// hierarchical solver child group. Every field is nullopt when its column
/// is NULL, meaning "inherit the parent tracker_configs row's value" -- see
/// docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md.
/// min_inliers_ratio/max_innovation_norm are accepted here (the schema has
/// them) but not yet consumed by TrackerConfig -- no corresponding field
/// exists there yet.
struct StageConfigOverrides {
    std::string group_name;
    std::optional<double> process_noise_std;
    std::optional<double> process_noise_vel_std;
    std::optional<double> velocity_half_life_s;
    std::optional<double> pose_noise_std;
    std::optional<double> calib_noise_std;
    std::optional<double> outlier_threshold;
    std::optional<double> min_inliers_ratio;
    std::optional<double> max_innovation_norm;
    std::optional<double> init_joint_std;
    std::optional<double> init_velocity_std;
};

/// @brief Time range for a pose observation sequence
struct SequenceInfo {
    double time_start_s = 0.0;
    double time_end_s = -1.0;
};

/// @brief Key IDs associated with a pose observation sequence
struct SequenceMetadata {
    std::string session_id;
    std::string extrinsic_calibration_id;
    std::string sync_config_id;
};

/// @brief One anonymous reflective-dot candidate detection, undistorted and
/// resolved to a Camera, but not yet resolved to a marker identity -- that
/// resolution is what the shared dot-assignment phase does at tracking time
/// (see docs/roadmap/features/marker-based-mocap/dot-assignment-architecture-design.md).
/// Deliberately not an Observation: there is no marker_id yet.
struct UnlabeledCandidate {
    int camera_id;
    int frame_idx;
    double timestamp;
    Eigen::Vector2d position;            ///< Undistorted pixels, matches Observation::position
    Eigen::Vector2d position_distorted;  ///< Original distorted pixels, for diagnostics
    /// No per-candidate detector confidence exists in the underlying blob (unlike a
    /// pose keypoint) -- always 1.0. Kept as a field for shape-parity with Observation
    /// rather than dropped, in case a future detector version adds a real one.
    double confidence;
    double area;         ///< Blob area in pixels, from the detector (diagnostics/tuning only)
    double compactness;  ///< Blob compactness, from the detector (diagnostics/tuning only)
};

/// @brief Reads tracking data from a per-session SQLite database
///
/// Opens the DB read-only and provides typed accessors for all data needed
/// to run the tracker.
class SessionReader {
   public:
    /// @brief Open a session database file
    /// @param db_path Path to the SQLite session DB
    /// @throws std::runtime_error if the file cannot be opened
    explicit SessionReader(std::string const& db_path);
    ~SessionReader();

    // Non-copyable, movable
    SessionReader(SessionReader const&) = delete;
    SessionReader& operator=(SessionReader const&) = delete;
    SessionReader(SessionReader&&) noexcept;
    SessionReader& operator=(SessionReader&&) noexcept;

    /// @brief Resolve a UUID prefix to a full ID within a table
    ///
    /// @param table Table name (must have an `id` TEXT PRIMARY KEY column)
    /// @param prefix Full UUID or a unique prefix
    /// @return Full UUID matching the prefix
    /// @throws std::runtime_error if zero or more than one record matches
    std::string resolve_id(std::string const& table, std::string const& prefix);

    /// @brief Load skeleton YAML content from the skeletons table
    /// @param skeleton_id Primary key or unique prefix of the skeleton record
    /// @return Raw YAML content string
    std::string load_skeleton_yaml(std::string const& skeleton_id);

    /// @brief Load tracker configuration from the tracker_configs table
    /// @param config_id Primary key of the tracker_config record
    /// @return DbTrackerConfig with TrackerConfig defaults overridden by non-NULL columns
    DbTrackerConfig load_tracker_config(std::string const& config_id);

    /// @brief Load every tracker_config_stages row for config_id.
    ///
    /// An empty return means config_id runs monolithic -- this is the
    /// existence-based hierarchical-mode toggle the design doc specifies:
    /// "A tracker_config_id with any tracker_config_stages rows is what
    /// selects hierarchical mode; one without runs monolithic, unchanged."
    /// @param config_id tracker_configs primary key
    /// @return One entry per stage (group_name), in no particular order
    std::vector<StageConfigOverrides> load_tracker_config_stages(std::string const& config_id);

    /// @brief Load time range information for a pose observation sequence
    /// @param sequence_id Primary key of the pose_observation_sequences record
    /// @return SequenceInfo with start and end timestamps
    SequenceInfo load_sequence_info(std::string const& sequence_id);

    /// @brief Load key IDs (session, extrinsic calibration, sync config) for a sequence
    /// @param sequence_id Primary key of the pose_observation_sequences record
    /// @return SequenceMetadata containing associated IDs
    /// @throws std::runtime_error if the sequence is not found
    SequenceMetadata load_sequence_metadata(std::string const& sequence_id);

    /// @brief Load cameras by resolving all IDs from a pose observation sequence
    ///
    /// Derives session_id, extrinsic_calibration_id, and sync_config_id by following
    /// the link chain: pose_observation_sequences → shots → (session, extrinsics, sync).
    ///
    /// @param sequence_id pose_observation_sequences primary key
    /// @return Map from camera label to Camera (ordered by label for deterministic ID assignment)
    std::map<std::string, Camera> load_cameras_for_sequence(std::string const& sequence_id);

    /// @brief Load cameras for a session with calibration and sync data applied
    ///
    /// Camera integer IDs are assigned in ascending camera_key (label) order.
    ///
    /// @param session_id mocap_sessions primary key
    /// @param extrinsic_calibration_id extrinsic_calibrations primary key
    /// @param sync_config_id sync_configs primary key
    /// @return Map from camera label to Camera (ordered by label for deterministic ID assignment)
    std::map<std::string, Camera> load_cameras(std::string const& session_id,
                                               std::string const& extrinsic_calibration_id,
                                               std::string const& sync_config_id);

    /// @brief Load pose observations for a sequence
    ///
    /// @param sequence_id pose_observation_sequences primary key
    /// @param cameras Camera map (label → Camera) as returned by load_cameras()
    /// @param skeleton Skeleton whose markers have COCO IDs to map keypoints
    /// @param min_confidence Minimum keypoint confidence to include (default 0.1)
    /// @param person_id Person index within each frame blob (default 0)
    /// @param use_relative_obs Emit RELATIVE (child-minus-parent) observations alongside POSITION
    /// @param relative_min_conf Minimum confidence for both markers of a RELATIVE pair
    /// @param pose_noise_std Pose estimation error std (pixels); sets RELATIVE noise_std_override
    /// @param cross_pair_max_px Pixel radius for spatial cross-pair RELATIVE obs (0 = disabled)
    /// @param cross_pair_max_n  Max cross-pairs per frame per camera (sorted by proximity)
    /// @param edited_kp_noise_std When > 0, keypoints overridden by a pose_observation_edits row
    ///   get this as noise_std_override and Observation::force_inlier = true (see
    ///   TrackerConfig::edited_kp_noise_std). 0 = disabled (edits use the normal formula/gate).
    /// @return ObservationSet ready for the tracker
    ObservationSet load_observations(std::string const& sequence_id,
                                     std::map<std::string, Camera> const& cameras,
                                     Skeleton const& skeleton, double min_confidence = 0.1,
                                     int person_id = 0, bool use_relative_obs = false,
                                     double relative_min_conf = 0.5, double pose_noise_std = 0.0,
                                     double cross_pair_max_px = 0.0, int cross_pair_max_n = 10,
                                     double edited_kp_noise_std = 0.0);

    /// @brief Load anonymous reflective-dot candidates for a sequence.
    ///
    /// Reads every `pose_observations` row with `source='dots'` (the
    /// finalized-data counterpart of a `detection_keypoints` row with
    /// `region_type='dots'`), decodes each row's variable-length
    /// `float32[N,4]` blob via `db::decode_dot_candidates()`, and undistorts
    /// positions the same way `load_observations()` does for labeled
    /// keypoints.
    ///
    /// Not filtered by `person_id`: dot candidates are scene-wide detections,
    /// not yet tied to any one tracked subject -- that is exactly the
    /// ambiguity the shared dot-assignment phase resolves at tracking time.
    ///
    /// @param sequence_id pose_observation_sequences primary key
    /// @param cameras Camera map (label → Camera) as returned by load_cameras()
    /// @return Every candidate across every camera/frame with a `source='dots'`
    ///         row, ordered by camera then frame. Empty if the sequence has no
    ///         such rows (every sequence before the dot-detection write path
    ///         exists).
    std::vector<UnlabeledCandidate>
    load_unlabeled_candidates(std::string const& sequence_id,
                              std::map<std::string, Camera> const& cameras);

   private:
    /// @brief Read pose_observation_sequences.pixels_are_undistorted for sequence_id.
    /// @return true (safe default for pre-flag data) if NULL or the row is absent.
    bool load_pixels_are_undistorted(std::string const& sequence_id);

    /// @brief Build camera_instance_id -> Camera const* for every camera actually
    /// used by this sequence's captured videos, resolved against `cameras`.
    /// Shared by load_observations() and load_unlabeled_candidates() -- both need
    /// the identical instance-id-to-Camera resolution.
    /// @note The returned pointers alias `cameras`; callers must not outlive it.
    std::unordered_map<std::string, Camera const*>
    load_instance_camera_map(std::string const& sequence_id,
                             std::map<std::string, Camera> const& cameras);

    sqlite3* db_{};

    /// @brief RAII wrapper around a prepared SQLite statement
    struct Stmt {
        sqlite3_stmt* ptr = nullptr;

        Stmt() = default;
        explicit Stmt(sqlite3* db, char const* sql);
        ~Stmt();

        Stmt(Stmt const&) = delete;
        Stmt& operator=(Stmt const&) = delete;
        Stmt(Stmt&&) noexcept;
        Stmt& operator=(Stmt&&) noexcept;

        /// @brief Step the statement
        /// @return true if a row is available, false if done
        /// @throws std::runtime_error on SQLite error
        bool step();

        /// @brief Reset the statement and clear bindings for reuse
        void reset();
    };
};

}  // namespace posetrak
