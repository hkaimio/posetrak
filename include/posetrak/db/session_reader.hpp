#pragma once

#include <posetrak/core/camera.hpp>
#include <posetrak/core/config.hpp>
#include <posetrak/core/observation.hpp>
#include <posetrak/core/skeleton.hpp>

#include <sqlite3.h>

#include <map>
#include <string>

namespace posetrak {

/// @brief TrackerConfig loaded from DB plus the tracker sample rate
struct DbTrackerConfig {
    TrackerConfig tracker;
    double tracker_fps = 100.0;
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
    /// @return ObservationSet ready for the tracker
    ObservationSet load_observations(std::string const& sequence_id,
                                     std::map<std::string, Camera> const& cameras,
                                     Skeleton const& skeleton, double min_confidence = 0.1,
                                     int person_id = 0);

   private:
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
