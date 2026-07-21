#pragma once
#include <posetrak/core/camera.hpp>
#include <posetrak/core/skeleton.hpp>
#include <posetrak/filters/update_result.hpp>

#include <Eigen/Core>

#include <sqlite3.h>

#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace posetrak {

/// @brief Writes tracking results to a session SQLite database.
///
/// Creates a tracking_runs row on construction and batches tracking_results
/// inserts for efficiency. Call flush() (or let the destructor do it) when done.
class ResultWriter {
   public:
    /// @brief Open db_path read-write and create a tracking_runs row.
    /// @param db_path Path to the session SQLite database
    /// @param sequence_id pose_observation_sequences primary key
    /// @param skeleton_id skeletons primary key
    /// @param config_id tracker_configs primary key
    /// @param extrinsic_calibration_id extrinsic_calibrations primary key
    /// @param sync_config_id sync_configs primary key
    /// @param person_id Person index stored in tracking_results rows
    /// @param cameras Label-keyed Camera map (from load_cameras_for_sequence)
    /// @param skeleton Skeleton whose markers provide the marker_names JSON
    ResultWriter(std::string const& db_path, std::string const& sequence_id,
                 std::string const& skeleton_id, std::string const& config_id,
                 std::string const& extrinsic_calibration_id, std::string const& sync_config_id,
                 int person_id, std::map<std::string, Camera> const& cameras,
                 Skeleton const& skeleton);

    /// @brief Attach to an existing tracking run for read-modify-write patching.
    ///
    /// Does not insert a tracking_runs row -- used by a hierarchical solver's
    /// child stage to patch into the rows a parent-stage ResultWriter already
    /// created for the same person. Loads camera_labels_/marker_names_ from
    /// the existing tracking_runs row's active_camera_ids/marker_names JSON
    /// columns, so patch_obs_results() works without the caller re-deriving
    /// them. Only patch_frame()/patch_obs_results() are valid to call on an
    /// instance constructed this way; write_frame()/write_smoothed_frame()/
    /// write_obs_results() would insert duplicate-key rows and are not meant
    /// to be used with it.
    /// @param db_path Path to the session SQLite database
    /// @param run_id Existing tracking_runs row to patch into
    /// @param person_id Person index (must match the parent's tracking_results rows)
    ResultWriter(std::string const& db_path, std::string const& run_id, int person_id);

    ~ResultWriter();

    ResultWriter(ResultWriter const&) = delete;
    ResultWriter& operator=(ResultWriter const&) = delete;

    /// @brief Return the tracking run UUID created in the constructor
    std::string const& run_id() const { return run_id_; }

    /// @brief Write one forward-pass frame. Batches internally; call flush() when done.
    void write_frame(int step, double timestamp, Eigen::VectorXd const& state,
                     Eigen::MatrixXd const& covariance, bool tracking_lost,
                     int n_inlier_observations, double cov_condition_number, double nis_value,
                     int nis_dof);

    /// @brief Write one smoothed frame (is_smoothed=1).
    /// Pass an empty MatrixXd when covariance is unavailable — encodes as zero-length blob.
    void write_smoothed_frame(int step, double timestamp, Eigen::VectorXd const& state,
                              Eigen::MatrixXd const& covariance);

    /// @brief Write per-observation results for one forward-pass frame.
    ///
    /// Encodes observations as a float32 blob shaped [n_cameras, n_markers, 8]:
    ///   [obs_x, obs_y, pred_x, pred_y, mahal_dist, used_in_update, is_outlier, pad]
    /// NaN is written for slots with no observation for that (camera, marker) pair.
    ///
    /// @param step Tracker step index (must match a previously written write_frame() step)
    /// @param observations Per-observation results from UpdateResult::observations
    void write_obs_results(int step, std::vector<ObservationResult> const& observations);

    /// @brief Read-modify-write patch of an existing tracking_results row's
    /// state/cov_diag blobs.
    ///
    /// Reads the current row for (run_id(), person_id(), step, is_smoothed),
    /// decodes state/cov_diag as float64 vectors, overwrites the given
    /// indices with the given values, and writes the row back. Index
    /// semantics are entirely up to the caller (e.g. an offset derived from
    /// SkeletonLayout::build_index_map_from() into the flat layout
    /// State::to_error_vector() produces) -- ResultWriter has no
    /// skeleton/layout knowledge, generically enough that no group/joint
    /// name is ever hardcoded here.
    ///
    /// @param step Tracker step index of an already-written row
    /// @param is_smoothed Which row family to patch (false=filtered, true=smoothed)
    /// @param state_indices Indices into the row's decoded state vector to overwrite
    /// @param state_values Replacement values, same length as state_indices
    /// @param cov_diag_indices Indices into the row's decoded cov_diag vector to overwrite
    /// @param cov_diag_values Replacement values, same length as cov_diag_indices
    /// @throws std::runtime_error if no matching row exists, or the row's state is NULL
    /// @throws std::invalid_argument if an indices/values pair has mismatched length,
    ///         an index is out of range, or cov_diag_indices is non-empty but the row's
    ///         cov_diag is NULL
    void patch_frame(int step, bool is_smoothed, std::vector<int> const& state_indices,
                     std::vector<double> const& state_values,
                     std::vector<int> const& cov_diag_indices = {},
                     std::vector<double> const& cov_diag_values = {});

    /// @brief Read-modify-write patch of an existing tracking_obs_results row's obs_blob.
    ///
    /// Reads the current row's obs_blob for (run_id(), person_id(), step),
    /// and for each entry in @p observations writes into its (camera, marker)
    /// slot -- UNLESS the entry's marker_name is in @p parent_owned_markers,
    /// in which case that slot is left untouched (parent-wins rule for
    /// markers a parent and child stage both solve, e.g. a wrist marker
    /// shared between a body and a hand group). The pad field (blob index 7,
    /// unused before this method existed) is set per entry from @p
    /// pair_diff_reconstructed: 1.0 marks an entry whose actual/predicted
    /// pixels were reconstructed from a PAIR_DIFF difference (see
    /// reconstruct_pair_diff_absolute() in relative_observations.hpp) rather
    /// than measured directly, 0.0 marks a native absolute-pixel entry.
    ///
    /// @param step Tracker step index of an already-written tracking_obs_results row
    /// @param observations Per-observation results to patch in (child stage's own results,
    ///        already in absolute pixels -- see reconstruct_pair_diff_absolute())
    /// @param pair_diff_reconstructed Same length as observations; the pad-field mode flag
    ///        per entry
    /// @param parent_owned_markers Marker names never overwritten (shared-marker slots)
    /// @throws std::runtime_error if no matching row exists, or its obs_blob size doesn't
    ///         match this run's camera/marker counts
    /// @throws std::invalid_argument if pair_diff_reconstructed's length doesn't match
    ///         observations'
    void patch_obs_results(int step, std::vector<ObservationResult> const& observations,
                           std::vector<uint8_t> const& pair_diff_reconstructed,
                           std::vector<std::string> const& parent_owned_markers = {});

    /// @brief Flush any pending batched rows to the database.
    void flush();

   private:
    sqlite3* db_{};
    std::string run_id_;
    int person_id_;

    static constexpr int kBatchSize = 500;

    // Pending rows: (step, timestamp, state_blob, cov_diag_blob,
    //                tracking_lost, n_inliers, cov_condition, nis_value, nis_dof, is_smoothed)
    using FrameRow = std::tuple<int, double, std::vector<uint8_t>, std::vector<uint8_t>, int, int,
                                double, double, int, int>;
    std::vector<FrameRow> pending_;

    void flush_pending();
    static std::vector<uint8_t> encode_vector(Eigen::VectorXd const& v);
    static std::vector<double> decode_doubles(void const* blob, int n_bytes);
    static std::vector<uint8_t> encode_doubles(std::vector<double> const& v);
    static void apply_patch(std::vector<double>& vec, std::vector<int> const& indices,
                            std::vector<double> const& values, char const* field_name);

    // Metadata for obs_blob encoding (set during construction)
    std::vector<std::string> camera_labels_;  ///< Sorted camera labels (obs_blob camera axis)
    std::vector<std::string>
        marker_names_;  ///< Marker names in skeleton order (obs_blob marker axis)
};

}  // namespace posetrak
