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

    // Metadata for obs_blob encoding (set during construction)
    std::vector<std::string> camera_labels_;  ///< Sorted camera labels (obs_blob camera axis)
    std::vector<std::string>
        marker_names_;  ///< Marker names in skeleton order (obs_blob marker axis)
};

}  // namespace posetrak
