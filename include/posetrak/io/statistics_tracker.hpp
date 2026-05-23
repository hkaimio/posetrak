#pragma once

#include <Eigen/Dense>

#include <nlohmann/json.hpp>

#include "posetrak/filters/update_result.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace posetrak {

/**
 * @brief Per-frame tracking statistics
 */
struct FrameStatistics {
    int frame;                           ///< Frame number
    double timestamp;                    ///< Timestamp in seconds
    int num_observations;                ///< Total number of observations
    int num_inliers;                     ///< Number of accepted observations
    int num_outliers;                    ///< Number of rejected observations
    double mean_reprojection_error;      ///< Mean reprojection error (pixels)
    double max_reprojection_error;       ///< Max reprojection error (pixels)
    double covariance_min_eigenvalue;    ///< Minimum eigenvalue of covariance
    double covariance_condition_number;  ///< Condition number (max/min eigenvalue)
    double nis_value;                    ///< Normalized Innovation Squared
    bool tracking_lost;                  ///< Whether tracking was lost this frame
    double predict_ms = 0.0;             ///< Wall time for predict step (ms)
    double update_ms = 0.0;              ///< Wall time for update step (ms)

    // Predict sub-step timings
    double p_sigma_gen_ms = 0.0;
    double p_propagate_ms = 0.0;
    double p_mean_cov_ms = 0.0;
    double p_rts_ms = 0.0;

    // Update sub-step timings
    double u_fk1_ms = 0.0;
    double u_s_ms = 0.0;
    double u_outlier_ms = 0.0;
    double u_fk2_ms = 0.0;
    double u_inlier_ms = 0.0;
    double u_kalman_ms = 0.0;
    double u_cov_update_ms = 0.0;
};

/**
 * @brief Accumulates and exports tracking statistics
 *
 * Tracks per-frame statistics during tracking and exports:
 * - tracking_stats.csv: per-frame metrics
 * - overall_stats.json: summary statistics
 */
class StatisticsTracker {
   public:
    /**
     * @brief Add statistics for a single frame
     *
     * @param frame Frame number
     * @param timestamp Timestamp in seconds
     * @param update_result UKF update result with diagnostics
     * @param covariance Current state covariance matrix
     * @param tracking_lost Whether tracking was lost
     */
    void add_frame_stats(int frame, double timestamp, UpdateResult const& update_result,
                         Eigen::MatrixXd const& covariance, bool tracking_lost,
                         double predict_ms = 0.0, double update_ms = 0.0,
                         double p_sigma_gen_ms = 0.0, double p_propagate_ms = 0.0,
                         double p_mean_cov_ms = 0.0, double p_rts_ms = 0.0, double u_fk1_ms = 0.0,
                         double u_s_ms = 0.0, double u_outlier_ms = 0.0, double u_fk2_ms = 0.0,
                         double u_inlier_ms = 0.0, double u_kalman_ms = 0.0,
                         double u_cov_update_ms = 0.0);

    /**
     * @brief Write per-frame statistics to CSV
     *
     * @param output_path Path to output CSV file
     */
    void write_frame_stats(std::filesystem::path const& output_path) const;

    /**
     * @brief Write summary statistics to JSON
     *
     * @param output_path Path to output JSON file
     * @param metadata Additional metadata to include (sequence name, skeleton info, etc.)
     */
    void write_summary_stats(std::filesystem::path const& output_path,
                             nlohmann::json const& metadata) const;

    /**
     * @brief Get mean reprojection error across all frames
     * @return Mean reprojection error in pixels
     */
    double mean_reprojection_error() const;

    /**
     * @brief Get mean number of inliers across all frames
     * @return Mean number of inliers
     */
    double mean_num_inliers() const;

    /**
     * @brief Get outlier rate (num_outliers / num_observations)
     * @return Outlier rate [0, 1]
     */
    double outlier_rate() const;

    /**
     * @brief Get number of frames tracked
     * @return Number of frames with statistics
     */
    size_t num_frames() const { return frame_stats_.size(); }

    /**
     * @brief Get number of frames where tracking was lost
     * @return Count of lost frames
     */
    size_t num_lost_frames() const;

   private:
    std::vector<FrameStatistics> frame_stats_;  ///< Per-frame statistics
};

}  // namespace posetrak
