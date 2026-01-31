#include "posetrak/io/statistics_tracker.hpp"

#include <fmt/core.h>

#include <fstream>
#include <numeric>
#include <stdexcept>

namespace posetrak {

void StatisticsTracker::add_frame_stats(int frame, double timestamp,
                                        UpdateResult const& update_result,
                                        Eigen::MatrixXd const& covariance, bool tracking_lost) {
    FrameStatistics stats;
    stats.frame = frame;
    stats.timestamp = timestamp;
    stats.num_observations = update_result.num_observations;
    stats.num_inliers = update_result.num_inliers;
    stats.num_outliers = update_result.num_outliers;
    stats.tracking_lost = tracking_lost;

    // Compute reprojection errors from observation results
    double sum_error = 0.0;
    double max_error = 0.0;
    int count = 0;

    for (auto const& obs_result : update_result.observations) {
        if (!obs_result.is_outlier) {
            double error = obs_result.innovation.norm();
            sum_error += error;
            max_error = std::max(max_error, error);
            count++;
        }
    }

    stats.mean_reprojection_error = count > 0 ? sum_error / count : 0.0;
    stats.max_reprojection_error = max_error;

    // Compute covariance statistics
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(covariance);
    auto eigenvalues = solver.eigenvalues();

    stats.covariance_min_eigenvalue = eigenvalues.minCoeff();
    double max_eigenvalue = eigenvalues.maxCoeff();
    stats.covariance_condition_number = stats.covariance_min_eigenvalue > 1e-12
                                            ? max_eigenvalue / stats.covariance_min_eigenvalue
                                            : 1e12;

    // NIS value
    stats.nis_value = update_result.nis;

    frame_stats_.push_back(stats);
}

void StatisticsTracker::write_frame_stats(std::filesystem::path const& output_path) const {
    std::ofstream file(output_path);
    if (!file) {
        throw std::runtime_error(fmt::format("Failed to open {}", output_path.string()));
    }

    // Write header
    file << "frame,timestamp,num_observations,num_inliers,num_outliers,"
            "mean_reprojection_error,max_reprojection_error,"
            "covariance_min_eigenvalue,covariance_condition_number,"
            "nis_value,tracking_lost\n";

    // Write data rows
    for (auto const& stats : frame_stats_) {
        file << fmt::format("{},{},{},{},{},{},{},{},{},{},{}\n", stats.frame, stats.timestamp,
                            stats.num_observations, stats.num_inliers, stats.num_outliers,
                            stats.mean_reprojection_error, stats.max_reprojection_error,
                            stats.covariance_min_eigenvalue, stats.covariance_condition_number,
                            stats.nis_value, stats.tracking_lost ? "true" : "false");
    }
}

void StatisticsTracker::write_summary_stats(std::filesystem::path const& output_path,
                                            nlohmann::json const& metadata) const {
    nlohmann::json summary = metadata;

    // Overall statistics
    summary["total_frames"] = frame_stats_.size();
    summary["frames_tracked"] = frame_stats_.size() - num_lost_frames();
    summary["frames_lost"] = num_lost_frames();

    summary["mean_reprojection_error"] = mean_reprojection_error();
    summary["mean_num_inliers"] = mean_num_inliers();
    summary["outlier_rate"] = outlier_rate();

    // Additional statistics
    if (!frame_stats_.empty()) {
        // Min/max covariance condition number
        double min_condition = std::numeric_limits<double>::max();
        double max_condition = 0.0;
        for (auto const& stats : frame_stats_) {
            min_condition = std::min(min_condition, stats.covariance_condition_number);
            max_condition = std::max(max_condition, stats.covariance_condition_number);
        }
        summary["covariance_condition_number_min"] = min_condition;
        summary["covariance_condition_number_max"] = max_condition;

        // Min/max NIS
        double min_nis = std::numeric_limits<double>::max();
        double max_nis = 0.0;
        double sum_nis = 0.0;
        for (auto const& stats : frame_stats_) {
            min_nis = std::min(min_nis, stats.nis_value);
            max_nis = std::max(max_nis, stats.nis_value);
            sum_nis += stats.nis_value;
        }
        summary["nis_min"] = min_nis;
        summary["nis_max"] = max_nis;
        summary["nis_mean"] = sum_nis / frame_stats_.size();

        // Time span
        summary["start_timestamp"] = frame_stats_.front().timestamp;
        summary["end_timestamp"] = frame_stats_.back().timestamp;
        summary["duration_seconds"] =
            frame_stats_.back().timestamp - frame_stats_.front().timestamp;
    }

    // Write to file
    std::ofstream file(output_path);
    if (!file) {
        throw std::runtime_error(fmt::format("Failed to open {}", output_path.string()));
    }
    file << summary.dump(2) << "\n";
}

double StatisticsTracker::mean_reprojection_error() const {
    if (frame_stats_.empty()) {
        return 0.0;
    }

    double sum = std::accumulate(
        frame_stats_.begin(), frame_stats_.end(), 0.0,
        [](double acc, FrameStatistics const& s) { return acc + s.mean_reprojection_error; });
    return sum / frame_stats_.size();
}

double StatisticsTracker::mean_num_inliers() const {
    if (frame_stats_.empty()) {
        return 0.0;
    }

    double sum =
        std::accumulate(frame_stats_.begin(), frame_stats_.end(), 0.0,
                        [](double acc, FrameStatistics const& s) { return acc + s.num_inliers; });
    return sum / frame_stats_.size();
}

double StatisticsTracker::outlier_rate() const {
    if (frame_stats_.empty()) {
        return 0.0;
    }

    int total_observations = 0;
    int total_outliers = 0;

    for (auto const& stats : frame_stats_) {
        total_observations += stats.num_observations;
        total_outliers += stats.num_outliers;
    }

    return total_observations > 0 ? static_cast<double>(total_outliers) / total_observations : 0.0;
}

size_t StatisticsTracker::num_lost_frames() const {
    return std::count_if(frame_stats_.begin(), frame_stats_.end(),
                         [](FrameStatistics const& s) { return s.tracking_lost; });
}

}  // namespace posetrak
