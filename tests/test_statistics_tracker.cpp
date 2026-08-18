// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/filters/update_result.hpp"
#include "posetrak/io/statistics_tracker.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace posetrak;
using Catch::Matchers::WithinAbs;

TEST_CASE("StatisticsTracker basic functionality", "[io][statistics]") {
    StatisticsTracker tracker;

    SECTION("Empty tracker") {
        REQUIRE(tracker.num_frames() == 0);
        REQUIRE(tracker.mean_reprojection_error() == 0.0);
        REQUIRE(tracker.mean_num_inliers() == 0.0);
        REQUIRE(tracker.outlier_rate() == 0.0);
    }

    SECTION("Add frame statistics") {
        // Create mock update result
        UpdateResult result;
        result.num_observations = 10;
        result.num_inliers = 8;
        result.num_outliers = 2;
        result.nis = 15.5;

        // Add some observation results
        for (int i = 0; i < 8; ++i) {
            ObservationResult obs;
            obs.marker_name = "marker_" + std::to_string(i);
            obs.camera_id = 0;
            obs.is_outlier = false;
            obs.mahalanobis_distance = 2.0;
            obs.innovation = Eigen::Vector2d(1.0, 1.0);  // Error = sqrt(2) pixels
            obs.predicted = Eigen::Vector2d(100, 100);
            obs.actual = Eigen::Vector2d(101, 101);
            result.observations.push_back(obs);
        }

        // Add outliers
        for (int i = 0; i < 2; ++i) {
            ObservationResult obs;
            obs.marker_name = "marker_outlier_" + std::to_string(i);
            obs.camera_id = 0;
            obs.is_outlier = true;
            obs.mahalanobis_distance = 5.0;
            obs.innovation = Eigen::Vector2d(10.0, 10.0);
            obs.predicted = Eigen::Vector2d(100, 100);
            obs.actual = Eigen::Vector2d(110, 110);
            result.observations.push_back(obs);
        }

        // Create covariance matrix
        Eigen::MatrixXd covariance = Eigen::MatrixXd::Identity(10, 10) * 0.1;

        // Add frame
        tracker.add_frame_stats(0, 0.0, result, covariance, false);

        REQUIRE(tracker.num_frames() == 1);
        REQUIRE(tracker.num_lost_frames() == 0);
        REQUIRE_THAT(tracker.mean_reprojection_error(), WithinAbs(std::sqrt(2.0), 1e-6));
        REQUIRE_THAT(tracker.mean_num_inliers(), WithinAbs(8.0, 1e-6));
        REQUIRE_THAT(tracker.outlier_rate(), WithinAbs(0.2, 1e-6));  // 2/10
    }

    SECTION("Multiple frames with tracking lost") {
        Eigen::MatrixXd covariance = Eigen::MatrixXd::Identity(5, 5) * 0.1;

        // Frame 0: good tracking
        UpdateResult result1;
        result1.num_observations = 10;
        result1.num_inliers = 9;
        result1.num_outliers = 1;
        result1.nis = 12.0;

        ObservationResult obs1;
        obs1.is_outlier = false;
        obs1.innovation = Eigen::Vector2d(0.5, 0.5);
        for (int i = 0; i < 9; ++i) {
            result1.observations.push_back(obs1);
        }

        tracker.add_frame_stats(0, 0.0, result1, covariance, false);

        // Frame 1: tracking lost
        UpdateResult result2;
        result2.num_observations = 5;
        result2.num_inliers = 2;
        result2.num_outliers = 3;
        result2.nis = 50.0;

        tracker.add_frame_stats(1, 0.033, result2, covariance, true);

        REQUIRE(tracker.num_frames() == 2);
        REQUIRE(tracker.num_lost_frames() == 1);
    }

    SECTION("Export frame statistics CSV") {
        // Add frame
        UpdateResult result;
        result.num_observations = 5;
        result.num_inliers = 4;
        result.num_outliers = 1;
        result.nis = 10.5;

        ObservationResult obs;
        obs.is_outlier = false;
        obs.innovation = Eigen::Vector2d(1.0, 0.0);
        for (int i = 0; i < 4; ++i) {
            result.observations.push_back(obs);
        }

        Eigen::MatrixXd covariance = Eigen::MatrixXd::Identity(3, 3);
        covariance(0, 0) = 0.01;
        covariance(1, 1) = 0.1;
        covariance(2, 2) = 1.0;

        tracker.add_frame_stats(0, 0.0, result, covariance, false);

        // Write to temp file
        std::filesystem::path temp_path = std::filesystem::temp_directory_path() / "test_stats.csv";
        tracker.write_frame_stats(temp_path);

        // Verify file exists and has correct header
        REQUIRE(std::filesystem::exists(temp_path));

        std::ifstream file(temp_path);
        std::string header;
        std::getline(file, header);
        REQUIRE(header.find("frame") != std::string::npos);
        REQUIRE(header.find("timestamp") != std::string::npos);
        REQUIRE(header.find("num_observations") != std::string::npos);
        REQUIRE(header.find("mean_reprojection_error") != std::string::npos);
        REQUIRE(header.find("covariance_min_eigenvalue") != std::string::npos);

        // Read data line
        std::string data_line;
        std::getline(file, data_line);
        REQUIRE(data_line.find("0,0") != std::string::npos);    // frame, timestamp
        REQUIRE(data_line.find("5,4,1") != std::string::npos);  // obs, inliers, outliers

        // Cleanup -- close the handle first: Windows (unlike POSIX) refuses to
        // remove a file while it's still open.
        file.close();
        std::filesystem::remove(temp_path);
    }

    SECTION("Export summary statistics JSON") {
        // Add multiple frames
        Eigen::MatrixXd covariance = Eigen::MatrixXd::Identity(5, 5) * 0.1;

        for (int i = 0; i < 10; ++i) {
            UpdateResult result;
            result.num_observations = 8;
            result.num_inliers = 7;
            result.num_outliers = 1;
            result.nis = 12.0 + i;

            ObservationResult obs;
            obs.is_outlier = false;
            obs.innovation = Eigen::Vector2d(0.5, 0.5);
            for (int j = 0; j < 7; ++j) {
                result.observations.push_back(obs);
            }

            tracker.add_frame_stats(i, i * 0.033, result, covariance, false);
        }

        // Write summary
        std::filesystem::path temp_path =
            std::filesystem::temp_directory_path() / "test_summary.json";

        nlohmann::json metadata;
        metadata["sequence_name"] = "test_sequence";
        metadata["num_cameras"] = 3;

        tracker.write_summary_stats(temp_path, metadata);

        // Verify file exists and parse JSON
        REQUIRE(std::filesystem::exists(temp_path));

        std::ifstream file(temp_path);
        nlohmann::json summary = nlohmann::json::parse(file);

        REQUIRE(summary["sequence_name"] == "test_sequence");
        REQUIRE(summary["num_cameras"] == 3);
        REQUIRE(summary["total_frames"] == 10);
        REQUIRE(summary["frames_tracked"] == 10);
        REQUIRE(summary["frames_lost"] == 0);
        REQUIRE(summary.contains("mean_reprojection_error"));
        REQUIRE(summary.contains("mean_num_inliers"));
        REQUIRE(summary.contains("outlier_rate"));
        REQUIRE(summary.contains("duration_seconds"));

        // Cleanup -- close the handle first: Windows (unlike POSIX) refuses to
        // remove a file while it's still open.
        file.close();
        std::filesystem::remove(temp_path);
    }
}
