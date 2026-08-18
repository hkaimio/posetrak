// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Eigen/Dense>

#include "posetrak/core/camera.hpp"
#include "posetrak/core/observation.hpp"
#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/core/state.hpp"
#include "posetrak/filters/ukf.hpp"
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>

namespace posetrak {

/**
 * @brief Exports tracking results to CSV files for analysis and visualization
 *
 * Creates 5 CSV files:
 * - tracking_results.csv: 3D marker positions per frame
 * - joint_angles.csv: Joint angles and velocities per frame
 * - root_pose.csv: Root position, orientation, and velocities per frame
 * - marker_projections.csv: 2D projections vs observations with errors
 * - observations.csv: Raw OpenPose detections
 */
class TrackingExporter {
   public:
    /**
     * @brief Construct exporter with output directory and metadata
     *
     * @param output_dir Directory to write CSV files
     * @param skeleton Skeleton model (for marker names and joint names)
     * @param cameras Map of camera ID to Camera (for projections)
     */
    TrackingExporter(std::filesystem::path const& output_dir, Skeleton const& skeleton,
                     SkeletonLayout const& layout, std::unordered_map<int, Camera> const& cameras);

    /**
     * @brief Open all CSV files for writing
     *
     * Creates the output directory if needed and writes CSV headers
     */
    void open();

    /**
     * @brief Write a single frame's tracking results
     *
     * @param frame_number Frame number (0-indexed)
     * @param timestamp Timestamp in seconds
     * @param state Estimated state from tracker
     * @param marker_positions_3d Map of marker name to 3D position
     * @param observations Raw observations from all cameras
     * @param update_result UKF update result (for outlier info)
     */
    void write_frame(int frame_number, double timestamp, State const& state,
                     std::map<std::string, Eigen::Vector3d> const& marker_positions_3d,
                     std::vector<Observation> const& observations,
                     UpdateResult const& update_result);

    /**
     * @brief Close all CSV files
     */
    void close();

   private:
    std::filesystem::path output_dir_;
    Skeleton const& skeleton_;
    SkeletonLayout const& layout_;
    std::unordered_map<int, Camera> const& cameras_;

    // Output file streams
    std::ofstream tracking_results_;
    std::ofstream joint_angles_;
    std::ofstream root_pose_;
    std::ofstream marker_projections_;
    std::ofstream observations_;

    // Helper methods
    void write_tracking_results_header();
    void write_joint_angles_header();
    void write_root_pose_header();
    void write_marker_projections_header();
    void write_observations_header();

    void write_tracking_results_row(int frame, double timestamp, std::string const& marker_name,
                                    int marker_id, Eigen::Vector3d const& position_3d,
                                    bool is_visible);

    void write_joint_angles_row(int frame, double timestamp, std::string const& joint_name,
                                Eigen::Vector3d const& angles, Eigen::Vector3d const& velocities);

    void write_root_pose_row(int frame, double timestamp, Eigen::Vector3d const& position,
                             Eigen::Quaterniond const& orientation,
                             Eigen::Vector3d const& linear_velocity,
                             Eigen::Vector3d const& angular_velocity);

    void write_marker_projection_row(int frame, double timestamp, int marker_id,
                                     std::string const& marker_name, int camera_id,
                                     Eigen::Vector2d const& projection,
                                     Eigen::Vector2d const& observation, bool is_outlier);

    void write_observation_row(int frame, double timestamp, int marker_id,
                               std::string const& marker_name, int camera_id,
                               Eigen::Vector2d const& pixel, double confidence,
                               bool used_in_tracking);
};

}  // namespace posetrak
