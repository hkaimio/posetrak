#include "posetrak/io/tracking_export.hpp"

#include <fmt/core.h>

#include "posetrak/kinematics/forward_kinematics.hpp"
#include <stdexcept>

namespace posetrak {

TrackingExporter::TrackingExporter(std::filesystem::path const& output_dir,
                                   Skeleton const& skeleton,
                                   std::unordered_map<int, Camera> const& cameras)
    : output_dir_(output_dir), skeleton_(skeleton), cameras_(cameras) {}

void TrackingExporter::open() {
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir_);

    // Open all CSV files
    tracking_results_.open(output_dir_ / "tracking_results.csv");
    if (!tracking_results_) {
        throw std::runtime_error(
            fmt::format("Failed to open {}", (output_dir_ / "tracking_results.csv").string()));
    }

    joint_angles_.open(output_dir_ / "joint_angles.csv");
    if (!joint_angles_) {
        throw std::runtime_error(
            fmt::format("Failed to open {}", (output_dir_ / "joint_angles.csv").string()));
    }

    root_pose_.open(output_dir_ / "root_pose.csv");
    if (!root_pose_) {
        throw std::runtime_error(
            fmt::format("Failed to open {}", (output_dir_ / "root_pose.csv").string()));
    }

    marker_projections_.open(output_dir_ / "marker_projections.csv");
    if (!marker_projections_) {
        throw std::runtime_error(
            fmt::format("Failed to open {}", (output_dir_ / "marker_projections.csv").string()));
    }

    observations_.open(output_dir_ / "observations.csv");
    if (!observations_) {
        throw std::runtime_error(
            fmt::format("Failed to open {}", (output_dir_ / "observations.csv").string()));
    }

    // Write headers
    write_tracking_results_header();
    write_joint_angles_header();
    write_root_pose_header();
    write_marker_projections_header();
    write_observations_header();
}

void TrackingExporter::close() {
    tracking_results_.close();
    joint_angles_.close();
    root_pose_.close();
    marker_projections_.close();
    observations_.close();
}

void TrackingExporter::write_tracking_results_header() {
    tracking_results_ << "frame,timestamp,marker_id,marker_name,x_3d,y_3d,z_3d,is_visible\n";
}

void TrackingExporter::write_joint_angles_header() {
    joint_angles_ << "frame,timestamp,joint_name,angle_x,angle_y,angle_z,velocity_x,velocity_y,"
                     "velocity_z\n";
}

void TrackingExporter::write_root_pose_header() {
    root_pose_ << "frame,timestamp,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z,vel_x,vel_y,"
                  "vel_z,omega_x,omega_y,omega_z\n";
}

void TrackingExporter::write_marker_projections_header() {
    marker_projections_ << "frame,timestamp,marker_id,marker_name,camera_id,proj_x,proj_y,obs_x,"
                           "obs_y,error_x,error_y,is_outlier\n";
}

void TrackingExporter::write_observations_header() {
    observations_ << "frame,timestamp,marker_id,marker_name,camera_id,pixel_x,pixel_y,"
                     "confidence,used_in_tracking\n";
}

void TrackingExporter::write_tracking_results_row(int frame, double timestamp,
                                                  std::string const& marker_name, int marker_id,
                                                  Eigen::Vector3d const& position_3d,
                                                  bool is_visible) {
    tracking_results_ << fmt::format("{},{},{},{},{},{},{},{}\n", frame, timestamp, marker_id,
                                     marker_name, position_3d.x(), position_3d.y(), position_3d.z(),
                                     is_visible ? "true" : "false");
}

void TrackingExporter::write_joint_angles_row(int frame, double timestamp,
                                              std::string const& joint_name,
                                              Eigen::Vector3d const& angles,
                                              Eigen::Vector3d const& velocities) {
    joint_angles_ << fmt::format("{},{},{},{},{},{},{},{},{}\n", frame, timestamp, joint_name,
                                 angles.x(), angles.y(), angles.z(), velocities.x(), velocities.y(),
                                 velocities.z());
}

void TrackingExporter::write_root_pose_row(int frame, double timestamp,
                                           Eigen::Vector3d const& position,
                                           Eigen::Quaterniond const& orientation,
                                           Eigen::Vector3d const& linear_velocity,
                                           Eigen::Vector3d const& angular_velocity) {
    root_pose_ << fmt::format("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n", frame, timestamp,
                              position.x(), position.y(), position.z(), orientation.w(),
                              orientation.x(), orientation.y(), orientation.z(),
                              linear_velocity.x(), linear_velocity.y(), linear_velocity.z(),
                              angular_velocity.x(), angular_velocity.y(), angular_velocity.z());
}

void TrackingExporter::write_marker_projection_row(int frame, double timestamp, int marker_id,
                                                   std::string const& marker_name, int camera_id,
                                                   Eigen::Vector2d const& projection,
                                                   Eigen::Vector2d const& observation,
                                                   bool is_outlier) {
    Eigen::Vector2d error = projection - observation;
    marker_projections_ << fmt::format("{},{},{},{},{},{},{},{},{},{},{},{}\n", frame, timestamp,
                                       marker_id, marker_name, camera_id, projection.x(),
                                       projection.y(), observation.x(), observation.y(), error.x(),
                                       error.y(), is_outlier ? "true" : "false");
}

void TrackingExporter::write_observation_row(int frame, double timestamp, int marker_id,
                                             std::string const& marker_name, int camera_id,
                                             Eigen::Vector2d const& pixel, double confidence,
                                             bool used_in_tracking) {
    observations_ << fmt::format("{},{},{},{},{},{},{},{},{}\n", frame, timestamp, marker_id,
                                 marker_name, camera_id, pixel.x(), pixel.y(), confidence,
                                 used_in_tracking ? "true" : "false");
}

void TrackingExporter::write_frame(
    int frame_number, double timestamp, State const& state,
    std::map<std::string, Eigen::Vector3d> const& marker_positions_3d,
    std::vector<Observation> const& observations, UpdateResult const& update_result) {
    // 1. Write tracking results (3D marker positions)
    auto const& markers = skeleton_.markers();
    for (size_t i = 0; i < markers.size(); ++i) {
        auto const& marker = markers[i];
        if (marker_positions_3d.count(marker.name)) {
            auto const& position_3d = marker_positions_3d.at(marker.name);
            write_tracking_results_row(frame_number, timestamp, marker.name, static_cast<int>(i),
                                       position_3d, true);
        }
    }

    // 2. Write joint angles
    auto const& joints = skeleton_.joints();
    auto const& joint_angles_vec = state.joint_angles();
    auto const& joint_velocities_vec = state.joint_velocities();

    int angle_idx = 0;
    for (size_t i = 0; i < joints.size(); ++i) {
        auto const& joint = joints[i];

        // Skip root joint (pelvis) - handled separately in root_pose
        if (joint.type == JointType::FIXED || !joint.parent_index.has_value()) {
            continue;
        }
        // Followers share the leader's state slot; don't write a separate row or advance angle_idx
        if (joint.is_scale_follower) {
            continue;
        }

        // Get joint angles (3 values for spherical joints)
        if (joint.type == JointType::SPHERICAL && angle_idx + 2 < joint_angles_vec.size()) {
            Eigen::Vector3d angles = joint_angles_vec.segment<3>(angle_idx);
            Eigen::Vector3d velocities = Eigen::Vector3d::Zero();
            if (angle_idx + 2 < joint_velocities_vec.size()) {
                velocities = joint_velocities_vec.segment<3>(angle_idx);
            }
            write_joint_angles_row(frame_number, timestamp, joint.name, angles, velocities);
            angle_idx += 3;
        } else if ((joint.type == JointType::REVOLUTE || joint.type == JointType::PRISMATIC) &&
                   angle_idx < joint_angles_vec.size()) {
            Eigen::Vector3d angles(joint_angles_vec(angle_idx), 0.0, 0.0);
            Eigen::Vector3d velocities(
                angle_idx < joint_velocities_vec.size() ? joint_velocities_vec(angle_idx) : 0.0,
                0.0, 0.0);
            write_joint_angles_row(frame_number, timestamp, joint.name, angles, velocities);
            angle_idx += 1;
        }
    }

    // 3. Write root pose
    Eigen::Vector3d const& root_position = state.root_position();
    Eigen::Quaterniond root_orientation = state.root_orientation();
    root_orientation.normalize();
    Eigen::Vector3d const& linear_velocity = state.root_velocity();
    Eigen::Vector3d const& angular_velocity = state.root_angular_velocity();

    write_root_pose_row(frame_number, timestamp, root_position, root_orientation, linear_velocity,
                        angular_velocity);

    // 4. Write marker projections and observations
    // Build a map of observations for quick lookup
    std::map<std::pair<int, int>, Observation> obs_map;  // (marker_id, camera_id) -> Observation
    std::set<std::pair<int, int>> used_observations;     // Track which were used (not outliers)

    for (auto const& obs : observations) {
        obs_map[{obs.marker_id, obs.camera_id}] = obs;
    }

    // Mark inliers as used (non-outliers from update_result)
    for (auto const& obs_result : update_result.observations) {
        if (!obs_result.is_outlier) {
            // Find the marker by name
            for (size_t i = 0; i < markers.size(); ++i) {
                if (markers[i].name == obs_result.marker_name) {
                    used_observations.insert({static_cast<int>(i), obs_result.camera_id});
                    break;
                }
            }
        }
    }

    // For each marker and camera, write projection if we have an observation
    for (size_t marker_idx = 0; marker_idx < markers.size(); ++marker_idx) {
        auto const& marker = markers[marker_idx];
        if (!marker_positions_3d.count(marker.name)) {
            continue;
        }

        auto const& position_3d = marker_positions_3d.at(marker.name);

        for (auto const& [camera_id, camera] : cameras_) {
            // Check if we have an observation for this marker/camera
            auto obs_key = std::make_pair(static_cast<int>(marker_idx), camera_id);
            if (obs_map.count(obs_key)) {
                auto const& obs = obs_map.at(obs_key);

                // Project marker to camera
                auto projection_opt = camera.project(position_3d);
                Eigen::Vector2d projection = projection_opt.value_or(Eigen::Vector2d(-1.0, -1.0));
                bool is_outlier = !used_observations.count(obs_key);

                write_marker_projection_row(frame_number, timestamp, static_cast<int>(marker_idx),
                                            marker.name, camera_id, projection, obs.position,
                                            is_outlier);
            }
        }
    }

    // 5. Write all observations (raw detections)
    for (auto const& obs : observations) {
        auto const& marker = markers[obs.marker_id];
        bool used = used_observations.count({obs.marker_id, obs.camera_id}) > 0;
        write_observation_row(frame_number, timestamp, obs.marker_id, marker.name, obs.camera_id,
                              obs.position, obs.confidence, used);
    }
}

}  // namespace posetrak
