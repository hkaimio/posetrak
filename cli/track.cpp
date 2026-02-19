#include <CLI/CLI.hpp>
#include <fmt/core.h>

#include "fmt/base.h"
#include "posetrak/core/config.hpp"
#include "posetrak/io/camera_loader.hpp"
#include "posetrak/io/observation_loader.hpp"
#include "posetrak/io/skeleton_loader.hpp"
#include "posetrak/io/statistics_tracker.hpp"
#include "posetrak/io/sync_loader.hpp"
#include "posetrak/io/tracking_export.hpp"
#include "posetrak/kinematics/forward_kinematics.hpp"
#include "posetrak/kinematics/pinocchio_model_builder.hpp"
#include "posetrak/kinematics/triangulation.hpp"
#include "posetrak/tracking/tracker.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace posetrak;

// Helper: Export predicted observations for comparison with Python
void export_predicted_observations(std::ofstream& file, int frame_idx, double timestamp,
                                   std::vector<Observation> const& observations, State const& state,
                                   ForwardKinematics& fk,
                                   std::unordered_map<int, Camera> const& cameras,
                                   Skeleton const& skeleton) {
    // Compute 3D marker positions from current state
    auto marker_positions_3d = fk.compute(state);

    // For each observation, project and compute residual
    for (auto const& obs : observations) {
        // Find marker name from marker_id (marker_id is the index in skeleton.markers())
        if (obs.marker_id < 0 || obs.marker_id >= static_cast<int>(skeleton.markers().size()))
            continue;

        std::string marker_name = skeleton.markers()[obs.marker_id].name;

        // Get 3D position
        auto it = marker_positions_3d.find(marker_name);
        if (it == marker_positions_3d.end())
            continue;

        Eigen::Vector3d pos_3d = it->second;

        // Find camera
        auto cam_it = cameras.find(obs.camera_id);
        if (cam_it == cameras.end())
            continue;

        auto const& camera = cam_it->second;

        // Project to camera
        auto predicted_opt = camera.project(pos_3d);
        if (!predicted_opt.has_value()) {
            // Skip failed projections
            continue;
        }
        Eigen::Vector2d predicted = *predicted_opt;

        // Compute residual
        double residual_u = obs.position.x() - predicted.x();
        double residual_v = obs.position.y() - predicted.y();
        double residual_norm = std::sqrt(residual_u * residual_u + residual_v * residual_v);

        // Write: frame,camera,marker,obs_u,obs_v,pred_u,pred_v,res_u,res_v,res_norm
        file << frame_idx << "," << camera.name() << "," << obs.position.x() << ","
             << obs.position.y() << "," << predicted.x() << "," << predicted.y() << ","
             << residual_u << "," << residual_v << "," << residual_norm << "\n";
    }
}

// Helper: Export complete state vector for comparison with Python
void export_state_vector(std::ofstream& file, int frame_idx, double timestamp, State const& state,
                         Skeleton const& skeleton) {
    // Format matches Python: tracker_frame_idx,timestamp,root_position_x/y/z,
    // root_quaternion_w/x/y/z, root_velocity_x/y/z, root_angular_velocity_x/y/z,
    // joint_<name>_angle_<n>, joint_<name>_velocity_<n>

    file << frame_idx << "," << timestamp << ",";

    // Root position
    file << state.root_position().x() << "," << state.root_position().y() << ","
         << state.root_position().z() << ",";

    // Root orientation (quaternion: w,x,y,z)
    file << state.root_orientation().w() << "," << state.root_orientation().x() << ","
         << state.root_orientation().y() << "," << state.root_orientation().z() << ",";

    // Root velocity
    file << state.root_velocity().x() << "," << state.root_velocity().y() << ","
         << state.root_velocity().z() << ",";

    // Root angular velocity
    file << state.root_angular_velocity().x() << "," << state.root_angular_velocity().y() << ","
         << state.root_angular_velocity().z();

    // Joint angles and velocities (in skeleton order)
    size_t state_idx = 0;
    for (auto const& joint : skeleton.joints()) {
        for (int i = 0; i < joint.dof; ++i) {
            file << "," << state.joint_angles()[state_idx + i];
        }
        for (int i = 0; i < joint.dof; ++i) {
            file << "," << state.joint_velocities()[state_idx + i];
        }
        state_idx += joint.dof;
    }

    file << "\n";
}

// Helper: Generate state vector CSV header
std::string generate_state_header(Skeleton const& skeleton) {
    std::string header = "tracker_frame_idx,timestamp,";
    header += "root_position_x,root_position_y,root_position_z,";
    header += "root_quaternion_w,root_quaternion_x,root_quaternion_y,root_quaternion_z,";
    header += "root_velocity_x,root_velocity_y,root_velocity_z,";
    header += "root_angular_velocity_x,root_angular_velocity_y,root_angular_velocity_z";

    // Joint angles and velocities
    for (auto const& joint : skeleton.joints()) {
        for (int i = 0; i < joint.dof; ++i) {
            header += ",joint_" + joint.name + "_angle_" + std::to_string(i);
        }
        for (int i = 0; i < joint.dof; ++i) {
            header += ",joint_" + joint.name + "_velocity_" + std::to_string(i);
        }
    }

    return header;
}

// Helper: Load Python tracker state from CSV (for validation/comparison)
std::optional<State> load_python_state(std::string const& csv_path, Skeleton const& skeleton,
                                       int frame = 0) {
    fmt::print("\n=== Loading Python Tracker State from CSV: {} ===\n", csv_path);
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        fmt::print("Warning: Could not open Python state CSV: {}\n", csv_path);
        return std::nullopt;
    }

    std::string line;
    // Read header
    if (!std::getline(file, line)) {
        return std::nullopt;
    }

    // Read frames until we find the target frame
    int current_frame = -1;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;

        // Parse frame number (tracker_frame_idx - first column)
        if (!std::getline(ss, token, ',')) {
            continue;
        }
        current_frame = std::stoi(token);

        if (current_frame != frame) {
            continue;
        }

        // Skip timestamp
        if (!std::getline(ss, token, ',')) {
            continue;
        }

        // Found target frame, parse state
        // Format: ,tracker_frame_idx,timestamp,root_position_x,root_position_y,root_position_z,
        //         root_quaternion_w,root_quaternion_x,root_quaternion_y,root_quaternion_z,
        //         ...velocities..., joint_angles...

        // Root position
        Eigen::Vector3d root_position;
        for (int i = 0; i < 3; ++i) {
            if (!std::getline(ss, token, ',')) {
                return std::nullopt;
            }
            root_position[i] = std::stod(token);
        }

        // Root orientation (quaternion: w,x,y,z)
        Eigen::Quaterniond root_orientation;
        if (!std::getline(ss, token, ','))
            return std::nullopt;
        root_orientation.w() = std::stod(token);
        if (!std::getline(ss, token, ','))
            return std::nullopt;
        root_orientation.x() = std::stod(token);
        if (!std::getline(ss, token, ','))
            return std::nullopt;
        root_orientation.y() = std::stod(token);
        if (!std::getline(ss, token, ','))
            return std::nullopt;
        root_orientation.z() = std::stod(token);

        // Root velocity (x,y,z)
        Eigen::Vector3d root_velocity;
        for (int i = 0; i < 3; ++i) {
            if (!std::getline(ss, token, ',')) {
                return std::nullopt;
            }
            root_velocity[i] = std::stod(token);
        }

        // Root angular velocity (x,y,z)
        Eigen::Vector3d root_angular_velocity;
        for (int i = 0; i < 3; ++i) {
            if (!std::getline(ss, token, ',')) {
                return std::nullopt;
            }
            root_angular_velocity[i] = std::stod(token);
        }

        // Joint angles and velocities (interleaved per joint)
        // Format: angle_0, angle_1, angle_2, velocity_0, velocity_1, velocity_2 for each joint
        // NOTE: CSV does NOT include the root/hips joint - it only has body joints
        int num_dof = skeleton.total_dof_count();
        Eigen::VectorXd joint_angles = Eigen::VectorXd::Zero(num_dof);
        Eigen::VectorXd joint_velocities = Eigen::VectorXd::Zero(num_dof);

        int dof_idx = 0;
        for (auto const& joint : skeleton.joints()) {
            // Skip root joint - CSV doesn't include it (root is handled separately above)
            // Note: total_dof_count() already excludes root, so don't increment dof_idx
            if (!joint.parent_index.has_value()) {
                continue;
            }

            int num_joint_dof = joint.dof;  // CSV has ALL DoFs, not just active ones

            if (!skeleton.is_joint_active(joint.name)) {
                dof_idx += num_joint_dof;  // Skip inactive joint DOFs
                continue;
            }

            // Read angles for this joint
            for (int i = 0; i < num_joint_dof; ++i) {
                if (!std::getline(ss, token, ',')) {
                    break;
                }
                joint_angles[dof_idx + i] = std::stod(token);
            }

            // Read velocities for this joint
            for (int i = 0; i < num_joint_dof; ++i) {
                if (!std::getline(ss, token, ',')) {
                    break;
                }
                joint_velocities[dof_idx + i] = std::stod(token);
            }

            dof_idx += num_joint_dof;
        }

        return State(root_position, root_orientation, joint_angles, root_velocity,
                     root_angular_velocity, joint_velocities);
    }

    return std::nullopt;
}

// Helper: Triangulate markers from first frame and compare with Python
void validate_camera_model(std::vector<Observation> const& observations,
                           std::unordered_map<int, Camera> const& cameras, Skeleton const& skeleton,
                           std::string const& python_csv_path) {
    fmt::print("\n=== Camera Model Validation ===\n");

    // Load Python marker positions for frame 0
    std::map<std::string, Eigen::Vector3d> python_positions;
    if (!python_csv_path.empty()) {
        std::ifstream file(python_csv_path);
        if (file.is_open()) {
            std::string line;
            // Read header
            std::getline(file, line);

            // Read frame 0 data
            while (std::getline(file, line)) {
                std::istringstream ss(line);
                std::string token;

                // Skip index column
                if (!std::getline(ss, token, ','))
                    continue;

                // Parse frame number
                if (!std::getline(ss, token, ','))
                    continue;
                int frame = std::stoi(token);
                if (frame != 0)
                    continue;

                // Skip timestamp
                if (!std::getline(ss, token, ','))
                    continue;

                // Skip person_id
                if (!std::getline(ss, token, ','))
                    continue;

                // Parse marker name
                std::string marker_name;
                if (!std::getline(ss, token, ','))
                    continue;
                marker_name = token;

                // Parse x, y, z
                Eigen::Vector3d pos;
                for (int i = 0; i < 3; ++i) {
                    if (!std::getline(ss, token, ','))
                        break;
                    pos[i] = std::stod(token);
                }

                python_positions[marker_name] = pos;
            }
        }
    }

    // Group observations by marker
    std::map<int, std::vector<Observation>> obs_by_marker;
    for (auto const& obs : observations) {
        obs_by_marker[obs.marker_id].push_back(obs);
    }

    // Triangulate each marker
    Triangulator triangulator(Triangulator::Method::DLT);
    int num_triangulated = 0;
    double total_error = 0.0;
    int num_compared = 0;

    for (auto const& [marker_id, marker_obs] : obs_by_marker) {
        if (marker_obs.size() < 2) {
            continue;  // Need at least 2 cameras
        }

        // Get marker name
        if (marker_id >= static_cast<int>(skeleton.markers().size())) {
            continue;
        }
        std::string marker_name = skeleton.markers()[marker_id].name;

        // Prepare for triangulation
        std::vector<Eigen::Vector2d> pixel_coords;
        std::vector<Camera const*> marker_cameras;
        std::vector<double> confidences;

        for (auto const& obs : marker_obs) {
            auto it = cameras.find(obs.camera_id);
            if (it == cameras.end()) {
                continue;
            }
            pixel_coords.push_back(obs.position);
            marker_cameras.push_back(&it->second);
            confidences.push_back(obs.confidence);
        }

        // Triangulate
        auto result = triangulator.triangulate(pixel_coords, marker_cameras, confidences);
        if (result.success) {
            num_triangulated++;
            Eigen::Vector3d cpp_pos = result.position;

            // Compare with Python if available
            auto it = python_positions.find(marker_name);
            if (it != python_positions.end()) {
                Eigen::Vector3d python_pos = it->second;
                double error = (cpp_pos - python_pos).norm();
                total_error += error;
                num_compared++;

                fmt::print(
                    "  {}: C++ ({:.3f}, {:.3f}, {:.3f}) vs Python ({:.3f}, {:.3f}, "
                    "{:.3f}) - error: {:.4f}m\n",
                    marker_name, cpp_pos.x(), cpp_pos.y(), cpp_pos.z(), python_pos.x(),
                    python_pos.y(), python_pos.z(), error);
            } else {
                fmt::print("  {}: C++ ({:.3f}, {:.3f}, {:.3f}) - no Python comparison\n",
                           marker_name, cpp_pos.x(), cpp_pos.y(), cpp_pos.z());
            }
        }
    }

    fmt::print("\nTriangulation summary:\n");
    fmt::print("  Triangulated markers: {}\n", num_triangulated);
    if (num_compared > 0) {
        fmt::print("  Compared with Python: {}\n", num_compared);
        fmt::print("  Mean position error: {:.4f}m\n", total_error / num_compared);
    }
    fmt::print("==============================\n\n");
}

int main(int argc, char* argv[]) {
    CLI::App app{"Posetrak - Motion Capture Tracker"};

    std::string config_path;
    app.add_option("config", config_path, "Configuration file (TOML)")
        ->required()
        ->check(CLI::ExistingFile);

    bool verbose = false;
    app.add_flag("-v,--verbose", verbose, "Verbose output (show per-frame statistics)");

    bool quiet = false;
    app.add_flag("-q,--quiet", quiet, "Quiet mode (only show errors)");

    CLI11_PARSE(app, argc, argv);

    try {
        // Load configuration
        if (!quiet) {
            fmt::print("Loading configuration: {}\n", config_path);
        }
        auto config = TrackerAppConfig::load(config_path);
        config.validate();

        // Load skeleton
        if (!quiet) {
            fmt::print("Loading skeleton: {}\n", config.skeleton_path.string());
        }
        auto skeleton = load_skeleton_from_yaml(config.skeleton_path.string());
        if (!quiet) {
            fmt::print("  Loaded {} joints\n", skeleton.joints().size());
        }

        // Apply active joint groups filter (if specified)
        if (!config.active_joint_groups.empty()) {
            skeleton.set_active_groups(config.active_joint_groups);
            if (!quiet) {
                std::string groups_str;
                for (size_t i = 0; i < config.active_joint_groups.size(); ++i) {
                    if (i > 0)
                        groups_str += ", ";
                    groups_str += config.active_joint_groups[i];
                }
                fmt::print("  Active joint groups: {}\n", groups_str);
                fmt::print("  Active DOFs: {}\n", skeleton.active_dof());
            }
        }

        // Load cameras
        if (!quiet) {
            fmt::print("Loading cameras: {}\n", config.cameras_path.string());
        }
        auto cameras_by_name = load_cameras_from_toml(config.cameras_path.string());
        if (!quiet) {
            fmt::print("  Loaded {} cameras\n", cameras_by_name.size());
        }

        // Load sync (optional)
        if (config.sync_path) {
            if (!quiet) {
                fmt::print("Loading sync: {}\n", config.sync_path->string());
            }
            auto sync_data = load_sync_metadata(config.sync_path->string());
            // Apply sync to cameras (modifies cameras_by_name in-place)
            apply_sync_metadata(cameras_by_name, sync_data, false);
        }

        // Load observations (camera IDs will be correct from the start)
        if (!quiet) {
            fmt::print("Loading observations: {}\n", config.observations_dir.string());
        }

        // Determine end time
        double end_time = config.end_time;
        if (end_time < 0.0) {
            // Load all available data - we'll determine end_time after loading
            end_time = std::numeric_limits<double>::max();
        }

        auto observations_set =
            load_openpose_sequence(config.observations_dir.string(), cameras_by_name, skeleton,
                                   config.start_time, end_time, 0.1, config.person_id);

        if (observations_set.empty()) {
            throw std::runtime_error("No observations found in time range");
        }

        // Auto-detect end time if not specified
        if (config.end_time < 0.0) {
            end_time = observations_set.max_time();
        }

        // Calculate number of tracker steps
        double dt = 1.0 / config.tracker_fps;
        int num_steps = static_cast<int>((end_time - config.start_time) / dt);

        if (num_steps <= 0) {
            throw std::runtime_error(
                "No time steps to process - check start_time, end_time, and tracker_fps");
        }

        if (!quiet) {
            fmt::print("  Time range: [{:.3f}, {:.3f}) seconds\n", config.start_time, end_time);
            fmt::print("  Tracker sample rate: {:.1f} Hz (dt = {:.6f} s)\n", config.tracker_fps,
                       dt);
            fmt::print("  Will process {} time steps\n", num_steps);
        }

        // Create tracker
        if (!quiet) {
            fmt::print("\nInitializing tracker...\n");
        }

        // Convert cameras to ID-keyed map for Tracker
        std::unordered_map<int, Camera> cameras;
        for (auto const& [name, cam] : cameras_by_name) {
            cameras.emplace(cam.id(), cam);
        }

        auto tracker_config = config.to_tracker_config();
        Tracker tracker(skeleton, cameras, tracker_config);

        // Validate camera model by triangulating first frame
        double t_first_window = config.start_time + dt;
        auto first_frame_obs = observations_set.get_all_in_range(config.start_time, t_first_window);
        std::string python_markers_csv =
            "tracking_tests/kotegaeshi/makers_person0_python_tracker.csv";
        validate_camera_model(first_frame_obs, cameras, skeleton, python_markers_csv);

        // Try to load Python state for initialization (if available)
        std::string python_state_csv =
            config.python_state_path.value_or("tracking_tests/kotegaeshi/python_tracker_state.csv");
        auto python_state = load_python_state(python_state_csv, skeleton, 0);

        if (python_state.has_value()) {
            if (!quiet) {
                fmt::print("  Initializing from Python tracker state (frame 0)\n");
                auto const& s = python_state.value();
                fmt::print("    Root position: ({:.3f}, {:.3f}, {:.3f})\n", s.root_position().x(),
                           s.root_position().y(), s.root_position().z());
                fmt::print("    Root orientation: w={:.3f}, x={:.3f}, y={:.3f}, z={:.3f}\n",
                           s.root_orientation().w(), s.root_orientation().x(),
                           s.root_orientation().y(), s.root_orientation().z());
                fmt::print("    Joint angles (first 5):");
                for (int i = 0; i < 5 && i < s.joint_angles().size(); ++i) {
                    fmt::print(" {:.4f}", s.joint_angles()[i]);
                }
                fmt::print("\n");
            }
            tracker.initialize_from_state(python_state.value(), config.start_time);
        } else {
            // Fall back to rest pose if Python state not available
            if (!quiet) {
                fmt::print("  Python state not available, initializing from rest pose\n");
            }
            tracker.initialize_from_rest_pose(config.start_time);
        }

        if (!quiet) {
            fmt::print("  Initialization successful\n\n");
        }

        // Create FK for computing marker positions
        pinocchio::Model model;
        pinocchio::Data data;
        PinocchioModelBuilder::build_model_and_data(skeleton, model, data);
        auto marker_frame_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
        ForwardKinematics fk(model, data, marker_frame_map, skeleton);

        // Create exporters
        std::unique_ptr<TrackingExporter> exporter;
        std::unique_ptr<StatisticsTracker> stats_tracker;

        if (config.export_tracking_results) {
            exporter = std::make_unique<TrackingExporter>(config.output_dir, skeleton, cameras);
            exporter->open();
        }

        if (config.export_statistics) {
            stats_tracker = std::make_unique<StatisticsTracker>();
        }

        // Create diagnostic export files
        std::ofstream pred_obs_file;
        std::ofstream state_vec_file;
        if (config.export_tracking_results) {
            pred_obs_file.open(config.output_dir / "predicted_observations.csv");
            pred_obs_file << "frame,camera,marker,obs_u,obs_v,pred_u,pred_v,res_u,res_v,res_norm\n";

            state_vec_file.open(config.output_dir / "state_vectors.csv");
            state_vec_file << generate_state_header(skeleton) << "\n";
        }

        // Enable UKF debug mode
        if (auto* ukf = tracker.get_ukf()) {
            ukf->enable_debug(true, (config.output_dir / "debug").string());
        }

        // Track sequence
        if (!quiet) {
            fmt::print("Tracking:\n");
        }

        auto start_time = std::chrono::steady_clock::now();
        int frames_tracked = 0;
        int frames_lost = 0;

        // Note: State vectors will start at frame 1 (step 1 posterior) to align with
        // tracking_results.csv Initialization state (frame 0) is not exported to maintain alignment

        // Process first update (correct the initialization with first observations)
        {
            auto frame_0_obs = observations_set.get_all_in_range(config.start_time, t_first_window);
            if (!frame_0_obs.empty()) {
                if (auto* ukf = tracker.get_ukf()) {
                    ukf->set_frame_number(0);
                }

                // Debug: count unique marker-camera pairs
                std::set<std::pair<int, int>> unique_pairs;
                std::map<int, int> camera_counts;
                for (const auto& obs : frame_0_obs) {
                    unique_pairs.insert({obs.marker_id, obs.camera_id});
                    camera_counts[obs.camera_id]++;
                }

                if (!quiet) {
                    fmt::print(
                        "  Step 0: Updating initialization with {} observations ({} unique "
                        "marker-camera pairs) in time [{:.6f}, {:.6f})\n",
                        frame_0_obs.size(), unique_pairs.size(), config.start_time, t_first_window);
                    fmt::print("    Camera distribution:");
                    for (const auto& [cam_id, count] : camera_counts) {
                        fmt::print(" cam{}={}", cam_id, count);
                    }
                    fmt::print("\n");
                }

                // Use window midpoint as effective timestamp
                // double t_effective = config.start_time + dt / 2.0;
                double t_effective = config.start_time - dt * 0.5;
                auto result = tracker.track_frame(frame_0_obs, t_effective);

                // Note: Step 0 posterior not exported - tracking_results starts at frame 1 (step 1)

                if (result.tracking_lost) {
                    fmt::print(stderr, "Warning: Tracking lost on first update\n");
                } else {
                    frames_tracked++;
                }
            } else {
                if (!quiet) {
                    fmt::print(
                        "  No observations in time [{:.6f}, {:.6f}), skipping first update\n",
                        config.start_time, t_first_window);
                }
            }
        }

        // Main tracking loop - process remaining time steps
        for (int step = 1; step < num_steps; ++step) {
            double t_start =
                config.start_time + step * dt - dt / 2.0;  // Center the window around the step time
            double t_end = t_start + dt;

            // Set frame number for UKF debug
            if (auto* ukf = tracker.get_ukf()) {
                ukf->set_frame_number(step);
            }

            // Get all observations in this time window
            auto frame_obs = observations_set.get_all_in_range(t_start, t_end);

            if (frame_obs.empty()) {
                if (verbose) {
                    fmt::print("  Step {}: t=[{:.3f}, {:.3f}): No observations, skipping\n", step,
                               t_start, t_end);
                }
                continue;
            }

            // Use window midpoint as effective timestamp
            double t_effective = t_start + dt / 2.0;

            // Export predicted observations BEFORE tracking update (uses prior state)
            if (pred_obs_file.is_open()) {
                export_predicted_observations(pred_obs_file, step + 1, t_effective, frame_obs,
                                              tracker.state(), fk, cameras, skeleton);
            }

            // DEBUG: Export step 1 prior state for comparison with Python frame 1
            if (step == 1) {
                auto debug_dir = config.output_dir / "debug";
                std::filesystem::create_directories(debug_dir);
                auto prior_state_path = debug_dir / "step_0001_prior_state.csv";
                std::ofstream prior_file(prior_state_path);
                if (prior_file.is_open()) {
                    // Write header (same as state_vectors.csv)
                    prior_file << generate_state_header(skeleton) << "\n";
                    // Write prior state (before update)
                    export_state_vector(prior_file, 2, t_effective, tracker.state(), skeleton);
                    prior_file.close();
                    if (!quiet) {
                        fmt::print("  [DEBUG] Exported step 1 prior state to {}\n",
                                   prior_state_path.string());
                    }
                }
            }

            // Track using observations in window
            auto result = tracker.track_frame(frame_obs, t_effective);

            if (result.tracking_lost) {
                frames_lost++;
                if (verbose) {
                    fmt::print("  Step {}: t=[{:.3f}, {:.3f}): Tracking LOST ({} obs)\n", step,
                               t_start, t_end, result.update_info.num_observations);
                }
            } else {
                frames_tracked++;
                if (true) {
                    double sum_error = 0.0;
                    double max_error = 0.0;
                    uint32_t count = 0;
                    for (auto const& obs_result : result.update_info.observations) {
                        if (!obs_result.is_outlier) {
                            double error = obs_result.innovation.norm();
                            sum_error += error;
                            max_error = std::max(max_error, error);
                            count++;
                        }
                    }

                    fmt::print(
                        "  Step {}: t=[{:.3f}, {:.3f}): {} inliers, {} outliers mean reproj error "
                        "{:.4f} max {:.4f}\n",
                        step, t_start, t_end, result.update_info.num_inliers,
                        result.update_info.num_outliers, count > 0 ? sum_error / count : 0.0,
                        max_error);
                }
            }

            // Export state vector AFTER tracking update (posterior state)
            // CRITICAL: Use same frame index as tracking_results to keep them synchronized
            if (state_vec_file.is_open()) {
                export_state_vector(state_vec_file, step, t_effective, result.state, skeleton);
            }

            // Export
            if (exporter) {
                // Compute marker positions using FK
                auto marker_positions_3d_map = fk.compute(result.state);

                // Convert to std::map (exporter expects std::map, not unordered_map)
                std::map<std::string, Eigen::Vector3d> marker_positions_3d(
                    marker_positions_3d_map.begin(), marker_positions_3d_map.end());

                exporter->write_frame(step, t_effective, result.state, marker_positions_3d,
                                      frame_obs, result.update_info);
            }

            if (stats_tracker) {
                stats_tracker->add_frame_stats(step, t_effective, result.update_info,
                                               result.covariance, result.tracking_lost);
            }

            // Progress indicator
            if (!quiet && !verbose && step % 10 == 0) {
                double percent = 100.0 * step / num_steps;
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                double elapsed_sec =
                    std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() / 1000.0;
                double steps_per_sec = step / elapsed_sec;
                int eta_sec = static_cast<int>((num_steps - step) / steps_per_sec);

                fmt::print("  Progress: {}/{} ({:.1f}%) | {:.1f} steps/s | ETA: {}s\r", step,
                           num_steps, percent, steps_per_sec, eta_sec);
                std::cout.flush();
            }
        }

        if (!quiet && !verbose) {
            fmt::print("\n");
        }

        // Close exporters
        if (exporter) {
            exporter->close();
        }

        // Close diagnostic files
        if (pred_obs_file.is_open()) {
            pred_obs_file.close();
        }
        if (state_vec_file.is_open()) {
            state_vec_file.close();
        }

        // Write statistics
        if (stats_tracker) {
            stats_tracker->write_frame_stats(config.output_dir / "tracking_stats.csv");

            nlohmann::json metadata;
            metadata["sequence_name"] = config.observations_dir.filename().string();
            metadata["skeleton_file"] = config.skeleton_path.filename().string();
            metadata["num_cameras"] = cameras.size();
            metadata["num_markers"] = skeleton.markers().size();
            metadata["start_time"] = config.start_time;
            metadata["end_time"] = end_time;
            metadata["num_steps"] = num_steps;
            metadata["tracker_fps"] = config.tracker_fps;
            metadata["config"] = {
                {"process_noise_std", config.process_noise_std},
                {"measurement_noise_std", config.measurement_noise_std},
                {"outlier_threshold", config.outlier_threshold},
                {"ukf_alpha", config.ukf_alpha},
            };

            stats_tracker->write_summary_stats(config.output_dir / "overall_stats.json", metadata);
        }

        // Final summary
        auto tracking_end_time = std::chrono::steady_clock::now();
        auto total_elapsed = tracking_end_time - start_time;
        double total_sec =
            std::chrono::duration_cast<std::chrono::milliseconds>(total_elapsed).count() / 1000.0;
        double avg_steps_per_sec = frames_tracked / total_sec;

        if (!quiet) {
            fmt::print("\nTracking complete!\n");
            fmt::print("  Tracked: {}/{} steps ({:.1f}%)\n", frames_tracked, num_steps,
                       100.0 * frames_tracked / num_steps);
            fmt::print("  Lost: {} steps\n", frames_lost);
            fmt::print("  Average rate: {:.1f} steps/s\n", avg_steps_per_sec);
            fmt::print("  Total time: {:.1f}s\n", total_sec);

            if (stats_tracker) {
                fmt::print("  Mean reprojection error: {:.2f}px\n",
                           stats_tracker->mean_reprojection_error());
                fmt::print("  Outlier rate: {:.1f}%\n", stats_tracker->outlier_rate() * 100.0);
            }

            fmt::print("\nResults exported to: {}\n", config.output_dir.string());
            if (config.export_tracking_results) {
                fmt::print("  - tracking_results.csv\n");
                fmt::print("  - joint_angles.csv\n");
                fmt::print("  - root_pose.csv\n");
                fmt::print("  - marker_projections.csv\n");
                fmt::print("  - observations.csv\n");
                fmt::print("  - predicted_observations.csv (diagnostic)\n");
                fmt::print("  - state_vectors.csv (diagnostic)\n");
            }
            if (config.export_statistics) {
                fmt::print("  - tracking_stats.csv\n");
                fmt::print("  - overall_stats.json\n");
            }
        }

        return 0;

    } catch (std::exception const& e) {
        fmt::print(stderr, "Error: {}\n", e.what());
        return 1;
    }
}
