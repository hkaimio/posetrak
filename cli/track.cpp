#include <CLI/CLI.hpp>
#include <fmt/core.h>

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
#include "posetrak/tracking/tracker.hpp"  // Must be before config.hpp for inline function
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

        // Joint angles
        int num_dof = skeleton.total_dof_count();
        Eigen::VectorXd joint_angles = Eigen::VectorXd::Zero(num_dof);
        for (int i = 0; i < num_dof; ++i) {
            if (!std::getline(ss, token, ',')) {
                // If we run out of columns, leave remaining angles at zero
                break;
            }
            joint_angles[i] = std::stod(token);
        }

        // Initialize velocities to zero (Python CSV doesn't include velocities)
        Eigen::Vector3d root_velocity = Eigen::Vector3d::Zero();
        Eigen::Vector3d root_angular_velocity = Eigen::Vector3d::Zero();
        Eigen::VectorXd joint_velocities = Eigen::VectorXd::Zero(num_dof);

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

// Helper: Convert camera map with string keys to int keys
std::unordered_map<int, Camera>
convert_camera_map(std::unordered_map<std::string, Camera> const& cameras_by_name,
                   std::unordered_map<std::string, int>& name_to_id) {
    std::unordered_map<int, Camera> cameras_by_id;
    int next_id = 0;
    for (auto const& [name, cam] : cameras_by_name) {
        name_to_id[name] = next_id;
        cameras_by_id.emplace(next_id, cam);
        next_id++;
    }
    return cameras_by_id;
}

// Helper: Update observation camera IDs from names
void update_observation_camera_ids(ObservationSet& obs_set,
                                   std::unordered_map<std::string, int> const& name_to_id) {
    for (auto& [cam_name, sequence] :
         const_cast<std::map<std::string, ObservationSequence>&>(obs_set.sequences())) {
        auto it = name_to_id.find(cam_name);
        if (it != name_to_id.end()) {
            sequence.camera_id = it->second;
            for (auto& obs : sequence.observations) {
                obs.camera_id = it->second;
            }
        }
    }
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

        // Load cameras
        if (!quiet) {
            fmt::print("Loading cameras: {}\n", config.cameras_path.string());
        }
        auto cameras_by_name = load_cameras_from_toml(config.cameras_path.string());
        if (!quiet) {
            fmt::print("  Loaded {} cameras\n", cameras_by_name.size());
        }

        // Convert camera map to use integer IDs
        std::unordered_map<std::string, int> camera_name_to_id;
        auto cameras = convert_camera_map(cameras_by_name, camera_name_to_id);

        // Load sync (optional)
        if (config.sync_path) {
            if (!quiet) {
                fmt::print("Loading sync: {}\n", config.sync_path->string());
            }
            auto sync_data = load_sync_metadata(config.sync_path->string());
            // Apply sync to cameras (modifies cameras_by_name in-place)
            apply_sync_metadata(cameras_by_name, sync_data, false);
            // Re-convert to int keys
            cameras = convert_camera_map(cameras_by_name, camera_name_to_id);
        }

        // Load observations
        if (!quiet) {
            fmt::print("Loading observations: {}\n", config.observations_dir.string());
        }

        uint32_t end_frame =
            config.max_frames < 0 ? UINT32_MAX : config.start_frame + config.max_frames;
        auto observations_set =
            load_openpose_sequence(config.observations_dir.string(), cameras_by_name, skeleton,
                                   {config.start_frame, end_frame}, 0.1, config.person_id);

        // Update observation camera IDs to match the integer IDs
        update_observation_camera_ids(observations_set, camera_name_to_id);

        // Determine frame range from observations
        int start_frame = config.start_frame;
        int num_frames_to_process = 0;

        // Get unique timestamps
        auto timestamps = observations_set.get_unique_timestamps();
        if (timestamps.empty()) {
            throw std::runtime_error("No observations found");
        }

        num_frames_to_process = static_cast<int>(timestamps.size());
        if (config.max_frames > 0) {
            num_frames_to_process = std::min(num_frames_to_process, config.max_frames);
        }

        if (!quiet) {
            fmt::print("  Found {} timesteps\n", timestamps.size());
            fmt::print("  Will process {} frames starting at frame {}\n", num_frames_to_process,
                       start_frame);
        }

        // Create tracker
        if (!quiet) {
            fmt::print("\nInitializing tracker...\n");
        }

        auto tracker_config = config.to_tracker_config();
        Tracker tracker(skeleton, cameras, tracker_config);

        // Validate camera model by triangulating first frame
        auto first_frame_obs = observations_set.get_all_at_time(timestamps[0]);
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
            }
            tracker.initialize_from_state(python_state.value(), timestamps[0]);
        } else {
            // Fall back to rest pose if Python state not available
            if (!quiet) {
                fmt::print("  Python state not available, initializing from rest pose\n");
            }
            tracker.initialize_from_rest_pose(timestamps[0]);
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

        // Export initialization state as frame 0 at t=0.0
        if (state_vec_file.is_open()) {
            export_state_vector(state_vec_file, 0, 0.0, tracker.state(), skeleton);
        }

        // Process first update (correct the initialization with first observations)
        {
            auto frame_0_obs = observations_set.get_all_at_time(timestamps[0]);
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
                        "  Frame 0: Updating initialization with {} observations ({} unique "
                        "marker-camera pairs) at t={:.6f}\n",
                        frame_0_obs.size(), unique_pairs.size(), timestamps[0]);
                    fmt::print("    Camera distribution:");
                    for (const auto& [cam_id, count] : camera_counts) {
                        fmt::print(" cam{}={}", cam_id, count);
                    }
                    fmt::print("\n");
                }

                auto result = tracker.track_frame(frame_0_obs, timestamps[0]);

                // Export posterior as frame 1
                if (state_vec_file.is_open()) {
                    export_state_vector(state_vec_file, 1, timestamps[0], tracker.state(),
                                        skeleton);
                }

                if (result.tracking_lost) {
                    fmt::print(stderr, "Warning: Tracking lost on first update\n");
                } else {
                    frames_tracked++;
                }
            } else {
                if (!quiet) {
                    fmt::print("  No observations at t={:.6f}, skipping first update\n",
                               timestamps[0]);
                }
            }
        }

        for (int i = 1; i < num_frames_to_process; ++i) {
            double frame_timestamp = timestamps[i];

            // Set frame number for UKF debug
            if (auto* ukf = tracker.get_ukf()) {
                ukf->set_frame_number(i);
            }

            // Get observations at this time
            auto frame_obs = observations_set.get_all_at_time(frame_timestamp);

            if (frame_obs.empty()) {
                if (verbose) {
                    fmt::print("  t={:.3f}: No observations, skipping\n", frame_timestamp);
                }
                continue;
            }

            // Export predicted observations BEFORE tracking update (uses prior state)
            if (pred_obs_file.is_open()) {
                export_predicted_observations(pred_obs_file, i + 1, frame_timestamp, frame_obs,
                                              tracker.state(), fk, cameras, skeleton);
            }

            // DEBUG: Export frame 2 prior state for comparison with Python frame 1
            if (i == 1) {
                auto debug_dir = config.output_dir / "debug";
                std::filesystem::create_directories(debug_dir);
                auto prior_state_path = debug_dir / "frame_0001_prior_state.csv";
                std::ofstream prior_file(prior_state_path);
                if (prior_file.is_open()) {
                    // Write header (same as state_vectors.csv)
                    prior_file << generate_state_header(skeleton) << "\n";
                    // Write prior state (before update) as frame 2 (Python frame 1)
                    export_state_vector(prior_file, 2, frame_timestamp, tracker.state(), skeleton);
                    prior_file.close();
                    if (!quiet) {
                        fmt::print("  [DEBUG] Exported frame 2 prior state to {}\n",
                                   prior_state_path.string());
                    }
                }
            }

            // Track
            auto result = tracker.track_frame(frame_obs, frame_timestamp);

            if (result.tracking_lost) {
                frames_lost++;
                if (verbose) {
                    fmt::print("  t={:.3f}: Tracking LOST ({} obs)\n", frame_timestamp,
                               result.update_info.num_observations);
                }
            } else {
                frames_tracked++;
                if (verbose) {
                    fmt::print("  t={:.3f}: {} inliers, {} outliers\n", frame_timestamp,
                               result.update_info.num_inliers, result.update_info.num_outliers);
                }
            }

            // Export state vector AFTER tracking update (posterior state) as frame i+1
            if (state_vec_file.is_open()) {
                export_state_vector(state_vec_file, i + 1, frame_timestamp, result.state, skeleton);
            }

            // Export
            if (exporter) {
                // Compute marker positions using FK
                auto marker_positions_3d_map = fk.compute(result.state);

                // Convert to std::map (exporter expects std::map, not unordered_map)
                std::map<std::string, Eigen::Vector3d> marker_positions_3d(
                    marker_positions_3d_map.begin(), marker_positions_3d_map.end());

                exporter->write_frame(start_frame + i, frame_timestamp, result.state,
                                      marker_positions_3d, frame_obs, result.update_info);
            }

            if (stats_tracker) {
                stats_tracker->add_frame_stats(start_frame + i, frame_timestamp, result.update_info,
                                               result.covariance, result.tracking_lost);
            }

            // Progress indicator
            if (!quiet && !verbose && i % 10 == 0) {
                double percent = 100.0 * i / num_frames_to_process;
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                double elapsed_sec =
                    std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() / 1000.0;
                double fps = i / elapsed_sec;
                int eta_sec = static_cast<int>((num_frames_to_process - i) / fps);

                fmt::print("  Progress: {}/{} ({:.1f}%) | {:.1f} fps | ETA: {}s\r", i,
                           num_frames_to_process, percent, fps, eta_sec);
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
            metadata["start_frame"] = start_frame;
            metadata["num_frames"] = num_frames_to_process;
            metadata["config"] = {
                {"process_noise_std", config.process_noise_std},
                {"measurement_noise_std", config.measurement_noise_std},
                {"outlier_threshold", config.outlier_threshold},
                {"ukf_alpha", config.ukf_alpha},
            };

            stats_tracker->write_summary_stats(config.output_dir / "overall_stats.json", metadata);
        }

        // Final summary
        auto end_time = std::chrono::steady_clock::now();
        auto total_elapsed = end_time - start_time;
        double total_sec =
            std::chrono::duration_cast<std::chrono::milliseconds>(total_elapsed).count() / 1000.0;
        double avg_fps = frames_tracked / total_sec;

        if (!quiet) {
            fmt::print("\nTracking complete!\n");
            fmt::print("  Tracked: {}/{} frames ({:.1f}%)\n", frames_tracked, num_frames_to_process,
                       100.0 * frames_tracked / num_frames_to_process);
            fmt::print("  Lost: {} frames\n", frames_lost);
            fmt::print("  Average FPS: {:.1f}\n", avg_fps);
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
